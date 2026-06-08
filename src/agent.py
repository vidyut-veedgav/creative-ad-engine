"""Layer 4 — Agent Layer.

A conversational assistant that answers questions about the creative-ad pipeline
by *interpreting* the precomputed artifacts from Layers 1–3 — it never recomputes
or derives a metric. Every number it cites is pulled, live, from one of four
typed accessor tools over ``theme_signals`` (Layer 1), ``theme_diagnosis``
(Layer 2), and ``ranked_backlog`` (Layer 3). That keeps every answer traceable
to a specific, already-validated value and time window.

**Provider note — a deliberate divergence from the spec.** ``docs/system_architecture.md``
and ``CLAUDE.md`` describe an Anthropic/Claude agent. This implementation instead
uses Google's ``google-genai`` SDK with ``gemini-2.5-flash`` (per an explicit
project decision). The architecture is otherwise unchanged: the LLM is strictly
downstream of the deterministic layers and reaches data only through the tool
surface below. The API key is read from the ``GEMINI_API_KEY`` (or
``GOOGLE_API_KEY``) environment variable by the SDK's default ``genai.Client()``.

Public surface (what ``app/`` wires in at startup):

- ``build_tools(...)`` → the dict of four accessor closures over the artifacts.
- ``generate_backlog_narrative(ranked_backlog, theme_signals)`` → a one-time,
  cached natural-language summary of the backlog (spec contract).
- ``answer_query(query, tools)`` → a stateless single-shot answer (spec contract).
- ``GiantAgent`` → a stateful, multi-turn assistant for the chat UI; holds
  conversation history so follow-ups ("why?", "what about that ad?") work.

Design notes worth knowing when reading the code:

- **Manual tool loop, not auto-calling.** We declare the tools as
  ``FunctionDeclaration``s and run the call→execute→respond loop ourselves
  (``_run_tool_loop``) rather than handing the SDK Python callables. This lets us
  *record which tools fired* per answer (``last_tool_calls``) — a cheap form of
  the spec's traceability requirement — and bound the loop.
- **Tools return JSON-safe slices.** The artifacts carry pandas/numpy scalars and
  ``NaN``; ``_to_native`` coerces them to plain Python (``NaN``/``NA`` → ``None``)
  so they survive serialization into a Gemini function response.
- **``get_ad_detail`` is the one fused tool.** It joins the static creative
  metadata (``ads.csv`` row: hook/headline/copy) with the ad's signal row, which
  is what lets the agent answer "is it the hook, the format, or the placement?".
"""

from __future__ import annotations

import math

import pandas as pd
from google import genai
from google.genai import types

# ── Constants ──────────────────────────────────────────────────────────────
DEFAULT_MODEL = "gemini-2.5-flash"
_MAX_TOOL_ITERATIONS = 6        # cap the call→execute→respond loop
_TEMPERATURE = 0.2              # low: answers are grounded, not creative

_THEMES = ("wonder", "safety", "confidence", "connection")
_HYPOTHESIS_TYPES = (
    "amplification", "optimization", "replication", "intervention", "retirement",
)


# ── JSON-safety ─────────────────────────────────────────────────────────────

def _to_native(obj):
    """Recursively coerce pandas/numpy values to JSON-serializable Python.

    numpy scalars → their Python equivalent; ``NaN``/``NaT``/``pd.NA`` → ``None``;
    dicts/lists are recursed. Plain Python values pass through untouched.
    """
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, str):
        return obj
    # numpy / pandas scalar → Python scalar (np.float64, np.int64, np.bool_, …).
    if hasattr(obj, "item") and not isinstance(obj, bytes):
        try:
            obj = obj.item()
        except (ValueError, TypeError):
            pass
    if obj is None:
        return None
    if isinstance(obj, float) and math.isnan(obj):
        return None
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    return obj


# ── Tool factory (the agent's only window onto the data) ────────────────────

def build_tools(
    theme_signals: dict,
    theme_diagnosis: dict,
    ranked_backlog: list[dict],
    ad_signals_df: pd.DataFrame,
    ads_df: pd.DataFrame,
) -> dict:
    """Build the four spec accessors as closures over the precomputed artifacts.

    Returns a ``{name: callable}`` dict. Each callable returns plain JSON-safe
    Python (via ``_to_native``) and returns an ``{"error": ...}`` dict for bad
    inputs rather than raising, so the model can recover within the tool loop.
    """
    ads_by_id = ads_df.set_index("ad_id")
    signals_by_id = ad_signals_df.set_index("ad_id")

    def get_theme_signals(theme: str | None = None):
        """theme_signals for one theme, or all four themes when ``theme`` is None."""
        if theme is None:
            return _to_native(theme_signals)
        key = theme.strip().lower()
        if key not in theme_signals:
            return {"error": f"unknown theme '{theme}'; valid themes: {list(_THEMES)}"}
        return _to_native(theme_signals[key])

    def get_theme_diagnosis(theme: str):
        """Ad-level diagnosis for one theme (incl. the deterministic pattern_summary)."""
        key = theme.strip().lower()
        if key not in theme_diagnosis:
            return {"error": f"unknown theme '{theme}'; valid themes: {list(_THEMES)}"}
        return _to_native(theme_diagnosis[key])

    def get_ranked_backlog(theme: str | None = None, hypothesis_type: str | None = None):
        """The ICE-ranked backlog, optionally filtered by theme and/or type."""
        items = ranked_backlog
        if theme is not None:
            key = theme.strip().lower()
            items = [it for it in items if it["theme"] == key]
        if hypothesis_type is not None:
            htype = hypothesis_type.strip().lower()
            items = [it for it in items if it["hypothesis_type"] == htype]
        return _to_native(items)

    def get_ad_detail(ad_id: str):
        """Static creative metadata fused with the ad's Layer 1 signal row."""
        key = ad_id.strip()
        if key not in signals_by_id.index:
            return {"error": f"unknown ad_id '{ad_id}'"}
        metadata = ads_by_id.loc[key].to_dict() if key in ads_by_id.index else {}
        signals = signals_by_id.loc[key].to_dict()
        # signals overwrites the shared theme/format/placement/platform keys with
        # identical values; the union is the full picture of one ad.
        return _to_native({"ad_id": key, **metadata, **signals})

    return {
        "get_theme_signals": get_theme_signals,
        "get_theme_diagnosis": get_theme_diagnosis,
        "get_ranked_backlog": get_ranked_backlog,
        "get_ad_detail": get_ad_detail,
    }


# ── Gemini tool declarations ────────────────────────────────────────────────

def _function_declarations() -> list[types.FunctionDeclaration]:
    """Declare the four tools to Gemini (names mirror ``build_tools`` keys)."""
    theme_param = types.Schema(
        type=types.Type.STRING, enum=list(_THEMES),
        description="One of the four creative themes.",
    )
    return [
        types.FunctionDeclaration(
            name="get_theme_signals",
            description=(
                "Trend signals for a theme: trend_direction, trend_strength, uniformity, "
                "7-day CPI/CTR/CVR deltas (negative CPI delta = improving), per-ad "
                "rising/declining/stable counts, diagnosis_level, hypothesis_type, and the "
                "efficiency_ceiling flag. Omit `theme` to get all four themes at once."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={"theme": theme_param},
            ),
        ),
        types.FunctionDeclaration(
            name="get_theme_diagnosis",
            description=(
                "Ad-level diagnosis for one theme: leading_ads, lagging_ads, outlier_ads "
                "(with distinguishing_features), dominant/weak format and placement, and a "
                "deterministic one-sentence pattern_summary. Use this to explain WHY a theme "
                "is moving the way it is."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={"theme": theme_param},
                required=["theme"],
            ),
        ),
        types.FunctionDeclaration(
            name="get_ranked_backlog",
            description=(
                "The ICE-scored, ranked action backlog. Each item has rank, theme, "
                "hypothesis_type, level, recommendation, evidence, ice_impact/confidence/ease, "
                "ice_score (1-10), and open_questions. Optionally filter by theme and/or "
                "hypothesis_type."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "theme": theme_param,
                    "hypothesis_type": types.Schema(
                        type=types.Type.STRING, enum=list(_HYPOTHESIS_TYPES),
                        description="Filter to a single hypothesis type.",
                    ),
                },
            ),
        ),
        types.FunctionDeclaration(
            name="get_ad_detail",
            description=(
                "Full detail for one ad by ad_id: static creative metadata (hook_text, "
                "headline, primary_text, call_to_action, format, placement, platform, "
                "aspect_ratio, video_length_sec) fused with its signal row (7-day deltas, "
                "current CPI/CTR/CVR levels, 30-day slopes, direction/strength, efficiency "
                "ceiling, outlier flag). Use this to tell whether a result is driven by the "
                "hook, the format, or the placement, and to read creative copy."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "ad_id": types.Schema(
                        type=types.Type.STRING, description="An ad identifier, e.g. 'ad_019'.",
                    ),
                },
                required=["ad_id"],
            ),
        ),
    ]


def _build_tool() -> types.Tool:
    """Bundle the function declarations into a single Gemini ``Tool``."""
    return types.Tool(function_declarations=_function_declarations())


# ── System instruction ──────────────────────────────────────────────────────

_GLOSSARY = """\
Domain glossary (so you can explain implications, never recompute):
- uniformity: >60% of a theme's ads moving together = `uniform` (a strategy-level
  story); <40% = `isolated` (an execution-level story, a few ads carry it); else `mixed`.
- diagnosis_level: `strategy` (act on the whole theme) vs `execution` (act on specific ads).
- efficiency_ceiling: CPI has stopped improving despite rising spend — scaling budget
  won't help; refresh the creative instead.
- hypothesis_type → action: amplification = scale spend/expand placements;
  optimization = refresh creative / monitor; replication = copy an isolated winner across
  the theme; intervention = fix specific laggard ads' format/placement; retirement = wind
  the theme down and reallocate budget.
- ICE: ice_score (1-10) = impact x confidence x ease; higher ranks first in the backlog.
- Deltas are 7-day week-over-week unless a tool says otherwise; slopes are 30-day. A
  negative CPI delta means CPI is FALLING, i.e. the theme is IMPROVING."""

_RULES = """\
How you must answer:
- Pull every fact from the tools. Never invent, estimate, or recompute a metric — if a
  tool can't supply something, say so plainly.
- Cite the specific metric and its time window for each claim, e.g. "CPI -12% wk/wk (7-day)"
  or "30-day CPI slope flat".
- Distinguish theme-level claims (from get_theme_signals/diagnosis) from ad-level claims
  (from get_ad_detail), and name specific ad_ids when the answer is about specific ads.
- Always end with a concrete recommended action, not a general observation.
- Be warm, concise, and plain-spoken. You're a friendly analyst, not a report generator."""


def _system_instruction(theme_signals: dict | None, backlog_narrative: str | None) -> str:
    """Assemble the system prompt: role + rules + glossary + cached backlog context."""
    parts = [
        "You are the analyst assistant for Giant, an AI storytelling app for children. "
        "Giant's paid ads target PARENTS (the buyer), not children — CTR/CVR reflect parent "
        "behavior. Four creative themes run: wonder, safety, confidence, connection. You sit "
        "downstream of a deterministic analytics pipeline and answer questions by calling "
        "tools over its precomputed outputs.",
        _RULES,
        _GLOSSARY,
    ]
    if theme_signals is not None:
        snapshot = "; ".join(
            f"{t}: {s['trend_direction']}/{s['trend_strength']}, {s['uniformity']}, "
            f"{s['hypothesis_type']}"
            for t, s in theme_signals.items()
        )
        parts.append(f"Current theme snapshot (call tools for the numbers): {snapshot}.")
    if backlog_narrative:
        parts.append("Precomputed backlog narrative for context:\n" + backlog_narrative)
    return "\n\n".join(parts)


# ── Tool-call loop ──────────────────────────────────────────────────────────

def _dispatch(tools: dict, name: str, args: dict):
    """Execute one tool call, returning an error dict rather than raising."""
    fn = tools.get(name)
    if fn is None:
        return {"error": f"unknown tool '{name}'"}
    try:
        return fn(**args)
    except Exception as exc:  # noqa: BLE001 — surface to the model, don't crash the turn
        return {"error": f"{type(exc).__name__}: {exc}"}


def _run_tool_loop(
    client: "genai.Client",
    model: str,
    system_instruction: str,
    history: list,
    tool: types.Tool,
    tools: dict,
    max_iter: int = _MAX_TOOL_ITERATIONS,
) -> tuple[str, list, list[str]]:
    """Drive the call→execute→respond loop until the model returns text.

    Appends each model turn (and the function-response turns) to ``history`` so a
    stateful caller can keep the conversation going. Returns
    ``(answer_text, history, tool_names_called)``.
    """
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        tools=[tool],
        temperature=_TEMPERATURE,
        # We run the loop manually, so disable the SDK's automatic calling.
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    tool_calls: list[str] = []
    response = None
    for _ in range(max_iter):
        response = client.models.generate_content(
            model=model, contents=history, config=config
        )
        if not response.candidates:
            return ("I couldn't generate a response for that.", history, tool_calls)
        history.append(response.candidates[0].content)

        calls = response.function_calls
        if not calls:
            return ((response.text or "").strip(), history, tool_calls)

        response_parts = []
        for call in calls:
            tool_calls.append(call.name)
            result = _dispatch(tools, call.name, dict(call.args or {}))
            response_parts.append(
                types.Part.from_function_response(name=call.name, response={"result": result})
            )
        history.append(types.Content(role="user", parts=response_parts))

    # Tool budget exhausted — return whatever text the last turn carried.
    trailing = (getattr(response, "text", "") or "").strip()
    return (trailing or "I ran out of tool-call steps before finishing that answer.",
            history, tool_calls)


# ── Spec-contract functions ─────────────────────────────────────────────────

def generate_backlog_narrative(
    ranked_backlog: list[dict],
    theme_signals: dict,
    *,
    client: "genai.Client | None" = None,
    model: str = DEFAULT_MODEL,
) -> str:
    """Produce a one-time, cacheable plain-language summary of the ranked backlog.

    No tools are needed: the backlog already carries deterministic ``recommendation``
    and ``evidence`` strings, so they're embedded directly in the prompt. Called once
    at app startup and injected into the agent's system prompt as grounding context.
    """
    client = client or genai.Client()
    lines = [
        f"#{it['rank']} [{it['theme']}/{it['hypothesis_type']}, {it['level']}-level, "
        f"ICE {it['ice_score']}] {it['recommendation']} (evidence: {it['evidence']})"
        for it in ranked_backlog
    ]
    prompt = (
        "Summarize this precomputed, ICE-ranked creative action backlog for the Giant "
        "paid-ads team in a friendly, concise narrative (~150 words). Walk the items in "
        "priority order; for each, name the theme, the recommended action, and the one-line "
        "reason. Use only the facts given — do not invent numbers. Close with the single "
        "highest-priority next step.\n\nRANKED BACKLOG:\n" + "\n".join(lines)
    )
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.3),
    )
    return (response.text or "").strip()


def answer_query(
    query: str,
    tools: dict,
    *,
    client: "genai.Client | None" = None,
    model: str = DEFAULT_MODEL,
    system_instruction: str | None = None,
    backlog_narrative: str | None = None,
) -> str:
    """Answer a single query statelessly (the spec's ``answer_query`` contract).

    The literal 2-arg call works; the optional keyword args let a caller inject a
    shared client, model, or grounding context without breaking the signature.
    """
    client = client or genai.Client()
    if system_instruction is None:
        system_instruction = _system_instruction(None, backlog_narrative)
    history = [types.Content(role="user", parts=[types.Part(text=query)])]
    text, _history, _calls = _run_tool_loop(
        client, model, system_instruction, history, _build_tool(), tools
    )
    return text


# ── Stateful assistant (for the chat UI) ────────────────────────────────────

class GiantAgent:
    """A multi-turn conversational assistant over the pipeline artifacts.

    Holds conversation history so follow-ups resolve naturally, and exposes
    ``last_tool_calls`` after each turn for traceability/debugging. The ``app/``
    layer builds one of these at startup and calls ``chat()`` per user message.
    """

    def __init__(
        self,
        theme_signals: dict,
        theme_diagnosis: dict,
        ranked_backlog: list[dict],
        ad_signals_df: pd.DataFrame,
        ads_df: pd.DataFrame,
        *,
        model: str = DEFAULT_MODEL,
        client: "genai.Client | None" = None,
    ) -> None:
        self.model = model
        self._client = client or genai.Client()
        self.tools = build_tools(
            theme_signals, theme_diagnosis, ranked_backlog, ad_signals_df, ads_df
        )
        self._tool = _build_tool()
        self.backlog_narrative = generate_backlog_narrative(
            ranked_backlog, theme_signals, client=self._client, model=model
        )
        self._system_instruction = _system_instruction(theme_signals, self.backlog_narrative)
        self._history: list = []
        self.last_tool_calls: list[str] = []

    def chat(self, query: str) -> str:
        """Answer ``query`` in the context of the running conversation."""
        self._history.append(types.Content(role="user", parts=[types.Part(text=query)]))
        text, history, calls = _run_tool_loop(
            self._client, self.model, self._system_instruction,
            self._history, self._tool, self.tools,
        )
        self._history = history
        self.last_tool_calls = calls
        return text

    def reset(self) -> None:
        """Start a fresh conversation (clears history; keeps the cached narrative)."""
        self._history = []
        self.last_tool_calls = []
