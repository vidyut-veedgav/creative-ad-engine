"""Layer 5: Streamlit UI for the Giant Creative Ad Engine.

Ties the four upstream layers together into a single demo-ready page:

    data/*.csv
        → signal  (Layer 1)  → theme_signals, ad_signals_df
        → diagnosis (Layer 2) → theme_diagnosis
        → hypothesis (Layer 3) → ranked_backlog
        → agent  (Layer 4)    → GiantAgent (conversational)
        → this file (Layer 5) → dashboard + chat

Design contract preserved here: **the UI never computes a metric.** Layers 1–3
run exactly once (cached), the agent's startup narrative runs exactly once
(cached), and every widget below only *reads* those precomputed artifacts, so
every number on screen is traceable to a verified pipeline output.

Run with:  streamlit run app/app.py
"""

from __future__ import annotations

import os
import sys

import pandas as pd
import streamlit as st

# ── Path bootstrap ────────────────────────────────────────────────────────
# app/ is a subdirectory; put the repo root on sys.path so `src.*` / `util.*`
# import the same way the verify_*.py harnesses do.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.agent import GiantAgent  # noqa: E402
from src.diagnosis import build_theme_diagnosis  # noqa: E402
from src.hypothesis import build_ranked_backlog  # noqa: E402
from src.signal import build_theme_signals  # noqa: E402

ADS_PATH = os.path.join(_REPO_ROOT, "data", "ads.csv")
ENG_PATH = os.path.join(_REPO_ROOT, "data", "engagement.csv")


# ── API key bootstrap ─────────────────────────────────────────────────────
# Streamlit does not read .env files, so the agent's key would otherwise have
# to be exported in the shell before every `streamlit run`. Load the repo-root
# .env into os.environ once at import time so the chat works out of the box.
# Real environment variables win; we only fill in keys that aren't already set.
def _load_dotenv(path: str = os.path.join(_REPO_ROOT, ".env")) -> None:
    try:
        with open(path, encoding="utf-8") as fh:
            lines = fh.readlines()
    except OSError:
        return  # no .env present; rely on the ambient environment
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv()

# Canonical theme order (left→right on the dashboard).
THEME_ORDER = ["wonder", "safety", "confidence", "connection"]

# Trend → accent colour. Negative CPI delta = improving = "rising" = good.
TREND_COLOR = {"rising": "#1a7f37", "declining": "#cf222e", "stable": "#6e7781"}
# A short, plain-language gloss per hypothesis type (shown under the badge).
HYPOTHESIS_GLOSS = {
    "amplification": "scale spend · expand placements",
    "optimization": "refresh creative · monitor",
    "replication": "copy the isolated winner",
    "intervention": "fix specific laggard ads",
    "retirement": "wind down · reallocate budget",
}

EXAMPLE_QUESTIONS = [
    "Which themes are declining and what should we test next?",
    "Why is Safety losing efficiency, and should we retire it?",
    "What's behind Wonder's efficiency ceiling?",
    "Which Connection ads are outliers worth replicating?",
    "How is Confidence trending, and can we scale it?",
    "What are the highest-ICE recommendations right now?",
]


# ── Cached pipeline + agent ───────────────────────────────────────────────

@st.cache_resource(show_spinner="Running the analysis pipeline…")
def load_pipeline():
    """Run Layers 1–3 once. Returns every artifact the UI and agent need."""
    theme_signals, ads_df, eng_df, ad_signals_df = build_theme_signals(ADS_PATH, ENG_PATH)
    theme_diagnosis = build_theme_diagnosis(theme_signals, ad_signals_df, ads_df)
    ranked_backlog = build_ranked_backlog(theme_signals, theme_diagnosis, ad_signals_df)
    return theme_signals, ads_df, eng_df, ad_signals_df, theme_diagnosis, ranked_backlog


@st.cache_resource(show_spinner="Starting the conversational agent…")
def load_agent(_pipeline):
    """Build the GiantAgent once (its startup narrative is one LLM call).

    Returns ``(agent, error)``. ``error`` is a string when construction failed
    (e.g. no ``GEMINI_API_KEY``) so the dashboard can still render and the chat
    can degrade gracefully. The ``_pipeline`` arg is underscored so Streamlit
    doesn't try to hash the DataFrames inside it.
    """
    theme_signals, ads_df, _eng_df, ad_signals_df, theme_diagnosis, ranked_backlog = _pipeline
    try:
        agent = GiantAgent(
            theme_signals, theme_diagnosis, ranked_backlog, ad_signals_df, ads_df
        )
        return agent, None
    except Exception as exc:  # missing key, SDK/auth error, etc.
        return None, str(exc)


# ── Small render helpers ──────────────────────────────────────────────────

def _fmt_pct(value: float | None, places: int = 1) -> str:
    """Signed percentage, e.g. ``-2.5%``; dash for missing."""
    return "-" if value is None else f"{value:+.{places}f}%"


def _badge(text: str, color: str) -> str:
    """Inline coloured pill (HTML)."""
    return (
        f"<span style='background:{color};color:#fff;border-radius:6px;"
        f"padding:2px 8px;font-size:0.72rem;font-weight:600;"
        f"text-transform:uppercase;letter-spacing:.04em;'>{text}</span>"
    )


def render_card(theme: str, sig: dict, selected: bool) -> bool:
    """Render one theme card; return True if its button was clicked this run."""
    color = TREND_COLOR.get(sig["trend_direction"], "#6e7781")
    htype = sig["hypothesis_type"]
    with st.container(border=True):
        st.markdown(
            f"<div style='font-size:1.25rem;font-weight:700;"
            f"border-left:4px solid {color};padding-left:8px;'>{theme.title()}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            _badge(htype, color)
            + f"<div style='color:#6e7781;font-size:0.74rem;margin-top:4px;'>"
            f"{HYPOTHESIS_GLOSS.get(htype, '')}</div>",
            unsafe_allow_html=True,
        )
        # Big headline metric (the 7-day CPI delta). The trend word is rendered as
        # our own coloured badge below — st.metric's delta_color only tints numeric
        # deltas, so a string like "rising"/"declining"/"stable" would all paint the
        # same colour. TREND_COLOR gives three distinct ones (green/red/grey).
        st.metric(label="7-day CPI", value=_fmt_pct(sig["cpi_7d_delta_pct"]))
        st.markdown(_badge(sig["trend_direction"], color), unsafe_allow_html=True)
        st.caption(
            f"**{sig['trend_strength']}** · {sig['uniformity']} · "
            f"{sig['ads_rising']}↑ / {sig['ads_declining']}↓ / {sig['ads_stable']}→"
        )
        if sig.get("efficiency_ceiling"):
            st.caption("⚠️ efficiency ceiling reached")
        label = "✓ Showing details" if selected else "View ad-level detail →"
        return st.button(label, key=f"card_{theme}", width="stretch")


def render_drilldown(theme: str, pipeline) -> None:
    """Inline ad-level analysis for the selected theme."""
    theme_signals, _ads_df, eng_df, ad_signals_df, theme_diagnosis, ranked_backlog = pipeline
    sig = theme_signals[theme]
    diag = theme_diagnosis[theme]
    color = TREND_COLOR.get(sig["trend_direction"], "#6e7781")

    st.markdown(
        f"### {theme.title()} &nbsp; {_badge(sig['hypothesis_type'], color)}",
        unsafe_allow_html=True,
    )
    st.info(diag["pattern_summary"], icon="🧭")

    # Dominant vs weak format/placement.
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dominant format", diag.get("dominant_format") or "-")
    c2.metric("Dominant placement", diag.get("dominant_placement") or "-")
    c3.metric("Weak format", diag.get("weak_format") or "-")
    c4.metric("Weak placement", diag.get("weak_placement") or "-")

    # Per-ad trend charts over the full 90 days (one line per ad).
    theme_eng = eng_df[eng_df["theme"] == theme]
    st.markdown("#### Per-ad trends (90 days)")
    cc1, cc2 = st.columns(2)
    with cc1:
        st.caption("CPI by ad (lower is better)")
        st.line_chart(theme_eng.pivot_table(index="date", columns="ad_id", values="CPI"))
    with cc2:
        st.caption("CTR by ad (higher is better)")
        st.line_chart(theme_eng.pivot_table(index="date", columns="ad_id", values="CTR"))

    # Ad-level signal table, annotated with each ad's diagnosis role.
    st.markdown("#### Ad-level signals")
    role_by_ad: dict[str, str] = {}
    for ad in diag.get("leading_ads", []):
        role_by_ad[ad["ad_id"]] = "leading"
    for ad in diag.get("lagging_ads", []):
        role_by_ad[ad["ad_id"]] = "lagging"
    for ad in diag.get("outlier_ads", []):
        role_by_ad[ad["ad_id"]] = "outlier"

    cols = [
        "ad_id", "format", "placement", "cpi_7d_delta_pct",
        "ctr_7d_delta_pct", "last7d_mean_cpi", "recent_spend", "direction",
    ]
    table = ad_signals_df[ad_signals_df["theme"] == theme][cols].copy()
    table.insert(1, "role", table["ad_id"].map(role_by_ad).fillna("-"))
    table = table.sort_values("cpi_7d_delta_pct")
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={
            "ad_id": "Ad",
            "role": "Role",
            "format": "Format",
            "placement": "Placement",
            "cpi_7d_delta_pct": st.column_config.NumberColumn("CPI Δ 7d %", format="%.1f"),
            "ctr_7d_delta_pct": st.column_config.NumberColumn("CTR Δ 7d %", format="%.1f"),
            "last7d_mean_cpi": st.column_config.NumberColumn("CPI ($)", format="$%.2f"),
            "recent_spend": st.column_config.NumberColumn("Spend 7d ($)", format="$%.0f"),
            "direction": "Trend",
        },
    )

    # This theme's recommendations from the ranked backlog.
    theme_items = [it for it in ranked_backlog if it["theme"] == theme]
    if theme_items:
        st.markdown("#### Recommended next steps")
        for it in theme_items:
            with st.expander(
                f"#{it['rank']} · {it['hypothesis_type'].title()} "
                f"({it['level']}) · ICE {it['ice_score']}/10",
                expanded=False,
            ):
                st.markdown(f"**Recommendation.** {it['recommendation']}")
                points = it.get("evidence_points")
                if points:
                    st.markdown("**Evidence**")
                    for p in points:
                        st.markdown(f"- {p}")
                else:
                    st.markdown(f"**Evidence.** {it['evidence']}")
                st.caption(
                    f"Impact: {it['ice_impact']} · Confidence: {it['ice_confidence']} "
                    f"· Ease: {it['ice_ease']}"
                )
                if it.get("open_questions"):
                    st.markdown("**Open questions (to answer with a test):**")
                    for q in it["open_questions"]:
                        st.markdown(f"- {q}")


def render_chat(agent, agent_error: str | None) -> None:
    """Global conversational interface over all four themes."""
    st.markdown("## Ask the analyst")
    st.caption(
        "The agent answers only from the precomputed signals, diagnosis, and "
        "backlog."
    )

    if agent is None:
        st.warning(
            "Chat is disabled because the conversational agent could not start "
            "(usually a missing `GEMINI_API_KEY` / `GOOGLE_API_KEY`). The "
            "dashboard above works without a key.\n\n"
            f"_Details: {agent_error}_"
        )
        return

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Replay history.
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("tools"):
                st.caption(f"Sourced via: {', '.join(msg['tools'])}")

    if not st.session_state.messages:
        st.markdown(
            "**Try asking:**\n"
            + "\n".join(f'- *"{q}"*' for q in EXAMPLE_QUESTIONS)
        )

    prompt = st.chat_input("Ask about themes, ads, or what to test next…")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing…"):
            answer = agent.chat(prompt)
            tools = list(agent.last_tool_calls)
        st.markdown(answer)
        if tools:
            st.caption(f"Sourced via: {', '.join(tools)}")
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "tools": tools}
    )


# ── Page ──────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(page_title="Giant Creative Ad Engine", layout="wide", page_icon="📊")
    st.title("Giant Creative Ad Engine")
    st.caption(
        "Which creative themes are rising or declining, and what to test next. "
        "Click a theme for ad-level analysis."
    )

    pipeline = load_pipeline()
    agent, agent_error = load_agent(pipeline)
    theme_signals = pipeline[0]

    if "selected_theme" not in st.session_state:
        st.session_state.selected_theme = None

    # Dashboard: one clickable card per theme.
    cols = st.columns(len(THEME_ORDER))
    for col, theme in zip(cols, THEME_ORDER):
        with col:
            clicked = render_card(
                theme, theme_signals[theme],
                selected=(st.session_state.selected_theme == theme),
            )
        if clicked:
            # Toggle: clicking the open card again collapses the drill-down.
            st.session_state.selected_theme = (
                None if st.session_state.selected_theme == theme else theme
            )
            st.rerun()

    # Inline drill-down for the selected theme.
    if st.session_state.selected_theme:
        st.divider()
        render_drilldown(st.session_state.selected_theme, pipeline)

    # Chat.
    st.divider()
    render_chat(agent, agent_error)


if __name__ == "__main__":
    main()
