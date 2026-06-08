"""Live smoke test for the agent layer (Layer 4).

Unlike the layer 1–3 harnesses, the agent's output is LLM-generated and therefore
non-deterministic, so this script does **not** assert exact answer text. Instead it
exercises the real path end to end: it builds the deterministic artifacts, stands up
a ``GiantAgent`` (which makes a live Gemini call to generate the backlog narrative),
asks a handful of questions spanning the spec's 13 required queries, and prints each
answer alongside the tools the agent chose to call. The only hard assertions are
structural — a non-empty answer, and at least one tool call on the data questions —
so a human can eyeball that answers cite real metrics and name specific ads.

Requires a Gemini API key. If ``GEMINI_API_KEY`` (or ``GOOGLE_API_KEY``) is not set,
the script prints a SKIPPED notice and exits 0, so a full ``verify_*`` sweep never
breaks just because no key is configured.

Run from the repo root with the project venv::

    python -m util.verify_agent
"""

from __future__ import annotations

import os
import sys

# Allow running as a plain script as well as a module.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.diagnosis import build_theme_diagnosis  # noqa: E402
from src.hypothesis import build_ranked_backlog  # noqa: E402
from src.signal import build_theme_signals  # noqa: E402

ADS_PATH = os.path.join(_REPO_ROOT, "data", "ads.csv")
ENG_PATH = os.path.join(_REPO_ROOT, "data", "engagement.csv")

# Representative questions spanning the spec's 13 (snapshot, ad-level drill-down,
# uniformity, backlog/traceability, CEO summary). `expect_tool` is True where a
# grounded answer must hit at least one accessor.
_QUESTIONS = [
    ("Which themes are performing best right now, and which are declining?", True),
    ("What's driving the decline in safety — which specific ads, and is it the hook, "
     "the format, or the placement?", True),
    ("Is connection's signal uniform across its ads or isolated to specific ones? "
     "Are there standout ads worth replicating?", True),
    ("Should we scale spend on wonder, or have we hit an efficiency ceiling? "
     "Walk me through the evidence.", True),
    ("Give me a 2-minute summary of where our paid creative strategy stands.", False),
]


def _check(label: str, condition: bool, failures: list[str]) -> None:
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}")
    if not condition:
        failures.append(label)


def main() -> int:
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        print("=" * 100)
        print("SKIPPED: set GEMINI_API_KEY (or GOOGLE_API_KEY) to run the live agent smoke test.")
        print("=" * 100)
        return 0

    # Import here so the SKIP path above needs no google-genai install.
    from src.agent import GiantAgent  # noqa: E402

    theme_signals, ads_df, _eng_df, ad_signals_df = build_theme_signals(ADS_PATH, ENG_PATH)
    diagnosis = build_theme_diagnosis(theme_signals, ad_signals_df, ads_df)
    backlog = build_ranked_backlog(theme_signals, diagnosis, ad_signals_df)

    print("=" * 100)
    print("BUILDING AGENT (live Gemini call for the backlog narrative)…")
    agent = GiantAgent(theme_signals, diagnosis, backlog, ad_signals_df, ads_df)

    print("\n" + "=" * 100)
    print("BACKLOG NARRATIVE")
    print("=" * 100)
    print(agent.backlog_narrative)

    print("\n" + "=" * 100)
    print("SAMPLE Q&A")
    print("=" * 100)
    failures: list[str] = []
    for i, (question, expect_tool) in enumerate(_QUESTIONS, start=1):
        print(f"\n--- Q{i}: {question}")
        agent.reset()  # each question is independent for the smoke test
        answer = agent.chat(question)
        print(f"tools called: {agent.last_tool_calls or '(none)'}")
        print(f"A: {answer}")
        _check(f"Q{i} returned a non-empty answer", bool(answer.strip()), failures)
        if expect_tool:
            _check(f"Q{i} called at least one tool", len(agent.last_tool_calls) > 0, failures)

    print("\n" + "=" * 100)
    if failures:
        print(f"RESULT: {len(failures)} structural check(s) FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("RESULT: all structural checks PASSED. Eyeball the answers above for metric grounding.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
