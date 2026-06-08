"""Validation harness for the hypothesis layer (Layer 3).

Mirrors ``util/verify_signals.py`` / ``util/verify_diagnosis.py``: builds the
signal → diagnosis → backlog artifacts, prints the full ranked backlog as
evidence, then asserts both the output contract and that the ranking matches the
narratives baked into the dataset. Exits non-zero on any failure.

Run from the repo root with the project venv::

    python -m util.verify_hypothesis
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

_ITEM_KEYS = {
    "rank", "hypothesis_type", "level", "theme", "recommendation", "evidence",
    "evidence_points", "ice_impact", "ice_confidence", "ice_ease", "ice_score",
    "open_questions",
}
_LABELS = {"high", "medium", "low"}
_TYPES = {"amplification", "optimization", "replication", "intervention", "retirement"}


def _check(label: str, condition: bool, failures: list[str]) -> None:
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}")
    if not condition:
        failures.append(label)


def _print_evidence(backlog: list[dict]) -> None:
    print("=" * 100)
    print("RANKED BACKLOG")
    print("=" * 100)
    for it in backlog:
        print(
            f"#{it['rank']}  {it['theme']:11s} {it['hypothesis_type']:13s} "
            f"[{it['level']:9s}] ICE={it['ice_score']:2d} "
            f"(I={it['ice_impact']}/C={it['ice_confidence']}/E={it['ice_ease']})"
        )
        print(f"      rec: {it['recommendation']}")
        print(f"      ev : {it['evidence']}")
        for p in it["evidence_points"]:
            print(f"        • {p}")
        print(f"      q  : {it['open_questions']}")


def _verify_contract(backlog: list[dict], theme_signals: dict, failures: list[str]) -> None:
    print("\nGLOBAL - contract shape:")
    _check("backlog is non-empty", len(backlog) > 0, failures)
    _check("every item has the contract keys", all(set(it) == _ITEM_KEYS for it in backlog), failures)
    _check("hypothesis_type in the enum", all(it["hypothesis_type"] in _TYPES for it in backlog), failures)
    _check("ICE labels valid",
           all({it["ice_impact"], it["ice_confidence"], it["ice_ease"]} <= _LABELS for it in backlog),
           failures)
    _check("ice_score in 1..10", all(1 <= it["ice_score"] <= 10 for it in backlog), failures)
    _check("open_questions non-empty list",
           all(isinstance(it["open_questions"], list) and it["open_questions"] for it in backlog),
           failures)
    _check("evidence_points non-empty list of strings",
           all(isinstance(it["evidence_points"], list) and it["evidence_points"]
               and all(isinstance(p, str) for p in it["evidence_points"]) for it in backlog),
           failures)
    _check("level matches diagnosis_level",
           all(it["level"] == theme_signals[it["theme"]]["diagnosis_level"] for it in backlog), failures)
    scores = [it["ice_score"] for it in backlog]
    _check("sorted by ice_score desc", scores == sorted(scores, reverse=True), failures)
    ranks = [it["rank"] for it in backlog]
    _check("ranks are 1..N contiguous", ranks == list(range(1, len(backlog) + 1)), failures)


def _verify_narratives(backlog: list[dict], failures: list[str]) -> None:
    by_theme: dict[str, list[dict]] = {}
    for it in backlog:
        by_theme.setdefault(it["theme"], []).append(it)

    print("\nGRANULARITY - primary + secondary:")
    _check("5 items (one per theme + connection's secondary)", len(backlog) == 5, failures)
    conn_types = {it["hypothesis_type"] for it in by_theme.get("connection", [])}
    _check("connection yields replication AND intervention",
           conn_types == {"replication", "intervention"}, failures)
    _check("strategy themes yield exactly one item each",
           all(len(by_theme.get(t, [])) == 1 for t in ("wonder", "safety", "confidence")), failures)

    print("\nCOVERAGE - all five hypothesis types appear:")
    present = {it["hypothesis_type"] for it in backlog}
    _check("all five hypothesis types present",
           present == {"amplification", "optimization", "replication", "intervention", "retirement"},
           failures)

    print("\nNARRATIVE - ranking reflects the baked stories:")
    # The two easy, high-leverage strategy plays lead the backlog (tie at the top):
    # scale the uniform riser (confidence -> amplification) and retire the uniform
    # decliner (safety -> retirement). Assert the pair, not which wins the tie-break.
    top2 = {(backlog[0]["theme"], backlog[0]["hypothesis_type"]),
            (backlog[1]["theme"], backlog[1]["hypothesis_type"])}
    _check("top two are confidence-amplification + safety-retirement",
           top2 == {("confidence", "amplification"), ("safety", "retirement")}, failures)

    conf = by_theme["confidence"][0]
    _check("confidence is amplification (uniform rise, no ceiling)",
           conf["hypothesis_type"] == "amplification", failures)

    wonder = by_theme["wonder"][0]
    _check("wonder is optimization (ceiling), not amplification",
           wonder["hypothesis_type"] == "optimization", failures)
    _check("wonder evidence mentions the efficiency ceiling",
           "ceiling" in wonder["evidence"].lower(), failures)
    _check("wonder recommendation names the video winner ad_001 + a static/feed laggard",
           "ad_001" in wonder["recommendation"]
           and ("ad_005" in wonder["recommendation"] or "ad_002" in wonder["recommendation"]),
           failures)
    _check("wonder evidence_points name a specific laggard ad",
           any("ad_005" in p or "ad_002" in p for p in wonder["evidence_points"]), failures)

    repl = next(it for it in by_theme["connection"] if it["hypothesis_type"] == "replication")
    interv = next(it for it in by_theme["connection"] if it["hypothesis_type"] == "intervention")
    _check("connection replication names the reels-video winners ad_019 + ad_022",
           "ad_019" in repl["recommendation"] and "ad_022" in repl["recommendation"], failures)
    _check("connection replication is execution-level", repl["level"] == "execution", failures)
    _check("connection intervention names the laggards ad_023 + ad_024",
           "ad_023" in interv["recommendation"] and "ad_024" in interv["recommendation"], failures)
    _check("connection primary (replication) outranks its secondary (intervention)",
           repl["rank"] < interv["rank"], failures)


def main() -> int:
    theme_signals, ads_df, _eng_df, ad_signals_df = build_theme_signals(ADS_PATH, ENG_PATH)
    diagnosis = build_theme_diagnosis(theme_signals, ad_signals_df, ads_df)
    backlog = build_ranked_backlog(theme_signals, diagnosis, ad_signals_df)

    _print_evidence(backlog)

    print("\n" + "=" * 100)
    print("ASSERTIONS")
    print("=" * 100)
    failures: list[str] = []
    _verify_contract(backlog, theme_signals, failures)
    _verify_narratives(backlog, failures)

    print("\n" + "=" * 100)
    if failures:
        print(f"RESULT: {len(failures)} check(s) FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("RESULT: all hypothesis checks PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
