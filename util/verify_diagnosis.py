"""Validation harness for the diagnosis layer (Layer 2).

Mirrors ``util/verify_signals.py``: builds the signal + diagnosis artifacts,
prints the per-theme diagnosis as evidence, then asserts that each theme's
diagnosis matches the narrative baked into the dataset. Exits non-zero on any
failure.

Run from the repo root with the project venv::

    python -m util.verify_diagnosis
"""

from __future__ import annotations

import os
import sys

# Allow running as a plain script as well as a module.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.diagnosis import build_theme_diagnosis  # noqa: E402
from src.signal import build_theme_signals  # noqa: E402

ADS_PATH = os.path.join(_REPO_ROOT, "data", "ads.csv")
ENG_PATH = os.path.join(_REPO_ROOT, "data", "engagement.csv")

_REQUIRED_KEYS = {
    "leading_ads", "lagging_ads", "outlier_ads",
    "dominant_format", "dominant_placement", "weak_format", "weak_placement",
    "pattern_summary",
}
_ENTRY_KEYS = {"ad_id", "format", "placement", "cpi_7d_delta_pct", "ctr_7d_delta_pct"}


def _check(label: str, condition: bool, failures: list[str]) -> None:
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}")
    if not condition:
        failures.append(label)


def _ids(entries: list[dict]) -> set[str]:
    return {e["ad_id"] for e in entries}


def _print_evidence(diagnosis: dict) -> None:
    for theme in ("wonder", "safety", "confidence", "connection"):
        d = diagnosis[theme]
        print("=" * 100)
        print(
            f"{theme.upper():11s} dominant_format={d['dominant_format']} "
            f"dominant_placement={d['dominant_placement']} "
            f"weak_format={d['weak_format']} weak_placement={d['weak_placement']}"
        )
        print("  leading:", [(a["ad_id"], a["format"], a["placement"]) for a in d["leading_ads"]])
        print("  lagging:", [(a["ad_id"], a["format"], a["placement"]) for a in d["lagging_ads"]])
        print("  outliers:", [(a["ad_id"], a["distinguishing_features"]) for a in d["outlier_ads"]])
        print("  summary:", d["pattern_summary"])


def _verify_global(diagnosis: dict, ad: "dict[str, float]", failures: list[str]) -> None:
    """Shape/contract checks that apply to every theme."""
    print("\nGLOBAL - contract shape across all themes:")
    _check("4 themes present", set(diagnosis) == {"wonder", "safety", "confidence", "connection"}, failures)
    level = ad  # ad_id -> last7d_mean_cpi
    for theme, d in diagnosis.items():
        ok_keys = set(d) == _REQUIRED_KEYS
        _check(f"{theme}: required keys present", ok_keys, failures)
        _check(f"{theme}: 2 leading + 2 lagging",
               len(d["leading_ads"]) == 2 and len(d["lagging_ads"]) == 2, failures)
        entries_ok = all(_ENTRY_KEYS <= set(e) for e in d["leading_ads"] + d["lagging_ads"])
        _check(f"{theme}: ad entries have contract fields", entries_ok, failures)
        # Every leading ad must be cheaper than every lagging ad (level ranking).
        lead_max = max(level[e["ad_id"]] for e in d["leading_ads"])
        lag_min = min(level[e["ad_id"]] for e in d["lagging_ads"])
        _check(f"{theme}: leading cheaper than lagging", lead_max < lag_min, failures)
        summary = d["pattern_summary"]
        _check(f"{theme}: pattern_summary is one sentence",
               bool(summary) and "\n" not in summary and summary.endswith("."), failures)


def _verify_narratives(diagnosis: dict, failures: list[str]) -> None:
    w, s, c, n = (diagnosis[t] for t in ("wonder", "safety", "confidence", "connection"))

    print("\nWONDER - expect video/reels driving, static weak, ceiling -> optimize:")
    _check("dominant_format == video", w["dominant_format"] == "video", failures)
    _check("dominant_placement == reels", w["dominant_placement"] == "reels", failures)
    _check("weak_format == static", w["weak_format"] == "static", failures)
    _check("leading == {ad_001, ad_004}", _ids(w["leading_ads"]) == {"ad_001", "ad_004"}, failures)
    _check("outlier_ads empty (uniform)", w["outlier_ads"] == [], failures)
    _check("summary mentions ceiling/optimize",
           "ceiling" in w["pattern_summary"] and "optimize" in w["pattern_summary"], failures)

    print("\nSAFETY - expect video cheapest but deteriorating, static weak, no placement signal:")
    _check("dominant_format == video", s["dominant_format"] == "video", failures)
    _check("weak_format == static", s["weak_format"] == "static", failures)
    _check("dominant_placement is None", s["dominant_placement"] is None, failures)
    _check("weak_placement is None", s["weak_placement"] is None, failures)
    _check("leading == {ad_010, ad_007}", _ids(s["leading_ads"]) == {"ad_010", "ad_007"}, failures)
    _check("leading video ads are deteriorating (cpi_7d_delta_pct > 0)",
           all(e["cpi_7d_delta_pct"] is not None and e["cpi_7d_delta_pct"] > 0 for e in s["leading_ads"]),
           failures)
    _check("summary captures video deterioration",
           "video" in s["pattern_summary"] and "deteriorat" in s["pattern_summary"], failures)
    _check("outlier_ads empty (uniform)", s["outlier_ads"] == [], failures)

    print("\nCONFIDENCE - expect video/stories cheapest, static/search weak, flat -> monitor:")
    _check("dominant_format == video", c["dominant_format"] == "video", failures)
    _check("dominant_placement == stories", c["dominant_placement"] == "stories", failures)
    _check("weak_format == static", c["weak_format"] == "static", failures)
    _check("weak_placement == search", c["weak_placement"] == "search", failures)
    _check("outlier_ads empty (uniform)", c["outlier_ads"] == [], failures)

    print("\nCONNECTION - expect isolated reels-video winners to replicate:")
    _check("dominant_format == video", n["dominant_format"] == "video", failures)
    _check("dominant_placement == reels", n["dominant_placement"] == "reels", failures)
    _check("weak_format == static", n["weak_format"] == "static", failures)
    _check("weak_placement == feed", n["weak_placement"] == "feed", failures)
    _check("outlier_ads non-empty", len(n["outlier_ads"]) > 0, failures)
    top_two = [a["ad_id"] for a in n["outlier_ads"][:2]]
    _check("top two outliers by extremity == [ad_022, ad_019]", top_two == ["ad_022", "ad_019"], failures)
    # The two winners are distinguished BY video+reels (the core signal); in this
    # small cluster an extra platform tag can also appear, so check by subset.
    _check("winner outliers distinguished by format:video + placement:reels",
           all({"format:video", "placement:reels"} <= set(a["distinguishing_features"])
               for a in n["outlier_ads"][:2]), failures)
    _check("leading == {ad_022, ad_019}", _ids(n["leading_ads"]) == {"ad_022", "ad_019"}, failures)
    _check("summary mentions replicate", "replicate" in n["pattern_summary"], failures)


def main() -> int:
    theme_signals, ads_df, _eng_df, ad_signals_df = build_theme_signals(ADS_PATH, ENG_PATH)
    diagnosis = build_theme_diagnosis(theme_signals, ad_signals_df, ads_df)
    level = dict(zip(ad_signals_df["ad_id"], ad_signals_df["last7d_mean_cpi"]))

    _print_evidence(diagnosis)

    print("\n" + "=" * 100)
    print("ASSERTIONS")
    print("=" * 100)
    failures: list[str] = []
    _verify_global(diagnosis, level, failures)
    _verify_narratives(diagnosis, failures)

    print("\n" + "=" * 100)
    if failures:
        print(f"RESULT: {len(failures)} check(s) FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("RESULT: all diagnosis checks PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
