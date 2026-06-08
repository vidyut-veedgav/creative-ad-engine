"""Validation harness for the signal layer (Layer 1).

There is no unit-test suite yet, so this script is the executable check that the
signal layer reproduces the four narratives deliberately baked into the dataset
by ``util/generate_dataset.py``. It prints the raw per-ad and per-theme numbers
first (so thresholds can be tuned against reality), then asserts the narratives.

Run from the repo root with the project venv::

    python -m util.verify_signals

Exits non-zero if any narrative assertion fails.
"""

from __future__ import annotations

import os
import sys

import pandas as pd

# Allow running as a plain script (``python util/verify_signals.py``) as well as
# a module, by ensuring the repo root is importable.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.signal import build_theme_signals  # noqa: E402

ADS_PATH = os.path.join(_REPO_ROOT, "data", "ads.csv")
ENG_PATH = os.path.join(_REPO_ROOT, "data", "engagement.csv")

# Columns to surface per ad when printing the evidence table.
_AD_COLS = [
    "ad_id", "theme", "format", "placement",
    "cpi_7d_delta_pct", "cpi_30d_norm_slope", "cpi_prior30d_norm_slope",
    "spend_30d_norm_slope", "last7d_mean_cpi", "direction", "ceiling",
    "is_outlier", "cpi_vs_theme_median_pct",
]


def _print_evidence(theme_signals: dict, ad_signals_df: pd.DataFrame) -> None:
    """Print the raw ad-level and theme-level numbers behind the verdicts."""
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)

    print("=" * 100)
    print("AD-LEVEL SIGNALS")
    print("=" * 100)
    print(ad_signals_df[_AD_COLS].round(2).to_string(index=False))

    levels = ad_signals_df.groupby("theme")["last7d_mean_cpi"]
    print("\nWithin-theme CPI level CV (std/mean) - drives uniformity:")
    print((levels.std() / levels.mean()).round(3).to_string())

    print("\n" + "=" * 100)
    print("THEME-LEVEL SIGNALS")
    print("=" * 100)
    for theme in ("wonder", "safety", "confidence", "connection"):
        v = theme_signals[theme]
        print(
            f"{theme:11s} dir={v['trend_direction']:9s} str={v['trend_strength']:8s} "
            f"unif={v['uniformity']:8s} hyp={v['hypothesis_type']:13s} "
            f"ceil={str(v['efficiency_ceiling']):5s} diag={v['diagnosis_level']:9s} "
            f"cpi_delta={v['cpi_7d_delta_pct']} "
            f"R/D/S={v['ads_rising']}/{v['ads_declining']}/{v['ads_stable']}"
        )


def _check(label: str, condition: bool, failures: list[str]) -> None:
    """Record and print a single named assertion."""
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}")
    if not condition:
        failures.append(label)


def _verify_narratives(theme_signals: dict, ad: pd.DataFrame) -> list[str]:
    """Assert each theme's baked narrative. Returns the list of failed checks."""
    from src.signal import MODERATE_PCT

    failures: list[str] = []
    by_theme = {t: g for t, g in ad.groupby("theme")}
    min_cpi_theme = min(theme_signals, key=lambda t: by_theme[t]["last7d_mean_cpi"].mean())

    # ── Wonder: rising winner that has hit an efficiency ceiling. ──
    w = theme_signals["wonder"]
    wads = by_theme["wonder"]
    print("\nWONDER - expect efficiency ceiling -> optimization, best current CPI:")
    _check("efficiency_ceiling is True", w["efficiency_ceiling"] is True, failures)
    _check("hypothesis_type == optimization", w["hypothesis_type"] == "optimization", failures)
    _check("diagnosis_level == strategy", w["diagnosis_level"] == "strategy", failures)
    _check("trend_direction not declining (plateaued, not falling)",
           w["trend_direction"] in ("rising", "stable"), failures)
    _check("lowest current CPI of all themes", min_cpi_theme == "wonder", failures)
    _check("CPI was improving earlier (prior-30d slope < 0)",
           wads["cpi_prior30d_norm_slope"].mean() < 0, failures)

    # ── Safety: uniform decline from mid-flight; mid-video dropout. ──
    s = theme_signals["safety"]
    print("\nSAFETY - expect uniform decline -> retirement, hold decays faster than CTR:")
    _check("trend_direction == declining", s["trend_direction"] == "declining", failures)
    _check("cpi_7d_delta_pct > MODERATE_PCT (CPI worsening)",
           s["cpi_7d_delta_pct"] is not None and s["cpi_7d_delta_pct"] > MODERATE_PCT, failures)
    _check("uniformity == uniform", s["uniformity"] == "uniform", failures)
    _check("hypothesis_type == retirement", s["hypothesis_type"] == "retirement", failures)
    _check("efficiency_ceiling is False", s["efficiency_ceiling"] is False, failures)
    _check("hold_rate decays faster than CTR (hold_delta < ctr_delta)",
           s["hold_rate_7d_delta_pct"] is not None
           and s["hold_rate_7d_delta_pct"] < s["ctr_7d_delta_pct"], failures)

    # ── Confidence: uniform rise with no ceiling -> amplification. ──
    c = theme_signals["confidence"]
    print("\nCONFIDENCE - expect uniform rise, no ceiling -> amplification:")
    _check("trend_direction == rising", c["trend_direction"] == "rising", failures)
    _check("cpi_7d_delta_pct < -MODERATE_PCT (CPI improving materially)",
           c["cpi_7d_delta_pct"] is not None and c["cpi_7d_delta_pct"] < -MODERATE_PCT, failures)
    _check("uniformity == uniform", c["uniformity"] == "uniform", failures)
    _check("efficiency_ceiling is False (no spend ramp -> no ceiling)",
           c["efficiency_ceiling"] is False, failures)
    _check("hypothesis_type == amplification", c["hypothesis_type"] == "amplification", failures)

    # ── Connection: volatile, isolated reels-video winners to replicate. ──
    n = theme_signals["connection"]
    nads = by_theme["connection"].sort_values("last7d_mean_cpi")
    cheapest_two_formats = set(nads.head(2)["format"])
    print("\nCONNECTION - expect isolated outliers -> replication, the 2 video ads are the winners:")
    _check("uniformity == isolated", n["uniformity"] == "isolated", failures)
    _check("diagnosis_level == execution", n["diagnosis_level"] == "execution", failures)
    _check("hypothesis_type == replication", n["hypothesis_type"] == "replication", failures)
    _check("efficiency_ceiling is False", n["efficiency_ceiling"] is False, failures)
    _check("the two cheapest ads in-theme are both video",
           cheapest_two_formats == {"video"}, failures)

    return failures


def main() -> int:
    theme_signals, _ads_df, _eng_df, ad_signals_df = build_theme_signals(ADS_PATH, ENG_PATH)
    _print_evidence(theme_signals, ad_signals_df)

    print("\n" + "=" * 100)
    print("NARRATIVE ASSERTIONS")
    print("=" * 100)
    failures = _verify_narratives(theme_signals, ad_signals_df)

    print("\n" + "=" * 100)
    if failures:
        print(f"RESULT: {len(failures)} check(s) FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("RESULT: all narrative checks PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
