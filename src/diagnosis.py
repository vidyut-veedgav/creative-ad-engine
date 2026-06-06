"""Layer 2 — Diagnosis Layer.

Explains *why* each theme is in the state Layer 1 reported, by drilling into its
ad-level patterns. For every theme (not just the declining ones — rising themes
need this as the evidence base for amplification) it identifies:

- ``leading_ads`` / ``lagging_ads`` — the theme's best and worst performers,
- ``outlier_ads`` — ads diverging from the theme cluster (only meaningful, and
  only populated, when the theme's signal is isolated/mixed),
- ``dominant_format`` / ``dominant_placement`` — what's driving the theme,
- ``weak_format`` / ``weak_placement`` — where it underperforms its own average,
- ``pattern_summary`` — one deterministic, metric-grounded sentence injected
  verbatim into the agent's context (NOT LLM-generated, so it is verifiable).

All logic is deterministic Python — no LLM. Everything is read from Layer 1's
``ad_signals_df``; no Layer 1 metric is recomputed.

**Ranking axis — a deliberate divergence from the spec's literal wording.**
The spec describes leading/lagging and dominant/weak "by CPI delta". But the
7-day CPI delta is dominated by the dataset's weekly noise: ranking by it makes
the cheapest ads (e.g. Connection's $0.52/$0.67 reels videos) surface as
*laggards*, which is incoherent. Layer 1 already established CPI **level**
(``last7d_mean_cpi``) as the noise-robust performance axis (see ``src/signal.py``
docstring). Layer 2 inherits that precedent: it ranks by level, still reports the
7-day deltas in every entry so trajectory is never lost, and lets
``pattern_summary`` contrast level against trajectory — which is what preserves
the "cheapest but deteriorating" story for a declining theme like Safety.
"""

from __future__ import annotations

import math

import pandas as pd

from util.stats import weighted_nanmean

# ── Constants ─────────────────────────────────────────────────────────────
LEADING_N = 2                       # 2 best + 2 worst (N=3 would leave no middle of 6)
MATERIAL_GAP_PCT = 15.0             # relative %, on CPI LEVEL (mirrors signal.OUTLIER_PCT)
RANK_METRIC = "last7d_mean_cpi"     # rank on current CPI level, not delta
DISTINGUISHING_ATTRS = ("format", "placement", "platform")

# Outlier ads are only meaningful where the theme signal isn't a single story.
_OUTLIER_UNIFORMITIES = ("isolated", "mixed")


# ── Small formatting / rounding helpers ───────────────────────────────────

def _round(value: float | None, places: int = 2) -> float | None:
    """Round, mapping None/NaN to None (keeps spec-typed optional floats clean)."""
    if value is None:
        return None
    try:
        if math.isnan(value):
            return None
    except TypeError:
        return value
    return round(float(value), places)


def _money(value: float | None) -> str:
    """Format a CPI level as a dollar figure."""
    return "n/a" if value is None else f"${value:.2f}"


def _pct(value: float | None) -> str:
    """Format a percentage delta with an explicit sign."""
    return "n/a" if value is None else f"{value:+.0f}%"


# ── Ad / group helpers (operate on one theme's ad-signal rows) ─────────────

def _ad_entry(rec: dict, features: list[str] | None = None) -> dict:
    """Build a spec contract entry from an ad-signal record.

    ``features`` is included (as ``distinguishing_features``) only for outliers.
    """
    entry = {
        "ad_id": rec["ad_id"],
        "format": rec["format"],
        "placement": rec["placement"],
        "cpi_7d_delta_pct": _round(rec["cpi_7d_delta_pct"]),
        "ctr_7d_delta_pct": _round(rec["ctr_7d_delta_pct"]),
    }
    if features is not None:
        entry["distinguishing_features"] = features
    return entry


def _rank_by_level(theme_rows: pd.DataFrame, n: int) -> tuple[list[dict], list[dict]]:
    """Return (leading, lagging) ad entries by current CPI level.

    Leading = the ``n`` cheapest ads (best-first); lagging = the ``n`` most
    expensive (worst-first). ``ad_id`` breaks ties for determinism.
    """
    ordered = theme_rows.sort_values([RANK_METRIC, "ad_id"])
    leading = [_ad_entry(r) for r in ordered.head(n).to_dict("records")]
    lagging = [_ad_entry(r) for r in ordered.tail(n).iloc[::-1].to_dict("records")]
    return leading, lagging


def _group_mean_cpi(theme_rows: pd.DataFrame, dim: str) -> dict[str, float]:
    """Mean current CPI level per value of ``dim`` (format/placement)."""
    means: dict[str, float] = {}
    for value, group in theme_rows.groupby(dim):
        mean = weighted_nanmean(group[RANK_METRIC].values)
        if mean is not None:
            means[value] = mean
    return means


def _dominant_and_weak(theme_rows: pd.DataFrame, dim: str) -> tuple[str | None, str | None]:
    """Find the dominant and weak value of ``dim`` by material CPI-level gap.

    A value is dominant if its mean CPI is >15% *below* the theme mean (cheaper =
    driving performance) and weak if >15% *above* it. The most extreme qualifying
    value on each side wins; either may be ``None``.
    """
    theme_mean = weighted_nanmean(theme_rows[RANK_METRIC].values)
    if not theme_mean:
        return None, None
    dominant = weak = None
    best_gap = -MATERIAL_GAP_PCT     # dominant must beat this (more negative)
    worst_gap = MATERIAL_GAP_PCT     # weak must beat this (more positive)
    for value, mean in _group_mean_cpi(theme_rows, dim).items():
        gap = (mean - theme_mean) / theme_mean * 100.0
        if gap < best_gap:
            best_gap, dominant = gap, value
        if gap > worst_gap:
            worst_gap, weak = gap, value
    return dominant, weak


def _outlier_rows(theme_rows: pd.DataFrame, uniformity: str) -> pd.DataFrame:
    """Outlier ads for the theme, most extreme first.

    Empty unless the theme is isolated/mixed (per spec, outliers are only
    meaningful then). Reuses Layer 1's level-based ``is_outlier`` flag — a
    direction-based "against the modal trend" test misses level winners whose
    momentum reads flat (e.g. Connection's stable-but-cheapest reels videos).
    """
    if uniformity not in _OUTLIER_UNIFORMITIES:
        return theme_rows.iloc[0:0]
    outliers = theme_rows[theme_rows["is_outlier"]].copy()
    outliers["_extremity"] = outliers["cpi_vs_theme_median_pct"].abs()
    return outliers.sort_values(["_extremity", "ad_id"], ascending=[False, True])


def _distinguishing_features(theme_rows: pd.DataFrame, outlier_ids: list[str]) -> dict[str, list[str]]:
    """Cross-tab outliers vs non-outliers; per outlier, the attribute values that
    set it apart.

    A value is distinguishing if it appears among the outliers but not among the
    non-outliers (a clean signal for this small dataset). Falls back to
    over-representation (higher share among outliers) if the strict rule yields
    nothing for an ad.
    """
    if not outlier_ids:
        return {}
    outlier_mask = theme_rows["ad_id"].isin(outlier_ids)
    outliers = theme_rows[outlier_mask]
    non_outliers = theme_rows[~outlier_mask]
    n_out = len(outliers)
    n_non = len(non_outliers)

    features: dict[str, list[str]] = {rec["ad_id"]: [] for rec in outliers.to_dict("records")}
    for attr in DISTINGUISHING_ATTRS:
        non_values = set(non_outliers[attr])
        non_counts = non_outliers[attr].value_counts().to_dict()
        out_counts = outliers[attr].value_counts().to_dict()
        for rec in outliers.to_dict("records"):
            value = rec[attr]
            unique = value not in non_values
            over = n_non and (out_counts.get(value, 0) / n_out) > (non_counts.get(value, 0) / n_non)
            if unique or over:
                features[rec["ad_id"]].append(f"{attr}:{value}")
    return features


# ── Pattern summary (deterministic, template-assembled) ────────────────────

def _drivers_phrase(facts: dict) -> str:
    """Render '<format> on <placement>' (or just the format) for the dominant pair."""
    fmt = facts.get("dominant_format")
    plc = facts.get("dominant_placement")
    if fmt and plc:
        return f"{fmt} on {plc}"
    return fmt or "no single format"


def _collect_facts(
    theme: str, theme_rows: pd.DataFrame, theme_sig: dict, outlier_df: pd.DataFrame,
    dominant_format: str | None, weak_format: str | None,
    dominant_placement: str | None, weak_placement: str | None,
) -> dict:
    """Gather every number the pattern-summary templates might slot in."""
    levels = theme_rows[RANK_METRIC]
    fmt_means = _group_mean_cpi(theme_rows, "format")
    videos = theme_rows[theme_rows["format"] == "video"]
    winners = outlier_df[outlier_df["cpi_vs_theme_median_pct"] < 0].head(2)
    return {
        "n_ads": len(theme_rows),
        "ads_rising": theme_sig["ads_rising"],
        "ads_declining": theme_sig["ads_declining"],
        "ads_stable": theme_sig["ads_stable"],
        "cpi_delta": theme_sig["cpi_7d_delta_pct"],
        "hold_delta": theme_sig["hold_rate_7d_delta_pct"],
        "theme_mean_cpi": weighted_nanmean(levels.values),
        "theme_median_cpi": float(levels.median()),
        "dominant_format": dominant_format,
        "weak_format": weak_format,
        "dominant_placement": dominant_placement,
        "weak_placement": weak_placement,
        "dominant_format_cpi": fmt_means.get(dominant_format) if dominant_format else None,
        "video_mean_cpi": weighted_nanmean(videos[RANK_METRIC].values) if len(videos) else None,
        "video_mean_cpi_delta": weighted_nanmean(videos["cpi_7d_delta_pct"].values) if len(videos) else None,
        "winner_ids": sorted(winners["ad_id"].tolist()),
        "winner_mean_cpi": weighted_nanmean(winners[RANK_METRIC].values) if len(winners) else None,
        "winner_below_pct": weighted_nanmean(winners["cpi_vs_theme_median_pct"].values) if len(winners) else None,
    }


def _pattern_summary(theme: str, theme_sig: dict, f: dict) -> str:
    """Assemble the one-sentence, metric-grounded summary for a theme.

    Selected by ``hypothesis_type`` (which already encodes direction + uniformity
    + ceiling), with the two ``optimization`` cases split by ``efficiency_ceiling``.
    Every number comes from ``f``/``theme_sig`` — nothing is hard-coded.
    """
    htype = theme_sig["hypothesis_type"]
    cap = theme.capitalize()
    drivers = _drivers_phrase(f)

    if htype == "optimization" and theme_sig["efficiency_ceiling"]:
        return (
            f"{cap} has reached its efficiency ceiling: CPI is flat ({_pct(f['cpi_delta'])} "
            f"wk/wk) despite rising spend, with {drivers} leading at "
            f"{_money(f['dominant_format_cpi'])} vs {_money(f['theme_mean_cpi'])} theme mean; "
            f"optimize creative rather than scale spend."
        )

    if htype == "retirement":
        hold = f", hold-rate {_pct(f['hold_delta'])}" if f["hold_delta"] is not None else ""
        return (
            f"All {f['n_ads']} {theme} ads are declining (theme CPI {_pct(f['cpi_delta'])} wk/wk); "
            f"video stays cheapest at {_money(f['video_mean_cpi'])} vs {_money(f['theme_mean_cpi'])} "
            f"theme mean but is deteriorating fastest (CPI {_pct(f['video_mean_cpi_delta'])}{hold}), "
            f"pointing to mid-video drop-off; retire or rebuild rather than optimize."
        )

    if htype == "replication":
        ids = ", ".join(f["winner_ids"])
        below = abs(f["winner_below_pct"]) if f["winner_below_pct"] is not None else 0.0
        return (
            f"{cap} is volatile and execution-level: {len(f['winner_ids'])} {drivers} ads ({ids}) "
            f"run ~{below:.0f}% below the theme median CPI ({_money(f['winner_mean_cpi'])} vs "
            f"{_money(f['theme_median_cpi'])}) while the rest cluster higher; replicate the {drivers} formula."
        )

    if htype == "intervention":
        ids = ", ".join(f["winner_ids"]) or "specific ads"
        return (
            f"{cap} is execution-level with isolated laggards ({ids}) running well above the theme's "
            f"CPI; fix their format/placement rather than the theme."
        )

    if htype == "amplification":
        return (
            f"{f['ads_rising']} of {f['n_ads']} {theme} ads are rising uniformly "
            f"(CPI {_pct(f['cpi_delta'])} wk/wk); {drivers} is driving the gain at "
            f"{_money(f['dominant_format_cpi'])} vs {_money(f['theme_mean_cpi'])} theme mean; "
            f"scale spend and expand placements."
        )

    # optimization without a ceiling == stable theme: monitor.
    weak = ""
    if f["weak_format"]:
        where = f" on {f['weak_placement']}" if f["weak_placement"] else ""
        weak = f" while {f['weak_format']}{where} is most expensive"
    return (
        f"{cap} is flat with no theme-level trend (CPI {_pct(f['cpi_delta'])} wk/wk); "
        f"{drivers} runs cheapest at {_money(f['dominant_format_cpi'])} vs "
        f"{_money(f['theme_mean_cpi'])} theme mean{weak}; monitor, no structural change indicated."
    )


# ── Per-theme assembly + orchestrator ──────────────────────────────────────

def _diagnose_theme(theme: str, theme_rows: pd.DataFrame, theme_sig: dict) -> dict:
    """Build the diagnosis dict for a single theme."""
    n = min(LEADING_N, len(theme_rows) // 2)
    leading, lagging = _rank_by_level(theme_rows, n)

    dominant_format, weak_format = _dominant_and_weak(theme_rows, "format")
    dominant_placement, weak_placement = _dominant_and_weak(theme_rows, "placement")

    outlier_df = _outlier_rows(theme_rows, theme_sig["uniformity"])
    outlier_ids = outlier_df["ad_id"].tolist()
    features = _distinguishing_features(theme_rows, outlier_ids)
    outlier_ads = [
        _ad_entry(rec, features.get(rec["ad_id"], [])) for rec in outlier_df.to_dict("records")
    ]

    facts = _collect_facts(
        theme, theme_rows, theme_sig, outlier_df,
        dominant_format, weak_format, dominant_placement, weak_placement,
    )
    return {
        "leading_ads": leading,
        "lagging_ads": lagging,
        "outlier_ads": outlier_ads,
        "dominant_format": dominant_format,
        "dominant_placement": dominant_placement,
        "weak_format": weak_format,
        "weak_placement": weak_placement,
        "pattern_summary": _pattern_summary(theme, theme_sig, facts),
    }


def build_theme_diagnosis(
    theme_signals: dict, ad_signals_df: pd.DataFrame, ads_df: pd.DataFrame
) -> dict:
    """Build the ``theme_diagnosis`` dict for all themes.

    ``ads_df`` is part of the spec contract but functionally unused here: Layer 1
    already merged ``format``/``placement``/``platform`` onto ``ad_signals_df``,
    and the remaining ``ads.csv`` columns are collinear with format/placement or
    are high-cardinality text. It is accepted for contract stability and future use.
    """
    theme_diagnosis: dict = {}
    for theme in sorted(theme_signals):
        theme_rows = ad_signals_df[ad_signals_df["theme"] == theme]
        theme_diagnosis[theme] = _diagnose_theme(theme, theme_rows, theme_signals[theme])
    return theme_diagnosis
