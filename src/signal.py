"""Layer 1 — Signal Layer.

Converts the raw engagement/ads CSVs into trend *evidence*, at two levels:

- **Ad level** (one row per ad in ``ad_signals_df``): 7-day deltas for the core
  metrics, absolute current levels, 30-day trend slopes, a spend-gated
  efficiency-ceiling flag, and a trend direction/strength classification.
- **Theme level** (``theme_signals`` dict): spend-weighted deltas, direction
  counts, uniformity, the diagnosis level, an efficiency-ceiling verdict, and
  the derived hypothesis type.

All logic here is deterministic Python — no LLM. The signal layer reads the
*stored* CPI/CTR/CVR columns and computes deltas from them; it never recomputes
those metrics from spend/clicks/installs (per the architecture spec). Every
downstream layer consumes these artifacts, so the public entry point
``build_theme_signals`` returns the four objects they need together to avoid
recomputation.

Design notes worth knowing when reading the code:

- The "7-day delta" compares the mean of the last 7 daily rows against the mean
  of the previous 7. A full-week mean cancels the dataset's deliberate period-7
  weekly volatility (notably for the ``connection`` theme), so adjacent-week
  comparison *suppresses* that noise rather than aliasing to it.
- ``efficiency_ceiling`` is gated on **spend**: CPI no longer improving *while
  spend is materially rising* (and was improving earlier). Only the ``wonder``
  theme ramps budget in the tail, which cleanly separates a true ceiling from a
  merely flat theme (``confidence``) or a declining one (``safety``).
- The reported theme ``trend_direction`` is a *right-now* (7-day) read. A
  long-run winner that has plateaued (``wonder``) may read ``stable`` here; its
  strength as the best performer is carried by absolute levels in
  ``ad_signals_df`` (lowest current CPI), not by forcing the direction.
"""

from __future__ import annotations

import pandas as pd

from util.data_loader import load_dataset
from util.stats import normalized_slope, pct_change, weighted_nanmean

# ── Constants / thresholds ────────────────────────────────────────────────
# Window sizes (in daily rows).
SHORT_WINDOW = 7          # "current" week
PRIOR_WINDOW = 7          # comparison week immediately before it
LONG_WINDOW = 30          # rolling window for trend slopes / ceiling

# Trend classification thresholds on the CPI 7-day delta (% change).
# Spec: >15% change = strong, 5–15% = moderate, <5% = weak/stable.
STRONG_PCT = 15.0
MODERATE_PCT = 5.0

# Uniformity thresholds on the dominant-direction share of a theme's ads.
# Spec: >60% same direction = uniform, <40% = isolated, 40–60% = mixed.
UNIFORM_SHARE = 0.60
ISOLATED_SHARE = 0.40

# Uniformity is also judged by how *concentrated* performance is across a theme's
# ads — the coefficient of variation (std/mean) of current CPI level. The 7-day
# direction counts are dominated by the dataset's weekly noise (every theme's
# delta-std is similar), so they can't tell a volatile, outlier-driven theme from
# a flat one; level dispersion can. High CV → performance is isolated to a few
# ads (execution-level); low CV → the ads behave alike (strategy-level). Tuned so
# the lone volatile theme (CV ~0.70) separates from the homogeneous ones (~0.3–0.4).
ISOLATED_CV = 0.55
UNIFORM_CV = 0.45

# Efficiency-ceiling gates (normalized OLS slopes, in %/day over LONG_WINDOW).
CEILING_CPI_SLOPE_MAX = -0.10    # CPI improving slower than this = "flattened"
CEILING_SPEND_SLOPE_MIN = 0.40   # spend rising faster than this = "materially up"

# An ad whose current CPI level is more than this % from its theme median is a
# theme-relative outlier (mirrors the diagnosis layer's material threshold).
OUTLIER_PCT = 15.0

# Metrics that only exist for video ads (null for static/carousel).
VIDEO_METRICS = ("hold_rate", "thumbstop_rate")


# ── Ad-level helpers ──────────────────────────────────────────────────────
# Each takes a single ad's engagement slice (chronologically sorted) and
# returns a scalar or classification string.

def _window_delta_pct(
    ad_eng: pd.DataFrame, metric: str, short: int = SHORT_WINDOW, prior: int = PRIOR_WINDOW
) -> float | None:
    """Percentage change of ``metric``: mean(last ``short``) vs mean(prior ``prior``).

    Nan-aware. Returns ``None`` for metrics that don't apply to this ad's format
    (all-NaN column) or when there isn't enough history.
    """
    series = ad_eng[metric]
    recent = series.iloc[-short:]
    earlier = series.iloc[-(short + prior):-short]
    return pct_change(weighted_nanmean(recent.values), weighted_nanmean(earlier.values))


def _window_mean(ad_eng: pd.DataFrame, metric: str, n: int = SHORT_WINDOW) -> float | None:
    """Nan-aware mean of ``metric`` over the last ``n`` rows (absolute level)."""
    return weighted_nanmean(ad_eng[metric].iloc[-n:].values)


def _window_sum(ad_eng: pd.DataFrame, metric: str, n: int = SHORT_WINDOW) -> float:
    """Sum of ``metric`` over the last ``n`` rows (used for spend weighting)."""
    return float(ad_eng[metric].iloc[-n:].sum())


def _norm_slope(
    ad_eng: pd.DataFrame, metric: str, n: int = LONG_WINDOW, offset: int = 0
) -> float | None:
    """Normalized OLS slope (%/day) of ``metric`` over an ``n``-row window.

    ``offset=0`` is the most recent window; ``offset=n`` is the window before it
    (used to confirm CPI *was* improving prior to a suspected ceiling).
    """
    stop = len(ad_eng) - offset
    start = max(0, stop - n)
    window = ad_eng.iloc[start:stop]
    if len(window) < 2:
        return None
    return normalized_slope(window[metric].values, window["day_index"].values)


def _classify_direction(cpi_delta_pct: float | None) -> str:
    """Trend direction from the CPI 7-day delta. Negative CPI delta = improving."""
    if cpi_delta_pct is None or abs(cpi_delta_pct) < MODERATE_PCT:
        return "stable"
    return "rising" if cpi_delta_pct < 0 else "declining"


def _classify_strength(cpi_delta_pct: float | None) -> str:
    """Trend strength from the magnitude of the CPI 7-day delta."""
    if cpi_delta_pct is None:
        return "weak"
    magnitude = abs(cpi_delta_pct)
    if magnitude > STRONG_PCT:
        return "strong"
    if magnitude >= MODERATE_PCT:
        return "moderate"
    return "weak"


def _detect_ceiling(ad_eng: pd.DataFrame) -> bool:
    """Spend-gated efficiency ceiling for a single ad.

    True when, over the last 30 days, CPI is no longer meaningfully improving
    while spend is materially rising, *and* CPI had been improving in the prior
    30 days. The spend gate is load-bearing: it distinguishes a real ceiling
    (wonder, whose budget ramps) from a merely flat theme (confidence).
    """
    cpi_slope = _norm_slope(ad_eng, "CPI", LONG_WINDOW)
    spend_slope = _norm_slope(ad_eng, "spend", LONG_WINDOW)
    prior_cpi_slope = _norm_slope(ad_eng, "CPI", LONG_WINDOW, offset=LONG_WINDOW)
    if cpi_slope is None or spend_slope is None or prior_cpi_slope is None:
        return False
    return (
        cpi_slope > CEILING_CPI_SLOPE_MAX            # CPI improvement has flattened
        and spend_slope > CEILING_SPEND_SLOPE_MIN    # despite materially rising spend
        and prior_cpi_slope < CEILING_CPI_SLOPE_MAX  # and it was improving before
    )


def compute_ad_signals(eng_df: pd.DataFrame, ads_df: pd.DataFrame) -> pd.DataFrame:
    """Compute the ad-level signal row for every ad.

    Returns one row per ad: the spec's 7-day deltas plus enrichment columns
    (absolute current levels, 30-day slopes, direction/strength/ceiling, and a
    theme-relative outlier flag) that downstream layers use to answer "best
    right now" and "standout ads" questions the deltas alone can't.
    """
    rows = []
    for ad_id, group in eng_df.groupby("ad_id", sort=True):
        group = group.sort_values("date")
        record = {
            "ad_id": ad_id,
            "theme": group["theme"].iloc[0],
            "format": group["format"].iloc[0],
            "placement": group["placement"].iloc[0],
            "platform": group["platform"].iloc[0],
            # Spec 7-day deltas (negative CPI = improving).
            "cpi_7d_delta_pct": _window_delta_pct(group, "CPI"),
            "ctr_7d_delta_pct": _window_delta_pct(group, "CTR"),
            "cvr_7d_delta_pct": _window_delta_pct(group, "CVR"),
            "hold_rate_7d_delta_pct": _window_delta_pct(group, "hold_rate"),
            "thumbstop_rate_7d_delta_pct": _window_delta_pct(group, "thumbstop_rate"),
            # Enrichment: absolute current-window levels + spend weight.
            "last7d_mean_cpi": _window_mean(group, "CPI"),
            "last7d_mean_ctr": _window_mean(group, "CTR"),
            "last7d_mean_cvr": _window_mean(group, "CVR"),
            "recent_spend": _window_sum(group, "spend"),
            # Enrichment: 30-day trend slopes + ceiling.
            "cpi_30d_norm_slope": _norm_slope(group, "CPI", LONG_WINDOW),
            "spend_30d_norm_slope": _norm_slope(group, "spend", LONG_WINDOW),
            "cpi_prior30d_norm_slope": _norm_slope(group, "CPI", LONG_WINDOW, offset=LONG_WINDOW),
            "ceiling": _detect_ceiling(group),
        }
        record["direction"] = _classify_direction(record["cpi_7d_delta_pct"])
        record["strength"] = _classify_strength(record["cpi_7d_delta_pct"])
        rows.append(record)

    ad_signals_df = pd.DataFrame(rows)

    # Theme-relative outlier flag, on absolute current CPI level. Negative pct =
    # cheaper than the theme median (a positive outlier, e.g. connection's reels
    # video winners); positive = more expensive.
    theme_median = ad_signals_df.groupby("theme")["last7d_mean_cpi"].transform("median")
    ad_signals_df["cpi_vs_theme_median_pct"] = (
        (ad_signals_df["last7d_mean_cpi"] - theme_median) / theme_median * 100.0
    )
    ad_signals_df["is_outlier"] = ad_signals_df["cpi_vs_theme_median_pct"].abs() > OUTLIER_PCT
    return ad_signals_df


# ── Theme-level helpers ───────────────────────────────────────────────────
# Each takes the ad-signal rows for one theme and returns a scalar/string.

def _weighted_delta(theme_rows: pd.DataFrame, col: str, weight_col: str = "recent_spend") -> float | None:
    """Spend-weighted, nan-aware mean of an ad-level delta column."""
    return weighted_nanmean(theme_rows[col].values, theme_rows[weight_col].values)


def _direction_counts(theme_rows: pd.DataFrame) -> tuple[int, int, int]:
    """Counts of (rising, declining, stable) ad directions in the theme."""
    counts = theme_rows["direction"].value_counts()
    return (
        int(counts.get("rising", 0)),
        int(counts.get("declining", 0)),
        int(counts.get("stable", 0)),
    )


def _classify_uniformity(
    counts: tuple[int, int, int], n_ads: int, weighted_dir: str, level_cv: float | None
) -> str:
    """Classify a theme's uniformity.

    Leads with performance concentration (CV of current CPI level), which is
    robust to the dataset's weekly noise, then falls back to direction
    agreement:
    - High level CV -> ``isolated``: a few ads carry (or drag) the theme.
    - A decisive directional consensus -> ``uniform``.
    - A theme whose blended trend is flat with low concentration is telling one
      (flat) story -> ``uniform`` (rather than letting 7-day sign noise fragment
      it into ``mixed``).
    - Otherwise ``mixed``.
    """
    if level_cv is not None and level_cv >= ISOLATED_CV:
        return "isolated"
    dominant_share = max(counts) / n_ads if n_ads else 0.0
    if dominant_share > UNIFORM_SHARE:
        return "uniform"
    if weighted_dir == "stable" and (level_cv is None or level_cv <= UNIFORM_CV):
        return "uniform"
    if dominant_share < ISOLATED_SHARE:
        return "isolated"
    return "mixed"


def _theme_direction(counts: tuple[int, int, int], weighted_cpi_delta: float | None) -> str:
    """Theme direction from the spend-weighted CPI delta (the theme's blended
    trend), reporting ``stable`` if the ad-level plurality contradicts it.

    The weighted mean is preferred over a raw modal count because the 7-day
    direction counts are noisy around zero for the flat/volatile themes; the
    budget-weighted delta is the theme's actual blended movement.
    """
    weighted_dir = _classify_direction(weighted_cpi_delta)
    if weighted_dir == "stable":
        return "stable"
    rising, declining, _ = counts
    if rising > declining:
        modal = "rising"
    elif declining > rising:
        modal = "declining"
    else:
        modal = weighted_dir
    # Only honour a decisive weighted move if the ad plurality doesn't oppose it.
    return weighted_dir if modal in (weighted_dir, "stable") else "stable"


def _theme_strength(weighted_cpi_delta: float | None) -> str:
    """Theme strength from the magnitude of the spend-weighted CPI delta."""
    return _classify_strength(weighted_cpi_delta)


def _theme_ceiling(theme_rows: pd.DataFrame) -> bool:
    """True when a majority of the theme's ads show the ceiling signal."""
    n_ads = len(theme_rows)
    return int(theme_rows["ceiling"].sum()) * 2 > n_ads


def _diagnosis_level(uniformity: str) -> str:
    """uniform → strategy-level; isolated/mixed → execution-level."""
    return "strategy" if uniformity == "uniform" else "execution"


def _hypothesis_type(
    uniformity: str, direction: str, ceiling: bool, outlier_is_winner: bool
) -> str:
    """Derive the hypothesis type from ceiling + uniformity + direction.

    Precedence:
    1. An efficiency ceiling -> ``optimization`` (refresh/optimize rather than
       scale or kill), whatever the noisy 7-day direction reads.
    2. Execution-level (``isolated``) themes are keyed on the standout ad: if the
       most extreme outlier is a *winner* (cheaper than the theme), replicate it;
       otherwise intervene on the laggard — the spec's "isolated + rising/declining
       outliers" rows, judged on level rather than a volatile theme's modal trend.
    3. Strategy-level (``uniform``/``mixed``): declining -> retirement,
       rising -> amplification, stable -> optimization (monitor).
    """
    if ceiling:
        return "optimization"
    if uniformity == "isolated":
        return "replication" if outlier_is_winner else "intervention"
    if direction == "declining":
        return "retirement"
    if direction == "rising":
        return "amplification"
    return "optimization"  # stable / mixed


def _round(value: float | None, places: int = 2) -> float | None:
    """Round, preserving ``None`` (so the dict keeps spec-typed optional floats)."""
    return None if value is None else round(value, places)


def aggregate_theme_signals(ad_signals_df: pd.DataFrame) -> dict:
    """Build the ``theme_signals`` dict (the spec output contract) from ad rows."""
    theme_signals: dict = {}
    for theme, group in ad_signals_df.groupby("theme", sort=True):
        n_ads = len(group)
        counts = _direction_counts(group)
        ads_rising, ads_declining, ads_stable = counts

        cpi_delta = _weighted_delta(group, "cpi_7d_delta_pct")
        ctr_delta = _weighted_delta(group, "ctr_7d_delta_pct")
        cvr_delta = _weighted_delta(group, "cvr_7d_delta_pct")

        # Video-only deltas aggregate over the theme's video ads alone.
        videos = group[group["format"] == "video"]
        hold_delta = _weighted_delta(videos, "hold_rate_7d_delta_pct") if len(videos) else None
        thumb_delta = _weighted_delta(videos, "thumbstop_rate_7d_delta_pct") if len(videos) else None

        # Concentration of performance across the theme's ads (CV of current CPI
        # level) drives the uniformity classification — robust to weekly noise.
        levels = group["last7d_mean_cpi"]
        level_mean = weighted_nanmean(levels.values)
        level_cv = (
            float(levels.std()) / level_mean
            if level_mean and level_mean > 0 and levels.notna().sum() > 1
            else None
        )
        # The theme's most extreme outlier (by current CPI vs theme median): a
        # winner (cheaper) implies replicate, a laggard implies intervene.
        extreme_idx = group["cpi_vs_theme_median_pct"].abs().idxmax()
        outlier_is_winner = bool(group.loc[extreme_idx, "cpi_vs_theme_median_pct"] < 0)

        direction = _theme_direction(counts, cpi_delta)
        strength = _theme_strength(cpi_delta)
        uniformity = _classify_uniformity(counts, n_ads, direction, level_cv)
        ceiling = _theme_ceiling(group)

        theme_signals[theme] = {
            "trend_direction": direction,
            "trend_strength": strength,
            "uniformity": uniformity,
            "cpi_7d_delta_pct": _round(cpi_delta),
            "ctr_7d_delta_pct": _round(ctr_delta),
            "cvr_7d_delta_pct": _round(cvr_delta),
            "hold_rate_7d_delta_pct": _round(hold_delta),
            "thumbstop_rate_7d_delta_pct": _round(thumb_delta),
            "ads_rising": ads_rising,
            "ads_declining": ads_declining,
            "ads_stable": ads_stable,
            "diagnosis_level": _diagnosis_level(uniformity),
            "hypothesis_type": _hypothesis_type(
                uniformity, direction, ceiling, outlier_is_winner
            ),
            "efficiency_ceiling": ceiling,
        }
    return theme_signals


# ── Public orchestrator ───────────────────────────────────────────────────

def build_theme_signals(
    ads_path: str, engagement_path: str
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the full signal layer.

    Returns ``(theme_signals, ads_df, eng_df, ad_signals_df)``. All four are
    returned together because downstream layers need the raw frames and the
    ad-level signals without recomputing them.
    """
    ads_df, eng_df = load_dataset(ads_path, engagement_path)
    ad_signals_df = compute_ad_signals(eng_df, ads_df)
    theme_signals = aggregate_theme_signals(ad_signals_df)
    return theme_signals, ads_df, eng_df, ad_signals_df
