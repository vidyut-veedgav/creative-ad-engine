"""CSV loading for the Giant Creative Ad Engine.

This is the single place that knows the quirks of ``data/ads.csv`` and
``data/engagement.csv`` so the rest of the pipeline can assume clean, typed
frames:

- Missing metrics are encoded as **empty strings** (video-only, carousel-only,
  and Meta-social columns are blank for the formats they don't apply to). We
  coerce every numeric column with ``errors="coerce"`` so blanks become NaN and
  downstream nan-aware math just works.
- ``date`` is ISO ``YYYY-MM-DD``; we parse it to datetime and sort by
  ``(ad_id, date)`` so per-ad ``.tail(n)`` windows are chronological.
- A per-ad ``day_index`` (0..89) is added as a clean integer x-axis for slope
  fits, independent of calendar gaps.
"""

from __future__ import annotations

import pandas as pd

# Numeric columns in engagement.csv. Everything else (ad_id, date) is left as-is.
# Listed explicitly rather than inferred so a malformed cell fails loudly here
# rather than silently poisoning a metric downstream.
_ENGAGEMENT_NUMERIC = [
    "spend", "impressions", "clicks", "installs",
    "CTR", "CPI", "CPM", "CPC", "CVR",
    "thumbstop_rate", "hold_rate", "completion_rate",
    "card_swipe_rate", "last_card_reach_rate",
    "video_plays_25pct", "video_plays_75pct", "avg_play_time_sec",
    "likes", "comments", "shares", "saves",
]

# ads.csv columns carried onto engagement rows for grouping/diagnosis.
_AD_DIMENSIONS = ["ad_id", "theme", "format", "placement", "platform"]


def load_ads(path: str) -> pd.DataFrame:
    """Load ads.csv (one row per ad, static metadata)."""
    ads = pd.read_csv(path, dtype=str)
    # video_length_sec is the only numeric ads column; blank for non-video.
    if "video_length_sec" in ads.columns:
        ads["video_length_sec"] = pd.to_numeric(
            ads["video_length_sec"], errors="coerce"
        )
    return ads


def load_engagement(path: str) -> pd.DataFrame:
    """Load engagement.csv: parse dates, coerce numerics, sort, add day_index."""
    eng = pd.read_csv(path, parse_dates=["date"])
    for col in _ENGAGEMENT_NUMERIC:
        if col in eng.columns:
            eng[col] = pd.to_numeric(eng[col], errors="coerce")
    eng = eng.sort_values(["ad_id", "date"]).reset_index(drop=True)
    # Per-ad chronological index (0-based) for OLS slope x-axes.
    eng["day_index"] = eng.groupby("ad_id").cumcount()
    return eng


def load_dataset(
    ads_path: str, engagement_path: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load both tables and attach ad dimensions to each engagement row.

    Returns ``(ads_df, eng_df)`` where ``eng_df`` has ``theme``/``format``/
    ``placement``/``platform`` merged in so layers can group engagement by any
    creative dimension without re-joining.
    """
    ads_df = load_ads(ads_path)
    eng_df = load_engagement(engagement_path)
    dims = ads_df[[c for c in _AD_DIMENSIONS if c in ads_df.columns]]
    eng_df = eng_df.merge(dims, on="ad_id", how="left")
    return ads_df, eng_df
