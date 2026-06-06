"""
generate_dataset.py
===================================================================
Synthetic ad-performance dataset generator for the Giant creative
analytics system (Giant = a personalized AI storytelling app for kids).

Outputs two CSVs:
  - ads.csv          24 ads  (1 row per ad, static metadata)
  - engagement.csv   2,160 rows (1 row per ad per day, 90-day time series)

-------------------------------------------------------------------
NARRATIVE STRUCTURE BAKED INTO THE DATA
-------------------------------------------------------------------
The dataset is NOT random noise. Four theme-level stories are encoded
as trend curves, with day-of-week + warmup + per-ad variance layered
on top, so that aggregations produce directionally meaningful signal:

  WONDER     -> The control / winner. Consistently strong and RISING, but
               with REALISTIC DIMINISHING RETURNS rather than a straight line:
                 days  1-30 (warmup):  CPI improves meaningfully as spend ramps
                 days 31-60 (scaling): CPI keeps improving, but slower
                 days 61-90 (mature):  CPI flattens and ticks back up ~8-12%
               The mature-phase worsening is SPEND-DRIVEN: Wonder's budget is
               scaled up over the window, and once daily spend crosses ~$190/day
               per ad an audience-saturation penalty drags CVR (hence CPI). So a
               spend-vs-CPI scatter for Wonder is a CURVE, not a line.
               This is the anchor every other theme is compared against.

  SAFETY     -> The cautionary tale. Strong days 1-30, an inflection
               around day ~40, then steady decline. CTR falls, CPI rises.
               CRITICAL SIGNAL: on VIDEO ads, hold_rate degrades FASTER
               than CTR from day 35 onward -> the visual execution is
               failing, not merely the theme. The analytics layer is
               meant to detect this divergence.

  CONFIDENCE -> The flat one. Mid-tier metrics with natural variance and
               no real trend in either direction. The "stable" baseline.

  CONNECTION -> The noisy one. High week-over-week volatility. A couple of
               strong ads mask several weak ones (large WITHIN-theme
               variance). Net trend is slightly positive but jagged --
               the theme most likely to mislead a naive average.
               THE TWO OUTPERFORMERS ARE THE VIDEO ADS (a child-to-character
               interaction mechanic): CVR ~1.6x the video baseline AND damped
               week-over-week variance, so they read as the ONLY consistent
               performers in an otherwise jagged cluster -- detectable outliers.

-------------------------------------------------------------------
FORMAT & PLACEMENT DIFFERENTIALS (stable properties, not theme trends)
-------------------------------------------------------------------
FORMAT is a creative property applied identically across all themes/days:
    video    -> CTR 1.40x baseline, CVR 1.00x  (hooks hard, converts average)
    static   -> CTR 0.85x baseline, CVR 0.75x  (interest without context)
    carousel -> CTR 0.80x baseline, CVR 1.25x  (sequential story -> best CVR)

PLACEMENT is assigned DELIBERATELY per theme (not uniformly) so cross-theme
and cross-placement comparisons are possible:
    wonder     -> feed, reels, youtube      (no stories)
    safety     -> feed, stories             (NO reels, no youtube)
    confidence -> feed, stories, search
    connection -> feed, reels, stories      (no youtube)
Placement-level CPI differentials (relative to that theme's feed baseline):
    reels   ~ -16% CPI  (short-form video performs well; wonder + connection)
    stories ~ +10% CPI  (higher CPM, weaker conversion efficiency)
    youtube ~ -12% CPI  (wonder video only -- youtube exists only there)
    search  ~ highest CPI but HIGHEST CVR (expensive, but high-intent clicks)
Because Safety never runs on reels while Wonder/Connection show strong reels
CPI, "test Safety on reels" becomes an evidence-backed, defensible hypothesis.

-------------------------------------------------------------------
HOW THE NUMBERS ARE BUILT (consistency by construction)
-------------------------------------------------------------------
We generate the four PRIMITIVES per ad-day:
    spend, impressions, clicks, installs
and DERIVE every ratio from them:
    CTR = clicks / impressions
    CPC = spend  / clicks
    CPM = spend  / impressions * 1000
    CVR = installs / clicks
    CPI = spend  / installs
Because installs = impressions * CTR * CVR and impressions = spend/CPM*1000,
spend cancels out of CPI, i.e. CPI ~ CPM / (1000 * CTR * CVR). That is why
raising CTR/CVR (Wonder) mechanically lowers CPI -- no separate CPI curve.
The Wonder diminishing-returns "ceiling" is likewise NOT a hand-drawn CPI
curve: it falls out of (a) a piecewise CTR/CVR efficiency curve that rolls
over after day ~60 and (b) a spend-saturation haircut on CVR that only bites
once daily spend is high -- so CPI emerges as a function of spend, not time.
Floors (>=1 impression, >=1 click) guarantee every ratio is computable.

Format-specific engagement metrics are null where they don't apply:
    video    -> thumbstop_rate, hold_rate, completion_rate,
                video_plays_25pct, video_plays_75pct, avg_play_time_sec
    carousel -> card_swipe_rate, last_card_reach_rate
    static   -> (none of the above)

Meta social actions (likes, comments, shares, saves) are populated ONLY for
Meta static + carousel ads; they are null for every video ad and every Google
ad. likes track CTR (so they inherit each theme's CTR narrative); comments,
saves, shares are ordered fractions of likes (comments < saves < shares < likes).

The added video deep-play columns carry their own narrative leg: the 25% (hook)
count stays stable for Safety while the 75% (deep) count and avg_play_time_sec
collapse after day ~35 -- viewers bailing deep in the video, not at the hook.
By construction video_plays_75pct < video_plays_25pct < impressions and
avg_play_time_sec <= video_length_sec on every row.

Reproducible: fixed RANDOM_SEED. Standard library only (no numpy/pandas).
===================================================================
"""

import csv
import math
import random
from datetime import date, timedelta

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
RANDOM_SEED = 42
DAYS = 90
START_DATE = date(2026, 3, 1)          # day 0 of the window
COUNTRY = "US"

THEMES = ["wonder", "safety", "confidence", "connection"]
FORMATS = ["video", "static", "carousel"]
PLATFORMS = ["meta", "google"]

# Each theme gets exactly one ad per (platform, format) combo = 6 ads/theme.
ADS_PER_THEME = len(PLATFORMS) * len(FORMATS)   # 6
TOTAL_ADS = len(THEMES) * ADS_PER_THEME         # 24

rng = random.Random(RANDOM_SEED)

# A SECOND, independent stream for the engagement columns added later (deep-play
# counts, watch time, Meta social actions). Drawing these from `rng` would shift
# every subsequent draw and silently alter the original columns; an offset seed
# keeps the legacy stream byte-for-byte identical while staying fully reproducible.
aux_rng = random.Random(RANDOM_SEED + 1)


# ------------------------------------------------------------------
# Creative copy, themed for a children's AI storytelling app
# ------------------------------------------------------------------
COPY = {
    "wonder": {
        "hooks": [
            "What if every bedtime opened a new world?",
            "Your child becomes the hero tonight",
            "A story that knows your child's name",
        ],
        "headlines": ["Stories That Spark Wonder", "Magic, Made for Them"],
        "primary": [
            "Giant turns your child's imagination into a personalized adventure, every single night.",
            "Endless worlds, gentle magic, and a hero who looks and sounds just like your kid.",
        ],
    },
    "safety": {
        "hooks": [
            "Screen time you can finally feel good about",
            "No ads. No surprises. Just stories.",
            "A safe little world, built for small humans",
        ],
        "headlines": ["Storytime, Without the Worry", "Safe by Design"],
        "primary": [
            "Giant is a walled garden of kind, age-appropriate stories -- no ads, no chat, no strangers.",
            "Parent-approved, kid-safe tales you can hand over with total peace of mind.",
        ],
    },
    "confidence": {
        "hooks": [
            "Stories where your child saves the day",
            "Help them believe they can",
            "Every tale ends with 'I did it!'",
        ],
        "headlines": ["Adventures That Build Brave Kids", "They Can Do Hard Things"],
        "primary": [
            "Giant writes your child into stories where they solve, lead, and triumph -- one chapter at a time.",
            "Confidence grows when kids are the hero. Giant makes them the hero every night.",
        ],
    },
    "connection": {
        "hooks": [
            "Read together, even when you're apart",
            "A story you build with your child",
            "Snuggle up to a tale made for two",
        ],
        "headlines": ["Stories You Share", "Closer, One Chapter at a Time"],
        "primary": [
            "Giant creates stories you and your child shape together -- the best part of the day, every day.",
            "Grandparents, parents, kids -- one shared adventure that brings everyone closer.",
        ],
    },
}

CTAS = ["Download Now", "Try Free", "Start the Adventure", "Begin the Story"]

# Placement is assigned DELIBERATELY per (theme, platform, format) -- not
# uniformly -- so the placement mix differs by theme and enables cross-theme /
# cross-placement comparisons (e.g. reels is strong in wonder + connection but
# DELIBERATELY ABSENT from safety, which is the untested-placement signal).
# Allowed sets:  wonder {feed,reels,youtube}  safety {feed,stories}
#                confidence {feed,stories,search}  connection {feed,reels,stories}
PLACEMENT_MAP = {
    ("wonder", "meta", "video"): "reels",
    ("wonder", "meta", "static"): "feed",
    ("wonder", "meta", "carousel"): "feed",
    ("wonder", "google", "video"): "youtube",
    ("wonder", "google", "static"): "feed",
    ("wonder", "google", "carousel"): "feed",

    ("safety", "meta", "video"): "stories",
    ("safety", "meta", "static"): "feed",
    ("safety", "meta", "carousel"): "stories",
    ("safety", "google", "video"): "feed",
    ("safety", "google", "static"): "feed",
    ("safety", "google", "carousel"): "stories",

    ("confidence", "meta", "video"): "stories",
    ("confidence", "meta", "static"): "feed",
    ("confidence", "meta", "carousel"): "feed",
    ("confidence", "google", "video"): "stories",
    ("confidence", "google", "static"): "search",
    ("confidence", "google", "carousel"): "search",

    ("connection", "meta", "video"): "reels",
    ("connection", "meta", "static"): "feed",
    ("connection", "meta", "carousel"): "stories",
    ("connection", "google", "video"): "reels",
    ("connection", "google", "static"): "feed",
    ("connection", "google", "carousel"): "stories",
}
ASPECT_MAP = {
    "video": "9:16",
    "static": "1:1",
    "carousel": "4:5",
}


# ------------------------------------------------------------------
# Build the static ad table (ads.csv)
# ------------------------------------------------------------------
def build_ads():
    ads = []
    ad_counter = 1
    for t_idx, theme in enumerate(THEMES):
        # Per-theme "quality" spread across the 6 ads. Connection gets a wide
        # spread (a couple of strong ads masking several weak ones); the other
        # themes are tighter. This multiplier scales an ad's CTR/CVR baseline.
        # Combo order is (meta video, meta static, meta carousel, google video,
        # google static, google carousel) -> indices 0 and 3 are the VIDEO ads.
        # For connection those two video ads are deliberately the strong pair
        # (the child-to-character mechanic), so the spread peaks at 0 and 3.
        if theme == "connection":
            quality_spread = [1.30, 0.95, 0.88, 1.25, 0.78, 0.70]
        else:
            quality_spread = [1.08, 1.04, 1.00, 0.98, 0.95, 0.92]

        combo_idx = 0
        for platform in PLATFORMS:
            for fmt in FORMATS:
                ad_id = f"ad_{ad_counter:03d}"
                campaign_id = f"cmp_{platform}_{theme}"
                adset_id = f"set_{platform}_{theme}_{fmt}"

                hooks = COPY[theme]["hooks"]
                heads = COPY[theme]["headlines"]
                prim = COPY[theme]["primary"]

                placement = PLACEMENT_MAP[(theme, platform, fmt)]
                aspect = ASPECT_MAP[fmt]
                # Desktop only really shows up for Google search; everything
                # else is a mobile-first children's app audience.
                device = "desktop" if placement == "search" else "mobile"

                video_len = ""  # null for non-video
                if fmt == "video":
                    video_len = rng.choice([6, 15, 30])

                # Connection's two video ads are the standout creatives: a big
                # CVR lift on top of the video baseline AND damped volatility, so
                # they stay consistent while the rest of the cluster swings.
                is_conn_outlier = (theme == "connection" and fmt == "video")
                cvr_outlier_mult = 1.60 if is_conn_outlier else 1.00
                volatility = 0.25 if is_conn_outlier else 1.00

                ads.append({
                    "record": {
                        "ad_id": ad_id,
                        "platform": platform,
                        "campaign_id": campaign_id,
                        "adset_id": adset_id,
                        "theme": theme,
                        "format": fmt,
                        "hook_text": rng.choice(hooks),
                        "headline": rng.choice(heads),
                        "primary_text": rng.choice(prim),
                        "call_to_action": rng.choice(CTAS),
                        "video_length_sec": video_len,
                        "aspect_ratio": aspect,
                        "placement": placement,
                        "country": COUNTRY,
                        "device_type": device,
                    },
                    # internal generation state (not written to ads.csv):
                    "theme": theme,
                    "format": fmt,
                    "platform": platform,
                    "quality": quality_spread[combo_idx],
                    # a stable per-ad phase so connection ads peak on
                    # different weeks (de-synchronized volatility):
                    "phase": rng.uniform(0, 2 * math.pi),
                    "base_spend": rng.uniform(45.0, 85.0),
                    # CVR lift for the connection video outliers (1.0 otherwise):
                    "cvr_outlier_mult": cvr_outlier_mult,
                    # week-over-week swing damping (1.0 = full volatility,
                    # 0.25 = steady, used for the connection outliers):
                    "volatility": volatility,
                })
                combo_idx += 1
                ad_counter += 1
    return ads


# ------------------------------------------------------------------
# Trend curves -- the heart of the narrative
# Each returns a multiplier on the ad's baseline rate for a given day.
# d is the day index (0..89); progress p = d / (DAYS-1) in [0, 1].
# ------------------------------------------------------------------
def wonder_ctr_mult(d):
    # Engagement keeps climbing through scaling, then PLATEAUS in the mature
    # window (it does not roll over -- the CPI ceiling comes from CVR + spend).
    if d < 30:
        return 1.00 + 0.15 * (d / 30.0)          # warmup:  1.00 -> 1.15
    if d < 60:
        return 1.15 + 0.07 * ((d - 30) / 30.0)   # scaling: 1.15 -> 1.22
    return 1.22                                   # mature:  plateau


def wonder_cvr_mult(d):
    # Diminishing-returns conversion efficiency. This is the curve that gives
    # CPI its three phases: steep gain, then slower, then a mature ROLLOVER --
    # CPI bottoms near day 60, then the efficiency ceiling pushes it back up.
    if d < 30:
        return 1.00 + 0.15 * (d / 30.0)          # warmup:  1.00 -> 1.15 (steep)
    if d < 60:
        return 1.15 + 0.04 * ((d - 30) / 30.0)   # scaling: 1.15 -> 1.19 (slower)
    return 1.19 - 0.09 * ((d - 60) / 29.0)       # mature:  1.19 -> 1.10 (rollover)


def theme_ctr_mult(theme, d, ad):
    p = d / (DAYS - 1)
    if theme == "wonder":
        # rising with diminishing returns (plateaus in the mature window)
        return wonder_ctr_mult(d)
    if theme == "safety":
        # strong & slightly rising through ~day 35, then steady decline
        if d < 35:
            return 1.10 + 0.10 * (d / 35.0)
        decay = (d - 35) / (DAYS - 1 - 35)
        return 1.20 - 0.50 * decay        # falls from ~1.20 to ~0.70
    if theme == "confidence":
        # flat, mild slow wobble only (no trend)
        return 1.00 + 0.03 * math.sin(2 * math.pi * d / 30.0)
    if theme == "connection":
        # slight positive drift + strong week-over-week oscillation,
        # phase-shifted per ad so peaks don't line up. ad["volatility"] damps
        # the swing for the outlier video ads so they stay steady.
        drift = 1.00 + 0.15 * p
        weekly = 0.22 * ad["volatility"] * math.sin(2 * math.pi * d / 7.0 + ad["phase"])
        return drift + weekly
    return 1.0


def theme_cvr_mult(theme, d, ad):
    p = d / (DAYS - 1)
    if theme == "wonder":
        return wonder_cvr_mult(d)         # diminishing-returns funnel -> CPI floor
    if theme == "safety":
        if d < 35:
            return 1.06
        decay = (d - 35) / (DAYS - 1 - 35)
        return 1.06 - 0.34 * decay        # funnel weakens -> CPI rises
    if theme == "confidence":
        return 1.00 + 0.02 * math.sin(2 * math.pi * d / 45.0)
    if theme == "connection":
        weekly = 0.12 * ad["volatility"] * math.sin(2 * math.pi * d / 7.0 + ad["phase"] + 1.0)
        return 1.00 + 0.08 * p + weekly
    return 1.0


def theme_spend_scale(theme, d):
    # WONDER ONLY: the winner gets scaled up over the window, so mature-phase
    # daily spend climbs past the saturation threshold. This is what makes the
    # efficiency ceiling SPEND-driven rather than a pure time trend. All other
    # themes return 1.0 -> diminishing-returns logic never leaks into them.
    if theme == "wonder":
        if d < 30:
            return 1.00 + 0.50 * (d / 30.0)          # 1.0 -> 1.5
        if d < 60:
            return 1.50 + 1.00 * ((d - 30) / 30.0)   # 1.5 -> 2.5
        return 2.50 + 0.80 * ((d - 60) / 29.0)       # 2.5 -> 3.3
    return 1.00


def spend_saturation_penalty(theme, spend):
    # WONDER ONLY: once daily spend pushes past ~$190/day per ad, incremental
    # budget reaches lower-intent users, so CVR (hence CPI) degrades. The size of
    # the haircut scales with how far over the threshold spend is -> a spend-vs-CPI
    # scatter bends into a curve. Capped at a 20% CVR drag.
    if theme != "wonder":
        return 1.00
    THRESHOLD = 190.0
    if spend <= THRESHOLD:
        return 1.00
    overshoot = (spend - THRESHOLD) / THRESHOLD
    return max(0.75, 1.00 - 0.35 * overshoot)


def theme_cpm_mult(theme, d):
    # Auction pressure. Mostly stable; declining theme costs a bit more to
    # serve as relevance drops (Safety late window), winner gets cheaper.
    p = d / (DAYS - 1)
    if theme == "wonder":
        # CPM efficiency improves while the campaign is learning, then BOTTOMS
        # OUT around day 60 -- no more serving-cost gains once mature, which
        # contributes to the CPI ceiling rather than fighting it.
        if d < 60:
            return 1.00 - 0.10 * (d / 59.0)   # 1.00 -> ~0.90
        return 0.90
    if theme == "safety":
        if d < 35:
            return 1.00
        return 1.00 + 0.18 * ((d - 35) / (DAYS - 1 - 35))
    return 1.00


def hold_rate_mult(theme, d):
    """
    Video hold_rate curve. For SAFETY this is the key signal: from day 35
    hold_rate must decline FASTER than CTR. Compare slopes:
      safety CTR  late: 1.20 -> 0.70  (drop of 0.50 over the tail)
      safety hold late: 1.15 -> 0.45  (drop of 0.70 over the tail)  <-- steeper
    """
    if theme == "safety":
        if d < 35:
            return 1.15
        decay = (d - 35) / (DAYS - 1 - 35)
        return 1.15 - 0.70 * decay
    if theme == "wonder":
        return 1.00 + 0.15 * (d / (DAYS - 1))
    if theme == "confidence":
        return 1.00
    if theme == "connection":
        return 1.00 + 0.10 * math.sin(2 * math.pi * d / 7.0)
    return 1.0


def plays25_mult(theme, d, ad):
    """
    Retention to the 25% mark -- i.e. survival past the HOOK. For SAFETY this is
    the deliberately STABLE leg: the hook keeps landing all 90 days, so the 25%
    count holds steady even as the theme declines. The collapse shows up only in
    the *deep* (75%) watch below -- "viewers drop off deeper in, not at the hook."
      WONDER  -> gently rising.   CONFIDENCE -> flat with tiny wobble.
      CONNECTION -> strong week-over-week swing (phase-shifted per ad).
    """
    if theme == "wonder":
        return 1.00 + 0.12 * (d / (DAYS - 1))
    if theme == "safety":
        return 1.00
    if theme == "confidence":
        return 1.00 + 0.02 * math.sin(2 * math.pi * d / 30.0)
    if theme == "connection":
        return 1.00 + 0.18 * ad["volatility"] * math.sin(2 * math.pi * d / 7.0 + ad["phase"])
    return 1.0


def plays75_ratio_mult(theme, d, ad):
    """
    Multiplier on the 75%/25% deep-retention RATIO. Because plays75 is built as
    plays25 * (ratio clamped < 1), the invariant plays75 < plays25 holds by
    construction. This curve carries the deep-watch narrative:
      WONDER  -> ratio RISES: viewers increasingly watch through to the end,
                 so video_plays_75pct trends UP over the window.
      SAFETY  -> flat to day 35, then DROPS SHARPLY -> 75% plays (and watch time)
                 fall off a cliff while the 25% hook count stays flat. This is the
                 deep-dropoff signal that pairs with the steeper hold_rate decay.
      CONFIDENCE -> flat.   CONNECTION -> high week-over-week swing.
    """
    if theme == "wonder":
        return 1.00 + 0.30 * (d / (DAYS - 1))
    if theme == "safety":
        if d < 35:
            return 1.00
        decay = (d - 35) / (DAYS - 1 - 35)
        return 1.00 - 0.60 * decay
    if theme == "confidence":
        return 1.00
    if theme == "connection":
        return 1.00 + 0.15 * ad["volatility"] * math.sin(2 * math.pi * d / 7.0 + ad["phase"] + 0.5)
    return 1.0


# ------------------------------------------------------------------
# Environmental modulation: warmup ramp + day-of-week seasonality
# ------------------------------------------------------------------
def spend_ramp(d):
    # Campaign warmup: budgets scale up from ~55% to 100% over days 0-29,
    # then hold flat. Simulates a growth team easing into spend.
    if d < 30:
        return 0.55 + 0.45 * (d / 30.0)
    return 1.00


def weekend_spend_factor(the_day):
    # Sat=5, Sun=6. Weekends pace down slightly.
    return 0.85 if the_day.weekday() >= 5 else 1.00


def weekend_demand_factor(the_day):
    # Conversion intent also dips a touch on weekends (parents busier).
    return 0.93 if the_day.weekday() >= 5 else 1.00


def noise(scale):
    # Multiplicative noise centered on 1.0, clamped to stay positive.
    return max(0.4, rng.gauss(1.0, scale))


def aux_noise(scale):
    # Same shape as noise(), but drawn from the independent aux_rng so the
    # original columns' random stream is never perturbed.
    return max(0.4, aux_rng.gauss(1.0, scale))


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


# ------------------------------------------------------------------
# Baseline rates by theme/format (pre-trend, pre-noise)
# ------------------------------------------------------------------
BASE_CTR = {"wonder": 0.0220, "safety": 0.0205, "confidence": 0.0160, "connection": 0.0175}
BASE_CVR = {"wonder": 0.180, "safety": 0.165, "confidence": 0.140, "connection": 0.150}
BASE_CPM = {"meta": 9.0, "google": 7.0}

# Format differentials -- stable creative properties applied across ALL themes
# and all 90 days (they are format facts, not theme trends):
#   video    hooks hardest (high CTR) but converts at baseline
#   static   moderate CTR, weakest CVR (interest without enough context)
#   carousel lowest CTR but BEST CVR (sequential cards tell a story pre-CTA)
FORMAT_CTR_MULT = {"video": 1.40, "static": 0.85, "carousel": 0.80}
FORMAT_CVR_MULT = {"video": 1.00, "static": 0.75, "carousel": 1.25}

# Placement differentials -- stable per-placement level shifts realised through
# the primitives (never by editing CPI directly). Relative to feed:
#   reels   ctr*cvr up   -> ~-16% CPI    stories cpm up / cvr down -> ~+10% CPI
#   youtube ctr*cvr up   -> ~-12% CPI    search  cpm up + best cvr -> highest CPI
PLACEMENT_CTR_MULT = {"feed": 1.00, "reels": 1.10, "stories": 1.00, "youtube": 1.07, "search": 0.90}
PLACEMENT_CVR_MULT = {"feed": 1.00, "reels": 1.09, "stories": 0.965, "youtube": 1.06, "search": 1.25}
PLACEMENT_CPM_MULT = {"feed": 1.00, "reels": 1.00, "stories": 1.06, "youtube": 1.00, "search": 1.50}

# Format engagement baselines (video / carousel only)
BASE_THUMBSTOP = 0.32
BASE_HOLD = 0.46
BASE_COMPLETION = 0.55
BASE_CARD_SWIPE = 0.40
BASE_LAST_CARD = 0.20

# Deep-play retention (video only). ~25% of impressions reach the 25% mark for a
# strong performer; of those, ~45% reach 75% before the narrative reshapes it.
BASE_PLAYS25_FRAC = 0.25
BASE_PLAYS75_RATIO = 0.45

# Meta social actions (Meta static + carousel only). likes scale with CTR; the
# rest are fractions of likes, ordered comments < saves < shares < likes.
BASE_LIKE_K = 0.20      # likes-per-impression per unit CTR (keeps likes a small %)
COMMENT_RATIO = 0.06    # comments as a fraction of likes (much smaller)
SAVE_RATIO = 0.10       # saves slightly above comments
SHARE_RATIO = 0.18      # shares sit between comments and likes


# ------------------------------------------------------------------
# Build the time series (engagement.csv)
# ------------------------------------------------------------------
def fmt_float(x, places):
    return "" if x is None else f"{round(x, places):.{places}f}"


def build_engagement(ads):
    rows = []
    for ad in ads:
        theme = ad["theme"]
        fmt = ad["format"]
        platform = ad["platform"]
        quality = ad["quality"]
        placement = ad["record"]["placement"]

        for d in range(DAYS):
            the_day = START_DATE + timedelta(days=d)

            # ----- spend -----
            # theme_spend_scale ramps Wonder's budget up over the window (1.0 for
            # every other theme) so its mature-phase spend crosses the saturation
            # threshold -- the lever that makes Wonder's CPI a function of spend.
            spend = (ad["base_spend"]
                     * spend_ramp(d)
                     * theme_spend_scale(theme, d)
                     * weekend_spend_factor(the_day)
                     * noise(0.10))
            spend = max(5.0, spend)

            # ----- effective rates -----
            # Each rate stacks: theme baseline x format property x placement
            # property x per-ad quality x theme trend x environment x noise.
            ctr = (BASE_CTR[theme]
                   * FORMAT_CTR_MULT[fmt]
                   * PLACEMENT_CTR_MULT[placement]
                   * quality
                   * theme_ctr_mult(theme, d, ad)
                   * noise(0.07))
            ctr = clamp(ctr, 0.0015, 0.12)

            # CVR additionally carries: the format CVR differential (carousel best,
            # static worst), the placement CVR differential (search best), the
            # connection-outlier 1.6x lift, and the Wonder spend-saturation haircut.
            cvr = (BASE_CVR[theme]
                   * FORMAT_CVR_MULT[fmt]
                   * PLACEMENT_CVR_MULT[placement]
                   * quality
                   * theme_cvr_mult(theme, d, ad)
                   * ad["cvr_outlier_mult"]
                   * spend_saturation_penalty(theme, spend)
                   * weekend_demand_factor(the_day)
                   * noise(0.06))
            cvr = clamp(cvr, 0.01, 0.45)

            cpm = (BASE_CPM[platform]
                   * PLACEMENT_CPM_MULT[placement]
                   * theme_cpm_mult(theme, d)
                   * noise(0.08))
            cpm = max(1.5, cpm)

            # ----- primitives (derive counts, then recompute ratios) -----
            # Floors guarantee no divide-by-zero downstream: >=1 impression and
            # >=1 click before any ratio is taken.
            impressions = int(round(spend / cpm * 1000.0))
            impressions = max(1, impressions)
            clicks = int(round(impressions * ctr))
            clicks = max(1, clicks)
            installs = int(round(clicks * cvr))
            installs = max(0, installs)

            # ----- derived metrics (guarded against divide-by-zero) -----
            d_ctr = clicks / impressions if impressions else None
            d_cpm = spend / impressions * 1000.0 if impressions else None
            d_cpc = spend / clicks if clicks else None
            d_cvr = installs / clicks if clicks else None
            d_cpi = spend / installs if installs else None

            # ----- format-specific engagement -----
            thumbstop = hold = completion = None
            card_swipe = last_card = None
            # NEW video deep-play columns (null for static/carousel):
            video_plays_25pct = video_plays_75pct = avg_play_time_sec = None
            # NEW Meta social columns (null for video + all Google):
            likes = comments = shares = saves = None
            if fmt == "video":
                thumbstop = clamp(BASE_THUMBSTOP * quality
                                  * (0.5 * theme_ctr_mult(theme, d, ad) + 0.5)
                                  * noise(0.06), 0.05, 0.95)
                hold = clamp(BASE_HOLD * quality
                             * hold_rate_mult(theme, d)
                             * noise(0.06), 0.03, 0.95)
                completion = clamp(BASE_COMPLETION * quality
                                   * (0.6 * hold_rate_mult(theme, d) + 0.4)
                                   * noise(0.05), 0.02, 0.98)

                # 25% mark = hook survival. Built off impressions so it can never
                # exceed them. frac clamped < 1 -> plays25 < impressions always.
                frac25 = clamp(BASE_PLAYS25_FRAC * quality
                               * plays25_mult(theme, d, ad)
                               * aux_noise(0.06), 0.02, 0.60)
                video_plays_25pct = clamp(int(round(impressions * frac25)),
                                          0, impressions)

                # 75% mark = deep watch, built as a sub-fraction of the 25% count.
                # ratio clamped < 1 guarantees plays75 < plays25 by construction;
                # the final min() guard keeps it STRICT even after int rounding.
                ratio75 = clamp(BASE_PLAYS75_RATIO * plays75_ratio_mult(theme, d, ad)
                                * aux_noise(0.07), 0.05, 0.90)
                video_plays_75pct = int(round(video_plays_25pct * ratio75))
                if video_plays_25pct > 0:
                    video_plays_75pct = min(video_plays_75pct,
                                            video_plays_25pct - 1)
                video_plays_75pct = max(0, video_plays_75pct)

                # avg watch time rides the hold_rate value directly (so it is
                # correlated with hold_rate and inherits every theme curve: rising
                # for Wonder, the sharp post-35 cliff for Safety) and scales with
                # the ad's length. The fraction is capped < 1, so the result is
                # always <= video_length_sec.
                vlen = ad["record"]["video_length_sec"]
                play_fraction = clamp(hold * 1.3 * aux_noise(0.05), 0.03, 0.98)
                avg_play_time_sec = vlen * play_fraction
            elif fmt == "carousel":
                card_swipe = clamp(BASE_CARD_SWIPE * quality
                                   * theme_ctr_mult(theme, d, ad)
                                   * noise(0.07), 0.05, 0.95)
                last_card = clamp(BASE_LAST_CARD * quality
                                  * theme_ctr_mult(theme, d, ad)
                                  * noise(0.08), 0.02, 0.85)

            # Meta social actions: Meta platform, static + carousel only (never
            # video, never Google). likes track realized CTR -> they inherit each
            # theme's CTR narrative for free (Wonder up, Safety down, Connection
            # jagged, Confidence flat). The others are ordered fractions of likes.
            if platform == "meta" and fmt in ("static", "carousel"):
                ctr_for_social = d_ctr if d_ctr else 0.0
                like_rate = clamp(BASE_LIKE_K * ctr_for_social * quality
                                  * aux_noise(0.10), 0.0, 0.05)
                likes = max(0, int(round(impressions * like_rate)))
                comments = max(0, int(round(likes * COMMENT_RATIO * aux_noise(0.15))))
                saves = max(0, int(round(likes * SAVE_RATIO * aux_noise(0.15))))
                shares = max(0, int(round(likes * SHARE_RATIO * aux_noise(0.15))))

            rows.append({
                "ad_id": ad["record"]["ad_id"],
                "date": the_day.isoformat(),
                "spend": f"{spend:.2f}",
                "impressions": impressions,
                "clicks": clicks,
                "installs": installs,
                "CTR": fmt_float(d_ctr, 5),
                "CPI": fmt_float(d_cpi, 4),
                "CPM": fmt_float(d_cpm, 4),
                "CPC": fmt_float(d_cpc, 4),
                "CVR": fmt_float(d_cvr, 5),
                "thumbstop_rate": fmt_float(thumbstop, 4),
                "hold_rate": fmt_float(hold, 4),
                "completion_rate": fmt_float(completion, 4),
                "card_swipe_rate": fmt_float(card_swipe, 4),
                "last_card_reach_rate": fmt_float(last_card, 4),
                "video_plays_25pct": video_plays_25pct,
                "video_plays_75pct": video_plays_75pct,
                "avg_play_time_sec": fmt_float(avg_play_time_sec, 2),
                "likes": likes,
                "comments": comments,
                "shares": shares,
                "saves": saves,
            })
    return rows


# ------------------------------------------------------------------
# Writers
# ------------------------------------------------------------------
ADS_COLUMNS = [
    "ad_id", "platform", "campaign_id", "adset_id", "theme", "format",
    "hook_text", "headline", "primary_text", "call_to_action",
    "video_length_sec", "aspect_ratio", "placement", "country", "device_type",
]

ENGAGEMENT_COLUMNS = [
    "ad_id", "date", "spend", "impressions", "clicks", "installs",
    "CTR", "CPI", "CPM", "CPC", "CVR",
    "thumbstop_rate", "hold_rate", "completion_rate",
    "card_swipe_rate", "last_card_reach_rate",
    # added: video deep-play (video only) + Meta social (Meta static/carousel only)
    "video_plays_25pct", "video_plays_75pct", "avg_play_time_sec",
    "likes", "comments", "shares", "saves",
]


def write_csv(path, columns, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    ads = build_ads()
    engagement = build_engagement(ads)

    write_csv("ads.csv", ADS_COLUMNS, [a["record"] for a in ads])
    write_csv("engagement.csv", ENGAGEMENT_COLUMNS, engagement)

    print(f"Wrote ads.csv         ({len(ads)} ads)")
    print(f"Wrote engagement.csv  ({len(engagement)} rows = "
          f"{len(ads)} ads x {DAYS} days)")


if __name__ == "__main__":
    main()
