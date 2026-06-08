"""Layer 3 - Hypothesis Layer.

Turns the precomputed ``theme_signals`` (Layer 1) and ``theme_diagnosis``
(Layer 2) artifacts into an **ICE-scored, ranked action backlog** - the list of
"what to test/do next" that the agent (Layer 4) reads back in plain language.

All logic here is deterministic Python - no LLM. This layer **scores and ranks
existing facts**; it never re-derives a metric. Every recommendation is
traceable to a signal/diagnosis value the earlier layers already computed and
validated, which is what keeps the whole pipeline auditable.

Design notes worth knowing when reading the code:

- **Primary + secondary granularity.** Each theme always emits its *primary*
  hypothesis - the ``hypothesis_type`` Layer 1 already derived. An
  *execution-level* theme (uniformity ``isolated``/``mixed``) additionally emits
  a *secondary* ``intervention`` keyed on its laggards: the same theme can both
  reward replicating its winners and demand fixing its laggards (e.g. Connection
  - replicate the reels-video winners, intervene on the static/feed laggards).
  Replication keys on ``leading_ads``; intervention on ``lagging_ads``. No new
  signal is needed - only fields the diagnosis layer already exposes.

- **ICE impact is spend-aware** (a deliberate divergence from the spec's 2-arg
  ``build_ranked_backlog(theme_signals, theme_diagnosis)``). The spec defines
  impact as "CPI-delta magnitude × spend level," but neither input dict carries
  spend, so we accept Layer 1's ``ad_signals_df`` as a third argument and read
  its per-ad ``recent_spend``. ``app.py`` already holds ``ad_signals_df`` at
  startup, so wiring it in is trivial. Spend matters even on today's data:
  Wonder's budget ramps ~3.3× over the window, so it carries by far the most
  spend-at-stake and a flat-but-expensive plateau there is a bigger opportunity
  than a large % swing on a small-spend theme. The spend tier is taken relative
  to the **median** theme spend (robust to the one ramped outlier), so the
  "rounding-error theme shouldn't outrank a real budget" guard is in place and
  will bite harder once per-ad spend is widened in the dataset.

- **ICE is multiplicative.** ``raw = impact × confidence × ease`` on 1–3 points
  each, rescaled to the spec's 1–10 ``ice_score``. A single weak dimension drags
  the score down hard - the standard ICE behaviour - so e.g. an ``optimization``
  (hard to execute → low ease) never tops the backlog on ease alone.

- **Narrative strings are templated, never hard-coded.** ``recommendation`` /
  ``evidence`` / ``open_questions`` are assembled per ``hypothesis_type`` with
  every number slotted from the artifacts; ``evidence`` is anchored on the
  diagnosis layer's deterministic ``pattern_summary`` so it stays verbatim-
  traceable. All five types are templated (incl. ``amplification``, which the
  current dataset never triggers) because the contract enumerates them and the
  agent may surface any.
"""

from __future__ import annotations

import pandas as pd

from util.stats import weighted_nanmean

# ── ICE points / label mapping ────────────────────────────────────────────
# Every ICE dimension is scored on a 1–3 scale; the public high/medium/low
# label is derived back from the points so the two never drift apart.
_PTS_TO_LABEL = {1: "low", 2: "medium", 3: "high"}
_STRENGTH_PTS = {"strong": 3, "moderate": 2, "weak": 1}

# Ease is a pure function of the action type (spec: amplification/retirement are
# cheap to execute; anything needing net-new creative is harder).
_EASE_PTS = {
    "amplification": 3,
    "retirement": 3,
    "replication": 2,
    "intervention": 2,
    "optimization": 1,
}

# Spend tier thresholds, on a theme's last-week spend / median theme spend.
SPEND_HIGH_RATIO = 1.30   # well above the typical theme budget
SPEND_LOW_RATIO = 0.60    # well below it (a near-rounding-error theme)

# Impact bucket cutoffs on the (magnitude + spend)/2 blend (each term 1–3).
IMPACT_HIGH_RAW = 2.25
IMPACT_MED_RAW = 1.75


# ── Small formatting helpers (mirrors diagnosis.py's local formatters) ─────

def _pct(value: float | None) -> str:
    """Percentage delta with an explicit sign."""
    return "n/a" if value is None else f"{value:+.0f}%"


def _money(value: float | None) -> str:
    """A dollar figure (used for last-week spend)."""
    return "n/a" if value is None else f"${value:,.0f}"


def _cpi(value: float | None) -> str:
    """A CPI level to the cent (spend is whole-dollar; CPI is not)."""
    return "n/a" if value is None else f"${value:.2f}"


def _ids(entries: list[dict]) -> list[str]:
    """ad_ids out of a list of diagnosis ad-entries (leading/lagging)."""
    return [e["ad_id"] for e in entries]


def _ad_phrase(entry: dict) -> str:
    """Name one ad with its format/placement and current CPI level.

    e.g. ``ad_005 (static/feed) at $2.71 CPI``. CPI level and hook come from the
    diagnosis ad-entry verbatim — nothing is recomputed here.
    """
    cpi = entry.get("last7d_mean_cpi")
    cpi_str = f" at {_cpi(cpi)} CPI" if cpi is not None else ""
    return f"{entry['ad_id']} ({entry['format']}/{entry['placement']}){cpi_str}"


def _ad_list(entries: list[dict]) -> str:
    """Comma-joined ``_ad_phrase`` for several ads (empty → generic fallback)."""
    return ", ".join(_ad_phrase(e) for e in entries) or "the relevant ads"


def _hook(entry: dict | None) -> str:
    """The ad's creative hook in quotes (for benchmarking copy), or a fallback."""
    text = entry.get("hook_text") if entry else None
    return f'"{text}"' if text else "its current hook"


def _drivers(diag: dict) -> str:
    """Render '<format> on <placement>' for a theme's dominant pair."""
    fmt = diag.get("dominant_format")
    plc = diag.get("dominant_placement")
    if fmt and plc:
        return f"{fmt} on {plc}"
    return fmt or "no single format"


def _weak_where(diag: dict) -> str:
    """Render '<weak_format>/<weak_placement>' for intervention targeting."""
    fmt = diag.get("weak_format")
    plc = diag.get("weak_placement")
    if fmt and plc:
        return f"{fmt}/{plc}"
    return fmt or plc or "their current format/placement"


# ── ICE scoring (deterministic) ────────────────────────────────────────────

def _spend_pts(theme_spend: float | None, median_spend: float | None) -> int:
    """Spend tier (1–3) relative to the median theme spend.

    Median-relative so the single budget-ramped theme (Wonder) doesn't drag the
    reference up and flatten everyone else to "below average". A theme far below
    the median scores 1 - the spec's "small % move on a rounding-error line item
    shouldn't outrank a real budget" guard.
    """
    if not theme_spend or not median_spend or median_spend <= 0:
        return 2  # no basis to discriminate → neutral
    ratio = theme_spend / median_spend
    if ratio >= SPEND_HIGH_RATIO:
        return 3
    if ratio < SPEND_LOW_RATIO:
        return 1
    return 2


def _impact_pts(strength: str, ceiling: bool, spend_pts: int) -> int:
    """Impact (1–3): CPI-delta magnitude blended with spend-at-stake.

    A confirmed efficiency ceiling counts as at least a moderate magnitude even
    when the noisy 7-day delta reads weak - the plateau itself is the signal.
    """
    mag = _STRENGTH_PTS.get(strength, 1)
    if ceiling:
        mag = max(mag, 2)
    raw = (mag + spend_pts) / 2.0
    if raw >= IMPACT_HIGH_RAW:
        return 3
    if raw >= IMPACT_MED_RAW:
        return 2
    return 1


def _confidence_pts(strength: str, uniformity: str, ceiling: bool) -> int:
    """Confidence (1–3) from trend strength × uniformity.

    A ``uniform`` theme is trusted at its raw strength; ``mixed`` is discounted a
    notch; an ``isolated`` (execution) read rests on clear outlier separation
    rather than theme-wide strength, so it sits at a steady medium. A confirmed
    efficiency ceiling is a clear structural signal and floors confidence at
    medium.
    """
    base = _STRENGTH_PTS.get(strength, 1)
    if uniformity == "uniform":
        pts = base
    elif uniformity == "mixed":
        pts = max(1, base - 1)
    else:  # isolated
        pts = 2
    if ceiling:
        pts = max(pts, 2)
    return pts


def _ice_score(impact_pts: int, confidence_pts: int, ease_pts: int) -> int:
    """Classic multiplicative ICE (1–27) rescaled to the spec's 1–10."""
    raw = impact_pts * confidence_pts * ease_pts
    score = round(1 + (raw - 1) * 9 / 26)
    return max(1, min(10, int(score)))


# ── Narrative templates (one per hypothesis_type) ──────────────────────────

def _recommendation(htype: str, f: dict) -> str:
    cap, drivers = f["cap"], f["drivers"]
    if htype == "amplification":
        return (
            f"Scale spend on {cap} and expand {drivers} into adjacent placements - "
            f"the theme is rising uniformly with headroom and no efficiency ceiling yet."
        )
    if htype == "optimization" and f["efficiency_ceiling"]:
        winner = f["winners"][0] if f["winners"] else None
        bench = (
            f" against the {drivers} winner {winner['ad_id']} "
            f"({_cpi(winner.get('last7d_mean_cpi'))} CPI, hook {_hook(winner)})"
            if winner else ""
        )
        return (
            f"Refresh {cap}'s laggards — {_ad_list(f['laggards'])} — with net-new hooks/angles"
            f"{bench}, and hold spend flat: it has hit its efficiency ceiling, so added budget "
            f"no longer lowers CPI."
        )
    if htype == "optimization":
        cut = f" Cut or rework {f['weak_format']} first." if f["weak_format"] else ""
        return (
            f"Hold {cap} steady and monitor - it is flat with no theme-level trend, so no "
            f"structural change is indicated; optionally test one new angle against the baseline.{cut}"
        )
    if htype == "retirement":
        return (
            f"Wind down {cap} and reallocate its budget: the whole theme is declining and even "
            f"its cheapest format ({f['dominant_format'] or 'video'}) is deteriorating fastest "
            f"(mid-video drop-off), so optimization won't rescue it. Worst performers: "
            f"{_ad_list(f['laggards'])}."
        )
    if htype == "replication":
        return (
            f"Replicate the {drivers} formula from {_ad_list(f['winners'])} across the rest of "
            f"{cap} — these isolated winners run far below the theme-median CPI while the cluster "
            f"lags behind them."
        )
    if htype == "intervention":
        return (
            f"Fix the execution on {cap}'s laggards — {_ad_list(f['laggards'])}: they run well "
            f"above the theme CPI on {f['weak_where']} — swap their format/placement rather than "
            f"touching the theme strategy."
        )
    return f"Review {cap}."  # unreachable; all five types covered above


def _evidence(htype: str, f: dict) -> str:
    """Anchor on the deterministic pattern_summary, then append the precise
    metrics + ad_ids + last-week spend that ground this specific hypothesis."""
    head = (
        f"{f['pattern_summary']} (theme CPI {_pct(f['cpi_delta'])} wk/wk; "
        f"{f['ads_rising']}/{f['ads_declining']}/{f['ads_stable']} ads rising/declining/stable; "
        f"last-week spend {_money(f['theme_spend'])})."
    )
    if htype == "retirement":
        tail = f" Lagging ads: {', '.join(f['lagging_ids'])}."
    elif htype == "replication":
        tail = f" Winners to copy: {', '.join(f['winner_ids'])}; the rest of the cluster sits higher."
    elif htype == "intervention":
        tail = f" Laggards: {', '.join(f['lagging_ids'])} on {f['weak_where']}."
    elif htype == "amplification":
        tail = f" Driver: {f['drivers']}; leaders {', '.join(f['winner_ids'])}."
    else:  # optimization (ceiling or monitor)
        tail = f" Driver: {f['drivers']}."
    return head + tail


def _evidence_points(htype: str, f: dict) -> list[str]:
    """The same grounding as the ``evidence`` string, split into discrete,
    ad-specific bullet lines for the UI.

    Every line is a *settled* number from Layers 1–2 (theme CPI delta, per-ad CPI
    level, the ad's hook copy, last-week spend). What the pipeline genuinely can't
    answer — e.g. *which* creative element is saturating, or whether a net-new
    concept resets the ceiling — stays in ``open_questions`` rather than being
    fabricated here.
    """
    counts = (
        f"Theme CPI {_pct(f['cpi_delta'])} wk/wk; "
        f"{f['ads_rising']}↑ / {f['ads_declining']}↓ / {f['ads_stable']}→ ads."
    )
    spend = f"Last-week spend {_money(f['theme_spend'])}."
    winner = f["winners"][0] if f["winners"] else None
    pts: list[str] = []

    if htype == "optimization" and f["efficiency_ceiling"]:
        pts.append(
            f"Efficiency ceiling: CPI is flat ({_pct(f['cpi_delta'])} wk/wk) despite a rising "
            f"30-day spend ramp — extra budget no longer buys lower CPI."
        )
        if winner:
            pts.append(f"Winner to beat: {_ad_phrase(winner)} — hook {_hook(winner)}.")
        if f["laggards"]:
            pts.append(f"Dragging the theme mean: {_ad_list(f['laggards'])}.")
        if f["weak_format"]:
            pts.append(f"Weakest combo: {f['weak_where']}.")
        pts.append(spend)
        return pts

    if htype == "optimization":  # flat / monitor
        pts.append(f"Flat with no theme-level trend ({_pct(f['cpi_delta'])} wk/wk).")
        if winner:
            pts.append(f"Cheapest right now: {_ad_phrase(winner)}.")
        if f["laggards"]:
            pts.append(f"Most expensive: {_ad_list(f['laggards'])}.")
        pts.append(spend)
        return pts

    if htype == "retirement":
        pts.append(counts)
        pts.append(
            "Even the cheapest format (video) is deteriorating fastest → mid-video drop-off."
        )
        if f["laggards"]:
            pts.append(f"Worst performers: {_ad_list(f['laggards'])}.")
        pts.append(f"{spend} Reallocate it to a healthier theme.")
        return pts

    if htype == "replication":
        if f["winners"]:
            pts.append(f"Isolated winners far below theme median: {_ad_list(f['winners'])}.")
        if winner:
            pts.append(f"Winning formula: {f['drivers']}; e.g. {winner['ad_id']} hook {_hook(winner)}.")
        if f["laggards"]:
            pts.append(f"The rest of the cluster lags: {_ad_list(f['laggards'])}.")
        pts.append(spend)
        return pts

    if htype == "intervention":
        if f["laggards"]:
            pts.append(f"Laggards running well above theme CPI: {_ad_list(f['laggards'])}.")
        pts.append(f"They sit on {f['weak_where']} — the theme's weak combo.")
        if winner:
            pts.append(
                f"Benchmark against the theme winner {winner['ad_id']} "
                f"({_cpi(winner.get('last7d_mean_cpi'))} CPI, hook {_hook(winner)})."
            )
        pts.append(spend)
        return pts

    if htype == "amplification":
        pts.append(f"Rising uniformly with no ceiling — {counts}")
        if f["winners"]:
            pts.append(f"Driver: {f['drivers']}; leaders {_ad_list(f['winners'])}.")
        pts.append("Headroom to scale spend and widen placements.")
        pts.append(spend)
        return pts

    return [counts, spend]


def _open_questions(htype: str, f: dict) -> list[str]:
    if htype == "amplification":
        return [
            "At what daily spend does CPI start to flatten - where is the efficiency ceiling?",
            f"Do the {f['drivers']} placements saturate as budget expands, or is there room to widen them?",
        ]
    if htype == "optimization" and f["efficiency_ceiling"]:
        return [
            "Which creative element is saturating - the hook, the video length, or the format?",
            "Would a net-new concept reset the ceiling, or only refresh the existing winners?",
        ]
    if htype == "optimization":
        return [
            "What untested narrative angle could break the theme out of its flat band?",
            "Is the flatness a true plateau, or just under-spend masking real movement?",
        ]
    if htype == "retirement":
        return [
            "Is the mid-video drop-off fixable with a re-cut, or is the theme creatively exhausted?",
            "Where should the freed budget go - the rising theme or the isolated-winner theme?",
        ]
    if htype == "replication":
        return [
            "Does the winning formula hold for new hooks, or is it specific to these ads?",
            "What audience are the winners reaching that the laggards in the same theme are not?",
        ]
    if htype == "intervention":
        return [
            "Is the laggard's problem the placement, the format, or the creative itself?",
            "Would moving them onto the theme's dominant placement recover CPI?",
        ]
    return []


# ── Per-theme assembly ─────────────────────────────────────────────────────

def _facts(theme: str, sig: dict, diag: dict, theme_spend: float | None) -> dict:
    """Gather every value the templates and ICE scorer slot in for a theme."""
    return {
        "theme": theme,
        "cap": theme.capitalize(),
        "drivers": _drivers(diag),
        "weak_where": _weak_where(diag),
        "cpi_delta": sig["cpi_7d_delta_pct"],
        "ads_rising": sig["ads_rising"],
        "ads_declining": sig["ads_declining"],
        "ads_stable": sig["ads_stable"],
        "efficiency_ceiling": sig["efficiency_ceiling"],
        "dominant_format": diag["dominant_format"],
        "weak_format": diag["weak_format"],
        "pattern_summary": diag["pattern_summary"],
        "winner_ids": _ids(diag["leading_ads"]),   # cheapest in-theme = the winners
        "lagging_ids": _ids(diag["lagging_ads"]),   # most expensive = intervention targets
        "winners": diag["leading_ads"],             # full entries: CPI level + hook copy
        "laggards": diag["lagging_ads"],
        "theme_spend": theme_spend,
    }


def _build_item(htype: str, sig: dict, facts: dict, spend_pts: int) -> dict:
    """Build one backlog item (sans rank) plus its ICE points for sorting.

    For the secondary ``intervention`` item ``htype`` is forced to
    ``"intervention"`` while ``sig`` stays the theme's signal - the level and
    confidence inputs are the theme's, the action and ease are intervention's.
    """
    ceiling = sig["efficiency_ceiling"]
    impact = _impact_pts(sig["trend_strength"], ceiling, spend_pts)
    confidence = _confidence_pts(sig["trend_strength"], sig["uniformity"], ceiling)
    ease = _EASE_PTS[htype]
    item = {
        "hypothesis_type": htype,
        "level": sig["diagnosis_level"],
        "theme": facts["theme"],
        "recommendation": _recommendation(htype, facts),
        "evidence": _evidence(htype, facts),
        "evidence_points": _evidence_points(htype, facts),
        "ice_impact": _PTS_TO_LABEL[impact],
        "ice_confidence": _PTS_TO_LABEL[confidence],
        "ice_ease": _PTS_TO_LABEL[ease],
        "ice_score": _ice_score(impact, confidence, ease),
        "open_questions": _open_questions(htype, facts),
    }
    return {"item": item, "pts": (impact, confidence, ease)}


# ── Public orchestrator ────────────────────────────────────────────────────

def build_ranked_backlog(
    theme_signals: dict,
    theme_diagnosis: dict,
    ad_signals_df: pd.DataFrame,
) -> list[dict]:
    """Build the ICE-scored, ranked action backlog.

    ``ad_signals_df`` (Layer 1) is the spend source: ICE impact is grounded in
    each theme's last-week spend (``recent_spend``), per the spec's "impact =
    delta magnitude × spend level". This third argument is the one deliberate
    divergence from the spec's 2-arg signature; ``app.py`` has the frame at
    startup, so the call site stays a one-liner.

    Returns the backlog sorted by ``ice_score`` descending, with a fully
    deterministic tie-break (impact, confidence, ease, then emission order so a
    theme's primary hypothesis precedes its secondary one). Ranks are 1..N.
    """
    theme_spend = ad_signals_df.groupby("theme")["recent_spend"].sum().to_dict()
    spends = [s for s in theme_spend.values() if s and s > 0]
    median_spend = float(pd.Series(spends).median()) if spends else None

    staged: list[tuple] = []   # (sort_key, emission_index, item)
    emit = 0
    for theme in sorted(theme_signals):
        sig = theme_signals[theme]
        diag = theme_diagnosis[theme]
        facts = _facts(theme, sig, diag, theme_spend.get(theme))
        sp = _spend_pts(theme_spend.get(theme), median_spend)

        # Primary: the theme's own derived hypothesis_type.
        built = [_build_item(sig["hypothesis_type"], sig, facts, sp)]
        # Secondary: execution-level themes also get an intervention on laggards.
        if sig["diagnosis_level"] == "execution" and facts["lagging_ids"]:
            if sig["hypothesis_type"] != "intervention":
                built.append(_build_item("intervention", sig, facts, sp))

        for b in built:
            item, (imp, conf, ease) = b["item"], b["pts"]
            # Higher is better → negate for an ascending sort; emission_index is
            # the final, stable tie-break (primary emitted before secondary).
            sort_key = (-item["ice_score"], -imp, -conf, -ease)
            staged.append((sort_key, emit, item))
            emit += 1

    staged.sort(key=lambda t: (t[0], t[1]))
    backlog = []
    for rank, (_key, _emit, item) in enumerate(staged, start=1):
        item["rank"] = rank
        backlog.append(item)
    return backlog
