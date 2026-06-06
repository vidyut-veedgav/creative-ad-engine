"""Nan-aware numeric primitives shared across the pipeline.

These are deliberately small, pure functions that operate on plain sequences
(lists, numpy arrays, or pandas Series) and return plain Python floats or
``None``. Keeping them framework-agnostic means every layer computes deltas,
weighted means, and trends the same way, and the behaviour around missing data
(NaN) is defined in exactly one place.

The CSVs encode missing metrics as empty strings, which pandas reads as NaN
(video-only / carousel-only / Meta-social columns are blank for the other
formats). Every helper here therefore drops NaN inputs before computing, and
returns ``None`` when there is nothing meaningful left to compute.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

# Denominators smaller than this are treated as zero to avoid blow-ups in
# percentage / normalisation math (e.g. a near-zero prior CPI mean).
_EPS = 1e-9


def _clean_pairs(
    values: Sequence[float], weights: Sequence[float] | None = None
) -> tuple[np.ndarray, np.ndarray | None]:
    """Return finite values (and aligned weights), dropping NaN/inf entries."""
    v = np.asarray(values, dtype=float)
    if weights is None:
        return v[np.isfinite(v)], None
    w = np.asarray(weights, dtype=float)
    mask = np.isfinite(v) & np.isfinite(w)
    return v[mask], w[mask]


def pct_change(recent: float | None, prior: float | None) -> float | None:
    """Percentage change from ``prior`` to ``recent``: ``(recent-prior)/prior*100``.

    Returns ``None`` if either input is missing or ``prior`` is ~0 (the change
    would be undefined / explode).
    """
    if recent is None or prior is None:
        return None
    if not (np.isfinite(recent) and np.isfinite(prior)):
        return None
    if abs(prior) < _EPS:
        return None
    return (recent - prior) / prior * 100.0


def weighted_nanmean(
    values: Sequence[float], weights: Sequence[float] | None = None
) -> float | None:
    """Weighted mean ignoring NaN value/weight pairs.

    Falls back to a plain mean when ``weights`` is None or the surviving weights
    sum to ~0. Returns ``None`` when no finite values remain.
    """
    v, w = _clean_pairs(values, weights)
    if v.size == 0:
        return None
    if w is None:
        return float(v.mean())
    total = w.sum()
    if abs(total) < _EPS:
        return float(v.mean())
    return float(np.dot(v, w) / total)


def ols_slope(
    y: Sequence[float], x: Sequence[float] | None = None
) -> float | None:
    """Ordinary-least-squares slope of ``y`` against ``x`` (per unit x).

    ``x`` defaults to ``0, 1, 2, ...`` (i.e. per-row, which is per-day here).
    NaN pairs are dropped. Returns ``None`` with fewer than two finite points or
    when ``x`` has no spread.
    """
    yv = np.asarray(y, dtype=float)
    xv = np.arange(yv.size, dtype=float) if x is None else np.asarray(x, dtype=float)
    mask = np.isfinite(yv) & np.isfinite(xv)
    yv, xv = yv[mask], xv[mask]
    if yv.size < 2:
        return None
    xmean = xv.mean()
    denom = float(((xv - xmean) ** 2).sum())
    if denom < _EPS:
        return None
    return float(((xv - xmean) * (yv - yv.mean())).sum() / denom)


def normalized_slope(
    y: Sequence[float], x: Sequence[float] | None = None
) -> float | None:
    """OLS slope expressed as a percentage of the mean level, i.e. %/unit-x.

    This makes slopes comparable across metrics on different scales (CPI in
    dollars vs spend in dollars vs rates in [0,1]). Returns ``None`` if the
    slope is undefined or the mean level is ~0.
    """
    slope = ols_slope(y, x)
    if slope is None:
        return None
    level = weighted_nanmean(y)
    if level is None or abs(level) < _EPS:
        return None
    return slope / level * 100.0
