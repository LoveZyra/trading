"""Conformal prediction intervals → position sizing.

A point forecast ("MSFT +2.1% next month") hides the only thing risk management
cares about: how wrong it might be. Split conformal prediction wraps ANY model's
predictions in a distribution-free interval with finite-sample coverage — no
normality assumption, no refitting, a handful of numpy lines. The trading use
(CPPS, arXiv:2410.16333) is then simple and defensible: **the wider the interval
relative to the prediction, the smaller the position.** A model that says "+2% ± 1%"
deserves capital; one that says "+2% ± 15%" deserves almost none, and pretending
otherwise is how confident-sounding backtests die live.

Markets are non-stationary, so plain split conformal (which assumes exchangeability)
under-covers in stress. `adaptive_alpha` implements the ACI update (Gibbs & Candès):
each step the miscoverage target is nudged by whether the last interval actually
covered — coverage self-corrects online without any distributional assumption.

Everything here is model-agnostic and CPU-trivial. Hooks:
  * models.ml_factor_backtest(conformal_alpha=0.2) gates each name's weight by
    prediction/uncertainty (calibration split inside each walk-forward fit).
  * sizing: multiply any weights row by `conviction_scale(pred, qhat)`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def split_conformal_qhat(residuals, alpha: float = 0.2) -> float:
    """The conformal quantile q̂ from calibration |residuals|.

    q̂ is the ceil((n+1)(1-alpha))/n empirical quantile of the absolute residuals on a
    CALIBRATION set the model never trained on. Then [pred - q̂, pred + q̂] covers the
    truth with probability >= 1-alpha (finite-sample, distribution-free) — as long as
    calibration and test points are exchangeable. Returns NaN when there's too little
    calibration data to say anything (caller should then not gate).
    """
    r = np.abs(np.asarray(residuals, float))
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 8:
        return float("nan")
    q = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(r, q))


def adaptive_alpha(alpha_target: float, alpha_t: float, covered: bool,
                   gamma: float = 0.02) -> float:
    """One ACI step (Gibbs & Candès 2021): alpha_{t+1} = alpha_t + gamma*(alpha_target - err_t)
    where err_t = 1 if the last interval FAILED to cover. Feed the returned alpha into the
    next `split_conformal_qhat` call. Under distribution shift the interval automatically
    widens after misses and tightens after streaks of easy coverage — long-run coverage
    tracks the target with no stationarity assumption."""
    err = 0.0 if covered else 1.0
    return float(np.clip(alpha_t + gamma * (alpha_target - err), 0.01, 0.99))


def conviction_scale(pred, qhat: float, floor: float = 0.0, cap: float = 1.0):
    """Signal-to-uncertainty position multiplier in [floor, cap]: |pred| / q̂, clipped.

    |pred| >= q̂ means even the pessimistic edge of the interval agrees on the sign —
    full size. |pred| << q̂ means the interval straddles zero — the model itself is
    telling you it doesn't know the direction; size accordingly. This is the CPPS
    prescription reduced to its honest core. Works on scalars or Series/arrays.
    NaN/invalid q̂ -> neutral 1.0 (no gating: don't fake precision you don't have)."""
    if qhat is None or not np.isfinite(qhat) or qhat <= 0:
        return pd.Series(1.0, index=pred.index) if isinstance(pred, pd.Series) else 1.0
    scale = np.clip(np.abs(pred) / qhat, floor, cap)
    return pd.Series(scale, index=pred.index) if isinstance(pred, pd.Series) else float(scale)


def calibrated_interval(pred, qhat: float) -> pd.DataFrame:
    """[lo, hi] band per name for a report's uncertainty display. NaN q̂ -> NaN band."""
    p = pd.Series(pred, dtype=float)
    if not np.isfinite(qhat):
        return pd.DataFrame({"pred": p, "lo": np.nan, "hi": np.nan})
    return pd.DataFrame({"pred": p, "lo": p - qhat, "hi": p + qhat})
