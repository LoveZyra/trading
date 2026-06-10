"""Risk-based position sizing — turn raw target weights into risk-aware ones.

Equal-weighting ignores that a 40%-vol name and a 12%-vol name contribute wildly
different risk. These functions re-weight so risk, not dollars, is what's balanced:

  * inverse_vol_weights   — weight ∝ 1/volatility (simple, robust risk parity).
  * risk_parity_weights   — equal *risk contribution* (ERC) via Newton/CCD iteration,
                            accounting for correlations (Choi & Chen 2022).
  * vol_target_scale      — scale the whole book so realized portfolio vol hits a
                            target (e.g. 10%/yr), the lever that makes drawdowns more
                            uniform across regimes.

All operate on a wide weights panel (index=date, cols=symbols) plus the price panel,
and are causal (use only trailing data). Feed the result to backtest_portfolio.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _trailing_vol(panel_close: pd.DataFrame, lookback: int = 63) -> pd.DataFrame:
    return panel_close.pct_change().rolling(lookback).std(ddof=0) * np.sqrt(TRADING_DAYS)


def inverse_vol_weights(weights: pd.DataFrame, panel_close: pd.DataFrame,
                        lookback: int = 63) -> pd.DataFrame:
    """Re-scale each date's nonzero weights by 1/trailing-vol, renormalized to keep the
    same gross exposure. Low-vol names get more capital, high-vol less."""
    vol = _trailing_vol(panel_close, lookback).reindex_like(weights)
    inv = 1.0 / vol.replace(0, np.nan)
    sized = weights.where(weights == 0, weights.abs() * inv) * np.sign(weights)
    # renormalize per row to original gross
    gross_old = weights.abs().sum(axis=1).replace(0, np.nan)
    gross_new = sized.abs().sum(axis=1).replace(0, np.nan)
    scale = (gross_old / gross_new).fillna(0.0)
    return sized.mul(scale, axis=0).fillna(0.0)


def risk_parity_weights(panel_close: pd.DataFrame, universe: list | None = None,
                        lookback: int = 126, rebalance_index: pd.DatetimeIndex | None = None,
                        max_iter: int = 200, tol: float = 1e-8) -> pd.DataFrame:
    """Equal-risk-contribution (ERC) long-only weights, recomputed on each rebalance
    date from the trailing covariance. Each asset contributes equal risk to the
    portfolio — the textbook risk-parity portfolio. Uses cyclical coordinate descent.

    Returns a weights panel forward-filled between rebalances.
    """
    close = panel_close[universe] if universe else panel_close
    rets = close.pct_change()
    if rebalance_index is not None:
        idx = [d for d in rebalance_index if d in close.index]
    else:
        # last actual trading day per month -- calendar 'ME' labels silently skip
        # ~28% of months on a trading-day index (see scripts/rebalance.py)
        from .rebalance import rebalance_dates
        idx = rebalance_dates(close.index, "ME")
    cols = close.columns
    out = pd.DataFrame(0.0, index=close.index, columns=cols)

    for t in idx:
        window = rets.loc[:t].tail(lookback).dropna(how="all", axis=1)
        if len(window) < lookback // 2 or window.shape[1] < 2:
            continue
        cov = window.cov().values
        names = window.columns
        w = _erc_solve(cov, max_iter=max_iter, tol=tol)
        out.loc[t, names] = w
    return out.reindex(close.index).replace(0.0, np.nan).ffill().fillna(0.0)


def _erc_solve(cov: np.ndarray, max_iter: int = 200, tol: float = 1e-8) -> np.ndarray:
    """Solve equal-risk-contribution weights for covariance `cov` (long-only, sum=1)
    via cyclical coordinate descent (Choi & Chen 2022, Spinu)."""
    n = cov.shape[0]
    w = np.ones(n) / n
    vol = np.sqrt(w @ cov @ w)
    for _ in range(max_iter):
        w_old = w.copy()
        for i in range(n):
            # marginal risk excluding i
            ci = cov[i] @ w - cov[i, i] * w[i]
            # solve quadratic: cov_ii*w_i^2 + ci*w_i - vol^2/n = 0
            a, b, c = cov[i, i], ci, -vol**2 / n
            w[i] = (-b + np.sqrt(b*b - 4*a*c)) / (2*a) if a > 0 else w[i]
        w = np.clip(w, 1e-8, None)
        w /= w.sum()
        vol = np.sqrt(w @ cov @ w)
        if np.abs(w - w_old).max() < tol:
            break
    return w


def vol_target_scale(returns: pd.Series, target_vol: float = 0.10,
                     lookback: int = 21, max_leverage: float = 2.0) -> pd.Series:
    """Per-bar leverage multiplier so trailing realized vol tracks `target_vol`
    (annualized). Causal: uses vol estimated up to the *previous* bar. Apply by
    multiplying a strategy's position/weights by this series (lagged by the engine).
    """
    realized = returns.rolling(lookback).std(ddof=0) * np.sqrt(TRADING_DAYS)
    scale = (target_vol / realized.shift(1)).clip(upper=max_leverage)
    # Warm-up: no vol estimate yet -> neutral 1.0 (unscaled), NOT 0.0. fillna(0)
    # forced the whole book flat for the first `lookback` bars of every backtest /
    # walk-forward fold.
    return scale.fillna(1.0)
