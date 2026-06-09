"""Market-regime detection & exposure scaling.

Markets are non-stationary: a trend system that prints money in a calm uptrend gets
shredded in a high-vol chop. Rather than pretend one parameter set fits all regimes,
these tools detect the current regime and scale exposure accordingly. FINSABER
(2505.07078) found that ignoring regimes is exactly why many timing strategies look
great in backtest and disappoint live; regime-aware risk controls are the fix.

Everything here is CAUSAL — a regime label at bar t uses only data up to t (volatility
and trend estimated from the past, classified against an *expanding* history so we
never use a threshold computed from the future).

  * vol_regime    — low / mid / high volatility state (the dominant risk driver).
  * trend_regime  — bull / bear via a long moving average.
  * regime_scale  — a 0..1 exposure multiplier: cut risk in high-vol/bear states.
  * cusum_changepoints — simple CUSUM break detector for regime-shift dates.

Use regime_scale to multiply a strategy's signal before backtesting; the engine still
lags it, so this stays look-ahead-free.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ind

TRADING_DAYS = 252


def vol_regime(close: pd.Series, lookback: int = 21, n_states: int = 3,
               min_history: int = 252) -> pd.Series:
    """Classify each bar's volatility into states 0..n_states-1 (0=calmest).

    Realized vol is compared to its OWN expanding quantiles, so the thresholds are
    learned only from the past — causal. Until `min_history` bars accumulate, returns
    the middle state."""
    rv = close.pct_change().rolling(lookback).std(ddof=0) * np.sqrt(TRADING_DAYS)
    qs = np.linspace(0, 1, n_states + 1)[1:-1]
    state = pd.Series(np.nan, index=close.index)
    for i in range(len(rv)):
        if i < min_history or not np.isfinite(rv.iloc[i]):
            continue
        hist = rv.iloc[:i + 1].dropna()
        thr = hist.quantile(qs).values
        state.iloc[i] = int(np.searchsorted(thr, rv.iloc[i]))
    return state.ffill().fillna((n_states - 1) / 2)


def trend_regime(close: pd.Series, slow: int = 200) -> pd.Series:
    """+1 bull (price above its slow MA), -1 bear. Causal by construction."""
    ma = ind.sma(close, slow)
    return np.sign(close - ma).fillna(0.0)


def regime_scale(close: pd.Series, *, vol_lookback: int = 21, n_states: int = 3,
                 bear_scale: float = 0.5, high_vol_scale: float = 0.4,
                 slow: int = 200) -> pd.Series:
    """Exposure multiplier in [0, 1]: full risk in calm bull regimes, reduced risk in
    high-vol and/or bear regimes. Multiply your signal by this.

    The high-vol state scales exposure toward `high_vol_scale`; a bear trend scales
    toward `bear_scale`; they compound. This is a risk overlay, not a forecast —
    it shrinks the book when the environment is hostile."""
    vs = vol_regime(close, vol_lookback, n_states)
    # map vol state 0..n-1 to a multiplier from 1.0 (calm) down to high_vol_scale (wild)
    frac = vs / max(1, (n_states - 1))
    vol_mult = 1.0 - frac * (1.0 - high_vol_scale)
    tr = trend_regime(close, slow)
    trend_mult = np.where(tr < 0, bear_scale, 1.0)
    return (vol_mult * pd.Series(trend_mult, index=close.index)).clip(0.0, 1.0).fillna(0.0)


def apply_regime(signal: pd.Series, close: pd.Series, **kw) -> pd.Series:
    """Convenience: scale a target-position signal by the regime overlay."""
    return signal * regime_scale(close, **kw).reindex(signal.index).fillna(0.0)


def cusum_changepoints(close: pd.Series, threshold: float = 5.0) -> list:
    """Detect regime-shift dates with a symmetric CUSUM filter on log returns
    (Page 1954). Returns the dates where cumulative drift exceeds `threshold` std —
    handy for marking when a strategy's assumptions may have broken. Causal: each
    flag uses only prior returns."""
    r = np.log(close).diff().fillna(0.0)
    sd = r.expanding(min_periods=20).std(ddof=0).replace(0, np.nan)
    z = (r / sd).fillna(0.0)
    s_pos = s_neg = 0.0
    points = []
    for dt, x in z.items():
        s_pos = max(0.0, s_pos + x)
        s_neg = min(0.0, s_neg + x)
        if s_pos > threshold or s_neg < -threshold:
            points.append(dt)
            s_pos = s_neg = 0.0
    return points


def gmm_regime(close: pd.Series, n_states: int = 3, lookback: int = 504) -> dict:
    """Statistical regime via a Gaussian mixture on (return, vol) features — a pragmatic
    'HMM-lite' richer than the vol+trend heuristic. Needs scikit-learn (optional). Labels
    states by mean return so 0=worst..n-1=best, and returns the current state + its stats.
    Falls back to vol_regime if sklearn is unavailable."""
    try:
        from sklearn.mixture import GaussianMixture
    except Exception:  # noqa: BLE001
        st = int(vol_regime(close, n_states=n_states).iloc[-1])
        return {"state": st, "n_states": n_states, "method": "vol_regime (sklearn absent)"}
    r = close.pct_change()
    feat = pd.DataFrame({"r": r, "vol": r.rolling(10).std()}).dropna().tail(lookback)
    if len(feat) < 50:
        return {"state": None, "note": "insufficient history"}
    gm = GaussianMixture(n_components=n_states, covariance_type="full", random_state=0)
    lab = gm.fit_predict(feat.values)
    order = np.argsort([gm.means_[k, 0] for k in range(n_states)])    # by mean return
    remap = {old: new for new, old in enumerate(order)}
    cur = remap[int(lab[-1])]
    means = {remap[k]: round(float(gm.means_[k, 0]) * 252, 3) for k in range(n_states)}
    return {"state": cur, "n_states": n_states, "ann_ret_by_state": means,
            "regime": ("熊/risk-off" if cur == 0 else "牛/risk-on" if cur == n_states - 1 else "震荡"),
            "method": "GaussianMixture"}


def hedge_suggestion(portfolio_beta: float | None = None, macro_score: float | None = None,
                     market_regime: float | None = None) -> dict:
    """When the environment is risk-off, suggest a hedge size. Combines portfolio beta
    (how exposed you are) with macro/market risk scores (how hostile it is). Returns a
    suggested hedge ratio (fraction of book to neutralize) + plain-language actions.
    NOT advice — a mechanical risk overlay."""
    risk = 0.0
    n = 0
    for x in (macro_score, market_regime):
        if x is not None:
            risk += -float(x); n += 1               # negative score => risk-off => + risk
    risk = risk / n if n else 0.0
    beta = float(portfolio_beta) if portfolio_beta is not None else 1.0
    hedge_ratio = float(np.clip(risk, 0, 1) * np.clip(beta, 0.5, 2.0) / 2.0)
    actions = []
    if hedge_ratio < 0.1:
        actions.append("环境中性偏多，无需对冲，按 regime 仓位持有")
    else:
        actions.append(f"建议对冲约 {hedge_ratio:.0%} 敞口：可减仓高beta标的 / 配置反向ETF(如SH、SQQQ)或买入看跌期权保护")
        if beta > 1.3:
            actions.append(f"组合 beta {beta:.1f} 偏高，risk-off 时跌幅会放大，优先降高beta头寸")
    return {"hedge_ratio": round(hedge_ratio, 2), "env_risk": round(risk, 2),
            "beta": round(beta, 2), "actions": actions}
