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

from ..core import indicators as ind

TRADING_DAYS = 252


def vol_regime(close: pd.Series, lookback: int = 21, n_states: int = 3,
               min_history: int = 252) -> pd.Series:
    """Classify each bar's volatility into states 0..n_states-1 (0=calmest).

    Realized vol is compared to its OWN expanding quantiles, so the thresholds are
    learned only from the past — causal. Until `min_history` bars accumulate, returns
    the middle state."""
    rv = close.pct_change().rolling(lookback).std(ddof=0) * np.sqrt(TRADING_DAYS)
    qs = np.linspace(0, 1, n_states + 1)[1:-1]
    # Vectorized: one expanding-quantile column per threshold, then count how many
    # thresholds today's vol exceeds (== the old searchsorted state). The previous
    # per-bar dropna()+quantile() loop was O(n^2) and a hotspot under autoresearch.
    thr_cols = [rv.expanding(min_periods=1).quantile(q) for q in qs]
    st = sum((rv > th).astype(float) for th in thr_cols)
    state = pd.Series(np.nan, index=close.index)
    mask = rv.notna().to_numpy()
    mask[:min_history] = False
    state[mask] = st[mask]
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


def regime_conditional_weights(member_returns: pd.DataFrame, close: pd.Series, *,
                               n_states: int = 3, vol_lookback: int = 21, slow: int = 200,
                               halflife: int = 63, min_obs: int = 20,
                               temp: float = 1.0) -> pd.DataFrame:
    """Per-date ensemble-member weights conditioned on the CURRENT market regime.

    The StockBench failure mode is that everything that worked in the up-leg fails
    together in the down-leg. A flat (unconditional) performance-weighted ensemble
    can't fix that: the members that dominated the bull run keep their high weights
    into the bear. This weights each member by its EWMA Sharpe measured ONLY on past
    bars of the SAME regime state (vol tercile × bull/bear), so when the market flips
    risk-off, the weights flip to whichever members historically coped with risk-off.

    Causal by construction: the stat at bar t uses same-state bars strictly before t
    (within-state shift). Until a state has `min_obs` observations, members get equal
    weight there. Returns a weights frame (rows sum to 1) — multiply member signals by
    it and sum for the ensemble signal.
    """
    rets = member_returns.astype(float).fillna(0.0)
    px = close.reindex(rets.index).ffill()
    vs = vol_regime(px, vol_lookback, n_states).reindex(rets.index).ffill()
    tr = trend_regime(px, slow).reindex(rets.index).fillna(0.0)
    state = vs.fillna((n_states - 1) / 2) * 2 + (tr < 0).astype(float)

    sh = pd.DataFrame(np.nan, index=rets.index, columns=rets.columns)
    ann = np.sqrt(TRADING_DAYS)
    for s in pd.unique(state.dropna()):
        m = (state == s)
        if int(m.sum()) < 3:
            continue
        seq = rets[m]
        mu = seq.ewm(halflife=halflife, min_periods=min_obs).mean()
        sd = seq.ewm(halflife=halflife, min_periods=min_obs).std()
        val = (mu / sd.replace(0.0, np.nan)) * ann
        sh.loc[m] = val.shift(1).values          # within-state shift -> strictly past info
    # NaN sharpe (unseen state / flat member) = neutral 0 INSIDE the softmax; filling after
    # normalization inflated row sums past 1 -> hidden leverage (P1 fix, 2026-07-02).
    z = np.exp((sh.fillna(0.0) / max(temp, 1e-6)).clip(-10, 10))
    w = z.div(z.sum(axis=1), axis=0)
    return w.fillna(1.0 / max(rets.shape[1], 1))


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


# ========================================================================
# Round10 §2.13:per-stock drift regime gate(因子级,与 Round9 的集成级
# regime_conditional_weights 互补——那个按"市场"状态调成员权重,这个按
# "个股自身"漂移状态开/关它的因子值)。
# ========================================================================

def stock_drift_regime(close: pd.Series, *, window: int = 63,
                       threshold: float = 0.6) -> pd.Series:
    """Per-stock drift state: 1 when the fraction of up days over the trailing
    `window` exceeds `threshold`, else 0 — returned AFTER shift(1), so the label
    at bar t only uses closes through t-1 and is safe to act on same-day.

    Why: cross-sectional momentum factors implicitly assume the stock's return
    process has persistent drift; on stocks that are mean-reverting/choppy the
    same factor value is noise. Gating by a cheap drift proxy (up-day fraction)
    lets the factor speak only where its assumption plausibly holds.

    Provenance caveat (read before trusting): the idea comes from
    arXiv 2511.12490, a SINGLE-AUTHOR, NON-PEER-REVIEWED preprint whose
    self-reported Sharpe > 13 is not credible. Treat the mechanism as a
    hypothesis only — the (window, threshold) pair MUST be walk-forward
    sensitivity-tested on your own data before the gate goes anywhere near
    production. Warmup (< window bars) is labeled 0 (non-drift) — conservative:
    a stock with too little history never activates the gate.
    """
    up = (close.pct_change() > 0).astype(float)
    frac = up.rolling(window, min_periods=window).mean()
    state = (frac > threshold).astype(float)            # NaN warmup -> False -> 0
    return state.shift(1).fillna(0.0)


def drift_regime_gate(factor_panel: pd.DataFrame, data: dict, *, window: int = 63,
                      threshold: float = 0.6, activate_in: str = "drift") -> pd.DataFrame:
    """Per-stock factor gate (§2.13): keep a stock's factor value only in the
    regime where the factor's premise holds; set it to NaN otherwise so the name
    simply DROPS OUT of the cross-sectional ranking instead of polluting it with
    a score whose meaning has flipped.

      activate_in="drift"     -> keep values where stock_drift_regime == 1
      activate_in="non_drift" -> keep values where stock_drift_regime == 0
                                 (e.g. for reversal factors, which want chop)

    Stocks absent from `data` (or without a close column) have an unknowable
    state -> their whole column becomes NaN (conservative: unknown regime never
    trades). Causality: the state series is already shift(1)-ed inside
    stock_drift_regime, so bar t's gate uses information through t-1 only.

    Provenance caveat: mechanism from arXiv 2511.12490 — single-author,
    unreviewed, self-reported Sharpe > 13 (not credible). Do NOT lift its
    parameters; walk-forward the (window, threshold) sensitivity yourself.
    """
    if activate_in not in ("drift", "non_drift"):
        raise ValueError(f"activate_in must be 'drift' or 'non_drift', got {activate_in!r}")
    out = factor_panel.astype(float).copy()
    for s in factor_panel.columns:
        df = data.get(s)
        if df is None or "close" not in getattr(df, "columns", []):
            out[s] = np.nan
            continue
        st = stock_drift_regime(df["close"], window=window,
                                threshold=threshold).reindex(factor_panel.index)
        keep = (st == 1.0) if activate_in == "drift" else (st == 0.0)
        out[s] = out[s].where(keep)                     # st==NaN -> False -> NaN
    return out
