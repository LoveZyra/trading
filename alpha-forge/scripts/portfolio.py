"""Portfolio-level risk & concentration — the dimension single-name analysis misses.

A watchlist heavy in one theme (e.g. AI/semis/optical) can look like 15 names but be
1–2 independent bets. This module measures that and the portfolio's risk:

  * correlation_matrix / avg_correlation — how alike the holdings move.
  * effective_num_bets (Meucci, PCA) — # of UNCORRELATED bets you really hold.
  * diversification_ratio — weighted-avg vol / portfolio vol (higher = better).
  * concentration (HHI -> effective N of weights) — weight lumpiness.
  * sector_exposure — % risk/weight per sector (uses data.sectors).
  * risk_contributions — each name's share of portfolio variance.
  * portfolio_var_cvar — historical 1-day VaR & CVaR (tail loss).
  * beta_to — portfolio beta to a market index.
  * stress_test — apply a sector/market shock and read the P&L hit.
  * portfolio_health — one dict for the report's "🧩 组合体检" section.

All take a price panel (wide, index=date, cols=symbols) and optional weights
(defaults to equal-weight). Pure numpy/pandas.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _returns(panel_close: pd.DataFrame) -> pd.DataFrame:
    return panel_close.sort_index().pct_change().dropna(how="all")


def _weights(weights, cols) -> pd.Series:
    if weights is None:
        return pd.Series(1.0 / len(cols), index=cols)
    w = pd.Series(weights).reindex(cols).fillna(0.0)
    s = w.abs().sum()
    return w / s if s else w


def correlation_matrix(panel_close: pd.DataFrame, lookback: int = 120) -> pd.DataFrame:
    return _returns(panel_close).tail(lookback).corr()


def avg_correlation(panel_close: pd.DataFrame, lookback: int = 120) -> float:
    c = correlation_matrix(panel_close, lookback).values
    n = c.shape[0]
    if n < 2:
        return float("nan")
    iu = np.triu_indices(n, 1)
    return float(np.nanmean(c[iu]))


def effective_num_bets(panel_close: pd.DataFrame, weights=None, lookback: int = 120) -> float:
    """Meucci's Effective Number of Bets via PCA: decorrelate the assets, compute the
    portfolio's variance spread across principal components, ENB = exp(entropy(p)).
    ENB≈N means N independent bets; ENB≈1 means everything is really one bet."""
    r = _returns(panel_close).tail(lookback).dropna(axis=1, how="any")
    if r.shape[1] < 2:
        return 1.0
    w = _weights(weights, r.columns).values
    cov = np.cov(r.values, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 1e-12, None)
    # portfolio exposure to each principal component
    wt = vecs.T @ w
    contrib = (wt ** 2) * vals
    p = contrib / contrib.sum() if contrib.sum() > 0 else np.ones_like(contrib) / len(contrib)
    p = np.clip(p, 1e-12, None)
    enb = float(np.exp(-np.sum(p * np.log(p))))
    return round(enb, 2)


def diversification_ratio(panel_close: pd.DataFrame, weights=None, lookback: int = 120) -> float:
    r = _returns(panel_close).tail(lookback).dropna(axis=1, how="any")
    if r.shape[1] < 2:
        return 1.0
    w = _weights(weights, r.columns).values
    cov = np.cov(r.values, rowvar=False)
    vol = np.sqrt(np.diag(cov))
    port_vol = np.sqrt(w @ cov @ w)
    return round(float((w @ vol) / port_vol), 2) if port_vol > 0 else 1.0


def concentration(weights=None, cols=None) -> dict:
    """Herfindahl (HHI) of weights and its effective N (=1/HHI). Equal weight over N
    gives effective N = N; concentrated gives much less."""
    if weights is None and cols is not None:
        w = _weights(None, cols)
    else:
        w = pd.Series(weights); w = w.abs() / w.abs().sum()
    hhi = float((w ** 2).sum())
    return {"hhi": round(hhi, 4), "effective_n": round(1 / hhi, 2) if hhi else None,
            "max_weight": round(float(w.max()), 3), "top_name": w.idxmax()}


def sector_exposure(weights, sector_map_fn=None) -> dict:
    """% weight per sector. sector_map_fn(symbol)->sector (defaults to data.sectors)."""
    if sector_map_fn is None:
        from .data.sectors import sector_of as sector_map_fn
    w = pd.Series(weights); w = w.abs() / w.abs().sum()
    out: dict = {}
    for sym, wt in w.items():
        out[sector_map_fn(sym)] = out.get(sector_map_fn(sym), 0.0) + float(wt)
    return {k: round(v, 3) for k, v in sorted(out.items(), key=lambda x: -x[1])}


def risk_contributions(panel_close: pd.DataFrame, weights=None, lookback: int = 120) -> pd.Series:
    r = _returns(panel_close).tail(lookback).dropna(axis=1, how="any")
    w = _weights(weights, r.columns)
    cov = r.cov().values
    wv = w.values
    port_var = wv @ cov @ wv
    if port_var <= 0:
        return pd.Series(0.0, index=r.columns)
    mrc = cov @ wv
    rc = wv * mrc / port_var          # fraction of variance from each name
    return pd.Series(rc, index=r.columns).sort_values(ascending=False).round(3)


def portfolio_var_cvar(panel_close: pd.DataFrame, weights=None, alpha: float = 0.05,
                       lookback: int = 252) -> dict:
    r = _returns(panel_close).tail(lookback).dropna(axis=1, how="any")
    w = _weights(weights, r.columns)
    pr = (r * w).sum(axis=1)
    var = float(np.percentile(pr, 100 * alpha))
    cvar = float(pr[pr <= var].mean()) if (pr <= var).any() else var
    return {"var_1d": round(var, 4), "cvar_1d": round(cvar, 4),
            "var_pct": round(var * 100, 2), "cvar_pct": round(cvar * 100, 2),
            "alpha": alpha}


def beta_to(panel_close: pd.DataFrame, market_close: pd.Series, weights=None,
            lookback: int = 120) -> float:
    r = _returns(panel_close).tail(lookback).dropna(axis=1, how="any")
    w = _weights(weights, r.columns)
    pr = (r * w).sum(axis=1)
    m = market_close.pct_change().reindex(pr.index).dropna()
    pr = pr.reindex(m.index)
    if len(m) < 10 or m.var() == 0:
        return float("nan")
    return round(float(np.cov(pr, m)[0, 1] / m.var()), 2)


def stress_test(panel_close: pd.DataFrame, weights=None, *, market_shock: float = -0.10,
                beta=None, market_close: pd.Series | None = None,
                sector_shocks: dict | None = None, sector_map_fn=None) -> dict:
    """Estimate portfolio P&L under shocks.

    market_shock: a broad index move (e.g. -10%); applied via portfolio beta (computed
    from market_close if beta not given). sector_shocks: {sector: shock} applied to the
    weight in that sector. Returns the estimated hit(s)."""
    cols = panel_close.columns
    w = _weights(weights, cols)
    out = {}
    if beta is None and market_close is not None:
        beta = beta_to(panel_close, market_close, weights)
    if beta is not None and np.isfinite(beta):
        out["market_shock_pnl"] = round(float(beta) * market_shock, 4)
        out["assumptions"] = f"market {market_shock:+.0%} × beta {beta}"
    if sector_shocks:
        if sector_map_fn is None:
            from .data.sectors import sector_of as sector_map_fn
        hit = 0.0
        for sym, wt in w.items():
            sh = sector_shocks.get(sector_map_fn(sym), 0.0)
            hit += float(wt) * sh
        out["sector_shock_pnl"] = round(hit, 4)
    return out


def portfolio_health(panel_close: pd.DataFrame, weights=None, *,
                     market_close: pd.Series | None = None, lookback: int = 120) -> dict:
    """One-stop summary for the report's 🧩 组合体检 section."""
    cols = list(panel_close.columns)
    enb = effective_num_bets(panel_close, weights, lookback)
    out = {
        "n_names": len(cols),
        "effective_bets": enb,
        "diversification_ratio": diversification_ratio(panel_close, weights, lookback),
        "avg_correlation": round(avg_correlation(panel_close, lookback), 2),
        "concentration": concentration(weights, cols),
        "sector_exposure": sector_exposure(weights if weights is not None else
                                           {c: 1.0 for c in cols}),
        "var_cvar": portfolio_var_cvar(panel_close, weights),
        "top_risk": risk_contributions(panel_close, weights, lookback).head(5).to_dict(),
    }
    if market_close is not None:
        out["beta"] = beta_to(panel_close, market_close, weights)
        out["stress_market_-10%"] = stress_test(panel_close, weights,
                                                 market_shock=-0.10, market_close=market_close)
    # verdict
    msgs = []
    # The risk math (ENB/VaR/correlation) keeps only names with COMPLETE returns in the
    # window; a name with one missing bar is dropped. If that happens silently, the
    # figures describe a SUBSET of the book — often understating risk, since the dropped
    # names are exactly the new/illiquid ones. Surface it instead of hiding it.
    used = set(_returns(panel_close).tail(lookback).dropna(axis=1, how="any").columns)
    dropped = [c for c in cols if c not in used]
    if dropped:
        shown = ", ".join(map(str, dropped[:6])) + ("…" if len(dropped) > 6 else "")
        msgs.append(f"⚠️ 风险指标仅基于 {len(used)}/{len(cols)} 只(数据不全已剔除:{shown})"
                    f" → ENB/VaR/相关性为子集口径,可能低估风险")
    if enb < max(2, len(cols) * 0.25):
        msgs.append(f"⚠️ 有效押注仅 {enb}（共{len(cols)}只）→ 高度同质、集中风险大")
    if out["avg_correlation"] > 0.6:
        msgs.append(f"⚠️ 平均相关性 {out['avg_correlation']} 偏高 → 一起涨一起跌")
    big = out["sector_exposure"]
    if big and max(big.values()) > 0.5:
        s0 = max(big, key=big.get)
        msgs.append(f"⚠️ {s0} 板块占 {big[s0]:.0%} → 单板块押注过重")
    out["verdict"] = msgs or ["组合分散度尚可"]
    return out
