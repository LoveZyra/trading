"""Performance & risk metrics for an equity curve / return stream.

All functions take a Series of *periodic returns* (not prices) unless noted.
`periods_per_year` defaults to 252 (daily trading days); use 52 weekly, 12 monthly,
or 252*6.5*60 for 1-min bars, etc.

The headline function is `summary()` -- it returns every metric a research report
needs in one dict, matching what professional tools (quantstats, pyfolio) report.

Convention: volatility/Sharpe/Sortino use population std (ddof=0). quantstats and
pyfolio use sample std (ddof=1), so small-sample Sharpes here are marginally higher;
the difference vanishes for n >> 1. Documented here so cross-tool comparisons are
explained, not mysterious.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def to_returns(equity: pd.Series) -> pd.Series:
    return equity.pct_change().dropna()


def total_return(returns: pd.Series) -> float:
    return float((1 + returns).prod() - 1)


def cagr(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """Compound annual growth rate.

    Guards against a real footgun: annualizing a *very short* window. Raising a few
    bars' return to the ``periods_per_year/n`` power turns a +10% week into a
    >10,000,000% 'CAGR', which then poisons ``calmar`` and any mean/sort over short
    walk-forward folds. So for windows shorter than ~a month we return the plain
    (un-annualized) window return -- bounded and still sortable. Normal-length windows
    annualize exactly as before.
    """
    n = len(returns)
    if n == 0:
        return 0.0
    growth = (1 + returns).prod()
    if growth <= 0:
        return -1.0
    if n < max(5, periods_per_year // 12):   # < ~1 month of bars: don't extrapolate
        return total_return(returns)
    return float(growth ** (periods_per_year / n) - 1)


def ann_volatility(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    return float(returns.std(ddof=0) * np.sqrt(periods_per_year))


def sharpe(returns: pd.Series, rf: float = 0.0, periods_per_year: int = TRADING_DAYS) -> float:
    """Annualized Sharpe. rf is the *annual* risk-free rate."""
    if returns.std(ddof=0) == 0 or len(returns) == 0:
        return 0.0
    excess = returns - rf / periods_per_year
    return float(excess.mean() / returns.std(ddof=0) * np.sqrt(periods_per_year))


def sortino(returns: pd.Series, rf: float = 0.0, periods_per_year: int = TRADING_DAYS) -> float:
    """Like Sharpe but penalizes only downside deviation. Preferred for asymmetric
    strategies -- most practitioners trust it more than Sharpe."""
    excess = returns - rf / periods_per_year
    downside = excess.clip(upper=0)
    dd = np.sqrt((downside ** 2).mean())
    if dd == 0:
        return 0.0
    return float(excess.mean() / dd * np.sqrt(periods_per_year))


def drawdown_series(equity: pd.Series) -> pd.Series:
    """Drawdown at each point (<=0). Pass an equity curve, not returns."""
    peak = equity.cummax()
    return equity / peak - 1


def max_drawdown(equity: pd.Series) -> float:
    return float(drawdown_series(equity).min())


def calmar(returns: pd.Series, equity: pd.Series | None = None,
           periods_per_year: int = TRADING_DAYS) -> float:
    """CAGR / |max drawdown|. Rewards strategies that keep drawdowns shallow."""
    if equity is None:
        equity = (1 + returns).cumprod()
    mdd = abs(max_drawdown(equity))
    if mdd == 0:
        return 0.0
    return float(cagr(returns, periods_per_year) / mdd)


def win_rate(returns: pd.Series) -> float:
    nz = returns[returns != 0]
    return float((nz > 0).mean()) if len(nz) else 0.0


def profit_factor(returns: pd.Series, cap: float = 100.0) -> float:
    """Gross gains / gross losses. Zero-loss windows (common in short walk-forward
    segments) used to return inf, which poisoned downstream means/sorts; now they
    return NaN when there are no gains either, and `cap` (finite, sortable) when
    gains exist but losses don't."""
    gains = returns[returns > 0].sum()
    losses = -returns[returns < 0].sum()
    if losses <= 0:
        return float("nan") if gains <= 0 else float(cap)
    return float(min(gains / losses, cap))


def exposure(position: pd.Series) -> float:
    """Fraction of time with a non-zero position (time in market)."""
    return float((position != 0).mean()) if len(position) else 0.0


def summary(returns: pd.Series, equity: pd.Series | None = None,
            position: pd.Series | None = None, rf: float = 0.0,
            periods_per_year: int = TRADING_DAYS) -> dict:
    """One-stop metrics dict for a report or a backtest result."""
    returns = returns.dropna()
    if equity is None:
        equity = (1 + returns).cumprod()
    out = {
        "total_return": total_return(returns),
        "cagr": cagr(returns, periods_per_year),
        "ann_volatility": ann_volatility(returns, periods_per_year),
        "sharpe": sharpe(returns, rf, periods_per_year),
        "sortino": sortino(returns, rf, periods_per_year),
        "max_drawdown": max_drawdown(equity),
        "calmar": calmar(returns, equity, periods_per_year),
        "win_rate": win_rate(returns),
        "profit_factor": profit_factor(returns),
        "n_periods": int(len(returns)),
    }
    if position is not None:
        out["exposure"] = exposure(position)
    return out


def format_summary(s: dict) -> str:
    """Human-readable one-per-line block for printing in a report."""
    pct = {"total_return", "cagr", "ann_volatility", "max_drawdown", "win_rate", "exposure"}
    lines = []
    for k, v in s.items():
        if k in pct:
            lines.append(f"{k:>16}: {v:+.2%}")
        elif isinstance(v, float):
            lines.append(f"{k:>16}: {v:,.3f}")
        else:
            lines.append(f"{k:>16}: {v}")
    return "\n".join(lines)
