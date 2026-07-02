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


def cost_stressed_sharpe(stats: dict, extra_bps: float = 10.0) -> float:
    """Sharpe after charging EXTRA one-way costs on the realized turnover.

    Selecting parameters on raw Sharpe quietly favors high-turnover configs whose edge
    evaporates the moment real-world costs are a few bps worse than assumed. The exact
    first-order Sharpe deterioration from `extra_bps` more cost is
    (turnover_annual * extra_bps/1e4) / ann_volatility — computable from a backtest's
    stats without re-running it. Use this as the SELECTION score (turnover-regularized
    objective, arXiv:2407.21791) while still REPORTING the raw Sharpe at true costs.
    Needs `turnover_annual` and `ann_volatility` in `stats` (the backtest engine
    provides both); falls back to the raw Sharpe when they're absent.
    """
    sh = stats.get("sharpe", float("nan"))
    if not np.isfinite(sh):
        return float("nan")
    to, vol = stats.get("turnover_annual"), stats.get("ann_volatility")
    if not to or not vol or not np.isfinite(to) or not np.isfinite(vol) or vol <= 0:
        return float(sh)
    return float(sh - (to * extra_bps / 1e4) / vol)


def downturn_slices(returns: pd.Series, benchmark_returns: pd.Series,
                    periods_per_year: int = TRADING_DAYS) -> list[dict]:
    """Strategy vs buy-and-hold on the UGLIEST stretches of the benchmark — the slices a
    cherry-picked backtest window hides.

    StockBench's core finding is that strategies (and LLM agents especially) fail
    together in down markets while looking fine on the full sample. So every report
    should carry a forced downturn slice: the benchmark's worst calendar quarter, its
    worst rolling ~quarter (63-bar) window, and its deepest drawdown episode
    (peak→trough). Returns [{label, period, strategy, benchmark, excess}] with
    percentage returns, ready for a report table. Empty list when history is too short.
    """
    r = pd.Series(returns).dropna()
    b = pd.Series(benchmark_returns).reindex(r.index).fillna(0.0)
    if len(r) < 70:
        return []
    if not isinstance(b.index, pd.DatetimeIndex):   # bar-count / Period indexes: the
        try:                                         # calendar slices below are meaningless
            b.index = r.index = pd.DatetimeIndex(pd.to_datetime(b.index))
        except Exception:  # noqa: BLE001
            return []
    out = []

    def _fmt(x):
        s = str(x)
        return s[:10] if len(s) >= 10 else s

    # Everything below is POSITIONAL (iloc): label-based .loc slicing explodes or
    # raises on duplicate dates, and duplicates do occur in stitched/backfilled data.
    def _slice_row(label, i0, i1):
        if i1 - i0 + 1 < 5:
            return
        sr = float((1 + r.iloc[i0:i1 + 1]).prod() - 1)
        br = float((1 + b.iloc[i0:i1 + 1]).prod() - 1)
        out.append({"label": label,
                    "period": f"{_fmt(b.index[i0])} ~ {_fmt(b.index[i1])}",
                    "strategy": round(sr, 4), "benchmark": round(br, 4),
                    "excess": round(sr - br, 4)})

    # 1) worst benchmark calendar quarter
    try:
        periods = pd.PeriodIndex(b.index, freq="Q")
        q = b.groupby(periods).apply(lambda x: (1 + x).prod() - 1)
        if len(q) >= 2:
            worst_q = q.idxmin()
            pos = np.where(periods == worst_q)[0]
            if len(pos):
                _slice_row(f"基准最差季度 {worst_q}", int(pos[0]), int(pos[-1]))
    except Exception:  # noqa: BLE001 — irregular index; skip the calendar slice
        pass

    # 2) worst rolling 63-bar benchmark window
    win = min(63, max(20, len(b) // 4))
    roll = (1 + b).rolling(win).apply(np.prod, raw=True) - 1
    if roll.notna().any():
        i1 = int(np.nanargmin(roll.to_numpy()))
        _slice_row(f"基准最差滚动{win}日", max(0, i1 - win + 1), i1)

    # 3) deepest benchmark drawdown episode (peak -> trough)
    eq = (1 + b).cumprod()
    dd = eq / eq.cummax() - 1
    if float(dd.min()) < 0:
        i_tr = int(np.nanargmin(dd.to_numpy()))
        i_pk = int(np.nanargmax(eq.to_numpy()[:i_tr + 1])) if i_tr > 0 else 0
        _slice_row(f"基准最深回撤段({float(dd.min()):.0%})", i_pk, i_tr)
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


# ---------------------------------------------------------------- trade excursions

def mae_mfe(trades: pd.DataFrame, prices: pd.Series) -> pd.DataFrame:
    """Per-holding MAE/MFE from a backtest's trade ledger (date/from_pos/to_pos).

    Why: Sharpe tells you IF a strategy works; excursion analysis tells you WHERE its
    stops belong. MAE (max adverse excursion, most negative % move against the entry
    while held) says how much pain winning trades typically endure — the empirical
    stop level; MFE (max favorable) says how much open profit gets given back — the
    empirical take-profit level.

    Holding segments are reconstructed positionally: each trade sets to_pos from its
    date until the next trade date (or the end of `prices`). Excursions are direction-
    signed relative to the entry bar's close, so for shorts a falling price is
    favorable. Returns one row per non-flat segment:
    {entry_date, exit_date, position, mae, mfe, ret, bars}; mae <= 0 <= mfe by
    construction (the entry bar itself is excursion 0).
    """
    cols = ["entry_date", "exit_date", "position", "mae", "mfe", "ret", "bars"]
    if trades is None or len(trades) == 0:
        return pd.DataFrame(columns=cols)
    px = pd.Series(prices).dropna().sort_index()
    t = trades.sort_values("date").reset_index(drop=True)
    rows = []
    dates, tos = list(t["date"]), list(t["to_pos"])
    for i, (d0, pos) in enumerate(zip(dates, tos)):
        if not np.isfinite(pos) or pos == 0:
            continue                       # went flat: not a holding segment
        d1 = dates[i + 1] if i + 1 < len(dates) else px.index[-1]
        path = px[(px.index >= d0) & (px.index <= d1)]
        if len(path) == 0:
            continue
        entry = float(path.iloc[0])
        if entry <= 0:
            continue
        sgn = 1.0 if pos > 0 else -1.0
        exc = sgn * (path / entry - 1.0)   # direction-signed % excursion from entry
        rows.append({"entry_date": path.index[0], "exit_date": path.index[-1],
                     "position": float(pos), "mae": float(exc.min()),
                     "mfe": float(exc.max()), "ret": float(exc.iloc[-1]),
                     "bars": int(len(path) - 1)})
    return pd.DataFrame(rows, columns=cols)


def capm_decompose(returns: pd.Series, market_returns: pd.Series, rf: float = 0.0,
                   periods_per_year: int = TRADING_DAYS) -> dict:
    """Single-factor CAPM decomposition of a return stream against a market index.

    Why: a 20% year means nothing until you know how much was beta (the market would
    have paid anyone) vs alpha (skill). Returns
    {alpha_ann, beta, corr, r2, treynor, information_ratio}:

      * alpha_ann — annualized regression intercept on excess returns;
      * beta / corr / r2 — exposure and how much of the variance the market explains;
      * treynor — annualized excess return per unit of beta (systematic risk);
      * information_ratio — mean(active return)/std(active return), annualized,
        where active = returns − market (tracking-error-adjusted skill).

    rf is the ANNUAL risk-free rate (same convention as sharpe()).
    """
    df = pd.concat([pd.Series(returns).rename("r"),
                    pd.Series(market_returns).rename("m")], axis=1).dropna()
    nan = float("nan")
    if len(df) < 3:
        return {"alpha_ann": nan, "beta": nan, "corr": nan, "r2": nan,
                "treynor": nan, "information_ratio": nan}
    rp = rf / periods_per_year
    re, me = df["r"] - rp, df["m"] - rp
    var_m = float(me.var(ddof=0))
    beta = float(((re - re.mean()) * (me - me.mean())).mean() / var_m) if var_m > 0 else nan
    alpha_ann = float((re.mean() - beta * me.mean()) * periods_per_year) if np.isfinite(beta) else nan
    corr = float(df["r"].corr(df["m"]))
    r2 = float(corr ** 2) if np.isfinite(corr) else nan
    treynor = float(re.mean() * periods_per_year / beta) if np.isfinite(beta) and abs(beta) > 1e-9 else nan
    active = df["r"] - df["m"]
    te = float(active.std(ddof=0))
    ir = float(active.mean() / te * np.sqrt(periods_per_year)) if te > 0 else 0.0
    return {"alpha_ann": alpha_ann, "beta": beta, "corr": corr, "r2": r2,
            "treynor": treynor, "information_ratio": ir}
