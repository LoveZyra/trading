"""Vectorized single-asset backtest engine.

Design choices and WHY they matter
-----------------------------------
* **Signal lag (the #1 source of fake profits).** A strategy outputs a target
  position for bar *t* using data available *at* bar *t*. You cannot trade on
  that until the *next* bar. So the engine shifts the position by `lag` (default 1)
  before computing returns. This single line is what separates an honest backtest
  from a look-ahead fantasy.
* **Costs are real.** Commission (per-trade bps on traded notional) and slippage
  (bps on each fill) are charged on every change in position. Ignoring them makes
  high-turnover strategies look amazing and lose money live. An optional square-root
  market-impact term (cost_model='sqrt') makes large/!high-turnover books pay more.
* **Returns, not prices.** We compound bar returns weighted by the (lagged)
  position. Works for long-only (0..1), long/short (-1..1) and leverage (>1).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import metrics as M


def _validate_bt_args(lag, commission_bps, slippage_bps, periods_per_year, cost_model):
    """Fail fast on nonsense arguments instead of silently producing fiction."""
    if lag < 0:
        raise ValueError(f"lag must be >= 0, got {lag} (negative lag = trading the past)")
    if commission_bps < 0 or slippage_bps < 0:
        raise ValueError("commission_bps / slippage_bps must be non-negative")
    if periods_per_year <= 0:
        raise ValueError(f"periods_per_year must be positive, got {periods_per_year}")
    if cost_model not in ("linear", "sqrt"):
        raise ValueError(f"cost_model must be 'linear' or 'sqrt', got {cost_model!r}")


@dataclass
class BacktestResult:
    equity: pd.Series           # cumulative equity, starts at 1.0
    returns: pd.Series          # per-bar strategy returns (net of costs)
    position: pd.Series         # realized position each bar (already lagged)
    signal: pd.Series           # raw target position before lag
    trades: pd.DataFrame        # one row per position change
    stats: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return self.stats

    def __repr__(self) -> str:
        return "BacktestResult(\n" + M.format_summary(self.stats) + "\n)"


def backtest(prices: pd.DataFrame | pd.Series,
             signal: pd.Series,
             *,
             lag: int = 1,
             commission_bps: float = 1.0,
             slippage_bps: float = 1.0,
             periods_per_year: int = M.TRADING_DAYS,
             allow_short: bool = True,
             cost_model: str = "linear",
             impact_coef: float = 10.0,
             capital: float = 1e6) -> BacktestResult:
    """Run a vectorized backtest.

    prices : OHLCV DataFrame (uses 'close') or a close Series.
    signal : target position per bar in [-1, 1] (or beyond for leverage).
             Computed from info available AT that bar; the engine handles the lag.
    lag    : bars between signal and execution. 1 = trade on next bar's close.
    commission_bps / slippage_bps : cost per unit traded notional, in basis points.
    cost_model : 'linear' (flat bps on turnover, the default) or 'sqrt' (square-root
             market-impact: extra cost proportional to sqrt(participation), where
             participation is traded notional vs. the bar's dollar volume). The sqrt
             law (Almgren; Bugaenko 2020) is the empirically standard impact shape and
             stops high-turnover/large-size strategies from looking free. Needs a
             'volume' column; falls back to linear for a price-only Series.
    impact_coef : strength of the sqrt impact, in bps per unit sqrt(participation).
    capital : notional book size to translate fractional turnover into a participation
             rate against dollar volume (only used by cost_model='sqrt').
    """
    _validate_bt_args(lag, commission_bps, slippage_bps, periods_per_year, cost_model)
    close = prices["close"] if isinstance(prices, pd.DataFrame) else prices
    close = close.astype(float)

    if cost_model == "sqrt" and not (isinstance(prices, pd.DataFrame) and "volume" in prices.columns):
        warnings.warn("cost_model='sqrt' needs a 'volume' column; falling back to "
                      "linear costs (market impact NOT charged)", stacklevel=2)

    signal = signal.reindex(close.index).astype(float).fillna(0.0)
    if not allow_short:
        signal = signal.clip(lower=0.0)

    # Execution lag: today's signal becomes tomorrow's position.
    position = signal.shift(lag).fillna(0.0)

    bar_ret = close.pct_change().fillna(0.0)
    gross = position * bar_ret

    # Trading costs on every change in position.
    turnover = position.diff().abs().fillna(position.abs())
    cost_rate = (commission_bps + slippage_bps) / 1e4
    costs = turnover * cost_rate

    # Optional square-root market-impact on top of the linear bps. participation =
    # traded notional / bar dollar-volume; impact_bps = impact_coef * sqrt(participation).
    if cost_model == "sqrt" and isinstance(prices, pd.DataFrame) and "volume" in prices.columns:
        dollar_vol = (close * prices["volume"]).replace(0, np.nan)
        traded_notional = turnover * capital
        participation = (traded_notional / dollar_vol).clip(lower=0).fillna(0.0)
        impact = (impact_coef / 1e4) * np.sqrt(participation)
        costs = costs + turnover * impact

    net = gross - costs
    equity = (1 + net).cumprod()

    # Trade ledger: each bar where position changes.
    changes = position.diff().fillna(position)
    tr_idx = changes[changes != 0].index
    trades = pd.DataFrame({
        "date": tr_idx,
        "price": close.reindex(tr_idx).values,
        "from_pos": position.shift(1).reindex(tr_idx).fillna(0.0).values,
        "to_pos": position.reindex(tr_idx).values,
        "cost": costs.reindex(tr_idx).values,
    }).reset_index(drop=True)

    stats = M.summary(net, equity=equity, position=position,
                      periods_per_year=periods_per_year)
    stats["n_trades"] = int(len(trades))
    stats["total_costs"] = float(costs.sum())
    stats["turnover_annual"] = float(turnover.sum() / len(turnover) * periods_per_year) if len(turnover) else 0.0

    return BacktestResult(equity=equity, returns=net, position=position,
                          signal=signal, trades=trades, stats=stats)


def backtest_portfolio(panel_close: pd.DataFrame,
                       weights: pd.DataFrame,
                       *,
                       lag: int = 1,
                       commission_bps: float = 1.0,
                       slippage_bps: float = 1.0,
                       periods_per_year: int = M.TRADING_DAYS,
                       cost_model: str = "linear",
                       panel_volume: pd.DataFrame | None = None,
                       impact_coef: float = 10.0,
                       capital: float = 1e6) -> BacktestResult:
    """Backtest a multi-asset weight panel (e.g. from multi_factor_signal).

    panel_close : wide close-price frame (index=date, cols=symbols).
    weights     : target weight per asset per bar, same shape. Rows need not sum to 1.
    Same lag + cost discipline as the single-asset engine, applied per asset then
    aggregated. The 'position' on the result is gross exposure (sum |w|).
    cost_model='sqrt' adds per-asset square-root market impact (needs `panel_volume`,
    a wide share-volume frame aligned to panel_close); see the single-asset engine.
    The result's `trades` is a per-rebalance ledger (date, names changed, turnover, cost).
    """
    _validate_bt_args(lag, commission_bps, slippage_bps, periods_per_year, cost_model)
    if cost_model == "sqrt" and panel_volume is None:
        warnings.warn("cost_model='sqrt' needs panel_volume; falling back to linear "
                      "costs (market impact NOT charged)", stacklevel=2)
    panel_close = panel_close.sort_index()
    # ffill so a rebalance-only weight panel (rows only on rebalance dates, NaN between)
    # HOLDS its positions until the next rebalance instead of silently going flat. Dense
    # per-bar panels are unaffected (no gaps to fill); leading NaNs -> 0.0 (start flat).
    weights = weights.reindex_like(panel_close).ffill().fillna(0.0)

    asset_ret = panel_close.pct_change().fillna(0.0)
    held = weights.shift(lag).fillna(0.0)

    gross = (held * asset_ret).sum(axis=1)
    asset_turnover = held.diff().abs()
    asset_turnover.iloc[0] = held.abs().iloc[0]
    turnover = asset_turnover.sum(axis=1)
    cost_rate = (commission_bps + slippage_bps) / 1e4
    costs = turnover * cost_rate
    if cost_model == "sqrt" and panel_volume is not None:
        dollar_vol = (panel_close * panel_volume.reindex_like(panel_close)).replace(0, np.nan)
        participation = (asset_turnover * capital / dollar_vol).clip(lower=0).fillna(0.0)
        impact = (impact_coef / 1e4) * np.sqrt(participation)
        costs = costs + (asset_turnover * impact).sum(axis=1)
    net = gross - costs
    equity = (1 + net).cumprod()

    exposure = held.abs().sum(axis=1)
    n_trades = int((turnover > 1e-9).sum())

    # Per-rebalance ledger (the old version always returned an empty frame).
    reb_idx = turnover[turnover > 1e-9].index
    ledger = pd.DataFrame({
        "date": reb_idx,
        "n_changed": (asset_turnover.loc[reb_idx] > 1e-9).sum(axis=1).values,
        "turnover": turnover.reindex(reb_idx).values,
        "cost": costs.reindex(reb_idx).values,
    }).reset_index(drop=True)

    stats = M.summary(net, equity=equity, position=exposure,
                      periods_per_year=periods_per_year)
    stats["n_trades"] = n_trades
    stats["total_costs"] = float(costs.sum())
    stats["turnover_annual"] = float(turnover.sum() / len(turnover) * periods_per_year) if len(turnover) else 0.0

    return BacktestResult(equity=equity, returns=net, position=exposure,
                          signal=exposure, trades=ledger, stats=stats)


def buy_and_hold(prices: pd.DataFrame | pd.Series,
                 periods_per_year: int = M.TRADING_DAYS) -> BacktestResult:
    """Benchmark: always long. Every strategy should be compared against this."""
    close = prices["close"] if isinstance(prices, pd.DataFrame) else prices
    sig = pd.Series(1.0, index=close.index)
    return backtest(close, sig, lag=0, commission_bps=0, slippage_bps=0,
                    periods_per_year=periods_per_year)
