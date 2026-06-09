"""A-share microstructure — price limits (涨跌停), T+1, no-short. Make CN backtests real.

The US backtest rules don't fit A-shares: ±10% (±20% ChiNext/STAR, ±5% ST) daily price
limits mean you often CAN'T fill at the limit; T+1 means you can't sell what you bought
today; most names can't be shorted. Ignoring these flatters CN backtests. Apply these to
a signal/series before backtesting CN names.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LIMIT = {"main": 0.10, "chinext": 0.20, "star": 0.20, "st": 0.05, "bj": 0.30}


def limit_for(symbol: str) -> float:
    s = str(symbol)
    if s.startswith(("300", "301")):  # ChiNext
        return LIMIT["chinext"]
    if s.startswith(("688", "689")):  # STAR
        return LIMIT["star"]
    if s.startswith(("8", "4")):       # Beijing
        return LIMIT["bj"]
    return LIMIT["main"]


def limit_blocked(df: pd.DataFrame, limit: float = 0.10) -> pd.Series:
    """Days where the close is at/through the prior-close ±limit (can't reliably transact
    — limit-up has no sellers to buy from is the opposite; here we flag |move|>=limit*0.98
    as a non-tradable bar for the side that's locked)."""
    prev = df["close"].shift(1)
    move = df["close"] / prev - 1
    up = move >= limit * 0.98
    down = move <= -limit * 0.98
    return pd.DataFrame({"limit_up": up.fillna(False), "limit_down": down.fillna(False)})


def apply_cn_rules(signal: pd.Series, df: pd.DataFrame, *, symbol: str | None = None,
                   limit: float | None = None, allow_short: bool = False) -> pd.Series:
    """Adjust a target-position signal for A-share rules:
    - no shorting (clip >=0) unless allow_short;
    - can't INCREASE position on a limit-up bar (no sellers), can't DECREASE on limit-down
      (no buyers) -> hold prior position on locked bars.
    Returns the rule-adjusted signal; feed it to backtest(... lag=1) as usual (lag handles
    T+1: you act next bar, so same-day buy isn't sold same day)."""
    lim = limit if limit is not None else (limit_for(symbol) if symbol else 0.10)
    sig = signal.copy()
    if not allow_short:
        sig = sig.clip(lower=0.0)
    lb = limit_blocked(df, lim).reindex(sig.index).fillna(False)
    out = sig.copy()
    prev = 0.0
    for i, t in enumerate(sig.index):
        cur = sig.iloc[i]
        if lb["limit_up"].iloc[i] and cur > prev:    # can't add into limit-up
            cur = prev
        if lb["limit_down"].iloc[i] and cur < prev:  # can't cut into limit-down
            cur = prev
        out.iloc[i] = cur
        prev = cur
    return out
