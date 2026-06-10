"""A-share microstructure — price limits (涨跌停), T+1, no-short. Make CN backtests real.

The US backtest rules don't fit A-shares: ±10% (±20% ChiNext/STAR, ±5% ST) daily price
limits mean you often CAN'T fill at the limit; T+1 means you can't sell what you bought
today; most names can't be shorted. Ignoring these flatters CN backtests. Apply these to
a signal/series before backtesting CN names.
"""
from __future__ import annotations

import pandas as pd

LIMIT = {"main": 0.10, "chinext": 0.20, "star": 0.20, "st": 0.05, "bj": 0.30}


def limit_for(symbol: str, *, is_st: bool = False) -> float:
    """Daily price-limit fraction for an A-share. ST / *ST names are ±5% but that can't be
    inferred from the numeric code (the ST/*ST marker lives in the name), so pass
    is_st=True for them. ChiNext/STAR ±20%, Beijing ±30%, everything else ±10%."""
    s = str(symbol)
    if is_st:                          # ST / *ST: ±5% (caller must flag; the code can't tell)
        return LIMIT["st"]
    if s.startswith(("300", "301")):  # ChiNext
        return LIMIT["chinext"]
    if s.startswith(("688", "689")):  # STAR
        return LIMIT["star"]
    if s.startswith(("8", "4")):       # Beijing
        return LIMIT["bj"]
    return LIMIT["main"]


def limit_blocked(df: pd.DataFrame, limit: float = 0.10) -> pd.DataFrame:
    """Limit-up/down flags per bar.

    limit_up / limit_down: the bar CLOSED at/through prior-close ±limit (sealed at
    the close -- you can't buy a sealed limit-up / sell a sealed limit-down).
    touched_up / touched_down: the intraday HIGH/LOW reached the limit price even if
    the seal broke later -- fills are possible but unreliable; useful as a softer
    warning flag. (0.98 tolerance absorbs rounding of the exchange's limit price.)"""
    prev = df["close"].shift(1)
    up_px, down_px = prev * (1 + limit), prev * (1 - limit)
    sealed_up = df["close"] >= prev * (1 + limit * 0.98)
    sealed_down = df["close"] <= prev * (1 - limit * 0.98)
    touched_up = df["high"] >= up_px * 0.998 if "high" in df else sealed_up
    touched_down = df["low"] <= down_px * 1.002 if "low" in df else sealed_down
    return pd.DataFrame({"limit_up": sealed_up.fillna(False),
                         "limit_down": sealed_down.fillna(False),
                         "touched_up": touched_up.fillna(False),
                         "touched_down": touched_down.fillna(False)})


def apply_cn_rules(signal: pd.Series, df: pd.DataFrame, *, symbol: str | None = None,
                   limit: float | None = None, allow_short: bool = False,
                   is_st: bool = False, lag: int = 1) -> pd.Series:
    """Adjust a target-position signal for A-share rules:
    - no shorting (clip >=0) unless allow_short;
    - can't INCREASE position on a limit-up bar (no sellers), can't DECREASE on limit-down
      (no buyers) -> hold prior position on locked bars.
    Returns the rule-adjusted signal; feed it to backtest(... lag=1) as usual (lag handles
    T+1: you act next bar, so same-day buy isn't sold same day). Pass the SAME `lag`
    here as you pass to backtest() so limit checks look at the execution bar."""
    lim = limit if limit is not None else (limit_for(symbol, is_st=is_st) if symbol else 0.10)
    sig = signal.copy()
    if not allow_short:
        sig = sig.clip(lower=0.0)
    lb = limit_blocked(df, lim).reindex(sig.index, fill_value=False)
    if lag:
        # The trade implied by the signal at t executes at t+lag, so what matters is
        # whether the EXECUTION bar is locked. Align the limit flags to the signal
        # bar; the engine's own shift(lag) then re-aligns everything to execution.
        lb = lb.shift(-lag)
        lb = lb.where(lb.notna(), False).astype(bool)
    # State-dependent (each bar's allowed position depends on the previous bar's
    # realized one), so it can't be fully vectorized -- but looping over plain numpy
    # arrays instead of .iloc is 10-50x faster and exact.
    vals = sig.to_numpy(dtype=float, copy=True)
    up = lb["limit_up"].to_numpy(dtype=bool)
    down = lb["limit_down"].to_numpy(dtype=bool)
    prev = 0.0
    for i in range(len(vals)):
        cur = vals[i]
        if up[i] and cur > prev:      # can't add into limit-up
            cur = prev
        if down[i] and cur < prev:    # can't cut into limit-down
            cur = prev
        vals[i] = cur
        prev = cur
    return pd.Series(vals, index=sig.index)
