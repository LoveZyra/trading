"""Trend / momentum strategy templates.

These bet that moves persist. They tend to have low win rates but large average
wins -- the opposite shape to mean-reversion. Compare both on the same asset.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import indicators as ind
from .base import Strategy


class MACrossover(Strategy):
    """Long when fast MA > slow MA. Optionally short the other side.

    The canonical trend system. Whipsaws in range-bound markets -- pairing it with
    an ADX trend filter (see adx) cuts the worst of that.
    """
    name = "ma_crossover"

    def __init__(self, fast: int = 20, slow: int = 50, allow_short: bool = False,
                 ma: str = "ema"):
        self.fast, self.slow, self.allow_short, self.ma = fast, slow, allow_short, ma

    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        f = getattr(ind, self.ma)(df["close"], self.fast)
        s = getattr(ind, self.ma)(df["close"], self.slow)
        long = (f > s).astype(float)
        if self.allow_short:
            return long * 2 - 1            # {0,1} -> {-1,+1}
        return long


class Breakout(Strategy):
    """Donchian channel breakout (Turtle-style).

    Enter long on a new n-bar high, exit on a new exit-bar low. Trades the fat
    tail of sustained trends; expect long flat periods between them.
    """
    name = "breakout"

    def __init__(self, entry: int = 20, exit: int = 10, allow_short: bool = False):
        self.entry, self.exit, self.allow_short = entry, exit, allow_short

    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        hi = df["high"].rolling(self.entry).max()
        lo = df["low"].rolling(self.exit).min()
        pos = pd.Series(np.nan, index=df.index)
        pos[df["close"] >= hi.shift(1)] = 1.0       # break prior high -> long
        pos[df["close"] <= lo.shift(1)] = 0.0       # break exit low  -> flat
        if self.allow_short:
            lo_s = df["low"].rolling(self.entry).min()
            hi_s = df["high"].rolling(self.exit).max()
            pos[df["close"] <= lo_s.shift(1)] = -1.0
            pos[df["close"] >= hi_s.shift(1)] = 0.0
        return pos.ffill().fillna(0.0)


class TimeSeriesMomentum(Strategy):
    """Long if trailing n-bar return is positive (else flat/short).

    The classic 'TSMOM' factor (Moskowitz, Ooi, Pedersen 2012). A trend filter on
    the asset's own past return -- simple and surprisingly robust across markets.
    """
    name = "ts_momentum"

    def __init__(self, lookback: int = 90, allow_short: bool = False):
        self.lookback, self.allow_short = lookback, allow_short

    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        mom = df["close"] / df["close"].shift(self.lookback) - 1
        long = (mom > 0).astype(float)
        if self.allow_short:
            return np.sign(mom).fillna(0.0)
        return long


class MACDTrend(Strategy):
    """Long while MACD histogram is positive (fast EMA above signal line)."""
    name = "macd_trend"

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9,
                 allow_short: bool = False):
        self.fast, self.slow, self.signal, self.allow_short = fast, slow, signal, allow_short

    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        h = ind.macd(df["close"], self.fast, self.slow, self.signal)["hist"]
        long = (h > 0).astype(float)
        if self.allow_short:
            return np.sign(h).fillna(0.0)
        return long


def with_trend_filter(signal: pd.Series, df: pd.DataFrame, adx_n: int = 14,
                      adx_min: float = 20.0) -> pd.Series:
    """Zero out a signal when ADX says the market isn't trending. Helps trend
    systems avoid death-by-a-thousand-whipsaws in choppy regimes."""
    strength = ind.adx(df, adx_n)
    return signal.where(strength >= adx_min, 0.0)
