"""Mean-reversion & statistical-arbitrage templates.

These bet that stretched prices snap back. High win rate, small average wins, and
the ever-present tail risk that "stretched" becomes "trending". Always cost- and
drawdown-test these -- they trade often.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import indicators as ind
from .base import Strategy, positions_from_signals


class ZScoreReversion(Strategy):
    """Fade rolling z-score extremes. Short when price is z>+entry above its mean,
    long when z<-entry below, exit back toward the mean (|z| < exit)."""
    name = "zscore_reversion"

    def __init__(self, lookback: int = 20, entry: float = 1.5, exit: float = 0.5,
                 allow_short: bool = True):
        self.lookback, self.entry, self.exit, self.allow_short = lookback, entry, exit, allow_short

    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        z = ind.zscore(df["close"], self.lookback)
        # Long and short legs as separate entry/exit state machines. Exits are
        # ONE-SIDED (long exits once z recovers to >= -exit, short once z falls to
        # <= +exit) rather than |z|<=exit: a gap straight from one extreme to the
        # other then closes the old leg on the same bar the new one opens, instead
        # of leaving both legs alive and netting the position to 0.
        pos = positions_from_signals(z <= -self.entry, z >= -self.exit, 1.0)
        if self.allow_short:
            pos = pos + positions_from_signals(z >= self.entry, z <= self.exit, -1.0)
        return pos


class BollingerReversion(Strategy):
    """Buy at/below the lower Bollinger band, exit at the mid band; mirror for shorts."""
    name = "bollinger_reversion"

    def __init__(self, n: int = 20, k: float = 2.0, allow_short: bool = True):
        self.n, self.k, self.allow_short = n, k, allow_short

    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        b = ind.bollinger(df["close"], self.n, self.k)
        c = df["close"]
        # Long and short legs built independently, then added. Long entry
        # (c<=lower) implies the short exit (c<=mid) and short entry (c>=upper)
        # implies the long exit (c>=mid), so the legs can't overlap. The old
        # flat-mask version zeroed shorts as soon as mid<=c<upper -- shorts barely
        # lasted one bar instead of covering at the mid band.
        pos = positions_from_signals(c <= b["lower"], c >= b["mid"], 1.0)
        if self.allow_short:
            pos = pos + positions_from_signals(c >= b["upper"], c <= b["mid"], -1.0)
        return pos


class RSIReversion(Strategy):
    """Long when RSI < oversold, exit when RSI recovers past `exit_level`."""
    name = "rsi_reversion"

    def __init__(self, n: int = 14, oversold: float = 30, overbought: float = 70,
                 exit_level: float = 50, allow_short: bool = False):
        self.n, self.oversold, self.overbought = n, oversold, overbought
        self.exit_level, self.allow_short = exit_level, allow_short

    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        r = ind.rsi(df["close"], self.n)
        # Separate long/short state machines (see positions_from_signals). The old
        # single-mask version zeroed every r<=exit_level bar in short mode, which
        # erased all long entries (r<=oversold) -- the strategy degenerated to
        # short-only. Long entry implies short exit and vice versa, so adding the
        # legs is safe.
        pos = positions_from_signals(r <= self.oversold, r >= self.exit_level, 1.0)
        if self.allow_short:
            pos = pos + positions_from_signals(r >= self.overbought, r <= self.exit_level, -1.0)
        return pos


def pair_spread(a: pd.Series, b: pd.Series, lookback: int = 60):
    """Hedge-ratio spread for a cointegration / pairs trade.

    Returns (spread, zscore). Long the spread = long A, short hedge*B. Rolling OLS
    hedge ratio keeps it adaptive. Run a cointegration test (see references) before
    trusting any pair.
    """
    # Guard: if B is flat over the window (A-share halt / consecutive limit days),
    # var()==0 would make hedge inf and poison the whole spread/z-score. NaN instead:
    # the z-score is NaN there, so no signal fires -- the safe behaviour.
    b_var = b.rolling(lookback).var().replace(0, np.nan)
    hedge = a.rolling(lookback).cov(b) / b_var
    spread = a - hedge * b
    z = ind.zscore(spread, lookback)
    return spread, z


class PairsTrading(Strategy):
    """Trade the z-score of a two-leg spread. `generate_signal` returns the position
    on leg A; the caller takes -hedge*that on leg B. Provide the partner series at
    construction (aligned to the same index as the df passed to generate_signal)."""
    name = "pairs_trading"

    def __init__(self, partner_close: pd.Series, lookback: int = 60,
                 entry: float = 2.0, exit: float = 0.5):
        self._partner = partner_close
        self.lookback, self.entry, self.exit = lookback, entry, exit

    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        b = self._partner.reindex(df.index).ffill()
        _, z = pair_spread(df["close"], b, self.lookback)
        # One-sided exits per leg (see ZScoreReversion): a gap across the band
        # flips the position instead of netting to 0.
        pos = positions_from_signals(z <= -self.entry, z >= -self.exit, 1.0)
        pos = pos + positions_from_signals(z >= self.entry, z <= self.exit, -1.0)
        return pos
