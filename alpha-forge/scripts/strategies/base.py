"""Strategy base class.

A Strategy is a thin object that turns an OHLCV frame into a *signal* Series
(target position per bar, in [-1, 1]). It deliberately knows nothing about costs,
lagging or metrics -- the backtest engine owns all of that. This separation means
the same strategy runs unchanged on any market or data source, and can be reused
to generate a LIVE signal (just take the last row).

Subclasses implement `generate_signal(df) -> pd.Series`. Keep them causal: never
use information from a future bar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def positions_from_signals(entries: pd.Series, exits: pd.Series, value: float = 1.0) -> pd.Series:
    """Build a stateful position series from boolean entry/exit masks.

    Entry bars set the position to `value`, exit bars set it to 0, everything in
    between forward-fills the last state. Entries win when an entry and an exit fire
    on the same bar. NaN comparisons evaluate False, so indicator warm-up periods
    yield no signal (position 0) by construction.

    This is THE safe way to express "enter at X, exit at Y" rules: writing
    overlapping boolean masks straight into one Series (long entry, long exit, short
    entry, short exit) lets a later mask silently overwrite an earlier one -- the
    bug class where allow_short=True erased every long entry. Build the long and
    short legs separately with this helper, then add them.
    """
    pos = pd.Series(np.nan, index=entries.index)
    pos[exits.fillna(False).astype(bool)] = 0.0
    pos[entries.fillna(False).astype(bool)] = value
    return pos.ffill().fillna(0.0)


class Strategy:
    name = "strategy"

    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    def latest_signal(self, df: pd.DataFrame) -> float:
        """The target position implied by the most recent bar -- this is what you
        act on for live trading. Returns a float in [-1, 1]."""
        sig = self.generate_signal(df)
        return float(sig.iloc[-1])

    def params(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def __repr__(self) -> str:
        p = ", ".join(f"{k}={v}" for k, v in self.params().items())
        return f"{self.__class__.__name__}({p})"
