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

from dataclasses import dataclass

import pandas as pd


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
