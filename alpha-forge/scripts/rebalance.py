"""Rebalance-date helpers shared by multi_factor, models and sizing.

Why this exists: `close.resample("ME").last().index` yields CALENDAR period-end
labels (Jan 31, Feb 28, ...). On a trading-day index ~28% of those labels fall on
a weekend/holiday; code that then does `reindex(labels)` (all-NaN rows dropped) or
`[d for d in labels if d in index]` silently SKIPS those rebalances and holds the
old book an extra period. Always rebalance on the last actual trading day instead.
"""
from __future__ import annotations

import pandas as pd


def rebalance_dates(index: pd.DatetimeIndex, freq: str = "ME") -> pd.DatetimeIndex:
    """Last ACTUAL trading day of each period in `index`.

    freq: pandas offset alias -- 'ME' month-end, 'W-FRI' weekly, 'QE' quarterly.
    Every returned date is guaranteed to be a member of `index`.
    """
    s = pd.Series(index, index=index)
    last = s.resample(freq).last().dropna()
    return pd.DatetimeIndex(last.values)
