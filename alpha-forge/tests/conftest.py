"""Shared pytest fixtures + import path for the alpha-forge test suite.

Run from the skill root:   pytest -q
The suite uses only pandas/numpy/matplotlib (the always-present deps) plus monkeypatched
stand-ins for yfinance/akshare, so it runs offline with no broker or network access.
"""
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

# Make `import scripts...` work no matter where pytest is invoked from.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))


@pytest.fixture
def ohlcv():
    """A clean 400-bar canonical OHLCV frame with a mild upward drift."""
    np.random.seed(7)
    idx = pd.date_range("2020-01-01", periods=400, freq="B")
    close = pd.Series(100 * np.exp(np.cumsum(np.random.normal(0.0003, 0.012, len(idx)))), index=idx)
    return pd.DataFrame({
        "open": close.shift(1).fillna(close.iloc[0]),
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": 1e6,
    }, index=idx)


@pytest.fixture
def panel():
    """A {symbol: OHLCV} dict of 5 names over 300 bars (enough for 12-1 momentum)."""
    np.random.seed(11)
    idx = pd.date_range("2021-01-01", periods=300, freq="B")
    out = {}
    for s in ["AAA", "BBB", "CCC", "DDD", "EEE"]:
        c = 100 * np.exp(np.cumsum(np.random.normal(0.0002, 0.013, len(idx))))
        out[s] = pd.DataFrame({"open": c, "high": c * 1.01, "low": c * 0.99,
                               "close": c, "volume": 1e6}, index=idx)
    return out
