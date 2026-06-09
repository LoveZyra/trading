"""Unified OHLCV data contract.

Every data source in this skill returns a pandas DataFrame with the SAME shape so
that indicators, the backtest engine and strategies never need to know where the
data came from. This is the single most important design decision in the skill:
one canonical format -> everything downstream is source-agnostic.

Canonical contract
------------------
- index: pandas.DatetimeIndex, tz-naive, sorted ascending, no duplicates
- columns (lowercase): open, high, low, close, volume
- optional extra column: adj_close  (split/dividend adjusted close)
- one row per bar, no gaps filled with NaN
- dtype: float64 for prices, int64/float64 for volume

Keeping the contract strict means a look-ahead bug or a misaligned index gets
caught here, at the boundary, instead of silently poisoning a backtest.
"""
from __future__ import annotations

import pandas as pd

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


def validate_ohlcv(df: pd.DataFrame, *, name: str = "data") -> pd.DataFrame:
    """Coerce and validate a DataFrame against the canonical OHLCV contract.

    Raises ValueError with a clear message if the frame can't be made to comply.
    Returns a cleaned copy (sorted, de-duplicated, correct dtypes).
    """
    if df is None or len(df) == 0:
        raise ValueError(f"{name}: empty dataset")

    df = df.copy()
    df.columns = [str(c).lower().strip() for c in df.columns]

    # Map a few common aliases so adapters can be sloppy.
    aliases = {
        "date": None, "datetime": None, "time": None, "timestamp": None,
        "vol": "volume", "adj close": "adj_close", "adjclose": "adj_close",
    }
    for src, dst in aliases.items():
        if src in df.columns and dst and dst not in df.columns:
            df = df.rename(columns={src: dst})

    # Establish a DatetimeIndex.
    if not isinstance(df.index, pd.DatetimeIndex):
        for cand in ("date", "datetime", "time", "timestamp"):
            if cand in df.columns:
                df = df.set_index(cand)
                break
    df.index = pd.to_datetime(df.index, utc=False, errors="coerce")
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index.name = "date"

    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing required columns {missing}; got {list(df.columns)}")

    for c in OHLCV_COLUMNS + (["adj_close"] if "adj_close" in df.columns else []):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    keep = OHLCV_COLUMNS + (["adj_close"] if "adj_close" in df.columns else [])
    df = df[keep]

    df = df[~df.index.isna()]
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=["close"])

    if len(df) == 0:
        raise ValueError(f"{name}: no valid rows after cleaning")
    return df


def from_columnar(payload: dict, *, name: str = "data") -> pd.DataFrame:
    """Build an OHLCV frame from the columnar JSON the IBKR-style MCP returns.

    The brokerage `get_price_history` tool returns parallel arrays:
        {"time": [...], "open": [...], "high": [...], "low": [...],
         "close": [...], "volume": [...]}
    `time` entries are ISO-8601 strings. See references/data_sources.md.
    """
    cols = {k: payload[k] for k in ("time", "open", "high", "low", "close", "volume") if k in payload}
    if "time" not in cols:
        raise ValueError(f"{name}: columnar payload has no 'time' array")
    df = pd.DataFrame(cols).rename(columns={"time": "date"})
    return validate_ohlcv(df, name=name)
