"""Single entry point for loading price data, with on-disk caching.

    from data.loader import load
    df = load("AAPL", source="yfinance", start="2020-01-01")
    df = load("600519", source="akshare", market="cn")
    df = load("broker_aapl.json", source="ibkr")   # a file Claude dumped

Caching: parquet files under .cache/ keyed by the request. Re-running a backtest
doesn't re-hit the network. Delete the .cache folder to force a refresh.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from . import free, ibkr
from .base import validate_ohlcv

CACHE_DIR = Path(__file__).resolve().parent.parent.parent / ".cache"


def _cache_key(**kw) -> Path:
    raw = "|".join(f"{k}={v}" for k, v in sorted(kw.items()) if v is not None)
    h = hashlib.md5(raw.encode()).hexdigest()[:16]
    return CACHE_DIR / f"{kw.get('symbol','x')}_{kw.get('source','x')}_{h}.parquet".replace("/", "_")


def load(symbol: str, source: str = "yfinance", *, use_cache: bool = True,
         **kwargs) -> pd.DataFrame:
    """Load OHLCV for one instrument from the requested source.

    source: 'yfinance' | 'akshare' | 'pykrx' | 'ibkr'
    kwargs are forwarded to the underlying adapter (start, end, period, interval,
    market, adjust, ...). For source='ibkr', `symbol` is the path to a JSON file
    Claude dumped from the brokerage MCP.
    """
    if source == "ibkr":
        return ibkr.from_mcp_json_file(symbol, **kwargs)

    cache = _cache_key(symbol=symbol, source=source, **kwargs)
    if use_cache and cache.exists():
        return validate_ohlcv(pd.read_parquet(cache), name=f"cache:{symbol}")

    if source == "yfinance":
        df = free.from_yfinance(symbol, **kwargs)
    elif source == "akshare":
        df = free.from_akshare(symbol, **kwargs)
    elif source == "pykrx":
        df = free.from_pykrx(symbol, **kwargs)
    else:
        raise ValueError(f"unknown source {source!r}; use yfinance|akshare|pykrx|ibkr")

    if use_cache:
        CACHE_DIR.mkdir(exist_ok=True)
        try:
            df.to_parquet(cache)
        except Exception:
            pass  # parquet engine optional; caching is best-effort
    return df


def load_many(symbols: list[str], source: str = "yfinance", **kwargs) -> dict[str, pd.DataFrame]:
    """Load several instruments; returns {symbol: frame}. Skips ones that error."""
    out = {}
    for s in symbols:
        try:
            out[s] = load(s, source=source, **kwargs)
        except Exception as e:  # noqa: BLE001
            print(f"[load_many] skipped {s}: {e}")
    return out
