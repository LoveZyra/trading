"""Single entry point for loading price data, with on-disk caching.

    from data.loader import load
    df = load("AAPL", source="yfinance", start="2020-01-01")
    df = load("600519", source="akshare", market="cn")
    df = load("broker_aapl.json", source="ibkr")   # a file Claude dumped

Caching: parquet files under .cache/ keyed by the request. Re-running a backtest
doesn't re-hit the network. Delete the .cache folder to force a refresh.
Open-ended requests (no explicit `end`) additionally key the cache by today's date,
so yesterday's download can't silently serve today's session forever. Corrupt cache
files are deleted and re-downloaded instead of raising.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

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
        # Local JSON dump -> no network, nothing to cache.
        return ibkr.from_mcp_json_file(symbol, **kwargs)

    cache_kw = dict(symbol=symbol, source=source, **kwargs)
    if not kwargs.get("end"):
        # Open-ended request ("latest 2y"): without this, the first download would
        # be served forever -- a daily review would silently run on stale prices.
        cache_kw["asof"] = pd.Timestamp.today().strftime("%Y-%m-%d")
    cache = _cache_key(**cache_kw)
    if use_cache and cache.exists():
        try:
            return validate_ohlcv(pd.read_parquet(cache), name=f"cache:{symbol}")
        except Exception:  # noqa: BLE001  corrupt/partial cache -> refetch
            log.warning("corrupt cache %s; deleting and re-downloading", cache, exc_info=True)
            try:
                cache.unlink()
            except OSError:
                pass

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
            log.warning("[load_many] skipped %s: %s", s, e)
            print(f"[load_many] skipped {s}: {e}")
    return out
