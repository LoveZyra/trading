"""Free / open data adapters: yfinance (global), akshare (A-share & HK), pykrx (Korea).

These run entirely inside the sandbox -- no broker login needed -- so they are the
default source for *backtesting*. For live/delayed quotes the IBKR MCP path
(see ibkr.py + references/data_sources.md) is preferred.

Each adapter returns the canonical OHLCV frame defined in base.py. Libraries are
imported lazily so the skill works even if only some are installed.
"""
from __future__ import annotations

import pandas as pd

from .base import validate_ohlcv


def from_yfinance(symbol: str, start: str | None = None, end: str | None = None,
                  period: str = "2y", interval: str = "1d",
                  auto_adjust: bool = True) -> pd.DataFrame:
    """US and most global tickers. e.g. 'AAPL', 'MSFT', '0700.HK', '005930.KS', '600519.SS'.

    yfinance already understands market suffixes:
      .SS / .SZ  Shanghai / Shenzhen A-shares      .HK  Hong Kong
      .KS / .KQ  KOSPI / KOSDAQ (Korea)            (none) US
    auto_adjust=True folds splits & dividends into OHLC (recommended for backtests).
    """
    import yfinance as yf

    kw = dict(interval=interval, auto_adjust=auto_adjust, progress=False)
    if start:
        df = yf.download(symbol, start=start, end=end, **kw)
    else:
        df = yf.download(symbol, period=period, **kw)
    if df is None or len(df) == 0:
        raise ValueError(f"yfinance returned no data for {symbol}")
    # yfinance may return a MultiIndex column frame for a single ticker.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)
    if "adj close" in df.columns:
        df = df.rename(columns={"adj close": "adj_close"})
    return validate_ohlcv(df, name=f"yfinance:{symbol}")


def from_akshare(symbol: str, start: str | None = None, end: str | None = None,
                 market: str = "cn", adjust: str = "qfq") -> pd.DataFrame:
    """A-share / HK via akshare (better China coverage than yfinance).

    symbol: A-share code like '600519' or '000001'; HK code like '00700'.
    market: 'cn' for mainland A-share, 'hk' for Hong Kong.
    adjust: 'qfq' forward-adjusted (前复权, recommended), 'hfq' back-adjusted, '' raw.
    yfinance-style suffixes ('.SS'/'.SZ'/'.HK') are stripped so suffixed watchlists
    work with every adapter.
    """
    import akshare as ak

    symbol = (str(symbol).upper().removesuffix(".SS").removesuffix(".SZ")
              .removesuffix(".HK"))

    s = start.replace("-", "") if start else "20150101"
    e = end.replace("-", "") if end else pd.Timestamp.today().strftime("%Y%m%d")
    if market == "hk":
        raw = ak.stock_hk_hist(symbol=symbol, period="daily",
                               start_date=s, end_date=e, adjust=adjust)
    else:
        raw = ak.stock_zh_a_hist(symbol=symbol, period="daily",
                                 start_date=s, end_date=e, adjust=adjust)
    rename = {"日期": "date", "开盘": "open", "最高": "high", "最低": "low",
              "收盘": "close", "成交量": "volume"}
    raw = raw.rename(columns=rename)
    return validate_ohlcv(raw, name=f"akshare:{symbol}")


def from_pykrx(symbol: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Korean equities via pykrx. symbol is the 6-digit code, e.g. '005930' (Samsung).
    A yfinance-style suffix ('005930.KS' / '.KQ') is stripped, so suffixed watchlists
    (which make the market unambiguous, see data/market.py) work with every adapter."""
    from pykrx import stock

    symbol = str(symbol).upper().removesuffix(".KS").removesuffix(".KQ")

    s = start.replace("-", "") if start else "20150101"
    e = end.replace("-", "") if end else pd.Timestamp.today().strftime("%Y%m%d")
    raw = stock.get_market_ohlcv(s, e, symbol)
    rename = {"시가": "open", "고가": "high", "저가": "low", "종가": "close", "거래량": "volume"}
    raw = raw.rename(columns=rename)
    raw.index.name = "date"
    return validate_ohlcv(raw, name=f"pykrx:{symbol}")
