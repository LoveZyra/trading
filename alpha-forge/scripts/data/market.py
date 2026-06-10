"""Broad-market index environment — read each stock against ITS OWN market's index.

The macro layer (macro.py) is US-centric (VIX, US Treasuries, US data). But a Korean
stock trades with the KOSPI, an A-share with the CSI 300, a Japanese name with the
Nikkei. "Don't fight the tape" — a stock's home-market trend is a first-order driver.
This module turns a broad index into a 'market beta / 大盘' regime score in [-1, 1]
(positive = market uptrend/risk-on for that market), plus a breadth proxy, and blends
them with the (global) macro score.

Fetch the index OHLCV like any instrument:
  - US:  S&P500 ^GSPC / Nasdaq100 ^NDX (yfinance), or SPX/NDX (broker IND), or ETF SPY/QQQ
  - KR:  KOSPI ^KS11, KOSDAQ ^KQ11 (yfinance); pykrx index also works
  - JP:  Nikkei ^N225, TOPIX ^TPX (yfinance)
  - CN:  CSI300 / SSE — akshare `stock_zh_index_daily`/`index_zh_a_hist` (000300, 000001)
  - HK:  HSI ^HSI
Then `index_regime(index_close)`; route each stock with `market_of` / `index_for`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import indicators as ind

MARKET_INDEX = {
    "US": {"sp500": "^GSPC", "nasdaq100": "^NDX", "etf": "SPY/QQQ", "broker": ("SPX", "IND")},
    "KR": {"kospi": "^KS11", "kosdaq": "^KQ11"},
    "JP": {"nikkei": "^N225", "topix": "^TPX"},
    "CN": {"csi300": "000300", "sse": "000001", "akshare": "index_zh_a_hist"},
    "HK": {"hsi": "^HSI"},
}


_SUFFIX_MARKET = {"SS": "CN", "SZ": "CN", "BJ": "CN", "HK": "HK",
                  "KS": "KR", "KQ": "KR", "T": "JP"}


def market_of(symbol: str, default: str = "US") -> str:
    """Best-effort market inference from a ticker.

    Exchange suffixes ('005930.KS', '600519.SS', '0700.HK', '7203.T') are
    authoritative and checked first. Bare numeric codes are ambiguous: a 6-digit
    code starting with 0/3 could be Shenzhen OR Korea ('005930' is Samsung, not a
    Shenzhen name) -- prefer suffixed tickers or pass an explicit market where you
    know it; bare 6-digit codes are read as A-share by convention here.
    """
    s = str(symbol).upper().strip()
    if "." in s:                                  # exchange suffix wins
        suf = s.rsplit(".", 1)[1]
        if suf in _SUFFIX_MARKET:
            return _SUFFIX_MARKET[suf]
    if s.isalpha():
        return "US"
    if s.isdigit():
        if len(s) == 6 and s[0] in "0369":       # 600/000/300/688 ... A-share
            return "CN"
        if len(s) <= 5:                           # 0700, 1928 ... HK
            return "HK"
        return default
    if len(s) == 4 and (s.isdigit() or s[:3].isdigit() and s[-1].isalpha()):
        return "JP"                               # 7203, 285A ...
    return default


def index_for(market: str) -> dict:
    return MARKET_INDEX.get(str(market).upper(), {})


def index_regime(index_close: pd.Series, ma_long: int = 200, mom_lookback: int = 63) -> dict:
    """Broad-index regime score in [-1, 1] for the '大盘' overlay.

    Blends: (a) price vs long MA (trend, sign + distance), (b) medium-term momentum,
    (c) realized-vol band (calm vs stressed). Positive = market uptrend/risk-on.
    """
    c = index_close.dropna()
    if len(c) < 30:
        return {"score": 0.0, "note": "insufficient index history"}
    px = float(c.iloc[-1])
    maL = float(ind.sma(c, min(ma_long, len(c) - 1)).iloc[-1])
    trend = np.tanh((px / maL - 1) * 8) if maL else 0.0          # +distance above MA
    mom = float(c.iloc[-1] / c.iloc[-mom_lookback] - 1) if len(c) > mom_lookback else 0.0
    mom_s = np.tanh(mom * 6)
    rv = float(c.pct_change().rolling(20).std(ddof=0).iloc[-1] * np.sqrt(252)) if len(c) > 20 else 0.0
    vol_s = -np.tanh((rv - 0.15) / 0.15)                          # >15%/yr starts to drag
    score = 0.5 * trend + 0.35 * mom_s + 0.15 * vol_s
    return {"score": round(float(np.clip(score, -1, 1)), 3),
            "above_ma": bool(px > maL), "mom_3m": round(mom, 4), "ann_vol": round(rv, 3),
            "regime": ("大盘多头/risk-on" if score > 0.25 else
                       "大盘空头/risk-off" if score < -0.25 else "大盘震荡/中性")}


def market_breadth(universe_data: dict, ma: int = 50) -> dict:
    """Breadth proxy: fraction of the universe trading above its `ma`-day average.
    >0.6 healthy participation, <0.4 narrow/weak. Returns {breadth, score in [-1,1]}."""
    above = tot = 0
    for sym, df in (universe_data or {}).items():
        c = df["close"].dropna()
        if len(c) > ma:
            tot += 1
            above += int(float(c.iloc[-1]) > float(ind.sma(c, ma).iloc[-1]))
    if tot == 0:
        return {"breadth": None, "score": 0.0, "n": 0}
    b = above / tot
    return {"breadth": round(b, 3), "score": round(2 * b - 1, 3), "n": tot}


def market_overlay(index_close: pd.Series | None = None, breadth_score: float | None = None,
                   global_macro: float | None = None, weights: dict | None = None) -> dict:
    """Combine home-market index regime + breadth + (global) macro into one market-layer
    score. Use as the `market=` layer of sentiment.composite_sentiment, or show in the
    report's 📊 大盘环境 section. Only the components you pass are used."""
    w = {"index": 0.5, "breadth": 0.2, "macro": 0.3}
    if weights:
        w.update(weights)
    comp = {}
    if index_close is not None and len(index_close):
        comp["index"] = index_regime(index_close)["score"]
    if breadth_score is not None:
        comp["breadth"] = float(breadth_score)
    if global_macro is not None:
        comp["macro"] = float(global_macro)
    if not comp:
        return {"score": 0.0}
    tot = sum(w[k] for k in comp) or 1.0
    score = sum(w[k] * comp[k] for k in comp) / tot
    out = {"score": round(float(np.clip(score, -1, 1)), 3)}
    out.update({k: round(v, 3) for k, v in comp.items()})
    return out
