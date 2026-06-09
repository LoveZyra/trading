"""Fundamental data: valuation, quality, growth metrics -- source-agnostic.

Every adapter returns a flat dict keyed by a CANONICAL set of field names, so the
factor layer never cares whether a number came from yfinance, akshare or a broker
JSON. Missing fields are simply absent (None) rather than faked -- a factor that
needs them will skip that name, which is the honest behaviour.

Canonical fields (all optional, float unless noted)
---------------------------------------------------
  symbol            : str
  name              : str
  market_cap        : market capitalization (local currency)
  pe                : trailing P/E            (lower = cheaper)  [value]
  pb                : price / book            (lower = cheaper)  [value]
  ps                : price / sales           (lower = cheaper)  [value]
  dividend_yield    : trailing dividend yield (higher = richer payout)
  roe               : return on equity        (higher = better) [quality]
  roa               : return on assets        (higher = better) [quality]
  gross_margin      : gross profit margin     (higher = better) [quality]
  net_margin        : net profit margin       (higher = better) [quality]
  debt_to_equity    : leverage                (lower = safer)   [quality, inverted]
  revenue_growth    : YoY revenue growth      (higher = better) [growth]
  earnings_growth   : YoY earnings growth     (higher = better) [growth]
  as_of             : ISO date the snapshot reflects

All ratio / margin / growth fields are stored as FRACTIONS (0.25 means 25%), normalized
across sources: yfinance is already fractional; akshare's (%) values and yfinance's
percent-scaled debtToEquity are divided by 100 on the way in, so a mixed-market panel
z-scores on one consistent scale (otherwise A-share names, at ~100x, swamp every factor).

See references/fundamentals_news.md for source quirks and the Web-search hand-off.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

CANONICAL_FIELDS = [
    "symbol", "name", "market_cap", "pe", "pb", "ps", "dividend_yield",
    "roe", "roa", "gross_margin", "net_margin", "debt_to_equity",
    "revenue_growth", "earnings_growth", "as_of",
]

# Direction each metric pushes a factor score: +1 = higher is better, -1 = lower is
# better. Used by the factor layer to sign the z-scores consistently.
FACTOR_DIRECTION = {
    "pe": -1, "pb": -1, "ps": -1, "debt_to_equity": -1,
    "roe": +1, "roa": +1, "gross_margin": +1, "net_margin": +1,
    "dividend_yield": +1, "revenue_growth": +1, "earnings_growth": +1,
}


def _f(v):
    """Coerce to float or None (drop NaN/inf/strings cleanly)."""
    try:
        x = float(v)
        if x != x or x in (float("inf"), float("-inf")):
            return None
        return x
    except (TypeError, ValueError):
        return None


def from_yfinance(symbol: str) -> dict:
    """US & global fundamentals via yfinance .info / financial statements."""
    import yfinance as yf

    t = yf.Ticker(symbol)
    info = {}
    try:
        info = t.info or {}
    except Exception:  # noqa: BLE001
        info = getattr(t, "fast_info", {}) or {}

    out = {k: None for k in CANONICAL_FIELDS}
    out.update(
        symbol=symbol,
        name=info.get("shortName") or info.get("longName"),
        market_cap=_f(info.get("marketCap")),
        pe=_f(info.get("trailingPE")),
        pb=_f(info.get("priceToBook")),
        ps=_f(info.get("priceToSalesTrailing12Months")),
        dividend_yield=_f(info.get("dividendYield")),
        roe=_f(info.get("returnOnEquity")),
        roa=_f(info.get("returnOnAssets")),
        gross_margin=_f(info.get("grossMargins")),
        net_margin=_f(info.get("profitMargins")),
        debt_to_equity=_f(info.get("debtToEquity")),
        revenue_growth=_f(info.get("revenueGrowth")),
        earnings_growth=_f(info.get("earningsGrowth")),
        as_of=pd.Timestamp.today().strftime("%Y-%m-%d"),
    )
    # yfinance reports debtToEquity as a percent (e.g. 150.0 = 1.5x); store a fraction so it
    # matches akshare's (converted) 资产负债率 and the fraction convention documented above.
    if out["debt_to_equity"] is not None:
        out["debt_to_equity"] /= 100.0
    return out


def from_akshare(symbol: str, market: str = "cn") -> dict:
    """A-share fundamentals via akshare. Combines valuation (PE/PB via
    stock_a_indicator_lg / stock_individual_info_em) with financial-statement
    ratios (ROE, margins via stock_financial_analysis_indicator). Degrades
    gracefully if any endpoint is unavailable."""
    import akshare as ak

    out = {k: None for k in CANONICAL_FIELDS}
    out["symbol"] = symbol
    out["as_of"] = pd.Timestamp.today().strftime("%Y-%m-%d")

    # Valuation snapshot (东方财富 individual info has PE/PB/总市值/名称)
    try:
        info = ak.stock_individual_info_em(symbol=symbol)
        kv = dict(zip(info["item"], info["value"]))
        out["name"] = kv.get("股票简称")
        out["market_cap"] = _f(kv.get("总市值"))
        out["pe"] = _f(kv.get("市盈率(动)") or kv.get("市盈率"))
        out["pb"] = _f(kv.get("市净率"))
    except Exception:  # noqa: BLE001
        pass

    # Financial ratios (净资产收益率, 销售毛利率, 资产负债率, 营收增长率)
    try:
        fin = ak.stock_financial_analysis_indicator(symbol=symbol)
        if fin is not None and len(fin):
            row = fin.iloc[0]  # most recent period

            def g(*names):
                for n in names:
                    if n in row.index:
                        return _f(row[n])
                return None

            def gp(*names):
                """Like g() but converts a percent (25.0) to a fraction (0.25), so akshare
                ratios share yfinance's scale. A mixed-source panel z-scores consistently."""
                v = g(*names)
                return v / 100.0 if v is not None else None

            out["roe"] = gp("净资产收益率(%)", "加权净资产收益率(%)")
            out["gross_margin"] = gp("销售毛利率(%)")
            out["net_margin"] = gp("销售净利率(%)")
            out["debt_to_equity"] = gp("资产负债率(%)")
            out["revenue_growth"] = gp("主营业务收入增长率(%)", "营业收入增长率(%)")
            out["earnings_growth"] = gp("净利润增长率(%)")
    except Exception:  # noqa: BLE001
        pass
    return out


def from_json_file(path: str | Path) -> dict:
    """Read a fundamentals snapshot that Claude saved from the broker MCP or a
    Web search (canonical keys, or close enough). Unknown keys are ignored."""
    raw = json.loads(Path(path).read_text())
    out = {k: None for k in CANONICAL_FIELDS}
    for k in CANONICAL_FIELDS:
        if k in raw:
            out[k] = _f(raw[k]) if k not in ("symbol", "name", "as_of") else raw[k]
    return out


def load(symbol: str, source: str = "yfinance", **kwargs) -> dict:
    """Unified entry. source: 'yfinance' | 'akshare' | 'json'."""
    if source == "yfinance":
        return from_yfinance(symbol)
    if source == "akshare":
        return from_akshare(symbol, **kwargs)
    if source == "json":
        return from_json_file(symbol)
    raise ValueError(f"unknown fundamentals source {source!r}")


def load_panel(symbols: list[str], source: str = "yfinance", **kwargs) -> pd.DataFrame:
    """Fundamentals for many names as a DataFrame (index=symbol, cols=canonical
    fields). This is what the multi-factor layer consumes for value/quality."""
    rows = []
    for s in symbols:
        try:
            rows.append(load(s, source=source, **kwargs))
        except Exception as e:  # noqa: BLE001
            print(f"[fundamentals] skipped {s}: {e}")
    df = pd.DataFrame(rows)
    if "symbol" in df.columns:
        df = df.set_index("symbol")
    return df
