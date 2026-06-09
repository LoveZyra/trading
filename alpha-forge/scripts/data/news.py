"""Company news fetching -- free libraries + a Web-search hand-off, with sentiment.

Three ways in, all returning a uniform list of dicts:
  {"title", "publisher", "time" (Timestamp), "url", "summary"}

1. yfinance  -- US & global headlines (Ticker.news).
2. akshare   -- A-share news from 东方财富 (stock_news_em).
3. Web search / broker JSON hand-off -- because the WebSearch and broker MCP tools
   can only be called by Claude, not by this sandbox script. Claude searches, dumps
   the results to a JSON file, and `from_json_file` reads them. Same pattern as the
   price/fundamentals hand-off. See references/fundamentals_news.md.

Attach sentiment with sentiment.score_headlines / aggregate_sentiment.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import sentiment as _sent

NEWS_FIELDS = ["title", "publisher", "time", "url", "summary"]


def _norm(items: list[dict]) -> list[dict]:
    out = []
    for it in items:
        out.append({k: it.get(k) for k in NEWS_FIELDS})
    return out


def from_yfinance(symbol: str, limit: int = 30) -> list[dict]:
    """Latest headlines via yfinance. Handles both the old flat schema and the
    newer {'content': {...}} schema yfinance has shipped."""
    import yfinance as yf

    raw = yf.Ticker(symbol).news or []
    items = []
    for n in raw[:limit]:
        c = n.get("content", n)  # newer yfinance nests under 'content'
        ts = n.get("providerPublishTime") or c.get("pubDate")
        try:
            time = (pd.to_datetime(ts, unit="s") if isinstance(ts, (int, float))
                    else pd.to_datetime(ts))
        except Exception:  # noqa: BLE001
            time = pd.NaT
        items.append({
            "title": c.get("title") or n.get("title"),
            "publisher": (c.get("provider", {}) or {}).get("displayName") or n.get("publisher"),
            "time": time,
            "url": (c.get("canonicalUrl", {}) or {}).get("url") or n.get("link"),
            "summary": c.get("summary") or c.get("description"),
        })
    return _norm(items)


def from_akshare(symbol: str, limit: int = 50) -> list[dict]:
    """A-share company news via 东方财富 (akshare.stock_news_em)."""
    import akshare as ak

    df = ak.stock_news_em(symbol=symbol)
    rename = {"新闻标题": "title", "新闻内容": "summary", "发布时间": "time",
              "文章来源": "publisher", "新闻链接": "url"}
    df = df.rename(columns=rename)
    df["time"] = pd.to_datetime(df.get("time"), errors="coerce")
    items = df.head(limit).to_dict("records")
    return _norm(items)


def from_json_file(path: str | Path) -> list[dict]:
    """Read news Claude saved from a WebSearch / broker MCP call.

    Accepts either a list of dicts, or {"results"/"news"/"items": [...]} . Each item
    should have at least a title; other fields are mapped if present (headline/链接/
    publishedAt etc.)."""
    raw = json.loads(Path(path).read_text())
    if isinstance(raw, dict):
        for key in ("results", "news", "items", "data"):
            if key in raw and isinstance(raw[key], list):
                raw = raw[key]
                break
    items = []
    for it in raw if isinstance(raw, list) else []:
        items.append({
            "title": it.get("title") or it.get("headline") or it.get("新闻标题"),
            "publisher": it.get("publisher") or it.get("source") or it.get("文章来源"),
            "time": pd.to_datetime(it.get("time") or it.get("publishedAt")
                                   or it.get("date") or it.get("发布时间"), errors="coerce"),
            "url": it.get("url") or it.get("link") or it.get("新闻链接"),
            "summary": it.get("summary") or it.get("snippet") or it.get("description"),
        })
    return _norm(items)


def fetch(symbol: str, source: str = "yfinance", **kwargs) -> list[dict]:
    """Unified entry. source: 'yfinance' | 'akshare' | 'json'."""
    if source == "yfinance":
        return from_yfinance(symbol, **kwargs)
    if source == "akshare":
        return from_akshare(symbol, **kwargs)
    if source == "json":
        return from_json_file(symbol)
    raise ValueError(f"unknown news source {source!r}")


def fetch_with_sentiment(symbol: str, source: str = "yfinance", **kwargs):
    """Fetch news and attach per-headline + aggregate sentiment in one call.
    Returns (DataFrame_of_scored_headlines, aggregate_dict)."""
    items = fetch(symbol, source=source, **kwargs)
    scored = _sent.score_headlines(items)
    agg = _sent.aggregate_sentiment(items)
    return scored, agg
