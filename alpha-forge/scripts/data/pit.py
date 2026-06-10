"""Point-in-time (PIT) snapshot store — free path to honest historical factor backtests.

Free data gives only the LATEST fundamentals/sentiment, so backtesting them is a
look-ahead approximation (see fundamentals_news.md §5). The fix without buying a PIT
database: snapshot today's values every day. Over time you accumulate a dated archive
that IS point-in-time going forward. The daily tasks call save_snapshot(); backtests
read it as a {date: panel} history.

Layout: <base>/fundamentals/YYYY-MM-DD.json, <base>/sentiment/YYYY-MM-DD.json, etc.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

DEFAULT_BASE = "pit_store"


def save_snapshot(date, *, fundamentals_panel=None, sentiment_by_symbol=None,
                  macro=None, market=None, base: str = DEFAULT_BASE) -> dict:
    """Persist today's snapshot. `date` like '2026-06-09'. Returns the paths written."""
    base = Path(base)
    d = pd.Timestamp(date).strftime("%Y-%m-%d")
    written = {}
    if fundamentals_panel is not None and len(fundamentals_panel):
        p = base / "fundamentals"; p.mkdir(parents=True, exist_ok=True)
        fp = p / f"{d}.json"
        fundamentals_panel.reset_index().to_json(fp, orient="records", force_ascii=False)
        written["fundamentals"] = str(fp)
    if sentiment_by_symbol:
        p = base / "sentiment"; p.mkdir(parents=True, exist_ok=True)
        fp = p / f"{d}.json"; fp.write_text(json.dumps(sentiment_by_symbol, ensure_ascii=False), encoding="utf-8")
        written["sentiment"] = str(fp)
    if macro is not None or market is not None:
        p = base / "env"; p.mkdir(parents=True, exist_ok=True)
        fp = p / f"{d}.json"; fp.write_text(json.dumps({"macro": macro, "market": market}, ensure_ascii=False, default=float), encoding="utf-8")
        written["env"] = str(fp)
    return written


def load_pit_sentiment(base: str = DEFAULT_BASE) -> dict:
    """{date(Timestamp): {symbol: sentiment}} from accumulated snapshots."""
    p = Path(base) / "sentiment"
    out = {}
    if p.exists():
        for f in sorted(p.glob("*.json")):
            out[pd.Timestamp(f.stem)] = json.loads(f.read_text(encoding="utf-8"))
    return out


def load_pit_fundamentals(base: str = DEFAULT_BASE) -> dict:
    """{date(Timestamp): fundamentals_panel(DataFrame, index=symbol)}."""
    p = Path(base) / "fundamentals"
    out = {}
    if p.exists():
        for f in sorted(p.glob("*.json")):
            df = pd.read_json(f, dtype={"symbol": str})
            if "symbol" in df.columns:
                # Keep leading zeros on A-share / HK codes (000063, 00700): never let the
                # symbol be parsed as an int, or the join to string tickers silently fails.
                df["symbol"] = df["symbol"].astype(str)
                df = df.set_index("symbol")
            out[pd.Timestamp(f.stem)] = df
    return out


def asof_sentiment(history: dict, date) -> dict:
    """The most recent snapshot AT OR BEFORE `date` — the value you'd actually have known.
    This is what makes a PIT backtest honest (no peeking at a future snapshot)."""
    date = pd.Timestamp(date)
    keys = sorted(k for k in history if k <= date)
    return history[keys[-1]] if keys else {}


def pit_coverage(base: str = DEFAULT_BASE) -> dict:
    """How much history you've accumulated so far (so you know when PIT backtests get
    meaningful)."""
    out = {}
    for kind in ("fundamentals", "sentiment", "env"):
        p = Path(base) / kind
        files = sorted(p.glob("*.json")) if p.exists() else []
        out[kind] = {"days": len(files),
                     "from": files[0].stem if files else None,
                     "to": files[-1].stem if files else None}
    return out
