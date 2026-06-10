"""Signal tracking & calibration — does the daily report actually work?

The daily tasks emit signals/highlights/buy-zones, but nothing checks whether they pan
out. This logs each day's signals to a file and, later, scores them against realized
forward returns — turning the system from "prints a report" into "knows its own
hit-rate and recalibrates". This is the RD-Agent feedback loop applied to live signals.

Workflow:
  1. Each run: signal_tracker.log_signals([...], path) appends today's signals.
  2. Periodically: signal_tracker.evaluate(path, price_lookup) scores past signals whose
     horizon has elapsed -> hit-rate, avg forward return, calibration by signal strength.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def log_signals(records: list, path: str | Path) -> int:
    """Append signal records (list of dicts) as JSON lines. Each record SHOULD have at
    least: date (YYYY-MM-DD), symbol, signal (-1/0/1 or score), price. Optional: buy_lo,
    buy_hi, stop, target, composite_sentiment, regime, note. Returns total lines."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(path, encoding="utf-8") as f:
        return sum(1 for _ in f)


def load_log(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    df = pd.DataFrame(rows)
    if "date" in df:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def evaluate(path: str | Path, price_lookup, horizon: int = 5, today=None) -> dict:
    """Score logged signals whose `horizon` trading days have elapsed.

    price_lookup(symbol) -> a close Series indexed by date (e.g. from data.loader.load).
    For each matured signal, forward return = price[t+horizon]/price[t]-1; a 'hit' is
    a long signal (>0) with positive forward return, or a flat/short (<=0) that avoided
    a drop. Returns hit-rate, mean forward return by signal bucket, and calibration.
    """
    df = load_log(path)
    if df.empty:
        return {"n": 0, "note": "no signals logged yet"}
    today = pd.Timestamp(today) if today is not None else pd.Timestamp.now()
    cache: dict = {}
    rows = []
    for _, r in df.iterrows():
        sym, d = r.get("symbol"), r.get("date")
        sig = float(r.get("signal", 0))
        if pd.isna(d) or sym is None:
            continue
        if sym not in cache:
            try:
                cache[sym] = price_lookup(sym)
            except Exception:  # noqa: BLE001
                cache[sym] = None
        c = cache[sym]
        if c is None or len(c) == 0:
            continue
        idx = c.index.searchsorted(d)
        if idx + horizon >= len(c):
            continue                       # not matured yet
        fwd = float(c.iloc[idx + horizon] / c.iloc[idx] - 1)
        direction = np.sign(sig)
        hit = (direction > 0 and fwd > 0) or (direction <= 0 and fwd <= 0)
        rows.append({"symbol": sym, "date": d, "signal": sig, "fwd_ret": fwd,
                     "long": direction > 0, "hit": bool(hit)})
    if not rows:
        return {"n": 0, "note": "no matured signals yet (need >= horizon days)"}
    e = pd.DataFrame(rows)
    longs = e[e["long"]]
    out = {
        "n_matured": int(len(e)),
        "hit_rate": round(float(e["hit"].mean()), 3),
        "long_hit_rate": round(float(longs["hit"].mean()), 3) if len(longs) else None,
        "mean_fwd_ret": round(float(e["fwd_ret"].mean()), 4),
        "long_mean_fwd_ret": round(float(longs["fwd_ret"].mean()), 4) if len(longs) else None,
        "horizon": horizon,
    }
    # calibration: stronger signals should have higher forward returns
    if e["signal"].abs().max() > 1.5:        # continuous scores
        e["bucket"] = pd.qcut(e["signal"], min(4, e["signal"].nunique()), duplicates="drop")
        out["calibration"] = {str(k): round(float(v), 4)
                              for k, v in e.groupby("bucket", observed=True)["fwd_ret"].mean().items()}
    return out
