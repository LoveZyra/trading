"""compare.py — cross-sectional comparison across multiple instruments.

The rest of alpha-forge is single-name first, but a very common question is
comparative: "SK海力士 vs 三星 vs 美光 — who's strongest, how alike do they move,
what's the relative posture?" This is the thin convenience layer for exactly that.

Nothing new is computed here — it ASSEMBLES existing pieces (indicators / metrics /
regime / portfolio) cross-sectionally into one tidy table + a correlation/ENB read +
a relative-strength ranking. Pair it with `data.ibkr.from_columnar` (broker hand-off)
to go from several get_price_history payloads to a comparison in a few lines:

    from scripts.data import ibkr
    from scripts.research import compare
    universe = {"SK海力士": ibkr.from_columnar(payload_skh),
                "三星":     ibkr.from_columnar(payload_ss),
                "美光":     ibkr.from_columnar(payload_mu)}
    cmp = compare.compare_tickers(universe, bars_per_year=52)   # weekly bars
    print(cmp["table"]); print(cmp["correlation"]); print(cmp["rs_rank"])

Works on daily OR weekly bars — pass bars_per_year (252 daily, 52 weekly) so the
return windows and vol annualization scale correctly. Mechanical readings, not advice.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import indicators as ind
from ..risk import regime as RG, portfolio as PF


def _ret(c: pd.Series, w: int) -> float:
    return float(c.iloc[-1] / c.iloc[-1 - w] - 1) if len(c) > w else float("nan")


def compare_tickers(data: dict, *, bars_per_year: int = 252) -> dict:
    """Cross-sectional posture + ranking across `data` = {name: canonical OHLCV df}.

    Returns a dict:
      table          : per-name DataFrame (last, rsi, returns 1m/1q/half/1y, ann_vol,
                       pct_from_high, above quarter/year MA, trend bull/bear)
      correlation    : DataFrame of pairwise return correlations (alike-ness)
      effective_bets : Meucci ENB across the names (≈1 ⇒ really one bet)
      rs_rank        : names ordered by trailing 1-year (or longest) return
      rs_scores      : {name: that return %}
    """
    if not data:
        raise ValueError("compare_tickers: empty universe")
    q = max(5, bars_per_year // 4)        # ~quarter
    yr = max(20, bars_per_year)           # ~year
    mo = max(3, bars_per_year // 12)      # ~month
    half = max(10, bars_per_year // 2)    # ~half year

    rows, closes = {}, {}
    for name, df in data.items():
        c = df["close"].astype(float)
        closes[name] = c
        rets = c.pct_change().dropna()
        last = float(c.iloc[-1]); peak = float(c.cummax().iloc[-1])
        smaq = ind.sma(c, q).iloc[-1]; smay = ind.sma(c, yr).iloc[-1]
        slow = min(yr, max(10, len(c) // 2))
        trd = float(RG.trend_regime(c, slow=slow).iloc[-1])
        rows[name] = {
            "last": round(last, 2),
            "rsi": round(float(ind.rsi(c, 14).iloc[-1]), 1),
            "ret_1m_%": round(_ret(c, mo) * 100, 1),
            "ret_1q_%": round(_ret(c, q) * 100, 1),
            "ret_half_%": round(_ret(c, half) * 100, 1),
            "ret_1y_%": round(_ret(c, yr) * 100, 1),
            "ann_vol_%": round(float(rets.tail(yr).std(ddof=0) * np.sqrt(bars_per_year) * 100), 1),
            "pct_from_high_%": round((last / peak - 1) * 100, 1),
            "above_q_ma": (bool(last > smaq) if np.isfinite(smaq) else None),
            "above_y_ma": (bool(last > smay) if np.isfinite(smay) else None),
            "trend": "bull" if trd > 0 else ("bear" if trd < 0 else "flat"),
        }
    table = pd.DataFrame(rows).T

    panel = pd.DataFrame({n: closes[n] for n in data})
    lookback = min(len(panel), max(20, bars_per_year))
    try:
        corr = PF.correlation_matrix(panel, lookback=lookback).round(2)
    except Exception:  # noqa: BLE001
        corr = None
    try:
        enb = PF.effective_num_bets(panel, lookback=lookback)
    except Exception:  # noqa: BLE001
        enb = float("nan")

    rs = table["ret_1y_%"].astype(float).sort_values(ascending=False)
    return {"table": table, "correlation": corr, "effective_bets": enb,
            "rs_rank": list(rs.index), "rs_scores": rs.round(1).to_dict()}
