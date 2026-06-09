"""Suggested trade levels — rule-based entry / stop / target from price structure.

Turns a chart into concrete, *reproducible* numbers a trader can act on: support &
resistance, a buy zone, an ATR-based stop, and take-profit targets with a risk/reward
ratio. These are mechanical readings of recent price structure + volatility, NOT
predictions and NOT advice — they just make "where would I buy/sell and where am I
wrong" explicit so position sizing and stops are disciplined.

Design:
  * Support/resistance from recent swing lows/highs (Donchian) + moving averages +
    Bollinger band edges — the levels everyone watches.
  * Stop from ATR (volatility-scaled) and the nearest structural level, whichever is
    tighter-but-sane. Wider stops in high-vol names, by construction.
  * Targets: the nearest resistance, plus an R-multiple target so reward is framed
    against the risk you're taking.
  * A plain-language bias that combines the trend, where price sits in its range, and
    (optionally) the latest strategy signal + regime exposure.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ind


def _nearest_below(price, levels):
    cand = [l for l in levels if np.isfinite(l) and l < price]
    return max(cand) if cand else np.nan


def _nearest_above(price, levels):
    cand = [l for l in levels if np.isfinite(l) and l > price]
    return min(cand) if cand else np.nan


def trade_levels(df: pd.DataFrame, *, signal: float | None = None,
                 regime_scale: float | None = None, atr_mult: float = 2.0,
                 rr: float = 2.0) -> dict:
    """Compute suggested levels for the most recent bar of an OHLCV frame.

    signal       : latest strategy signal in [-1,1] (optional), to colour the bias.
    regime_scale : current regime exposure multiplier in [0,1] (optional).
    atr_mult     : stop distance in ATRs below the entry.
    rr           : reward/risk multiple for the secondary target.
    Returns a dict of levels + a 'bias' string. All prices rounded to 2dp.
    """
    c = df["close"]
    price = float(c.iloc[-1])
    atr = float(ind.atr(df).iloc[-1])
    ma20 = float(ind.sma(c, 20).iloc[-1])
    ma50 = float(ind.sma(c, 50).iloc[-1])
    ma200 = float(ind.sma(c, 200).iloc[-1]) if len(c) >= 200 else np.nan
    bb = ind.bollinger(c).iloc[-1]
    don20 = ind.donchian(df, 20).iloc[-1]
    don60 = ind.donchian(df, 60).iloc[-1]
    hi252 = float(df["high"].tail(252).max())
    lo252 = float(df["low"].tail(252).min())

    support_pool = [ma20, ma50, ma200, bb["lower"], don20["lower"], don60["lower"], lo252]
    resist_pool = [bb["upper"], don20["upper"], don60["upper"], hi252]
    sup1 = _nearest_below(price, support_pool)
    sup2 = _nearest_below(sup1, support_pool) if np.isfinite(sup1) else np.nan
    res1 = _nearest_above(price, resist_pool)
    res2 = _nearest_above(res1, resist_pool) if np.isfinite(res1) else np.nan

    # Buy zone: between the nearest support and a shallow pullback (~0.5 ATR under price).
    buy_hi = round(price - 0.5 * atr, 2)
    buy_lo = round(max(sup1, price - 1.5 * atr) if np.isfinite(sup1) else price - 1.5 * atr, 2)
    if buy_lo > buy_hi:
        buy_lo, buy_hi = buy_hi, buy_lo

    entry = (buy_lo + buy_hi) / 2
    # Stop: the tighter-but-sane of ATR-stop and just under nearest support.
    atr_stop = entry - atr_mult * atr
    struct_stop = (sup1 - 0.3 * atr) if np.isfinite(sup1) else atr_stop
    stop = round(min(atr_stop, struct_stop), 2)
    risk = entry - stop
    target1 = round(res1, 2) if np.isfinite(res1) else round(entry + rr * risk, 2)
    target2 = round(entry + rr * risk, 2)
    reward_risk = round((target1 - entry) / risk, 2) if risk > 0 else np.nan

    # Bias text
    trend = "上升" if (np.isfinite(ma200) and price > ma200) else "下降/震荡"
    pctb = bb["pctb"]
    loc = ("区间上沿(偏贵)" if pctb > 0.8 else "区间下沿(偏便宜)" if pctb < 0.2 else "区间中部")
    bits = [f"趋势{trend}", f"价格处于{loc}"]
    if signal is not None:
        bits.append({1: "策略信号:多", 0: "策略信号:观望", -1: "策略信号:空"}.get(int(np.sign(signal)), f"信号{signal:.2f}"))
    if regime_scale is not None:
        bits.append(f"regime建议仓位×{regime_scale:.2f}")

    return {
        "price": round(price, 2), "atr": round(atr, 2),
        "support1": round(sup1, 2) if np.isfinite(sup1) else None,
        "support2": round(sup2, 2) if np.isfinite(sup2) else None,
        "resistance1": round(res1, 2) if np.isfinite(res1) else None,
        "resistance2": round(res2, 2) if np.isfinite(res2) else None,
        "buy_zone": [buy_lo, buy_hi],
        "stop_loss": stop,
        "target1": target1, "target2": target2,
        "reward_risk": reward_risk,
        "bias": " | ".join(bits),
    }


def format_levels(lv: dict, symbol: str = "") -> str:
    """One-line-per-field human readout for a report."""
    bz = lv["buy_zone"]
    return (f"{symbol} 现价 {lv['price']} | 建议买入区 {bz[0]}–{bz[1]} | 止损 {lv['stop_loss']} "
            f"| 目标1 {lv['target1']} 目标2 {lv['target2']} (盈亏比 {lv['reward_risk']})\n"
            f"  支撑 {lv['support1']}/{lv['support2']}  阻力 {lv['resistance1']}/{lv['resistance2']}\n"
            f"  {lv['bias']}")
