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
  * Every price level is annotated with its signed % move vs the current price
    (`pct`): positive = 上涨空间, negative = 回撤幅度.
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


def _safe_round(v, nd: int = 2):
    """round() for maybe-NaN levels: finite -> rounded float, else None."""
    return round(float(v), nd) if v is not None and np.isfinite(v) else None


def _pct(level, price):
    """Signed % move from current price to `level` (+涨幅 / −跌幅). None if N/A."""
    if level is None or not np.isfinite(level) or not price:
        return None
    return round((level / price - 1.0) * 100.0, 2)


def trade_levels(df: pd.DataFrame, *, signal: float | None = None,
                 regime_scale: float | None = None, atr_mult: float = 2.0,
                 rr: float = 2.0) -> dict:
    """Compute suggested levels for the most recent bar of an OHLCV frame.

    signal       : latest strategy signal in [-1,1] (optional), to colour the bias.
    regime_scale : current regime exposure multiplier in [0,1] (optional).
    atr_mult     : stop distance in ATRs below the entry.
    rr           : reward/risk multiple for the secondary target.
    Returns a dict of levels + per-level % moves (`pct`) + a 'bias' string.
    All prices rounded to 2dp.
    """
    if len(df) < 60:
        raise ValueError(f"trade_levels needs >=60 bars of history, got {len(df)} "
                         "(Donchian-60 / MA50 / ATR would all be NaN)")
    c = df["close"]
    price = float(c.iloc[-1])
    atr = float(ind.atr(df).iloc[-1])
    if not np.isfinite(atr) or not np.isfinite(price) or price <= 0:
        raise ValueError("trade_levels: price/ATR not computable from the given frame")
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

    sup1, sup2 = _safe_round(sup1), _safe_round(sup2)
    res1, res2 = _safe_round(res1), _safe_round(res2)

    # 每个价位相对现价的涨/跌幅(%)，正=上涨空间，负=回撤幅度
    pct = {
        "support1": _pct(sup1, price), "support2": _pct(sup2, price),
        "resistance1": _pct(res1, price), "resistance2": _pct(res2, price),
        "buy_low": _pct(buy_lo, price), "buy_high": _pct(buy_hi, price),
        "stop_loss": _pct(stop, price),
        "target1": _pct(target1, price), "target2": _pct(target2, price),
    }

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
        "support1": sup1, "support2": sup2,
        "resistance1": res1, "resistance2": res2,
        "buy_zone": [buy_lo, buy_hi],
        "stop_loss": stop,
        "target1": target1, "target2": target2,
        "reward_risk": reward_risk,
        "pct": pct,                       # 各价位相对现价涨跌幅(%)
        "bias": " | ".join(bits),
    }


def _fp(v):
    """Format a signed percent like '+8.7%' / '-19.0%' / '—'."""
    return "—" if v is None else f"{v:+.1f}%"


def format_levels(lv: dict, symbol: str = "") -> str:
    """One-line-per-field human readout for a report, each price annotated with its
    move vs current price (+涨幅 / −跌幅)."""
    bz = lv["buy_zone"]
    p = lv.get("pct", {})
    return (f"{symbol} 现价 {lv['price']} | 建议买入区 {bz[0]}({_fp(p.get('buy_low'))})–{bz[1]}({_fp(p.get('buy_high'))}) "
            f"| 止损 {lv['stop_loss']}({_fp(p.get('stop_loss'))}) "
            f"| 目标1 {lv['target1']}({_fp(p.get('target1'))}) 目标2 {lv['target2']}({_fp(p.get('target2'))}) "
            f"(盈亏比 {lv['reward_risk']})\n"
            f"  支撑 {lv['support1']}({_fp(p.get('support1'))})/{lv['support2']}({_fp(p.get('support2'))})  "
            f"阻力 {lv['resistance1']}({_fp(p.get('resistance1'))})/{lv['resistance2']}({_fp(p.get('resistance2'))})\n"
            f"  {lv['bias']}")
