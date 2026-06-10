"""Order-execution planner — turn a target position into a smart order PLAN.

Builds (does NOT submit) an execution plan: how many shares to trade, limit vs market,
and a TWAP/VWAP child-order schedule capped by a participation limit so you don't move
the market. Submitting is a separate, explicit step via the broker's order tools with
the user's confirmation.
"""
from __future__ import annotations

import numpy as np


def shares_to_trade(target_weight: float, current_weight: float, account_value: float,
                    price: float) -> int:
    delta_w = float(target_weight) - float(current_weight)
    return int(round(delta_w * account_value / price)) if price > 0 else 0


def order_style(spread_bps: float, urgency: str = "normal") -> str:
    """Marketable-limit vs patient-limit vs market, from spread & urgency. Wide spreads
    or low urgency -> patient limit; tight & urgent -> market."""
    if urgency == "high" or spread_bps <= 3:
        return "market/marketable-limit"
    if spread_bps >= 20:
        return "patient-limit (挂单等成交)"
    return "limit @ mid±spread/2"


def twap_schedule(total_shares: int, n_slices: int = 6) -> list:
    """Equal child slices over the window (time-weighted)."""
    if total_shares == 0 or n_slices <= 0:
        return []
    base = total_shares // n_slices
    rem = total_shares - base * n_slices
    return [base + (1 if i < rem else 0) for i in range(n_slices)]


def vwap_schedule(total_shares: int, volume_profile: list) -> list:
    """Child slices proportional to an intraday volume profile (front/back-loaded)."""
    vp = np.array(volume_profile, float)
    if vp.sum() <= 0 or total_shares == 0:
        return twap_schedule(total_shares, len(volume_profile) or 6)
    raw = total_shares * vp / vp.sum()
    out = np.floor(raw).astype(int)
    out[np.argmax(raw - out)] += total_shares - out.sum()
    return out.tolist()


def order_plan(target_weight: float, current_weight: float, account_value: float,
               price: float, *, adv_shares: float | None = None,
               max_participation: float = 0.10, spread_bps: float = 5.0,
               urgency: str = "normal", style: str = "twap", n_slices: int = 6,
               volume_profile: list | None = None) -> dict:
    """Full plan for one name. Caps order at `max_participation` of ADV (splits across
    more days if it would exceed), chooses order style, and schedules child orders.
    Returns a dict; nothing is sent."""
    qty = shares_to_trade(target_weight, current_weight, account_value, price)
    side = "BUY" if qty > 0 else ("SELL" if qty < 0 else "HOLD")
    aq = abs(qty)
    days = 1
    capped = aq
    if adv_shares and adv_shares > 0:
        cap = max_participation * adv_shares
        if aq > cap:
            days = int(np.ceil(aq / cap))
            capped = int(cap)
    per_day = capped
    # Explicit day-by-day quantities (last day takes the remainder), so a plan that
    # spans days is complete instead of only describing day 1.
    day_shares = ([per_day] * (days - 1) + [aq - per_day * (days - 1)]) if days > 1 else [aq]
    sched = (vwap_schedule(per_day, volume_profile) if style == "vwap" and volume_profile
             else twap_schedule(per_day, n_slices))
    return {
        "side": side, "total_shares": aq, "notional": round(aq * price, 2),
        "order_style": order_style(spread_bps, urgency),
        "participation_cap": max_participation,
        "spread_over_days": days,
        "per_day_shares": per_day, "day_shares": day_shares, "child_orders": sched,
        "note": (f"{side} {aq}股(${aq*price:,.0f})；" +
                 (f"超过{max_participation:.0%} ADV，分{days}天执行；" if days > 1 else "") +
                 f"{'TWAP' if style!='vwap' else 'VWAP'} {len([x for x in sched if x])}笔子单。仅为计划，需确认后下单。"),
    }
