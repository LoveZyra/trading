"""Alternative-data signals — information beyond price. All hand-off: Claude pulls the
numbers (broker / WebSearch / akshare), these normalize them to a [-1,1] tilt.
"""
from __future__ import annotations

import numpy as np


def _t(x, scale):
    return float(np.tanh(x / scale))


def insider_signal(net_buy_value: float, market_cap: float | None = None) -> dict:
    """Net insider buying (buys - sells, $). Buying = bullish, selling = bearish.
    Scaled by market cap if given."""
    base = market_cap * 0.0005 if market_cap else 5e6
    return {"signal": round(_t(net_buy_value, base), 3),
            "read": "内部人增持" if net_buy_value > 0 else "内部人减持"}


def short_interest_signal(si_pct_float: float, change_pp: float = 0.0) -> dict:
    """Short interest as % of float + its change. High & rising SI = bearish pressure
    (but extreme SI can squeeze). Returns a risk tilt (negative = bearish)."""
    level = -_t(si_pct_float - 5, 8)            # >5% float short starts to weigh
    momentum = -_t(change_pp, 3)                # rising SI -> more bearish
    return {"signal": round(0.6 * level + 0.4 * momentum, 3), "si_pct": si_pct_float,
            "read": ("空头沉重" if si_pct_float > 12 else "空头一般" if si_pct_float > 5 else "空头轻")}


def northbound_signal(net_inflow_yi: float) -> dict:
    """A-share northbound (北向资金) net inflow in 亿元. Inflow = risk-on for A-shares."""
    return {"signal": round(_t(net_inflow_yi, 50), 3),
            "read": "北向净流入" if net_inflow_yi > 0 else "北向净流出"}


def trends_signal(interest_now: float, interest_avg: float) -> dict:
    """Search/attention (e.g. Google/百度 index) vs its average. Spiking attention often
    accompanies tops/froth; mild pickup can precede moves. Returns attention z-tilt."""
    if not interest_avg:
        return {"signal": 0.0}
    ratio = interest_now / interest_avg - 1
    return {"signal": round(_t(ratio, 1.0), 3), "attention_ratio": round(ratio, 2)}
