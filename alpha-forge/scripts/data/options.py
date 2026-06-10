"""Options-implied signals — expected moves, IV skew, put/call ratio.

Price-only analysis misses what the options market is pricing: how big a move it
expects (esp. into earnings) and which way it's leaning. The broker exposes OPT chains;
Claude pulls ATM implied vol / a straddle price / put & call IVs and feeds them here.
Research backs this: put-call IV spread & skew predict stock returns (they proxy the
borrow fee), and the ATM straddle gives the market's expected earnings move.

All functions take simple numeric inputs (no live chain needed in the sandbox).
"""
from __future__ import annotations

import numpy as np


def expected_move(price: float, atm_iv: float, days: int) -> dict:
    """Expected ±move over `days` from at-the-money implied vol (annualized, e.g. 0.6).
    move = price * iv * sqrt(days/365). Returns the move and the implied range."""
    price = float(price)
    mv = price * float(atm_iv) * np.sqrt(max(days, 0) / 365.0)
    move_pct = round(mv / price * 100, 2) if price else 0.0
    return {"expected_move": round(mv, 2), "move_pct": move_pct,
            "low": round(price - mv, 2), "high": round(price + mv, 2), "days": days}


def expected_move_from_straddle(straddle_price: float, price: float) -> dict:
    """Market's expected move ≈ the ATM straddle (call+put) price. Quick read for
    earnings: if the at-the-money straddle costs $40 on a $400 stock, the options market
    is pricing a ~±10% earnings move."""
    mv = float(straddle_price); price = float(price)
    move_pct = round(mv / price * 100, 2) if price else 0.0
    return {"expected_move": round(mv, 2), "move_pct": move_pct,
            "low": round(price - mv, 2), "high": round(price + mv, 2)}


def iv_skew_signal(put_iv: float, call_iv: float) -> dict:
    """Put-call IV spread / skew. Puts richer than calls (positive spread) = demand for
    downside protection / higher borrow = BEARISH tilt. Returns a signal in [-1,1]
    (negative = bearish skew). Academic: the spread predicts returns (borrow-fee proxy).
    """
    spread = float(put_iv) - float(call_iv)
    base = 0.5 * (put_iv + call_iv) or 1e-6
    sig = -np.tanh((spread / base) * 4)        # rich puts -> negative
    return {"skew_signal": round(float(sig), 3), "put_call_iv_spread": round(spread, 4),
            "bias": ("看跌(下行保护需求高)" if sig < -0.15 else
                     "看涨(call偏贵)" if sig > 0.15 else "中性")}


def put_call_ratio_signal(pcr: float, *, contrarian: bool = False) -> dict:
    """Put/Call volume ratio as sentiment. >1 = fear (more puts). Default reads it as a
    RISK gauge (high PCR -> risk-off); set contrarian=True to fade extremes (very high
    PCR can mark capitulation lows)."""
    pcr = float(pcr)
    risk = -np.tanh((pcr - 0.9) / 0.4)         # high PCR -> negative (fearful)
    sig = -risk if contrarian else risk
    return {"pcr": pcr, "signal": round(float(sig), 3),
            "read": ("恐慌/避险" if pcr > 1.1 else "乐观/贪婪" if pcr < 0.7 else "中性")
            + ("（逆向：极端处反向）" if contrarian else "")}


def earnings_setup(price: float, atm_iv: float, days_to_earnings: int,
                   realized_vol: float | None = None) -> dict:
    """Pre-earnings read: expected move + whether options are 'expensive' (IV >> realized
    => big move priced in; a beat that's already priced can still sell off). Use to size
    down / avoid chasing into a print."""
    em = expected_move(price, atm_iv, max(days_to_earnings, 1))
    rich = None
    if realized_vol:
        rich = round(float(atm_iv) / float(realized_vol), 2)
    note = []
    if rich and rich > 1.3:
        note.append(f"IV/实际波动={rich} → 期权偏贵、大幅波动已被定价，财报后或'见光死'")
    note.append(f"期权定价财报±{em['move_pct']}%（{em['low']}–{em['high']}）")
    return {**em, "days_to_earnings": days_to_earnings, "iv_rv_ratio": rich, "note": "；".join(note)}
