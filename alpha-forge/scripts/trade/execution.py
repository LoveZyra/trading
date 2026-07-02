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
               volume_profile: list | None = None,
               executor: "ExecutionPolicy | None" = None) -> dict:
    """Full plan for one name. Caps order at `max_participation` of ADV (splits across
    more days if it would exceed), chooses order style, and schedules child orders.
    Returns a dict; nothing is sent.

    executor: optional ExecutionPolicy (e.g. load_trained_executor(path)) -- when
    given, the child-order schedule comes from the policy's per-slice decisions
    instead of the static TWAP/VWAP split (roadmap v3 SS2.21). executor=None keeps
    the exact legacy behaviour."""
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
    if executor is not None:
        sched = executor.schedule(per_day, n_slices, participation_cap=max_participation,
                                  spread_bps=spread_bps, urgency=urgency)
        sched_label = "RL"
    else:
        sched = (vwap_schedule(per_day, volume_profile) if style == "vwap" and volume_profile
                 else twap_schedule(per_day, n_slices))
        sched_label = "TWAP" if style != "vwap" else "VWAP"
    return {
        "side": side, "total_shares": aq, "notional": round(aq * price, 2),
        "order_style": order_style(spread_bps, urgency),
        "participation_cap": max_participation,
        "spread_over_days": days,
        "per_day_shares": per_day, "day_shares": day_shares, "child_orders": sched,
        "note": (f"{side} {aq}股(${aq*price:,.0f})；" +
                 (f"超过{max_participation:.0%} ADV，分{days}天执行；" if days > 1 else "") +
                 f"{sched_label} {len([x for x in sched if x])}笔子单。仅为计划，需确认后下单。"),
    }


# ============================================================================
# Execution quality (roadmap v3 SS2.21): implementation shortfall + IS report
# ============================================================================
def implementation_shortfall(fills, decision_price: float, *, side: str = "buy",
                             arrival_price: float | None = None,
                             final_price: float | None = None,
                             target_qty: float | None = None) -> dict:
    """Perold implementation-shortfall decomposition, in bps of the decision price.

    fills: DataFrame with columns {time, qty, price} (qty in shares, all positive).
    decision_price: price when the trade decision was made (paper-portfolio price).
    arrival_price:  price when the order hit the market; defaults to the FIRST fill
        price (=> delay cost collapses to 0 unless you supply the true arrival).
    final_price:    price used to mark the unfilled remainder (defaults to last fill).
    target_qty:     intended shares; defaults to filled shares (=> opportunity cost 0).

    With s = +1 for buy / -1 for sell and f = filled fraction:
        delay_cost_bps       = f * s * (arrival - decision) / decision * 1e4
        trading_cost_bps     = f * s * (avg_fill - arrival) / decision * 1e4
        opportunity_cost_bps = (1-f) * s * (final - decision) / decision * 1e4
        is_bps               = sum of the three (total shortfall vs paper portfolio)
    Positive numbers = cost. For a buy, prices drifting UP before/while you execute
    make delay/trading costs positive -- you paid up for the delay.
    """
    import pandas as pd
    fills = pd.DataFrame(fills)
    for col in ("qty", "price"):
        if col not in fills.columns:
            raise ValueError(f"fills needs a {col!r} column")
    if decision_price is None or decision_price <= 0:
        raise ValueError("decision_price must be a positive price")
    s = {"buy": 1.0, "sell": -1.0}.get(str(side).lower())
    if s is None:
        raise ValueError("side must be 'buy' or 'sell'")
    fills = fills.sort_values("time") if "time" in fills.columns else fills
    qty = fills["qty"].astype(float).abs()
    px = fills["price"].astype(float)
    filled = float(qty.sum())
    avg_fill = float((qty * px).sum() / filled) if filled > 0 else float("nan")
    p_arr = float(arrival_price) if arrival_price is not None else (
        float(px.iloc[0]) if filled > 0 else float(decision_price))
    p_fin = float(final_price) if final_price is not None else (
        float(px.iloc[-1]) if filled > 0 else float(decision_price))
    tgt = float(target_qty) if target_qty is not None else filled
    f = min(filled / tgt, 1.0) if tgt > 0 else 0.0
    dec = float(decision_price)
    delay = f * s * (p_arr - dec) / dec * 1e4
    trading = f * s * (avg_fill - p_arr) / dec * 1e4 if filled > 0 else 0.0
    opp = (1.0 - f) * s * (p_fin - dec) / dec * 1e4
    return {"is_bps": delay + trading + opp,
            "delay_cost_bps": delay, "trading_cost_bps": trading,
            "opportunity_cost_bps": opp,
            "avg_fill_price": avg_fill, "filled_qty": filled, "fill_rate": f,
            "decision_price": dec, "arrival_price": p_arr, "final_price": p_fin,
            "side": side}


def execution_quality_report(fills, prices, *, benchmark: str = "vwap",
                             side: str = "buy", volumes=None,
                             decision_price: float | None = None,
                             target_qty: float | None = None) -> dict:
    """Post-trade execution scorecard: IS decomposition + slippage vs interval
    VWAP/TWAP benchmarks + participation rate + fill speed.

    fills:  DataFrame {time, qty, price} of child-order executions.
    prices: Series of market prices over the execution window (index = time).
    volumes: optional Series aligned with prices -- enables true interval VWAP and
        participation rate; without it VWAP falls back to TWAP (flagged in output).
    decision_price: defaults to the first price of the window (arrival-as-decision).

    Sign convention: positive *_bps = cost (you did worse than the benchmark).
    Keys: is decomposition (is_bps/delay/trading/opportunity...), vs_vwap_bps,
    vs_twap_bps, interval_vwap, interval_twap, participation_rate, fill_span,
    n_fills, fill_qty_per_period.
    """
    import numpy as np
    import pandas as pd
    fills = pd.DataFrame(fills)
    prices = pd.Series(prices).astype(float)
    if prices.empty:
        raise ValueError("prices must be a non-empty Series")
    s = {"buy": 1.0, "sell": -1.0}.get(str(side).lower())
    if s is None:
        raise ValueError("side must be 'buy' or 'sell'")
    dec = float(decision_price) if decision_price is not None else float(prices.iloc[0])
    out = implementation_shortfall(fills, dec, side=side,
                                   final_price=float(prices.iloc[-1]),
                                   target_qty=target_qty)
    twap = float(prices.mean())
    vwap_is_twap = False
    if volumes is not None:
        vol = pd.Series(volumes).astype(float).reindex(prices.index).fillna(0.0)
        vwap = float((prices * vol).sum() / vol.sum()) if vol.sum() > 0 else twap
    else:
        vwap, vwap_is_twap = twap, True
    avg_fill = out["avg_fill_price"]
    out["interval_twap"] = twap
    out["interval_vwap"] = vwap
    out["vwap_is_twap_fallback"] = vwap_is_twap
    out["vs_twap_bps"] = s * (avg_fill - twap) / twap * 1e4 if np.isfinite(avg_fill) else float("nan")
    out["vs_vwap_bps"] = s * (avg_fill - vwap) / vwap * 1e4 if np.isfinite(avg_fill) else float("nan")
    out["benchmark"] = benchmark
    out["vs_benchmark_bps"] = out["vs_vwap_bps"] if benchmark == "vwap" else out["vs_twap_bps"]
    # participation: our shares vs market volume over the window
    if volumes is not None and float(pd.Series(volumes).sum()) > 0:
        out["participation_rate"] = out["filled_qty"] / float(pd.Series(volumes).sum())
    else:
        out["participation_rate"] = float("nan")
    # fill speed
    n_periods = len(prices)
    out["n_fills"] = int(len(fills))
    out["fill_qty_per_period"] = out["filled_qty"] / n_periods if n_periods else float("nan")
    if "time" in fills.columns and len(fills) >= 2:
        tt = pd.Series(fills["time"]).sort_values()
        span = tt.iloc[-1] - tt.iloc[0]
        out["fill_span"] = float(span.total_seconds()) if hasattr(span, "total_seconds") else float(span)
    else:
        out["fill_span"] = 0.0
    return out


# ============================================================================
# RL execution bridge (roadmap v3 SS2.21): trained policy -> child-order schedule
# ============================================================================
class ExecutionPolicy:
    """Interface for a (trained) execution policy plugged into order_plan(executor=).

    decide(state) -> shares for the NEXT child slice. state dict keys provided by
    order_plan: {"remaining", "total", "slice_idx", "n_slices", "participation_cap",
    "spread_bps", "urgency"}. Return values are clamped to [0, remaining] and the
    last slice always takes the remainder, so a policy cannot under/over-fill.

    Subclass for rule-based policies, or get one from load_trained_executor(path)
    (torch state_dict trained offline by scripts/train_rl_executor.py -- note the
    training evidence is SIMULATOR-based, see that script's docstring)."""

    def decide(self, state: dict) -> float:
        raise NotImplementedError

    def schedule(self, total_shares: int, n_slices: int, *, participation_cap: float = 0.10,
                 spread_bps: float = 5.0, urgency: str = "normal") -> list:
        """Roll the policy over n_slices to produce a full child-order schedule."""
        total = int(abs(total_shares))
        if total == 0 or n_slices <= 0:
            return []
        out, remaining = [], total
        for i in range(n_slices):
            if i == n_slices - 1:
                q = remaining
            else:
                q = self.decide({"remaining": remaining, "total": total, "slice_idx": i,
                                 "n_slices": n_slices, "participation_cap": participation_cap,
                                 "spread_bps": spread_bps, "urgency": urgency})
                q = int(max(0, min(round(float(q)), remaining)))
            out.append(q)
            remaining -= q
        return out


class _TorchExecutionPolicy(ExecutionPolicy):
    """Wraps a small torch policy net: state features -> fraction of remaining to
    trade this slice. Built by load_trained_executor; needs torch at decide() time."""

    def __init__(self, net, meta: dict | None = None):
        self._net = net
        self.meta = meta or {}

    def decide(self, state: dict) -> float:
        import torch
        feats = torch.tensor([[state["slice_idx"] / max(state["n_slices"] - 1, 1),
                               state["remaining"] / max(state["total"], 1),
                               float(state.get("spread_bps", 5.0)) / 100.0,
                               float(state.get("participation_cap", 0.10))]],
                             dtype=torch.float32)
        with torch.no_grad():
            frac = float(torch.sigmoid(self._net(feats)).squeeze())
        return frac * state["remaining"]


def load_trained_executor(path) -> ExecutionPolicy:
    """Load an RL execution policy trained offline (scripts/train_rl_executor.py,
    local GPU) and return an ExecutionPolicy for order_plan(executor=...).

    Expects either a directory containing policy.pt + config.json, or a *.pt file
    with a sibling <stem>.json. torch imports lazily: without torch this raises a
    clear ImportError with the install hint (the static TWAP/VWAP planner keeps
    working regardless)."""
    import json as _json
    from pathlib import Path as _Path
    try:
        import torch
        import torch.nn as nn
    except ImportError as e:
        raise ImportError(
            "load_trained_executor needs torch (`pip install torch`). Train the policy "
            "on a local GPU with scripts/train_rl_executor.py; without torch, "
            "order_plan's static TWAP/VWAP scheduling remains fully available.") from e
    p = _Path(str(path))
    pt_path = p if p.suffix == ".pt" else p / "policy.pt"
    cfg_path = pt_path.with_suffix(".json") if p.suffix == ".pt" else p / "config.json"
    cfg = _json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    hidden = int(cfg.get("hidden", 32))
    n_feat = int(cfg.get("n_features", 4))
    net = nn.Sequential(nn.Linear(n_feat, hidden), nn.Tanh(), nn.Linear(hidden, 1))
    net.load_state_dict(torch.load(str(pt_path), map_location="cpu"))
    net.eval()
    return _TorchExecutionPolicy(net, meta=cfg)
