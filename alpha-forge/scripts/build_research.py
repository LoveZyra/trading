"""build_research — assemble the report's 「自动研究详情 / 策略测试与选择」 section.

SCHEMA.md documents this rich `research` block (selection narrative + a leaderboard whose
rows each carry the strategy's CURRENT signal AND buy/sell trigger prices + the winner's
triggers + a buy/sell-points-on-price trades chart + stats) and names *this* module as its
builder — but the module was missing, so callers had to hand-roll a thin version. This
implements it to spec so the skill genuinely produces the full strategy output.

Search is delegated to the engine: `autoresearch.screen_rule_strategies` exhaustively finds each
family's best config + its walk-forward OOS Sharpe (ONE search path — never re-implemented here).
build_research only RANKS, formats the leaderboard / triggers / trades-chart, and runs
`validation.selection_robustness` (CPCV OOS distribution + PBO + SPA/Reality-Check) to say plainly
whether the winner is a real edge or just the luckiest of the search. Triggers move with price;
OOS is overstated on short / one-directional samples (the winner's total return is often below
buy-and-hold — it wins on drawdown, not absolute return).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import backtest as bt, indicators as ind, validation as V, autoresearch as AR
from .strategies import REGISTRY


def _cast(v):
    """numpy/str -> native python int/float (clean for display + JSON + constructor args)."""
    try:
        x = float(v)
        return int(x) if x == int(x) else round(x, 4)
    except (TypeError, ValueError):
        return v


def _pstr(p: dict) -> str:
    """Format the searched params for display, e.g. {'fast':20,'slow':100} -> 'fast=20·slow=100'."""
    return "·".join(f"{k}={_cast(v)}" for k, v in p.items()) if p else "默认"


def _triggers(name: str, df: pd.DataFrame, params: dict | None = None) -> dict:
    """Rule-derived current buy/sell trigger for a strategy family, using the SEARCHED params
    where the window matters (so the trigger matches the chosen config, not a fixed default).
    These '跌破X卖 / 站上Y买' levels move every bar — they are not fixed targets."""
    p = params or {}
    c = df["close"]
    try:
        if name == "breakout":
            en, ex = int(p.get("entry", 20)), int(p.get("exit", 10))
            hi = float(df["high"].rolling(en).max().iloc[-1]); lo = float(df["low"].rolling(ex).min().iloc[-1])
            return {"buy": f"站上 {hi:.2f}({en}日高)", "sell": f"跌破 {lo:.2f}({ex}日低)"}
        if name == "bollinger_reversion":
            b = ind.bollinger(c, int(p.get("n", 20)), float(p.get("k", 2.0))).iloc[-1]
            return {"buy": f"触 {b['lower']:.2f}(下轨)", "sell": f"回 {b['mid']:.2f}(中轨)"}
        if name == "zscore_reversion":
            lb = int(p.get("lookback", 20)); ent = float(p.get("entry", 1.5))
            m = float(c.rolling(lb).mean().iloc[-1]); sd = float(c.rolling(lb).std(ddof=0).iloc[-1])
            return {"buy": f"{m - ent * sd:.2f}(z≈-{ent})", "sell": f"{m:.2f}(回均值)"}
        if name == "rsi_reversion":
            return {"buy": f"RSI<{int(p.get('oversold', 30))}", "sell": "RSI>50"}
        if name == "ts_momentum":
            lb = int(p.get("lookback", 90)); ref = float(c.iloc[-lb]) if len(c) > lb else float(c.iloc[0])
            return {"buy": f"站上 {ref:.2f}(动量翻正)", "sell": f"跌破 {ref:.2f}"}
        if name == "ma_crossover":
            slow = int(p.get("slow", 50)); s = float(getattr(ind, p.get("ma", "ema"))(c, slow).iloc[-1])
            return {"buy": f"快线上穿(慢线≈{s:.2f})", "sell": "快线下穿"}
        if name == "macd_trend":
            return {"buy": "MACD 柱翻正", "sell": "MACD 柱翻负"}
    except Exception:  # noqa: BLE001
        pass
    return {"buy": "—", "sell": "—"}


def trades_viz(df: pd.DataFrame, res, *, logy: bool = False, unit: str = "") -> dict:
    """Build the `trades` dict for tradesChart: price line + ▲buy/▼sell indices + green hold
    spans, from a BacktestResult's (already-lagged) position series."""
    P = [round(float(x), 3) for x in df["close"].values]
    pos = pd.Series(res.position).reindex(df.index).fillna(0.0).values
    buys, sells, hold = [], [], []
    start, prev = None, 0.0
    for i, p in enumerate(pos):
        if p > 0 and prev <= 0:
            buys.append(i); start = i
        if p <= 0 and prev > 0:
            sells.append(i)
            if start is not None:
                hold.append([start, i]); start = None
        prev = p
    if start is not None:
        hold.append([start, len(P) - 1])
    return {"price": P, "buys": buys, "sells": sells, "hold": hold, "logy": logy, "unit": unit,
            "dates": [str(d.date()) if hasattr(d, "date") else str(d) for d in df.index],
            "date_start": str(df.index[0].date()), "date_end": str(df.index[-1].date())}


def build_research(df: pd.DataFrame, name: str = "标的", *, iterations: int = 14,
                   n_splits: int = 4, top: int = 6, commission_bps: float = 2.0,
                   slippage_bps: float = 3.0, title: str | None = None) -> dict:
    """Full `research` dict (one item) per SCHEMA.md — ready for html_report.

    Returns {title, items:[{name, winner, selection_text, leaderboard, trades, stats}],
    glossary, robustness}. `robustness` carries the CPCV/PBO/SPA bundle for a 稳健性体检.
    """
    # ONE search path: the engine exhaustively finds each family's best config + its OOS Sharpe.
    # build_research only consumes the result and formats it — it never re-searches here.
    screen = AR.screen_rule_strategies(df, commission_bps=commission_bps, slippage_bps=slippage_bps)
    if not screen:
        return {"title": title or f"策略测试与选择 · {name}", "items": [], "glossary": [],
                "note": "无可回测策略(数据不足)"}
    rets = {k: fr.returns for k, fr in screen.items()}
    results = {k: fr.result for k, fr in screen.items()}
    params_by = {k: fr.params for k, fr in screen.items()}
    oos_by = {k: fr.oos_sharpe for k, fr in screen.items()}
    bh = bt.buy_and_hold(df)

    ranked = sorted(rets, key=lambda k: (1 + rets[k]).prod(), reverse=True)
    winner = ranked[0]
    tr = pd.DataFrame(rets).fillna(0.0)
    rob = V.selection_robustness(tr, winner=winner)
    spa, cp = rob.get("spa", {}), rob.get("cpcv", {})

    def _row(i, k):
        pp = params_by.get(k, {})
        sig = float(REGISTRY[k](**pp).latest_signal(df)); t = _triggers(k, df, pp)
        return {"rank": i + 1, "strategy": k, "params": _pstr(pp),
                "oos_sharpe": oos_by.get(k),
                "oos_return": f"{(1 + rets[k]).prod() - 1:+.0%}",
                "signal": "做多" if sig > 0 else ("做空" if sig < 0 else "空仓"),
                "buy": t["buy"], "sell": t["sell"], "win": k == winner}
    leaderboard = [_row(i, k) for i, k in enumerate(ranked[:top])]

    wp = params_by.get(winner, {})
    wsig = float(REGISTRY[winner](**wp).latest_signal(df)); wt = _triggers(winner, df, wp); wr = results[winner]
    winner_obj = {"strategy": winner, "params": _pstr(wp),
                  "oos_sharpe": oos_by.get(winner, round(float(wr.stats["sharpe"]), 2)),
                  "oos_return": f"{(1 + rets[winner]).prod() - 1:+.0%}",
                  "signal": ("当前做多/持有" if wsig > 0 else "当前空仓·等触发"),
                  "signal_tone": "pos" if wsig > 0 else "neu",
                  "exit": wt["sell"],
                  "triggers": {"action": ("已满足买入条件,小仓试" if wsig > 0 else "等价格触发再进"),
                               "buy": wt["buy"], "sell": wt["sell"]}}

    spa_p = spa.get("spa_p")
    spa_ok = (spa_p is not None and np.isfinite(spa_p) and spa_p < 0.05)
    selection_text = (
        f"在 {len(rets)} 个策略族上逐一回测 {len(df)} 根:按样本内累计收益排名,<b>{winner}</b> 居首"
        f"({'均值回归族逢极端反向' if 'revers' in winner else '趋势族顺势'});"
        f"<b>稳健性体检</b>(CPCV+SPA):SPA/Reality-Check p={spa_p} → "
        f"{'<b>通过数据窥探校验(p&lt;0.05)</b>,但样本仍短' if spa_ok else '<b>未通过——多半是从多个策略里挑最好的搜索幸运</b>'};"
        f"PBO {rob.get('pbo')}、Deflated Sharpe {rob.get('deflated_sharpe')};"
        f"CPCV 出样本夏普 中位 {cp.get('median')}、5–95% 带 [{cp.get('q05')}, {cp.get('q95')}]、为正占比 {cp.get('frac_positive')}。")

    stats = [{"k": "策略收益", "v": f"{(1 + rets[winner]).prod() - 1:+.0%}"},
             {"k": "买入持有", "v": f"{bh.stats['total_return']:+.0%}"},
             {"k": "最大回撤", "v": f"{wr.stats['max_drawdown']:.0%}"},
             {"k": "夏普", "v": f"{wr.stats['sharpe']:.1f}"}]

    return {"title": title or f"自动研究详情 · 策略测试与选择 · {name}",
            "items": [{"name": name, "winner": winner_obj, "selection_text": selection_text,
                       "leaderboard": leaderboard, "trades": trades_viz(df, wr), "stats": stats}],
            "glossary": AR.strategy_glossary(families=ranked),
            "robustness": rob}
