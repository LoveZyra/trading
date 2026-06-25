"""Multi-method trade-signal judgments — 6 complementary lenses for one instrument.

Each lens answers a DIFFERENT question, so a name that one method calls "观望" another
may call "买入"/"持有". Designed for the report's 「信号多法对照」 section. Mechanical,
NOT advice. All degrade gracefully (return label 'N/A') when history is too short.

  1 tech_rating       — TradingView 式综合技术评级:均线族 + 振荡器各投 -1/0/+1 -> 5 档
  2 strength_score    — 0–99 自有历史多周期技术强度(SCTR 思路,单标的口径)
  3 walkforward_signal— MA20/50 趋势策略的样本外(walk-forward)edge + 当前方向
  4 regime_signal     — 趋势/波动择时态 + 建议敞口
  5 autoresearch_signal— 自动研究(多策略搜索, 样本外验证)选出的最优策略 + OOS
  6 breakout_signal   — 趋势跟随/突破口径(与"回踩买"的旧规则对照)
  old rule_signal     — 旧 classify_signal(回踩买点口径)做对照
"""
from __future__ import annotations
import numpy as np, pandas as pd
from . import indicators as IND
from . import optimize as opt, autoresearch as AR, regime as RG
from .strategies import MACrossover


def _label5(x):
    return "强力买入" if x >= 0.5 else "买入" if x >= 0.1 else "中性" if x > -0.1 else "卖出" if x > -0.5 else "强力卖出"
def _tone(x):
    return "pos" if x >= 0.1 else ("neg" if x <= -0.1 else "neu")


def tech_rating(df):
    c, h, l = df["close"], df["high"], df["low"]; last = c.iloc[-1]
    ma = []
    for n in (10, 20, 30, 50, 100, 200):
        for fn in (IND.sma, IND.ema):
            try:
                m = fn(c, n).iloc[-1]
                if pd.notna(m): ma.append(1.0 if last > m else -1.0)
            except Exception: pass
    osc = []
    try:
        r = IND.rsi(c, 14).iloc[-1]
        if pd.notna(r): osc.append(1.0 if r < 30 else -1.0 if r > 70 else 0.0)
    except Exception: pass
    try:
        ll = l.rolling(14).min(); hh = h.rolling(14).max()
        k = ((c - ll) / (hh - ll) * 100).iloc[-1]
        if pd.notna(k): osc.append(1.0 if k < 20 else -1.0 if k > 80 else 0.0)
    except Exception: pass
    try:
        tp = (h + l + c) / 3; sm = tp.rolling(20).mean(); md = (tp - sm).abs().rolling(20).mean()
        cci = ((tp - sm) / (0.015 * md)).iloc[-1]
        if pd.notna(cci): osc.append(1.0 if cci < -100 else -1.0 if cci > 100 else 0.0)
    except Exception: pass
    try:
        hh = h.rolling(14).max(); ll = l.rolling(14).min()
        wr = ((hh - c) / (hh - ll) * -100).iloc[-1]
        if pd.notna(wr): osc.append(1.0 if wr < -80 else -1.0 if wr > -20 else 0.0)
    except Exception: pass
    try:
        macd = IND.ema(c, 12) - IND.ema(c, 26); sig = macd.ewm(span=9, adjust=False).mean()
        osc.append(1.0 if macd.iloc[-1] > sig.iloc[-1] else -1.0)
    except Exception: pass
    try:
        roc = (c / c.shift(10) - 1).iloc[-1]
        if pd.notna(roc): osc.append(1.0 if roc > 0 else -1.0)
    except Exception: pass
    ma_s = float(np.mean(ma)) if ma else 0.0
    osc_s = float(np.mean(osc)) if osc else 0.0
    score = (ma_s + osc_s) / 2
    return {"score": round(score, 2), "label": _label5(score), "tone": _tone(score),
            "detail": f"均线 {round(ma_s,2)} · 振荡器 {round(osc_s,2)}(共 {len(ma)+len(osc)} 指标)"}


def strength_score(df):
    c = df["close"]; last = c.iloc[-1]; comps = []
    clamp = lambda v: max(0.0, min(100.0, v))
    def pctl(val, hist):
        hh = hist.dropna(); return float((hh <= val).mean() * 100) if len(hh) >= 20 else None
    try:
        s = IND.sma(c, 200).iloc[-1]
        if pd.notna(s): comps.append((clamp(50 + (last / s - 1) * 100), 0.25))
    except Exception: pass
    try:
        rc = (c / c.shift(125) - 1); p = pctl(rc.iloc[-1], rc)
        if p is not None: comps.append((p, 0.20))
    except Exception: pass
    try:
        s = IND.sma(c, 50).iloc[-1]
        if pd.notna(s): comps.append((clamp(50 + (last / s - 1) * 250), 0.20))
    except Exception: pass
    try:
        rc = (c / c.shift(63) - 1); p = pctl(rc.iloc[-1], rc)
        if p is not None: comps.append((p, 0.15))
    except Exception: pass
    try:
        r = IND.rsi(c, 14).iloc[-1]
        if pd.notna(r): comps.append((float(r), 0.10))
    except Exception: pass
    try:
        lo = c.rolling(20).min().iloc[-1]; hi = c.rolling(20).max().iloc[-1]
        comps.append((clamp((last - lo) / (hi - lo) * 100) if hi > lo else 50.0, 0.10))
    except Exception: pass
    if not comps: return {"score": None, "label": "N/A", "tone": "neu", "detail": "样本不足"}
    tw = sum(w for _, w in comps); score = sum(v * w for v, w in comps) / tw
    lab = "极强" if score >= 80 else "强" if score >= 60 else "中性" if score >= 40 else "弱" if score >= 20 else "极弱"
    tone = "pos" if score >= 60 else "neg" if score < 40 else "neu"
    return {"score": round(score), "label": f"{round(score)}/99 · {lab}", "tone": tone,
            "detail": f"多周期技术强度自评({len(comps)} 项,自有历史口径)"}


def walkforward_signal(df):
    # Adapt the fold count to the available history: walk_forward needs >=30 bars/fold, so a
    # fixed n_splits=4 silently N/As on ~120-bar histories. Use the MOST folds that fit.
    wf = None
    for nsplits in (4, 3, 2):
        try:
            wf = opt.walk_forward(MACrossover, df, {"fast": [10, 20, 30], "slow": [50, 100, 150]},
                                  n_splits=nsplits, metric="sharpe")
            break
        except Exception:
            continue
    if wf is None:
        return {"label": "N/A", "tone": "neu", "detail": "历史过短,无法做样本外验证(需 ~60+ 根)"}
    try: oos = float(wf.oos_stats.get("sharpe"))
    except Exception: oos = None
    sig = float(MACrossover(fast=20, slow=50).latest_signal(df))
    lab = "做多" if sig > 0 else "做空" if sig < 0 else "空仓"
    edge = oos is not None and oos > 0.3
    tone = "pos" if (sig > 0 and edge) else ("neg" if sig < 0 else "neu")
    return {"signal": lab, "oos_sharpe": round(oos, 2) if oos is not None else None, "tone": tone,
            "label": lab + ("" if edge else "(OOS弱)"),
            "detail": f"MA20/50 · OOS Sharpe {round(oos,2) if oos is not None else 'NA'} · 当前 {lab}"}


def regime_signal(df):
    c = df["close"]
    try: tr = RG.trend_regime(c).iloc[-1]
    except Exception: tr = np.nan
    if pd.isna(tr):
        try: tr = 1.0 if c.iloc[-1] > IND.sma(c, 200).iloc[-1] else -1.0
        except Exception: tr = 0.0
    try: sc = float(RG.regime_scale(c).iloc[-1])
    except Exception: sc = None
    try:
        rv = c.pct_change().rolling(21).std() * (252 ** 0.5)
        volp = float((rv.dropna() <= rv.iloc[-1]).mean() * 100)
    except Exception: volp = None
    up = tr > 0; hivol = volp is not None and volp > 70
    if up and not hivol: lab, tone = "顺势做多(上升·波动可控)", "pos"
    elif up and hivol: lab, tone = "持有但降敞口(上升·高波)", "warn"
    elif not up: lab, tone = "收手/观望(非上升态)", "neg"
    else: lab, tone = "中性", "neu"
    return {"label": lab, "tone": tone, "scale": round(sc, 2) if sc is not None else None,
            "detail": f"趋势 {'上升' if up else '非上升'} · 波动分位 {round(volp) if volp is not None else 'NA'} · 敞口×{round(sc,2) if sc is not None else 'NA'}"}


def autoresearch_signal(df, iterations=8):
    try:
        nsplits = max(2, min(4, len(df) // 35))   # adapt folds to history so OOS can compute
        rep = AR.research_single(df, iterations=iterations, n_splits=nsplits)
        best = rep.best
        name = getattr(best, "direction", None) or getattr(best, "name", None) or (best.get("name") if isinstance(best, dict) else str(best))
        oos = None
        for getter in (lambda: getattr(best, "oos_sharpe", None), lambda: best.extra.get("holdout_sharpe"),
                       lambda: best.extra.get("oos_sharpe"), lambda: best.get("sharpe")):
            try:
                v = getter()
                if v is not None: oos = float(v); break
            except Exception: pass
        tone = "pos" if (oos is not None and oos > 0.5) else "neu"
        return {"label": str(name), "oos_sharpe": round(oos, 2) if oos is not None else None, "tone": tone,
                "detail": f"自动搜索最优:{name} · OOS Sharpe {round(oos,2) if oos is not None else 'NA'}"}
    except Exception as e:
        return {"label": "N/A", "tone": "neu", "detail": f"err:{str(e)[:60]}"}


def breakout_signal(df):
    c, h = df["close"], df["high"]; last = c.iloc[-1]
    try: s50 = IND.sma(c, 50).iloc[-1]; s200 = IND.sma(c, 200).iloc[-1]
    except Exception: s50 = s200 = np.nan
    up = pd.notna(s50) and pd.notna(s200) and last > s50 > s200
    try: dhi = h.rolling(20).max().iloc[-2]
    except Exception: dhi = np.nan
    brk = pd.notna(dhi) and last >= dhi * 0.99
    if up and brk: lab, tone = "突破/趋势跟进", "pos"
    elif up: lab, tone = "持有(趋势内)", "pos"
    elif pd.notna(s50) and last < s50: lab, tone = "跌破SMA50/离场", "neg"
    else: lab, tone = "观望", "neu"
    return {"label": lab, "tone": tone, "detail": f"价 vs SMA50/200 {'多头排列' if up else '—'} · 近20日高 {'已破' if brk else '未破'}"}


def rule_signal(df):
    from .levels import trade_levels, classify_signal
    lv = trade_levels(df, atr_mult=2.0, rr=2.0); c = df["close"]; last = c.iloc[-1]
    try: rsi = float(IND.rsi(c, 14).iloc[-1])
    except Exception: rsi = 50.0
    bz = lv["buy_zone"]; inz = bz[0] <= last <= bz[1]
    sig = classify_signal(rsi=rsi, rr=lv["reward_risk"], trend="bull", in_buy_zone=inz, event=False, leveraged=False)
    m = {"long": "做多", "watch": "观望", "short": "做空"}
    return {"label": m.get(sig, sig), "tone": ("pos" if sig == "long" else "neg" if sig == "short" else "neu"),
            "detail": f"R/R {round(float(lv['reward_risk']),2)} · {'在买区' if inz else ('买区上方' if last>bz[1] else '买区下方')}"}


def all_methods(df, *, iterations=8, heavy=True):
    """6-lens signal read for one instrument.

    heavy=False skips the two EXPENSIVE lenses (m3 walk-forward, m5 autoresearch — each
    runs a full backtest search) and returns a 'skipped' placeholder for them. Use it for
    multi-name 信号多法对照 tables, where running ~32 fits per name would make a 50-name
    table take 10s+. Single-name reports should keep heavy=True for the full read.
    """
    skipped = {"label": "—", "tone": "neu", "detail": "(大规模筛选·已显式关闭)", "signal": "—"}
    return {"m1": tech_rating(df), "m2": strength_score(df),
            "m3": walkforward_signal(df) if heavy else dict(skipped),
            "m4": regime_signal(df),
            "m5": autoresearch_signal(df, iterations) if heavy else dict(skipped),
            "m6": breakout_signal(df), "old": rule_signal(df)}


# One-line intro per judging method — so the report's 「信号多法对照」 explains every lens.
METHOD_INFO = {
    "m1": {"name": "技术评级", "desc": "均线族 + 振荡器各投 -1/0/+1,综合成 5 档评级(TradingView 式)。"},
    "m2": {"name": "技术强度分", "desc": "0–99 的多周期相对强度自评(SCTR 思路,单标的口径)。"},
    "m3": {"name": "样本外择时", "desc": "MA20/50 趋势策略的 walk-forward 出样本 edge + 当前方向。"},
    "m4": {"name": "regime 择时", "desc": "趋势/波动状态判定 + 建议敞口(高波/熊市自动降仓)。"},
    "m5": {"name": "自动研究最优", "desc": "多策略自动搜索(样本外验证)选出的最优策略族 + OOS 夏普。"},
    "m6": {"name": "突破/趋势", "desc": "价 vs SMA50/200 多头排列 + 近 20 日高点突破口径。"},
    "old": {"name": "回踩买点(对照)", "desc": "旧规则:按盈亏比 + 是否在买区给做多/观望/做空(留作对照)。"},
}
_METHOD_ORDER = ["m1", "m2", "m3", "m4", "m5", "m6", "old"]


def methods_report(data, *, symbols=None, iterations=8, heavy=None, title="信号多法对照"):
    """Assemble the html_report `methods` dict from all_methods() across one or more names.

    data: a single OHLCV df (one symbol) OR {name: df}. Every method row carries its
    one-line `desc` (from METHOD_INFO) so the report EXPLAINS each lens, not just shows a
    verdict. By default ALL six lenses run for EVERY name (heavy=True). Pass heavy=False
    only for a LARGE universe where a full walk-forward + autoresearch search per name
    would be slow — then the two expensive lenses (m3/m5) are skipped, with a note saying so.
    """
    uni = data if isinstance(data, dict) else {(symbols[0] if symbols else "标的"): data}
    hv = True if heavy is None else heavy
    per = {name: all_methods(df, iterations=iterations, heavy=hv) for name, df in uni.items()}
    rows = [{"key": k, "m": METHOD_INFO[k]["name"], "desc": METHOD_INFO[k]["desc"]} for k in _METHOD_ORDER]
    syms = [{"key": name, "name": name} for name in uni]
    note = ("本表显式关闭了重型计算(heavy=False),「样本外择时 / 自动研究」两栏留空;默认全跑。"
            if (not hv and len(uni) > 1) else None)
    return {"title": title, "symbols": syms, "rows": rows, "data": per, "note": note}
