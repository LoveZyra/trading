"""横截面预测力评测(AI 选股的"诚实记分卡")。
purged 扩窗 walk-forward 训练排序器 -> 逐调仓日打分整截面 -> 收集预测/真实 ->
逐日 IC(Pearson)/RankIC(Spearman) -> meanIC、ICIR、RankIC、RankICIR、IC>0 占比;
分位桶平均前向收益(看单调);Top-K 多空(去重叠按 horizon、扣成本)价差/年化/Sharpe。
判定阈值参考业界:RankIC≳0.03 且 RankICIR≳0.3 才算"可用"。非投资建议。
"""
from __future__ import annotations
import warnings, copy
import numpy as np, pandas as pd
from ..strategies import multi_factor as mf
from ..core.rebalance import rebalance_dates
from ..research import models as M
from . import panel as PN

def _pear(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float); m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m] - a[m].mean(), b[m] - b[m].mean(); d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 0 else np.nan
def _spear(a, b):
    a = pd.Series(a); b = pd.Series(b); m = a.notna() & b.notna()
    return _pear(a[m].rank().values, b[m].rank().values)
def _r(x, n=4):
    try: return round(float(x), n)
    except Exception: return None

def _verdict(rankic, rankicir):
    if not np.isfinite(rankic): return "样本不足"
    if rankic >= 0.03 and rankicir >= 0.3: return "有可用横截面信号"
    if rankic > 0.01: return "弱/不稳(达不到可用门槛)"
    return "无横截面排序能力"

def evaluate_cross_section(data, model=None, panels=None, horizon=21, rebalance="ME",
                          train_window=252, min_train=120, top_frac=0.2, n_quantiles=5,
                          commission_bps=1.0, slippage_bps=1.0, min_names=8,
                          neutralize=None, sector_map=None, style_panels=None):
    """返回 {scorecard, daily, quantile_fwd, preds}。data={symbol: OHLCV}。
    neutralize∈{None,"industry","style","both"}:训练/打分前把每个因子面板过
    panel.neutralize_panel(行业组内去均值 / 风格逐日截面回归取残差)。只动预测
    特征、不动前向收益标签——评的是"剥掉行业/风格暴露后还剩多少选股 alpha",
    标签保持原始收益才是这个问题的正确口径。scorecard["neutralize"] 记录口径。"""
    model = model or M.RidgeModel(alpha=1.0)
    close = mf.build_panel(data, "close")
    if panels is None:
        panels = PN.price_factor_panels(data)
    if neutralize is not None:
        if neutralize not in ("industry", "style", "both"):
            raise ValueError(f"neutralize 须为 None/'industry'/'style'/'both',got {neutralize!r}")
        sm = sector_map if neutralize in ("industry", "both") else None
        sp = style_panels if neutralize in ("style", "both") else None
        if neutralize in ("industry", "both") and not sm:
            raise ValueError("neutralize='industry'/'both' 需要 sector_map={symbol: 行业}")
        if neutralize in ("style", "both") and not sp:
            raise ValueError("neutralize='style'/'both' 需要 style_panels={name: 宽表}")
        panels = {k: PN.neutralize_panel(v, sector_map=sm, style_panels=sp)
                  for k, v in panels.items()}
    fnames = list(panels)
    if close.shape[1] < min_names:
        warnings.warn(f"仅 {close.shape[1]} 只标的;横截面指标会很吵,需 ~30+ 只才稳。", stacklevel=2)
    fwd = close.shift(-horizon) / close - 1.0
    rebal = rebalance_dates(close.index, rebalance)
    loc = {d: i for i, d in enumerate(close.index)}
    recs = []
    for t in rebal:
        hist = close.index[close.index <= t]
        if len(hist) < min_train: continue
        tw = hist[max(0, len(hist) - train_window):]; tl = loc[t]
        usable = [s for s in tw if loc[s] + horizon < tl]          # purge:标签须已实现
        if len(usable) < max(20, min_train // 3): continue
        Xtr, ytr = [], []
        for s in usable:
            row = np.column_stack([panels[f].loc[s].values for f in fnames]); yy = fwd.loc[s].values
            mk = np.isfinite(row).all(1) & np.isfinite(yy)
            if mk.any(): Xtr.append(row[mk]); ytr.append(yy[mk])
        if not Xtr: continue
        Xtr = np.vstack(Xtr); ytr = np.concatenate(ytr)
        if len(ytr) < 20: continue
        mt = copy.deepcopy(model); mt.fit(Xtr, ytr)
        cur = np.column_stack([panels[f].loc[t].values for f in fnames]); valid = np.isfinite(cur).all(1)
        if valid.sum() < 3: continue
        pred = np.full(close.shape[1], np.nan); pred[valid] = mt.predict(cur[valid]); rr = fwd.loc[t].values
        for j, sym in enumerate(close.columns):
            if np.isfinite(pred[j]) and np.isfinite(rr[j]):
                recs.append((t, sym, float(pred[j]), float(rr[j])))
    preds = pd.DataFrame(recs, columns=["date", "symbol", "pred", "fwd"])
    out = _scorecard(preds, top_frac, n_quantiles, horizon, commission_bps + slippage_bps)
    out["scorecard"]["neutralize"] = neutralize          # 记录评测口径,报告可追溯
    out["preds"] = preds
    return out

def _scorecard(preds, top_frac, nq, horizon, cost_bps):
    nan_sc = dict(n_dates=0, n_names_avg=0, meanIC=None, ICIR=None, IC_hit=None,
                  RankIC=None, RankICIR=None, LS_mean=None, LS_ann=None, LS_sharpe=None,
                  quantile_monotonicity=None, verdict="样本不足")
    if preds.empty:
        return {"scorecard": nan_sc, "daily": pd.DataFrame(), "quantile_fwd": pd.Series(dtype=float)}
    ic, ric, ls, dates, qrows = [], [], [], [], []
    cost = 2 * cost_bps / 1e4
    for d, g in preds.groupby("date"):
        if len(g) < 3: continue
        ic.append(_pear(g.pred, g.fwd)); ric.append(_spear(g.pred, g.fwd)); dates.append(d)
        gs = g.sort_values("pred", ascending=False); k = max(1, int(round(len(gs) * top_frac)))
        ls.append(gs.head(k).fwd.mean() - gs.tail(k).fwd.mean() - cost)
        ranks = g.pred.rank(method="first"); q = ((ranks - 1) / len(g) * nq).astype(int).clip(0, nq - 1)
        qrows.append(g.assign(q=q).groupby("q").fwd.mean())
    ic, ric, ls = np.array(ic), np.array(ric), np.array(ls)
    pyr = 252.0 / horizon
    qmean = pd.concat(qrows, axis=1).mean(axis=1) if qrows else pd.Series(dtype=float)
    mono = _spear(np.asarray(qmean.index, float), qmean.values) if len(qmean) > 2 else np.nan
    sc = dict(n_dates=int(len(ic)), n_names_avg=_r(preds.groupby("date").size().mean(), 1),
              meanIC=_r(np.nanmean(ic)), ICIR=_r(np.nanmean(ic) / (np.nanstd(ic) + 1e-9), 3), IC_hit=_r((ic > 0).mean(), 3),
              RankIC=_r(np.nanmean(ric)), RankICIR=_r(np.nanmean(ric) / (np.nanstd(ric) + 1e-9), 3),
              LS_mean=_r(np.mean(ls)), LS_ann=_r(np.mean(ls) * pyr), LS_sharpe=_r(np.mean(ls) / (np.std(ls) + 1e-9) * np.sqrt(pyr), 3),
              quantile_monotonicity=_r(mono, 3), verdict=_verdict(np.nanmean(ric), np.nanmean(ric) / (np.nanstd(ric) + 1e-9)))
    daily = pd.DataFrame({"date": dates, "IC": ic, "RankIC": ric, "LS": ls})
    return {"scorecard": sc, "daily": daily, "quantile_fwd": qmean}
