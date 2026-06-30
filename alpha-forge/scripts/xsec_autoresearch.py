"""自动选股研究:搜因子子集 × 排序器,目标函数 = RankICIR(或 LS_sharpe),walk-forward。
复用 xsec_eval(本身就是 purged WF),返回排行榜。非投资建议。
"""
from __future__ import annotations
from itertools import combinations
import pandas as pd
from . import models as M
from . import panel as PN
from .xsec_eval import evaluate_cross_section

def search(data, factor_panels=None, candidate_models=None, horizon=21, rebalance="ME",
           objective="RankICIR", top_n=8, **kw):
    if factor_panels is None: factor_panels = PN.price_factor_panels(data)
    fn = list(factor_panels)
    if candidate_models is None:
        candidate_models = {"ridge": M.RidgeModel(alpha=1.0)}
    subsets = [tuple(fn)]
    if len(fn) > 1:
        subsets += [tuple(c) for c in combinations(fn, len(fn) - 1)]   # leave-one-out
        subsets += [(f,) for f in fn]                                  # 单因子
    seen, rows = set(), []
    for sub in subsets:
        if sub in seen: continue
        seen.add(sub)
        pj = {k: factor_panels[k] for k in sub}
        for mname, mdl in candidate_models.items():
            res = evaluate_cross_section(data, model=mdl, panels=pj, horizon=horizon, rebalance=rebalance, **kw)
            sc = res["scorecard"]
            rows.append({"factors": "+".join(sub), "model": mname,
                         **{k: sc[k] for k in ("RankIC", "RankICIR", "ICIR", "LS_sharpe", "n_dates")}})
    lb = pd.DataFrame(rows)
    if objective in lb.columns:
        lb = lb.sort_values(objective, ascending=False, na_position="last").reset_index(drop=True)
    return lb.head(top_n) if top_n else lb
