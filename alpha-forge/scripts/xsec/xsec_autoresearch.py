"""自动选股研究:搜因子子集 × 排序器,目标函数 = RankICIR(或 LS_sharpe),walk-forward。
复用 xsec_eval(本身就是 purged WF),返回排行榜。非投资建议。

Round 11(§2.9 接入):search 新增 factor_source="price"|"zoo"|"all" —— 因子面板
可以直接从 factor_zoo 表达式库构建,先过 research.prescreen 的质量预筛
(RankIC/衰减/冗余/拥挤四道闸)再进搜索;大因子池的子集生成自动切换为
"全选 + top-N 嵌套组 + 单因子"策略(封顶 ~25 个子集)。默认参数
(factor_source=None)的行为与旧版逐路径一致,零破坏。
"""
from __future__ import annotations
from itertools import combinations
import pandas as pd
from ..research import models as M
from ..strategies import multi_factor as mf
from . import panel as PN
from .xsec_eval import evaluate_cross_section

#: 大因子池时的子集总数上限(见 _adaptive_subsets 的理由)。
SUBSET_CAP = 25


def build_factor_source_panels(data, factor_source="all", zoo_which="alpha158",
                               zoo_max=40):
    """按来源构建 {name: date×symbol 宽表} 因子面板。

    "price" = panel.price_factor_panels(现有 8 个价格因子,已截面 z);
    "zoo"   = factor_zoo.compute_library(表达式库,取前 zoo_max 条)再逐日
              截面 z-score —— 与价格面板同尺度,ridge/stacking 的正则才公平;
    "all"   = 两者合并(名字空间不重叠)。全 NaN 的面板直接丢弃(库因子在
    小样本/合成数据上可能退化),它只会浪费模型容量。
    """
    if factor_source not in ("price", "zoo", "all"):
        raise ValueError(f"factor_source 须为 'price'/'zoo'/'all',got {factor_source!r}")
    out = {}
    if factor_source in ("price", "all"):
        out.update(PN.price_factor_panels(data))
    if factor_source in ("zoo", "all"):
        from ..research import factor_zoo as FZ
        for name, raw in FZ.compute_library(data, which=zoo_which,
                                            max_factors=zoo_max).items():
            z = PN.cs_zscore(raw)
            if z.notna().any().any():
                out[name] = z
    return out


def _adaptive_subsets(fn, score, cap=SUBSET_CAP):
    """大因子池(>8)的子集策略:全选 + |RankIC| top-3/5/8 嵌套组 + 单因子,封顶 cap。

    why 放弃 leave-one-out 枚举:k 个因子的 LOO 是 k 个 k-1 维、彼此 96%+ 重叠的
    子集,k=40 时评测数线性膨胀而信息几乎不增——统计上等于对同一组合做几十次
    近似相同的检验(多重检验白白消耗显著性预算),计算上是组合爆炸的入口。
    top-N 嵌套组覆盖"少而精 vs 全都要"的谱系(prescreen 已按质量排序,前缀组
    就是最有希望的组合),单因子测每条因子的独立贡献,总数封顶 cap 控制
    走查成本与假阳性率。
    """
    order = sorted(fn, key=lambda n: -abs(score.get(n, 0.0)))
    subsets = [tuple(fn)]
    for k in (3, 5, 8):
        if k < len(fn):
            subsets.append(tuple(order[:k]))
    room = max(0, cap - len(subsets))
    subsets += [(f,) for f in order[:room]]
    seen, out = set(), []
    for s in subsets:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def default_candidate_models(include_listwise=False):
    """自动研究的默认模型空间。ridge 永远在(纯 numpy 基线,任何模型必须先打赢它);
    include_listwise=True 追加 ListNet/ListMLE(纯 numpy 线性 listwise 排序器——
    选股本质是排序,listwise 直接优化序而非逐点 MSE)。LambdaMART 需要 lightgbm,
    装了才进空间(缺库静默跳过而不是让整个搜索炸掉)。"""
    out = {"ridge": M.RidgeModel(alpha=1.0)}
    if include_listwise:
        from .xsec_models import ListNetModel, ListMLEModel
        out["listnet"] = ListNetModel()
        out["listmle"] = ListMLEModel()
        try:
            import lightgbm  # noqa: F401
            from .xsec_models import LambdaMARTModel
            out["lambdamart"] = LambdaMARTModel()
        except ImportError:
            pass
    return out


def search(data, factor_panels=None, candidate_models=None, horizon=21, rebalance="ME",
           objective="RankICIR", top_n=8, factor_source=None, zoo_which="alpha158",
           zoo_max=40, prescreen=True, prescreen_kwargs=None,
           include_listwise=False, **kw):
    """因子子集 × 模型 的排行榜搜索(purged walk-forward 打分)。

    factor_source=None(默认):行为与旧版完全一致 —— 用传入的 factor_panels
    (缺省 price_factor_panels),子集 = 全选 + leave-one-out + 单因子。

    factor_source="price"|"zoo"|"all":factor_panels 自动构建(见
    build_factor_source_panels);prescreen=True 时先过
    research.prescreen.prescreen_factors 四道闸(prescreen_kwargs 透传,
    horizon 缺省与本函数一致),只有 selected 进搜索;全军覆没时回退到
    |RankIC| 前 3(搜索总得有原料,报告里仍能看到全体被剔的原因)。
    因子数 > 8 时子集生成切换为 _adaptive_subsets(封顶 ~25)。

    返回:leaderboard DataFrame;factor_source 非 None 时附
    lb.attrs["prescreen_report"](预筛报告,未预筛为 None)与
    lb.attrs["factor_panels"](实际参战的面板,可直接喂 ensemble_top /
    ml_factor_backtest)——用 attrs 而不是改返回类型,向后兼容。
    """
    prescreen_report = None
    rank_score = None
    if factor_source is not None:
        factor_panels = build_factor_source_panels(data, factor_source,
                                                   zoo_which=zoo_which, zoo_max=zoo_max)
        if prescreen:
            from ..research.prescreen import prescreen_factors
            pk = dict(horizon=horizon)
            pk.update(prescreen_kwargs or {})
            close = mf.build_panel(data, "close")
            scr = prescreen_factors(factor_panels, close, **pk)
            prescreen_report = scr["report"]
            rank_score = dict(zip(prescreen_report["name"],
                                  prescreen_report["rankic"].fillna(0.0)))
            if scr["selected"]:
                factor_panels = scr["selected"]
            else:                                   # 全剔 -> 回退到 |RankIC| 前 3
                best = (prescreen_report.assign(a=prescreen_report["rankic"].abs())
                        .sort_values("a", ascending=False)["name"].head(3))
                factor_panels = {n: factor_panels[n] for n in best if n in factor_panels}
    if factor_panels is None:
        factor_panels = PN.price_factor_panels(data)
    fn = list(factor_panels)
    if candidate_models is None:
        candidate_models = default_candidate_models(include_listwise=include_listwise)
    if factor_source is not None and len(fn) > 8:
        if rank_score is None:                      # prescreen=False 也要有排序依据
            from ..research.factor_lab import daily_rank_ic
            close = mf.build_panel(data, "close")
            fwd = close.shift(-horizon) / close - 1.0
            rank_score = {f: abs(float(daily_rank_ic(factor_panels[f], fwd).mean()) or 0.0)
                          for f in fn}
            rank_score = {k: (v if v == v else 0.0) for k, v in rank_score.items()}
        subsets = _adaptive_subsets(fn, rank_score)
    else:
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
    lb = lb.head(top_n) if top_n else lb
    if factor_source is not None:
        lb.attrs["prescreen_report"] = prescreen_report
        lb.attrs["factor_panels"] = factor_panels
    return lb

def ensemble_top(data, lb, k=3, factor_panels=None, candidate_models=None, horizon=21,
                 rebalance="ME", top_frac=0.2, n_quantiles=5,
                 commission_bps=1.0, slippage_bps=1.0, **kw):
    """标准化 rank-average 集成:把 `search` 排行榜前 k 个 (因子子集×模型) 配置的预测,
    逐日转 rank 百分位后平均成一个集成排序,再用同一张诚实记分卡重打分。

    为什么用 rank 而不是原始预测值平均:不同模型的预测尺度天差地别(ridge 输出 ±2%,
    lgbm 可能 ±20%),直接平均等于让尺度最大的模型独裁;rank 百分位把每个模型都归一到
    [0,1],成员平权,这正是"简单组合稳定优于单个复杂模型"在横截面上的标准做法。
    返回 {scorecard, daily, quantile_fwd, preds, members}(members 为实际用到的配置)。
    """
    from .xsec_eval import _scorecard
    if factor_panels is None:
        factor_panels = PN.price_factor_panels(data)
    if candidate_models is None:
        candidate_models = {"ridge": M.RidgeModel(alpha=1.0)}
    frames, members = [], []
    for _, row in lb.head(k).iterrows():
        sub = tuple(str(row["factors"]).split("+"))
        mdl = candidate_models.get(row["model"], M.RidgeModel(alpha=1.0))
        pj = {f: factor_panels[f] for f in sub if f in factor_panels}
        if not pj:
            continue
        res = evaluate_cross_section(data, model=mdl, panels=pj, horizon=horizon,
                                     rebalance=rebalance, top_frac=top_frac,
                                     n_quantiles=n_quantiles, commission_bps=commission_bps,
                                     slippage_bps=slippage_bps, **kw)
        p = res.get("preds")
        if p is None or p.empty:
            continue
        p = p.copy()
        p["rk"] = p.groupby("date")["pred"].rank(pct=True)   # 归一到 [0,1] 的截面百分位
        frames.append(p.set_index(["date", "symbol"])[["rk", "fwd"]])
        members.append({"factors": row["factors"], "model": row["model"]})
    if not frames:
        return {"scorecard": {"verdict": "样本不足"}, "daily": pd.DataFrame(),
                "quantile_fwd": pd.Series(dtype=float), "preds": pd.DataFrame(), "members": []}
    ranks = pd.concat([f["rk"] for f in frames], axis=1)
    fwd = pd.concat([f["fwd"] for f in frames], axis=1).bfill(axis=1).iloc[:, 0]  # first non-null
    ens = pd.DataFrame({"pred": ranks.mean(axis=1), "fwd": fwd}).dropna().reset_index()
    out = _scorecard(ens, top_frac, n_quantiles, horizon, commission_bps + slippage_bps)
    out["preds"] = ens
    out["members"] = members
    return out
