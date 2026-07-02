"""因子质量预筛管线(Round 11,优化方案 §2.9 + §2.2/2.3/2.4 的"接入"条款)。

把 Round 10 的质量套件串成一条"进模型之前"的流水线:
  factor_lab.daily_rank_ic   -> 预测力闸(弱因子剔除)
  decay_monitor.half_life    -> 寿命闸(信号活不过调仓周期的剔除)
  逐日截面 spearman 相关       -> 冗余闸(贪心去克隆)
  crowding.crowding_score    -> 拥挤闸(不剔、降权)

why 需要这层:factor_zoo 一次能产出几十条表达式因子,全部直接塞进
xsec_autoresearch.search 是组合爆炸(子集数指数增长)+ 多重检验(几十次
几乎相同的回测烧光显著性)+ 模型容量浪费(冗余特征让 ridge 权重不稳)。
先用便宜的统计闸把池子收敛到 max_factors 个"各有各的用处"的因子,
再让 walk-forward 评测花大钱。

无前视:所有统计只用传入面板窗口内的数据(RankIC 的前向收益在窗口尾部
horizon 天自然缺失,不会用窗口外数据补齐)。注意这仍是"研究窗内"的筛选
——若要严格 OOS,请在训练窗上 prescreen、在留出窗上评测(xsec_eval 的
purged walk-forward 已保证模型层无前视)。

冗余闸实现取舍(§2.2 给了两条路,这里选相关阈值而非 incremental_ic ratio):
incremental_ic 要对每个候选逐日 lstsq 正交化(纯 Python 按日循环,慢一个
量级),且 raw_ic 很小时 ratio 发散(NaN/爆炸)导致判定不稳;成对"逐日截面
spearman 相关的时间均值"(与 factor_lab.factor_correlation_matrix 同口径)
是全向量化的,阈值含义直白,而且对主要冗余形态——"0.99 相关的克隆因子"
——最直接。按 |RankIC| 降序贪心保证留下的是每个相关簇里预测力最强的代表。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .factor_lab import daily_rank_ic
from .decay_monitor import half_life
from .crowding import crowding_score

REPORT_COLUMNS = ("name", "rankic", "half_life", "max_corr_selected",
                  "crowding", "decision")


def prescreen_factors(panels: dict, close_panel: pd.DataFrame, *,
                      horizon: int = 21, max_factors: int = 12,
                      min_abs_rankic: float = 0.01,
                      max_incremental_corr: float = 0.85,
                      drop_fast_decay: bool = True, rebalance_days: int = 21,
                      crowding_penalty: bool = True,
                      verbose: bool = False) -> dict:
    """四道闸的因子预筛。返回 {selected, report, dropped}。

    panels        {name: date×symbol 宽表};close_panel 同口径收盘宽表。
    流程(顺序即优先级,先便宜后昂贵):
      a) |时均 RankIC| < min_abs_rankic(或 NaN)     -> dropped["weak"]
         (取绝对值:负 IC 因子照留,符号交给下游模型学——ridge 的权重
          可以为负,人工翻号反而多一个出错的地方);
      b) half_life(三口径中位数)< rebalance_days     -> dropped["fast_decay"]
         (信号还没等到下次调仓就死了一半,拿到的只是残值;half_life 测不出
          [NaN/inf] 时不剔——"没测出衰减"不等于"衰减快");
      c) 按 |RankIC| 降序贪心:与任一已选因子的逐日截面 spearman 相关
         (时间均值)的绝对值 > max_incremental_corr  -> dropped["redundant"];
      d) crowding_score > 0.7 的已选因子不剔,但排序分折半(拥挤因子仍有
         信息,只是踩踏风险高,降权让位给同等预测力的冷门因子)并记入 report;
      e) 按(可能折半后的)|RankIC| 排序取前 max_factors,溢出的
         dropped["capacity"]。
    report 每行:name / rankic / half_life / max_corr_selected / crowding /
    decision(selected|weak|fast_decay|redundant|capacity)。
    selected 按最终排序分降序,可直接作为 evaluate_cross_section /
    ml_factor_backtest(panels=...) 的输入。
    """
    close = close_panel
    fwd = close.shift(-horizon) / close - 1.0
    aligned = {str(k): v.reindex(index=close.index, columns=close.columns)
               for k, v in panels.items()}
    rows = {name: dict(name=name, rankic=np.nan, half_life=np.nan,
                       max_corr_selected=np.nan, crowding=np.nan, decision=None)
            for name in aligned}
    dropped: dict = {}

    def _say(msg):
        if verbose:
            print(f"[prescreen] {msg}")

    # ---- a) 预测力闸 -------------------------------------------------------
    rankic: dict = {}
    survivors: list = []
    for name, p in aligned.items():
        ic = float(daily_rank_ic(p, fwd).mean())
        rankic[name] = ic
        rows[name]["rankic"] = ic
        if not np.isfinite(ic) or abs(ic) < min_abs_rankic:
            dropped[name] = "weak"
            rows[name]["decision"] = "weak"
            _say(f"{name}: RankIC={ic:+.4f} -> weak")
        else:
            survivors.append(name)

    # ---- b) 寿命闸 ---------------------------------------------------------
    kept: list = []
    for name in survivors:
        hl = float(half_life(aligned[name], close)["median"])
        rows[name]["half_life"] = hl
        if drop_fast_decay and np.isfinite(hl) and hl < rebalance_days:
            dropped[name] = "fast_decay"
            rows[name]["decision"] = "fast_decay"
            _say(f"{name}: half_life={hl:.1f}d < {rebalance_days}d -> fast_decay")
        else:
            kept.append(name)

    # ---- c) 冗余闸(|RankIC| 降序贪心) ------------------------------------
    kept.sort(key=lambda n: -abs(rankic[n]))
    selected: list = []
    for name in kept:
        max_corr = 0.0
        for s in selected:
            c = float(daily_rank_ic(aligned[name], aligned[s]).mean())
            if np.isfinite(c):
                max_corr = max(max_corr, abs(c))
        rows[name]["max_corr_selected"] = max_corr
        if selected and max_corr > max_incremental_corr:
            dropped[name] = "redundant"
            rows[name]["decision"] = "redundant"
            _say(f"{name}: corr_with_selected={max_corr:.3f} -> redundant")
        else:
            selected.append(name)

    # ---- d) 拥挤闸(降权不剔) ---------------------------------------------
    eff_score = {}
    for name in selected:
        score = abs(rankic[name])
        if crowding_penalty and len(selected) > 1:
            others = {s: aligned[s] for s in selected if s != name}
            cs = crowding_score(aligned[name], others, close)
            rows[name]["crowding"] = cs["score"]
            if np.isfinite(cs["score"]) and cs["score"] > 0.7:
                score *= 0.5
                _say(f"{name}: crowding={cs['score']:.2f} -> score halved")
        eff_score[name] = score

    # ---- e) 容量闸 ---------------------------------------------------------
    final = sorted(selected, key=lambda n: -eff_score[n])[:max(int(max_factors), 1)]
    for name in selected:
        if name in final:
            rows[name]["decision"] = "selected"
        else:
            dropped[name] = "capacity"
            rows[name]["decision"] = "capacity"
            _say(f"{name}: beyond max_factors={max_factors} -> capacity")

    report = pd.DataFrame([rows[n] for n in aligned], columns=list(REPORT_COLUMNS))
    _say(f"selected {len(final)}/{len(aligned)}: {final}")
    return {"selected": {n: aligned[n] for n in final},
            "report": report, "dropped": dropped}
