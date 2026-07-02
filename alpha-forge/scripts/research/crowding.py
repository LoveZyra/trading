"""因子拥挤度监控(优化方案 §2.3)。

一个因子"有效"和"拥挤"是两回事:拥挤的因子照样有 IC,但持仓和别人高度重合、
多空收益和别人同涨同跌,一旦资金撤退就是踩踏(2007 年 quant quake 的教训)。
这里给三个可观察的拥挤代理 + 一个综合分:
  1. holdings_overlap          —— 多头桶 Jaccard(和别的因子买的是不是同一批票);
  2. factor_return_correlation —— 因子多空日收益的滚动相关(收益来源是否同源);
  3. valuation_spread          —— 多空桶估值价差(拥挤资金涌入会把价差买窄)。
所有 panel 均为 date×symbol 宽表,与 xsec/panel.py 口径一致。
另附 fit_hyperbolic_decay:α(t)=K/(1+λt) 的双曲衰减拟合工具。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _bucket_masks(panel: pd.DataFrame, top_frac: float):
    """按截面百分位秩切多头/空头桶(布尔宽表)。用 pct rank 而不是固定名额,
    是为了对缺失票的日子仍然稳定——桶宽随当日有效截面自适应。"""
    pct = panel.rank(axis=1, pct=True)
    return pct >= 1.0 - top_frac, pct <= top_frac


def holdings_overlap(factor_a_panel: pd.DataFrame, factor_b_panel: pd.DataFrame,
                     top_frac: float = 0.2) -> pd.Series:
    """逐日多头桶 Jaccard = |A∩B| / |A∪B|,返回按日 Series ∈ [0,1]。

    why Jaccard 而不是相关系数:拥挤是"持仓层面"的现象——两个因子值相关 0.5
    可以对应几乎相同的 top 桶,也可以几乎不重叠;直接数票最诚实。
    截面有效名字太少(<5)的日子置 NaN。
    """
    a, b = factor_a_panel.align(factor_b_panel, join="inner")
    ta, _ = _bucket_masks(a, top_frac)
    tb, _ = _bucket_masks(b, top_frac)
    inter = (ta & tb).sum(axis=1).astype(float)
    union = (ta | tb).sum(axis=1).astype(float)
    out = inter / union.replace(0.0, np.nan)
    n_valid = (a.notna() & b.notna()).sum(axis=1)
    out[n_valid < 5] = np.nan
    return out


def long_short_returns(factor_panel: pd.DataFrame, close_panel: pd.DataFrame,
                       top_frac: float = 0.2) -> pd.Series:
    """因子多空组合的日收益:用 T-1 的因子分桶,吃 T 日收益(因果:今天收盘才
    知道的因子值不能决定今天的持仓)。等权多头桶 - 等权空头桶。"""
    close = close_panel.reindex(index=factor_panel.index, columns=factor_panel.columns)
    ret1 = close.pct_change()
    top, bot = _bucket_masks(factor_panel.shift(1), top_frac)
    return (ret1.where(top).mean(axis=1) - ret1.where(bot).mean(axis=1)).rename("ls")


def factor_return_correlation(factor_panel: pd.DataFrame, other_panels: dict,
                              close_panel: pd.DataFrame, lookback: int = 63) -> pd.Series:
    """本因子多空收益 vs 各其他因子多空收益的滚动相关,再对"其他因子"取均值。

    why 用多空收益相关而不是因子值相关:两个因子值构造迥异也可能靠同一条
    收益来源赚钱(收益同源才是拥挤的实质);反之值相关高但收益不同源不算挤。
    """
    mine = long_short_returns(factor_panel, close_panel)
    cors = []
    for _name, p in (other_panels or {}).items():
        other = long_short_returns(p, close_panel)
        cors.append(mine.rolling(lookback, min_periods=max(20, lookback // 2)).corr(other))
    if not cors:
        return pd.Series(np.nan, index=factor_panel.index)
    return pd.concat(cors, axis=1).mean(axis=1)


def valuation_spread(factor_panel: pd.DataFrame, value_panel: pd.DataFrame | None,
                     top_frac: float = 0.2) -> pd.Series:
    """多头桶与空头桶的估值因子差(value(多头) - value(空头)),按日 Series。

    逻辑:拥挤资金涌向因子多头,会把多头买贵、空头打便宜,估值价差收窄
    (|spread| 掉到历史低位)是经典拥挤信号(参考 AQR 的 value spread 研究)。
    value_panel 缺省(None)时优雅降级为全 NaN——没有估值数据就不硬编。
    """
    if value_panel is None:
        return pd.Series(np.nan, index=factor_panel.index)
    v = value_panel.reindex(index=factor_panel.index, columns=factor_panel.columns)
    top, bot = _bucket_masks(factor_panel, top_frac)
    return (v.where(top).mean(axis=1) - v.where(bot).mean(axis=1)).rename("value_spread")


def crowding_score(factor_panel: pd.DataFrame, other_panels: dict,
                   close_panel: pd.DataFrame, *, lookback: int = 252,
                   value_panel: pd.DataFrame | None = None,
                   top_frac: float = 0.2) -> dict:
    """综合拥挤度 ∈ [0,1] + 分项 + warning(>0.7)。

    分项口径(全部映射到 [0,1],越高越拥挤):
      holdings_overlap   近 lookback 日、对各 other 的均值取最大(最像谁就按谁算);
      return_correlation 近 lookback 日滚动相关均值,负相关(对冲)不算拥挤,截到 0;
      valuation_spread   近端 |价差| 在历史分布中的分位取反(价差越窄越拥挤),可选。
    综合 = 可得分项均值。宁可少一个分项也不用假数据凑——缺 value_panel 就两项。
    """
    components: dict = {}
    tail = int(min(lookback, len(factor_panel)))

    if other_panels:
        ovs = [float(holdings_overlap(factor_panel, p, top_frac).tail(tail).mean())
               for p in other_panels.values()]
        ovs = [o for o in ovs if np.isfinite(o)]
        if ovs:
            components["holdings_overlap"] = max(ovs)
        rc = factor_return_correlation(factor_panel, other_panels, close_panel,
                                       lookback=min(63, max(20, tail // 4)))
        rc_recent = float(rc.tail(tail).mean())
        if np.isfinite(rc_recent):
            components["return_correlation"] = float(np.clip(rc_recent, 0.0, 1.0))

    if value_panel is not None:
        sp = valuation_spread(factor_panel, value_panel, top_frac).abs().dropna()
        if len(sp) > 40:
            recent = float(sp.tail(63).mean())
            pct = float((sp <= recent).mean())          # 价差分位:低 → 拥挤
            components["valuation_spread"] = float(np.clip(1.0 - pct, 0.0, 1.0))

    score = float(np.mean(list(components.values()))) if components else float("nan")
    warning = bool(np.isfinite(score) and score > 0.7)
    return {"score": score, "components": components, "warning": warning}


def fit_hyperbolic_decay(ic_series) -> dict:
    """拟合双曲衰减 α(t) = K / (1 + λ t),返回 {K, lam, half_life, r2}。

    函数形式取自 arXiv 2512.11913(注意:该文 v2 已撤稿,这里只借它的函数形式
    作曲线拟合工具,不外推其"alpha 必然双曲衰减"的结论——结论要靠自己的数据说话)。
    做法:1/α = 1/K + (λ/K)·t 是线性的,对 α>0 的观测做纯 numpy 最小二乘,
    避免非线性优化的初值依赖(也避免依赖 scipy)。half_life = 1/λ
    (α 掉到 K/2 的时刻)。观测太少或拟合出非正 K 时全 NaN——不硬给数。
    """
    y = pd.Series(ic_series).astype(float).reset_index(drop=True)
    t = np.arange(len(y), dtype=float)
    m = y.notna().values & (y.values > 1e-12)
    nan_out = {"K": float("nan"), "lam": float("nan"),
               "half_life": float("nan"), "r2": float("nan")}
    if m.sum() < 4:
        return nan_out
    tt, yy = t[m], y.values[m]
    slope, intercept = np.polyfit(tt, 1.0 / yy, 1)
    if intercept <= 0:
        return nan_out
    K = 1.0 / intercept
    lam = slope * K
    pred = K / (1.0 + lam * tt)
    ss_res = float(np.sum((yy - pred) ** 2))
    ss_tot = float(np.sum((yy - yy.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    half_life = 1.0 / lam if lam > 0 else float("inf")
    return {"K": float(K), "lam": float(lam),
            "half_life": float(half_life), "r2": float(r2)}
