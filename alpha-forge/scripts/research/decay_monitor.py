"""因子衰减监控(优化方案 §2.4)。

因子不是资产,是会折旧的耗材:发表、拥挤、市场结构变化都会磨掉它的 IC。
这里回答三个运营问题:
  1. 它现在还活着吗?      rolling_ic / decay_warning
  2. 信号能撑多少天?      ic_decay / half_life(信号寿命 vs 调仓周期的匹配)
  3. 换了市况还活着吗?    mrp(Minimum Regime Performance:最差 regime 的年化)
所有 panel 均为 date×symbol 宽表,与 xsec/panel.py 口径一致。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .factor_lab import daily_rank_ic


def _fwd(close_panel: pd.DataFrame, horizon: int) -> pd.DataFrame:
    return close_panel.shift(-horizon) / close_panel - 1.0


def rolling_ic(factor_panel: pd.DataFrame, close_panel: pd.DataFrame,
               horizon: int = 21, window: int = 63) -> pd.Series:
    """逐日截面 RankIC 的滚动均值。单日截面 IC 噪声极大(±0.3 常见),
    只有滚动均值才能看出"因子在变钝"这种慢变量。"""
    daily = daily_rank_ic(factor_panel, _fwd(close_panel, horizon))
    return daily.rolling(window, min_periods=max(10, window // 3)).mean()


def ic_decay(factor_panel: pd.DataFrame, close_panel: pd.DataFrame,
             horizons=(1, 5, 10, 21, 42, 63)) -> dict:
    """各持有期 h 的时均 RankIC:{h: mean_rankic}。

    读法:曲线在哪个 h 见顶/掉到一半,决定了合理的调仓频率——信号 5 天就没了
    却按月调仓,等于拿着过期票据交易。
    """
    return {int(h): float(daily_rank_ic(factor_panel, _fwd(close_panel, int(h))).mean())
            for h in horizons}


def _lagged_daily_ic(factor_panel: pd.DataFrame, close_panel: pd.DataFrame,
                     lag: int) -> float:
    """因子(T)对 [T+lag, T+lag+1] 单日收益的时均 RankIC。

    why 用"滞后单日 IC"而不是累计 h 日 IC 来测寿命:累计 IC 把"信号累积"和
    "信号衰减"混在一起(持久因子的累计 IC 先升后降,拟合不出衰减率);
    滞后单日 IC 隔离出"第 lag 天还剩多少预测力",对 AR(1) 型信号正好按 ρ^lag 衰减。
    """
    r1 = close_panel.pct_change().shift(-lag)  # T+lag 日的单日收益,对齐到 T
    return float(daily_rank_ic(factor_panel, r1).mean())


def _exp_half_life(xs: dict) -> float:
    """对 {lag: 值} 拟合 ln|值| = a + b·lag,半衰期 = ln2/(-b)。只用与首项同号
    且非零的点(翻号点进 log 就是造数);衰减不显著(b>=0)返回 inf——"测不出
    衰减"和"衰减极快"必须可区分。"""
    items = sorted((int(k), float(v)) for k, v in xs.items())
    if not items:
        return float("nan")
    s = np.sign(items[0][1])
    pts = [(k, abs(v)) for k, v in items if np.isfinite(v) and np.sign(v) == s and abs(v) > 1e-12]
    if len(pts) < 3 or s == 0:
        return float("nan")
    ks = np.array([p[0] for p in pts], float)
    ys = np.log(np.array([p[1] for p in pts], float))
    b, _a = np.polyfit(ks, ys, 1)
    if b >= -1e-12:
        return float("inf")
    return float(np.log(2.0) / -b)


def half_life(factor_panel: pd.DataFrame, close_panel: pd.DataFrame,
              method: str = "auto") -> dict:
    """三种口径的信号半衰期(天)+ 中位数:{ar1, ic_decay, quantile_spread, median}。

    ar1            因子自身的截面持续性:各标的 lag-1 自相关的中位 ρ,
                   半衰期 = ln2 / ln(1/ρ)。测的是"因子值翻篇多快"(换手上限);
    ic_decay       滞后单日 RankIC 随 lag 的指数衰减拟合。测"预测力活多久";
    quantile_spread 按因子分桶后,第 lag 天多空单日价差的衰减拟合。测"可交易的
                   钱活多久"(IC 可能靠小票撑着,价差口径更接近组合层)。
    三法答案常不一致(这正是分开算的意义),median 取有限正值的中位数做汇总。
    method 目前仅 "auto"(全算),参数保留给未来单法快速路径。
    """
    if method != "auto":
        raise ValueError(f"unknown method {method!r}; only 'auto' is implemented")
    out: dict = {}

    with np.errstate(invalid="ignore", divide="ignore"):
        # 常数列的自相关是 0/0 —— 该列本就没有"翻篇速度"可言,静默给 NaN 即可
        rhos = [factor_panel[c].autocorr(lag=1) for c in factor_panel.columns
                if factor_panel[c].notna().sum() > 10]
    rho = float(np.nanmedian(rhos)) if rhos else float("nan")
    out["ar1"] = float(np.log(2.0) / -np.log(rho)) if 0.0 < rho < 1.0 else float("nan")

    lags = (1, 3, 5, 10, 21, 42)
    out["ic_decay"] = _exp_half_life(
        {lag: _lagged_daily_ic(factor_panel, close_panel, lag) for lag in lags})

    r1 = close_panel.pct_change()
    pct = factor_panel.rank(axis=1, pct=True)
    top, bot = pct >= 0.8, pct <= 0.2
    spreads = {}
    for lag in lags:
        fut = r1.shift(-lag)
        spreads[lag] = float((fut.where(top).mean(axis=1)
                              - fut.where(bot).mean(axis=1)).mean())
    out["quantile_spread"] = _exp_half_life(spreads)

    vals = [v for v in out.values() if np.isfinite(v) and v > 0]
    out["median"] = float(np.median(vals)) if vals else float("nan")
    return out


def decay_warning(factor_panel: pd.DataFrame, close_panel: pd.DataFrame, *,
                  rebalance_days: int = 21, horizon: int = 21) -> dict:
    """衰减告警:{warning: bool, reasons: [...]}(附 recent_ic / longterm_ic / half_life)。

    触发条件(任一即告警):
      1. rolling IC 近端均值跌破长期均值的一半(长期均值为正才有意义——
         本来就没有 IC 的因子谈不上"衰减");
      2. 信号半衰期中位数 < 调仓周期:还没等到下次调仓,信号已经死了一半,
         实际拿到的是曲线尾部的残值。
    """
    reasons: list = []
    roll = rolling_ic(factor_panel, close_panel, horizon=horizon).dropna()
    recent = longterm = float("nan")
    if len(roll) >= 40:
        longterm = float(roll.mean())
        recent = float(roll.tail(max(10, len(roll) // 10)).mean())
        if longterm > 0.005 and recent < 0.5 * longterm:
            reasons.append(f"rolling IC 近端 {recent:.4f} 跌破长期均值 "
                           f"{longterm:.4f} 的一半 — 因子在钝化")
    hl = half_life(factor_panel, close_panel)
    if np.isfinite(hl["median"]) and hl["median"] < rebalance_days:
        reasons.append(f"信号半衰期 {hl['median']:.1f} 天 < 调仓周期 "
                       f"{rebalance_days} 天 — 调仓频率跟不上信号寿命")
    return {"warning": bool(reasons), "reasons": reasons,
            "recent_ic": recent, "longterm_ic": longterm,
            "half_life": hl["median"]}


def mrp(strategy_returns, regime_series) -> float:
    """Minimum Regime Performance:各 regime 状态内年化收益(mean×252)的最小值。

    why 取最小而不是平均:平均会让牛市里躺赢的策略掩盖"熊市清零"的事实;
    MRP 是短板逻辑——策略的可持续性由它最差的市况决定(§2.4 的验收口径)。
    regime_series 按索引对齐到收益;对不上的日子丢弃,空集返回 NaN。
    """
    r = pd.Series(strategy_returns).astype(float)
    g = pd.Series(regime_series).reindex(r.index)
    m = r.notna() & g.notna()
    if not m.any():
        return float("nan")
    ann = r[m].groupby(g[m]).mean() * 252.0
    return float(ann.min())
