"""横截面因子面板:{symbol: OHLCV} -> {factor_name: 宽表(date×symbol)},均做截面 z-score。
价格类、尺度无关因子,所有市场可比;与 models.build_factor_panels(基本面/舆情) 可合并。
"""
from __future__ import annotations
import numpy as np, pandas as pd
from ..strategies import multi_factor as mf

def price_factor_panels(data: dict) -> dict:
    close = mf.build_panel(data, "close")
    lc = np.log(close); r1 = lc.diff()
    z = lambda df: df.apply(mf._cross_section_z, axis=1)
    P = {
        "mom20":  z(lc.diff(20)),  "mom60": z(lc.diff(60)), "mom120": z(lc.diff(120)),
        "rev5":   z(lc.diff(5)),   "rev10": z(lc.diff(10)),
        "vol20":  z(r1.rolling(20).std()),
        "dist_high60": z(close / close.rolling(60).max() - 1),
        "dist_low60":  z(close / close.rolling(60).min() - 1),
    }
    return P

def forward_return(data: dict, horizon: int = 21) -> pd.DataFrame:
    close = mf.build_panel(data, "close")
    return close.shift(-horizon) / close - 1.0

def merge_panels(*panel_dicts) -> dict:
    out = {}
    for d in panel_dicts:
        if d: out.update(d)
    return out


# ========================================================================
# Round10 §2.7:横截面标准化 / 中性化(对标 Qlib CSZScoreNorm / Barra 残差化)
# ========================================================================

def cs_zscore(panel: pd.DataFrame, groupby: dict | None = None) -> pd.DataFrame:
    """截面 z-score(Qlib CSZScoreNorm 对标)。

    why:横截面排序只关心"同一天谁比谁强",z-score 把不同量纲的因子拉到同一
    尺度才能混合/回归。groupby={symbol: group} 时按组内做 z(如按行业/市场),
    消除组间水平差异,只留组内相对强弱——A股与美股、金融与科技的因子原始值
    没有可比性,组内 z 是最便宜的修正。
    NaN 保持 NaN(零方差退化路径不会把缺失值填成 0 混进排序)。
    """
    if groupby:
        out = pd.DataFrame(np.nan, index=panel.index, columns=panel.columns)
        key = pd.Series({c: groupby.get(c, "__nogroup__") for c in panel.columns})
        for _, cols in key.groupby(key).groups.items():
            cols = list(cols)
            out[cols] = panel[cols].apply(mf._cross_section_z, axis=1)
        return out.where(panel.notna())
    return panel.apply(mf._cross_section_z, axis=1).where(panel.notna())


def neutralize_panel(factor_panel: pd.DataFrame, *, sector_map: dict | None = None,
                     style_panels: dict | None = None) -> pd.DataFrame:
    """因子中性化:行业=组内去均值;风格=逐日截面 OLS(含截距)取残差;可叠加
    (先行业后风格)。

    why:未中性化的因子 IC 往往是行业贝塔/风格暴露(市值、波动)的搭便车——
    看似动量有预测力,其实是"押了一个行业/一类股票"。去掉这些系统性暴露后
    剩下的残差才是真正的选股 alpha,这是 Barra/Qlib 流程的标准步骤。
    因果性:每天的去均值/回归只用当天截面,不跨日、不碰未来。
    实现细节:
      * sector_map={symbol: 行业};不在 map 里的标的归入同一 "__nogroup__" 组。
      * style_panels={name: 宽表(date×symbol)};回归用 np.linalg.lstsq。
      * 某日有效样本数 <= 自变量数+1 时无法回归,该日保留原值(而非清空),
        避免风格因子 rolling warmup 期把整段历史抹掉;
      * y 有效但 style 缺失的标的,该日置 NaN(残差无定义,不能混入排序)。
    """
    out = factor_panel.astype(float).copy()
    if sector_map:
        key = pd.Series({c: sector_map.get(c, "__nogroup__") for c in out.columns})
        for _, cols in key.groupby(key).groups.items():
            cols = list(cols)
            sub = out[cols]
            out[cols] = sub.sub(sub.mean(axis=1), axis=0)
    if style_panels:
        styles = [v.reindex(index=out.index, columns=out.columns).astype(float)
                  for v in style_panels.values()]
        res = out.copy()
        for i in range(len(out.index)):
            y = out.iloc[i].to_numpy()
            X = np.column_stack([s.iloc[i].to_numpy() for s in styles] + [np.ones(len(y))])
            m = np.isfinite(y) & np.isfinite(X).all(axis=1)
            if int(m.sum()) <= X.shape[1]:
                continue                                # 样本不足:该日保留原值
            beta, *_ = np.linalg.lstsq(X[m], y[m], rcond=None)
            row = np.full(len(y), np.nan)
            row[m] = y[m] - X[m] @ beta
            res.iloc[i] = row
        out = res
    return out
