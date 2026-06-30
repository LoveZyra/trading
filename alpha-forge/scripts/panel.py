"""横截面因子面板:{symbol: OHLCV} -> {factor_name: 宽表(date×symbol)},均做截面 z-score。
价格类、尺度无关因子,所有市场可比;与 models.build_factor_panels(基本面/舆情) 可合并。
"""
from __future__ import annotations
import numpy as np, pandas as pd
from .strategies import multi_factor as mf

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
