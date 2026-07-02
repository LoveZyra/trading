"""数据驱动选池(市值/流动性/指数成分 基座 + 热门/龙头 软打分)单测。全部离线合成。"""
import os, sys, warnings
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.xsec import universe as U


def _synth_prices(sym, trend, n=300, seed=0):
    rng = np.random.default_rng(seed)
    r = rng.normal(trend, 0.01, n)
    c = 100 * np.exp(np.cumsum(r))
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame({"close": c}, index=idx)


def _meta(sym, mktcap, adv, sector, price=100, hi=110, a200=95, idx=None):
    return {"mktcap": mktcap, "adv_usd": adv, "sector": sector, "price": price,
            "high52": hi, "low52": 50, "avg200": a200, "in_index": idx or []}


def test_compute_scores_columns_and_order():
    meta = {"BIGHOT": _meta("BIGHOT", 5e11, 5e9, "A", price=109, a200=80),
            "SMALLWEAK": _meta("SMALLWEAK", 3e9, 3e7, "A", price=60, a200=100)}
    df = U.compute_scores(meta)
    for c in ["size_z", "liq_z", "hot_z", "hi52_z", "lead_z", "score"]:
        assert c in df.columns
    # 大而强热门的应排在前
    assert df.index[0] == "BIGHOT"


def test_gate_filters_cap_and_adv():
    meta = {"KEEP": _meta("KEEP", 1e10, 1e8, "A"),
            "TINYCAP": _meta("TINYCAP", 5e8, 1e8, "A"),      # 市值不足
            "ILLIQ": _meta("ILLIQ", 1e10, 1e6, "A")}          # 流动性不足
    res = U.build_scored_universe(meta, per_sector=10, cap_min=2e9, adv_min=2e7)
    assert "KEEP" in res["selected"]
    assert "TINYCAP" not in res["gate_pass"]
    assert "ILLIQ" not in res["gate_pass"]


def test_per_sector_topn():
    meta = {}
    for i in range(15):
        meta[f"A{i}"] = _meta(f"A{i}", 1e10 + i * 1e9, 1e8, "SecA", price=100 + i)
    for i in range(3):
        meta[f"B{i}"] = _meta(f"B{i}", 1e10, 1e8, "SecB")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = U.build_scored_universe(meta, per_sector=10, cap_min=2e9, adv_min=2e7)
    sel = res["selected_table"]
    assert (sel["sector"] == "SecA").sum() == 10   # A 板块截断到 10
    assert (sel["sector"] == "SecB").sum() == 3    # B 板块只有 3 只,全取


def test_sectors_override():
    meta = {"X": _meta("X", 1e10, 1e8, "Technology"), "Y": _meta("Y", 1e10, 1e8, "Technology")}
    res = U.build_scored_universe(meta, per_sector=10, cap_min=2e9, adv_min=2e7,
                                  sectors_override={"X": "核电/电力"})
    assert res["table"].loc["X", "sector"] == "核电/电力"
    assert res["table"].loc["Y", "sector"] == "Technology"


def test_require_index_gate():
    meta = {"IN": _meta("IN", 1e10, 1e8, "A", idx=["SP500"]),
            "OUT": _meta("OUT", 1e10, 1e8, "A", idx=[])}
    res = U.build_scored_universe(meta, per_sector=10, cap_min=2e9, adv_min=2e7, require_index="SP500")
    assert "IN" in res["gate_pass"] and "OUT" not in res["gate_pass"]


def test_meta_from_fmp_parse():
    quotes = [{"symbol": "NVDA", "price": 200.0, "volume": 1_000_000, "marketCap": 4.8e12,
               "yearHigh": 236.0, "yearLow": 151.0, "priceAvg50": 210.0, "priceAvg200": 190.0}]
    profiles = {"NVDA": {"sector": "Technology", "averageVolume": 2_000_000}}
    meta = U.meta_from_fmp(quotes, profiles=profiles, index_members={"SP500": {"NVDA"}, "NDX": {"NVDA"}})
    m = meta["NVDA"]
    assert m["mktcap"] == 4.8e12
    assert m["adv_usd"] == 200.0 * 2_000_000          # 用 profile.averageVolume
    assert m["sector"] == "Technology"
    assert set(m["in_index"]) == {"SP500", "NDX"}


def test_prices_drive_hotness():
    # 有价格时,热门度来自动量:强趋势应比弱趋势 hot_z 高
    prices = {"UP": _synth_prices("UP", 0.004, seed=1), "DOWN": _synth_prices("DOWN", -0.002, seed=2)}
    meta = {"UP": _meta("UP", 1e10, 1e8, "A"), "DOWN": _meta("DOWN", 1e10, 1e8, "A")}
    df = U.compute_scores(meta, prices=prices)
    assert df.loc["UP", "hot_z"] > df.loc["DOWN", "hot_z"]


def test_empty_meta_safe():
    res = U.build_scored_universe({}, per_sector=10)
    assert res["selected"] == []


def test_merge_manual_overlay():
    meta = {"BIG": _meta("BIG", 5e11, 1e9, "S"), "CONV": _meta("CONV", 3e9, 5e7, "S")}
    res = U.build_scored_universe(meta, per_sector=1, cap_min=2e9, adv_min=2e7)
    assert res["selected"] == ["BIG"]                       # 数据池每板块只取1只
    m = U.merge_manual(res, ["CONV"])
    assert m["selected"] == ["BIG", "CONV"]                 # 主观名附加在后
    assert m["manual"] == ["CONV"] and m["source"]["CONV"] == "manual" and m["source"]["BIG"] == "data"


def test_merge_manual_skips_unknown():
    meta = {"BIG": _meta("BIG", 5e11, 1e9, "S")}
    res = U.build_scored_universe(meta, per_sector=10, cap_min=2e9, adv_min=2e7)
    m = U.merge_manual(res, ["NOPE"])                       # 无 meta 的主观名被跳过
    assert "NOPE" not in m["selected"] and m["manual"] == []
