"""Round 11 §2.9 全链路测试:因子库 -> 质量预筛 -> 自动研究 -> 模型训练/预测 ->
选股产出,加研究记忆与跨市场热启动。只用 numpy/pandas,离线、固定 seed。

合成宇宙(参考 test_round9._universe,但植入可控信号):
  r[t,i] = mu_i + fast_beta * s[t-1,i] + noise
  mu_i    持续漂移  -> 慢衰减的横截面动量(真因子,应被留下);
  s       rho=0.25 的快速 AR(1) -> 有真实预测力但半衰期 <1 天(应被 fast_decay 剔);
另外构造纯噪声因子(应被 weak 剔)和动量的高相关克隆(应被 redundant 剔)。
作弊因子(用未来收益构造)不出现——那是 validate_factor 的职责,不是 prescreen 的。
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from scripts.research import models as Mdl
from scripts.research.prescreen import prescreen_factors
from scripts.research.research_memory import (ResearchMemory, cluster_strategies,
                                              warm_start_search)
from scripts.strategies import multi_factor as mf
from scripts.xsec import panel as PN, xsec_autoresearch as XAR
from scripts.xsec.xsec_eval import evaluate_cross_section

warnings.filterwarnings("ignore", message=".*只标的.*")
warnings.filterwarnings("ignore", message=".*skipped.*factors.*")


def _xuniverse(n_sym=24, T=300, seed=11, fast_beta=0.006):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=T)
    mu = rng.normal(0, 0.002, n_sym)
    s = np.zeros((T, n_sym))
    e = rng.standard_normal((T, n_sym))
    for t in range(1, T):
        s[t] = 0.25 * s[t - 1] + e[t]
    r = mu + 0.011 * rng.standard_normal((T, n_sym))
    r[1:] += fast_beta * s[:-1]                      # 因果:s(t-1) 驱动 r(t)
    close = 100 * np.exp(np.cumsum(r, axis=0))
    data = {}
    for i in range(n_sym):
        c = pd.Series(close[:, i], index=idx)
        data[f"S{i:02d}"] = pd.DataFrame(
            {"open": c.shift(1).fillna(c), "high": c * 1.01, "low": c * 0.99,
             "close": c, "volume": 1e6 + rng.integers(0, 1e5, T)}, index=idx)
    return data, pd.DataFrame(s, index=idx, columns=[f"S{i:02d}" for i in range(n_sym)])


_CACHE: dict = {}


def _shared():
    """模块级缓存:大头(合成宇宙 + 四类质量因子面板)只建一次。"""
    if not _CACHE:
        data, spanel = _xuniverse()
        close = mf.build_panel(data, "close")
        lc = np.log(close)
        z = lambda df: df.apply(mf._cross_section_z, axis=1)   # noqa: E731
        rng = np.random.default_rng(99)
        momz = z(lc.diff(60))                                   # 真因子:慢衰减动量
        noise = z(pd.DataFrame(rng.standard_normal(close.shape),
                               index=close.index, columns=close.columns))
        rw = z(pd.DataFrame(rng.normal(0, 1, close.shape).cumsum(axis=0),
                            index=close.index, columns=close.columns))
        clone = 0.95 * momz + 0.35 * rw                         # 高相关克隆(慢衰减)
        _CACHE.update(data=data, close=close,
                      panels={"mom": momz, "noise": noise,
                              "mom_clone": clone, "fast": spanel})
    return _CACHE


# ---- 1) prescreen:四道闸 ----------------------------------------------------

def test_prescreen_gates_strong_noise_clone_decay():
    c = _shared()
    res = prescreen_factors(c["panels"], c["close"], horizon=21, rebalance_days=21)
    assert "mom" in res["selected"]                       # 真因子留下
    assert res["dropped"]["noise"] == "weak"              # 纯噪声剔 weak
    assert res["dropped"]["mom_clone"] == "redundant"     # 0.9+ 相关克隆剔 redundant
    assert res["dropped"]["fast"] == "fast_decay"         # 快衰减剔 fast_decay
    rep = res["report"].set_index("name")
    assert abs(rep.loc["mom", "rankic"]) > 0.05           # 强信号确实强
    assert rep.loc["mom_clone", "max_corr_selected"] > 0.85
    assert rep.loc["fast", "half_life"] < 21


def test_prescreen_report_complete_and_capacity():
    c = _shared()
    res = prescreen_factors(c["panels"], c["close"], horizon=21)
    rep = res["report"]
    assert list(rep.columns) == ["name", "rankic", "half_life",
                                 "max_corr_selected", "crowding", "decision"]
    assert set(rep["name"]) == set(c["panels"])           # 每个入池因子都有一行
    assert rep["decision"].notna().all()
    # 放开冗余闸、收紧容量:克隆活过前三道闸,但被 max_factors=1 挤出 -> capacity
    res2 = prescreen_factors(c["panels"], c["close"], horizon=21,
                             max_incremental_corr=0.99, max_factors=1)
    assert list(res2["selected"]) == ["mom"]
    assert res2["dropped"]["mom_clone"] == "capacity"


def test_prescreen_switches_and_crowding_column():
    c = _shared()
    res = prescreen_factors(c["panels"], c["close"], horizon=21, drop_fast_decay=False)
    assert "fast" in res["selected"]                      # 关闸后快衰减因子放行
    rep = res["report"].set_index("name")
    cr = rep["crowding"].dropna()
    assert ((cr >= 0) & (cr <= 1)).all()                  # 拥挤分在 [0,1](或 NaN)
    # 全噪声池:全体 weak,selected 为空但不抛异常
    noi = {"n1": c["panels"]["noise"], "n2": -c["panels"]["noise"] * 0.5}
    res3 = prescreen_factors(noi, c["close"], horizon=21)
    assert res3["selected"] == {} and set(res3["dropped"].values()) == {"weak"}


# ---- 2) search 大因子池子集策略(单元) ---------------------------------------

def test_adaptive_subsets_capped_and_ranked():
    fn = [f"f{i:02d}" for i in range(40)]
    score = {f: 40 - i for i, f in enumerate(fn)}
    subs = XAR._adaptive_subsets(fn, score)
    assert len(subs) <= XAR.SUBSET_CAP
    assert tuple(fn) in subs                              # 全选组合在
    assert ("f00", "f01", "f02") in subs                  # top-3(按分数序)
    assert ("f00",) in subs and len(set(subs)) == len(subs)
    # 小池(<=8)不该走这条路:search 里由 len(fn)>8 门控,这里只验单调不炸
    assert len(XAR._adaptive_subsets(fn[:9], score)) <= XAR.SUBSET_CAP


# ---- 3) search(factor_source=...):库因子进自动研究 --------------------------

def test_search_zoo_runs_with_prescreen_report():
    data = _shared()["data"]
    lb = XAR.search(data, factor_source="zoo", zoo_max=20, horizon=21,
                    rebalance="ME", min_names=10, top_n=None)
    assert len(lb) > 0 and "RankICIR" in lb.columns
    assert lb["factors"].nunique() <= XAR.SUBSET_CAP      # 子集数封顶
    rep = lb.attrs["prescreen_report"]
    assert rep is not None and "decision" in rep.columns
    assert set(lb.attrs["factor_panels"]) == set(
        rep.loc[rep["decision"] == "selected", "name"])   # 只有幸存者参战


def test_search_default_behavior_unchanged():
    """默认参数(factor_source=None)的 leaderboard 必须与旧算法逐帧相等。"""
    data = _shared()["data"]
    small = {k: data[k] for k in list(data)[:12]}
    fp = {k: v for k, v in PN.price_factor_panels(small).items()
          if k in ("mom20", "mom60", "vol20")}
    lb = XAR.search(small, factor_panels=fp, horizon=21, rebalance="ME",
                    min_names=8, top_n=6)
    from itertools import combinations
    fn = list(fp)
    subsets = ([tuple(fn)] + [tuple(x) for x in combinations(fn, len(fn) - 1)]
               + [(f,) for f in fn])                      # 旧版子集逻辑原样重放
    rows = []
    for sub in subsets:
        res = evaluate_cross_section(small, model=Mdl.RidgeModel(alpha=1.0),
                                     panels={k: fp[k] for k in sub}, horizon=21,
                                     rebalance="ME", min_names=8)
        sc = res["scorecard"]
        rows.append({"factors": "+".join(sub), "model": "ridge",
                     **{k: sc[k] for k in ("RankIC", "RankICIR", "ICIR",
                                           "LS_sharpe", "n_dates")}})
    exp = (pd.DataFrame(rows).sort_values("RankICIR", ascending=False,
                                          na_position="last")
           .reset_index(drop=True).head(6))
    pd.testing.assert_frame_equal(lb, exp)
    assert "prescreen_report" not in lb.attrs             # 默认路径不带附件


# ---- 4) 端到端:库因子 -> 训练 -> 预测 -> 选股名单 ----------------------------

def test_end_to_end_zoo_factors_to_stock_picks():
    """闭环证明:factor_zoo 表达式因子经 prescreen 进 search,模型 walk-forward
    训练+预测,rank-average 集成后产出最新截面的多头名单。"""
    data = _shared()["data"]
    lb = XAR.search(data, factor_source="all", zoo_max=10, horizon=21,
                    rebalance="ME", min_names=10, top_n=4,
                    prescreen_kwargs={"max_factors": 5})
    assert len(lb) > 0
    panels = lb.attrs["factor_panels"]
    ens = XAR.ensemble_top(data, lb, k=2, factor_panels=panels,
                           horizon=21, rebalance="ME", min_names=10)
    assert ens["members"]                                  # 集成确实吸收了配置
    preds = ens["preds"]
    assert not preds.empty
    last = preds[preds["date"] == preds["date"].max()]
    picks = last.nlargest(5, "pred")["symbol"].tolist()    # 最新截面多头名单
    assert len(picks) == 5 and all(p in data for p in picks)
    assert np.isfinite(ens["scorecard"]["RankIC"])


# ---- 5) ml_factor_backtest 吃外部因子面板 ------------------------------------

def test_ml_backtest_external_panels_extend_and_replace():
    data = _shared()["data"]
    zoo = XAR.build_factor_source_panels(data, "zoo", zoo_max=16)
    sub = {k: zoo[k] for k in list(zoo)[:3]}
    r_ext = Mdl.ml_factor_backtest(data, panels=sub)                     # extend
    assert {"momentum", "low_vol"} <= set(r_ext.feature_names)
    assert set(sub) <= set(r_ext.feature_names)            # 库因子进了特征矩阵
    assert np.isfinite(r_ext.ic)
    r_rep = Mdl.ml_factor_backtest(data, panels=sub, panels_mode="replace")
    assert set(r_rep.feature_names) == set(sub)            # replace 只用库因子
    with pytest.raises(ValueError):
        Mdl.ml_factor_backtest(data, panels=sub, panels_mode="bogus")


# ---- 6) 研究记忆 --------------------------------------------------------------

def _trials():
    return [
        {"market": "US", "universe_hash": "h1", "factors": ["mom", "vol"],
         "model": "ridge", "horizon": 21, "oos_rankicir": 0.8,
         "regime": "bull", "verdict": "ok"},
        {"market": "CN", "universe_hash": "h2", "factors": "mom+rev",
         "model": "ridge", "horizon": 21, "oos_rankicir": -0.2,
         "regime": "bear", "verdict": "fail"},
        {"market": "US", "universe_hash": "h1", "factors": ["vol"],
         "model": "stacking", "horizon": 21, "oos_rankicir": 0.5,
         "regime": "bull", "verdict": "ok"},
    ]


def test_memory_log_query_suggest_roundtrip(tmp_path):
    mem = ResearchMemory(tmp_path / "trials.jsonl")
    for t in _trials():
        rec = mem.log(t)
        assert "ts" in rec                                 # 自动补时间戳
    us = mem.query(market="US")
    assert [r["oos_rankicir"] for r in us] == [0.8, 0.5]   # 按 OOS 降序
    assert len(mem.query(regime="bear")) == 1
    assert mem.query(market="US", top=1)[0]["model"] == "ridge"
    # vol 成功率 2/2 > mom 1/2 > rev 0/1;"a+b" 字符串因子也被解析
    assert mem.suggest_factors(top=3) == ["vol", "mom", "rev"]
    assert mem.suggest_factors(regime="bull", top=1) == ["mom"]  # bull 内按均值破平局


def test_memory_graceful_fallbacks(tmp_path):
    missing = ResearchMemory(tmp_path / "nope" / "void.jsonl")
    assert missing.query() == [] and missing.suggest_factors() == []
    disabled = ResearchMemory(None)                        # 禁用:log 不落盘不报错
    rec = disabled.log({"market": "US", "oos_rankicir": 1.0})
    assert rec["market"] == "US" and disabled.query() == []
    p = tmp_path / "dirty.jsonl"                           # 坏行跳过,好行照读
    p.write_text('{"market": "US", "oos_rankicir": 0.3}\nNOT-JSON{{{\n',
                 encoding="utf-8")
    assert len(ResearchMemory(p).query()) == 1


def test_cluster_strategies_families():
    rng = np.random.default_rng(5)
    idx = pd.date_range("2023-01-02", periods=250, freq="B")
    g = rng.standard_normal((250, 3))
    df = pd.DataFrame({
        "a1": g[:, 0] + 0.1 * rng.standard_normal(250),
        "a2": g[:, 0] + 0.1 * rng.standard_normal(250),
        "b1": g[:, 1] + 0.1 * rng.standard_normal(250),
        "b2": g[:, 1] + 0.1 * rng.standard_normal(250),
        "c1": g[:, 2] + 0.1 * rng.standard_normal(250)}, index=idx)
    out = cluster_strategies(df, n_clusters=3)
    labels, medoids = out["labels"], out["medoids"]
    assert len(labels) == 5 and labels.nunique() == 3 and len(medoids) == 3
    assert labels["a1"] == labels["a2"] and labels["b1"] == labels["b2"]
    assert labels["c1"] not in (labels["a1"], labels["b1"])
    assert set(medoids) <= set(df.columns)                 # 类中心是真实策略
    assert cluster_strategies(df.iloc[:, :0])["medoids"] == []   # 空输入优雅返回


# ---- 7) 跨市场热启动 ----------------------------------------------------------

def test_warm_start_search_transfers_top_configs():
    data, _ = _xuniverse(n_sym=14, T=280, seed=23)         # "目标市场":另一组数据
    src_lb = pd.DataFrame([                                 # "源市场"排行榜 top
        {"factors": "mom60+mom120", "model": "ridge", "RankICIR": 1.2},
        {"factors": "mom20", "model": "ridge", "RankICIR": 0.9},
        {"factors": "ghost_factor", "model": "ridge", "RankICIR": 0.5},
    ])
    lb = warm_start_search(data, src_lb, top=3, horizon=21, rebalance="ME",
                           min_names=10)
    assert len(lb) == 2                                    # ghost 无面板 -> 静默跳过
    assert (lb["origin"] == "transfer").all()
    assert {"mom60+mom120", "mom20"} == set(lb["factors"])
    assert lb["RankICIR"].notna().all()
    empty = warm_start_search(data, src_lb.head(0), horizon=21, rebalance="ME",
                              min_names=10)
    assert list(empty.columns) and len(empty) == 0         # 空源表 -> 空同构表
