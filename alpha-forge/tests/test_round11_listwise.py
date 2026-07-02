"""Round11 §2.5:Listwise 排序损失模型(ListNet / ListMLE / LSList / LambdaMART)。
纯 numpy/pandas、离线、确定性(固定 seed);只测 scripts/xsec/xsec_models.py 的追加部分。"""
import copy

import numpy as np
import pandas as pd
import pytest

from scripts.research.models import FactorModel, RidgeModel
from scripts.xsec import xsec_eval
from scripts.xsec.xsec_models import (
    LISTWISE_REGISTRY, LambdaMARTModel, ListMLEModel, ListNetModel, LSListModel)

TRAIN_G, N_GROUPS, N_PER = 20, 30, 20      # 前 20 组训练,后 10 组评估


def _make_data(n_groups=N_GROUPS, n_per=N_PER, k=5, noise=0.5, seed=0):
    """合成排序数据:y = Xβ + 噪声,n_groups 组 × n_per 行(组 = 一天的截面)。"""
    rng = np.random.default_rng(seed)
    beta = rng.normal(size=k)
    X = rng.normal(size=(n_groups * n_per, k))
    y = X @ beta + noise * rng.standard_normal(len(X))
    groups = np.repeat(np.arange(n_groups), n_per)
    return X, y, groups


def _group_spearman(scores, y, groups):
    """组内 Spearman 的均值。"""
    out = []
    for v in np.unique(groups):
        m = groups == v
        out.append(np.corrcoef(pd.Series(scores[m]).rank(),
                               pd.Series(y[m]).rank())[0, 1])
    return float(np.mean(out))


# ---------- 三个纯 numpy listwise 模型:恢复排序 + 显著优于随机 ----------

@pytest.mark.parametrize("cls", [ListNetModel, ListMLEModel, LSListModel])
def test_listwise_recovers_ranking(cls):
    X, y, groups = _make_data()
    tr, te = groups < TRAIN_G, groups >= TRAIN_G
    m = cls().fit(X[tr], y[tr], groups=groups[tr])
    rho = _group_spearman(m.predict(X[te]), y[te], groups[te])
    assert rho > 0.5, f"{cls.__name__} 组内 Spearman={rho:.3f} 未过 0.5"
    rand = _group_spearman(np.random.default_rng(123).normal(size=int(te.sum())),
                           y[te], groups[te])
    assert rho > rand + 0.3, f"未显著优于随机打分({rho:.3f} vs {rand:.3f})"


# ---------- 与 pointwise RidgeModel 对照:不低于 ridge 的 80% ----------

def test_listwise_not_much_worse_than_ridge():
    X, y, groups = _make_data(seed=1)
    tr, te = groups < TRAIN_G, groups >= TRAIN_G
    rho_ridge = _group_spearman(RidgeModel(alpha=1.0).fit(X[tr], y[tr]).predict(X[te]),
                                y[te], groups[te])
    for cls in (ListNetModel, ListMLEModel, LSListModel):
        m = cls().fit(X[tr], y[tr], groups=groups[tr])
        rho = _group_spearman(m.predict(X[te]), y[te], groups[te])
        assert rho >= 0.8 * rho_ridge, \
            f"{cls.__name__}: {rho:.3f} < 0.8 × ridge {rho_ridge:.3f}(实现疑似坏了)"


# ---------- groups=None:单一大组近似路径可跑且仍有排序能力 ----------

def test_groups_none_single_big_group():
    X, y, groups = _make_data(seed=2)
    tr, te = groups < TRAIN_G, groups >= TRAIN_G
    for cls in (ListNetModel, ListMLEModel, LSListModel):
        m = cls().fit(X[tr], y[tr])                 # 不传 groups —— xsec_eval 的现状
        s = m.predict(X[te])
        assert np.isfinite(s).all()
        assert _group_spearman(s, y[te], groups[te]) > 0.3, cls.__name__


# ---------- shift-invariance:y 平移常数不改变排序输出(LSList 的卖点) ----------

@pytest.mark.parametrize("cls", [ListMLEModel, LSListModel])
def test_shift_invariance(cls):
    X, y, groups = _make_data(seed=3)
    s1 = cls().fit(X, y, groups=groups).predict(X)
    s2 = cls().fit(X, y + 7.3, groups=groups).predict(X)
    assert np.allclose(s1, s2, rtol=1e-5, atol=1e-8)   # 确定性 GD + 平移不变损失
    assert list(np.argsort(s1)) == list(np.argsort(s2))


# ---------- deepcopy 安全(ml_factor_backtest 每折 deepcopy 模型原型) ----------

def test_deepcopy_fit_predict_consistent():
    X, y, groups = _make_data(seed=4)
    proto = ListNetModel(epochs=200)
    m1 = copy.deepcopy(proto).fit(X, y, groups=groups)
    m2 = copy.deepcopy(proto).fit(X, y, groups=groups)
    assert np.allclose(m1.predict(X), m2.predict(X))    # 原型 deepcopy 后各自 fit 一致
    m3 = copy.deepcopy(m1)                              # 已 fit 模型 deepcopy 后预测一致
    assert np.allclose(m1.predict(X), m3.predict(X))


# ---------- LambdaMART:lazy import,缺 lightgbm 时清晰报错 ----------

def test_lambdamart_import_error_without_lightgbm():
    try:
        import lightgbm  # noqa: F401
        pytest.skip("lightgbm 已安装;本测试针对未安装环境的报错路径")
    except ImportError:
        pass
    X, y, groups = _make_data(n_groups=3)
    m = LambdaMARTModel()                               # 构造/deepcopy 不需要 lightgbm
    copy.deepcopy(m)
    with pytest.raises(ImportError, match="lightgbm"):
        m.fit(X, y, groups=groups)


# ---------- 注册表完备 ----------

def test_registry_complete():
    assert set(LISTWISE_REGISTRY) == {"listnet", "listmle", "ls_list", "lambdamart"}
    for cls in LISTWISE_REGISTRY.values():
        assert issubclass(cls, FactorModel)
        m = cls()
        assert hasattr(m, "fit") and hasattr(m, "predict")


# ---------- 端到端:evaluate_cross_section(model=ListNetModel()) 出 scorecard ----------

def _ohlcv12(T=280, seed=9):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-03", periods=T, freq="B")
    mu = rng.normal(0, 0.0015, 12)
    data = {}
    for i in range(12):
        c = 100 * np.exp(np.cumsum(mu[i] + rng.normal(0, 0.01, T)))
        data[f"S{i:02d}"] = pd.DataFrame({"open": c, "high": c * 1.01, "low": c * 0.99,
                                          "close": c, "volume": 1e6}, index=idx)
    return data


def test_end_to_end_evaluate_cross_section_listnet():
    res = xsec_eval.evaluate_cross_section(_ohlcv12(), model=ListNetModel(epochs=150),
                                           horizon=21, rebalance="ME", min_names=10)
    sc = res["scorecard"]
    assert sc["n_dates"] > 0
    assert sc["RankIC"] is not None and np.isfinite(sc["RankIC"])
    assert sc["verdict"] in ("有可用横截面信号", "弱/不稳(达不到可用门槛)",
                             "无横截面排序能力", "样本不足")
    assert not res["preds"].empty
