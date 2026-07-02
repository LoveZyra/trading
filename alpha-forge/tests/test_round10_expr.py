"""Round-10 tests: factor expression engine + Alpha101/Alpha158 factor zoo.

Covers: operator correctness against hand computations, mechanical causality
(truncating the future must not change the past), whitelist enforcement, both
input shapes (single OHLCV frame / {symbol: OHLCV} panel), full-library compute
robustness, and the two integration seams (factor_lab.validate_factor and
xsec_eval.evaluate_cross_section).
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from scripts.research.factor_expr import (available_operators, eval_expr,
                                          expr_to_callable)
from scripts.research import factor_zoo as zoo


@pytest.fixture
def rich_panel():
    """5-symbol panel with NON-constant volume (the conftest panel's flat volume
    would legitimately NaN out every volume-correlation factor)."""
    np.random.seed(42)
    idx = pd.date_range("2021-01-01", periods=300, freq="B")
    out = {}
    for s in ["AAA", "BBB", "CCC", "DDD", "EEE"]:
        c = 100 * np.exp(np.cumsum(np.random.normal(0.0002, 0.013, len(idx))))
        v = np.exp(np.random.normal(13.0, 0.4, len(idx)))
        out[s] = pd.DataFrame({"open": c * (1 + np.random.normal(0, 0.003, len(idx))),
                               "high": c * 1.012, "low": c * 0.988,
                               "close": c, "volume": v}, index=idx)
    return out


# ---------------------------------------------------------------- correctness

def test_delta_delay_ts_mean_hand_check(ohlcv):
    c = ohlcv["close"]
    assert np.allclose(eval_expr("delta(close, 3)", ohlcv).dropna(), c.diff(3).dropna())
    assert np.allclose(eval_expr("delay(close, 5)", ohlcv).dropna(), c.shift(5).dropna())
    got = eval_expr("ts_mean(close, 10)", ohlcv)
    assert abs(got.iloc[50] - c.iloc[41:51].mean()) < 1e-10
    # alias spelling + case-insensitive lookup resolve to the same operator
    assert np.allclose(eval_expr("SMA(close, 10)", ohlcv).dropna(), got.dropna())


def test_wma_ema_ts_rank_hand_check(ohlcv):
    c = ohlcv["close"]
    w = np.arange(1, 6, dtype=float); w /= w.sum()
    assert abs(eval_expr("wma(close, 5)", ohlcv).iloc[30]
               - float(np.dot(c.iloc[26:31].values, w))) < 1e-10
    ema = c.ewm(span=12, adjust=False, min_periods=12).mean()
    assert np.allclose(eval_expr("ema(close, 12)", ohlcv).dropna(), ema.dropna())
    window = c.iloc[21:31].values
    assert abs(eval_expr("ts_rank(close, 10)", ohlcv).iloc[30]
               - (window <= window[-1]).mean()) < 1e-12


def test_cross_section_rank_zscore_hand_check(rich_panel):
    r = eval_expr("rank(close)", rich_panel)
    z = eval_expr("zscore(close)", rich_panel)
    row = pd.DataFrame({s: df["close"] for s, df in rich_panel.items()}).iloc[100]
    assert np.allclose(r.iloc[100].values, row.rank(pct=True).values)
    assert np.allclose(z.iloc[100].values,
                       (row - row.mean()) / (row.std() + 1e-12), atol=1e-9)
    assert abs(eval_expr("scale(delta(close, 5))", rich_panel).iloc[-1].abs().sum() - 1.0) < 1e-9


def test_kbar_where_and_arithmetic(ohlcv):
    o, h, l, c = (ohlcv[k] for k in ("open", "high", "low", "close"))
    assert np.allclose(eval_expr("KMID()", ohlcv), (c - o) / o)
    assert np.allclose(eval_expr("KLEN()", ohlcv), (h - l) / o)
    cond = eval_expr("where(close > delay(close, 1), 1, -1)", ohlcv)
    up = (c > c.shift(1))
    assert ((cond.dropna() == 1) == up[cond.notna()]).all()
    assert cond.iloc[0] != cond.iloc[0]  # NaN condition -> NaN, not an invented 0/±1


# ----------------------------------------------------------------- causality

@pytest.mark.parametrize("expr", [
    "ts_rank(close, 15)",
    "resi(close, 20) / close",
    "ts_corr(close, log(volume + 1), 10)",
    "ema(close, 10) / close",
    "ts_decay_linear(returns, 8)",
])
def test_causality_truncating_future_keeps_past(ohlcv, expr):
    """Core no-look-ahead property: values at t must be identical whether or not
    bars after t exist. This is exactly what factor_lab.validate_factor checks."""
    full = eval_expr(expr, ohlcv)
    for K in (200, 300, 380):
        trunc = eval_expr(expr, ohlcv.iloc[:K])
        head = full.iloc[:K]
        mask = head.notna()
        assert not (mask & trunc.isna()).any(), f"{expr}: value vanished at cut {K}"
        assert np.allclose(head[mask], trunc[mask]), f"{expr}: history changed at cut {K}"


def test_validate_factor_integration(ohlcv):
    from scripts.research.factor_lab import validate_factor
    f = expr_to_callable("ts_zscore(close, 20) + roc(close, 10)")
    chk = validate_factor(f, ohlcv)
    assert chk.causal and chk.ok
    assert isinstance(f(ohlcv), pd.Series)


# ----------------------------------------------------------- safety/whitelist

def test_whitelist_rejects_non_dsl_code(ohlcv):
    for bad in ["__import__('os').system('true')",
                "close.iloc[0]",
                "evil_func(close)",
                "[x for x in close]",
                "lambda: 1"]:
        with pytest.raises(ValueError):
            eval_expr(bad, ohlcv)
    with pytest.raises(ValueError) as e:
        eval_expr("not_an_op(close)", ohlcv)
    assert "ts_corr" in str(e.value)          # error message lists available ops
    assert "ts_corr" in available_operators()


def test_cross_sectional_op_on_single_symbol_raises(ohlcv):
    with pytest.raises(ValueError, match="panel"):
        eval_expr("rank(close)", ohlcv)


# ------------------------------------------------------------- input shapes

def test_single_vs_panel_shapes(ohlcv, rich_panel):
    s = eval_expr("ts_mean(close, 5) / close", ohlcv)
    assert isinstance(s, pd.Series) and s.index.equals(ohlcv.index)
    w = eval_expr("ts_mean(close, 5) / close", rich_panel)
    assert isinstance(w, pd.DataFrame) and list(w.columns) == list(rich_panel)
    # per-column result must equal the single-symbol result on the same data
    one = eval_expr("ts_mean(close, 5) / close", rich_panel["AAA"])
    assert np.allclose(w["AAA"].dropna(), one.dropna())


def test_vwap_fallback_and_returns(ohlcv):
    v = eval_expr("vwap", ohlcv)   # no vwap column -> (h+l+c)/3
    assert np.allclose(v, (ohlcv["high"] + ohlcv["low"] + ohlcv["close"]) / 3)
    r = eval_expr("returns", ohlcv)
    assert np.allclose(r.dropna(), ohlcv["close"].pct_change().dropna())


def test_group_ops(rich_panel):
    groups = {"AAA": "tech", "BBB": "tech", "CCC": "fin", "DDD": "fin", "EEE": "fin"}
    gr = eval_expr("group_rank(delta(close, 5))", rich_panel, groups=groups)
    row = gr.iloc[-1]
    assert row[["AAA", "BBB"]].max() == 1.0 and row[["CCC", "DDD", "EEE"]].max() == 1.0
    gn = eval_expr("group_neutralize(delta(close, 5))", rich_panel, groups=groups)
    assert abs(gn.iloc[-1][["AAA", "BBB"]].mean()) < 1e-9
    with pytest.raises(ValueError, match="groups"):
        eval_expr("group_rank(close)", rich_panel)


# ------------------------------------------------------------------ factor zoo

def test_alpha_libraries_compute_mostly_valid(rich_panel):
    for which, lib in (("alpha101", zoo.ALPHA101), ("alpha158", zoo.ALPHA158)):
        errs = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            panels = zoo.compute_library(rich_panel, which, errors=errs)
        assert len(panels) >= 0.9 * len(lib), f"{which}: too many failures {errs}"
        for name, p in panels.items():
            assert isinstance(p, pd.DataFrame) and p.notna().any().any(), name
    assert len(zoo.load_factor_library("all")) == len({**zoo.ALPHA158, **zoo.ALPHA101})


def test_compute_library_skips_bad_factor_and_records(rich_panel, monkeypatch):
    broken = dict(zoo.ALPHA101, BAD="this_is_not_an_operator(close)")
    monkeypatch.setattr(zoo, "ALPHA101", broken)
    errs = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        panels = zoo.compute_library(rich_panel, "alpha101", errors=errs)
    assert "BAD" in errs and "BAD" not in panels and len(panels) >= len(broken) - 1


def test_alpha360_panel(rich_panel):
    p = zoo.alpha360_panel(rich_panel, seq_len=4)
    assert len(p) == 6 * 4
    assert np.allclose(p["CLOSE0"].dropna().values, 1.0)      # close/close == 1
    aaa_close = rich_panel["AAA"]["close"]
    expect = (aaa_close.shift(2) / aaa_close)
    assert np.allclose(p["CLOSE2"]["AAA"].dropna(), expect.dropna())


# --------------------------------------------------------------- e2e smoke

def test_end_to_end_xsec_smoke(rich_panel):
    """expression strings -> wide factor panels -> evaluate_cross_section."""
    from scripts.xsec.xsec_eval import evaluate_cross_section
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        panels = zoo.compute_library(rich_panel, "alpha101", max_factors=6)
        panels["ROC20"] = eval_expr(zoo.ALPHA158["ROC20"], rich_panel)
        out = evaluate_cross_section(rich_panel, panels=panels, horizon=21,
                                     rebalance="ME", min_train=100)
    sc = out["scorecard"]
    assert "RankIC" in sc and "verdict" in sc and out["preds"].shape[1] == 4
    assert sc["n_dates"] >= 1                 # walk-forward actually produced scores
