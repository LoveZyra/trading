"""Regression + invariant tests for alpha-forge.

Two jobs:
  1. lock in the correctness fixes from the June-2026 audit (each test names the bug it
     guards), so they can never silently regress;
  2. assert the core honesty invariants the whole skill rests on — signal lag, costs,
     out-of-sample validation direction, factor causality, the canonical data contract.

Pure offline: pandas/numpy/matplotlib + monkeypatched yfinance/akshare. `pytest -q`.
"""
import sys
import types

import numpy as np
import pandas as pd
import pytest


# ======================================================================== #
#  Data contract (scripts/data/base.py)
# ======================================================================== #
def test_validate_ohlcv_canonicalizes():
    from scripts.data.base import validate_ohlcv
    raw = pd.DataFrame({
        "Date": ["2020-01-03", "2020-01-02", "2020-01-02"],   # unsorted + duplicate
        "Open": [1, 2, 3], "High": [2, 3, 4], "Low": [0.5, 1, 2],
        "Close": [1.5, 2.5, 3.5], "Vol": [10, 20, 30],
    })
    df = validate_ohlcv(raw)
    assert list(df.columns)[:5] == ["open", "high", "low", "close", "volume"]
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.is_monotonic_increasing
    assert not df.index.has_duplicates
    assert df.index.tz is None


def test_validate_ohlcv_missing_columns_raises():
    from scripts.data.base import validate_ohlcv
    with pytest.raises(ValueError):
        validate_ohlcv(pd.DataFrame({"close": [1, 2, 3]}))


def test_from_columnar_ibkr_handoff():
    from scripts.data.base import from_columnar
    payload = {"time": ["2020-01-01", "2020-01-02"], "open": [1, 2], "high": [2, 3],
               "low": [0.5, 1], "close": [1.5, 2.5], "volume": [10, 20]}
    df = from_columnar(payload)
    assert len(df) == 2 and df["close"].iloc[-1] == 2.5
    assert list(df.columns)[:5] == ["open", "high", "low", "close", "volume"]


# ======================================================================== #
#  Backtest engine — the lag + cost honesty (scripts/backtest.py)
# ======================================================================== #
def test_backtest_lags_signal_by_one_bar(ohlcv):
    """The #1 honesty guarantee: today's signal becomes tomorrow's position."""
    from scripts import backtest as bt
    sig = pd.Series(np.random.RandomState(0).uniform(-1, 1, len(ohlcv)), index=ohlcv.index)
    r = bt.backtest(ohlcv, sig, lag=1, commission_bps=0, slippage_bps=0)
    expected = sig.shift(1).fillna(0.0)
    assert np.allclose(r.position.values, expected.values)
    assert r.position.iloc[0] == 0.0


def test_backtest_costs_only_reduce_return(ohlcv):
    from scripts import backtest as bt
    from scripts.strategies import MACrossover
    sig = MACrossover(fast=10, slow=30).generate_signal(ohlcv)
    free = bt.backtest(ohlcv, sig, commission_bps=0, slippage_bps=0)
    costly = bt.backtest(ohlcv, sig, commission_bps=10, slippage_bps=10)
    assert costly.stats["total_return"] <= free.stats["total_return"]
    assert costly.stats["total_costs"] > 0


def test_buy_and_hold_equals_price_return(ohlcv):
    from scripts import backtest as bt
    bh = bt.buy_and_hold(ohlcv)
    expected = ohlcv["close"].iloc[-1] / ohlcv["close"].iloc[0] - 1
    assert abs(bh.stats["total_return"] - expected) < 1e-9


def test_portfolio_holds_between_rebalances(panel):
    """Guards the Tier-2 fix: a rebalance-only weight panel must HOLD, not go flat."""
    from scripts import backtest as bt
    from scripts.strategies import multi_factor as mf
    close = mf.build_panel(panel, "close")
    w = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    w.iloc[0] = 1.0 / close.shape[1]            # set weights only on day 0, NaN after
    r = bt.backtest_portfolio(close, w)
    assert r.position.mean() > 0.8              # ~fully invested, not ~0


# ======================================================================== #
#  Metrics (scripts/metrics.py)
# ======================================================================== #
def test_total_return_exact():
    from scripts import metrics as M
    assert abs(M.total_return(pd.Series([0.1, 0.1])) - 0.21) < 1e-12


def test_cagr_doubling_in_one_year():
    from scripts import metrics as M
    r = pd.Series([2 ** (1 / 252) - 1] * 252)   # exactly doubles over 252 bars
    assert abs(M.cagr(r) - 1.0) < 1e-6


def test_sharpe_zero_when_flat():
    from scripts import metrics as M
    assert M.sharpe(pd.Series([0.0] * 50)) == 0.0


def test_max_drawdown_non_positive(ohlcv):
    from scripts import metrics as M
    eq = (1 + ohlcv["close"].pct_change().fillna(0)).cumprod()
    assert M.max_drawdown(eq) <= 0


# ======================================================================== #
#  Validation — the two audited statistics bugs (scripts/validation.py)
# ======================================================================== #
def test_pbo_is_low_for_a_genuinely_robust_set():
    """REGRESSION: pbo_cscv sign was inverted — a robust set reported pbo~0.89 'overfit'."""
    from scripts import validation as V
    np.random.seed(1)
    df = pd.DataFrame({f"s{j}": np.random.normal(0.002 if j == 0 else 0.0, 0.01, 600)
                       for j in range(8)})
    out = V.pbo_cscv(df, n_splits=10)
    assert out["pbo"] < 0.5
    assert "robust" in out["interpretation"]


def test_pbo_noise_not_below_robust():
    """A pure-noise set must not look MORE robust than a real-edge set (catches re-inversion)."""
    from scripts import validation as V
    np.random.seed(3)
    robust = pd.DataFrame({f"s{j}": np.random.normal(0.002 if j == 0 else 0.0, 0.01, 600)
                           for j in range(8)})
    noise = pd.DataFrame({f"s{j}": np.random.normal(0, 0.01, 600) for j in range(8)})
    assert V.pbo_cscv(noise, 10)["pbo"] >= V.pbo_cscv(robust, 10)["pbo"]


def test_dsr_not_pinned_at_zero_and_monotone_in_trials():
    """REGRESSION: default sr_std=1.0 pinned DSR at 0 for any realistic strategy."""
    from scripts import validation as V
    np.random.seed(2)
    strong = pd.Series(np.random.normal(0.0012, 0.01, 750))
    assert 0.0 < V.deflated_sharpe_ratio(strong, 50) < 1.0
    seq = [V.deflated_sharpe_ratio(strong, n) for n in (1, 10, 100, 1000)]
    assert seq[0] >= seq[1] >= seq[2] >= seq[3]      # more search -> harder haircut


def test_dsr_strong_beats_weak():
    from scripts import validation as V
    np.random.seed(5)
    strong = pd.Series(np.random.normal(0.0012, 0.01, 750))
    weak = pd.Series(np.random.normal(0.0, 0.01, 750))
    assert V.deflated_sharpe_ratio(strong, 50) > V.deflated_sharpe_ratio(weak, 50)


# ======================================================================== #
#  Factor lab — causality guard (scripts/factor_lab.py)
# ======================================================================== #
def test_validate_factor_passes_causal(ohlcv):
    from scripts import factor_lab as FL
    assert FL.validate_factor(lambda d: d["close"].pct_change(20), ohlcv).causal


def test_validate_factor_catches_lookahead(ohlcv):
    from scripts import factor_lab as FL
    peek = lambda d: d["close"].shift(-1) / d["close"] - 1     # uses tomorrow's bar
    assert not FL.validate_factor(peek, ohlcv).causal


def test_backtest_custom_factor_refuses_lookahead(ohlcv):
    from scripts import factor_lab as FL
    with pytest.raises(ValueError):
        FL.backtest_custom_factor(lambda d: d["close"].shift(-1), ohlcv)


# ======================================================================== #
#  Multi-factor + walk-forward (scripts/strategies/multi_factor.py, optimize.py)
# ======================================================================== #
def test_multi_factor_weights_ffilled(panel):
    from scripts.strategies import multi_factor as mf
    w = mf.multi_factor_signal(panel, rebalance="ME", top=0.4)
    close = mf.build_panel(panel, "close")
    assert list(w.columns) == list(close.columns)
    assert (w.abs().sum(axis=1).iloc[150:] > 0).mean() > 0.9   # invested, ffilled


def test_walk_forward_produces_oos(ohlcv):
    from scripts import optimize as opt
    from scripts.strategies import MACrossover
    wf = opt.walk_forward(MACrossover, ohlcv,
                          grid={"fast": [10, 20], "slow": [40, 60]}, n_splits=4)
    assert "sharpe" in wf.oos_stats
    assert len(wf.folds) >= 1


# ======================================================================== #
#  Data layer fixes (fundamentals units, PIT leading zeros)
# ======================================================================== #
def test_pit_preserves_leading_zero_codes():
    """REGRESSION: read_json turned 000063 -> int 63, breaking the symbol join.

    Manages its own temp dir (avoids pytest's tmp_path, whose teardown can't rmdir on
    some mounted/sandbox filesystems) so the suite stays green everywhere.
    """
    import pathlib
    import shutil
    from scripts.data import pit
    base = pathlib.Path(__file__).resolve().parent / "_pit_tmp"
    shutil.rmtree(base, ignore_errors=True)
    base.mkdir(parents=True, exist_ok=True)
    try:
        panel = pd.DataFrame({"roe": [0.25, 0.10], "pe": [15, 20]},
                             index=pd.Index(["000063", "00700"], name="symbol"))
        pit.save_snapshot("2026-06-09", fundamentals_panel=panel, base=str(base))
        df = list(pit.load_pit_fundamentals(base=str(base)).values())[0]
        assert "000063" in df.index and "00700" in df.index
        assert df.loc["000063", "roe"] == 0.25
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_fundamentals_yfinance_debt_to_equity_is_fraction(monkeypatch):
    """REGRESSION: yfinance debtToEquity (a percent, 150.0) must be stored as 1.5x."""
    from scripts.data import fundamentals as F

    class FakeTicker:
        def __init__(self, *a, **k):
            pass

        @property
        def info(self):
            return {"debtToEquity": 150.0, "returnOnEquity": 0.25,
                    "trailingPE": 18.0, "shortName": "X"}

    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(Ticker=FakeTicker))
    out = F.from_yfinance("X")
    assert abs(out["debt_to_equity"] - 1.5) < 1e-9      # 150% -> 1.5x
    assert abs(out["roe"] - 0.25) < 1e-9                # already a fraction, untouched


def test_fundamentals_akshare_percent_to_fraction(monkeypatch):
    """REGRESSION: akshare (%) ratios must be /100 so a mixed panel z-scores consistently."""
    from scripts.data import fundamentals as F
    fake = types.SimpleNamespace(
        stock_individual_info_em=lambda symbol: pd.DataFrame(
            {"item": ["股票简称", "总市值", "市盈率(动)", "市净率"],
             "value": ["测试", 1e9, 18.0, 2.0]}),
        stock_financial_analysis_indicator=lambda symbol: pd.DataFrame([{
            "净资产收益率(%)": 25.0, "销售毛利率(%)": 40.0, "销售净利率(%)": 18.0,
            "资产负债率(%)": 60.0, "主营业务收入增长率(%)": 12.0, "净利润增长率(%)": 30.0}]),
    )
    monkeypatch.setitem(sys.modules, "akshare", fake)
    out = F.from_akshare("000063")
    assert abs(out["roe"] - 0.25) < 1e-9
    assert abs(out["gross_margin"] - 0.40) < 1e-9
    # 资产负债率 60% (debt/assets) must be CONVERTED to debt/equity = 0.6/0.4 = 1.5
    # so it shares the canonical field's semantics with yfinance (debt/equity).
    assert abs(out["debt_to_equity"] - 1.5) < 1e-9
    assert abs(out["revenue_growth"] - 0.12) < 1e-9


# ======================================================================== #
#  Sentiment (scripts/data/sentiment.py)
# ======================================================================== #
def test_sentiment_directional_bilingual():
    from scripts.data import sentiment as S
    assert S.score("Apple beats estimates, surges to record high") > 0
    assert S.score("shares plunge on fraud probe and bankruptcy fears") < 0
    assert S.score("业绩预增 涨停 利好") > 0
    assert S.score("跌停 爆雷 立案调查") < 0


def test_sentiment_positional_negation():
    """REGRESSION: negation was whole-string; a far-away 不/未 wrongly flipped a term."""
    from scripts.data import sentiment as S
    assert S.score("公司未发布公告 但是 大涨") > 0     # 未 is far from 大涨 -> not flipped
    assert S.score("股价不大涨") <= 0                  # 不 adjacent to 大涨 -> flipped


# ======================================================================== #
#  Report + options + macro robustness
# ======================================================================== #
def test_markdown_report_missing_metrics_safe():
    """REGRESSION: hard s['key'] indexing crashed / printed nan% on absent metrics."""
    from scripts import report as rpt

    class R:
        stats = {"total_return": 0.1, "sharpe": 1.2}    # most keys absent

    md = rpt.markdown_report(R(), name="partial")
    assert "nan%" not in md and "—" in md


def test_options_zero_price_guarded():
    from scripts.data import options as O
    assert O.expected_move(0, 0.5, 30)["move_pct"] == 0.0
    assert O.expected_move_from_straddle(5, 0)["move_pct"] == 0.0


def test_macro_empty_series_is_neutral():
    from scripts.data import macro as MAC
    assert MAC.vix_signal(pd.Series([], dtype=float)) == 0.0
    assert MAC.rates_signal(pd.Series([np.nan] * 80)) == 0.0
    assert MAC.curve_signal(pd.Series([], dtype=float), pd.Series([], dtype=float)) == 0.0


# ======================================================================== #
#  Microstructure (scripts/data/microstructure.py)
# ======================================================================== #
def test_microstructure_limits_incl_st():
    """REGRESSION: the ST ±5% limit was unreachable from limit_for()."""
    from scripts.data import microstructure as mc
    assert mc.limit_for("600519") == 0.10
    assert mc.limit_for("600519", is_st=True) == 0.05
    assert mc.limit_for("300750") == 0.20
    assert mc.limit_for("830799") == 0.30


def test_apply_cn_rules_blocks_shorts(ohlcv):
    from scripts.data import microstructure as mc
    out = mc.apply_cn_rules(pd.Series(-1.0, index=ohlcv.index), ohlcv, symbol="600519")
    assert (out >= 0).all()


# ======================================================================== #
#  HTML report (scripts/html_report.py)
# ======================================================================== #
def test_html_title_is_escaped():
    """REGRESSION: <title> was raw-concatenated — markup in a title broke <head>."""
    from scripts import html_report as H
    html = H.render({"meta": {"title": "X & </title><script>boom()</script>"}})
    assert "<script>boom()" not in html
    assert "&amp;" in html


def test_html_render_tolerates_missing_keys():
    from scripts import html_report as H
    html = H.render({"meta": {"title": "t", "report_type": "single"}})
    assert "<html" in html and "report-data" in html
