"""Regression tests for the 2026-06 optimization pass.

Each pins a specific fix so the issue can't silently return:
  * metrics.cagr short-window blow-up (poisoned calmar / fold aggregates)
  * optimize metric-direction footgun (ann_volatility must minimize)
  * factor_lab multi-point causality (catch a few-bar look-ahead)
  * portfolio silent universe shrinkage (warn on dropped newly-listed names)
  * autoresearch portfolio/co-opt OOS honesty (winner judged on a held-out tail)
"""
import numpy as np
import pandas as pd
import pytest


def _frame(close: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"open": close, "high": close * 1.005,
                         "low": close * 0.995, "close": close, "volume": 1e6})


# ---------------------------------------------------------------- metrics.cagr
def test_cagr_short_window_no_blowup():
    """A few-bar window must NOT annualize to an astronomical CAGR (it used to:
    1.1*1.05*1.2 over 3 bars -> ~8e11). Now it returns the plain window return."""
    from scripts import metrics as M
    short = pd.Series([0.1, 0.05, 0.2])
    assert M.cagr(short) == pytest.approx(M.total_return(short))
    assert abs(M.cagr(short)) < 5.0


def test_cagr_normal_window_unchanged():
    """A normal (>=1 month) window still annualizes with the classic formula."""
    from scripts import metrics as M
    r = pd.Series(np.full(252, 0.0004))           # ~ +10.6%/yr compounded
    expected = (1 + r).prod() ** (252 / len(r)) - 1
    assert M.cagr(r) == pytest.approx(expected)


# ------------------------------------------------------------- optimize metric
def test_grid_search_minimizes_ann_volatility():
    """metric='ann_volatility' must put the LEAST volatile config first, not the most."""
    from scripts import optimize as opt
    from scripts.strategies import MACrossover
    idx = pd.date_range("2020-01-01", periods=600, freq="B")
    close = pd.Series(100 * np.exp(np.cumsum(np.random.default_rng(1).normal(0.0004, 0.01, 600))), index=idx)
    df = _frame(close)
    tbl = opt.grid_search(MACrossover, df, {"fast": [10, 20], "slow": [50, 100]}, metric="ann_volatility")
    assert tbl["ann_volatility"].iloc[0] == tbl["ann_volatility"].min()


def test_grid_search_still_maximizes_sharpe():
    """Default (and other) metrics stay larger-is-better."""
    from scripts import optimize as opt
    from scripts.strategies import MACrossover
    idx = pd.date_range("2020-01-01", periods=600, freq="B")
    close = pd.Series(100 * np.exp(np.cumsum(np.random.default_rng(2).normal(0.0004, 0.01, 600))), index=idx)
    df = _frame(close)
    tbl = opt.grid_search(MACrossover, df, {"fast": [10, 20], "slow": [50, 100]}, metric="sharpe")
    assert tbl["sharpe"].iloc[0] == tbl["sharpe"].max()


# ---------------------------------------------------------- factor_lab causality
def test_validate_factor_catches_few_bar_lookahead():
    """A factor peeking only a few bars ahead must be flagged non-causal."""
    from scripts import factor_lab as FL
    idx = pd.date_range("2020-01-01", periods=400, freq="B")
    df = _frame(pd.Series(100 * np.exp(np.cumsum(np.random.default_rng(3).normal(0, 0.01, 400))), index=idx))
    assert FL.validate_factor(lambda d: d["close"].pct_change(5), df).causal is True
    assert FL.validate_factor(lambda d: d["close"].shift(-3), df).causal is False
    assert FL.validate_factor(lambda d: d["close"].shift(-1), df).causal is False


# --------------------------------------------------------- portfolio coverage
def test_portfolio_health_warns_on_dropped_names():
    """A newly-listed name (leading NaNs) is dropped from the risk math; the health
    verdict must SAY SO rather than silently reporting a subset portfolio."""
    from scripts import portfolio as PF
    idx = pd.date_range("2020-01-01", periods=200, freq="B")
    p = pd.DataFrame({s: 100 * np.exp(np.cumsum(np.random.default_rng(i).normal(0, 0.01, 200)))
                      for i, s in enumerate(["A", "B", "C"])}, index=idx)
    p.iloc[:150, p.columns.get_loc("C")] = np.nan          # C lists only in last 50 bars
    verdict = PF.portfolio_health(p, lookback=120)["verdict"]
    assert any("仅基于" in m for m in verdict), verdict


# --------------------------------------------------------- autoresearch holdout
@pytest.fixture
def big_panel():
    idx = pd.date_range("2019-01-01", periods=400, freq="B")
    out = {}
    for i, s in enumerate(["AAA", "BBB", "CCC", "DDD"]):
        c = 100 * np.exp(np.cumsum(np.random.default_rng(i + 9).normal(0.0003, 0.01, 400)))
        out[s] = pd.DataFrame({"open": c, "high": c * 1.01, "low": c * 0.99,
                               "close": c, "volume": 1e6}, index=idx)
    return out


def test_research_portfolio_reports_holdout(big_panel):
    """The winner must carry a held-out (OOS) Sharpe distinct from the selection score."""
    from scripts import autoresearch as AR
    rep = AR.research_portfolio(big_panel, iterations=6, use_ml=False, seed=1)
    assert "holdout_sharpe" in rep.best.extra
    assert "holdout_return" in rep.best.extra
    assert "holdout(OOS)" in repr(rep)


def test_cooptimize_reports_holdout(big_panel):
    from scripts import autoresearch as AR
    co = AR.cooptimize_factor_model(big_panel, rounds=2)
    assert "holdout" in co and "sharpe" in co["holdout"]
    # history rows expose the TRAIN (selection) sharpe, not a mislabeled OOS one
    assert all("train_sharpe" in row for row in co["history"])


# ====================== round 2: ML/sizing/tracker hardening ======================

def test_signal_tracker_snaps_nontrading_date_to_prior_bar(tmp_path):
    """A signal dated on a weekend must use the LAST trading bar on/before it as the
    entry (searchsorted alone would grab the NEXT bar)."""
    from scripts import signal_tracker as ST
    idx = pd.date_range("2024-01-01", periods=60, freq="B")
    close = pd.Series(np.linspace(100, 160, 60), index=idx)
    # find a Friday in the index, sign the *Saturday* after it (not a trading day)
    fri = [d for d in idx if d.weekday() == 4][3]
    sat = fri + pd.Timedelta(days=1)
    log = tmp_path / "sig.jsonl"
    ST.log_signals([{"date": sat.strftime("%Y-%m-%d"), "symbol": "X", "signal": 1}], log)
    out = ST.evaluate(log, lambda s: close, horizon=5)
    entry = idx.searchsorted(sat, side="right") - 1        # -> the Friday position
    expected = close.iloc[entry + 5] / close.iloc[entry] - 1
    assert out["n_matured"] == 1
    assert out["mean_fwd_ret"] == pytest.approx(round(float(expected), 4))


def test_ml_factor_backtest_warns_on_snapshot_factors(big_panel):
    """Using static fundamentals/sentiment as trained features is point-in-time
    look-ahead; the function must warn (and still return a sane IC)."""
    from scripts import models as Mdl
    senti = {s: 0.1 * i for i, s in enumerate(big_panel)}
    with pytest.warns(UserWarning, match="point-in-time"):
        res = Mdl.ml_factor_backtest(big_panel, sentiment_by_symbol=senti, rebalance="ME")
    assert (not np.isfinite(res.ic)) or (-1.0 <= res.ic <= 1.0)


def test_risk_parity_weights_valid(big_panel):
    """ERC weights are long-only and (on a rebalance) sum to ~1; collinear assets must
    not crash the solver."""
    from scripts import sizing as SZ
    close = pd.DataFrame({s: big_panel[s]["close"] for s in big_panel})
    w = SZ.risk_parity_weights(close, lookback=126)
    last = w.iloc[-1]
    assert (last >= -1e-9).all()
    assert last.sum() == pytest.approx(1.0, abs=1e-6)
    # near-collinear: duplicate a column; should still return valid weights, no exception
    close2 = close.copy(); close2["DUP"] = close.iloc[:, 0]
    w2 = SZ.risk_parity_weights(close2, lookback=126)
    assert w2.iloc[-1].sum() == pytest.approx(1.0, abs=1e-6)
