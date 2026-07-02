"""Round 11: sequence-model pipeline (SS2.6) + execution quality / RL bridge (SS2.21).

All tests pass WITHOUT torch (torch-only pieces use importorskip / conditional skips);
the numpy layer -- sequence panels, robust stats, checkpointing, GPU routing, IS
decomposition, execution report, order_plan zero-breakage, CLI validation -- always runs.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.research import seq_models as sm
from scripts.trade import execution as ex

ROOT = Path(__file__).resolve().parent.parent
HAS_TORCH = False
try:  # noqa: SIM105
    import torch  # noqa: F401
    HAS_TORCH = True
except ImportError:
    pass


def _panel(n=120, syms=("AAA", "BBB", "CCC"), seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-03", periods=n, freq="B")
    out = {}
    for s in syms:
        c = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, n)))
        out[s] = pd.DataFrame({"open": c * 0.999, "high": c * 1.01, "low": c * 0.99,
                               "close": c, "volume": rng.uniform(5e5, 2e6, n)}, index=idx)
    return out


# ---------------------------------------------------------------- sequence panel
def test_build_sequence_panel_shapes_and_alignment():
    data = _panel(80)
    p = sm.build_sequence_panel(data, seq_len=20)
    n_dates = 80 - 20 + 1
    assert p["X"].shape == (3 * n_dates, 20, 5)
    assert p["feature_names"] == ["open", "high", "low", "close", "volume"]
    assert len(p["index"]) == len(p["X"])
    # index tuple (end_date, symbol) really is the window's last bar
    d0, s0 = p["index"][0]
    loc = data[s0].index.get_loc(d0)
    assert loc == 19  # first full window ends at bar seq_len-1
    # relative transform: close's last element == 1 (divided by itself)
    ci = p["feature_names"].index("close")
    assert np.allclose(p["X"][:, -1, ci], 1.0)
    # date filter restricts samples
    some = data["AAA"].index[40]
    p2 = sm.build_sequence_panel(data, seq_len=20, dates=[some])
    assert len(p2["X"]) == 3 and all(d == some for d, _ in p2["index"])


def test_fit_apply_stats_train_test_separation():
    data = _panel(100)
    p = sm.build_sequence_panel(data, seq_len=15)
    X = p["X"]
    n_tr = len(X) // 2
    stats = sm.fit_stats(X[:n_tr])
    # stats reflect the TRAIN half only
    stats_full = sm.fit_stats(X)
    assert not np.allclose(stats["median"], stats_full["median"])
    Z_test = sm.apply_stats(X[n_tr:], stats)
    # apply is a pure transform: same input + same stats => identical output,
    # and it must NOT equal the self-fitted normalization of the test half
    assert np.array_equal(Z_test, sm.apply_stats(X[n_tr:], stats))
    Z_self = sm.apply_stats(X[n_tr:], sm.fit_stats(X[n_tr:]))
    assert not np.allclose(Z_test, Z_self)
    assert np.abs(Z_test).max() <= 5.0 + 1e-9  # clipped


# ---------------------------------------------------------------- checkpointing
def test_checkpoint_roundtrip(tmp_path):
    state = {"done_epochs": 3, "best_epoch": 2, "best_val": 0.123,
             "train_loss": [1.0, 0.5, 0.3], "note": "abc",
             "arr": np.arange(6, dtype=float).reshape(2, 3),
             "model_state": {"w.0": np.ones((4, 2)), "b": np.zeros(4)}}
    sm.save_checkpoint(state, tmp_path / "ckpt")
    back = sm.load_checkpoint(tmp_path / "ckpt")
    assert back["done_epochs"] == 3 and back["note"] == "abc"
    assert back["train_loss"] == [1.0, 0.5, 0.3]
    assert np.array_equal(back["arr"], state["arr"])
    assert set(back["model_state"]) == {"w.0", "b"}
    assert np.array_equal(back["model_state"]["w.0"], np.ones((4, 2)))


def test_checkpoint_resume_semantics(tmp_path):
    ck = tmp_path / "ckpt"
    assert sm.load_checkpoint(ck) is None            # fresh start -> None
    # simulate an interrupted trainer: epochs land one at a time, kill after 2
    for epoch in range(2):
        sm.save_checkpoint({"done_epochs": epoch + 1, "epochs_target": 5,
                            "model_state": {"w": np.full(3, float(epoch))}}, ck)
    st = sm.load_checkpoint(ck)
    assert st["done_epochs"] == 2                    # resume continues at epoch 2
    assert np.array_equal(st["model_state"]["w"], np.full(3, 1.0))  # LAST epoch's weights
    # a resumed run overwrites with later state
    sm.save_checkpoint({"done_epochs": 5, "epochs_target": 5,
                        "model_state": {"w": np.full(3, 4.0)}}, ck)
    assert sm.load_checkpoint(ck)["done_epochs"] == 5


# ---------------------------------------------------------------- routing + torch gate
def test_needs_local_gpu_routing():
    assert sm.needs_local_gpu("master", 1_000) is True          # heavy: always local
    assert sm.needs_local_gpu("stockformer", 10) is True
    assert sm.needs_local_gpu("gru", 50_000) is False           # light + small: in-skill
    assert sm.needs_local_gpu("lstm", 10 ** 6) is True          # light + huge: local
    assert sm.needs_local_gpu("mystery_model", 10) is True      # unknown: conservative


@pytest.mark.skipif(HAS_TORCH, reason="torch installed; ImportError path not reachable")
def test_torch_models_raise_clear_importerror():
    X = np.zeros((30, 5, 3)); y = np.zeros(30)
    for cls in (sm.GRUModel, sm.LSTMModel, sm.TransformerModel):
        with pytest.raises(ImportError, match="torch"):
            cls(epochs=1).fit(X, y)
    with pytest.raises(ImportError, match="torch"):
        sm.load_trained_seq_model("/nonexistent/model.pt")
    with pytest.raises(ImportError, match="torch"):
        ex.load_trained_executor("/nonexistent/policy.pt")
    # ml_seq_backtest surfaces the same actionable error for torch models
    with pytest.raises(ImportError, match="torch"):
        sm.ml_seq_backtest(_panel(140), sm.GRUModel(epochs=1), seq_len=15,
                           horizon=5, min_train=40, train_window=80)


def test_ml_seq_backtest_numpy_baseline_runs():
    res = sm.ml_seq_backtest(_panel(160), sm.FlattenRidgeSeqModel(), seq_len=15,
                             horizon=5, min_train=40, train_window=90)
    assert res.extra["n_fits"] >= 2
    assert np.isfinite(res.ic) or res.extra["n_ics"] == 0
    w = res.weights
    assert (w.abs().sum(axis=1) > 0).any()           # actually holds positions
    assert res.backtest is not None and hasattr(res.backtest, "stats")
    # 2.6f gate is documented where users will read it
    assert "ridge" in sm.ml_seq_backtest.__doc__ and "DSR" in sm.ml_seq_backtest.__doc__


def test_gru_train_loop_with_checkpoint_and_resume(tmp_path):
    pytest.importorskip("torch")
    data = _panel(90)
    p = sm.build_sequence_panel(data, seq_len=10)
    X = sm.apply_stats(p["X"], sm.fit_stats(p["X"]))
    rng = np.random.default_rng(0)
    y = rng.normal(0, 0.01, len(X))
    g = np.array([i % 30 for i in range(len(X))])
    ds = sm.SequenceDataset(X.astype(np.float32), y, g)
    m = sm.GRUModel(hidden=8, n_layers=1, epochs=3, batch_size=64, patience=10, val_frac=0.2)
    out = sm.train_loop(m, ds, loss="mse", epochs=3, ckpt_path=tmp_path / "ck")
    assert out["epochs_run"] == 3 and len(out["train_loss"]) == 3
    st = sm.load_checkpoint(tmp_path / "ck")
    assert st["done_epochs"] == 3
    preds = m.predict(X[:7])
    assert preds.shape == (7,) and np.isfinite(preds).all()
    # resume: asking for 5 epochs continues from 3, not from scratch
    out2 = sm.train_loop(m, ds, loss="mse", epochs=5, ckpt_path=tmp_path / "ck", resume=True)
    assert out2["epochs_run"] == 5 and len(out2["train_loss"]) == 5
    # save/load round-trip preserves predictions
    sm.save_model(m, tmp_path / "art")
    m2 = sm.load_trained_seq_model(tmp_path / "art")
    assert np.allclose(m2.predict(X[:7]), m.predict(X[:7]), atol=1e-5)


# ---------------------------------------------------------------- IS decomposition
def test_implementation_shortfall_hand_computed():
    # Buy 300 sh: decided at 100, arrived at 100.20, filled avg 101, target 400,
    # final price 102. Rising tape -> every component positive for the buyer.
    fills = pd.DataFrame({"time": [1, 2, 3], "qty": [100, 100, 100],
                          "price": [100.5, 101.0, 101.5]})
    r = ex.implementation_shortfall(fills, 100.0, side="buy", arrival_price=100.2,
                                    final_price=102.0, target_qty=400)
    f = 0.75
    assert r["fill_rate"] == pytest.approx(f)
    assert r["avg_fill_price"] == pytest.approx(101.0)
    assert r["delay_cost_bps"] == pytest.approx(f * (100.2 - 100) / 100 * 1e4)      # 15
    assert r["trading_cost_bps"] == pytest.approx(f * (101 - 100.2) / 100 * 1e4)    # 60
    assert r["opportunity_cost_bps"] == pytest.approx(0.25 * (102 - 100) / 100 * 1e4)  # 50
    assert r["is_bps"] == pytest.approx(15 + 60 + 50)
    assert min(r["delay_cost_bps"], r["trading_cost_bps"], r["opportunity_cost_bps"]) > 0
    # sell side flips the sign: for a sell, filling above the decision price is a gain
    r2 = ex.implementation_shortfall(fills, 100.0, side="sell", arrival_price=100.2,
                                     final_price=102.0, target_qty=400)
    assert r2["is_bps"] == pytest.approx(-(15 + 60 + 50))


def test_execution_quality_report_numbers():
    fills = pd.DataFrame({"time": pd.to_datetime(["2024-01-02 10:00", "2024-01-02 10:30",
                                                  "2024-01-02 11:00"]),
                          "qty": [100, 100, 100], "price": [100.5, 101.0, 101.5]})
    prices = pd.Series([100.0, 100.5, 101.0, 101.5, 101.0],
                       index=pd.date_range("2024-01-02 09:30", periods=5, freq="30min"))
    vols = pd.Series([10_000, 20_000, 20_000, 20_000, 30_000], index=prices.index)
    q = ex.execution_quality_report(fills, prices, volumes=vols, side="buy")
    twap = prices.mean()                       # 100.8
    vwap = (prices * vols).sum() / vols.sum()  # 100.95
    assert q["interval_twap"] == pytest.approx(twap)
    assert q["interval_vwap"] == pytest.approx(vwap)
    assert q["vs_twap_bps"] == pytest.approx((101.0 - twap) / twap * 1e4)
    assert q["vs_vwap_bps"] == pytest.approx((101.0 - vwap) / vwap * 1e4)
    assert q["participation_rate"] == pytest.approx(300 / 100_000)
    assert q["n_fills"] == 3 and q["fill_span"] == pytest.approx(3600.0)
    # decision defaults to first price (100) -> full IS vs 100
    assert q["is_bps"] == pytest.approx((101.0 - 100.0) / 100.0 * 1e4)
    assert q["vs_benchmark_bps"] == q["vs_vwap_bps"]
    # without volumes VWAP falls back to TWAP and says so
    q2 = ex.execution_quality_report(fills, prices)
    assert q2["vwap_is_twap_fallback"] and q2["interval_vwap"] == pytest.approx(twap)
    assert np.isnan(q2["participation_rate"])


# ---------------------------------------------------------------- order_plan bridge
def test_order_plan_default_unchanged_zero_breakage():
    kw = dict(adv_shares=20_000, max_participation=0.10, spread_bps=5.0,
              urgency="normal", style="twap", n_slices=6)
    plan = ex.order_plan(0.05, 0.0, 1_000_000, 50.0, **kw)
    # replicate the legacy pipeline field by field
    qty = ex.shares_to_trade(0.05, 0.0, 1_000_000, 50.0)
    assert plan["side"] == "BUY" and plan["total_shares"] == qty == 1000
    assert plan["notional"] == round(qty * 50.0, 2)
    assert plan["order_style"] == ex.order_style(5.0, "normal")
    assert plan["participation_cap"] == 0.10
    assert plan["spread_over_days"] == 1 and plan["per_day_shares"] == qty
    assert plan["day_shares"] == [qty]
    assert plan["child_orders"] == ex.twap_schedule(qty, 6)
    assert "TWAP" in plan["note"] and "仅为计划" in plan["note"]
    # executor=None is the explicit spelling of the same default
    assert ex.order_plan(0.05, 0.0, 1_000_000, 50.0, executor=None, **kw) == plan
    # vwap style + multi-day ADV split also unchanged
    vp = [3, 1, 1, 3]
    plan_v = ex.order_plan(0.20, 0.0, 1_000_000, 10.0, adv_shares=50_000,
                           max_participation=0.10, style="vwap", volume_profile=vp)
    assert plan_v["spread_over_days"] == 4 and plan_v["per_day_shares"] == 5000
    assert plan_v["day_shares"] == [5000, 5000, 5000, 5000]
    assert plan_v["child_orders"] == ex.vwap_schedule(5000, vp)


def test_order_plan_executor_hook():
    class Half(ex.ExecutionPolicy):
        def decide(self, state):
            return 0.5 * state["remaining"]
    plan = ex.order_plan(0.05, 0.0, 1_000_000, 50.0, executor=Half())
    assert sum(plan["child_orders"]) == plan["total_shares"] == 1000
    assert plan["child_orders"][0] == 500 and plan["child_orders"][-1] > 0
    assert "RL" in plan["note"]
    # policy can never over/under-fill even if it misbehaves
    class Greedy(ex.ExecutionPolicy):
        def decide(self, state):
            return 10 * state["total"]
    plan2 = ex.order_plan(0.05, 0.0, 1_000_000, 50.0, executor=Greedy())
    assert sum(plan2["child_orders"]) == 1000 and min(plan2["child_orders"]) >= 0


# ---------------------------------------------------------------- CLIs
def test_train_seq_model_cli_help_and_no_torch(tmp_path):
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "train_seq_model.py"),
                        "--help"], capture_output=True, text=True, cwd=str(ROOT))
    assert r.returncode == 0 and "--seq-len" in r.stdout and "--resume" in r.stdout
    # bad data path -> exit 1 with message
    r1 = subprocess.run([sys.executable, str(ROOT / "scripts" / "train_seq_model.py"),
                         "--model", "gru", "--data", str(tmp_path / "nope.pkl"),
                         "--output", str(tmp_path / "out")],
                        capture_output=True, text=True, cwd=str(ROOT))
    assert r1.returncode == 1 and "does not exist" in r1.stderr
    if HAS_TORCH:
        pytest.skip("torch installed; missing-torch exit code not reachable")
    # real data + no torch -> data loads fine, then exit 2 with install guidance
    import pickle
    pkl = tmp_path / "panel.pkl"
    with open(pkl, "wb") as fh:
        pickle.dump(_panel(100), fh)
    r2 = subprocess.run([sys.executable, str(ROOT / "scripts" / "train_seq_model.py"),
                         "--model", "gru", "--data", str(pkl), "--seq-len", "20",
                         "--horizon", "5", "--output", str(tmp_path / "out")],
                        capture_output=True, text=True, cwd=str(ROOT))
    assert r2.returncode == 2
    assert "pip install torch" in r2.stderr
    assert "samples" in r2.stdout                     # data-loading path executed


def test_train_rl_executor_cli_skeleton(tmp_path):
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "train_rl_executor.py"),
                        "--help"], capture_output=True, text=True, cwd=str(ROOT))
    assert r.returncode == 0 and "--episodes" in r.stdout
    # dry-run rolls the toy env without torch and flags the simulator caveat
    r2 = subprocess.run([sys.executable, str(ROOT / "scripts" / "train_rl_executor.py"),
                         "--output", str(tmp_path), "--dry-run"],
                        capture_output=True, text=True, cwd=str(ROOT))
    assert r2.returncode == 0
    out = json.loads(r2.stdout)
    assert out["baseline"] == "twap" and np.isfinite(out["sim_is_bps"])
    if not HAS_TORCH:
        r3 = subprocess.run([sys.executable, str(ROOT / "scripts" / "train_rl_executor.py"),
                             "--output", str(tmp_path)],
                            capture_output=True, text=True, cwd=str(ROOT))
        assert r3.returncode == 2 and "pip install torch" in r3.stderr
