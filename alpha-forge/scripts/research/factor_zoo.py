"""Built-in factor library: Alpha101 / Alpha158 subsets expressed in the factor DSL.

Why strings, not functions: each factor is a `factor_expr` expression, so the whole
library is inspectable data. `compute_library` turns it into {name: wide DataFrame}
panels ready for `scripts.xsec.xsec_eval.evaluate_cross_section(panels=...)`, and any
single expression can become a single-symbol callable via `factor_expr.expr_to_callable`
for `factor_lab.validate_factor`.

Selection notes:
- ALPHA101: ~26 of WorldQuant's "101 Formulaic Alphas" (Kakushadze 2015) that need
  ONLY price/volume — the ones requiring industry classification, market cap, or
  IndNeutralize are deliberately excluded (we don't ship that data). adv20 is
  rewritten as ts_mean(volume, 20). `?:` ternaries become where(cond, a, b).
- ALPHA158: a representative subset (~60) of Qlib's Alpha158 covering EVERY family:
  K-bar shape, price ratios, ROC momentum, SMA/EMA means, Std volatility, regression
  (SLOPE/RSQR/RESI), extremes (MAX/MIN/QTLU/QTLD/RANK/RSV/IMAX/IMIN), price-volume
  correlation (CORR/CORD), count/sum ratios (CNTP/SUMP), and volume (VMA/VSTD/WVMA/
  VSUMP). Most are normalized by current close/volume, per Qlib, so they are
  scale-free and comparable across symbols.
- Every formula uses only causal operators, so the whole zoo passes
  factor_lab.validate_factor's truncation test by construction.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .factor_expr import eval_expr

# --------------------------------------------------------------------------- #
# ALPHA101 — price/volume-only subset of WorldQuant's 101 Formulaic Alphas
# --------------------------------------------------------------------------- #

ALPHA101 = {
    "alpha001": "rank(ts_arg_max(signedpower(where(returns < 0, ts_std(returns, 20), close), 2.0), 5)) - 0.5",
    "alpha002": "-1 * ts_corr(rank(delta(log(volume), 2)), rank((close - open) / open), 6)",
    "alpha003": "-1 * ts_corr(rank(open), rank(volume), 10)",
    "alpha004": "-1 * ts_rank(rank(low), 9)",
    "alpha005": "rank(open - ts_mean(vwap, 10)) * (-1 * abs(rank(close - vwap)))",
    "alpha006": "-1 * ts_corr(open, volume, 10)",
    "alpha012": "sign(delta(volume, 1)) * (-1 * delta(close, 1))",
    "alpha014": "-1 * rank(delta(returns, 3)) * ts_corr(open, volume, 10)",
    "alpha018": "-1 * rank(ts_std(abs(close - open), 5) + (close - open) + ts_corr(close, open, 10))",
    "alpha022": "-1 * delta(ts_corr(high, volume, 5), 5) * rank(ts_std(close, 20))",
    "alpha023": "where(ts_mean(high, 20) < high, -1 * delta(high, 2), 0)",
    "alpha024": ("where(delta(ts_mean(close, 100), 100) / delay(close, 100) <= 0.05, "
                 "-1 * (close - ts_min(close, 100)), -1 * delta(close, 3))"),
    "alpha026": "-1 * ts_max(ts_corr(ts_rank(volume, 5), ts_rank(high, 5), 5), 3)",
    "alpha028": "scale(ts_corr(ts_mean(volume, 20), low, 5) + (high + low) / 2 - close)",
    "alpha033": "rank(-1 * (1 - open / close))",
    "alpha034": "rank(2 - rank(ts_std(returns, 2) / ts_std(returns, 5)) - rank(delta(close, 1)))",
    "alpha035": "ts_rank(volume, 32) * (1 - ts_rank(close + high - low, 16)) * (1 - ts_rank(returns, 32))",
    "alpha038": "-1 * rank(ts_rank(close, 10)) * rank(close / open)",
    "alpha040": "-1 * rank(ts_std(high, 10)) * ts_corr(high, volume, 10)",
    "alpha041": "power(high * low, 0.5) - vwap",
    "alpha042": "rank(vwap - close) / rank(vwap + close)",
    "alpha043": "ts_rank(volume / ts_mean(volume, 20), 20) * ts_rank(-1 * delta(close, 7), 8)",
    "alpha044": "-1 * ts_corr(high, rank(volume), 5)",
    "alpha045": ("-1 * rank(ts_mean(delay(close, 5), 20)) * ts_corr(close, volume, 2) "
                 "* rank(ts_corr(ts_sum(close, 5), ts_sum(close, 20), 2))"),
    "alpha051": ("where((delay(close, 20) - delay(close, 10)) / 10 - (delay(close, 10) - close) / 10 < -0.05, "
                 "1, -1 * (close - delay(close, 1)))"),
    "alpha053": "-1 * delta(((close - low) - (high - close)) / (close - low), 9)",
    "alpha054": "-1 * (low - close) * power(open, 5) / ((low - high) * power(close, 5))",
    "alpha055": ("-1 * ts_corr(rank((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12))), "
                 "rank(volume), 6)"),
    "alpha060": ("-1 * (2 * scale(rank(((close - low) - (high - close)) / (high - low) * volume)) "
                 "- scale(rank(ts_arg_max(close, 10))))"),
    "alpha101": "(close - open) / (high - low + 0.001)",
}

# --------------------------------------------------------------------------- #
# ALPHA158 — representative Qlib Alpha158 subset, every family covered
# --------------------------------------------------------------------------- #

ALPHA158 = {}

# K-bar shape (9)
for _k in ("KMID", "KLEN", "KMID2", "KUP", "KUP2", "KLOW", "KLOW2", "KSFT", "KSFT2"):
    ALPHA158[_k] = f"{_k}()"

# current price ratios (4)
for _f in ("open", "high", "low", "vwap"):
    ALPHA158[_f.upper() + "0"] = f"{_f} / close"

_W = (5, 10, 20, 30, 60)
for _w in _W:                                   # momentum / mean / volatility
    ALPHA158[f"ROC{_w}"] = f"roc(close, {_w})"
    ALPHA158[f"MA{_w}"] = f"ts_mean(close, {_w}) / close"
    ALPHA158[f"STD{_w}"] = f"ts_std(close, {_w}) / close"
for _w in (5, 20):                              # exponential mean (EMA family)
    ALPHA158[f"EMA{_w}"] = f"ema(close, {_w}) / close"
for _w in (10, 30, 60):                         # trailing OLS on time
    ALPHA158[f"SLOPE{_w}"] = f"slope(close, {_w}) / close"
    ALPHA158[f"RSQR{_w}"] = f"rsquare(close, {_w})"
    ALPHA158[f"RESI{_w}"] = f"resi(close, {_w}) / close"
for _w in (5, 20, 60):                          # extremes / quantiles / rank / RSV
    ALPHA158[f"MAX{_w}"] = f"ts_max(high, {_w}) / close"
    ALPHA158[f"MIN{_w}"] = f"ts_min(low, {_w}) / close"
    ALPHA158[f"QTLU{_w}"] = f"QTLU({_w}) / close"
    ALPHA158[f"QTLD{_w}"] = f"QTLD({_w}) / close"
    ALPHA158[f"TSRANK{_w}"] = f"ts_rank(close, {_w})"
    ALPHA158[f"RSV{_w}"] = (f"(close - ts_min(low, {_w})) / "
                            f"(ts_max(high, {_w}) - ts_min(low, {_w}) + 1e-12)")
for _w in (10, 60):                             # days-since-extreme (normalized)
    ALPHA158[f"IMAX{_w}"] = f"ts_arg_max(high, {_w}) / {_w}"
    ALPHA158[f"IMIN{_w}"] = f"ts_arg_min(low, {_w}) / {_w}"
    ALPHA158[f"IMXD{_w}"] = f"(ts_arg_max(high, {_w}) - ts_arg_min(low, {_w})) / {_w}"
for _w in (5, 20, 60):                          # price-volume correlation
    ALPHA158[f"CORR{_w}"] = f"ts_corr(close, log(volume + 1), {_w})"
for _w in (5, 20):
    ALPHA158[f"CORD{_w}"] = f"ts_corr(close / delay(close, 1), log(volume / delay(volume, 1) + 1), {_w})"
for _w in (5, 20):                              # up/down-day counts & gain ratios
    ALPHA158[f"CNTP{_w}"] = f"ts_mean(close > delay(close, 1), {_w})"
    ALPHA158[f"CNTN{_w}"] = f"ts_mean(close < delay(close, 1), {_w})"
    ALPHA158[f"SUMP{_w}"] = (f"ts_sum(greater(close - delay(close, 1), 0), {_w}) / "
                             f"(ts_sum(abs(close - delay(close, 1)), {_w}) + 1e-12)")
ALPHA158["CNTD20"] = "ts_mean(close > delay(close, 1), 20) - ts_mean(close < delay(close, 1), 20)"
for _w in (5, 20, 60):                          # volume family
    ALPHA158[f"VMA{_w}"] = f"ts_mean(volume, {_w}) / (volume + 1e-12)"
for _w in (5, 20):
    ALPHA158[f"VSTD{_w}"] = f"ts_std(volume, {_w}) / (volume + 1e-12)"
    ALPHA158[f"WVMA{_w}"] = f"WVMA({_w})"
    ALPHA158[f"VSUMP{_w}"] = (f"ts_sum(greater(volume - delay(volume, 1), 0), {_w}) / "
                              f"(ts_sum(abs(volume - delay(volume, 1)), {_w}) + 1e-12)")

# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #

def load_factor_library(which: str = "alpha158") -> dict:
    """Return {name: expression} for 'alpha101', 'alpha158', or 'all' (both merged;
    Alpha101 names are already distinct from Alpha158's, so no prefixing needed)."""
    key = which.lower()
    if key == "alpha101":
        return dict(ALPHA101)
    if key == "alpha158":
        return dict(ALPHA158)
    if key == "all":
        return {**ALPHA158, **ALPHA101}
    raise ValueError(f"unknown library {which!r}; use 'alpha101', 'alpha158' or 'all'")


def compute_library(data, which: str = "alpha158", max_factors: int | None = None,
                    groups: dict | None = None, errors: dict | None = None) -> dict:
    """Evaluate a library on a {symbol: OHLCV} panel -> {name: wide DataFrame}.

    Robustness over purity: each factor is computed in its own try/except, so one
    degenerate formula (e.g. div-by-zero on a flat synthetic series) never kills a
    research run. Failures are warned about and, if `errors` (a dict) is passed,
    recorded there as {name: message} for post-mortems. Factors that come back
    all-NaN are dropped too — an empty panel only wastes model capacity downstream.
    """
    lib = load_factor_library(which)
    names = list(lib)[:max_factors] if max_factors else list(lib)
    out, failed = {}, {}
    for name in names:
        try:
            panel = eval_expr(lib[name], data, groups=groups)
            if not panel.notna().any().any():
                raise ValueError("all-NaN result")
            out[name] = panel
        except Exception as e:  # noqa: BLE001 — deliberately broad: skip & record
            failed[name] = f"{type(e).__name__}: {e}"
    if failed:
        warnings.warn(f"compute_library({which}): skipped {len(failed)}/{len(names)} "
                      f"factors: {sorted(failed)}", stacklevel=2)
        if errors is not None:
            errors.update(failed)
    return out


def alpha360_panel(data, seq_len: int = 60) -> dict:
    """Qlib-Alpha360-style raw input panel for (future) deep sequence models.

    For each field and lag k in [0, seq_len): price fields are divided by the
    CURRENT close and volume by the CURRENT volume — the normalization Qlib uses so
    each observation is a scale-free window ending "now" (causal: lags only look
    back). Returns {f"{FIELD}{k}": wide DataFrame} — seq_len * 6 panels; with
    seq_len=60 that is the classic 360 features.
    """
    fields = ("close", "open", "high", "low", "vwap", "volume")
    base = {f: eval_expr(f, data) for f in fields}
    out = {}
    for f in fields:
        denom = base["volume"] if f == "volume" else base["close"]
        denom = denom.replace(0.0, np.nan) if isinstance(denom, (pd.Series, pd.DataFrame)) else denom
        for k in range(seq_len):
            out[f"{f.upper()}{k}"] = (base[f].shift(k) / denom).replace([np.inf, -np.inf], np.nan)
    return out
