"""Lightweight ML prediction layer: factors -> expected forward return.

This is the "model" half of RD-Agent-style factor+model co-optimization. Where the
rule strategies in strategies/ are hand-coded mappings from data to position, a
FactorModel *learns* the mapping from a panel of factor exposures to next-period
cross-sectional returns, then ranks names by predicted return into a portfolio.

Two hard rules keep it honest (see references/pitfalls.md):
  1. **No look-ahead in labels.** The target for a sample at date s is the forward
     return s -> s+horizon. When we predict weights for rebalance date t, the model
     is trained ONLY on samples whose label was fully observable by t (s+horizon<=t).
     A purge gap enforces this. Get this wrong and the backtest is fiction.
  2. **Walk-forward, not one fit.** The model is re-fit on an expanding/rolling
     window before each rebalance, never fit once on the whole history.

Dependencies: the RidgeModel uses only numpy and always works. SklearnModel and
LGBMModel are optional wrappers — they import lazily and raise a clear message if the
library isn't installed (`pip install scikit-learn lightgbm`).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import backtest as bt
from .strategies import multi_factor as mf


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------
class FactorModel:
    """Maps a (n_samples, n_factors) feature matrix to predicted returns."""
    name = "model"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FactorModel":
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class RidgeModel(FactorModel):
    """Closed-form ridge regression in pure numpy -- always available.

    w = (X'X + alpha*I)^{-1} X'y on standardized features (+ intercept). Ridge (not
    plain OLS) because factor exposures are collinear and noisy; the penalty keeps
    weights stable, which matters far more than squeezing in-sample fit."""
    name = "ridge"

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self._mu = self._sd = self._w = self._b = None

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        self._mu = X.mean(0)
        self._sd = X.std(0)
        self._sd[self._sd == 0] = 1.0
        Xs = (X - self._mu) / self._sd
        n, k = Xs.shape
        A = Xs.T @ Xs + self.alpha * np.eye(k)
        self._w = np.linalg.solve(A, Xs.T @ y)
        self._b = y.mean()
        return self

    def predict(self, X):
        Xs = (np.asarray(X, float) - self._mu) / self._sd
        return Xs @ self._w + self._b


class SklearnModel(FactorModel):
    """Wrap any sklearn-style regressor (Ridge, Lasso, RandomForest, GBRT...).

    Example: SklearnModel("RandomForestRegressor", n_estimators=200, max_depth=4).
    Requires scikit-learn."""
    name = "sklearn"

    def __init__(self, estimator="Ridge", **kwargs):
        self.estimator_name = estimator if isinstance(estimator, str) else type(estimator).__name__
        self._kwargs = kwargs
        self._est = estimator

    def _build(self):
        if isinstance(self._est, str):
            import importlib
            for mod in ("sklearn.linear_model", "sklearn.ensemble", "sklearn.tree"):
                try:
                    cls = getattr(importlib.import_module(mod), self._est)
                    return cls(**self._kwargs)
                except (ImportError, AttributeError):
                    continue
            raise ImportError(
                f"could not find sklearn estimator {self._est!r}; install scikit-learn "
                f"(`pip install scikit-learn`)")
        return self._est

    def fit(self, X, y):
        self._est = self._build()
        self._est.fit(np.asarray(X, float), np.asarray(y, float))
        return self

    def predict(self, X):
        return self._est.predict(np.asarray(X, float))


class LGBMModel(FactorModel):
    """Gradient-boosted trees via LightGBM (optional). Good for non-linear factor
    interactions. Requires lightgbm (`pip install lightgbm`)."""
    name = "lightgbm"

    def __init__(self, **kwargs):
        self._kwargs = dict(n_estimators=200, max_depth=4, learning_rate=0.05,
                            subsample=0.8, verbose=-1)
        self._kwargs.update(kwargs)
        self._est = None

    def fit(self, X, y):
        try:
            import lightgbm as lgb
        except ImportError as e:
            raise ImportError("LGBMModel needs lightgbm (`pip install lightgbm`)") from e
        self._est = lgb.LGBMRegressor(**self._kwargs)
        self._est.fit(np.asarray(X, float), np.asarray(y, float))
        return self

    def predict(self, X):
        return self._est.predict(np.asarray(X, float))


MODEL_REGISTRY = {"ridge": RidgeModel, "sklearn": SklearnModel, "lightgbm": LGBMModel}


# ----------------------------------------------------------------------------
# Cross-sectional ML backtest
# ----------------------------------------------------------------------------
def build_factor_panels(data: dict, fundamentals_panel: pd.DataFrame | None = None,
                        sentiment_by_symbol: dict | None = None) -> dict:
    """Compute the standard factor exposures as {factor_name: wide DataFrame}.

    Price factors vary by date; fundamental/news factors are broadcast (static
    snapshot -- same point-in-time caveat as multi_factor; see references)."""
    close = mf.build_panel(data, "close")
    panels = {
        "momentum": mf.momentum_factor(close).apply(mf._cross_section_z, axis=1),
        "low_vol": mf.low_vol_factor(close).apply(mf._cross_section_z, axis=1),
    }
    if fundamentals_panel is not None and len(fundamentals_panel):
        fp = fundamentals_panel.reindex(close.columns)
        for fac, ser in (("value", mf.value_factor(fp)), ("quality", mf.quality_factor(fp)),
                         ("growth", mf.growth_factor(fp))):
            if ser is not None and len(ser):
                panels[fac] = pd.DataFrame([ser.reindex(close.columns).fillna(0.0).values] * len(close),
                                           index=close.index, columns=close.columns)
    if sentiment_by_symbol:
        s = mf.sentiment_factor(sentiment_by_symbol).reindex(close.columns).fillna(0.0)
        panels["sentiment"] = pd.DataFrame([s.values] * len(close), index=close.index, columns=close.columns)
    return panels


@dataclass
class MLResult:
    weights: pd.DataFrame
    backtest: object
    ic: float                       # mean cross-sectional information coefficient
    feature_names: list = field(default_factory=list)

    @property
    def stats(self):
        return self.backtest.stats


def ml_factor_backtest(data: dict, model: FactorModel | None = None,
                       fundamentals_panel: pd.DataFrame | None = None,
                       sentiment_by_symbol: dict | None = None,
                       rebalance: str = "ME", horizon: int = 21,
                       train_window: int = 252, min_train: int = 120,
                       top: float = 0.3, long_short: bool = False,
                       commission_bps: float = 1.0, slippage_bps: float = 1.0,
                       weight_smoothing: float = 0.0) -> MLResult:
    """Train a model to predict forward returns from factors, walk-forward.

    At each rebalance date t: assemble training rows (date s, symbol) with features =
    factor exposures at s and label = return s->s+horizon, for s in the trailing
    `train_window` AND s+horizon <= t (purge -> no look-ahead). Fit `model`, predict
    the cross-section at t, rank into weights, hold to next rebalance.

    Reports the strategy backtest plus the realized Information Coefficient (rank-ish
    correlation of prediction vs realized forward return) -- the cleanest measure of
    whether the model has any edge.
    """
    model = model or RidgeModel(alpha=1.0)
    close = mf.build_panel(data, "close")
    panels = build_factor_panels(data, fundamentals_panel, sentiment_by_symbol)
    fnames = list(panels)

    fwd = close.shift(-horizon) / close - 1.0       # forward return label
    # Last actual trading day per period (calendar labels skipped ~28% of months).
    from .rebalance import rebalance_dates as _rebal
    rebal_dates = _rebal(close.index, rebalance)

    weights = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    ic_list = []
    _last_w = None
    _loc = {d: i for i, d in enumerate(close.index)}   # O(1) lookups in the loop

    def feat_matrix(dates, syms):
        cols = [panels[f].reindex(index=dates, columns=syms) for f in fnames]
        return cols

    for t in rebal_dates:
        hist = close.index[close.index <= t]
        if len(hist) < min_train:
            continue
        train_dates = hist[max(0, len(hist) - train_window):]
        # purge: label must be realized by t
        t_loc = _loc[t]
        usable = [s for s in train_dates if _loc[s] + horizon < t_loc]
        if len(usable) < max(20, min_train // 3):
            continue

        # stack training samples
        Xtr, ytr = [], []
        fcols = {f: panels[f] for f in fnames}
        for s in usable:
            row = np.column_stack([fcols[f].loc[s].values for f in fnames])  # (n_sym, n_fac)
            yy = fwd.loc[s].values
            mask = np.isfinite(row).all(1) & np.isfinite(yy)
            if mask.any():
                Xtr.append(row[mask]); ytr.append(yy[mask])
        if not Xtr:
            continue
        Xtr = np.vstack(Xtr); ytr = np.concatenate(ytr)
        if len(ytr) < 20:
            continue
        # Fit a fresh clone each rebalance so stateful custom models can't leak
        # information (or fitted state) across walk-forward folds.
        import copy as _copy
        model_t = _copy.deepcopy(model)
        model_t.fit(Xtr, ytr)

        # predict current cross-section
        cur = np.column_stack([fcols[f].loc[t].values for f in fnames])
        valid = np.isfinite(cur).all(1)
        if valid.sum() < 2:
            continue
        pred = np.full(close.shape[1], np.nan)
        pred[valid] = model_t.predict(cur[valid])
        pred_s = pd.Series(pred, index=close.columns)

        # realized IC for this date (corr of prediction vs realized forward return)
        real = fwd.loc[t]
        both = pd.concat([pred_s, real], axis=1).dropna()
        if len(both) >= 3:
            ic_list.append(both.corr().iloc[0, 1])

        w_row = mf.rank_and_weight(pred_s.to_frame().T, top=top,
                                   bottom=(top if long_short else 0.0),
                                   long_short=long_short)
        new_w = w_row.iloc[0]
        # Turnover penalty (Finance-Grounded Optimization, 2509.04541): shrink the new
        # target toward the previous rebalance's weights so the model only trades when
        # its conviction changed enough to be worth the cost. smoothing in [0,1].
        if weight_smoothing > 0 and _last_w is not None:
            new_w = weight_smoothing * _last_w + (1 - weight_smoothing) * new_w
            g = new_w.abs().sum()
            if g > 0:
                new_w = new_w / g
        weights.loc[t] = new_w
        _last_w = new_w

    weights = weights.reindex(close.index).ffill().fillna(0.0)
    res = bt.backtest_portfolio(close, weights, commission_bps=commission_bps,
                                slippage_bps=slippage_bps)
    ic = float(np.nanmean(ic_list)) if ic_list else float("nan")
    return MLResult(weights=weights, backtest=res, ic=ic, feature_names=fnames)


class MLPModel(FactorModel):
    """Small neural network (sklearn MLPRegressor) — a non-linear factor->return learner
    that runs in-sandbox (no GPU). A pragmatic stand-in for deep models: for heavier
    sequence/transformer models, train externally and ingest scores via load_external_scores.
    Requires scikit-learn."""
    name = "mlp"

    def __init__(self, hidden=(32, 16), alpha: float = 1e-3, max_iter: int = 400, **kw):
        self._kw = dict(hidden_layer_sizes=hidden, alpha=alpha, max_iter=max_iter,
                        random_state=0, early_stopping=True)
        self._kw.update(kw)
        self._mu = self._sd = self._est = None

    def fit(self, X, y):
        from sklearn.neural_network import MLPRegressor
        X = np.asarray(X, float)
        self._mu, self._sd = X.mean(0), X.std(0); self._sd[self._sd == 0] = 1
        self._est = MLPRegressor(**self._kw).fit((X - self._mu) / self._sd, np.asarray(y, float))
        return self

    def predict(self, X):
        return self._est.predict((np.asarray(X, float) - self._mu) / self._sd)


def load_external_scores(json_path):
    """Ingest a {symbol: score} dict from an EXTERNALLY-trained model (Stockformer, an
    LSTM, a TS-foundation model, etc. — trained on your machine/GPU) and return it as a
    sentiment-style per-symbol factor you can drop into multi_factor_signal
    (sentiment_by_symbol=) or blend with other factors. Keeps heavy training out of the
    sandbox while still using the model's signal."""
    import json
    from pathlib import Path
    raw = json.loads(Path(json_path).read_text(encoding="utf-8"))
    return {k: float(v) for k, v in raw.items()}
