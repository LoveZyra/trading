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

from ..core import backtest as bt
from ..strategies import multi_factor as mf


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


class StackingModel(FactorModel):
    """Out-of-fold stacking: several base models -> ridge meta-learner on their OOF
    predictions.

    Cross-sectional research keeps finding that a simple COMBINATION of predictors is
    more stable than any single complex one. Stacking done naively (fit members and the
    meta-learner on the same rows) lets the meta-learner reward whichever member
    memorized the training set; out-of-fold discipline fixes that — each member's
    meta-training predictions come from folds it never saw. Folds are contiguous blocks
    of the (time-ordered) training rows, respecting the serial structure.

    members: list of FactorModels. Default = ridge(0.3) + ridge(3.0) (+ LightGBM when
    installed — degrades gracefully without). Works anywhere a FactorModel does:
    ml_factor_backtest(model=StackingModel()), xsec evaluate_cross_section, etc.
    """
    name = "stacking"

    def __init__(self, members: list | None = None, meta_alpha: float = 1.0, n_folds: int = 4):
        self.members = members
        self.meta_alpha = meta_alpha
        self.n_folds = max(int(n_folds), 2)
        self._members = self._meta = None

    def _default_members(self):
        mem = [RidgeModel(0.3), RidgeModel(3.0)]
        try:
            import lightgbm  # noqa: F401
            mem.append(LGBMModel(n_estimators=120, max_depth=3))
        except ImportError:
            pass
        return mem

    def fit(self, X, y):
        import copy
        X = np.asarray(X, float); y = np.asarray(y, float)
        proto = self.members or self._default_members()
        n = len(y)
        folds = np.array_split(np.arange(n), self.n_folds)   # contiguous time blocks
        oof = np.full((n, len(proto)), np.nan)
        for j, m in enumerate(proto):
            for f in folds:
                tr = np.setdiff1d(np.arange(n), f)
                if len(tr) < 10 or len(f) == 0:
                    continue
                try:
                    mm = copy.deepcopy(m)
                    oof[f, j] = mm.fit(X[tr], y[tr]).predict(X[f])
                except ImportError:
                    oof[:, j] = np.nan
                    break
        keep = [j for j in range(len(proto)) if np.isfinite(oof[:, j]).any()]
        if not keep:
            # ValueError, NOT ImportError: autoresearch treats ImportError as "optional lib
            # missing -> skip quietly", which would silently swallow a too-short dataset.
            raise ValueError("StackingModel: not enough training rows for any base model's "
                             "OOF folds (need ~%d+); shrink n_folds or supply more data"
                             % (10 * self.n_folds))
        oof = oof[:, keep]
        proto = [proto[j] for j in keep]
        mask = np.isfinite(oof).all(axis=1)
        self._meta = RidgeModel(self.meta_alpha).fit(oof[mask], y[mask]) if mask.sum() >= 10 else None
        self._members = [copy.deepcopy(m).fit(X, y) for m in proto]   # refit on ALL rows
        return self

    def predict(self, X):
        P = np.column_stack([m.predict(X) for m in self._members])
        return self._meta.predict(P) if self._meta is not None else P.mean(axis=1)


MODEL_REGISTRY = {"ridge": RidgeModel, "sklearn": SklearnModel, "lightgbm": LGBMModel,
                  "stacking": StackingModel}


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
    extra: dict = field(default_factory=dict)   # e.g. conformal q̂ history

    @property
    def stats(self):
        return self.backtest.stats


def ml_factor_backtest(data: dict, model: FactorModel | None = None,
                       fundamentals_panel: pd.DataFrame | None = None,
                       sentiment_by_symbol: dict | None = None,
                       panels: dict | None = None, panels_mode: str = "extend",
                       rebalance: str = "ME", horizon: int = 21,
                       train_window: int = 252, min_train: int = 120,
                       top: float = 0.3, long_short: bool = False,
                       commission_bps: float = 1.0, slippage_bps: float = 1.0,
                       weight_smoothing: float = 0.0,
                       conformal_alpha: float | None = None) -> MLResult:
    """Train a model to predict forward returns from factors, walk-forward.

    At each rebalance date t: assemble training rows (date s, symbol) with features =
    factor exposures at s and label = return s->s+horizon, for s in the trailing
    `train_window` AND s+horizon <= t (purge -> no look-ahead). Fit `model`, predict
    the cross-section at t, rank into weights, hold to next rebalance.

    Reports the strategy backtest plus the realized Information Coefficient (rank-ish
    correlation of prediction vs realized forward return) -- the cleanest measure of
    whether the model has any edge.

    conformal_alpha (e.g. 0.2): calibrated-uncertainty position gating. The trailing
    window is split into fit + calibration slices; the calibration residuals give a
    distribution-free interval q̂ (scripts/conformal.py), and each name's weight is
    scaled by |prediction|/q̂ (clipped to 1). When the model's own uncertainty
    swamps its prediction, the book shrinks instead of pretending conviction —
    gross exposure becomes a function of how much the model actually knows.
    None = off. The gate costs nothing extra to fit and never ADDS exposure.

    panels / panels_mode (Round 11, §2.9 接入): 表达式/库因子由此进入 ML 训练与
    预测 —— 任何 {name: date×symbol 宽表} 外部因子面板(factor_zoo.compute_library
    的产出、prescreen_factors 的 selected、xsec panel 的价格因子、你手写的宽表)
    都可以直接作为特征喂给所有模型(Ridge/Stacking/Sklearn/LGBM/MLP,任何
    FactorModel 都自动吃到,走同一条 purged walk-forward 训练->预测->组合链路)。
      panels_mode="extend"  (默认) 与内置 momentum/low_vol(及基本面/舆情)因子
                            取并集,同名时外部面板覆盖内置;
      panels_mode="replace" 只用外部面板(纯库因子/表达式因子模型)。
    面板会 reindex 到 close 的日期×标的网格;因果性由面板作者保证(factor_zoo
    全库只用因果算子,factor_lab.validate_factor 可机械校验自定义因子)。
    """
    model = model or RidgeModel(alpha=1.0)
    if fundamentals_panel is not None or sentiment_by_symbol:
        import warnings as _w
        _w.warn("ml_factor_backtest: fundamental/sentiment factors are a single CURRENT "
                "snapshot broadcast across all history; as a TRAINED feature against a "
                "forward-return label that is point-in-time look-ahead and can inflate "
                "IC/returns. Supply dated PIT panels (data/pit.py) for an honest multi-year "
                "backtest. The price factors (momentum/low_vol) are causal.", stacklevel=2)
    close = mf.build_panel(data, "close")
    ext_panels = panels
    panels = build_factor_panels(data, fundamentals_panel, sentiment_by_symbol)
    if ext_panels:
        if panels_mode not in ("extend", "replace"):
            raise ValueError(f"panels_mode 须为 'extend'/'replace',got {panels_mode!r}")
        ext = {str(k): v.reindex(index=close.index, columns=close.columns)
               for k, v in ext_panels.items()}
        panels = {**panels, **ext} if panels_mode == "extend" else ext
    fnames = list(panels)

    fwd = close.shift(-horizon) / close - 1.0       # forward return label
    # Last actual trading day per period (calendar labels skipped ~28% of months).
    from ..core.rebalance import rebalance_dates as _rebal
    rebal_dates = _rebal(close.index, rebalance)

    # NaN (not 0.0) init: only rebalance rows get values, the final ffill then HOLDS
    # positions between rebalances. A 0.0 init made ffill a no-op -> the book was only
    # in the market 1 bar per rebalance (P0 fix, 2026-07-02).
    weights = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    ic_list = []
    qhat_list = []
    _last_w = None
    _loc = {d: i for i, d in enumerate(close.index)}   # O(1) lookups in the loop

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

        fcols = {f: panels[f] for f in fnames}

        def _stack(dates):
            Xs, ys = [], []
            for s in dates:
                row = np.column_stack([fcols[f].loc[s].values for f in fnames])  # (n_sym, n_fac)
                yy = fwd.loc[s].values
                mask = np.isfinite(row).all(1) & np.isfinite(yy)
                if mask.any():
                    Xs.append(row[mask]); ys.append(yy[mask])
            if not Xs:
                return None, None
            return np.vstack(Xs), np.concatenate(ys)

        # conformal: reserve the trailing ~20% of usable dates for calibration —
        # residuals the model never fit give the honest interval q̂.
        cal_dates: list = []
        fit_dates = usable
        if conformal_alpha:
            n_cal = max(3, len(usable) // 5)
            if len(usable) - n_cal >= max(15, min_train // 4):
                fit_dates, cal_dates = usable[:-n_cal], usable[-n_cal:]

        Xtr, ytr = _stack(fit_dates)
        if Xtr is None or len(ytr) < 20:
            continue
        # Fit a fresh clone each rebalance so stateful custom models can't leak
        # information (or fitted state) across walk-forward folds.
        import copy as _copy
        model_t = _copy.deepcopy(model)
        model_t.fit(Xtr, ytr)

        qhat = float("nan")
        if cal_dates:
            Xc, yc = _stack(cal_dates)
            if Xc is not None and len(yc) >= 8:
                from ..risk import conformal as C
                qhat = C.split_conformal_qhat(yc - model_t.predict(Xc), conformal_alpha)
                if np.isfinite(qhat):
                    qhat_list.append(qhat)

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
            ic_list.append(both.rank().corr().iloc[0, 1])   # Spearman (rank) IC: robust to outliers, matches the "rank-ish" intent

        w_row = mf.rank_and_weight(pred_s.to_frame().T, top=top,
                                   bottom=(top if long_short else 0.0),
                                   long_short=long_short)
        new_w = w_row.iloc[0]
        # Conformal gate: scale each held name by |pred|/q̂ (never renormalized back up —
        # low conviction should REDUCE gross exposure, that's the point).
        if conformal_alpha and np.isfinite(qhat):
            from ..risk import conformal as C
            conv = C.conviction_scale(pred_s, qhat)
            new_w = new_w * conv.reindex(new_w.index).fillna(1.0)
        # Turnover penalty (Finance-Grounded Optimization, 2509.04541): shrink the new
        # target toward the previous rebalance's weights so the model only trades when
        # its conviction changed enough to be worth the cost. smoothing in [0,1].
        if weight_smoothing > 0 and _last_w is not None:
            # Rescale to the NEW target's own gross (like multi_factor), never to 1.0:
            # renormalizing to 1 silently undid the conformal gate's exposure cut and
            # halved long-short's gross-2 book (P1 fix, 2026-07-02).
            g_new = new_w.abs().sum()
            mixed = weight_smoothing * _last_w + (1 - weight_smoothing) * new_w
            g_mix = mixed.abs().sum()
            if g_mix > 0 and g_new > 0:
                mixed = mixed * (g_new / g_mix)
            new_w = mixed
        weights.loc[t] = new_w
        _last_w = new_w

    weights = weights.reindex(close.index).ffill().fillna(0.0)
    res = bt.backtest_portfolio(close, weights, commission_bps=commission_bps,
                                slippage_bps=slippage_bps)
    ic = float(np.nanmean(ic_list)) if ic_list else float("nan")
    extra = {}
    if conformal_alpha:
        extra = {"conformal_alpha": conformal_alpha,
                 "qhat_mean": float(np.mean(qhat_list)) if qhat_list else float("nan"),
                 "n_calibrations": len(qhat_list)}
    return MLResult(weights=weights, backtest=res, ic=ic, feature_names=fnames, extra=extra)


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
