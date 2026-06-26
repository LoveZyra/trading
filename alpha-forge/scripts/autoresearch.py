"""Automated research loop — RD-Agent-style Research -> Develop -> Feedback.

Inspired by Microsoft's RD-Agent(Q) (arXiv:2505.15155): decompose quant R&D into a
loop that proposes a hypothesis, implements & backtests it, evaluates the result, and
uses a bandit scheduler to adaptively pick the next direction — alternating between
*factor* search and *model* search (factor-model co-optimization).

What's native here vs. the paper:
  * The "Research agent" that proposes hypotheses is YOU (Claude). The loop samples
    from a structured search space; you make it smart by editing the space, adding
    factors, or seeding promising directions between runs. That's the intended use —
    the loop does the bookkeeping and honest OOS scoring; you supply the ideas.
  * "Development" = the existing strategy/factor/model code; nothing is hand-rolled
    per run.
  * "Feedback" = out-of-sample metrics. HOW each driver stays honest:
      - research_single: walk-forward OOS per trial (select on train, score on test).
      - research_portfolio / cooptimize_factor_model: the search runs only on a TRAIN
        slice; the single winner is then re-scored on a held-out TEST tail it never saw
        during selection (`holdout_sharpe`). That tail is the number to trust.
  * The scheduler is a UCB1 multi-armed bandit over directions, exactly as the paper
    uses a bandit to balance exploration vs. exploitation across research avenues.

Why the train/test split matters: searching hard *creates* false positives. If you
score every factor-weight blend on the full history and keep the best, you've simply
curve-fit to noise. Confining the search to a train slice and judging the winner on an
untouched tail is what makes the leaderboard's headline number trustworthy. For an
extra multiple-testing haircut on the winner, pass its returns to
`validation.deflated_sharpe_ratio(n_trials=<#trials>)`.
"""
from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

import numpy as np
import pandas as pd

from . import optimize as opt, models as Mdl
from .strategies import REGISTRY, multi_factor as mf
from . import backtest as bt


# ----------------------------------------------------------------------------
# UCB1 bandit scheduler
# ----------------------------------------------------------------------------
class UCB1:
    """Upper-Confidence-Bound bandit. Picks the arm maximizing
    mean_reward + sqrt(2 ln(total) / pulls). Unpulled arms are tried first.
    Balances exploring new research directions against exploiting good ones."""

    def __init__(self, arms: list[str]):
        self.arms = list(arms)
        self.counts = {a: 0 for a in arms}
        self.values = {a: 0.0 for a in arms}   # running mean reward
        self.total = 0

    def select(self) -> str:
        for a in self.arms:
            if self.counts[a] == 0:
                return a
        log_t = math.log(self.total)
        return max(self.arms, key=lambda a: self.values[a] + math.sqrt(2 * log_t / self.counts[a]))

    def update(self, arm: str, reward: float):
        self.counts[arm] += 1
        self.total += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n


def _reward(sharpe: float) -> float:
    """Map an OOS Sharpe to a bounded reward in (0,1) for the bandit."""
    if not np.isfinite(sharpe):
        return 0.0
    return 1.0 / (1.0 + math.exp(-sharpe))


# ----------------------------------------------------------------------------
# Search spaces
# ----------------------------------------------------------------------------
# Per rule-strategy family: a small, economically-sensible grid to sample from.
# Shared with run_backtest --walk-forward via scripts/param_grids.py.
from .param_grids import PARAM_GRIDS as RULE_SPACE


def _sample(grid: dict, rng: random.Random) -> dict:
    return {k: rng.choice(v) for k, v in grid.items()}


@dataclass
class Trial:
    direction: str
    params: dict
    oos_sharpe: float
    oos_return: float
    extra: dict = field(default_factory=dict)


@dataclass
class ResearchReport:
    trials: list                         # every hypothesis tried, with OOS result
    leaderboard: pd.DataFrame            # sorted best-first
    best: Trial
    bandit_summary: dict                 # pulls + mean reward per direction

    def __repr__(self):
        b = self.best
        hold = ""
        h = b.extra.get("holdout_sharpe", float("nan"))
        if np.isfinite(h):
            hold = f" | holdout(OOS) sharpe={h:.2f} ret={b.extra.get('holdout_return', float('nan')):+.1%}"
        return (f"ResearchReport(best={b.direction} {b.params} "
                f"select sharpe={b.oos_sharpe:.2f} ret={b.oos_return:+.1%}{hold}; "
                f"{len(self.trials)} trials)")


# ----------------------------------------------------------------------------
# Driver 1: rule-strategy research on a single asset
# ----------------------------------------------------------------------------
MIN_ITERATIONS = 30  # 自动研究每只标的的最低搜索次数(skill 下限);低于此值自动抬到 30
RESEARCH_DEPTHS = {"quick": 30, "standard": 60, "deep": 150}  # 迭代档位预设;传 depth= 即用,iterations= 仍可显式覆盖


def research_single(df: pd.DataFrame, iterations: int = 30, *, depth: str = None, seed: int = 0,
                    n_splits: int = 4, commission_bps: float = 1.0,
                    slippage_bps: float = 1.0) -> ResearchReport:
    """Bandit-driven search over rule-strategy families on one asset.

    Each iteration: the bandit picks a strategy family, a random param set is sampled
    (the "hypothesis"), it's scored by walk-forward OOS Sharpe ("feedback"), and the
    bandit is updated. Returns a leaderboard + the hypothesis log.
    """
    if depth is not None:
        iterations = RESEARCH_DEPTHS.get(str(depth).lower().strip(), iterations)  # 档位优先于 iterations
    iterations = max(int(iterations), MIN_ITERATIONS)  # enforce skill minimum
    rng = random.Random(seed)
    bandit = UCB1(list(RULE_SPACE))
    trials: list[Trial] = []

    for _ in range(iterations):
        fam = bandit.select()
        params = _sample(RULE_SPACE[fam], rng)
        try:
            # single-combo grid -> walk_forward gives a clean OOS estimate
            grid = {k: [v] for k, v in params.items()}
            wf = opt.walk_forward(REGISTRY[fam], df, grid, n_splits=n_splits,
                                  commission_bps=commission_bps, slippage_bps=slippage_bps)
            sh = wf.oos_stats.get("sharpe", float("nan"))
            ret = wf.oos_stats.get("total_return", float("nan"))
        except Exception:  # noqa: BLE001
            log.debug("trial failed: %s %s", fam, params, exc_info=True)
            sh, ret = float("nan"), float("nan")
        bandit.update(fam, _reward(sh))
        trials.append(Trial(fam, params, sh, ret))

    return _finalize(trials, bandit)


# ----------------------------------------------------------------------------
# Driver 2: factor + model co-optimization on a universe
# ----------------------------------------------------------------------------
FACTOR_KEYS = ["momentum", "low_vol", "value", "quality", "growth", "sentiment"]
MODEL_DIRECTIONS = ["ridge_low", "ridge_high", "rf", "lightgbm", "equal_weight"]


def _make_model(direction: str):
    if direction == "ridge_low":
        return Mdl.RidgeModel(alpha=0.3)
    if direction == "ridge_high":
        return Mdl.RidgeModel(alpha=3.0)
    if direction == "rf":
        return Mdl.SklearnModel("RandomForestRegressor", n_estimators=120, max_depth=4, random_state=0)
    if direction == "lightgbm":
        return Mdl.LGBMModel(n_estimators=150, max_depth=3)
    return None  # equal_weight handled separately


def _score_factor_weights(data, weights_dict, fundamentals_panel, sentiment_by_symbol,
                          rebalance, top, commission_bps, slippage_bps):
    w = mf.multi_factor_signal(data, weights_dict, rebalance=rebalance, top=top,
                               fundamentals_panel=fundamentals_panel,
                               sentiment_by_symbol=sentiment_by_symbol)
    res = bt.backtest_portfolio(mf.build_panel(data, "close"), w,
                                commission_bps=commission_bps, slippage_bps=slippage_bps)
    return res.stats, w


def _split_data(data: dict, oos_frac: float):
    """Split each symbol's history at one common date: search on `train`, judge the
    winner on the held-out `test` tail. Returns (train, test); test is None when there
    isn't enough history to carve an honest holdout (caller then scores on full data
    and the holdout is simply reported as NaN)."""
    if not (0.0 < oos_frac < 0.9):
        return data, None
    panel = mf.build_panel(data, "close")
    if len(panel) < 80:
        return data, None
    split = panel.index[int(len(panel) * (1 - oos_frac))]
    train = {s: df[df.index < split] for s, df in data.items()}
    test = {s: df[df.index >= split] for s, df in data.items()}
    if (min((len(v) for v in train.values()), default=0) < 40 or
            min((len(v) for v in test.values()), default=0) < 30):
        return data, None
    return train, test


def _holdout_score(best: "Trial", test_data, avail, fundamentals_panel,
                   sentiment_by_symbol, rebalance, top, commission_bps, slippage_bps):
    """Re-score the winning config on the untouched test tail -> honest OOS Sharpe."""
    if test_data is None:
        return float("nan"), float("nan")
    try:
        if best.direction == "factor":
            stats, _ = _score_factor_weights(test_data, best.params, fundamentals_panel,
                                             sentiment_by_symbol, rebalance, top,
                                             commission_bps, slippage_bps)
        elif best.direction.startswith("model:"):
            md = best.params.get("model")
            if md == "equal_weight":
                stats, _ = _score_factor_weights(test_data, {k: 1.0 for k in avail},
                                                 fundamentals_panel, sentiment_by_symbol,
                                                 rebalance, top, commission_bps, slippage_bps)
            else:
                res = Mdl.ml_factor_backtest(test_data, model=_make_model(md),
                                             fundamentals_panel=fundamentals_panel,
                                             sentiment_by_symbol=sentiment_by_symbol,
                                             rebalance=rebalance, top=top,
                                             commission_bps=commission_bps,
                                             slippage_bps=slippage_bps)
                stats = res.stats
        else:
            return float("nan"), float("nan")
        return stats.get("sharpe", float("nan")), stats.get("total_return", float("nan"))
    except Exception:  # noqa: BLE001
        log.debug("holdout eval failed: %s", best.direction, exc_info=True)
        return float("nan"), float("nan")


def research_portfolio(data: dict, iterations: int = 24, *, seed: int = 0,
                       fundamentals_panel: pd.DataFrame | None = None,
                       sentiment_by_symbol: dict | None = None,
                       rebalance: str = "ME", top: float = 0.3,
                       commission_bps: float = 1.0, slippage_bps: float = 1.0,
                       use_ml: bool = True, oos_frac: float = 0.3) -> ResearchReport:
    """Bandit over two kinds of direction: 'factor' (sample factor-weight blends) and
    'model' (try ML predictors). Factor trials use the linear multi-factor portfolio;
    model trials use the cross-sectional ML backtest. The bandit learns which avenue
    is paying off and concentrates there — RD-Agent's adaptive scheduling, natively.

    Honesty: the whole search runs on a TRAIN slice (the first `1-oos_frac` of history);
    the single winner is then re-scored on the held-out TEST tail it never saw, reported
    as `report.best.extra['holdout_sharpe']`. The per-trial `oos_sharpe` on the
    leaderboard is the *selection* score (train) — compare configs by it, but trust the
    winner's holdout. With too little history to split, it falls back to full-sample
    scoring and the holdout is NaN.

    Only factors you supply data for are sampled (fundamentals/news optional). ML
    model directions are skipped gracefully if sklearn/lightgbm aren't installed.
    """
    rng = random.Random(seed)
    train_data, test_data = _split_data(data, oos_frac)
    # which factors are actually available
    avail = ["momentum", "low_vol"]
    if fundamentals_panel is not None and len(fundamentals_panel):
        avail += ["value", "quality", "growth"]
    if sentiment_by_symbol:
        avail += ["sentiment"]

    directions = ["factor"]
    if use_ml:
        directions.append("model")
    bandit = UCB1(directions)
    trials: list[Trial] = []

    for _ in range(iterations):
        d = bandit.select()
        if d == "factor":
            # sample a random nonnegative weight blend over available factors
            wd = {k: round(rng.random(), 2) for k in avail if rng.random() > 0.3}
            if not wd:
                wd = {avail[0]: 1.0}
            try:
                stats, _ = _score_factor_weights(train_data, wd, fundamentals_panel,
                                                  sentiment_by_symbol, rebalance, top,
                                                  commission_bps, slippage_bps)
                sh, ret = stats.get("sharpe", float("nan")), stats.get("total_return", float("nan"))
            except Exception:  # noqa: BLE001
                log.debug("factor trial failed: %s", wd, exc_info=True)
                sh, ret = float("nan"), float("nan")
            trials.append(Trial("factor", wd, sh, ret))
            bandit.update(d, _reward(sh))
        else:
            md = rng.choice(MODEL_DIRECTIONS)
            try:
                if md == "equal_weight":
                    stats, _ = _score_factor_weights(train_data, {k: 1.0 for k in avail},
                                                     fundamentals_panel, sentiment_by_symbol,
                                                     rebalance, top, commission_bps, slippage_bps)
                    sh, ret, ic = stats.get("sharpe", float("nan")), stats.get("total_return", float("nan")), float("nan")
                else:
                    res = Mdl.ml_factor_backtest(train_data, model=_make_model(md),
                                                 fundamentals_panel=fundamentals_panel,
                                                 sentiment_by_symbol=sentiment_by_symbol,
                                                 rebalance=rebalance, top=top,
                                                 commission_bps=commission_bps,
                                                 slippage_bps=slippage_bps)
                    sh, ret, ic = res.stats.get("sharpe", float("nan")), res.stats.get("total_return", float("nan")), res.ic
            except ImportError:
                sh, ret, ic = float("nan"), float("nan"), float("nan")  # lib missing -> skip
            except Exception:  # noqa: BLE001
                log.debug("model trial failed: %s", md, exc_info=True)
                sh, ret, ic = float("nan"), float("nan"), float("nan")
            trials.append(Trial(f"model:{md}", {"model": md}, sh, ret, {"ic": ic}))
            bandit.update(d, _reward(sh))

    rep = _finalize(trials, bandit)
    # Judge the winner on the held-out tail it never saw during the search.
    h_sh, h_ret = _holdout_score(rep.best, test_data, avail, fundamentals_panel,
                                 sentiment_by_symbol, rebalance, top,
                                 commission_bps, slippage_bps)
    rep.best.extra["holdout_sharpe"] = h_sh
    rep.best.extra["holdout_return"] = h_ret
    return rep


def cooptimize_factor_model(data: dict, rounds: int = 3, *,
                            fundamentals_panel: pd.DataFrame | None = None,
                            sentiment_by_symbol: dict | None = None,
                            rebalance: str = "ME", top: float = 0.3,
                            seed: int = 0, oos_frac: float = 0.3) -> dict:
    """Alternating factor<->model optimization (the heart of RD-Agent(Q)).

    Round-robin: (1) fix the model, search factor-weight blends; (2) fix the best
    factors, search models. Repeat. A compact, honest version of co-optimization you
    can run in seconds. Like research_portfolio, the search runs on a TRAIN slice and
    the final (best_weights, best_model) pair is re-scored on the held-out TEST tail —
    returned as `holdout` (sharpe + total_return), the number to trust. `history` holds
    the per-round train (selection) Sharpes.
    """
    rng = random.Random(seed)
    train_data, test_data = _split_data(data, oos_frac)
    avail = ["momentum", "low_vol"]
    if fundamentals_panel is not None and len(fundamentals_panel):
        avail += ["value", "quality", "growth"]
    if sentiment_by_symbol:
        avail += ["sentiment"]

    best_weights = {k: 1.0 for k in avail}
    best_model = "equal_weight"
    history = []

    for r in range(rounds):
        # (1) factor search given current model = linear blend (equal_weight proxy)
        best_sh = -np.inf
        for _ in range(8):
            wd = {k: round(rng.random(), 2) for k in avail if rng.random() > 0.3} or {avail[0]: 1.0}
            try:
                stats, _ = _score_factor_weights(train_data, wd, fundamentals_panel,
                                                 sentiment_by_symbol, rebalance, top, 1.0, 1.0)
                if stats.get("sharpe", -np.inf) > best_sh:
                    best_sh, best_weights = stats["sharpe"], wd
            except Exception:  # noqa: BLE001
                continue
        history.append({"round": r, "phase": "factor", "train_sharpe": round(best_sh, 3),
                        "weights": dict(best_weights)})

        # (2) model search given factors fixed
        best_msh, chosen = -np.inf, best_model
        for md in MODEL_DIRECTIONS:
            try:
                if md == "equal_weight":
                    stats, _ = _score_factor_weights(train_data, best_weights, fundamentals_panel,
                                                     sentiment_by_symbol, rebalance, top, 1.0, 1.0)
                    sh = stats.get("sharpe", -np.inf)
                else:
                    res = Mdl.ml_factor_backtest(train_data, model=_make_model(md),
                                                 fundamentals_panel=fundamentals_panel,
                                                 sentiment_by_symbol=sentiment_by_symbol,
                                                 rebalance=rebalance, top=top)
                    sh = res.stats.get("sharpe", -np.inf)
                if sh > best_msh:
                    best_msh, chosen = sh, md
            except Exception:  # noqa: BLE001
                continue
        best_model = chosen
        history.append({"round": r, "phase": "model", "train_sharpe": round(best_msh, 3),
                        "model": best_model})

    # Honest OOS read on the final co-optimized pair.
    winner = Trial(f"model:{best_model}", {"model": best_model}, float("nan"), float("nan")) \
        if best_model != "equal_weight" else Trial("factor", dict(best_weights), float("nan"), float("nan"))
    h_sh, h_ret = _holdout_score(winner, test_data, avail, fundamentals_panel,
                                 sentiment_by_symbol, rebalance, top, 1.0, 1.0)
    return {"best_weights": best_weights, "best_model": best_model, "history": history,
            "holdout": {"sharpe": h_sh, "total_return": h_ret}}


# ----------------------------------------------------------------------------
def _finalize(trials: list, bandit: UCB1) -> ResearchReport:
    rows = [{"direction": t.direction, "params": t.params, "oos_sharpe": t.oos_sharpe,
             "oos_return": t.oos_return, **t.extra} for t in trials]
    lb = pd.DataFrame(rows)
    if len(lb):
        lb = lb.sort_values("oos_sharpe", ascending=False, na_position="last").reset_index(drop=True)
    valid = [t for t in trials if np.isfinite(t.oos_sharpe)]
    best = max(valid, key=lambda t: t.oos_sharpe) if valid else trials[0]
    summary = {a: {"pulls": bandit.counts[a], "mean_reward": round(bandit.values[a], 3)}
               for a in bandit.arms}
    return ResearchReport(trials=trials, leaderboard=lb, best=best, bandit_summary=summary)


# ----------------------------------------------------------------------------
# Strategy ensembling (Ensembling Portfolio Strategies, 2406.03652)
# ----------------------------------------------------------------------------
def ensemble_top_k(report: ResearchReport, df: pd.DataFrame, k: int = 3, *,
                   commission_bps: float = 1.0, slippage_bps: float = 1.0):
    """Blend the top-k rule-strategy configs from a research_single report into one
    equal-weight ensemble signal and backtest it on `df`.

    Why ensemble instead of picking the single best? The #1 OOS config is partly luck;
    averaging several decorrelated winners keeps most of the edge with less variance
    and lower overfitting risk (a free lunch when the strategies disagree). Returns a
    BacktestResult for the ensemble plus the list of members used.

    Note: this re-backtests the blend on the `df` you pass. For a clean OOS read, pass a
    hold-out slice not used when the members were selected (the members came from a
    walk-forward research_single, but re-scoring on the same series still has mild
    selection bias).
    """
    members = []
    for t in report.leaderboard.itertuples():
        if np.isfinite(getattr(t, "oos_sharpe", float("nan"))) and t.direction in REGISTRY:
            members.append((t.direction, t.params))
        if len(members) >= k:
            break
    if not members:
        raise ValueError("no usable members in report leaderboard")

    sigs = []
    for fam, params in members:
        try:
            sigs.append(REGISTRY[fam](**params).generate_signal(df).reindex(df.index).fillna(0.0))
        except Exception:  # noqa: BLE001
            continue
    if not sigs:
        raise ValueError("could not build any ensemble signals")
    blended = sum(sigs) / len(sigs)          # equal-weight average of target positions
    res = bt.backtest(df, blended, commission_bps=commission_bps, slippage_bps=slippage_bps)
    return res, members


def strategy_glossary(report: "ResearchReport | None" = None, families: list | None = None) -> list:
    """One-line intros for the strategy families a research run actually tested — feeds the
    report's 「策略测试选择」 glossary so every tested strategy carries an explanation.

    Pass a `report` to glossary just the families on its leaderboard (deduped, in rank
    order), or an explicit `families` list, or neither to describe all known families.
    Returns [{family, name, intro, edge}] (only families with a known description).
    """
    from .param_grids import STRATEGY_INFO
    if families is None:
        families = []
        if report is not None and getattr(report, "leaderboard", None) is not None:
            for t in report.leaderboard.itertuples():
                d = getattr(t, "direction", None)
                if d in STRATEGY_INFO and d not in families:
                    families.append(d)
        families = families or list(STRATEGY_INFO)
    out = []
    for f in families:
        info = STRATEGY_INFO.get(f)
        if info:
            out.append({"family": f, "name": info["name"], "intro": info["intro"],
                        "edge": info.get("edge", "")})
    return out


# ----------------------------------------------------------------------------
# Deterministic, full-coverage screen of EVERY rule-strategy family
# ----------------------------------------------------------------------------
@dataclass
class FamilyResult:
    family: str
    params: dict                 # best config found (native python types)
    in_sharpe: float             # in-sample Sharpe of that config
    oos_sharpe: float            # walk-forward OOS Sharpe (falls back to in-sample if too short)
    returns: object              # per-bar returns Series of the best config
    result: object               # the BacktestResult (equity / position / stats)


def _native(v):
    try:
        x = float(v)
        return int(x) if x == int(x) else round(x, 4)
    except (TypeError, ValueError):
        return v


def screen_rule_strategies(df: pd.DataFrame, *, commission_bps: float = 1.0,
                           slippage_bps: float = 1.0, metric: str = "sharpe") -> dict:
    """Exhaustively grid-search EVERY rule-strategy family for its best config, backtest it, and
    score THAT config out-of-sample (walk-forward, with the fold count adapted to the history so
    it actually runs on short series; falls back to in-sample only when it truly can't).

    This is the deterministic, full-coverage counterpart to `research_single`'s bandit sampling —
    the SINGLE search path the report builder reuses, so strategy search is never re-implemented
    elsewhere. Returns ``{family: FamilyResult}`` (params are real searched values, never a
    placeholder).
    """
    out: dict = {}
    for k, Cls in REGISTRY.items():
        grid = RULE_SPACE.get(k)
        try:
            params: dict = {}
            if grid:
                tbl = opt.grid_search(Cls, df, grid, metric=metric,
                                      commission_bps=commission_bps, slippage_bps=slippage_bps)
                if len(tbl) and metric in tbl.columns and pd.notna(tbl.iloc[0].get(metric)):
                    params = {pk: _native(tbl.iloc[0][pk]) for pk in grid if pk in tbl.columns}
            strat = Cls(**params) if params else Cls()
            r = bt.backtest(df, strat.generate_signal(df), commission_bps=commission_bps,
                            slippage_bps=slippage_bps)
            oos = None
            if params:
                combo = {pk: [pv] for pk, pv in params.items()}
                for ns in (4, 3, 2):       # adapt folds to history (>=30 bars/fold needed)
                    try:
                        oos = float(opt.walk_forward(Cls, df, combo, n_splits=ns,
                                                     metric=metric).oos_stats.get("sharpe"))
                        break
                    except Exception:  # noqa: BLE001
                        continue
            in_sh = round(float(r.stats.get("sharpe", float("nan"))), 2)
            out[k] = FamilyResult(k, params, in_sh,
                                  round(oos, 2) if (oos is not None and np.isfinite(oos)) else in_sh,
                                  r.returns.fillna(0.0), r)
        except Exception:  # noqa: BLE001
            continue
    return out
