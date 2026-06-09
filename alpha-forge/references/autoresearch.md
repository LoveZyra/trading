# Automated research loop & ML model layer (RD-Agent-inspired)

This is the skill's take on the ideas in Microsoft's **RD-Agent(Q)**
(arXiv:2505.15155): automate the quant R&D loop — propose a hypothesis, implement &
backtest it, evaluate out-of-sample, and let a bandit scheduler adaptively pick the
next direction, alternating between **factor** search and **model** search.

It is deliberately a *native, lightweight* re-implementation, not a wrapper around
RD-Agent. See "Using the real RD-Agent" at the bottom for when to reach for the full
system instead.

## Table of contents
1. The loop: Research → Develop → Feedback
2. ML model layer (`models.py`)
3. The bandit drivers (`autoresearch.py`)
4. Factor–model co-optimization
5. Honesty guarantees
6. Using the real RD-Agent (Qlib)

---

## 1. The loop

```
        ┌──────────── Research ────────────┐
        │  propose a hypothesis             │   <- YOU (Claude) shape the space
        │  (a strategy/factor/model config) │
        └───────────────┬───────────────────┘
                        v
        ┌──────────── Develop ─────────────┐
        │  implement = existing code        │   <- strategies / factors / models
        │  execute = walk-forward backtest  │
        └───────────────┬───────────────────┘
                        v
        ┌──────────── Feedback ────────────┐
        │  OOS metrics + IC                 │   <- never in-sample
        │  UCB1 bandit updates direction    │
        └───────────────┬───────────────────┘
                        └────────► next iteration (exploit/explore)
```

The "research agent" is you. The loop samples a structured search space and does the
honest bookkeeping; you make it smart by editing the space, adding factors you
extracted from a report (see factor_extraction.md), or seeding directions you believe
in. That division — human/LLM ideas + automated honest evaluation — is the whole point.

---

## 2. ML model layer (`models.py`)

Where rule strategies hand-code data→position, a **FactorModel** *learns* the map
from factor exposures to next-period cross-sectional return, then ranks names by
predicted return.

```python
from scripts import models as Mdl

Mdl.RidgeModel(alpha=1.0)                       # pure numpy, always available
Mdl.SklearnModel("RandomForestRegressor", n_estimators=200, max_depth=4)  # needs sklearn
Mdl.LGBMModel(n_estimators=200, max_depth=4)    # needs lightgbm

res = Mdl.ml_factor_backtest(prices,            # {symbol: OHLCV}
                             model=Mdl.RidgeModel(1.0),
                             fundamentals_panel=funds, sentiment_by_symbol=senti,
                             rebalance="ME", horizon=21, train_window=252, top=0.3)
print(res.stats, "IC =", res.ic)                # IC = realized predictive correlation
```

`ml_factor_backtest` re-fits the model before every rebalance on a trailing window,
with a **purge**: the label for a training sample (forward `horizon`-bar return) must
be fully realized before the rebalance date, so the model never trains on the future.
It reports the portfolio backtest **and** the Information Coefficient — the cleanest
read on whether the model predicts anything.

Optional libs install fine anywhere (incl. the sandbox): `pip install scikit-learn
lightgbm`. Without them, RidgeModel still gives you a real linear model.

---

## 3. The bandit drivers (`autoresearch.py`)

```python
from scripts import autoresearch as AR

# (a) rule-strategy search on one asset
rep = AR.research_single(df, iterations=30)
print(rep.best, rep.leaderboard.head())        # best OOS config + full log
print(rep.bandit_summary)                       # how the bandit spent its pulls

# (b) factor + model search on a universe
rep2 = AR.research_portfolio(prices, iterations=24,
                             fundamentals_panel=funds, sentiment_by_symbol=senti)
```

`UCB1` (upper-confidence-bound) picks the direction maximizing
`mean_reward + sqrt(2 ln T / pulls)` — it tries every arm once, then concentrates on
what's paying off while still occasionally exploring. Reward = squashed OOS Sharpe.

---

## 4. Factor–model co-optimization

```python
co = AR.cooptimize_factor_model(prices, rounds=3,
                                fundamentals_panel=funds, sentiment_by_symbol=senti)
print(co["best_weights"], co["best_model"], co["history"])
```

Alternates: (1) fix the model, search factor-weight blends; (2) fix the best factors,
search models (ridge α, random forest, LightGBM, or plain equal-weight). Repeats. This
is the compact, seconds-not-hours version of RD-Agent(Q)'s alternating optimization.
On pure noise it correctly prefers the simplest model — a good sign the OOS scoring
isn't being gamed.

---

## 5. Honesty guarantees

Everything is scored out-of-sample, so the leaderboard can't be won by overfitting:
- rule trials → walk-forward OOS Sharpe;
- model trials → purged, rolling-train cross-sectional backtest + IC;
- the bandit reward is the OOS metric, never in-sample.

This matters more here than anywhere: an *automated* search will find in-sample flukes
fast if you let it score in-sample. Don't. See pitfalls.md.

---

## 6. Using the real RD-Agent (Qlib)

The native loop captures most of the value for a connected-broker, single-user
workflow. If you want the full system — LLM-driven hypothesis generation, Co-STEER
code generation, factor extraction from reports at scale — use Microsoft's package:

```bash
pip install rdagent          # Linux + Docker required
rdagent fin_quant            # joint factor+model evolution on Qlib data
rdagent fin_factor           # factor evolution only
rdagent fin_factor_report --report-folder=<your reports>   # extract factors from PDFs
```

It needs Docker, an LLM API key (OpenAI/Azure/DeepSeek via LiteLLM) + an embedding
model, and runs on Qlib's data layer (~$10/run in the paper). Bridge pattern: run
RD-Agent to discover factors/models, then port the winning factor formulas into this
skill (implement as causal factor functions, validate with factor_lab, and backtest /
trade through your broker connection). That keeps the day-to-day workflow lightweight
and broker-connected while borrowing RD-Agent's discovery power when you want it.
