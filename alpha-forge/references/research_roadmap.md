# Research roadmap — arXiv papers mapped to skill features

Recent quant/backtesting literature, and exactly where each idea lives in this skill.
The first block is **implemented**; the second is **deliberately out of scope** for a
lightweight, broker-connected, offline-capable skill (with pointers if you want to go
there).

## Implemented (paper -> module)

| Idea | Paper | Where in the skill |
|---|---|---|
| Deflated Sharpe / PSR / Probability of Backtest Overfitting | Bailey & López de Prado; backtest-overfitting literature | `validation.py` — `deflated_sharpe_ratio`, `probabilistic_sharpe_ratio`, `pbo_cscv`. Deflates the best-of-N Sharpe the auto-research loop finds. |
| Risk parity / equal-risk-contribution allocation | Choi & Chen 2022 ([2203.00148](https://hf.co/papers/2203.00148)) | `sizing.py` — `risk_parity_weights`, `inverse_vol_weights`. |
| Volatility targeting | TS-momentum multi-task ([2306.13661](https://hf.co/papers/2306.13661)) | `sizing.py` — `vol_target_scale`. |
| Square-root market-impact costs | Bugaenko 2020 ([2004.08290](https://hf.co/papers/2004.08290)); Almgren | `backtest.py` — `cost_model="sqrt"`. |
| Turnover-regularized / finance-grounded objective | Khubiev et al. 2026 ([2509.04541](https://hf.co/papers/2509.04541)) | `models.py` — `weight_smoothing` in `ml_factor_backtest`. |
| Factor quality scorecard (IC stability, diversity, decay) | AlphaEval ([2508.13174](https://hf.co/papers/2508.13174)); AlphaAgent ([2502.16789](https://hf.co/papers/2502.16789)) | `factor_lab.py` — `factor_scorecard`. |
| LLM news sentiment (predicts next-day, underreaction) | Lopez-Lira & Tang ([2304.07619](https://hf.co/papers/2304.07619)) | `data/sentiment.py` — `headlines_for_llm` / `apply_llm_scores`. |
| Market-regime detection & exposure scaling | FINSABER ([2505.07078](https://hf.co/papers/2505.07078)); BOCPD ([0710.3742](https://hf.co/papers/0710.3742)); HMM regimes ([2407.19858](https://hf.co/papers/2407.19858)) | `regime.py` — `vol_regime`, `trend_regime`, `regime_scale`, `cusum_changepoints`. |
| Strategy ensembling | Lam 2024 ([2406.03652](https://hf.co/papers/2406.03652)) | `autoresearch.py` — `ensemble_top_k`. |
| Factor–model co-optimization, bandit-scheduled R&D loop | RD-Agent(Q) ([2505.15155](https://hf.co/papers/2505.15155)) | `autoresearch.py` — `research_*`, `cooptimize_factor_model`. |
| LLM/RL/evolutionary alpha mining (formulaic factors) | Alpha Jungle/MCTS ([2505.11122](https://hf.co/papers/2505.11122)); AlphaAgent; CogAlpha ([2511.18850](https://hf.co/papers/2511.18850)); QuantaAlpha ([2602.07085](https://hf.co/papers/2602.07085)) | `factor_lab.py` workflow — Claude proposes formulaic factors, validates causality, scores, registers. |

## Out of scope (heavier; bridge if you want it)

| Idea | Paper | Why deferred / how to bridge |
|---|---|---|
| Deep sequence models for return prediction | Stockformer ([2401.06139](https://hf.co/papers/2401.06139)); ResNLS ([2312.01020](https://hf.co/papers/2312.01020)); MTMD ([2212.08656](https://hf.co/papers/2212.08656)) | Needs GPUs/heavy training; the skill stays light. Bridge: train externally, expose the model's per-name score as a factor column for `ml_factor_backtest`. |
| Time-series foundation models | Re(Visiting) TSFMs in Finance ([2511.18578](https://hf.co/papers/2511.18578)) | Domain pre-training required; same bridge as above. |
| Deep RL portfolio optimization | DRL China ([2412.18563](https://hf.co/papers/2412.18563)); RL w/ dynamic embedding ([2501.17992](https://hf.co/papers/2501.17992)) | Reward-hacking & non-stationarity risks; heavy. Our bandit + risk overlays cover the pragmatic 80%. |
| Full multi-agent trading systems | TradingGroup ([2508.17565](https://hf.co/papers/2508.17565)) | The scheduled-task agents already give a lightweight version (news + signals + report). |

## A caution worth keeping (FINSABER, 2505.07078)
Empirically, LLM/timing strategies that look great in-sample often underperform passive
benchmarks over long horizons and across regimes. This is *why* the skill leans so hard
on out-of-sample validation, deflated Sharpe/PBO, regime-aware risk controls, and
buy-and-hold benchmarking. Treat every backtest edge as a hypothesis, not a promise.
