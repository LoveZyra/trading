# Pitfalls — why most backtests lie

Read this before trusting any result. These are the failure modes that turn a
beautiful equity curve into a losing live strategy. The code in this skill defends
against several of them automatically, but it can't save you from all of them.

## 1. Look-ahead bias (the silent killer)
Using information that wasn't available yet. The classic forms:
- Computing a signal on bar *t*'s close and "executing" at that same close.
- An indicator that references future bars (`shift(-1)`).
- Using adjusted prices/fundamentals that were restated after the fact.

**Defense in this skill:** `backtest()` shifts the signal by `lag=1` before
computing returns, so a signal formed on today's close is executed on the *next*
bar. Indicators in `indicators.py` are all causal. You can break this by passing a
forward-shifted signal — don't.

## 2. Overfitting / curve-fitting
With enough parameters or enough tries, you can fit any history perfectly and learn
nothing. Symptoms: a strategy with 6 finely-tuned thresholds; the "best" grid-search
combo that's wildly better than its neighbours.

**Defenses:**
- Constrain parameter ranges to economically sensible values (don't scan MA from 2
  to 500).
- Prefer fewer parameters and smooth parameter surfaces (neighbours should perform
  similarly).
- **Use `optimize.walk_forward`** — optimize on in-sample, measure on the next unseen
  window. If `oos_stats` is far worse than the in-sample grid winner, it's overfit.
  The out-of-sample number is the only one worth quoting.

## 3. Ignoring transaction costs
Commission + slippage + spread. A strategy that flips position daily can look great
gross and bleed to death net. High-frequency mean-reversion is especially exposed.

**Defense:** costs are ON by default (`commission_bps` + `slippage_bps`, charged on
turnover). Always report `turnover_annual` and `total_costs`. Stress-test by doubling
the cost assumption — a real edge survives it.

## 4. Survivorship bias
Backtesting only today's surviving tickers ignores the ones that delisted/went to
zero. It quietly inflates returns, especially for multi-year equity screens.

**Mitigation:** use a point-in-time universe if you can; at minimum, acknowledge it
and be skeptical of long-horizon single-name results on hand-picked winners.

## 5. Data-snooping / multiple testing
Test 200 strategies, the best one looks "significant" by luck. Keep a hold-out
period you never touch until the very end. Don't iterate against your test set.

## 6. Regime dependence
A trend system shines in trending years and dies in chop. Check performance across
sub-periods (`wf.folds` shows per-fold results) and across different assets, not just
one lucky ticker in one lucky decade.

## 7. Unrealistic fills & liquidity
Assuming you fill at the close, in size, with no market impact. Fine for liquid
large caps in modest size; dangerous for small caps, illiquid names, or large
notional. Cap position size relative to average volume.

## 8. Non-stationarity
Markets change; a relationship that held for a decade can vanish. Re-validate
periodically, size positions by volatility (`indicators.atr`), and don't assume the
future resembles the backtest.

---

### A healthy checklist before believing a result
- [ ] Signal is lagged (no same-bar execution).
- [ ] Costs included and stress-tested.
- [ ] Compared against buy-and-hold.
- [ ] Validated out-of-sample (walk-forward), not just in-sample.
- [ ] Works on more than one asset / sub-period.
- [ ] Parameters are few, sensible, and on a smooth surface.
- [ ] Assumptions (universe, period, costs, slippage) stated explicitly.

If several boxes are unchecked, the strategy is a hypothesis, not an edge.

## 9. Multiple testing — deflate what you found
When you search N strategies/parameters (the auto-research loop does exactly this), the
*best* Sharpe is upward-biased even if nothing has real edge. Use `validation.py`:
`deflated_sharpe_ratio` (is the winner better than the best of N coin-flips?) and
`pbo_cscv` (Probability of Backtest Overfitting). A great in-sample Sharpe with DSR≈0.5
or PBO>0.5 is a mirage. Always report these alongside the headline number.

## 10. Regimes & the long-run reality check
FINSABER (arXiv:2505.07078) found timing/LLM strategies that shine in backtest often
trail buy-and-hold over long horizons and across regimes. Validate across sub-periods,
scale exposure with `regime.py`, and never assume the regime you fit on will persist.
