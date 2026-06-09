# Extracting factors from research reports, financial reports & papers

Inspired by RD-Agent's `fin_factor_report`: turn a *described* factor in a document
into a *runnable, validated, backtested* factor. The reading and translation is LLM
work (you, Claude); `factor_lab.py` provides the safety rails so what you implement is
causal and actually predictive before it touches a strategy.

## The workflow

1. **Read the source.** A sell-side research note, a 10-K / 年报, an academic paper, or
   even a news cluster. Use the `pdf` skill for PDFs, the data/news layer for news.
   Pull out the precise factor definition: the formula, the inputs (price? volume?
   a fundamental line item?), the horizon, and the claimed direction (does a high
   value predict higher or lower forward returns?).

2. **Translate to a causal function.** Implement it as `f(df) -> pd.Series` using only
   past/current bars. Reuse `scripts.indicators`. Examples:
   ```python
   # "12-1 momentum": last 12 months return skipping the most recent month
   def mom_12_1(df): return df["close"].shift(21)/df["close"].shift(21+252) - 1

   # "Amihud illiquidity": |return| / dollar volume, 21-day average (higher = illiquid)
   def amihud(df):
       r = df["close"].pct_change().abs()
       dollar = (df["close"]*df["volume"]).replace(0, float("nan"))
       return (r/dollar).rolling(21).mean()

   # "52-week-high proximity": close / trailing 252-day high (Hou et al.)
   def high52(df): return df["close"]/df["high"].rolling(252).max()
   ```

3. **VALIDATE causality — do not skip.** Many described factors are easy to implement
   with an accidental look-ahead. The validator proves the factor uses no future data:
   ```python
   from scripts import factor_lab as FL
   chk = FL.validate_factor(mom_12_1, df)
   print(chk)          # FactorCheck[OK] causal=True ...  (or PROBLEM with the reason)
   ```
   It recomputes the factor on a truncated history and checks that past values are
   unchanged when future bars are appended. A centered rolling window, a forward
   `shift(-k)`, using a restated/point-in-time-violating field — all get caught.

4. **Quick edge check (IC).** Before a full backtest, see if the factor correlates
   with forward returns at all:
   ```python
   FL.factor_ic(mom_12_1, df, horizon=21)     # |IC| > ~0.03 on real data is interesting
   ```

5. **Backtest it.** Single asset, as a continuous signal:
   ```python
   res, chk = FL.backtest_custom_factor(mom_12_1, df, mode="momentum")  # refuses cheaters
   print(res.stats)
   ```
   `mode="momentum"` longs high-factor names; `mode="reversion"` shorts them — match it
   to the claimed direction. For a cross-sectional version, register the factor and add
   it to the multi-factor model (see below).

6. **Promote to the factor model / auto-research.** Register it so it's reusable:
   ```python
   FL.register_custom_factor("mom_12_1", mom_12_1)
   ```
   To use it cross-sectionally across a universe, compute it per symbol into a panel
   and z-score it the same way `multi_factor` treats price factors, then include it in
   the `factor_weights` blend or feed it as a feature column to `models.ml_factor_backtest`.

## Cautions
- **Point-in-time.** A factor built from a *current* fundamental snapshot applied over
  history is a look-ahead approximation (see fundamentals_news.md §5). For an honest
  historical test you need the value as known on each past date.
- **Survivorship & universe.** A factor that "works" only on today's survivors is
  suspect (see pitfalls.md).
- **One factor, many tries.** If you implement 50 factors and keep the best, you're
  data-snooping. Hold out a period; prefer factors with an economic story.

The discipline: a described factor is a *hypothesis*. `factor_lab` turns it into a
causal implementation and an out-of-sample test — that's what makes it trustworthy.
