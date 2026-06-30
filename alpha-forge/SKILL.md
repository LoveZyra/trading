---
name: alpha-forge
description: >-
  Build, backtest, validate & run systematic stock-trading strategies and research factors/models:
  backtesting (trend/mean-reversion/momentum/pairs), indicators, Sharpe/drawdown, walk-forward
  optimization, multi-factor screens, custom-factor mining/validation, ML factor models,
  cross-sectional AI stock-selection (IC/ICIR/RankIC, quantile, long-short), automated
  factor/model research, fundamentals & news/sentiment, portfolio sizing & risk,
  signal-to-broker-order. Triggers: 回测, 量化, 交易策略, 选股, AI选股, 横截面选股, 横截面排序, 多空组合, 因子, 多因子, 夏普, 择时,
  基本面, 舆情, 自动研究, 组合风险, 仓位, backtest, trading strategy, quant, mean reversion, momentum, pairs
  trading, factor model, walk-forward, ML model, cross-sectional ranking, stock ranking, IC, ICIR,
  RankIC, long-short, news sentiment, portfolio risk, position sizing, RD-Agent, IBKR/盈透. Markets:
  US, A-share (沪深), HK, Korea via broker MCP or free libs (yfinance/akshare/pykrx). No-strategy
  company/sector writeup → use equity-research-report.
---

# Alpha-Forge

A toolkit for the full systematic-trading research loop: **get data → compute
indicators → run a strategy → measure performance honestly → validate
out-of-sample → (optionally) generate a live order.**

The guiding principle of everything here: a backtest is only worth as much as its
honesty. Most "amazing" strategies are look-ahead bugs or overfit curves. The code
is built so the honest path is the easy path — signals are lagged automatically,
costs are charged on every trade, and walk-forward validation is one flag away.

## When to reach for this

Any request to design, test, or reason about a rules-based trading strategy on
equities — single-name timing, a portfolio screen, or a stat-arb pair — or to pull
fundamentals/news and fold them into a factor model. If the user just wants a company
writeup or sector summary with no strategy/backtest, prefer the
`equity-research-report` skill instead.

## Layout

```
alpha-forge/
├── scripts/
│   ├── data/            # source-agnostic loaders:
│   │   ├── (prices)     #   IBKR MCP, yfinance, akshare, pykrx -> canonical OHLCV
│   │   ├── fundamentals.py  # PE/PB/ROE/margins/growth (yfinance, akshare, JSON)
│   │   ├── news.py      #   company headlines (yfinance, akshare, Web-search JSON)
│   │   └── sentiment.py #   bilingual (EN+中文) finance-lexicon sentiment scoring
│   ├── indicators.py    # SMA/EMA, RSI, MACD, Bollinger, ATR, ADX, z-score, Donchian...
│   ├── backtest.py      # vectorized single-asset + portfolio engine (auto lag + costs)
│   ├── metrics.py       # Sharpe, Sortino, Calmar, max drawdown, profit factor...
│   ├── strategies/      # trend, mean-reversion, multi-factor (price+value+quality+news)
│   ├── models.py        # ML predictors: factors->forward return (ridge/sklearn/lightgbm)
│   ├── autoresearch.py  # RD-Agent-style auto research loop + UCB bandit + co-opt + ensemble
│   ├── factor_lab.py    # implement/VALIDATE(causality)/scorecard/backtest a custom factor
│   ├── validation.py    # deflated Sharpe / PSR / PBO — multiple-testing haircut
│   ├── sizing.py        # risk parity / inverse-vol / vol-target position sizing
│   ├── regime.py        # vol/trend regime detection + exposure scaling
│   ├── levels.py        # suggested buy/stop/target levels from price structure + ATR
│   ├── portfolio.py     # concentration/correlation/ENB/VaR/beta/stress — portfolio risk
│   ├── signal_tracker.py # log daily signals + evaluate hit-rate/calibration (feedback loop)
│   ├── optimize.py      # grid search + walk-forward out-of-sample validation
│   ├── rebalance.py     # last-actual-trading-day rebalance dates (never calendar labels)
│   ├── param_grids.py   # shared default parameter grids (autoresearch + CLI)
│   ├── report.py        # equity/drawdown chart + markdown report
│   └── run_backtest.py  # CLI tying it all together
├── references/
│   ├── data_sources.md  # HOW to pull PRICE data from each source (incl. IBKR hand-off)
│   ├── fundamentals_news.md # fundamentals, news, sentiment, macro, market, recency, calendar
│   ├── optimization_roadmap.md # portfolio risk, options, feedback, A-share microstructure
│   ├── autoresearch.md   # auto research loop, ML model layer, factor-model co-opt (RD-Agent)
│   ├── factor_extraction.md # turn a report/paper factor into a validated, backtested one
│   ├── research_roadmap.md  # recent arXiv papers mapped to skill features (what & why)
│   ├── strategies.md     # strategy design notes & parameter guidance
│   └── pitfalls.md       # look-ahead, overfitting, costs, survivorship — read this
├── sectors.json         # editable ticker->sector map (auto-merged at import)
└── examples/
    └── quickstart.py
```

## The workflow

### 1. Get the data

All price loaders return ONE canonical OHLCV DataFrame (lowercase
`open/high/low/close/volume`, DatetimeIndex). Downstream code never cares about the
source.

- **Free data (default for backtesting)** — runs in the sandbox, no login:
  ```python
  from scripts.data.loader import load
  df = load("AAPL", source="yfinance", start="2020-01-01")     # US & global
  df = load("600519", source="akshare", market="cn")            # 沪深 A-share
  df = load("00700", source="akshare", market="hk")             # Hong Kong
  df = load("005930", source="pykrx")                           # Korea
  ```
- **IBKR-style broker MCP (for live/delayed quotes)** — the broker tools can only
  be called by *you* (Claude), not by the Python sandbox. So the pattern is a
  hand-off: you call the MCP, save the result to a JSON file, and the loader reads
  it. **Read `references/data_sources.md` for the exact steps.**

### 2. Pick or build a strategy

Ready-made templates live in `scripts/strategies/` (registry keys in parentheses):

- **Trend / 趋势**: `MACrossover` (ma_crossover), `Breakout` (breakout, Donchian),
  `TimeSeriesMomentum` (ts_momentum), `MACDTrend` (macd_trend)
- **Mean reversion / 均值回归**: `ZScoreReversion`, `BollingerReversion`,
  `RSIReversion`, `PairsTrading` (cointegration spread — import-only: it needs a partner
  price series, so it has no registry key / `--strategy` slug; use it via direct import)
- **Multi-factor / 多因子选股**: `multi_factor.multi_factor_signal(...)` blends price,
  fundamental and news factors (see 2b).

To build a new one, subclass `Strategy` and implement `generate_signal(df) -> Series`
returning a target position in `[-1, 1]`. See `references/strategies.md`. Keep it
causal — never read a future bar; the engine handles execution lag for you.

### 2b. Fundamentals, news & sentiment (优先 free libs, broker/Web 补充)

Beyond price, the skill pulls **fundamentals** (PE/PB/ROE/margins/growth) and
**company news**, and scores news **sentiment** with a bilingual EN+中文 finance
lexicon. All three plug into the multi-factor model as `value` / `quality` /
`growth` / `sentiment` factors.

```python
from scripts.data import fundamentals as F, news, sentiment as S

F.load("AAPL", source="yfinance")                 # canonical fundamentals dict
F.load("600519", source="akshare", market="cn")    # A-share fundamentals
scored, agg = news.fetch_with_sentiment("AAPL", source="yfinance")  # news + sentiment
S.score("业绩预增 大涨 利好")                        # +; "跌停 爆雷 立案" -> -
```

`news.py`, `fundamentals.py` and the broker quotes share one rule: **WebSearch and
the broker/IBKR MCP tools can only be called by you (Claude), not by the sandbox.**
So for the widest, freshest coverage, you search/pull, save the results to a JSON
file, and the script reads it via `source="json"`. **Read
`references/fundamentals_news.md`** for the canonical fields, the hand-off format,
and — importantly — the point-in-time caveat (a single current fundamentals/news
snapshot applied across history is a look-ahead approximation; treat it as a
present-day screen unless you supply dated snapshots).

To build the combined factor model:

```python
from scripts.strategies import multi_factor as mf
from scripts import backtest as bt

weights = mf.multi_factor_signal(
    prices,                                  # {symbol: OHLCV}
    factor_weights={"momentum":0.3,"low_vol":0.2,"value":0.2,"quality":0.2,"sentiment":0.1},
    rebalance="ME", top=0.4,
    fundamentals_panel=funds,                # from F.load_panel(universe)
    sentiment_by_symbol=senti,               # {symbol: mean_sentiment}
)
result = bt.backtest_portfolio(mf.build_panel(prices, "close"), weights)
```

### 2c. Automated research, ML models & factor extraction (RD-Agent-inspired)

Beyond hand-running strategies, the skill can **automate the research loop** in the
spirit of Microsoft's RD-Agent(Q): propose a hypothesis → backtest it → score it
out-of-sample → let a bandit pick the next direction, alternating factor vs. model
search. Read `references/autoresearch.md` and `references/factor_extraction.md`.

```python
from scripts import autoresearch as AR, models as Mdl, factor_lab as FL

# (a) auto-search rule strategies on one asset (UCB bandit over families, walk-forward OOS)
rep = AR.research_single(df, iterations=30); print(rep.best, rep.leaderboard.head())

# (a2) auto-search factor blends + ML on a universe — search on train, winner judged
#      on a HELD-OUT tail (rep.best.extra['holdout_sharpe'] is the number to trust)
repp = AR.research_portfolio(prices, iterations=24, fundamentals_panel=funds); print(repp)

# (b) ML model: learn factors -> forward return, walk-forward, with purge + IC
res = Mdl.ml_factor_backtest(prices, model=Mdl.RidgeModel(1.0),
                             fundamentals_panel=funds, sentiment_by_symbol=senti)
print(res.stats, "IC=", res.ic)              # ridge=numpy(always); sklearn/lightgbm optional

# (c) factor-model co-optimization (alternates factor weights <-> model choice)
co = AR.cooptimize_factor_model(prices, rounds=3, fundamentals_panel=funds)

# (d) extract a factor from a report/paper, VALIDATE causality, then backtest
def mom_12_1(d): return d["close"].shift(21)/d["close"].shift(21+252) - 1
FL.validate_factor(mom_12_1, df)             # catches look-ahead before you trust it
FL.backtest_custom_factor(mom_12_1, df, mode="momentum")
```

Every driver is scored out-of-sample, but knowing *how* matters: `research_single`
walk-forwards each trial; `research_portfolio` and `cooptimize_factor_model` run the
*search* on a train slice and judge the single winner on a held-out tail
(`rep.best.extra['holdout_sharpe']`, or the co-opt's `holdout`) — trust that tail, not
the per-trial leaderboard score (which is the train-side *selection* metric). For an
extra multiple-testing haircut, feed the winner's returns to
`validation.deflated_sharpe_ratio(n_trials=…)`. `factor_lab.validate_factor` refuses
look-ahead factors (now probed at several cut points, not one). The "research agent"
proposing ideas is you (Claude) — you shape the search space and seed factors; the loop
does the honest evaluation.

### 2d. Honest-backtest upgrades (recent arXiv literature)

Several modules harden the core "is this edge real?" question. See
`references/research_roadmap.md` for the full paper→feature map and
`references/pitfalls.md` §9-10.

```python
from scripts import validation as V, sizing as SZ, regime as RG, autoresearch as AR

# Deflate the best Sharpe a search found (multiple-testing haircut)
V.deflated_sharpe_ratio(result.returns, n_trials=200)   # ~0.5 => the 'winner' is luck
V.pbo_cscv(trial_returns_df)                             # Probability of Backtest Overfitting

# Risk-based sizing instead of equal weight
w_rp = SZ.risk_parity_weights(panel_close)               # equal risk contribution
w_iv = SZ.inverse_vol_weights(weights, panel_close)      # 1/vol tilt
scale = SZ.vol_target_scale(result.returns, target_vol=0.10)

# Size-aware costs: square-root market impact
bt.backtest(df, sig, cost_model="sqrt", impact_coef=10, capital=1e6)

# Regime overlay: cut exposure in high-vol / bear states (shallower drawdowns)
sig_safe = RG.apply_regime(sig, df["close"])

# Ensemble the top-k auto-research winners rather than betting on one
ens, members = AR.ensemble_top_k(report, df, k=3)
```

These exist because searching hard *creates* false positives: an automated loop will
find in-sample flukes unless you deflate (validation), control risk (sizing/regime) and
prefer robust ensembles. FINSABER (2505.07078) is the cautionary evidence.

### 2e. AI 选股 / 横截面排序 (cross-sectional stock-selection)

Use this path when the question is **"of these N stocks, which are relatively stronger?"**
(rank a universe, build a long/long-short book) rather than timing one name. Orthogonal to
the single-asset path and **composes** with it: the ranker picks *what* to hold, then
`levels`/`sizing`/`regime`/`backtest` decide *when/how much*.

```python
from scripts import universe, panel as PN, xsec_eval, xsec_autoresearch, xsec_report
from scripts.data import loader
uni  = universe.build_universe({"market": "US", "source": "list", "symbols": [...]})
data = loader.load_many(uni["symbols"])                       # {symbol: OHLCV}
res  = xsec_eval.evaluate_cross_section(data, horizon=21, rebalance="ME", top_frac=0.2)
print(res["scorecard"]); print(xsec_report.scorecard_markdown(res))
lb   = xsec_autoresearch.search(data, horizon=21, rebalance="ME")   # 因子×模型 排行榜
```

Models reuse the `FactorModel` interface (`RidgeModel`/`LGBMModel`/`MLPModel`, optional
`xsec_models.TorchRanker`); a deep model (RAVEN/LSTM/foundation) trained on your machine
plugs in via `models.load_external_scores`, judged by the **same** scorecard.

**Honesty rails (see `references/ai_stock_selection.md`):** RankIC is primary (Spearman);
"usable" ~ RankIC >= 0.03 **and** RankICIR >= 0.3. The builder **warns** when the universe
is < ~30 names or one sector — same-sector names co-move, leaving little to rank
(empirically 9 AI-semis over 5y gave RankIC ~ 0 even for ridge). Always purge labels,
charge costs, mind survivorship bias.

### 3. Backtest

```python
from scripts import backtest as bt
from scripts.strategies import MACrossover

strat = MACrossover(fast=20, slow=50)
result = bt.backtest(df, strat.generate_signal(df),
                     commission_bps=1.0, slippage_bps=1.0)   # costs ON by default
print(result.stats)                  # full metrics dict
bench = bt.buy_and_hold(df)          # ALWAYS compare against this
```

`result.stats` carries `total_return`, `cagr`, `ann_volatility`, `sharpe`, `sortino`,
`max_drawdown`, `calmar`, `win_rate`, `profit_factor`, `exposure`, `turnover_annual`,
`n_trades` and `total_costs` (those are the exact dict keys). The engine shifts the
signal by one bar (`lag=1`) so you can never trade on information you didn't have.

For a multi-asset weight panel use `bt.backtest_portfolio(panel_close, weights)`.

### 4. Validate out-of-sample (do not skip)

A single in-sample backtest tells you almost nothing. Prove the edge survives on
unseen data:

```python
from scripts import optimize as opt
from scripts.strategies import MACrossover

wf = opt.walk_forward(MACrossover, df,
                      grid={"fast": [10, 20, 30], "slow": [50, 100, 150]},
                      n_splits=5, metric="sharpe")
print(wf.oos_stats)   # out-of-sample metrics — THIS is the number to trust
print(wf.folds)       # params chosen per fold + train-vs-test gap
```

If the train metric is great but `oos_stats` collapses, the strategy is overfit.
`references/pitfalls.md` explains why and what to do.

### 5. Report

```python
from scripts import report as rpt
rpt.plot_result(result, benchmark=bench, path="equity.png")   # equity + drawdown
md = rpt.markdown_report(result, name="MA crossover", benchmark=bench)
```

**结构化 HTML 报告（复盘/分析报告的最终产出，取代 markdown）。** 不再把报告写成
markdown 再转 HTML——直接构造一个 **结构化 report dict** 交给 `scripts.html_report`，
它会渲染成一份**机构研报风、自包含单文件、可打印为 PDF** 的 HTML：买卖点为核心
（单标的=价格阶梯 止损→买区→现价→目标 + 盈亏比；组合=带内嵌 R/R 条的密集大表），
配大盘/宏观评分计、三层情绪条、策略买卖点图（▲买/▼卖 + 持仓阴影 + 日期），**红涨绿跌**（A股惯例）。
字段「给了就渲染、不给就跳过」，完整契约见 `SKILL` 同级的 **`SCHEMA.md`**。
无需第三方库（CSS/JS 已内联，中文走系统字体回退）。

```python
from scripts import html_report as H

report = {"meta": {...}, "verdict": {...}, "levels": [...], "backtest": {...}, ...}
H.save_html(report, "trading/reports/美股复盘_2026-06-09.html")   # -> 单文件 .html
html = H.render(report)                                          # 或直接拿字符串
```

> **每个 key 的完整形状、配色与打印设置都在同级 `SCHEMA.md` ——构造 report dict 前先读它**
> （那里有逐字段的契约与示例，本处不再重复）。同一套模板按 `report_type` 切换：`single`
> （个股）/`portfolio`（组合自选池）/`market`（市场扫描）/`backtest`（回测）/`attribution`/
> `macro`。`scripts.report` 仍可生成 markdown 草稿留痕，但**最终展示走上面的 dict**——版式
> 稳定、买卖点层级清晰，不再有 markdown 表格错位 / 内联 span 渲染差的问题。

Or do steps 1–5 from the command line:

```bash
python scripts/run_backtest.py --symbol AAPL --strategy ma_crossover \
    --fast 20 --slow 50 --out ./report
python scripts/run_backtest.py --symbol AAPL --strategy ma_crossover --walk-forward
```

### 6. From signal to live order (optional)

`strat.latest_signal(df)` returns the target position implied by the most recent
bar — that's what you'd act on today. To actually place it through the broker, use
the MCP order tools (`create_order_instruction`, `get_account_positions`,
`get_account_balances`). **Read `references/data_sources.md` → "Placing orders"
first.** Treat live trading with extreme care: confirm the contract, size against
real balances, and never auto-submit without the user's explicit go-ahead. Paper
trade before risking capital.

## Setup

The free loaders need their libraries:
```bash
pip install yfinance akshare pykrx pandas numpy matplotlib pyarrow markdown --break-system-packages
```
Only install what the chosen market needs (yfinance for US/global, akshare for
China/HK news+fundamentals, pykrx for Korea). The broker MCP path needs none of these.
The ML model layer's ridge is pure numpy; for random forests / gradient boosting add
`pip install scikit-learn lightgbm` (optional — the skill degrades gracefully without).
`validation.py` uses scipy if present but falls back to pure-numpy normal CDFs, so
`scipy` is optional too.

## A note on responsible use

This skill helps research and test ideas. It does not predict the future and a good
backtest is not a guarantee. Always state assumptions (costs, slippage, universe,
period), compare against buy-and-hold, prefer out-of-sample evidence, and remember
the point-in-time caveat for fundamental/news factors. Nothing here is investment
advice.
