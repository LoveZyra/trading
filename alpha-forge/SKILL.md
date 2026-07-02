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
├── scripts/                 # 按职责分包(2026-07 重组;旧平铺路径 scripts.backtest 等已移除)
│   ├── core/                # 引擎与数学基座
│   │   ├── backtest.py      #   vectorized single-asset + portfolio engine (auto lag + costs)
│   │   ├── metrics.py       #   Sharpe/Sortino/Calmar/回撤 + 成本压力夏普 + 下跌切片
│   │   ├── indicators.py    #   SMA/EMA, RSI, MACD, Bollinger, ATR, ADX, z-score, Donchian...
│   │   ├── optimize.py      #   grid search + walk-forward OOS(cost_stress_bps 成本压力选择)
│   │   ├── validation.py    #   deflated Sharpe / PSR / PBO / CPCV / SPA — 过拟合与窥探校正
│   │   ├── rebalance.py     #   last-actual-trading-day rebalance dates
│   │   └── calendars.py     #   重建真实交易日(NYSE 口径)
│   ├── data/                # source-agnostic loaders:
│   │   ├── (prices)         #   IBKR MCP, yfinance, akshare, pykrx -> canonical OHLCV
│   │   ├── fundamentals.py  #   PE/PB/ROE/margins/growth (yfinance, akshare, JSON)
│   │   ├── news.py          #   company headlines (yfinance, akshare, Web-search JSON)
│   │   └── sentiment.py     #   bilingual (EN+中文) finance-lexicon sentiment scoring
│   ├── strategies/          # trend, mean-reversion, multi-factor (price+value+quality+news)
│   ├── research/            # 研究层
│   │   ├── autoresearch.py  #   RD-Agent 式研究循环 + bandit + co-opt + 时变/regime 集成
│   │   ├── models.py        #   ML: factors->forward return (ridge/sklearn/lgbm/MLP + OOF stacking)
│   │   ├── factor_expr.py   #   因子表达式 DSL(80 算子,ast 白名单,时序全因果)
│   │   ├── factor_zoo.py    #   Alpha101(30) + Alpha158 子集(84) 因子库 + alpha360_panel
│   │   ├── factor_lab.py    #   implement/VALIDATE(causality)/正交化/增量IC/AST正则/五维 alpha_eval
│   │   ├── crowding.py      #   因子拥挤度(持仓重合/收益相关/估值价差)+ 双曲衰减拟合
│   │   ├── decay_monitor.py #   滚动IC/多horizon衰减/half-life三法/自动预警/MRP
│   │   ├── prescreen.py     #   因子四道闸预筛(弱信号/快衰减/冗余/拥挤降权)→ 搜索空间
│   │   ├── research_memory.py #  研究经验记忆(JSONL)+ 策略聚类 + 跨市场 warm_start
│   │   ├── seq_models.py    #   序列模型管线(GRU/LSTM/Transformer, torch lazy;checkpoint 续跑)
│   │   ├── llm_factor_reasoning.py # LLM prompt 组装+JSON 解析(推理/辩论/对齐/可解释)
│   │   ├── signals.py       #   信号多法对照(6 lenses)
│   │   ├── signal_tracker.py#   log daily signals + hit-rate/calibration (feedback loop)
│   │   ├── param_grids.py   #   shared parameter grids + STRATEGY_INFO
│   │   └── compare.py       #   多标的横向对比
│   ├── xsec/                # 横截面选股(AI 选股)
│   │   ├── universe.py      #   数据驱动选池(市值/流动性基座 + 软打分)
│   │   ├── panel.py         #   价格类因子面板(截面 z-score)
│   │   ├── xsec_eval.py     #   诚实记分卡:IC/RankIC/ICIR/分位/多空
│   │   ├── xsec_autoresearch.py # 因子×模型搜索 + rank-average 集成(ensemble_top)
│   │   ├── xsec_models.py   #   TorchRanker + ListNet/ListMLE/LSList(纯numpy listwise)/LambdaMART
│   │   └── xsec_report.py   #   选股 HTML 报告
│   ├── risk/                # 风险层
│   │   ├── sizing.py        #   risk parity / inverse-vol / vol-target position sizing
│   │   ├── regime.py        #   vol/trend regime + regime 条件化集成权重
│   │   ├── conformal.py     #   共形预测区间 -> 不确定性仓位门(CPPS)
│   │   ├── optimization.py  #   MinVar/MaxSharpe/有效前沿/Black-Litterman/带约束MVO(纯numpy)
│   │   ├── portfolio.py     #   concentration/correlation/ENB/VaR/beta/stress
│   │   └── levels.py        #   suggested buy/stop/target levels from price structure + ATR
│   ├── reporting/           # 报告层
│   │   ├── html_report.py   #   结构化研报(含稳健性体检/下跌切片/成本敏感度区块)
│   │   ├── build_research.py#   「策略测试与选择」生成器(自动带上面三块)
│   │   ├── factor_tearsheet.py # Alphalens 式因子体检页(分位/IC/多空/换手,自包含HTML)
│   │   ├── report.py        #   equity/drawdown chart + markdown 草稿
│   │   ├── attribution.py   #   P&L 归因
│   │   └── newsfeed.py      #   新闻 -> 报告 alerts/groups
│   ├── trade/
│   │   └── execution.py     #   TWAP/VWAP 计划 + IS 执行质量分解 + RL 执行策略钩子
│   ├── run_backtest.py      # CLI tying it all together
│   ├── train_seq_model.py   # 序列模型本地训练入口(路径B:本地GPU训练→线上加载推理)
│   └── train_rl_executor.py # RL 执行策略训练入口骨架(模拟器证据,非实盘)
├── prompts/                 # LLM 模板:因子推理/四视角辩论/假设对齐/可解释性(agent 侧执行)
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
from scripts.core import backtest as bt

weights = mf.multi_factor_signal(
    prices,                                  # {symbol: OHLCV}
    factor_weights={"momentum":0.3,"low_vol":0.2,"value":0.2,"quality":0.2,"sentiment":0.1},
    rebalance="ME", top=0.4,
    fundamentals_panel=funds,                # from F.load_panel(universe)
    sentiment_by_symbol=senti,               # {symbol: mean_sentiment}
    weight_smoothing=0.3,                    # L1 换手收缩:桶边缘股不再每月进出(成本杀手)
)
result = bt.backtest_portfolio(mf.build_panel(prices, "close"), weights)
```

### 2c. Automated research, ML models & factor extraction (RD-Agent-inspired)

Beyond hand-running strategies, the skill can **automate the research loop** in the
spirit of Microsoft's RD-Agent(Q): propose a hypothesis → backtest it → score it
out-of-sample → let a bandit pick the next direction, alternating factor vs. model
search. Read `references/autoresearch.md` and `references/factor_extraction.md`.

```python
from scripts.research import autoresearch as AR, models as Mdl, factor_lab as FL

# (a) auto-search rule strategies on one asset (UCB bandit over families, walk-forward OOS)
#     选择默认按「成本压力夏普」(cost_stress_bps=10):高换手的纸面冠军自动降权,展示仍是真实成本下的夏普
rep = AR.research_single(df, iterations=30); print(rep.best, rep.leaderboard.head())

# (a2) auto-search factor blends + ML on a universe — search on train, winner judged
#      on a HELD-OUT tail (rep.best.extra['holdout_sharpe'] is the number to trust)
repp = AR.research_portfolio(prices, iterations=24, fundamentals_panel=funds); print(repp)

# (b) ML model: learn factors -> forward return, walk-forward, with purge + IC
res = Mdl.ml_factor_backtest(prices, model=Mdl.RidgeModel(1.0),
                             fundamentals_panel=funds, sentiment_by_symbol=senti)
print(res.stats, "IC=", res.ic)              # ridge=numpy(always); sklearn/lightgbm optional
#     model=Mdl.StackingModel() -> OOF stacking 集成(ridge×2+lgbm, 组合稳于单模型);
#     conformal_alpha=0.2 -> 共形不确定性门:|预测|<区间宽的名字自动缩仓(risk/conformal.py)

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
from scripts.core import validation as V
from scripts.risk import sizing as SZ, regime as RG, conformal as C
from scripts.research import autoresearch as AR

# Deflate the best Sharpe a search found (multiple-testing haircut)
V.deflated_sharpe_ratio(result.returns, n_trials=200)   # ~0.5 => the 'winner' is luck
V.pbo_cscv(trial_returns_df)                             # Probability of Backtest Overfitting
V.selection_robustness(trial_returns_df)                 # DSR+PBO+SPA+CPCV 一站式体检

# Risk-based sizing instead of equal weight
w_rp = SZ.risk_parity_weights(panel_close)               # equal risk contribution
w_iv = SZ.inverse_vol_weights(weights, panel_close)      # 1/vol tilt
scale = SZ.vol_target_scale(result.returns, target_vol=0.10)

# Size-aware costs: square-root market impact
bt.backtest(df, sig, cost_model="sqrt", impact_coef=10, capital=1e6)

# Regime overlay: cut exposure in high-vol / bear states (shallower drawdowns)
sig_safe = RG.apply_regime(sig, df["close"])

# Ensemble the top-k winners rather than betting on one. weighting=
#   'ewma'  (默认) 时变权重:近期表现好的成员多拿权重(AlphaForge 动态组合器)
#   'regime' regime 条件化:按当前 vol/牛熊状态下的历史表现加权 —— 修下跌段集体失效
ens, members = AR.ensemble_top_k(report, df, k=3, weighting="regime")

# 共形不确定性 -> 仓位(模型无关、分布无关;CPPS)
q = C.split_conformal_qhat(calib_residuals, alpha=0.2)   # 校准区间宽
w = w * C.conviction_scale(pred, q)                       # |pred|/q̂ 截断到 [0,1]
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
from scripts.xsec import universe, panel as PN, xsec_eval, xsec_autoresearch, xsec_report
from scripts.data import loader
uni  = universe.build_universe({"market": "US", "source": "list", "symbols": [...]})
data = loader.load_many(uni["symbols"])                       # {symbol: OHLCV}
res  = xsec_eval.evaluate_cross_section(data, horizon=21, rebalance="ME", top_frac=0.2)
print(res["scorecard"]); print(xsec_report.scorecard_markdown(res))
lb   = xsec_autoresearch.search(data, horizon=21, rebalance="ME")   # 因子×模型 排行榜
ens  = xsec_autoresearch.ensemble_top(data, lb, k=3)          # rank-average 集成前k配置,同一记分卡重打分
html = xsec_report.render_html_report({"H=21":res}, current_rank, sectors, out_path="report.html")  # 选股报告(排名先行+图表)
```

**数据驱动选池（市值/流动性/指数成分 基座 + 热门/龙头 软打分）** —— 回答“每个板块选谁”。
`universe.build_scored_universe(meta, prices, per_sector=10, cap_min=2e9, adv_min=2e7, require_index=None, sectors_override=SEC)`：
先按真实**市值 / 美元日均额 / （可选）指数成分**圈基座，再用 **size/liq/hot/hi52/lead** 软打分在每板块取 TopN。
真实数据由 `universe.meta_from_fmp(quotes, profiles, index_members)` 从行情连接器（quote/market-cap/indexes/profile）解析。详见 `references/ai_stock_selection.md`「选池方法」。 主观叠加：`universe.merge_manual(res, [syms])` 在数据驱动基础池上附加 conviction 标的（未被数据池选中者标 `source='manual'`），基础池与主观分开、可复现。

Models reuse the `FactorModel` interface (`RidgeModel`/`LGBMModel`/`MLPModel`, optional
`xsec_models.TorchRanker`); a deep model (RAVEN/LSTM/foundation) trained on your machine
plugs in via `models.load_external_scores`, judged by the **same** scorecard.

**Honesty rails (see `references/ai_stock_selection.md`):** RankIC is primary (Spearman);
"usable" ~ RankIC >= 0.03 **and** RankICIR >= 0.3. The builder **warns** when the universe
is < ~30 names or one sector — same-sector names co-move, leaving little to rank
(empirically 9 AI-semis over 5y gave RankIC ~ 0 even for ridge). Always purge labels,
charge costs, mind survivorship bias.

### 2f. 因子工程:表达式引擎 · 因子库 · 质量体检(Round 10)

因子的完整生命周期在 skill 内闭环:**写(DSL/库)→ 验(因果+五维)→ 查(冗余/拥挤/衰减)→ 用(进 xsec 记分卡)**。

```python
from scripts.research import factor_expr as FE, factor_zoo as FZ, factor_lab as FL
from scripts.research import crowding as CW, decay_monitor as DM

# 表达式因子(80 算子,ast 白名单安全求值,时序全因果;截面算子仅对 {symbol: OHLCV} 宽表生效)
p = FE.eval_expr("rank(ts_delta(close, 5) / ts_std(returns, 20))", data)   # date×symbol
FL.validate_factor(FE.expr_to_callable("ts_zscore(close, 20)"), df)        # 因果性体检

# 因子库:Alpha101(30 个纯价量) + Alpha158 代表子集(84 个,8 类全覆盖)
panels = FZ.compute_library(data, which="alpha158", max_factors=40)        # {name: 宽表}
res = xsec_eval.evaluate_cross_section(data, panels=panels)                # 同一张诚实记分卡

# 质量体检:冗余 / 拥挤 / 衰减 —— 数字面前再决定要不要这个因子
FL.incremental_ic(p, existing_panels=panels, close_panel=close)   # 正交后还剩多少预测力
CW.crowding_score(p, panels, close)                               # 综合拥挤度,>0.7 预警
DM.decay_warning(p, close, rebalance_days=21)                     # half-life < 调仓周期 → 弃
FL.alpha_eval(p, data, existing_panels=panels)                    # 五维(AlphaEval 式)综合分
FL.complexity_control("rank(ts_corr(open, volume, 10))")          # AST 深度/参数正则(AlphaAgent)

# 中性化口径:行业/风格剥离后还有没有纯选股 alpha
res_n = xsec_eval.evaluate_cross_section(data, panels=panels, neutralize="industry",
                                         sector_map=SEC)

# 因子 tearsheet(自包含 HTML:分位曲线/IC 时序/多空/换手)
from scripts.reporting import factor_tearsheet as FT
FT.factor_tearsheet(p, close, out_path="tearsheet.html")

# 组合优化(纯 numpy 解析/投影解):MinVar / MaxSharpe / 有效前沿 / Black-Litterman
from scripts.risk import optimization as OPT
w  = OPT.max_sharpe_weights(cov, mu)               # 切线组合(long_only 投影)
bl = OPT.black_litterman(cov, mkt_w, P, Q)         # ML 预测经 views_from_predictions 转 views
```

**因子 → 自动研究全链路(Round 11 已接通)**:因子库/表达式因子会自动喂给自动研究的
全部模型做训练、预测与选股产出——

```python
# 一行式:库因子 → 四道闸预筛 → 因子子集×模型搜索(ridge+listwise)→ 排行榜
lb = xsec_autoresearch.search(data, factor_source="all",      # price 8 因子 + Alpha158 库
                              zoo_max=40, prescreen=True,      # 弱/快衰减/冗余 剔,拥挤降权
                              include_listwise=True)           # 模型空间 ridge+ListNet+ListMLE
lb.attrs["prescreen_report"]                                   # 每个因子的去留原因
ens  = xsec_autoresearch.ensemble_top(data, lb, k=3)           # top-k rank-average 集成
                                                               # → ens["preds"] 最新截面=多头名单
# ML 组合回测同样直接吃库因子(任何 FactorModel:Ridge/Stacking/listwise)
res = Mdl.ml_factor_backtest(data, model=Mdl.StackingModel(),
                             panels=lb.attrs["factor_panels"], panels_mode="extend")

# 研究记忆与跨市场热启动
from scripts.research.research_memory import ResearchMemory, warm_start_search
mem = ResearchMemory("research_memory.jsonl"); mem.log({...})  # 每轮研究落盘
lb_kr = warm_start_search(data_kr, lb, top=5)                  # 美股冠军配置热启动韩股搜索
```

动态选池与反幸存者(`universe.dynamic_universe` / `anti_survivorship_pool` /
`rolling_universe`)让回测只用 as-of 当日可知的标的;`regime.stock_drift_regime` /
`drift_regime_gate` 提供逐股 drift 状态门(思路来源的数值不可信,阈值须自行敏感性测试,
见 `references/optimization_roadmap_v3.md` §2.13)。评价补件:`metrics.mae_mfe` /
`capm_decompose`、`sizing.kelly_fraction/kelly_weights`、`attribution.brinson`。

### 3. Backtest

```python
from scripts.core import backtest as bt
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
from scripts.core import optimize as opt
from scripts.strategies import MACrossover

wf = opt.walk_forward(MACrossover, df,
                      grid={"fast": [10, 20, 30], "slow": [50, 100, 150]},
                      n_splits=5, metric="sharpe",
                      cost_stress_bps=10)   # 选参用成本压力夏普,OOS 仍按真实成本报告
print(wf.oos_stats)   # out-of-sample metrics — THIS is the number to trust
print(wf.folds)       # params chosen per fold + train-vs-test gap
```

If the train metric is great but `oos_stats` collapses, the strategy is overfit.
`references/pitfalls.md` explains why and what to do.

### 5. Report

```python
from scripts.reporting import report as rpt
rpt.plot_result(result, benchmark=bench, path="equity.png")   # equity + drawdown
md = rpt.markdown_report(result, name="MA crossover", benchmark=bench)
```

**结构化 HTML 报告（复盘/分析报告的最终产出，取代 markdown）。** 不再把报告写成
markdown 再转 HTML——直接构造一个 **结构化 report dict** 交给 `scripts.reporting.html_report`，
它会渲染成一份**机构研报风、自包含单文件、可打印为 PDF** 的 HTML：买卖点为核心
（单标的=价格阶梯 止损→买区→现价→目标 + 盈亏比；组合=带内嵌 R/R 条的密集大表），
配大盘/宏观评分计、三层情绪条、策略买卖点图（▲买/▼卖 + 持仓阴影 + 日期），**红涨绿跌**（A股惯例）。
字段「给了就渲染、不给就跳过」，完整契约见 `SKILL` 同级的 **`SCHEMA.md`**。
无需第三方库（CSS/JS 已内联，中文走系统字体回退）。

```python
from scripts.reporting import html_report as H

report = {"meta": {...}, "verdict": {...}, "levels": [...], "backtest": {...}, ...}
H.save_html(report, "trading/reports/美股复盘_2026-06-09.html")   # -> 单文件 .html
html = H.render(report)                                          # 或直接拿字符串
```

> 复盘/回测类报告建议直接用 `scripts.reporting.build_research.build_research(df, name)` 产出
> `research` 块——它自动附带 **稳健性体检**(CPCV 出样本夏普分布图 + Deflated Sharpe/PBO/SPA)、
> **下跌切片**(基准最差季度/滚动窗/最深回撤段的策略对照)与**成本敏感度**(0/10/30bps 三档),
> html_report 会把这三块自动上提为独立区块渲染,无需手工搬运。

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
