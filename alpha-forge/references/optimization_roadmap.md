# Optimization roadmap — portfolio risk, options, feedback, microstructure

This round added the dimensions that single-name analysis misses. New modules:

## Implemented this round
| Dimension | Module | Key functions |
|---|---|---|
| **组合集中度/相关性** | `portfolio.py` | `effective_num_bets` (Meucci ENB), `diversification_ratio`, `avg_correlation`, `concentration` (HHI), `sector_exposure`, `risk_contributions`, `portfolio_health` |
| **组合风险/压力** | `portfolio.py` | `portfolio_var_cvar` (1d VaR/CVaR), `beta_to`, `stress_test` (sector/market shock) |
| **期权隐含** | `data/options.py` | `expected_move`, `expected_move_from_straddle`, `iv_skew_signal` (put-call IV spread → borrow-fee proxy), `put_call_ratio_signal`, `earnings_setup` |
| **信号回看闭环** | `signal_tracker.py` | `log_signals` (daily JSONL) + `evaluate` (hit-rate, calibration vs realized fwd returns) |
| **A股微结构** | `data/microstructure.py` | `limit_for` (±10/20/5/30%), `apply_cn_rules` (涨跌停不能追/砍, 禁空, T+1 via lag) |
| **另类数据** | `data/altdata.py` | `insider_signal`, `short_interest_signal`, `northbound_signal` (北向), `trends_signal` |
| **统计 regime** | `regime.py` | `gmm_regime` (HMM-lite Gaussian-mixture states, needs sklearn) |
| **对冲建议** | `regime.py` | `hedge_suggestion` (beta × env-risk → hedge ratio + actions) |

### Daily-report usage
```python
from scripts import portfolio as PF, signal_tracker as ST
from scripts.data import options as O
# 🧩 组合体检 (holdings + watchlist)
h = PF.portfolio_health(panel_close, weights=positions_w, market_close=index_close)
#   -> effective_bets / diversification_ratio / avg_correlation / sector_exposure /
#      var_cvar / beta / stress_market_-10% / verdict（自动警告集中度）
# 期权（事件票，如财报临近）
O.earnings_setup(price=949, atm_iv=0.80, days_to_earnings=15, realized_vol=1.11)
# 信号回看：每天 log 今日信号；展示历史命中率
ST.log_signals(today_records, "trading/reports/signal_log.jsonl")
ST.evaluate("trading/reports/signal_log.jsonl", price_lookup, horizon=5)
```

## Why these matter (esp. for a concentrated AI/semi book)
A 15-name AI/semiconductor watchlist can have an **effective number of bets ≈ 1.2** —
i.e. it's really one bet. `portfolio_health` surfaces this (ENB, avg correlation, sector
exposure, beta, a −10% sector stress) so the report can warn before a single semi
selloff takes the whole book down. Options add what price can't: the earnings expected
move and the put-call skew. The signal tracker closes the loop — the system learns
whether its own calls work.

## Previously "still open" — now CLOSED (see Round 2 below)
These were the bigger lifts flagged for later. All are now implemented:
- DONE **Point-in-time fundamentals/sentiment** -> `data/pit.py` (`save_snapshot` daily +
  `asof_sentiment`/`load_pit_*`). Wired into the **US** post-close task only
  (`base="trading/pit_store"`), so the dated PIT archive accumulates automatically;
  A-share intentionally does NOT store PIT. (True multi-year history still needs paid
  PIT data or months of free accumulation — a *data* limit, not a code gap.)
- DONE **Live execution** (smart routing, TWAP/VWAP) -> `execution.py`. *Borrow cost for
  shorts is the one deliberately deferred item.*
- DONE **HTML dashboard artifact** -> cowork artifact `portfolio-dashboard` (holdings + PnL +
  concentration + live watchlist quotes, refreshes from the broker connector each open).
- DONE **PnL attribution** -> `attribution.py` (`pnl_by_symbol`, `attribute_by_sector`,
  `factor_attribution`).
- DONE **Deep sequence / TS-foundation models** -> `models.py` (`MLPModel` +
  `load_external_scores` to ingest an externally-trained model's scores as a factor).

## Only remaining items (both are data/policy limits, not code gaps)
- **Short-borrow / 融券 cost** — deliberately deferred per user.
- **True multi-year point-in-time data** — code path ready (`pit.py`); needs paid data
  or months of free `save_snapshot` accumulation to become meaningful.

## Round 2 (this batch) — execution, attribution, PIT, deep-model bridge
| Item | Module | Notes |
|---|---|---|
| **Point-in-time 历史回测(免费起步)** | `data/pit.py` | `save_snapshot` daily → accumulates a dated archive; `asof_sentiment`/`load_pit_*` read the value known AS OF each date (honest PIT). Wire `save_snapshot` into the daily tasks; backtests get meaningful once months accumulate. |
| **智能执行计划** | `execution.py` | `order_plan` → shares, limit-vs-market, TWAP/VWAP child orders, participation cap, multi-day split. Plan only — submit separately with confirmation. |
| **PnL 归因** | `attribution.py` | `pnl_by_symbol` (from broker positions/trades), `attribute_by_sector`, `factor_attribution` (regress port returns on factor returns → betas + R²). |
| **深度/时序模型桥接** | `models.py` | `MLPModel` (sklearn neural net, in-sandbox) + `load_external_scores` (ingest a {symbol:score} from an externally-trained Stockformer/LSTM/TS-foundation model as a factor — heavy training stays off the sandbox). |
| **HTML 实时看板** | cowork artifact | Holdings/PnL/concentration page that refreshes from the broker connector each open. |

Only **真·历史 PIT 数据** still costs money (or months of free accumulation); everything
else here needs no new data source. Short-borrow/融券 cost is the one deferred item.
