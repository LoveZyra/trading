# Alpha-Forge optimization round — 2026-06-23

Two independent code reviews (core engine + ML/research/data layer) plus a doc &
triggering pass. Verdict going in: unusually solid code, **no critical look-ahead
bugs**, all 59 tests green. This round fixes the real correctness/honesty gaps the
reviews surfaced and hardens a few edge cases. After: **67 tests green** (8 new).

## Code fixes

| # | File | Issue | Fix |
|---|------|-------|-----|
| 1 | `autoresearch.py` | **HIGH.** Docstring promised "everything OOS", but `research_portfolio` / `cooptimize_factor_model` scored factor-blend & `equal_weight` trials on the **full sample** → the bandit could win by overfitting. | Search now runs on a **train slice**; the single winner is re-scored on a **held-out tail** (`best.extra['holdout_sharpe']` / co-opt `holdout`). Docstrings corrected. New `oos_frac` param. |
| 2 | `portfolio.py` | **HIGH.** Risk fns drop any name with one NaN then renormalize → ENB/VaR/correlation silently describe a *different* portfolio. | `portfolio_health` now emits a ⚠️ coverage warning naming dropped names (fires for newly-listed/illiquid names — the real case, since `pct_change` ffills mid-series gaps). |
| 3 | `metrics.py` | **MED.** `cagr` on a few-bar window annualized to ~8×10¹¹ %, poisoning `calmar` & walk-forward fold aggregates. | Windows shorter than ~a month return the plain (un-annualized) window return; normal windows unchanged. |
| 4 | `optimize.py` | **MED.** `grid_search`/`walk_forward` always *maximized* → `metric="ann_volatility"` selected the **most** volatile params. | `_LOWER_IS_BETTER` set; sort/compare direction respected (sharpe etc. still maximize). |
| 5 | `factor_lab.py` | **MED.** Causality check used a **single** cut point; a few-bar look-ahead late in the series could slip through. | Now probes **3 cut points** (50/70/90%); much harder to fool. |

Also: `ensemble_top_k` docstring now flags its re-backtest is illustrative, not a clean OOS read.

## Before / after (same synthetic universe, old snapshot vs fixed)
```
OLD: ResearchReport(best=factor {'momentum':0.63} OOS sharpe=-0.16 ...)   cagr_3bar=8.09e11
NEW: ResearchReport(best=factor {'momentum':0.63} select sharpe=0.33 | holdout(OOS) sharpe=-0.10 ...)   cagr_3bar=0.39
```
The new run **exposes that the "winner" fails out-of-sample** (holdout −0.10) — exactly the overfitting the old version hid.

## Docs / triggering
- `SKILL.md` §2c rewritten so the OOS claim matches the code (train-search + holdout; deflated-Sharpe pointer).
- `description` frontmatter: added genuinely-missing trigger coverage (portfolio risk / position sizing, 组合风险 / 仓位), a should-NOT-trigger boundary vs `equity-research-report`, and a nudge to fire even when the user doesn't say "backtest".

## Tests
`tests/test_opt_round.py` — 8 new regression tests pinning each fix. `pytest -q` → **67 passed**.

## Notes / not done in this environment
- Automated triggering loop (`skill-creator/run_loop.py`) needs nested `claude -p` auth, unavailable in this sandbox → description optimized manually (same principles). Re-runnable in your environment for quantitative trigger scores.
- Full subagent eval-viewer needs network/data; replaced with the deterministic before/after above. yfinance is network-blocked here.
- Deferred (data/policy, not code): short-borrow cost; true multi-year point-in-time fundamentals (still a snapshot approximation — now documented at call sites).

---

## Round 2 — further hardening (same day)

Four more fixes from the lower-severity review findings (all bash-applied, tested):

| # | File | Issue | Fix |
|---|------|-------|-----|
| 6 | `models.py` | ML uses static fundamentals/sentiment snapshot as **trained features** vs a forward-return label → point-in-time look-ahead that inflates IC/returns. | `ml_factor_backtest` now **warns** when fundamentals/sentiment are supplied without dated PIT panels (price factors stay causal). |
| 7 | `models.py` | Realized IC used **Pearson** corr though documented "rank-ish" → outliers dominate. | Switched to **Spearman (rank) IC**. |
| 8 | `signal_tracker.py` | `searchsorted` on a non-trading-day signal date used the **NEXT** bar as the entry price → forward-return misalignment in the feedback loop. | Snap to the last bar **≤ date** (`side="right"-1`), guard `entry≥0`. |
| 9 | `sizing.py` | Risk-parity (ERC) returned the last iterate as if converged; `sqrt` of a possibly-negative discriminant. | Added diagonal **ridge** conditioning, `sqrt(max(disc,0))`, and a **non-convergence warning**. |

Tests: `tests/test_opt_round.py` now has **11** regression tests. Full suite → **70 passed**.

### Still deferred (intentional)
`ensemble_top_k` re-backtests on the supplied series (now documented as illustrative, not a clean OOS read); VaR/CVaR keep the negative-loss sign convention (documented); `debtToEquity` /100 heuristic left as-is. None are correctness-critical.

---

## Round 3 — dogfooding the skill on a real task (存储三巨头 SK海力士/三星/美光)

Ran a full live analysis through the skill (broker data → indicators/levels/regime/backtest →
institutional HTML report). What that surfaced:

- **Already there (discoverability, not a gap):** the broker hand-off is built — `data/ibkr.from_columnar`
  parses the exact `get_price_history` parallel-array payload and `validate_ohlcv` already drops the
  `close=0` gap bars and parses ISO timestamps. Lesson: load via `source="ibkr"`, don't hand-build frames.
- **Genuinely missing → added `scripts/compare.py`:** a cross-sectional `compare_tickers({name: df})`
  convenience (trend / RSI / multi-period returns / ann-vol / distance-from-high + correlation + Meucci
  **ENB** + relative-strength rank). Thin layer over indicators/regime/portfolio. On the 3 memory names it
  printed **ENB ≈ 1.05** — quantifying "3 tickers = 1 bet" automatically.
- **Docs:** `references/data_sources.md` now documents the broker hand-off, the `compare_tickers` workflow,
  and the broker **context** tools — `get_price_snapshot` (52w/YTD/vol) and `get_company_themes`
  (sector/peers + capex/market-share/HBM4 evidence) as a fundamental-context source when yfinance/akshare
  fundamentals aren't reachable (e.g. no network). Limitation noted: broker gives no structured PE/PB/ROE.

Tests: `tests/test_compare.py` (3) — `from_columnar` on a realistic ISO+gap payload, and `compare_tickers`
shape/ranking on daily & weekly. Full suite → **73 passed**.

Deliverables from the task (in `reports/`): `存储三巨头_技术情景分析_2026-06-23.html` (weekly trend/levels)
and `存储三巨头_深度补充_日线回测产业_2026-06-23.html` (daily levels + MA4/13 backtest vs buy&hold + HBM 产业).

---

## Round 4 — report gaps found while reviewing the memory report (user feedback)

- **买卖点 %-vs-current-price (was a real gap):** `levels.trade_levels` already computes `pct` for every
  level, but `html_report` rendered the buy zone / stop / target as **absolute prices only**. Fixed:
  the levels table AND the single-name ladder now auto-append the signed %-vs-current-price for
  **buy zone, support, stop, target** (computed from each row's `price`). New JS helpers `pctStr`/`vsPrice`.
- **支撑位 (was missing from the table):** the engine returns `support1`/`support2` but there was no
  column. Added a **支撑** column (support1 + optional support2, each with %).
- **宏观 / VIX (template was fine; data wasn't fetched):** `html_report.macroPanel` already supports a VIX
  row — it just wasn't populated. The broker DOES carry VIX (CBOE, IND). Now fetched and the reports carry
  a 🌐 macro panel (today VIX 19.97, +15.6% — the risk-off spike behind the synchronized −8~−11% drop).
  Documented the broker→VIX/macro path.
- **News — honest limitation:** the template has `alerts`/`calendar`/`sentiment`, but no live news source is
  reachable in this sandbox (yfinance/akshare/web blocked; broker has no news feed). Stated plainly in the
  report disclaimer; VIX used as the available objective risk proxy.

SCHEMA.md documents `support1`/`support2` and the auto-% behavior. Test added
(`test_html_report_levels_show_pct_and_support`). Full suite → **74 passed**. Both memory reports
regenerated with buy% + support + VIX.

---

## Round 5 — live news wired in (MT Newswires) + `newsfeed` helper

- **News connectors evaluated:** recommended **MT Newswires** (dedicated real-time financial news;
  dataset `search`/`fetch` model) and **FMP** (broad data). FMP's `news`+`calendar` are **plan-gated**
  (free tier → ACCESS DENIED); MT Newswires works. Korean names have no dedicated MT items (US-tape
  coverage), so the macro/sector tape stands in for them (stated, not fabricated).
- **`scripts/newsfeed.py` (new):** `to_alerts(items)` and `to_news_group(items, source=…)` turn curated
  headlines into the report's `alerts` + a 🗞 news `groups` block in one call. Pass-through only (keeps the
  report honest; every headline keeps its date + source).
- **Report now carries live news:** the memory report's 🔴 重点关注 + 🗞 头条 show real MT Newswires items —
  Micron×Anthropic AI deal, analyst PT hikes (BofA $950→$1,500 / Needham→$1,550 / Bernstein→$1,300),
  Micron earnings this week, and the chip-rout/Iran/PCE macro tape — which directly explain the −8~−11% day.
- **Docs:** `references/data_sources.md` documents the MT Newswires `fetch` flow (symbols by ticker/ISIN,
  big-response→file extraction), the FMP plan gating, and the `newsfeed` → report mapping.

Test `test_newsfeed_to_alerts_and_group` added (renders through html_report). Full suite → **75 passed**.

---

## Round 6 — report layout/consistency (user feedback: "排版有些乱" + 右侧空白)

- **右侧大空白 = 空的 `regime` 面板槽**:`市场环境` 用 2 列网格(regime | macro);只给了 macro → 右列空。
  - **skill fix:** 单面板时 `env-grid` 自动占满整行(新增 `.env-grid.one`);不再留空白。
  - **report fix:** 实际填上 `regime`(板块趋势/超买/ENB 评分)+ `calendar`(美光财报本周/PCE/美伊和谈)。
- **公司顺序不一致**(alerts 美光在前、levels 海力士在前):新增顶层 **`symbol_order`**,稳定重排 `levels`
  与带 symbol 的 `alerts` 到统一顺序(海力士→三星→美光);`factor_rank` 作为排名不动,「宏观」类条目排最前。
- Docs: `SCHEMA.md` 增「排版/一致性」节;测试 `test_symbol_order_and_single_env_panel`。全套 **76 passed**。
