# Data sources & broker hand-off

Every loader returns the same canonical OHLCV frame, so the rest of the skill is
source-agnostic. This file covers how to actually pull the data from each source —
especially the IBKR-style broker MCP, which needs a hand-off because the Python
sandbox can't call MCP tools directly.

## Table of contents
1. Free libraries (yfinance / akshare / pykrx)
2. IBKR-style broker MCP: the hand-off pattern
3. Symbol / market cheat-sheet
4. Placing orders (live trading)

---

## 1. Free libraries

These run inside the sandbox; the user just needs them installed and network access.

```python
from scripts.data.loader import load

load("AAPL", source="yfinance", start="2020-01-01", end="2024-12-31")
load("AAPL", source="yfinance", period="5y", interval="1d")   # period instead of dates
load("600519", source="akshare", market="cn", adjust="qfq")   # 贵州茅台, forward-adjusted
load("00700", source="akshare", market="hk")                  # Tencent
load("005930", source="pykrx", start="2018-01-01")            # Samsung
```

Results are cached as parquet under `.cache/`. Delete that folder to force a refresh.
`adjust="qfq"` (前复权) is recommended for A-shares so splits/dividends don't create
fake gaps; yfinance's `auto_adjust=True` does the equivalent and is the default.

---

## 2. IBKR-style broker MCP: the hand-off pattern

The broker connector exposes tools like `search_contracts`, `get_price_history`,
`get_price_snapshot`, `get_account_*` and `create_order_instruction`. **Only Claude
can call these** — a script in the sandbox cannot. So data flows like this:

```
Claude calls MCP  ->  saves JSON to a file  ->  loader reads file  ->  canonical OHLCV
```

### Step A — find the contract id

Call `search_contracts(query="AAPL", security_type="STK")`. Pick the row with the
right `exchange` / `country_code` and note `underlying_contract_id`. Example:
`AAPL` on `NASDAQ`, `country_code="US"` → contract id `265598`.

### Step B — pull history

Call `get_price_history` with the contract id. Provide **either** `period` **or**
`step_count`, never both, plus a `step` (bar size):

```
get_price_history(contract_id=265598, exchange="NASDAQ", security_type="STK",
                  step="ONE_DAY", period="FIVE_YEARS", outside_rth=false)
```

Valid `period`: ONE_DAY, TWO_DAYS, THREE_DAYS, ONE_WEEK, TWO_WEEKS, ONE_MONTH,
THREE_MONTHS, SIX_MONTHS, ONE_YEAR, TWO_YEARS, FIVE_YEARS.
Valid `step`: THIRTY_SECS, ONE_MIN, … ONE_HOUR, FOUR_HOURS, ONE_DAY, ONE_WEEK, ONE_MONTH.

The response is **columnar**: parallel arrays `time / open / high / low / close /
volume`, plus metadata (`delayed` is the quote delay in seconds — often 900 = 15 min).

### Step C — get it into a DataFrame

Large pulls (e.g. 5 years daily ≈ 1250 bars) exceed the inline tool-result limit and
are **auto-saved to a `.txt`/`.json` file** by the harness; the error message tells
you the path. Either way:

```python
# (a) the result was saved to a file:
from scripts.data.loader import load
df = load("/path/to/saved_result.txt", source="ibkr")

# (b) you have the JSON payload in hand (small pulls):
from scripts.data.ibkr import from_mcp_payload
df = from_mcp_payload(payload_dict)
```

Then proceed exactly as with free data — indicators, backtest, etc. don't know or
care that it came from the broker.

> Tip: for backtests prefer free, split-adjusted history (longer, cleaner, free).
> Use the broker for recent/live/delayed bars and for anything you intend to trade.

---

## 3. Symbol / market cheat-sheet

| Market        | yfinance suffix | akshare                         | pykrx |
|---------------|-----------------|---------------------------------|-------|
| US            | `AAPL`          | —                               | —     |
| Shanghai A    | `600519.SS`     | `600519` market="cn"            | —     |
| Shenzhen A    | `000001.SZ`     | `000001` market="cn"            | —     |
| Hong Kong     | `0700.HK`       | `00700` market="hk"             | —     |
| Korea (KOSPI) | `005930.KS`     | —                               | `005930` |

For the broker MCP, always resolve the symbol to a `contract_id` via
`search_contracts` first — the same ticker exists on many exchanges.

---

## 4. Placing orders (live trading)

Turning a backtested signal into a real order:

1. `strat.latest_signal(df)` → target position in [-1, 1] from the latest bar.
2. Check real state: `get_account_balances`, `get_account_positions`.
3. Size the order against actual buying power — never hard-code share counts.
4. `create_order_instruction(...)` to stage it; review before submitting.
5. Confirm fills with `get_account_orders` / `get_account_trades`.

Guardrails that matter: confirm the exact contract, respect the quote delay
(`delayed`), paper-trade first, and **never auto-submit an order without the user's
explicit confirmation**. A backtest edge is a hypothesis, not a guarantee.

---

## 多标的对比 + broker 上下文工具(2026-06 新增)

### 经纪商行情 hand-off(已内置,优先用它,别手搓 DataFrame)
`get_price_history` 返回并行数组 `{time, open, high, low, close, volume}`(`time` 为 ISO-8601;
偶有 `close=0` 的缺口 bar)。**不要**手动拼 DataFrame——直接交给适配器,它会解析 ISO 时间、
去重排序、剔除 `close<=0` 的缺口、修复微小 OHLC 不一致:

```python
from scripts.data import ibkr
df = ibkr.from_columnar(payload)                 # payload = get_price_history 的返回 dict
# 或先把 payload 存成 .json,再:
from scripts.data.loader import load
df = load("trading/data/skh_weekly.json", source="ibkr")
```

### 多标的横截面对比(`scripts/compare.py`)
单名分析之外,"A vs B vs C 谁更强/多像/什么姿态"用这个便捷层(复用 indicators/regime/portfolio):

```python
from scripts import compare
universe = {"SK海力士": df_skh, "三星": df_ss, "美光": df_mu}   # canonical OHLCV
cmp = compare.compare_tickers(universe, bars_per_year=52)        # 周线传 52,日线传 252
cmp["table"]          # 每名:last/rsi/各期收益/年化波动/距高/趋势
cmp["correlation"]    # 两两相关(多像)
cmp["effective_bets"] # Meucci ENB(≈1 = 其实是一个押注)
cmp["rs_rank"]        # 按近一年收益的相对强弱排名
```

### 经纪商「上下文」工具(不止价格)
当 yfinance/akshare 基本面取不到(如沙箱无外网)时,broker MCP 仍能给出有用上下文:
- **`get_price_snapshot`**(字段 `misc_statistics`/`year_to_date_change`/`historical_vol`):
  52 周高/低、年初至今涨幅、近 30 日年化波动 → 一行「当前定位」。
- **`get_company_themes`**:板块 / 同业 / 主题,且每个同业附 `evidence`(含营收、资本开支、
  市占、HBM4 进度等 2025/2026 口径事实)→ 可作**基本面/产业 context**(虽非 PE/ROE 结构化字段)。
  例:存储板块一次性带出 SK海力士(HBM4 率先量产)、三星(DRAM 34% 份额)、美光、以及
  Lam/AMAT/Rambus/SanDisk 等资本开支受益链。
> 局限:broker 不提供结构化 PE/PB/ROE;真·实时基本面仍需付费数据源或可达的 yfinance/akshare。

---

## 实时新闻接入(news connector → 报告 alerts / 🗞 新闻栏)(2026-06 新增)

新闻数据来自只有 Claude 能调的 connector;fetch+筛选是 Claude 的活,`scripts/newsfeed.py` 负责把
筛好的头条变成报告的 `alerts` 与 🗞 `groups` 块(一步到位,且只透传你筛的文本、逐条带日期+来源,保持诚实)。

**MT Newswires**(数据集模型):`search("news")` 找到数据集(`mt_newswires_global` / `_north_america`)→
`fetch(dataset_name=..., product="EDGE", symbols="MU,KR7000660001,...", from_date, to_date, limit)`。
- symbols 支持 ticker / ISIN / FIGI / LEI(韩股用 ISIN,如 SK海力士 `KR7000660001`、三星 `KR7005930003`)。
- 响应可能很大(带全文 body)而被存到文件 → 用 jq/python 只取 `headline/date/key/isPrimary/metadata`,丢弃 body。
- 覆盖偏美股/北美:`MU` 头条丰富(分析师目标价、Anthropic 合作、财报前瞻、芯片板块 tape);韩股个股头条常缺。

**FMP**:`news`(general/stock/press)与 `calendar`(财报日)需 **Starter+ 付费档**;免费档会 ACCESS DENIED。

**接进报告:**
```python
from scripts import newsfeed as NF
items = [ {"date":"2026-06-23","headline":"...","symbol":"宏观","name":"芯片板块","level":"high","detail":"...","action":"..."},
          {"date":"2026-06-22","headline":"Micron signs AI deal with Anthropic"} ]   # 你从 feed 筛出来的
report["alerts"] = NF.to_alerts(items) + report.get("alerts", [])
report.setdefault("groups", []).insert(0, NF.to_news_group(items, source="MT Newswires"))
```
> 韩股缺个股头条时,用「宏观/板块 tape」条目代替(注明覆盖局限),不要编造个股新闻。

---

## 期权隐含 · 财报预期波动(broker IV → options.earnings_setup)(2026-06 补充)

FMP 免费档拿不到行情/财报日历(quote/news/calendar 均需付费档),但 broker 能给期权隐含波动:
```python
# Claude 调 broker:get_price_snapshot(contract, market_data_names=["implied_vol_underlying","historical_vol","last"])
from scripts.data import options as O
O.earnings_setup(price=1098.0, atm_iv=1.066, days_to_earnings=3, realized_vol=1.035)
# -> {'move_pct': 9.66, 'low': 991.9, 'high': 1204.1, 'iv_rv_ratio': 1.03, 'note': '期权定价财报±9.66%…'}
```
把结果作为一个 `groups`/`technical` 块接进报告——尤其在财报临近时,给出市场定价的双向跳空区间。
> 即:免费 news 连接器(MT Newswires)给"何时财报/催化",broker 期权 IV 给"财报会跳多大"。

---

## 事件日期(财报 / 宏观)——免费做法:直接 WebSearch(无需付费连接器)

报告 `calendar` 的具体日期,**最省事且免费**的来源是 `WebSearch`(一次性查),不必连付费数据商:
- 财报日:搜 "Micron earnings date fiscal Q3 2026" → "6/24 盘后";
- 宏观:搜 "US PCE release date June 2026" → "6/25"(BEA Personal Income & Outlays)。
把查到的日期(+按今天算 `in_days` 倒计时)填进 `report["calendar"]` 即可。

各路径对比(本项目实测):
- **WebSearch**:免费、覆盖广、适合一次性查具体日期(✅ 推荐默认)。
- **MT Newswires**:已连、免费,但周前瞻只到「本周」粒度(无逐日)。
- **FMP `calendar`/`economics`、Quartr `list_events`**:能自动化且结构化,但**均需付费订阅**(免费档 ACCESS DENIED / Pro required)。
> 结论:要"具体某天"先用 WebSearch;要"每天自动刷新整张日历"再上付费连接器 + 定时任务。
