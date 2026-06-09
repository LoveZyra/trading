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
