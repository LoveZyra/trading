# Fundamentals & news — data, sentiment, and using them as factors

This covers the fundamental-metrics layer (`data/fundamentals.py`), the news layer
(`data/news.py`), the bilingual sentiment scorer (`data/sentiment.py`), and how all
three feed the multi-factor model.

## Table of contents
1. Fundamentals — fields & sources
2. News — sources & the Web-search hand-off
3. Sentiment scoring
4. Using them as factors (value / quality / growth / sentiment)
5. The point-in-time caveat (important)

---

## 1. Fundamentals

`data.fundamentals.load(symbol, source=...)` returns a flat dict of **canonical
fields** (same names regardless of source): `market_cap, pe, pb, ps,
dividend_yield, roe, roa, gross_margin, net_margin, debt_to_equity,
revenue_growth, earnings_growth`. Missing values are `None`, never faked.

```python
from scripts.data import fundamentals as F
F.load("AAPL", source="yfinance")              # US & global
F.load("600519", source="akshare", market="cn")   # A-share (东方财富 + 财务指标)
panel = F.load_panel(["AAPL","MSFT","NVDA"], source="yfinance")  # index=symbol
```

`FACTOR_DIRECTION` encodes which way each metric points (PE/PB/PS/debt: lower is
better → negated; ROE/margins/growth: higher is better). The factor layer uses it so
every score reads "higher = more attractive".

**Broker / Web hand-off:** if you (Claude) pulled fundamentals from the broker MCP or
a Web search, save a dict with canonical keys to JSON and use `source="json"`.

---

## 2. News

`data.news.fetch(symbol, source=...)` returns a uniform list of
`{title, publisher, time, url, summary}`.

```python
from scripts.data import news
news.fetch("AAPL", source="yfinance")               # Yahoo headlines
news.fetch("600519", source="akshare")              # 东方财富 stock_news_em
```

### Web-search hand-off (the important one)
`WebSearch` and the broker news tools can only be called by **Claude**, not by the
sandbox script. So for the broadest, freshest coverage:

1. Claude runs `WebSearch` (e.g. `"NVDA news June 2026 earnings guidance"`) and/or
   the broker news tool.
2. Claude saves the results to a JSON file — a list of `{title, source, url, time,
   summary}` (key aliases like `headline`/`新闻标题`/`publishedAt` are auto-mapped).
3. The script reads it: `news.fetch("/path/to/news.json", source="json")`.

This is the same hand-off pattern as price/fundamentals — keeps everything
reproducible and lets the sandbox do the scoring.

---

## 3. Sentiment scoring

`data.sentiment.score(text) -> [-1, 1]` uses a curated **bilingual finance lexicon**
(English in the spirit of Loughran-McDonald + a Chinese 金融情绪 word list) with
simple negation handling. It's deterministic, fast and auditable — the right tool for
scoring thousands of headlines across a backtest.

```python
from scripts.data import sentiment as S
S.score("Apple beats earnings, stock surges to record high")   # +0.88
S.score("某公司爆雷 业绩预减 跌停 遭立案调查")                    # -0.96

scored, agg = news.fetch_with_sentiment("AAPL", source="yfinance")
# scored: per-headline DataFrame with a 'sentiment' column
# agg: {'mean_sentiment', 'n', 'n_pos', 'n_neg', 'max', 'min'}
```

For a nuanced read on a *handful* of fresh headlines, Claude can add its own judgment
on top — but the number that goes into the backtest factor is the lexicon score, so
it stays reproducible.

---

## 4. Using them as factors

`multi_factor.multi_factor_signal` now accepts two extra inputs:

```python
from scripts.data import fundamentals as F, news, sentiment as S
from scripts.strategies import multi_factor as mf
from scripts import backtest as bt

universe = ["AAPL","MSFT","NVDA","AMD","GOOGL"]
prices = {s: load(s, source="yfinance", period="3y") for s in universe}   # dict of OHLCV
funds  = F.load_panel(universe, source="yfinance")                        # index=symbol
senti  = {s: S.aggregate_sentiment(news.fetch(s, source="yfinance"))["mean_sentiment"]
          for s in universe}

weights = mf.multi_factor_signal(
    prices,
    factor_weights={"momentum":0.3,"low_vol":0.2,"value":0.2,"quality":0.2,"sentiment":0.1},
    rebalance="ME", top=0.4,
    fundamentals_panel=funds,
    sentiment_by_symbol=senti,
)
result = bt.backtest_portfolio(mf.build_panel(prices,"close"), weights)
print(result.stats)
```

Available factor keys: price → `momentum`, `low_vol`; fundamental → `value`,
`quality`, `growth`; news → `sentiment`. Supply only the data you have; unsupplied
factors are dropped and weights renormalize.

---

## 5. The point-in-time caveat (read this)

The price factors (momentum, low-vol) are genuinely time-varying — recomputed each
bar from history. But a **single fundamentals/news snapshot is *current*** data. When
used in a multi-year backtest it's applied as a *static tilt* across all dates, which
means the backtest assumes today's PE/ROE/sentiment held throughout. That's a
**look-ahead approximation** — fine for a present-day screen or a tilt study, but it
will overstate an edge if you read it as a true historical backtest.

To do it honestly over history you need **point-in-time** fundamentals (the value
known *on each past date*, with reporting lag) and **dated** news sentiment. The code
is structured so you can extend `multi_factor_signal` to consume dated snapshots:
build a `{date: fundamentals_panel}` / `{date: sentiment}` history and index it per
rebalance date instead of broadcasting one snapshot. Until then, treat fundamental/
news factors as a **current-day screen**, and lean on the price factors + walk-forward
for historical evidence. See `pitfalls.md` on look-ahead bias.

---

## 6. Multi-level sentiment — stock + sector/peers + macro (don't stop at company news)

A common mistake: scoring only a company's OWN headlines. But a stock's news-driven
move is rarely idiosyncratic — the **sector cycle** (the whole semi complex moving on
an NVDA print) and **macro** (Fed/CPI/rates, risk-on/off) usually dominate. Scoring
only single-name news misses the main drivers.

The skill now supports a 3-layer composite:

```python
from scripts.data import sentiment as S, sectors as SEC, news

# 1) company news (as before)
stock = news.fetch("MRVL", source="yfinance")              # or Web-search JSON
# 2) sector / peer news — search the THEME, not just the ticker
#    (e.g. "semiconductor AI chip demand June 2026"), save JSON, then:
sector = news.fetch("/path/semi_news.json", source="json")
# 3) macro / market news — "Fed CPI rates stock market June 2026", save JSON:
macro  = news.fetch("/path/macro_news.json", source="json")

blend = S.blend_news_layers(stock, sector, macro)          # default 0.5/0.3/0.2
#   -> {"composite": .., "stock": .., "sector": .., "market": .., "n_*": ..}
sentiment_by_symbol["MRVL"] = blend["composite"]           # feed the factor model
```

- `sectors.sector_of(sym)` / `peers_of(sym, universe)` / `group_by_sector(universe)`
  map tickers to a theme so you fetch sector news ONCE per group, not per name.
- `composite_sentiment(stock, sector, market, w_stock=, w_sector=, w_market=)` lets you
  reweight: ETFs / cyclicals -> more macro weight; single-name event stocks -> more
  stock weight.
- Workflow tip for daily reports: pull macro news once, sector news once per sector
  group, company news per name; blend. That's a handful of searches covering the whole
  watchlist, and the composite reflects what actually moves the stock.

The hand-off rule is unchanged: WebSearch/broker tools are Claude-only, so Claude
searches each layer, saves JSON, and the script scores + blends.

---

## 7. Macro layer = real indicators, not just headlines (`data/macro.py`)

The 'market' layer of composite sentiment should reflect HARD macro data, not only
news text. `data.macro.macro_score(...)` turns indicators into one risk-on/off score
in [-1, 1] (negative = risk-off):

| Input | How to fetch | Effect |
|---|---|---|
| **VIX** (恐慌指数) | broker `search_contracts("VIX", security_type="IND")` + `get_price_history`, or yfinance `^VIX` | high/spiking -> risk-off |
| **10y / 2y Treasury yields** (美债) | yfinance `^TNX`/`^FVX`(÷10), or ETF TLT/IEF/SHY via broker | rising yields -> headwind; **10y-2y inverted -> recession risk** |
| **CPI / PPI / NFP / unemployment** | Web-search the release ("US CPI June 2026 actual vs consensus") -> pass numbers to `econ_surprise` | hot inflation -> risk-off; strong jobs -> hawkish tilt |
| **War / geopolitics** (战争) | score geopolitics headlines with `sentiment` -> pass as `geo_sentiment` | escalation -> risk-off |
| **通胀/通缩** | read off CPI/PPI trend & surprise | hot=通胀/risk-off, soft/negative=通缩担忧 |

```python
from scripts.data import macro as MAC, sentiment as S
m = MAC.macro_score(
    vix=vix_close,                 # pd.Series of VIX closes
    y10=y10_close, y2=y2_close,    # yield series (percent)
    releases=[{"name":"cpi","actual":3.6,"consensus":3.1},
              {"name":"nfp","actual":250,"consensus":180}],
    geo_sentiment=S.aggregate_sentiment(geo_headlines)["mean_sentiment"],
)
# m -> {"score": -0.60, "vix":-1.0, "rates":-0.39, "curve":-0.38, "econ":-0.68,
#       "geo":-0.23, "regime":"risk-off/避险"}

# feed it as the MARKET layer of composite sentiment:
sentiment_by_symbol[sym] = S.composite_sentiment(stock=own, sector=sec, market=m["score"])
```

`INDICATOR_SIGN` holds the default equity-risk sign of an upside surprise per release
(CPI/PPI/PCE: -1 risk-off; NFP: mild -0.3 hawkish; GDP/retail/ISM: + risk-on). **These
are regime-dependent** (a hot NFP is risk-on in a soft-landing regime, risk-off under a
hawkish Fed) — override `sign` per release when the regime shifts. Treat macro as a
risk overlay, not a precise forecast.

---

## 8. Recency (时效性) + forward econ calendar (consider CPI/PPI/NFP BEFORE they print)

**Time-decay**: markets price the freshest info, so weight news by recency.
`sentiment.time_weighted_sentiment(items, halflife_hours=48, now=None)` weights each
headline by `0.5 ** (age/halflife)` using its timestamp — a 2-hour-old miss outweighs a
month-old beat. Use it instead of `aggregate_sentiment` for daily reports (default
half-life 48h; shorten to 12–24h for fast-moving names/event days).

**Forward econ calendar**: don't only react to released data — anticipate scheduled
prints. `macro.pre_event_risk(upcoming, today, window_days=4)` takes a Web-searched
calendar `[{"name":"cpi","date":"2026-06-11"}, {"name":"fomc","date":"2026-06-17"}]`
and returns a small risk-off nudge that grows as a HIGH-impact release (CPI/PPI/NFP/
FOMC) nears (markets de-risk into big prints), plus the imminent-event list. It's wired
into `macro_score(..., upcoming=[...], today=...)` as an `event` component, and the
report should carry a "📅 事件前瞻" line (e.g. "CPI 明天公布，盘前注意；FOMC 在 6 天后").

**Event propagation**: a big single-name event spills to peers.
`sectors.propagate_event("NVDA", universe, impact=0.8)` returns `{peer: spillover}` for
same-sector names (0.5× transmission) so an NVDA beat flags the whole semi complex.
`sectors.sector_score_map(universe, {sector: score})` broadcasts a sector score to each
member — use it as the `sector` layer of composite_sentiment, or as its own factor.

Daily-report workflow: (1) Web-search the econ calendar for the next ~1 week →
`pre_event_risk`; (2) score each news layer with `time_weighted_sentiment`; (3) feed
the macro `event` component + recency-weighted layers into the composite. Now the
report both reacts to fresh news and warns ahead of CPI/PPI/NFP.

---

## 9. Broad-market index environment per market (大盘β — don't fight the tape)

The macro layer (§7) is US-centric. But a stock trades with ITS OWN market: a Korean
name with the KOSPI, an A-share with the CSI 300, a Japanese name with the Nikkei.
`data/market.py` adds a per-market "大盘" regime so the report reads each stock against
its home index, not only US macro.

| Market | Index (fetch) |
|---|---|
| US | ^GSPC / ^NDX (yfinance) · SPX/NDX (broker IND) · SPY/QQQ |
| 韩股 KR | ^KS11 (KOSPI) / ^KQ11 (KOSDAQ) |
| 日股 JP | ^N225 (Nikkei) / ^TPX (TOPIX) |
| A股 CN | 000300 (沪深300) / 000001 (上证) via akshare index |
| 港股 HK | ^HSI (恒指) |

```python
from scripts.data import market as MK, macro as MAC, sentiment as S
idx = load("^NDX"/SPX..., source=...)          # the home-market index OHLCV
reg = MK.index_regime(idx["close"])            # {"score","above_ma","mom_3m","ann_vol","regime"}
br  = MK.market_breadth(universe_data, ma=50)  # % of names above MA50 -> breadth score
ov  = MK.market_overlay(index_close=idx["close"], breadth_score=br["score"],
                        global_macro=MAC.macro_score(...)["score"])
# use the market overlay as the MARKET layer of composite sentiment:
sentiment_by_symbol[sym] = S.composite_sentiment(stock=own, sector=sec, market=ov["score"])
```

- `index_regime` blends trend (price vs 200MA), 3-month momentum, and vol band into a
  [-1,1] score (+ = 大盘多头/risk-on). `market_breadth` is a participation proxy
  (>0.6 healthy, <0.4 narrow). `market_overlay` blends index + breadth + global macro.
- **Routing caveat**: `market_of(symbol)` is best-effort and CANNOT tell a Korean
  6-digit code (005930) from an A-share 6-digit (600519). In the daily tasks, route by
  the **watchlist file** the ticker came from (watchlist_us → US/^NDX, watchlist_kr →
  KR/^KS11, watchlist_jp → JP/^N225, watchlist_cn → CN/沪深300). Show a "📊 大盘环境"
  line per market in the report.

Two-tier picture: **global macro (VIX/US rates/US data, §7)** sets the world risk tone;
the **home-market index (§9)** sets the local tape. A US stock keys off both; an A-share
keys off CSI 300 first, global macro second.
