# Strategy design notes

How the built-in templates think, when each tends to work, and how to build your own.
All templates produce a **signal**: a target position per bar in `[-1, 1]`
(0 = flat, 1 = fully long, -1 = fully short). The backtest engine owns lag, costs
and sizing — strategies stay pure.

## The two families (and why you want both)

**Trend / momentum** bets that moves persist. Low win rate, big average winners,
long flat/whipsaw stretches in range-bound markets. Shines in sustained trends.

**Mean reversion** bets that stretched prices snap back. High win rate, small
average winners, and a fat left tail when "stretched" becomes "trending". Trades
often, so it's cost-sensitive.

They're complementary — strong where the other is weak. Testing both on the same
asset tells you what regime that asset has been in.

## Built-in templates

### Trend (`strategies/trend.py`)
- **MACrossover(fast, slow, ma, allow_short)** — long when fast MA > slow MA. The
  canonical system. Pair with `with_trend_filter(...)` (ADX) to mute whipsaws.
- **Breakout(entry, exit, allow_short)** — Donchian channel; enter on an n-bar high,
  exit on the exit-bar low. Turtle-style; captures the fat tail of big trends.
- **TimeSeriesMomentum(lookback, allow_short)** — long if trailing return > 0. The
  robust TSMOM factor; simple and travels well across markets.
- **MACDTrend(fast, slow, signal)** — long while the MACD histogram is positive.

### Mean reversion (`strategies/mean_reversion.py`)
- **ZScoreReversion(lookback, entry, exit)** — fade rolling z-score extremes;
  enter at |z|≥entry, exit toward the mean at |z|≤exit.
- **BollingerReversion(n, k)** — buy the lower band, exit at the mid band.
- **RSIReversion(n, oversold, overbought, exit_level)** — buy oversold, exit on
  recovery.
- **PairsTrading(partner_close, lookback, entry, exit)** — trade the z-score of a
  rolling-hedge spread between two cointegrated names. Run a cointegration test
  (e.g. `statsmodels.tsa.stattools.coint`) before trusting any pair; a spurious pair
  will quietly diverge forever.

### Multi-factor (`strategies/multi_factor.py`)
`multi_factor_signal(data, factor_weights, rebalance, top, bottom, long_short)`
ranks a *universe* each rebalance date by a z-scored blend of factors (momentum +
low-vol out of the box), longs the top bucket, optionally shorts the bottom, equal-
weights within a bucket, and holds until the next rebalance. Feed the resulting
weights panel to `backtest.backtest_portfolio`. Add your own factors by extending the
`factors` dict (value, quality, size…) — supply the fundamental data yourself.

## Sensible starting parameters
- MA crossover: fast 20 / slow 50 (or 50/200 for slow trend).
- Donchian breakout: entry 20–55 / exit 10–20.
- TSMOM lookback: 60–180 bars.
- Z-score reversion: lookback 20, entry 1.5–2.0, exit 0.5.
- Bollinger: n 20, k 2.0.
- RSI: n 14, oversold 30, exit 50.
- Multi-factor: monthly rebalance (`"ME"`), top 20%.

Keep ranges economically motivated — see `pitfalls.md` on overfitting.

## Building your own

```python
from scripts.strategies.base import Strategy
from scripts.core import indicators as ind

class MyStrategy(Strategy):
    name = "my_strategy"
    def __init__(self, lookback=20):
        self.lookback = lookback
    def generate_signal(self, df):
        # MUST be causal: only use past/current bars. Return a Series in [-1, 1].
        z = ind.zscore(df["close"], self.lookback)
        sig = (-z.clip(-2, 2) / 2)           # scale into [-1, 1], fade extremes
        return sig.fillna(0.0)
```

Then: `backtest(df, MyStrategy().generate_signal(df))`. Register it in
`strategies/__init__.py:REGISTRY` to use it from the CLI.

## Position sizing beyond ±1
Signals can exceed ±1 for leverage, or scale continuously (e.g. proportional to
signal strength or inversely to volatility — divide by `indicators.realized_vol` for
vol-targeting). The engine multiplies position by bar return, so a 0.5 means half
exposure. Volatility-scaled sizing usually improves risk-adjusted returns and keeps
drawdowns more uniform across assets.
