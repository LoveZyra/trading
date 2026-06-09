"""Strategy templates. All produce a target-position signal; the engine handles the rest."""
from .base import Strategy
from .trend import (MACrossover, Breakout, TimeSeriesMomentum, MACDTrend,
                    with_trend_filter)
from .mean_reversion import (ZScoreReversion, BollingerReversion, RSIReversion,
                             PairsTrading, pair_spread)
from . import multi_factor

# Registry so the CLI / Claude can pick a strategy by name string.
REGISTRY = {
    "ma_crossover": MACrossover,
    "breakout": Breakout,
    "ts_momentum": TimeSeriesMomentum,
    "macd_trend": MACDTrend,
    "zscore_reversion": ZScoreReversion,
    "bollinger_reversion": BollingerReversion,
    "rsi_reversion": RSIReversion,
}

__all__ = ["Strategy", "MACrossover", "Breakout", "TimeSeriesMomentum", "MACDTrend",
           "ZScoreReversion", "BollingerReversion", "RSIReversion", "PairsTrading",
           "with_trend_filter", "pair_spread", "multi_factor", "REGISTRY"]
