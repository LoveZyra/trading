"""Default parameter grids per rule-strategy family -- single source of truth.

Used by autoresearch (as its sampling space) and run_backtest --walk-forward (as
its default grid). Previously duplicated in both files; editing one and missing
the other silently desynced research from the CLI.
"""

PARAM_GRIDS = {
    "ma_crossover": {"fast": [10, 20, 30], "slow": [50, 100, 150]},
    "breakout": {"entry": [20, 40, 55], "exit": [10, 20]},
    "ts_momentum": {"lookback": [60, 90, 120, 180]},
    "macd_trend": {"fast": [8, 12], "slow": [21, 26], "signal": [9]},
    "zscore_reversion": {"lookback": [10, 20, 30], "entry": [1.0, 1.5, 2.0]},
    "bollinger_reversion": {"n": [10, 20, 30], "k": [1.5, 2.0, 2.5]},
    "rsi_reversion": {"n": [7, 14, 21], "oversold": [20, 30]},
}
