"""Technical indicators -- pure pandas/numpy, no TA-Lib dependency.

Every function takes a Series/DataFrame and returns a Series/DataFrame *aligned to
the same index*, computed using ONLY past and current data. None of these peek into
the future. The backtest engine still lags signals by one bar on top of this, which
is where execution-timing realism comes from -- but it helps that the indicators
themselves are causal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---- moving averages -------------------------------------------------------
def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def wma(s: pd.Series, n: int) -> pd.Series:
    w = np.arange(1, n + 1)
    return s.rolling(n).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)


# ---- momentum / oscillators ------------------------------------------------
def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    """Wilder's RSI. >70 overbought, <30 oversold by convention."""
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def macd(s: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Returns columns: macd, signal, hist."""
    macd_line = ema(s, fast) - ema(s, slow)
    sig = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"macd": macd_line, "signal": sig, "hist": macd_line - sig})


def momentum(s: pd.Series, n: int = 90) -> pd.Series:
    """Simple price ratio momentum over n bars (1.0 = flat)."""
    return s / s.shift(n)


def roc(s: pd.Series, n: int = 12) -> pd.Series:
    """Rate of change in percent."""
    return s.pct_change(n) * 100


def zscore(s: pd.Series, n: int = 20) -> pd.Series:
    """Rolling z-score -- the workhorse of mean-reversion signals."""
    m = s.rolling(n).mean()
    sd = s.rolling(n).std(ddof=0)
    return (s - m) / sd.replace(0, np.nan)


# ---- volatility / bands ----------------------------------------------------
def bollinger(s: pd.Series, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    """Returns columns: mid, upper, lower, pctb (%B position within band)."""
    mid = sma(s, n)
    sd = s.rolling(n).std(ddof=0)
    upper, lower = mid + k * sd, mid - k * sd
    pctb = (s - lower) / (upper - lower).replace(0, np.nan)
    return pd.DataFrame({"mid": mid, "upper": upper, "lower": lower, "pctb": pctb})


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Average True Range -- used for volatility-scaled position sizing & stops."""
    return true_range(df).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def realized_vol(s: pd.Series, n: int = 20, annualize: int = 252) -> pd.Series:
    """Annualized rolling volatility of returns."""
    return s.pct_change().rolling(n).std(ddof=0) * np.sqrt(annualize)


# ---- trend strength --------------------------------------------------------
def donchian(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Donchian channel -- classic breakout system (Turtle traders)."""
    upper = df["high"].rolling(n).max()
    lower = df["low"].rolling(n).min()
    return pd.DataFrame({"upper": upper, "lower": lower, "mid": (upper + lower) / 2})


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Average Directional Index -- trend strength, >25 = trending."""
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = true_range(df)
    atr_ = tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / atr_
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / atr_
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
