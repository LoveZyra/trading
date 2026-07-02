"""Factor expression engine — a safe DSL for price/volume alpha formulas.

Why a DSL instead of plain Python functions? Three reasons:
1. **Auditability**: a factor stored as a string ("-1 * ts_corr(rank(open), rank(volume), 10)")
   can be logged, diffed, and re-run years later; a closure cannot.
2. **Safety**: factor libraries (Alpha101/Alpha158) are data, not code. We parse with
   `ast` and evaluate against a *whitelist* of operators — never `eval()` — so a factor
   string can't touch the filesystem, network, or anything outside the operator table.
3. **Causality by construction**: every time-series operator here is built on
   shift(+n)/rolling/ewm, which only look BACKWARD. If a formula parses, it cannot
   look ahead. `factor_lab.validate_factor` still mechanically verifies this.

Two input shapes, one expression language:
- single-symbol OHLCV DataFrame  -> returns a pd.Series (dates) — for factor_lab.
- {symbol: OHLCV} dict (a panel) -> returns a wide DataFrame (date x symbol) — for
  xsec_eval. Cross-sectional operators (rank/zscore/scale/...) act per-row and are
  therefore only meaningful (and only allowed) on panel input.

Variables: open, high, low, close, volume, vwap, returns.
vwap falls back to (high+low+close)/3 when the column is absent (typical for free
daily data); returns = close.pct_change().
"""
from __future__ import annotations

import ast

import numpy as np
import pandas as pd

_EPS = 1e-12

# --------------------------------------------------------------------------- #
# input normalization
# --------------------------------------------------------------------------- #

_FIELDS = ("open", "high", "low", "close", "volume")


def _norm_df(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case column lookup so 'Close'/'close' both work."""
    out = df.copy()
    out.columns = [str(c).lower() for c in out.columns]
    if "close" not in out.columns:
        raise ValueError("OHLCV frame must at least have a 'close' column")
    return out


def _single_env(df: pd.DataFrame) -> dict:
    d = _norm_df(df)
    env = {}
    for f in _FIELDS:
        if f in d.columns:
            env[f] = d[f].astype(float)
    close = env["close"]
    # graceful degradation: synthesize missing price fields from close so that
    # library formulas still run on close-only data (coverage over strictness).
    for f in ("open", "high", "low"):
        env.setdefault(f, close.copy())
    env.setdefault("volume", pd.Series(np.nan, index=close.index))
    if "vwap" in d.columns:
        env["vwap"] = d["vwap"].astype(float)
    else:
        env["vwap"] = (env["high"] + env["low"] + env["close"]) / 3.0
    env["returns"] = close.pct_change()
    return env


def _panel_env(data: dict) -> dict:
    envs = {sym: _single_env(df) for sym, df in data.items()}
    names = list(envs)
    env = {}
    for var in ("open", "high", "low", "close", "volume", "vwap", "returns"):
        env[var] = pd.DataFrame({s: envs[s][var] for s in names}).sort_index()
    return env


# --------------------------------------------------------------------------- #
# operator implementations (ALL time-series ops are causal: shift/rolling/ewm)
# --------------------------------------------------------------------------- #

def _roll(x, n):
    return x.rolling(int(n), min_periods=int(n))


def _apply(x, n, fn):
    """rolling.apply(raw=True) helper — per-column for wide frames."""
    return _roll(x, n).apply(fn, raw=True)


def _delay(x, n):     return x.shift(int(n))
def _delta(x, n):     return x.diff(int(n))
def _ts_mean(x, n):   return _roll(x, n).mean()
def _ts_std(x, n):    return _roll(x, n).std()
def _ts_sum(x, n):    return _roll(x, n).sum()
def _ts_max(x, n):    return _roll(x, n).max()
def _ts_min(x, n):    return _roll(x, n).min()
def _ts_skew(x, n):   return _roll(x, n).skew()
def _ts_kurt(x, n):   return _roll(x, n).kurt()
def _ts_median(x, n): return _roll(x, n).median()


def _ema(x, n):
    # adjust=False = recursive EMA: past values never change when new bars arrive.
    return x.ewm(span=int(n), adjust=False, min_periods=int(n)).mean()


def _wma(x, n):
    n = int(n)
    w = np.arange(1, n + 1, dtype=float)   # most recent bar gets the largest weight
    w /= w.sum()
    return _apply(x, n, lambda a: float(np.dot(a, w)))


def _ts_rank(x, n):
    # percentile rank of the CURRENT value within its own trailing window
    return _apply(x, n, lambda a: float((a <= a[-1]).mean()) if np.isfinite(a[-1]) else np.nan)


def _ts_zscore(x, n):
    return (x - _ts_mean(x, n)) / (_ts_std(x, n) + _EPS)


def _ts_arg_max(x, n):
    # position of the max inside the window: 0 = oldest bar, n-1 = current bar
    return _apply(x, n, lambda a: float(np.argmax(a)) if np.isfinite(a).all() else np.nan)


def _ts_arg_min(x, n):
    return _apply(x, n, lambda a: float(np.argmin(a)) if np.isfinite(a).all() else np.nan)


def _ts_corr(x, y, n):
    with np.errstate(all="ignore"):
        out = _roll(x, n).corr(y)
    return out.replace([np.inf, -np.inf], np.nan)


def _ts_cov(x, y, n):
    return _roll(x, n).cov(y)


def _ts_quantile(x, n, q):
    return _roll(x, n).quantile(float(q))


def _decay_linear(x, n):
    return _wma(x, n)


def _roc(x, n):
    return x / x.shift(int(n)) - 1.0


def _slope_arr(a):
    t = np.arange(len(a), dtype=float)
    if not np.isfinite(a).all():
        return np.nan
    tc = t - t.mean(); ac = a - a.mean()
    denom = float(tc @ tc)
    return float(tc @ ac / denom) if denom > 0 else np.nan


def _slope(x, n):
    return _apply(x, n, _slope_arr)


def _rsquare(x, n):
    def f(a):
        if not np.isfinite(a).all():
            return np.nan
        t = np.arange(len(a), dtype=float)
        sa, st = a.std(), t.std()
        if sa <= 0 or st <= 0:
            return 0.0
        c = float(np.corrcoef(a, t)[0, 1])
        return c * c
    return _apply(x, n, f)


def _resi(x, n):
    """Residual of the CURRENT bar from a trailing OLS-on-time fit (Qlib RESI)."""
    def f(a):
        b = _slope_arr(a)
        if not np.isfinite(b):
            return np.nan
        t = np.arange(len(a), dtype=float)
        alpha = a.mean() - b * t.mean()
        return float(a[-1] - (alpha + b * t[-1]))
    return _apply(x, n, f)


# ---- elementwise -----------------------------------------------------------

def _log(x):
    with np.errstate(all="ignore"):
        out = np.log(x)
    return _clean(out)


def _sign(x):      return np.sign(x)
def _abs(x):       return abs(x) if not np.isscalar(x) else float(np.abs(x))
def _sqrt(x):      return _clean(np.sqrt(_abs(x)))
def _exp(x):       return _clean(np.exp(x))
def _power(x, p):
    with np.errstate(all="ignore"):
        return _clean(np.power(x, p))
def _signedpower(x, p): return _sign(x) * np.power(_abs(x), p)


def _greater(x, y):
    if isinstance(x, (pd.Series, pd.DataFrame)):
        return x.where(x >= y, y)
    if isinstance(y, (pd.Series, pd.DataFrame)):
        return y.where(y >= x, x)
    return max(x, y)


def _less(x, y):
    if isinstance(x, (pd.Series, pd.DataFrame)):
        return x.where(x <= y, y)
    if isinstance(y, (pd.Series, pd.DataFrame)):
        return y.where(y <= x, x)
    return min(x, y)


def _where(cond, a, b):
    """If(cond, a, b) — cond may be a 0/1 float frame (Compare results are floats)."""
    if isinstance(cond, (pd.Series, pd.DataFrame)):
        mask = cond.fillna(0.0).astype(bool)
        ref = a if isinstance(a, (pd.Series, pd.DataFrame)) else b
        if not isinstance(ref, (pd.Series, pd.DataFrame)):   # both scalars
            return (mask * a + (~mask) * b).where(cond.notna())
        aa = a if isinstance(a, (pd.Series, pd.DataFrame)) else ref * 0 + a
        bb = b if isinstance(b, (pd.Series, pd.DataFrame)) else ref * 0 + b
        out = aa.where(mask, bb)
        # NaN in cond -> NaN out (don't invent values where the condition is unknown)
        return out.where(cond.notna())
    return a if cond else b


def _clean(x):
    if isinstance(x, (pd.Series, pd.DataFrame)):
        return x.replace([np.inf, -np.inf], np.nan)
    if np.isscalar(x) and not np.isfinite(x):
        return np.nan
    return x


# ---- cross-sectional (panel/wide-frame only, act per ROW = per date) --------

def _need_panel(x, opname):
    if not isinstance(x, pd.DataFrame):
        raise ValueError(
            f"cross-sectional operator '{opname}' needs panel input "
            "({symbol: OHLCV} dict -> wide frame); got a single-symbol Series. "
            "Use time-series operators (ts_rank/ts_zscore/...) for one symbol.")


def _cs_rank(x):
    _need_panel(x, "rank")
    return x.rank(axis=1, pct=True)


def _cs_zscore(x):
    _need_panel(x, "zscore")
    mu = x.mean(axis=1)
    sd = x.std(axis=1)
    return x.sub(mu, axis=0).div(sd + _EPS, axis=0)


def _cs_scale(x, a=1.0):
    _need_panel(x, "scale")
    denom = x.abs().sum(axis=1)
    return x.mul(float(a), axis=0).div(denom.replace(0.0, np.nan), axis=0)


def _cs_quantile(x, q=0.05):
    """Winsorize each date's cross-section to its [q, 1-q] quantiles (outlier clip)."""
    _need_panel(x, "quantile")
    q = float(q)
    lo = x.quantile(q, axis=1)
    hi = x.quantile(1.0 - q, axis=1)
    return x.clip(lower=lo, upper=hi, axis=0)


def _cs_group_rank(x, groups):
    _need_panel(x, "group_rank")
    out = pd.DataFrame(np.nan, index=x.index, columns=x.columns)
    for g in set(groups.get(c, None) for c in x.columns):
        cols = [c for c in x.columns if groups.get(c, None) == g]
        if cols:
            out[cols] = x[cols].rank(axis=1, pct=True)
    return out


def _cs_group_neutralize(x, groups):
    _need_panel(x, "group_neutralize")
    out = x.copy()
    for g in set(groups.get(c, None) for c in x.columns):
        cols = [c for c in x.columns if groups.get(c, None) == g]
        if cols:
            out[cols] = x[cols].sub(x[cols].mean(axis=1), axis=0)
    return out


# --------------------------------------------------------------------------- #
# whitelist tables
# --------------------------------------------------------------------------- #

_FUNCS = {
    # time-series (causal)
    "delay": _delay, "ref": _delay,
    "delta": _delta,
    "ts_mean": _ts_mean, "sma": _ts_mean, "mean": _ts_mean,
    "ts_std": _ts_std, "std": _ts_std,
    "ts_sum": _ts_sum, "sum": _ts_sum,
    "ts_max": _ts_max, "max": _ts_max,
    "ts_min": _ts_min, "min": _ts_min,
    "ts_median": _ts_median,
    "ts_skew": _ts_skew, "skew": _ts_skew,
    "ts_kurt": _ts_kurt, "kurt": _ts_kurt,
    "ema": _ema, "wma": _wma,
    "ts_rank": _ts_rank, "ts_zscore": _ts_zscore,
    "ts_arg_max": _ts_arg_max, "ts_argmax": _ts_arg_max, "imax": _ts_arg_max, "idxmax": _ts_arg_max,
    "ts_arg_min": _ts_arg_min, "ts_argmin": _ts_arg_min, "imin": _ts_arg_min, "idxmin": _ts_arg_min,
    "ts_corr": _ts_corr, "corr": _ts_corr,
    "ts_cov": _ts_cov, "cov": _ts_cov,
    "ts_decay_linear": _decay_linear, "decay_linear": _decay_linear,
    "ts_quantile": _ts_quantile,
    "roc": _roc,
    "slope": _slope,
    "rsquare": _rsquare, "rsqr": _rsquare,
    "resi": _resi,
    # elementwise
    "abs": _abs, "log": _log, "sign": _sign, "sqrt": _sqrt, "exp": _exp,
    "power": _power, "pow": _power, "signedpower": _signedpower,
    "greater": _greater, "less": _less,
    "where": _where, "if": _where,
    # cross-sectional
    "rank": _cs_rank, "zscore": _cs_zscore, "scale": _cs_scale, "quantile": _cs_quantile,
}

_GROUP_FUNCS = {"group_rank": _cs_group_rank, "group_neutralize": _cs_group_neutralize}

# macros expand against the variable environment (K-bar shape features & composites).
# Zero-arg macros keep library formulas short: KMID() instead of (close-open)/open.
_MACROS = {
    "kmid":  lambda e: (e["close"] - e["open"]) / e["open"],
    "klen":  lambda e: (e["high"] - e["low"]) / e["open"],
    "kmid2": lambda e: (e["close"] - e["open"]) / (e["high"] - e["low"] + _EPS),
    "kup":   lambda e: (e["high"] - _greater(e["open"], e["close"])) / e["open"],
    "kup2":  lambda e: (e["high"] - _greater(e["open"], e["close"])) / (e["high"] - e["low"] + _EPS),
    "klow":  lambda e: (_less(e["open"], e["close"]) - e["low"]) / e["open"],
    "klow2": lambda e: (_less(e["open"], e["close"]) - e["low"]) / (e["high"] - e["low"] + _EPS),
    "ksft":  lambda e: (2 * e["close"] - e["high"] - e["low"]) / e["open"],
    "ksft2": lambda e: (2 * e["close"] - e["high"] - e["low"]) / (e["high"] - e["low"] + _EPS),
    # WVMA(n): volume-weighted return volatility (Qlib) — Std/Mean of |ret|*vol
    "wvma":  lambda e, n: _clean(_ts_std(abs(_roc(e["close"], 1)) * e["volume"], n)
                                 / (_ts_mean(abs(_roc(e["close"], 1)) * e["volume"], n) + _EPS)),
    # rolling quantile shorthands (Qlib QTLU/QTLD on close)
    "qtlu":  lambda e, n: _ts_quantile(e["close"], n, 0.8),
    "qtld":  lambda e, n: _ts_quantile(e["close"], n, 0.2),
}

_VARS = ("open", "high", "low", "close", "volume", "vwap", "returns")

_BINOPS = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b,
           ast.Mult: lambda a, b: a * b, ast.Div: lambda a, b: a / b,
           ast.Pow: lambda a, b: a ** b, ast.Mod: lambda a, b: a % b}
_CMPOPS = {ast.Gt: lambda a, b: a > b, ast.Lt: lambda a, b: a < b,
           ast.GtE: lambda a, b: a >= b, ast.LtE: lambda a, b: a <= b,
           ast.Eq: lambda a, b: a == b, ast.NotEq: lambda a, b: a != b}


def available_operators() -> list:
    """Every name the DSL accepts — surfaced in error messages so a typo is cheap."""
    return sorted(set(_FUNCS) | set(_MACROS) | set(_GROUP_FUNCS) | set(_VARS))


class _Evaluator(ast.NodeVisitor):
    """Whitelist AST walker. Anything not explicitly handled -> ValueError.

    Compare nodes return 0/1 floats (NaN where either side is NaN) so that
    conditions compose with arithmetic and rolling ops (e.g. CNTP = ts_mean(close >
    delay(close,1), n)) without boolean-dtype surprises.
    """

    def __init__(self, env, groups=None):
        self.env = env
        self.groups = groups

    def _err(self, msg):
        raise ValueError(f"{msg}. Available operators/variables: "
                         + ", ".join(available_operators()))

    def visit(self, node):  # route only through whitelisted node types
        h = getattr(self, "visit_" + type(node).__name__, None)
        if h is None:
            self._err(f"expression element '{type(node).__name__}' is not allowed")
        return h(node)

    def visit_Expression(self, node):
        return self.visit(node.body)

    def visit_Constant(self, node):
        if isinstance(node.value, (int, float)):
            return node.value
        self._err(f"constant {node.value!r} not allowed (numbers only)")

    def visit_Name(self, node):
        key = node.id.lower()
        if key in self.env:
            return self.env[key]
        self._err(f"unknown variable '{node.id}'")

    def visit_UnaryOp(self, node):
        v = self.visit(node.operand)
        if isinstance(node.op, ast.USub):
            return -v
        if isinstance(node.op, ast.UAdd):
            return v
        self._err("unary operator not allowed")

    def visit_BinOp(self, node):
        op = _BINOPS.get(type(node.op))
        if op is None:
            self._err(f"binary operator '{type(node.op).__name__}' not allowed")
        with np.errstate(all="ignore"):
            return op(self.visit(node.left), self.visit(node.right))

    def visit_Compare(self, node):
        if len(node.ops) != 1:
            self._err("chained comparisons (a < b < c) are not supported")
        op = _CMPOPS.get(type(node.ops[0]))
        if op is None:
            self._err("comparison operator not allowed")
        a, b = self.visit(node.left), self.visit(node.comparators[0])
        res = op(a, b)
        if isinstance(res, (pd.Series, pd.DataFrame)):
            res = res.astype(float)
            for side in (a, b):        # NaN-in -> NaN-out, comparisons shouldn't invent 0s
                if isinstance(side, (pd.Series, pd.DataFrame)):
                    res = res.where(side.notna())
        return res

    def visit_IfExp(self, node):
        return _where(self.visit(node.test), self.visit(node.body), self.visit(node.orelse))

    def visit_Call(self, node):
        if not isinstance(node.func, ast.Name):
            self._err("only plain function calls f(...) are allowed")
        if node.keywords:
            self._err("keyword arguments are not supported in expressions")
        name = node.func.id.lower()
        args = [self.visit(a) for a in node.args]
        if name in _MACROS:
            return _MACROS[name](self.env, *args)
        if name in _GROUP_FUNCS:
            if self.groups is None:
                raise ValueError(f"'{name}' needs a symbol->group mapping: "
                                 "eval_expr(expr, data, groups={symbol: group})")
            return _GROUP_FUNCS[name](*args, self.groups)
        fn = _FUNCS.get(name)
        if fn is None:
            self._err(f"unknown function '{node.func.id}'")
        return fn(*args)


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #

def eval_expr(expr: str, data, *, groups: dict | None = None):
    """Evaluate a factor expression.

    data: single OHLCV DataFrame -> pd.Series;  {symbol: OHLCV} dict -> wide
    DataFrame (date x symbol). ±inf is mapped to NaN at the end so downstream
    IC/regression code never chokes on a divide-by-zero bar.
    """
    if isinstance(data, dict):
        env = _panel_env(data)
    elif isinstance(data, pd.DataFrame):
        env = _single_env(data)
    else:
        raise ValueError("data must be an OHLCV DataFrame or a {symbol: OHLCV} dict")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"cannot parse factor expression {expr!r}: {e}") from None
    out = _Evaluator(env, groups=groups).visit(tree)
    if not isinstance(out, (pd.Series, pd.DataFrame)):
        # a constant expression — broadcast so callers always get an aligned object
        out = env["close"] * 0 + float(out)
    return _clean(out.astype(float))


def expr_to_callable(expr: str, *, groups: dict | None = None):
    """Wrap an expression as f(df) -> Series, the exact signature factor_lab's
    validate_factor / register_custom_factor expect for single-symbol factors."""
    def f(df):
        return eval_expr(expr, df, groups=groups)
    f.__name__ = "expr_factor"
    f.__doc__ = f"factor expression: {expr}"
    return f
