"""Factor lab — implement, VALIDATE, and backtest a custom factor.

This is the landing pad for RD-Agent's "extract a factor from a research report /
financial report / paper" idea (their fin_factor_report). The extraction itself is
LLM work — YOU (Claude) read the document and translate the described factor into a
function. This module gives you the safety rails and the test harness so the factor
you implement is causal and actually has predictive content before it ever enters a
strategy. See references/factor_extraction.md for the end-to-end workflow.

A "factor function" has signature:  f(df: OHLCV DataFrame) -> pd.Series
aligned to df.index. It must use ONLY past/current bars (causal). The validator below
mechanically checks that.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..core import backtest as bt
from ..core import indicators as ind  # noqa: F401  (handy for factor authors)

CUSTOM_FACTORS: dict = {}


def register_custom_factor(name: str, func):
    """Add a factor function to the registry so the CLI / autoresearch can use it."""
    CUSTOM_FACTORS[name] = func
    return func


@dataclass
class FactorCheck:
    causal: bool
    nan_ratio: float
    coverage: int
    messages: list

    @property
    def ok(self) -> bool:
        return self.causal and self.nan_ratio < 0.9

    def __repr__(self):
        tag = "OK" if self.ok else "PROBLEM"
        return (f"FactorCheck[{tag}] causal={self.causal} nan_ratio={self.nan_ratio:.2%} "
                f"coverage={self.coverage}\n  " + "\n  ".join(self.messages))


def validate_factor(func, df: pd.DataFrame, *, tol: float = 1e-8) -> FactorCheck:
    """Mechanically check a factor is causal (no look-ahead) and well-formed.

    Causality test: compute the factor on the full history, and again on the history
    truncated to the first K bars. A causal factor at position i<K uses only bars
    <=i, all present in the truncated series, so trunc[i] MUST equal full[i] wherever
    full[i] is non-NaN -- including being non-NaN there. If full has a value but trunc
    is NaN (it needed a bar >=K) or the values differ, the factor peeked at the
    future. (We must NOT dropna, since look-ahead shows up exactly at the boundary
    positions a dropna would discard.)

    We probe several cut points (50%, 70%, 90% of length), not one. A factor that only
    peeks a few bars ahead, or that conditionally looks ahead late in the series, can
    slip past a single cut at 60% but gets caught at one of the others — the boundary
    is where look-ahead shows up, so testing several boundaries is much harder to fool.
    """
    msgs = []
    full = func(df)
    if not isinstance(full, pd.Series):
        return FactorCheck(False, 1.0, 0, ["factor must return a pd.Series"])
    full = full.reindex(df.index)

    n = len(df)
    cut_points = sorted({max(30, int(n * f)) for f in (0.5, 0.7, 0.9)})
    causal = True
    checked_any = False
    worst_leak, worst_diff, worst_K = 0, 0.0, None
    for K in cut_points:
        if K >= n:        # need at least one future bar removed for the test to bite
            continue
        trunc = func(df.iloc[:K]).reindex(df.index[:K])
        overlap = full.iloc[:K]
        mask = overlap.notna()
        if mask.sum() == 0:
            continue
        checked_any = True
        leaked_to_nan = int((mask & trunc.reindex(overlap.index).isna()).sum())
        diffs = (overlap[mask] - trunc.reindex(overlap.index)[mask]).abs()
        max_diff = float(diffs.max()) if len(diffs.dropna()) else 0.0
        if leaked_to_nan or max_diff > tol:
            causal = False
            if leaked_to_nan >= worst_leak and max_diff >= worst_diff:
                worst_leak, worst_diff, worst_K = leaked_to_nan, max_diff, K
    if not checked_any:
        msgs.append("could not establish overlap for causality check (too few values)")
    elif not causal:
        why = []
        if worst_leak:
            why.append(f"{worst_leak} past values became undefined when future bars "
                       f"were removed (factor needs future data)")
        if worst_diff > tol:
            why.append(f"past values shifted by up to {worst_diff:.2e}")
        msgs.append(f"NOT CAUSAL (cut@{worst_K}): " + "; ".join(why) +
                    " -> the factor is looking ahead. Fix it.")
    else:
        msgs.append(f"causality check passed at {len(cut_points)} cut points "
                    "(past values stable when future appended)")

    nan_ratio = float(full.isna().mean())
    coverage = int(full.notna().sum())
    if nan_ratio > 0.5:
        msgs.append(f"high NaN ratio {nan_ratio:.0%} — factor is sparse; check warm-up/inputs")
    return FactorCheck(causal=causal, nan_ratio=nan_ratio, coverage=coverage, messages=msgs)


def factor_to_signal(factor: pd.Series, *, lookback: int = 60, mode: str = "momentum",
                     clip: float = 2.0) -> pd.Series:
    """Standardize a raw factor into a target position in [-1, 1].

    mode='momentum'  -> high factor = long  (trend/quality/positive-alpha factors)
    mode='reversion' -> high factor = short (overbought/expensive factors)
    Uses a rolling z-score so the factor's own scale/drift doesn't matter.
    """
    z = (factor - factor.rolling(lookback).mean()) / factor.rolling(lookback).std(ddof=0)
    z = z.clip(-clip, clip) / clip
    return (-z if mode == "reversion" else z).fillna(0.0)


def backtest_custom_factor(func, df: pd.DataFrame, *, mode: str = "momentum",
                           lookback: int = 60, commission_bps: float = 1.0,
                           slippage_bps: float = 1.0, validate: bool = True):
    """Validate (optional) then backtest a single-asset factor as a continuous signal.

    Returns (BacktestResult, FactorCheck|None). Refuses to backtest a non-causal
    factor — a look-ahead factor's backtest is meaningless and dangerous.
    """
    check = validate_factor(func, df) if validate else None
    if check is not None and not check.causal:
        raise ValueError(f"factor failed causality check — refusing to backtest.\n{check}")
    factor = func(df)
    sig = factor_to_signal(factor, lookback=lookback, mode=mode)
    res = bt.backtest(df, sig, commission_bps=commission_bps, slippage_bps=slippage_bps)
    return res, check


def factor_ic(func, df: pd.DataFrame, horizon: int = 21) -> float:
    """Single-asset information coefficient: correlation of the factor with the
    forward `horizon`-bar return. A quick read on whether the factor has any edge
    before you bother backtesting. |IC| > ~0.03 is already interesting on real data.
    """
    factor = func(df)
    fwd = df["close"].shift(-horizon) / df["close"] - 1.0
    both = pd.concat([factor, fwd], axis=1).dropna()
    if len(both) < 10:
        return float("nan")
    return float(both.corr().iloc[0, 1])


def factor_scorecard(func, df: pd.DataFrame, *, horizon: int = 21,
                     existing: dict | None = None) -> dict:
    """Score a factor on the dimensions that matter — not just one backtest Sharpe.

    Inspired by AlphaEval (2508.13174) and AlphaAgent's decay/diversity controls
    (2502.16789). A factor can have a great backtest and still be useless: unstable,
    redundant with what you already have, or so high-turnover it can't be traded.

      ic            : mean information coefficient vs forward return (predictive power)
      ic_ir         : IC information ratio = mean(rolling IC)/std(rolling IC) — STABILITY
      autocorr      : lag-1 autocorrelation of the factor — smoothness / turnover proxy
      coverage      : fraction of bars with a value
      max_corr_existing : largest |correlation| to factors in `existing` (DIVERSITY)
    `existing`: {name: factor_func} of factors you already use, for the diversity check.
    Returns a dict plus a short 'verdict'.
    """
    f = func(df)
    ic = factor_ic(func, df, horizon)
    fwd = df["close"].shift(-horizon) / df["close"] - 1.0
    win = max(63, horizon * 3)
    roll = f.rolling(win).corr(fwd)
    ic_ir = float(roll.mean() / roll.std(ddof=0)) if roll.std(ddof=0) else float("nan")
    autocorr = float(f.autocorr(lag=1)) if f.notna().sum() > 3 else float("nan")
    coverage = float(f.notna().mean())
    max_corr = 0.0
    if existing:
        for nm, g in existing.items():
            try:
                both = pd.concat([f, g(df)], axis=1).dropna()
                if len(both) > 10:
                    max_corr = max(max_corr, abs(both.corr().iloc[0, 1]))
            except Exception:  # noqa: BLE001
                continue
    verdict = []
    if abs(ic) < 0.02:
        verdict.append("weak predictive power")
    if np.isfinite(ic_ir) and abs(ic_ir) < 0.3:
        verdict.append("unstable IC")
    if np.isfinite(autocorr) and autocorr < 0.5:
        verdict.append("high turnover")
    if max_corr > 0.7:
        verdict.append(f"redundant (corr {max_corr:.2f} with existing)")
    return {"ic": round(ic, 4), "ic_ir": round(ic_ir, 3) if np.isfinite(ic_ir) else None,
            "autocorr": round(autocorr, 3) if np.isfinite(autocorr) else None,
            "coverage": round(coverage, 3), "max_corr_existing": round(max_corr, 3),
            "verdict": "; ".join(verdict) if verdict else "looks promising"}


# =============================================================================
# Round 10 — 因子质量套件
# §2.2 正交化 / 增量评估、§2.14 AST 复杂度与新颖度、§2.16 五维 alpha_eval。
# 本节所有 "panel" 参数都是宽表 DataFrame(date × symbol),与 xsec/panel.py 口径
# 一致。只追加、不改动上方任何既有函数。
# =============================================================================
import ast as _ast


def daily_rank_ic(a: pd.DataFrame, b: pd.DataFrame, min_names: int = 5) -> pd.Series:
    """逐日截面 Spearman 相关(a、b 均为 date×symbol 宽表),返回按日 Series。

    why 向量化按行做:横截面因子评估的基本原子是"每天一个截面相关",逐日循环
    scipy.spearmanr 既慢又引入非必装依赖;先把双方共同缺失置 NaN 再按行 rank,
    然后按行去均值做 Pearson,数学上等价于逐日 spearman(含并列的平均秩修正)。
    截面名字数 < min_names 的日子置 NaN——3 只票的"相关"只有噪声,不如缺失诚实。
    """
    a, b = a.align(b, join="inner")
    mask = a.notna() & b.notna()
    ar = a.where(mask).rank(axis=1)
    br = b.where(mask).rank(axis=1)
    am = ar.sub(ar.mean(axis=1), axis=0)
    bm = br.sub(br.mean(axis=1), axis=0)
    num = (am * bm).sum(axis=1)
    den = np.sqrt((am ** 2).sum(axis=1) * (bm ** 2).sum(axis=1))
    ic = num / den.replace(0.0, np.nan)
    ic[mask.sum(axis=1) < min_names] = np.nan
    return ic


def _forward_return_panel(close_panel: pd.DataFrame, horizon: int) -> pd.DataFrame:
    return close_panel.shift(-horizon) / close_panel - 1.0


def orthogonalize(factor_panel: pd.DataFrame, reference_panels: dict,
                  method: str = "schmidt") -> pd.DataFrame:
    """逐日截面正交化:每个交易日把因子对全部参照因子(加截距)做 OLS,取残差。

    why 联合回归而非逐个 Gram-Schmidt 迭代:对残差而言,一次性投影到
    span(参照因子) 的正交补,等于按任意顺序做完整 Gram-Schmidt 的最后结果,
    且数值上更稳(lstsq 用 SVD,参照因子共线时不爆炸)。加截距是为了把
    "截面均值"也吸收掉——不然残差会带着水平项污染后续 rank 相关。
    某日有效样本太少(< 参照数+3)时整行置 NaN,而不是硬拟合出过参数化的 0 残差。
    """
    if method != "schmidt":
        raise ValueError(f"unknown method {method!r}; only 'schmidt' is implemented")
    refs = [p.reindex(index=factor_panel.index, columns=factor_panel.columns)
            for p in reference_panels.values()]
    F = factor_panel.values.astype(float)
    R = [r.values.astype(float) for r in refs]
    out = np.full_like(F, np.nan, dtype=float)
    ones = np.ones(F.shape[1])
    for i in range(F.shape[0]):
        y = F[i]
        X = np.column_stack([ones] + [r[i] for r in R])
        m = np.isfinite(y) & np.isfinite(X).all(axis=1)
        if m.sum() < X.shape[1] + 3:
            continue
        beta, *_ = np.linalg.lstsq(X[m], y[m], rcond=None)
        out[i, m] = y[m] - X[m] @ beta
    return pd.DataFrame(out, index=factor_panel.index, columns=factor_panel.columns)


def incremental_ic(factor_panel: pd.DataFrame, existing_panels: dict,
                   close_panel: pd.DataFrame, horizon: int = 21) -> dict:
    """增量评估:因子对既有因子库正交化后还剩多少预测力(§2.2)。

    raw_ic          原始因子的时均 RankIC;
    orthogonal_ic   对 existing_panels 逐日正交化后的残差 RankIC;
    incremental_ratio = orthogonal_ic / raw_ic —— 接近 1 说明预测力基本独立于
    既有因子;接近 0 说明"新因子"只是旧因子换皮(AlphaAgent 式冗余控制)。
    |raw_ic| 太小时比值没有意义,返回 NaN 而不是一个爆炸的数。
    """
    fwd = _forward_return_panel(close_panel, horizon)
    raw = float(daily_rank_ic(factor_panel, fwd).mean())
    orth = orthogonalize(factor_panel, existing_panels)
    oic = float(daily_rank_ic(orth, fwd).mean())
    if not (np.isfinite(raw) and abs(raw) > 1e-8):
        ratio = float("nan")
    else:
        ratio = oic / raw
    return {"raw_ic": raw, "orthogonal_ic": oic, "incremental_ratio": ratio}


def factor_correlation_matrix(panels: dict) -> pd.DataFrame:
    """因子两两"逐日截面 spearman 相关的时间均值"矩阵(对称、对角为 1)。

    why 逐日截面而不是把宽表拉平:拉平后的相关被"日期效应"(整个截面同涨同跌)
    主导,而截面选股关心的是同一天内的排序一致性。
    """
    names = list(panels)
    M = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    for i, a in enumerate(names):
        for j in range(i + 1, len(names)):
            c = float(daily_rank_ic(panels[a], panels[names[j]]).mean())
            M.iloc[i, j] = M.iloc[j, i] = c
    return M


# ---- §2.14 表达式复杂度与新颖度(AST) ---------------------------------------

_AST_SKIP = (_ast.expr_context, _ast.operator, _ast.unaryop, _ast.cmpop, _ast.boolop)


def _ast_depth(node) -> int:
    kids = [c for c in _ast.iter_child_nodes(node) if not isinstance(c, _AST_SKIP)]
    return 1 + (max(_ast_depth(c) for c in kids) if kids else 0)


def complexity_control(expr_str: str, max_depth: int = 5, max_params: int = 3) -> dict:
    """用 Python ast 给因子表达式做复杂度门禁(§2.14)。

    depth    = 表达式树的嵌套深度(剔除 Load/Add 之类的语法糖节点,数的是
               "真实嵌套层数":f(g(x)) 深 3,不受 +、索引写法的噪声干扰);
    n_params = 数值常数个数(魔法数字越多越像过拟合出来的曲线救国)。
    why:AlphaAgent(2502.16789)的经验是,复杂度不设闸,LLM 挖因子会朝着
    "backtest 上刚好好看"的深嵌套怪物收敛——先拒掉,比事后 OOS 崩了再排查便宜。
    解析失败直接 ok=False:连语法都不合法的表达式没资格进库。
    """
    try:
        tree = _ast.parse(expr_str, mode="eval")
    except SyntaxError as e:
        return {"ok": False, "depth": None, "n_params": None, "error": str(e)}
    depth = _ast_depth(tree.body)
    n_params = sum(1 for nd in _ast.walk(tree)
                   if isinstance(nd, _ast.Constant)
                   and isinstance(nd.value, (int, float))
                   and not isinstance(nd.value, bool))
    return {"ok": depth <= max_depth and n_params <= max_params,
            "depth": depth, "n_params": n_params}


def _ast_tokens(expr_str: str) -> list:
    """前序遍历的 AST 节点标签序列。常数统一记作 'Const'(把 10 调成 12 不算
    新因子),变量/函数名保留(mom 换成 vol 才是结构性差异)。"""
    tree = _ast.parse(expr_str, mode="eval")
    out: list = []

    def visit(n):
        if isinstance(n, _AST_SKIP):
            return
        if isinstance(n, _ast.Name):
            out.append(f"Name:{n.id}")
        elif isinstance(n, _ast.Attribute):
            out.append(f"Attr:{n.attr}")
        elif isinstance(n, _ast.Constant):
            out.append("Const")
        else:
            out.append(type(n).__name__)
        for c in _ast.iter_child_nodes(n):
            visit(c)

    visit(tree.body)
    return out


def _norm_edit_distance(a: list, b: list) -> float:
    """归一化 Levenshtein(除以较长序列长度,落在 [0,1])。O(len_a·len_b),
    因子表达式的 token 数是几十级别,DP 足够快且比 multiset Jaccard 更能
    区分"同一批算子换了嵌套顺序"的伪新颖。"""
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1] / max(len(a), len(b))


def novelty_check(expr_str: str, library: dict, method: str = "ast",
                  threshold: float = 0.25) -> dict:
    """新因子表达式 vs 因子库的结构新颖度(§2.14)。

    对库里每条表达式算 AST token 序列的归一化编辑距离,取最小值:
    min_distance <= threshold 判为"不新颖"(novel=False)。对自身距离恒为 0。
    库中某条表达式解析失败就跳过它(库脏不应该拦住新因子的评估)。
    """
    if method != "ast":
        raise ValueError(f"unknown method {method!r}; only 'ast' is implemented")
    toks = _ast_tokens(expr_str)
    best, nearest = 1.0, None
    for name, expr in (library or {}).items():
        try:
            d = _norm_edit_distance(toks, _ast_tokens(expr))
        except SyntaxError:
            continue
        if d < best:
            best, nearest = d, name
    return {"min_distance": float(best), "nearest": nearest,
            "novel": bool(best > threshold)}


# ---- §2.16 五维 alpha_eval ---------------------------------------------------

_INTERPRETABILITY_PROMPT = (
    "请以资深量化研究员的身份评估该因子的可解释性,给出 [0,1] 分数:\n"
    "1) 经济学直觉:因子捕捉的是什么行为/风险溢价?讲得通吗?\n"
    "2) 简洁性:构造是否简单到能向投委会一句话说清?\n"
    "3) 已知谱系:它和文献中的动量/价值/质量/低波等经典因子是什么关系?\n"
    "只有三条都答得上来才给高分;'backtest 好看但说不出为什么'应给低分。\n"
    "因子描述:{desc}"
)


def _sign_stability_score(vals: list) -> float:
    """把一组 IC 观察压成 [0,1] 稳健分:先按均值符号对齐,再取 min/max 比。
    全部同号且幅度接近 → 接近 1;有观察翻号 → min<0 → 截到 0。"""
    vals = [v for v in vals if np.isfinite(v)]
    if len(vals) < 2:
        return 0.0
    s = np.sign(np.mean(vals)) or 1.0
    aligned = [v * s for v in vals]
    hi = max(aligned)
    if hi <= 0:
        return 0.0
    return float(np.clip(min(aligned) / hi, 0.0, 1.0))


def alpha_eval(factor, data, *, existing_panels: dict | None = None,
               horizon: int = 21) -> dict:
    """五维因子评估(§2.16,参考 AlphaEval 2508.13174 的维度划分)。

    factor:callable f(df)->Series(逐标的应用到 data={symbol: OHLCV})或
    直接给宽表 panel(此时 data 可以是 close 宽表或 {symbol: OHLCV})。

      predictive       时均 RankIC,|IC|/0.05 截断到 [0,1](0.05 已是很强的日频因子);
      robustness       callable → horizon×{0.7,1,1.3} 三档 IC 的符号/幅度一致性;
                       panel   → 截面随机分桶(子采样)的 IC 一致性;
      diversity        1 - vs existing_panels 的最大|时均截面spearman|;无库给 1;
      stability        rolling(63) RankIC 的 IR,经 x/(1+x) 压到 [0,1)(平滑饱和,
                       不设"IR>2 就是 1"这种拍脑袋硬截断);
      interpretability 返回 None + interpretability_prompt —— 语义判断是 LLM(agent)
                       的活,库内伪造一个数字分数反而是污染。
    composite = 四个数值维度的均值(可解释性缺省不计入)。
    """
    if callable(factor):
        if not isinstance(data, dict):
            raise ValueError("callable factor needs data={symbol: OHLCV DataFrame}")
        close = pd.DataFrame({s: df["close"] for s, df in data.items()})
        panel = pd.DataFrame({s: factor(df) for s, df in data.items()})
        desc = getattr(factor, "__name__", repr(factor))
    else:
        panel = factor
        close = data if isinstance(data, pd.DataFrame) else \
            pd.DataFrame({s: df["close"] for s, df in data.items()})
        desc = "factor panel (expression unavailable)"
    close = close.reindex(index=panel.index, columns=panel.columns)

    fwd = _forward_return_panel(close, horizon)
    daily = daily_rank_ic(panel, fwd)
    mean_ic = float(daily.mean())
    predictive = float(np.clip(abs(mean_ic) / 0.05, 0.0, 1.0)) if np.isfinite(mean_ic) else 0.0

    # robustness:callable 用 horizon 扰动(因子本身不变,标签口径扰动);
    # panel 用截面子采样(没法重算因子,只能问"换一半票还灵吗")。
    if callable(factor):
        hs = sorted({max(1, int(round(horizon * m))) for m in (0.7, 1.0, 1.3)})
        ics = [float(daily_rank_ic(panel, _forward_return_panel(close, h)).mean())
               for h in hs]
    else:
        rng = np.random.default_rng(42)
        cols = list(panel.columns)
        rng.shuffle(cols)
        n_chunks = 3 if len(cols) >= 15 else 2
        chunks = [cols[i::n_chunks] for i in range(n_chunks)]
        ics = [float(daily_rank_ic(panel[ch], fwd[ch], min_names=3).mean())
               for ch in chunks if len(ch) >= 3]
    robustness = _sign_stability_score(ics)

    diversity = 1.0
    max_corr, max_name = 0.0, None
    for name, p in (existing_panels or {}).items():
        c = abs(float(daily_rank_ic(panel, p).mean()))
        if np.isfinite(c) and c > max_corr:
            max_corr, max_name = c, name
    diversity = float(np.clip(1.0 - max_corr, 0.0, 1.0))

    win = int(min(63, max(20, len(daily.dropna()) // 4))) or 20
    roll = daily.rolling(win, min_periods=max(10, win // 2)).mean()
    sd = float(roll.std(ddof=0))
    ir = abs(float(roll.mean())) / sd if sd > 0 else 0.0
    stability = float(ir / (1.0 + ir)) if np.isfinite(ir) else 0.0

    dims = [predictive, robustness, diversity, stability]
    composite = float(np.mean(dims))
    return {"predictive": predictive, "robustness": robustness,
            "diversity": diversity, "stability": stability,
            "interpretability": None,
            "interpretability_prompt": _INTERPRETABILITY_PROMPT.format(desc=desc),
            "composite": composite,
            "detail": {"mean_rank_ic": mean_ic, "robustness_ics": ics,
                       "max_corr_existing": max_corr, "most_similar": max_name,
                       "rolling_ic_ir": ir, "horizon": horizon}}
