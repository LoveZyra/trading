"""标的池构建(AI 选股 / 横截面)。
模式:list | csv | index | sector。返回去重后的代码清单 + 可复现 universe.json 规格。
诚实护栏:数量过少告警、单一板块告警、低流动性过滤。
"""
from __future__ import annotations
import json, warnings
from pathlib import Path
import pandas as pd

MIN_NAMES_OK = 30   # 低于此,横截面排序统计意义弱(见 references/ai_stock_selection.md)

def _warn_breadth(symbols, sectors=None):
    n = len(symbols)
    if n < MIN_NAMES_OK:
        warnings.warn(f"universe 仅 {n} 只;横截面排序需 ~{MIN_NAMES_OK}+ 只分散标的才有统计意义。", stacklevel=3)
    if sectors:
        uniq = {s for s in sectors.values() if s}
        if uniq and len(uniq) == 1:
            warnings.warn(f"全部属于同一板块({next(iter(uniq))});同业高度同涨同跌,横截面可排序结构很弱。", stacklevel=3)

def from_list(symbols, sectors=None):
    syms = [str(s).strip().upper() for s in symbols if s and str(s).strip()]
    syms = list(dict.fromkeys(syms))                       # 去重保序
    _warn_breadth(syms, sectors)
    return syms

def from_csv(path, column="symbol", sector_column=None):
    df = pd.read_csv(path)
    col = column if column in df.columns else df.columns[0]
    sectors = dict(zip(df[col].astype(str), df[sector_column])) if sector_column and sector_column in df.columns else None
    return from_list(df[col].astype(str).tolist(), sectors)

def from_index(name, market="CN"):
    """指数成分:A股/港股用 akshare(可选依赖);美股需提供成分快照(走 list/csv)。"""
    if str(market).upper() in ("CN", "A", "ASHARE", "HK"):
        try:
            import akshare as ak
        except Exception as e:
            raise RuntimeError(f"A股/港股指数成分需要 akshare:{e}")
        df = ak.index_stock_cons(symbol=str(name))
        col = [c for c in df.columns if "代码" in c or "code" in c.lower()][0]
        return from_list(df[col].astype(str).tolist())
    raise RuntimeError("美股指数成分无免费稳定 API:请用 source='csv' 提供成分快照,或用券商扫描。")

def liquidity_filter(data: dict, min_dollar_vol=2e6, min_days=120):
    """剔除流动性过低/历史过短的标的。"""
    keep = []
    for s, df in data.items():
        if df is None or len(df) < min_days:
            continue
        if {"close", "volume"}.issubset(df.columns):
            dv = (df["close"] * df["volume"]).tail(60).median()
            if pd.notna(dv) and dv < min_dollar_vol:
                continue
        keep.append(s)
    return keep

def build_universe(spec: dict) -> dict:
    """spec={market, source:list|csv|index, symbols/path/name, sectors?}. 返回可 JSON 序列化的规格。"""
    src = spec.get("source", "list"); market = spec.get("market", "US")
    if src == "list":   syms = from_list(spec["symbols"], spec.get("sectors"))
    elif src == "csv":  syms = from_csv(spec["path"], spec.get("column", "symbol"), spec.get("sector_column"))
    elif src == "index":syms = from_index(spec["name"], market)
    else: raise ValueError(f"未知 source: {src}")
    return {"symbols": syms, "market": market, "source": src, "n": len(syms), "spec": spec}

def save_spec(uni: dict, path="universe.json"):
    Path(path).write_text(json.dumps(uni, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
