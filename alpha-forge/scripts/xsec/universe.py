"""标的池构建(AI 选股 / 横截面)。
模式:list | csv | index | sector。返回去重后的代码清单 + 可复现 universe.json 规格。
诚实护栏:数量过少告警、单一板块告警、低流动性过滤。
数据驱动选池:见文件末 build_scored_universe(市值/流动性/指数成分 基座 + 热门/龙头 软打分)。
"""
from __future__ import annotations
import json, warnings
from pathlib import Path
import pandas as pd
import numpy as np

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
    syms = list(dict.fromkeys(syms))
    _warn_breadth(syms, sectors)
    return syms

def from_csv(path, column="symbol", sector_column=None):
    df = pd.read_csv(path)
    col = column if column in df.columns else df.columns[0]
    sectors = dict(zip(df[col].astype(str), df[sector_column])) if sector_column and sector_column in df.columns else None
    return from_list(df[col].astype(str).tolist(), sectors)

def from_index(name, market="CN"):
    """指数成分:A股/港股用 akshare(可选依赖);美股需提供成分快照(走 list/csv/连接器)。"""
    if str(market).upper() in ("CN", "A", "ASHARE", "HK"):
        try:
            import akshare as ak
        except Exception as e:
            raise RuntimeError(f"A股/港股指数成分需要 akshare:{e}")
        df = ak.index_stock_cons(symbol=str(name))
        col = [c for c in df.columns if "代码" in c or "code" in c.lower()][0]
        return from_list(df[col].astype(str).tolist())
    raise RuntimeError("美股指数成分:用 source='csv' 提供成分快照,或用行情连接器(见 meta_from_fmp)。")

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


# ========================================================================
# 数据驱动选池:市值/流动性/指数成分(硬基座) + 热门度/龙头度(软打分)
# 逻辑见 references/ai_stock_selection.md「选池方法」。核心为纯函数(吃 meta+prices),
# 便于离线单测;真实数据由 meta_from_fmp() 从行情连接器解析后喂入。非投资建议。
# ========================================================================

DEFAULT_SCORE_WEIGHTS = {"size": 0.20, "liq": 0.15, "hot": 0.30, "hi52": 0.15, "lead": 0.20}


def _z(s):
    s = pd.Series(s, dtype="float64")
    sd = s.std(ddof=0)
    if not sd or pd.isna(sd):
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / sd


def _close_of(prices, sym):
    if prices and sym in prices and prices[sym] is not None:
        df = prices[sym]
        if "close" in getattr(df, "columns", []):
            return df["close"].dropna()
    return None


def _mom(close, lb):
    if close is None or len(close) <= lb:
        return float("nan")
    return float(close.iloc[-1] / close.iloc[-1 - lb] - 1.0)


def compute_scores(meta: dict, prices: dict | None = None, weights: dict | None = None,
                   hot_lbs=(63, 126)) -> pd.DataFrame:
    """把每只标的的真实基本面(meta)+价量(prices)算成可解释的分量与综合分。
    meta[sym] = {mktcap, adv_usd, sector, price, high52, avg200, in_index:[...]}。
    热门度=近 63/126 日动量均值(缺价则用 price/avg200-1 代理);
    52周高贴近度=price/high52;龙头度=板块内相对强弱(长动量减板块中位)x0.6 + 规模 z x0.4。"""
    weights = {**DEFAULT_SCORE_WEIGHTS, **(weights or {})}
    rows = {}
    for s, m in meta.items():
        price, hi, a200 = m.get("price"), m.get("high52"), m.get("avg200")
        close = _close_of(prices, s)
        moms = [x for x in (_mom(close, lb) for lb in hot_lbs) if x == x] if close is not None else []
        if moms:
            hot = sum(moms) / len(moms)
        elif price and a200:
            hot = price / a200 - 1.0
        else:
            hot = float("nan")
        if price and hi:
            hi52 = price / hi
        elif close is not None and len(close):
            hi52 = float(close.iloc[-1] / close.max())
        else:
            hi52 = float("nan")
        rs = _mom(close, max(hot_lbs)) if close is not None else (price / a200 - 1.0 if price and a200 else float("nan"))
        rows[s] = {"mktcap": m.get("mktcap"), "adv_usd": m.get("adv_usd"), "sector": m.get("sector", "") or "",
                   "in_index": ",".join(m.get("in_index") or []), "hot": hot, "hi52": hi52, "rs": rs}
    df = pd.DataFrame(rows).T
    if df.empty:
        return df
    df["size_z"] = _z(np.log10(df["mktcap"].astype(float).clip(lower=1.0)))
    df["liq_z"] = _z(np.log10(df["adv_usd"].astype(float).clip(lower=1.0)))
    df["hot_z"] = _z(df["hot"].astype(float).fillna(df["hot"].astype(float).median()))
    df["hi52_z"] = _z(df["hi52"].astype(float).fillna(df["hi52"].astype(float).median()))
    rs = df["rs"].astype(float)
    lead_rs = rs.fillna(rs.median()).groupby(df["sector"]).transform(lambda x: x - x.median())
    df["lead_z"] = _z(lead_rs) * 0.6 + df["size_z"] * 0.4
    df["score"] = (weights["size"] * df["size_z"] + weights["liq"] * df["liq_z"]
                   + weights["hot"] * df["hot_z"] + weights["hi52"] * df["hi52_z"]
                   + weights["lead"] * df["lead_z"])
    return df.sort_values("score", ascending=False)


def build_scored_universe(meta: dict, prices: dict | None = None, per_sector=10,
                          weights: dict | None = None, cap_min=2e9, adv_min=2e7,
                          require_index=None, sectors_override: dict | None = None,
                          hot_lbs=(63, 126)) -> dict:
    """数据驱动选池:先按 市值>=cap_min / 美元日均额>=adv_min /(可选)指数成分 圈基座,
    再用软打分在每个板块内取 TopN(per_sector)。require_index 可为 'SP500' 或 ['SP500','NDX']。
    sectors_override={sym:板块} 用主题板块覆盖连接器的 GICS 行业(如 AI基建/核电电力)。"""
    meta = {s: dict(m) for s, m in meta.items()}
    if sectors_override:
        for s, sec in sectors_override.items():
            if s in meta:
                meta[s]["sector"] = sec
    df = compute_scores(meta, prices, weights, hot_lbs)
    if df.empty:
        warnings.warn("build_scored_universe: 无候选标的。", stacklevel=2)
        return {"selected": [], "gate_pass": [], "table": df, "selected_table": df, "dropped": []}
    gate = pd.Series(True, index=df.index)
    if cap_min:
        gate &= df["mktcap"].astype(float) >= cap_min
    if adv_min:
        gate &= df["adv_usd"].astype(float) >= adv_min
    if require_index:
        need = [require_index] if isinstance(require_index, str) else list(require_index)
        gate &= df["in_index"].apply(lambda x: any(i in str(x).split(",") for i in need))
    df["gate"] = gate
    chosen = []
    for sec, g in df[gate].groupby("sector"):
        g = g.sort_values("score", ascending=False)
        chosen.append(g.head(per_sector))
        if len(g) < per_sector:
            warnings.warn(f"板块[{sec}] 过基座门槛仅 {len(g)} 只(<{per_sector});已全取。", stacklevel=2)
    sel = pd.concat(chosen) if chosen else df.head(0)
    _warn_breadth(list(sel.index), {s: sel.loc[s, "sector"] for s in sel.index})
    return {"selected": list(sel.index), "gate_pass": list(df[gate].index),
            "table": df, "selected_table": sel,
            "dropped": [s for s in df.index if s not in set(sel.index)]}


def meta_from_fmp(quotes, profiles: dict | None = None, index_members: dict | None = None,
                  adv_from="volume") -> dict:
    """把行情连接器(FMP 系)的 batch-quote(+可选 profile,+可选指数成分集合)解析成 meta。
    adv_usd = price x (profile.averageVolume 优先, 否则 quote.volume)。非投资建议。"""
    profiles = profiles or {}
    index_members = index_members or {}
    meta = {}
    for q in quotes:
        s = q.get("symbol")
        if not s:
            continue
        prof = profiles.get(s, {}) or {}
        price = q.get("price") or 0.0
        vol = prof.get("averageVolume") or q.get("volume") or 0.0
        meta[s] = {"mktcap": q.get("marketCap"), "adv_usd": float(price) * float(vol),
                   "sector": prof.get("sector") or q.get("sector") or "",
                   "price": q.get("price"), "high52": q.get("yearHigh"), "low52": q.get("yearLow"),
                   "avg50": q.get("priceAvg50"), "avg200": q.get("priceAvg200"),
                   "in_index": [k for k, v in index_members.items() if s in v]}
    return meta


def merge_manual(res: dict, manual_add, tag_only=False) -> dict:
    """在数据驱动选池结果上叠加"主观 conviction"标的,不改基础池逻辑(基础池仍由
    build_scored_universe 按 市值/流动性/成分 + 软打分 产生)。manual_add=[sym,...];
    要求 sym 已在 res['table'](即有 meta+价量)。主观名即使被门槛/每板块Top截断也强制纳入,
    并标 source='manual'。返回新增 selected(数据池在前、主观按分数附后)/manual/source。"""
    tab = res.get("table")
    base = list(res.get("selected", []))
    man = list(dict.fromkeys(manual_add or []))
    if tab is None or len(tab) == 0:
        return {**res, "manual": [], "source": {s: "data" for s in base}}
    baseset = set(base); added = []
    for s in man:
        if s in tab.index and s not in baseset:
            added.append(s); baseset.add(s)
    added = sorted(added, key=lambda x: (-(float(tab.loc[x, "score"]) if tab.loc[x, "score"] == tab.loc[x, "score"] else -9)))
    selected = base + added
    if added:
        miss = [s for s in man if s not in tab.index]
        if miss:
            warnings.warn(f"merge_manual: 这些主观标的无 meta/价量,已跳过:{miss}", stacklevel=2)
    return {**res, "selected": selected, "manual": added,
            "source": {s: ("manual" if s in set(added) else "data") for s in selected}}


# ========================================================================
# Round10 §2.8:动态选池(因果)+ 反幸存者偏差 + 逐期滚动池
# ========================================================================

def dynamic_universe(meta: dict, prices: dict, *, date, cap_min=None, adv_min=None,
                     min_history: int = 60, adv_window: int = 63) -> list:
    """as-of `date` 的动态标的池——只用当日已知信息(因果性第一)。

    why:静态名单贯穿全程 = 拿"今天还活着且够大"的股票回测三年前,幸存者/
    规模偏差直接虚高业绩。这里每个 as-of 日独立判定:
      1) 截至 date 已有 >= min_history(默认60)根 K 线——未上市/刚上市的自然
         进不来,也保证因子 warmup 有数据;
      2) adv_min:截至 date 的 adv_window(默认63)日均成交额(close*volume);
         缺量价列时无法判定,保守保留(不误杀数据源缺字段的标的);
      3) cap_min:meta[sym]['mktcap'];meta 缺市值时跳过该滤条(同上保守原则)。
    注意:mktcap 若来自"当前"快照而非历史市值,严格说仍有轻微前视——调用方
    应尽量喂 as-of 市值;价格/ADV 滤条则是完全因果的。
    """
    date = pd.Timestamp(date)
    keep = []
    for s in sorted(prices):
        df = prices[s]
        if df is None or not len(df):
            continue
        hist = df[df.index <= date]
        if len(hist) < min_history:
            continue
        if adv_min is not None and {"close", "volume"}.issubset(hist.columns):
            adv = float((hist["close"] * hist["volume"]).tail(adv_window).mean())
            if np.isfinite(adv) and adv < adv_min:
                continue
        if cap_min is not None:
            cap = (meta.get(s) or {}).get("mktcap")
            if cap is not None and np.isfinite(float(cap)) and float(cap) < cap_min:
                continue
        keep.append(s)
    return keep


def anti_survivorship_pool(meta: dict, *, asof_date, listed_key: str = "list_date",
                           delisted_key: str = "delist_date"):
    """剔除 asof 后才上市、或 asof 时已退市的标的;返回 (symbols, warnings)。

    why:幸存者偏差是回测虚高的头号来源之一——用"今天还活着"的池子回测过去,
    等于偷看了谁没死。反向操作(asof 后上市的也剔)同时防了 IPO 前视。
    字段缺失时保守保留(宁可多留、让 dynamic_universe 的价格滤条兜底,也不
    静默剔除),并在 warnings 里说明,让调用方知道口径不完整、自己补数据。
    delisted_key 值为 None 视为"仍在上市"(数据源常规表示),不告警;
    键整个缺失才告警(说明数据源根本没提供退市信息)。
    """
    asof = pd.Timestamp(asof_date)
    syms, warns = [], []
    for s, m in meta.items():
        m = m or {}
        ld = m.get(listed_key)
        if ld is None:
            warns.append(f"{s}: 缺 {listed_key},无法确认 as-of 已上市,保守保留")
        elif pd.Timestamp(ld) > asof:
            continue                                    # asof 后才上市:剔除
        if delisted_key not in m:
            warns.append(f"{s}: 缺 {delisted_key} 字段,默认视为未退市,保守保留")
        else:
            dd = m.get(delisted_key)
            if dd is not None and pd.Timestamp(dd) <= asof:
                continue                                # asof 时已退市:剔除
        syms.append(s)
    return syms, warns


def rolling_universe(meta: dict, prices: dict, *, rebalance_dates, **filters) -> dict:
    """逐调仓日的动态池 {date: [symbols]}。

    每个调仓日都以当期 as-of 信息重算 dynamic_universe——池子随时间生长/收缩,
    xsec 回测按期换池,而不是一张静态名单贯穿全程(那是幸存者偏差的温床)。
    filters 原样透传给 dynamic_universe(cap_min/adv_min/min_history/adv_window)。
    """
    return {pd.Timestamp(d): dynamic_universe(meta, prices, date=d, **filters)
            for d in rebalance_dates}
