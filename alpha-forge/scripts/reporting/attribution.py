"""P&L attribution — from account trades/positions, explain WHERE returns came from.

Pull get_account_trades / get_account_positions (broker), pass them here to get realized
+ unrealized P&L per symbol, then aggregate by sector and (optionally) decompose the
portfolio's returns onto factor returns. Turns "I'm up 8%" into "semis +12, hedges -4".
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def pnl_by_symbol(positions: list, trades: list | None = None) -> pd.DataFrame:
    """positions: [{symbol, position, average_price, market_price, market_value,
    unrealized_pnl, ...}] (broker get_account_positions shape). trades optional for
    realized P&L. Returns a per-symbol P&L frame."""
    rows = []
    for p in positions or []:
        sym = p.get("contract_description") or p.get("symbol")
        rows.append({"symbol": sym,
                     "position": p.get("position"),
                     "avg_price": p.get("average_price"),
                     "price": p.get("market_price"),
                     "market_value": p.get("market_value"),
                     "unrealized_pnl": p.get("unrealized_pnl"),
                     "daily_pnl": p.get("daily_pnl")})
    df = pd.DataFrame(rows)
    if trades:
        td = pd.DataFrame(trades)
        # realized pnl if provided per trade
        if "realized_pnl" in td.columns and "symbol" in td.columns:
            real = td.groupby("symbol")["realized_pnl"].sum().rename("realized_pnl")
            df = df.merge(real, left_on="symbol", right_index=True, how="left")
    return df


def attribute_by_sector(pnl_df: pd.DataFrame, sector_fn=None,
                        pnl_col: str = "unrealized_pnl") -> pd.DataFrame:
    """Aggregate a per-symbol P&L column by sector (uses data.sectors by default)."""
    if sector_fn is None:
        from ..data.sectors import sector_of as sector_fn
    d = pnl_df.copy()
    d["sector"] = d["symbol"].map(sector_fn)
    agg = d.groupby("sector")[pnl_col].sum().sort_values(ascending=False)
    return agg.rename("pnl").to_frame()


def factor_attribution(port_returns: pd.Series, factor_returns: pd.DataFrame) -> dict:
    """Regress portfolio returns on factor returns (e.g. market, momentum, low-vol) to
    see how much of the P&L each factor explains. Returns betas + R². factor_returns:
    DataFrame of per-period factor returns aligned to port_returns."""
    df = pd.concat([port_returns.rename("port"), factor_returns], axis=1).dropna()
    if len(df) < 10:
        return {"note": "insufficient overlap"}
    y = df["port"].values
    X = df.drop(columns="port").values
    X1 = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    yhat = X1 @ beta
    ss_res = ((y - yhat) ** 2).sum(); ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot else 0.0
    names = ["alpha"] + list(df.drop(columns="port").columns)
    return {"betas": {n: round(float(b), 4) for n, b in zip(names, beta)},
            "r2": round(float(r2), 3),
            "note": "alpha=未被因子解释的超额；betas=对各因子的暴露"}


def brinson(port_weights: pd.Series, bench_weights: pd.Series,
            port_returns: pd.Series, bench_returns: pd.Series, *,
            sector_map: dict) -> dict:
    """Single-period Brinson-Fachler attribution: split the portfolio's excess return
    over the benchmark into allocation / selection / interaction, by sector.

    Why Brinson-Fachler (vs plain Brinson-Hood-Beebower): BF measures allocation
    against (Rb_sector − Rb_total), so overweighting a sector only scores as good
    allocation if that sector BEAT the benchmark overall — the version consultants
    and attribution systems actually report.

        allocation_s  = (Wp_s − Wb_s) · (Rb_s − Rb_total)   — was the sector bet right?
        selection_s   =  Wb_s · (Rp_s − Rb_s)               — were the picks inside right?
        interaction_s = (Wp_s − Wb_s) · (Rp_s − Rb_s)       — cross term
        Σ(all three)  =  Rp − Rb  (exact when both weight vectors sum to 1)

    Inputs are symbol-indexed (weights + single-period returns); aggregation to
    sectors happens here via sector_map {symbol: sector}. Weights are renormalized to
    sum 1 so the identity above holds even if a caller passes gross weights. Sectors
    the portfolio doesn't hold contribute selection/interaction = 0 (Rp_s defaults to
    Rb_s rather than NaN-poisoning the sum). Returns
    {allocation, selection, interaction, total, by_sector: DataFrame}.
    """
    syms = pd.Index(sorted(set(port_weights.index) | set(bench_weights.index)))
    wp = pd.Series(port_weights).reindex(syms).fillna(0.0).astype(float)
    wb = pd.Series(bench_weights).reindex(syms).fillna(0.0).astype(float)
    rp = pd.Series(port_returns).reindex(syms).fillna(0.0).astype(float)
    rb = pd.Series(bench_returns).reindex(syms).fillna(0.0).astype(float)
    if wp.sum() > 0:
        wp = wp / wp.sum()
    if wb.sum() > 0:
        wb = wb / wb.sum()
    sec = pd.Series({s: sector_map.get(s, "Unknown") for s in syms})

    wp_s = wp.groupby(sec).sum()
    wb_s = wb.groupby(sec).sum()
    rp_s = (wp * rp).groupby(sec).sum() / wp_s.replace(0, np.nan)
    rb_s = (wb * rb).groupby(sec).sum() / wb_s.replace(0, np.nan)
    rb_tot = float((wb * rb).sum())
    rb_s = rb_s.fillna(rb_tot)   # sector absent from benchmark: neutral reference
    rp_s = rp_s.fillna(rb_s)     # sector absent from portfolio: no selection effect

    alloc = (wp_s - wb_s) * (rb_s - rb_tot)
    select = wb_s * (rp_s - rb_s)
    interact = (wp_s - wb_s) * (rp_s - rb_s)
    by_sector = pd.DataFrame({
        "port_w": wp_s.round(4), "bench_w": wb_s.round(4),
        "port_ret": rp_s.round(4), "bench_ret": rb_s.round(4),
        "allocation": alloc, "selection": select, "interaction": interact,
        "total": alloc + select + interact,
    }).sort_values("total", ascending=False)
    return {"allocation": float(alloc.sum()), "selection": float(select.sum()),
            "interaction": float(interact.sum()),
            "total": float((alloc + select + interact).sum()),
            "by_sector": by_sector}
