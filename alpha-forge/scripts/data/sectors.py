"""Sector / theme map + peer groups — so sentiment & news aren't purely single-name.

A stock's news-driven move is rarely idiosyncratic: the sector cycle (the whole semi
complex moving on an NVDA print) and macro (Fed/CPI/rates) usually dominate. This map
lets the news layer aggregate *peer* and *sector* headlines, not just the company's own.
Extend SECTOR_MAP for any ticker you follow; unknown tickers fall back to 'other'.
"""
from __future__ import annotations

SECTOR_MAP = {
    # semiconductors / AI chips
    "NVDA": "semiconductors", "AMD": "semiconductors", "MRVL": "semiconductors",
    "QCOM": "semiconductors", "AVGO": "semiconductors", "INTC": "semiconductors",
    "ARM": "semiconductors", "NVTS": "semiconductors", "SOXX": "semiconductors",
    "SOXL": "semiconductors",
    # optical networking / photonics
    "LITE": "optical_networking", "COHR": "optical_networking", "AAOI": "optical_networking",
    "POET": "optical_networking", "LITX": "optical_networking", "AAOX": "optical_networking",
    # mega-cap tech / internet
    "AAPL": "mega_tech", "MSFT": "mega_tech", "GOOG": "mega_tech", "GOOGL": "mega_tech",
    "META": "mega_tech", "AMZN": "mega_tech",
    # AI infrastructure / cloud GPU
    "NBIS": "ai_infra", "CRWV": "ai_infra",
    # autos / EV
    "TSLA": "autos_ev",
    # crypto / miners
    "IREN": "crypto_mining",
    # consumer staples
    "KO": "consumer_staples", "MCD": "consumer_staples",
    # financials / brokers
    "IBKR": "financials",
    # broad index ETFs (macro-driven)
    "QQQ": "index", "VOO": "index",
    # China A-share examples
    "600519": "cn_liquor", "300750": "cn_battery", "002594": "cn_auto",
    "600036": "cn_bank", "601318": "cn_insurance",
}


def sector_of(symbol: str) -> str:
    return SECTOR_MAP.get(str(symbol).upper(), SECTOR_MAP.get(str(symbol), "other"))


def peers_of(symbol: str, universe) -> list:
    """Other tickers in `universe` sharing the same sector (excluding `symbol`)."""
    s = sector_of(symbol)
    return [u for u in universe if u != symbol and sector_of(u) == s]


def group_by_sector(universe) -> dict:
    out: dict = {}
    for u in universe:
        out.setdefault(sector_of(u), []).append(u)
    return out


# ---- event propagation & sector-score mapping ------------------------------
def propagate_event(trigger_symbol: str, universe, impact: float = 1.0) -> dict:
    """A big single-name event spills over to peers (NVDA earnings -> whole semi
    complex). Returns {peer: spillover_score} for same-sector names, scaled by
    `impact` (the event's own sentiment/magnitude in [-1,1]) and a 0.5 transmission
    factor (peers move less than the source). Use it to flag peers for attention or to
    nudge their sector-sentiment layer."""
    peers = peers_of(trigger_symbol, universe)
    spill = 0.5 * float(impact)
    return {p: round(spill, 3) for p in peers}


def sector_score_map(universe, sector_scores: dict) -> dict:
    """Broadcast per-SECTOR scores to per-SYMBOL, so a sector-sentiment value can be
    used as its own factor or as the `sector` layer of composite_sentiment.

    sector_scores: {"semiconductors": 0.4, "mega_tech": -0.1, ...}
    Returns {symbol: score_of_its_sector} for every symbol in `universe`.
    """
    return {u: float(sector_scores.get(sector_of(u), 0.0)) for u in universe}


# ---- A-share AI-compute / semiconductor watchlist sector tags --------------
SECTOR_MAP.update({
    # PCB / copper-clad laminate (AI server boards)
    "300476": "cn_pcb", "002463": "cn_pcb", "600183": "cn_ccl",
    # passive components / MLCC / ceramics
    "000636": "cn_passive", "300408": "cn_ceramic_mlcc",
    # optical modules / fiber (AI datacenter interconnect)
    "300502": "cn_optical", "300308": "cn_optical", "300394": "cn_optical",
    "600487": "cn_optical_fiber", "601869": "cn_optical_fiber",
    # AI compute chips / servers
    "688041": "cn_ai_chip", "688256": "cn_ai_chip", "603019": "cn_ai_server",
    # foundry / telecom equipment
    "688981": "cn_foundry", "000063": "cn_telecom_equip",
})
