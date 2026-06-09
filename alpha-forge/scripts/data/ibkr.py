"""IBKR-style brokerage MCP adapter.

IMPORTANT: the brokerage data lives behind an MCP connector that only *Claude*
can call (search_contracts / get_price_history / get_price_snapshot ...). A plain
Python script in the sandbox cannot reach it. So the workflow is a hand-off:

    1. Claude calls the MCP tools and saves the raw JSON to a file, OR pastes the
       columnar payload into a dict.
    2. This module turns that payload into the canonical OHLCV frame.

See references/data_sources.md for the exact Claude-side steps and the period/step
cheat-sheet. Keeping the parsing here means the rest of the pipeline treats broker
data identically to free data.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .base import from_columnar, validate_ohlcv


def from_mcp_payload(payload: dict, *, name: str = "ibkr") -> pd.DataFrame:
    """Parse the dict returned by the brokerage get_price_history tool.

    The payload has parallel arrays: time/open/high/low/close/volume, plus metadata
    like chart_step, delayed, source. We only need the OHLCV arrays.
    """
    return from_columnar(payload, name=name)


def from_mcp_json_file(path: str | Path, *, name: str | None = None) -> pd.DataFrame:
    """Load broker history that Claude previously dumped to a .json file."""
    path = Path(path)
    payload = json.loads(path.read_text())
    return from_columnar(payload, name=name or f"ibkr:{path.stem}")


# Period/step values accepted by the brokerage MCP, surfaced here so Claude (or a
# user reading the code) doesn't have to guess. These are passed to the MCP tool,
# not used by Python directly.
MCP_PERIODS = [
    "ONE_DAY", "TWO_DAYS", "THREE_DAYS", "ONE_WEEK", "TWO_WEEKS",
    "ONE_MONTH", "THREE_MONTHS", "SIX_MONTHS", "ONE_YEAR", "TWO_YEARS", "FIVE_YEARS",
]
MCP_STEPS = [
    "THIRTY_SECS", "ONE_MIN", "TWO_MINS", "FIVE_MINS", "TEN_MINS", "FIFTEEN_MINS",
    "THIRTY_MINS", "ONE_HOUR", "TWO_HOURS", "FOUR_HOURS", "ONE_DAY", "ONE_WEEK", "ONE_MONTH",
]
