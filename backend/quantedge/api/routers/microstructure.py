"""Microstructure screen endpoints.

The study runs offline (``quantedge micro study``) over roughly 300 million
quote updates, so the API serves its published results rather than recomputing
anything per request. The document is read once and cached for the life of
the process; a redeploy picks up a regenerated study.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query

from quantedge.api.deps import require_api_key

router = APIRouter(
    prefix="/microstructure",
    tags=["microstructure"],
    dependencies=[Depends(require_api_key)],
)

STUDY_PATH = Path(__file__).resolve().parents[2] / "microstructure" / "study.json"


@lru_cache(maxsize=1)
def load_study() -> dict:
    if not STUDY_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="No microstructure study published. Run `quantedge micro study`.",
        )
    return json.loads(STUDY_PATH.read_text())


def _symbol(study: dict, symbol: str) -> dict:
    try:
        return study["symbols"][symbol.upper()]
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"{symbol} is not in the study; available: {sorted(study['symbols'])}",
        ) from None


@router.get("/summary")
def summary() -> dict:
    """Cross-asset headline table plus each symbol's dataset description."""
    study = load_study()
    return {
        "symbols": sorted(study["symbols"]),
        "cross_asset": study["cross_asset"],
        "datasets": {s: v["dataset"] for s, v in study["symbols"].items()},
    }


@router.get("/study/{symbol}")
def symbol_study(
    symbol: str,
    include_hourly: bool = Query(default=False, description="Include hour-by-hour IC"),
) -> dict:
    """Full research output for one symbol."""
    data = dict(_symbol(load_study(), symbol))
    if not include_hourly:
        data["stability"] = {k: v for k, v in data["stability"].items() if k != "hourly"}
    data.pop("manifest", None)
    return data


@router.get("/stability/{symbol}")
def stability(symbol: str) -> dict:
    """Hour-by-hour IC series, for the stability chart."""
    return _symbol(load_study(), symbol)["stability"]
