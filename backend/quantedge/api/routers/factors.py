"""Factor Explorer endpoints (screen 2)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select

from quantedge.api.deps import cache, require_api_key
from quantedge.db.models import Security
from quantedge.db.session import session_scope
from quantedge.factors.composite import CORE_FACTORS, CompositeFactor, get_factor
from quantedge.factors.diagnostics import (
    cross_factor_correlation,
    ic_decay,
)
from quantedge.logging_config import get_recent_logs
from quantedge.strategy import FACTOR_WEIGHTS, load_panel

router = APIRouter(prefix="/factors", tags=["factors"])

PRODUCTION_ORIENTATION = {"volatility": -1}


def _signals() -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    """Price panel, per-factor signals and the composite (cached)."""
    cached = cache.get("factor_signals")
    if cached is not None:
        return cached

    panel, _ = load_panel()
    composite = CompositeFactor(
        weights=FACTOR_WEIGHTS, orientations=PRODUCTION_ORIENTATION
    )
    components = composite.component_signals(panel)
    blended = composite.compute(panel)

    payload = (panel, components, blended)
    cache.set("factor_signals", payload)
    return payload


def warm_cache() -> None:
    """Compute factor signals ahead of the first request.

    Building the panel and every factor takes several seconds; doing it lazily
    made the first Factor Explorer request an outlier that dominated the p95.
    """
    try:
        _signals()
        _sector_map()
    except Exception as exc:  # a cold cache must not stop the API booting
        from quantedge.logging_config import get_logger

        get_logger(__name__).warning("factors.warm_cache_failed error=%s", exc)


def _sector_map() -> dict[str, str]:
    cached = cache.get("sector_map")
    if cached is None:
        with session_scope() as s:
            rows = s.scalars(select(Security)).all()
        cached = {r.ticker: r.sector for r in rows if r.sector}
        cache.set("sector_map", cached)
    return cached


@router.get("/table", dependencies=[Depends(require_api_key)])
def factor_table(
    limit: int = Query(default=100, le=600),
    sort_by: str = Query(default="composite"),
    descending: bool = True,
    search: str | None = None,
) -> dict:
    """Sortable per-ticker factor ranks as of the latest bar."""
    panel, components, blended = _signals()
    sectors = _sector_map()

    latest_date = panel.index[-1]
    rows = []
    for ticker in panel.columns:
        composite_score = blended.iloc[-1].get(ticker, np.nan)
        if pd.isna(composite_score):
            continue
        row = {
            "ticker": ticker,
            "sector": sectors.get(ticker),
            "composite": round(float(composite_score), 4),
        }
        for name, sig in components.items():
            value = sig.iloc[-1].get(ticker, np.nan)
            row[name] = round(float(value), 4) if pd.notna(value) else None
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return {"as_of": str(latest_date.date()), "n_total": 0, "rows": []}

    if search:
        df = df[df["ticker"].str.contains(search.upper(), na=False)]

    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=not descending, na_position="last")

    return {
        "as_of": str(latest_date.date()),
        "n_total": len(df),
        "universe_size": int(panel.shape[1]),
        "factors": list(components),
        "rows": df.head(limit).to_dict("records"),
    }


@router.get("/correlation", dependencies=[Depends(require_api_key)])
def correlation_matrix() -> dict:
    """Average cross-sectional correlation between factor signals."""
    cached = cache.get("factor_correlation")
    if cached is not None:
        return cached

    _, components, _ = _signals()
    matrix = cross_factor_correlation(components)

    payload = {
        "factors": list(matrix.columns),
        "matrix": matrix.round(4).to_dict(),
        "note": "Spearman rank correlation, averaged across dates.",
    }
    cache.set("factor_correlation", payload)
    return payload


@router.get("/ic", dependencies=[Depends(require_api_key)])
def factor_ic(
    horizons: str = Query(default="1,5,10,21,42,63"),
) -> dict:
    """Information Coefficient and its decay curve per factor."""
    key = f"factor_ic:{horizons}"
    cached = cache.get(key)
    if cached is not None:
        return cached

    panel, components, blended = _signals()
    try:
        parsed = tuple(int(h) for h in horizons.split(",") if h.strip())
    except ValueError:
        raise HTTPException(400, "horizons must be comma-separated integers") from None

    out = {}
    for name, sig in components.items():
        out[name] = ic_decay(sig, panel, horizons=parsed, factor=name).to_dict("records")
    out["composite"] = ic_decay(blended, panel, horizons=parsed, factor="composite").to_dict(
        "records"
    )

    payload = {
        "horizons": list(parsed),
        "ic_by_factor": out,
        "note": (
            "Spearman IC of the one-bar-lagged signal against forward returns. "
            "A mean IC of 0.02-0.05 is typical for a daily equity factor."
        ),
    }
    cache.set(key, payload)
    return payload


@router.get("/{ticker}/detail", dependencies=[Depends(require_api_key)])
def ticker_detail(ticker: str, lookback: int = Query(default=252, le=1500)) -> dict:
    """Factor history for one ticker — the detail pane."""
    panel, components, blended = _signals()
    ticker = ticker.upper()

    if ticker not in panel.columns:
        raise HTTPException(404, f"{ticker} is not in the universe")

    sectors = _sector_map()
    window = slice(-lookback, None)
    dates = [str(d.date()) for d in panel.index[window]]

    history = {"date": dates}
    for name, sig in components.items():
        series = sig[ticker].iloc[window]
        history[name] = [None if pd.isna(v) else round(float(v), 4) for v in series]
    history["composite"] = [
        None if pd.isna(v) else round(float(v), 4) for v in blended[ticker].iloc[window]
    ]

    prices = panel[ticker].iloc[window]
    return {
        "ticker": ticker,
        "sector": sectors.get(ticker),
        "as_of": dates[-1] if dates else None,
        "current": {
            name: (None if pd.isna(sig.iloc[-1].get(ticker)) else round(float(sig.iloc[-1][ticker]), 4))
            for name, sig in components.items()
        },
        "composite": None
        if pd.isna(blended.iloc[-1].get(ticker))
        else round(float(blended.iloc[-1][ticker]), 4),
        "price_history": [
            {"date": d, "close": round(float(p), 4)}
            for d, p in zip(dates, prices, strict=False)
            if pd.notna(p)
        ],
        "factor_history": history,
    }


@router.get("/list", dependencies=[Depends(require_api_key)])
def list_factors() -> dict:
    """Registry of available factors and the production weighting."""
    return {
        "core_factors": list(CORE_FACTORS),
        "production_weights": FACTOR_WEIGHTS,
        "production_orientations": PRODUCTION_ORIENTATION,
        "available": [get_factor(name).describe() for name in FACTOR_WEIGHTS],
        "note": (
            "The volatility factor runs inverted: the low-volatility anomaly "
            "reversed over 2020-2026. Orientation is selected on training data "
            "inside walk-forward, not fixed from full-sample hindsight."
        ),
    }


@router.get("/diagnostics/logs", dependencies=[Depends(require_api_key)])
def diagnostic_logs(limit: int = Query(default=30, le=200)) -> dict:
    return {"logs": get_recent_logs(limit=limit)}
