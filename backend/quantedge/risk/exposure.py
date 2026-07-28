"""Exposure monitoring: sector concentration, gross/net, breach detection."""

from __future__ import annotations

import pandas as pd

from quantedge.config import settings


def sector_exposure(weights: pd.Series, sectors: pd.Series) -> pd.DataFrame:
    """Long, short, net and gross weight per sector."""
    active = weights[weights.abs() > 1e-12]
    if active.empty:
        return pd.DataFrame()

    df = pd.DataFrame(
        {
            "ticker": active.index,
            "weight": active.to_numpy(),
            "sector": sectors.reindex(active.index).fillna("Unknown").to_numpy(),
        }
    )

    grouped = df.groupby("sector")["weight"]
    out = pd.DataFrame(
        {
            "long": grouped.apply(lambda x: x[x > 0].sum()),
            "short": grouped.apply(lambda x: x[x < 0].sum()),
            "net": grouped.sum(),
            "gross": grouped.apply(lambda x: x.abs().sum()),
            "n_positions": grouped.count(),
        }
    )
    return out.sort_values("gross", ascending=False).reset_index()


def exposure_summary(weights: pd.Series) -> dict:
    active = weights[weights.abs() > 1e-12]
    longs = active[active > 0]
    shorts = active[active < 0]

    return {
        "gross_exposure": round(float(active.abs().sum()), 4),
        "net_exposure": round(float(active.sum()), 4),
        "long_exposure": round(float(longs.sum()), 4),
        "short_exposure": round(float(shorts.sum()), 4),
        "n_positions": int(len(active)),
        "n_long": int(len(longs)),
        "n_short": int(len(shorts)),
        "largest_position": round(float(active.abs().max()), 4) if len(active) else 0.0,
        # Herfindahl index: 1/n for equal weights, 1.0 for a single holding.
        "concentration_hhi": round(float((active.abs() ** 2).sum()), 6),
    }


def check_breaches(
    weights: pd.Series,
    sectors: pd.Series | None = None,
    max_position: float = settings.max_position_weight,
    max_sector: float = settings.max_sector_weight,
    max_gross: float = 2.5,
) -> list[dict]:
    """Detect risk-limit violations. Feeds the Risk Monitor alert banner."""
    breaches: list[dict] = []
    active = weights[weights.abs() > 1e-12]
    if active.empty:
        return breaches

    over = active[active.abs() > max_position + 1e-9]
    for ticker, w in over.items():
        breaches.append(
            {
                "type": "POSITION_LIMIT",
                "severity": "HIGH",
                "subject": ticker,
                "message": f"{ticker} at {w:.2%} exceeds the {max_position:.2%} position limit",
                "value": float(w),
                "limit": max_position,
            }
        )

    gross = float(active.abs().sum())
    if gross > max_gross:
        breaches.append(
            {
                "type": "GROSS_EXPOSURE",
                "severity": "HIGH",
                "subject": "PORTFOLIO",
                "message": f"Gross exposure {gross:.2f}x exceeds the {max_gross:.2f}x limit",
                "value": gross,
                "limit": max_gross,
            }
        )

    if sectors is not None and max_sector:
        se = sector_exposure(active, sectors)
        for row in se.itertuples():
            if row.gross > max_sector + 1e-9:
                breaches.append(
                    {
                        "type": "SECTOR_CONCENTRATION",
                        "severity": "MEDIUM",
                        "subject": row.sector,
                        "message": (
                            f"Sector {row.sector} at {row.gross:.1%} gross exceeds "
                            f"the {max_sector:.0%} limit"
                        ),
                        "value": float(row.gross),
                        "limit": max_sector,
                    }
                )

    return breaches


def exposure_timeseries(weights: pd.DataFrame) -> pd.DataFrame:
    """Gross/net/long/short exposure through time."""
    if weights.empty:
        return pd.DataFrame()

    return pd.DataFrame(
        {
            "gross": weights.abs().sum(axis=1),
            "net": weights.sum(axis=1),
            "long": weights.clip(lower=0).sum(axis=1),
            "short": weights.clip(upper=0).sum(axis=1),
            "n_positions": (weights.abs() > 1e-12).sum(axis=1),
        }
    )
