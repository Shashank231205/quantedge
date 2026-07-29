"""Precompute factor output so the API never has to.

Building the composite needs the full price panel resident -- roughly 270MB on
top of a ~240MB library baseline. That fits comfortably on a workstation and
not at all on a 512MB instance, where it OOMs the process on the first request
to the Factor Explorer.

So the work moves to where the memory exists. This module runs during seeding
and after each ingest, writes the result to `factor_snapshots` and
`factor_diagnostics`, and the API reads rows. Beyond fitting in memory it is
the more honest shape: the factor scores for a given as-of date are a fact
about that date, not something to recompute per page view.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
from sqlalchemy import delete, select

from quantedge.db.models import FactorDiagnostic, FactorSnapshot
from quantedge.db.session import session_scope
from quantedge.logging_config import get_logger

log = get_logger(__name__)

#: How many trailing as-of dates to keep. The screens only ever show the
#: latest, but a short history makes it possible to see whether the ranking is
#: stable without storing six years of every ticker.
KEEP_SNAPSHOTS = 5


def build_snapshot(top_n: int = 600) -> dict:
    """Compute the current factor state and persist it.

    Returns a summary rather than the frame -- the caller is a CLI step, and
    the data it just wrote is the point, not the return value.
    """
    from quantedge.factors.composite import CompositeFactor
    from quantedge.strategy import DEFAULT_SPEC, load_panel

    panel, _ = load_panel()
    if panel.empty:
        log.warning("factors.snapshot_skipped reason=empty_panel")
        return {"rows": 0, "as_of": None}

    composite = CompositeFactor(
        weights=DEFAULT_SPEC.factor_weights, orientations={"volatility": -1}
    )
    blended = composite.compute(panel)
    components = composite.component_signals(panel)

    as_of = pd.Timestamp(blended.index.max()).date()
    latest = blended.iloc[-1].dropna().sort_values(ascending=False)

    # The long book is the top decile; everything else is ranked but not held.
    cutoff = max(1, int(len(latest) * DEFAULT_SPEC.long_quantile))

    rows = []
    for rank, (ticker, score) in enumerate(latest.head(top_n).items()):
        per_factor = {}
        for name, frame in (components or {}).items():
            try:
                value = frame.iloc[-1].get(ticker)
                if pd.notna(value):
                    per_factor[f"{name}_rank"] = round(float(value), 4)
            except Exception:  # pragma: no cover - a missing factor is not fatal
                continue

        rows.append(
            {
                "as_of": as_of,
                "ticker": str(ticker),
                "composite_score": round(float(score), 6),
                "components": per_factor or None,
                "bias": "LONG" if rank < cutoff else "NEUTRAL",
            }
        )

    with session_scope() as s:
        s.execute(delete(FactorSnapshot).where(FactorSnapshot.as_of == as_of))
        s.bulk_insert_mappings(FactorSnapshot, rows)

        # Trim history so this table cannot grow without bound.
        kept = s.scalars(
            select(FactorSnapshot.as_of)
            .distinct()
            .order_by(FactorSnapshot.as_of.desc())
            .limit(KEEP_SNAPSHOTS)
        ).all()
        if kept:
            s.execute(delete(FactorSnapshot).where(FactorSnapshot.as_of.notin_(kept)))

    # The Risk screen needs the live book's weights and per-name volatility.
    # Deriving them here costs nothing extra -- the panel is already loaded --
    # and saves the API from running a full backtest per request.
    try:
        from quantedge.backtest.vectorized import VectorizedBacktestEngine
        from quantedge.risk.position_sizing import realized_volatility

        engine = VectorizedBacktestEngine(
            config=DEFAULT_SPEC.portfolio_config(), cost_model=DEFAULT_SPEC.cost_model()
        )
        result = engine.run(panel, blended)
        weights = result.weights.iloc[-1]
        vols = realized_volatility(panel, window=60).iloc[-1]

        store_diagnostic(
            "live_book",
            {
                "as_of": str(as_of),
                "weights": {
                    str(k): round(float(v), 6)
                    for k, v in weights.items()
                    if abs(float(v)) > 1e-9
                },
                "volatility": {
                    str(k): round(float(v), 6)
                    for k, v in vols.items()
                    if pd.notna(v)
                },
            },
            as_of=as_of,
        )
        log.info("factors.live_book_written positions=%s", int((weights.abs() > 1e-9).sum()))
    except Exception:  # pragma: no cover - risk view degrades, snapshot stands
        log.warning("factors.live_book_failed", exc_info=True)

    log.info("factors.snapshot_written as_of=%s rows=%s", as_of, len(rows))
    return {"rows": len(rows), "as_of": str(as_of)}


def store_diagnostic(kind: str, payload: dict, as_of: date | None = None) -> None:
    """Persist an IC or correlation result for the API to serve."""
    as_of = as_of or date.today()
    with session_scope() as s:
        s.execute(
            delete(FactorDiagnostic).where(
                FactorDiagnostic.kind == kind, FactorDiagnostic.as_of == as_of
            )
        )
        s.add(FactorDiagnostic(as_of=as_of, kind=kind, payload=payload))


def read_snapshot(limit: int = 600) -> dict:
    """The stored ranking, newest as-of date first. Reads rows, computes nothing."""
    with session_scope() as s:
        as_of = s.scalar(select(FactorSnapshot.as_of).order_by(FactorSnapshot.as_of.desc()).limit(1))
        if as_of is None:
            return {"as_of": None, "rows": [], "n_total": 0}

        rows = s.execute(
            select(
                FactorSnapshot.ticker,
                FactorSnapshot.composite_score,
                FactorSnapshot.components,
                FactorSnapshot.bias,
            )
            .where(FactorSnapshot.as_of == as_of)
            .order_by(FactorSnapshot.composite_score.desc())
            .limit(limit)
        ).all()

        total = s.scalar(
            select(FactorSnapshot.id)
            .where(FactorSnapshot.as_of == as_of)
            .order_by(FactorSnapshot.id.desc())
            .limit(1)
        )

    return {
        "as_of": str(as_of),
        "n_total": len(rows) if total is None else len(rows),
        "rows": [
            {
                "ticker": r.ticker,
                "composite_score": r.composite_score,
                "bias": r.bias,
                **(r.components or {}),
            }
            for r in rows
        ],
    }


def read_diagnostic(kind: str) -> dict | None:
    with session_scope() as s:
        row = s.scalars(
            select(FactorDiagnostic)
            .where(FactorDiagnostic.kind == kind)
            .order_by(FactorDiagnostic.as_of.desc())
            .limit(1)
        ).first()
        return row.payload if row else None
