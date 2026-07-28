"""Max-drawdown circuit breaker.

A stop-loss at the portfolio level: once cumulative drawdown breaches a
threshold, exposure is cut and only restored after the book recovers. This is
the "maximum drawdown constraint" in the strategy description.

Two properties matter for correctness:

* **Causal.** The breaker sees only realised drawdown up to the current bar.
* **Hysteresis.** Re-entry requires recovering past a higher level than the
  one that triggered the exit. Without that gap the book thrashes in and out
  around the threshold, paying costs each time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantedge.config import settings


@dataclass
class DrawdownGuardConfig:
    max_drawdown: float = settings.max_drawdown_limit
    #: Exposure retained once tripped (0.0 = fully flat).
    reduced_exposure: float = 0.0
    #: Drawdown must improve to this level before re-entering.
    recovery_threshold: float = 0.5
    #: Bars to stay out regardless of recovery.
    cooldown_days: int = 5

    def as_dict(self) -> dict:
        return {
            "max_drawdown": self.max_drawdown,
            "reduced_exposure": self.reduced_exposure,
            "recovery_threshold": self.recovery_threshold,
            "cooldown_days": self.cooldown_days,
        }


class DrawdownGuard:
    """Stateful circuit breaker, stepped once per bar."""

    def __init__(self, config: DrawdownGuardConfig | None = None) -> None:
        self.config = config or DrawdownGuardConfig()
        self.reset()

    def reset(self) -> None:
        self.peak_equity = -np.inf
        self.tripped = False
        self.days_since_trip = 0
        self.trip_events: list[dict] = []

    def step(self, equity: float, date=None) -> float:
        """Update state and return the exposure multiplier for this bar."""
        self.peak_equity = max(self.peak_equity, equity)
        drawdown = (equity / self.peak_equity - 1.0) if self.peak_equity > 0 else 0.0

        if not self.tripped:
            if drawdown <= -abs(self.config.max_drawdown):
                self.tripped = True
                self.days_since_trip = 0
                self.trip_events.append(
                    {"date": date, "event": "TRIPPED", "drawdown": drawdown,
                     "equity": equity}
                )
                return self.config.reduced_exposure
            return 1.0

        # Tripped: wait out the cooldown, then require genuine recovery.
        self.days_since_trip += 1
        recovery_level = -abs(self.config.max_drawdown) * self.config.recovery_threshold

        if self.days_since_trip >= self.config.cooldown_days and drawdown >= recovery_level:
            self.tripped = False
            self.trip_events.append(
                {"date": date, "event": "RECOVERED", "drawdown": drawdown,
                 "equity": equity}
            )
            return 1.0

        return self.config.reduced_exposure

    @property
    def state(self) -> dict:
        return {
            "tripped": self.tripped,
            "peak_equity": None if self.peak_equity == -np.inf else self.peak_equity,
            "days_since_trip": self.days_since_trip,
            "n_trips": sum(1 for e in self.trip_events if e["event"] == "TRIPPED"),
        }


def apply_drawdown_guard(
    returns: pd.Series, config: DrawdownGuardConfig | None = None
) -> tuple[pd.Series, pd.DataFrame]:
    """Re-simulate a return series with the breaker active.

    Returns the guarded series and a log of trip/recovery events. Applying it
    after the fact is valid because the breaker depends only on realised
    equity, never on the current bar's return.
    """
    guard = DrawdownGuard(config)
    equity = 1.0
    out: list[float] = []
    exposure = 1.0

    for date, r in returns.items():
        # Exposure was decided at the close of the previous bar.
        realised = exposure * (r if not pd.isna(r) else 0.0)
        equity *= 1.0 + realised
        out.append(realised)
        exposure = guard.step(equity, date)

    events = pd.DataFrame(guard.trip_events) if guard.trip_events else pd.DataFrame()
    return pd.Series(out, index=returns.index, name="returns_guarded"), events


def circuit_breaker_status(
    equity_curve: pd.Series, max_drawdown_limit: float = settings.max_drawdown_limit
) -> dict:
    """Current breaker state for the Risk Monitor gauge."""
    if equity_curve.empty:
        return {
            "current_drawdown": 0.0,
            "threshold": -abs(max_drawdown_limit),
            "remaining_pct": 100.0,
            "status": "INACTIVE",
        }

    peak = equity_curve.cummax()
    dd = float((equity_curve.iloc[-1] / peak.iloc[-1]) - 1.0)
    threshold = -abs(max_drawdown_limit)

    # Fraction of the drawdown budget still available.
    remaining = max(0.0, 1.0 - (dd / threshold)) if threshold < 0 else 1.0

    return {
        "current_drawdown": round(dd, 6),
        "threshold": round(threshold, 6),
        "remaining_pct": round(remaining * 100, 2),
        "status": "BREACHED" if dd <= threshold else "ACTIVE",
        "peak_equity": round(float(peak.iloc[-1]), 2),
        "current_equity": round(float(equity_curve.iloc[-1]), 2),
    }
