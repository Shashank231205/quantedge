"""What INU can look up.

Each tool reads the same functions the screens read, so an answer about the
Sharpe ratio comes from the row the Backtest screen renders rather than from
anything the model remembers. That is the whole point: the model supplies
language, the platform supplies facts.

Tools are deliberately coarse -- one per screen rather than one per field.
A model choosing between six well-described tools is reliable; choosing
between forty is not, and the extra calls cost latency the chat cannot spare.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from sqlalchemy import func, select

from quantedge.db.models import BacktestRun, OhlcvClean, Security
from quantedge.db.session import session_scope
from quantedge.logging_config import get_logger

log = get_logger(__name__)


def _latest_run() -> BacktestRun | None:
    with session_scope() as s:
        return s.scalars(
            select(BacktestRun)
            .where(BacktestRun.is_walk_forward.is_(True))
            .order_by(BacktestRun.created_at.desc())
            .limit(1)
        ).first()


def get_performance() -> dict:
    """Headline performance and risk figures for the validated run."""
    with session_scope() as s:
        run = s.scalars(
            select(BacktestRun)
            .where(BacktestRun.is_walk_forward.is_(True))
            .order_by(BacktestRun.created_at.desc())
            .limit(1)
        ).first()
        if run is None:
            return {"error": "No walk-forward run exists yet."}
        m = run.metrics or {}
        return {
            "run_name": run.name,
            "period": {"start": str(run.start_date), "end": str(run.end_date)},
            "is_out_of_sample": run.is_walk_forward,
            "risk_adjusted": m.get("risk_adjusted", {}),
            "returns": m.get("returns", {}),
            "risk": m.get("risk", {}),
            "benchmark_relative": m.get("benchmark_relative", {}),
            "note": (
                "Walk-forward out-of-sample, net of 1bp commission and 5bp "
                "slippage per side."
            ),
        }


def get_significance() -> dict:
    """Whether the edge survives multiple-testing correction.

    Separated from performance on purpose. A caller asking 'is this good'
    needs this answer, and burying it inside a metrics blob invites the model
    to quote the headline Sharpe without the caveat that undoes it.
    """
    run = _latest_run()
    if run is None:
        return {"error": "No walk-forward run exists yet."}
    ds = (run.metrics or {}).get("deflated_sharpe", {})
    return {
        "deflated_sharpe": ds.get("deflated_sharpe"),
        "observed_sharpe": ds.get("observed_sharpe"),
        "expected_max_sharpe_from_trials": ds.get("expected_max_sharpe"),
        "n_trials": ds.get("n_trials"),
        "n_observations": ds.get("n_obs"),
        "is_significant": ds.get("is_significant"),
        "decision_rule": "Deflated Sharpe must exceed 0.95 to be significant.",
        "plain_meaning": (
            "The deflated Sharpe corrects the headline Sharpe for how many "
            "strategy variants were tried. Below 0.95 the result cannot be "
            "distinguished from the best of several lucky draws."
        ),
    }


def get_factors() -> dict:
    """The factors in the composite, their weights, and their IC diagnostics."""
    from quantedge.strategy import DEFAULT_SPEC

    spec = DEFAULT_SPEC.as_dict()
    out: dict[str, Any] = {
        "factors": spec.get("factors"),
        "weights": spec.get("weights"),
        "orientations": spec.get("orientations"),
        "rebalance": spec.get("rebalance_frequency"),
    }
    try:
        from quantedge.factors.diagnostics import load_diagnostics

        out["ic_diagnostics"] = load_diagnostics()
    except Exception:  # pragma: no cover - diagnostics are supplementary
        out["ic_diagnostics"] = "Not computed. Run `quantedge factors`."
    return out


def get_signals(top_n: int = 10) -> dict:
    """The current factor ranking -- which names the strategy favours today."""
    from quantedge.strategy import current_signals

    try:
        df = current_signals(top_n=min(top_n, 50))
        with session_scope() as s:
            as_of = s.scalar(select(func.max(OhlcvClean.date)))
        return {
            "as_of": str(as_of) if as_of else None,
            "signals": df.to_dict("records") if not df.empty else [],
            "note": "Ranked by composite factor score. Not a recommendation.",
        }
    except Exception as exc:
        return {"error": f"Ranking unavailable: {exc}"}


def get_risk() -> dict:
    """Live exposure, concentration and circuit-breaker state."""
    try:
        from quantedge.api.routers.risk import summary as risk_summary

        return risk_summary()
    except Exception as exc:
        return {"error": f"Risk state unavailable: {exc}"}


def get_data_coverage() -> dict:
    """What data the platform holds: tickers, rows, date range."""
    with session_scope() as s:
        return {
            "total_rows": s.scalar(select(func.count()).select_from(OhlcvClean)) or 0,
            "n_tickers": s.scalar(
                select(func.count(func.distinct(OhlcvClean.ticker)))
            )
            or 0,
            "n_securities": s.scalar(select(func.count()).select_from(Security)) or 0,
            "first_date": str(s.scalar(select(func.min(OhlcvClean.date)))),
            "last_date": str(s.scalar(select(func.max(OhlcvClean.date)))),
            "source": "yfinance, point-in-time S&P 500 membership",
        }


def get_methodology() -> dict:
    """How the platform works: definitions, formulas, and design choices.

    Answers 'what is a Sharpe ratio' or 'how do you avoid lookahead bias'
    without a web call, and in this platform's own terms rather than a
    generic textbook's.
    """
    return {
        "formulas": {
            "sharpe_ratio": "(annualised return - risk-free) / annualised volatility",
            "sortino_ratio": "(annualised return - risk-free) / downside deviation",
            "calmar_ratio": "annualised return / abs(max drawdown)",
            "information_ratio": "(portfolio return - benchmark return) / tracking error",
            "profit_factor": "gross profit / gross loss",
            "deflated_sharpe": (
                "Probability the observed Sharpe exceeds what the best of N "
                "trials would produce by chance, adjusted for skew, kurtosis "
                "and sample length (Bailey & Lopez de Prado)."
            ),
            "ulcer_index": "root mean square of drawdown depth over the period",
        },
        "pipeline": [
            "Reconstruct point-in-time S&P 500 membership so delisted names "
            "are present in the periods they actually traded.",
            "Ingest and clean daily OHLCV, dropping bad bars and flagging "
            "extreme returns.",
            "Compute momentum, volatility and mean-reversion factors, then "
            "cross-sectionally rank and blend them into a composite.",
            "Walk forward: train on 2 years, embargo 10 days, test on 6 "
            "months, roll. Parameters are chosen inside each training window "
            "and never with knowledge of the test window.",
            "Score the result honestly, including the multiple-testing "
            "correction that the headline Sharpe does not survive.",
        ],
        "bias_controls": {
            "survivorship": (
                "Point-in-time membership. 46 of 612 ever-members have no "
                "available history, which is a residual bias the README states."
            ),
            "lookahead": (
                "Factors use only data available at the rebalance date; an "
                "embargo separates train and test windows. Enforced by tests."
            ),
            "costs": "1bp commission and 5bp slippage per side, inside returns.",
        },
        "screens": {
            "Dashboard": "Headline performance, equity curve vs SPY, live signals.",
            "Factors": "Per-factor IC, decay, cross-factor correlation, ticker detail.",
            "Backtest": "Walk-forward folds, trade log, full metric set.",
            "Risk": "Exposure, concentration, VaR, circuit-breaker state.",
            "Pipeline": "Ingestion health, job history, API latency.",
            "Analyst": "Scored assessment with reasons, gaps and citations.",
        },
    }


#: Tool name -> (callable, JSON schema). The schema is what the model sees, so
#: descriptions are written for it: what the tool answers, and when to reach
#: for it rather than another.
TOOLS: dict[str, tuple[Callable[..., dict], dict]] = {
    "get_performance": (
        get_performance,
        {
            "type": "function",
            "function": {
                "name": "get_performance",
                "description": (
                    "Headline out-of-sample performance and risk for the "
                    "validated strategy run: Sharpe, Sortino, Calmar, returns, "
                    "volatility, drawdown, and comparison to SPY. Use for any "
                    "question about how the strategy performed."
                ),
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ),
    "get_significance": (
        get_significance,
        {
            "type": "function",
            "function": {
                "name": "get_significance",
                "description": (
                    "Whether the strategy's edge is statistically real: "
                    "deflated Sharpe, trial count, and the significance "
                    "verdict. Use whenever asked if the strategy is good, "
                    "works, or is trustworthy -- performance alone does not "
                    "answer that."
                ),
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ),
    "get_factors": (
        get_factors,
        {
            "type": "function",
            "function": {
                "name": "get_factors",
                "description": (
                    "The factors driving the strategy, their weights and "
                    "orientations, and their information-coefficient "
                    "diagnostics. Use for questions about what the strategy "
                    "trades on or which factor works best."
                ),
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ),
    "get_signals": (
        get_signals,
        {
            "type": "function",
            "function": {
                "name": "get_signals",
                "description": (
                    "Today's ranked names by composite factor score. Use when "
                    "asked what the strategy currently favours or which "
                    "tickers rank highest."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "top_n": {
                            "type": "integer",
                            "description": "How many names to return, max 50.",
                        }
                    },
                },
            },
        },
    ),
    "get_risk": (
        get_risk,
        {
            "type": "function",
            "function": {
                "name": "get_risk",
                "description": (
                    "Current risk state: gross and net exposure, position "
                    "concentration, VaR, and how close the portfolio sits to "
                    "its drawdown circuit breaker."
                ),
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ),
    "get_data_coverage": (
        get_data_coverage,
        {
            "type": "function",
            "function": {
                "name": "get_data_coverage",
                "description": (
                    "What market data the platform holds: row count, ticker "
                    "count, and date range. Use for questions about data size, "
                    "history length or coverage."
                ),
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ),
    "get_methodology": (
        get_methodology,
        {
            "type": "function",
            "function": {
                "name": "get_methodology",
                "description": (
                    "Definitions, formulas and design choices: what each metric "
                    "means, how the pipeline runs, how lookahead and "
                    "survivorship bias are controlled, and what each screen "
                    "shows. Use for 'what is', 'how does', and 'why' questions "
                    "about the platform or its metrics."
                ),
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ),
}


def schemas() -> list[dict]:
    return [schema for _, schema in TOOLS.values()]


def execute(name: str, arguments: str | dict) -> str:
    """Run a tool and return its result as JSON text for the model.

    Errors are returned rather than raised: a tool that fails should let the
    model say so in context, not abort the whole turn.
    """
    entry = TOOLS.get(name)
    if entry is None:
        return json.dumps({"error": f"Unknown tool: {name}"})

    fn, _ = entry
    try:
        args = json.loads(arguments) if isinstance(arguments, str) else (arguments or {})
    except json.JSONDecodeError:
        args = {}

    try:
        return json.dumps(fn(**args), default=str)
    except TypeError:
        # Model supplied arguments the tool does not take.
        try:
            return json.dumps(fn(), default=str)
        except Exception as exc:  # pragma: no cover
            return json.dumps({"error": str(exc)})
    except Exception as exc:
        log.warning("inu.tool_failed name=%s %s", name, exc)
        return json.dumps({"error": f"{name} failed: {exc}"})
