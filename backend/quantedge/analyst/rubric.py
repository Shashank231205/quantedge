"""Deterministic per-metric scoring.

The analyst agent writes prose, not verdicts. Every score in the report is
computed here from thresholds that are visible in the source, so the same run
always scores the same and any number in the report can be traced back to a
comparison a reader can check.

Thresholds are the conventional bars used in manager evaluation (Sharpe > 1 is
respectable, > 2 is rare and usually a sign of undisclosed leverage or a short
sample). Where a metric has a real statistical decision rule rather than a
convention -- deflated Sharpe -- the rule is used and the convention ignored.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Citation:
    """A number the report cites, with enough context to verify it.

    `points` are the supporting figures shown when the citation is expanded --
    they exist so a reader can see what the headline number is made of without
    leaving the report.
    """

    title: str
    description: str
    points: list[str]
    source: str  # which screen/endpoint the figure came from


@dataclass
class MetricScore:
    key: str
    label: str
    score: int  # 0-100
    band: str  # STRONG | ADEQUATE | WEAK | FAILING
    value: float | None
    citations: list[Citation] = field(default_factory=list)
    # Facts the agent must build its prose from. Not prose itself -- the agent
    # turns these into reasons/gaps. Anything not in here is not licensed.
    evidence: dict[str, Any] = field(default_factory=dict)


def _band(score: int) -> str:
    if score >= 80:
        return "STRONG"
    if score >= 60:
        return "ADEQUATE"
    if score >= 40:
        return "WEAK"
    return "FAILING"


def _scale(value: float, floor: float, ceiling: float) -> int:
    """Linear 0-100 between two thresholds, clamped at both ends."""
    if ceiling == floor:
        return 50
    pct = (value - floor) / (ceiling - floor)
    return max(0, min(100, round(pct * 100)))


def score_risk_adjusted(m: dict) -> MetricScore:
    ra = m.get("risk_adjusted", {})
    sharpe = ra.get("sharpe_ratio")
    if sharpe is None:
        return MetricScore("risk_adjusted", "Risk-Adjusted Return", 0, "FAILING", None)

    # 0.0 -> 0, 2.0 -> 100. A Sharpe above 2 on six years of daily equity data
    # is more often a bug or a leverage artefact than an edge, so the scale
    # tops out rather than rewarding further.
    score = _scale(sharpe, 0.0, 2.0)
    return MetricScore(
        key="risk_adjusted",
        label="Risk-Adjusted Return",
        score=score,
        band=_band(score),
        value=sharpe,
        citations=[
            Citation(
                title="Risk-adjusted ratios (out-of-sample)",
                description=(
                    "Computed from walk-forward out-of-sample daily returns, "
                    "net of 1bp commission and 5bp slippage per side."
                ),
                points=[
                    f"Sharpe ratio: {sharpe}",
                    f"Sortino ratio: {ra.get('sortino_ratio')}",
                    f"Calmar ratio: {ra.get('calmar_ratio')}",
                    f"Omega ratio: {ra.get('omega_ratio')}",
                ],
                source="Backtest Analysis",
            )
        ],
        evidence={
            "sharpe_ratio": sharpe,
            "sortino_ratio": ra.get("sortino_ratio"),
            "calmar_ratio": ra.get("calmar_ratio"),
            "omega_ratio": ra.get("omega_ratio"),
            "scale": "0.0 Sharpe scores 0; 2.0 or above scores 100",
            "sortino_exceeds_sharpe": (
                (ra.get("sortino_ratio") or 0) > sharpe
            ),
        },
    )


def score_statistical_significance(m: dict) -> MetricScore:
    """The metric that decides whether the rest of the report means anything.

    Deflated Sharpe corrects the observed Sharpe for the number of strategy
    variants tried. A high raw Sharpe chosen from 16 trials is partly selection
    luck, and this is the only metric here that quantifies how much.
    """
    ds = m.get("deflated_sharpe", {})
    deflated = ds.get("deflated_sharpe")
    if deflated is None:
        return MetricScore(
            "significance", "Statistical Significance", 0, "FAILING", None
        )

    # The decision rule is the 0.95 significance bar, not a smooth preference:
    # a result either survives multiple-testing correction or it does not.
    # Scoring is capped below ADEQUATE when it does not, because no amount of
    # headline performance compensates for an unproven edge.
    is_significant = bool(ds.get("is_significant"))
    if is_significant:
        score = _scale(deflated, 0.95, 1.0) // 2 + 50  # 50-100 band
    else:
        score = _scale(deflated, 0.0, 0.95) // 2  # 0-47 band, cannot reach 60

    return MetricScore(
        key="significance",
        label="Statistical Significance",
        score=score,
        band=_band(score),
        value=deflated,
        citations=[
            Citation(
                title="Deflated Sharpe ratio",
                description=(
                    "Adjusts the observed Sharpe for the number of strategy "
                    "variants tested, sample length, and return non-normality. "
                    "Values below 0.95 mean the result does not survive "
                    "multiple-testing correction."
                ),
                points=[
                    f"Observed Sharpe: {ds.get('observed_sharpe')}",
                    f"Deflated Sharpe: {deflated} (significant: {is_significant})",
                    f"Expected max Sharpe from {ds.get('n_trials')} trials: "
                    f"{ds.get('expected_max_sharpe')}",
                    f"Observations: {ds.get('n_obs')}, "
                    f"skew {ds.get('skewness')}, "
                    f"excess kurtosis {ds.get('excess_kurtosis')}",
                ],
                source="Backtest Analysis",
            )
        ],
        evidence={
            "deflated_sharpe": deflated,
            "observed_sharpe": ds.get("observed_sharpe"),
            "probabilistic_sharpe": ds.get("probabilistic_sharpe"),
            "expected_max_sharpe": ds.get("expected_max_sharpe"),
            "n_trials": ds.get("n_trials"),
            "n_obs": ds.get("n_obs"),
            "is_significant": is_significant,
            "decision_rule": "deflated Sharpe must exceed 0.95 to be significant",
            "why_capped": (
                None
                if is_significant
                else "score is capped below 60 because the edge is unproven"
            ),
            "observed_exceeds_expected_max": (
                (ds.get("observed_sharpe") or 0) > (ds.get("expected_max_sharpe") or 0)
            ),
        },
    )


def score_drawdown(m: dict) -> MetricScore:
    r = m.get("risk", {})
    mdd = r.get("max_drawdown")
    if mdd is None:
        return MetricScore("drawdown", "Drawdown Control", 0, "FAILING", None)

    depth = abs(mdd)
    # -40% scores 0, -5% scores 100. The configured circuit breaker is at 20%,
    # so a drawdown at or past that bar cannot score as STRONG.
    score = _scale(-depth, -0.40, -0.05)
    return MetricScore(
        key="drawdown",
        label="Drawdown Control",
        score=score,
        band=_band(score),
        value=mdd,
        citations=[
            Citation(
                title="Drawdown profile",
                description=(
                    "Peak-to-trough decline on the out-of-sample equity curve, "
                    "with recovery time and the share of days spent below a "
                    "prior peak."
                ),
                points=[
                    f"Max drawdown: {mdd:.2%}",
                    f"Average drawdown: {r.get('avg_drawdown', 0):.2%}",
                    f"Longest drawdown: {r.get('max_drawdown_duration_days')} days",
                    f"Time underwater: {r.get('time_underwater_pct')}% of days",
                ],
                source="Risk Monitor",
            )
        ],
        evidence={
            "max_drawdown": mdd,
            "avg_drawdown": r.get("avg_drawdown"),
            "max_drawdown_duration_days": r.get("max_drawdown_duration_days"),
            "time_underwater_pct": r.get("time_underwater_pct"),
            "ulcer_index": r.get("ulcer_index"),
            "configured_circuit_breaker": 0.20,
            "breached_circuit_breaker": depth >= 0.20,
            "scale": "-40% scores 0; -5% or shallower scores 100",
        },
    )


def score_return_distribution(m: dict) -> MetricScore:
    """Volatility and tail shape -- how the return was earned, not how much."""
    r = m.get("risk", {})
    vol = r.get("annualized_volatility")
    if vol is None:
        return MetricScore("distribution", "Return Distribution", 0, "FAILING", None)

    # 40% vol scores 0, 10% scores 100 (10% is the configured target).
    vol_score = _scale(-vol, -0.40, -0.10)
    skew = r.get("skewness") or 0.0
    kurt = r.get("excess_kurtosis") or 0.0
    # Negative skew and fat tails both mean the average return understates the
    # bad days, so each costs points.
    penalty = (10 if skew < 0 else 0) + (10 if kurt > 1.0 else 0)
    score = max(0, vol_score - penalty)

    return MetricScore(
        key="distribution",
        label="Return Distribution",
        score=score,
        band=_band(score),
        value=vol,
        citations=[
            Citation(
                title="Volatility and tail shape",
                description=(
                    "Realised volatility against the 10% annual target, plus "
                    "the skew and kurtosis that determine how badly the worst "
                    "days behave relative to a normal distribution."
                ),
                points=[
                    f"Annualised volatility: {vol:.2%} (target 10%)",
                    f"Downside deviation: {r.get('downside_deviation', 0):.2%}",
                    f"Skewness: {skew} (negative means fat left tail)",
                    f"Excess kurtosis: {kurt} (above 0 means fatter tails than normal)",
                ],
                source="Risk Monitor",
            )
        ],
        evidence={
            "annualized_volatility": vol,
            "target_volatility": 0.10,
            "vol_overshoot_multiple": round(vol / 0.10, 2),
            "downside_deviation": r.get("downside_deviation"),
            "skewness": skew,
            "excess_kurtosis": kurt,
            "penalties_applied": {
                "negative_skew": 10 if skew < 0 else 0,
                "fat_tails": 10 if kurt > 1.0 else 0,
            },
        },
    )


def score_benchmark(m: dict) -> MetricScore:
    br = m.get("benchmark_relative", {})
    ir = br.get("information_ratio")
    if ir is None:
        return MetricScore("benchmark", "Benchmark Relative", 0, "FAILING", None)

    # IR of 0 scores 0, 1.0 scores 100. An IR of 0.5 is the usual bar for a
    # manager worth paying; 1.0 sustained is exceptional.
    score = _scale(ir, 0.0, 1.0)
    return MetricScore(
        key="benchmark",
        label="Benchmark Relative",
        score=score,
        band=_band(score),
        value=ir,
        citations=[
            Citation(
                title="Performance against SPY",
                description=(
                    "Excess return over the benchmark per unit of tracking "
                    "error. Answers whether the strategy beat a passive "
                    "alternative by enough to justify its activity."
                ),
                points=[
                    f"Information ratio: {ir}",
                    f"Tracking error: {br.get('tracking_error', 0):.2%}",
                    f"Strategy vs benchmark Sharpe: "
                    f"{m.get('risk_adjusted', {}).get('sharpe_ratio')} vs "
                    f"{br.get('benchmark_sharpe')}",
                    f"Excess total return: {br.get('excess_return', 0):.2%}",
                ],
                source="Portfolio Dashboard",
            )
        ],
        evidence={
            "information_ratio": ir,
            "tracking_error": br.get("tracking_error"),
            "benchmark_sharpe": br.get("benchmark_sharpe"),
            "benchmark_return": br.get("benchmark_return"),
            "excess_return": br.get("excess_return"),
            "beat_benchmark_sharpe": (
                (m.get("risk_adjusted", {}).get("sharpe_ratio") or 0)
                > (br.get("benchmark_sharpe") or 0)
            ),
        },
    )


def score_trade_quality(m: dict) -> MetricScore:
    t = m.get("trades", {})
    pf = t.get("profit_factor")
    if pf is None:
        return MetricScore("trades", "Trade Quality", 0, "FAILING", None)

    # Profit factor 1.0 (breakeven) scores 0, 2.5 scores 100.
    score = _scale(pf, 1.0, 2.5)
    return MetricScore(
        key="trades",
        label="Trade Quality",
        score=score,
        band=_band(score),
        value=pf,
        citations=[
            Citation(
                title="Trade-level statistics",
                description=(
                    "Gross profit divided by gross loss, with the win rate and "
                    "payoff ratio that produce it. Shows whether the edge comes "
                    "from winning often or from winning big."
                ),
                points=[
                    f"Profit factor: {pf}",
                    f"Win rate: {t.get('win_rate', 0):.2%} "
                    f"({t.get('n_wins')} wins / {t.get('n_losses')} losses)",
                    f"Payoff ratio: {t.get('payoff_ratio')} "
                    f"(avg win {t.get('avg_win', 0):.2%} vs "
                    f"avg loss {t.get('avg_loss', 0):.2%})",
                    f"Expectancy per trade: {t.get('expectancy', 0):.2%} "
                    f"across {t.get('n_trades')} trades",
                ],
                source="Backtest Analysis",
            )
        ],
        evidence={
            "profit_factor": pf,
            "win_rate": t.get("win_rate"),
            "payoff_ratio": t.get("payoff_ratio"),
            "expectancy": t.get("expectancy"),
            "n_trades": t.get("n_trades"),
            "edge_source": (
                "payoff size"
                if (t.get("payoff_ratio") or 0) > 1.2
                else "win frequency"
            ),
            "sample_adequate": (t.get("n_trades") or 0) >= 100,
        },
    )


def score_cost_resilience(m: dict) -> MetricScore:
    """Turnover decides how much of the paper edge survives real costs."""
    tn = m.get("turnover", {})
    annual = tn.get("annual_turnover")
    if annual is None:
        return MetricScore("costs", "Cost Resilience", 0, "FAILING", None)

    # 30x annual turnover scores 0, 2x scores 100. At 6bp round-trip, every
    # turn of the portfolio costs ~12bp of annual return.
    score = _scale(-annual, -30.0, -2.0)
    cost_drag = annual * 0.0012  # 6bp per side, both sides
    return MetricScore(
        key="costs",
        label="Cost Resilience",
        score=score,
        band=_band(score),
        value=annual,
        citations=[
            Citation(
                title="Turnover and cost drag",
                description=(
                    "How often the portfolio is replaced per year, and the "
                    "resulting drag at the modelled 1bp commission plus 5bp "
                    "slippage per side."
                ),
                points=[
                    f"Annual turnover: {annual:.1f}x",
                    f"Implied cost drag: {cost_drag:.2%} per year",
                    f"Average rebalance turnover: "
                    f"{tn.get('avg_rebalance_turnover', 0):.2%}",
                    f"Rebalances: {tn.get('n_rebalances')} (weekly schedule)",
                ],
                source="Backtest Analysis",
            )
        ],
        evidence={
            "annual_turnover": annual,
            "implied_annual_cost_drag": round(cost_drag, 4),
            "avg_rebalance_turnover": tn.get("avg_rebalance_turnover"),
            "n_rebalances": tn.get("n_rebalances"),
            "modelled_cost_bps_per_side": 6,
            "costs_already_in_returns": True,
        },
    )


SCORERS = (
    score_risk_adjusted,
    score_statistical_significance,
    score_drawdown,
    score_return_distribution,
    score_benchmark,
    score_trade_quality,
    score_cost_resilience,
)


def score_all(metrics: dict) -> list[MetricScore]:
    return [scorer(metrics) for scorer in SCORERS]


def overall_score(scores: list[MetricScore]) -> int:
    """Significance is weighted heaviest because it gates everything else.

    A strategy with a great Sharpe and no statistical significance is a
    hypothesis, not a result, and the headline number should say so.
    """
    weights = {
        "significance": 3.0,
        "risk_adjusted": 2.0,
        "drawdown": 1.5,
        "benchmark": 1.5,
        "distribution": 1.0,
        "trades": 1.0,
        "costs": 1.0,
    }
    total = sum(weights.get(s.key, 1.0) for s in scores)
    weighted = sum(s.score * weights.get(s.key, 1.0) for s in scores)
    return round(weighted / total) if total else 0
