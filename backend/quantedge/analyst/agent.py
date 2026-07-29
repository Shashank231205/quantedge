"""The analyst agent: turns scored evidence into a readable report.

Division of labour, which the prompt enforces and the code guarantees:

    rubric.py  decides every score, band, citation and figure
    agent.py   decides only the wording

The agent is handed the evidence and told what each score already is. It is
never asked whether a number is good -- that judgement is made before it runs.
This keeps the report reproducible (same run, same verdict) and means a weaker
fallback model degrades the prose without touching the analysis.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from quantedge.analyst import providers
from quantedge.analyst.rubric import (
    Citation,
    MetricScore,
    gap_citation,
    overall_score,
    score_all,
)
from quantedge.logging_config import get_logger

log = get_logger(__name__)


SYSTEM_PROMPT = """\
You are the analyst for QUANTEDGE, an equity factor research platform. You write \
the explanatory prose of a strategy assessment for a reader who evaluates \
managers for a living -- a portfolio manager, a risk officer, or an interviewer \
probing whether the numbers hold up.

WHAT IS ALREADY DECIDED, AND NOT YOURS TO CHANGE

Every score, band and figure in the input was computed by the platform's scoring \
rubric before you were called. You are not being asked whether a result is good. \
That is settled. You explain why it landed where it did.

Never contradict a score. If a metric scored 12/100 you may not call it strong, \
encouraging, or promising. If it scored 100 you may not hedge it into mediocrity. \
Your prose and its score must read as the same assessment.

THE ONE RULE THAT MATTERS MOST

Use only numbers present in the `evidence` and `citations` you are given. Do not \
introduce a figure, ratio, date, benchmark or comparison that is not in the \
input. Do not estimate, extrapolate, annualise, or convert. If you want to say \
something the evidence does not support, leave it out.

This platform's credibility rests on every number being traceable to a \
computation. One invented figure destroys that, and a reader who checks will \
find it.

VOICE

Write the way a practitioner talks to a peer: plain, direct, unhedged. State \
what happened and what it implies.

Avoid the register of marketing and of chatbots. No "robust", "impressive", \
"solid", "it's worth noting that", "in conclusion", "delve", "leverage" as a \
verb, "showcases", "underscores". Do not open with a restatement of the \
question. Do not close with a summary of what you just said.

Prefer the concrete over the abstract: "the strategy lost 18.9% peak to trough \
and took 130 days to recover" beats "drawdown characteristics were unfavourable".

Contractions are fine. Short sentences are fine. A paragraph that makes one \
point and stops is better than one that makes three and blurs them.

WHAT EACH FIELD IS FOR

`reasons` -- why the score is what it is. Cite the specific figures that drove \
it. When a score is high, say what earned it. When it is low, this is still the \
place to note what the metric does show, before the gaps explain what it does \
not. One to three sentences.

`gaps` -- what is missing, unproven, or working against the result; concretely, \
why the score is not 100. This is where honesty lives. Name the specific \
shortfall and, where the evidence supports it, what would resolve it (more \
data, fewer trials, lower turnover). If a score is near-perfect, say plainly \
what the remaining distance represents rather than inventing a flaw. One to \
three sentences.

Both fields must be grounded in the supplied evidence keys. When evidence \
contains a flag such as `breached_circuit_breaker: true` or \
`is_significant: false`, address it -- those flags exist because they matter.

OUTPUT

Return a single JSON object, no markdown fence, no commentary:

{
  "summary": "3-5 sentences. Lead with the verdict the overall score implies, \
then the single strongest result and the single most serious problem. A reader \
who stops here should not be surprised by anything below.",
  "metrics": {
    "<metric key>": {"reasons": "...", "gaps": "..."}
  }
}

Include every metric key from the input, spelled exactly as given.
"""


@dataclass
class MetricReport:
    key: str
    label: str
    score: int
    band: str
    value: float | None
    reasons: str
    gaps: str
    citations: list[dict]
    gap_citations: list[dict]


@dataclass
class AnalystReport:
    run_id: int
    run_name: str
    overall_score: int
    verdict: str
    summary: str
    metrics: list[MetricReport]
    provider: str
    model: str
    generated_at: str
    universe_size: int
    is_template: bool = False
    notes: list[str] = field(default_factory=list)


def _verdict(score: int) -> str:
    """Deliberately blunt labels -- a hedged verdict helps nobody."""
    if score >= 80:
        return "VALIDATED"
    if score >= 65:
        return "PROMISING"
    if score >= 50:
        return "UNPROVEN"
    if score >= 35:
        return "WEAK"
    return "NOT SUPPORTED"


def _build_user_prompt(
    scores: list[MetricScore], overall: int, context: dict
) -> str:
    payload = {
        "strategy": context,
        "overall_score": overall,
        "overall_verdict": _verdict(overall),
        "metrics": [
            {
                "key": s.key,
                "label": s.label,
                "score": s.score,
                "band": s.band,
                "value": s.value,
                "evidence": s.evidence,
                "citations": [asdict(c) for c in s.citations],
            }
            for s in scores
        ],
    }
    return (
        "Write the assessment for this strategy run.\n\n"
        f"{json.dumps(payload, indent=2, default=str)}\n\n"
        "Return the JSON object described in your instructions. Every metric "
        "key above must appear in your `metrics` object."
    )


# --- Template fallback ------------------------------------------------------
# Used when no provider is reachable. Mechanical by design: these sentences are
# assembled from the same evidence the agent would have received, so they can
# state less but cannot state anything untrue.

_BAND_PHRASE = {
    "STRONG": "clears the bar comfortably",
    "ADEQUATE": "clears the bar without much margin",
    "WEAK": "falls short of the bar",
    "FAILING": "does not meet the bar",
}


def _template_prose(s: MetricScore) -> tuple[str, str]:
    phrase = _BAND_PHRASE.get(s.band, "was scored")
    value = f"{s.value}" if s.value is not None else "not available"
    reasons = (
        f"{s.label} scored {s.score}/100 and {phrase}. "
        f"The driving figure is {value}."
    )

    gap_bits: list[str] = []
    if s.evidence.get("is_significant") is False:
        gap_bits.append(
            "the result does not survive multiple-testing correction, so the "
            "edge is unproven rather than demonstrated"
        )
    if s.evidence.get("breached_circuit_breaker"):
        gap_bits.append("the drawdown reached the configured 20% circuit breaker")
    if s.evidence.get("vol_overshoot_multiple"):
        gap_bits.append(
            f"realised volatility ran {s.evidence['vol_overshoot_multiple']}x the "
            "10% target"
        )
    if s.evidence.get("sample_adequate") is False:
        gap_bits.append("the trade sample is too small to be conclusive")
    if not gap_bits and s.score < 100:
        gap_bits.append(
            f"the remaining {100 - s.score} points reflect distance from the "
            "threshold this metric scores against"
        )

    gaps = (
        ("Specifically, " + "; ".join(gap_bits) + ".")
        if gap_bits
        else "No material gap identified for this metric."
    )
    return reasons, gaps


def _template_summary(scores: list[MetricScore], overall: int) -> str:
    best = max(scores, key=lambda s: s.score)
    worst = min(scores, key=lambda s: s.score)
    return (
        f"Overall assessment: {_verdict(overall)} at {overall}/100. "
        f"The strongest result is {best.label} at {best.score}/100; the weakest "
        f"is {worst.label} at {worst.score}/100. "
        "This summary was generated from the scoring rubric without a language "
        "model, so it states the figures rather than interpreting them."
    )


def generate_report(
    metrics: dict, context: dict, universe_size: int = 0
) -> AnalystReport:
    """Score deterministically, then ask a provider to explain the scores."""
    from datetime import UTC, datetime

    scores = score_all(metrics)
    overall = overall_score(scores)
    notes: list[str] = []

    completion = providers.complete(
        SYSTEM_PROMPT, _build_user_prompt(scores, overall, context)
    )
    parsed = providers.parse_json(completion.text)
    is_template = not parsed

    if is_template and completion.provider != "template":
        # A provider answered but the payload was unusable. Say so in the
        # report rather than silently serving templates that look like prose.
        notes.append(
            f"The {completion.provider} response could not be parsed; "
            "prose fell back to the rule-based templates."
        )

    prose = parsed.get("metrics", {}) if parsed else {}
    metric_reports: list[MetricReport] = []
    for s in scores:
        written = prose.get(s.key) or {}
        reasons, gaps = _template_prose(s)
        metric_reports.append(
            MetricReport(
                key=s.key,
                label=s.label,
                score=s.score,
                band=s.band,
                value=s.value,
                reasons=(written.get("reasons") or reasons).strip(),
                gaps=(written.get("gaps") or gaps).strip(),
                citations=[asdict(c) for c in s.citations],
                gap_citations=[asdict(c) for c in gap_citation(s)],
            )
        )

    summary = (parsed.get("summary") or "").strip() or _template_summary(scores, overall)

    return AnalystReport(
        run_id=context.get("run_id", 0),
        run_name=context.get("run_name", "unknown"),
        overall_score=overall,
        verdict=_verdict(overall),
        summary=summary,
        metrics=metric_reports,
        provider=completion.provider,
        model=completion.model,
        generated_at=datetime.now(UTC).isoformat(),
        universe_size=universe_size,
        is_template=is_template,
        notes=notes,
    )


def report_to_dict(report: AnalystReport) -> dict[str, Any]:
    return asdict(report)


__all__ = [
    "AnalystReport",
    "Citation",
    "MetricReport",
    "generate_report",
    "report_to_dict",
]
