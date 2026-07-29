"""The free-model roster INU AI routes across.

Every model here is free to call and open-weight. They are not
interchangeable, and the limits that separate them were measured from each
provider's own rate-limit headers rather than assumed: Groq answers in a
fraction of a second but meters tokens per minute, while OpenRouter's free
models are slower with context windows up to a million tokens.

Routing by task rather than by preference is what lets the whole thing stay
free without feeling degraded -- a one-line question does not need a 550B
model, and a 200-page PDF cannot be answered by a model with a 6k/min ceiling.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Task(StrEnum):
    """What the router is being asked for, not which model it should use."""

    #: Short conversational turn. Latency dominates quality here.
    CHAT = "chat"
    #: Multi-step reasoning over platform numbers -- explaining a metric,
    #: comparing runs, walking through a formula.
    REASONING = "reasoning"
    #: Needs information this platform does not hold: news, current prices,
    #: what a term means outside our own docs.
    WEB = "web"
    #: The user attached an image or a chart screenshot.
    VISION = "vision"
    #: A long document that will not fit in a small context window.
    LONG_DOC = "long_doc"


@dataclass(frozen=True)
class ModelSpec:
    """One callable model, and the facts the router needs to choose it."""

    id: str
    provider: str
    context: int
    #: Measured per-minute token ceiling, read from the provider's own
    #: rate-limit headers rather than assumed.
    tpm: int | None = None
    #: Measured requests-per-day ceiling. Some models trade a generous token
    #: budget for a small request budget, which changes what they suit.
    rpd: int | None = None
    notes: str = ""
    #: Tasks this model is a sensible first choice for.
    good_for: tuple[Task, ...] = ()


#: Groq. Fastest inference on a free tier -- measured at 0.3s for a short
#: completion against ~2.2s for Gemini. Limits below are read from Groq's own
#: rate-limit headers, and they differ enough per model to change routing: the
#: 8B model allows 14,400 requests a day but only 6k tokens a minute, while
#: compound allows 70k tokens a minute across just 250 requests. Cheap chatter
#: therefore goes to the 8B model and anything bulky to a roomier one.
GROQ_MODELS = (
    ModelSpec(
        id="llama-3.1-8b-instant",
        provider="groq",
        context=131072,
        tpm=6000,
        rpd=14400,
        notes="Sub-second. First choice for ordinary conversational turns.",
        good_for=(Task.CHAT,),
    ),
    ModelSpec(
        id="openai/gpt-oss-120b",
        provider="groq",
        context=131072,
        tpm=8000,
        rpd=1000,
        notes="Open-weight 120B. Best free reasoning that still answers fast.",
        good_for=(Task.REASONING,),
    ),
    ModelSpec(
        id="qwen/qwen3.6-27b",
        provider="groq",
        context=131072,
        tpm=8000,
        rpd=1000,
        notes="Strong on structured output and tool arguments.",
        good_for=(Task.REASONING, Task.CHAT),
    ),
    ModelSpec(
        id="groq/compound",
        provider="groq",
        context=131072,
        tpm=70000,
        rpd=250,
        notes=(
            "Web search runs inside the model -- no separate search API key "
            "and no scraping. This is how INU answers questions about the "
            "world rather than about this platform."
        ),
        good_for=(Task.WEB,),
    ),
)

#: OpenRouter free tier. Slower than Groq, but the context windows are an order
#: of magnitude larger and there is no comparable per-minute token ceiling, so
#: everything bulky routes here.
OPENROUTER_MODELS = (
    ModelSpec(
        id="google/gemma-4-31b-it:free",
        provider="openrouter",
        context=262144,
        notes="Open-weight vision. Reads uploaded charts and screenshots.",
        good_for=(Task.VISION,),
    ),
    ModelSpec(
        id="nvidia/nemotron-nano-12b-v2-vl:free",
        provider="openrouter",
        context=128000,
        notes="Second vision model, so an image request has a fallback.",
        good_for=(Task.VISION,),
    ),
    ModelSpec(
        id="nvidia/nemotron-3-ultra-550b-a55b:free",
        provider="openrouter",
        context=1000000,
        notes="1M context. Whole documents, or a full conversation history.",
        good_for=(Task.LONG_DOC, Task.REASONING),
    ),
    ModelSpec(
        id="nvidia/nemotron-3-super-120b-a12b:free",
        provider="openrouter",
        context=262144,
        notes="Faster than the 550B at similar quality for most questions.",
        good_for=(Task.LONG_DOC, Task.REASONING),
    ),
    ModelSpec(
        id="openai/gpt-oss-20b:free",
        provider="openrouter",
        context=131072,
        notes="Reliable JSON. The safe fallback when a bigger model stalls.",
        good_for=(Task.CHAT, Task.REASONING),
    ),
    ModelSpec(
        id="inclusionai/ling-3.0-flash:free",
        provider="openrouter",
        context=262144,
        notes="Fast open-weight generalist.",
        good_for=(Task.CHAT,),
    ),
)

ALL_MODELS: tuple[ModelSpec, ...] = GROQ_MODELS + OPENROUTER_MODELS


#: Ordered candidates per task. The router walks these in order, skipping any
#: provider currently in cooldown, so a rate-limited model costs nothing.
ROUTES: dict[Task, tuple[ModelSpec, ...]] = {
    Task.CHAT: (
        GROQ_MODELS[0],  # llama-3.1-8b-instant
        GROQ_MODELS[2],  # qwen3.6-27b
        OPENROUTER_MODELS[5],  # ling-3.0-flash
        OPENROUTER_MODELS[4],  # gpt-oss-20b
    ),
    Task.REASONING: (
        GROQ_MODELS[1],  # gpt-oss-120b
        GROQ_MODELS[2],  # qwen3.6-27b
        OPENROUTER_MODELS[3],  # nemotron-super-120b
        OPENROUTER_MODELS[4],  # gpt-oss-20b
    ),
    Task.WEB: (
        GROQ_MODELS[3],  # compound, search built in
        # No OpenRouter equivalent carries live search, so a web question that
        # cannot reach compound is answered from context with that stated.
        OPENROUTER_MODELS[3],
    ),
    Task.VISION: (
        OPENROUTER_MODELS[0],  # gemma-4-31b
        OPENROUTER_MODELS[1],  # nemotron-nano-vl
    ),
    Task.LONG_DOC: (
        OPENROUTER_MODELS[2],  # nemotron-ultra-550b, 1M context
        OPENROUTER_MODELS[3],  # nemotron-super-120b
    ),
}

#: Below this a request fits the tightest Groq model (the 8B, at 6k TPM) and
#: can take the fast path. Above it the router picks a model whose measured
#: ceiling actually accommodates the prompt, rather than bouncing off a 429.
GROQ_TOKEN_CEILING = 5000


def estimate_tokens(text: str) -> int:
    """Rough token count, deliberately pessimistic.

    Four characters per token is the usual English approximation; overshooting
    routes a borderline request to the roomier model, which is the cheaper
    mistake of the two.
    """
    return len(text) // 3


def route(task: Task, prompt_chars: int = 0, *, has_image: bool = False) -> tuple[ModelSpec, ...]:
    """Candidate models for a task, best first.

    Size overrides task: a reasoning question carrying a long document is a
    long-document request, because no amount of reasoning quality helps if the
    input does not fit.
    """
    if has_image:
        return ROUTES[Task.VISION]

    approx = prompt_chars // 3
    if approx > GROQ_TOKEN_CEILING and task in (Task.CHAT, Task.REASONING):
        # Drop any candidate whose measured TPM cannot hold this prompt, then
        # fall back to the roomy OpenRouter models.
        fits = tuple(
            m for m in ROUTES[task] if m.tpm is None or m.tpm > approx
        )
        return fits + ROUTES[Task.LONG_DOC]

    return ROUTES.get(task, ROUTES[Task.CHAT])
