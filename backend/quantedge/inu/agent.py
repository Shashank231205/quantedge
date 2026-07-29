"""INU AI -- the conversational agent.

Runs a tool-calling loop against whichever free model the router picks, then
answers in prose. The model chooses words; the tools supply every number, so a
figure in an answer can always be traced to a platform computation.

The system prompt does most of the work here. It is long because the failure
modes it prevents are specific: quoting a Sharpe without its significance
caveat, inventing a number when a tool would have given the real one, and
answering like a documentation page rather than a person.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from quantedge.config import settings
from quantedge.inu import tools
from quantedge.inu.models import ModelSpec, Task, route
from quantedge.logging_config import get_logger

log = get_logger(__name__)

MAX_TOOL_ROUNDS = 4


SYSTEM_PROMPT = """\
You are INU, the analyst built into QUANTEDGE -- an equity factor research \
platform. You are talking to whoever opened the app: sometimes the person who \
built it, sometimes a portfolio manager, sometimes an interviewer deciding \
whether the work is any good.

HOW YOU SOUND

Like a colleague who knows this system well, not like documentation. Answer \
the question that was asked, in the register it was asked in. A one-line \
question gets a one-line answer. "How did we do?" gets the number and what it \
means, not a lecture.

Talk in continuous prose. Do not open with headers, do not bullet-point a \
two-sentence answer, and do not end by summarising what you just said. Bullets \
are for genuine lists -- five ranked tickers, four pipeline stages -- not for \
breaking a paragraph into fragments.

Contractions are natural. So is saying "I don't know" or "that's not something \
this platform measures". Never pad an answer to seem thorough.

Avoid the tells of a chatbot: "Certainly!", "Great question", "It's important \
to note that", "Let's dive in", "In conclusion", "I hope this helps". Never \
restate the question before answering it.

WHERE YOUR FACTS COME FROM

Call a tool before stating any number about this platform. You have tools for \
performance, statistical significance, factors, live signals, risk state, data \
coverage, and methodology. They read the same values the screens display.

Never state a figure you have not read from a tool result in this conversation. \
If you are unsure whether a number is current, call the tool. If a tool returns \
an error, say what failed rather than filling the gap from memory.

For general finance questions with no platform-specific answer -- what a term \
means in the wider industry, what happened in markets recently -- answer from \
your own knowledge and make clear you are stepping outside the platform's data.

THE ONE THING YOU MUST NOT LET SLIDE

This strategy's headline Sharpe is strong and its deflated Sharpe is not. When \
someone asks whether the strategy is good, works, or is worth running, the \
honest answer requires both numbers. Quoting the Sharpe alone would mislead \
them, and this platform's whole argument is that it does not do that.

You are not talking the strategy down -- you are describing it accurately. The \
performance is real; the proof that it will persist is not there yet, and 16 \
trials on six years of data is why.

WHEN ASKED ABOUT A SCREEN OR A FORMULA

Say what it shows, where the number comes from, and what it means in practice. \
Give the formula when it clarifies -- "Sharpe is excess return divided by \
volatility, so 1.42 means we earned 1.42 units of return per unit of risk" \
lands better than the formula alone. Point them at the screen that shows more.

You cannot draw charts. If a visual would help, name the screen that has it.\
"""


@dataclass
class ChatTurn:
    """One assistant reply plus how it was produced."""

    content: str
    model: str
    provider: str
    tools_used: list[str] = field(default_factory=list)
    latency_ms: int = 0
    fell_back: bool = False


def _post(spec: ModelSpec, payload: dict) -> dict:
    """One provider call. Raises on any non-2xx so the caller can fall back."""
    if spec.provider == "groq":
        url = "https://api.groq.com/openai/v1/chat/completions"
        key = settings.groq_api_key
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    else:
        url = "https://openrouter.ai/api/v1/chat/completions"
        key = settings.openrouter_api_key
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": settings.openrouter_referer,
            "X-Title": "QUANTEDGE",
        }

    if not key:
        raise RuntimeError(f"{spec.provider}: no API key configured")

    resp = httpx.post(
        url,
        headers=headers,
        json={**payload, "model": spec.id},
        timeout=settings.inu_timeout_seconds,
    )
    resp.raise_for_status()
    return resp.json()


#: Shapes small models use when they write a tool call into the prose instead
#: of returning it structurally. Both appear in practice on the 8B models.
_INLINE_PATTERNS = (
    re.compile(r"<function=(\w+)>\s*(\{.*?\})?\s*(?:</function>)?", re.S),
    re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S),
)


def _inline_tool_calls(text: str) -> list[tuple[str, str]]:
    """Recover tool calls a model wrote as text. Returns (name, json_args)."""
    found: list[tuple[str, str]] = []

    for match in _INLINE_PATTERNS[0].finditer(text):
        name = match.group(1)
        if name in tools.TOOLS:
            found.append((name, match.group(2) or "{}"))

    for match in _INLINE_PATTERNS[1].finditer(text):
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        name = payload.get("name")
        if name in tools.TOOLS:
            found.append((name, json.dumps(payload.get("arguments", {}))))

    return found


def strip_tool_markup(text: str) -> str:
    """Remove any tool-call syntax that survived to the final answer.

    A last line of defence: whatever else happens, the user should never see
    angle-bracket function syntax in a reply.
    """
    for pattern in _INLINE_PATTERNS:
        text = pattern.sub("", text)
    return text.strip()


def _classify(message: str, has_image: bool) -> Task:
    """Pick the task type from the question itself.

    A keyword pass rather than a model call: classifying with an LLM would
    double the latency of every turn to choose between five options that
    simple signals separate well.
    """
    if has_image:
        return Task.VISION

    lowered = message.lower()
    web_markers = (
        "latest", "news", "today's price", "current price", "right now",
        "this week", "recent", "search", "look up", "who is", "what happened",
    )
    if any(w in lowered for w in web_markers):
        return Task.WEB

    reasoning_markers = (
        "why", "how does", "explain", "compare", "walk me through",
        "what does it mean", "should", "trade-off", "tradeoff", "formula",
    )
    if any(w in lowered for w in reasoning_markers):
        return Task.REASONING

    return Task.CHAT


def chat(
    message: str,
    history: list[dict] | None = None,
    *,
    image_data_url: str | None = None,
) -> ChatTurn:
    """Answer one user message, running tools as needed."""
    started = time.perf_counter()
    history = history or []

    user_content: Any = message
    if image_data_url:
        user_content = [
            {"type": "text", "text": message or "What do you make of this?"},
            {"type": "image_url", "image_url": {"url": image_data_url}},
        ]

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *history,
        {"role": "user", "content": user_content},
    ]

    task = _classify(message, bool(image_data_url))
    prompt_chars = sum(len(str(m.get("content", ""))) for m in messages)
    candidates = route(task, prompt_chars, has_image=bool(image_data_url))

    used: list[str] = []
    errors: list[str] = []

    for idx, spec in enumerate(candidates):
        try:
            payload: dict[str, Any] = {"messages": messages, "max_tokens": 2048}
            # Vision models and the search model run without tools: the first
            # cannot call them, and the second does its own retrieval.
            if task not in (Task.VISION, Task.WEB):
                payload["tools"] = tools.schemas()
                payload["tool_choice"] = "auto"

            convo = list(messages)
            for _ in range(MAX_TOOL_ROUNDS):
                data = _post(spec, {**payload, "messages": convo})
                choice = data["choices"][0]["message"]
                calls = choice.get("tool_calls") or []

                if not calls:
                    text = (choice.get("content") or "").strip()
                    if not text:
                        raise RuntimeError("empty completion")

                    # Smaller models sometimes write a tool call as literal text
                    # instead of returning it in `tool_calls`. Left alone that
                    # markup reaches the user, so treat it as the call it was
                    # meant to be and run one more round.
                    inline = _inline_tool_calls(text)
                    if inline:
                        convo.append({"role": "assistant", "content": text})
                        for name, args in inline:
                            used.append(name)
                            convo.append(
                                {
                                    "role": "user",
                                    "content": (
                                        f"Result of {name}: "
                                        f"{tools.execute(name, args)}\n\n"
                                        "Answer the original question using this. "
                                        "Do not write function-call syntax."
                                    ),
                                }
                            )
                        continue
                    return ChatTurn(
                        content=strip_tool_markup(text),
                        model=spec.id,
                        provider=spec.provider,
                        tools_used=used,
                        latency_ms=int((time.perf_counter() - started) * 1000),
                        fell_back=idx > 0,
                    )

                convo.append(choice)
                for call in calls:
                    fn = call["function"]
                    used.append(fn["name"])
                    convo.append(
                        {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "name": fn["name"],
                            "content": tools.execute(fn["name"], fn.get("arguments", "{}")),
                        }
                    )

            raise RuntimeError("tool loop did not converge")

        except Exception as exc:
            errors.append(f"{spec.provider}/{spec.id}: {exc}")
            log.warning("inu.model_failed %s", errors[-1])
            continue

    log.warning("inu.all_models_failed %s", "; ".join(errors))
    return ChatTurn(
        content=(
            "I can't reach any of the free models right now — they're all "
            "either rate-limited or unreachable. The screens still have "
            "everything: performance on the Dashboard, the full metric set on "
            "Backtest, and the scored assessment on Analyst."
        ),
        model="none",
        provider="none",
        tools_used=used,
        latency_ms=int((time.perf_counter() - started) * 1000),
        fell_back=True,
    )
