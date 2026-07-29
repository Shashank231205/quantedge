"""LLM backends for the analyst, tried in order until one answers.

The analyst is deliberately provider-agnostic. Scores, bands and citations are
computed in `rubric.py` before any model is called, so a provider only ever
chooses wording -- and the report's numbers are identical whether the prose
came from Gemini, Groq, a free OpenRouter model, or the local templates.

That property is what makes the fallback chain safe: degrading to a weaker
model degrades the writing, never the analysis. The template provider at the
end of the chain has no network dependency, so a deployment with no API keys
still serves a complete report instead of an error.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Protocol

import httpx

from quantedge.config import settings
from quantedge.logging_config import get_logger

log = get_logger(__name__)


class ProviderError(RuntimeError):
    """Raised when a provider cannot answer, so the chain moves to the next."""


@dataclass
class Completion:
    text: str
    provider: str
    model: str


class Provider(Protocol):
    name: str

    def available(self) -> bool: ...

    def complete(self, system: str, user: str) -> Completion: ...


def _key(setting_value: str | None, env_name: str) -> str | None:
    """Settings first, then the raw environment.

    Reading the environment directly as a fallback means a key exported in a
    shell works without editing .env, which is the common case when trying a
    provider out.
    """
    return setting_value or os.environ.get(env_name) or None


class GeminiProvider:
    """Google AI Studio. Free tier is rate-limited rather than credit-limited."""

    name = "gemini"

    def __init__(self) -> None:
        self.api_key = _key(settings.gemini_api_key, "GEMINI_API_KEY")
        self.model = settings.gemini_model

    def available(self) -> bool:
        return bool(self.api_key)

    def complete(self, system: str, user: str) -> Completion:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent"
        )
        try:
            resp = httpx.post(
                url,
                headers={"x-goog-api-key": self.api_key or "", "Content-Type": "application/json"},
                json={
                    "system_instruction": {"parts": [{"text": system}]},
                    "contents": [{"role": "user", "parts": [{"text": user}]}],
                    "generationConfig": {
                        "responseMimeType": "application/json",
                        "maxOutputTokens": 8192,
                    },
                },
                timeout=settings.analyst_timeout_seconds,
            )
            resp.raise_for_status()
            payload = resp.json()
            parts = payload["candidates"][0]["content"]["parts"]
            return Completion(parts[0]["text"], self.name, self.model)
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
            raise ProviderError(f"gemini: {exc}") from exc


class _OpenAICompatProvider:
    """Groq, OpenRouter and most open-model hosts share this wire format."""

    name = "openai-compatible"
    base_url = ""

    def __init__(self, api_key: str | None, model: str) -> None:
        self.api_key = api_key
        self.model = model

    def available(self) -> bool:
        return bool(self.api_key)

    def complete(self, system: str, user: str) -> Completion:
        try:
            resp = httpx.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "response_format": {"type": "json_object"},
                    "max_tokens": 8192,
                },
                timeout=settings.analyst_timeout_seconds,
            )
            resp.raise_for_status()
            payload = resp.json()
            return Completion(
                payload["choices"][0]["message"]["content"], self.name, self.model
            )
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
            raise ProviderError(f"{self.name}: {exc}") from exc


class GroqProvider(_OpenAICompatProvider):
    name = "groq"
    base_url = "https://api.groq.com/openai/v1"

    def __init__(self) -> None:
        super().__init__(
            _key(settings.groq_api_key, "GROQ_API_KEY"), settings.groq_model
        )


class OpenRouterProvider(_OpenAICompatProvider):
    name = "openrouter"
    base_url = "https://openrouter.ai/api/v1"

    def __init__(self) -> None:
        super().__init__(
            _key(settings.openrouter_api_key, "OPENROUTER_API_KEY"),
            settings.openrouter_model,
        )


class TemplateProvider:
    """The terminal provider: always available, never calls out.

    It returns an empty payload rather than prose. The caller detects that and
    renders sentences built from the rubric's own evidence, which is why a
    key-less deployment still produces a full report -- one that is more
    mechanical to read but cannot invent anything.
    """

    name = "template"
    model = "rule-based"

    def available(self) -> bool:
        return True

    def complete(self, system: str, user: str) -> Completion:
        return Completion("", self.name, self.model)


_REGISTRY = {
    "gemini": GeminiProvider,
    "groq": GroqProvider,
    "openrouter": OpenRouterProvider,
    "template": TemplateProvider,
}


def resolve_chain() -> list[Provider]:
    chain: list[Provider] = []
    for name in settings.analyst_providers:
        factory = _REGISTRY.get(name)
        if factory is None:
            log.warning("analyst.unknown_provider name=%s", name)
            continue
        chain.append(factory())
    return chain


def complete(system: str, user: str) -> Completion:
    """Walk the chain, returning the first successful completion.

    A provider that is configured but failing (rate limit, outage, revoked key)
    is logged and skipped rather than surfaced, because the next provider or
    the templates can still produce a report.
    """
    errors: list[str] = []
    for provider in resolve_chain():
        if not provider.available():
            continue
        if isinstance(provider, TemplateProvider):
            if errors:
                log.warning("analyst.falling_back_to_templates errors=%s", "; ".join(errors))
            return provider.complete(system, user)
        try:
            completion = provider.complete(system, user)
            log.info("analyst.completed provider=%s model=%s", completion.provider, completion.model)
            return completion
        except ProviderError as exc:
            errors.append(str(exc))
            log.warning("analyst.provider_failed %s", exc)

    log.warning("analyst.no_provider_available errors=%s", "; ".join(errors))
    return TemplateProvider().complete(system, user)


def active_provider_name() -> str:
    """What the next report would use -- surfaced in the UI, not guessed at."""
    for provider in resolve_chain():
        if provider.available():
            return provider.name
    return "template"


def parse_json(text: str) -> dict:
    """Models occasionally wrap JSON in prose or a code fence despite the
    response-format request. Recover the object rather than failing the run."""
    text = text.strip()
    if not text:
        return {}
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return {}
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}
