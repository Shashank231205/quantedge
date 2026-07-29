"""Central configuration, driven by environment variables.

Every tunable that affects backtest results lives here so a run can be
reproduced from its recorded config alone.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- Database -------------------------------------------------------
    database_url: str = Field(
        default="postgresql+psycopg://quantedge:quantedge@localhost:5432/quantedge"
    )
    db_echo: bool = False

    @field_validator("database_url", mode="after")
    @classmethod
    def _normalize_driver(cls, v: str) -> str:
        """Managed hosts hand out bare `postgresql://` (or legacy `postgres://`)
        URLs. SQLAlchemy would then load psycopg2, which is not installed —
        pin the driver we actually depend on."""
        for prefix in ("postgresql://", "postgres://"):
            if v.startswith(prefix):
                return "postgresql+psycopg://" + v[len(prefix) :]
        return v

    # --- Data sources ---------------------------------------------------
    data_source: str = Field(default="yfinance", description="yfinance | polygon")
    polygon_api_key: str | None = None
    history_years: int = 6
    universe_size: int = 500
    benchmark_ticker: str = "SPY"

    # Local parquet cache so a re-run never re-hits the provider.
    cache_dir: str = "../data/raw"
    processed_dir: str = "../data/processed"

    # --- Ingestion behaviour --------------------------------------------
    batch_size: int = 50
    max_retries: int = 3
    retry_backoff_seconds: float = 2.0
    request_delay_seconds: float = 0.3

    # --- Trading calendar -----------------------------------------------
    trading_days_per_year: int = 252
    risk_free_rate: float = 0.02

    # --- Costs (round-trip realism) --------------------------------------
    commission_bps: float = Field(default=1.0, description="per side, basis points")
    slippage_bps: float = Field(default=5.0, description="per side, basis points")

    # --- Portfolio construction ------------------------------------------
    long_quantile: float = 0.1
    short_quantile: float = 0.1
    target_annual_vol: float = 0.10
    max_position_weight: float = 0.05
    max_sector_weight: float = 0.25
    max_drawdown_limit: float = 0.20
    rebalance_frequency: str = "W-FRI"

    # --- Walk-forward -----------------------------------------------------
    train_years: int = 2
    test_months: int = 6
    embargo_days: int = 10

    # --- API ---------------------------------------------------------------
    api_key: str = Field(default="quantedge-dev-key")
    api_prefix: str = "/v1"
    # Deployed, the UI is served from a different origin than the API, so the
    # allowed set has to be configurable. CORS_ORIGINS accepts a JSON list or a
    # comma-separated string; preview deployments get a regex instead, since
    # their hostnames are generated per commit and cannot be enumerated.
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]
    cors_origin_regex: str | None = None
    log_buffer_size: int = 500

    # --- Analyst mode -------------------------------------------------------
    # The analyst writes the prose of a report; it never decides a score. Every
    # number and verdict comes from quantedge.analyst.rubric, so swapping the
    # provider changes the wording and nothing else.
    #
    # Providers are tried in order and the first one holding a key wins. With
    # no keys at all the report still renders from templates, which is why a
    # deployment without credentials degrades rather than breaks.
    # Groq leads on latency: measured at ~0.7s against ~2.2s for Gemini on the
    # same prompt, and a full report is the difference between a wait and a
    # pause. Gemini stays in the chain for its larger free quota and as the
    # multimodal path, and OpenRouter behind it for breadth.
    analyst_providers: list[str] = [
        "groq",
        "gemini",
        "gemini2",
        "openrouter",
        "template",
    ]
    #: How long a provider that returned 429 is skipped before being retried.
    #: Without this every request pays a guaranteed-failing round trip to an
    #: exhausted key before reaching one that works.
    analyst_cooldown_seconds: float = 900.0

    # --- INU AI chat --------------------------------------------------------
    #: Chat turns must feel immediate, so the ceiling is far tighter than the
    #: analyst's -- a model that has not answered in 45s should be abandoned
    #: for the next one in the chain rather than waited on.
    inu_timeout_seconds: float = 45.0
    inu_history_turns: int = 12

    # --- Deployment sizing ---------------------------------------------------
    #: Pre-building the factor and risk caches costs a full copy of the price
    #: panel each. Worth it on a machine with headroom, fatal on a small
    #: instance -- set WARM_CACHE_ON_STARTUP=false there and pay the first
    #: request's latency instead of restarting under memory pressure.
    warm_cache_on_startup: bool = True
    gemini_api_key: str | None = None
    # A second Gemini key on a different Google project is a second free-tier
    # quota, not just a spare credential: when the first key exhausts its daily
    # limit the chain moves to this one instead of degrading to templates.
    gemini_api_key_2: str | None = None
    # Alias rather than a pinned version. Google retires pinned models for new
    # projects while keeping them for existing ones, so a hard-coded version
    # works on an older key and 404s on a newer one -- which looks like a bad
    # credential and is not.
    gemini_model: str = "gemini-flash-latest"
    groq_api_key: str | None = None
    groq_model: str = "llama-3.3-70b-versatile"
    openrouter_api_key: str | None = None
    # OpenRouter's free catalogue turns over -- a model id that worked last
    # month may simply not exist today, and an unknown id fails at request time
    # rather than at startup. Check /api/v1/models before changing this.
    openrouter_model: str = "openai/gpt-oss-20b:free"
    openrouter_referer: str = "https://github.com/Shashank231205/quantedge"
    # Reports are recomputed rarely and cost a provider call, so a generous
    # cache keeps repeat views free.
    analyst_cache_seconds: float = 900.0
    # Free-tier models queue behind paid traffic and a full seven-metric report
    # can take minutes on a busy one. The timeout is generous because giving up
    # early costs a provider that would have answered; the chain still moves on
    # if it genuinely stalls.
    analyst_timeout_seconds: float = 180.0

    @classmethod
    def settings_customise_sources(cls, settings_cls, **kwargs):  # noqa: ANN206
        """pydantic-settings JSON-decodes complex fields straight from the
        environment, so a plain `a.com,b.com` in CORS_ORIGINS raises before any
        validator runs. Decode that comma form here, where the raw string is
        still available."""
        sources = super().settings_customise_sources(settings_cls, **kwargs)

        def patch(source):
            original = getattr(source, "decode_complex_value", None)
            if original is None:
                return source

            def decode(field_name, field, value):
                if field_name == "cors_origins" and not value.strip().startswith("["):
                    return [o.strip() for o in value.split(",") if o.strip()]
                return original(field_name, field, value)

            source.decode_complex_value = decode
            return source

        return tuple(patch(s) for s in sources)


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
