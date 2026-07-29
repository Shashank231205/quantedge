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
