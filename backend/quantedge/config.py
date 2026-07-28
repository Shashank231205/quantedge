"""Central configuration, driven by environment variables.

Every tunable that affects backtest results lives here so a run can be
reproduced from its recorded config alone.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
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
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]
    log_buffer_size: int = 500


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
