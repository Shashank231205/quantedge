"""FastAPI application."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from quantedge.api.middleware import LatencyMiddleware, flush_on_shutdown
from quantedge.api.routers import analyst, backtest, factors, portfolio, risk, system
from quantedge.api.ws import router as ws_router
from quantedge.config import settings
from quantedge.logging_config import configure_logging, get_logger

configure_logging()
log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("api.startup source=%s prefix=%s", settings.data_source, settings.api_prefix)

    # Building the price panel, factor signals and live portfolio takes several
    # seconds. Doing it on first request made that request a ~15s outlier that
    # dominated the p95 the System Health screen reports. Warm it in a worker
    # thread so startup stays non-blocking.
    import asyncio

    async def warm() -> None:
        await asyncio.gather(
            asyncio.to_thread(factors.warm_cache),
            asyncio.to_thread(risk.warm_cache),
        )
        log.info("api.cache_warm")

    task = asyncio.create_task(warm())

    yield

    task.cancel()
    flush_on_shutdown()
    log.info("api.shutdown")


app = FastAPI(
    title="QUANTEDGE API",
    description=(
        "Equity research and signal backtesting platform.\n\n"
        "Every metric served here is computed from real market data by this "
        "codebase. Results labelled *out-of-sample* come from walk-forward "
        "validation; anything labelled *in-sample* is explicitly marked as "
        "such and should not be quoted as a validated figure."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(LatencyMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_origin_regex=settings.cors_origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Response-Time-Ms"],
)

for r in (
    portfolio.router,
    factors.router,
    backtest.router,
    risk.router,
    system.router,
    analyst.router,
):
    app.include_router(r, prefix=settings.api_prefix)

app.include_router(ws_router)


@app.get("/health", tags=["meta"])
def health_probe() -> dict:
    """Unauthenticated liveness probe for Docker and load balancers.

    Delegates to the same implementation the versioned endpoint uses, so the
    two can never report different things.
    """
    return system.health()


@app.get("/", tags=["meta"])
def root() -> dict:
    return {
        "name": "QUANTEDGE",
        "version": "0.1.0",
        "docs": "/docs",
        "api_prefix": settings.api_prefix,
        "health": "/health",
        "screens": {
            "dashboard": f"{settings.api_prefix}/portfolio/summary",
            "factors": f"{settings.api_prefix}/factors/table",
            "backtest": f"{settings.api_prefix}/backtest/metrics",
            "risk": f"{settings.api_prefix}/risk/summary",
            "system": f"{settings.api_prefix}/system/status",
        },
    }


@app.exception_handler(Exception)
async def unhandled_exception(request, exc):  # pragma: no cover - safety net
    log.exception("api.unhandled path=%s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "path": str(request.url.path)},
    )
