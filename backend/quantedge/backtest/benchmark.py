"""Head-to-head runtime benchmark: naive loop vs. vectorized engine.

The reported speedup is only credible if the two engines produce identical
results, so this harness *verifies parity as part of the measurement* rather
than trusting the separate test suite. A run that produces different numbers
is reported as invalid regardless of how fast it was.

Both engines run on the same data, in the same process, with the same config.
"""

from __future__ import annotations

import platform
import sys
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from quantedge.backtest.costs import DEFAULT_COST_MODEL, CostModel
from quantedge.backtest.naive import NaiveBacktestEngine
from quantedge.backtest.portfolio import PortfolioConfig
from quantedge.backtest.vectorized import VectorizedBacktestEngine
from quantedge.logging_config import get_logger

log = get_logger(__name__)


@dataclass
class BenchmarkResult:
    n_bars: int
    n_tickers: int
    n_cells: int
    naive_seconds: float
    vectorized_seconds: float
    speedup: float
    reduction_pct: float
    results_match: bool
    max_abs_difference: float
    n_repeats: int = 1
    naive_times: list[float] = field(default_factory=list)
    vectorized_times: list[float] = field(default_factory=list)
    environment: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        verdict = "VERIFIED IDENTICAL" if self.results_match else "!! RESULTS DIFFER !!"
        return (
            f"\n{'=' * 62}\n"
            f"  BACKTEST ENGINE BENCHMARK\n"
            f"{'=' * 62}\n"
            f"  Workload      : {self.n_bars:,} bars x {self.n_tickers:,} tickers "
            f"= {self.n_cells:,} cells\n"
            f"  Repeats       : {self.n_repeats}\n"
            f"  Naive loop    : {self.naive_seconds:8.3f}s\n"
            f"  Vectorized    : {self.vectorized_seconds:8.3f}s\n"
            f"  Speedup       : {self.speedup:8.1f}x\n"
            f"  Reduction     : {self.reduction_pct:8.1f}%\n"
            f"  Parity        : {verdict} (max diff {self.max_abs_difference:.2e})\n"
            f"{'=' * 62}\n"
        )


def _environment() -> dict:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }


def compare_engines(
    prices: pd.DataFrame,
    scores: pd.DataFrame,
    config: PortfolioConfig | None = None,
    cost_model: CostModel | None = None,
    sectors: pd.Series | None = None,
    repeats: int = 1,
) -> BenchmarkResult:
    """Run both engines on identical inputs and compare time and output."""
    config = config or PortfolioConfig()
    cost_model = cost_model or DEFAULT_COST_MODEL

    naive_engine = NaiveBacktestEngine(config=config, cost_model=cost_model)
    vec_engine = VectorizedBacktestEngine(config=config, cost_model=cost_model)

    naive_times: list[float] = []
    vec_times: list[float] = []
    naive_res = vec_res = None

    for i in range(repeats):
        log.info("benchmark.repeat %s/%s", i + 1, repeats)
        naive_res = naive_engine.run(prices, scores, sectors)
        naive_times.append(naive_res.runtime_seconds)

        vec_res = vec_engine.run(prices, scores, sectors)
        vec_times.append(vec_res.runtime_seconds)

    assert naive_res is not None and vec_res is not None

    # Best-of, to reduce the influence of unrelated system load.
    naive_t = min(naive_times)
    vec_t = min(vec_times)

    diff = float(
        np.nanmax(np.abs(naive_res.returns.to_numpy() - vec_res.returns.to_numpy()))
    )
    matches = bool(diff < 1e-9)
    if not matches:
        log.error("benchmark.parity_failed max_diff=%.3e", diff)

    speedup = naive_t / vec_t if vec_t > 0 else float("inf")

    return BenchmarkResult(
        n_bars=len(prices),
        n_tickers=prices.shape[1],
        n_cells=int(prices.size),
        naive_seconds=naive_t,
        vectorized_seconds=vec_t,
        speedup=speedup,
        # The headline figure: how much wall-clock time vectorization removed.
        reduction_pct=100.0 * (1.0 - vec_t / naive_t) if naive_t > 0 else 0.0,
        results_match=matches,
        max_abs_difference=diff,
        n_repeats=repeats,
        naive_times=naive_times,
        vectorized_times=vec_times,
        environment=_environment(),
    )


def scaling_benchmark(
    prices: pd.DataFrame,
    scores: pd.DataFrame,
    ticker_counts: tuple[int, ...] = (50, 100, 250, 500),
    config: PortfolioConfig | None = None,
    repeats: int = 3,
    warmup: bool = True,
) -> pd.DataFrame:
    """Measure how each engine scales with universe size.

    The naive engine is O(bars x tickers) in interpreted Python, so its cost
    grows roughly linearly with the universe; the vectorized engine pushes the
    same work into BLAS-backed array ops and stays close to flat.

    A warmup pass and best-of-N repeats matter here: the first run in a
    process pays import, JIT and cache costs that would otherwise show up as
    a nonsensical result (a small universe appearing slower than a large one).
    """
    if warmup and len(prices.columns) >= 20:
        cols = prices.columns[:20]
        compare_engines(prices[cols], scores[cols], config=config, repeats=1)

    rows = []
    for n in ticker_counts:
        if n > prices.shape[1]:
            continue
        cols = prices.columns[:n]
        res = compare_engines(prices[cols], scores[cols], config=config, repeats=repeats)
        rows.append(
            {
                "n_tickers": n,
                "n_bars": res.n_bars,
                "naive_seconds": round(res.naive_seconds, 4),
                "vectorized_seconds": round(res.vectorized_seconds, 4),
                "speedup": round(res.speedup, 1),
                "reduction_pct": round(res.reduction_pct, 2),
                "results_match": res.results_match,
            }
        )
        log.info(
            "scaling n=%s naive=%.2fs vec=%.3fs speedup=%.0fx",
            n, res.naive_seconds, res.vectorized_seconds, res.speedup,
        )
    return pd.DataFrame(rows)
