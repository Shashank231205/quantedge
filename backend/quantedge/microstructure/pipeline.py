"""End-to-end microstructure study: archives to a single results document.

Bars are cached per symbol-day as parquet. Building them means streaming 15-40
million quote updates, which takes on the order of a minute a day; the
research itself runs on the cached bars in seconds, so iterating on features
never re-pays the aggregation.
"""

from __future__ import annotations

import time
from datetime import date
from pathlib import Path

import pandas as pd

from quantedge.logging_config import get_logger
from quantedge.microstructure import binance
from quantedge.microstructure.bars import aggregate_book, aggregate_trades, build_bars
from quantedge.microstructure.execution import REGULAR_FEES, FeeSchedule, evaluate_execution
from quantedge.microstructure.features import FEATURES, build_features
from quantedge.microstructure.research import (
    contemporaneous_vs_predictive,
    ic_by_horizon,
    ic_by_regime,
    ic_stability,
    walk_forward_model,
)

log = get_logger(__name__)

#: Binance USDⓈ-M futures fee tiers as published during the sample period:
#: the regular tier every new account starts on, a mid tier, and the top tier.
FEE_TIERS = {
    "regular": REGULAR_FEES,
    "vip3": FeeSchedule(maker_bps=1.2, taker_bps=3.2),
    "vip9": FeeSchedule(maker_bps=0.0, taker_bps=1.7),
}


def day_bars(symbol: str, day: date, raw_dir: Path, bars_dir: Path) -> tuple[pd.DataFrame, dict]:
    """One day's one-second bars, from cache when present."""
    bars_dir.mkdir(parents=True, exist_ok=True)
    cached = bars_dir / f"{symbol}-{day.isoformat()}-1s.parquet"
    if cached.exists():
        bars = pd.read_parquet(cached)
        return bars, {"day": day.isoformat(), "cached": True, **bars.attrs.get("stats", {})}

    started = time.perf_counter()
    book_zip = binance.download(symbol, "bookTicker", day, raw_dir)
    trade_zip = binance.download(symbol, "aggTrades", day, raw_dir)

    book = aggregate_book(binance.read_book_ticker(book_zip))
    trades_raw = binance.read_agg_trades(trade_zip)
    trades = aggregate_trades(trades_raw)
    bars = build_bars(book.bars, trades)

    stats = {
        "book_events": int(book.events),
        "trades": int(len(trades_raw)),
        "seconds": int(len(bars)),
        "build_seconds": round(time.perf_counter() - started, 1),
    }
    bars.attrs["stats"] = stats
    bars.to_parquet(cached)
    log.info("micro.bars day=%s events=%s trades=%s", day, stats["book_events"], stats["trades"])
    return bars, {"day": day.isoformat(), "cached": False, **stats}


def load_dataset(
    symbol: str, start: date, end: date, raw_dir: Path, bars_dir: Path
) -> tuple[pd.DataFrame, list[dict]]:
    frames, manifest = [], []
    for day in binance.date_range(start, end):
        bars, info = day_bars(symbol, day, raw_dir, bars_dir)
        frames.append(bars)
        manifest.append(info)
    return pd.concat(frames), manifest


def run_study(
    symbol: str,
    start: date,
    end: date,
    raw_dir: Path,
    bars_dir: Path,
    fees: FeeSchedule = REGULAR_FEES,
    execution_horizons: tuple[int, ...] = (10, 30),
) -> dict:
    """Everything the Microstructure screen and README report, in one pass."""
    bars, manifest = load_dataset(symbol, start, end, raw_dir, bars_dir)
    df = build_features(bars)
    tick = df.attrs["tick"]

    valid = df.dropna(subset=["mid"])
    spread = valid["spread_ticks"]
    summary = {
        "symbol": symbol,
        "venue": "Binance USDⓈ-M futures",
        "start": start.isoformat(),
        "end": end.isoformat(),
        "days": len(manifest),
        "seconds": int(len(valid)),
        "book_events": int(sum(m.get("book_events", 0) for m in manifest)),
        "trades": int(sum(m.get("trades", 0) for m in manifest)),
        "tick_size": tick,
        "median_mid": round(float(valid["mid"].median()), 4),
        "pct_seconds_one_tick_spread": round(float((spread <= 1).mean()), 4),
        "half_spread_bps_median": round(
            float(((valid["ask_px"] - valid["bid_px"]) / 2 / valid["mid"] * 1e4).median()), 4
        ),
        "pct_seconds_mid_unchanged": round(float((valid["mid_change_ticks"] == 0).mean()), 4),
    }

    log.info("micro.research rows=%s days=%s", len(df), len(manifest))
    execution = {}
    for wait in execution_horizons:
        stride = max(60, wait + 30)
        execution[f"wait_{wait}s"] = {
            model: evaluate_execution(
                df, FEATURES, horizon=wait, stride=stride, fees=fees, fill_model=model
            )
            for model in ("conservative", "optimistic")
        }

    # Whether prediction can pay for itself depends on the gap between taker
    # and maker fees: at the regular tier a limit order saves 3bps before any
    # price move, which is more than a few-second forecast can recover. Re-run
    # the routing study across the published tiers to show where that changes.
    fee_sensitivity = []
    for tier, tier_fees in FEE_TIERS.items():
        result = evaluate_execution(
            df, FEATURES, horizon=10, stride=60, fees=tier_fees, fill_model="conservative"
        )
        fee_sensitivity.append(
            {
                "tier": tier,
                **tier_fees.as_dict(),
                **{f"{k}_mean_bps": v["mean_cost_bps"] for k, v in result["policies"].items()},
                "routed_market_pct": result["signal_detail"]["routed_market_pct"],
                "signal_vs_best_static": result["signal_detail"]["vs_best_static"],
            }
        )

    return {
        "dataset": summary,
        "manifest": manifest,
        "features": list(FEATURES),
        "ic_by_horizon": ic_by_horizon(df),
        "contemporaneous_vs_predictive": contemporaneous_vs_predictive(df),
        "ic_by_regime": ic_by_regime(df),
        "stability": ic_stability(df),
        "walk_forward_model": walk_forward_model(df),
        "execution": execution,
        "fee_sensitivity": fee_sensitivity,
    }


def cross_asset_summary(studies: dict[str, dict]) -> list[dict]:
    """One headline row per symbol, so robustness across assets is one glance."""
    rows = []
    for symbol, study in studies.items():
        ic = {(r["feature"], r["horizon_s"]): r for r in study["ic_by_horizon"]}
        r2 = {r["interval_s"]: r for r in study["contemporaneous_vs_predictive"]}
        wf = study["walk_forward_model"]
        conservative = study["execution"]["wait_10s"]["conservative"]
        rows.append(
            {
                "symbol": symbol,
                "book_events": study["dataset"]["book_events"],
                "pct_one_tick_spread": study["dataset"]["pct_seconds_one_tick_spread"],
                "qi_ic_10s": ic[("qi", 10)]["ic"],
                "qi_days_positive": f'{ic[("qi", 10)]["days_positive"]}/{ic[("qi", 10)]["days"]}',
                "ofi_r2_contemporaneous_10s": r2[10]["r2_contemporaneous"],
                "ofi_r2_predictive_10s": r2[10]["r2_predictive"],
                "oos_ic_10s": wf.get("pooled_oos_ic"),
                "oos_folds_positive": f'{wf.get("folds_positive_ic", 0)}/{len(wf["folds"])}',
                "limit_saving_vs_market_bps": round(
                    conservative["policies"]["market"]["mean_cost_bps"]
                    - conservative["policies"]["limit"]["mean_cost_bps"],
                    4,
                ),
                "limit_fill_rate": conservative["limit_detail"]["fill_rate"],
                "fill_markout_bps": conservative["limit_detail"]["fill_markout_bps"],
            }
        )
    return rows
