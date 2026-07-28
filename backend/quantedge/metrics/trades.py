"""Trade-level statistics.

Derived from the weight panel rather than an order blotter: a "trade" is a
position held from the bar it becomes non-zero until it returns to zero or
flips sign. This keeps the trade log consistent with the equity curve, which
would not be guaranteed if trades were reconstructed separately.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def extract_trades(
    weights: pd.DataFrame, prices: pd.DataFrame, capital: float = 1_000_000.0
) -> pd.DataFrame:
    """Reconstruct discrete trades from a panel of daily target weights."""
    records: list[dict] = []

    for ticker in weights.columns:
        w = weights[ticker]
        if (w.abs() < 1e-12).all():
            continue

        px = prices[ticker] if ticker in prices.columns else None
        if px is None:
            continue

        in_trade = False
        side = 0
        entry_i = 0

        values = w.to_numpy()
        for i, weight in enumerate(values):
            active = abs(weight) > 1e-12
            current_side = int(np.sign(weight)) if active else 0

            if not in_trade and active:
                in_trade, side, entry_i = True, current_side, i
            elif in_trade and (not active or current_side != side):
                records.append(
                    _make_trade(ticker, w, px, entry_i, i, side, capital)
                )
                if active and current_side != side:
                    # A sign flip closes one trade and opens the opposite.
                    in_trade, side, entry_i = True, current_side, i
                else:
                    in_trade = False

        if in_trade:
            rec = _make_trade(ticker, w, px, entry_i, len(values) - 1, side, capital)
            rec["status"] = "OPEN"
            rec["exit_date"] = None
            records.append(rec)

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records).sort_values("entry_date").reset_index(drop=True)


def _make_trade(
    ticker: str,
    weights: pd.Series,
    prices: pd.Series,
    entry_i: int,
    exit_i: int,
    side: int,
    capital: float,
) -> dict:
    entry_date = weights.index[entry_i]
    exit_date = weights.index[exit_i]

    entry_px = float(prices.iloc[entry_i]) if not pd.isna(prices.iloc[entry_i]) else np.nan
    exit_px = float(prices.iloc[exit_i]) if not pd.isna(prices.iloc[exit_i]) else np.nan

    avg_weight = float(weights.iloc[entry_i:exit_i].abs().mean()) if exit_i > entry_i else abs(float(weights.iloc[entry_i]))

    if np.isnan(entry_px) or np.isnan(exit_px) or entry_px <= 0:
        pnl_pct = 0.0
    else:
        pnl_pct = side * (exit_px / entry_px - 1.0)

    return {
        "ticker": ticker,
        "side": "LONG" if side > 0 else "SHORT",
        "entry_date": entry_date,
        "exit_date": exit_date,
        "entry_price": entry_px,
        "exit_price": exit_px,
        "weight": avg_weight,
        "pnl_pct": float(pnl_pct),
        "pnl_abs": float(pnl_pct * avg_weight * capital),
        "holding_days": int(exit_i - entry_i),
        "status": "CLOSED",
    }


def trade_statistics(trades: pd.DataFrame) -> dict:
    """Win rate, profit factor, expectancy and friends."""
    if trades.empty:
        return {
            "n_trades": 0, "win_rate": 0.0, "profit_factor": 0.0,
            "avg_win": 0.0, "avg_loss": 0.0, "payoff_ratio": 0.0,
            "expectancy": 0.0, "avg_holding_days": 0.0,
        }

    closed = trades[trades["status"] == "CLOSED"] if "status" in trades else trades
    if closed.empty:
        closed = trades

    pnl = closed["pnl_pct"]
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    gross_profit = float(wins.sum())
    gross_loss = float(-losses.sum())

    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0
    win_rate = float(len(wins) / len(pnl)) if len(pnl) else 0.0

    return {
        "n_trades": int(len(closed)),
        "n_wins": int(len(wins)),
        "n_losses": int(len(losses)),
        "win_rate": round(win_rate, 4),
        # Total won per unit lost; > 1 means the book made money.
        "profit_factor": round(gross_profit / gross_loss, 4) if gross_loss > 1e-12 else 0.0,
        "avg_win": round(avg_win, 6),
        "avg_loss": round(avg_loss, 6),
        "payoff_ratio": round(abs(avg_win / avg_loss), 4) if avg_loss < -1e-12 else 0.0,
        # Expected P&L per trade — the number that decides long-run viability.
        "expectancy": round(win_rate * avg_win + (1 - win_rate) * avg_loss, 6),
        "best_trade": round(float(pnl.max()), 6),
        "worst_trade": round(float(pnl.min()), 6),
        "avg_holding_days": round(float(closed["holding_days"].mean()), 2),
        "long_trades": int((closed["side"] == "LONG").sum()),
        "short_trades": int((closed["side"] == "SHORT").sum()),
    }


def turnover_statistics(
    turnover: pd.Series, trading_days: int = 252
) -> dict:
    """Portfolio churn. Drives transaction costs and capacity."""
    t = turnover.dropna()
    if t.empty:
        return {"avg_daily_turnover": 0.0, "annual_turnover": 0.0}

    active = t[t > 1e-12]
    return {
        "avg_daily_turnover": round(float(t.mean()), 6),
        "avg_rebalance_turnover": round(float(active.mean()), 6) if len(active) else 0.0,
        "annual_turnover": round(float(t.mean() * trading_days), 4),
        "max_turnover": round(float(t.max()), 6),
        "n_rebalances": int(len(active)),
    }
