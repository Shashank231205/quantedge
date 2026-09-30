"""Microstructure pipeline correctness.

The research numbers are only as good as three mechanical things: OFI computed
exactly as defined, no feature seeing a future second, and a fill simulator
that never credits a fill the book did not prove. Each has a hand-checkable
test here, plus an end-to-end run on a synthetic market whose signal is known.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantedge.microstructure.bars import (
    aggregate_book,
    aggregate_trades,
    build_bars,
    ofi_increments,
)
from quantedge.microstructure.execution import FeeSchedule, evaluate_execution, simulate_orders
from quantedge.microstructure.features import FEATURES, build_features, infer_tick_size
from quantedge.microstructure.research import (
    contemporaneous_vs_predictive,
    ic_by_horizon,
    walk_forward_days,
    walk_forward_model,
)

T0 = 1_696_204_800  # 2023-10-02 00:00:00 UTC


def book_events(rows: list[tuple], start_ms: int = T0 * 1000) -> pd.DataFrame:
    """(ms offset, bid, bid_qty, ask, ask_qty) tuples to a bookTicker frame."""
    return pd.DataFrame(
        [
            {"bid_px": b, "bid_qty": bq, "ask_px": a, "ask_qty": aq, "ts_ms": start_ms + off}
            for off, b, bq, a, aq in rows
        ]
    )


class TestOrderFlowImbalance:
    def test_hand_computed_increments(self):
        prev = pd.Series({"bid_px": 100.0, "bid_qty": 5.0, "ask_px": 101.0, "ask_qty": 5.0})
        events = book_events(
            [
                (0, 100.0, 7.0, 101.0, 5.0),  # bid size +2 at same price       -> +2
                (1, 100.5, 3.0, 101.0, 5.0),  # bid improves, adds 3             -> +3
                (2, 100.5, 3.0, 100.8, 4.0),  # ask improves, adds 4 supply      -> -4
                (3, 100.0, 9.0, 101.0, 6.0),  # bid drops (-3), ask retreats (+4) -> +1
            ]
        )
        np.testing.assert_allclose(ofi_increments(events, prev), [2.0, 3.0, -4.0, 1.0])

    def test_first_event_without_history_is_zero(self):
        events = book_events([(0, 100.0, 7.0, 101.0, 5.0), (1, 100.0, 9.0, 101.0, 5.0)])
        np.testing.assert_allclose(ofi_increments(events), [0.0, 2.0])

    def test_chunking_does_not_change_the_bars(self):
        """Seconds that straddle chunk boundaries must merge to the one-pass answer."""
        rng = np.random.default_rng(7)
        n = 5_000
        mid = 100 + np.cumsum(rng.choice([-0.01, 0, 0.01], size=n, p=[0.1, 0.8, 0.1]))
        events = pd.DataFrame(
            {
                "bid_px": np.round(mid - 0.005, 3),
                "bid_qty": rng.uniform(0.1, 10, n),
                "ask_px": np.round(mid + 0.005, 3),
                "ask_qty": rng.uniform(0.1, 10, n),
                "ts_ms": T0 * 1000 + np.sort(rng.integers(0, 120_000, n)),
            }
        )
        whole = aggregate_book([events]).bars
        chunked = aggregate_book(
            [events.iloc[i : i + 333] for i in range(0, n, 333)]
        ).bars
        pd.testing.assert_frame_equal(whole, chunked)
        assert whole["ofi"].sum() == pytest.approx(ofi_increments(events).sum())


class TestBars:
    def test_grid_is_gap_free_with_carried_state(self):
        book = aggregate_book(
            [book_events([(0, 100.0, 1.0, 100.1, 2.0), (3_500, 100.1, 4.0, 100.2, 1.0)])]
        ).bars
        bars = build_bars(book, pd.DataFrame())
        assert len(bars) == 4
        # Seconds 1 and 2 had no update: state carried, flow zero.
        assert bars["bid_px"].tolist() == [100.0, 100.0, 100.0, 100.1]
        assert bars["n_updates"].tolist() == [1, 0, 0, 1]
        assert bars["ofi"].iloc[1:3].eq(0).all()
        # The book at the start of second 3 was the old ask, so it counts.
        assert bars["ask_min"].iloc[3] == 100.1

    def test_trade_aggressor_split(self):
        trades = pd.DataFrame(
            {
                "price": [100.0, 99.9, 100.2],
                "qty": [1.0, 2.0, 3.0],
                "ts_ms": [T0 * 1000, T0 * 1000 + 10, T0 * 1000 + 20],
                "is_buyer_maker": [False, True, False],
            }
        )
        agg = aggregate_trades(trades).iloc[0]
        assert agg["buy_qty"] == 4.0
        assert agg["sell_qty"] == 2.0
        assert agg["sell_min_px"] == 99.9
        assert agg["buy_max_px"] == 100.2


def synthetic_bars(days: int = 3, edge: float = 0.35, seed: int = 11) -> pd.DataFrame:
    """A one-tick-spread market in which queue imbalance genuinely predicts.

    Each second the mid moves one tick with a probability that tilts toward the
    heavier side of the book by ``edge``. With ``edge=0`` it is a fair coin and
    nothing should look predictive.
    """
    rng = np.random.default_rng(seed)
    n = days * 86_400
    tick = 0.01
    qi = rng.uniform(-0.9, 0.9, n)
    move = rng.random(n) < 0.3
    up_prob = 0.5 + edge * qi
    direction = np.where(rng.random(n) < up_prob, 1, -1)
    steps = np.concatenate(([0], (move * direction)[:-1]))  # move at t+1 driven by qi at t
    bid = 1000.0 + np.cumsum(steps) * tick
    ask = bid + tick
    depth = rng.uniform(5, 50, n)
    bid_qty = depth * (1 + qi) / 2
    ask_qty = depth * (1 - qi) / 2
    buy = rng.exponential(1.0, n)
    sell = rng.exponential(1.0, n)
    index = pd.RangeIndex(T0, T0 + n, name="second")
    next_bid = np.concatenate((bid[1:], [bid[-1]]))
    next_ask = np.concatenate((ask[1:], [ask[-1]]))
    return pd.DataFrame(
        {
            "time": pd.to_datetime(index, unit="s", utc=True),
            "ofi": rng.normal(0, 5, n) + 10 * qi,
            "depth_sum": depth * 10,
            "n_updates": np.full(n, 10),
            "bid_px": bid,
            "bid_qty": bid_qty,
            "ask_px": ask,
            "ask_qty": ask_qty,
            "ask_min": np.minimum(ask, np.concatenate(([ask[0]], ask[:-1]))),
            "bid_max": np.maximum(bid, np.concatenate(([bid[0]], bid[:-1]))),
            "buy_qty": buy,
            "sell_qty": sell,
            "n_trades": np.full(n, 3),
            # Sells print at the bid, and through it when the price is about to fall.
            "sell_min_px": np.where(next_bid < bid, bid - tick, bid),
            "buy_max_px": np.where(next_ask > ask, ask + tick, ask),
        },
        index=index,
    )


class TestFeatures:
    def test_tick_size_is_read_from_prices(self):
        assert infer_tick_size(pd.Series([1733.39, 1733.40, 1733.42])) == 0.01

    def test_queue_imbalance_value(self):
        bars = synthetic_bars(days=1).iloc[:600]
        df = build_features(bars, tick=0.01)
        row = bars.iloc[400]
        expected = (row["bid_qty"] - row["ask_qty"]) / (row["bid_qty"] + row["ask_qty"])
        assert df["qi"].iloc[400] == pytest.approx(expected)

    def test_features_never_see_the_future(self):
        """Rewriting every bar after t must leave features at t untouched."""
        bars = synthetic_bars(days=1).iloc[:2_000]
        cut = 1_000
        tampered = bars.copy()
        rng = np.random.default_rng(0)
        for col in ("bid_qty", "ask_qty", "ofi", "buy_qty", "sell_qty", "depth_sum"):
            tampered.iloc[cut + 1 :, tampered.columns.get_loc(col)] = rng.uniform(
                1, 99, len(bars) - cut - 1
            )
        shift = tampered.iloc[cut + 1 :, tampered.columns.get_loc("bid_px")] + 5
        tampered.iloc[cut + 1 :, tampered.columns.get_loc("bid_px")] = shift
        tampered.iloc[cut + 1 :, tampered.columns.get_loc("ask_px")] = shift + 0.01

        a = build_features(bars, tick=0.01)
        b = build_features(tampered, tick=0.01)
        cols = [*FEATURES, "rv_300", "spread_ticks"]
        pd.testing.assert_frame_equal(a[cols].iloc[: cut + 1], b[cols].iloc[: cut + 1])

    def test_labels_are_forward_mid_returns(self):
        df = build_features(synthetic_bars(days=1).iloc[:500], tick=0.01)
        t = 100
        expected = (df["mid"].iloc[t + 10] / df["mid"].iloc[t] - 1) * 1e4
        assert df["fwd_10"].iloc[t] == pytest.approx(expected)

    def test_gap_between_days_breaks_windows(self):
        bars = synthetic_bars(days=1).iloc[:1_000]
        gapped = pd.concat([bars.iloc[:400], bars.iloc[700:]])
        df = build_features(gapped, tick=0.01)
        # Labels that would reach across the missing block are undefined.
        assert np.isnan(df["fwd_10"].loc[T0 + 395])
        assert np.isnan(df["ofi_30"].loc[T0 + 710])


@pytest.fixture(scope="module")
def predictive():
    return build_features(synthetic_bars(days=3, edge=0.35), tick=0.01)


@pytest.fixture(scope="module")
def noise():
    return build_features(synthetic_bars(days=3, edge=0.0), tick=0.01)


class TestResearch:
    def test_recovers_a_real_signal(self, predictive):
        ic = {(r["feature"], r["horizon_s"]): r for r in ic_by_horizon(predictive)}
        assert ic[("qi", 1)]["ic"] > 0.1
        assert ic[("qi", 1)]["t_stat"] > 10
        assert ic[("qi", 1)]["days_positive"] == ic[("qi", 1)]["days"]

    def test_finds_nothing_in_noise(self, noise):
        ic = {(r["feature"], r["horizon_s"]): r for r in ic_by_horizon(noise)}
        assert abs(ic[("qi", 1)]["t_stat"]) < 4

    def test_non_overlapping_sample_sizes(self, predictive):
        rows = {(r["feature"], r["horizon_s"]): r["n"] for r in ic_by_horizon(predictive)}
        # Sampling every h seconds: the 60s horizon has ~1/60th the observations.
        assert rows[("qi", 60)] == pytest.approx(rows[("qi", 1)] / 60, rel=0.05)

    def test_walk_forward_trains_strictly_before_testing(self, predictive):
        for day, train, test in walk_forward_days(predictive, embargo=10):
            assert train["day"].max() < day
            assert (test["day"] == day).all()
            # The embargo removes the tail whose labels reach into the test day.
            assert train.index.max() + 10 < test.index.min()

    def test_walk_forward_model_is_out_of_sample(self, predictive):
        wf = walk_forward_model(predictive, horizon=1)
        assert len(wf["folds"]) == 2
        assert wf["pooled_oos_ic"] > 0.1

    def test_contemporaneous_block_sums(self, predictive):
        rows = contemporaneous_vs_predictive(predictive, intervals=(1, 10))
        assert rows[0]["n_intervals"] > rows[1]["n_intervals"] * 9


def exec_frame(bid: list[float], sell_min: list[float], ask_min: list[float] | None = None):
    """Minimal frame for the fill simulator: one-cent spread, flat book otherwise."""
    n = len(bid)
    bid_a = np.array(bid, dtype=float)
    ask = bid_a + 0.01
    return pd.DataFrame(
        {
            "bid_px": bid_a,
            "ask_px": ask,
            "mid": (bid_a + ask) / 2,
            "ask_min": np.array(ask_min, dtype=float) if ask_min else ask,
            "bid_max": bid_a,
            "sell_min_px": np.array(sell_min, dtype=float),
            "buy_max_px": np.full(n, np.nan),
        },
        index=pd.RangeIndex(n),
    )


class TestExecution:
    FEES = FeeSchedule(maker_bps=2.0, taker_bps=5.0)
    N = 40

    def buy(self, frame, fill_model="conservative"):
        sim = simulate_orders(frame, horizon=5, stride=100, fees=self.FEES, fill_model=fill_model)
        return sim[sim["side"] == 1].iloc[0]

    def test_trade_through_fills_conservatively(self):
        sell = [np.nan] * self.N
        sell[3] = 99.99  # a sell prints below our 100.00 bid
        row = self.buy(exec_frame([100.0] * self.N, sell))
        assert row["limit_filled"]
        assert row["limit_wait_s"] == 3
        half_spread = 0.005 / 100.005 * 1e4
        assert row["limit_cost_bps"] == pytest.approx(-half_spread + 2.0)
        assert row["market_cost_bps"] == pytest.approx(half_spread + 5.0)

    def test_trade_at_our_price_needs_the_optimistic_model(self):
        sell = [np.nan] * self.N
        sell[2] = 100.0  # trades at our price: we may or may not be in the queue
        frame = exec_frame([100.0] * self.N, sell)
        assert not self.buy(frame, "conservative")["limit_filled"]
        assert self.buy(frame, "optimistic")["limit_filled"]

    def test_ask_dropping_to_our_bid_is_a_fill(self):
        ask_min = [100.01] * self.N
        ask_min[4] = 100.0
        row = self.buy(exec_frame([100.0] * self.N, [np.nan] * self.N, ask_min))
        assert row["limit_filled"]

    def test_unfilled_order_chases_at_the_later_ask(self):
        bid = [100.0] * 5 + [100.05] * (self.N - 5)  # price runs away from us
        row = self.buy(exec_frame(bid, [np.nan] * self.N))
        assert not row["limit_filled"]
        expected = (100.06 - 100.005) / 100.005 * 1e4 + 5.0
        assert row["limit_cost_bps"] == pytest.approx(expected)

    def test_fills_outside_the_wait_window_do_not_count(self):
        sell = [np.nan] * self.N
        sell[6] = 99.0  # one second after a 5s wait has expired
        assert not self.buy(exec_frame([100.0] * self.N, sell))["limit_filled"]

    def test_walk_forward_execution_runs_end_to_end(self):
        df = build_features(synthetic_bars(days=3, edge=0.35), tick=0.01)
        out = evaluate_execution(df, FEATURES, horizon=10, stride=60)
        assert out["n_decisions"] > 1_000
        assert set(out["policies"]) == {"market", "limit", "signal"}
        assert 0 < out["limit_detail"]["fill_rate"] < 1
        assert len(out["signal_detail"]["thresholds"]) == 2

    def test_overlapping_decisions_are_rejected(self):
        df = build_features(synthetic_bars(days=1).iloc[:2_000], tick=0.01)
        with pytest.raises(ValueError, match="stride"):
            evaluate_execution(df, FEATURES, horizon=30, stride=30)
