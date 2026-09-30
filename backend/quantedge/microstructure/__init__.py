"""Short-horizon market microstructure research.

The equity side of this platform works on daily bars and rebalances monthly.
This package works at the other end of the time scale: top-of-book quotes and
individual trades, aggregated to one-second bars, asking whether liquidity at
the touch and the direction of order flow carry information about the next few
seconds of price action — and whether that information is worth anything once
it has to pay for its own execution.

Modules, in pipeline order:

* ``binance``   — download and checksum-verify public L1 + trade archives
* ``bars``      — stream tens of millions of events into 1-second bars,
                  computing order-flow imbalance event by event
* ``features``  — queue imbalance, normalised OFI, trade-flow imbalance,
                  regime variables and forward-return labels
* ``research``  — IC by horizon, contemporaneous vs predictive R², regime
                  splits, intraday stability, walk-forward model
* ``execution`` — market vs limit order simulation, with a signal-conditioned
                  policy chosen on one day and evaluated on the next
"""
