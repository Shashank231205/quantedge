#!/usr/bin/env bash
# One-time seed of a deployed QUANTEDGE database.
#
# The API serves nothing but empty states until this has run: a fresh cloud
# Postgres has the schema (migrations run on deploy) but no market data.
#
# Run it from a Render shell on the quantedge-api service, where DATABASE_URL
# already points at the managed database:
#
#     bash scripts/seed_cloud.sh
#
# Takes roughly 15-25 minutes, dominated by the yfinance backfill. It is
# idempotent — safe to re-run if a step fails partway.

set -euo pipefail

if [[ -z "${DATABASE_URL:-}" ]]; then
    echo "DATABASE_URL is not set — refusing to seed an unknown database." >&2
    exit 1
fi

# Show which host we are about to write to, with the password stripped.
echo "==> Target: $(printf '%s' "$DATABASE_URL" | sed -E 's#//[^@]*@#//***@#')"

step() { echo; echo "==> $1"; }

step "Applying migrations"
alembic upgrade head

step "Reconstructing point-in-time universe"
python -m quantedge.cli universe

step "Backfilling OHLCV (this is the slow one)"
python -m quantedge.cli ingest

step "Computing factors and IC diagnostics"
python -m quantedge.cli factors

step "Running walk-forward backtest"
python -m quantedge.cli backtest

step "Coverage"
python -m quantedge.cli status

echo
echo "Seed complete. The API now has data to serve."
