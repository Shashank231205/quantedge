#!/usr/bin/env bash
# Seed a remote database from this machine.
#
# Render's free tier has no shell, so scripts/seed_cloud.sh -- which assumes it
# runs inside the container -- cannot be used there. This does the same work
# over the database's public endpoint instead: the pipeline only needs a
# DATABASE_URL, and it does not care whether the database is local or hosted.
#
# Put the provider's external connection string in backend/.env.render:
#
#     RENDER_DATABASE_URL=postgresql://user:pass@host.oregon-postgres.render.com/db
#
# That file is gitignored. Then:
#
#     bash scripts/seed_remote.sh
#
# Takes 20-30 minutes, most of it the OHLCV backfill, and is slower than the
# in-container version because every write crosses the network. It is
# idempotent, so a failed step can be re-run without starting over.

set -euo pipefail
cd "$(dirname "$0")/.."

if [[ ! -f .env.render ]]; then
    echo "Missing backend/.env.render — add RENDER_DATABASE_URL to it first." >&2
    exit 1
fi

# shellcheck disable=SC1091
set -a; source .env.render; set +a

if [[ -z "${RENDER_DATABASE_URL:-}" ]]; then
    echo "RENDER_DATABASE_URL is not set in backend/.env.render." >&2
    exit 1
fi

# The app normalises the driver prefix itself, but alembic reads the URL
# directly, so hand it a form SQLAlchemy can resolve without psycopg2.
export DATABASE_URL="${RENDER_DATABASE_URL/postgresql:\/\//postgresql+psycopg://}"

# Show the target with the password stripped, so a mistyped host is caught
# before twenty minutes of ingest go to the wrong database.
echo "==> Target: $(printf '%s' "$DATABASE_URL" | sed -E 's#//[^@]*@#//***@#')"
echo

PY=.venv/bin/python
step() { echo; echo "==> $1"; }

step "Applying migrations"
.venv/bin/alembic upgrade head

step "Reconstructing point-in-time universe"
$PY -m quantedge.cli universe

step "Backfilling OHLCV (the slow one — expect ~15 minutes)"
$PY -m quantedge.cli ingest

step "Computing factors and IC diagnostics"
$PY -m quantedge.cli factors

step "Running walk-forward backtest"
$PY -m quantedge.cli backtest

step "Coverage"
$PY -m quantedge.cli status

echo
echo "Seed complete. The deployed API now has data to serve."
