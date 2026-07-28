.PHONY: help install db-up migrate revision universe ingest factors backtest \
        benchmark signals status serve schedule ui test test-fast lint fmt \
        up down clean all

PY  := backend/.venv/bin/python
PIP := backend/.venv/bin/pip
CLI := cd backend && .venv/bin/python -m quantedge.cli

help:
	@echo "QUANTEDGE"
	@echo ""
	@echo "  Setup"
	@echo "    make install     create the venv and install dependencies"
	@echo "    make db-up       start PostgreSQL"
	@echo "    make migrate     apply database migrations"
	@echo ""
	@echo "  Pipeline"
	@echo "    make universe    reconstruct point-in-time S&P 500 membership"
	@echo "    make ingest      backfill OHLCV (~3 min, 567 tickers x 6 years)"
	@echo "    make factors     compute IC, decay and cross-factor correlation"
	@echo ""
	@echo "  Research"
	@echo "    make backtest    walk-forward validation with honest metrics"
	@echo "    make benchmark   naive vs vectorized, with parity verification"
	@echo "    make signals     current strategy ranking"
	@echo "    make status      data coverage and pipeline uptime"
	@echo ""
	@echo "  Services"
	@echo "    make serve       API on :8000  (docs at /docs)"
	@echo "    make ui          dashboard on :5173"
	@echo "    make schedule    run the job scheduler"
	@echo "    make up          full stack via Docker"
	@echo ""
	@echo "  Quality"
	@echo "    make test        full test suite"
	@echo "    make lint        ruff checks"

# --- setup ---------------------------------------------------------------

install:
	python3.11 -m venv backend/.venv
	$(PIP) install --upgrade pip
	cd backend && .venv/bin/pip install -e ".[dev]"
	cd frontend && npm install

db-up:
	brew services start postgresql@18 || true
	@sleep 2
	@pg_isready

migrate:
	cd backend && .venv/bin/alembic upgrade head

revision:
	cd backend && .venv/bin/alembic revision --autogenerate -m "$(m)"

# --- pipeline ---------------------------------------------------------------

universe:
	$(CLI) universe

ingest:
	$(CLI) ingest

ingest-daily:
	$(CLI) ingest --incremental

factors:
	$(CLI) factors

# --- research ----------------------------------------------------------------

backtest:
	$(CLI) backtest

backtest-insample:
	$(CLI) backtest --no-walk-forward

benchmark:
	$(CLI) benchmark

signals:
	$(CLI) signals

status:
	$(CLI) status

# Full cold-start: schema through to validated results.
all: migrate universe ingest factors backtest benchmark

# --- services -------------------------------------------------------------------

serve:
	cd backend && .venv/bin/uvicorn quantedge.api.main:app --reload --port 8000

schedule:
	$(CLI) schedule

ui:
	cd frontend && npm run dev

up:
	docker compose up --build

down:
	docker compose down

# --- quality ----------------------------------------------------------------------

test:
	cd backend && .venv/bin/pytest -v

test-fast:
	cd backend && .venv/bin/pytest -q -m "not slow and not integration"

lint:
	cd backend && .venv/bin/ruff check quantedge tests
	cd frontend && npx tsc -b --noEmit

fmt:
	cd backend && .venv/bin/ruff check --fix quantedge tests
	cd backend && .venv/bin/ruff format quantedge tests

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf backend/.pytest_cache backend/.ruff_cache frontend/dist
