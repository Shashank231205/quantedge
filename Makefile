.PHONY: help install db-up db-init migrate ingest factors backtest benchmark test lint ui up down clean

PY := backend/.venv/bin/python
PIP := backend/.venv/bin/pip
CLI := cd backend && .venv/bin/python -m quantedge.cli

help:
	@echo "QUANTEDGE"
	@echo "  make install    - create venv + install deps"
	@echo "  make db-up      - start postgres"
	@echo "  make migrate    - apply alembic migrations"
	@echo "  make ingest     - backfill OHLCV for the universe"
	@echo "  make factors    - compute all factors"
	@echo "  make backtest   - run walk-forward backtest"
	@echo "  make benchmark  - naive vs vectorized runtime"
	@echo "  make test       - pytest suite"
	@echo "  make lint       - ruff + mypy"
	@echo "  make up         - docker compose up"

install:
	python3.11 -m venv backend/.venv
	$(PIP) install --upgrade pip
	cd backend && .venv/bin/pip install -e ".[dev]"

db-up:
	brew services start postgresql@18 || true
	@sleep 2
	@pg_isready

migrate:
	cd backend && .venv/bin/alembic upgrade head

revision:
	cd backend && .venv/bin/alembic revision --autogenerate -m "$(m)"

ingest:
	$(CLI) ingest

factors:
	$(CLI) factors

backtest:
	$(CLI) backtest

benchmark:
	$(CLI) benchmark

serve:
	cd backend && .venv/bin/uvicorn quantedge.api.main:app --reload --port 8000

test:
	cd backend && .venv/bin/pytest -v

test-fast:
	cd backend && .venv/bin/pytest -v -m "not slow and not integration"

lint:
	cd backend && .venv/bin/ruff check quantedge tests
	cd backend && .venv/bin/ruff format --check quantedge tests

fmt:
	cd backend && .venv/bin/ruff check --fix quantedge tests
	cd backend && .venv/bin/ruff format quantedge tests

ui:
	cd frontend && npm run dev

up:
	docker compose up --build

down:
	docker compose down

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf backend/.pytest_cache backend/.ruff_cache
