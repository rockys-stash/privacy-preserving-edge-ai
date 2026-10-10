.PHONY: help setup data test test-ui lint format web serve benchmark smoke report screenshots e1 e2 e3 e4 e5 clean

SMOKE_SUBJECTS ?= 1,2,3,5,6,7,8,9

help: ## List targets
	@grep -E '^[a-z0-9-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

setup: ## Install Python (uv) and console (npm) dependencies
	uv sync --locked
	npm --prefix web ci

data: ## Download the UCI HAR archive and verify its SHA-256 (data/raw/, not committed)
	uv run edgebench fetch

benchmark: data ## THE one command: run E1-E5 on all 30 subjects, then regenerate docs/REPORT.md and figures
	OMP_NUM_THREADS=1 uv run edgebench experiment all
	uv run edgebench report

e1: data ## E1 frontier grid (3 architectures x 5 levels, 5 folds x 3 seeds)
	OMP_NUM_THREADS=1 uv run edgebench experiment e1
e2: data ## E2 adversary strength (needs E1)
	OMP_NUM_THREADS=1 uv run edgebench experiment e2
e3: data ## E3 epsilon sweep (needs E1)
	OMP_NUM_THREADS=1 uv run edgebench experiment e3
e4: data ## E4 cost and break-even
	OMP_NUM_THREADS=1 uv run edgebench experiment e4
e5: data ## E5 streaming replay against batch
	OMP_NUM_THREADS=1 uv run edgebench experiment e5

report: ## Regenerate docs/generated/results.md and docs/figures/ from the latest runs
	uv run edgebench report

smoke: data ## E1-E5 on 8 subjects with tiny models into results/scratch (checks the pipeline; not a result)
	EDGEBENCH_SUBJECTS=$(SMOKE_SUBJECTS) uv run edgebench experiment all --configs configs/smoke --results results/scratch
	uv run edgebench report --results results/scratch --out results/scratch/report

test: ## Unit, streaming-consistency and API tests (8-subject subset)
	uv run pytest

test-ui: web ## Browser, keyboard and axe accessibility tests (needs Chromium)
	uv run pytest -m ui

lint: ## Ruff, mypy, ESLint and TypeScript checks
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy src tests scripts
	npm --prefix web run lint
	npm --prefix web run typecheck

format: ## Auto-format Python
	uv run ruff format .
	uv run ruff check --fix .

web: ## Build the console into web/dist
	npm --prefix web run build

serve: web ## Serve the console and API on http://127.0.0.1:8000 (reads .env if present)
	set -a; [ -f .env ] && . ./.env; set +a; uv run edgebench serve

screenshots: web ## Capture console screenshots into docs/screenshots (needs Chromium)
	uv run python scripts/screenshots.py

clean: ## Remove build output, caches and smoke runs (keeps data/ and results/)
	rm -rf web/dist results/scratch .pytest_cache .mypy_cache .ruff_cache
