API := apps/api
WEB := apps/web

.DEFAULT_GOAL := help
.PHONY: help setup up down logs reset dev-api dev-worker dev-web test lint fix types check eval eval-retrieval

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-16s\033[0m %s\n", $$1, $$2}'

setup: ## Install dependencies for both apps
	cd $(API) && uv sync --group dev
	cd $(WEB) && npm install
	@test -f .env || (cp .env.example .env && echo "Created .env — set OLLAMA_API_KEY in it")

# --- docker -------------------------------------------------------------------

up: ## Start the whole stack (qdrant, redis, api, worker, web)
	docker compose up --build

down: ## Stop the stack
	docker compose down

logs: ## Tail API and worker logs
	docker compose logs -f api worker

reset: ## Stop the stack and delete all data, indexes and uploads
	docker compose down -v

# --- local development --------------------------------------------------------
# Needs qdrant and redis reachable; `docker compose up qdrant redis` is enough.

dev-api: ## Run the API with reload
	cd $(API) && uv run uvicorn rag.main:app --reload --port 8000

dev-worker: ## Run the ingestion worker
	cd $(API) && uv run arq rag.jobs.worker.WorkerSettings

dev-web: ## Run the web app (needs Node 20+)
	cd $(WEB) && npm run dev

# --- quality ------------------------------------------------------------------

test: ## Run the API test suite
	cd $(API) && uv run pytest

lint: ## Lint both apps
	cd $(API) && uv run ruff check src tests && uv run ruff format --check src tests
	cd $(WEB) && npx biome check .

fix: ## Auto-fix lint and formatting in both apps
	cd $(API) && uv run ruff check --fix src tests && uv run ruff format src tests
	cd $(WEB) && npx biome check --write .

types: ## Typecheck both apps
	cd $(API) && uv run mypy src
	cd $(WEB) && npx tsc --noEmit

check: lint types test ## Everything CI runs

# --- evaluation ---------------------------------------------------------------

eval: ## Score the golden set end to end (needs an ingested corpus)
	cd $(API) && uv run rag-eval run --dataset ../../evals/golden.jsonl --judge

eval-retrieval: ## Score retrieval only — fast, free, no LLM calls
	cd $(API) && uv run rag-eval run --dataset ../../evals/golden.jsonl --retrieval-only
