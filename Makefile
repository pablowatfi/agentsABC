.PHONY: help setup install env pull-model pull-embed-model run-backend run-ui \
        test test-health test-full test-missing test-nonweather \
        eval observability observability-stop clean

PYTHON  := python3
VENV    := .venv
BIN     := $(VENV)/bin
PIP     := $(BIN)/pip
UV      := $(BIN)/uvicorn
STREAM  := $(BIN)/streamlit
APP_DIR := weather_agents
BACKEND := http://localhost:8000

help:
	@echo ""
	@echo "  setup          create venv, install deps, copy .env"
	@echo "  install        install/sync dependencies only"
	@echo "  env            copy .env.example → .env (skip if exists)"
	@echo "  pull-model        pull the Ollama LLM set in .env"
	@echo "  pull-embed-model  pull the embedding model for semantic cache (nomic-embed-text)"
	@echo ""
	@echo "  run-backend    start FastAPI server (port 8000)"
	@echo "  run-ui         start Streamlit UI  (port 8501)"
	@echo ""
	@echo "  test           run all API tests"
	@echo "  test-health    check backend is alive"
	@echo "  test-full      question with full location (no clarification)"
	@echo "  test-missing   multi-turn: missing city → follow-up"
	@echo "  test-nonweather non-weather rejection"
	@echo ""
	@echo "  eval           run LLM-as-a-judge on all reference examples"
	@echo ""
	@echo "  observability  start Langfuse + Postgres via Docker"
	@echo "  observability-stop  stop and remove containers"
	@echo ""
	@echo "  clean          remove venv and __pycache__"
	@echo ""

# ── Setup ────────────────────────────────────────────────────────────────────

setup: $(VENV)/bin/activate env install
	@echo "✓ Setup complete. Edit $(APP_DIR)/.env then run: make pull-model pull-embed-model"

$(VENV)/bin/activate:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip

install: $(VENV)/bin/activate
	$(PIP) install -r $(APP_DIR)/requirements.txt

env:
	@if [ ! -f $(APP_DIR)/.env ]; then \
		cp $(APP_DIR)/.env.example $(APP_DIR)/.env; \
		echo "✓ Created $(APP_DIR)/.env — fill in TAVILY_API_KEY"; \
	else \
		echo "✓ $(APP_DIR)/.env already exists, skipping"; \
	fi

pull-model:
	@. $(APP_DIR)/.env && ollama pull $${OLLAMA_MODEL:-llama3}

pull-embed-model:
	@. $(APP_DIR)/.env && ollama pull $${CACHE_EMBED_MODEL:-nomic-embed-text}

# ── Run ──────────────────────────────────────────────────────────────────────

run-backend:
	@cd $(APP_DIR) && . .env && ../$(UV) app.main:app --reload

run-ui:
	$(STREAM) run $(APP_DIR)/ui/streamlit_app.py

# ── Tests ────────────────────────────────────────────────────────────────────

test: test-health test-full test-missing test-nonweather

test-health:
	@echo "\n── Health check ──────────────────────────"
	@curl -sf $(BACKEND)/health | python3 -m json.tool

test-full:
	@echo "\n── Full question (NYC temperature) ───────"
	@curl -sf -X POST $(BACKEND)/chat \
		-H "Content-Type: application/json" \
		-d '{"message": "What will be the temperature tomorrow in New York City?", "session_id": ""}' \
		| python3 -m json.tool

test-missing:
	@echo "\n── Multi-turn: missing city ──────────────"
	$(eval SESSION := $(shell curl -sf -X POST $(BACKEND)/chat \
		-H "Content-Type: application/json" \
		-d '{"message": "Will it rain tomorrow?", "session_id": ""}' \
		| python3 -c "import sys,json; print(json.load(sys.stdin)['session_id'])"))
	@echo "  Clarification response:"
	@curl -sf -X POST $(BACKEND)/chat \
		-H "Content-Type: application/json" \
		-d '{"message": "Will it rain tomorrow?", "session_id": ""}' \
		| python3 -c "import sys,json; print('  >', json.load(sys.stdin)['response'])"
	@echo "  Follow-up with city → Chicago:"
	@curl -sf -X POST $(BACKEND)/chat \
		-H "Content-Type: application/json" \
		-d "{\"message\": \"Chicago\", \"session_id\": \"$(SESSION)\"}" \
		| python3 -m json.tool

test-nonweather:
	@echo "\n── Non-weather rejection ─────────────────"
	@curl -sf -X POST $(BACKEND)/chat \
		-H "Content-Type: application/json" \
		-d '{"message": "Tell me a joke", "session_id": ""}' \
		| python3 -m json.tool

# ── Evaluation ───────────────────────────────────────────────────────────────

eval:
	@echo "Running LLM-as-a-judge evaluation (requires make run-backend in another terminal)…"
	@cd $(APP_DIR) && . .env && ../$(BIN)/python -m app.evaluation.run_eval

# ── Observability ────────────────────────────────────────────────────────────

observability:
	docker compose up -d
	@echo ""
	@echo "✓ Langfuse running at http://localhost:3000"
	@echo "  Sign up for a local account, then copy the API keys into $(APP_DIR)/.env"

observability-stop:
	docker compose down

# ── Clean ────────────────────────────────────────────────────────────────────

clean:
	rm -rf $(VENV)
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; true
	@echo "✓ Cleaned"
