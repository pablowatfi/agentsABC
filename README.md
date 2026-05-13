# Weather Agent App

A simple multi-agent weather chatbot using LangGraph, Ollama, Tavily, FastAPI, and Streamlit.

---

## Prerequisites

| Tool | Install |
|------|---------|
| Python 3.10+ | [python.org](https://python.org) |
| Ollama | [ollama.com](https://ollama.com) |
| Tavily API key | [app.tavily.com](https://app.tavily.com) (free tier available) |

---

## Setup

### 1. Clone and enter the project

```bash
cd weather_agents
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in your values:

```env
TAVILY_API_KEY=tvly-xxxxxxxxxxxxxxxx
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3
```

Get a free Tavily key at [app.tavily.com](https://app.tavily.com).

### 5. Pull the Ollama model

```bash
ollama pull llama3
```

Verify Ollama is running:

```bash
ollama list   # should show llama3
```

---

## Running the App

Open **two terminals**, both with the virtualenv activated and inside `weather_agents/`.

**Terminal 1 — Backend:**

```bash
uvicorn app.main:app --reload
```

You should see:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
```

**Terminal 2 — Frontend:**

```bash
streamlit run ui/streamlit_app.py
```

Streamlit will open `http://localhost:8501` in your browser automatically.

---

## Testing

### Manual — via the UI

Open `http://localhost:8501` and try these cases:

| Input | Expected output |
|-------|----------------|
| `What will be the temperature tomorrow in New York?` | Temperature forecast for NYC |
| `Will it rain tomorrow?` | Asks: *"Which city are you asking about?"* |
| → `Chicago` | Rain forecast for Chicago |
| `Tell me a joke` | *"I can only answer weather-related questions."* |

### Manual — via the API directly

```bash
# Health check
curl http://localhost:8000/health

# Full question (no clarification needed)
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Will it snow tomorrow in Denver?", "session_id": ""}' | python -m json.tool

# Multi-turn: missing location
SESSION=$(curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Will it rain tomorrow?", "session_id": ""}' | python -c "import sys,json; d=json.load(sys.stdin); print(d['session_id'])")

echo "Session ID: $SESSION"

# Follow up with the city
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"Chicago\", \"session_id\": \"$SESSION\"}" | python -m json.tool

# Non-weather question
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Who won the World Cup?", "session_id": ""}' | python -m json.tool
```

---

## How It Works

```
User message
     │
     ▼
[Agent 1: Classifier]
     │
     ├── not weather ──► "I can only answer weather-related questions."
     │
     └── weather
          │
          ▼
     [Agent 2: Resolver]
          │
          ├── missing location ──► "Which city are you asking about?"
          │                              (session holds state; waits for reply)
          └── complete
               │
               ▼
          [Tavily Search]
               │
               ▼
          [LLM Summarization]
               │
               ▼
          Final weather answer
```

- **Classifier** (Ollama): decides if the question is weather-related with a yes/no prompt.
- **Resolver** (Ollama): extracts `location`, `date_ref`, and `weather_intent` from the question. If `location` is missing, returns a clarification question. The session ID keeps the context alive for the follow-up.
- **Tavily Search**: performs a real-time web search for the weather query.
- **Summarizer** (Ollama): produces a concise natural-language answer from the search results.

---

## Troubleshooting

**Ollama not responding**
```bash
ollama serve   # start the server if it's not running
```

**`ModuleNotFoundError` for app.***
Run uvicorn from inside the `weather_agents/` directory, not the repo root.

**Tavily returns no results**
Check that `TAVILY_API_KEY` is set correctly in `.env` and that you have remaining API credits.

**LLM returns bad JSON (resolver)**
Try switching to `mistral` which follows JSON instructions more reliably:
```env
OLLAMA_MODEL=mistral
```
Then `ollama pull mistral`.
