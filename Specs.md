Here’s a **very simple spec** you can hand to a code assistant.

---

# Weather Agent App — Simple Spec

## Goal

Build a very simple app with:

* a **UI** where a user asks weather-related questions in natural language
* **2 agents** using **LangGraph + LangChain**
* **Ollama** as the LLM
* **Tavily** for web search

Example inputs:

* “What will be the temperature tomorrow?”
* “Is it going to rain tomorrow in San Francisco?”

---

# Functional Requirements

## 1. Frontend

Create a minimal chat UI:

* input text box
* send button
* conversation history window

User flow:

1. User submits a question
2. Backend runs agents
3. UI displays:

   * final answer, or
   * clarification question if needed

---

## 2. Agent Architecture

### Agent 1: Intent Classifier

Responsibilities:

* determine if question is **weather-related**

Examples:

* ✅ “Will it rain tomorrow?”
* ✅ “Temperature in London?”
* ❌ “Who won the World Cup?”

If NOT weather:
Return:

> “I can only answer weather-related questions.”

---

### Agent 2: Weather Resolver

Responsibilities:

1. extract required weather parameters:

   * location
   * date/time reference
   * weather intent (rain, temperature, wind, etc.)

2. detect missing info

Example:
Input:

> “Will it rain tomorrow?”

Missing:

* city/location

Ask user:

> “Which city are you asking about?”

3. once complete:

   * call Tavily search
   * retrieve weather data
   * summarize answer

Example:

> “Yes, there is a 70% chance of rain tomorrow in San Francisco.”

---

# LangGraph Flow

```text
START
  |
Classifier Agent
  |
  |-- not weather --> return rejection
  |
  |-- weather -->
         |
   Missing Info?
      | yes --> ask user --> wait
      | no
         |
    Tavily Search
         |
    LLM Summarization
         |
        END
```

---

# Tech Stack

## Backend

* Python
* [LangGraph](https://www.langchain.com/langgraph?utm_source=chatgpt.com)
* [LangChain](https://www.langchain.com/?utm_source=chatgpt.com)
* [Ollama](https://ollama.com/?utm_source=chatgpt.com)
* [Tavily](https://tavily.com/?utm_source=chatgpt.com)
* FastAPI

Suggested Ollama model:

* `llama3`
  or
* `mistral`

---

## Frontend

Simple option:

* React

or simpler:

* Streamlit

Preferred for MVP:

* [Streamlit](https://streamlit.io/?utm_source=chatgpt.com)

---

# Suggested File Structure

```text
weather_agents/
│── app/
│   ├── main.py              # FastAPI entrypoint
│   ├── graph.py             # LangGraph flow
│   ├── agents/
│   │    ├── classifier.py
│   │    └── weather_agent.py
│   ├── tools/
│   │    └── tavily_search.py
│   └── models/
│        └── ollama_llm.py
│
│── ui/
│   └── streamlit_app.py
│
│── requirements.txt
│── README.md
```

---

# Environment Variables

```env
TAVILY_API_KEY=xxx
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3
```

---

# Success Criteria

Must support:

### Case 1

Input:

> “What will be temperature tomorrow in New York City?”

Output:

> “Tomorrow’s high is 24°C and low is 16°C.”

---

### Case 2

Input:

> “Will it rain tomorrow?”

Output:

> “Which city are you asking about?”

User:

> “Chicago”

Output:

> “Yes, 40% chance of rain.”

---

### Case 3

Input:

> “Tell me a joke”

Output:

> “I can only answer weather-related questions.”

---

That’s enough for a code assistant to scaffold the whole project.
