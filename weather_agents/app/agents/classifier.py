"""
Combined classify + extract node.

One LLM call replaces the old separate classify and _extract_params calls,
cutting the happy-path from 3 LLM calls to 2.
"""

import json
import re
from langchain_core.messages import HumanMessage
from app.models.ollama_llm import get_llm

_PROMPT = """Analyze the user's message and return ONLY a JSON object with these exact keys:

- "is_weather": true if the question is about weather (current, forecast, rain, temperature, wind, snow, humidity, etc.), false otherwise
- "language": ISO 639-1 code of the language the user wrote in (e.g. "en", "es", "fr", "de", "pt")
- "rejection_message": if is_weather is false, a short friendly message in the user's language saying you only answer weather questions; null if is_weather is true
- "location": the city or place name mentioned, or null if none
- "location_type": "city" if a specific city/town, "country" if an entire country, "region" if a vague area (e.g. "the north", "the coast"), null if no location
- "date_ref": the time reference (e.g. "today", "tomorrow", "Friday", "this weekend") or null
- "weather_intent": what the user wants to know — e.g. "rain", "temperature", "wind", "snow", "humidity", "general weather" — or null
- "country": the country if explicitly stated OR unambiguously implied (e.g. "New York" → "United States"), null otherwise
- "location_ambiguous": true ONLY if the place name could realistically refer to very different places in different countries (e.g. "Monterrey", "San Jose"); false for well-known places
- "suggested_country": if location_ambiguous is true, the most likely country; null otherwise

User message: {question}

JSON:"""


def classify_node(state: dict) -> dict:
    llm = get_llm()
    response = llm.invoke([HumanMessage(content=_PROMPT.format(question=state["question"]))])
    text = response.content.strip()

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            p = json.loads(match.group())
        except json.JSONDecodeError:
            p = {}
    else:
        p = {}

    is_weather = bool(p.get("is_weather", False))
    language = p.get("language") or "en"

    if not is_weather:
        answer = p.get("rejection_message") or "I can only answer weather-related questions."
        return {
            **state,
            "is_weather": False,
            "language": language,
            "answer": answer,
            "location_type": None,
            "location_ambiguous": False,
            "suggested_country": None,
        }

    # Carry extracted params forward so resolve_node doesn't need another LLM call
    return {
        **state,
        "is_weather": True,
        "language": language,
        "location":           state.get("location") or p.get("location"),
        "location_type":      p.get("location_type"),
        "date_ref":           state.get("date_ref") or p.get("date_ref"),
        "weather_intent":     state.get("weather_intent") or p.get("weather_intent"),
        "country":            state.get("country") or p.get("country"),
        "location_ambiguous": bool(p.get("location_ambiguous", False)),
        "suggested_country":  p.get("suggested_country"),
    }
