import json
import os
import re
import logging
from langchain_core.messages import HumanMessage, SystemMessage
from app.models.ollama_llm import get_llm
from app.tools.tavily_search import search_weather
from app.cache import get_cache

log = logging.getLogger(__name__)

# Two-tier cache thresholds:
#   >= ANSWER_THRESHOLD  → return cached answer verbatim (virtually identical question)
#   >= SEARCH_THRESHOLD  → reuse cached search results, regenerate answer for new phrasing
_ANSWER_THRESHOLD = float(os.getenv("CACHE_ANSWER_THRESHOLD", "0.97"))


_EXTRACT_PROMPT = """Extract weather query parameters from the user's question.
Return ONLY a JSON object with these keys:
- "location": city/place name or null if not mentioned
- "date_ref": time reference (e.g. "tomorrow", "today", "Friday") or null if not mentioned
- "weather_intent": what they want to know (e.g. "rain", "temperature", "wind") or null

Question: {question}

JSON:"""


def _extract_params(question: str) -> dict:
    llm = get_llm()
    response = llm.invoke([HumanMessage(content=_EXTRACT_PROMPT.format(question=question))])
    text = response.content.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {"location": None, "date_ref": None, "weather_intent": None}


def resolve_node(state: dict) -> dict:
    params = _extract_params(state["question"])

    location = state.get("location") or params.get("location")
    date_ref = state.get("date_ref") or params.get("date_ref") or "today"
    weather_intent = state.get("weather_intent") or params.get("weather_intent") or "weather"

    if not location:
        return {
            **state,
            "needs_clarification": True,
            "clarification_question": "Which city are you asking about?",
            "missing_field": "location",
            "location": None,
            "date_ref": date_ref,
            "weather_intent": weather_intent,
        }

    return {
        **state,
        "needs_clarification": False,
        "location": location,
        "date_ref": date_ref,
        "weather_intent": weather_intent,
    }


def _llm_answer(question: str, search_results: str) -> str:
    llm = get_llm()
    response = llm.invoke([
        SystemMessage(content=(
            "You are a helpful weather assistant. "
            "Based on the search results below, answer the user's question concisely. "
            "If the data is unclear, give your best estimate from the results."
        )),
        HumanMessage(content=(
            f"User question: {question}\n\n"
            f"Search results:\n{search_results}"
        )),
    ])
    return response.content.strip()


def search_and_answer_node(state: dict) -> dict:
    query = f"{state['weather_intent']} {state['date_ref']} in {state['location']}"
    cache = get_cache()

    entry, score = cache.get(query)
    if entry:
        if score >= _ANSWER_THRESHOLD:
            # Virtually identical question — return the exact same answer
            log.info("Cache HIT (exact) score=%.3f — returning cached answer", score)
            return {
                **state,
                "search_query": query,
                "search_results": entry.search_results,
                "answer": entry.answer,
                "cache_hit": True,
                "cache_score": score,
            }
        else:
            # Similar topic — reuse Tavily data but regenerate answer for the new phrasing
            log.info("Cache HIT (reuse data) score=%.3f — regenerating answer", score)
            answer = _llm_answer(state["question"], entry.search_results)
            return {
                **state,
                "search_query": query,
                "search_results": entry.search_results,
                "answer": answer,
                "cache_hit": True,
                "cache_score": score,
            }

    raw_data = search_weather(query)
    answer = _llm_answer(state["question"], raw_data)

    cache.put(
        canonical_query=query,
        search_results=raw_data,
        answer=answer,
        location=state["location"],
        date_ref=state["date_ref"],
        weather_intent=state["weather_intent"],
    )

    return {
        **state,
        "search_query": query,
        "search_results": raw_data,
        "answer": answer,
        "cache_hit": False,
        "cache_score": score,
    }
