import os
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


_COUNTRY_PROMPT = """Extract the country name from the user's text. Return ONLY the country name.
If no country is mentioned, return null.

Text: "{text}"
Country:"""

_CLARIFICATION_PROMPT = """A user asked a weather question but you need more information about the location.
Write a SHORT, FRIENDLY clarification in the SAME LANGUAGE the user wrote in.

User's question: "{question}"
Detected location: "{location}"
Problem: {problem}

Rules:
- Write in the SAME LANGUAGE as the user's question (detected: {language})
- Maximum 2 sentences
- Be specific: reference what the user said, explain what you need
- If the location is a country or vague region: explain you need a specific city, not a country
- If the location is ambiguous across countries: ask which country
- If no location at all: ask for a city

Return ONLY the clarification question, nothing else."""



def _extract_country(text: str) -> str | None:
    """Parse a clean country name from a free-form user response ('Yes, Mexico' → 'Mexico')."""
    llm = get_llm()
    try:
        response = llm.invoke([HumanMessage(content=_COUNTRY_PROMPT.format(text=text))])
        result = response.content.strip().strip('"\'')
        return result if result.lower() not in ("null", "none", "") else None
    except Exception:
        return None


def _generate_clarification(question: str, location: str | None, problem: str, language: str) -> str:
    """Generate a contextual clarification question in the user's language."""
    llm = get_llm()
    try:
        response = llm.invoke([HumanMessage(content=_CLARIFICATION_PROMPT.format(
            question=question,
            location=location or "(none mentioned)",
            problem=problem,
            language=language,
        ))])
        return response.content.strip().strip('"\'')
    except Exception:
        # Fallback to English if generation fails
        return "Which city are you asking about?"


def resolve_node(state: dict) -> dict:
    # classify_node already extracted all params — no second LLM call needed.
    # location_was_clarified is True when the user provided a location in a prior clarification
    # turn (stored in session and pre-set in state before classify ran). In that case we skip
    # the country/region and ambiguity checks: trust what the user explicitly said.
    location_was_clarified = bool(state.get("_location_clarified"))

    location = state.get("location")
    date_ref = state.get("date_ref") or "today"
    weather_intent = state.get("weather_intent") or "weather"
    language = state.get("language") or "en"
    location_type = state.get("location_type")

    # No location at all
    if not location:
        q = _generate_clarification(
            state["question"], None,
            problem="The user did not mention any city or location.",
            language=language,
        )
        return {
            **state,
            "needs_clarification": True,
            "clarification_question": q,
            "missing_field": "location",
            "location": None,
            "date_ref": date_ref,
            "weather_intent": weather_intent,
            "country": None,
        }

    # Location is a whole country or vague region — need a specific city.
    # Skip this check if the user just provided this location as a direct clarification answer:
    # "salta" typed in response to "which city?" should be trusted as a city.
    if not location_was_clarified and location_type in ("country", "region"):
        q = _generate_clarification(
            state["question"], location,
            problem=(
                f'"{location}" is a {"country" if location_type == "country" else "broad region"}, '
                f"not a specific city. Weather data requires a city."
            ),
            language=language,
        )
        log.info("Location is %s: %r — asking for city", location_type, location)
        return {
            **state,
            "needs_clarification": True,
            "clarification_question": q,
            "missing_field": "location",
            "location": None,
            "date_ref": date_ref,
            "weather_intent": weather_intent,
            "country": None,
        }

    # Resolve country: clean up user's free-form clarification, or use what classify extracted
    raw_country = state.get("country")
    if raw_country:
        country = _extract_country(raw_country)
        log.info("Country clarification received: raw=%r → parsed=%r", raw_country, country)
    else:
        country = None  # classify_node already set state["country"] if unambiguously implied

    # Location is ambiguous across countries — ask which one before searching.
    # Skip if the user already clarified (they know which place they mean).
    if not location_was_clarified and not country and state.get("location_ambiguous"):
        suggested = state.get("suggested_country")
        problem = (
            f'"{location}" could refer to places in different countries'
            + (f", most likely {suggested}" if suggested else "")
            + ". Need to confirm the country to look up the right location."
        )
        q = _generate_clarification(
            state["question"], location, problem=problem, language=language,
        )
        log.info("Location ambiguous: location=%r suggested=%r", location, suggested)
        return {
            **state,
            "needs_clarification": True,
            "clarification_question": q,
            "missing_field": "country",
            "location": location,
            "date_ref": date_ref,
            "weather_intent": weather_intent,
            "country": None,
        }

    full_location = f"{location}, {country}" if country else location

    return {
        **state,
        "needs_clarification": False,
        "location": full_location,
        "date_ref": date_ref,
        "weather_intent": weather_intent,
        "country": country,
        "language": language,
    }


def _llm_answer(question: str, search_results: str, language: str = "en") -> str:
    llm = get_llm()
    response = llm.invoke([
        SystemMessage(content=(
            "You are a helpful weather assistant. "
            "Based on the search results below, answer the user's question concisely. "
            "If the data is unclear, give your best estimate from the results. "
            f"IMPORTANT: always reply in the same language as the user's question (detected: {language})."
        )),
        HumanMessage(content=(
            f"User question: {question}\n\n"
            f"Search results:\n{search_results}"
        )),
    ])
    return response.content.strip()


def search_and_answer_node(state: dict) -> dict:
    query = f"{state['weather_intent']} {state['date_ref']} in {state['location']}"
    language = state.get("language", "en")
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
            answer = _llm_answer(state["question"], entry.search_results, language)
            return {
                **state,
                "search_query": query,
                "search_results": entry.search_results,
                "answer": answer,
                "cache_hit": True,
                "cache_score": score,
            }

    raw_data = search_weather(query)
    answer = _llm_answer(state["question"], raw_data, language)

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
