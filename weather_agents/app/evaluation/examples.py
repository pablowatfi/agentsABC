"""
Reference Q&A pairs used to calibrate the LLM judge.

Single-turn examples: just `question` + `ideal_answer`.
Multi-turn examples:  also have `follow_up`, `ideal_clarification`, and
                      `why_clarification_good` so both conversation turns
                      can be evaluated independently.
"""

REFERENCE_EXAMPLES = [
    # ── Single-turn ───────────────────────────────────────────────────────────
    {
        "id": "temp_full_en",
        "question": "What will be the temperature tomorrow in New York City?",
        "ideal_search_query": "temperature tomorrow in New York City",
        "ideal_answer": (
            "Tomorrow in New York City, expect a high of around 24°C (75°F) "
            "and a low of 16°C (61°F), with mostly sunny skies."
        ),
        "why_good": (
            "Includes specific numeric values, confirms the location, covers both "
            "high and low temperatures, and mentions conditions."
        ),
    },
    {
        "id": "rain_full_en",
        "question": "Is it going to rain tomorrow in San Francisco?",
        "ideal_search_query": "rain tomorrow in San Francisco",
        "ideal_answer": (
            "Yes, there is a 70% chance of rain in San Francisco tomorrow, "
            "with light showers expected in the afternoon."
        ),
        "why_good": "Gives a probability, confirms the city, specifies timing.",
    },
    {
        "id": "wind_en",
        "question": "How strong are the winds in Miami today?",
        "ideal_search_query": "wind today in Miami",
        "ideal_answer": (
            "Winds in Miami today are around 15 mph (24 km/h) from the southeast, "
            "gusting up to 25 mph in the afternoon."
        ),
        "why_good": "Includes speed, units, direction, and peak gust.",
    },
    {
        "id": "temp_es",
        "question": "¿Qué temperatura va a hacer en Buenos Aires?",
        "ideal_search_query": "temperature today in Buenos Aires",
        "ideal_answer": (
            "Hoy en Buenos Aires se espera una máxima de 22°C y una mínima de 14°C, "
            "con cielos parcialmente nublados."
        ),
        "why_good": "Specific numeric values, confirms city, mentions conditions.",
    },
    {
        "id": "non_weather_en",
        "question": "Tell me a joke",
        "ideal_search_query": None,
        "ideal_answer": "I can only answer weather-related questions.",
        "why_good": "Correctly rejects an off-topic request with a clear, short message.",
    },
    {
        "id": "non_weather_es",
        "question": "¿Quién ganó el Mundial?",
        "ideal_search_query": None,
        "ideal_answer": "Solo puedo responder preguntas sobre el clima.",
        "why_good": "Rejects the off-topic question cleanly in the user's language.",
    },

    # ── Multi-turn (clarification required) ───────────────────────────────────
    {
        "id": "multiturn_rain_en",
        "question": "Will it rain tomorrow?",
        "follow_up": "Chicago",
        "ideal_clarification": "Which city are you asking about?",
        "why_clarification_good": (
            "Identifies exactly the missing piece (location) in one short, "
            "natural question — no extra words."
        ),
        "ideal_search_query": "rain tomorrow in Chicago",
        "ideal_answer": "Yes, there is a 40% chance of rain in Chicago tomorrow.",
        "why_good": "Correct probability, confirms city, directly answers the question.",
    },
    {
        "id": "multiturn_temp_en",
        "question": "What's the temperature today?",
        "follow_up": "London",
        "ideal_clarification": "Which city are you asking about?",
        "why_clarification_good": "Asks for the only missing piece without over-explaining.",
        "ideal_search_query": "temperature today in London",
        "ideal_answer": (
            "In London today, the temperature is around 18°C (64°F), "
            "with cloudy skies and a slight breeze."
        ),
        "why_good": "Confirms city, gives current temperature, adds context.",
    },
    {
        "id": "multiturn_lluvia_es",
        "question": "¿Va a llover mañana?",
        "follow_up": "Rosario",
        "ideal_clarification": "¿En qué ciudad?",
        "why_clarification_good": "Minimal, natural clarification in the same language as the question.",
        "ideal_search_query": "rain tomorrow in Rosario",
        "ideal_answer": (
            "Mañana en Rosario hay un 55% de probabilidad de lluvia, "
            "con lluvias moderadas por la tarde."
        ),
        "why_good": "Probability, city confirmation, timing — complete and in Spanish.",
    },
    {
        "id": "multiturn_viento_es",
        "question": "¿Cómo está el viento?",
        "follow_up": "Mendoza",
        "ideal_clarification": "¿En qué ciudad querés consultar el viento?",
        "why_clarification_good": "Asks for the city in conversational Spanish, referencing the topic (wind).",
        "ideal_search_query": "wind today in Mendoza",
        "ideal_answer": (
            "En Mendoza hoy el viento sopla del oeste a unos 20 km/h, "
            "con ráfagas de hasta 35 km/h por la tarde."
        ),
        "why_good": "Speed, direction, gusts — all the wind details in Spanish.",
    },
    {
        "id": "multiturn_snow_en",
        "question": "Is it going to snow this weekend?",
        "follow_up": "Denver",
        "ideal_clarification": "Which city are you asking about?",
        "why_clarification_good": "Short and precise — asks only what is needed.",
        "ideal_search_query": "snow this weekend in Denver",
        "ideal_answer": (
            "Yes, Denver is expecting 4–8 inches of snow this weekend, "
            "with the heaviest snowfall on Saturday night."
        ),
        "why_good": "Includes amount range, confirms city, specifies timing within the weekend.",
    },

    # ── Never-clear (give-up required) ───────────────────────────────────────
    {
        "id": "never_clear_en",
        "question": "What's the weather like?",
        "follow_ups": [
            "I don't know",
            "somewhere nice",
            "not sure really",
            "you decide",
            "doesn't matter",
        ],
        "expected_behavior": "give_up",
        "ideal_giveup_message": (
            "I wasn't able to determine the location after several attempts. "
            "Please start a new question and include a specific city."
        ),
        "why_good": (
            "After 5 failed attempts to extract a location, gives up with a clear, "
            "actionable message — does not loop forever."
        ),
    },
    {
        "id": "never_clear_es",
        "question": "¿Cómo va a estar el tiempo?",
        "follow_ups": [
            "no sé",
            "en algún lugar",
            "donde sea",
            "cualquier ciudad",
            "me da igual",
        ],
        "expected_behavior": "give_up",
        "ideal_giveup_message": (
            "No pude determinar la ubicación luego de varios intentos. "
            "Por favor iniciá una nueva pregunta e incluí una ciudad específica."
        ),
        "why_good": (
            "Gives up gracefully in Spanish after 5 vague follow-ups, "
            "with a clear instruction on what to do next."
        ),
    },
]
