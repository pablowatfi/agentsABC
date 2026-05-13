"""
LLM-as-a-judge evaluator for weather agent responses.

Two evaluation modes:

  evaluate(question, answer)
    Judges a FINAL answer on: relevance, grounding, completeness, clarity, overall.

  evaluate_clarification(question, clarification)
    Judges a CLARIFICATION REQUEST on: accuracy, naturalness, brevity, language_match, overall.
    Criteria are different because a clarification has no weather data to ground — its
    job is solely to ask for the one missing piece as naturally as possible.

Few-shot examples anchor the judge's scale for consistency across runs.
"""

import json
import re

from langchain_core.messages import HumanMessage

from app.evaluation.examples import REFERENCE_EXAMPLES
from app.models.ollama_llm import get_llm

# ── Few-shot blocks ───────────────────────────────────────────────────────────

_ANSWER_FEW_SHOT = "\n\n".join(
    f'Q: "{ex["question"]}"\n'
    f'IDEAL A: "{ex["ideal_answer"]}"\n'
    f'WHY GOOD: {ex["why_good"]}'
    for ex in REFERENCE_EXAMPLES
    if "ideal_answer" in ex
)

_SEARCH_FEW_SHOT = "\n\n".join(
    f'Question: "{ex["question"]}"'
    + (f'\nFollow-up city: "{ex["follow_up"]}"' if ex.get("follow_up") else "")
    + f'\nIDEAL query: "{ex["ideal_search_query"]}"'
    for ex in REFERENCE_EXAMPLES
    if ex.get("ideal_search_query")
)

_CLARIF_FEW_SHOT = "\n\n".join(
    f'Q: "{ex["question"]}"\n'
    f'IDEAL CLARIFICATION: "{ex["ideal_clarification"]}"\n'
    f'WHY GOOD: {ex["why_clarification_good"]}'
    for ex in REFERENCE_EXAMPLES
    if "ideal_clarification" in ex
)

# ── Final-answer judge prompt ─────────────────────────────────────────────────

_ANSWER_PROMPT = f"""You are an expert evaluator for a real-time weather chatbot.

## Reference examples of EXCELLENT final answers (score 9-10):

{_ANSWER_FEW_SHOT}

## Scoring rubric (each criterion 0–10):
- relevance:    Answer directly addresses what was asked. Off-topic = 0.
- grounding:    Uses specific data: temperatures, %, mm of rain, wind speed. Vague = 0.
- completeness: Covers all aspects the user asked about (location, time, metric).
- clarity:      Concise, no fluff, easy to understand in one read.
- overall:      Holistic quality considering all criteria.

## Important calibration:
- "I can only answer weather-related questions" for a non-weather question → 10 on ALL criteria.
- No specific data (e.g. "it might rain") → grounding ≤ 3.
- Answer in different language than question → deduct 2 from clarity.

## Now evaluate:
User question: {{question}}
Chatbot answer: {{answer}}

Respond with ONLY a JSON object — no text before or after:
{{{{
  "relevance": <int 0-10>,
  "grounding": <int 0-10>,
  "completeness": <int 0-10>,
  "clarity": <int 0-10>,
  "overall": <int 0-10>,
  "reasoning": "<one concise sentence>"
}}}}"""

# ── Clarification judge prompt ────────────────────────────────────────────────

_CLARIF_PROMPT = f"""You are an expert evaluator for a real-time weather chatbot.
Your task is to judge the quality of a CLARIFICATION REQUEST — not a final answer.
A clarification request is what the bot says when required info (e.g. city) is missing.

## Reference examples of EXCELLENT clarification requests (score 9-10):

{_CLARIF_FEW_SHOT}

## Scoring rubric (each criterion 0–10):
- accuracy:        Did it identify the correct missing piece (location/city)?
                   Asking for something already provided = 0.
- naturalness:     Does it sound like a natural human question, not robotic?
- brevity:         Is it as short as possible while still being clear? One sentence max = 10.
- language_match:  Is it in the same language as the user's question?
                   Wrong language = 0, correct language = 10.
- overall:         Holistic quality considering all criteria.

## Important calibration:
- "Which city are you asking about?" is a 10/10 clarification for a missing-city case.
- Asking for info the user already provided → accuracy = 0, overall ≤ 2.
- A long explanation instead of a short question → brevity ≤ 3.

## Now evaluate:
User question (that triggered the clarification): {{question}}
Chatbot clarification request: {{clarification}}

Respond with ONLY a JSON object — no text before or after:
{{{{
  "accuracy": <int 0-10>,
  "naturalness": <int 0-10>,
  "brevity": <int 0-10>,
  "language_match": <int 0-10>,
  "overall": <int 0-10>,
  "reasoning": "<one concise sentence>"
}}}}"""


# ── Public API ────────────────────────────────────────────────────────────────

def _invoke(prompt: str) -> dict | None:
    llm = get_llm()
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        match = re.search(r"\{.*\}", response.content.strip(), re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception:
        pass
    return None


def evaluate(question: str, answer: str) -> dict | None:
    """Judge a final weather answer. Returns scores dict or None on failure."""
    return _invoke(_ANSWER_PROMPT.format(question=question, answer=answer))


def evaluate_clarification(question: str, clarification: str) -> dict | None:
    """Judge a clarification request. Returns scores dict or None on failure."""
    return _invoke(_CLARIF_PROMPT.format(question=question, clarification=clarification))


# ── Search-query judge prompt ─────────────────────────────────────────────────

_SEARCH_PROMPT = f"""You are an expert evaluator for a weather chatbot's search pipeline.
Your task is to judge the quality of the Tavily search query the system constructed.
A bad query means bad search results, which means a bad final answer — this is the most
critical step in the pipeline.

## Reference examples of IDEAL search queries:

{_SEARCH_FEW_SHOT}

## Scoring rubric (each criterion 0–10):
- location_correct:  Does the query contain the right city/location from the question?
                     Wrong city = 0. Missing city = 0.
- date_correct:      Is the time reference right (today/tomorrow/this weekend)?
                     Missing when needed = 3. Wrong date = 0.
- intent_correct:    Does the query target the right weather aspect (rain/temperature/wind/snow)?
                     Generic "weather" when something specific was asked = 5.
- query_quality:     Is it concise and well-formed for a search engine?
                     Redundant words or garbled syntax = lower score.
- overall:           Holistic quality. If location or intent is wrong → overall ≤ 3.

## Important calibration:
- "rain tomorrow in Chicago" for "Will it rain tomorrow in Chicago?" → 10 on all criteria.
- "weather today in Paris" when asked about snow → intent_correct = 4, overall ≤ 6.
- "temperature in London" when asked for tomorrow's forecast → date_correct = 3, overall ≤ 7.
- Wrong city entirely → location_correct = 0, overall = 0.

## Now evaluate:
Original user question: {{question}}
Search query constructed by the system: {{search_query}}

Respond with ONLY a JSON object — no text before or after:
{{{{
  "location_correct": <int 0-10>,
  "date_correct": <int 0-10>,
  "intent_correct": <int 0-10>,
  "query_quality": <int 0-10>,
  "overall": <int 0-10>,
  "reasoning": "<one concise sentence>"
}}}}"""


def evaluate_search_query(question: str, search_query: str) -> dict | None:
    """Judge the Tavily search query. Returns scores dict or None on failure."""
    return _invoke(_SEARCH_PROMPT.format(question=question, search_query=search_query))


# ── Give-up judge prompt ──────────────────────────────────────────────────────

_GIVEUP_PROMPT = """You are an expert evaluator for a weather chatbot.
The chatbot tried to get the user's location multiple times but the user never provided
a clear answer. The chatbot eventually gave up. Evaluate how well it handled that.

## What a GOOD give-up message looks like (score 9-10):
- Explains clearly that it could not get the needed info
- Tells the user exactly what to do next (include a specific city)
- Is friendly and non-blaming — the user may genuinely not know
- Is short (1-2 sentences max)
- Matches the language the user was writing in

## What a BAD give-up message looks like (score 0-3):
- Keeps asking for the city again (did not give up)
- Is rude or blaming ("You never answered my question")
- Is so vague the user doesn't know what went wrong
- Is in the wrong language

## Scoring rubric (each criterion 0–10):
- clarity:         Does it clearly explain what went wrong and what to do next?
- tone:            Is it friendly, patient, non-blaming?
- language_match:  Is it in the same language as the user's original question?
- timing:          Did it give up at the right moment — not too early (after 1 ask),
                   not too late (after 10 asks)? 5 attempts is ideal = 10.
- overall:         Holistic quality.

## Context:
Original user question: {question}
Number of clarification attempts made before giving up: {attempts}
Final give-up message from the chatbot: {giveup_message}

Respond with ONLY a JSON object — no text before or after:
{{
  "clarity": <int 0-10>,
  "tone": <int 0-10>,
  "language_match": <int 0-10>,
  "timing": <int 0-10>,
  "overall": <int 0-10>,
  "reasoning": "<one concise sentence>"
}}"""


def evaluate_giveup(question: str, giveup_message: str, attempts: int) -> dict | None:
    """Judge a give-up message after too many failed clarifications."""
    return _invoke(_GIVEUP_PROMPT.format(
        question=question,
        giveup_message=giveup_message,
        attempts=attempts,
    ))
