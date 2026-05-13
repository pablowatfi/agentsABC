import os
import uuid
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

from app.graph import graph, WeatherState
from app.observability import get_langfuse
from app.evaluation.runner import run_async as evaluate_async

app = FastAPI(title="Weather Agent")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

sessions: dict[str, dict] = {}

MAX_CLARIFICATIONS = 5
GIVEUP_MESSAGE = (
    "I wasn't able to determine the location after several attempts. "
    "Please start a new question and include a specific city."
)


class ChatRequest(BaseModel):
    message: str
    session_id: str = ""


class ChatResponse(BaseModel):
    response: str
    session_id: str
    search_query: str | None = None
    cache_hit: bool = False
    cache_score: float = 0.0


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    session = sessions.get(session_id, {})

    lf = get_langfuse()

    if session.get("awaiting_field"):
        field = session.pop("awaiting_field")
        session[field] = req.message
        trace = lf.trace(id=session["langfuse_trace_id"]) if lf and session.get("langfuse_trace_id") else None
        if trace:
            trace.span(name="user-clarification", input={"field": field, "value": req.message})
    else:
        session = {"original_question": req.message}
        if lf:
            trace = lf.trace(
                name="weather-query",
                session_id=session_id,
                input={"question": req.message},
            )
            session["langfuse_trace_id"] = trace.id
        else:
            trace = None

    question = session.get("original_question", req.message)

    initial_state: WeatherState = {
        "question": question,
        "location": session.get("location"),
        "date_ref": session.get("date_ref"),
        "weather_intent": session.get("weather_intent"),
        "is_weather": False,
        "needs_clarification": False,
        "clarification_question": "",
        "missing_field": None,
        "answer": "",
        "search_query": "",
        "search_results": "",
        "cache_hit": False,
        "cache_score": 0.0,
    }

    result = graph.invoke(initial_state)
    search_query   = result.get("search_query") or None
    search_results = result.get("search_results") or None

    if result["needs_clarification"]:
        clarification_count = session.get("clarification_count", 0) + 1

        if clarification_count >= MAX_CLARIFICATIONS:
            # Give up — too many failed attempts to extract location
            sessions.pop(session_id, None)
            response_text = GIVEUP_MESSAGE
            if trace:
                trace.update(
                    output={"response": response_text},
                    metadata={"given_up": True, "clarification_count": clarification_count},
                )
        else:
            response_text = result["clarification_question"]
            session["clarification_count"] = clarification_count
            session["awaiting_field"] = result["missing_field"]
            session["location"] = result.get("location")
            session["date_ref"] = result.get("date_ref")
            session["weather_intent"] = result.get("weather_intent")
            sessions[session_id] = session
            if trace:
                trace.span(
                    name="clarification-requested",
                    input={"question": question},
                    output={
                        "missing_field": result["missing_field"],
                        "question_asked": response_text,
                        "attempt": clarification_count,
                    },
                )
    else:
        response_text = result["answer"]
        sessions.pop(session_id, None)
        cache_hit   = result.get("cache_hit", False)
        cache_score = result.get("cache_score", 0.0)
        if trace:
            if search_query and search_results:
                span_name = "cache-hit" if cache_hit else "tavily-search"
                trace.span(
                    name=span_name,
                    input={"query": search_query},
                    output={"results": search_results},
                    metadata={"cache_hit": cache_hit, "cache_score": round(cache_score, 4)},
                )
            trace.update(
                output={"response": response_text},
                metadata={
                    "is_weather": result["is_weather"],
                    "location": result.get("location"),
                    "date_ref": result.get("date_ref"),
                    "weather_intent": result.get("weather_intent"),
                    "search_query": search_query,
                    "cache_hit": cache_hit,
                    "cache_score": round(cache_score, 4),
                },
            )
        evaluate_async(
            question=question,
            answer=response_text,
            search_query=search_query,
            trace_id=session.get("langfuse_trace_id") if lf else None,
            lf=lf,
        )

    if lf:
        lf.flush()

    return ChatResponse(
        response=response_text,
        session_id=session_id,
        search_query=search_query,
        cache_hit=result.get("cache_hit", False),
        cache_score=round(result.get("cache_score", 0.0), 4),
    )


@app.get("/health")
def health():
    return {"status": "ok"}
