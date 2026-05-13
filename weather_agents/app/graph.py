from typing import Optional
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from app.agents.classifier import classify_node
from app.agents.weather_agent import resolve_node, search_and_answer_node


class WeatherState(TypedDict):
    question: str
    location: Optional[str]
    date_ref: Optional[str]
    weather_intent: Optional[str]
    is_weather: bool
    needs_clarification: bool
    clarification_question: str
    missing_field: Optional[str]
    answer: str
    search_query: str
    search_results: str
    cache_hit: bool
    cache_score: float


def _after_classify(state: WeatherState) -> str:
    return "resolve" if state["is_weather"] else END


def _after_resolve(state: WeatherState) -> str:
    return END if state["needs_clarification"] else "search_answer"


def build_graph():
    g = StateGraph(WeatherState)
    g.add_node("classify", classify_node)
    g.add_node("resolve", resolve_node)
    g.add_node("search_answer", search_and_answer_node)

    g.set_entry_point("classify")
    g.add_conditional_edges("classify", _after_classify)
    g.add_conditional_edges("resolve", _after_resolve)
    g.add_edge("search_answer", END)

    return g.compile()


graph = build_graph()
