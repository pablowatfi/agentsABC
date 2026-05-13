from langchain_core.messages import HumanMessage, SystemMessage
from app.models.ollama_llm import get_llm


def classify_node(state: dict) -> dict:
    llm = get_llm()
    response = llm.invoke([
        SystemMessage(content=(
            "You are a classifier. Determine if the user's question is weather-related. "
            "Reply with only 'yes' or 'no'. No other words."
        )),
        HumanMessage(content=state["question"]),
    ])
    is_weather = "yes" in response.content.strip().lower()
    if not is_weather:
        state["answer"] = "I can only answer weather-related questions."
    return {**state, "is_weather": is_weather}
