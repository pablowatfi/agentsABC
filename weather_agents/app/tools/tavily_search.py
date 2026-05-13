import os
from tavily import TavilyClient


def search_weather(query: str) -> str:
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    result = client.search(query, search_depth="basic", max_results=3)
    snippets = [r["content"] for r in result.get("results", [])]
    return "\n\n".join(snippets) if snippets else "No weather data found."
