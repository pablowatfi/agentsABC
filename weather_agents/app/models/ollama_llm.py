import os
from langchain_ollama import ChatOllama


def get_llm() -> ChatOllama:
    return ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "llama3"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )
