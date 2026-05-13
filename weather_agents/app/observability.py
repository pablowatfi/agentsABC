import os
from langfuse import Langfuse

_client: Langfuse | None = None


def get_langfuse() -> Langfuse | None:
    if not os.getenv("LANGFUSE_PUBLIC_KEY"):
        return None
    global _client
    if _client is None:
        _client = Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_HOST", "http://localhost:3000"),
        )
    return _client
