"""
Semantic cache for weather search results.

Cache key: embedding of the canonical query string  "{intent} {date_ref} in {location}"
           Always in English, always normalized — language-independent.

Lookup:    cosine similarity against all stored embeddings.
           If best score >= threshold → cache HIT, return stored results + answer.

Expiry:    each entry carries a TTL (seconds). Expired entries are evicted on lookup.

Storage:   in-memory list. Drop-in replaceable with Redis + vector search for production.
"""

import os
import time
import logging
from dataclasses import dataclass, field

import numpy as np
from langchain_ollama import OllamaEmbeddings

log = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    embedding:       list[float]
    canonical_query: str
    search_results:  str
    answer:          str
    location:        str
    date_ref:        str
    weather_intent:  str
    cached_at:       float = field(default_factory=time.time)
    ttl:             int   = 7200  # seconds

    @property
    def is_expired(self) -> bool:
        return time.time() - self.cached_at > self.ttl

    @property
    def age_minutes(self) -> float:
        return (time.time() - self.cached_at) / 60


class SemanticCache:
    def __init__(
        self,
        similarity_threshold: float = 0.92,
        default_ttl: int = 7200,
        embed_model: str = "nomic-embed-text",
    ):
        self._entries:   list[CacheEntry] = []
        self._threshold  = similarity_threshold
        self._default_ttl = default_ttl
        self._embeddings = OllamaEmbeddings(
            model=embed_model,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )

    # ── internals ─────────────────────────────────────────────────────────────

    def _embed(self, text: str) -> list[float]:
        return self._embeddings.embed_query(text)

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        va, vb = np.array(a), np.array(b)
        denom = np.linalg.norm(va) * np.linalg.norm(vb)
        return float(np.dot(va, vb) / denom) if denom else 0.0

    def _evict(self) -> None:
        before = len(self._entries)
        self._entries = [e for e in self._entries if not e.is_expired]
        if len(self._entries) < before:
            log.debug("Cache: evicted %d expired entries", before - len(self._entries))

    # ── public API ────────────────────────────────────────────────────────────

    def get(self, canonical_query: str) -> tuple["CacheEntry | None", float]:
        """Return (entry, score) on HIT, or (None, best_score) on MISS."""
        self._evict()
        if not self._entries:
            return None, 0.0

        try:
            q_emb = self._embed(canonical_query)
        except Exception as exc:
            log.warning("Cache: embedding failed (%s) — bypassing cache", exc)
            return None, 0.0

        best_score, best_entry = 0.0, None
        for entry in self._entries:
            score = self._cosine(q_emb, entry.embedding)
            if score > best_score:
                best_score, best_entry = score, entry

        if best_score >= self._threshold and best_entry:
            log.info(
                "Cache HIT  query='%s'  score=%.3f  age=%.1f min",
                canonical_query, best_score, best_entry.age_minutes,
            )
            return best_entry, best_score

        log.info("Cache MISS query='%s'  best=%.3f", canonical_query, best_score)
        return None, best_score

    def put(
        self,
        canonical_query: str,
        search_results:  str,
        answer:          str,
        location:        str,
        date_ref:        str,
        weather_intent:  str,
        ttl:             int | None = None,
    ) -> None:
        try:
            embedding = self._embed(canonical_query)
        except Exception as exc:
            log.warning("Cache: store skipped — embedding failed: %s", exc)
            return

        self._entries.append(CacheEntry(
            embedding=embedding,
            canonical_query=canonical_query,
            search_results=search_results,
            answer=answer,
            location=location,
            date_ref=date_ref,
            weather_intent=weather_intent,
            ttl=ttl or self._default_ttl,
        ))
        log.info(
            "Cache STORE query='%s'  ttl=%ds  size=%d",
            canonical_query, ttl or self._default_ttl, len(self._entries),
        )

    @property
    def size(self) -> int:
        self._evict()
        return len(self._entries)


# ── Singleton ─────────────────────────────────────────────────────────────────

_cache: SemanticCache | None = None


def get_cache() -> SemanticCache:
    global _cache
    if _cache is None:
        _cache = SemanticCache(
            similarity_threshold=float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.92")),
            default_ttl=int(os.getenv("CACHE_TTL_SECONDS", "7200")),
            embed_model=os.getenv("CACHE_EMBED_MODEL", "nomic-embed-text"),
        )
    return _cache
