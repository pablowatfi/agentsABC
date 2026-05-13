"""
Runs the LLM judge asynchronously (daemon thread) so it never blocks the
user-facing response. Scores are posted to Langfuse once available.
"""

import logging
import threading

from app.evaluation.judge import evaluate, evaluate_search_query

log = logging.getLogger(__name__)

_ANSWER_CRITERIA = ("relevance", "grounding", "completeness", "clarity", "overall")
_SEARCH_CRITERIA = ("location_correct", "date_correct", "intent_correct", "query_quality", "overall")


def _log_scores(lf, trace_id: str, prefix: str, scores: dict) -> None:
    reasoning = scores.get("reasoning", "")
    for criterion, raw in scores.items():
        if criterion == "reasoning" or raw is None:
            continue
        try:
            lf.score(
                trace_id=trace_id,
                name=f"{prefix}.{criterion}",
                value=float(raw),
                comment=reasoning if criterion == "overall" else None,
            )
        except Exception as exc:
            log.warning("Failed to log score %s.%s: %s", prefix, criterion, exc)


def run_async(
    question: str,
    answer: str,
    search_query: str | None,
    trace_id: str | None,
    lf,
) -> None:
    """Fire-and-forget: evaluate answer + search query in background, log to Langfuse."""
    if not trace_id or not lf:
        return

    def _run() -> None:
        # Judge the final answer
        a_scores = evaluate(question, answer)
        if a_scores:
            _log_scores(lf, trace_id, "judge.answer", a_scores)
            log.info("Answer eval done — trace %s overall=%s", trace_id, a_scores.get("overall"))

        # Judge the search query (only if one was made)
        if search_query:
            s_scores = evaluate_search_query(question, search_query)
            if s_scores:
                _log_scores(lf, trace_id, "judge.search", s_scores)
                log.info("Search eval done — trace %s overall=%s", trace_id, s_scores.get("overall"))

        lf.flush()

    thread = threading.Thread(target=_run, daemon=True, name=f"judge-{trace_id[:8]}")
    thread.start()
