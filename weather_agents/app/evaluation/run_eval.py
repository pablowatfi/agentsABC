"""
Batch evaluation runner.

For each reference example:
  - Single-turn:  sends the question, judges the final answer.
  - Multi-turn:   sends the question, judges the CLARIFICATION REQUEST,
                  then sends the follow-up city, judges the FINAL ANSWER.

Output:
  1. Brief summary table  (one row per example, search / clarify / answer overall)
  2. Detailed tables per judge with per-criterion scores and reasoning
  3. All scores saved to Langfuse as a tagged batch-eval run

Requires the backend to be running:  make run-backend
"""

import sys
import uuid
import datetime
import requests

from app.evaluation.examples import REFERENCE_EXAMPLES
from app.evaluation.judge import evaluate, evaluate_clarification, evaluate_search_query, evaluate_giveup
from app.observability import get_langfuse

BACKEND = "http://localhost:8000"
TIMEOUT = 180

ANSWER_CRITERIA  = ("relevance", "grounding", "completeness", "clarity", "overall")
CLARIF_CRITERIA  = ("accuracy", "naturalness", "brevity", "language_match", "overall")
SEARCH_CRITERIA  = ("location_correct", "date_correct", "intent_correct", "query_quality", "overall")
GIVEUP_CRITERIA  = ("clarity", "tone", "language_match", "timing", "overall")


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def _ask(message: str, session_id: str = "") -> tuple[str, str, str | None]:
    resp = requests.post(
        f"{BACKEND}/chat",
        json={"message": message, "session_id": session_id},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["response"], data["session_id"], data.get("search_query")


def _check_backend() -> None:
    try:
        requests.get(f"{BACKEND}/health", timeout=5).raise_for_status()
    except Exception:
        print("✗ Backend is not reachable at http://localhost:8000")
        print("  Run  make run-backend  in another terminal first.")
        sys.exit(1)


# ── Formatting helpers ────────────────────────────────────────────────────────

def _trunc(text: str, n: int) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"


def _bar(value) -> str:
    if not isinstance(value, (int, float)):
        return "─ err ─"
    return "█" * int(round(value)) + "░" * (10 - int(round(value)))


def _fmt(value) -> str:
    return f"{value:>3}" if isinstance(value, (int, float)) else " — "


def _avg(values: list) -> str:
    nums = [v for v in values if isinstance(v, (int, float))]
    return f"{sum(nums)/len(nums):.1f}" if nums else "—"


def _print_summary(summary: list[dict]) -> None:
    id_w, type_w, col_w = 24, 8, 8
    sep = "+" + "+".join([
        "-" * (id_w + 2), "-" * (type_w + 2),
        "-" * (col_w + 2), "-" * (col_w + 2), "-" * (col_w + 2), "-" * (col_w + 2),
    ]) + "+"

    def row(cells, widths):
        return "|" + "|".join(f" {str(c):<{w}} " for c, w in zip(cells, widths)) + "|"

    widths = (id_w, type_w, col_w, col_w, col_w, col_w)
    print(f"\n{'═'*70}")
    print("  SUMMARY")
    print(f"{'═'*70}")
    print(sep)
    print(row(("ID", "Type", "Search", "Clarify", "Answer", "Give-up"), widths))
    print(sep)
    for s in summary:
        print(row((
            _trunc(s["id"], id_w),
            s["type"],
            _fmt(s.get("search_overall")),
            _fmt(s.get("clarif_overall")),
            _fmt(s.get("answer_overall")),
            _fmt(s.get("giveup_overall")),
        ), widths))
    print(sep)

    avgs = (
        "AVERAGES", "",
        _avg([s.get("search_overall") for s in summary]),
        _avg([s.get("clarif_overall") for s in summary]),
        _avg([s.get("answer_overall") for s in summary]),
        _avg([s.get("giveup_overall") for s in summary]),
    )
    print(row(avgs, widths))
    print(sep)


def _print_detail(title: str, rows: list[dict], criteria: tuple) -> None:
    id_w, q_w, score_w = 24, 36, 6
    widths = [id_w, q_w] + [score_w] * len(criteria)
    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"

    def row(cells):
        return "|" + "|".join(f" {str(c):<{w}} " for c, w in zip(cells, widths)) + "|"

    print(f"\n{'─'*70}")
    print(f"  {title}")
    print(f"{'─'*70}")
    print(sep)
    print(row(["ID", "Question/Turn"] + list(criteria)))
    print(sep)
    for r in rows:
        print(row([_trunc(r["id"], id_w), _trunc(r["label"], q_w)]
                  + [r.get(c, "?") for c in criteria]))
    print(sep)
    print("  Reasoning:")
    for r in rows:
        print(f"    [{r['id']}] {r.get('reasoning', '')}")
    print("  Averages:")
    for c in criteria:
        vals = [r[c] for r in rows if isinstance(r.get(c), (int, float))]
        if vals:
            avg = sum(vals) / len(vals)
            print(f"    {c:<16} {avg:4.1f}/10  {_bar(avg)}")


# ── Langfuse logging ──────────────────────────────────────────────────────────

def _log_to_langfuse(summary: list[dict], run_id: str) -> None:
    lf = get_langfuse()
    if not lf:
        print("\n  (Langfuse not configured — skipping remote logging)")
        return

    for s in summary:
        trace = lf.trace(
            name="batch-eval",
            session_id=run_id,
            input={"question": s["question"]},
            output={"answer": s.get("answer", "")},
            tags=["batch-eval"],
            metadata={
                "example_id": s["id"],
                "type": s["type"],
                "search_query": s.get("search_query"),
                "run_id": run_id,
            },
        )
        scores = {
            f"judge.search.{k}": s["search_scores"].get(k)
            for k in SEARCH_CRITERIA if s.get("search_scores")
        } | {
            f"judge.clarif.{k}": s["clarif_scores"].get(k)
            for k in CLARIF_CRITERIA if s.get("clarif_scores")
        } | {
            f"judge.answer.{k}": s["answer_scores"].get(k)
            for k in ANSWER_CRITERIA if s.get("answer_scores")
        } | {
            f"judge.giveup.{k}": s["giveup_scores"].get(k)
            for k in GIVEUP_CRITERIA if s.get("giveup_scores")
        }
        for name, value in scores.items():
            if value is not None:
                lf.score(trace_id=trace.id, name=name, value=float(value))

    lf.flush()
    print(f"\n  Saved to Langfuse — filter by tag 'batch-eval', session '{run_id}'")


# ── Core runner ───────────────────────────────────────────────────────────────

def run() -> None:
    _check_backend()

    run_id = f"eval-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    total = len(REFERENCE_EXAMPLES)
    print(f"\nRunning evaluation on {total} examples  [{run_id}]\n")

    summary:     list[dict] = []
    answer_rows: list[dict] = []
    clarif_rows: list[dict] = []
    search_rows: list[dict] = []
    giveup_rows: list[dict] = []

    for i, ex in enumerate(REFERENCE_EXAMPLES, 1):
        ex_id     = ex["id"]
        question  = ex["question"]
        follow_up = ex.get("follow_up")
        follow_ups_list = ex.get("follow_ups")
        ex_type = "give-up" if follow_ups_list else ("multi" if ex.get("follow_up") else "single")
        s = {"id": ex_id, "type": ex_type,
             "question": question, "answer": "",
             "search_query": None,
             "search_scores": None, "clarif_scores": None,
             "answer_scores": None, "giveup_scores": None}

        expected = ex.get("expected_behavior")

        if follow_ups_list and expected == "give_up":
            # ── Never-clear (give-up) ────────────────────────────────────────
            print(f"[{i}/{total}] {ex_id}  (never-clear — expect give-up after ≤5 turns)")
            try:
                session_id = str(uuid.uuid4())
                giveup_text = None
                attempts = 0
                for turn, reply in enumerate(follow_ups_list, 1):
                    label = _trunc(question if turn == 1 else reply, 35)
                    print(f"         turn {turn}: '{label}'", end="", flush=True)
                    response, session_id, sq = _ask(
                        question if turn == 1 else reply,
                        "" if turn == 1 else session_id,
                    )
                    attempts = turn
                    print(f" → '{_trunc(response, 40)}'")
                    if sq:  # got a search query = system answered instead of giving up
                        print(f"         ⚠ system answered instead of giving up (search: {sq})")
                        break
                    # Check if this looks like a give-up (no more session = cleared)
                    giveup_text = response

                if giveup_text:
                    print(f"         judging give-up message", end="", flush=True)
                    g_scores = evaluate_giveup(question, giveup_text, attempts) or {}
                    s["giveup_scores"] = g_scores
                    s["giveup_overall"] = g_scores.get("overall")
                    s["answer"] = giveup_text
                    giveup_rows.append({
                        "id": ex_id,
                        "label": f"{_trunc(question, 28)} ({attempts} turns)",
                        "reasoning": g_scores.get("reasoning", ""),
                        **{k: g_scores.get(k, "?") for k in GIVEUP_CRITERIA},
                    })
                    print(f" ✓  overall={g_scores.get('overall')}")
            except Exception as exc:
                print(f" ✗  {exc}")

        elif follow_up:
            print(f"[{i}/{total}] {ex_id}  (multi-turn)")

            # Turn 1 — clarification
            print(f"         turn 1: '{_trunc(question, 38)}'", end="", flush=True)
            try:
                session_id = str(uuid.uuid4())
                clarif_text, session_id, _ = _ask(question, session_id)
                print(" … judging clarification", end="", flush=True)
                c_scores = evaluate_clarification(question, clarif_text) or {}
                s["clarif_scores"] = c_scores
                s["clarif_overall"] = c_scores.get("overall")
                clarif_rows.append({
                    "id": ex_id,
                    "label": f"[clarify] {_trunc(question, 26)}",
                    "reasoning": c_scores.get("reasoning", ""),
                    **{k: c_scores.get(k, "?") for k in CLARIF_CRITERIA},
                })
                print(f" ✓  overall={c_scores.get('overall')}")
            except Exception as exc:
                print(f" ✗  {exc}")
                clarif_rows.append({"id": ex_id, "label": f"[clarify] {question}",
                                    "reasoning": str(exc),
                                    **{k: "err" for k in CLARIF_CRITERIA}})
                session_id = ""

            # Turn 2 — final answer
            print(f"         turn 2: '{follow_up}'", end="", flush=True)
            try:
                answer, _, search_query = _ask(follow_up, session_id)
                s["answer"] = answer
                s["search_query"] = search_query
                print(" … judging search + answer", end="", flush=True)

                if search_query:
                    sq_scores = evaluate_search_query(question, search_query) or {}
                    s["search_scores"] = sq_scores
                    s["search_overall"] = sq_scores.get("overall")
                    search_rows.append({
                        "id": ex_id,
                        "label": f"{_trunc(question, 20)} → {follow_up}",
                        "reasoning": sq_scores.get("reasoning", ""),
                        **{k: sq_scores.get(k, "?") for k in SEARCH_CRITERIA},
                    })

                a_scores = evaluate(question, answer) or {}
                s["answer_scores"] = a_scores
                s["answer_overall"] = a_scores.get("overall")
                answer_rows.append({
                    "id": ex_id,
                    "label": f"[answer] {_trunc(question, 22)} → {follow_up}",
                    "reasoning": a_scores.get("reasoning", ""),
                    **{k: a_scores.get(k, "?") for k in ANSWER_CRITERIA},
                })
                print(f" ✓  overall={a_scores.get('overall')}")
            except Exception as exc:
                print(f" ✗  {exc}")

        else:
            # Single-turn
            print(f"[{i}/{total}] {ex_id}  (single-turn)", end="", flush=True)
            try:
                answer, _, search_query = _ask(question)
                s["answer"] = answer
                s["search_query"] = search_query
                print(" … judging", end="", flush=True)

                if search_query:
                    sq_scores = evaluate_search_query(question, search_query) or {}
                    s["search_scores"] = sq_scores
                    s["search_overall"] = sq_scores.get("overall")
                    search_rows.append({
                        "id": ex_id,
                        "label": _trunc(question, 36),
                        "reasoning": sq_scores.get("reasoning", ""),
                        **{k: sq_scores.get(k, "?") for k in SEARCH_CRITERIA},
                    })

                a_scores = evaluate(question, answer) or {}
                s["answer_scores"] = a_scores
                s["answer_overall"] = a_scores.get("overall")
                answer_rows.append({
                    "id": ex_id,
                    "label": _trunc(question, 36),
                    "reasoning": a_scores.get("reasoning", ""),
                    **{k: a_scores.get(k, "?") for k in ANSWER_CRITERIA},
                })
                print(f" ✓  overall={a_scores.get('overall')}")
            except Exception as exc:
                print(f" ✗  {exc}")

        summary.append(s)

    # ── Output ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  EVALUATION RESULTS")
    print("=" * 70)

    _print_summary(summary)

    if search_rows:
        _print_detail(
            "SEARCH QUERY QUALITY  ⚠ wrong here = wrong everywhere",
            search_rows, SEARCH_CRITERIA,
        )
    if clarif_rows:
        _print_detail("CLARIFICATION QUALITY", clarif_rows, CLARIF_CRITERIA)

    _print_detail("FINAL ANSWER QUALITY", answer_rows, ANSWER_CRITERIA)

    if giveup_rows:
        _print_detail(
            "GIVE-UP QUALITY  (never-clear conversations — max 5 turns)",
            giveup_rows, GIVEUP_CRITERIA,
        )

    _log_to_langfuse(summary, run_id)


if __name__ == "__main__":
    run()
