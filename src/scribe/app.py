"""Scribe — the Urdu desk. Web UI over the engines in `engine.py`.

    uv run python -m scribe.app --port 8810

Everything is served from one process: the corpus is loaded once at startup
(12 s for 90,016 rows) and every request is answered from memory.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .engine import (
    Corpus,
    correct,
    fill_blank,
    search,
    suggest_heading,
    transliterate_to_roman,
    transliterate_to_urdu,
)
from .web import App

UI_DIR = Path(__file__).resolve().parent.parent.parent / "ui"
# How many lines one batch request will do. Beyond this the caller is told what
# was left rather than being made to wait.
MAX_BATCH_LINES = 300


def build_app(corpus: Corpus, *, examples_split: str = "val") -> App:
    app = App(UI_DIR, title="scribe")

    examples: dict[str, list[dict]] = {}
    ex_path = corpus.data_dir / f"{examples_split}.jsonl"
    if ex_path.exists():
        with ex_path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    row = json.loads(line)
                    examples.setdefault(row["task"], []).append(row)

    dataset_stats = {}
    stats_path = corpus.data_dir / "stats.json"
    if stats_path.exists():
        dataset_stats = json.loads(stats_path.read_text(encoding="utf-8"))

    baseline = {}
    base_path = corpus.data_dir / "baseline_test.json"
    if base_path.exists():
        baseline = json.loads(base_path.read_text(encoding="utf-8"))

    @app.route("/api/stats")
    def _stats(_payload: dict) -> dict:
        return {
            "corpus": corpus.stats,
            "dataset": {
                k: dataset_stats.get(k)
                for k in ("rows", "source_documents", "by_task", "by_split",
                          "target_human_share", "input_human_share", "licence")
            },
            "examples_available": {k: len(v) for k, v in sorted(examples.items())},
        }

    @app.route("/api/benchmark")
    def _benchmark(_payload: dict) -> dict:
        return baseline or {"note": "No baseline file yet. Run: python -m scribe.evaluate --split test"}

    @app.route("/api/example")
    def _example(payload: dict) -> dict:
        task = payload.get("task", "cloze")
        pool = examples.get(task) or []
        if not pool:
            return {"error": f"no held-out examples for {task}"}
        row = random.choice(pool)
        return {
            "task": task,
            "input": row["input"],
            "gold": row["output"],
            "split": examples_split,
            "doc_id": row["doc_id"],
            "meta": row.get("meta", {}),
        }

    def _text(payload: dict) -> str:
        return (payload.get("text") or "").strip()

    @app.route("/api/translit/roman")
    def _to_roman(payload: dict) -> dict:
        return transliterate_to_roman(_text(payload), corpus).to_dict()

    @app.route("/api/translit/urdu")
    def _to_urdu(payload: dict) -> dict:
        return transliterate_to_urdu(_text(payload), corpus).to_dict()

    @app.route("/api/cloze")
    def _cloze(payload: dict) -> dict:
        return fill_blank(_text(payload), corpus).to_dict()

    @app.route("/api/correct")
    def _correct(payload: dict) -> dict:
        return correct(_text(payload), corpus).to_dict()

    @app.route("/api/heading")
    def _heading(payload: dict) -> dict:
        return suggest_heading(_text(payload), corpus).to_dict()

    @app.route("/api/batch")
    def _batch(payload: dict) -> dict:
        """Run one task over many lines at once.

        A desk that answers one sentence at a time is a demonstration; a person
        correcting a document has three hundred of them. The work is the same per
        line, so this caps the number of lines rather than the time: a caller
        gets a clear limit instead of a request that quietly takes four minutes.
        """
        task = payload.get("task") or "correct"
        runner = {
            "correct": correct,
            "cloze": fill_blank,
            "roman": transliterate_to_roman,
            "urdu": transliterate_to_urdu,
            "heading": suggest_heading,
        }.get(task)
        if runner is None:
            return {"error": f"unknown task {task!r}"}

        raw = payload.get("text") or ""
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        limit = int(payload.get("limit", MAX_BATCH_LINES))
        over = max(0, len(lines) - limit)
        rows = []
        for line in lines[:limit]:
            answer = runner(line, corpus)
            rows.append({
                "input": line,
                "output": answer.output,
                "method": answer.method,
                "confidence": round(answer.confidence, 3),
                "changed": answer.output.strip() != line.strip(),
            })
        return {
            "task": task,
            "rows": rows,
            "lines_given": len(lines),
            "lines_done": len(rows),
            "lines_skipped": over,
            "changed": sum(1 for r in rows if r["changed"]),
        }

    @app.route("/api/search")
    def _search(payload: dict) -> dict:
        query = (payload.get("query") or payload.get("text") or "").strip()
        if not query:
            return {"query": "", "hits": [], "searched": len(corpus.sentences)}
        return search(query, corpus, limit=int(payload.get("limit", 10)))

    return app


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run the Scribe desk")
    ap.add_argument("--port", type=int, default=8810)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--limit-rows", type=int, default=None, help="load fewer rows (faster startup, for tests)")
    args = ap.parse_args(argv)

    corpus = Corpus(args.data, limit=args.limit_rows) if args.data else Corpus(limit=args.limit_rows)
    print(f"corpus ready: {corpus.stats['rows_loaded']} rows", flush=True)
    build_app(corpus).serve(args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
