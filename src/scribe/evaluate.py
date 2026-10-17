"""Measure the desk's engines on a held-out split.

    python -m scribe.evaluate --split test

This is the no-model baseline. It exists for two reasons: the UI shows real
numbers rather than adjectives, and the trained arm (when the GPU frees) has to
beat something that was measured the same way.

Every figure carries its denominator, and each task reports more than one metric
where one metric would flatter it:

* transliteration - exact match, "any attested romanisation" match, and the
  share of inputs the lexicon even covered. Coverage is the headline here: the
  dataset splits word-level transliteration *by word*, so a test word is by
  construction absent from the train lexicon. A lookup engine cannot win that,
  and the number says so instead of hiding it.
* cloze - exact match, and top-5 (did the right word appear among the
  alternatives the engine offered).
* correction - exact sentence match, and the weaker "found the broken token"
  rate, which is what a user actually sees highlighted.
* headline - exact match and token F1, because exact match on a free-text
  heading is a harsh and partly arbitrary target.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from .engine import (
    Corpus,
    correct,
    fill_blank,
    suggest_heading,
    transliterate_to_roman,
    transliterate_to_urdu,
)
from .urdu import strip_marks, words


def _f1(pred: str, gold: str) -> float:
    p, g = [strip_marks(w) for w in words(pred)], [strip_marks(w) for w in words(gold)]
    if not p or not g:
        return 0.0
    common = collections.Counter(p) & collections.Counter(g)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(p), overlap / len(g)
    return 2 * precision * recall / (precision + recall)


def load_split(data_dir: Path, split: str) -> list[dict]:
    path = Path(data_dir) / f"{split}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def evaluate(corpus: Corpus, rows: list[dict], *, limit_per_task: int | None = None) -> dict:
    by_task: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)

    report: dict[str, dict] = {}

    # --- transliteration, both directions ---------------------------------------
    for task, fn in (("translit_ur2roman", transliterate_to_roman), ("translit_roman2ur", transliterate_to_urdu)):
        items = by_task.get(task, [])[: limit_per_task or None]
        exact = attested = covered = 0
        for r in items:
            ans = fn(r["input"], corpus)
            pred = ans.output.strip()
            if pred == r["output"].strip():
                exact += 1
            accepted = {v["roman"] for v in r.get("meta", {}).get("accepted_outputs", [])} or {r["output"]}
            if pred in accepted:
                attested += 1
            if ans.confidence > 0:
                covered += 1
        n = len(items)
        report[task] = {
            "items": n,
            "exact_match": round(exact / n, 4) if n else None,
            "any_attested_match": round(attested / n, 4) if n else None,
            "lexicon_coverage": round(covered / n, 4) if n else None,
            "note": (
                "The dataset splits word-level transliteration by word, so a test word is "
                "absent from the train lexicon by construction. Coverage near zero is the "
                "design of the split, not a bug in the engine."
            ),
        }

    # --- cloze --------------------------------------------------------------------
    items = by_task.get("cloze", [])[: limit_per_task or None]
    exact = top5 = ctx = 0
    for r in items:
        ans = fill_blank(r["input"], corpus)
        gold = strip_marks(r["output"])
        if strip_marks(ans.output) == gold:
            exact += 1
        if gold in {strip_marks(a) for a in [ans.output, *ans.alternatives]}:
            top5 += 1
        if ans.method.startswith("context index"):
            ctx += 1
    n = len(items)
    report["cloze"] = {
        "items": n,
        "exact_match": round(exact / n, 4) if n else None,
        "top5_match": round(top5 / n, 4) if n else None,
        "answered_from_context_index": round(ctx / n, 4) if n else None,
    }

    # --- correction ---------------------------------------------------------------
    items = by_task.get("correction", [])[: limit_per_task or None]
    exact = located = unchanged = 0
    for r in items:
        ans = correct(r["input"], corpus)
        if ans.output.strip() == r["output"].strip():
            exact += 1
        if ans.output.strip() == r["input"].strip():
            unchanged += 1
        else:
            # Did it at least touch the token that was broken?
            bad = {a for a, b in zip(words(r["input"]), words(r["output"])) if a != b}
            touched = {a for a, b in zip(words(r["input"]), words(ans.output)) if a != b}
            if bad & touched:
                located += 1
    n = len(items)
    report["correction"] = {
        "items": n,
        "exact_match": round(exact / n, 4) if n else None,
        "located_the_broken_token": round(located / n, 4) if n else None,
        "left_unchanged": round(unchanged / n, 4) if n else None,
    }

    # --- headline -----------------------------------------------------------------
    items = by_task.get("headline", [])[: limit_per_task or None]
    exact = 0
    f1s: list[float] = []
    borrowed = 0
    for r in items:
        ans = suggest_heading(r["input"], corpus)
        if ans.output.strip() == r["output"].strip():
            exact += 1
        f1s.append(_f1(ans.output, r["output"]))
        if ans.method.startswith("heading of the most similar"):
            borrowed += 1
    n = len(items)
    report["headline"] = {
        "items": n,
        "exact_match": round(exact / n, 4) if n else None,
        "token_f1": round(sum(f1s) / n, 4) if n else None,
        "answered_by_retrieval": round(borrowed / n, 4) if n else None,
    }

    report["_meta"] = {
        "rows_evaluated": sum(v.get("items", 0) for k, v in report.items() if k != "_meta"),
        "corpus": corpus.stats,
        "model_used": None,
        "methodology": (
            "Cache-augmented: the whole corpus is indexed in memory at startup, no retrieval "
            "hop for lookups. BM25 retrieval where similarity decides the answer (cloze "
            "neighbours, headline suggestion). No reranker: that needs a model, and the GPU is "
            "reserved - the slot is explicit in engine.rerank_slot."
        ),
    }
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Measure the Scribe engines on a split")
    ap.add_argument("--split", default="test", choices=("val", "test"))
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--limit-per-task", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    corpus = Corpus(args.data) if args.data else Corpus()
    rows = load_split(corpus.data_dir, args.split)
    report = evaluate(corpus, rows, limit_per_task=args.limit_per_task)
    report["_meta"]["split"] = args.split

    text = json.dumps(report, ensure_ascii=False, indent=2)
    out = args.out or (corpus.data_dir / f"baseline_{args.split}.json")
    Path(out).write_text(text, encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))
    print(f"\nwritten to {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
