"""Build scribe-100k.

    python -m scribe.build --target 100000 --out "C:/d drive data/scribe-100k/v1"

Writes, into ``--out``:

* ``train.jsonl`` / ``val.jsonl`` / ``test.jsonl`` - 90/7/3, split by **source
  document**, so no sentence contributes to two splits.
* ``eval_dakshina_sentences.jsonl`` - sentence-level transliteration, evaluation
  only, never part of the three splits above (see builders/dakshina.py).
* ``stats.json`` - totals *with their denominators*: how many candidates each
  builder saw, how many rows it produced, the achieved share per task against
  its cap, and the human-provenance shares.

Pools are disjoint by construction: a Wikipedia sentence used for a cloze item is
never also used for a correction item, because the same sentence in two tasks
leaks one task's answer into the other.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

from .builders import dakshina, wiki_sentences
from .schema import TASKS

# Share caps, checked against the achieved total. No task may dominate: a
# 70%-transliteration corpus scores well on transliteration and teaches nothing.
#
# The numbers come from measured pool sizes, not from taste. Measured 2026-10-17:
#   Urdu words with a romanisation attested >=2 times : 17,593  -> ur2roman ceiling
#   romanisations mapping to exactly one Urdu word    : 20,259  -> roman2ur ceiling
#   distinct usable non-boilerplate wiki sentences    : 353,754 -> split between
#                                                       cloze and correction
# ur2roman is therefore capped at its ceiling and the wiki tasks absorb the rest.
CAPS = {
    "translit_ur2roman": 0.175,
    "translit_roman2ur": 0.15,
    "cloze": 0.375,
    "correction": 0.30,
    "headline": 0.25,  # builder lands with the full dump; 0 until then
}

# A pool can come in under its cap - the stricter mutual-best rule for reverse
# transliteration cut its candidates from 20,259 to 14,055, below a 15% cap of a
# 100k target. Rather than hand-tune the caps after every such measurement, the
# wiki tasks absorb the shortfall up to their own hard ceilings, and the build
# reports what it absorbed.
CEILINGS = {"cloze": 0.42, "correction": 0.34}

SPLITS = (("train", 0.90), ("val", 0.07), ("test", 0.03))

DEFAULT_DAKSHINA = Path("D:/github/urdu-nlp-toolkit/data/dakshina/ur")
DEFAULT_WIKI_GZ = DEFAULT_DAKSHINA / "native_script_wikipedia" / "ur.wiki-filt.train.text.shuf.txt.gz"


def split_of(doc_id: str) -> str:
    """Deterministic split by document, stable across runs and across row counts.

    Hashing the doc id (not the row) is what keeps a document whole: every item
    derived from one sentence lands in the same split, whatever order it was built in.
    """
    h = int(hashlib.sha256(doc_id.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    acc = 0.0
    for name, frac in SPLITS:
        acc += frac
        if h < acc:
            return name
    return SPLITS[-1][0]


def build(target: int, dakshina_root: Path, wiki_gz: Path):
    budget = {task: int(target * CAPS[task]) for task in TASKS}
    rows, seen, prompts = [], set(), set()
    seen_counts: collections.Counter = collections.Counter()

    def take(stream, builder: str):
        kept = 0
        for row in stream:
            seen_counts[builder] += 1
            if budget.get(row.task, 0) <= 0:
                continue  # this task is full; another task from the same stream may not be
            if row.row_id in seen or row.input in prompts:
                # A prompt may only appear once in the whole dataset. Two rows with
                # the same prompt and different gold answers make an unanswerable
                # item, and if they land in different splits they also leak.
                seen_counts[builder + ":duplicate_prompt_or_row"] += 1
                continue
            seen.add(row.row_id)
            prompts.add(row.input)
            budget[row.task] -= 1
            rows.append(row)
            kept += 1
            if all(budget.get(t, 0) <= 0 for t in TASKS):
                break
        return kept

    produced = {}
    produced["dakshina_lexicon"] = take(dakshina.lexicon_rows(dakshina_root), "dakshina_lexicon")

    # One pass over the corpus to find the boilerplate before either wiki builder runs.
    freq = wiki_sentences.sentence_frequency(wiki_gz)
    boilerplate = {s: n for s, n in freq.items() if n > 3}

    # Disjoint sentence pools, decided by the sentence's content hash (see pool_of).
    produced["wiki_cloze"] = take(
        wiki_sentences.cloze_rows(wiki_gz, pool=0, freq=freq), "wiki_cloze"
    )
    produced["wiki_correction"] = take(
        wiki_sentences.correction_rows(wiki_gz, pool=1, freq=freq), "wiki_correction"
    )
    # Top-up pass: fill any shortfall from the wiki pools, which are large enough
    # to absorb it, without letting either exceed its hard ceiling.
    topped_up = {}
    for task in ("cloze", "correction"):
        shortfall = target - len(rows)
        if shortfall <= 0:
            break
        room = int(target * CEILINGS[task]) - sum(1 for r in rows if r.task == task)
        if room <= 0:
            continue
        budget[task] += min(shortfall, room)
        fn = wiki_sentences.cloze_rows if task == "cloze" else wiki_sentences.correction_rows
        pool = 0 if task == "cloze" else 1
        topped_up[task] = take(fn(wiki_gz, pool=pool, freq=freq), f"topup_{task}")
    produced["topped_up"] = topped_up

    corpus = {
        "wiki_usable_sentence_instances": sum(freq.values()),
        "wiki_distinct_usable_sentences": len(freq),
        "wiki_boilerplate_sentences_dropped": len(boilerplate),
        "wiki_boilerplate_instances_dropped": sum(boilerplate.values()),
        "wiki_boilerplate_max_repeats": max(freq.values()) if freq else 0,
        "wiki_boilerplate_threshold": 3,
    }
    return rows, produced, dict(seen_counts), corpus


def stats(rows, produced, seen_counts, target: int, dakshina_root: Path, eval_rows, corpus: dict) -> dict:
    by_task = collections.Counter(r.task for r in rows)
    by_split = collections.Counter(split_of(r.doc_id) for r in rows)
    by_corpus = collections.Counter(r.corpus for r in rows)
    err_types = collections.Counter(
        r.meta.get("error_type", "") for r in rows if r.task == "correction"
    )
    docs = {r.doc_id for r in rows}
    n = len(rows) or 1
    return {
        "rows": len(rows),
        "target": target,
        "source_documents": len(docs),
        "unique_row_ids": len({r.row_id for r in rows}),
        "verified_rows": sum(1 for r in rows if r.verified),
        "target_human_share": round(sum(1 for r in rows if r.target_human) / n, 4),
        "input_human_share": round(sum(1 for r in rows if r.input_human) / n, 4),
        "by_task": dict(by_task),
        "achieved_share_vs_cap": {
            # Two limits per task: the cap it is filled to in the first pass, and
            # the ceiling it may reach when it absorbs another pool's shortfall.
            k: {
                "share": round(v / n, 4),
                "cap": CAPS[k],
                "ceiling": CEILINGS.get(k, CAPS[k]),
                "within_ceiling": (v / n) <= CEILINGS.get(k, CAPS[k]) + 1e-9,
            }
            for k, v in by_task.items()
        },
        "by_split": dict(by_split),
        "by_split_share": {k: round(v / n, 4) for k, v in by_split.items()},
        "split_doc_overlap": _doc_overlap(rows),
        "by_corpus": dict(by_corpus),
        "corpus_pool": corpus,
        "candidates_seen_per_builder": seen_counts,
        "rows_per_builder": produced,
        "correction_error_types": {k: v for k, v in err_types.items() if k},
        "eval_only_rows": len(eval_rows),
        "dakshina_heldout_sentences": len(dakshina.held_out_natives(dakshina_root)),
        "caps": CAPS,
        "licence": "CC BY-SA 4.0 (inherited from Dakshina and Urdu Wikipedia)",
        "model_in_label_path": False,
    }


def _doc_overlap(rows) -> dict:
    """Documents appearing in more than one split. Must be empty; printed anyway."""
    where: dict[str, set[str]] = collections.defaultdict(set)
    for r in rows:
        where[r.doc_id].add(split_of(r.doc_id))
    bad = {d: sorted(s) for d, s in where.items() if len(s) > 1}
    return {"documents_in_multiple_splits": len(bad), "examples": list(bad)[:5]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build scribe-100k rows")
    ap.add_argument("--target", type=int, default=1000)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dakshina", type=Path, default=DEFAULT_DAKSHINA)
    ap.add_argument("--wiki-gz", type=Path, default=DEFAULT_WIKI_GZ)
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    rows, produced, seen, corpus = build(args.target, args.dakshina, args.wiki_gz)

    handles = {name: (args.out / f"{name}.jsonl").open("w", encoding="utf-8", newline="\n") for name, _ in SPLITS}
    try:
        for row in rows:
            handles[split_of(row.doc_id)].write(row.to_json() + "\n")
    finally:
        for fh in handles.values():
            fh.close()

    eval_rows = list(dakshina.eval_sentence_rows(args.dakshina))
    with (args.out / "eval_dakshina_sentences.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for row in eval_rows:
            fh.write(row.to_json() + "\n")

    s = stats(rows, produced, seen, args.target, args.dakshina, eval_rows, corpus)
    (args.out / "stats.json").write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")
    # Windows consoles are cp1252; stats.json keeps the Urdu, the console gets escapes.
    print(json.dumps(s, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
