"""Independent re-verification of a built scribe-100k directory.

This does not import the builders and does not trust their `verified` flag. It
reads the written JSONL back and re-derives every check from the row itself,
because the failure mode worth catching is a builder that marked a row verified
by a path that no longer holds.

    python -m scribe.verify --dir "C:/d drive data/scribe-100k/v1"

Exit code is non-zero when any check fails, so this can gate a build. Every
check prints its denominator: a check that silently applied to nothing is the
defect this file exists to catch.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from .urdu import LATIN, is_urdu_word, strip_marks, urdu_share, usable_sentence, words

MASK = "____"
SPLIT_FILES = ("train", "val", "test")


def load(d: Path):
    rows = []
    for name in SPLIT_FILES:
        p = d / f"{name}.jsonl"
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                r["_split"] = name
                rows.append(r)
    return rows


def check_cloze(r) -> str | None:
    if r["input"].count(MASK) != 1:
        return "mask_count"
    restored = r["input"].replace(MASK, r["output"], 1)
    if len(words(restored)) != len(words(r["input"])):
        return "answer_changes_word_count"
    if not usable_sentence(restored):
        return "restored_not_usable"
    if strip_marks(r["output"]) in [strip_marks(w) for w in words(r["input"])]:
        return "answer_also_present_in_prompt"
    if not is_urdu_word(r["output"]):
        return "answer_not_urdu_word"
    return None


def check_correction(r) -> str | None:
    if r["input"] == r["output"]:
        return "no_error_introduced"
    if not usable_sentence(r["output"]):
        return "target_not_usable"
    wi, wo = words(r["input"]), words(r["output"])
    if abs(len(wi) - len(wo)) > 1:
        return "word_count_drift"
    kind = r.get("meta", {}).get("error_type", "")
    if not kind:
        return "error_type_missing"
    # Exactly one token should differ, except space_split which splits one into two.
    diffs = sum(1 for a, b in zip(wi, wo) if a != b)
    if kind == "space_split":
        if len(wi) != len(wo) + 1:
            return "space_split_did_not_split"
    elif diffs != 1:
        return f"expected_one_changed_token_got_{diffs}"
    if r.get("input_human") is not False:
        return "corrupted_input_claimed_human"
    return None


def check_translit(r) -> str | None:
    ur, ro = (r["input"], r["output"]) if r["task"] == "translit_ur2roman" else (r["output"], r["input"])
    if urdu_share(ur) < 0.80:
        return "urdu_side_not_urdu"
    if LATIN.search(ur):
        return "latin_in_urdu_side"
    if not LATIN.search(ro):
        return "roman_side_has_no_latin"
    if urdu_share(ro) >= 0.05:
        return "urdu_in_roman_side"
    if r["input"].strip() == r["output"].strip():
        return "identity_pair"
    return None


CHECKS = {
    "cloze": check_cloze,
    "correction": check_correction,
    "translit_ur2roman": check_translit,
    "translit_roman2ur": check_translit,
}


def audit(rows) -> dict:
    checked = collections.Counter()
    failed: collections.Counter = collections.Counter()
    unchecked = collections.Counter()
    for r in rows:
        fn = CHECKS.get(r["task"])
        if fn is None:
            unchecked[r["task"]] += 1
            continue
        checked[r["task"]] += 1
        why = fn(r)
        if why:
            failed[f"{r['task']}:{why}"] += 1

    # Cross-row leakage. These are the checks a per-row verifier cannot make.
    by_input = collections.defaultdict(set)
    for r in rows:
        by_input[r["input"]].add(r["_split"])
    input_in_two_splits = sum(1 for s in by_input.values() if len(s) > 1)

    doc_splits = collections.defaultdict(set)
    for r in rows:
        doc_splits[r["doc_id"]].add(r["_split"])
    doc_in_two_splits = sum(1 for s in doc_splits.values() if len(s) > 1)

    # A sentence must not feed two tasks: the correction target is the cloze source.
    doc_tasks = collections.defaultdict(set)
    for r in rows:
        doc_tasks[r["doc_id"]].add(r["task"])
    wiki_shared = sum(
        1
        for d, t in doc_tasks.items()
        if d.startswith("urwiki") and {"cloze", "correction"} <= t
    )

    ids = collections.Counter(r["row_id"] for r in rows)
    return {
        "rows_read": len(rows),
        "rows_checked": sum(checked.values()),
        "rows_unchecked_no_check_defined": dict(unchecked),
        "checked_per_task": dict(checked),
        "failures": dict(failed),
        "failure_rate": round(sum(failed.values()) / max(1, sum(checked.values())), 6),
        "duplicate_row_ids": sum(v - 1 for v in ids.values() if v > 1),
        "identical_input_in_two_splits": input_in_two_splits,
        "documents_in_two_splits": doc_in_two_splits,
        "wiki_sentences_feeding_both_cloze_and_correction": wiki_shared,
        "distinct_inputs": len(by_input),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Re-verify a built scribe-100k directory")
    ap.add_argument("--dir", type=Path, required=True)
    args = ap.parse_args(argv)

    rows = load(args.dir)
    if not rows:
        print("no rows read - nothing was verified")
        return 2
    report = audit(rows)
    (args.dir / "verify.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=True, indent=2))
    bad = (
        sum(report["failures"].values())
        + report["duplicate_row_ids"]
        + report["documents_in_two_splits"]
        + report["wiki_sentences_feeding_both_cloze_and_correction"]
    )
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
