"""Transliteration items from Dakshina's human romanisations.

Dakshina's romanised side was typed by people, which is the point: a model
trained on romanisation produced by rules alone learns a scheme nobody
uses. Licence CC BY-SA 4.0, so scribe-100k inherits share-alike.

**Contamination, measured not assumed.** `ur.romanized.rejoined.tsv` holds 9,759
sentence pairs. The dev and test native files hold 9,631 of those same
sentences, and `ur.romanized.rejoined.aligned.tsv` is the word alignment *of
those sentences*. So Dakshina's Urdu sentence-level romanisation is evaluation
data almost in its entirety, and the one genuinely trainable file is
`lexicons/ur.translit.sampled.train.tsv` (106,260 attested word forms).

Therefore:

* ``lexicon_rows``       -> training rows, word level, both directions.
* ``eval_sentence_rows`` -> an evaluation-only stream, written to its own file
  and never mixed into the training rows. It stays comparable with the number
  the `urdunlp` package already reports on this same held-out set.
"""

from __future__ import annotations

import collections
from pathlib import Path

from ..schema import Row, nfc
from ..urdu import LATIN, has_arabic_only_form, urdu_share, words

INSTR_UR2ROMAN = "اس اردو جملے کو رومن اردو میں لکھیں۔"
INSTR_ROMAN2UR = "Write this Roman Urdu sentence in Urdu script."
INSTR_WORD_UR2ROMAN = "اس اردو لفظ کو رومن اردو میں لکھیں۔"
INSTR_WORD_ROMAN2UR = "Write this Roman Urdu word in Urdu script."


def held_out_natives(root: Path) -> set[str]:
    """The dev+test native sentences. Public so the build can report the overlap."""
    out: set[str] = set()
    for name in ("ur.romanized.rejoined.test.native.txt", "ur.romanized.rejoined.dev.native.txt"):
        p = root / "romanized" / name
        if p.exists():
            out |= {nfc(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()}
    return out


def _ok_pair(urdu: str, roman: str) -> bool:
    if not urdu or not roman:
        return False
    if urdu_share(urdu) < 0.80:
        return False
    if LATIN.search(urdu):
        return False
    if not LATIN.search(roman):
        return False
    # A romanised side with Urdu letters in it is a misaligned row.
    return urdu_share(roman) < 0.05


def eval_sentence_rows(root: Path):
    """Sentence-level pairs, both directions, for EVALUATION ONLY.

    Yields only sentences that are in Dakshina's dev/test native files, i.e. the
    opposite filter to a training builder. Anything outside them is skipped
    rather than quietly promoted to training data.
    """
    path = root / "romanized" / "ur.romanized.rejoined.tsv"
    held = held_out_natives(root)
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split("\t")
        if len(parts) != 2:
            continue
        urdu, roman = nfc(parts[0]), nfc(parts[1])
        if not _ok_pair(urdu, roman) or urdu not in held:
            continue
        if not (5 <= len(words(urdu)) <= 60):
            continue
        doc_id = f"dakshina-rejoined:{lineno}"
        for task, instruction, src, tgt in (
            ("translit_ur2roman", INSTR_UR2ROMAN, urdu, roman),
            ("translit_roman2ur", INSTR_ROMAN2UR, roman, urdu),
        ):
            yield Row(
                task=task,
                instruction=instruction,
                input=src,
                output=tgt,
                corpus="dakshina",
                doc_id=doc_id,
                verify_method="pair_script_check+in_dakshina_heldout",
                verified=True,
                target_human=True,
                input_human=True,
                source_url="https://github.com/google-research-datasets/dakshina",
                meta={"level": "sentence", "split": "eval_only", "words": len(words(urdu))},
            )


def _lexicon(root: Path):
    """Parse the train lexicon into urdu -> [(roman, attestations)], descending."""
    path = root / "lexicons" / "ur.translit.sampled.train.tsv"
    table: dict[str, list[tuple[str, int]]] = collections.defaultdict(list)
    lineno_of: dict[str, int] = {}
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        urdu, roman = nfc(parts[0]), nfc(parts[1])
        try:
            count = int(parts[2]) if len(parts) > 2 else 1
        except ValueError:
            count = 1
        if not _ok_pair(urdu, roman) or len(words(urdu)) != 1 or len(urdu) < 3:
            continue
        if has_arabic_only_form(urdu):
            continue  # keep the Urdu side in Urdu orthography
        table[urdu].append((roman, count))
        lineno_of.setdefault(urdu, lineno)
    for variants in table.values():
        variants.sort(key=lambda rc: (-rc[1], rc[0]))
    return table, lineno_of


def lexicon_rows(root: Path, *, min_count: int = 2, limit: int | None = None):
    """Word-level training rows, both directions.

    Forward (Urdu -> Roman) emits **one row per Urdu word**, not one per
    romanisation. The first version emitted every attested variant as its own
    row, which put the same input in 13,416 rows with different gold answers -
    an exact-match score would then mark a correct, attested romanisation wrong.
    The variants are kept where they belong: ``meta.accepted_outputs`` lists
    every romanisation attested at least ``min_count`` times, with its count, so
    an evaluation can score "the most attested" and "any attested" separately.

    Reverse (Roman -> Urdu) needs two conditions, not one. "Unambiguous in this
    lexicon" was the first attempt and it produced wrong-sense gold: `subah` ->
    صوبہ, when a reader would say صبح, because صوبہ happened to be the only word
    in the sampled lexicon with that romanisation attested. So a reverse row is
    emitted only when the pairing is **mutual best**: the romanisation maps to
    one Urdu word, *and* it is that word's most-attested romanisation. The
    measured cost of the stricter rule is 20,259 -> 14,055 candidate pairs.
    """
    table, lineno_of = _lexicon(root)
    reverse: dict[str, set[str]] = collections.defaultdict(set)
    for urdu, variants in table.items():
        for roman, count in variants:
            if count >= min_count:
                reverse[roman].add(urdu)

    made = 0
    for urdu, variants in table.items():
        if limit is not None and made >= limit:
            return
        attested = [(r, c) for r, c in variants if c >= min_count]
        if not attested:
            continue
        roman, count = attested[0]  # _lexicon sorts by descending attestation
        yield Row(
            task="translit_ur2roman",
            instruction=INSTR_WORD_UR2ROMAN,
            input=urdu,
            output=roman,
            corpus="dakshina",
            doc_id=f"dakshina-lexicon:{lineno_of[urdu]}",
            verify_method="pair_script_check+attestation",
            verified=True,
            target_human=True,
            input_human=True,
            source_url="https://github.com/google-research-datasets/dakshina",
            meta={
                "level": "word",
                "attestations": count,
                "accepted_outputs": [{"roman": r, "attestations": c} for r, c in attested],
                "variants_attested": len(attested),
                "variants_total": len(variants),
            },
        )
        made += 1

    best = {u: v[0][0] for u, v in table.items() if v and v[0][1] >= min_count}
    for roman, urdus in reverse.items():
        if limit is not None and made >= limit:
            return
        if len(urdus) != 1:
            continue
        urdu = next(iter(urdus))
        if best.get(urdu) != roman:
            continue  # not this word's own preferred romanisation
        yield Row(
            task="translit_roman2ur",
            instruction=INSTR_WORD_ROMAN2UR,
            input=roman,
            output=urdu,
            corpus="dakshina",
            doc_id=f"dakshina-lexicon:{lineno_of[urdu]}",
            verify_method="pair_script_check+mutual_best_reverse",
            verified=True,
            target_human=True,
            input_human=True,
            source_url="https://github.com/google-research-datasets/dakshina",
            meta={"level": "word", "unambiguous": True, "mutual_best": True},
        )
        made += 1
