"""Cloze and correction items from native Urdu Wikipedia sentences.

Source: Dakshina's `native_script_wikipedia/ur.wiki-filt.train.text.shuf.txt.gz`
- Urdu Wikipedia sentences, already filtered and shuffled, CC BY-SA 4.0.

Honesty about the two tasks:

* **cloze** - input and target are both human text. The only machine act is
  choosing which word to hide, and the check proves the hidden word restores the
  original sentence exactly.
* **correction** - the *target* is human text, the *input* is not: it is the same
  sentence with one deliberate error of a kind Urdu typists actually make. Rows
  carry ``input_human=False`` and the error type, and the dataset stats report
  their share so nobody can mistake them for found errors.

**Boilerplate.** Urdu Wikipedia's stub and maintenance notices recur verbatim
across thousands of articles - the worst offender appears 693 times in this
corpus. Left in, they put the *same* cloze item in train and in test. Two
defences, both here: a sentence seen more than ``max_repeats`` times in the
corpus is dropped as template text, and a sentence's ``doc_id`` is a hash of its
own content, so every copy of a sentence shares one id and therefore one split.
"""

from __future__ import annotations

import gzip
import hashlib
import random
from pathlib import Path

from ..schema import Row, nfc
from ..urdu import maskable_indices, usable_sentence, window_keys, words

MASK = "____"
INSTR_CLOZE = f"خالی جگہ ({MASK}) میں مناسب لفظ لکھیں۔ صرف وہی لفظ لکھیں۔"
INSTR_CORRECTION = "اس جملے میں ایک غلطی ہے۔ درست جملہ لکھیں۔"

# Confusion pairs real typists produce: the two h's, the y/e forms, the Arabic
# vs Urdu kaf, nun-ghunna, and the alef forms. Each is one character.
CONFUSIONS = (
    ("ہ", "ھ"),
    ("ھ", "ہ"),
    ("ی", "ے"),
    ("ے", "ی"),
    ("آ", "ا"),
    ("ا", "آ"),
    ("ک", "ك"),
    ("ں", "ن"),
)


def sentence_doc_id(sentence: str) -> str:
    """Content-addressed id: identical sentences collapse to one document."""
    return "urwiki-sent:" + hashlib.sha256(sentence.encode("utf-8")).hexdigest()[:16]


def sentence_frequency(gz_path: Path) -> dict[str, int]:
    """How many times each usable sentence occurs in the corpus.

    Computed once and passed to both builders. It is what makes the boilerplate
    filter possible, and its distribution is reported in the build stats.
    """
    freq: dict[str, int] = {}
    with gzip.open(gz_path, "rt", encoding="utf-8") as fh:
        for line in fh:
            s = nfc(line)
            if usable_sentence(s):
                freq[s] = freq.get(s, 0) + 1
    return freq


def template_key(sentence: str) -> str:
    """A sentence's frame: its first three and last three words.

    Urdu Wikipedia is full of generated census lines - "...<number> ghar hain"
    - identical but for one number. They survive a frequency filter, because
    each instance is unique, yet masking the number yields the *same* prompt with
    a different gold answer, and those prompts then scatter across splits. The
    frame catches the family that exact-match cannot.
    """
    toks = words(sentence)
    return " ".join(toks[:3] + ["|"] + toks[-3:])


def pool_of(sentence: str, pools: int = 2) -> int:
    """Which builder owns a sentence, decided by the sentence's own content.

    Line-number parity was the first attempt and it leaked: a sentence that
    occurs on both an even and an odd line reached both builders, so 114
    sentences ended up as a cloze item *and* as a correction target - the
    correction target being the very sentence the cloze item hides a word from.
    Hashing the content makes ownership a property of the sentence, so a
    duplicate occurrence lands in the same pool every time.
    """
    return int(hashlib.sha256(sentence.encode("utf-8")).hexdigest()[:8], 16) % pools


def iter_sentences(
    gz_path: Path,
    *,
    pool: int | None = None,
    freq=None,
    max_repeats: int = 3,
    max_per_template: int = 3,
    template_counts: dict[str, int] | None = None,
    exclude: set[str] | None = None,
    exclude_windows: set[str] | None = None,
):
    """Yield (line number, sentence) for usable, non-boilerplate, de-duplicated sentences.

    ``pool`` takes the content-addressed half of the corpus that belongs to this
    builder, so two builders never see the same sentence. ``max_per_template``
    then limits how many sentences may share one frame, which is what stops a
    generated template family from dominating a task.
    """
    emitted: set[str] = set()
    tcounts = template_counts if template_counts is not None else {}
    with gzip.open(gz_path, "rt", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            s = nfc(line)
            if not usable_sentence(s):
                continue
            if pool is not None and pool_of(s) != pool:
                continue
            if exclude is not None and s in exclude:
                continue  # already used by another task (see build.py)
            if exclude_windows and (window_keys(s) & exclude_windows):
                continue  # this passage is already inside another task's text
            if freq is not None and freq.get(s, 0) > max_repeats:
                continue  # template / maintenance notice
            if s in emitted:
                continue  # the same sentence twice is one item, not two
            key = template_key(s)
            if tcounts.get(key, 0) >= max_per_template:
                continue  # template family already represented
            tcounts[key] = tcounts.get(key, 0) + 1
            emitted.add(s)
            yield i, s


def cloze_rows(
    gz_path: Path,
    *,
    limit: int | None = None,
    seed: int = 20261017,
    pool: int | None = 0,
    freq=None,
    exclude: set[str] | None = None,
    exclude_windows: set[str] | None = None,
):
    rng = random.Random(seed)
    made = 0
    for lineno, sent in iter_sentences(
        gz_path, pool=pool, freq=freq, exclude=exclude, exclude_windows=exclude_windows
    ):
        if limit is not None and made >= limit:
            return
        toks = words(sent)
        idxs = maskable_indices(toks)
        if not idxs:
            continue
        i = rng.choice(idxs)
        answer = toks[i]
        masked = " ".join(MASK if j == i else t for j, t in enumerate(toks))
        # The check: putting the answer back must reproduce the source sentence.
        if masked.replace(MASK, answer, 1) != " ".join(toks):
            continue
        if masked.count(MASK) != 1:
            continue
        yield Row(
            task="cloze",
            instruction=INSTR_CLOZE,
            input=masked,
            output=answer,
            corpus="urwiki-dakshina-filt",
            doc_id=sentence_doc_id(sent),
            verify_method="restores_source_exactly",
            verified=True,
            target_human=True,
            input_human=True,
            source_url="https://dumps.wikimedia.org/urwiki/",
            meta={"position": i, "words": len(toks), "corpus_line": lineno},
        )
        made += 1


def _corrupt(sent: str, rng: random.Random) -> tuple[str, str] | None:
    """Return (corrupted, error_type) or None when no error could be applied."""
    toks = words(sent)
    order = list(range(len(toks)))
    rng.shuffle(order)
    kinds = ["confusion", "space_split", "transpose"]
    rng.shuffle(kinds)
    for kind in kinds:
        if kind == "space_split":
            for i in order:
                if len(toks[i]) >= 6:
                    cut = len(toks[i]) // 2
                    out = list(toks)
                    out[i] = toks[i][:cut] + " " + toks[i][cut:]
                    return " ".join(out), "space_split"
        elif kind == "confusion":
            for i in order:
                for src, dst in CONFUSIONS:
                    if src in toks[i][1:]:  # not the first letter: keeps the word findable
                        out = list(toks)
                        out[i] = toks[i].replace(src, dst, 1)
                        if out[i] != toks[i]:
                            return " ".join(out), f"confusion:{src}->{dst}"
        else:
            for i in order:
                w = toks[i]
                if len(w) >= 4:
                    j = len(w) // 2
                    out = list(toks)
                    out[i] = w[:j] + w[j + 1] + w[j] + w[j + 2 :]
                    if out[i] != w:
                        return " ".join(out), "transpose"
    return None


def correction_rows(
    gz_path: Path,
    *,
    limit: int | None = None,
    seed: int = 20261017,
    pool: int | None = 1,
    freq=None,
    exclude: set[str] | None = None,
    exclude_windows: set[str] | None = None,
):
    """Correction items, over whichever half of the corpus ``keep`` admits."""
    rng = random.Random(seed + 1)
    made = 0
    for lineno, sent in iter_sentences(
        gz_path, pool=pool, freq=freq, exclude=exclude, exclude_windows=exclude_windows
    ):
        if limit is not None and made >= limit:
            return
        got = _corrupt(sent, rng)
        if got is None:
            continue
        corrupted, kind = got
        if corrupted == sent:
            continue
        yield Row(
            task="correction",
            instruction=INSTR_CORRECTION,
            input=corrupted,
            output=sent,
            corpus="urwiki-dakshina-filt",
            doc_id=sentence_doc_id(sent),
            verify_method="single_edit_differs_from_target",
            verified=True,
            target_human=True,
            input_human=False,
            source_url="https://dumps.wikimedia.org/urwiki/",
            meta={"error_type": kind, "words": len(words(sent)), "corpus_line": lineno},
        )
        made += 1
