"""Deterministic Urdu text helpers.

Everything here is rule-based. No model, no network. These functions decide
which words a cloze item may mask and which sentences are usable at all, so
their mistakes become dataset defects - they are tested in tests/.
"""

from __future__ import annotations

import re

# Urdu uses the Arabic block plus the Arabic Extended/Presentation ranges.
URDU_CHARS = re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")
# Diacritics (zer, zabar, pesh, ...) are optional in running text and must not
# count as letters when deciding whether a token is a real word.
URDU_MARKS = re.compile(r"[ً-ٰٟۖ-ۭ]")
LATIN = re.compile(r"[A-Za-z]")
DIGITS = re.compile(r"[0-9٠-٩۰-۹]")
# Urdu full stop, question mark, comma, semicolon, plus Latin equivalents.
SENT_END = re.compile(r"(?<=[۔?!؟.])\s+")
WORD_SPLIT = re.compile(r"\s+")

# Closed-class words: a cloze item that masks one of these is a grammar quiz
# with many right answers, not a comprehension item. Hand-listed, not learned.
FUNCTION_WORDS = frozenset(
    """
    کا کی کے کو کہ کہا میں سے پر نے ہے ہیں تھا تھی تھے ہوا ہوئی ہوئے ہو ہوں
    اور یا بھی تو ہی نہ نہیں جو جس جن اس ان یہ وہ ایک دو اپنے اپنی اپنا
    لیے لئے بعد پہلے ساتھ بغیر طور علاوہ باعث ذریعے مگر لیکن اگر جب تک
    گا گی گے رہا رہی رہے کر کرنے کرنا کیا گیا گئی گئے دیا دی دیے لگا
    """.split()  # noqa: SIM905 - a readable block beats a 70-item list literal
)


# Arabic-only letter forms that Urdu spells differently: Arabic yeh, Arabic kaf,
# teh marbuta and alef maksura. They appear in quoted Arabic inside Urdu
# Wikipedia. Rejecting them from gold answers keeps the dataset self-consistent:
# the correction task treats Urdu kaf -> Arabic kaf as an *error*, so a cloze gold
# answer must not contain one.
#
# NOT in this set, deliberately: ؤ and ئ. Both are ordinary Urdu letters
# (کوئی, ہوئے), and listing them would discard a large share of correct Urdu.
ARABIC_ONLY_FORMS = frozenset("يكةى")


def has_arabic_only_form(text: str) -> bool:
    return bool(ARABIC_ONLY_FORMS & set(text))


def strip_marks(text: str) -> str:
    return URDU_MARKS.sub("", text)


def is_urdu_word(token: str) -> bool:
    """True when the token is a plain Urdu word: Urdu letters only, no Latin, no digits."""
    core = strip_marks(token).strip("۔،؛؟.,;:!?()[]{}\"'«»”“’‘-–—")
    if len(core) < 2:
        return False
    if LATIN.search(core) or DIGITS.search(core):
        return False
    return all(URDU_CHARS.match(ch) for ch in core)


def urdu_share(text: str) -> float:
    """Share of non-space characters that are Urdu. Used to reject mixed-script lines."""
    chars = [ch for ch in strip_marks(text) if not ch.isspace()]
    if not chars:
        return 0.0
    return sum(1 for ch in chars if URDU_CHARS.match(ch)) / len(chars)


def sentences(text: str) -> list[str]:
    return [s.strip() for s in SENT_END.split(text) if s.strip()]


def words(text: str) -> list[str]:
    return [w for w in WORD_SPLIT.split(text) if w]


def usable_sentence(text: str, *, min_words: int = 8, max_words: int = 60) -> bool:
    """A sentence is usable as a task source when it is clean, Urdu, and sized.

    The caps matter: a 3-word sentence gives a cloze item with no context, and a
    200-word line in a Wikipedia dump is nearly always a broken table row.
    """
    toks = words(text)
    if not (min_words <= len(toks) <= max_words):
        return False
    if urdu_share(text) < 0.80:
        return False
    if any(ch in text for ch in "{}|=<>[]"):  # wiki markup leftovers
        return False
    return sum(1 for t in toks if is_urdu_word(t)) >= max(5, len(toks) // 2)


def maskable_indices(toks: list[str]) -> list[int]:
    """Positions whose word is a content word and unique in the sentence.

    Uniqueness is required: masking a word that occurs twice makes the answer
    recoverable by copying, which would inflate any score measured on the item.
    """
    counts: dict[str, int] = {}
    for t in toks:
        counts[strip_marks(t)] = counts.get(strip_marks(t), 0) + 1
    out = []
    for i, t in enumerate(toks):
        if not is_urdu_word(t):
            continue
        if strip_marks(t) in FUNCTION_WORDS:
            continue
        if counts[strip_marks(t)] != 1:
            continue
        if len(strip_marks(t)) < 3:
            continue
        if has_arabic_only_form(t):
            continue
        out.append(i)
    return out


WINDOW = 8


def window_keys(text: str) -> set[str]:
    """Every WINDOW-word window of a text.

    Used to decide whether two differently segmented extracts of the same
    Wikipedia contain the same passage. Sentence-level comparison fails for that:
    the text matches, the boundaries do not.
    """
    toks = words(text)
    if len(toks) < WINDOW:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i : i + WINDOW]) for i in range(len(toks) - WINDOW + 1)}

