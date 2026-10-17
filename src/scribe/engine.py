"""The Urdu desk's engines: five tasks, no model, no GPU.

Methodology, and why it is this and not something heavier
---------------------------------------------------------
The corpus is **bounded and stable** - 100,000 rows that do not change between
builds - which is the case where cache-augmented generation beats retrieval:
load the knowledge once, answer with no retrieval hop. So every index here is
built at startup and held in memory (``Corpus``), and the per-request path does
no I/O at all.

Retrieval still earns its place where the answer depends on *which* passage is
most similar: cloze context matching and headline suggestion both run BM25 over
the corpus. Hybrid lexical+dense retrieval and a cross-encoder reranker are the
documented next step (a 2026 benchmark put hybrid+rerank at 0.816 Recall@5
against 0.644 for BM25 alone), but both need a model, and the GPU is reserved -
so the reranker slot is left explicit in ``rerank_slot`` rather than pretended.

Every engine returns its evidence: which attested variants, which neighbour
sentence, which rule. A prediction with no evidence is a guess, and the UI shows
the difference.
"""

from __future__ import annotations

import collections
import itertools
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

from .schema import nfc
from .urdu import (
    CONFUSIONS_HINT,
    is_urdu_word,
    strip_marks,
    words,
)

MASK = "____"
DEFAULT_DATA = Path("C:/d drive data/scribe-100k/v2")


@dataclass
class Answer:
    """One engine result. ``evidence`` is what the UI shows under the answer."""

    task: str
    output: str
    method: str
    confidence: float
    evidence: list[dict] = field(default_factory=list)
    alternatives: list[str] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "output": self.output,
            "method": self.method,
            "confidence": round(self.confidence, 3),
            "evidence": self.evidence,
            "alternatives": self.alternatives,
            "note": self.note,
        }


class BM25:
    """Okapi BM25 over tokenised documents. Stdlib only, built once, queried hot."""

    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = docs
        self.len = [len(d) for d in docs]
        self.avg = (sum(self.len) / len(self.len)) if docs else 0.0
        self.postings: dict[str, list[tuple[int, int]]] = collections.defaultdict(list)
        for i, doc in enumerate(docs):
            for term, tf in collections.Counter(doc).items():
                self.postings[term].append((i, tf))
        self.n = len(docs)

    def idf(self, term: str) -> float:
        df = len(self.postings.get(term, ()))
        if df == 0:
            return 0.0
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def search(self, query: list[str], limit: int = 5) -> list[tuple[int, float]]:
        scores: dict[int, float] = collections.defaultdict(float)
        for term in query:
            idf = self.idf(term)
            if idf == 0.0:
                continue
            for i, tf in self.postings[term]:
                denom = tf + self.k1 * (1 - self.b + self.b * self.len[i] / (self.avg or 1))
                scores[i] += idf * tf * (self.k1 + 1) / denom
        return sorted(scores.items(), key=lambda kv: -kv[1])[:limit]


def rerank_slot(candidates: list[tuple[int, float]]) -> list[tuple[int, float]]:
    """Where a cross-encoder reranker goes when there is a GPU to run it on.

    Today it is the identity function, and the UI says so rather than implying a
    reranking step that is not happening.
    """
    return candidates


class Corpus:
    """Every index the desk needs, built once from the dataset's train split."""

    def __init__(self, data_dir: Path = DEFAULT_DATA, *, split: str = "train", limit: int | None = None):
        self.data_dir = Path(data_dir)
        self.rows: list[dict] = []
        path = self.data_dir / f"{split}.jsonl"
        if path.exists():
            with path.open(encoding="utf-8") as fh:
                for i, line in enumerate(fh):
                    if limit is not None and i >= limit:
                        break
                    if line.strip():
                        self.rows.append(json.loads(line))

        # --- transliteration tables -------------------------------------------------
        self.ur2roman: dict[str, list[tuple[str, int]]] = {}
        self.roman2ur: dict[str, str] = {}
        for r in self.rows:
            if r["task"] == "translit_ur2roman":
                variants = [
                    (v["roman"], v["attestations"])
                    for v in r.get("meta", {}).get("accepted_outputs", [])
                ] or [(r["output"], 1)]
                self.ur2roman[r["input"]] = variants
            elif r["task"] == "translit_roman2ur":
                self.roman2ur[r["input"]] = r["output"]

        # --- sentence material ------------------------------------------------------
        self.sentences: list[str] = []
        for r in self.rows:
            if r["task"] == "cloze":
                self.sentences.append(r["input"].replace(MASK, r["output"], 1))
            elif r["task"] == "correction":
                self.sentences.append(r["output"])

        self.unigram: collections.Counter = collections.Counter()
        self.bigram: collections.Counter = collections.Counter()
        for s in self.sentences:
            toks = words(s)
            self.unigram.update(strip_marks(t) for t in toks)
            self.bigram.update(zip((strip_marks(t) for t in toks), (strip_marks(t) for t in toks[1:])))
        self.vocab = {w for w, c in self.unigram.items() if c >= 2}
        # Indexed by first letter, so snapping a guess to a real word does not scan
        # all 29k words on every request.
        self.vocab_by_first: dict[str, list[str]] = collections.defaultdict(list)
        for w in self.vocab:
            if w:
                self.vocab_by_first[w[0]].append(w)

        # Context index for cloze: (left word, right word) -> answers seen there.
        self.context: dict[tuple[str, str], collections.Counter] = collections.defaultdict(
            collections.Counter
        )
        for s in self.sentences:
            toks = [strip_marks(t) for t in words(s)]
            for i in range(1, len(toks) - 1):
                self.context[(toks[i - 1], toks[i + 1])][toks[i]] += 1

        # --- headline material ------------------------------------------------------
        self.headline_bodies: list[str] = []
        self.headline_heads: list[str] = []
        for r in self.rows:
            if r["task"] == "headline":
                self.headline_bodies.append(r["input"])
                self.headline_heads.append(r["output"])
        self.headline_bm25 = BM25([[strip_marks(t) for t in words(b)] for b in self.headline_bodies])
        self.sentence_bm25 = BM25([[strip_marks(t) for t in words(s)] for s in self.sentences])
        self.heading_freq = collections.Counter(self.headline_heads)

    @property
    def stats(self) -> dict:
        return {
            "rows_loaded": len(self.rows),
            "transliteration_entries": len(self.ur2roman),
            "reverse_entries": len(self.roman2ur),
            "sentences": len(self.sentences),
            "vocabulary": len(self.vocab),
            "context_pairs": len(self.context),
            "headline_examples": len(self.headline_bodies),
            "data_dir": str(self.data_dir),
        }


# --------------------------------------------------------------------------------------
# Engines
# --------------------------------------------------------------------------------------


def transliterate_to_roman(text: str, corpus: Corpus) -> Answer:
    """Word-by-word, lexicon first, then a character transducer for unseen words."""
    out, evidence, covered = [], [], 0
    toks = words(nfc(text))
    for tok in toks:
        key = nfc(tok)
        variants = corpus.ur2roman.get(key) or corpus.ur2roman.get(strip_marks(key))
        if variants:
            out.append(variants[0][0])
            covered += 1
            evidence.append(
                {
                    "word": tok,
                    "source": "human romanisations in the corpus",
                    "variants": [{"roman": v, "attestations": c} for v, c in variants[:5]],
                }
            )
        else:
            guess = _char_transliterate(key)
            out.append(guess)
            evidence.append({"word": tok, "source": "character rules (word not in corpus)", "variants": []})
    conf = covered / len(toks) if toks else 0.0
    return Answer(
        task="translit_ur2roman",
        output=" ".join(out),
        method="lexicon lookup, character transducer for unseen words",
        confidence=conf,
        evidence=evidence,
        note=f"{covered} of {len(toks)} words came from human romanisations.",
    )


# Character map for words the corpus has never seen. Longest match first.
_CHAR_MAP = [
    ("ایں", "ain"), ("اؤ", "ao"), ("آ", "aa"), ("ای", "ai"), ("او", "au"),
    ("بھ", "bh"), ("پھ", "ph"), ("تھ", "th"), ("ٹھ", "th"), ("جھ", "jh"),
    ("چھ", "chh"), ("دھ", "dh"), ("ڈھ", "dh"), ("کھ", "kh"), ("گھ", "gh"),
    ("ا", "a"), ("ب", "b"), ("پ", "p"), ("ت", "t"), ("ٹ", "t"), ("ث", "s"),
    ("ج", "j"), ("چ", "ch"), ("ح", "h"), ("خ", "kh"), ("د", "d"), ("ڈ", "d"),
    ("ذ", "z"), ("ر", "r"), ("ڑ", "r"), ("ز", "z"), ("ژ", "zh"), ("س", "s"),
    ("ش", "sh"), ("ص", "s"), ("ض", "z"), ("ط", "t"), ("ظ", "z"), ("ع", "a"),
    ("غ", "gh"), ("ف", "f"), ("ق", "q"), ("ک", "k"), ("گ", "g"), ("ل", "l"),
    ("م", "m"), ("ن", "n"), ("ں", "n"), ("و", "o"), ("ہ", "h"), ("ھ", "h"),
    ("ء", ""), ("ی", "i"), ("ے", "e"), ("ؤ", "o"), ("ئ", "y"),
]


_VOWELS = set("aeiou")


def _char_transliterate(word: str) -> str:
    """Map an unseen Urdu word character by character, then restore short vowels.

    Urdu does not write short vowels, so a straight character map returns
    "hsptal" for ہسپتال, which no Roman Urdu reader would type. Two insertions
    recover the common cases: an "a" inside a word-initial consonant cluster
    (کتاب -> kataab) and inside a word-final one (ڈاکٹر -> daaktar).

    A transducer cannot know vowel *quality* - kitaab, not kataab - and guessing
    it would need a model. So this stays a labelled fallback: the engine marks
    these words as rule-derived, reports the share, and the UI shows it. The real
    fix for them is corpus coverage, not a cleverer rule.
    """
    s, out = strip_marks(word), []
    i = 0
    while i < len(s):
        for src, dst in _CHAR_MAP:
            if s.startswith(src, i):
                out.append(dst)
                i += len(src)
                break
        else:
            out.append(s[i])
            i += 1
    text = "".join(out)

    def is_cons(ch: str) -> bool:
        return ch.isalpha() and ch not in _VOWELS

    chars = list(text)
    # word-initial cluster: C C -> C a C
    if len(chars) >= 3 and is_cons(chars[0]) and is_cons(chars[1]):
        chars.insert(1, "a")
    # word-final cluster: ... C C -> ... C a C
    if len(chars) >= 3 and is_cons(chars[-1]) and is_cons(chars[-2]):
        chars.insert(len(chars) - 1, "a")
    return "".join(chars)


# Roman -> Urdu character rules, longest match first. Digraphs before letters, or
# "sh" becomes س+ہ.
_ROMAN_MAP = [
    ("chh", "چھ"), ("sh", "ش"), ("ch", "چ"), ("kh", "خ"), ("gh", "غ"), ("th", "تھ"),
    ("ph", "پھ"), ("bh", "بھ"), ("dh", "دھ"), ("jh", "جھ"), ("rh", "ڑھ"), ("ng", "نگ"),
    ("aa", "ا"), ("ee", "ی"), ("ii", "ی"), ("oo", "و"), ("uu", "و"), ("ai", "ے"),
    ("ay", "ے"), ("au", "و"), ("ou", "و"), ("ea", "ی"),
    ("a", "ا"), ("b", "ب"), ("c", "ک"), ("d", "د"), ("e", "ے"), ("f", "ف"),
    ("g", "گ"), ("h", "ہ"), ("i", "ی"), ("j", "ج"), ("k", "ک"), ("l", "ل"),
    ("m", "م"), ("n", "ن"), ("o", "و"), ("p", "پ"), ("q", "ق"), ("r", "ر"),
    ("s", "س"), ("t", "ت"), ("u", "و"), ("v", "و"), ("w", "و"), ("x", "کس"),
    ("y", "ی"), ("z", "ز"),
]


# Word-initial vowels are written with an alef in Urdu: "aur" is اور, not ور.
# Without this the digraph rules consume "au" and the alef is simply lost, which
# is what turned اور into ور.
_INITIAL_VOWEL = [
    ("aa", "آ"), ("ai", "ای"), ("au", "او"), ("ee", "ای"), ("oo", "او"),
    ("a", "ا"), ("e", "ای"), ("i", "ا"), ("o", "او"), ("u", "ا"),
]


def _roman_to_urdu_chars(word: str) -> str:
    s, out = word.lower(), []
    i = 0
    for src, dst in _INITIAL_VOWEL:
        if s.startswith(src):
            out.append(dst)
            i = len(src)
            break
    while i < len(s):
        for src, dst in _ROMAN_MAP:
            if s.startswith(src, i):
                out.append(dst)
                i += len(src)
                break
        else:
            i += 1
    return "".join(out)


# Characters that are orthographic variants of each other rather than different
# letters. Substituting inside a group is half the cost of substituting across
# groups, so آدمی beats ادبی as the snap target for "admi": both are one plain
# edit away from the guess ادمی, but only one of them is a spelling variant.
_CONFUSABLE_GROUPS = ("اآ", "یےئ", "ہھ", "کك", "نں", "سصث", "زذضظ", "تط", "وؤ", "حہ")
_CONFUSABLE: dict[str, int] = {}
for _gi, _group in enumerate(_CONFUSABLE_GROUPS):
    for _ch in _group:
        _CONFUSABLE.setdefault(_ch, _gi)


def _sub_cost(a: str, b: str) -> float:
    if a == b:
        return 0.0
    ga, gb = _CONFUSABLE.get(a), _CONFUSABLE.get(b)
    if ga is not None and ga == gb:
        return 0.5
    return 1.0


def _edit_distance(a: str, b: str, cap: float = 3) -> float:
    """Levenshtein with a cheaper substitution for orthographic variants."""
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = [float(i) for i in range(len(b) + 1)]
    for i, ca in enumerate(a, 1):
        cur = [float(i)] + [0.0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + _sub_cost(ca, cb))
        if min(cur) > cap:
            return cap + 1
        prev = cur
    return prev[-1]


def _snap_to_vocabulary(guess: str, corpus: Corpus, cap: int | None = None) -> tuple[str, int] | None:
    """Pull a transduced guess onto the nearest real Urdu word in the corpus.

    The character rules cannot know vowel quality, so the raw guess is often a
    near-miss of a real word. The corpus has 29,460 words attested twice or more;
    snapping to the closest one within an edit distance of 2, preferring the more
    frequent word, turns a plausible-looking non-word into a real one.
    """
    if cap is None:
        # Deliberately tight. A cap of 3 did find حکومت-length words, but it also
        # snapped ہوکومات to ہندومت and ادمی to ادبی - a *wrong real word*, which
        # reads as confident and correct, while a non-word at least looks unsure.
        cap = 2
    # An exact hit needs no snapping at all.
    if guess in corpus.vocab:
        return guess, corpus.unigram.get(guess, 0)
    # Buckets for the first letter *and* its orthographic variants. Looking only in
    # the guess's own bucket hid آدمی from the guess ادمی, because آ and ا are
    # different keys - the exact case the confusable cost exists to handle.
    first = guess[:1]
    group = _CONFUSABLE_GROUPS[_CONFUSABLE[first]] if first in _CONFUSABLE else first
    pool: list[str] = []
    for letter in group:
        pool.extend(corpus.vocab_by_first.get(letter, ()))

    best: tuple[float, int, str] | None = None
    for cand in pool:
        if abs(len(cand) - len(guess)) > cap:
            continue
        d = _edit_distance(guess, cand, cap)
        if d > cap:
            continue
        freq = corpus.unigram.get(cand, 0)
        key = (d, -freq, cand)
        if best is None or key < (best[0], best[1], best[2]):
            best = key
    if best is None:
        return None
    return best[2], int(-best[1])


# Roman letters with more than one honest Urdu reading. "kh" is خ in خان and کھ
# in کھانا, and nothing in the spelling says which. Rather than pick one and be
# wrong half the time, the engine transduces every reading and lets the corpus
# vocabulary decide which one is a word.
_AMBIGUOUS = {
    "kh": ("خ", "کھ"),
    "gh": ("غ", "گھ"),
    "h": ("ہ", "ح"),
    "s": ("س", "ص"),
    "t": ("ت", "ط"),
    "z": ("ز", "ذ", "ض"),
    "q": ("ق", "ک"),
}
_MAX_VARIANTS = 24


def _roman_variants(word: str) -> list[str]:
    """Every plausible Urdu spelling of a Roman word, capped so this stays cheap."""
    s = word.lower()
    variants = [""]
    i = 0
    initial = ""
    for src, dst in _INITIAL_VOWEL:
        if s.startswith(src):
            initial, i = dst, len(src)
            break
    while i < len(s):
        for src, dst in _ROMAN_MAP:
            if s.startswith(src, i):
                options = _AMBIGUOUS.get(src, (dst,))
                if len(variants) * len(options) > _MAX_VARIANTS:
                    options = options[:1]
                variants = [v + o for v in variants for o in options]
                i += len(src)
                break
        else:
            i += 1
    return [initial + v for v in variants]


def transliterate_to_urdu(text: str, corpus: Corpus) -> Answer:
    out, evidence, covered, snapped_count = [], [], 0, 0
    toks = words(text.strip())
    for tok in toks:
        key = tok.lower().strip(".,;:!?\"'()")
        urdu = corpus.roman2ur.get(key)
        if urdu:
            out.append(urdu)
            covered += 1
            evidence.append({"word": tok, "source": "mutual-best pairing in the corpus", "urdu": urdu})
            continue
        variants = _roman_variants(key)
        guess = variants[0] if variants else ""
        picked: tuple[float, int, str, str] | None = None
        for v in variants:
            if not v:
                continue
            if v in corpus.vocab:
                cand = (0.0, -corpus.unigram.get(v, 0), v, v)
            else:
                snapped_v = _snap_to_vocabulary(v, corpus)
                if snapped_v is None:
                    continue
                word, freq = snapped_v
                cand = (_edit_distance(v, word, 3), -freq, word, v)
            if picked is None or cand < picked:
                picked = cand
        if picked is not None:
            dist, negfreq, word, from_variant = picked
            out.append(word)
            # A snapped word is a real corpus word, just not a lexicon pairing, so it
            # counts partially. Counting it as zero showed 0% confidence next to a
            # perfectly correct answer, which is worse than being approximately right.
            snapped_count += 1
            evidence.append({
                "word": tok,
                "source": (
                    f"character rules ({len(variants)} readings tried), "
                    f"matched a corpus word seen {-negfreq} times"
                ),
                "urdu": word,
                "raw_guess": from_variant,
                "edit_distance": dist,
            })
        elif guess:
            out.append(guess)
            evidence.append({"word": tok, "source": "character rules (no close corpus word)", "urdu": guess})
        else:
            out.append(tok)
            evidence.append({"word": tok, "source": "could not transliterate - left as typed", "urdu": ""})
    return Answer(
        task="translit_roman2ur",
        output=" ".join(out),
        method="reverse lexicon, then character rules snapped to a corpus word",
        confidence=((covered + 0.6 * snapped_count) / len(toks)) if toks else 0.0,
        evidence=evidence,
        note=(
            f"{covered} of {len(toks)} words came from a human pairing; "
            f"{snapped_count} were derived by rule and matched to a word in the corpus."
        ),
    )


def fill_blank(sentence: str, corpus: Corpus) -> Answer:
    """Cloze. Context index first, BM25 neighbour second, frequency last."""
    s = nfc(sentence)
    if MASK not in s:
        return Answer("cloze", "", "no blank found", 0.0, note=f"Put {MASK} where the word should go.")
    toks = words(s)
    i = next(j for j, t in enumerate(toks) if MASK in t)
    left = strip_marks(toks[i - 1]) if i > 0 else ""
    right = strip_marks(toks[i + 1]) if i + 1 < len(toks) else ""

    cands = corpus.context.get((left, right))
    if cands:
        best, count = cands.most_common(1)[0]
        total = sum(cands.values())
        return Answer(
            task="cloze",
            output=best,
            method="context index: the same words either side of the blank",
            confidence=count / total,
            evidence=[{"left": left, "right": right, "seen": total,
                       "candidates": [{"word": w, "count": c} for w, c in cands.most_common(5)]}],
            alternatives=[w for w, _ in cands.most_common(5)[1:]],
        )

    query = [strip_marks(t) for t in toks if MASK not in t]
    hits = rerank_slot(corpus.sentence_bm25.search(query, limit=5))
    for idx, score in hits:
        neighbour = corpus.sentences[idx]
        ntoks = words(neighbour)
        # Take the word in the neighbour that sits between the same two words.
        for j in range(1, len(ntoks) - 1):
            if strip_marks(ntoks[j - 1]) == left and strip_marks(ntoks[j + 1]) == right:
                return Answer(
                    task="cloze",
                    output=ntoks[j],
                    method="BM25 neighbour sentence, same surrounding words",
                    confidence=min(0.6, score / 40),
                    evidence=[{"neighbour": neighbour, "bm25": round(score, 2)}],
                )

    pool = [(w, c) for w, c in corpus.unigram.items() if c > 5 and is_urdu_word(w) and len(w) >= 3]
    pool.sort(key=lambda wc: -wc[1])
    best = pool[0][0] if pool else ""
    return Answer(
        task="cloze",
        output=best,
        method="corpus frequency (no context match found)",
        confidence=0.05,
        evidence=[{"left": left, "right": right, "seen": 0}],
        alternatives=[w for w, _ in pool[1:6]],
        note="Neither the context index nor a neighbour sentence matched, so this is a frequency guess.",
    )


_SPLIT_FIX_RE = re.compile(r"\s+")


def correct(sentence: str, corpus: Corpus) -> Answer:
    """Single-error correction: generate candidates by rule, score by corpus evidence."""
    s = nfc(sentence)
    toks = words(s)
    best: tuple[float, str, dict] | None = None

    def score_tokens(cand: list[str]) -> float:
        uni = sum(math.log1p(corpus.unigram.get(strip_marks(t), 0)) for t in cand)
        bi = sum(
            math.log1p(corpus.bigram.get((strip_marks(a), strip_marks(b)), 0)) * 2
            for a, b in itertools.pairwise(cand)
        )
        return uni + bi

    baseline = score_tokens(toks)

    # 1. join a word split in two
    for i in range(len(toks) - 1):
        cand = toks[:i] + [toks[i] + toks[i + 1]] + toks[i + 2 :]
        if strip_marks(cand[i]) in corpus.vocab:
            sc = score_tokens(cand)
            if best is None or sc > best[0]:
                best = (sc, " ".join(cand), {"rule": "joined a split word", "from": f"{toks[i]} {toks[i+1]}", "to": cand[i]})

    # 2. undo a letter confusion
    for i, tok in enumerate(toks):
        for src, dst in CONFUSIONS_HINT:
            if src not in tok:
                continue
            for pos in [m.start() for m in re.finditer(re.escape(src), tok)]:
                alt = tok[:pos] + dst + tok[pos + len(src) :]
                if strip_marks(alt) not in corpus.vocab or alt == tok:
                    continue
                cand = toks[:i] + [alt] + toks[i + 1 :]
                sc = score_tokens(cand)
                if best is None or sc > best[0]:
                    best = (sc, " ".join(cand), {"rule": f"letter confusion {src} -> {dst}", "from": tok, "to": alt})

    # 3. undo a transposition
    for i, tok in enumerate(toks):
        core = strip_marks(tok)
        for j in range(len(core) - 1):
            alt = core[:j] + core[j + 1] + core[j] + core[j + 2 :]
            if alt == core or alt not in corpus.vocab:
                continue
            cand = toks[:i] + [alt] + toks[i + 1 :]
            sc = score_tokens(cand)
            if best is None or sc > best[0]:
                best = (sc, " ".join(cand), {"rule": "undid a transposition", "from": tok, "to": alt})

    if best is None or best[0] <= baseline:
        return Answer(
            task="correction",
            output=s,
            method="no change: the sentence already scores best against the corpus",
            confidence=0.3,
            evidence=[{"baseline_score": round(baseline, 2)}],
            note="Nothing in the corpus suggests a single-edit improvement.",
        )
    sc, out, rule = best
    return Answer(
        task="correction",
        output=out,
        method="rule candidates scored by corpus unigram and bigram evidence",
        confidence=min(0.95, (sc - baseline) / 10),
        evidence=[rule | {"score": round(sc, 2), "baseline": round(baseline, 2)}],
    )


_STOP_FOR_HEADLINE = {"میں", "سے", "کے", "کی", "کا", "اور", "ہے", "ہیں", "کو", "پر", "جو", "یہ", "تھا"}


def suggest_heading(body: str, corpus: Corpus) -> Answer:
    """Heading for a passage: borrow from the most similar passage, or extract."""
    s = nfc(body)
    query = [strip_marks(t) for t in words(s)]
    hits = rerank_slot(corpus.headline_bm25.search(query, limit=5))
    evidence = []
    for idx, score in hits:
        evidence.append(
            {
                "heading": corpus.headline_heads[idx],
                "bm25": round(score, 2),
                "passage": corpus.headline_bodies[idx][:160] + "...",
            }
        )
    if hits and hits[0][1] > 12:
        idx, score = hits[0]
        return Answer(
            task="headline",
            output=corpus.headline_heads[idx],
            method="heading of the most similar passage in the corpus (BM25)",
            confidence=min(0.9, score / 60),
            evidence=evidence,
            alternatives=[corpus.headline_heads[i] for i, _ in hits[1:4]],
        )

    # Extractive fallback: the most distinctive content words in the passage.
    counts = collections.Counter(
        strip_marks(t) for t in words(s) if is_urdu_word(t) and strip_marks(t) not in _STOP_FOR_HEADLINE
    )
    scored = [(w, c * corpus.sentence_bm25.idf(w)) for w, c in counts.items() if len(w) >= 3]
    scored.sort(key=lambda wc: -wc[1])
    picked = [w for w, _ in scored[:3]]
    return Answer(
        task="headline",
        output=" ".join(picked),
        method="extractive: most distinctive words in the passage (TF-IDF)",
        confidence=0.25,
        evidence=evidence,
        note="No sufficiently similar passage in the corpus, so the heading is extracted from the text.",
    )


def search(query: str, corpus: Corpus, limit: int = 10) -> dict:
    """Corpus search, so a claim about the dataset can be checked in the UI."""
    q = [strip_marks(t) for t in words(nfc(query))]
    hits = corpus.sentence_bm25.search(q, limit=limit)
    return {
        "query": query,
        "hits": [{"sentence": corpus.sentences[i], "bm25": round(s, 2)} for i, s in hits],
        "searched": len(corpus.sentences),
    }
