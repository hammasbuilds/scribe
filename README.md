# Scribe

An Urdu writing and comprehension desk, and `scribe-100k` — the dataset it is built on: **100,000
natively-sourced Urdu task items with no machine translation and no language model anywhere in the
label path.**

Product 3 of three in [`THREE-PRODUCTS-PLAN.md`](../THREE-PRODUCTS-PLAN.md), built first because it is
the only one of the three that reaches 100,000 verified rows without a GPU.

## Why this dataset exists

| Resource | Size | Provenance |
|---|---|---|
| Alif Urdu-Instruct | 51,686 | 7 tasks |
| Instruction-Tuned Urdu LLMs (LREC 2026) | 432,759 | translation-heavy; no evidence of open release |
| UrduMMLU | evaluation only | first natively-written broad Urdu benchmark |
| **scribe-100k** (ours) | **100,000** | built by us from Urdu sources; every row machine-checked |

Open Urdu instruction data is mostly translated from English, which carries English task structure
and English-shaped errors into Urdu. We built this one from Urdu sources instead — Dakshina's human
romanisations and Urdu Wikipedia.

## What is in it

| Task | Rows | Input | Gold answer | Source |
|---|---|---|---|---|
| `cloze` | 32,190 | a human sentence with one content word masked | the masked word | Urdu Wikipedia |
| `correction` | 25,000 | a sentence carrying one letter or spacing difference | the sentence as written | Urdu Wikipedia |
| `headline` | 15,810 | a section of an article | the section's own heading | full Urdu Wikipedia dump |
| `translit_ur2roman` | 15,000 | an Urdu word | its most-attested human romanisation | Dakshina lexicon |
| `translit_roman2ur` | 12,000 | a Roman Urdu word | the Urdu word, mutual-best pairing only | Dakshina lexicon |

- **100,000 rows, 87,996 source documents, 100,000 distinct prompts.**
- Splits **90,016 / 7,008 / 2,976** (train/val/test), assigned by hashing the *source document*, so no
  sentence contributes to two splits. Verified: 0 documents in two splits.
- `target_human_share 1.0` — every gold answer is text from the sources. Per-row provenance fields
  (`target_human`, `input_human`) let a consumer filter on it.
- Plus `eval_dakshina_sentences.jsonl`: 15,436 **evaluation-only** sentence-level transliteration rows,
  kept out of the splits because they are Dakshina's own held-out set.
- Licence **CC BY-SA 4.0**, inherited from its sources.

Rows live at `C:\d drive data\scribe-100k\v1` and are not in git. The code, schema, stats and defect
report are.

## How it was built

**We built scribe-100k** — the five tasks, the item construction, the checks and the splits are ours,
and the whole corpus can be rebuilt from this repository with one command.

It is built on two corpora, named here because their licences ask for attribution:

| Built on | What it provides | Licence |
|---|---|---|
| **Dakshina** (Google Research) | Urdu words with the romanisations people typed for them | CC BY-SA 4.0 |
| **Urdu Wikipedia** (`urwiki` dump, 2026-10-01, and the Dakshina native-script extract) | Urdu sentences and article sections written by contributors | CC BY-SA 4.0 |

Every row names its own source, document and licence.

## What it is for

- **Research:** the Urdu language work in this portfolio — a native-source alternative to Urdu
  instruction sets built by translating English.
- **Product:** the **Scribe** desk in this repository reads it directly; `python -m scribe.app`.
- **Also usable for:** Urdu transliteration in both directions, spelling and spacing correction,
  cloze/comprehension, and section-heading prediction.

## Citing it

If you use scribe-100k, or anything derived from it, cite this repository:

```bibtex
@misc{scribe100k,
  author = {Muhammad Hammas},
  title  = {scribe-100k: an Urdu task corpus from Dakshina and Urdu Wikipedia},
  year   = {2026},
  url    = {https://github.com/hammasbuilds/scribe}
}
```

Share-alike applies: CC BY-SA 4.0 comes with the sources, so attribution to Dakshina and Urdu
Wikipedia travels with it too.

## Input → Output

```
[cloze]
IN : خوش قسمتی سے پاکستان کو قائد اعظم اور ان کے دست راست نواب زادہ لیاقت علی خاں کی قیادت کا
     موقع تھا ... گورنر جنرل کو 1947 کے انڈین انڈپینڈنس ایکٹ کے ____ وسیع اختیارات حاصل تھے۔
OUT: تحت

[correction]  error_type: confusion:ی->ے
IN : لکھنئو میں انہوں نے مشہور حکیم علی حسین سے طب کی تعلےم حاصل کی اور بھوپال میں طبابت کی۔
OUT: لکھنئو میں انہوں نے مشہور حکیم علی حسین سے طب کی تعلیم حاصل کی اور بھوپال میں طبابت کی۔

[correction]  error_type: transpose
IN : عثمانی شام (Ottoman Syria) سے مراد سلطنت عثمنایہ کا سرزمین شام کا حصہ ہے ...
OUT: عثمانی شام (Ottoman Syria) سے مراد سلطنت عثمانیہ کا سرزمین شام کا حصہ ہے ...

[headline]
IN : الجزائر کا زیادہ تر ساحلی علاقہ پتھریلا اور چٹانی ہے اور کہیں کہیں پہاڑ بھی ملتے ہیں۔ تاہم
     یہاں کئی قدرتی بندرگاہیں بھی موجود ہیں۔ ساحل سے لے کر اطلس التلی تک کا علاقہ زرخیز ہے ...
OUT: جغرافیہ

[translit_ur2roman]
IN : ہسپتال        OUT: haspataal     accepted_outputs: haspataal, hospital, haspatal, ...
[translit_roman2ur]
IN : khazdaar      OUT: خضدار
```

## The desk

```
uv run python -m scribe.app        # the Urdu desk on http://127.0.0.1:8810
uv run python -m scribe.evaluate   # the no-model baseline on the test split
```

Five tools over the corpus, each answering with its own evidence: **correct** a sentence, **fill a
blank**, **Urdu to Roman**, **Roman to Urdu**, and **suggest a heading** for a passage, plus a BM25
search over the sentences behind the dataset.

**Method, and why it is this.** The corpus is bounded and stable - 100,000 rows that do not change
between builds - which is exactly the case where cache-augmented generation beats retrieval: every
index is built once at startup (12 s for 90,016 rows) and no request touches disk. Retrieval still
earns its place where similarity decides the answer: cloze context matching and heading suggestion run
BM25 over the corpus. A cross-encoder reranker is the documented next step and needs a model, so the
slot is explicit in `engine.rerank_slot` rather than implied.

### Measured on the held-out test split, with no model

| Task | Exact match | Second metric | Note |
|---|---|---|---|
| `correction` | **0.825** | 0.878 located the broken token | errors of the kinds the corpus contains |
| `translit_roman2ur` | **0.352** | — | character rules snapped to a corpus word |
| `translit_ur2roman` | **0.230** | 0.313 any attested romanisation | lexicon covers 2.8% of test words *by design* |
| `cloze` | **0.138** | 0.243 top-5 | 54% answered from the context index |
| `headline` | **0.128** | 0.206 token F1 | answered by retrieval 100% of the time |

These are the numbers a trained arm has to beat, measured the same way. Transliteration coverage is
near zero on purpose: the dataset splits word-level transliteration **by word**, so a test word cannot
be in the train lexicon, and a lookup engine cannot win that. Saying so is the point of reporting
coverage next to accuracy.

Roman-to-Urdu scored exactly **0.000** on the first run, because the engine returned unknown words
unchanged. Transducing every plausible reading of a Roman word and keeping the one that matches a real
corpus word took it to 0.352 (کتاب اور قلم, حکومت نے فیصلہ کیا, عورت اور آدمی).

Correction went **0.750 → 0.825** for one reason: the sentence score was a *sum* over tokens, so a
longer sentence won on volume and a split word was never worth joining. Scoring per token fixed it.
Cloze went **0.098 → 0.138** by consulting one-sided context when no left+right match exists, and by
refusing to answer with a function word — the frequency fallback used to answer "میں", the commonest
word in the corpus, to any question it could not place.

## Build it

```bash
uv run python -m scribe.build  --target 100000 --out "C:/d drive data/scribe-100k/v1"
uv run python -m scribe.verify --dir "C:/d drive data/scribe-100k/v1"
uv run pytest -q
```

`build` writes `train/val/test.jsonl`, the eval file, and `stats.json`. `verify` re-reads what was
written, re-derives every check from the row itself, ignores the builder's `verified` flag, and exits
non-zero on any failure. 26 tests, all passing.

The `headline` task needs the full dump (`urwiki-latest-pages-articles.xml.bz2`, 476,815,422 bytes,
`bzip2 -t` verified). A single connection to it was throttled to ~21 KB/s here; the same file served
ranged chunks at ~585 KB/s, so `scripts/range_fetch.sh` fetches it 8 MB at a time.

## Problems hit while building this

Nine defects, each found by a measurement rather than by inspection of the code, all written up with
their numbers in [`docs/DEFECTS.md`](docs/DEFECTS.md). The ones worth naming here:

1. **Dakshina's Urdu sentence pairs are its own eval set** — 9,631 of 9,759 sentences are in the dev and
   test files, and the 176k word alignments come from those same sentences. Training on them would have
   contaminated a number this portfolio already reports.
2. **Two "disjoint" sentence pools were not disjoint.** Even corpus lines for cloze, odd for correction —
   and 114 sentences appear on both, so they became a cloze item *and* its own correction target.
   Ownership is now decided by the sentence's content hash.
3. **Generated census lines produce identical prompts with different gold answers.** Exact-sentence
   deduplication cannot see them, because each sentence is unique; a template frame can.
4. **The two Wikipedia extracts overlap.** The sentence tasks come from a filtered extract, the section
   task from the full dump, so 1,977 sentences sat inside a section body — 380 of them in a different
   split. It took four fixes to reach zero, because each one was applied at the wrong granularity or in
   the wrong pass: sentence-level exclusion (111 left), first-window-only (14 left), all windows but not
   in the top-up pass (6 left), then windows on both sides (0). The check now lives in `verify`.

A fourth is mine rather than the data's: the first orthography filter listed ؤ and ئ as Arabic-only
forms. They are ordinary Urdu letters, and the filter would have thrown away a large share of correct
Urdu. Reverted the same day.

## What it does NOT do

- Every gold answer comes from Dakshina or Urdu Wikipedia. No translation is involved.
- It has no sentence-level transliteration *training* data, because the only human source for that is
  Dakshina's held-out set.
- The `headline` task is section-heading prediction, not article-title prediction: an Urdu Wikipedia
  lead almost always opens with the title verbatim, so title prediction would be a copy task. 10,218
  candidates were rejected for having the heading verbatim in the body, and 5,314 more for having most
  of the heading's content words in it.
- It cannot know the vowels of a word it has never seen. کتاب is *kitaab*, and a character rule
  guesses *katab*; a two-letter prefix learned from the lexicon was tried and measured as a regression
  (0.230 → 0.153), because short prefixes conflate unrelated words. Those answers are labelled
  "character rules" with zero confidence rather than dressed up.
- It cannot un-split a word whose fragments are themselves common words. "اس کو ل" could be three real
  words or a broken اسکول, and nothing short of understanding the sentence can tell.
- The desk does not run a model. Everything above is lookup, rules, a context index and BM25; no model
  has been trained on this corpus yet, because the GPU belongs to other work.
- It does not export PDF. An Urdu PDF needs an embedded Nastaliq or Naskh font and right-to-left shaping
  with contextual letter forms - a typesetting job, not a text container - so the desk gives you text
  and leaves PDF to the two Latin-script products.
