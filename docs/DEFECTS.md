# scribe-100k — defect report on the generator

Built 2026-10-17. Dataset at `C:\d drive data\scribe-100k\v1` (rows are not in git).

A dataset's own defects belong beside the product's numbers, not behind them. Every item below was
found while building this corpus, and each is either fixed in the code or stated as a standing limit.
The clean verifier report at the end is only meaningful because of the sensitivity tests, so those are
described too.

## Fixed, with the measurement that found it

### 1. Dakshina's sentence-level Urdu romanisations are evaluation data, nearly all of them

The first builder read `ur.romanized.rejoined.tsv` (9,759 human sentence pairs) as training data. The
dev and test native files hold **9,631 of those same 9,759 sentences**, and
`ur.romanized.rejoined.aligned.tsv` (176,266 word pairs) is the word alignment *of those sentences*.
Training on any of it would have contaminated the held-out set the `urdunlp` package already reports a
transliteration number on.

**Fix:** sentence-level pairs moved to `eval_dakshina_sentences.jsonl` (15,436 rows, evaluation only,
never in the three splits). The one genuinely trainable file is the train lexicon.

**Visible as:** `dakshina_heldout_sentences: 9631` in `stats.json`.

### 2. Wikipedia boilerplate put identical items in different splits

Urdu Wikipedia's stub and maintenance notices recur verbatim; the worst appears **693 times**. With
line-numbered document ids, copies of one sentence received different ids, so the same cloze item
landed in train *and* in test.

**Fix:** sentences occurring more than 3 times are dropped as template text (183 sentences, 2,860
instances), and a sentence's `doc_id` is now a hash of its own content, so every copy shares one id and
therefore one split.

### 3. Even/odd line partitioning leaked sentences between tasks

Cloze took even corpus lines and correction odd ones, which looked disjoint and was not: a sentence
occurring on both an even and an odd line reached both builders. **114 sentences** became a cloze item
*and* a correction target — and the correction target is the very sentence the cloze item hides a word
from.

**Fix:** pool ownership is decided by the sentence's content hash (`pool_of`), so a duplicate occurrence
always lands in the same pool. Now 0.

### 4. Generated census lines produced identical prompts with different gold answers

Urdu Wikipedia carries families of generated sentences differing only in a number. Masking the number
yields the *same* prompt with a different correct answer; three such prompts were split across train,
val and test. Exact-sentence deduplication cannot see this, because each sentence is unique.

**Fix:** a template frame (first three + last three words) may contribute at most 3 sentences, and a
prompt may appear only once in the whole dataset. `distinct_inputs` is now exactly 100,000.

### 5. One romanisation, one Urdu word, wrong word

The reverse direction (Roman → Urdu) first required only that a romanisation map to a single Urdu word
*in this lexicon*. A hand read found `subah` → صوبہ, where a reader would answer صبح: صوبہ was simply
the only word in the sampled lexicon with that romanisation attested.

**Fix:** a reverse row needs the pairing to be **mutual best** — the romanisation maps to one Urdu word
*and* is that word's most-attested romanisation. Measured cost: 20,259 → 14,055 candidate pairs, which
is why the task caps are elastic (see below).

### 6. The forward direction had 13,416 rows sharing an input

Emitting one row per attested romanisation put the same Urdu word in up to a dozen rows with different
gold answers, so an exact-match score would mark a correct, attested romanisation wrong.

**Fix:** one row per Urdu word, gold = most attested, and every attested variant with its count in
`meta.accepted_outputs`, so an evaluation can score "most attested" and "any attested" separately.

### 7. The dataset disagreed with itself about Urdu orthography

The correction task treats Urdu kaf → Arabic kaf as an error to fix. Meanwhile cloze gold answers could
contain Arabic-only forms (ي, ك, ة, ى) from quoted Arabic — so the same substitution was an error in one
task and the right answer in another.

**Fix:** gold answers and transliteration inputs reject those four forms. ؤ and ئ are deliberately *not*
in that set: they are ordinary Urdu letters (کوئی, ہوئے), and excluding them would discard a large share
of correct Urdu. That mistake was made and reverted during this build.

### 8. Article-title summarisation would have been a copy task

The obvious summarisation pair is lead paragraph → article title. In Urdu Wikipedia a lead almost always
opens with the title verbatim ("فیصل آباد ایک شہر ہے..."), so a model scores well by extracting the first
noun phrase and nothing about summarisation is measured.

**Fix:** the task is **section heading** prediction instead — a human-written abstraction over a section
that usually does not appear in it. Triviality is then checked rather than assumed: **10,218** candidates
were rejected for the heading appearing verbatim in the body, and **5,314** more for more than half of the
heading's content words appearing in it. Two further filters came out of reading the samples: bullet-list
sections (**11,274** rejected, plus 69 inline lists) and `زمرہ:` category markup surviving the cleaner.

The list filter was itself broken when first written. `clean_wikitext` stripped the list markers *before*
`list_share` measured them, so the check ran on text with no markers in it and rejected 3 sections out of
11,000. Moving the measurement before the stripping took it to 821 in the same sample.

### 9. The two Wikipedia extracts overlap, and it took four fixes to close

The sentence tasks read Dakshina's filtered extract; the section task reads the full dump. Same
encyclopaedia, so a section body can contain a sentence another task already used:
**1,977 sentences, 380 of them in a different split** — a cloze answer sitting in a test-set body.

Each fix was correct about the leak and wrong about where to apply it:

| Fix | Overlap left | Cross-split left | Why it was not enough |
|---|---|---|---|
| exclude the sentence tasks' sentences from section bodies, matched sentence-to-sentence | 111 | 24 | the two extracts segment sentences differently: the text matches, the boundaries do not |
| match on the first 8-word window instead | 14 | 4 | the extracts do not start a sentence in the same place either |
| index every 8-word window of every excluded sentence | 6 | 4 | all six were from the **top-up pass**, which ran after the sections were emitted |
| exclude by window in the top-up pass too | **0** | **0** | — |

**Fix:** `window_keys` in `urdu.py`, used by both wiki builders, and the measurement itself is now a
check in `scribe.verify` (`sentences_also_inside_a_headline_body`) with two tests asserting it fires.
A one-off probe found this leak; leaving it as a probe would have let the next builder change restore it.

## Standing limits, not fixed

| Limit | Why it stays | How it is disclosed |
|---|---|---|
| Correction inputs are machine-corrupted, not found errors | No public corpus of real Urdu typing errors at this scale exists without a budget | `input_human: false` per row, `input_human_share` in stats, error type per row |
| Formulaic collocations make easy cloze items (`محمد صلی ____ علیہ وسلم` → اللہ) | They are genuine Urdu text, and removing religious and fixed phrases would distort the register | Per-item position and length in `meta`; a difficulty split is future work |
| Word-level transliteration dominates the transliteration tasks | The only non-contaminated Dakshina source is the word lexicon | `meta.level` is `word` on every such row; sentence level exists only in the eval file |
| Register skews encyclopaedic | Source is Wikipedia | `by_corpus` in stats; a news and literature slice is still future work |
| Section headings are short and sometimes generic even after filtering (`تاریخ`, `جغرافیہ`) | They are what the writer chose, and a stricter filter would keep only unusual headings | 1,258,108 navigational headings rejected by name; at most 40 sections per heading |
| Exotic punctuation (e.g. a Syriac full stop) survives the sentence filter | Rare and harmless to the task | Noted here; not filtered |

## Why the clean verifier report should be believed

`python -m scribe.verify` re-reads the written JSONL, re-derives every check from the row itself, and
ignores the builder's own `verified` flag. It reports:

```
rows_read 100000 | rows_checked 100000 | failures {} | failure_rate 0.0
duplicate_row_ids 0 | identical_input_in_two_splits 0 | documents_in_two_splits 0
wiki_sentences_feeding_both_cloze_and_correction 0 | distinct_inputs 100000
sentences_also_inside_a_headline_body 0 | ..._in_another_split 0
```

A 0.000 failure rate is normally a smell, so `tests/test_verify_sensitivity.py` plants one defect of
each known kind — no mask, two masks, an answer still visible in the prompt, a non-Urdu answer, a
two-edit "single" correction, a `space_split` that did not split, a corrupted input claiming to be
human, Latin in the Urdu side, Urdu in the Roman side, a swapped direction, a heading sitting in its
own body, and a sentence shared between a section body and a cloze item in another split — and asserts
the matching check names it. 26 tests, all passing. Two of those tests initially failed because *my expectations*
were wrong about which check fires first, not the checks.

`rows_unchecked_no_check_defined` is printed on every run: a task with no check would show up as rows
nobody looked at, rather than as a smaller, cleaner-looking denominator.
