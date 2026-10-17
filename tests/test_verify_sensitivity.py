"""The verifier must be shown to discriminate before a clean report means anything.

`python -m scribe.verify` reported a 0.000 failure rate over 100,000 rows. That
is only evidence if the same checks reject bad rows, so each test here plants
one defect of a known kind and asserts the matching check names it. A test that
merely asserted "good rows pass" would pass even if every check were `return None`.
"""

from __future__ import annotations

import pytest

from scribe.verify import check_cloze, check_correction, check_translit

GOOD_SENT = "یہ شہر دریائے سندھ کے کنارے آباد ایک پرانا تجارتی مرکز سمجھا جاتا ہے"
ANSWER = "تجارتی"


def cloze_row(**over):
    row = {
        "task": "cloze",
        "input": GOOD_SENT.replace(ANSWER, "____"),
        "output": ANSWER,
        "input_human": True,
    }
    row.update(over)
    return row


def correction_row(**over):
    row = {
        "task": "correction",
        "input": GOOD_SENT.replace("کنارے", "کنارہ"),
        "output": GOOD_SENT,
        "input_human": False,
        "meta": {"error_type": "confusion:ے->ہ"},
    }
    row.update(over)
    return row


def translit_row(**over):
    row = {"task": "translit_ur2roman", "input": "کتاب", "output": "kitaab"}
    row.update(over)
    return row


def test_good_rows_pass():
    assert check_cloze(cloze_row()) is None
    assert check_correction(correction_row()) is None
    assert check_translit(translit_row()) is None


@pytest.mark.parametrize(
    "over,expected",
    [
        ({"input": GOOD_SENT}, "mask_count"),  # no mask at all
        ({"input": GOOD_SENT.replace(ANSWER, "____").replace("شہر", "____")}, "mask_count"),
        ({"output": "تجارتی مرکز"}, "answer_changes_word_count"),
        ({"output": "شہر"}, "answer_also_present_in_prompt"),  # a word still visible in the prompt
        ({"output": "kitaab"}, "answer_not_urdu_word"),
    ],
)
def test_cloze_defects_are_caught(over, expected):
    assert check_cloze(cloze_row(**over)) == expected


def test_cloze_answer_leak_is_caught():
    """The answer still visible in the prompt is the defect that inflates any score."""
    leaky = GOOD_SENT + " " + ANSWER
    row = cloze_row(input=leaky.replace(ANSWER, "____", 1), output=ANSWER)
    assert check_cloze(row) == "answer_also_present_in_prompt"


@pytest.mark.parametrize(
    "over,expected",
    [
        ({"input": GOOD_SENT}, "no_error_introduced"),
        ({"meta": {}}, "error_type_missing"),
        ({"input_human": True}, "modified_input_claimed_human"),
        ({"input": "یہ شہر"}, "word_count_drift"),
    ],
)
def test_correction_defects_are_caught(over, expected):
    assert check_correction(correction_row(**over)) == expected


def test_correction_with_two_edits_is_caught():
    two = GOOD_SENT.replace("کنارے", "کنارہ").replace("پرانا", "پرانی")
    assert check_correction(correction_row(input=two)) == "expected_one_changed_token_got_2"


def test_space_split_that_did_not_split_is_caught():
    row = correction_row(
        input=GOOD_SENT.replace("کنارے", "کنارہ"), meta={"error_type": "space_split"}
    )
    assert check_correction(row) == "space_split_did_not_split"


@pytest.mark.parametrize(
    "over,expected",
    [
        ({"input": "kitaab"}, "urdu_side_not_urdu"),
        # One Latin word in a long Urdu sentence: the share check passes, so the
        # Latin check is the one that must fire.
        ({"input": GOOD_SENT + " book"}, "latin_in_urdu_side"),
        ({"output": "کتاب"}, "roman_side_has_no_latin"),
        ({"output": "kitaab کتاب"}, "urdu_in_roman_side"),
    ],
)
def test_translit_defects_are_caught(over, expected):
    assert check_translit(translit_row(**over)) == expected


def test_translit_identity_pair_is_caught():
    assert check_translit({"task": "translit_ur2roman", "input": "abc", "output": "abc"}) in (
        "urdu_side_not_urdu",
        "identity_pair",
    )


def test_reverse_direction_checks_the_right_side():
    """roman2ur swaps which field must be Urdu; the check must swap with it."""
    ok = {"task": "translit_roman2ur", "input": "kitaab", "output": "کتاب"}
    assert check_translit(ok) is None
    swapped = {"task": "translit_roman2ur", "input": "کتاب", "output": "kitaab"}
    assert check_translit(swapped) is not None


# --- headline (section heading) ---------------------------------------------

BODY = (
    "الجزائر کا زیادہ تر ساحلی علاقہ پتھریلا اور چٹانی ہے اور کہیں کہیں پہاڑ بھی ملتے ہیں۔ "
    "تاہم یہاں کئی قدرتی بندرگاہیں بھی موجود ہیں۔ ساحل سے لے کر اطلس التلی تک کا علاقہ زرخیز "
    "ہے اور اس کے بعد گھاس کے وسیع میدان شروع ہو جاتے ہیں جو مشرقی حصے تک پھیلے ہوئے ہیں۔"
)


def headline_row(**over):
    row = {"task": "headline", "input": BODY, "output": "محل وقوع"}
    row.update(over)
    return row


def test_good_headline_passes():
    from scribe.verify import check_headline

    assert check_headline(headline_row()) is None


@pytest.mark.parametrize(
    "over,expected",
    [
        # The heading sitting in the prompt is the defect that turns this task into copying.
        ({"output": "ساحلی علاقہ"}, "heading_verbatim_in_body"),
        ({"output": "اطلس التلی میدان"}, "heading_words_mostly_in_body"),
        ({"output": "محل وقوع اور جغرافیائی حدود اربعہ کی تفصیل مکمل"}, "heading_length"),
        ({"input": "الجزائر کا ساحل پتھریلا ہے۔"}, "body_too_short"),
    ],
)
def test_headline_defects_are_caught(over, expected):
    from scribe.verify import check_headline

    assert check_headline(headline_row(**over)) == expected


# --- cross-extract leakage (the audit-level check) --------------------------


def _rows_for_audit(cloze_split="test"):
    """A headline row in train, and a cloze row whose source sits in that body."""
    body = BODY
    first = " ".join(body.split()[:12])
    return [
        {
            "task": "headline",
            "input": body,
            "output": "محل وقوع",
            "doc_id": "urwiki-sec:aaaa",
            "row_id": "a1",
            "_split": "train",
            "input_human": True,
            "meta": {},
        },
        {
            "task": "cloze",
            "input": first.replace(first.split()[3], "____", 1),
            "output": first.split()[3],
            "doc_id": "urwiki-sent:bbbb",
            "row_id": "b1",
            "_split": cloze_split,
            "input_human": True,
            "meta": {},
        },
    ]


def test_audit_catches_a_sentence_inside_a_headline_body():
    """The leak that cost four rebuilds must fail the audit, not just the probe."""
    from scribe.verify import audit

    report = audit(_rows_for_audit(cloze_split="test"))
    assert report["sentences_also_inside_a_headline_body"] == 1
    assert report["sentences_inside_a_headline_body_in_another_split"] == 1


def test_audit_is_clean_when_the_passage_is_not_shared():
    from scribe.verify import audit

    rows = _rows_for_audit()
    rows[1]["input"] = "یہ ایک بالکل مختلف جملہ ہے جس کا متن کسی اور ____ سے لیا گیا ہے۔"
    rows[1]["output"] = "مضمون"
    report = audit(rows)
    assert report["sentences_also_inside_a_headline_body"] == 0


# --- engine behaviours found by feeding it real input -----------------------


def _corpus():
    import pathlib

    """A small corpus built in memory, so these tests need no dataset on disk."""
    from scribe.engine import Corpus

    c = Corpus.__new__(Corpus)
    import collections
    import itertools

    from scribe.engine import BM25
    from scribe.urdu import strip_marks
    from scribe.urdu import words as _w

    c.rows = []
    # build_app reads the data directory for examples, stats and the baseline;
    # pointing it at nothing keeps these tests independent of the dataset on disk.
    c.data_dir = pathlib.Path("tests") / "no-such-dataset"
    c.ur2roman, c.roman2ur, c.urdu2roman_from_reverse = {}, {}, {}
    c.sentences = [
        "انہوں نے طب کی تعلیم حاصل کی اور بھوپال میں طبابت کی",
        "یہ شہر دریائے سندھ کے کنارے ایک پرانا تجارتی مرکز ہے",
        "وہ ہر روز اسکول جاتا ہے اور شام کو کھیلتا ہے",
    ] * 3
    c.unigram, c.bigram = collections.Counter(), collections.Counter()
    for s in c.sentences:
        toks = _w(s)
        c.unigram.update(strip_marks(t) for t in toks)
        c.bigram.update(itertools.pairwise([strip_marks(t) for t in toks]))
    c.vocab = {w for w, n in c.unigram.items() if n >= 2}
    c.vocab_by_first = collections.defaultdict(list)
    for w in c.vocab:
        c.vocab_by_first[w[0]].append(w)
    c.context = collections.defaultdict(collections.Counter)
    c.after = collections.defaultdict(collections.Counter)
    c.before = collections.defaultdict(collections.Counter)
    for s in c.sentences:
        toks = [strip_marks(t) for t in _w(s)]
        for i in range(1, len(toks) - 1):
            c.context[(toks[i - 1], toks[i + 1])][toks[i]] += 1
        for a, b in itertools.pairwise(toks):
            c.after[a][b] += 1
            c.before[b][a] += 1
    c.headline_bodies, c.headline_heads = [], []
    c.headline_bm25 = BM25([])
    c.sentence_bm25 = BM25([[strip_marks(t) for t in _w(s)] for s in c.sentences])
    c.heading_freq = collections.Counter()
    return c


def test_corrector_joins_a_split_word():
    """A sum over tokens rewarded keeping two fragments; the score is per token now."""
    from scribe.engine import correct

    out = correct("انہوں نے طب کی تع لیم حاصل کی", _corpus()).output
    assert "تعلیم" in out, out


def test_corrector_leaves_correct_text_alone():
    from scribe.engine import correct

    text = "وہ ہر روز اسکول جاتا ہے"
    answer = correct(text, _corpus())
    assert answer.output == text
    assert "no change" in answer.method


def test_cloze_answer_never_carries_sentence_punctuation():
    """"ہے۔" is a correct token and a wrong answer."""
    from scribe.engine import fill_blank

    c = _corpus()
    c.sentences = [s + "۔" for s in c.sentences]
    answer = fill_blank("یہ شہر دریائے سندھ کے ____ ایک پرانا مرکز ہے", c)
    assert "۔" not in answer.output, answer.output


def test_cloze_reports_more_than_one_blank():
    from scribe.engine import fill_blank

    answer = fill_blank("یہ ____ شہر ____ ہے", _corpus())
    assert "2 blanks" in answer.note, answer.note


def test_cloze_without_a_blank_explains_itself():
    from scribe.engine import fill_blank

    answer = fill_blank("یہ جملہ مکمل ہے", _corpus())
    assert answer.output == ""
    assert "____" in answer.note


def test_cloze_fallback_does_not_answer_with_a_function_word():
    """The frequency fallback answered میں - the commonest word in the corpus."""
    from scribe.engine import fill_blank
    from scribe.urdu import FUNCTION_WORDS, strip_marks

    answer = fill_blank("کوئی ایسا جملہ ____ جو corpus میں نہیں", _corpus())
    if answer.output:
        assert strip_marks(answer.output) not in FUNCTION_WORDS, answer.output


def test_heading_query_is_capped():
    """A 92 KB paste took 60 seconds before the query was capped."""
    from scribe.engine import MAX_QUERY_WORDS, suggest_heading

    big = "یہ ایک لمبا جملہ ہے جو بار بار دہرایا گیا ہے۔ " * 500
    answer = suggest_heading(big, _corpus())
    assert answer.output is not None
    assert MAX_QUERY_WORDS <= 1000


# --- the batch endpoint -----------------------------------------------------


def test_batch_runs_every_line_and_counts_the_changes():
    """A desk that answers one sentence at a time is a demonstration; a person
    correcting a document has three hundred of them."""
    from scribe.app import build_app

    app = build_app(_corpus(), examples_split="val")
    handler = app.routes["/api/batch"]
    out = handler({"task": "correct", "text": "انہوں نے طب کی تع لیم حاصل کی\nوہ ہر روز اسکول جاتا ہے\n\n"})
    assert out["lines_given"] == 2
    assert out["lines_done"] == 2
    assert len(out["rows"]) == 2
    assert any(r["changed"] for r in out["rows"]), out["rows"]
    assert any(not r["changed"] for r in out["rows"]), "a correct line must be left alone"


def test_batch_caps_the_work_and_says_what_was_left():
    from scribe.app import build_app

    app = build_app(_corpus(), examples_split="val")
    handler = app.routes["/api/batch"]
    out = handler({"task": "correct", "text": "\n".join(["وہ ہر روز اسکول جاتا ہے"] * 12), "limit": 5})
    assert out["lines_done"] == 5
    assert out["lines_skipped"] == 7, "the caller has to be told what was not done"


def test_batch_rejects_an_unknown_task():
    from scribe.app import build_app

    app = build_app(_corpus(), examples_split="val")
    assert "error" in app.routes["/api/batch"]({"task": "nonsense", "text": "x"})
