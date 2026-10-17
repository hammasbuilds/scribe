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
        ({"input_human": True}, "corrupted_input_claimed_human"),
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
