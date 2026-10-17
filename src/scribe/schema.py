"""Row schema for scribe-100k.

One row is one task item. Every field is either copied from a human source or
computed by code; nothing here is produced by a language model.

Provenance is explicit and per row, because the dataset's whole claim against
Alif Urdu-Instruct is provenance rather than size. Two flags carry it:

* ``target_human``  - the gold answer is text a person wrote.
* ``input_human``   - the prompt text is text a person wrote. False when the
  prompt was derived from it (a sentence with a word masked, or carrying one
  error), which is reported as a share in the dataset stats.

A row is only counted when ``verified`` is true, i.e. the check named in
``verify_method`` was run and passed.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import asdict, dataclass, field

SCHEMA_VERSION = "1"

# Task names are stable identifiers; the stats table and the splits key off them.
TASKS = (
    "translit_ur2roman",
    "translit_roman2ur",
    "cloze",
    "correction",
    "headline",
)

LICENCES = {
    "dakshina": "CC BY-SA 4.0",
    "urwiki": "CC BY-SA 4.0",
    "urwiki-dakshina-filt": "CC BY-SA 4.0",
}


@dataclass(frozen=True)
class Row:
    task: str
    instruction: str
    input: str
    output: str
    corpus: str
    doc_id: str
    verify_method: str
    verified: bool
    target_human: bool
    input_human: bool
    licence: str = ""
    source_url: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.task not in TASKS:
            raise ValueError(f"unknown task: {self.task!r}")
        if not self.output.strip():
            raise ValueError("empty output")
        if not self.verify_method:
            raise ValueError("a row must name the check that produced `verified`")

    @property
    def row_id(self) -> str:
        """Content-addressed id, so the same item built twice collides instead of duplicating."""
        # An explicit separator join, not string formatting: the separator must stay
        # visible, and it must be a character that cannot occur in any field.
        payload = "\x1f".join(  # noqa: FLY002
            [self.task, self.input, self.output, self.corpus, self.doc_id]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def to_json(self) -> str:
        d = asdict(self)
        d["row_id"] = self.row_id
        d["schema_version"] = SCHEMA_VERSION
        if not d["licence"]:
            d["licence"] = LICENCES.get(self.corpus, "")
        return json.dumps(d, ensure_ascii=False, sort_keys=True)


def nfc(text: str) -> str:
    """Normalise to NFC and collapse whitespace.

    Urdu text from the wild arrives in several normalisation forms, and an
    un-normalised gold answer silently fails an exact-match check later.
    """
    return " ".join(unicodedata.normalize("NFC", text).split())
