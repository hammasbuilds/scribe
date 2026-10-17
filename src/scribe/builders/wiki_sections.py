"""Section-heading items from the full Urdu Wikipedia dump.

Source: `urwiki-latest-pages-articles.xml.bz2` (476,815,422 bytes, 2026-10-01,
`bzip2 -t` verified). CC BY-SA 4.0.

**Why section headings and not article titles.** The obvious summarisation pair -
lead paragraph to article title - is a copy task in Urdu Wikipedia, because a
lead almost always opens with the title verbatim ("فیصل آباد ایک شہر ہے..."). A
model would score well by extracting the first noun phrase. A section heading is
a human-written abstraction over its own section and usually does *not* appear in
the body, so the item requires reading the section rather than copying from it.

Both sides are human text: a person wrote the body, and a person wrote the
heading that summarises it.

Triviality is checked, not assumed. A candidate is rejected when the heading
appears verbatim in the body, and when more than half of the heading's content
words appear in the body. Both rejection counts are reported by the build.
"""

from __future__ import annotations

import bz2
import collections
import hashlib
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from ..schema import Row, nfc
from ..urdu import (
    has_arabic_only_form,
    is_urdu_word,
    urdu_share,
    usable_sentence,
    window_keys,
    words,
)

INSTR_HEADLINE = "اس اقتباس کے لیے مناسب عنوان لکھیں۔"

# Navigational and bibliographic headings. They are the most frequent headings on
# the wiki and they describe the page's furniture, not its content, so an item
# asking for them would be a frequency guess.
GENERIC_HEADINGS = frozenset(
    """
    حوالہ_جات حوالہ حوالے بیرونی_روابط بیرونی_ربط مزید_دیکھیے مزید_دیکھئے مزید_پڑھیے
    کتابیات تصاویر گیلری فہرست تعارف مزید اعداد_و_شمار نگار_خانہ نگارخانہ متعلقہ_مضامین
    """.split()  # noqa: SIM905 - a readable block beats a one-line list literal
)

HEADING_RE = re.compile(r"^==\s*([^=].*?)\s*==\s*$")
# Wikitext noise, stripped in this order.
REF_RE = re.compile(r"<ref[^>]*>.*?</ref>|<ref[^>]*/>", re.DOTALL)
COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
TAG_RE = re.compile(r"<[^>]+>")
TEMPLATE_RE = re.compile(r"\{\{[^{}]*\}\}")
TABLE_RE = re.compile(r"\{\|.*?\|\}", re.DOTALL)
PIPED_LINK_RE = re.compile(r"\[\[[^\]|]*\|([^\]]*)\]\]")
LINK_RE = re.compile(r"\[\[([^\]]*)\]\]")
EXTLINK_RE = re.compile(r"\[(?:https?|ftp)://[^\s\]]+\s*([^\]]*)\]")
FILE_RE = re.compile(r"\[\[\s*(?:File|Image|ملف|تصویر)\s*:[^\]]*\]\]", re.IGNORECASE)
QUOTES_RE = re.compile(r"'{2,}")
LIST_PREFIX_RE = re.compile(r"^[*#:;]+\s*", re.MULTILINE)
LIST_LINE_RE = re.compile(r"^\s*(?:[*#:;]|-\s)")
# Category links survive template stripping and arrive as stray text ("زمرہ:فیصلہ طلب").
CATEGORY_RE = re.compile(r"\[?\[?\s*(?:Category|زمرہ)\s*:[^\]\n]*\]?\]?", re.IGNORECASE)


def list_share(text: str) -> float:
    """Share of non-empty lines that are wiki list items.

    Measured *before* the markers are stripped. A section that is mostly a bullet
    list is not prose, and its heading ("اہم شخصیات") is guessable from the shape
    of the list rather than from reading it.
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return 1.0
    return sum(1 for ln in lines if LIST_LINE_RE.match(ln)) / len(lines)


def clean_wikitext(text: str) -> str:
    text = COMMENT_RE.sub(" ", text)
    text = CATEGORY_RE.sub(" ", text)
    text = REF_RE.sub(" ", text)
    text = TABLE_RE.sub(" ", text)
    text = FILE_RE.sub(" ", text)
    for _ in range(4):  # templates nest; four passes clears the common depth
        new = TEMPLATE_RE.sub(" ", text)
        if new == text:
            break
        text = new
    text = PIPED_LINK_RE.sub(r"\1", text)
    text = LINK_RE.sub(r"\1", text)
    text = EXTLINK_RE.sub(r"\1", text)
    text = TAG_RE.sub(" ", text)
    text = QUOTES_RE.sub("", text)
    # List markers are deliberately NOT stripped here: list_share has to see
    # them. The first version stripped them in this function and then measured
    # afterwards, so the "is this a list?" check ran on text with no markers
    # left in it and rejected 3 sections out of 11,000. _make_row strips them.
    return text


def iter_pages(dump_path: Path, *, max_pages: int | None = None):
    """Stream (title, wikitext) for main-namespace pages out of the bz2 dump."""
    with bz2.open(dump_path, "rb") as fh:
        seen = 0
        title = ns = text = None
        for event, elem in ET.iterparse(fh, events=("end",)):
            tag = elem.tag.rsplit("}", 1)[-1]
            if tag == "title":
                title = elem.text
            elif tag == "ns":
                ns = elem.text
            elif tag == "text":
                text = elem.text
            elif tag == "page":
                if ns == "0" and title and text:
                    yield title, text
                    seen += 1
                title = ns = text = None
                elem.clear()
                if max_pages is not None and seen >= max_pages:
                    return
            elif tag in ("revision", "redirect"):
                continue


def heading_key(heading: str) -> str:
    return heading.replace(" ", "_")


def _content_words(text: str) -> set[str]:
    return {w for w in words(text) if is_urdu_word(w) and len(w) >= 3}


def section_rows(
    dump_path: Path,
    *,
    rejected: collections.Counter | None = None,
    exclude_sentences: set[str] | None = None,
    limit: int | None = None,
    max_pages: int | None = None,
    min_body_words: int = 30,
    max_body_words: int = 140,
    max_per_heading: int = 40,
):
    """Yield (Row | None) is not used: yields Rows, and counts rejections in `stats`.

    ``max_per_heading`` stops one popular heading from becoming the whole task:
    "ابتدائی زندگی" is a good item once, not four thousand times.
    """
    made = 0
    rejected = rejected if rejected is not None else collections.Counter()
    per_heading: collections.Counter = collections.Counter()
    exclude_keys: set[str] = set()
    if exclude_sentences:
        # Every window of every excluded sentence, not just its first. Indexing only
        # the first window still left 14 overlaps (4 across a split), because the two
        # extracts do not start their sentences in the same place.
        for s in exclude_sentences:
            exclude_keys |= window_keys(s)
    for title, wikitext in iter_pages(dump_path, max_pages=max_pages):
        if limit is not None and made >= limit:
            return
        lines = clean_wikitext(wikitext).splitlines()
        current: str | None = None
        buf: list[str] = []
        for line in lines + ["== END =="]:
            m = HEADING_RE.match(line.strip())
            if m is None:
                if current is not None:
                    buf.append(line)
                continue
            # A heading closes the previous section: try to emit it.
            if current is not None:
                raw = "\n".join(buf)
                row = _make_row(
                    title,
                    current,
                    " ".join(buf),
                    min_body_words,
                    max_body_words,
                    list_share(raw),
                    rejected,
                    exclude_keys,
                )
                if row is not None and per_heading[heading_key(current)] >= max_per_heading:
                    rejected["heading_quota_reached"] += 1
                elif row is not None:
                    per_heading[heading_key(current)] += 1
                    yield row
                    made += 1
                    if limit is not None and made >= limit:
                        return
            current, buf = m.group(1), []


def _make_row(
    title: str,
    heading: str,
    body: str,
    min_body_words: int,
    max_body_words: int,
    body_list_share: float,
    rejected: collections.Counter,
    exclude_keys: set[str] | None = None,
):
    heading = nfc(heading)
    body = nfc(body)

    def no(reason: str) -> None:
        """Count a rejection reason and return the sentinel (None) for the caller."""
        rejected[reason] += 1

    if heading_key(heading) in GENERIC_HEADINGS:
        return no("heading_generic")
    body = LIST_PREFIX_RE.sub("", body)
    hw = words(heading)
    if not (1 <= len(hw) <= 6):
        return no("heading_length")
    if urdu_share(heading) < 0.80 or has_arabic_only_form(heading):
        return no("heading_not_urdu")
    if not any(is_urdu_word(w) for w in hw):
        return no("heading_no_urdu_word")

    bw = words(body)
    if not (min_body_words <= len(bw) <= max_body_words):
        return no("body_length")
    if urdu_share(body) < 0.80:
        return no("body_not_urdu")
    if body_list_share > 0.4:
        return no("body_is_a_list")
    # Bullets joined onto one line read as prose to a sentence splitter but are
    # still a list: "- item - item - item".
    if body.count(" - ") >= 3:
        return no("body_is_an_inline_list")
    # The body must read as prose: at least two usable sentences in it.
    from ..urdu import sentences

    if sum(1 for s in sentences(body) if usable_sentence(s)) < 2:
        return no("body_not_prose")

    # Cross-task exclusion. The sentence tasks draw from a filtered extract of this
    # same Wikipedia, so a body can contain a sentence another task already used:
    # measured at 1,977 sentences, 380 of them across a split boundary. A body that
    # contains one is dropped rather than split away, because the body is the larger
    # unit and the sentence tasks were filled first.
    if exclude_keys and (window_keys(body) & exclude_keys):
        return no("body_contains_a_sentence_used_by_another_task")

    # Triviality: the answer must not be sitting in the prompt.
    if heading in body:
        return no("heading_verbatim_in_body")
    hc = _content_words(heading)
    bc = _content_words(body)
    if not hc:
        return no("heading_no_content_words")
    overlap = len(hc & bc) / len(hc)
    if overlap > 0.5:
        return no("heading_words_mostly_in_body")

    return Row(
        task="headline",
        instruction=INSTR_HEADLINE,
        input=body,
        output=heading,
        corpus="urwiki",
        doc_id="urwiki-sec:"
        + hashlib.sha256(f"{title}\x1f{heading}".encode()).hexdigest()[:16],
        verify_method="heading_absent_from_body+content_overlap<=0.5",
        verified=True,
        target_human=True,
        input_human=True,
        source_url="https://dumps.wikimedia.org/urwiki/latest/",
        meta={
            "article": title,
            "body_words": len(bw),
            "heading_words": len(hw),
            "content_word_overlap": round(overlap, 3),
        },
    )
