"""Heading-aware HTML/text evidence extraction with noise removal."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from html.parser import HTMLParser

from evidencetrace.audit_models import (
    EVIDENCE_EXACT_MAX_BYTES,
    EVIDENCE_EXACT_MAX_CHARS,
    EvidenceChunk,
    bounded_evidence_metadata,
    bounded_heading_path,
)
from evidencetrace.models import SourceMetadata
from evidencetrace.retrieval.fetch import FetchedSource

NOISE_TAGS = {
    "script",
    "style",
    "nav",
    "header",
    "footer",
    "aside",
    "form",
    "noscript",
    "template",
}
VOID_TAGS = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "param",
    "source",
    "track",
    "wbr",
}
BLOCK_TAGS = {"p", "li", "dt", "dd", "pre"}
NOISE_WORDS = re.compile(
    r"(?<![A-Za-z0-9])(?:nav(?:bar|igation)?|sidebar|footer|header|"
    r"advert(?:isement|ising)?|cookie|breadcrumb|social)(?![A-Za-z0-9])",
    re.I,
)


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _bounded_text_ranges(value: str) -> tuple[tuple[int, int], ...]:
    """Split cleaned source text into exact, UTF-8-bounded source ranges."""

    ranges: list[tuple[int, int]] = []
    cursor = 0
    while cursor < len(value):
        while cursor < len(value) and value[cursor].isspace():
            cursor += 1
        if cursor >= len(value):
            break
        end = min(cursor + EVIDENCE_EXACT_MAX_CHARS, len(value))
        while (
            end > cursor
            and len(value[cursor:end].encode("utf-8")) > EVIDENCE_EXACT_MAX_BYTES
        ):
            end -= 1
        if end < len(value):
            boundary = max(
                value.rfind(" ", cursor + 1, end + 1),
                value.rfind("\n", cursor + 1, end + 1),
                value.rfind("\t", cursor + 1, end + 1),
            )
            if boundary > cursor:
                end = boundary
        while end > cursor and value[end - 1].isspace():
            end -= 1
        if end <= cursor:
            raise ValueError("source text cannot fit the evidence size limit")
        ranges.append((cursor, end))
        cursor = end
    return tuple(ranges)


class _Parser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.skip_depth = 0
        self.active_tag: str | None = None
        self.active_text: list[str] = []
        self.headings: list[str] = []
        self.title_parts: list[str] = []
        self.in_title = False
        self.loose_parts: list[str] = []
        self.paragraphs: list[tuple[str, tuple[str, ...]]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        values = {key.lower(): value or "" for key, value in attrs}
        attribute_text = " ".join(value or "" for _, value in attrs)
        hidden = "hidden" in values or values.get("aria-hidden", "").lower() == "true"
        noisy = tag in NOISE_TAGS or hidden or bool(NOISE_WORDS.search(attribute_text))
        if self.skip_depth:
            if tag not in VOID_TAGS:
                self.skip_depth += 1
            return
        if noisy:
            if tag not in VOID_TAGS:
                self.skip_depth = 1
            return
        if tag == "title":
            self.in_title = True
            return
        if tag in BLOCK_TAGS or (len(tag) == 2 and tag[0] == "h" and tag[1].isdigit()):
            self.active_tag = tag
            self.active_text = []

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if self.skip_depth:
            self.skip_depth -= 1
            return
        if tag == "title":
            self.in_title = False
            return
        if tag != self.active_tag:
            return
        text = _clean("".join(self.active_text))
        if tag.startswith("h") and len(tag) == 2 and tag[1].isdigit():
            level = int(tag[1])
            self.headings = self.headings[: level - 1]
            if text:
                self.headings.append(bounded_evidence_metadata(text))
        elif text:
            self.paragraphs.append(
                (text, bounded_heading_path(tuple(self.headings)))
            )
        self.active_tag = None
        self.active_text = []

    def handle_data(self, data: str) -> None:
        if self.skip_depth:
            return
        if self.in_title:
            self.title_parts.append(data)
        elif self.active_tag is not None:
            self.active_text.append(data)
        elif data.strip():
            self.loose_parts.append(data)


def extract_html(
    source: FetchedSource, *, source_id: str
) -> tuple[SourceMetadata, tuple[EvidenceChunk, ...]]:
    """Return sanitized chunks whose evidence text is literal extracted text."""

    parser = _Parser()
    paragraphs: list[tuple[str, tuple[str, ...]]]
    if source.content_type == "text/plain":
        paragraphs = [(source.content, ())] if source.content.strip() else []
        title = ""
    else:
        parser.feed(source.content)
        parser.close()
        paragraphs = parser.paragraphs
        if not paragraphs:
            visible = _clean(" ".join(parser.loose_parts))
            paragraphs = [(visible, ())] if visible else []
        title = bounded_evidence_metadata(_clean("".join(parser.title_parts)))
    counts: Counter[tuple[str, ...]] = Counter()
    cursor = 0
    chunks: list[EvidenceChunk] = []
    for text, heading_path in paragraphs:
        counts[heading_path] += 1
        if chunks:
            cursor += 2
        paragraph_start = cursor
        cursor += len(text)
        prefix = " > ".join(heading_path)
        paragraph_label = f"paragraph {counts[heading_path]}"
        ranges = _bounded_text_ranges(text)
        for part, (local_start, local_end) in enumerate(ranges, start=1):
            start = paragraph_start + local_start
            end = paragraph_start + local_end
            locator = (
                f"{prefix} > {paragraph_label}" if prefix else paragraph_label
            )
            if len(ranges) > 1:
                locator += f" part {part}"
            locator += f" [chars {start}-{end}]"
            chunks.append(
                EvidenceChunk(
                    source_id=source_id,
                    url=source.final_url,
                    text=text[local_start:local_end],
                    heading_path=heading_path,
                    locator=locator,
                    char_start=start,
                    char_end=end,
                )
            )
    metadata = SourceMetadata(
        source_id=source_id,
        url=source.final_url,
        title=title,
        retrieved_at=source.retrieved_at,
        content_hash=hashlib.sha256(source.content.encode("utf-8")).hexdigest(),
        mime_type=source.content_type,
        status="ok",
    )
    return metadata, tuple(chunks)


__all__ = ["extract_html"]
