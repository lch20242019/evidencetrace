"""Deterministic, source-mapped retrieval over local Markdown/TXT references.

Filesystem paths are runtime-only inputs.  Public results use caller-provided
safe identifiers, content hashes, and bounded exact source excerpts.
"""

from __future__ import annotations

import hashlib
import re
from bisect import bisect_right
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path, PurePosixPath
from typing import Literal, Self
from urllib.parse import quote

from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeInt,
    PositiveInt,
    field_validator,
    model_validator,
)

from evidencetrace.audit_models import (
    EVIDENCE_EXACT_MAX_CHARS,
    EvidenceChunk,
    RetrievedEvidence,
    bounded_heading_path,
)
from evidencetrace.markdown import parse_markdown_with_source_map
from evidencetrace.models import (
    AtomicClaim,
    CitationKind,
    CitationOccurrence,
    MarkdownParagraph,
    ModelTextSourceSegment,
    ParagraphKind,
    ParsedDocument,
    ParsedDocumentWithSourceMap,
    SourceMetadata,
    SourceSpan,
    TargetKind,
    TrustedModelVisibleParagraph,
)
from evidencetrace.retrieval.rank import LexicalRetriever

LOCAL_EVIDENCE_MAX_TOP_K = 5
LOCAL_REFERENCE_MAX_BYTES = 5 * 1024 * 1024
LOCAL_REFERENCE_CHUNK_CHARS = EVIDENCE_EXACT_MAX_CHARS
_LOCAL_REFERENCE_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_SAFE_SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_BARE_URL_RE = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)


class LocalEvidenceError(ValueError):
    """A local reference violated the bounded, privacy-safe input contract."""


def _safe_display_id(value: str) -> str:
    if not isinstance(value, str) or value != value.strip() or not value:
        raise LocalEvidenceError("safe display ID must be non-empty")
    if "\\" in value:
        raise LocalEvidenceError("safe display ID must use POSIX separators")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or value in {".", ".."}
        or ".." in path.parts
        or any(not _SAFE_SEGMENT_RE.fullmatch(part) for part in path.parts)
    ):
        raise LocalEvidenceError("safe display ID is invalid")
    return value


def _content_sha256(value: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise LocalEvidenceError("reference SHA-256 must be 64 hexadecimal characters")
    return value.lower()


class _SafeModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class LocalProtectedSpan(_SafeModel):
    """A private-path-free range that a repair planner must not modify."""

    safe_id: str
    content_sha256: str
    line_start: PositiveInt
    line_end: PositiveInt
    column_start: PositiveInt
    column_end: PositiveInt
    char_start: NonNegativeInt
    char_end: PositiveInt
    byte_start: NonNegativeInt
    byte_end: PositiveInt

    @field_validator("safe_id")
    @classmethod
    def validate_safe_id(cls, value: str) -> str:
        return _safe_display_id(value)

    @field_validator("content_sha256")
    @classmethod
    def validate_content_sha256(cls, value: str) -> str:
        return _content_sha256(value)

    @model_validator(mode="after")
    def validate_ranges(self) -> Self:
        if self.line_end < self.line_start:
            raise ValueError("protected span line range is reversed")
        if self.char_end <= self.char_start:
            raise ValueError("protected span character range must be non-empty")
        if self.byte_end <= self.byte_start:
            raise ValueError("protected span byte range must be non-empty")
        return self


class PlainTextParsedSource(ParsedDocumentWithSourceMap):
    """TXT parser result compatible with the existing trusted source-map API."""

    protected_spans: tuple[LocalProtectedSpan, ...] = ()


class LocalReferenceSummary(_SafeModel):
    """Serializable local-reference state without its filesystem path or body."""

    reference_id: str
    reference_sha256: str
    line_count: NonNegativeInt
    media_type: Literal["text/markdown", "text/plain"]
    source: SourceMetadata
    protected_spans: tuple[LocalProtectedSpan, ...] = ()

    @field_validator("reference_id")
    @classmethod
    def validate_reference_id(cls, value: str) -> str:
        return _safe_display_id(value)

    @field_validator("reference_sha256")
    @classmethod
    def validate_reference_sha256(cls, value: str) -> str:
        return _content_sha256(value)

    @model_validator(mode="after")
    def validate_source_identity(self) -> Self:
        if self.source.content_hash != self.reference_sha256:
            raise ValueError("local source hash does not match its reference")
        if any(
            span.safe_id != self.reference_id
            or span.content_sha256 != self.reference_sha256
            for span in self.protected_spans
        ):
            raise ValueError("protected span escaped its local reference")
        return self


class LocalEvidenceResult(_SafeModel):
    """Bounded evidence returned for one relevant local reference."""

    source: SourceMetadata
    evidence: tuple[RetrievedEvidence, ...]
    reference_id: str
    reference_sha256: str

    @field_validator("reference_id")
    @classmethod
    def validate_reference_id(cls, value: str) -> str:
        return _safe_display_id(value)

    @field_validator("reference_sha256")
    @classmethod
    def validate_reference_sha256(cls, value: str) -> str:
        return _content_sha256(value)

    @model_validator(mode="after")
    def validate_evidence_identity(self) -> Self:
        if self.source.content_hash != self.reference_sha256:
            raise ValueError("local evidence hash does not match its source")
        if any(
            item.chunk.source_id != self.source.source_id
            or item.chunk.url != self.source.url
            for item in self.evidence
        ):
            raise ValueError("local evidence escaped its source")
        return self


@dataclass(frozen=True, slots=True)
class LocalReferenceInput:
    """Runtime-only path plus safe, caller-owned reference identity."""

    path: Path = field(repr=False)
    reference_id: str
    reference_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(self, "reference_id", _safe_display_id(self.reference_id))
        object.__setattr__(
            self, "reference_sha256", _content_sha256(self.reference_sha256)
        )


@dataclass(frozen=True)
class _LineMap:
    text: str
    safe_id: str
    starts: tuple[int, ...]

    @classmethod
    def build(cls, text: str, safe_id: str) -> _LineMap:
        starts = [0]
        starts.extend(match.end() for match in re.finditer("\n", text))
        return cls(text=text, safe_id=safe_id, starts=tuple(starts))

    @property
    def line_count(self) -> int:
        if not self.text:
            return 0
        return len(self.starts) - int(self.text.endswith("\n"))

    def physical_line_offsets(self, line: int) -> tuple[int, int]:
        start = self.starts[line]
        end = self.starts[line + 1] if line + 1 < len(self.starts) else len(self.text)
        while end > start and self.text[end - 1] in "\r\n":
            end -= 1
        return start, end

    def span(self, start: int, end: int) -> SourceSpan:
        if not 0 <= start < end <= len(self.text):
            raise LocalEvidenceError("source span must be non-empty and in bounds")
        start_line = bisect_right(self.starts, start) - 1
        final = end - 1
        end_line = bisect_right(self.starts, final) - 1
        return SourceSpan(
            file=self.safe_id,
            line_start=start_line + 1,
            line_end=end_line + 1,
            column_start=start - self.starts[start_line] + 1,
            column_end=final - self.starts[end_line] + 1,
            offset_start=start,
            offset_end=end,
        )


def _trim_bare_url(value: str, start: int, end: int) -> int:
    while end > start and value[end - 1] in ".,;:!?":
        end -= 1
    pairs = (("(", ")"), ("[", "]"), ("{", "}"))
    changed = True
    while changed and end > start:
        changed = False
        for opening, closing in pairs:
            candidate = value[start:end]
            if candidate.endswith(closing) and candidate.count(
                closing
            ) > candidate.count(opening):
                end -= 1
                changed = True
    return end


def _url_ranges(value: str) -> tuple[tuple[int, int], ...]:
    ranges = []
    for match in _BARE_URL_RE.finditer(value):
        end = _trim_bare_url(value, match.start(), match.end())
        if end > match.start():
            ranges.append((match.start(), end))
    return tuple(ranges)


def _model_segments(
    raw: str,
    *,
    source_start: int,
    line_map: _LineMap,
    protected: tuple[tuple[int, int], ...],
) -> tuple[ModelTextSourceSegment, ...]:
    boundaries = {0, len(raw)}
    for start, end in protected:
        boundaries.update((start, end))
    points = sorted(boundaries)
    protected_set = set(protected)
    segments = []
    for start, end in pairwise(points):
        if end <= start:
            continue
        source = line_map.span(source_start + start, source_start + end)
        segments.append(
            ModelTextSourceSegment(
                model_start=start,
                model_end=end,
                source_line_start=source.line_start,
                source_line_end=source.line_end,
                split_protected=(start, end) in protected_set,
                kind="plain",
            )
        )
    return tuple(segments)


def parse_plain_text_with_source_map(
    text: str, display_path: str
) -> PlainTextParsedSource:
    """Parse UTF-8 plain text by blank-line paragraphs without Markdown rules.

    Bare HTTP(S) URLs remain literal model-visible text, become citation
    candidates, and are exposed as character/byte protected spans.
    """

    safe_id = _safe_display_id(display_path)
    try:
        encoded = text.encode("utf-8")
    except UnicodeEncodeError:
        raise LocalEvidenceError("plain text is not valid UTF-8") from None
    content_hash = hashlib.sha256(encoded).hexdigest()
    line_map = _LineMap.build(text, safe_id)
    blocks: list[tuple[int, int]] = []
    block_start: int | None = None
    block_end = 0
    for line in range(line_map.line_count):
        start, end = line_map.physical_line_offsets(line)
        if text[start:end].strip():
            if block_start is None:
                block_start = start
            block_end = end
        elif block_start is not None:
            blocks.append((block_start, block_end))
            block_start = None
    if block_start is not None:
        blocks.append((block_start, block_end))

    paragraphs: list[MarkdownParagraph] = []
    trusted: list[TrustedModelVisibleParagraph] = []
    citations: list[CitationOccurrence] = []
    protected_spans: list[LocalProtectedSpan] = []
    for paragraph_number, (start, end) in enumerate(blocks, 1):
        raw = text[start:end]
        local_urls = _url_ranges(raw)
        citation_ids = []
        citation_urls = []
        for local_start, local_end in local_urls:
            absolute_start = start + local_start
            absolute_end = start + local_end
            source = line_map.span(absolute_start, absolute_end)
            url = text[absolute_start:absolute_end]
            citation_id = f"cit_{len(citations) + 1:04d}"
            citation_ids.append(citation_id)
            citation_urls.append(url)
            citations.append(
                CitationOccurrence(
                    citation_id=citation_id,
                    kind=CitationKind.BARE_URL,
                    raw=url,
                    destination=url,
                    target_kind=TargetKind.WEB,
                    resolved_urls=(url,),
                    source=source,
                    resolved=True,
                )
            )
            protected_spans.append(
                LocalProtectedSpan(
                    safe_id=safe_id,
                    content_sha256=content_hash,
                    line_start=source.line_start,
                    line_end=source.line_end,
                    column_start=source.column_start,
                    column_end=source.column_end,
                    char_start=absolute_start,
                    char_end=absolute_end,
                    byte_start=len(text[:absolute_start].encode("utf-8")),
                    byte_end=len(text[:absolute_end].encode("utf-8")),
                )
            )

        source = line_map.span(start, end)
        paragraph_id = f"p_{paragraph_number:04d}"
        paragraphs.append(
            MarkdownParagraph(
                paragraph_id=paragraph_id,
                kind=ParagraphKind.PARAGRAPH,
                source=source,
                raw_text=raw,
                plain_text=raw,
                citation_ids=tuple(citation_ids),
            )
        )
        trusted.append(
            TrustedModelVisibleParagraph(
                paragraph_id=paragraph_id,
                text=raw,
                source=source,
                segments=_model_segments(
                    raw,
                    source_start=start,
                    line_map=line_map,
                    protected=local_urls,
                ),
                citation_ids=tuple(citation_ids),
                citation_urls=tuple(citation_urls),
            )
        )

    document = ParsedDocument(
        path=safe_id,
        content_sha256=content_hash,
        line_count=line_map.line_count,
        paragraphs=tuple(paragraphs),
        citations=tuple(citations),
    )
    return PlainTextParsedSource(
        document=document,
        model_visible_paragraphs=tuple(trusted),
        protected_spans=tuple(protected_spans),
    )


def _bounded_ranges(value: str, max_chars: int) -> tuple[tuple[int, int], ...]:
    ranges = []
    cursor = 0
    while cursor < len(value):
        while cursor < len(value) and value[cursor].isspace():
            cursor += 1
        if cursor >= len(value):
            break
        hard_end = min(cursor + max_chars, len(value))
        cut = hard_end
        if hard_end < len(value):
            candidates = (
                value.rfind("\n", cursor + 1, hard_end + 1),
                value.rfind(" ", cursor + 1, hard_end + 1),
                value.rfind("\t", cursor + 1, hard_end + 1),
            )
            boundary = max(candidates)
            if boundary > cursor:
                cut = boundary
        end = cut
        while end > cursor and value[end - 1].isspace():
            end -= 1
        if end > cursor:
            ranges.append((cursor, end))
        cursor = max(cut, cursor + 1)
    return tuple(ranges)


def _source_id(reference_id: str, reference_sha256: str) -> str:
    identity = f"{reference_id}\0{reference_sha256}".encode()
    return "s_local_" + hashlib.sha256(identity).hexdigest()[:12]


def _local_url(reference_id: str) -> str:
    return "local-reference://" + quote(reference_id, safe="")


def _document_chunks(
    text: str,
    bundle: ParsedDocumentWithSourceMap,
    *,
    source: SourceMetadata,
    max_chars: int,
) -> tuple[EvidenceChunk, ...]:
    line_map = _LineMap.build(text, bundle.document.path)
    chunks = []
    for paragraph in bundle.document.paragraphs:
        paragraph_start = paragraph.source.offset_start
        paragraph_end = paragraph.source.offset_end
        if text[paragraph_start:paragraph_end] != paragraph.raw_text:
            raise LocalEvidenceError(
                "parsed reference paragraph lost its exact source span"
            )
        for local_start, local_end in _bounded_ranges(paragraph.raw_text, max_chars):
            start = paragraph_start + local_start
            end = paragraph_start + local_end
            span = line_map.span(start, end)
            chunks.append(
                EvidenceChunk(
                    source_id=source.source_id,
                    url=source.url,
                    text=text[start:end],
                    heading_path=bounded_heading_path(paragraph.heading_path),
                    locator=(
                        f"{bundle.document.path}:{span.line_start}-{span.line_end}"
                    ),
                    char_start=start,
                    char_end=end,
                )
            )
    return tuple(chunks)


@dataclass(frozen=True)
class _ReferenceEntry:
    summary: LocalReferenceSummary
    retriever: LexicalRetriever


class LocalReferenceIndex:
    """Read-only, bounded lexical retrieval over independently hashed sources."""

    def __init__(
        self,
        references: Iterable[LocalReferenceInput],
        *,
        max_chunk_chars: int = LOCAL_REFERENCE_CHUNK_CHARS,
        observed_at: datetime = _LOCAL_REFERENCE_EPOCH,
    ) -> None:
        if (
            isinstance(max_chunk_chars, bool)
            or not isinstance(max_chunk_chars, int)
            or max_chunk_chars <= 0
            or max_chunk_chars > EVIDENCE_EXACT_MAX_CHARS
        ):
            raise LocalEvidenceError(
                "max_chunk_chars must fit the evidence size limit"
            )
        if observed_at.tzinfo is None or observed_at.utcoffset() is None:
            raise LocalEvidenceError("observed_at must be timezone-aware")

        entries = []
        seen = set()
        for reference in references:
            if reference.reference_id in seen:
                raise LocalEvidenceError("local reference IDs must be unique")
            seen.add(reference.reference_id)
            entries.append(
                self._read_reference(
                    reference,
                    max_chunk_chars=max_chunk_chars,
                    observed_at=observed_at,
                )
            )
        self._entries = tuple(entries)

    @staticmethod
    def _read_reference(
        reference: LocalReferenceInput,
        *,
        max_chunk_chars: int,
        observed_at: datetime,
    ) -> _ReferenceEntry:
        path = reference.path
        if path.is_symlink():
            raise LocalEvidenceError(
                f"reference {reference.reference_id} must not be a symlink"
            )
        if not path.is_file():
            raise LocalEvidenceError(
                f"reference {reference.reference_id} must be a regular file"
            )
        suffix = path.suffix.casefold()
        if suffix not in {".md", ".markdown", ".txt"}:
            raise LocalEvidenceError(
                f"reference {reference.reference_id} has an unsupported type"
            )
        try:
            payload = path.read_bytes()
        except OSError:
            raise LocalEvidenceError(
                f"reference {reference.reference_id} could not be read"
            ) from None
        if len(payload) > LOCAL_REFERENCE_MAX_BYTES:
            raise LocalEvidenceError(
                f"reference {reference.reference_id} exceeds the size limit"
            )
        actual_sha256 = hashlib.sha256(payload).hexdigest()
        if actual_sha256 != reference.reference_sha256:
            raise LocalEvidenceError(
                f"reference {reference.reference_id} SHA-256 changed"
            )
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError:
            raise LocalEvidenceError(
                f"reference {reference.reference_id} is not valid UTF-8"
            ) from None

        source_id = _source_id(reference.reference_id, reference.reference_sha256)
        media_type: Literal["text/markdown", "text/plain"]
        if suffix in {".md", ".markdown"}:
            media_type = "text/markdown"
            bundle: ParsedDocumentWithSourceMap = parse_markdown_with_source_map(
                text, reference.reference_id
            )
            protected_spans: tuple[LocalProtectedSpan, ...] = ()
        else:
            media_type = "text/plain"
            plain_bundle = parse_plain_text_with_source_map(
                text, reference.reference_id
            )
            bundle = plain_bundle
            protected_spans = plain_bundle.protected_spans

        source = SourceMetadata(
            source_id=source_id,
            url=_local_url(reference.reference_id),
            title=reference.reference_id,
            retrieved_at=observed_at,
            content_hash=reference.reference_sha256,
            mime_type=media_type,
            status="ok",
        )
        chunks = _document_chunks(
            text,
            bundle,
            source=source,
            max_chars=max_chunk_chars,
        )
        summary = LocalReferenceSummary(
            reference_id=reference.reference_id,
            reference_sha256=reference.reference_sha256,
            line_count=bundle.document.line_count,
            media_type=media_type,
            source=source,
            protected_spans=protected_spans,
        )
        return _ReferenceEntry(
            summary=summary,
            retriever=LexicalRetriever(chunks, neighbor_window=0),
        )

    @property
    def references(self) -> tuple[LocalReferenceSummary, ...]:
        return tuple(entry.summary for entry in self._entries)

    @property
    def protected_spans(self) -> tuple[LocalProtectedSpan, ...]:
        return tuple(
            span for entry in self._entries for span in entry.summary.protected_spans
        )

    def lookup(
        self, claim: AtomicClaim, *, top_k: int = 5
    ) -> tuple[LocalEvidenceResult, ...]:
        """Return at most ``top_k`` exact excerpts per relevant reference."""

        if top_k <= 0:
            return ()
        limit = min(top_k, LOCAL_EVIDENCE_MAX_TOP_K)
        results = []
        for entry in self._entries:
            evidence = tuple(
                item
                for item in entry.retriever.search(claim.text, top_k=limit)
                if item.score > 1e-9
            )
            if not evidence:
                continue
            results.append(
                LocalEvidenceResult(
                    source=entry.summary.source,
                    evidence=evidence,
                    reference_id=entry.summary.reference_id,
                    reference_sha256=entry.summary.reference_sha256,
                )
            )
        return tuple(results)


__all__ = [
    "LOCAL_EVIDENCE_MAX_TOP_K",
    "LOCAL_REFERENCE_CHUNK_CHARS",
    "LOCAL_REFERENCE_MAX_BYTES",
    "LocalEvidenceError",
    "LocalEvidenceResult",
    "LocalProtectedSpan",
    "LocalReferenceIndex",
    "LocalReferenceInput",
    "LocalReferenceSummary",
    "PlainTextParsedSource",
    "parse_plain_text_with_source_map",
]
