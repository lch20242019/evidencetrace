"""Deterministic Markdown structure, citation, and suppression parsing.

This module deliberately stops at source mapping.  It performs no fetching,
claim extraction, or evidence judgement.
"""

from __future__ import annotations

import hashlib
import html
import re
from bisect import bisect_right
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from markdown_it import MarkdownIt
from markdown_it.token import Token

from evidencetrace.models import (
    CitationDefinition,
    CitationKind,
    CitationOccurrence,
    DefinitionKind,
    DiagnosticSeverity,
    MarkdownParagraph,
    ModelTextSourceKind,
    ModelTextSourceSegment,
    ParagraphKind,
    ParsedDocument,
    ParsedDocumentWithSourceMap,
    ParseDiagnostic,
    SourceSpan,
    SuppressionDirective,
    SuppressionScope,
    TargetKind,
    TrustedModelVisibleParagraph,
)

_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_DIRECTIVE_RE = re.compile(
    r"^\s*evidencetrace:\s*([a-z_-]+)\b(?P<attributes>.*)\s*$", re.DOTALL
)
_REASON_RE = re.compile(r"\breason\s*=\s*(?:\"([^\"]*)\"|'([^']*)')", re.DOTALL)
_FOOTNOTE_DEFINITION_RE = re.compile(r"^[ \t]{0,3}\[\^([^]\n]+)\]:[ \t]*(.*)$")
_REFERENCE_LABEL_RE = re.compile(r"^[ \t]{0,3}\[([^]^][^]]*)\]:")
_AUTOLINK_RE = re.compile(
    r"<((?:https?://|mailto:)[^ <>\n]+|"
    r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@"
    r"[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?)>"
)
_BARE_URL_RE = re.compile(r"https?://[^\s<>\"']+")
_INLINE_CODE_IDENTIFIER_RE = re.compile(
    r"[A-Za-z][A-Za-z0-9]*(?:[._:-][A-Za-z0-9]+)+\Z"
)
_INLINE_CODE_URL_SCHEME_RE = re.compile(
    r"(?:https?|ftp|file|data|mailto):", re.IGNORECASE
)


@dataclass(frozen=True)
class _LineIndex:
    text: str
    path: str
    starts: tuple[int, ...]

    @classmethod
    def build(cls, text: str, path: str) -> _LineIndex:
        starts = [0]
        starts.extend(match.end() for match in re.finditer("\n", text))
        return cls(text=text, path=path, starts=tuple(starts))

    @property
    def line_count(self) -> int:
        if not self.text:
            return 0
        return len(self.starts) - int(self.text.endswith("\n"))

    def line_offsets(self, start_line: int, end_line: int) -> tuple[int, int]:
        """Return offsets for a zero-based, end-exclusive block line range."""

        start = self.starts[start_line]
        end = self.starts[end_line] if end_line < len(self.starts) else len(self.text)
        while end > start and self.text[end - 1] in "\r\n":
            end -= 1
        return start, end

    def physical_line_offsets(self, line: int) -> tuple[int, int]:
        end_line = line + 1
        return self.line_offsets(line, end_line)

    def span(self, start: int, end: int) -> SourceSpan:
        if not (0 <= start < end <= len(self.text)):
            raise ValueError(f"invalid non-empty source slice [{start}, {end})")
        start_line = bisect_right(self.starts, start) - 1
        final = end - 1
        end_line = bisect_right(self.starts, final) - 1
        return SourceSpan(
            file=self.path,
            line_start=start_line + 1,
            line_end=end_line + 1,
            column_start=start - self.starts[start_line] + 1,
            column_end=final - self.starts[end_line] + 1,
            offset_start=start,
            offset_end=end,
        )


@dataclass
class _DefinitionDraft:
    kind: DefinitionKind
    key: str
    raw: str
    start: int
    end: int
    destinations: tuple[str, ...] = ()
    definition_id: str = ""


@dataclass
class _CitationDraft:
    kind: CitationKind
    raw: str
    label: str | None
    reference_key: str | None
    destination: str | None
    target_kind: TargetKind | None
    resolved_urls: tuple[str, ...]
    definition_id: str | None
    start: int
    end: int
    resolved: bool
    paragraph_index: int = -1
    citation_id: str = ""


@dataclass
class _ParagraphDraft:
    kind: ParagraphKind
    start: int
    end: int
    raw: str
    plain: str
    plain_segments: tuple[ModelTextSourceSegment, ...]
    heading_path: tuple[str, ...]
    citation_ids: list[str] = field(default_factory=list)
    suppression_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _MappedCharacter:
    value: str
    line_start: int
    line_end: int
    split_protected: bool = False
    kind: ModelTextSourceKind = "plain"


def _inline_code_source_kind(value: str) -> ModelTextSourceKind:
    if (
        not value
        or value != value.strip()
        or any(character.isspace() for character in value)
        or _INLINE_CODE_URL_SCHEME_RE.match(value)
    ):
        return "inline_code"
    if _INLINE_CODE_IDENTIFIER_RE.fullmatch(value):
        return "inline_code_identifier"
    return "inline_code"


def _normalize_key(value: str) -> str:
    value = re.sub(r"\\([!\"#$%&'()*+,\-./:;<=>?@[\\\]^_`{|}~])", r"\1", value)
    return " ".join(html.unescape(value).split()).casefold()


def _unescape_destination(value: str) -> str:
    value = re.sub(r"\\([!\"#$%&'()*+,\-./:;<=>?@[\\\]^_`{|}~])", r"\1", value)
    return html.unescape(value)


def _target_kind(destination: str) -> TargetKind:
    lowered = destination.casefold()
    if lowered.startswith(("http://", "https://")):
        return TargetKind.WEB
    if lowered.startswith("mailto:") or (
        "@" in destination and ":" not in destination and "/" not in destination
    ):
        return TargetKind.EMAIL
    if destination.startswith("#"):
        return TargetKind.ANCHOR
    if ":" not in destination.split("/", 1)[0]:
        return TargetKind.RELATIVE
    return TargetKind.OTHER


def _is_escaped(value: str, index: int) -> bool:
    backslashes = 0
    index -= 1
    while index >= 0 and value[index] == "\\":
        backslashes += 1
        index -= 1
    return backslashes % 2 == 1


def _matching(value: str, start: int, opening: str, closing: str) -> int | None:
    depth = 0
    for index in range(start, len(value)):
        if _is_escaped(value, index):
            continue
        if value[index] == opening:
            depth += 1
        elif value[index] == closing:
            depth -= 1
            if depth == 0:
                return index
    return None


def _mark(mask: list[bool], start: int, end: int) -> None:
    mask[start:end] = [True] * (end - start)


def _initial_inline_mask(value: str) -> list[bool]:
    mask = [False] * len(value)
    for match in _COMMENT_RE.finditer(value):
        _mark(mask, match.start(), match.end())

    index = 0
    while index < len(value):
        if value[index] != "`" or _is_escaped(value, index):
            index += 1
            continue
        run_end = index + 1
        while run_end < len(value) and value[run_end] == "`":
            run_end += 1
        marker = value[index:run_end]
        close = value.find(marker, run_end)
        if close < 0:
            index = run_end
            continue
        _mark(mask, index, close + len(marker))
        index = close + len(marker)

    index = 0
    while index < len(value) - 1:
        if value[index : index + 2] != "![" or _is_escaped(value, index):
            index += 1
            continue
        label_end = _matching(value, index + 1, "[", "]")
        if label_end is None:
            index += 2
            continue
        end = label_end + 1
        if end < len(value) and value[end] == "(":
            destination_end = _matching(value, end, "(", ")")
            if destination_end is not None:
                end = destination_end + 1
        elif end < len(value) and value[end] == "[":
            reference_end = _matching(value, end, "[", "]")
            if reference_end is not None:
                end = reference_end + 1
        _mark(mask, index, end)
        index = end

    # Escaping the opening bracket renders a would-be link literally. Mask
    # its destination too, otherwise the URL would be emitted as a bare URL.
    index = 0
    while index < len(value):
        if value[index] != "[" or not _is_escaped(value, index):
            index += 1
            continue
        label_end = value.find("]", index + 1)
        while label_end >= 0 and _is_escaped(value, label_end):
            label_end = value.find("]", label_end + 1)
        if label_end < 0:
            index += 1
            continue
        end = label_end + 1
        if end < len(value) and value[end] == "(":
            destination_end = _matching(value, end, "(", ")")
            if destination_end is not None:
                end = destination_end + 1
        elif end < len(value) and value[end] == "[":
            reference_end = _matching(value, end, "[", "]")
            if reference_end is not None:
                end = reference_end + 1
        _mark(mask, index, end)
        index = end
    return mask


def _inline_destination(value: str) -> str | None:
    content = value.strip()
    if not content:
        return None
    if content.startswith("<"):
        close = content.find(">", 1)
        return _unescape_destination(content[1:close]) if close > 0 else None
    index = 0
    nested = 0
    while index < len(content):
        char = content[index]
        if _is_escaped(content, index):
            index += 1
        elif char == "(":
            nested += 1
        elif char == ")" and nested:
            nested -= 1
        elif char.isspace() and nested == 0:
            break
        index += 1
    return _unescape_destination(content[:index]) or None


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


def _scan_citations(
    value: str,
    base_offset: int,
    references: dict[str, _DefinitionDraft],
    footnotes: dict[str, _DefinitionDraft],
) -> list[_CitationDraft]:
    mask = _initial_inline_mask(value)
    found: list[_CitationDraft] = []

    index = 0
    while index < len(value):
        if value[index] != "[" or mask[index] or _is_escaped(value, index):
            index += 1
            continue
        label_end = _matching(value, index, "[", "]")
        if label_end is None or any(mask[index : label_end + 1]):
            index += 1
            continue
        label = value[index + 1 : label_end]
        end = label_end + 1
        kind: CitationKind | None = None
        key: str | None = None
        destination: str | None = None
        definition: _DefinitionDraft | None = None

        if label.startswith("^") and len(label) > 1:
            kind = CitationKind.FOOTNOTE
            key = _normalize_key(label[1:])
            definition = footnotes.get(key)
        elif end < len(value) and value[end] == "(":
            close = _matching(value, end, "(", ")")
            if close is not None:
                kind = CitationKind.INLINE_LINK
                destination = _inline_destination(value[end + 1 : close])
                end = close + 1
        elif end < len(value) and value[end] == "[":
            close = _matching(value, end, "[", "]")
            if close is not None:
                raw_key = value[end + 1 : close] or label
                key = _normalize_key(raw_key)
                kind = CitationKind.REFERENCE_LINK
                definition = references.get(key)
                end = close + 1
        else:
            key = _normalize_key(label)
            definition = references.get(key)
            if definition is not None:
                kind = CitationKind.REFERENCE_LINK

        if kind is None:
            index += 1
            continue
        if definition is not None and kind is CitationKind.REFERENCE_LINK:
            destination = (
                definition.destinations[0] if definition.destinations else None
            )
        resolved_urls = (
            definition.destinations
            if definition is not None
            else ((destination,) if destination is not None else ())
        )
        target = _target_kind(destination) if destination is not None else None
        found.append(
            _CitationDraft(
                kind=kind,
                raw=value[index:end],
                label=label[1:] if kind is CitationKind.FOOTNOTE else label,
                reference_key=key,
                destination=destination,
                target_kind=target,
                resolved_urls=resolved_urls,
                definition_id=definition.definition_id if definition else None,
                start=base_offset + index,
                end=base_offset + end,
                resolved=bool(resolved_urls),
            )
        )
        _mark(mask, index, end)
        index = end

    for match in _AUTOLINK_RE.finditer(value):
        if any(mask[match.start() : match.end()]):
            continue
        raw_destination = match.group(1)
        destination = (
            raw_destination
            if ":" in raw_destination.split("@", 1)[0]
            else f"mailto:{raw_destination}"
        )
        found.append(
            _CitationDraft(
                kind=CitationKind.AUTOLINK,
                raw=match.group(0),
                label=raw_destination,
                reference_key=None,
                destination=destination,
                target_kind=_target_kind(destination),
                resolved_urls=(destination,),
                definition_id=None,
                start=base_offset + match.start(),
                end=base_offset + match.end(),
                resolved=True,
            )
        )
        _mark(mask, match.start(), match.end())

    for match in _BARE_URL_RE.finditer(value):
        start = match.start()
        end = _trim_bare_url(value, start, match.end())
        if end == start or any(mask[start:end]):
            continue
        destination = value[start:end]
        found.append(
            _CitationDraft(
                kind=CitationKind.BARE_URL,
                raw=destination,
                label=None,
                reference_key=None,
                destination=destination,
                target_kind=TargetKind.WEB,
                resolved_urls=(destination,),
                definition_id=None,
                start=base_offset + start,
                end=base_offset + end,
                resolved=True,
            )
        )
        _mark(mask, start, end)
    return sorted(found, key=lambda item: (item.start, item.end))


def _append_mapped_text(
    mapped: list[_MappedCharacter],
    value: str,
    *,
    current_line: int,
    paragraph_line_start: int,
    paragraph_line_end: int,
    conservative: bool = False,
    split_protected: bool = False,
    kind: ModelTextSourceKind = "plain",
) -> int:
    line = current_line
    for character in value:
        if conservative:
            line_start, line_end = paragraph_line_start, paragraph_line_end
        elif character in {"\r", "\n"}:
            line_start = line
            line_end = min(line + 1, paragraph_line_end)
        else:
            line_start = line_end = line
        mapped.append(
            _MappedCharacter(
                character,
                line_start,
                line_end,
                split_protected=split_protected,
                kind=kind,
            )
        )
        if character == "\n":
            line = min(line + 1, paragraph_line_end)
    return line


def _remove_mapped_matches(
    mapped: list[_MappedCharacter], pattern: re.Pattern[str]
) -> list[_MappedCharacter]:
    value = "".join(item.value for item in mapped)
    removed = [False] * len(mapped)
    for match in pattern.finditer(value):
        removed[match.start() : match.end()] = [True] * (match.end() - match.start())
    return [item for index, item in enumerate(mapped) if not removed[index]]


def _collapse_mapped_whitespace(
    mapped: list[_MappedCharacter],
) -> list[_MappedCharacter]:
    collapsed: list[_MappedCharacter] = []
    whitespace: list[_MappedCharacter] = []
    for item in mapped:
        if item.value.isspace():
            whitespace.append(item)
            continue
        if collapsed and whitespace:
            kinds = {value.kind for value in whitespace}
            collapsed.append(
                _MappedCharacter(
                    " ",
                    min(value.line_start for value in whitespace),
                    max(value.line_end for value in whitespace),
                    split_protected=any(value.split_protected for value in whitespace),
                    kind=kinds.pop() if len(kinds) == 1 else "plain",
                )
            )
        whitespace.clear()
        collapsed.append(item)
    return collapsed


def _mapped_segments(
    mapped: list[_MappedCharacter],
) -> tuple[ModelTextSourceSegment, ...]:
    if not mapped:
        return ()
    segments: list[ModelTextSourceSegment] = []
    start = 0
    previous = mapped[0]
    for index, item in enumerate(mapped[1:], start=1):
        if (
            item.line_start == previous.line_start
            and item.line_end == previous.line_end
            and item.split_protected == previous.split_protected
            and item.kind == previous.kind
        ):
            continue
        segments.append(
            ModelTextSourceSegment(
                model_start=start,
                model_end=index,
                source_line_start=previous.line_start,
                source_line_end=previous.line_end,
                split_protected=previous.split_protected,
                kind=previous.kind,
            )
        )
        start, previous = index, item
    segments.append(
        ModelTextSourceSegment(
            model_start=start,
            model_end=len(mapped),
            source_line_start=previous.line_start,
            source_line_end=previous.line_end,
            split_protected=previous.split_protected,
            kind=previous.kind,
        )
    )
    return tuple(segments)


def _token_model_visible_text(
    token: Token,
) -> tuple[str, tuple[ModelTextSourceSegment, ...]]:
    paragraph_line_start = token.map[0] + 1 if token.map else 1
    paragraph_line_end = token.map[1] if token.map else paragraph_line_start
    current_line = paragraph_line_start
    line_is_uncertain = False
    link_depth = 0
    mapped: list[_MappedCharacter] = []
    for child in token.children or ():
        if child.type == "link_open":
            link_depth += 1
        elif child.type == "link_close":
            link_depth = max(0, link_depth - 1)
        elif child.type == "text":
            current_line = _append_mapped_text(
                mapped,
                child.content,
                current_line=current_line,
                paragraph_line_start=paragraph_line_start,
                paragraph_line_end=paragraph_line_end,
                conservative=line_is_uncertain,
                split_protected=link_depth > 0,
                kind="link_label" if link_depth > 0 else "plain",
            )
        elif child.type in {"softbreak", "hardbreak"}:
            next_line = min(current_line + 1, paragraph_line_end)
            mapped.append(
                _MappedCharacter(
                    " ",
                    (paragraph_line_start if line_is_uncertain else current_line),
                    paragraph_line_end if line_is_uncertain else next_line,
                    split_protected=link_depth > 0,
                    kind="link_label" if link_depth > 0 else "plain",
                )
            )
            current_line = next_line
        elif child.type == "code_inline":
            potentially_multiline = paragraph_line_start != paragraph_line_end and any(
                character.isspace() for character in child.content
            )
            current_line = _append_mapped_text(
                mapped,
                child.content,
                current_line=current_line,
                paragraph_line_start=paragraph_line_start,
                paragraph_line_end=paragraph_line_end,
                conservative=line_is_uncertain or potentially_multiline,
                split_protected=True,
                kind=(
                    "link_label"
                    if link_depth > 0
                    else _inline_code_source_kind(child.content)
                ),
            )
            line_is_uncertain = line_is_uncertain or potentially_multiline
        elif child.type in {"image", "html_inline"}:
            if "\n" in child.content:
                current_line = min(
                    current_line + child.content.count("\n"),
                    paragraph_line_end,
                )
            elif child.type == "image" and paragraph_line_start != paragraph_line_end:
                line_is_uncertain = True

    mapped = _remove_mapped_matches(mapped, re.compile(r"\[\^[^]]+\]"))
    mapped = _remove_mapped_matches(mapped, _BARE_URL_RE)
    mapped = _collapse_mapped_whitespace(mapped)
    return "".join(item.value for item in mapped), _mapped_segments(mapped)


def _token_plain_text(token: Token) -> str:
    return _token_model_visible_text(token)[0]


def _code_ranges(tokens: list[Token], index: _LineIndex) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for token in tokens:
        if token.type in {"fence", "code_block"} and token.map:
            start, end = index.line_offsets(token.map[0], token.map[1])
            ranges.append((start, end))
    return ranges


def _inside(offset: int, ranges: list[tuple[int, int]]) -> bool:
    return any(start <= offset < end for start, end in ranges)


def _footnote_ranges(
    text: str, index: _LineIndex, code: list[tuple[int, int]]
) -> list[_DefinitionDraft]:
    drafts: list[_DefinitionDraft] = []
    line = 0
    while line < index.line_count:
        start, end = index.physical_line_offsets(line)
        if _inside(start, code):
            line += 1
            continue
        match = _FOOTNOTE_DEFINITION_RE.match(text[start:end])
        if match is None:
            line += 1
            continue
        final_line = line + 1
        while final_line < index.line_count:
            next_start, next_end = index.physical_line_offsets(final_line)
            next_text = text[next_start:next_end]
            if re.match(r"^(?: {4}|\t)", next_text):
                final_line += 1
                continue
            if not next_text.strip() and final_line + 1 < index.line_count:
                after_start, after_end = index.physical_line_offsets(final_line + 1)
                if re.match(r"^(?: {4}|\t)", text[after_start:after_end]):
                    final_line += 1
                    continue
            break
        definition_start, definition_end = index.line_offsets(line, final_line)
        drafts.append(
            _DefinitionDraft(
                kind=DefinitionKind.FOOTNOTE,
                key=_normalize_key(match.group(1)),
                raw=text[definition_start:definition_end],
                start=definition_start,
                end=definition_end,
            )
        )
        line = final_line
    return drafts


def _definitions(
    text: str,
    env: dict[str, Any],
    tokens: list[Token],
    index: _LineIndex,
) -> tuple[
    list[CitationDefinition],
    dict[str, _DefinitionDraft],
    dict[str, _DefinitionDraft],
    list[tuple[int, int]],
    list[ParseDiagnostic],
]:
    code = _code_ranges(tokens, index)
    drafts: list[_DefinitionDraft] = []
    del env  # A line scan preserves duplicates omitted by markdown-it's env.
    for line in range(index.line_count):
        start, end = index.physical_line_offsets(line)
        if _inside(start, code):
            continue
        match = _REFERENCE_LABEL_RE.match(text[start:end])
        if match is None:
            continue
        destination = _inline_destination(text[start + match.end() : end])
        if destination is None:
            continue
        drafts.append(
            _DefinitionDraft(
                kind=DefinitionKind.REFERENCE,
                key=_normalize_key(match.group(1)),
                raw=text[start:end],
                start=start,
                end=end,
                destinations=(destination,),
            )
        )
    drafts.extend(_footnote_ranges(text, index, code))
    drafts.sort(key=lambda item: (item.start, item.end))
    for number, draft in enumerate(drafts, 1):
        draft.definition_id = f"def_{number:04d}"

    references: dict[str, _DefinitionDraft] = {}
    footnotes: dict[str, _DefinitionDraft] = {}
    diagnostics: list[ParseDiagnostic] = []
    for draft in drafts:
        definition_map = (
            references if draft.kind is DefinitionKind.REFERENCE else footnotes
        )
        if draft.key in definition_map:
            diagnostics.append(
                ParseDiagnostic(
                    code="duplicate_definition",
                    message=f"Duplicate {draft.kind.value} definition: {draft.key}",
                    severity=DiagnosticSeverity.WARNING,
                    source=index.span(draft.start, draft.end),
                )
            )
        else:
            definition_map[draft.key] = draft

    for draft in drafts:
        if draft.kind is not DefinitionKind.FOOTNOTE:
            continue
        nested = _scan_citations(draft.raw, draft.start, references, {})
        destinations: list[str] = []
        for citation in nested:
            if citation.kind is CitationKind.FOOTNOTE:
                continue
            for destination in citation.resolved_urls:
                if destination not in destinations:
                    destinations.append(destination)
        draft.destinations = tuple(destinations)

    models = [
        CitationDefinition(
            definition_id=draft.definition_id,
            kind=draft.kind,
            normalized_key=draft.key,
            raw_text=draft.raw,
            destinations=draft.destinations,
            source=index.span(draft.start, draft.end),
        )
        for draft in drafts
    ]
    excluded = [(draft.start, draft.end) for draft in drafts]
    return models, references, footnotes, excluded, diagnostics


def _overlaps(start: int, end: int, ranges: list[tuple[int, int]]) -> bool:
    return any(
        start < other_end and other_start < end for other_start, other_end in ranges
    )


def _paragraphs(
    text: str,
    tokens: list[Token],
    index: _LineIndex,
    excluded: list[tuple[int, int]],
    references: dict[str, _DefinitionDraft],
    footnotes: dict[str, _DefinitionDraft],
) -> tuple[list[_ParagraphDraft], list[_CitationDraft]]:
    paragraphs: list[_ParagraphDraft] = []
    citations: list[_CitationDraft] = []
    headings: list[str] = []
    list_depth = 0
    blockquote_depth = 0
    token_index = 0
    while token_index < len(tokens):
        token = tokens[token_index]
        if token.type in {"bullet_list_open", "ordered_list_open"}:
            list_depth += 1
        elif token.type in {"bullet_list_close", "ordered_list_close"}:
            list_depth -= 1
        elif token.type == "blockquote_open":
            blockquote_depth += 1
        elif token.type == "blockquote_close":
            blockquote_depth -= 1
        elif token.type == "heading_open" and token.map:
            inline = tokens[token_index + 1]
            level = int(token.tag[1:])
            title = _token_plain_text(inline)
            headings = headings[: level - 1]
            headings.append(title)
        elif token.type == "paragraph_open" and token.map:
            inline = tokens[token_index + 1]
            start, end = index.line_offsets(token.map[0], token.map[1])
            if start < end and not _overlaps(start, end, excluded):
                kind = ParagraphKind.PARAGRAPH
                if list_depth:
                    kind = ParagraphKind.LIST_ITEM
                elif blockquote_depth:
                    kind = ParagraphKind.BLOCKQUOTE
                paragraph_index = len(paragraphs)
                paragraph_citations = _scan_citations(
                    text[start:end], start, references, footnotes
                )
                for citation in paragraph_citations:
                    citation.paragraph_index = paragraph_index
                plain, plain_segments = _token_model_visible_text(inline)
                if plain or paragraph_citations:
                    paragraphs.append(
                        _ParagraphDraft(
                            kind=kind,
                            start=start,
                            end=end,
                            raw=text[start:end],
                            plain=plain,
                            plain_segments=plain_segments,
                            heading_path=tuple(headings),
                        )
                    )
                    citations.extend(paragraph_citations)
        token_index += 1
    return paragraphs, citations


def _valid_directive(comment: str) -> tuple[str | None, str | None]:
    inner = comment[4:-3]
    match = _DIRECTIVE_RE.match(inner)
    if match is None:
        return None, None
    action = match.group(1)
    reason_match = _REASON_RE.search(match.group("attributes"))
    reason = (
        next(
            (group.strip() for group in reason_match.groups() if group is not None), ""
        )
        if reason_match
        else None
    )
    return action, reason


def _suppressions(
    text: str,
    index: _LineIndex,
    code: list[tuple[int, int]],
    paragraphs: list[_ParagraphDraft],
) -> tuple[list[SuppressionDirective], list[ParseDiagnostic]]:
    applicable: list[tuple[int, int, str, SuppressionScope, int, int, int | None]] = []
    diagnostics: list[ParseDiagnostic] = []
    standalone_spans: list[tuple[int, int]] = []
    comments = list(_COMMENT_RE.finditer(text))
    for comment in comments:
        if _inside(comment.start(), code):
            continue
        inner = comment.group(0)[4:-3].lstrip()
        if not inner.startswith("evidencetrace:"):
            continue
        action, reason = _valid_directive(comment.group(0))
        source = index.span(comment.start(), comment.end())
        if action != "ignore":
            diagnostics.append(
                ParseDiagnostic(
                    code="unsupported_suppression_action",
                    message="Only the evidencetrace 'ignore' action is supported",
                    severity=DiagnosticSeverity.WARNING,
                    source=source,
                )
            )
            continue
        if reason is None or not reason:
            diagnostics.append(
                ParseDiagnostic(
                    code="invalid_suppression_reason",
                    message="Suppression requires a non-empty reason",
                    severity=DiagnosticSeverity.WARNING,
                    source=source,
                )
            )
            continue
        line = source.line_start - 1
        line_start, line_end = index.physical_line_offsets(line)
        before = text[line_start : comment.start()]
        after = text[comment.end() : line_end]
        if before.strip() or after.strip():
            paragraph_index = next(
                (
                    number
                    for number, paragraph in enumerate(paragraphs)
                    if paragraph.start <= comment.start() < paragraph.end
                ),
                None,
            )
            applicable.append(
                (
                    comment.start(),
                    comment.end(),
                    reason,
                    SuppressionScope.LINE,
                    line_start,
                    line_end,
                    paragraph_index,
                )
            )
            continue
        standalone_spans.append((comment.start(), comment.end()))
        candidate_index = next(
            (
                number
                for number, paragraph in enumerate(paragraphs)
                if paragraph.start > comment.end()
            ),
            None,
        )
        if candidate_index is None:
            diagnostics.append(
                ParseDiagnostic(
                    code="orphan_suppression",
                    message="Suppression is not followed by an auditable paragraph",
                    severity=DiagnosticSeverity.NOTICE,
                    source=source,
                )
            )
            continue
        paragraph = paragraphs[candidate_index]
        between = text[line_end : paragraph.start]
        for other_start, other_end in standalone_spans:
            relative_start = other_start - line_end
            relative_end = other_end - line_end
            if 0 <= relative_start < relative_end <= len(between):
                between = (
                    between[:relative_start]
                    + " " * (relative_end - relative_start)
                    + between[relative_end:]
                )
        if between.strip():
            diagnostics.append(
                ParseDiagnostic(
                    code="orphan_suppression",
                    message="A non-paragraph block interrupts suppression binding",
                    severity=DiagnosticSeverity.NOTICE,
                    source=source,
                )
            )
            continue
        applicable.append(
            (
                comment.start(),
                comment.end(),
                reason,
                SuppressionScope.PARAGRAPH,
                paragraph.start,
                paragraph.end,
                candidate_index,
            )
        )

    suppressions: list[SuppressionDirective] = []
    for number, item in enumerate(sorted(applicable), 1):
        start, end, reason, scope, target_start, target_end, paragraph_index = item
        suppression_id = f"sup_{number:04d}"
        suppressions.append(
            SuppressionDirective(
                suppression_id=suppression_id,
                reason=reason,
                scope=scope,
                directive_source=index.span(start, end),
                target_source=index.span(target_start, target_end),
            )
        )
        if paragraph_index is not None:
            paragraphs[paragraph_index].suppression_ids.append(suppression_id)
    return suppressions, diagnostics


def parse_markdown_with_source_map(text: str, path: str) -> ParsedDocumentWithSourceMap:
    """Parse Markdown and retain local model-text-to-source provenance."""

    index = _LineIndex.build(text, path)
    parser = MarkdownIt("commonmark", {"html": True})
    environment: dict[str, Any] = {}
    tokens = parser.parse(text, environment)
    definitions, references, footnotes, excluded, diagnostics = _definitions(
        text, environment, tokens, index
    )
    paragraphs, citation_drafts = _paragraphs(
        text, tokens, index, excluded, references, footnotes
    )
    citation_drafts.sort(key=lambda item: (item.start, item.end))
    citations: list[CitationOccurrence] = []
    for number, draft in enumerate(citation_drafts, 1):
        draft.citation_id = f"cit_{number:04d}"
        paragraphs[draft.paragraph_index].citation_ids.append(draft.citation_id)
        citations.append(
            CitationOccurrence(
                citation_id=draft.citation_id,
                kind=draft.kind,
                raw=draft.raw,
                label=draft.label,
                reference_key=draft.reference_key,
                destination=draft.destination,
                target_kind=draft.target_kind,
                resolved_urls=draft.resolved_urls,
                definition_id=draft.definition_id,
                source=index.span(draft.start, draft.end),
                resolved=draft.resolved,
            )
        )
        if not draft.resolved and draft.kind in {
            CitationKind.REFERENCE_LINK,
            CitationKind.FOOTNOTE,
        }:
            diagnostics.append(
                ParseDiagnostic(
                    code=(
                        "unresolved_footnote"
                        if draft.kind is CitationKind.FOOTNOTE
                        else "unresolved_reference"
                    ),
                    message=f"Citation definition not found: {draft.reference_key}",
                    severity=DiagnosticSeverity.WARNING,
                    source=index.span(draft.start, draft.end),
                )
            )

    suppressions, suppression_diagnostics = _suppressions(
        text, index, _code_ranges(tokens, index), paragraphs
    )
    diagnostics.extend(suppression_diagnostics)
    diagnostics.sort(
        key=lambda item: (
            item.source.offset_start if item.source else len(text),
            item.code,
        )
    )
    citations_by_id = {citation.citation_id: citation for citation in citations}
    paragraph_models: list[MarkdownParagraph] = []
    model_visible_paragraphs: list[TrustedModelVisibleParagraph] = []
    for number, paragraph in enumerate(paragraphs, 1):
        paragraph_id = f"p_{number:04d}"
        source = index.span(paragraph.start, paragraph.end)
        citation_ids = tuple(paragraph.citation_ids)
        citation_urls = tuple(
            url
            for citation_id in citation_ids
            for url in citations_by_id[citation_id].resolved_urls
        )
        paragraph_models.append(
            MarkdownParagraph(
                paragraph_id=paragraph_id,
                kind=paragraph.kind,
                source=source,
                raw_text=paragraph.raw,
                plain_text=paragraph.plain,
                heading_path=paragraph.heading_path,
                citation_ids=citation_ids,
                suppression_ids=tuple(paragraph.suppression_ids),
            )
        )
        model_visible_paragraphs.append(
            TrustedModelVisibleParagraph(
                paragraph_id=paragraph_id,
                text=paragraph.plain,
                source=source,
                segments=paragraph.plain_segments,
                heading_path=paragraph.heading_path,
                citation_ids=citation_ids,
                citation_urls=citation_urls,
            )
        )

    document = ParsedDocument(
        path=path,
        content_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        line_count=index.line_count,
        paragraphs=tuple(paragraph_models),
        citations=tuple(citations),
        definitions=tuple(definitions),
        suppressions=tuple(suppressions),
        diagnostics=tuple(diagnostics),
    )
    return ParsedDocumentWithSourceMap(
        document=document,
        model_visible_paragraphs=tuple(model_visible_paragraphs),
    )


def parse_markdown(text: str, path: str) -> ParsedDocument:
    """Parse Markdown into deterministic, source-mapped Phase 1 contracts."""

    return parse_markdown_with_source_map(text, path).document


def _read_markdown(
    path: str | Path, *, repo_root: str | Path | None
) -> tuple[str, str]:
    file_path = Path(path)
    root = Path.cwd() if repo_root is None else Path(repo_root)
    display_path = file_path
    if file_path.is_absolute():
        try:
            display_path = file_path.resolve().relative_to(root.resolve())
        except ValueError as error:
            raise ValueError(
                "absolute Markdown paths must be inside repo_root"
            ) from error
    return file_path.read_text(encoding="utf-8"), display_path.as_posix()


def parse_markdown_file_with_source_map(
    path: str | Path, *, repo_root: str | Path | None = None
) -> ParsedDocumentWithSourceMap:
    """Read Markdown and retain its local trusted model-visible source map."""

    text, display_path = _read_markdown(path, repo_root=repo_root)
    return parse_markdown_with_source_map(text, display_path)


def parse_markdown_file(
    path: str | Path, *, repo_root: str | Path | None = None
) -> ParsedDocument:
    """Read a UTF-8 Markdown file and preserve its project-relative path."""

    return parse_markdown_file_with_source_map(path, repo_root=repo_root).document


__all__ = [
    "parse_markdown",
    "parse_markdown_file",
    "parse_markdown_file_with_source_map",
    "parse_markdown_with_source_map",
]
