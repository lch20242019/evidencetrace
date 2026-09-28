"""Paragraph-local atomic claim extraction."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal, Self

from pydantic import NonNegativeInt, PositiveInt, model_validator

from evidencetrace.audit_models import (
    MINER_DRAFT_CONTRACT_VERSION,
    MINER_WINDOW_POLICY_VERSION,
    LiveMinedClaimDraft,
    LiveMinerDraftOutput,
    MinerInput,
    MinerOutput,
)
from evidencetrace.checks.deterministic import is_recommendation
from evidencetrace.model_client import ModelClient
from evidencetrace.models import (
    AtomicClaim,
    Checkability,
    ContractModel,
    TrustedModelVisibleParagraph,
)

SENTENCE_BOUNDARY_RE = re.compile(
    r"(?:(?<=[!?;\u3002\uff01\uff1f\uff1b])\s*|(?<=\.)\s+)"
)
OPINION_RE = re.compile(
    r"\b(?:we will|i think|could|might|may)\b",
    re.I,
)
INCOMPLETE_END_RE = re.compile(
    r"\b(?:am|is|are|was|were|be|been|being|to|of|in|on|at|for|from|"
    r"with|by|as|into|than)\s*[.!?;\u3002\uff01\uff1f\uff1b]*$",
    re.I,
)
TOKEN_RE = re.compile(
    r"\d+(?:[.,]\d+)?%|[A-Za-z0-9]+(?:[._][A-Za-z0-9]+)*"
    r"|[\u3400-\u4dbf\u4e00-\u9fff]+"
)
PROTECTED_RE = re.compile(
    r"(?<![A-Za-z0-9])\d+(?:[.,]\d+)?%"
    r"|\bv?\d+(?:\.\d+){1,3}\b"
    r"|\b20\d{2}[-/]\d{1,2}[-/]\d{1,2}\b"
    r"|(?<![A-Za-z0-9])\d+(?:[.,]\d+)?"
    r"|\b(?:no|not|never|without|cannot|doesn't|does\s+not|isn't|is\s+not)\b"
    r"|\b(?:all|only|every|more|less|higher|lower|faster|slower|"
    r"at\s+least|at\s+most)\b"
    r"|\b(?:ms|milliseconds?|seconds?|minutes?|hours?|days?|"
    r"bytes?|kb|mb|gb|tb)\b",
    re.I,
)


MinerScopeCode = Literal[
    "non_source_span",
    "ambiguous_span",
    "missing_protected_token",
    "invalid_claim_fragment",
]
_MINER_SCOPE_CODES: frozenset[str] = frozenset(
    {
        "non_source_span",
        "ambiguous_span",
        "missing_protected_token",
        "invalid_claim_fragment",
    }
)


class MinerScopeError(ValueError):
    """Payload-free rejection of a live Miner paragraph-scope violation."""

    def __init__(
        self,
        code: MinerScopeCode,
        *,
        withheld_exact_draft_count: int = 0,
        protected_occurrences_total: int = 0,
        protected_occurrences_covered: int = 0,
    ) -> None:
        if code not in _MINER_SCOPE_CODES:
            raise ValueError("unsupported Miner scope code")
        counts = (
            withheld_exact_draft_count,
            protected_occurrences_total,
            protected_occurrences_covered,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in counts
        ):
            raise ValueError("Miner scope counters must be non-negative integers")
        if protected_occurrences_covered > protected_occurrences_total:
            raise ValueError("Miner scope coverage cannot exceed its total")
        super().__init__("live Miner output violated paragraph scope")
        self.code = code
        self.withheld_exact_draft_count = withheld_exact_draft_count
        self.protected_occurrences_total = protected_occurrences_total
        self.protected_occurrences_covered = protected_occurrences_covered


@dataclass(frozen=True)
class _ProtectedOccurrence:
    start: int
    end: int
    key: str


@dataclass(frozen=True)
class MinerDraftValidation:
    """Local-only exact offsets and occurrence-aware coverage."""

    offsets: tuple[tuple[int, int], ...]
    protected_occurrences_total: int
    protected_occurrences_covered: int


@dataclass(frozen=True)
class ValidatedMinerOutput:
    """Canonical Miner output plus local validation facts."""

    output: MinerOutput
    validation: MinerDraftValidation


def _claim_type(text: str) -> str:
    if re.search(r"\bv?\d+(?:\.\d+){1,3}\b", text, re.I):
        return "versioned_capability"
    if re.search(r"\b20\d{2}[-/]\d{1,2}[-/]\d{1,2}\b", text):
        return "dated_event"
    if re.search(r"\d+(?:\.\d+)?%?", text):
        return "numeric_measurement"
    return "factual_statement"


def _tokens(value: str) -> set[str]:
    return {match.group(0).casefold() for match in TOKEN_RE.finditer(value)}


def _protected_occurrences(value: str) -> tuple[_ProtectedOccurrence, ...]:
    return tuple(
        _ProtectedOccurrence(
            start=match.start(),
            end=match.end(),
            key=re.sub(r"\s+", " ", match.group(0).casefold()),
        )
        for match in PROTECTED_RE.finditer(value)
    )


def miner_protected_occurrence_count(value: str) -> int:
    """Return only the safe count needed by Controller coverage outcomes."""

    return len(_protected_occurrences(value))


def _lexical_protected_coverage(
    occurrences: tuple[_ProtectedOccurrence, ...],
    drafts: tuple[LiveMinedClaimDraft, ...],
) -> int:
    available = Counter(occurrence.key for occurrence in occurrences)
    emitted = Counter(
        occurrence.key
        for draft in drafts
        for occurrence in _protected_occurrences(draft.text)
    )
    return sum(min(count, emitted[key]) for key, count in available.items())


def _span_protected_coverage(
    occurrences: tuple[_ProtectedOccurrence, ...],
    offsets: tuple[tuple[int, int], ...],
) -> int:
    return sum(
        any(
            start <= occurrence.start and occurrence.end <= end
            for start, end in offsets
        )
        for occurrence in occurrences
    )


def _individually_exact_draft_count(
    paragraph: str,
    drafts: tuple[LiveMinedClaimDraft, ...],
) -> int:
    count = 0
    for draft in drafts:
        if not _is_complete_claim(draft.text):
            continue
        start = paragraph.find(draft.text)
        if start >= 0 and start == paragraph.rfind(draft.text):
            count += 1
    return count


def _fragments(
    value: str,
    *,
    split_protected_ranges: tuple[tuple[int, int], ...] = (),
) -> Iterator[tuple[str, int, int]]:
    """Yield trimmed sentence fragments and their paragraph offsets."""

    start = 0
    for boundary in SENTENCE_BOUNDARY_RE.finditer(value):
        if any(
            protected_start < boundary.start() < protected_end
            for protected_start, protected_end in split_protected_ranges
        ):
            continue
        end = boundary.start()
        raw = value[start:end]
        left = len(raw) - len(raw.lstrip())
        right = len(raw.rstrip())
        if right > left:
            yield raw[left:right], start + left, start + right
        start = boundary.end()
    raw = value[start:]
    left = len(raw) - len(raw.lstrip())
    right = len(raw.rstrip())
    if right > left:
        yield raw[left:right], start + left, start + right


class MinerWindow(ContractModel):
    """One locally owned exact model-visible slice planned before dispatch."""

    window_id: str
    paragraph_id: str
    text: str
    start: NonNegativeInt
    end: PositiveInt
    source_file: str
    source_line_start: PositiveInt
    source_line_end: PositiveInt
    heading_path: tuple[str, ...] = ()
    citation_ids: tuple[str, ...] = ()
    citation_urls: tuple[str, ...] = ()
    inline_code_identifier_count: NonNegativeInt = 0

    @model_validator(mode="after")
    def validate_ranges(self) -> Self:
        if self.end <= self.start or len(self.text) != self.end - self.start:
            raise ValueError("Miner window offsets must exactly bound its text")
        if self.source_line_end < self.source_line_start:
            raise ValueError("Miner window source lines must be ordered")
        return self


class OversizedMinerFragment(ContractModel):
    """A fragment that cannot be safely dispatched under the explicit bound."""

    code: Literal["miner_window_fragment_oversized"] = "miner_window_fragment_oversized"
    window_id: str
    paragraph_id: str
    start: NonNegativeInt
    end: PositiveInt
    source_file: str
    source_line_start: PositiveInt
    source_line_end: PositiveInt
    fragment_length: PositiveInt
    max_window_chars: PositiveInt
    heading_path: tuple[str, ...] = ()
    citation_ids: tuple[str, ...] = ()
    citation_urls: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_oversized_fragment(self) -> Self:
        if self.end <= self.start or self.fragment_length != self.end - self.start:
            raise ValueError("oversized Miner fragment offsets are inconsistent")
        if self.fragment_length <= self.max_window_chars:
            raise ValueError("oversized Miner fragment must exceed its bound")
        if self.source_line_end < self.source_line_start:
            raise ValueError("oversized Miner fragment source lines must be ordered")
        return self


class MinerWindowPlan(ContractModel):
    """Complete deterministic plan for one model-visible paragraph."""

    policy_version: Literal["miner-window-policy-v1"] = MINER_WINDOW_POLICY_VERSION
    paragraph_id: str
    max_window_chars: PositiveInt
    status: Literal["ready", "intentional_empty", "oversized"]
    windows: tuple[MinerWindow, ...] = ()
    oversized_fragments: tuple[OversizedMinerFragment, ...] = ()

    @model_validator(mode="after")
    def validate_plan(self) -> Self:
        planned: list[MinerWindow | OversizedMinerFragment] = [
            *self.windows,
            *self.oversized_fragments,
        ]
        planned.sort(key=lambda item: (item.start, item.end))
        cursor = 0
        window_ids: set[str] = set()
        for item in planned:
            if item.paragraph_id != self.paragraph_id:
                raise ValueError("Miner plan contains a foreign paragraph")
            if item.window_id in window_ids:
                raise ValueError("Miner plan window IDs must be unique")
            window_ids.add(item.window_id)
            if item.start < cursor:
                raise ValueError(
                    "Miner plan ranges must be ordered and non-overlapping"
                )
            cursor = item.end
        if any(len(window.text) > self.max_window_chars for window in self.windows):
            raise ValueError("dispatchable Miner window exceeds the plan bound")
        if any(
            item.max_window_chars != self.max_window_chars
            for item in self.oversized_fragments
        ):
            raise ValueError("oversized Miner outcome uses a different plan bound")
        if self.status == "ready" and (not self.windows or self.oversized_fragments):
            raise ValueError("ready Miner plan requires only dispatchable windows")
        if self.status == "intentional_empty" and (
            self.windows or self.oversized_fragments
        ):
            raise ValueError("intentional-empty Miner plan cannot contain fragments")
        if self.status == "oversized" and not self.oversized_fragments:
            raise ValueError("oversized Miner plan requires an oversized outcome")
        return self


def plan_miner_windows(
    paragraph: TrustedModelVisibleParagraph,
    *,
    max_window_chars: int,
) -> MinerWindowPlan:
    """Plan every bounded Miner window without dispatching or truncating."""

    if (
        isinstance(max_window_chars, bool)
        or not isinstance(max_window_chars, int)
        or max_window_chars <= 0
    ):
        raise ValueError("max_window_chars must be a positive integer")

    windows: list[MinerWindow] = []
    oversized: list[OversizedMinerFragment] = []
    identifier_ranges = paragraph.inline_code_identifier_ranges()
    fragments = tuple(
        _fragments(
            paragraph.text,
            split_protected_ranges=paragraph.split_protected_ranges(),
        )
    )
    for number, (text, start, end) in enumerate(fragments, 1):
        window_id = f"{paragraph.paragraph_id}_w_{number:04d}"
        line_start, line_end = paragraph.source_line_range(start, end)
        intersecting_identifiers = tuple(
            (identifier_start, identifier_end)
            for identifier_start, identifier_end in identifier_ranges
            if start < identifier_end and identifier_start < end
        )
        if any(
            not (start <= identifier_start < identifier_end <= end)
            for identifier_start, identifier_end in intersecting_identifiers
        ):
            raise ValueError("Miner window split an inline-code identifier")
        if len(text) > max_window_chars:
            oversized.append(
                OversizedMinerFragment(
                    window_id=window_id,
                    paragraph_id=paragraph.paragraph_id,
                    start=start,
                    end=end,
                    source_file=paragraph.source.file,
                    source_line_start=line_start,
                    source_line_end=line_end,
                    fragment_length=len(text),
                    max_window_chars=max_window_chars,
                    heading_path=paragraph.heading_path,
                    citation_ids=paragraph.citation_ids,
                    citation_urls=paragraph.citation_urls,
                )
            )
            continue
        windows.append(
            MinerWindow(
                window_id=window_id,
                paragraph_id=paragraph.paragraph_id,
                text=text,
                start=start,
                end=end,
                source_file=paragraph.source.file,
                source_line_start=line_start,
                source_line_end=line_end,
                heading_path=paragraph.heading_path,
                citation_ids=paragraph.citation_ids,
                citation_urls=paragraph.citation_urls,
                inline_code_identifier_count=len(intersecting_identifiers),
            )
        )

    status: Literal["ready", "intentional_empty", "oversized"]
    if oversized:
        status = "oversized"
    elif windows:
        status = "ready"
    else:
        status = "intentional_empty"
    return MinerWindowPlan(
        paragraph_id=paragraph.paragraph_id,
        max_window_chars=max_window_chars,
        status=status,
        windows=tuple(windows),
        oversized_fragments=tuple(oversized),
    )


def _is_complete_claim(value: str) -> bool:
    """Reject link-removal residue such as 'The launch record is'."""

    return len(_tokens(value)) >= 2 and INCOMPLETE_END_RE.search(value) is None


def _fragment_lines(input_data: MinerInput, start: int, end: int) -> tuple[int, int]:
    line_start = input_data.line_start + input_data.text.count("\n", 0, start)
    last_character = max(start, end - 1)
    line_end = input_data.line_start + input_data.text.count("\n", 0, last_character)
    return min(line_start, input_data.line_end), min(line_end, input_data.line_end)


def _unique_monotonic_offsets(
    paragraph: str,
    drafts: tuple[LiveMinedClaimDraft, ...],
) -> tuple[tuple[int, int], ...]:
    """Require earliest and latest ordered exact-span mappings to agree."""

    if not drafts:
        return ()

    earliest: list[tuple[int, int]] = []
    cursor = 0
    for draft in drafts:
        start = paragraph.find(draft.text, cursor)
        if start < 0:
            raise MinerScopeError("non_source_span")
        end = start + len(draft.text)
        earliest.append((start, end))
        cursor = end

    latest_reversed: list[tuple[int, int]] = []
    boundary = len(paragraph)
    for draft in reversed(drafts):
        start = paragraph.rfind(draft.text, 0, boundary)
        if start < 0:
            raise MinerScopeError("non_source_span")
        end = start + len(draft.text)
        latest_reversed.append((start, end))
        boundary = start

    earliest_mapping = tuple(earliest)
    latest_mapping = tuple(reversed(latest_reversed))
    if earliest_mapping != latest_mapping:
        raise MinerScopeError("ambiguous_span")
    return earliest_mapping


class ClaimMinerAgent:
    """Receives one paragraph and never receives source content or tools."""

    def __init__(self, client: ModelClient | None = None) -> None:
        self.client = client

    def mine(self, input_data: MinerInput) -> MinerOutput:
        return self.mine_with_validation(input_data).output

    def mine_with_validation(
        self,
        input_data: MinerInput,
        *,
        trusted_window_context: bool = False,
    ) -> ValidatedMinerOutput:
        """Mine one bounded input and retain only local coverage counters."""

        if self.client is not None:
            payload = input_data.model_dump(mode="json")
            if trusted_window_context:
                payload = {
                    "text": input_data.text,
                    "heading_path": input_data.heading_path,
                    "citation_urls": input_data.citation_urls,
                }
            draft = LiveMinerDraftOutput.model_validate(
                self.client.complete_model(
                    "claim_mining",
                    {
                        **payload,
                        "miner_contract_version": MINER_DRAFT_CONTRACT_VERSION,
                    },
                    LiveMinerDraftOutput,
                )
            )
            validation = self._validate_drafts(draft, input_data)
            live_claims = tuple(
                AtomicClaim(
                    claim_id=f"c_{index:04d}",
                    text=item.text,
                    file=input_data.file,
                    line_start=_fragment_lines(input_data, start, end)[0],
                    line_end=_fragment_lines(input_data, start, end)[1],
                    claim_type=item.claim_type,
                    slots={},
                    checkability=item.checkability,
                    citation_urls=input_data.citation_urls,
                )
                for index, (item, (start, end)) in enumerate(
                    zip(draft.claims, validation.offsets, strict=True), start=1
                )
            )
            output = MinerOutput(
                claims=live_claims,
                model_id=self.client.model_id,
                prompt_version=MINER_DRAFT_CONTRACT_VERSION,
            )
            final_validation = self._validate_scope(output, input_data)
            return ValidatedMinerOutput(output=output, validation=final_validation)
        deterministic_claims: list[AtomicClaim] = []
        for text, start, end in _fragments(input_data.text):
            if not _is_complete_claim(text):
                continue
            checkability = (
                Checkability.NOT_CHECKABLE
                if OPINION_RE.search(text) or is_recommendation(text)
                else Checkability.CHECKABLE
            )
            line_start, line_end = _fragment_lines(input_data, start, end)
            deterministic_claims.append(
                AtomicClaim(
                    claim_id=f"c_{len(deterministic_claims) + 1:04d}",
                    text=text,
                    file=input_data.file,
                    line_start=line_start,
                    line_end=line_end,
                    claim_type=_claim_type(text),
                    checkability=checkability,
                    citation_urls=input_data.citation_urls,
                )
            )
        output = MinerOutput(claims=tuple(deterministic_claims))
        validation = self._validate_scope(output, input_data)
        return ValidatedMinerOutput(output=output, validation=validation)

    @staticmethod
    def _validate_drafts(
        output: LiveMinerDraftOutput,
        input_data: MinerInput,
    ) -> MinerDraftValidation:
        occurrences = _protected_occurrences(input_data.text)
        exact_count = _individually_exact_draft_count(input_data.text, output.claims)
        for claim in output.claims:
            if not _is_complete_claim(claim.text):
                raise MinerScopeError(
                    "invalid_claim_fragment",
                    withheld_exact_draft_count=exact_count,
                    protected_occurrences_total=len(occurrences),
                )
        lexical_covered = _lexical_protected_coverage(occurrences, output.claims)
        if lexical_covered < len(occurrences):
            raise MinerScopeError(
                "missing_protected_token",
                withheld_exact_draft_count=exact_count,
                protected_occurrences_total=len(occurrences),
                protected_occurrences_covered=lexical_covered,
            )
        try:
            offsets = _unique_monotonic_offsets(input_data.text, output.claims)
        except MinerScopeError as error:
            raise MinerScopeError(
                error.code,
                withheld_exact_draft_count=exact_count,
                protected_occurrences_total=len(occurrences),
                protected_occurrences_covered=lexical_covered,
            ) from None
        covered = _span_protected_coverage(occurrences, offsets)
        if covered < len(occurrences):
            raise MinerScopeError(
                "missing_protected_token",
                withheld_exact_draft_count=len(output.claims),
                protected_occurrences_total=len(occurrences),
                protected_occurrences_covered=covered,
            )
        return MinerDraftValidation(
            offsets=offsets,
            protected_occurrences_total=len(occurrences),
            protected_occurrences_covered=covered,
        )

    @classmethod
    def _validate_scope(
        cls,
        output: MinerOutput,
        input_data: MinerInput,
    ) -> MinerDraftValidation:
        drafts = LiveMinerDraftOutput(
            claims=tuple(
                LiveMinedClaimDraft(
                    text=claim.text,
                    claim_type=claim.claim_type,
                    checkability=claim.checkability,
                )
                for claim in output.claims
            )
        )
        validation = cls._validate_drafts(drafts, input_data)
        for claim, (start, end) in zip(output.claims, validation.offsets, strict=True):
            expected_lines = _fragment_lines(input_data, start, end)
            if (
                claim.file != input_data.file
                or (claim.line_start, claim.line_end) != expected_lines
                or claim.slots
            ):
                raise RuntimeError("locally enriched Miner metadata escaped scope")
            if claim.citation_urls != input_data.citation_urls:
                raise RuntimeError("locally enriched Miner citation scope changed")
        return validation


__all__ = [
    "MINER_WINDOW_POLICY_VERSION",
    "ClaimMinerAgent",
    "MinerDraftValidation",
    "MinerScopeCode",
    "MinerScopeError",
    "MinerWindow",
    "MinerWindowPlan",
    "OversizedMinerFragment",
    "ValidatedMinerOutput",
    "miner_protected_occurrence_count",
    "plan_miner_windows",
]
