"""Deterministic, evidence-gated scalar repair planning and atomic application.

This module intentionally has no Agent, provider, or terminal responsibilities.
It accepts already-selected bounded evidence and typed final outcomes, derives a
replacement from the evidence scalar, and applies only byte-exact edits that
survive all local safety checks.
"""

from __future__ import annotations

import difflib
import hashlib
import os
import re
import stat
from collections import Counter, defaultdict
from collections.abc import Callable, Collection, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from itertools import pairwise
from pathlib import Path, PurePosixPath

from evidencetrace.agents.challenger import ChallengeAction
from evidencetrace.models import Relation

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_MONTH_NAMES = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
_MONTH_PATTERN = "(?:" + "|".join(_MONTH_NAMES) + ")"
_MONTH_NUMBER = {name.casefold(): index for index, name in enumerate(_MONTH_NAMES, 1)}

_SEMVER_NUMBER = r"(?:0|[1-9]\d*)"
_SEMVER_PRERELEASE_ID = r"(?:0|[1-9]\d*|[A-Za-z-][0-9A-Za-z-]*)"
_SEMVER_RE = re.compile(
    rf"(?<![0-9A-Za-z.-])"
    rf"(?P<value>{_SEMVER_NUMBER}\.{_SEMVER_NUMBER}\.{_SEMVER_NUMBER}"
    rf"(?:-{_SEMVER_PRERELEASE_ID}(?:\.{_SEMVER_PRERELEASE_ID})*)?"
    rf"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?)"
    rf"(?![0-9A-Za-z-]|\.\d)"
)
_NUMERIC_DATE_RE = re.compile(
    r"(?<!\d)(?P<year>\d{4})(?P<separator>[-/])"
    r"(?P<month>\d{2})(?P=separator)(?P<day>\d{2})(?!\d)"
)
_MONTH_FIRST_DATE_RE = re.compile(
    rf"(?<![A-Za-z0-9])(?P<month>{_MONTH_PATTERN})"
    r"(?P<first_gap>[ \t]+)(?P<day>0?[1-9]|[12]\d|3[01]),"
    r"(?P<second_gap>[ \t]+)(?P<year>\d{4})(?!\d)",
    re.IGNORECASE,
)
_DAY_FIRST_DATE_RE = re.compile(
    rf"(?<![A-Za-z0-9])(?P<day>0?[1-9]|[12]\d|3[01])"
    rf"(?P<first_gap>[ \t]+)(?P<month>{_MONTH_PATTERN})"
    r"(?P<second_gap>[ \t]+)(?P<year>\d{4})(?!\d)",
    re.IGNORECASE,
)
_PERCENT_RE = re.compile(
    r"(?<![A-Za-z0-9_.])(?P<number>[+-]?\d+(?:\.\d+)?)"
    r"(?P<gap>[ \t]*)%(?![A-Za-z0-9_])"
)
_DECIMAL_RE = re.compile(r"(?<![A-Za-z0-9_.])(?P<value>[+-]?\d+\.\d+)(?!\d|\.\d)")
_INTEGER_RE = re.compile(r"(?<![A-Za-z0-9_.])(?P<value>[+-]?\d+)(?!\d|\.\d)")
_BROAD_NUMERIC_DATE_RE = re.compile(
    r"(?<!\d)(?:\d{4}[/-]\d{1,2}[/-]\d{1,2}|"
    r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4})(?!\d)"
)
_BROAD_ENGLISH_DATE_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:[A-Za-z]{3,9}[ \t]+\d{1,2},[ \t]+\d{4}|"
    r"\d{1,2}[ \t]+[A-Za-z]{3,9}[ \t]+\d{4})(?![A-Za-z0-9])"
)
_URL_RE = re.compile(r"(?:https?|ftp)://[^\s<>'\"]+", re.IGNORECASE)
_HTML_OPEN_TAG_RE = re.compile(r"<[A-Za-z][^<>]*>", re.DOTALL)
_REFERENCE_DEFINITION_RE = re.compile(
    r"^[ \t]{0,3}\[[^]\n]+\]:[ \t]*(?P<target>.*)$", re.MULTILINE
)
_INLINE_LINK_OPEN_RE = re.compile(r"\][ \t]*\(")
_FENCE_OPEN_RE = re.compile(r"^[ ]{0,3}(?P<fence>`{3,}|~{3,})")
_CURRENCY_PREFIX_RE = re.compile(
    r"(?:(?P<symbol>[$€£¥₹])[ \t]*|"
    r"(?P<code>USD|EUR|GBP|JPY|CNY|RMB|INR)[ \t]+)$"
)
_UNIT_SUFFIX_RE = re.compile(
    r"(?P<gap>[ \t]*)(?P<unit>[A-Za-zµμ°]"
    r"[A-Za-z0-9µμ°²³^*/_-]*)"
)
_NON_UNIT_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "in",
        "is",
        "may",
        "might",
        "must",
        "of",
        "on",
        "or",
        "should",
        "than",
        "that",
        "the",
        "then",
        "this",
        "to",
        "was",
        "were",
        "which",
        "will",
        "with",
        "would",
    }
)


class ScalarKind(StrEnum):
    """The deliberately small repairable scalar allowlist."""

    INTEGER = "integer"
    DECIMAL = "decimal"
    PERCENTAGE = "percentage"
    DATE = "date"
    SEMVER = "semver"


class DecisionStatus(StrEnum):
    """Planning outcome for one proposal."""

    ELIGIBLE = "eligible"
    INELIGIBLE = "ineligible"
    NEEDS_HUMAN = "needs_human"
    STALE_TARGET = "stale_target"
    STALE_REFERENCE = "stale_reference"


class RepairReason(StrEnum):
    """Stable, machine-readable reason for a planning outcome."""

    ELIGIBLE = "eligible"
    DUPLICATE_PROPOSAL_ID = "duplicate_proposal_id"
    FINAL_NOT_CONTRADICTED = "final_not_contradicted"
    CHALLENGER_INCOMPLETE = "challenger_incomplete"
    CHALLENGER_ABSTAINED = "challenger_abstained"
    CHALLENGER_REVISION_UNSAFE = "challenger_revision_unsafe"
    EVIDENCE_NOT_EXACT = "evidence_not_exact"
    TARGET_MISSING = "target_missing"
    STALE_TARGET = "stale_target"
    STALE_REFERENCE = "stale_reference"
    INVALID_UTF8 = "invalid_utf8"
    CLAIM_NOT_FOUND = "claim_not_found"
    CLAIM_AMBIGUOUS = "claim_ambiguous"
    SCALAR_NOT_FOUND = "scalar_not_found"
    SCALAR_AMBIGUOUS = "scalar_ambiguous"
    UNSUPPORTED_SCALAR = "unsupported_scalar"
    EVIDENCE_SCALAR_NOT_FOUND = "evidence_scalar_not_found"
    EVIDENCE_SCALAR_AMBIGUOUS = "evidence_scalar_ambiguous"
    TYPE_MISMATCH = "type_mismatch"
    DATE_FAMILY_MISMATCH = "date_family_mismatch"
    UNIT_MISMATCH = "unit_mismatch"
    NO_SEMANTIC_CHANGE = "no_semantic_change"
    PROTECTED_LOCATION = "protected_location"
    OVERLAP = "overlap"
    ROUND_TRIP_FAILED = "round_trip_failed"


class ApplicationStatus(StrEnum):
    """Per-proposal and per-file application status."""

    APPLIED = "applied"
    NOT_APPROVED = "not_approved"
    NOT_ELIGIBLE = "not_eligible"
    STALE_TARGET = "stale_target"
    STALE_REFERENCE = "stale_reference"
    APPLY_ERROR = "apply_error"


@dataclass(frozen=True, slots=True)
class FileDigest:
    """Expected SHA-256 for one project-relative target or reference."""

    path: str
    sha256: str

    def __post_init__(self) -> None:
        _validate_relative_path(self.path)
        object.__setattr__(self, "sha256", _validate_sha256(self.sha256))


@dataclass(frozen=True, slots=True)
class ExactEvidence:
    """A selected bounded evidence span and its exact scalar locator."""

    source_id: str
    text: str
    scalar: str
    exact: bool = True

    def __post_init__(self) -> None:
        if not self.source_id.strip():
            raise ValueError("evidence source_id must not be blank")
        if not self.text.strip():
            raise ValueError("evidence text must not be blank")
        if not self.scalar.strip():
            raise ValueError("evidence scalar must not be blank")


@dataclass(frozen=True, slots=True)
class ChallengerGate:
    """Typed Challenger completion information used by the local gate."""

    completed: bool
    action: ChallengeAction | None
    revised_relation: Relation | None = None
    revised_evidence_text: str | None = None
    revised_evidence_scalar: str | None = None


@dataclass(frozen=True, slots=True)
class RepairProposal:
    """Inputs needed to plan one scalar-only correction."""

    proposal_id: str
    target: FileDigest
    claim_text: str
    line_start: int
    line_end: int
    target_scalar: str
    evidence: ExactEvidence
    final_relation: Relation
    challenger: ChallengerGate
    references: tuple[FileDigest, ...] = ()
    claim_is_exact: bool = True

    def __post_init__(self) -> None:
        if not self.proposal_id.strip():
            raise ValueError("proposal_id must not be blank")
        if not self.claim_text:
            raise ValueError("claim_text must not be empty")
        if self.line_start < 1 or self.line_end < self.line_start:
            raise ValueError("claim line range must be positive and ordered")
        if not self.target_scalar.strip():
            raise ValueError("target_scalar must not be blank")
        if len({item.path for item in self.references}) != len(self.references):
            raise ValueError("reference paths must be unique per proposal")


@dataclass(frozen=True, slots=True)
class RepairCandidate:
    """A byte-located replacement that passed every deterministic gate."""

    proposal_id: str
    target: FileDigest
    references: tuple[FileDigest, ...]
    line_start: int
    line_end: int
    claim_text: str
    scalar_kind: ScalarKind
    original_scalar: str
    replacement_scalar: str
    unit: str | None
    char_start: int
    char_end: int
    byte_start: int
    byte_end: int
    evidence: ExactEvidence
    suggested_diff: str


@dataclass(frozen=True, slots=True)
class RepairDecision:
    """Deterministic eligibility decision for one proposal."""

    proposal_id: str
    status: DecisionStatus
    reason: RepairReason
    candidate: RepairCandidate | None = None
    detail: str = ""

    def __post_init__(self) -> None:
        eligible = self.status is DecisionStatus.ELIGIBLE
        if eligible != (self.candidate is not None):
            raise ValueError("only eligible decisions may contain a candidate")


@dataclass(frozen=True, slots=True)
class RepairPlan:
    """All proposal decisions plus the collective eligible suggestion diff."""

    decisions: tuple[RepairDecision, ...]
    suggested_diff: str

    @property
    def candidates(self) -> tuple[RepairCandidate, ...]:
        return tuple(
            decision.candidate
            for decision in self.decisions
            if decision.candidate is not None
        )


@dataclass(frozen=True, slots=True)
class ApplicationDecision:
    """Application result for one proposal."""

    proposal_id: str
    status: ApplicationStatus
    detail: str = ""


@dataclass(frozen=True, slots=True)
class FileApplyResult:
    """Atomic application result for one target file."""

    target_path: str
    status: ApplicationStatus
    proposal_ids: tuple[str, ...]
    before_sha256: str | None
    after_sha256: str | None
    suggested_diff: str
    applied_diff: str
    reverse_diff: str
    detail: str = ""


BeforeFileWrite = Callable[[FileApplyResult], None]


@dataclass(frozen=True, slots=True)
class ApplyResult:
    """Batch result; files are isolated from one another."""

    decisions: tuple[ApplicationDecision, ...]
    files: tuple[FileApplyResult, ...]
    suggested_diff: str
    applied_diff: str
    reverse_diff: str


@dataclass(frozen=True, slots=True)
class _ScalarToken:
    kind: ScalarKind
    raw: str
    start: int
    end: int
    semantic: str
    family: str | None = None
    separator: str | None = None
    first_gap: str = ""
    second_gap: str = ""
    day_width: int = 0
    month_case: str = ""
    percent_gap: str = ""
    number_raw: str = ""
    date_value: date | None = None


@dataclass(frozen=True, slots=True)
class _FileFingerprint:
    device: int
    inode: int
    size: int
    modified_ns: int


@dataclass(frozen=True, slots=True)
class _RuntimePathResolution:
    paths: Mapping[str, Path]
    invalid_ids: frozenset[str]


class _TargetBecameStale(RuntimeError):
    pass


class _BeforeFileWriteFailed(RuntimeError):
    pass


class _UnsafeTargetPath(ValueError):
    pass


def sha256_bytes(content: bytes) -> str:
    """Return the lowercase SHA-256 for exact file bytes."""

    return hashlib.sha256(content).hexdigest()


def derive_unique_scalar_pair(
    claim_text: str, bounded_evidence_text: str
) -> tuple[str, str] | None:
    """Return the sole conservative target/evidence scalar pair, if one exists.

    This helper is intentionally stricter than proposal planning: each input
    must contain exactly one supported scalar in total.  Their scalar types,
    date families, and adjacent units must agree, and their semantic values
    must differ.  It never guesses among multiple numbers or locale dates.
    """

    target_tokens = _extract_scalars(claim_text)
    evidence_tokens = _extract_scalars(bounded_evidence_text)
    if len(target_tokens) != 1 or len(evidence_tokens) != 1:
        return None
    target, evidence = target_tokens[0], evidence_tokens[0]
    if (
        _overlaps_unsupported_date(claim_text, target.start, target.end)
        or _overlaps_unsupported_date(
            bounded_evidence_text, evidence.start, evidence.end
        )
        or _is_protected(claim_text, target.start, target.end)
        or _is_protected(bounded_evidence_text, evidence.start, evidence.end)
        or target.kind is not evidence.kind
        or (target.kind is ScalarKind.DATE and target.family != evidence.family)
        or _unit_signature(claim_text, target)
        != _unit_signature(bounded_evidence_text, evidence)
        or target.semantic == evidence.semantic
    ):
        return None
    return target.raw, evidence.raw


def scalar_outcome_signature(
    claim_text: str,
    bounded_evidence_text: str,
) -> tuple[ScalarKind, str, str | None] | None:
    """Return a typed evidence scalar outcome for conservative source grouping.

    Unlike ``derive_unique_scalar_pair``, this also represents exact semantic
    agreement. It still requires one unambiguous scalar per input, compatible
    types/date families/units, and an unprotected target scalar.
    """

    target_tokens = _extract_scalars(claim_text)
    evidence_tokens = _extract_scalars(bounded_evidence_text)
    if len(target_tokens) != 1 or len(evidence_tokens) != 1:
        return None
    target, evidence = target_tokens[0], evidence_tokens[0]
    target_unit = _unit_signature(claim_text, target)
    evidence_unit = _unit_signature(bounded_evidence_text, evidence)
    if (
        _overlaps_unsupported_date(claim_text, target.start, target.end)
        or _overlaps_unsupported_date(
            bounded_evidence_text, evidence.start, evidence.end
        )
        or _is_protected(claim_text, target.start, target.end)
        or _is_protected(bounded_evidence_text, evidence.start, evidence.end)
        or target.kind is not evidence.kind
        or (target.kind is ScalarKind.DATE and target.family != evidence.family)
        or target_unit != evidence_unit
    ):
        return None
    return evidence.kind, evidence.semantic, evidence_unit


def plan_repairs(
    target_contents: Mapping[str, bytes],
    proposals: Sequence[RepairProposal],
    *,
    reference_contents: Mapping[str, bytes] | None = None,
) -> RepairPlan:
    """Plan conservative scalar repairs without touching the filesystem.

    ``target_contents`` and ``reference_contents`` are exact snapshots keyed by
    the project-relative paths carried by proposal digests.  Returned decisions
    preserve proposal order.  Every proposal participating in an overlap is
    rejected as ``needs_human``.
    """

    references = reference_contents or {}
    duplicate_ids = {
        proposal_id
        for proposal_id, count in Counter(
            proposal.proposal_id for proposal in proposals
        ).items()
        if count > 1
    }
    decisions: list[RepairDecision] = []
    for proposal in proposals:
        if proposal.proposal_id in duplicate_ids:
            decisions.append(
                _decision(
                    proposal,
                    DecisionStatus.INELIGIBLE,
                    RepairReason.DUPLICATE_PROPOSAL_ID,
                )
            )
            continue
        decisions.append(_plan_one(target_contents, references, proposal))

    overlap_indexes: set[int] = set()
    candidates_by_path: dict[str, list[tuple[int, RepairCandidate]]] = defaultdict(list)
    for index, decision in enumerate(decisions):
        if decision.candidate is not None:
            candidates_by_path[decision.candidate.target.path].append(
                (index, decision.candidate)
            )
    for entries in candidates_by_path.values():
        ordered = sorted(entries, key=lambda item: item[1].byte_start)
        for left_index, (decision_index, candidate) in enumerate(ordered):
            for other_index, other in ordered[left_index + 1 :]:
                if other.byte_start >= candidate.byte_end:
                    break
                if (
                    candidate.byte_start < other.byte_end
                    and other.byte_start < candidate.byte_end
                ):
                    overlap_indexes.update({decision_index, other_index})
    for index in overlap_indexes:
        proposal = proposals[index]
        decisions[index] = _decision(
            proposal,
            DecisionStatus.NEEDS_HUMAN,
            RepairReason.OVERLAP,
            detail="proposal span overlaps another repair; all overlaps were rejected",
        )

    candidates = tuple(
        decision.candidate for decision in decisions if decision.candidate is not None
    )
    try:
        suggested = _render_diff_set(target_contents, candidates)
    except (UnicodeDecodeError, ValueError) as error:
        affected_paths = {candidate.target.path for candidate in candidates}
        for index, decision in enumerate(decisions):
            maybe_candidate = decision.candidate
            if (
                maybe_candidate is not None
                and maybe_candidate.target.path in affected_paths
            ):
                decisions[index] = RepairDecision(
                    proposal_id=decision.proposal_id,
                    status=DecisionStatus.INELIGIBLE,
                    reason=RepairReason.ROUND_TRIP_FAILED,
                    detail=str(error),
                )
        suggested = ""
    return RepairPlan(decisions=tuple(decisions), suggested_diff=suggested)


def apply_repair_plan(
    root: Path,
    plan: RepairPlan,
    approved_proposal_ids: Collection[str],
    *,
    runtime_paths: Mapping[str, Path] | None = None,
    before_file_write: BeforeFileWrite | None = None,
) -> ApplyResult:
    """Atomically apply explicitly approved eligible candidates.

    Target and local-reference SHA-256 values are rechecked immediately before
    each file write.  Edits for a file are assembled and round-tripped in
    memory, written to a same-directory temporary file, flushed with ``fsync``,
    and installed with one ``os.replace``.  A failure before that final replace
    leaves the target's bytes untouched.  ``before_file_write`` receives the
    complete successful result preview after the dry-runs and hashes are ready,
    but before replacement; its exceptions fail only that target file.
    """

    approved = frozenset(approved_proposal_ids)
    candidates_by_id = {
        candidate.proposal_id: candidate for candidate in plan.candidates
    }
    application: dict[str, ApplicationDecision] = {}
    for decision in plan.decisions:
        if decision.candidate is None:
            application[decision.proposal_id] = ApplicationDecision(
                proposal_id=decision.proposal_id,
                status=ApplicationStatus.NOT_ELIGIBLE,
                detail=decision.reason.value,
            )
        elif decision.proposal_id not in approved:
            application[decision.proposal_id] = ApplicationDecision(
                proposal_id=decision.proposal_id,
                status=ApplicationStatus.NOT_APPROVED,
            )

    selected = tuple(
        candidate
        for proposal_id, candidate in candidates_by_id.items()
        if proposal_id in approved
    )
    grouped: dict[str, list[RepairCandidate]] = defaultdict(list)
    for candidate in selected:
        grouped[candidate.target.path].append(candidate)

    root_resolved = root.resolve(strict=True)
    required_runtime_ids = {item.target.path for item in selected} | {
        reference.path for item in selected for reference in item.references
    }
    runtime_resolution = _validated_runtime_paths(
        runtime_paths or {}, required_runtime_ids
    )
    file_results: list[FileApplyResult] = []
    successful_forward: list[str] = []
    successful_reverse: list[str] = []
    for target_path in sorted(grouped):
        file_candidates = tuple(
            sorted(grouped[target_path], key=lambda item: item.byte_start)
        )
        proposal_ids = tuple(item.proposal_id for item in file_candidates)
        try:
            path = _resolve_file(root_resolved, target_path, runtime_resolution)
            parent_fd, fingerprint, before = _pin_regular_file(path)
        except (OSError, ValueError, _TargetBecameStale) as error:
            detail = f"could not read target {target_path!r}: {type(error).__name__}"
            for proposal_id in proposal_ids:
                application[proposal_id] = ApplicationDecision(
                    proposal_id, ApplicationStatus.APPLY_ERROR, detail
                )
            file_results.append(
                _failed_file_result(
                    target_path,
                    ApplicationStatus.APPLY_ERROR,
                    proposal_ids,
                    detail,
                )
            )
            continue

        try:
            actual_sha = sha256_bytes(before)
            if any(
                candidate.target.sha256 != actual_sha for candidate in file_candidates
            ):
                detail = "target SHA-256 changed after planning"
                for proposal_id in proposal_ids:
                    application[proposal_id] = ApplicationDecision(
                        proposal_id, ApplicationStatus.STALE_TARGET, detail
                    )
                file_results.append(
                    _failed_file_result(
                        target_path,
                        ApplicationStatus.STALE_TARGET,
                        proposal_ids,
                        detail,
                        before_sha256=actual_sha,
                    )
                )
                continue

            current_candidates: list[RepairCandidate] = []
            stale_reference_ids: list[str] = []
            for candidate in file_candidates:
                if _references_current(
                    root_resolved,
                    candidate.references,
                    runtime_resolution,
                ):
                    current_candidates.append(candidate)
                else:
                    stale_reference_ids.append(candidate.proposal_id)
                    application[candidate.proposal_id] = ApplicationDecision(
                        candidate.proposal_id,
                        ApplicationStatus.STALE_REFERENCE,
                        "a dependent reference SHA-256 changed after planning",
                    )
            if not current_candidates:
                detail = "all approved candidates depend on stale references"
                file_results.append(
                    _failed_file_result(
                        target_path,
                        ApplicationStatus.STALE_REFERENCE,
                        tuple(stale_reference_ids),
                        detail,
                        before_sha256=actual_sha,
                    )
                )
                continue

            active = tuple(current_candidates)
            active_ids = tuple(candidate.proposal_id for candidate in active)
            try:
                after = _forward_bytes(before, active)
                restored = _reverse_bytes(after, active)
                if restored != before:
                    raise ValueError("reverse dry-run did not restore original bytes")
                forward_diff = _unified_diff(before, after, target_path, reverse=False)
                reverse_diff = _unified_diff(after, before, target_path, reverse=True)
                after_sha = sha256_bytes(after)
                preview = FileApplyResult(
                    target_path=target_path,
                    status=ApplicationStatus.APPLIED,
                    proposal_ids=active_ids,
                    before_sha256=actual_sha,
                    after_sha256=after_sha,
                    suggested_diff=forward_diff,
                    applied_diff=forward_diff,
                    reverse_diff=reverse_diff,
                )
                if before_file_write is not None:
                    try:
                        before_file_write(preview)
                    except Exception as error:
                        raise _BeforeFileWriteFailed(type(error).__name__) from error
                _atomic_replace(
                    path,
                    after,
                    parent_fd=parent_fd,
                    expected_before_sha256=actual_sha,
                    expected_fingerprint=fingerprint,
                )
            except _TargetBecameStale as error:
                detail = str(error)
                for proposal_id in active_ids:
                    application[proposal_id] = ApplicationDecision(
                        proposal_id, ApplicationStatus.STALE_TARGET, detail
                    )
                file_results.append(
                    _failed_file_result(
                        target_path,
                        ApplicationStatus.STALE_TARGET,
                        active_ids,
                        detail,
                        before_sha256=actual_sha,
                    )
                )
                continue
            except _BeforeFileWriteFailed as error:
                cause_name = type(error.__cause__).__name__
                detail = f"before_file_write failed for {target_path!r}: {cause_name}"
                for proposal_id in active_ids:
                    application[proposal_id] = ApplicationDecision(
                        proposal_id, ApplicationStatus.APPLY_ERROR, detail
                    )
                file_results.append(
                    _failed_file_result(
                        target_path,
                        ApplicationStatus.APPLY_ERROR,
                        active_ids,
                        detail,
                        before_sha256=actual_sha,
                    )
                )
                continue
            except (OSError, UnicodeDecodeError, ValueError) as error:
                detail = (
                    f"atomic apply failed for {target_path!r}: {type(error).__name__}"
                )
                for proposal_id in active_ids:
                    application[proposal_id] = ApplicationDecision(
                        proposal_id, ApplicationStatus.APPLY_ERROR, detail
                    )
                file_results.append(
                    _failed_file_result(
                        target_path,
                        ApplicationStatus.APPLY_ERROR,
                        active_ids,
                        detail,
                        before_sha256=actual_sha,
                    )
                )
                continue

            for proposal_id in active_ids:
                application[proposal_id] = ApplicationDecision(
                    proposal_id, ApplicationStatus.APPLIED
                )
            file_results.append(preview)
            successful_forward.append(forward_diff)
            successful_reverse.append(reverse_diff)
        finally:
            os.close(parent_fd)

    ordered_application = tuple(
        application[decision.proposal_id] for decision in plan.decisions
    )
    return ApplyResult(
        decisions=ordered_application,
        files=tuple(file_results),
        suggested_diff=plan.suggested_diff,
        applied_diff="".join(successful_forward),
        reverse_diff="".join(successful_reverse),
    )


def _plan_one(
    target_contents: Mapping[str, bytes],
    reference_contents: Mapping[str, bytes],
    proposal: RepairProposal,
) -> RepairDecision:
    if proposal.final_relation is not Relation.CONTRADICTED:
        return _decision(
            proposal,
            DecisionStatus.INELIGIBLE,
            RepairReason.FINAL_NOT_CONTRADICTED,
        )
    challenger_reason = _challenger_reason(proposal)
    if challenger_reason is not None:
        return _decision(proposal, DecisionStatus.INELIGIBLE, challenger_reason)
    if not proposal.claim_is_exact or not proposal.evidence.exact:
        return _decision(
            proposal,
            DecisionStatus.INELIGIBLE,
            RepairReason.EVIDENCE_NOT_EXACT,
            detail="claim and bounded evidence must both be exact source text",
        )

    content = target_contents.get(proposal.target.path)
    if content is None:
        return _decision(
            proposal,
            DecisionStatus.INELIGIBLE,
            RepairReason.TARGET_MISSING,
        )
    if sha256_bytes(content) != proposal.target.sha256:
        return _decision(
            proposal,
            DecisionStatus.STALE_TARGET,
            RepairReason.STALE_TARGET,
        )
    if not _snapshot_references_current(reference_contents, proposal.references):
        return _decision(
            proposal,
            DecisionStatus.STALE_REFERENCE,
            RepairReason.STALE_REFERENCE,
        )
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        return _decision(
            proposal,
            DecisionStatus.INELIGIBLE,
            RepairReason.INVALID_UTF8,
            detail=str(error),
        )

    window = _line_window(text, proposal.line_start, proposal.line_end)
    if window is None:
        return _decision(
            proposal,
            DecisionStatus.NEEDS_HUMAN,
            RepairReason.CLAIM_NOT_FOUND,
            detail="claim line range is outside the current target",
        )
    window_start, window_end = window
    claim_occurrences = _occurrences(
        text, proposal.claim_text, window_start, window_end
    )
    if not claim_occurrences:
        return _decision(
            proposal,
            DecisionStatus.NEEDS_HUMAN,
            RepairReason.CLAIM_NOT_FOUND,
        )
    if len(claim_occurrences) != 1:
        return _decision(
            proposal,
            DecisionStatus.NEEDS_HUMAN,
            RepairReason.CLAIM_AMBIGUOUS,
        )
    claim_start = claim_occurrences[0]
    target_matches = tuple(
        token
        for token in _extract_scalars(proposal.claim_text)
        if token.raw == proposal.target_scalar
    )
    if not target_matches:
        reason = (
            RepairReason.UNSUPPORTED_SCALAR
            if proposal.target_scalar in proposal.claim_text
            else RepairReason.SCALAR_NOT_FOUND
        )
        return _decision(proposal, DecisionStatus.NEEDS_HUMAN, reason)
    if len(target_matches) != 1:
        return _decision(
            proposal,
            DecisionStatus.NEEDS_HUMAN,
            RepairReason.SCALAR_AMBIGUOUS,
        )
    target_token = target_matches[0]
    absolute_start = claim_start + target_token.start
    absolute_end = claim_start + target_token.end
    if _overlaps_unsupported_date(text, absolute_start, absolute_end):
        return _decision(
            proposal,
            DecisionStatus.NEEDS_HUMAN,
            RepairReason.UNSUPPORTED_SCALAR,
            detail="scalar is inside a non-allowlisted date",
        )

    evidence_matches = tuple(
        token
        for token in _extract_scalars(proposal.evidence.text)
        if token.raw == proposal.evidence.scalar
    )
    if not evidence_matches:
        reason = (
            RepairReason.UNSUPPORTED_SCALAR
            if proposal.evidence.scalar in proposal.evidence.text
            else RepairReason.EVIDENCE_SCALAR_NOT_FOUND
        )
        return _decision(proposal, DecisionStatus.NEEDS_HUMAN, reason)
    if len(evidence_matches) != 1:
        return _decision(
            proposal,
            DecisionStatus.NEEDS_HUMAN,
            RepairReason.EVIDENCE_SCALAR_AMBIGUOUS,
        )
    evidence_token = evidence_matches[0]
    if _overlaps_unsupported_date(
        proposal.evidence.text, evidence_token.start, evidence_token.end
    ):
        return _decision(
            proposal,
            DecisionStatus.NEEDS_HUMAN,
            RepairReason.UNSUPPORTED_SCALAR,
            detail="evidence scalar is inside a non-allowlisted date",
        )
    if target_token.kind is not evidence_token.kind:
        return _decision(
            proposal,
            DecisionStatus.NEEDS_HUMAN,
            RepairReason.TYPE_MISMATCH,
        )
    if (
        target_token.kind is ScalarKind.DATE
        and target_token.family != evidence_token.family
    ):
        return _decision(
            proposal,
            DecisionStatus.NEEDS_HUMAN,
            RepairReason.DATE_FAMILY_MISMATCH,
        )

    target_unit = _unit_signature(proposal.claim_text, target_token)
    evidence_unit = _unit_signature(proposal.evidence.text, evidence_token)
    if target_unit != evidence_unit:
        return _decision(
            proposal,
            DecisionStatus.NEEDS_HUMAN,
            RepairReason.UNIT_MISMATCH,
            detail=f"target unit {target_unit!r} != evidence unit {evidence_unit!r}",
        )
    if target_token.semantic == evidence_token.semantic:
        return _decision(
            proposal,
            DecisionStatus.INELIGIBLE,
            RepairReason.NO_SEMANTIC_CHANGE,
        )

    replacement = _render_replacement(target_token, evidence_token)
    if replacement == target_token.raw:
        return _decision(
            proposal,
            DecisionStatus.INELIGIBLE,
            RepairReason.NO_SEMANTIC_CHANGE,
        )
    if _is_protected(text, absolute_start, absolute_end):
        return _decision(
            proposal,
            DecisionStatus.NEEDS_HUMAN,
            RepairReason.PROTECTED_LOCATION,
        )

    byte_start = len(text[:absolute_start].encode("utf-8"))
    byte_end = len(text[:absolute_end].encode("utf-8"))
    provisional = RepairCandidate(
        proposal_id=proposal.proposal_id,
        target=proposal.target,
        references=proposal.references,
        line_start=proposal.line_start,
        line_end=proposal.line_end,
        claim_text=proposal.claim_text,
        scalar_kind=target_token.kind,
        original_scalar=target_token.raw,
        replacement_scalar=replacement,
        unit=target_unit,
        char_start=absolute_start,
        char_end=absolute_end,
        byte_start=byte_start,
        byte_end=byte_end,
        evidence=proposal.evidence,
        suggested_diff="",
    )
    try:
        after = _forward_bytes(content, (provisional,))
        if _reverse_bytes(after, (provisional,)) != content:
            raise ValueError("single-candidate reverse dry-run failed")
        candidate = RepairCandidate(
            proposal_id=provisional.proposal_id,
            target=provisional.target,
            references=provisional.references,
            line_start=provisional.line_start,
            line_end=provisional.line_end,
            claim_text=provisional.claim_text,
            scalar_kind=provisional.scalar_kind,
            original_scalar=provisional.original_scalar,
            replacement_scalar=provisional.replacement_scalar,
            unit=provisional.unit,
            char_start=provisional.char_start,
            char_end=provisional.char_end,
            byte_start=provisional.byte_start,
            byte_end=provisional.byte_end,
            evidence=provisional.evidence,
            suggested_diff=_unified_diff(
                content, after, proposal.target.path, reverse=False
            ),
        )
    except (UnicodeDecodeError, ValueError) as error:
        return _decision(
            proposal,
            DecisionStatus.INELIGIBLE,
            RepairReason.ROUND_TRIP_FAILED,
            detail=str(error),
        )
    return RepairDecision(
        proposal_id=proposal.proposal_id,
        status=DecisionStatus.ELIGIBLE,
        reason=RepairReason.ELIGIBLE,
        candidate=candidate,
    )


def _challenger_reason(proposal: RepairProposal) -> RepairReason | None:
    challenger = proposal.challenger
    if not challenger.completed or challenger.action is None:
        return RepairReason.CHALLENGER_INCOMPLETE
    if challenger.action is ChallengeAction.ABSTAIN:
        return RepairReason.CHALLENGER_ABSTAINED
    if challenger.action is ChallengeAction.UPHOLD:
        return None
    if challenger.action is not ChallengeAction.REVISE:
        return RepairReason.CHALLENGER_INCOMPLETE
    if (
        challenger.revised_relation is not Relation.CONTRADICTED
        or challenger.revised_evidence_text != proposal.evidence.text
        or challenger.revised_evidence_scalar != proposal.evidence.scalar
    ):
        return RepairReason.CHALLENGER_REVISION_UNSAFE
    return None


def _decision(
    proposal: RepairProposal,
    status: DecisionStatus,
    reason: RepairReason,
    *,
    detail: str = "",
) -> RepairDecision:
    return RepairDecision(
        proposal_id=proposal.proposal_id,
        status=status,
        reason=reason,
        detail=detail,
    )


def _extract_scalars(text: str) -> tuple[_ScalarToken, ...]:
    accepted: list[_ScalarToken] = []

    def available(start: int, end: int) -> bool:
        return not any(start < item.end and item.start < end for item in accepted)

    for match in _SEMVER_RE.finditer(text):
        accepted.append(
            _ScalarToken(
                kind=ScalarKind.SEMVER,
                raw=match.group("value"),
                start=match.start("value"),
                end=match.end("value"),
                semantic=match.group("value"),
            )
        )
    for regex, family in (
        (_NUMERIC_DATE_RE, "numeric_ymd"),
        (_MONTH_FIRST_DATE_RE, "english_month_first"),
        (_DAY_FIRST_DATE_RE, "english_day_first"),
    ):
        for match in regex.finditer(text):
            if not available(match.start(), match.end()):
                continue
            token = _date_token(match, family)
            if token is not None:
                accepted.append(token)
    for match in _PERCENT_RE.finditer(text):
        if not available(match.start(), match.end()):
            continue
        try:
            semantic = str(Decimal(match.group("number")).normalize())
        except InvalidOperation:
            continue
        accepted.append(
            _ScalarToken(
                kind=ScalarKind.PERCENTAGE,
                raw=match.group(0),
                start=match.start(),
                end=match.end(),
                semantic=semantic,
                percent_gap=match.group("gap"),
                number_raw=match.group("number"),
            )
        )
    for match in _DECIMAL_RE.finditer(text):
        if not available(match.start("value"), match.end("value")):
            continue
        try:
            semantic = str(Decimal(match.group("value")).normalize())
        except InvalidOperation:
            continue
        accepted.append(
            _ScalarToken(
                kind=ScalarKind.DECIMAL,
                raw=match.group("value"),
                start=match.start("value"),
                end=match.end("value"),
                semantic=semantic,
                number_raw=match.group("value"),
            )
        )
    for match in _INTEGER_RE.finditer(text):
        if not available(match.start("value"), match.end("value")):
            continue
        accepted.append(
            _ScalarToken(
                kind=ScalarKind.INTEGER,
                raw=match.group("value"),
                start=match.start("value"),
                end=match.end("value"),
                semantic=str(int(match.group("value"))),
                number_raw=match.group("value"),
            )
        )
    return tuple(sorted(accepted, key=lambda item: (item.start, item.end)))


def _date_token(match: re.Match[str], family: str) -> _ScalarToken | None:
    try:
        month_name = match.groupdict().get("month")
        month = (
            _MONTH_NUMBER[month_name.casefold()]
            if month_name is not None and not month_name.isdigit()
            else int(match.group("month"))
        )
        parsed = date(int(match.group("year")), month, int(match.group("day")))
    except (KeyError, ValueError):
        return None
    return _ScalarToken(
        kind=ScalarKind.DATE,
        raw=match.group(0),
        start=match.start(),
        end=match.end(),
        semantic=parsed.isoformat(),
        family=family,
        separator=match.groupdict().get("separator"),
        first_gap=match.groupdict().get("first_gap", "") or "",
        second_gap=match.groupdict().get("second_gap", "") or "",
        day_width=len(match.group("day")),
        month_case=month_name or "",
        date_value=parsed,
    )


def _render_replacement(target: _ScalarToken, evidence: _ScalarToken) -> str:
    if target.kind is ScalarKind.PERCENTAGE:
        return f"{evidence.number_raw}{target.percent_gap}%"
    if target.kind is not ScalarKind.DATE:
        return evidence.raw
    if evidence.date_value is None:
        raise ValueError("date evidence has no semantic date")
    replacement_date = evidence.date_value
    if target.family == "numeric_ymd":
        separator = target.separator or "-"
        return (
            f"{replacement_date.year:04d}{separator}"
            f"{replacement_date.month:02d}{separator}{replacement_date.day:02d}"
        )
    month = _case_like(_MONTH_NAMES[replacement_date.month - 1], target.month_case)
    day = (
        f"{replacement_date.day:02d}"
        if target.day_width == 2
        else str(replacement_date.day)
    )
    if target.family == "english_month_first":
        return (
            f"{month}{target.first_gap}{day},"
            f"{target.second_gap}{replacement_date.year:04d}"
        )
    if target.family == "english_day_first":
        return (
            f"{day}{target.first_gap}{month}"
            f"{target.second_gap}{replacement_date.year:04d}"
        )
    raise ValueError("unsupported date family")


def _case_like(value: str, template: str) -> str:
    if template.isupper():
        return value.upper()
    if template.islower():
        return value.lower()
    return value


def _unit_signature(text: str, token: _ScalarToken) -> str | None:
    if token.kind is ScalarKind.PERCENTAGE:
        return "suffix:%"
    if token.kind not in {ScalarKind.INTEGER, ScalarKind.DECIMAL}:
        return None
    units: list[str] = []
    context_start = token.start
    context_end = token.end
    for inline_start, inline_end in _inline_code_spans(text):
        if inline_start < token.start and token.end < inline_end:
            before_scalar = text[inline_start : token.start]
            after_scalar = text[token.end : inline_end]
            if before_scalar and not before_scalar.strip("`"):
                context_start = inline_start
            if after_scalar and not after_scalar.strip("`"):
                context_end = inline_end
            break
    prefix = _CURRENCY_PREFIX_RE.search(text[:context_start])
    if prefix is not None:
        value = prefix.group("symbol") or prefix.group("code")
        units.append(f"prefix:{value}")
    suffix = _UNIT_SUFFIX_RE.match(text[context_end:])
    if suffix is not None:
        unit = suffix.group("unit")
        gap = suffix.group("gap")
        if not gap or unit.casefold() not in _NON_UNIT_WORDS:
            units.append(f"suffix:{unit}")
    return "|".join(units) or None


def _line_window(text: str, line_start: int, line_end: int) -> tuple[int, int] | None:
    lines = text.splitlines(keepends=True)
    if not lines or line_start > len(lines) or line_end > len(lines):
        return None
    starts = [0]
    for line in lines:
        starts.append(starts[-1] + len(line))
    return starts[line_start - 1], starts[line_end]


def _occurrences(text: str, needle: str, start: int, end: int) -> tuple[int, ...]:
    found: list[int] = []
    cursor = start
    while cursor <= end - len(needle):
        index = text.find(needle, cursor, end)
        if index < 0:
            break
        found.append(index)
        cursor = index + 1
    return tuple(found)


def _overlaps_unsupported_date(text: str, start: int, end: int) -> bool:
    broad = tuple(_BROAD_NUMERIC_DATE_RE.finditer(text)) + tuple(
        _BROAD_ENGLISH_DATE_RE.finditer(text)
    )
    for match in broad:
        if not (start < match.end() and match.start() < end):
            continue
        matched_text = match.group(0)
        exact_allowed = any(
            token.kind is ScalarKind.DATE
            and token.start == 0
            and token.end == len(matched_text)
            for token in _extract_scalars(matched_text)
        )
        if not exact_allowed:
            return True
    return False


def _is_protected(text: str, start: int, end: int) -> bool:
    return any(
        start < protected_end and protected_start < end
        for protected_start, protected_end in _protected_spans(text)
    )


def _protected_spans(text: str) -> tuple[tuple[int, int], ...]:
    protected: list[tuple[int, int]] = []
    protected.extend(_front_matter_spans(text))
    protected.extend(_fenced_code_spans(text))
    inline_code = _inline_code_spans(text)

    for match in _URL_RE.finditer(text):
        protected.append((match.start(), match.end()))
    for match in _HTML_OPEN_TAG_RE.finditer(text):
        if not _inside_any(match.start(), inline_code):
            protected.append((match.start(), match.end()))
    for match in _REFERENCE_DEFINITION_RE.finditer(text):
        if not _inside_any(match.start(), inline_code):
            protected.append((match.start("target"), match.end("target")))
    for match in _INLINE_LINK_OPEN_RE.finditer(text):
        if _inside_any(match.start(), inline_code):
            continue
        opening = match.end() - 1
        closing = _matching_parenthesis(text, opening)
        if closing is not None:
            protected.append((opening + 1, closing))
    return tuple(sorted(protected))


def _front_matter_spans(text: str) -> tuple[tuple[int, int], ...]:
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].rstrip("\r\n") != "---":
        return ()
    cursor = len(lines[0])
    for line in lines[1:]:
        cursor += len(line)
        if line.rstrip("\r\n") in {"---", "..."}:
            return ((0, cursor),)
    return ()


def _fenced_code_spans(text: str) -> tuple[tuple[int, int], ...]:
    spans: list[tuple[int, int]] = []
    lines = text.splitlines(keepends=True)
    offsets: list[int] = []
    cursor = 0
    for line in lines:
        offsets.append(cursor)
        cursor += len(line)
    index = 0
    while index < len(lines):
        logical = lines[index].rstrip("\r\n")
        opening = _FENCE_OPEN_RE.match(logical)
        if opening is None:
            index += 1
            continue
        marker = opening.group("fence")
        character = re.escape(marker[0])
        closing_re = re.compile(rf"^[ ]{{0,3}}{character}{{{len(marker)},}}[ \t]*$")
        end_index = index + 1
        while end_index < len(lines):
            if closing_re.match(lines[end_index].rstrip("\r\n")):
                end_index += 1
                break
            end_index += 1
        span_end = offsets[end_index] if end_index < len(offsets) else len(text)
        spans.append((offsets[index], span_end))
        index = end_index
    return tuple(spans)


def _inline_code_spans(text: str) -> tuple[tuple[int, int], ...]:
    spans: list[tuple[int, int]] = []
    index = 0
    while index < len(text):
        if text[index] != "`" or _is_escaped(text, index):
            index += 1
            continue
        marker_end = index + 1
        while marker_end < len(text) and text[marker_end] == "`":
            marker_end += 1
        marker = text[index:marker_end]
        closing = text.find(marker, marker_end)
        if closing < 0:
            index = marker_end
            continue
        spans.append((index, closing + len(marker)))
        index = closing + len(marker)
    return tuple(spans)


def _inside_any(index: int, spans: Sequence[tuple[int, int]]) -> bool:
    return any(start <= index < end for start, end in spans)


def _matching_parenthesis(text: str, opening: int) -> int | None:
    depth = 0
    for index in range(opening, len(text)):
        if _is_escaped(text, index):
            continue
        if text[index] == "(":
            depth += 1
        elif text[index] == ")":
            depth -= 1
            if depth == 0:
                return index
        elif text[index] == "\n" and depth == 1:
            return None
    return None


def _is_escaped(text: str, index: int) -> bool:
    backslashes = 0
    index -= 1
    while index >= 0 and text[index] == "\\":
        backslashes += 1
        index -= 1
    return backslashes % 2 == 1


def _forward_bytes(original: bytes, candidates: Sequence[RepairCandidate]) -> bytes:
    ordered = sorted(candidates, key=lambda item: item.byte_start)
    _require_non_overlapping(ordered)
    result = original
    for candidate in reversed(ordered):
        before = candidate.original_scalar.encode("utf-8")
        replacement = candidate.replacement_scalar.encode("utf-8")
        if result[candidate.byte_start : candidate.byte_end] != before:
            raise ValueError(f"target bytes no longer match {candidate.proposal_id!r}")
        result = (
            result[: candidate.byte_start] + replacement + result[candidate.byte_end :]
        )
    return result


def _reverse_bytes(result: bytes, candidates: Sequence[RepairCandidate]) -> bytes:
    ordered = sorted(candidates, key=lambda item: item.byte_start)
    _require_non_overlapping(ordered)
    reverse_edits: list[tuple[int, int, bytes, bytes, str]] = []
    delta = 0
    for candidate in ordered:
        replacement = candidate.replacement_scalar.encode("utf-8")
        original = candidate.original_scalar.encode("utf-8")
        start = candidate.byte_start + delta
        end = start + len(replacement)
        reverse_edits.append((start, end, replacement, original, candidate.proposal_id))
        delta += len(replacement) - len(original)
    restored = result
    for start, end, replacement, original, proposal_id in reversed(reverse_edits):
        if restored[start:end] != replacement:
            raise ValueError(f"replacement bytes no longer match {proposal_id!r}")
        restored = restored[:start] + original + restored[end:]
    return restored


def _require_non_overlapping(candidates: Sequence[RepairCandidate]) -> None:
    for left, right in pairwise(candidates):
        if right.byte_start < left.byte_end:
            raise ValueError("repair candidates overlap")


def _render_diff_set(
    target_contents: Mapping[str, bytes],
    candidates: Sequence[RepairCandidate],
) -> str:
    grouped: dict[str, list[RepairCandidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.target.path].append(candidate)
    rendered: list[str] = []
    for path in sorted(grouped):
        original = target_contents[path]
        after = _forward_bytes(original, grouped[path])
        if _reverse_bytes(after, grouped[path]) != original:
            raise ValueError(f"collective reverse dry-run failed for {path!r}")
        rendered.append(_unified_diff(original, after, path, reverse=False))
    return "".join(rendered)


def _unified_diff(before: bytes, after: bytes, path: str, *, reverse: bool) -> str:
    before_lines = before.decode("utf-8").splitlines(keepends=True)
    after_lines = after.decode("utf-8").splitlines(keepends=True)
    if reverse:
        fromfile, tofile = f"b/{path}", f"a/{path}"
    else:
        fromfile, tofile = f"a/{path}", f"b/{path}"
    pieces = difflib.unified_diff(
        before_lines,
        after_lines,
        fromfile=fromfile,
        tofile=tofile,
        lineterm="\n",
    )
    rendered: list[str] = []
    for piece in pieces:
        rendered.append(piece)
        if not piece.endswith(("\n", "\r")):
            rendered.append("\n\\ No newline at end of file\n")
    return "".join(rendered)


def _snapshot_references_current(
    contents: Mapping[str, bytes], digests: Sequence[FileDigest]
) -> bool:
    return all(
        digest.path in contents and sha256_bytes(contents[digest.path]) == digest.sha256
        for digest in digests
    )


def _references_current(
    root: Path,
    digests: Sequence[FileDigest],
    runtime_paths: _RuntimePathResolution,
) -> bool:
    for digest in digests:
        try:
            path = _resolve_file(root, digest.path, runtime_paths)
            content = path.read_bytes()
        except (OSError, ValueError):
            return False
        if sha256_bytes(content) != digest.sha256:
            return False
    return True


def _validated_runtime_paths(
    runtime_paths: Mapping[str, Path], required_ids: Collection[str]
) -> _RuntimePathResolution:
    resolved: dict[str, Path] = {}
    invalid_ids: set[str] = set()
    identities: dict[tuple[int, int], list[str]] = defaultdict(list)
    required = frozenset(required_ids)
    for safe_id, runtime_path in runtime_paths.items():
        if safe_id not in required:
            continue
        try:
            _validate_relative_path(safe_id)
            path = Path(runtime_path)
            if not path.is_absolute():
                raise ValueError
            canonical = path.resolve(strict=True)
            metadata = canonical.stat(follow_symlinks=False)
        except (OSError, RuntimeError, TypeError, ValueError):
            invalid_ids.add(safe_id)
            continue
        if stat.S_ISREG(metadata.st_mode):
            identities[(metadata.st_dev, metadata.st_ino)].append(safe_id)
        if canonical != path or not stat.S_ISREG(metadata.st_mode):
            invalid_ids.add(safe_id)
            continue
        resolved[safe_id] = canonical
    for safe_ids in identities.values():
        if len(safe_ids) > 1:
            invalid_ids.update(safe_ids)
    for safe_id in invalid_ids:
        resolved.pop(safe_id, None)
    return _RuntimePathResolution(
        paths=resolved,
        invalid_ids=frozenset(invalid_ids),
    )


def _resolve_file(
    root: Path,
    relative: str,
    runtime_paths: _RuntimePathResolution,
) -> Path:
    if relative in runtime_paths.invalid_ids:
        raise ValueError(f"runtime path is invalid: {relative!r}")
    runtime = runtime_paths.paths.get(relative)
    if runtime is not None:
        if runtime.is_symlink() or not runtime.is_file():
            raise ValueError(
                f"runtime path is no longer a regular non-symlink file: {relative!r}"
            )
        if runtime.resolve(strict=True) != runtime:
            raise ValueError(
                f"runtime path changed to a symlink or alias: {relative!r}"
            )
        return runtime
    return _resolve_regular_file(root, relative)


def _resolve_regular_file(root: Path, relative: str) -> Path:
    _validate_relative_path(relative)
    path = root.joinpath(*PurePosixPath(relative).parts)
    cursor = root
    for part in PurePosixPath(relative).parts:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError(f"symlink paths are not writable: {relative!r}")
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError:
        raise ValueError(f"path escaped root: {relative!r}") from None
    if not resolved.is_file():
        raise ValueError(f"path is not a regular file: {relative!r}")
    return resolved


def _fingerprint_metadata(metadata: os.stat_result) -> _FileFingerprint:
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError("target stopped being a regular file")
    return _FileFingerprint(
        device=metadata.st_dev,
        inode=metadata.st_ino,
        size=metadata.st_size,
        modified_ns=metadata.st_mtime_ns,
    )


def _open_directory_nofollow(path: Path) -> int:
    """Open an absolute directory component-by-component without symlinks."""

    if not path.is_absolute():
        raise _UnsafeTargetPath("target parent must be absolute")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path.anchor, flags)
    try:
        for part in path.parts[1:]:
            if part in {".", ".."}:
                raise _UnsafeTargetPath("target parent is not canonical")
            next_descriptor = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _require_parent_identity(path: Path, parent_fd: int) -> None:
    """Require the lexical parent to still identify the pinned directory."""

    current_fd = _open_directory_nofollow(path)
    try:
        expected = os.fstat(parent_fd)
        current = os.fstat(current_fd)
        if (
            not stat.S_ISDIR(expected.st_mode)
            or not stat.S_ISDIR(current.st_mode)
            or (expected.st_dev, expected.st_ino) != (current.st_dev, current.st_ino)
        ):
            raise _UnsafeTargetPath("target parent changed during application")
    finally:
        os.close(current_fd)


def _read_regular_at(parent_fd: int, name: str) -> tuple[_FileFingerprint, bytes]:
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(name, flags, dir_fd=parent_fd)
    try:
        opened = _fingerprint_metadata(os.fstat(descriptor))
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 64 * 1024):
            chunks.append(chunk)
        finished = _fingerprint_metadata(os.fstat(descriptor))
        if finished != opened:
            raise _TargetBecameStale("target changed while its bytes were read")
        entry = _fingerprint_metadata(
            os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        )
        if entry != opened:
            raise _TargetBecameStale("target directory entry changed while read")
        return opened, b"".join(chunks)
    finally:
        os.close(descriptor)


def _pin_regular_file(path: Path) -> tuple[int, _FileFingerprint, bytes]:
    parent_fd = _open_directory_nofollow(path.parent)
    try:
        _require_parent_identity(path.parent, parent_fd)
        fingerprint, content = _read_regular_at(parent_fd, path.name)
    except BaseException:
        os.close(parent_fd)
        raise
    return parent_fd, fingerprint, content


def _create_temporary_at(parent_fd: int) -> tuple[str, int]:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )
    for _ in range(128):
        name = f".evidencetrace-{os.urandom(16).hex()}.tmp"
        try:
            return name, os.open(name, flags, 0o600, dir_fd=parent_fd)
        except FileExistsError:
            continue
    raise OSError("could not allocate a unique temporary file")


def _write_descriptor(descriptor: int, content: bytes) -> None:
    offset = 0
    while offset < len(content):
        written = os.write(descriptor, content[offset:])
        if written <= 0:
            raise OSError("temporary file write made no progress")
        offset += written


def _atomic_replace(
    path: Path,
    content: bytes,
    *,
    parent_fd: int,
    expected_before_sha256: str,
    expected_fingerprint: _FileFingerprint,
) -> None:
    temporary_name: str | None = None
    temporary_fd: int | None = None
    try:
        _require_parent_identity(path.parent, parent_fd)
        metadata = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        if _fingerprint_metadata(metadata) != expected_fingerprint:
            raise _TargetBecameStale("target changed before atomic write preparation")

        temporary_name, temporary_fd = _create_temporary_at(parent_fd)
        os.fchmod(temporary_fd, stat.S_IMODE(metadata.st_mode))
        _write_descriptor(temporary_fd, content)
        os.fsync(temporary_fd)
        os.close(temporary_fd)
        temporary_fd = None

        _require_parent_identity(path.parent, parent_fd)
        current_fingerprint, current = _read_regular_at(parent_fd, path.name)
        if (
            current_fingerprint != expected_fingerprint
            or sha256_bytes(current) != expected_before_sha256
        ):
            raise _TargetBecameStale("target changed during atomic write preparation")
        _require_parent_identity(path.parent, parent_fd)
        os.replace(
            temporary_name,
            path.name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        temporary_name = None
    finally:
        if temporary_fd is not None:
            os.close(temporary_fd)
        if temporary_name is not None:
            with suppress(FileNotFoundError):
                os.unlink(temporary_name, dir_fd=parent_fd)


def _failed_file_result(
    target_path: str,
    status: ApplicationStatus,
    proposal_ids: tuple[str, ...],
    detail: str,
    *,
    before_sha256: str | None = None,
) -> FileApplyResult:
    return FileApplyResult(
        target_path=target_path,
        status=status,
        proposal_ids=proposal_ids,
        before_sha256=before_sha256,
        after_sha256=None,
        suggested_diff="",
        applied_diff="",
        reverse_diff="",
        detail=detail,
    )


def _validate_relative_path(value: str) -> None:
    if not value or value != value.strip() or "\\" in value:
        raise ValueError("path must be a non-empty project-relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or value in {".", ".."} or ".." in path.parts:
        raise ValueError("path must be project-relative and cannot contain '..'")


def _validate_sha256(value: str) -> str:
    if not _SHA256_RE.fullmatch(value):
        raise ValueError("sha256 must be exactly 64 hexadecimal characters")
    return value.lower()


__all__ = [
    "ApplicationDecision",
    "ApplicationStatus",
    "ApplyResult",
    "BeforeFileWrite",
    "ChallengerGate",
    "DecisionStatus",
    "ExactEvidence",
    "FileApplyResult",
    "FileDigest",
    "RepairCandidate",
    "RepairDecision",
    "RepairPlan",
    "RepairProposal",
    "RepairReason",
    "ScalarKind",
    "apply_repair_plan",
    "derive_unique_scalar_pair",
    "plan_repairs",
    "scalar_outcome_signature",
    "sha256_bytes",
]
