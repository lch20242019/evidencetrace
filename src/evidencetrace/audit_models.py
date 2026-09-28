"""Typed contracts used only by the Phase 2 citation-audit path."""

from __future__ import annotations

import hashlib
import math
from pathlib import PurePosixPath
from typing import Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveInt,
    field_validator,
    model_validator,
)

from evidencetrace.model_client import (
    DEFAULT_MAX_TOKENS,
    MODEL_PROVIDER_CONTRACT_VERSION,
    MODEL_REASON_MAX_CHARS,
)
from evidencetrace.models import (
    AtomicClaim,
    Checkability,
    ParagraphMiningReasonCode,
    Relation,
    SourceMetadata,
    Verdict,
)

MINER_DRAFT_CONTRACT_VERSION: Literal["live-miner-draft-v3"] = "live-miner-draft-v3"
MINER_WINDOW_POLICY_VERSION: Literal["miner-window-policy-v1"] = (
    "miner-window-policy-v1"
)
MinerCoveragePolicyVersion = Literal[
    "miner-coverage-policy-v1",
    "miner-coverage-policy-v2",
]
MINER_COVERAGE_POLICY_VERSION: Literal["miner-coverage-policy-v2"] = (
    "miner-coverage-policy-v2"
)
DEFAULT_MINER_MAX_WINDOW_CHARS = 384
JUDGE_SEMANTIC_CONTRACT_VERSION = "live-judge-semantic-v2"
JUDGE_VERDICT_OWNERSHIP_POLICY_VERSION = "local-verdict-ownership-v1"
EVIDENCE_EXACT_MAX_CHARS = 2_000
EVIDENCE_EXACT_MAX_BYTES = 8_000
EVIDENCE_CONTEXT_MAX_CHARS = 6_144
EVIDENCE_CONTEXT_MAX_BYTES = 24_576
EVIDENCE_HEADING_MAX_CHARS = 512
EVIDENCE_HEADING_MAX_BYTES = 2_048
EVIDENCE_LOCATOR_MAX_CHARS = 1_024
EVIDENCE_LOCATOR_MAX_BYTES = 4_096
EVIDENCE_COLLECTION_MAX_ITEMS = 5
EVIDENCE_COLLECTION_EXACT_MAX_CHARS = 6_000
EVIDENCE_COLLECTION_EXACT_MAX_BYTES = 24_000
EVIDENCE_COLLECTION_CONTEXT_MAX_CHARS = 12_288
EVIDENCE_COLLECTION_CONTEXT_MAX_BYTES = 49_152
EVIDENCE_RESOLUTION_MAX_OPTIONS = 10
EVIDENCE_RESOLUTION_EXACT_MAX_CHARS = 12_000
EVIDENCE_RESOLUTION_EXACT_MAX_BYTES = 48_000
EVIDENCE_RESOLUTION_CONTEXT_MAX_CHARS = 24_576
EVIDENCE_RESOLUTION_CONTEXT_MAX_BYTES = 98_304

SafeProviderFinishReason = Literal[
    "stop",
    "length",
    "content_filter",
    "tool_calls",
    "function_call",
    "insufficient_system_resource",
    "unknown",
]
MinerOutcomeFailureCode = ParagraphMiningReasonCode


def bounded_evidence_metadata(value: str) -> str:
    """Bound untrusted display metadata while retaining a deterministic identity."""

    if (
        len(value) <= EVIDENCE_HEADING_MAX_CHARS
        and len(value.encode("utf-8")) <= EVIDENCE_HEADING_MAX_BYTES
    ):
        return value
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    suffix = f" … [sha256:{digest}]"
    end = min(
        len(value),
        EVIDENCE_HEADING_MAX_CHARS - len(suffix),
    )
    while (
        end > 0
        and len((value[:end].rstrip() + suffix).encode("utf-8"))
        > EVIDENCE_HEADING_MAX_BYTES
    ):
        end -= 1
    return value[:end].rstrip() + suffix


def bounded_heading_path(values: tuple[str, ...]) -> tuple[str, ...]:
    joined = " > ".join(values)
    if (
        len(values) <= 6
        and len(joined) <= EVIDENCE_HEADING_MAX_CHARS
        and len(joined.encode("utf-8")) <= EVIDENCE_HEADING_MAX_BYTES
    ):
        return values
    return (bounded_evidence_metadata(joined),) if joined else ()


def evidence_collection_is_bounded(
    values: tuple[RetrievedEvidence, ...],
) -> bool:
    return (
        len(values) <= EVIDENCE_COLLECTION_MAX_ITEMS
        and sum(len(item.text) for item in values)
        <= EVIDENCE_COLLECTION_EXACT_MAX_CHARS
        and sum(len(item.text.encode("utf-8")) for item in values)
        <= EVIDENCE_COLLECTION_EXACT_MAX_BYTES
        and sum(len(item.chunk.text) for item in values)
        <= EVIDENCE_COLLECTION_CONTEXT_MAX_CHARS
        and sum(len(item.chunk.text.encode("utf-8")) for item in values)
        <= EVIDENCE_COLLECTION_CONTEXT_MAX_BYTES
    )


class Phase2Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EvidenceChunk(Phase2Model):
    source_id: str
    url: str
    text: str
    heading_path: tuple[str, ...] = Field(default=(), max_length=6)
    locator: str
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)

    @field_validator("source_id", "text", "locator", "url")
    @classmethod
    def non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("field must not be blank")
        return value

    @field_validator("text")
    @classmethod
    def context_is_bounded(cls, value: str) -> str:
        if (
            len(value) > EVIDENCE_CONTEXT_MAX_CHARS
            or len(value.encode("utf-8")) > EVIDENCE_CONTEXT_MAX_BYTES
        ):
            raise ValueError("evidence context exceeds the hard size limit")
        return value

    @field_validator("heading_path")
    @classmethod
    def heading_path_is_bounded(
        cls, value: tuple[str, ...]
    ) -> tuple[str, ...]:
        joined = " > ".join(value)
        if any(not item.strip() for item in value):
            raise ValueError("heading path entries must not be blank")
        if (
            len(joined) > EVIDENCE_HEADING_MAX_CHARS
            or len(joined.encode("utf-8")) > EVIDENCE_HEADING_MAX_BYTES
        ):
            raise ValueError("evidence heading path exceeds the hard size limit")
        return value

    @field_validator("locator")
    @classmethod
    def locator_is_bounded(cls, value: str) -> str:
        if (
            len(value) > EVIDENCE_LOCATOR_MAX_CHARS
            or len(value.encode("utf-8")) > EVIDENCE_LOCATOR_MAX_BYTES
        ):
            raise ValueError("evidence locator exceeds the hard size limit")
        return value

    @model_validator(mode="after")
    def offsets_are_ordered(self) -> Self:
        if self.char_end < self.char_start:
            raise ValueError("char_end cannot precede char_start")
        return self


class RetrievedEvidence(Phase2Model):
    chunk: EvidenceChunk
    text: str
    score: float

    @field_validator("text")
    @classmethod
    def text_is_real_substring(cls, value: str, info):  # type: ignore[no-untyped-def]
        if not value.strip():
            raise ValueError("retrieved evidence must not be blank")
        if (
            len(value) > EVIDENCE_EXACT_MAX_CHARS
            or len(value.encode("utf-8")) > EVIDENCE_EXACT_MAX_BYTES
        ):
            raise ValueError("retrieved evidence exceeds the hard size limit")
        chunk = info.data.get("chunk")
        if chunk is not None and value not in chunk.text:
            raise ValueError("retrieved evidence must be a source substring")
        return value

    @field_validator("score")
    @classmethod
    def score_is_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("retrieval score must be finite")
        return value


class CheckSignal(Phase2Model):
    code: Literal[
        "numeric_mismatch",
        "date_mismatch",
        "version_mismatch",
        "negation_mismatch",
        "entity_mismatch",
        "qualifier_mismatch",
        "conjunction_mismatch",
        "comparison_reminder",
        "span_missing",
    ]
    detail: str
    severity: Literal["notice", "warning", "error"] = "warning"

    @field_validator("detail")
    @classmethod
    def detail_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("signal detail must not be blank")
        return value

    @model_validator(mode="after")
    def hard_conflicts_are_errors(self) -> Self:
        hard_conflicts = {
            "numeric_mismatch",
            "date_mismatch",
            "version_mismatch",
            "entity_mismatch",
            "negation_mismatch",
        }
        if self.code in hard_conflicts and self.severity != "error":
            raise ValueError("hard-conflict signals must have error severity")
        return self


class MinerInput(Phase2Model):
    text: str
    file: str
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    heading_path: tuple[str, ...] = ()
    citation_urls: tuple[str, ...] = ()

    @field_validator("text")
    @classmethod
    def text_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("paragraph text must not be blank")
        return value

    @field_validator("file")
    @classmethod
    def file_is_project_relative(cls, value: str) -> str:
        if not value or value != value.strip() or "\\" in value:
            raise ValueError("file must be a non-empty project-relative POSIX path")
        path = PurePosixPath(value)
        if path.is_absolute() or value in {".", ".."} or ".." in path.parts:
            raise ValueError("file must be a project-relative path without '..'")
        return value

    @field_validator("heading_path")
    @classmethod
    def headings_are_not_blank(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not value.strip() for value in values):
            raise ValueError("heading path entries must not be blank")
        return values

    @field_validator("citation_urls")
    @classmethod
    def citations_are_not_blank(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not value.strip() for value in values):
            raise ValueError("citation URLs must not be blank")
        return tuple(dict.fromkeys(values))

    @model_validator(mode="after")
    def lines_are_ordered(self) -> Self:
        if self.line_end < self.line_start:
            raise ValueError("line_end cannot precede line_start")
        return self


class MinerOutput(Phase2Model):
    claims: tuple[AtomicClaim, ...] = ()
    model_id: str = "deterministic-miner-v1"
    prompt_version: str = "miner-v1"

    @model_validator(mode="after")
    def claim_ids_are_unique(self) -> Self:
        claim_ids = [claim.claim_id for claim in self.claims]
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("miner claim_ids must be unique")
        return self


class MinerWindowOutcome(Phase2Model):
    """Safe Controller-owned result for one preplanned Miner window."""

    window_id: str
    paragraph_id: str
    source_file: str
    source_line_start: PositiveInt
    source_line_end: PositiveInt
    citation_ids: tuple[str, ...] = ()
    status: Literal[
        "complete",
        "partial",
        "agent_error",
        "budget_exhausted",
        "needs_human",
    ]
    accepted_claim_ids: tuple[str, ...] = ()
    withheld_exact_draft_count: NonNegativeInt = 0
    failure_codes: tuple[MinerOutcomeFailureCode, ...] = ()
    protected_occurrences_total: NonNegativeInt = 0
    protected_occurrences_covered: NonNegativeInt = 0
    provider_attempts: int = Field(default=0, ge=0, le=2)
    schema_retry: bool = False
    finish_reason: SafeProviderFinishReason | None = None
    window_policy_version: Literal["miner-window-policy-v1"] = (
        MINER_WINDOW_POLICY_VERSION
    )
    coverage_policy_version: MinerCoveragePolicyVersion = MINER_COVERAGE_POLICY_VERSION

    @field_validator("source_file")
    @classmethod
    def source_file_is_project_relative(cls, value: str) -> str:
        if not value or value != value.strip() or "\\" in value:
            raise ValueError("source_file must be a project-relative POSIX path")
        path = PurePosixPath(value)
        if path.is_absolute() or value in {".", ".."} or ".." in path.parts:
            raise ValueError("source_file must not escape the project")
        return value

    @model_validator(mode="after")
    def outcome_is_consistent(self) -> Self:
        if self.source_line_end < self.source_line_start:
            raise ValueError("window source lines must be ordered")
        if len(self.accepted_claim_ids) != len(set(self.accepted_claim_ids)):
            raise ValueError("accepted claim IDs must be unique")
        if len(self.failure_codes) != len(set(self.failure_codes)):
            raise ValueError("window failure codes must be unique")
        if self.protected_occurrences_covered > self.protected_occurrences_total:
            raise ValueError("protected coverage cannot exceed its total")
        if self.schema_retry and self.provider_attempts != 2:
            raise ValueError("schema retry requires exactly two provider attempts")
        if self.provider_attempts == 0 and self.finish_reason is not None:
            raise ValueError("zero-attempt window cannot have a finish reason")
        if self.status == "complete":
            if self.failure_codes:
                raise ValueError("complete window cannot carry failure codes")
            if self.protected_occurrences_covered != self.protected_occurrences_total:
                raise ValueError("complete window requires full protected coverage")
        elif not self.failure_codes:
            raise ValueError("non-complete window requires an allowlisted failure code")
        return self


class ParagraphMiningOutcome(Phase2Model):
    """Safe aggregate coverage state for one trusted Markdown paragraph."""

    paragraph_id: str
    source_file: str
    source_line_start: PositiveInt
    source_line_end: PositiveInt
    status: Literal["complete", "partial", "needs_human"]
    window_count: NonNegativeInt
    completed_window_count: NonNegativeInt
    accepted_claim_ids: tuple[str, ...] = ()
    reason_codes: tuple[MinerOutcomeFailureCode, ...] = ()
    protected_occurrences_total: NonNegativeInt = 0
    protected_occurrences_covered: NonNegativeInt = 0
    windows: tuple[MinerWindowOutcome, ...] = ()
    window_policy_version: Literal["miner-window-policy-v1"] = (
        MINER_WINDOW_POLICY_VERSION
    )
    coverage_policy_version: MinerCoveragePolicyVersion = MINER_COVERAGE_POLICY_VERSION

    @field_validator("source_file")
    @classmethod
    def source_file_is_project_relative(cls, value: str) -> str:
        return MinerWindowOutcome.source_file_is_project_relative(value)

    @model_validator(mode="after")
    def paragraph_outcome_is_consistent(self) -> Self:
        if self.source_line_end < self.source_line_start:
            raise ValueError("paragraph source lines must be ordered")
        if self.completed_window_count > self.window_count:
            raise ValueError("completed window count cannot exceed the plan")
        if len(self.windows) != self.window_count:
            raise ValueError("paragraph window count must match its outcomes")
        if any(window.paragraph_id != self.paragraph_id for window in self.windows):
            raise ValueError("paragraph contains a foreign window outcome")
        window_claim_ids = tuple(
            claim_id
            for window in self.windows
            for claim_id in window.accepted_claim_ids
        )
        if window_claim_ids != self.accepted_claim_ids:
            raise ValueError("paragraph accepted claims must preserve window order")
        if len(self.reason_codes) != len(set(self.reason_codes)):
            raise ValueError("paragraph reason codes must be unique")
        if self.protected_occurrences_covered > self.protected_occurrences_total:
            raise ValueError("paragraph protected coverage cannot exceed its total")
        if self.status == "complete":
            if self.completed_window_count != self.window_count:
                raise ValueError("complete paragraph requires every window to complete")
            if self.reason_codes:
                raise ValueError("complete paragraph cannot carry reason codes")
            if self.protected_occurrences_covered != self.protected_occurrences_total:
                raise ValueError("complete paragraph requires full protected coverage")
        elif not self.reason_codes:
            raise ValueError("incomplete paragraph requires an allowlisted reason")
        if self.status == "partial" and not self.accepted_claim_ids:
            raise ValueError("partial paragraph requires at least one accepted claim")
        return self


class MinerExecutionProvenance(Phase2Model):
    """Safe policy dimensions that bound and invalidate windowed Miner work."""

    window_policy_version: Literal["miner-window-policy-v1"] = (
        MINER_WINDOW_POLICY_VERSION
    )
    coverage_policy_version: MinerCoveragePolicyVersion = MINER_COVERAGE_POLICY_VERSION
    miner_contract_version: Literal["live-miner-draft-v3"] = (
        MINER_DRAFT_CONTRACT_VERSION
    )
    schema_recovery_policy_version: Literal["schema-recovery-v1"] = "schema-recovery-v1"
    model_provider_contract_version: str = MODEL_PROVIDER_CONTRACT_VERSION
    model_max_tokens: PositiveInt = DEFAULT_MAX_TOKENS
    max_window_chars: PositiveInt = DEFAULT_MINER_MAX_WINDOW_CHARS
    miner_window_limit: NonNegativeInt
    miner_provider_attempt_limit: NonNegativeInt

    @model_validator(mode="after")
    def provider_attempt_limit_is_bounded(self) -> Self:
        if self.miner_provider_attempt_limit != self.miner_window_limit * 2:
            raise ValueError("Miner provider-attempt limit must be twice window limit")
        return self


class LiveMinedClaimDraft(Phase2Model):
    """Minimal model-authored claim fields before local scope enrichment."""

    text: str = Field(
        description=(
            "One standalone atomic factual claim copied from the current input.text. "
            "The value must be a single, contiguous, character-for-character "
            "substring of input.text. Preserve the original case, whitespace, "
            "punctuation, numbers, versions, and identifiers. Do not paraphrase, "
            "summarize, add words, delete words, or join non-contiguous fragments."
        )
    )
    claim_type: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    checkability: Checkability

    @field_validator("text")
    @classmethod
    def text_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("draft claim text must not be blank")
        return value


class LiveMinerDraftOutput(Phase2Model):
    """Strict live Miner response; all provenance is added locally."""

    claims: tuple[LiveMinedClaimDraft, ...] = Field(
        description=(
            "Claims from the current input.text, emitted in source-text order and "
            "without overlap. Return an empty array for headings, lead-ins, isolated "
            "list labels, or content that cannot independently form an atomic "
            "factual claim. If claims are returned, the collection must not omit "
            "numbers, versions, dates, negations, or comparison qualifiers from "
            "input.text that are related to those claims."
        )
    )


class JudgeInput(Phase2Model):
    claim: AtomicClaim
    evidence: tuple[RetrievedEvidence, ...] = Field(
        default=(),
        max_length=EVIDENCE_COLLECTION_MAX_ITEMS,
    )
    source: SourceMetadata
    signals: tuple[CheckSignal, ...] = ()

    @field_validator("evidence")
    @classmethod
    def evidence_payload_is_bounded(
        cls, value: tuple[RetrievedEvidence, ...]
    ) -> tuple[RetrievedEvidence, ...]:
        if not evidence_collection_is_bounded(value):
            raise ValueError("Judge evidence collection exceeds the hard size limit")
        return value

    @model_validator(mode="after")
    def evidence_belongs_to_source(self) -> Self:
        for item in self.evidence:
            if item.chunk.source_id != self.source.source_id:
                raise ValueError("candidate evidence source_id does not match source")
            if item.chunk.url != self.source.url:
                raise ValueError("candidate evidence URL does not match source")
        return self


class LiveJudgeSemanticOutput(Phase2Model):
    """Model-owned Judge fields before local identity and scope enrichment."""

    relation: Relation
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(max_length=MODEL_REASON_MAX_CHARS)
    evidence_span: str | None

    @field_validator("reason")
    @classmethod
    def reason_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("live Judge reason must not be blank")
        return value

    @model_validator(mode="after")
    def substantive_relation_has_evidence(self) -> Self:
        substantive = {
            Relation.ENTAILED,
            Relation.PARTIALLY_ENTAILED,
            Relation.CONTRADICTED,
        }
        if self.relation in substantive and not self.evidence_span:
            raise ValueError("substantive live Judge relation requires evidence")
        if self.evidence_span is not None and not self.evidence_span.strip():
            raise ValueError("live Judge evidence span must not be blank")
        return self


class JudgeOutput(Phase2Model):
    verdict: Verdict
    model_id: str = "deterministic-judge-v1"
    prompt_version: str = "judge-v1"


class AuditFinding(Phase2Model):
    claim: AtomicClaim
    verdict: Verdict
    signals: tuple[CheckSignal, ...] = ()
    source: SourceMetadata | None = None

    @model_validator(mode="after")
    def claim_and_verdict_match(self) -> Self:
        if self.claim.claim_id != self.verdict.claim_id:
            raise ValueError("finding claim_id does not match verdict claim_id")
        return self


__all__ = [
    "DEFAULT_MINER_MAX_WINDOW_CHARS",
    "EVIDENCE_COLLECTION_CONTEXT_MAX_BYTES",
    "EVIDENCE_COLLECTION_CONTEXT_MAX_CHARS",
    "EVIDENCE_COLLECTION_EXACT_MAX_BYTES",
    "EVIDENCE_COLLECTION_EXACT_MAX_CHARS",
    "EVIDENCE_COLLECTION_MAX_ITEMS",
    "EVIDENCE_CONTEXT_MAX_BYTES",
    "EVIDENCE_CONTEXT_MAX_CHARS",
    "EVIDENCE_EXACT_MAX_BYTES",
    "EVIDENCE_EXACT_MAX_CHARS",
    "EVIDENCE_HEADING_MAX_BYTES",
    "EVIDENCE_HEADING_MAX_CHARS",
    "EVIDENCE_LOCATOR_MAX_BYTES",
    "EVIDENCE_LOCATOR_MAX_CHARS",
    "EVIDENCE_RESOLUTION_CONTEXT_MAX_BYTES",
    "EVIDENCE_RESOLUTION_CONTEXT_MAX_CHARS",
    "EVIDENCE_RESOLUTION_EXACT_MAX_BYTES",
    "EVIDENCE_RESOLUTION_EXACT_MAX_CHARS",
    "EVIDENCE_RESOLUTION_MAX_OPTIONS",
    "JUDGE_SEMANTIC_CONTRACT_VERSION",
    "JUDGE_VERDICT_OWNERSHIP_POLICY_VERSION",
    "MINER_COVERAGE_POLICY_VERSION",
    "MINER_DRAFT_CONTRACT_VERSION",
    "MINER_WINDOW_POLICY_VERSION",
    "AuditFinding",
    "CheckSignal",
    "EvidenceChunk",
    "JudgeInput",
    "JudgeOutput",
    "LiveJudgeSemanticOutput",
    "LiveMinedClaimDraft",
    "LiveMinerDraftOutput",
    "MinerExecutionProvenance",
    "MinerInput",
    "MinerOutcomeFailureCode",
    "MinerOutput",
    "MinerWindowOutcome",
    "ParagraphMiningOutcome",
    "RetrievedEvidence",
    "SafeProviderFinishReason",
    "bounded_evidence_metadata",
    "bounded_heading_path",
    "evidence_collection_is_bounded",
]
