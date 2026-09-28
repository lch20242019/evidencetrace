"""Bounded v0.1 product orchestration over existing audit components."""

from __future__ import annotations

import difflib
import os
import re
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from evidencetrace import __version__
from evidencetrace.agents.challenger import (
    ChallengeAction,
    ChallengerAgent,
    is_high_risk_claim,
)
from evidencetrace.agents.coordinator import (
    COORDINATOR_CALL_LIMIT,
    AuditCoordinatorAgent,
    ExecutionPlan,
    PlanAction,
    PlanStage,
    PlanTask,
)
from evidencetrace.agents.judge import ClaimJudgeAgent
from evidencetrace.agents.miner import (
    ClaimMinerAgent,
    MinerScopeError,
    MinerWindow,
    OversizedMinerFragment,
    miner_protected_occurrence_count,
    plan_miner_windows,
)
from evidencetrace.agents.scout import (
    MAX_FETCHES_PER_CLAIM,
    MAX_QUERIES_PER_CLAIM,
    DiscoveryCandidate,
    DiscoveryResult,
    DiscoveryTools,
    EvidenceScoutAgent,
    SearchClient,
)
from evidencetrace.artifacts import ArtifactManager
from evidencetrace.audit_models import (
    DEFAULT_MINER_MAX_WINDOW_CHARS,
    EVIDENCE_COLLECTION_MAX_ITEMS,
    EVIDENCE_RESOLUTION_CONTEXT_MAX_BYTES,
    EVIDENCE_RESOLUTION_CONTEXT_MAX_CHARS,
    EVIDENCE_RESOLUTION_EXACT_MAX_BYTES,
    EVIDENCE_RESOLUTION_EXACT_MAX_CHARS,
    EVIDENCE_RESOLUTION_MAX_OPTIONS,
    MINER_COVERAGE_POLICY_VERSION,
    MINER_DRAFT_CONTRACT_VERSION,
    MINER_WINDOW_POLICY_VERSION,
    CheckSignal,
    JudgeInput,
    MinerExecutionProvenance,
    MinerInput,
    MinerOutcomeFailureCode,
    MinerWindowOutcome,
    ParagraphMiningOutcome,
    RetrievedEvidence,
    SafeProviderFinishReason,
    evidence_collection_is_bounded,
)
from evidencetrace.checks.deterministic import (
    bounded_evidence_text,
    deterministic_signals,
)
from evidencetrace.gitdiff import collect_git_diff, select_changed_paragraphs
from evidencetrace.markdown import parse_markdown_file_with_source_map
from evidencetrace.model_client import (
    DEFAULT_MAX_TOKENS,
    MODEL_PROVIDER_CONTRACT_VERSION,
    SCHEMA_RECOVERY_POLICY_VERSION,
    ModelCallBudgetExceeded,
    ModelCallTelemetry,
    ModelClient,
    ModelResponseError,
    ModelSchemaError,
    ModelTransportError,
    SchemaFailureCategory,
)
from evidencetrace.models import (
    AtomicClaim,
    AuditArtifact,
    Checkability,
    ClaimOperationalOutcome,
    ClaimOperationalReasonCode,
    ClaimOperationalStatus,
    EffectiveConfig,
    ParagraphMiningAuditOutcome,
    ParsedDocumentWithSourceMap,
    Relation,
    RunMetadata,
    SourceMetadata,
    TrustedModelVisibleParagraph,
    Verdict,
)
from evidencetrace.render import render_audit_markdown, render_terminal
from evidencetrace.repair import scalar_outcome_signature
from evidencetrace.retrieval.fetch import SafeFetcher

PRODUCT_MAX_PROVIDER_ATTEMPTS = 304
FailureCode = Literal[
    "budget_exhausted",
    "challenger_error",
    "coordinator_plan_rejected",
    "discovery_unavailable",
    "evidence_not_found",
    "human_review_requested",
    "judge_error",
    "miner_error",
    "miner_local_validation_error",
    "miner_model_response_error",
    "miner_schema_constraint",
    "miner_schema_empty_content",
    "miner_schema_enum",
    "miner_schema_extra_field",
    "miner_schema_finish_reason",
    "miner_schema_invalid_envelope",
    "miner_schema_invalid_json",
    "miner_schema_missing_field",
    "miner_schema_wrong_type",
    "miner_scope_ambiguous_span",
    "miner_scope_invalid_claim_fragment",
    "miner_scope_missing_protected_token",
    "miner_scope_non_source_span",
    "miner_transport_error",
    "miner_unknown_error",
    "miner_coverage_empty_identifier_window",
    "miner_coverage_incomplete",
    "miner_window_fragment_oversized",
    "scout_error",
    "source_error",
]
ProviderFinishReason = SafeProviderFinishReason
_ROUTING_DATE_RE = re.compile(
    r"\b(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|"
    r"(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+\d{1,2},\s+\d{4}|"
    r"\d{1,2}\s+(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+\d{4})\b",
    re.IGNORECASE,
)
_ROUTING_SEMVER_RE = re.compile(
    r"(?<![A-Za-z0-9])v?\d+\.\d+\.\d+"
    r"(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?(?![A-Za-z0-9])"
)
_ROUTING_PERCENT_RE = re.compile(r"(?<![\w.])\d+(?:\.\d+)?%(?!\w)")
_ROUTING_NUMBER_RE = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.%])")
_MINER_SCHEMA_FAILURE_CODES: dict[SchemaFailureCategory, FailureCode] = {
    "invalid_envelope": "miner_schema_invalid_envelope",
    "empty_content": "miner_schema_empty_content",
    "invalid_json": "miner_schema_invalid_json",
    "finish_reason": "miner_schema_finish_reason",
    "missing_field": "miner_schema_missing_field",
    "extra_field": "miner_schema_extra_field",
    "wrong_type": "miner_schema_wrong_type",
    "enum": "miner_schema_enum",
    "schema_constraint": "miner_schema_constraint",
}
_MINER_SCOPE_FAILURE_CODES: dict[str, FailureCode] = {
    "non_source_span": "miner_scope_non_source_span",
    "ambiguous_span": "miner_scope_ambiguous_span",
    "missing_protected_token": "miner_scope_missing_protected_token",
    "invalid_claim_fragment": "miner_scope_invalid_claim_fragment",
}


def _miner_failure_code(error: Exception) -> FailureCode:
    if isinstance(error, ModelTransportError):
        return "miner_transport_error"
    if isinstance(error, ModelSchemaError):
        diagnostic = error.diagnostic
        if diagnostic is not None:
            return _MINER_SCHEMA_FAILURE_CODES[diagnostic.failure_category]
        return "miner_model_response_error"
    if isinstance(error, MinerScopeError):
        return _MINER_SCOPE_FAILURE_CODES[error.code]
    if isinstance(error, ModelResponseError):
        return "miner_model_response_error"
    if isinstance(
        error,
        (ValidationError, ValueError, RuntimeError, AssertionError, KeyError),
    ):
        return "miner_local_validation_error"
    return "miner_unknown_error"


def _routing_scalar_tokens(value: str) -> tuple[tuple[str, str], ...]:
    """Return non-overlapping scalar tokens for conservative source grouping."""

    occupied: list[tuple[int, int]] = []
    tokens: list[tuple[int, str, str]] = []
    patterns = (
        ("date", _ROUTING_DATE_RE),
        ("semver", _ROUTING_SEMVER_RE),
        ("percent", _ROUTING_PERCENT_RE),
        ("number", _ROUTING_NUMBER_RE),
    )
    for kind, pattern in patterns:
        for match in pattern.finditer(value):
            if any(
                match.start() < end and start < match.end()
                for start, end in occupied
            ):
                continue
            occupied.append((match.start(), match.end()))
            tokens.append((match.start(), kind, match.group(0).casefold()))
    return tuple((kind, token) for _, kind, token in sorted(tokens))


def _routing_replacement_signature(
    claim_text: str, evidence_text: str
) -> tuple[str, str] | None:
    """Identify one typed scalar outcome, including unit compatibility."""

    signature = scalar_outcome_signature(claim_text, evidence_text)
    if signature is None:
        return None
    kind, semantic, unit = signature
    return kind.value, f"{semantic}|{unit or '-'}"


def _model_telemetry_events(
    client: object | None,
) -> tuple[ModelCallTelemetry, ...] | None:
    events = getattr(client, "telemetry_events", None)
    if not isinstance(events, tuple) or not all(
        isinstance(event, ModelCallTelemetry) for event in events
    ):
        return None
    return events


def _miner_attempt_details(
    client: object | None,
    before_count: int | None,
    *,
    assume_attempt: bool,
) -> tuple[int, bool, ProviderFinishReason | None]:
    current = _model_telemetry_events(client)
    if before_count is None or current is None or len(current) < before_count:
        return (int(client is not None and assume_attempt), False, None)
    events = current[before_count:]
    finish_reason = (
        cast(ProviderFinishReason | None, events[-1].finish_reason) if events else None
    )
    return (
        len(events),
        any(event.schema_retry for event in events),
        finish_reason,
    )


def _client_setting(client: object | None, name: str, default: Any) -> Any:
    """Read one safe request setting through bounded wrapper layers."""

    current = client
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        value = getattr(current, name, None)
        if value is not None:
            return value
        current = getattr(current, "wrapped", None)
    return default


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ClaimRunStatus(StrEnum):
    COMPLETED = "completed"
    AGENT_ERROR = "agent_error"
    SOURCE_ERROR = "source_error"
    BUDGET_EXHAUSTED = "budget_exhausted"
    NEEDS_HUMAN = "needs_human"


class DocumentRunStatus(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    FAILED = "failed"


class TraceState(StrEnum):
    PLANNED = "planned"
    DISPATCHED = "dispatched"
    COMPLETED = "completed"
    FAILED = "failed"
    FALLBACK = "fallback"


class AgentName(StrEnum):
    MINER = "miner"
    COORDINATOR = "coordinator"
    SCOUT = "scout"
    JUDGE = "judge"
    CHALLENGER = "challenger"
    CONTROLLER = "controller"


class EvidenceOrigin(StrEnum):
    CITATION = "citation"
    REFERENCE = "reference"
    TAVILY = "tavily"


class EvidenceResolutionReason(StrEnum):
    AUTOMATIC = "automatic"
    TAVILY_ONLY = "tavily_only"
    CONFLICT = "conflict"


class EvidenceResolutionStatus(StrEnum):
    SELECTED = "selected"
    HUMAN_CORROBORATED = "human_corroborated"
    HUMAN_RESOLVED_CONFLICT = "human_resolved_conflict"
    NEEDS_HUMAN = "needs_human"


class EvidenceOption(_Model):
    option_id: str
    origin: EvidenceOrigin
    source: SourceMetadata
    evidence: tuple[RetrievedEvidence, ...] = Field(
        default=(),
        max_length=EVIDENCE_COLLECTION_MAX_ITEMS,
    )
    reference_id: str | None = None
    reference_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )

    @model_validator(mode="after")
    def reference_metadata_matches_origin(self) -> EvidenceOption:
        is_reference = self.origin is EvidenceOrigin.REFERENCE
        if is_reference != (
            self.reference_id is not None and self.reference_sha256 is not None
        ):
            raise ValueError("reference metadata must match reference evidence")
        if not evidence_collection_is_bounded(self.evidence):
            raise ValueError("evidence option exceeds the hard size limit")
        return self


class EvidenceResolution(_Model):
    claim_id: str
    reason: EvidenceResolutionReason
    status: EvidenceResolutionStatus
    options: tuple[EvidenceOption, ...] = Field(
        default=(),
        max_length=EVIDENCE_RESOLUTION_MAX_OPTIONS,
    )
    selected_option_id: str | None = None
    selected_source_id: str | None = None
    selected_origin: EvidenceOrigin | None = None
    reference_id: str | None = None
    reference_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )

    @model_validator(mode="after")
    def selected_fields_are_consistent(self) -> EvidenceResolution:
        selected = self.status is not EvidenceResolutionStatus.NEEDS_HUMAN
        fields = (
            self.selected_option_id,
            self.selected_source_id,
            self.selected_origin,
        )
        if selected != all(value is not None for value in fields):
            raise ValueError("selected evidence resolution is incomplete")
        if not selected and any(value is not None for value in fields):
            raise ValueError("unresolved evidence cannot select an option")
        if (self.reference_id is None) != (self.reference_sha256 is None):
            raise ValueError("reference id and hash must be recorded together")
        if len({option.option_id for option in self.options}) != len(self.options):
            raise ValueError("evidence resolution option IDs must be unique")
        evidence = tuple(
            item for option in self.options for item in option.evidence
        )
        if (
            sum(len(item.text) for item in evidence)
            > EVIDENCE_RESOLUTION_EXACT_MAX_CHARS
            or sum(len(item.text.encode("utf-8")) for item in evidence)
            > EVIDENCE_RESOLUTION_EXACT_MAX_BYTES
            or sum(len(item.chunk.text) for item in evidence)
            > EVIDENCE_RESOLUTION_CONTEXT_MAX_CHARS
            or sum(len(item.chunk.text.encode("utf-8")) for item in evidence)
            > EVIDENCE_RESOLUTION_CONTEXT_MAX_BYTES
        ):
            raise ValueError("evidence resolution exceeds the hard size limit")
        chosen = next(
            (
                option
                for option in self.options
                if option.option_id == self.selected_option_id
            ),
            None,
        )
        if selected and chosen is None:
            raise ValueError("selected evidence must be one of the bounded options")
        if chosen is not None and (
            chosen.source.source_id != self.selected_source_id
            or chosen.origin is not self.selected_origin
            or chosen.reference_id != self.reference_id
            or chosen.reference_sha256 != self.reference_sha256
        ):
            raise ValueError("selected evidence metadata does not match its option")
        if (
            self.status is EvidenceResolutionStatus.HUMAN_CORROBORATED
            and (
                self.reason is not EvidenceResolutionReason.TAVILY_ONLY
                or self.selected_origin is not EvidenceOrigin.TAVILY
            )
        ):
            raise ValueError("human corroboration must select Tavily evidence")
        if (
            self.status is EvidenceResolutionStatus.HUMAN_RESOLVED_CONFLICT
            and self.reason is not EvidenceResolutionReason.CONFLICT
        ):
            raise ValueError("human conflict resolution requires a conflict")
        return self


class LocalEvidenceLookup(Protocol):
    def lookup(
        self, claim: AtomicClaim, *, top_k: int = 5
    ) -> tuple[Any, ...]: ...


class HumanEvidenceResolver(Protocol):
    def choose_evidence(
        self,
        claim: AtomicClaim,
        reason: EvidenceResolutionReason,
        options: tuple[EvidenceOption, ...],
    ) -> str | None: ...


class TraceEvent(_Model):
    sequence: int = Field(ge=1)
    state: TraceState
    agent: AgentName
    action: str
    claim_id: str | None = None
    code: FailureCode | None = None
    miner_dispatch_id: str | None = Field(default=None, pattern=r"^miner_[0-9]{4}$")
    paragraph_id: str | None = Field(default=None, pattern=r"^p_[0-9]{4}$")
    window_id: str | None = Field(default=None, pattern=r"^p_[0-9]{4}_w_[0-9]{4}$")
    provider_attempts: int | None = Field(default=None, ge=0, le=2)
    schema_retry: bool | None = None
    finish_reason: ProviderFinishReason | None = None


class ClaimRunRecord(_Model):
    claim_id: str
    status: ClaimRunStatus
    initial_action: PlanAction
    review_action: PlanAction
    error_code: FailureCode | None = None
    challenger_action: ChallengeAction | None = None

    @model_validator(mode="after")
    def challenger_state_is_typed(self) -> ClaimRunRecord:
        if self.challenger_action is None:
            return self
        if self.review_action is not PlanAction.CHALLENGE:
            raise ValueError("Challenger outcome requires a review challenge")
        if (
            self.challenger_action is ChallengeAction.ABSTAIN
            and self.status is not ClaimRunStatus.NEEDS_HUMAN
        ):
            raise ValueError("Challenger abstention must require human review")
        if (
            self.challenger_action
            in {ChallengeAction.UPHOLD, ChallengeAction.REVISE}
            and self.status is not ClaimRunStatus.COMPLETED
        ):
            raise ValueError("successful Challenger outcome must be completed")
        return self


class ProductRunArtifact(_Model):
    artifact_version: Literal[
        "bounded-multi-agent-product-v1",
        "bounded-multi-agent-product-v2",
    ] = "bounded-multi-agent-product-v2"
    document_status: DocumentRunStatus
    claim_runs: tuple[ClaimRunRecord, ...]
    trace: tuple[TraceEvent, ...]
    budget: dict[str, int]
    live_agents_enabled: bool
    discover_enabled: bool
    paragraph_mining_outcomes: tuple[ParagraphMiningOutcome, ...] = ()
    miner_provenance: MinerExecutionProvenance | None = None

    @model_validator(mode="after")
    def artifact_version_matches_miner_state(self) -> ProductRunArtifact:
        if self.artifact_version == "bounded-multi-agent-product-v1":
            if self.paragraph_mining_outcomes or self.miner_provenance is not None:
                raise ValueError(
                    "v1 product artifact cannot contain windowed Miner state"
                )
        elif self.miner_provenance is None:
            raise ValueError("v2 product artifact requires Miner provenance")
        return self


@dataclass(frozen=True)
class ProductRunResult:
    audit: AuditArtifact
    product: ProductRunArtifact
    audit_path: Path | None
    terminal: str
    evidence_resolutions: tuple[EvidenceResolution, ...] = ()


@dataclass
class _Claim:
    claim: AtomicClaim
    status: ClaimRunStatus = ClaimRunStatus.COMPLETED
    initial: PlanAction = PlanAction.REQUEST_HUMAN
    review: PlanAction = PlanAction.REQUEST_HUMAN
    verdict: Verdict | None = None
    evidence: tuple[RetrievedEvidence, ...] = ()
    signals: tuple[CheckSignal, ...] = ()
    used_scout: bool = False
    error: FailureCode | None = None
    judge_calls: int = 0
    challenger_calls: int = 0
    challenger_action: ChallengeAction | None = None
    evidence_options: list[EvidenceOption] = field(default_factory=list)
    resolution: EvidenceResolution | None = None


@dataclass(frozen=True)
class _MiningResult:
    claims: tuple[_Claim, ...]
    paragraphs: tuple[ParagraphMiningOutcome, ...]


@dataclass
class _Budget:
    max_claims: int
    max_searches: int
    max_fetches: int
    miner_window_limit: int
    searches: int = 0
    fetches: int = 0
    judges: int = 0
    challengers: int = 0
    miner_windows_planned: int = 0
    miner_windows_dispatched: int = 0
    miner_provider_attempts: int = 0

    @property
    def miner_provider_attempt_limit(self) -> int:
        return self.miner_window_limit * 2

    def discovery_slots(self) -> int:
        return min(
            max((self.max_searches - self.searches) // MAX_QUERIES_PER_CLAIM, 0),
            max((self.max_fetches - self.fetches) // MAX_FETCHES_PER_CLAIM, 0),
        )

    def freeze(self, coordinator_calls: int, trace: list[TraceEvent]) -> dict[str, int]:
        return {
            "claim_limit": self.max_claims,
            "miner_calls": self.miner_windows_dispatched,
            "miner_window_limit": self.miner_window_limit,
            "miner_windows_planned": self.miner_windows_planned,
            "miner_windows_dispatched": self.miner_windows_dispatched,
            "miner_provider_attempt_limit": self.miner_provider_attempt_limit,
            "miner_provider_attempts": self.miner_provider_attempts,
            "coordinator_call_limit": COORDINATOR_CALL_LIMIT,
            "coordinator_calls": coordinator_calls,
            "search_query_limit": self.max_searches,
            "search_queries": self.searches,
            "search_queries_remaining": max(self.max_searches - self.searches, 0),
            "fetch_limit": self.max_fetches,
            "fetches": self.fetches,
            "fetches_remaining": max(self.max_fetches - self.fetches, 0),
            "judge_calls": self.judges,
            "challenger_calls": self.challengers,
        }


@dataclass
class _Context:
    budget: _Budget
    trace: list[TraceEvent]
    sources: dict[str, SourceMetadata]
    tools: DiscoveryTools
    miner_global_budget_exhausted: bool = False

    def event(
        self,
        state: TraceState,
        agent: AgentName,
        action: str,
        claim_id: str | None = None,
        code: FailureCode | None = None,
        *,
        miner_dispatch_id: str | None = None,
        paragraph_id: str | None = None,
        window_id: str | None = None,
        provider_attempts: int | None = None,
        schema_retry: bool | None = None,
        finish_reason: ProviderFinishReason | None = None,
    ) -> None:
        self.trace.append(
            TraceEvent(
                sequence=len(self.trace) + 1,
                state=state,
                agent=agent,
                action=action,
                claim_id=claim_id,
                code=code,
                miner_dispatch_id=miner_dispatch_id,
                paragraph_id=paragraph_id,
                window_id=window_id,
                provider_attempts=provider_attempts,
                schema_retry=schema_retry,
                finish_reason=finish_reason,
            )
        )


class PlanRejectedError(RuntimeError):
    pass


class ProductPipelineError(RuntimeError):
    pass


def validate_execution_plan(
    plan: ExecutionPlan,
    claims: tuple[_Claim, ...],
    *,
    stage: PlanStage,
    discover: bool,
    discovery_slots: int,
) -> ExecutionPlan:
    if plan.stage is not stage:
        raise PlanRejectedError("wrong stage")
    expected = {item.claim.claim_id for item in claims}
    ids = [task.claim_id for task in plan.tasks]
    if len(ids) != len(set(ids)) or set(ids) != expected:
        raise PlanRejectedError("invalid claim ownership")
    by_id = {item.claim.claim_id: item for item in claims}
    discovery_count = 0
    unresolved = {
        Relation.NOT_IN_SOURCE,
        Relation.SOURCE_UNAVAILABLE,
        Relation.PARTIALLY_ENTAILED,
    }
    for task in plan.tasks:
        item = by_id[task.claim_id]
        if stage is PlanStage.INITIAL:
            if (
                task.action is PlanAction.VERIFY_CITATION
                and not item.claim.citation_urls
            ):
                raise PlanRejectedError("citation missing")
            if task.action is PlanAction.DISCOVER:
                discovery_count += 1
                if (
                    not discover
                    or bool(item.claim.citation_urls)
                    or item.claim.checkability is Checkability.NOT_CHECKABLE
                ):
                    raise PlanRejectedError("discovery not authorized")
            if (
                task.action is PlanAction.SKIP_NOT_CHECKABLE
                and item.claim.checkability is not Checkability.NOT_CHECKABLE
            ):
                raise PlanRejectedError("skip not authorized")
        elif (
            item.status is not ClaimRunStatus.COMPLETED
            and task.action is not PlanAction.REQUEST_HUMAN
        ):
            raise PlanRejectedError("failed claim action not authorized")
        elif task.action is PlanAction.CHALLENGE:
            if item.verdict is None or not is_high_risk_claim(
                item.claim, item.verdict, item.signals
            ):
                raise PlanRejectedError("challenge not authorized")
        elif task.action is PlanAction.COUNTER_SEARCH:
            discovery_count += 1
            if (
                not discover
                or item.used_scout
                or item.status is not ClaimRunStatus.COMPLETED
                or item.verdict is None
                or item.verdict.relation not in unresolved
            ):
                raise PlanRejectedError("counter-search not authorized")
    if discovery_count > discovery_slots:
        raise PlanRejectedError("discovery budget exceeded")
    return plan


def _fallback(
    claim: AtomicClaim,
    relation: Relation,
    reason: str,
    *,
    confidence: float = 0.0,
) -> Verdict:
    return Verdict(
        claim_id=claim.claim_id,
        relation=relation,
        confidence=confidence,
        reason=reason,
        judge_version="product-controller-v1",
    )


def _fallback_plan(
    stage: PlanStage,
    claims: tuple[_Claim, ...],
    *,
    discover: bool,
    limit: int,
) -> ExecutionPlan:
    used = 0
    tasks = []
    unresolved = {
        Relation.NOT_IN_SOURCE,
        Relation.SOURCE_UNAVAILABLE,
        Relation.PARTIALLY_ENTAILED,
    }
    for item in claims:
        if stage is PlanStage.INITIAL:
            if item.claim.checkability is Checkability.NOT_CHECKABLE:
                action = PlanAction.SKIP_NOT_CHECKABLE
            elif item.claim.citation_urls:
                action = PlanAction.VERIFY_CITATION
            elif discover and used < limit:
                action, used = PlanAction.DISCOVER, used + 1
            else:
                action = PlanAction.REQUEST_HUMAN
        elif item.status is not ClaimRunStatus.COMPLETED:
            action = PlanAction.REQUEST_HUMAN
        elif (
            discover
            and not item.used_scout
            and item.verdict
            and item.verdict.relation in unresolved
            and used < limit
        ):
            action, used = PlanAction.COUNTER_SEARCH, used + 1
        elif item.verdict and is_high_risk_claim(
            item.claim, item.verdict, item.signals
        ):
            action = PlanAction.CHALLENGE
        else:
            action = PlanAction.ACCEPT
        tasks.append(PlanTask(claim_id=item.claim.claim_id, action=action))
    return ExecutionPlan(stage=stage, tasks=tuple(tasks))


class ProductAuditPipeline:
    def __init__(
        self,
        *,
        project_root: Path,
        model: ModelClient | None = None,
        fetcher: SafeFetcher | None = None,
        search_client: SearchClient | None = None,
        reference_index: LocalEvidenceLookup | None = None,
        evidence_resolver: HumanEvidenceResolver | None = None,
        enforce_human_evidence_gates: bool = False,
        config: EffectiveConfig | None = None,
        max_window_chars: int = DEFAULT_MINER_MAX_WINDOW_CHARS,
        miner_window_limit: int | None = None,
    ) -> None:
        if (
            isinstance(max_window_chars, bool)
            or not isinstance(max_window_chars, int)
            or max_window_chars <= 0
        ):
            raise ValueError("max_window_chars must be a positive integer")
        if miner_window_limit is not None and (
            isinstance(miner_window_limit, bool)
            or not isinstance(miner_window_limit, int)
            or miner_window_limit < 0
        ):
            raise ValueError("miner_window_limit must be a non-negative integer")
        self.root = project_root.resolve()
        self.config = config or EffectiveConfig()
        self.max_window_chars = max_window_chars
        self.miner_window_limit = (
            self.config.budget.max_changed_claims
            if miner_window_limit is None
            else miner_window_limit
        )
        self.model = model
        self.fetcher = fetcher or SafeFetcher(
            timeout=self.config.budget.timeout_seconds
        )
        self.search_client = search_client
        self.reference_index = reference_index
        self.evidence_resolver = evidence_resolver
        self.enforce_human_evidence_gates = enforce_human_evidence_gates
        self.miner = ClaimMinerAgent(model)
        self.coordinator = AuditCoordinatorAgent(model)
        self.judge = ClaimJudgeAgent(model)
        self.scout = EvidenceScoutAgent(model)
        self.challenger = ChallengerAgent(model)

    def run(
        self,
        markdown_path: Path,
        *,
        discover: bool = False,
        changed_from: str | None = None,
        run_id: str | None = None,
        started_at: datetime | None = None,
        persist: bool = True,
        parsed_document: ParsedDocumentWithSourceMap | None = None,
    ) -> ProductRunResult:
        try:
            parsed = (
                parsed_document
                if parsed_document is not None
                else parse_markdown_file_with_source_map(
                    markdown_path, repo_root=self.root
                )
            )
            document = parsed.document
            paragraphs = self._paragraphs(
                document, parsed.model_visible_paragraphs, changed_from
            )
        except Exception:
            raise ProductPipelineError("document setup failed") from None
        self.coordinator.calls = 0
        context = _Context(
            budget=_Budget(
                self.config.budget.max_changed_claims,
                self.config.budget.max_searches,
                self.config.budget.max_fetches,
                self.miner_window_limit,
            ),
            trace=[],
            sources={},
            tools=DiscoveryTools(self.search_client, self.fetcher),
        )
        mining = self._mine(paragraphs, context)
        claims = mining.claims
        self._attach_local_reference_evidence(claims, context)
        initial = self._plan(PlanStage.INITIAL, claims, discover, context)
        for task in initial.tasks:
            item = self._claim(claims, task.claim_id)
            item.initial = task.action
            self._initial(item, task.action, discover, context)
        self._pre_review_counter_search(claims, discover, context)
        review = self._plan(PlanStage.REVIEW, claims, discover, context)
        for task in review.tasks:
            item = self._claim(claims, task.claim_id)
            item.review = task.action
            self._review(item, task.action, context)
        partial = any(
            paragraph.status != "complete" for paragraph in mining.paragraphs
        ) or any(item.status is not ClaimRunStatus.COMPLETED for item in claims)
        now = started_at or datetime.now(UTC)
        audit = AuditArtifact(
            run=RunMetadata(
                run_id=run_id
                or f"{now.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}",
                started_at=now,
                finished_at=now,
                tool_version=__version__,
            ),
            effective_config=self.config,
            documents=(document,),
            claims=tuple(item.claim for item in claims),
            verdicts=tuple(
                item.verdict
                for item in claims
                if item.verdict
                and item.status
                in {
                    ClaimRunStatus.COMPLETED,
                    ClaimRunStatus.SOURCE_ERROR,
                }
            ),
            sources=tuple(context.sources.values()),
            paragraph_mining_outcomes=tuple(
                ParagraphMiningAuditOutcome(
                    paragraph_id=paragraph.paragraph_id,
                    file=paragraph.source_file,
                    line_start=paragraph.source_line_start,
                    line_end=paragraph.source_line_end,
                    status=paragraph.status,
                    window_count=paragraph.window_count,
                    completed_window_count=paragraph.completed_window_count,
                    accepted_claim_ids=paragraph.accepted_claim_ids,
                    reason_codes=paragraph.reason_codes,
                    protected_occurrences_total=(paragraph.protected_occurrences_total),
                    protected_occurrences_covered=(
                        paragraph.protected_occurrences_covered
                    ),
                )
                for paragraph in mining.paragraphs
            ),
            claim_operational_outcomes=tuple(
                ClaimOperationalOutcome(
                    claim_id=item.claim.claim_id,
                    file=item.claim.file,
                    line_start=item.claim.line_start,
                    line_end=item.claim.line_end,
                    status=cast(ClaimOperationalStatus, item.status.value),
                    reason_code=cast(
                        ClaimOperationalReasonCode,
                        item.error or "human_review_requested",
                    ),
                )
                for item in claims
                if item.status
                not in {
                    ClaimRunStatus.COMPLETED,
                    ClaimRunStatus.SOURCE_ERROR,
                }
            ),
        )
        product = ProductRunArtifact(
            document_status=(
                DocumentRunStatus.PARTIAL if partial else DocumentRunStatus.COMPLETE
            ),
            claim_runs=tuple(
                ClaimRunRecord(
                    claim_id=item.claim.claim_id,
                    status=item.status,
                    initial_action=item.initial,
                    review_action=item.review,
                    error_code=item.error,
                    challenger_action=item.challenger_action,
                )
                for item in claims
            ),
            trace=tuple(context.trace),
            budget=context.budget.freeze(self.coordinator.calls, context.trace),
            live_agents_enabled=self.model is not None,
            discover_enabled=discover,
            paragraph_mining_outcomes=mining.paragraphs,
            miner_provenance=self._miner_provenance(),
        )
        audit_path = self._persist(audit, product) if persist else None
        return ProductRunResult(
            audit,
            product,
            audit_path,
            self._terminal(audit, product),
            tuple(
                item.resolution
                for item in claims
                if item.resolution is not None
            ),
        )

    def _paragraphs(
        self,
        document: Any,
        trusted: tuple[TrustedModelVisibleParagraph, ...],
        changed_from: str | None,
    ) -> tuple[TrustedModelVisibleParagraph, ...]:
        if changed_from is None:
            return trusted
        diff = collect_git_diff(self.root, changed_from, paths=(document.path,))
        change = next(
            (
                item
                for item in diff.files
                if document.path in {item.old_path, item.new_path}
            ),
            None,
        )
        if change is None:
            return ()
        selected = set(select_changed_paragraphs(document, change).paragraph_ids)
        return tuple(x for x in trusted if x.paragraph_id in selected)

    def _miner_provenance(self) -> MinerExecutionProvenance:
        client = getattr(self.miner, "client", None)
        model_max_tokens = _client_setting(client, "max_tokens", DEFAULT_MAX_TOKENS)
        if (
            isinstance(model_max_tokens, bool)
            or not isinstance(model_max_tokens, int)
            or model_max_tokens <= 0
        ):
            model_max_tokens = DEFAULT_MAX_TOKENS
        return MinerExecutionProvenance(
            window_policy_version=MINER_WINDOW_POLICY_VERSION,
            coverage_policy_version=MINER_COVERAGE_POLICY_VERSION,
            miner_contract_version=MINER_DRAFT_CONTRACT_VERSION,
            schema_recovery_policy_version=SCHEMA_RECOVERY_POLICY_VERSION,
            model_provider_contract_version=MODEL_PROVIDER_CONTRACT_VERSION,
            model_max_tokens=model_max_tokens,
            max_window_chars=self.max_window_chars,
            miner_window_limit=self.miner_window_limit,
            miner_provider_attempt_limit=self.miner_window_limit * 2,
        )

    @staticmethod
    def _localize_window_claims(
        paragraph: TrustedModelVisibleParagraph,
        window: MinerWindow,
        claims: tuple[AtomicClaim, ...],
    ) -> tuple[AtomicClaim, ...]:
        localized: list[AtomicClaim] = []
        cursor = 0
        for claim in claims:
            start = window.text.find(claim.text, cursor)
            if start < 0:
                raise RuntimeError("validated Miner claim lost its local source span")
            end = start + len(claim.text)
            line_start, line_end = paragraph.source_line_range(
                window.start + start, window.start + end
            )
            localized.append(
                claim.model_copy(
                    update={
                        "file": window.source_file,
                        "line_start": line_start,
                        "line_end": line_end,
                        "slots": {},
                        "citation_urls": window.citation_urls,
                    }
                )
            )
            cursor = end
        return tuple(localized)

    @staticmethod
    def _failed_window_outcome(
        item: MinerWindow | OversizedMinerFragment,
        *,
        status: Literal["partial", "agent_error", "budget_exhausted", "needs_human"],
        code: MinerOutcomeFailureCode,
        protected_total: int,
        protected_covered: int = 0,
        withheld_exact_draft_count: int = 0,
        provider_attempts: int = 0,
        schema_retry: bool = False,
        finish_reason: ProviderFinishReason | None = None,
    ) -> MinerWindowOutcome:
        return MinerWindowOutcome(
            window_id=item.window_id,
            paragraph_id=item.paragraph_id,
            source_file=item.source_file,
            source_line_start=item.source_line_start,
            source_line_end=item.source_line_end,
            citation_ids=item.citation_ids,
            status=status,
            withheld_exact_draft_count=withheld_exact_draft_count,
            failure_codes=(code,),
            protected_occurrences_total=protected_total,
            protected_occurrences_covered=protected_covered,
            provider_attempts=provider_attempts,
            schema_retry=schema_retry,
            finish_reason=finish_reason,
        )

    @staticmethod
    def _paragraph_mining_outcome(
        paragraph: TrustedModelVisibleParagraph,
        windows: tuple[MinerWindowOutcome, ...],
    ) -> ParagraphMiningOutcome:
        accepted_claim_ids = tuple(
            claim_id for window in windows for claim_id in window.accepted_claim_ids
        )
        reason_codes = tuple(
            dict.fromkeys(code for window in windows for code in window.failure_codes)
        )
        protected_total = sum(window.protected_occurrences_total for window in windows)
        protected_covered = sum(
            window.protected_occurrences_covered for window in windows
        )
        if protected_covered < protected_total and not reason_codes:
            reason_codes = ("miner_coverage_incomplete",)
        complete_count = sum(window.status == "complete" for window in windows)
        if not reason_codes and complete_count == len(windows):
            status: Literal["complete", "partial", "needs_human"] = "complete"
        elif accepted_claim_ids:
            status = "partial"
        else:
            status = "needs_human"
        return ParagraphMiningOutcome(
            paragraph_id=paragraph.paragraph_id,
            source_file=paragraph.source.file,
            source_line_start=paragraph.source.line_start,
            source_line_end=paragraph.source.line_end,
            status=status,
            window_count=len(windows),
            completed_window_count=complete_count,
            accepted_claim_ids=accepted_claim_ids,
            reason_codes=reason_codes,
            protected_occurrences_total=protected_total,
            protected_occurrences_covered=protected_covered,
            windows=windows,
        )

    def _mine(
        self,
        paragraphs: tuple[TrustedModelVisibleParagraph, ...],
        context: _Context,
    ) -> _MiningResult:
        claims: list[_Claim] = []
        paragraph_outcomes: list[ParagraphMiningOutcome] = []
        plans = tuple(
            (
                paragraph,
                plan_miner_windows(paragraph, max_window_chars=self.max_window_chars),
            )
            for paragraph in paragraphs
        )
        context.budget.miner_windows_planned = sum(
            len(plan.windows) + len(plan.oversized_fragments) for _, plan in plans
        )
        dispatch_number = 0

        for paragraph, plan in plans:
            window_outcomes: list[MinerWindowOutcome] = []
            planned_items: list[MinerWindow | OversizedMinerFragment] = [
                *plan.windows,
                *plan.oversized_fragments,
            ]
            planned_items.sort(key=lambda item: (item.start, item.end))

            for item in planned_items:
                item_text = paragraph.text[item.start : item.end]
                protected_total = miner_protected_occurrence_count(item_text)
                if isinstance(item, OversizedMinerFragment):
                    outcome = self._failed_window_outcome(
                        item,
                        status="needs_human",
                        code="miner_window_fragment_oversized",
                        protected_total=protected_total,
                    )
                    window_outcomes.append(outcome)
                    context.event(
                        TraceState.FAILED,
                        AgentName.CONTROLLER,
                        "mine_window",
                        code="miner_window_fragment_oversized",
                        paragraph_id=paragraph.paragraph_id,
                        window_id=item.window_id,
                        provider_attempts=0,
                        schema_retry=False,
                    )
                    continue

                window_budget_blocked = (
                    context.budget.miner_windows_dispatched
                    >= context.budget.miner_window_limit
                    or context.budget.miner_provider_attempts + 2
                    > context.budget.miner_provider_attempt_limit
                    or context.miner_global_budget_exhausted
                    or len(claims) >= context.budget.max_claims
                )
                if window_budget_blocked:
                    outcome = self._failed_window_outcome(
                        item,
                        status="budget_exhausted",
                        code="budget_exhausted",
                        protected_total=protected_total,
                    )
                    window_outcomes.append(outcome)
                    context.event(
                        TraceState.FAILED,
                        AgentName.CONTROLLER,
                        "mine_window",
                        code="budget_exhausted",
                        paragraph_id=paragraph.paragraph_id,
                        window_id=item.window_id,
                        provider_attempts=0,
                        schema_retry=False,
                    )
                    continue

                dispatch_number += 1
                dispatch_id = f"miner_{dispatch_number:04d}"
                context.budget.miner_windows_dispatched += 1
                client = getattr(self.miner, "client", None)
                before_events = _model_telemetry_events(client)
                before_count = len(before_events) if before_events is not None else None
                context.event(
                    TraceState.DISPATCHED,
                    AgentName.MINER,
                    "mine",
                    miner_dispatch_id=dispatch_id,
                    paragraph_id=paragraph.paragraph_id,
                    window_id=item.window_id,
                )
                try:
                    validated = self.miner.mine_with_validation(
                        MinerInput(
                            text=item.text,
                            file=item.source_file,
                            line_start=item.source_line_start,
                            line_end=item.source_line_end,
                            heading_path=item.heading_path,
                            citation_urls=item.citation_urls,
                        ),
                        trusted_window_context=True,
                    )
                    localized = self._localize_window_claims(
                        paragraph, item, validated.output.claims
                    )
                except ModelCallBudgetExceeded:
                    attempts, retried, finish_reason = _miner_attempt_details(
                        client, before_count, assume_attempt=False
                    )
                    context.budget.miner_provider_attempts += attempts
                    context.miner_global_budget_exhausted = True
                    outcome = self._failed_window_outcome(
                        item,
                        status="budget_exhausted",
                        code="budget_exhausted",
                        protected_total=protected_total,
                        provider_attempts=attempts,
                        schema_retry=retried,
                        finish_reason=finish_reason,
                    )
                    window_outcomes.append(outcome)
                    context.event(
                        TraceState.FAILED,
                        AgentName.MINER,
                        "mine",
                        code="budget_exhausted",
                        miner_dispatch_id=dispatch_id,
                        paragraph_id=paragraph.paragraph_id,
                        window_id=item.window_id,
                        provider_attempts=attempts,
                        schema_retry=retried,
                        finish_reason=finish_reason,
                    )
                    continue
                except Exception as error:
                    attempts, retried, finish_reason = _miner_attempt_details(
                        client, before_count, assume_attempt=True
                    )
                    context.budget.miner_provider_attempts += attempts
                    code = _miner_failure_code(error)
                    protected_total = int(
                        getattr(
                            error,
                            "protected_occurrences_total",
                            protected_total,
                        )
                    )
                    withheld_count = int(
                        getattr(error, "withheld_exact_draft_count", 0)
                    )
                    outcome = self._failed_window_outcome(
                        item,
                        status=(
                            "needs_human"
                            if code == "miner_scope_missing_protected_token"
                            else "agent_error"
                        ),
                        code=cast(MinerOutcomeFailureCode, code),
                        protected_total=protected_total,
                        protected_covered=0,
                        withheld_exact_draft_count=withheld_count,
                        provider_attempts=attempts,
                        schema_retry=retried,
                        finish_reason=finish_reason,
                    )
                    window_outcomes.append(outcome)
                    context.event(
                        TraceState.FAILED,
                        AgentName.MINER,
                        "mine",
                        code=code,
                        miner_dispatch_id=dispatch_id,
                        paragraph_id=paragraph.paragraph_id,
                        window_id=item.window_id,
                        provider_attempts=attempts,
                        schema_retry=retried,
                        finish_reason=finish_reason,
                    )
                    continue

                attempts, retried, finish_reason = _miner_attempt_details(
                    client, before_count, assume_attempt=True
                )
                context.budget.miner_provider_attempts += attempts
                context.event(
                    TraceState.COMPLETED,
                    AgentName.MINER,
                    "mine",
                    miner_dispatch_id=dispatch_id,
                    paragraph_id=paragraph.paragraph_id,
                    window_id=item.window_id,
                    provider_attempts=attempts,
                    schema_retry=retried,
                    finish_reason=finish_reason,
                )

                if (
                    client is not None
                    and not localized
                    and item.inline_code_identifier_count > 0
                ):
                    outcome = self._failed_window_outcome(
                        item,
                        status="needs_human",
                        code="miner_coverage_empty_identifier_window",
                        protected_total=(
                            validated.validation.protected_occurrences_total
                        ),
                        protected_covered=(
                            validated.validation.protected_occurrences_covered
                        ),
                        provider_attempts=attempts,
                        schema_retry=retried,
                        finish_reason=finish_reason,
                    )
                    window_outcomes.append(outcome)
                    context.event(
                        TraceState.FAILED,
                        AgentName.CONTROLLER,
                        "mine_window",
                        code="miner_coverage_empty_identifier_window",
                        paragraph_id=paragraph.paragraph_id,
                        window_id=item.window_id,
                        provider_attempts=0,
                        schema_retry=False,
                    )
                    continue

                if len(claims) + len(localized) > context.budget.max_claims:
                    outcome = self._failed_window_outcome(
                        item,
                        status="budget_exhausted",
                        code="budget_exhausted",
                        protected_total=(
                            validated.validation.protected_occurrences_total
                        ),
                        withheld_exact_draft_count=len(localized),
                        provider_attempts=attempts,
                        schema_retry=retried,
                        finish_reason=finish_reason,
                    )
                    window_outcomes.append(outcome)
                    context.event(
                        TraceState.FAILED,
                        AgentName.CONTROLLER,
                        "mine_window",
                        code="budget_exhausted",
                        paragraph_id=paragraph.paragraph_id,
                        window_id=item.window_id,
                        provider_attempts=0,
                        schema_retry=False,
                    )
                    continue

                accepted_ids: list[str] = []
                for value in localized:
                    claim_id = f"c_{len(claims) + 1:04d}"
                    claims.append(
                        _Claim(value.model_copy(update={"claim_id": claim_id}))
                    )
                    accepted_ids.append(claim_id)
                window_outcomes.append(
                    MinerWindowOutcome(
                        window_id=item.window_id,
                        paragraph_id=item.paragraph_id,
                        source_file=item.source_file,
                        source_line_start=item.source_line_start,
                        source_line_end=item.source_line_end,
                        citation_ids=item.citation_ids,
                        status="complete",
                        accepted_claim_ids=tuple(accepted_ids),
                        protected_occurrences_total=(
                            validated.validation.protected_occurrences_total
                        ),
                        protected_occurrences_covered=(
                            validated.validation.protected_occurrences_covered
                        ),
                        provider_attempts=attempts,
                        schema_retry=retried,
                        finish_reason=finish_reason,
                    )
                )

            paragraph_outcome = self._paragraph_mining_outcome(
                paragraph, tuple(window_outcomes)
            )
            paragraph_outcomes.append(paragraph_outcome)
            context.event(
                (
                    TraceState.COMPLETED
                    if paragraph_outcome.status == "complete"
                    else TraceState.FAILED
                ),
                AgentName.CONTROLLER,
                "mine_paragraph",
                code=(
                    cast(FailureCode, paragraph_outcome.reason_codes[0])
                    if paragraph_outcome.reason_codes
                    else None
                ),
                paragraph_id=paragraph.paragraph_id,
            )

        return _MiningResult(
            claims=tuple(claims),
            paragraphs=tuple(paragraph_outcomes),
        )

    def _plan(
        self,
        stage: PlanStage,
        claims: tuple[_Claim, ...],
        discover: bool,
        context: _Context,
    ) -> ExecutionPlan:
        context.event(TraceState.DISPATCHED, AgentName.COORDINATOR, stage.value)
        slots = context.budget.discovery_slots()
        try:
            payload: list[dict[str, Any]]
            if stage is PlanStage.INITIAL:
                payload = [
                    {
                        "claim_id": item.claim.claim_id,
                        "text": item.claim.text,
                        "checkability": item.claim.checkability.value,
                        "has_citation": bool(item.claim.citation_urls),
                    }
                    for item in claims
                ]
            else:
                payload = [
                    {
                        "claim_id": item.claim.claim_id,
                        "relation": item.verdict.relation.value
                        if item.verdict
                        else None,
                        "confidence": item.verdict.confidence if item.verdict else None,
                        "high_risk": bool(
                            item.verdict
                            and is_high_risk_claim(
                                item.claim, item.verdict, item.signals
                            )
                        ),
                        "used_scout": item.used_scout,
                        "claim_completed": (item.status is ClaimRunStatus.COMPLETED),
                    }
                    for item in claims
                ]
            plan = self.coordinator.plan(
                stage,
                payload,
                discover_enabled=discover,
                discovery_limit=slots,
            )
            return validate_execution_plan(
                plan,
                claims,
                stage=stage,
                discover=discover,
                discovery_slots=slots,
            )
        except Exception:
            context.event(
                TraceState.FAILED,
                AgentName.COORDINATOR,
                stage.value,
                code="coordinator_plan_rejected",
            )
            context.event(TraceState.FALLBACK, AgentName.CONTROLLER, stage.value)
            return _fallback_plan(stage, claims, discover=discover, limit=slots)
        finally:
            context.event(TraceState.PLANNED, AgentName.COORDINATOR, stage.value)

    def _attach_local_reference_evidence(
        self,
        claims: tuple[_Claim, ...],
        context: _Context,
    ) -> None:
        if self.reference_index is None:
            return
        for item in claims:
            try:
                results = self.reference_index.lookup(item.claim, top_k=5)
            except Exception:
                # Reference parsing and SHA validation happen during preflight/index
                # construction. A claim-local retrieval failure must remain isolated.
                continue
            for index, result in enumerate(results, start=1):
                source = getattr(result, "source", None)
                evidence = getattr(result, "evidence", ())
                reference_id = getattr(result, "reference_id", None)
                reference_sha256 = getattr(result, "reference_sha256", None)
                if (
                    not isinstance(source, SourceMetadata)
                    or not isinstance(evidence, tuple)
                    or not all(
                        isinstance(value, RetrievedEvidence) for value in evidence
                    )
                    or not isinstance(reference_id, str)
                    or not isinstance(reference_sha256, str)
                ):
                    continue
                option = EvidenceOption(
                    option_id=(
                        f"{item.claim.claim_id}:reference:"
                        f"{source.source_id}:{index}"
                    ),
                    origin=EvidenceOrigin.REFERENCE,
                    source=source,
                    evidence=evidence,
                    reference_id=reference_id,
                    reference_sha256=reference_sha256,
                )
                item.evidence_options.append(option)
                context.sources[source.source_id] = source

    def _initial(
        self,
        item: _Claim,
        action: PlanAction,
        discover: bool,
        context: _Context,
    ) -> None:
        if item.status is ClaimRunStatus.BUDGET_EXHAUSTED:
            item.verdict = _fallback(
                item.claim, Relation.NOT_IN_SOURCE, "Claim budget exhausted."
            )
        elif action is PlanAction.SKIP_NOT_CHECKABLE:
            item.verdict = _fallback(
                item.claim,
                Relation.NOT_CHECKABLE,
                "Claim is not checkable.",
                confidence=0.95,
            )
        elif item.evidence_options:
            if item.claim.citation_urls:
                resolver = DiscoveryTools(None, self.fetcher)
                result = resolver.resolve_urls(
                    item.claim,
                    item.claim.citation_urls[
                        : max(
                            context.budget.max_fetches - context.budget.fetches,
                            0,
                        )
                    ],
                )
                self._resolve(
                    item,
                    result,
                    context,
                    origin=EvidenceOrigin.CITATION,
                )
            else:
                self._route_evidence(item, context)
        elif action is PlanAction.REQUEST_HUMAN:
            item.verdict = _fallback(
                item.claim, Relation.NOT_IN_SOURCE, "Human review is required."
            )
            self._fail(
                item,
                ClaimRunStatus.NEEDS_HUMAN,
                "human_review_requested",
            )
        elif action is PlanAction.DISCOVER:
            if not discover:
                item.verdict = _fallback(
                    item.claim, Relation.NOT_IN_SOURCE, "Discovery is unavailable."
                )
                self._fail(item, ClaimRunStatus.NEEDS_HUMAN, "discovery_unavailable")
            else:
                self._discover(item, context)
        else:
            resolver = DiscoveryTools(None, self.fetcher)
            result = resolver.resolve_urls(
                item.claim,
                item.claim.citation_urls[
                    : max(context.budget.max_fetches - context.budget.fetches, 0)
                ],
            )
            self._resolve(
                item,
                result,
                context,
                origin=EvidenceOrigin.CITATION,
            )

    def _review(self, item: _Claim, action: PlanAction, context: _Context) -> None:
        if action is PlanAction.REQUEST_HUMAN:
            if item.status is ClaimRunStatus.COMPLETED:
                self._fail(
                    item,
                    ClaimRunStatus.NEEDS_HUMAN,
                    "human_review_requested",
                )
        elif action is PlanAction.COUNTER_SEARCH:
            self._discover(item, context)
        elif action is PlanAction.CHALLENGE and item.verdict:
            self._challenge(item, context)

    def _pre_review_counter_search(
        self,
        claims: tuple[_Claim, ...],
        discover: bool,
        context: _Context,
    ) -> None:
        """Finish bounded unresolved discovery before final Coordinator review."""

        if not discover:
            return
        unresolved = {
            Relation.NOT_IN_SOURCE,
            Relation.SOURCE_UNAVAILABLE,
            Relation.PARTIALLY_ENTAILED,
        }
        for item in claims:
            if context.budget.discovery_slots() <= 0:
                break
            retryable_source_gap = item.error in {
                "evidence_not_found",
                "source_error",
            }
            if (
                not item.used_scout
                and item.verdict is not None
                and item.verdict.relation in unresolved
                and (
                    item.status is ClaimRunStatus.COMPLETED
                    or retryable_source_gap
                )
            ):
                self._discover(item, context)

    def _discover(self, item: _Claim, context: _Context) -> None:
        if item.used_scout or context.budget.discovery_slots() <= 0:
            self._fail(item, ClaimRunStatus.BUDGET_EXHAUSTED, "budget_exhausted")
            return
        item.used_scout = True
        context.event(
            TraceState.DISPATCHED,
            AgentName.SCOUT,
            "discover",
            item.claim.claim_id,
        )
        try:
            result = context.tools.discover(item.claim, self.scout.plan(item.claim))
        except ModelCallBudgetExceeded:
            self._budget_failure(item, context, AgentName.SCOUT, "discover")
            return
        except Exception:
            item.verdict = _fallback(
                item.claim, Relation.NOT_IN_SOURCE, "Discovery failed safely."
            )
            self._fail(item, ClaimRunStatus.AGENT_ERROR, "scout_error")
            context.event(
                TraceState.FAILED,
                AgentName.SCOUT,
                "discover",
                item.claim.claim_id,
                "scout_error",
            )
            return
        context.event(
            TraceState.COMPLETED,
            AgentName.SCOUT,
            "discover",
            item.claim.claim_id,
        )
        self._resolve(
            item,
            result,
            context,
            origin=EvidenceOrigin.TAVILY,
        )

    def _resolve(
        self,
        item: _Claim,
        result: DiscoveryResult,
        context: _Context,
        *,
        origin: EvidenceOrigin,
    ) -> None:
        context.budget.searches += result.query_count
        context.budget.fetches += result.fetch_count
        context.sources.update({source.source_id: source for source in result.sources})
        if result.status == "discovery_unavailable":
            if item.evidence_options:
                self._route_evidence(item, context)
                return
            item.verdict = _fallback(
                item.claim, Relation.NOT_IN_SOURCE, "Discovery is unavailable."
            )
            self._fail(item, ClaimRunStatus.NEEDS_HUMAN, "discovery_unavailable")
        elif result.status == "source_error" or result.source is None:
            if item.evidence_options:
                self._route_evidence(item, context)
                return
            item.verdict = _fallback(
                item.claim, Relation.SOURCE_UNAVAILABLE, "Source is unavailable."
            )
            self._fail(item, ClaimRunStatus.SOURCE_ERROR, "source_error")
        elif not result.evidence and self.enforce_human_evidence_gates:
            if item.evidence_options:
                self._route_evidence(item, context)
                return
            option = EvidenceOption(
                option_id=(
                    f"{item.claim.claim_id}:{origin.value}:"
                    f"{result.source.source_id}:{len(item.evidence_options) + 1}"
                ),
                origin=origin,
                source=result.source,
                evidence=(),
            )
            reason = (
                EvidenceResolutionReason.TAVILY_ONLY
                if origin is EvidenceOrigin.TAVILY
                else EvidenceResolutionReason.AUTOMATIC
            )
            item.resolution = EvidenceResolution(
                claim_id=item.claim.claim_id,
                reason=reason,
                status=EvidenceResolutionStatus.NEEDS_HUMAN,
                options=(option,),
            )
            item.verdict = _fallback(
                item.claim,
                Relation.NOT_IN_SOURCE,
                "No bounded exact evidence is available.",
            )
            self._fail(item, ClaimRunStatus.NEEDS_HUMAN, "evidence_not_found")
        else:
            if result.retrieval_fallback:
                context.event(
                    TraceState.FALLBACK,
                    AgentName.CONTROLLER,
                    "retrieval_empty_full_context_fallback",
                    item.claim.claim_id,
                )
            candidates = tuple(
                candidate
                for candidate in result.candidates
                if candidate.evidence
            )
            if not candidates:
                candidates = (
                    DiscoveryCandidate(result.source, result.evidence),
                )
            for candidate in candidates:
                item.evidence_options.append(
                    EvidenceOption(
                        option_id=(
                            f"{item.claim.claim_id}:{origin.value}:"
                            f"{candidate.source.source_id}:"
                            f"{len(item.evidence_options) + 1}"
                        ),
                        origin=origin,
                        source=candidate.source,
                        evidence=candidate.evidence,
                    )
                )
            self._route_evidence(item, context)

    @staticmethod
    def _option_score(option: EvidenceOption) -> float:
        return option.evidence[0].score if option.evidence else float("-inf")

    @staticmethod
    def _option_signature(
        claim: AtomicClaim, option: EvidenceOption
    ) -> tuple[str, str] | None:
        if not option.evidence:
            return None
        return _routing_replacement_signature(
            claim.text,
            option.evidence[0].text,
        )

    @staticmethod
    def _split_conflicting_option(
        claim: AtomicClaim,
        option: EvidenceOption,
    ) -> tuple[EvidenceOption, ...]:
        groups: dict[
            tuple[str, str] | None,
            list[RetrievedEvidence],
        ] = {}
        for evidence in option.evidence:
            signals = deterministic_signals(claim.text, evidence.text)
            if any(signal.code == "entity_mismatch" for signal in signals):
                continue
            signature = _routing_replacement_signature(
                claim.text,
                evidence.text,
            )
            if signature is not None:
                groups.setdefault(signature, []).append(evidence)
        if len(groups) <= 1:
            return (option,)
        return tuple(
            EvidenceOption(
                option_id=f"{option.option_id}:evidence:{index}",
                origin=option.origin,
                source=option.source,
                evidence=tuple(evidence),
                reference_id=option.reference_id,
                reference_sha256=option.reference_sha256,
            )
            for index, evidence in enumerate(groups.values(), start=1)
        )

    @staticmethod
    def _bound_resolution_options(
        options: tuple[EvidenceOption, ...],
    ) -> tuple[tuple[EvidenceOption, ...], bool]:
        all_evidence = tuple(
            evidence for option in options for evidence in option.evidence
        )
        if (
            len(options) <= EVIDENCE_RESOLUTION_MAX_OPTIONS
            and sum(len(item.text) for item in all_evidence)
            <= EVIDENCE_RESOLUTION_EXACT_MAX_CHARS
            and sum(len(item.text.encode("utf-8")) for item in all_evidence)
            <= EVIDENCE_RESOLUTION_EXACT_MAX_BYTES
            and sum(len(item.chunk.text) for item in all_evidence)
            <= EVIDENCE_RESOLUTION_CONTEXT_MAX_CHARS
            and sum(
                len(item.chunk.text.encode("utf-8"))
                for item in all_evidence
            )
            <= EVIDENCE_RESOLUTION_CONTEXT_MAX_BYTES
        ):
            return options, False
        ordered = sorted(
            options,
            key=lambda option: (
                ProductAuditPipeline._option_score(option),
                option.origin is EvidenceOrigin.CITATION,
                option.option_id,
            ),
            reverse=True,
        )
        selected: list[EvidenceOption] = []
        exact_chars = exact_bytes = context_chars = context_bytes = 0
        for option in ordered:
            evidence = option.evidence
            proposed_exact_chars = exact_chars + sum(
                len(item.text) for item in evidence
            )
            proposed_exact_bytes = exact_bytes + sum(
                len(item.text.encode("utf-8")) for item in evidence
            )
            proposed_context_chars = context_chars + sum(
                len(item.chunk.text) for item in evidence
            )
            proposed_context_bytes = context_bytes + sum(
                len(item.chunk.text.encode("utf-8")) for item in evidence
            )
            if (
                len(selected) >= EVIDENCE_RESOLUTION_MAX_OPTIONS
                or proposed_exact_chars > EVIDENCE_RESOLUTION_EXACT_MAX_CHARS
                or proposed_exact_bytes > EVIDENCE_RESOLUTION_EXACT_MAX_BYTES
                or proposed_context_chars
                > EVIDENCE_RESOLUTION_CONTEXT_MAX_CHARS
                or proposed_context_bytes
                > EVIDENCE_RESOLUTION_CONTEXT_MAX_BYTES
            ):
                continue
            selected.append(option)
            exact_chars = proposed_exact_chars
            exact_bytes = proposed_exact_bytes
            context_chars = proposed_context_chars
            context_bytes = proposed_context_bytes
        return tuple(selected), len(selected) != len(options)

    def _route_evidence(self, item: _Claim, context: _Context) -> None:
        deduplicated = tuple(
            {
                option.option_id: option for option in item.evidence_options
            }.values()
        )
        all_options = tuple(
            split
            for option in deduplicated
            for split in self._split_conflicting_option(item.claim, option)
        )
        if not all_options:
            item.verdict = _fallback(
                item.claim,
                Relation.NOT_IN_SOURCE,
                "No bounded evidence is available.",
            )
            self._fail(
                item,
                ClaimRunStatus.NEEDS_HUMAN,
                "human_review_requested",
            )
            return
        signatures = {
            signature
            for option in all_options
            if (
                signature := self._option_signature(item.claim, option)
            )
            is not None
        }
        ambiguous_multi_source = (
            len(all_options) > 1
            and bool(_routing_scalar_tokens(item.claim.text))
            and any(
                self._option_signature(item.claim, option) is None
                for option in all_options
            )
        )
        reason = (
            EvidenceResolutionReason.CONFLICT
            if len(signatures) > 1 or ambiguous_multi_source
            else EvidenceResolutionReason.AUTOMATIC
        )
        non_tavily = tuple(
            option
            for option in all_options
            if option.origin is not EvidenceOrigin.TAVILY
        )
        tavily = tuple(
            option
            for option in all_options
            if option.origin is EvidenceOrigin.TAVILY
        )
        corroborated_tavily = False
        if tavily and non_tavily:
            non_tavily_signatures = {
                signature
                for option in non_tavily
                if (
                    signature := self._option_signature(item.claim, option)
                )
                is not None
            }
            corroborated_tavily = any(
                self._option_signature(item.claim, option)
                in non_tavily_signatures
                for option in tavily
            )
        if (
            tavily
            and reason is not EvidenceResolutionReason.CONFLICT
            and (not non_tavily or not corroborated_tavily)
        ):
            reason = EvidenceResolutionReason.TAVILY_ONLY
        options, options_overflow = self._bound_resolution_options(all_options)
        recorded_options = options
        non_tavily = tuple(
            option
            for option in options
            if option.origin is not EvidenceOrigin.TAVILY
        )
        requires_human = (
            self.enforce_human_evidence_gates
            and reason
            in {
                EvidenceResolutionReason.CONFLICT,
                EvidenceResolutionReason.TAVILY_ONLY,
            }
        ) or (
            options_overflow
            and reason is EvidenceResolutionReason.CONFLICT
        )
        selected: EvidenceOption | None = None
        human_selected = False
        if (
            requires_human
            and self.evidence_resolver is not None
            and not (
                options_overflow
                and reason is EvidenceResolutionReason.CONFLICT
            )
        ):
            selected_id = self.evidence_resolver.choose_evidence(
                item.claim,
                reason,
                options,
            )
            selected = next(
                (
                    option
                    for option in options
                    if option.option_id == selected_id
                ),
                None,
            )
            if (
                reason is EvidenceResolutionReason.TAVILY_ONLY
                and selected is not None
                and selected.origin is not EvidenceOrigin.TAVILY
            ):
                selected = None
            human_selected = selected is not None
        elif not requires_human:
            candidates = non_tavily or options
            selected = max(
                candidates,
                key=lambda option: (
                    self._option_score(option),
                    option.origin is EvidenceOrigin.CITATION,
                    option.option_id,
                ),
            )
        if selected is None:
            item.resolution = EvidenceResolution(
                claim_id=item.claim.claim_id,
                reason=reason,
                status=EvidenceResolutionStatus.NEEDS_HUMAN,
                options=recorded_options,
            )
            item.verdict = _fallback(
                item.claim,
                Relation.NOT_IN_SOURCE,
                "Bounded evidence requires human resolution.",
            )
            self._fail(
                item,
                ClaimRunStatus.NEEDS_HUMAN,
                "human_review_requested",
            )
            return
        status = EvidenceResolutionStatus.SELECTED
        if human_selected and reason is EvidenceResolutionReason.TAVILY_ONLY:
            status = EvidenceResolutionStatus.HUMAN_CORROBORATED
        elif human_selected and reason is EvidenceResolutionReason.CONFLICT:
            status = EvidenceResolutionStatus.HUMAN_RESOLVED_CONFLICT
        item.resolution = EvidenceResolution(
            claim_id=item.claim.claim_id,
            reason=reason,
            status=status,
            options=recorded_options,
            selected_option_id=selected.option_id,
            selected_source_id=selected.source.source_id,
            selected_origin=selected.origin,
            reference_id=selected.reference_id,
            reference_sha256=selected.reference_sha256,
        )
        self._judge(
            item,
            DiscoveryResult(
                status="ok" if selected.evidence else "evidence_not_found",
                source=selected.source,
                evidence=selected.evidence,
                sources=(selected.source,),
            ),
            context,
        )

    def _judge(self, item: _Claim, result: DiscoveryResult, context: _Context) -> None:
        if item.judge_calls >= 2:
            self._fail(item, ClaimRunStatus.BUDGET_EXHAUSTED, "budget_exhausted")
            return
        item.judge_calls += 1
        context.budget.judges += 1
        item.evidence = result.evidence
        best = result.evidence[0].text if result.evidence else ""
        item.signals = deterministic_signals(
            item.claim.text,
            best,
            bounded_evidence_context=bounded_evidence_text(result.evidence),
        )
        context.event(
            TraceState.DISPATCHED, AgentName.JUDGE, "judge", item.claim.claim_id
        )
        try:
            assert result.source is not None
            item.verdict = self.judge.judge(
                JudgeInput(
                    claim=item.claim,
                    evidence=result.evidence,
                    source=result.source,
                    signals=item.signals,
                )
            ).verdict
            item.status = ClaimRunStatus.COMPLETED
            item.error = (
                "evidence_not_found" if result.status == "evidence_not_found" else None
            )
            context.event(
                TraceState.COMPLETED,
                AgentName.JUDGE,
                "judge",
                item.claim.claim_id,
            )
        except ModelCallBudgetExceeded:
            self._budget_failure(item, context, AgentName.JUDGE, "judge")
        except Exception:
            item.verdict = _fallback(
                item.claim, Relation.NOT_IN_SOURCE, "Judge failed safely."
            )
            self._fail(item, ClaimRunStatus.AGENT_ERROR, "judge_error")
            context.event(
                TraceState.FAILED,
                AgentName.JUDGE,
                "judge",
                item.claim.claim_id,
                "judge_error",
            )

    def _challenge(self, item: _Claim, context: _Context) -> None:
        if item.challenger_calls:
            self._fail(item, ClaimRunStatus.BUDGET_EXHAUSTED, "budget_exhausted")
            return
        item.challenger_calls = 1
        context.budget.challengers += 1
        context.event(
            TraceState.DISPATCHED,
            AgentName.CHALLENGER,
            "challenge",
            item.claim.claim_id,
        )
        try:
            assert item.verdict is not None
            result = self.challenger.challenge(
                item.claim, item.verdict, item.signals, item.evidence
            )
            item.challenger_action = result.action
            item.verdict = result.verdict
            if result.action is ChallengeAction.ABSTAIN:
                self._fail(
                    item,
                    ClaimRunStatus.NEEDS_HUMAN,
                    "human_review_requested",
                )
            context.event(
                TraceState.COMPLETED,
                AgentName.CHALLENGER,
                result.action.value,
                item.claim.claim_id,
            )
        except ModelCallBudgetExceeded:
            self._budget_failure(item, context, AgentName.CHALLENGER, "challenge")
        except Exception:
            self._fail(item, ClaimRunStatus.AGENT_ERROR, "challenger_error")
            context.event(
                TraceState.FAILED,
                AgentName.CHALLENGER,
                "challenge",
                item.claim.claim_id,
                "challenger_error",
            )

    @staticmethod
    def _claim(claims: tuple[_Claim, ...], claim_id: str) -> _Claim:
        return next(item for item in claims if item.claim.claim_id == claim_id)

    @staticmethod
    def _fail(item: _Claim, status: ClaimRunStatus, code: FailureCode) -> None:
        item.status, item.error = status, code

    @classmethod
    def _budget_failure(
        cls, item: _Claim, context: _Context, agent: AgentName, action: str
    ) -> None:
        if item.verdict is None:
            item.verdict = _fallback(
                item.claim, Relation.NOT_IN_SOURCE, "Model budget exhausted."
            )
        cls._fail(item, ClaimRunStatus.BUDGET_EXHAUSTED, "budget_exhausted")
        context.event(
            TraceState.FAILED, agent, action, item.claim.claim_id, "budget_exhausted"
        )

    def _persist(self, audit: AuditArtifact, product: ProductRunArtifact) -> Path:
        try:
            manager = ArtifactManager(self.root)
            paths = manager.create_run(audit.run)
            manager.write_audit(paths, audit)
            _atomic_write(
                paths.run_dir / "product_run.json",
                product.model_dump_json(indent=2) + "\n",
            )
            _atomic_write(paths.run_dir / "audit.md", render_audit_markdown(audit))
            return paths.audit_json
        except Exception:
            raise ProductPipelineError("artifact write failed") from None

    @staticmethod
    def _terminal(audit: AuditArtifact, product: ProductRunArtifact) -> str:
        agents = [
            AgentName.MINER,
            AgentName.COORDINATOR,
            AgentName.SCOUT,
            AgentName.JUDGE,
            AgentName.CHALLENGER,
        ]
        counts = {
            agent: sum(
                event.agent is agent and event.state is TraceState.DISPATCHED
                for event in product.trace
            )
            for agent in agents
        }
        skipped = []
        if not counts[AgentName.SCOUT]:
            skipped.append(
                "Scout=discovery_disabled"
                if not product.discover_enabled
                else "Scout=not_needed"
            )
        if not counts[AgentName.CHALLENGER]:
            skipped.append("Challenger=no_high_risk_claim")
        if not product.live_agents_enabled:
            skipped.append("live_model=missing_provider_configuration")
        coordinator_remaining = (
            product.budget["coordinator_call_limit"]
            - product.budget["coordinator_calls"]
        )
        statuses = ", ".join(
            f"{item.claim_id}={item.status.value}" for item in product.claim_runs
        )
        return "\n".join(
            (
                render_terminal(audit),
                f"Document status: {product.document_status.value}",
                "Claim runs: " + (statuses or "none"),
                "Agents run: "
                + ", ".join(f"{agent.value}={counts[agent]}" for agent in agents),
                "Skipped: " + (", ".join(skipped) if skipped else "none"),
                "Remaining budget: "
                f"coordinator_calls={coordinator_remaining}, "
                f"search_queries={product.budget['search_queries_remaining']}, "
                f"fetches={product.budget['fetches_remaining']}",
            )
        )


def _atomic_write(path: Path, content: str) -> None:
    temporary = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_suggestion_patch(
    markdown_path: Path,
    audit: AuditArtifact,
    output: Path,
    *,
    project_root: Path,
) -> Path:
    root, source = project_root.resolve(), markdown_path.resolve()
    destination = (output if output.is_absolute() else root / output).resolve()
    try:
        relative = source.relative_to(root).as_posix()
        destination.relative_to(root)
    except ValueError:
        raise ProductPipelineError("patch path escaped project root") from None
    if source == destination:
        raise ProductPipelineError("patch cannot replace source")
    before = source.read_text(encoding="utf-8").splitlines(keepends=True)
    claims = {item.claim_id: item for item in audit.claims}
    insertions: dict[int, list[str]] = {}
    for verdict in audit.verdicts:
        claim = claims.get(verdict.claim_id)
        if (
            claim
            and verdict.relation is Relation.CONTRADICTED
            and verdict.evidence_spans
        ):
            evidence = verdict.evidence_spans[0]
            evidence_text = " ".join(evidence.text.split()).replace("--", "- -")
            locator = " ".join(evidence.locator.split()).replace("--", "- -")
            insertions.setdefault(claim.line_start - 1, []).append(
                "<!-- EvidenceTrace candidate correction: review contradicted "
                f"claim {claim.claim_id} using bounded evidence "
                f'({locator}): "{evidence_text}". This suggestion was not '
                "applied. -->\n"
            )
    after: list[str] = []
    for index, line in enumerate(before):
        after.extend(insertions.get(index, ()))
        after.append(line)
    patch = "".join(
        difflib.unified_diff(
            before, after, fromfile=f"a/{relative}", tofile=f"b/{relative}"
        )
    )
    _atomic_write(destination, patch)
    return destination


__all__ = [
    "PRODUCT_MAX_PROVIDER_ATTEMPTS",
    "AgentName",
    "ClaimRunStatus",
    "DocumentRunStatus",
    "EvidenceOption",
    "EvidenceOrigin",
    "EvidenceResolution",
    "EvidenceResolutionReason",
    "EvidenceResolutionStatus",
    "HumanEvidenceResolver",
    "LocalEvidenceLookup",
    "PlanRejectedError",
    "ProductAuditPipeline",
    "ProductPipelineError",
    "ProductRunArtifact",
    "ProductRunResult",
    "TraceState",
    "validate_execution_plan",
    "write_suggestion_patch",
]
