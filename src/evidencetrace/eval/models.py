"""Pydantic contracts for frozen evaluation data and artifacts."""

from __future__ import annotations

import re
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from evidencetrace.eval.errors import LOCAL_VALIDATION_CODES, FailureCode
from evidencetrace.eval.router import AdaptiveRoute, AdaptiveRouteReason
from evidencetrace.model_client import (
    MODEL_REASON_MAX_CHARS,
    ModelCallTelemetry,
    ModelSchemaDiagnostic,
    ModelTelemetrySummary,
    SchemaRecoverySummary,
)
from evidencetrace.models import Relation

AnnotationStatus = Literal[
    "deterministic_gold",
    "human_reviewed",
    "single_human_review",
    "provisional",
]
Split = Literal["dev", "test"]
DeterministicBaselineName = Literal[
    "lexical_rules",
    "lexical_full_source",
    "retrieval_judge_deterministic",
    "miner_judge_deterministic",
]
LiveBaselineName = Literal[
    "single_agent_live",
    "retrieval_judge_live",
    "adaptive_live",
    "miner_judge_live",
]
BaselineName = DeterministicBaselineName | LiveBaselineName
LiveBaselineStatus = Literal["not_requested", "skipped_missing_credentials", "executed"]
LiveFailureStage = Literal[
    "baseline_case",
    "baseline_post_processing",
    "run_post_processing",
    "artifact_build",
    "artifact_write",
]
GuardSignalCode = Literal[
    "numeric_mismatch",
    "date_mismatch",
    "version_mismatch",
    "entity_mismatch",
    "negation_mismatch",
    "qualifier_mismatch",
]
JudgeScopeCode = Literal[
    "claim_id_mismatch",
    "source_id_out_of_scope",
    "evidence_span_out_of_scope",
    "source_availability_mismatch",
    "missing_substantive_evidence",
]
MinerScopeCode = Literal[
    "non_source_span",
    "ambiguous_span",
    "missing_protected_token",
    "invalid_claim_fragment",
]
BenchmarkValidity = Literal[
    "provisional",
    "diagnostic_contaminated",
    "human_reviewed_holdout",
    "single_human_synthetic_holdout",
]
BenchmarkStatus = Literal["provisional", "human_reviewed", "single_human_review"]
SourceLengthStratum = Literal["short", "medium", "long", "unavailable"]

_CASE_ID_RE = re.compile(r"^[a-z][a-z0-9_-]+$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class EvalModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class BenchmarkArtifactModel(EvalModel):
    @model_validator(mode="after")
    def single_human_holdout_is_never_public(self) -> Self:
        if getattr(
            self, "benchmark_validity", None
        ) == "single_human_synthetic_holdout" and getattr(
            self, "public_benchmark_eligible", False
        ):
            raise ValueError("single-human synthetic holdout cannot be public")
        return self


class SourceFixture(EvalModel):
    source_id: str

    url: str
    content: str
    provenance: str
    content_hash: str
    available: bool = True

    @field_validator("source_id", "url", "provenance", "content_hash")
    @classmethod
    def non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("source fixture text fields must not be blank")
        return value

    @field_validator("content_hash")
    @classmethod
    def valid_hash(cls, value: str) -> str:
        if not _SHA256_RE.fullmatch(value):
            raise ValueError("source content_hash must be lowercase SHA-256")
        return value


class EvalCase(EvalModel):
    case_id: str
    document_path: str | None = None
    claim_text: str | None = None
    claim_line: int = Field(default=1, ge=1)
    source_fixture: str
    source_id: str

    source_url: str
    gold_relation: Relation
    gold_evidence_span: str | None = None
    claim_type: str
    mutation_type: str
    split: Split
    provenance: str
    annotation_status: AnnotationStatus
    annotation_notes: str
    source_sha256: str | None = None
    reviewer_record_sha256: str | None = None
    generation_code_sha256: str | None = None
    case_hash: str | None = None
    source_length_stratum: SourceLengthStratum | None = None
    expected_chunk_count: int | None = Field(default=None, ge=0)
    expected_route: AdaptiveRoute | None = None

    @field_validator("case_id")
    @classmethod
    def valid_case_id(cls, value: str) -> str:
        if not _CASE_ID_RE.fullmatch(value):
            raise ValueError("case_id must be a lowercase stable identifier")
        return value

    @field_validator(
        "source_fixture",
        "source_id",
        "source_url",
        "claim_type",
        "mutation_type",
        "provenance",
        "annotation_notes",
    )
    @classmethod
    def non_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("evaluation text fields must not be blank")
        return value

    @field_validator("claim_text", "document_path", "gold_evidence_span")
    @classmethod
    def optional_non_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("optional evaluation text fields must not be blank")
        return value

    @field_validator(
        "source_sha256",
        "reviewer_record_sha256",
        "generation_code_sha256",
        "case_hash",
    )
    @classmethod
    def valid_case_hash(cls, value: str | None) -> str | None:
        if value is not None and not _SHA256_RE.fullmatch(value):
            raise ValueError("evaluation hashes must be lowercase SHA-256")
        return value

    @model_validator(mode="after")
    def has_claim_or_document(self) -> Self:
        if self.claim_text is None and self.document_path is None:
            raise ValueError("each case requires claim_text or document_path")
        if (
            self.gold_relation
            in {
                Relation.ENTAILED,
                Relation.PARTIALLY_ENTAILED,
                Relation.CONTRADICTED,
            }
            and not self.gold_evidence_span
        ):
            raise ValueError("substantive gold relation requires gold_evidence_span")
        if self.annotation_status == "single_human_review" and not all(
            (
                self.source_sha256,
                self.reviewer_record_sha256,
                self.generation_code_sha256,
            )
        ):
            raise ValueError("single-human review requires complete provenance hashes")
        return self


class EvalPrediction(EvalModel):
    case_id: str
    baseline: BaselineName
    predicted_relation: Relation
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = "No reason recorded."
    predicted_evidence_span: str | None = None
    retrieved_texts: tuple[str, ...] = ()
    retrieval_rank: int | None = Field(default=None, ge=1)
    extraction_count: int = Field(default=1, ge=0)
    predicted_line: int | None = Field(default=None, ge=1)
    model_calls: int = Field(default=0, ge=0)
    model_call_latencies_ms: tuple[float, ...] = ()
    model_usage_status: Literal[
        "not_applicable", "reported", "missing", "malformed"
    ] = "not_applicable"
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    schema_failure_count: int = Field(default=0, ge=0)
    transport_failure_count: int = Field(default=0, ge=0)
    model_provider: str | None = None
    model_id: str | None = None
    model_temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    cache_hit: bool = False
    latency_ms: float = Field(ge=0.0)
    pricing_snapshot_id: str | None = None
    cost_usd: float | None = Field(default=None, ge=0.0)
    error_stage: Literal["none", "retrieval", "judge"] = "none"
    selected_route: AdaptiveRoute | None = None
    route_reason_code: AdaptiveRouteReason | None = None
    chunk_count: int | None = Field(default=None, ge=0)
    estimated_context_size: int | None = Field(default=None, ge=0)

    @field_validator("reason")
    @classmethod
    def reason_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prediction reason must not be blank")
        return value

    @model_validator(mode="after")
    def telemetry_is_consistent(self) -> Self:
        tokens = (self.input_tokens, self.output_tokens, self.total_tokens)
        if any(latency < 0.0 for latency in self.model_call_latencies_ms):
            raise ValueError("model call latencies cannot be negative")
        if self.model_calls == 0:
            if (
                self.model_call_latencies_ms
                or self.model_usage_status != "not_applicable"
            ):
                raise ValueError("zero-call prediction cannot contain model telemetry")
        else:
            if len(self.model_call_latencies_ms) != self.model_calls:
                raise ValueError("model latency count must equal model call count")
            if self.model_usage_status == "not_applicable":
                raise ValueError("live model calls require an explicit usage status")
        if self.model_usage_status == "reported" and any(
            value is None for value in tokens
        ):
            raise ValueError("reported model usage requires all token counts")
        if self.model_usage_status != "reported" and any(
            value is not None for value in tokens
        ):
            raise ValueError("unavailable or malformed usage cannot expose totals")
        if self.schema_failure_count + self.transport_failure_count > self.model_calls:
            raise ValueError("failure counts cannot exceed model calls")
        if self.cost_usd is not None and not self.pricing_snapshot_id:
            raise ValueError("cost requires an explicit pricing snapshot")
        route_values = (
            self.selected_route,
            self.route_reason_code,
            self.chunk_count,
            self.estimated_context_size,
        )
        if self.baseline == "adaptive_live":
            if any(value is None for value in route_values):
                raise ValueError(
                    "adaptive prediction requires complete route telemetry"
                )
            if self.model_calls > 2:
                raise ValueError("adaptive prediction cannot exceed two model calls")
        elif any(value is not None for value in route_values):
            raise ValueError("route telemetry is reserved for adaptive predictions")
        return self


class SingleAgentLiveOutput(EvalModel):
    """One-call full-source model output before source-grounding checks."""

    relation: Relation
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(max_length=MODEL_REASON_MAX_CHARS)
    evidence_span: str | None

    @field_validator("reason")
    @classmethod
    def live_reason_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("live baseline reason must not be blank")
        return value

    @model_validator(mode="after")
    def substantive_relation_has_evidence(self) -> Self:
        if (
            self.relation
            in {
                Relation.ENTAILED,
                Relation.PARTIALLY_ENTAILED,
                Relation.CONTRADICTED,
            }
            and not self.evidence_span
        ):
            raise ValueError("substantive live relation requires an evidence span")
        if self.evidence_span is not None and not self.evidence_span.strip():
            raise ValueError("live evidence span must not be blank")
        return self


class EvalRunManifest(BenchmarkArtifactModel):
    dataset_hash: str
    split_hash: str
    model_id: str
    model_provider: str | None = None
    model_temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    model_temperature_actual_effective: None = None
    model_temperature_effective_status: Literal["not_applicable", "unknown"] = (
        "not_applicable"
    )
    model_temperature_semantics: Literal["not_applicable", "requested_only"] = (
        "not_applicable"
    )
    model_thinking_mode: Literal["not_applicable", "provider_default"] = (
        "not_applicable"
    )
    automatic_retry_count: int = Field(default=0, ge=0)
    schema_recovery_policy_version: str | None = None
    schema_retry_limit: int = Field(default=0, ge=0, le=1)
    maximum_expected_live_model_calls: int = Field(default=0, ge=0)
    pricing_snapshot_id: str | None = None
    prompt_version: str
    miner_contract_version: str | None = None
    judge_output_validation_version: str | None = None
    deterministic_signal_policy_version: str | None = None
    retrieval_config: dict[str, str | int | float | bool]
    python_version: str
    git_commit: str
    started_at: str
    finished_at: str
    selected_split: Split | Literal["all"]
    benchmark_status: BenchmarkStatus
    benchmark_validity: BenchmarkValidity = "provisional"
    public_benchmark_eligible: bool = False
    baselines: tuple[BaselineName, ...]
    live_baseline: LiveBaselineStatus = "not_requested"
    live_baseline_statuses: dict[LiveBaselineName, LiveBaselineStatus] = Field(
        default_factory=lambda: {
            "single_agent_live": "not_requested",
            "retrieval_judge_live": "not_requested",
            "adaptive_live": "not_requested",
        }
    )
    skip_reason: str | None = None


class EvalFailureArtifact(BenchmarkArtifactModel):
    """Secret-free telemetry persisted when a live eval fails closed."""

    artifact_type: Literal["live_eval_failure"] = "live_eval_failure"
    artifact_version: Literal[
        "phase3d-live-failure-v1",
        "phase3d1-live-failure-v2",
        "phase3e-live-failure-v3",
        "phase3e2-live-failure-v4",
        "phase3e3-live-failure-v5",
        "phase3g-schema-recovery-failure-v6",
        "phase4a-local-validation-failure-v7",
    ] = "phase3d1-live-failure-v2"
    status: Literal["failed"] = "failed"
    dataset_hash: str
    split_hash: str
    selected_split: Split | Literal["all"]
    git_commit: str
    prompt_version: str
    miner_contract_version: str | None = None
    judge_output_validation_version: str | None = None
    started_at: str
    failed_at: str
    model_provider: str
    model_id: str
    model_temperature_requested: float = Field(ge=0.0, le=2.0)
    model_temperature_actual_effective: None = None
    model_temperature_effective_status: Literal["unknown"] = "unknown"
    model_temperature_semantics: Literal["requested_only"] = "requested_only"
    model_thinking_mode: Literal["provider_default"] = "provider_default"
    same_model_and_thinking_mode: Literal[True] = True
    maximum_expected_live_model_calls: int = Field(ge=0)
    actual_model_calls: int = Field(ge=0)
    actual_model_calls_by_baseline: dict[LiveBaselineName, int]
    hard_limit_enforced: Literal[True] = True
    automatic_retry_count: int = Field(default=0, ge=0)
    schema_recovery: SchemaRecoverySummary = Field(
        default_factory=lambda: SchemaRecoverySummary(retry_limit=0)
    )
    completed_live_predictions: int = Field(ge=0)
    run_failure_count: Literal[1] = 1
    canonical_artifacts_written: Literal[False] = False
    failure_stage: LiveFailureStage = "baseline_case"
    failed_baseline: BaselineName
    failed_case_id: str
    failure_kind: Literal[
        "transport", "schema", "local_validation", "budget", "unexpected"
    ]
    failure_code: FailureCode | None = None
    error_type: str
    schema_diagnostic: ModelSchemaDiagnostic | None = None
    guard_signal_codes: tuple[GuardSignalCode, ...] = ()
    miner_scope_error_code: MinerScopeCode | None = None
    judge_scope_error_code: JudgeScopeCode | None = None
    model_telemetry: ModelTelemetrySummary
    model_call_events: tuple[ModelCallTelemetry, ...]
    calls_with_reported_usage: int = Field(ge=0)
    reported_input_token_subtotal: int = Field(ge=0)
    reported_output_token_subtotal: int = Field(ge=0)
    reported_total_token_subtotal: int = Field(ge=0)
    model_latency_ms_p50: float | None = Field(default=None, ge=0.0)
    model_latency_ms_p95: float | None = Field(default=None, ge=0.0)
    cost_usd: None = None
    cost_explanation: str
    contains_sensitive_payloads: Literal[False] = False

    @model_validator(mode="after")
    def telemetry_matches_hard_budget(self) -> Self:
        if self.actual_model_calls > self.maximum_expected_live_model_calls:
            raise ValueError("actual model calls exceed the hard limit")
        if self.actual_model_calls != self.model_telemetry.calls:
            raise ValueError("call budget and provider telemetry diverged")
        if any(value < 0 for value in self.actual_model_calls_by_baseline.values()):
            raise ValueError("per-baseline model calls cannot be negative")
        if len(self.model_call_events) != self.actual_model_calls:
            raise ValueError("each actual model call requires one safe telemetry event")
        if sum(self.actual_model_calls_by_baseline.values()) != self.actual_model_calls:
            raise ValueError("per-baseline calls must sum to actual model calls")
        if self.schema_diagnostic is not None and self.failure_kind != "schema":
            raise ValueError("schema diagnostics require a schema failure")
        if (
            self.artifact_version == "phase3d-live-failure-v1"
            and self.schema_diagnostic is not None
        ):
            raise ValueError("historical v1 artifacts cannot contain diagnostics")
        if (
            self.artifact_version == "phase3d1-live-failure-v2"
            and self.error_type == "ModelSchemaError"
            and self.schema_diagnostic is None
        ):
            raise ValueError("new ModelSchemaError artifacts require diagnostics")
        if (
            self.artifact_version == "phase3g-schema-recovery-failure-v6"
            and self.schema_recovery.retry_limit != 1
        ):
            raise ValueError("v6 failures require the bounded recovery policy")
        if (
            self.artifact_version == "phase3g-schema-recovery-failure-v6"
            and self.error_type == "ModelSchemaError"
            and self.schema_diagnostic is None
        ):
            raise ValueError("v6 schema failures require safe diagnostics")
        if (
            self.artifact_version != "phase3e-live-failure-v3"
            and self.guard_signal_codes
        ):
            raise ValueError("historical artifacts cannot contain guard signals")
        if self.artifact_version == "phase3e-live-failure-v3":
            if (
                self.error_type != "DeterministicConflictError"
                or self.failure_kind != "local_validation"
                or not self.guard_signal_codes
            ):
                raise ValueError("v3 guard artifacts require a typed local conflict")
            if tuple(sorted(set(self.guard_signal_codes))) != (self.guard_signal_codes):
                raise ValueError("guard signal codes must be sorted and unique")
        if self.error_type == "DeterministicConflictError" and (
            self.artifact_version != "phase3e-live-failure-v3"
            or not self.guard_signal_codes
        ):
            raise ValueError("typed guard conflicts require a v3 artifact")
        if (
            self.artifact_version != "phase3e2-live-failure-v4"
            and self.miner_scope_error_code is not None
        ):
            raise ValueError("historical artifacts cannot contain Miner scope codes")
        if self.artifact_version == "phase3e2-live-failure-v4" and (
            self.error_type != "MinerScopeError"
            or self.failure_kind != "local_validation"
            or self.miner_scope_error_code is None
            or bool(self.guard_signal_codes)
        ):
            raise ValueError("v4 Miner artifacts require one typed local scope code")
        if self.error_type == "MinerScopeError" and (
            self.artifact_version != "phase3e2-live-failure-v4"
            or self.miner_scope_error_code is None
        ):
            raise ValueError("typed Miner scope failures require a v4 artifact")
        if (
            self.artifact_version != "phase3e3-live-failure-v5"
            and self.judge_scope_error_code is not None
        ):
            raise ValueError("historical artifacts cannot contain Judge scope codes")
        if self.artifact_version == "phase3e3-live-failure-v5" and (
            self.error_type != "JudgeScopeError"
            or self.failure_kind != "local_validation"
            or self.judge_scope_error_code is None
            or bool(self.guard_signal_codes)
            or self.miner_scope_error_code is not None
        ):
            raise ValueError("v5 Judge artifacts require one typed local scope code")
        if self.error_type == "JudgeScopeError" and (
            self.artifact_version != "phase3e3-live-failure-v5"
            or self.judge_scope_error_code is None
        ):
            raise ValueError("typed Judge scope failures require a v5 artifact")
        if self.artifact_version == "phase4a-local-validation-failure-v7" and (
            self.error_type != "LocalValidationError"
            or self.failure_kind != "local_validation"
            or self.failure_code not in LOCAL_VALIDATION_CODES
            or self.schema_diagnostic is not None
            or bool(self.guard_signal_codes)
            or self.miner_scope_error_code is not None
            or self.judge_scope_error_code is not None
        ):
            raise ValueError("v7 failures require one typed local validation code")
        if self.error_type == "LocalValidationError" and (
            self.artifact_version != "phase4a-local-validation-failure-v7"
            or self.failure_code not in LOCAL_VALIDATION_CODES
        ):
            raise ValueError("typed local validation failures require a v7 artifact")
        return self


class EvalMetricsArtifact(BenchmarkArtifactModel):
    """Schema for metrics.json top-level sections."""

    dataset: dict[str, Any]
    baselines: dict[str, dict[str, Any]]
    error_analysis: dict[str, dict[str, Any]]
    headline_baseline: BaselineName
    provisional_excluded_from_headline: Literal[True]
    benchmark_status: BenchmarkStatus
    benchmark_validity: BenchmarkValidity = "provisional"
    public_benchmark_eligible: bool = False
    live_execution: dict[str, Any] = Field(default_factory=dict)
    baseline_groups: dict[
        Literal["deterministic", "live"], tuple[BaselineName, ...]
    ] = Field(default_factory=dict)
    live_baseline_statuses: dict[LiveBaselineName, LiveBaselineStatus] = Field(
        default_factory=dict
    )


class EvalResult(EvalPrediction):
    """Schema for one JSONL result including its frozen gold annotation."""

    gold_relation: Relation
    gold_evidence_span: str | None = None
    gold_line: int = Field(default=1, ge=1)
    claim_type: str
    mutation_type: str
    split: Split
    annotation_status: AnnotationStatus
    case_hash: str
    source_id: str
    source_length_stratum: SourceLengthStratum | None = None
    expected_chunk_count: int | None = Field(default=None, ge=0)
    expected_route: AdaptiveRoute | None = None


__all__ = [
    "AnnotationStatus",
    "BaselineName",
    "BenchmarkStatus",
    "BenchmarkValidity",
    "DeterministicBaselineName",
    "EvalCase",
    "EvalFailureArtifact",
    "EvalMetricsArtifact",
    "EvalModel",
    "EvalPrediction",
    "EvalResult",
    "EvalRunManifest",
    "GuardSignalCode",
    "JudgeScopeCode",
    "LiveBaselineName",
    "LiveBaselineStatus",
    "LiveFailureStage",
    "MinerScopeCode",
    "SingleAgentLiveOutput",
    "SourceFixture",
    "SourceLengthStratum",
    "Split",
]
