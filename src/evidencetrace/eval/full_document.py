"""Full-document benchmark contracts, orchestration, and metrics."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from statistics import median
from time import perf_counter
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, Field, field_validator, model_validator

from evidencetrace.agents.judge import JudgeScopeError
from evidencetrace.agents.miner import ClaimMinerAgent, MinerScopeError
from evidencetrace.audit_models import MinerInput
from evidencetrace.checks.deterministic import DeterministicConflictError
from evidencetrace.eval.baselines import build_evidence_chunks, run_baseline
from evidencetrace.eval.errors import LocalValidationError, clear_exception_chain
from evidencetrace.eval.metrics import classification_metrics, safe_divide, token_f1
from evidencetrace.eval.models import EvalCase, EvalModel, SourceFixture
from evidencetrace.eval.router import AdaptiveRouterConfig
from evidencetrace.markdown import parse_markdown
from evidencetrace.model_client import (
    MODEL_REASON_MAX_CHARS,
    FailureKind,
    ModelCallBudgetExceeded,
    ModelCallTelemetry,
    ModelClient,
    ModelResponseError,
    ModelSchemaDiagnostic,
    ModelSchemaError,
    ModelTelemetrySummary,
    ModelTransportError,
    SchemaRecoveryClient,
    SchemaRecoverySummary,
    summarize_model_telemetry,
)
from evidencetrace.models import (
    AtomicClaim,
    AuditArtifact,
    Checkability,
    CorroborationStatus,
    EffectiveConfig,
    EvidenceSpan,
    MarkdownParagraph,
    ParsedDocument,
    Relation,
    RunMetadata,
    SourceMetadata,
    Verdict,
)
from evidencetrace.policy import decide_policy, should_fail
from evidencetrace.sarif import sarif_bytes, write_sarif

FullDocumentBaseline = Literal[
    "single_agent_document_live",
    "miner_adaptive_judge_live",
]
DocumentSplit = Literal["dev", "test", "all"]
DocumentStratum = Literal["short", "medium", "long"]
DocumentRoute = Literal[
    "deterministic_no_citation",
    "deterministic_source_unavailable",
    "deterministic_not_checkable",
    "full_context_single_agent",
    "retrieval_judge",
]
FailureStage = Literal[
    "dataset",
    "provider_contract",
    "single_agent",
    "miner",
    "citation_binding",
    "adaptive_judge",
    "artifact",
]
FailureCategory = Literal[
    "schema",
    "transport",
    "scope",
    "guard",
    "budget",
    "local_validation",
    "artifact",
    "unexpected",
]
FailureCode = Literal[
    "artifact_write_failed",
    "call_budget_exhausted",
    "citation_out_of_scope",
    "claim_count_exceeded",
    "claim_span_ambiguous",
    "claim_span_out_of_scope",
    "dataset_invalid",
    "deterministic_conflict",
    "evidence_span_out_of_scope",
    "judge_scope_violation",
    "local_validation_failed",
    "miner_scope_violation",
    "model_response_invalid",
    "model_schema_invalid",
    "model_transport_failed",
    "provider_output_budget_mismatch",
    "source_binding_missing",
    "token_budget_exhausted",
    "unexpected_failure",
]

MAX_CLAIMS_PER_DOCUMENT = 12
FULL_DOCUMENT_CONTRACT_VERSION = "full-document-semantic-v2"
FULL_DOCUMENT_BENCHMARK_VERSION = "full-document-benchmark-v1"
FULL_DOCUMENT_PROVIDER_CONTRACT_VERSION = "full-document-provider-contract-v2"
FULL_DOCUMENT_MAX_TOKENS = 8192
_SUBSTANTIVE = {
    Relation.ENTAILED,
    Relation.PARTIALLY_ENTAILED,
    Relation.CONTRADICTED,
}
_T = TypeVar("_T", bound=BaseModel)


class FullDocumentMetadata(EvalModel):
    document_id: str
    path: str
    split: Literal["dev", "test"]
    stratum: DocumentStratum
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    gold_claim_count: int = Field(ge=1, le=MAX_CLAIMS_PER_DOCUMENT)

    @field_validator("path")
    @classmethod
    def path_is_relative(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or "\\" in value:
            raise ValueError("document path must be project relative")
        return value


class FullDocumentGoldClaim(EvalModel):
    annotation_status: Literal["author_constructed_deterministic_provisional"]
    document_id: str
    document_path: str
    claim_id: str
    text: str
    offset_start: int = Field(ge=0)
    offset_end: int = Field(ge=0)
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    claim_type: str
    checkability: Checkability
    citation_urls: tuple[str, ...] = ()
    source_ids: tuple[str, ...] = ()
    gold_relation: Relation
    gold_evidence_span: str | None = None
    gold_evidence_source_id: str | None = None
    construction_note: str

    @model_validator(mode="after")
    def validate_gold_contract(self) -> FullDocumentGoldClaim:
        if self.offset_end <= self.offset_start or self.line_end < self.line_start:
            raise ValueError("gold source span is not ordered")
        if len(self.citation_urls) != len(self.source_ids):
            raise ValueError("gold citation and source bindings diverged")
        if self.gold_relation in _SUBSTANTIVE and (
            not self.gold_evidence_span or not self.gold_evidence_source_id
        ):
            raise ValueError("substantive gold requires evidence and source")
        if self.gold_relation not in _SUBSTANTIVE and (
            self.gold_evidence_span is not None
            or self.gold_evidence_source_id is not None
        ):
            raise ValueError("non-substantive gold cannot contain evidence")
        return self


class LiveDocumentSemanticClaim(EvalModel):
    """Model-owned document fields before local identity and location binding."""

    text: str
    claim_type: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    checkability: Checkability
    citation_urls: tuple[str, ...] = Field(max_length=4)
    relation: Relation
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(max_length=MODEL_REASON_MAX_CHARS)
    evidence_span: str | None

    @field_validator("text", "reason")
    @classmethod
    def non_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("document semantic text cannot be blank")
        return value

    @model_validator(mode="after")
    def evidence_is_present_when_required(self) -> LiveDocumentSemanticClaim:
        if self.relation in _SUBSTANTIVE and not self.evidence_span:
            raise ValueError("substantive relation requires evidence")
        if self.evidence_span is not None and not self.evidence_span.strip():
            raise ValueError("evidence span cannot be blank")
        if len(self.citation_urls) != len(set(self.citation_urls)):
            raise ValueError("citation URLs must be unique")
        return self


class SingleAgentDocumentOutput(EvalModel):
    claims: tuple[LiveDocumentSemanticClaim, ...] = Field(
        max_length=MAX_CLAIMS_PER_DOCUMENT
    )


class FullDocumentPrediction(EvalModel):
    prediction_id: str
    document_id: str
    offset_start: int = Field(ge=0)
    offset_end: int = Field(ge=0)
    claim: AtomicClaim
    verdict: Verdict
    selected_route: DocumentRoute | None = None
    route_reason_code: str | None = None

    @model_validator(mode="after")
    def local_fields_are_consistent(self) -> FullDocumentPrediction:
        if self.offset_end <= self.offset_start:
            raise ValueError("prediction source span is not ordered")
        if self.claim.claim_id != self.verdict.claim_id:
            raise ValueError("prediction claim and verdict IDs diverged")
        if (self.selected_route is None) != (self.route_reason_code is None):
            raise ValueError("route telemetry must be complete or absent")
        return self


class DocumentResult(EvalModel):
    document_id: str
    prediction_ids: tuple[str, ...]
    policy_failed: bool
    handoff_attempts: int = Field(ge=0)
    handoff_failures: int = Field(ge=0)

    @model_validator(mode="after")
    def valid_handoff_counts(self) -> DocumentResult:
        if self.handoff_failures > self.handoff_attempts:
            raise ValueError("handoff failures cannot exceed attempts")
        return self


class StageTelemetry(EvalModel):
    calls: int = Field(ge=0)
    usage_status: Literal["not_applicable", "reported", "missing", "malformed"]
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    reasoning_tokens: int | None = Field(default=None, ge=0)
    latency_ms: tuple[float, ...] = ()
    p50_latency_ms: float | None = Field(default=None, ge=0.0)
    p95_latency_ms: float | None = Field(default=None, ge=0.0)
    schema_failures: int = Field(default=0, ge=0)
    transport_failures: int = Field(default=0, ge=0)
    finish_reasons: tuple[str | None, ...] = ()
    max_tokens_requested: tuple[int | None, ...] = ()
    response_content_lengths: tuple[int | None, ...] = ()
    suspected_token_truncations: int = Field(default=0, ge=0)


class FullDocumentRequestProvenance(EvalModel):
    provider_contract_version: Literal["full-document-provider-contract-v2"] = (
        FULL_DOCUMENT_PROVIDER_CONTRACT_VERSION
    )
    provider_id: str
    model_id: str
    prompt_version: str
    response_format: Literal["json_object"] = "json_object"
    output_schema_names: tuple[str, ...]
    max_tokens_requested: int | None = Field(default=None, ge=1, le=8192)
    thinking_mode: Literal["provider_default", "not_available"]
    temperature_requested: float | None = Field(default=None, ge=0.0, le=2.0)
    temperature_effective: None = None
    schema_retry_limit: Literal[1] = 1
    semantic_repair_count: Literal[0] = 0
    cache_status: Literal["disabled_not_available"] = "disabled_not_available"


class FullDocumentRunArtifact(EvalModel):
    artifact_version: Literal["full-document-benchmark-v1"]
    baseline: FullDocumentBaseline
    split: DocumentSplit
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_id: str
    predictions: tuple[FullDocumentPrediction, ...]
    documents: tuple[DocumentResult, ...]
    audit: AuditArtifact
    metrics: dict[str, Any]
    stage_telemetry: dict[str, StageTelemetry]
    request_provenance: FullDocumentRequestProvenance
    schema_recovery: SchemaRecoverySummary
    schema_diagnostics: tuple[ModelSchemaDiagnostic, ...] = ()
    sarif_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class FullDocumentFailureArtifact(EvalModel):
    artifact_type: Literal["full_document_failure"] = "full_document_failure"
    artifact_version: Literal[
        "full-document-failure-v1",
        "full-document-failure-v2",
    ] = "full-document-failure-v2"
    baseline: FullDocumentBaseline
    stage: FailureStage
    exception_category: FailureCategory
    failure_code: FailureCode
    schema_diagnostic: ModelSchemaDiagnostic | None = None


class FullDocumentBenchmarkError(RuntimeError):
    """Payload-free document benchmark failure."""

    def __init__(
        self,
        stage: FailureStage,
        category: FailureCategory,
        code: FailureCode,
        *,
        schema_diagnostic: ModelSchemaDiagnostic | None = None,
    ) -> None:
        super().__init__("full-document benchmark operation failed")
        self.stage = stage
        self.category = category
        self.code = code
        self.schema_diagnostic = schema_diagnostic
        clear_exception_chain(self)


@dataclass(frozen=True)
class LoadedFullDocumentDataset:
    root: Path
    documents: tuple[FullDocumentMetadata, ...]
    gold_claims: tuple[FullDocumentGoldClaim, ...]
    sources: dict[str, SourceFixture]
    sources_by_url: dict[str, SourceFixture]
    document_text: dict[str, str]
    dataset_sha256: str

    def gold_for(self, document_id: str) -> tuple[FullDocumentGoldClaim, ...]:
        return tuple(
            item for item in self.gold_claims if item.document_id == document_id
        )


@dataclass(frozen=True)
class ClaimAlignment:
    gold_index: int
    prediction_index: int
    token_score: float
    exact: bool


@dataclass(frozen=True)
class _TaskEvent:
    task: str
    event: ModelCallTelemetry


def _jsonl(path: Path, schema: type[_T]) -> tuple[_T, ...]:
    values: list[_T] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            values.append(schema.model_validate_json(line))
    return tuple(values)


def _jsonl_prefix(path: Path, schema: type[_T], count: int) -> tuple[_T, ...]:
    values: list[_T] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for _ in range(count):
            line = handle.readline()
            if not line:
                raise ValueError("full-document dev prefix ended early")
            values.append(schema.model_validate_json(line))
    return tuple(values)


def _bundle_hash(root: Path, paths: list[str]) -> str:
    repo_root = root.parent.parent
    rows = []
    for relative in paths:
        path = PurePosixPath(relative)
        if path.is_absolute() or ".." in path.parts:
            raise FullDocumentBenchmarkError(
                "dataset", "local_validation", "dataset_invalid"
            )
        resolved = (repo_root / Path(*path.parts)).resolve()
        try:
            resolved.relative_to(repo_root.resolve())
        except ValueError:
            raise FullDocumentBenchmarkError(
                "dataset", "local_validation", "dataset_invalid"
            ) from None
        digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
        rows.append(f"{digest}  {relative}\n")
    return hashlib.sha256("".join(rows).encode()).hexdigest()


def _validated_dataset(
    resolved_root: Path,
    selected: tuple[FullDocumentMetadata, ...],
    selected_gold: tuple[FullDocumentGoldClaim, ...],
    source_values: tuple[SourceFixture, ...],
    expected_hash: str,
) -> LoadedFullDocumentDataset:
    sources = {item.source_id: item for item in source_values}
    by_url = {item.url: item for item in source_values}
    if len(sources) != len(source_values) or len(by_url) != len(source_values):
        raise ValueError("duplicate source fixture binding")
    if not selected:
        raise ValueError("selected full-document split is empty")
    texts: dict[str, str] = {}
    repo_root = resolved_root.parent.parent
    for document in selected:
        path = (repo_root / document.path).resolve()
        path.relative_to(repo_root.resolve())
        text = path.read_text(encoding="utf-8")
        if hashlib.sha256(text.encode()).hexdigest() != document.content_sha256:
            raise ValueError("document content hash mismatch")
        texts[document.document_id] = text
    for claim in selected_gold:
        text = texts[claim.document_id]
        if text[claim.offset_start : claim.offset_end] != claim.text:
            raise ValueError("gold claim is not a continuous document span")
        if claim.line_start != text.count("\n", 0, claim.offset_start) + 1:
            raise ValueError("gold claim start line mismatch")
        expected_end = (
            text.count("\n", 0, max(claim.offset_start, claim.offset_end - 1)) + 1
        )
        if claim.line_end != expected_end:
            raise ValueError("gold claim end line mismatch")
        for url, source_id in zip(claim.citation_urls, claim.source_ids, strict=True):
            if by_url.get(url) is None or by_url[url].source_id != source_id:
                raise ValueError("gold citation binding mismatch")
        if claim.gold_evidence_span:
            source = sources.get(str(claim.gold_evidence_source_id))
            if source is None or claim.gold_evidence_span not in source.content:
                raise ValueError("gold evidence is not a source substring")
    if any(
        sum(item.document_id == document.document_id for item in selected_gold)
        != document.gold_claim_count
        for document in selected
    ):
        raise ValueError("document gold claim count mismatch")
    return LoadedFullDocumentDataset(
        root=resolved_root,
        documents=selected,
        gold_claims=selected_gold,
        sources=sources,
        sources_by_url=by_url,
        document_text=texts,
        dataset_sha256=expected_hash,
    )


def load_full_document_dev_dataset(root: Path) -> LoadedFullDocumentDataset:
    """Load the frozen dev prefixes without parsing any test JSONL row."""

    try:
        resolved_root = root.resolve()
        manifest = json.loads((resolved_root / "manifest.json").read_text())
        expected_hash = str(manifest["dataset_bundle_sha256"])
        if (
            _bundle_hash(resolved_root, manifest["dataset_bundle_paths"])
            != expected_hash
        ):
            raise ValueError("dataset bundle hash mismatch")
        dev_count = int(manifest["split_counts"]["dev"])
        total_documents = int(manifest["case_counts"]["documents"])
        total_sources = int(manifest["case_counts"]["sources"])
        if total_documents <= 0 or total_sources % total_documents:
            raise ValueError("source-to-document ratio is invalid")
        documents = _jsonl_prefix(
            resolved_root / "documents.jsonl",
            FullDocumentMetadata,
            dev_count,
        )
        if any(
            item.split != "dev"
            or "/test/" in item.path
            or not item.path.startswith("eval_sets/full_document_v1/documents/dev/")
            for item in documents
        ):
            raise ValueError("dev prefix contains a non-dev document")
        gold_count = sum(item.gold_claim_count for item in documents)
        gold = _jsonl_prefix(
            resolved_root / "gold_claims.jsonl",
            FullDocumentGoldClaim,
            gold_count,
        )
        selected_ids = {item.document_id for item in documents}
        if any(
            item.document_id not in selected_ids or "/test/" in item.document_path
            for item in gold
        ):
            raise ValueError("dev prefix contains non-dev gold")
        source_count = dev_count * (total_sources // total_documents)
        source_values = _jsonl_prefix(
            resolved_root / "sources.jsonl",
            SourceFixture,
            source_count,
        )
        return _validated_dataset(
            resolved_root,
            documents,
            gold,
            source_values,
            expected_hash,
        )
    except FullDocumentBenchmarkError:
        raise
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        raise FullDocumentBenchmarkError(
            "dataset", "local_validation", "dataset_invalid"
        ) from None


def load_full_document_dataset(
    root: Path,
    *,
    split: DocumentSplit = "all",
) -> LoadedFullDocumentDataset:
    """Load full-document-v1 and validate every frozen local binding."""

    if split == "dev":
        return load_full_document_dev_dataset(root)
    try:
        resolved_root = root.resolve()
        manifest = json.loads((resolved_root / "manifest.json").read_text())
        documents = _jsonl(resolved_root / "documents.jsonl", FullDocumentMetadata)
        gold = _jsonl(resolved_root / "gold_claims.jsonl", FullDocumentGoldClaim)
        source_values = _jsonl(resolved_root / "sources.jsonl", SourceFixture)
        expected_hash = str(manifest["dataset_bundle_sha256"])
        if (
            _bundle_hash(resolved_root, manifest["dataset_bundle_paths"])
            != expected_hash
        ):
            raise ValueError("dataset bundle hash mismatch")
        selected = tuple(
            item for item in documents if split == "all" or item.split == split
        )
        selected_ids = {item.document_id for item in selected}
        selected_gold = tuple(item for item in gold if item.document_id in selected_ids)
        return _validated_dataset(
            resolved_root,
            selected,
            selected_gold,
            source_values,
            expected_hash,
        )
    except FullDocumentBenchmarkError:
        raise
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        raise FullDocumentBenchmarkError(
            "dataset", "local_validation", "dataset_invalid"
        ) from None


class DocumentCallBudgetClient:
    """Count provider attempts and stop before call or reported-token overflow."""

    def __init__(
        self,
        wrapped: ModelClient,
        *,
        max_calls: int,
        max_total_tokens: int,
    ) -> None:
        if max_calls < 0 or max_total_tokens < 0:
            raise ValueError("document model budgets cannot be negative")
        self.wrapped = wrapped
        self.model_id = wrapped.model_id
        self.prompt_version = wrapped.prompt_version
        self.provider_id = str(getattr(wrapped, "provider_id", "custom"))
        self.temperature = getattr(wrapped, "temperature", None)
        self.temperature_effective = getattr(wrapped, "temperature_effective", None)
        self.thinking_mode = str(getattr(wrapped, "thinking_mode", "not_available"))
        self.max_tokens = getattr(wrapped, "max_tokens", None)
        self.max_calls = max_calls
        self.max_total_tokens = max_total_tokens
        self._events: list[_TaskEvent] = []
        self._reported_tokens = 0

    @staticmethod
    def _provider_events(client: ModelClient) -> tuple[ModelCallTelemetry, ...]:
        value = getattr(client, "telemetry_events", ())
        return (
            tuple(item for item in value if isinstance(item, ModelCallTelemetry))
            if isinstance(value, tuple)
            else ()
        )

    @property
    def telemetry_events(self) -> tuple[ModelCallTelemetry, ...]:
        return tuple(item.event for item in self._events)

    @property
    def task_events(self) -> tuple[_TaskEvent, ...]:
        return tuple(self._events)

    def can_start_model_call(self) -> bool:
        return (
            len(self._events) < self.max_calls
            and self._reported_tokens < self.max_total_tokens
        )

    def _invoke(self, task: str, callback: Callable[[], _T]) -> _T:
        if not self.can_start_model_call():
            raise ModelCallBudgetExceeded("document model budget exhausted")
        before = len(self._provider_events(self.wrapped))
        started = perf_counter()
        failure_kind: FailureKind = "none"
        try:
            result = callback()
        except ModelTransportError:
            failure_kind = "transport"
            raise
        except ModelResponseError:
            failure_kind = "schema"
            raise
        finally:
            observed = self._provider_events(self.wrapped)[before:]
            if not observed:
                observed = (
                    ModelCallTelemetry(
                        latency_ms=(
                            0.0
                            if self.model_id == "deterministic-fake-v1"
                            else max((perf_counter() - started) * 1000.0, 0.0)
                        ),
                        usage_status="missing",
                        failure_kind=failure_kind,
                    ),
                )
            self._events.extend(_TaskEvent(task, event) for event in observed)
            self._reported_tokens += sum(
                int(event.total_tokens)
                for event in observed
                if event.usage_status == "reported" and event.total_tokens is not None
            )
        if self._reported_tokens > self.max_total_tokens:
            raise ModelCallBudgetExceeded("document token budget exhausted")
        return result

    def complete(self, task: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._invoke(task, lambda: self.wrapped.complete(task, payload))

    def complete_model(
        self,
        task: str,
        payload: dict[str, Any],
        schema: type[_T],
    ) -> _T:
        return self._invoke(
            task, lambda: self.wrapped.complete_model(task, payload, schema)
        )


def _safe_call(
    stage: FailureStage,
    callback: Callable[[], _T],
) -> _T:
    try:
        return callback()
    except FullDocumentBenchmarkError:
        raise
    except ModelCallBudgetExceeded:
        raise FullDocumentBenchmarkError(
            stage, "budget", "call_budget_exhausted"
        ) from None
    except ModelSchemaError as error:
        raise FullDocumentBenchmarkError(
            stage,
            "schema",
            "model_schema_invalid",
            schema_diagnostic=error.diagnostic,
        ) from None
    except ModelTransportError:
        raise FullDocumentBenchmarkError(
            stage, "transport", "model_transport_failed"
        ) from None
    except ModelResponseError:
        raise FullDocumentBenchmarkError(
            stage, "schema", "model_response_invalid"
        ) from None
    except MinerScopeError:
        raise FullDocumentBenchmarkError(
            stage, "scope", "miner_scope_violation"
        ) from None
    except JudgeScopeError:
        raise FullDocumentBenchmarkError(
            stage, "scope", "judge_scope_violation"
        ) from None
    except DeterministicConflictError:
        raise FullDocumentBenchmarkError(
            stage, "guard", "deterministic_conflict"
        ) from None
    except LocalValidationError:
        raise FullDocumentBenchmarkError(
            stage, "local_validation", "local_validation_failed"
        ) from None
    except (ValueError, AssertionError, KeyError):
        raise FullDocumentBenchmarkError(
            stage, "local_validation", "unexpected_failure"
        ) from None


def _citation_urls(
    document: ParsedDocument,
    paragraph: MarkdownParagraph,
) -> tuple[str, ...]:
    wanted = set(paragraph.citation_ids)
    return tuple(
        dict.fromkeys(
            url
            for citation in document.citations
            if citation.citation_id in wanted
            for url in citation.resolved_urls
        )
    )


def _paragraph_for_span(
    document: ParsedDocument,
    start: int,
    end: int,
) -> MarkdownParagraph:
    paragraph = next(
        (
            item
            for item in document.paragraphs
            if item.source.offset_start <= start and end <= item.source.offset_end
        ),
        None,
    )
    if paragraph is None:
        raise FullDocumentBenchmarkError(
            "citation_binding", "scope", "claim_span_out_of_scope"
        )
    return paragraph


def _local_claim(
    *,
    document_id: str,
    index: int,
    document: ParsedDocument,
    text: str,
    claim_type: str,
    checkability: Checkability,
    citation_urls: tuple[str, ...],
    offset_start: int,
    offset_end: int,
) -> AtomicClaim:
    return AtomicClaim(
        claim_id=f"{document_id}_prediction_{index:02d}",
        text=text,
        file=document.path,
        line_start=document_text_line(document, offset_start),
        line_end=document_text_line(document, max(offset_start, offset_end - 1)),
        claim_type=claim_type,
        checkability=checkability,
        citation_urls=citation_urls,
    )


def document_text_line(document: ParsedDocument, offset: int) -> int:
    paragraph = next(
        (
            item
            for item in document.paragraphs
            if item.source.offset_start <= offset <= item.source.offset_end
        ),
        None,
    )
    if paragraph is None:
        raise FullDocumentBenchmarkError(
            "citation_binding", "scope", "claim_span_out_of_scope"
        )
    local = max(0, offset - paragraph.source.offset_start)
    return min(
        paragraph.source.line_start + paragraph.raw_text.count("\n", 0, local),
        paragraph.source.line_end,
    )


def _source_metadata(source: SourceFixture, at: datetime) -> SourceMetadata:
    return SourceMetadata(
        source_id=source.source_id,
        url=source.url,
        title=source.source_id,
        retrieved_at=at,
        content_hash=source.content_hash,
        mime_type="text/plain",
        status="ok" if source.available else "unavailable",
    )


def _verdict_from_semantic(
    semantic: LiveDocumentSemanticClaim,
    claim: AtomicClaim,
    dataset: LoadedFullDocumentDataset,
) -> Verdict:
    bound_sources = tuple(
        dataset.sources_by_url[url].source_id for url in semantic.citation_urls
    )
    spans: tuple[EvidenceSpan, ...] = ()
    if semantic.evidence_span is not None:
        source = next(
            (
                dataset.sources_by_url[url]
                for url in semantic.citation_urls
                if semantic.evidence_span in dataset.sources_by_url[url].content
            ),
            None,
        )
        if source is None:
            raise FullDocumentBenchmarkError(
                "single_agent", "scope", "evidence_span_out_of_scope"
            )
        spans = (
            EvidenceSpan(
                source_id=source.source_id,
                text=semantic.evidence_span,
                locator=f"{source.source_id} > source fixture",
            ),
        )
    return Verdict(
        claim_id=claim.claim_id,
        relation=semantic.relation,
        corroboration=(
            CorroborationStatus.CITED_ONLY
            if bound_sources
            else CorroborationStatus.NOT_REQUESTED
        ),
        confidence=semantic.confidence,
        source_ids=bound_sources,
        evidence_spans=spans,
        reason=semantic.reason,
        judge_version=FULL_DOCUMENT_CONTRACT_VERSION,
    )


def _single_agent_payload(
    document: ParsedDocument,
    text: str,
    dataset: LoadedFullDocumentDataset,
) -> dict[str, Any]:
    citation_urls = tuple(
        dict.fromkeys(url for item in document.citations for url in item.resolved_urls)
    )
    return {
        "document": {"path": document.path, "markdown": text},
        "allowed_citations": [
            {
                "url": url,
                "available": dataset.sources_by_url[url].available,
                "content": dataset.sources_by_url[url].content,
            }
            for url in citation_urls
            if url in dataset.sources_by_url
        ],
        "contract_version": FULL_DOCUMENT_CONTRACT_VERSION,
        "requirements": {
            "claim_text": "exact continuous Markdown source substring",
            "citation_urls": "only URLs attached to the claim paragraph",
            "evidence_span": "exact continuous cited-source substring",
            "ordering": "document source order",
        },
    }


def _run_single_document(
    metadata: FullDocumentMetadata,
    text: str,
    document: ParsedDocument,
    dataset: LoadedFullDocumentDataset,
    client: ModelClient,
) -> tuple[FullDocumentPrediction, ...]:
    output = _safe_call(
        "single_agent",
        lambda: client.complete_model(
            "single_agent_document_verification",
            _single_agent_payload(document, text, dataset),
            SingleAgentDocumentOutput,
        ),
    )
    predictions: list[FullDocumentPrediction] = []
    cursor = 0
    for index, semantic in enumerate(output.claims, 1):
        start = text.find(semantic.text, cursor)
        if start < 0:
            raise FullDocumentBenchmarkError(
                "single_agent", "scope", "claim_span_out_of_scope"
            )
        end = start + len(semantic.text)
        cursor = end
        paragraph = _paragraph_for_span(document, start, end)
        allowed = set(_citation_urls(document, paragraph))
        if not set(semantic.citation_urls) <= allowed:
            raise FullDocumentBenchmarkError(
                "single_agent", "scope", "citation_out_of_scope"
            )
        if any(url not in dataset.sources_by_url for url in semantic.citation_urls):
            raise FullDocumentBenchmarkError(
                "single_agent", "scope", "source_binding_missing"
            )
        claim = _local_claim(
            document_id=metadata.document_id,
            index=index,
            document=document,
            text=semantic.text,
            claim_type=semantic.claim_type,
            checkability=semantic.checkability,
            citation_urls=semantic.citation_urls,
            offset_start=start,
            offset_end=end,
        )
        verdict = _verdict_from_semantic(semantic, claim, dataset)
        predictions.append(
            FullDocumentPrediction(
                prediction_id=claim.claim_id,
                document_id=metadata.document_id,
                offset_start=start,
                offset_end=end,
                claim=claim,
                verdict=verdict,
            )
        )
    return tuple(predictions)


def _bind_mined_claims(
    metadata: FullDocumentMetadata,
    text: str,
    document: ParsedDocument,
    paragraph: MarkdownParagraph,
    claims: tuple[AtomicClaim, ...],
    *,
    first_index: int,
) -> tuple[FullDocumentPrediction, ...]:
    locations: list[tuple[int, int]] = []
    raw_cursor = 0
    for claim in claims:
        pattern = re.compile(re.escape(claim.text).replace(r"\ ", r"\s+"))
        match = pattern.search(paragraph.raw_text, raw_cursor)
        if match is None:
            raise FullDocumentBenchmarkError(
                "citation_binding", "scope", "claim_span_out_of_scope"
            )
        start = paragraph.source.offset_start + match.start()
        end = paragraph.source.offset_start + match.end()
        locations.append((start, end))
        raw_cursor = match.end()
    paragraph_citations = tuple(
        item
        for item in document.citations
        if item.citation_id in set(paragraph.citation_ids)
    )
    predictions: list[FullDocumentPrediction] = []
    for offset, (claim, (start, end)) in enumerate(zip(claims, locations, strict=True)):
        next_start = (
            locations[offset + 1][0]
            if offset + 1 < len(locations)
            else paragraph.source.offset_end + 1
        )
        following = tuple(
            item
            for item in paragraph_citations
            if end <= item.source.offset_start < next_start
        )
        selected = (
            following
            if following
            else paragraph_citations
            if len(paragraph_citations) == 1
            else ()
        )
        urls = tuple(
            dict.fromkeys(url for item in selected for url in item.resolved_urls)
        )
        index = first_index + offset
        local = _local_claim(
            document_id=metadata.document_id,
            index=index,
            document=document,
            text=text[start:end],
            claim_type=claim.claim_type,
            checkability=claim.checkability,
            citation_urls=urls,
            offset_start=start,
            offset_end=end,
        )
        predictions.append(
            FullDocumentPrediction(
                prediction_id=local.claim_id,
                document_id=metadata.document_id,
                offset_start=start,
                offset_end=end,
                claim=local,
                verdict=Verdict(
                    claim_id=local.claim_id,
                    relation=Relation.NOT_IN_SOURCE,
                    confidence=0.0,
                    reason="Awaiting adaptive verification.",
                    judge_version=FULL_DOCUMENT_CONTRACT_VERSION,
                ),
                selected_route="deterministic_no_citation",
                route_reason_code="pending_handoff",
            )
        )
    return tuple(predictions)


def _eval_case(
    metadata: FullDocumentMetadata,
    claim: AtomicClaim,
    source: SourceFixture,
) -> EvalCase:
    return EvalCase(
        case_id=claim.claim_id,
        document_path=claim.file,
        claim_text=claim.text,
        claim_line=claim.line_start,
        source_fixture="eval_sets/full_document_v1/sources.jsonl",
        source_id=source.source_id,
        source_url=source.url,
        gold_relation=Relation.NOT_IN_SOURCE,
        claim_type=claim.claim_type,
        mutation_type="full_document_extracted",
        split=metadata.split,
        provenance="full-document-v1 local handoff",
        annotation_status="provisional",
        annotation_notes="Runtime prediction; gold is not provided to the verifier.",
    )


def _adaptive_verdict(
    draft: FullDocumentPrediction,
    metadata: FullDocumentMetadata,
    dataset: LoadedFullDocumentDataset,
    client: ModelClient,
    router_config: AdaptiveRouterConfig,
) -> FullDocumentPrediction:
    claim = draft.claim
    if not claim.citation_urls:
        relation = (
            Relation.NOT_CHECKABLE
            if claim.checkability is Checkability.NOT_CHECKABLE
            else Relation.NOT_IN_SOURCE
        )
        return draft.model_copy(
            update={
                "verdict": Verdict(
                    claim_id=claim.claim_id,
                    relation=relation,
                    confidence=0.95 if relation is Relation.NOT_CHECKABLE else 0.0,
                    reason="No citation was locally bound to this claim.",
                    judge_version=FULL_DOCUMENT_CONTRACT_VERSION,
                ),
                "selected_route": "deterministic_no_citation",
                "route_reason_code": "no_bound_citation",
            }
        )
    try:
        bound = tuple(dataset.sources_by_url[url] for url in claim.citation_urls)
    except KeyError:
        raise FullDocumentBenchmarkError(
            "adaptive_judge", "scope", "source_binding_missing"
        ) from None
    primary = bound[0]
    predicted = _safe_call(
        "adaptive_judge",
        lambda: run_baseline(
            _eval_case(metadata, claim, primary),
            primary,
            "adaptive_live",
            client=client,
            router_config=router_config,
        ),
    )
    evidence: tuple[EvidenceSpan, ...] = ()
    if predicted.predicted_evidence_span is not None:
        locator = next(
            (
                chunk.locator
                for chunk in build_evidence_chunks(primary)
                if predicted.predicted_evidence_span in chunk.text
            ),
            f"{primary.source_id} > full source",
        )
        evidence = (
            EvidenceSpan(
                source_id=primary.source_id,
                text=predicted.predicted_evidence_span,
                locator=locator,
            ),
        )
    verdict = Verdict(
        claim_id=claim.claim_id,
        relation=predicted.predicted_relation,
        corroboration=CorroborationStatus.CITED_ONLY,
        confidence=predicted.confidence,
        source_ids=tuple(item.source_id for item in bound),
        evidence_spans=evidence,
        reason=predicted.reason,
        judge_version=FULL_DOCUMENT_CONTRACT_VERSION,
    )
    return draft.model_copy(
        update={
            "verdict": verdict,
            "selected_route": predicted.selected_route,
            "route_reason_code": predicted.route_reason_code,
        }
    )


def _run_multi_agent_document(
    metadata: FullDocumentMetadata,
    text: str,
    document: ParsedDocument,
    dataset: LoadedFullDocumentDataset,
    client: ModelClient,
    router_config: AdaptiveRouterConfig,
) -> tuple[FullDocumentPrediction, ...]:
    drafts: list[FullDocumentPrediction] = []
    for paragraph in document.paragraphs:
        urls = _citation_urls(document, paragraph)
        mined = _safe_call(
            "miner",
            lambda paragraph=paragraph, urls=urls: ClaimMinerAgent(client).mine(
                MinerInput(
                    text=paragraph.plain_text,
                    file=document.path,
                    line_start=paragraph.source.line_start,
                    line_end=paragraph.source.line_end,
                    heading_path=paragraph.heading_path,
                    citation_urls=urls,
                )
            ),
        )
        drafts.extend(
            _bind_mined_claims(
                metadata,
                text,
                document,
                paragraph,
                mined.claims,
                first_index=len(drafts) + 1,
            )
        )
        if len(drafts) > MAX_CLAIMS_PER_DOCUMENT:
            raise FullDocumentBenchmarkError(
                "miner", "local_validation", "claim_count_exceeded"
            )
    return tuple(
        _adaptive_verdict(item, metadata, dataset, client, router_config)
        for item in drafts
    )


def align_document_claims(
    gold: tuple[FullDocumentGoldClaim, ...],
    predictions: tuple[FullDocumentPrediction, ...],
) -> tuple[ClaimAlignment, ...]:
    """Deterministic one-to-one exact-first alignment."""

    matches: list[ClaimAlignment] = []
    used_gold: set[int] = set()
    used_prediction: set[int] = set()
    for gold_index, expected in enumerate(gold):
        for prediction_index, actual in enumerate(predictions):
            if prediction_index in used_prediction:
                continue
            if (
                expected.text == actual.claim.text
                and expected.offset_start == actual.offset_start
                and expected.offset_end == actual.offset_end
            ):
                matches.append(ClaimAlignment(gold_index, prediction_index, 1.0, True))
                used_gold.add(gold_index)
                used_prediction.add(prediction_index)
                break
    candidates = sorted(
        (
            (
                -token_f1(expected.text, actual.claim.text),
                abs(expected.line_start - actual.claim.line_start),
                gold_index,
                prediction_index,
            )
            for gold_index, expected in enumerate(gold)
            if gold_index not in used_gold
            for prediction_index, actual in enumerate(predictions)
            if prediction_index not in used_prediction
        )
    )
    for negative_score, _, gold_index, prediction_index in candidates:
        score = -negative_score
        if score <= 0.0:
            continue
        if gold_index in used_gold or prediction_index in used_prediction:
            continue
        matches.append(ClaimAlignment(gold_index, prediction_index, score, False))
        used_gold.add(gold_index)
        used_prediction.add(prediction_index)
    return tuple(sorted(matches, key=lambda item: item.gold_index))


def _latency_percentile(values: tuple[float, ...], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return float(ordered[min(len(ordered) - 1, int(len(ordered) * fraction))])


def _stage_telemetry(events: tuple[ModelCallTelemetry, ...]) -> StageTelemetry:
    summary: ModelTelemetrySummary = summarize_model_telemetry(events)
    return StageTelemetry(
        calls=summary.calls,
        usage_status=summary.usage_status,
        input_tokens=summary.input_tokens,
        output_tokens=summary.output_tokens,
        total_tokens=summary.total_tokens,
        reasoning_tokens=summary.reasoning_tokens,
        latency_ms=summary.call_latencies_ms,
        p50_latency_ms=(
            float(median(summary.call_latencies_ms))
            if summary.call_latencies_ms
            else None
        ),
        p95_latency_ms=_latency_percentile(summary.call_latencies_ms, 0.95),
        schema_failures=summary.schema_failures,
        transport_failures=summary.transport_failures,
        finish_reasons=summary.finish_reasons,
        max_tokens_requested=summary.max_tokens_requested,
        response_content_lengths=summary.response_content_lengths,
        suspected_token_truncations=summary.suspected_token_truncations,
    )


def _policy_failed_for_gold(
    gold: tuple[FullDocumentGoldClaim, ...],
    config: EffectiveConfig,
) -> bool:
    return should_fail(
        [
            decide_policy(
                item.gold_relation,
                (
                    CorroborationStatus.CITED_ONLY
                    if item.source_ids
                    else CorroborationStatus.NOT_REQUESTED
                ),
                policy=config.policy,
                claim_type=item.claim_type,
            )
            for item in gold
        ],
        config.policy,
    )


def compute_full_document_metrics(
    dataset: LoadedFullDocumentDataset,
    predictions: tuple[FullDocumentPrediction, ...],
    documents: tuple[DocumentResult, ...],
    *,
    config: EffectiveConfig,
    stage_telemetry: dict[str, StageTelemetry],
) -> dict[str, Any]:
    exact = 0
    relaxed_sum = 0.0
    citation_correct = 0
    line_correct = 0
    evidence_scores: list[float] = []
    end_to_end_correct = 0
    atomicity_violations = 0
    relation_gold: list[str] = []
    relation_predicted: list[str] = []
    confidences: list[float] = []
    prediction_by_document = {
        document.document_id: tuple(
            item for item in predictions if item.document_id == document.document_id
        )
        for document in dataset.documents
    }
    result_by_document = {item.document_id: item for item in documents}
    for document in dataset.documents:
        gold = dataset.gold_for(document.document_id)
        actual = prediction_by_document[document.document_id]
        alignment = align_document_claims(gold, actual)
        aligned = {item.gold_index: item for item in alignment}
        exact += sum(item.exact for item in alignment)
        relaxed_sum += sum(item.token_score for item in alignment)
        for prediction in actual:
            overlaps = sum(
                prediction.offset_start < item.offset_end
                and item.offset_start < prediction.offset_end
                for item in gold
            )
            atomicity_violations += overlaps > 1
        for gold_index, expected in enumerate(gold):
            relation_gold.append(expected.gold_relation.value)
            match = aligned.get(gold_index)
            if match is None:
                relation_predicted.append("__missing__")
                confidences.append(0.0)
                if expected.gold_evidence_span:
                    evidence_scores.append(0.0)
                continue
            prediction = actual[match.prediction_index]
            relation_predicted.append(prediction.verdict.relation.value)
            confidences.append(prediction.verdict.confidence)
            citation_correct += prediction.claim.citation_urls == expected.citation_urls
            line_correct += (
                prediction.claim.line_start == expected.line_start
                and prediction.claim.line_end == expected.line_end
            )
            if expected.gold_evidence_span:
                evidence_scores.append(
                    token_f1(
                        (
                            prediction.verdict.evidence_spans[0].text
                            if prediction.verdict.evidence_spans
                            else None
                        ),
                        expected.gold_evidence_span,
                    )
                )
            end_to_end_correct += (
                match.exact and prediction.verdict.relation is expected.gold_relation
            )
    predicted_count = len(predictions)
    gold_count = len(dataset.gold_claims)
    exact_precision = safe_divide(exact, predicted_count)
    exact_recall = safe_divide(exact, gold_count)
    relaxed_precision = safe_divide(relaxed_sum, predicted_count)
    relaxed_recall = safe_divide(relaxed_sum, gold_count)
    end_precision = safe_divide(end_to_end_correct, predicted_count)
    end_recall = safe_divide(end_to_end_correct, gold_count)
    relation = classification_metrics(
        relation_gold,
        relation_predicted,
        confidences=confidences,
    )
    policy_correct = sum(
        result_by_document[item.document_id].policy_failed
        == _policy_failed_for_gold(dataset.gold_for(item.document_id), config)
        for item in dataset.documents
    )
    handoff_attempts = sum(item.handoff_attempts for item in documents)
    handoff_failures = sum(item.handoff_failures for item in documents)
    return {
        "extraction": {
            "exact_precision": exact_precision,
            "exact_recall": exact_recall,
            "exact_f1": safe_divide(
                2 * exact_precision * exact_recall,
                exact_precision + exact_recall,
            ),
            "relaxed_token_precision": relaxed_precision,
            "relaxed_token_recall": relaxed_recall,
            "relaxed_span_token_f1": safe_divide(
                2 * relaxed_precision * relaxed_recall,
                relaxed_precision + relaxed_recall,
            ),
            "atomicity_violation_rate": safe_divide(
                atomicity_violations, predicted_count
            ),
            "citation_binding_accuracy": safe_divide(citation_correct, gold_count),
            "line_locator_accuracy": safe_divide(line_correct, gold_count),
        },
        "relation": relation,
        "not_checkable_f1": relation["per_label"][Relation.NOT_CHECKABLE.value]["f1"],
        "evidence_span_f1": safe_divide(sum(evidence_scores), len(evidence_scores)),
        "end_to_end_claim_relation": {
            "precision": end_precision,
            "recall": end_recall,
            "f1": safe_divide(
                2 * end_precision * end_recall, end_precision + end_recall
            ),
        },
        "document_policy_decision_accuracy": safe_divide(
            policy_correct, len(dataset.documents)
        ),
        "miner_judge_handoff": {
            "attempts": handoff_attempts,
            "failures": handoff_failures,
            "failure_rate": safe_divide(handoff_failures, handoff_attempts),
        },
        "model": {
            "stages": {
                key: value.model_dump(mode="json")
                for key, value in sorted(stage_telemetry.items())
            },
            "calls": sum(value.calls for value in stage_telemetry.values()),
            "input_tokens": (
                sum(int(value.input_tokens or 0) for value in stage_telemetry.values())
                if all(
                    value.usage_status in {"reported", "not_applicable"}
                    for value in stage_telemetry.values()
                )
                else None
            ),
            "output_tokens": (
                sum(int(value.output_tokens or 0) for value in stage_telemetry.values())
                if all(
                    value.usage_status in {"reported", "not_applicable"}
                    for value in stage_telemetry.values()
                )
                else None
            ),
            "reasoning_tokens": (
                sum(
                    int(value.reasoning_tokens or 0)
                    for value in stage_telemetry.values()
                )
                if all(
                    value.usage_status == "not_applicable"
                    or value.reasoning_tokens is not None
                    for value in stage_telemetry.values()
                )
                else None
            ),
            "total_tokens": (
                sum(int(value.total_tokens or 0) for value in stage_telemetry.values())
                if all(
                    value.usage_status in {"reported", "not_applicable"}
                    for value in stage_telemetry.values()
                )
                else None
            ),
        },
    }


def _request_provenance(
    client: ModelClient,
    baseline: FullDocumentBaseline,
) -> FullDocumentRequestProvenance:
    provider_id = str(getattr(client, "provider_id", "custom"))
    raw_max_tokens = getattr(client, "max_tokens", None)
    max_tokens = (
        raw_max_tokens
        if isinstance(raw_max_tokens, int) and not isinstance(raw_max_tokens, bool)
        else None
    )
    if provider_id == "openai-compatible" and max_tokens != FULL_DOCUMENT_MAX_TOKENS:
        raise FullDocumentBenchmarkError(
            "provider_contract",
            "local_validation",
            "provider_output_budget_mismatch",
        )
    thinking_mode = str(getattr(client, "thinking_mode", "not_available"))
    return FullDocumentRequestProvenance(
        provider_id=provider_id,
        model_id=client.model_id,
        prompt_version=client.prompt_version,
        output_schema_names=(
            ("SingleAgentDocumentOutput",)
            if baseline == "single_agent_document_live"
            else ("LiveMinerDraftOutput", "LiveJudgeSemanticOutput")
        ),
        max_tokens_requested=max_tokens,
        thinking_mode=(
            "provider_default"
            if thinking_mode == "provider_default"
            else "not_available"
        ),
        temperature_requested=getattr(client, "temperature", None),
    )


def run_full_document_benchmark(
    dataset: LoadedFullDocumentDataset,
    baseline: FullDocumentBaseline,
    client: ModelClient,
    *,
    run_id: str,
    started_at: datetime,
    config: EffectiveConfig | None = None,
    router_config: AdaptiveRouterConfig | None = None,
    max_calls: int = 600,
    max_total_tokens: int = 1_000_000,
) -> FullDocumentRunArtifact:
    """Run one frozen full-document baseline with no network or source fetching."""

    effective_config = config or EffectiveConfig()
    effective_router = router_config or AdaptiveRouterConfig()
    request_provenance = _request_provenance(client, baseline)
    budgeted = DocumentCallBudgetClient(
        client,
        max_calls=max_calls,
        max_total_tokens=max_total_tokens,
    )
    recovered = SchemaRecoveryClient(budgeted)
    parsed_documents: list[ParsedDocument] = []
    predictions: list[FullDocumentPrediction] = []
    document_results: list[DocumentResult] = []
    for metadata in dataset.documents:
        text = dataset.document_text[metadata.document_id]
        document = parse_markdown(text, metadata.path)
        parsed_documents.append(document)
        if baseline == "single_agent_document_live":
            current = _run_single_document(metadata, text, document, dataset, recovered)
            handoff_attempts = 0
        else:
            current = _run_multi_agent_document(
                metadata,
                text,
                document,
                dataset,
                recovered,
                effective_router,
            )
            handoff_attempts = len(current)
        predictions.extend(current)
        decisions = [
            decide_policy(
                item.verdict.relation,
                item.verdict.corroboration,
                policy=effective_config.policy,
                claim_type=item.claim.claim_type,
                document=document,
            )
            for item in current
        ]
        document_results.append(
            DocumentResult(
                document_id=metadata.document_id,
                prediction_ids=tuple(item.prediction_id for item in current),
                policy_failed=should_fail(decisions, effective_config.policy),
                handoff_attempts=handoff_attempts,
                handoff_failures=0,
            )
        )
    audit = AuditArtifact(
        run=RunMetadata(
            run_id=run_id,
            started_at=started_at.astimezone(UTC),
            finished_at=started_at.astimezone(UTC),
            tool_version=FULL_DOCUMENT_BENCHMARK_VERSION,
        ),
        effective_config=effective_config,
        documents=tuple(parsed_documents),
        claims=tuple(item.claim for item in predictions),
        verdicts=tuple(item.verdict for item in predictions),
        sources=tuple(
            _source_metadata(source, started_at.astimezone(UTC))
            for source in sorted(
                dataset.sources.values(), key=lambda item: item.source_id
            )
        ),
    )
    miner_events = tuple(
        item.event for item in budgeted.task_events if item.task == "claim_mining"
    )
    single_events = tuple(
        item.event
        for item in budgeted.task_events
        if item.task == "single_agent_document_verification"
    )
    judge_events = tuple(
        item.event
        for item in budgeted.task_events
        if item.task not in {"claim_mining", "single_agent_document_verification"}
    )
    stage_telemetry = {
        "miner": _stage_telemetry(miner_events),
        "judge": _stage_telemetry(judge_events),
        "single_agent": _stage_telemetry(single_events),
    }
    metrics = compute_full_document_metrics(
        dataset,
        tuple(predictions),
        tuple(document_results),
        config=effective_config,
        stage_telemetry=stage_telemetry,
    )
    sarif_hash = hashlib.sha256(sarif_bytes(audit)).hexdigest()
    return FullDocumentRunArtifact(
        artifact_version=FULL_DOCUMENT_BENCHMARK_VERSION,
        baseline=baseline,
        split=(
            dataset.documents[0].split
            if len({item.split for item in dataset.documents}) == 1
            else "all"
        ),
        dataset_sha256=dataset.dataset_sha256,
        run_id=run_id,
        predictions=tuple(predictions),
        documents=tuple(document_results),
        audit=audit,
        metrics=metrics,
        stage_telemetry=stage_telemetry,
        request_provenance=request_provenance,
        schema_recovery=recovered.schema_recovery,
        schema_diagnostics=recovered.schema_diagnostics,
        sarif_sha256=sarif_hash,
    )


def compare_full_document_runs(
    single: FullDocumentRunArtifact,
    multi: FullDocumentRunArtifact,
) -> dict[str, float | int | None]:
    if single.dataset_sha256 != multi.dataset_sha256 or single.split != multi.split:
        raise ValueError("document run comparison requires identical dataset scope")
    single_tokens = single.metrics["model"]["total_tokens"]
    multi_tokens = multi.metrics["model"]["total_tokens"]
    return {
        "exact_claim_f1_delta": (
            multi.metrics["extraction"]["exact_f1"]
            - single.metrics["extraction"]["exact_f1"]
        ),
        "relation_fixed_six_macro_f1_delta": (
            multi.metrics["relation"]["fixed_taxonomy_macro_f1"]
            - single.metrics["relation"]["fixed_taxonomy_macro_f1"]
        ),
        "end_to_end_f1_delta": (
            multi.metrics["end_to_end_claim_relation"]["f1"]
            - single.metrics["end_to_end_claim_relation"]["f1"]
        ),
        "policy_accuracy_delta": (
            multi.metrics["document_policy_decision_accuracy"]
            - single.metrics["document_policy_decision_accuracy"]
        ),
        "model_calls_delta": (
            multi.metrics["model"]["calls"] - single.metrics["model"]["calls"]
        ),
        "total_tokens_delta": (
            int(multi_tokens) - int(single_tokens)
            if multi_tokens is not None and single_tokens is not None
            else None
        ),
    }


def full_document_artifact_bytes(artifact: FullDocumentRunArtifact) -> bytes:
    return (
        json.dumps(
            artifact.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode()


def _bounded_output(path: Path, project_root: Path) -> Path:
    root = project_root.resolve()
    resolved = (path if path.is_absolute() else root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        raise FullDocumentBenchmarkError(
            "artifact", "artifact", "artifact_write_failed"
        ) from None
    if resolved == root or (resolved.exists() and not resolved.is_file()):
        raise FullDocumentBenchmarkError(
            "artifact", "artifact", "artifact_write_failed"
        )
    return resolved


def write_full_document_outputs(
    artifact: FullDocumentRunArtifact,
    output_json: Path,
    output_sarif: Path,
    *,
    project_root: Path,
) -> tuple[Path, Path]:
    """Atomically write the canonical benchmark artifact and derived SARIF."""

    destination = _bounded_output(output_json, project_root)
    temporary_path: Path | None = None
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination = _bounded_output(destination, project_root)
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=".full-document.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(full_document_artifact_bytes(artifact))
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, destination)
        sarif_path = write_sarif(
            artifact.audit,
            output_sarif,
            project_root=project_root,
        )
    except FullDocumentBenchmarkError:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    except OSError:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise FullDocumentBenchmarkError(
            "artifact", "artifact", "artifact_write_failed"
        ) from None
    return destination, sarif_path


def safe_full_document_failure(
    baseline: FullDocumentBaseline,
    error: BaseException,
) -> FullDocumentFailureArtifact:
    clear_exception_chain(error)
    if isinstance(error, FullDocumentBenchmarkError):
        return FullDocumentFailureArtifact(
            baseline=baseline,
            stage=error.stage,
            exception_category=error.category,
            failure_code=error.code,
            schema_diagnostic=error.schema_diagnostic,
        )
    return FullDocumentFailureArtifact(
        baseline=baseline,
        stage="artifact",
        exception_category="unexpected",
        failure_code="unexpected_failure",
    )


__all__ = [
    "FULL_DOCUMENT_BENCHMARK_VERSION",
    "FULL_DOCUMENT_CONTRACT_VERSION",
    "FULL_DOCUMENT_MAX_TOKENS",
    "FULL_DOCUMENT_PROVIDER_CONTRACT_VERSION",
    "MAX_CLAIMS_PER_DOCUMENT",
    "ClaimAlignment",
    "DocumentCallBudgetClient",
    "DocumentResult",
    "FullDocumentBenchmarkError",
    "FullDocumentFailureArtifact",
    "FullDocumentGoldClaim",
    "FullDocumentMetadata",
    "FullDocumentPrediction",
    "FullDocumentRequestProvenance",
    "FullDocumentRunArtifact",
    "LiveDocumentSemanticClaim",
    "LoadedFullDocumentDataset",
    "SingleAgentDocumentOutput",
    "StageTelemetry",
    "align_document_claims",
    "compare_full_document_runs",
    "compute_full_document_metrics",
    "full_document_artifact_bytes",
    "load_full_document_dataset",
    "load_full_document_dev_dataset",
    "run_full_document_benchmark",
    "safe_full_document_failure",
    "write_full_document_outputs",
]
