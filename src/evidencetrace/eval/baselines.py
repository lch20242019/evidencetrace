"""Semantically explicit deterministic and live Phase 3 baselines."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from datetime import UTC, datetime
from time import perf_counter
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from evidencetrace.agents.judge import ClaimJudgeAgent
from evidencetrace.audit_models import EvidenceChunk, JudgeInput
from evidencetrace.checks.deterministic import (
    bounded_evidence_text,
    deterministic_signals,
    is_high_confidence_subjective,
)
from evidencetrace.eval.errors import LocalValidationError
from evidencetrace.eval.models import (
    BaselineName,
    EvalCase,
    EvalPrediction,
    SingleAgentLiveOutput,
    SourceFixture,
)
from evidencetrace.eval.router import (
    AdaptiveRouteDecision,
    AdaptiveRouterConfig,
    select_adaptive_route,
)
from evidencetrace.model_client import (
    FailureKind,
    ModelCallTelemetry,
    ModelClient,
    ModelResponseError,
    ModelTelemetrySummary,
    ModelTransportError,
    summarize_model_telemetry,
)
from evidencetrace.models import (
    AtomicClaim,
    Checkability,
    Relation,
    SourceMetadata,
)
from evidencetrace.relation_policy import RELATION_DEFINITIONS_BILINGUAL
from evidencetrace.retrieval.rank import LexicalRetriever

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[._-][A-Za-z0-9]+)*")
STOPWORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "of",
    "to",
    "in",
    "on",
    "and",
    "for",
    "with",
}
SUBSTANTIVE_RELATIONS = {
    Relation.ENTAILED,
    Relation.PARTIALLY_ENTAILED,
    Relation.CONTRADICTED,
}


R = TypeVar("R")
T = TypeVar("T", bound=BaseModel)


class _CountingClient:
    """Capture secret-free call telemetry around one narrow baseline run."""

    def __init__(self, wrapped: ModelClient) -> None:
        self.wrapped = wrapped
        self.model_id = wrapped.model_id
        self.prompt_version = wrapped.prompt_version
        self.provider_id = str(getattr(wrapped, "provider_id", "custom"))
        raw_temperature = getattr(wrapped, "temperature", None)
        self.temperature = (
            float(raw_temperature)
            if isinstance(raw_temperature, (int, float))
            and not isinstance(raw_temperature, bool)
            else None
        )
        self.calls = 0
        self._events: list[ModelCallTelemetry] = []

    @staticmethod
    def _provider_events(client: ModelClient) -> tuple[ModelCallTelemetry, ...]:
        value = getattr(client, "telemetry_events", ())
        if not isinstance(value, tuple):
            return ()
        return tuple(event for event in value if isinstance(event, ModelCallTelemetry))

    def _invoke(self, callback: Callable[[], R]) -> R:
        before = len(self._provider_events(self.wrapped))
        started = perf_counter()
        failure_kind: FailureKind = "none"
        try:
            return callback()
        except ModelTransportError:
            failure_kind = "transport"
            raise
        except (ModelResponseError, ValidationError):
            failure_kind = "schema"
            raise
        finally:
            provider_events = self._provider_events(self.wrapped)[before:]
            if provider_events:
                self._events.extend(provider_events)
                self.calls += len(provider_events)
            else:
                self.calls += 1
                self._events.append(
                    ModelCallTelemetry(
                        latency_ms=max((perf_counter() - started) * 1000.0, 0.0),
                        usage_status="missing",
                        failure_kind=failure_kind,
                    )
                )

    @property
    def telemetry(self) -> ModelTelemetrySummary:
        return summarize_model_telemetry(tuple(self._events))

    def complete(self, task: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._invoke(lambda: self.wrapped.complete(task, payload))

    def complete_model(
        self,
        task: str,
        payload: dict[str, Any],
        schema: type[T],
    ) -> T:
        return self._invoke(lambda: self.wrapped.complete_model(task, payload, schema))


def _model_fields(counted: _CountingClient) -> dict[str, Any]:
    summary = counted.telemetry
    if summary.calls != counted.calls:
        raise LocalValidationError("model_telemetry_inconsistent")
    return {
        "model_calls": summary.calls,
        "model_call_latencies_ms": summary.call_latencies_ms,
        "model_usage_status": summary.usage_status,
        "input_tokens": summary.input_tokens,
        "output_tokens": summary.output_tokens,
        "total_tokens": summary.total_tokens,
        "schema_failure_count": summary.schema_failures,
        "transport_failure_count": summary.transport_failures,
        "model_provider": counted.provider_id,
        "model_id": counted.model_id,
        "model_temperature": counted.temperature,
    }


def _canonical_prediction(**payload: Any) -> EvalPrediction:
    try:
        return EvalPrediction(**payload)
    except (ValueError, AssertionError, KeyError):
        pass
    raise LocalValidationError("canonical_prediction_invalid")


def maximum_expected_live_model_calls(
    case_count: int, *, schema_retry_limit: int = 0
) -> int:
    """Return the three live pair baselines' hard request ceiling."""

    if case_count < 0:
        raise ValueError("case_count cannot be negative")
    if schema_retry_limit not in {0, 1}:
        raise ValueError("schema_retry_limit must be zero or one")
    return case_count * 3 * (1 + schema_retry_limit)


def _words(value: str) -> set[str]:
    return {
        token.casefold()
        for token in WORD_RE.findall(value)
        if token.casefold() not in STOPWORDS
    }


def build_evidence_chunks(source: SourceFixture) -> tuple[EvidenceChunk, ...]:
    """Create stable source chunks while retaining legacy Latin splitting."""

    parts: list[EvidenceChunk] = []
    start = 0
    boundary_pattern = (
        r"(?<=[.!?])\s+|(?<=[\u3002\uff01\uff1f])|\r?\n\s*\r?\n"
        if re.search(r"[\u3400-\u9fff]", source.content)
        else r"(?<=[.!?])\s+"
    )
    boundaries = [*re.finditer(boundary_pattern, source.content), None]
    for boundary in boundaries:
        end = boundary.start() if boundary is not None else len(source.content)
        raw = source.content[start:end]
        text = raw.strip()
        leading = len(raw) - len(raw.lstrip())
        actual_start = start + leading
        if text:
            parts.append(
                EvidenceChunk(
                    source_id=source.source_id,
                    url=source.url,
                    text=text,
                    heading_path=(source.source_id,),
                    locator=f"{source.source_id} > paragraph {len(parts) + 1}",
                    char_start=actual_start,
                    char_end=actual_start + len(text),
                )
            )
        start = boundary.end() if boundary is not None else len(source.content)
    return tuple(parts)


def _chunks(source: SourceFixture) -> tuple[EvidenceChunk, ...]:
    return build_evidence_chunks(source)


def _full_source_chunk(source: SourceFixture) -> EvidenceChunk:
    return EvidenceChunk(
        source_id=source.source_id,
        url=source.url,
        text=source.content,
        heading_path=(source.source_id,),
        locator=f"{source.source_id} > full source",
        char_start=0,
        char_end=len(source.content),
    )


def _metadata(source: SourceFixture) -> SourceMetadata:
    return SourceMetadata(
        source_id=source.source_id,
        url=source.url,
        title=source.source_id,
        retrieved_at=datetime(2026, 7, 10, tzinfo=UTC),
        content_hash=source.content_hash,
        mime_type="text/plain",
        status="ok" if source.available else "unavailable",
    )


def _heuristic_verdict(
    case: EvalCase,
    source: SourceFixture,
    evidence: tuple[Any, ...],
    *,
    baseline: BaselineName,
) -> tuple[Relation, float, str | None, str, str]:
    if not source.available:
        return (
            Relation.SOURCE_UNAVAILABLE,
            0.0,
            None,
            "The source fixture is unavailable.",
            "none",
        )
    if not evidence:
        return (
            Relation.NOT_IN_SOURCE,
            0.0,
            None,
            "No lexical evidence was retrieved.",
            "retrieval",
        )
    best = evidence[0]
    claim_words = _words(case.claim_text or case.document_path or "")
    overlap = len(claim_words & _words(best.text)) / max(len(claim_words), 1)
    signals = deterministic_signals(
        case.claim_text or "",
        best.text,
        bounded_evidence_context=bounded_evidence_text(evidence),
    )
    mismatch = any(
        item.code in {"numeric_mismatch", "date_mismatch", "version_mismatch"}
        and item.severity == "error"
        for item in signals
    )
    negation = any(item.code == "negation_mismatch" for item in signals)
    if mismatch and overlap >= 0.25:
        relation = Relation.CONTRADICTED
        reason = f"{baseline} detected a deterministic slot mismatch."
    elif negation and overlap >= 0.25:
        relation = Relation.CONTRADICTED
        reason = f"{baseline} detected a negation mismatch."
    elif overlap >= 0.75:
        relation = Relation.ENTAILED
        reason = f"{baseline} found strong lexical support."
    elif overlap >= 0.35:
        relation = Relation.PARTIALLY_ENTAILED
        reason = f"{baseline} found partial lexical support."
    else:
        relation = Relation.NOT_IN_SOURCE
        reason = f"{baseline} abstained because lexical overlap was low."
    span = best.text if relation in SUBSTANTIVE_RELATIONS else None
    error_stage = "judge" if relation is Relation.NOT_IN_SOURCE else "none"
    return relation, min(max(overlap, 0.0), 0.99), span, reason, error_stage


def _single_agent_payload(
    case: EvalCase,
    source: SourceFixture,
) -> dict[str, Any]:
    return {
        "claim": {
            "text": case.claim_text or case.document_path or "",
            "line": case.claim_line,
            "claim_type": case.claim_type,
        },
        "source": {
            "source_id": source.source_id,
            "url": source.url,
            "content": source.content,
            "available": source.available,
            "provenance": source.provenance,
        },
        "allowed_relations": [relation.value for relation in Relation],
        "relation_definitions": RELATION_DEFINITIONS_BILINGUAL,
        "evidence_requirement": (
            "For entailed, partially_entailed, or contradicted, copy an "
            "exact substring from source.content into evidence_span."
        ),
    }


def estimate_full_context_size(case: EvalCase, source: SourceFixture) -> int:
    """Measure the UTF-8 bytes of the exact model payload used by this baseline."""

    serialized = json.dumps(
        _single_agent_payload(case, source),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return len(serialized.encode("utf-8"))


def _single_agent_live(
    case: EvalCase,
    source: SourceFixture,
    client: ModelClient,
    *,
    baseline: BaselineName = "single_agent_live",
) -> EvalPrediction:
    counted = _CountingClient(client)
    output = SingleAgentLiveOutput.model_validate(
        counted.complete_model(
            "single_agent_full_source_verification",
            _single_agent_payload(case, source),
            SingleAgentLiveOutput,
        )
    )
    if not 1 <= counted.calls <= 2:
        raise LocalValidationError("model_call_count_out_of_bounds")
    if output.evidence_span is not None and output.evidence_span not in source.content:
        raise LocalValidationError("evidence_span_not_in_source")
    if not source.available and output.relation is not Relation.SOURCE_UNAVAILABLE:
        raise LocalValidationError("source_availability_conflict")
    return _canonical_prediction(
        case_id=case.case_id,
        baseline=baseline,
        predicted_relation=output.relation,
        confidence=output.confidence,
        reason=output.reason,
        predicted_evidence_span=output.evidence_span,
        predicted_line=case.claim_line,
        extraction_count=0,
        **_model_fields(counted),
        latency_ms=0.0,
        error_stage="judge" if output.relation is Relation.NOT_IN_SOURCE else "none",
    )


def _pair_claim(case: EvalCase) -> AtomicClaim:
    if case.claim_text is None:
        raise ValueError("pair-level retrieval/Judge baseline requires claim_text")
    return AtomicClaim(
        claim_id=f"eval_{case.case_id}",
        text=case.claim_text,
        file=case.document_path or f"eval_sets/{case.case_id}.md",
        line_start=case.claim_line,
        line_end=case.claim_line,
        claim_type=case.claim_type,
        slots={},
        checkability=(
            Checkability.NOT_CHECKABLE
            if is_high_confidence_subjective(case.claim_text)
            else Checkability.CHECKABLE
        ),
        citation_urls=(case.source_url,),
    )


def _retrieval_judge(
    case: EvalCase,
    source: SourceFixture,
    *,
    baseline: BaselineName,
    client: ModelClient | None,
    retrieved: tuple[Any, ...] | None = None,
) -> EvalPrediction:
    counted = _CountingClient(client) if client is not None else None
    claim = _pair_claim(case)
    if retrieved is None:
        retrieved = (
            LexicalRetriever(_chunks(source), neighbor_window=0).search(
                claim.text, top_k=5
            )
            if source.available
            else ()
        )
    model_client = counted if counted is not None else None
    signals = deterministic_signals(
        claim.text,
        retrieved[0].text if retrieved else "",
        bounded_evidence_context=bounded_evidence_text(retrieved),
    )
    judged = (
        ClaimJudgeAgent(model_client)
        .judge(
            JudgeInput(
                claim=claim,
                evidence=retrieved,
                source=_metadata(source),
                signals=signals,
            )
        )
        .verdict
    )
    relation = judged.relation
    confidence = judged.confidence
    span = judged.evidence_spans[0].text if judged.evidence_spans else None
    reason = judged.reason
    stage = "judge" if relation is Relation.NOT_IN_SOURCE else "none"
    if counted is not None and counted.calls > 2:
        raise LocalValidationError("model_call_count_out_of_bounds")

    return _canonical_prediction(
        case_id=case.case_id,
        baseline=baseline,
        predicted_relation=relation,
        confidence=confidence,
        reason=reason,
        predicted_evidence_span=span,
        retrieved_texts=tuple(item.text for item in retrieved),
        retrieval_rank=1 if retrieved else None,
        extraction_count=0,
        predicted_line=case.claim_line,
        **(_model_fields(counted) if counted is not None else {}),
        latency_ms=0.0,
        error_stage=stage,
    )


def _with_route_telemetry(
    prediction: EvalPrediction,
    decision: AdaptiveRouteDecision,
) -> EvalPrediction:
    payload = prediction.model_dump(mode="json")
    payload.update(
        {
            "baseline": "adaptive_live",
            "selected_route": decision.selected_route,
            "route_reason_code": decision.route_reason_code,
            "chunk_count": decision.chunk_count,
            "estimated_context_size": decision.estimated_context_size,
        }
    )
    try:
        return EvalPrediction.model_validate(payload)
    except (ValueError, AssertionError, KeyError):
        pass
    raise LocalValidationError("canonical_prediction_invalid")


def _adaptive_live(
    case: EvalCase,
    source: SourceFixture,
    client: ModelClient,
    *,
    router_config: AdaptiveRouterConfig,
) -> EvalPrediction:
    chunks = build_evidence_chunks(source)
    decision = select_adaptive_route(
        source_available=source.available,
        purely_subjective=is_high_confidence_subjective(
            case.claim_text or case.document_path or ""
        ),
        chunk_count=len(chunks),
        estimated_context_size=estimate_full_context_size(case, source),
        config=router_config,
    )
    if decision.selected_route == "deterministic_source_unavailable":
        prediction = _canonical_prediction(
            case_id=case.case_id,
            baseline="single_agent_live",
            predicted_relation=Relation.SOURCE_UNAVAILABLE,
            confidence=0.0,
            reason="Adaptive route observed unavailable source metadata.",
            extraction_count=0,
            predicted_line=case.claim_line,
            model_calls=0,
            latency_ms=0.0,
            error_stage="none",
        )
    elif decision.selected_route == "deterministic_not_checkable":
        prediction = _canonical_prediction(
            case_id=case.case_id,
            baseline="single_agent_live",
            predicted_relation=Relation.NOT_CHECKABLE,
            confidence=0.95,
            reason="Adaptive route found a subjective claim without fact slots.",
            extraction_count=0,
            predicted_line=case.claim_line,
            model_calls=0,
            latency_ms=0.0,
            error_stage="none",
        )
    elif decision.selected_route == "full_context_single_agent":
        prediction = _single_agent_live(
            case,
            source,
            client,
        )
    else:
        # Run lexical retrieval once so an empty recall can be handled by a
        # policy-level full-context fallback.  A zero-score one-chunk result
        # is the retriever's explicit bounded fallback and is treated as empty
        # for this decision; no synthetic positive ranking score is created.
        retrieved = LexicalRetriever(
            _chunks(source), neighbor_window=0
        ).search(
            case.claim_text or case.document_path or "",
            top_k=5,
            fallback_to_full_context=False,
        )
        retrieval_empty = not retrieved or all(
            item.score == 0.0 for item in retrieved
        )
        can_use_full_context = (
            source.available
            and bool(source.content.strip())
            and decision.estimated_context_size
            <= router_config.full_context_budget_bytes
        )
        if retrieval_empty and can_use_full_context:
            prediction = _single_agent_live(
                case,
                source,
                client,
            )
            decision = AdaptiveRouteDecision(
                selected_route="full_context_single_agent",
                route_reason_code="empty_retrieval_full_context_fallback",
                chunk_count=decision.chunk_count,
                estimated_context_size=decision.estimated_context_size,
            )
        else:
            prediction = _retrieval_judge(
                case,
                source,
                baseline="retrieval_judge_live",
                client=client,
                retrieved=retrieved,
            )
    routed = _with_route_telemetry(prediction, decision)
    if routed.model_calls > 2:
        raise LocalValidationError("model_call_count_out_of_bounds")
    return routed


def run_baseline(
    case: EvalCase,
    source: SourceFixture,
    baseline: BaselineName,
    *,
    client: ModelClient | None = None,
    router_config: AdaptiveRouterConfig | None = None,
) -> EvalPrediction:
    """Run one named baseline; only live paths may call a model."""

    if baseline == "single_agent_live":
        if client is None:
            raise ValueError("single_agent_live requires a model client")
        return _single_agent_live(case, source, client)
    if baseline == "retrieval_judge_live":
        if client is None:
            raise ValueError("retrieval_judge_live requires a model client")
        return _retrieval_judge(case, source, baseline=baseline, client=client)
    if baseline == "adaptive_live":
        if client is None:
            raise ValueError("adaptive_live requires a model client")
        return _adaptive_live(
            case,
            source,
            client,
            router_config=router_config or AdaptiveRouterConfig(),
        )
    if baseline == "retrieval_judge_deterministic":
        return _retrieval_judge(case, source, baseline=baseline, client=None)
    if baseline in {"miner_judge_deterministic", "miner_judge_live"}:
        raise ValueError(
            "legacy miner_judge pair baseline names are artifact-only; "
            "use retrieval_judge_* for claim-source pairs"
        )

    if not source.available:
        evidence: tuple[Any, ...] = ()
    elif baseline == "lexical_full_source":
        evidence = (_full_source_chunk(source),)
    else:
        evidence = LexicalRetriever(_chunks(source), neighbor_window=0).search(
            case.claim_text or case.document_path or "", top_k=5
        )
    relation, confidence, span, reason, stage = _heuristic_verdict(
        case, source, evidence, baseline=baseline
    )
    return _canonical_prediction(
        case_id=case.case_id,
        baseline=baseline,
        predicted_relation=relation,
        confidence=confidence,
        reason=reason,
        predicted_evidence_span=span,
        retrieved_texts=tuple(item.text for item in evidence),
        retrieval_rank=1 if evidence else None,
        extraction_count=0,
        predicted_line=case.claim_line,
        model_calls=0,
        latency_ms=0.0,
        error_stage=stage,
    )


__all__ = [
    "build_evidence_chunks",
    "estimate_full_context_size",
    "maximum_expected_live_model_calls",
    "run_baseline",
]
