"""Apply the production ClaimJudgeAgent once to each frozen evidence document.

This adapter never retrieves, reranks, retries, or substitutes a heuristic judge.
Only exact spans returned by the real Judge become official SciFact citations.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

from pydantic import ValidationError

from evidencetrace.agents.judge import ClaimJudgeAgent, JudgeScopeError
from evidencetrace.audit_models import (
    JUDGE_SEMANTIC_CONTRACT_VERSION,
    EvidenceChunk,
    JudgeInput,
    RetrievedEvidence,
)
from evidencetrace.checks.deterministic import DeterministicConflictError
from evidencetrace.model_client import (
    ModelCallTelemetry,
    ModelClient,
    ModelSchemaError,
    ModelTransportError,
    SchemaRecoveryClient,
    summarize_model_telemetry,
)
from evidencetrace.models import AtomicClaim, Checkability, Relation, SourceMetadata

POLICY_VERSION = "frozen-production-claim-judge-v1"
_SOURCE_TIME = datetime(1970, 1, 1, tzinfo=UTC)
_SUPPORTED_LABELS = {Relation.ENTAILED: "SUPPORT", Relation.CONTRADICTED: "CONTRADICT"}


class FrozenJudgeContractError(ValueError):
    """Frozen input or single-call client accounting violates the experiment."""


class FrozenCitationError(JudgeScopeError):
    """An exact Judge span cannot identify unambiguous frozen sentences."""

    def __init__(self, citation_code: str) -> None:
        if citation_code not in {"ambiguous_citation", "citation_out_of_scope"}:
            raise ValueError("unknown frozen citation error")
        super().__init__("evidence_span_out_of_scope")
        self.citation_code = citation_code


def _digest(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_row(row: dict[str, Any]) -> None:
    if (
        row.get("source") not in {"oof", "synthetic_diagnostic"}
        or type(row.get("claim_id")) is not int
    ):
        raise FrozenJudgeContractError(
            "Judge requires frozen OOF or explicitly synthetic diagnostic identity"
        )
    context = row.get("context")
    if not isinstance(context, dict) or set(context) != {"claim", "documents"}:
        raise FrozenJudgeContractError("unexpected frozen context payload fields")
    if not isinstance(context["claim"], str) or not context["claim"].strip():
        raise FrozenJudgeContractError("frozen claim must be nonempty text")
    if _digest(context) != row.get("canonical_context_sha256"):
        raise FrozenJudgeContractError("frozen context hash mismatch")
    documents = context["documents"]
    if not isinstance(documents, list):
        raise FrozenJudgeContractError("frozen documents must be a list")
    order = row.get("citation_sentence_indices")
    if not isinstance(order, dict):
        raise FrozenJudgeContractError("original selector citation order is required")
    document_ids = []
    ranks = []
    for doc in documents:
        if set(doc) != {"doc_id", "rank", "title", "sentences"}:
            raise FrozenJudgeContractError("unexpected document fields")
        if type(doc["doc_id"]) is not int or type(doc["rank"]) is not int:
            raise FrozenJudgeContractError(
                "document identity and rank must be integers"
            )
        document_ids.append(doc["doc_id"])
        ranks.append(doc["rank"])
        sentences = doc["sentences"]
        if not isinstance(sentences, list) or not sentences:
            raise FrozenJudgeContractError(
                "selected document must contain selected sentences"
            )
        indices = []
        for sentence in sentences:
            if set(sentence) != {"sentence_index", "text"}:
                raise FrozenJudgeContractError("unexpected sentence fields")
            index, text = sentence["sentence_index"], sentence["text"]
            if (
                type(index) is not int
                or index < 0
                or not isinstance(text, str)
                or not text.strip()
            ):
                raise FrozenJudgeContractError("invalid frozen sentence")
            indices.append(index)
        citations = order.get(str(doc["doc_id"]))
        if (
            indices != sorted(set(indices))
            or not isinstance(citations, list)
            or any(type(index) is not int for index in citations)
            or len(citations) != len(set(citations))
            or set(citations) != set(indices)
        ):
            raise FrozenJudgeContractError(
                "sentence display or original citation order is invalid"
            )
    if (
        len(document_ids) != len(set(document_ids))
        or ranks != sorted(set(ranks))
        or set(order) != {str(value) for value in document_ids}
    ):
        raise FrozenJudgeContractError(
            "document coverage or ranking metadata is invalid"
        )


def _evidence_groups(document: dict[str, Any] | None) -> list[dict[str, Any]]:
    if document is None:
        return []
    runs: list[list[dict[str, Any]]] = []
    for sentence in document["sentences"]:
        if not runs or sentence["sentence_index"] != runs[-1][-1]["sentence_index"] + 1:
            runs.append([])
        runs[-1].append(sentence)
    groups = []
    for run in runs:
        text = "\n".join(sentence["text"] for sentence in run)
        offsets = []
        start = 0
        for sentence in run:
            end = start + len(sentence["text"])
            offsets.append(
                {
                    "sentence_index": sentence["sentence_index"],
                    "char_start": start,
                    "char_end": end,
                }
            )
            start = end + 1
        first, last = run[0]["sentence_index"], run[-1]["sentence_index"]
        groups.append(
            {
                "locator": f"scifact_doc_{document['doc_id']}:sentences:{first}-{last}",
                "text": text,
                "sentence_offsets": offsets,
            }
        )
    return groups


def build_judge_input(
    row: dict[str, Any], document: dict[str, Any] | None
) -> JudgeInput:
    _validate_row(row)
    documents = row["context"]["documents"]
    if (document is None and documents) or (
        document is not None and document not in documents
    ):
        raise FrozenJudgeContractError(
            "Judge source is not the supplied frozen document"
        )
    claim_id = row["claim_id"]
    source_id = (
        f"scifact_doc_{document['doc_id']}"
        if document is not None
        else f"scifact_empty_claim_{claim_id}"
    )
    url = f"https://scifact.invalid/{source_id}"
    groups = _evidence_groups(document)
    evidence = tuple(
        RetrievedEvidence(
            chunk=EvidenceChunk(
                source_id=source_id,
                url=url,
                text=group["text"],
                locator=group["locator"],
                char_start=0,
                char_end=len(group["text"]),
            ),
            text=group["text"],
            score=0.0,
        )
        for group in groups
    )
    return JudgeInput(
        claim=AtomicClaim(
            claim_id=f"scifact_claim_{claim_id}",
            text=row["context"]["claim"],
            file=f"scifact/claims/{claim_id}.md",
            line_start=1,
            line_end=1,
            claim_type="factual_claim",
            slots={},
            checkability=Checkability.CHECKABLE,
        ),
        source=SourceMetadata(
            source_id=source_id,
            url=url,
            title="",
            retrieved_at=_SOURCE_TIME,
            content_hash=_digest([group["text"] for group in groups]),
            mime_type="text/plain",
            status="ok",
        ),
        evidence=evidence,
        signals=(),
    )


def _telemetry(client: ModelClient) -> tuple[ModelCallTelemetry, ...]:
    events = getattr(client, "telemetry_events", None)
    if not isinstance(events, tuple) or any(
        not isinstance(event, ModelCallTelemetry) for event in events
    ):
        raise FrozenJudgeContractError(
            "single-call evaluation requires typed telemetry_events"
        )
    return events


def _telemetry_delta(
    before: tuple[ModelCallTelemetry, ...], after: tuple[ModelCallTelemetry, ...]
) -> tuple[ModelCallTelemetry, ...]:
    if len(after) < len(before) or after[: len(before)] != before:
        raise FrozenJudgeContractError(
            "client changed earlier telemetry during judgement"
        )
    delta = after[len(before) :]
    if len(delta) > 1 or any(
        event.schema_retry or event.attempt_index != 1 for event in delta
    ):
        raise FrozenJudgeContractError(
            "one Judge invocation exceeded the no-retry call budget"
        )
    return delta


def _mapped_citations(
    row: dict[str, Any],
    document: dict[str, Any] | None,
    input_data: JudgeInput,
    spans: Any,
) -> list[int]:
    groups = _evidence_groups(document)
    cited: set[int] = set()
    for span in spans:
        if span.source_id != input_data.source.source_id or span.locator not in {
            group["locator"] for group in groups
        }:
            raise FrozenCitationError("citation_out_of_scope")
        matches: set[tuple[int, ...]] = set()
        for group in groups:
            start = group["text"].find(span.text)
            while start >= 0:
                end = start + len(span.text)
                overlap = tuple(
                    offset["sentence_index"]
                    for offset in group["sentence_offsets"]
                    if any(
                        not character.isspace()
                        for character in group["text"][
                            max(start, offset["char_start"]) : min(
                                end, offset["char_end"]
                            )
                        ]
                    )
                )
                if overlap:
                    matches.add(overlap)
                start = group["text"].find(span.text, start + 1)
        if not matches:
            raise FrozenCitationError("citation_out_of_scope")
        if len(matches) != 1:
            raise FrozenCitationError("ambiguous_citation")
        cited.update(next(iter(matches)))
    if document is None:
        return []
    return [
        index
        for index in row["citation_sentence_indices"][str(document["doc_id"])]
        if index in cited
    ]


def _error_description(error: Exception) -> dict[str, Any]:
    if isinstance(error, FrozenCitationError):
        return {"category": "citation_error", "code": error.citation_code}
    if isinstance(error, ModelSchemaError):
        return {
            "category": "model_schema_error",
            "diagnostic": error.diagnostic.model_dump(mode="json")
            if error.diagnostic is not None
            else None,
        }
    if isinstance(error, ModelTransportError):
        return {"category": "model_transport_error"}
    if isinstance(error, JudgeScopeError):
        return {"category": "judge_scope_error", "code": error.code}
    if isinstance(error, DeterministicConflictError):
        return {
            "category": "deterministic_conflict",
            "signal_codes": list(error.signal_codes),
        }
    if isinstance(error, ValidationError):
        return {
            "category": "pydantic_validation_error",
            "error_types": sorted(
                {
                    item["type"]
                    for item in error.errors(include_input=False, include_url=False)
                }
            ),
        }
    raise error


def judge_frozen_row(row: dict[str, Any], client: ModelClient) -> dict[str, Any]:
    if client is None or isinstance(client, SchemaRecoveryClient):
        raise FrozenJudgeContractError(
            "an explicit non-retrying model client is required"
        )
    _validate_row(row)
    row_before = _telemetry(client)
    outcomes = []
    official_evidence = {}
    judge = ClaimJudgeAgent(client)
    judge_invocations = 0
    for document in row["context"]["documents"] or [None]:
        before = _telemetry(client)
        outcome: dict[str, Any] = {
            "doc_id": None if document is None else document["doc_id"],
            "status": "error",
            "relation": None,
            "judge_output": None,
            "explicit_not_in_source": False,
            "cited_sentence_indices": [],
            "error": None,
            "evidence_groups": _evidence_groups(document),
            "judge_input_sha256": None,
            "model_payload_sha256": None,
            "derived_signals": [],
            "judge_invoked": False,
        }
        input_data = None
        try:
            input_data = build_judge_input(row, document)
            outcome["judge_input_sha256"] = _digest(input_data.model_dump(mode="json"))
            outcome["judge_invoked"] = True
            judge_invocations += 1
            output = judge.judge(input_data)
            outcome["judge_output"] = output.model_dump(mode="json")
            relation = output.verdict.relation
            outcome["relation"] = relation.value
            citations = _mapped_citations(
                row, document, input_data, output.verdict.evidence_spans
            )
            outcome["cited_sentence_indices"] = citations
            if relation in _SUPPORTED_LABELS:
                if document is None or not citations:
                    raise FrozenCitationError("citation_out_of_scope")
                official_evidence[str(document["doc_id"])] = {
                    "label": _SUPPORTED_LABELS[relation],
                    "sentences": citations,
                }
                outcome["status"] = "ok"
            elif relation is Relation.NOT_IN_SOURCE:
                outcome["status"] = "explicit_not_in_source"
                outcome["explicit_not_in_source"] = True
            else:
                outcome["status"] = "unsupported_relation"
        except (
            ModelSchemaError,
            ModelTransportError,
            JudgeScopeError,
            DeterministicConflictError,
            ValidationError,
        ) as error:
            outcome["status"] = "error"
            outcome["error"] = _error_description(error)
        after = _telemetry(client)
        events = _telemetry_delta(before, after)
        if document is None and events:
            raise FrozenJudgeContractError("empty evidence must not call the model")
        live_output = (
            outcome["judge_output"] is not None
            and outcome["judge_output"]["prompt_version"]
            == JUDGE_SEMANTIC_CONTRACT_VERSION
        )
        if live_output and len(events) != 1:
            raise FrozenJudgeContractError(
                "a live Judge result requires exactly one model telemetry event"
            )
        if events and input_data is not None:
            derived_input = ClaimJudgeAgent._with_derived_signals(input_data)
            payload = ClaimJudgeAgent._live_payload(derived_input)
            outcome["model_payload_sha256"] = _digest(payload)
            outcome["derived_signals"] = [
                signal.model_dump(mode="json") for signal in derived_input.signals
            ]
        outcome["telemetry_before_count"] = len(before)
        outcome["telemetry_after_count"] = len(after)
        outcome["telemetry_delta"] = {
            "events": [event.model_dump(mode="json") for event in events],
            "summary": summarize_model_telemetry(events).model_dump(mode="json"),
        }
        outcomes.append(outcome)
    row_after = _telemetry(client)
    if row_after[: len(row_before)] != row_before:
        raise FrozenJudgeContractError("client altered pre-row telemetry")
    row_events = row_after[len(row_before) :]
    has_errors = any(outcome["status"] == "error" for outcome in outcomes)
    unsupported = any(
        outcome["status"] == "unsupported_relation" for outcome in outcomes
    )
    pure_nei_eligible = (
        not has_errors
        and not unsupported
        and all(outcome["explicit_not_in_source"] for outcome in outcomes)
    )
    return {
        "policy_version": POLICY_VERSION,
        "claim_id": row["claim_id"],
        "canonical_context_sha256": row["canonical_context_sha256"],
        "selector_source": row["source"],
        "status": "error"
        if has_errors
        else "unsupported_relation"
        if unsupported
        else "ok",
        "outcomes": outcomes,
        "prediction": {"id": row["claim_id"], "evidence": official_evidence},
        "pure_nei_eligible": pure_nei_eligible,
        "operational_success": not has_errors and not unsupported,
        "judge_invocations": judge_invocations,
        "schema_retry_enabled": False,
        "telemetry_before_count": len(row_before),
        "telemetry_after_count": len(row_after),
        "telemetry_delta": {
            "events": [event.model_dump(mode="json") for event in row_events],
            "summary": summarize_model_telemetry(row_events).model_dump(mode="json"),
        },
    }
