from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

import evidencetrace.eval.runner as eval_runner
from evidencetrace.agents.judge import ClaimJudgeAgent
from evidencetrace.agents.miner import MinerScopeError
from evidencetrace.audit_models import EvidenceChunk, JudgeInput, RetrievedEvidence
from evidencetrace.checks.deterministic import (
    DETERMINISTIC_SIGNAL_POLICY_VERSION,
    DeterministicConflictError,
    bounded_evidence_text,
    deterministic_signals,
)
from evidencetrace.eval.dataset import load_dataset
from evidencetrace.eval.models import EvalFailureArtifact
from evidencetrace.model_client import DeterministicFakeModel
from evidencetrace.models import (
    AtomicClaim,
    Checkability,
    Relation,
    SourceMetadata,
)

ROOT = Path(__file__).parents[1]
SOURCE_ID = "s_aurora"
SOURCE_URL = "https://reference.example.test/aurora"
LOCATOR = "Operations > Retention"


def _claim(text: str) -> AtomicClaim:
    return AtomicClaim(
        claim_id="c_phase3e",
        text=text,
        file="docs/retention.md",
        line_start=9,
        line_end=9,
        claim_type="factual_statement",
        checkability=Checkability.CHECKABLE,
        citation_urls=(SOURCE_URL,),
    )


def _source() -> SourceMetadata:
    return SourceMetadata(
        source_id=SOURCE_ID,
        url=SOURCE_URL,
        title="AuroraMesh operations reference",
        retrieved_at=datetime(2026, 7, 21, tzinfo=UTC),
        content_hash="offline-fixture-hash",
        mime_type="text/plain",
        status="ok",
    )


def _evidence(
    text: str,
    *,
    chunk_text: str | None = None,
    locator: str = LOCATOR,
    score: float = 9.5,
) -> RetrievedEvidence:
    context = chunk_text or text
    chunk = EvidenceChunk(
        source_id=SOURCE_ID,
        url=SOURCE_URL,
        text=context,
        heading_path=("Operations", "Retention"),
        locator=locator,
        char_start=0,
        char_end=len(context),
    )
    return RetrievedEvidence(chunk=chunk, text=text, score=score)


def _live_entailed(text: str) -> dict[str, Any]:
    return {
        "relation": "entailed",
        "confidence": 0.83,
        "evidence_span": text,
        "reason": "The supplied candidate directly states the claim.",
    }


@pytest.mark.parametrize(
    "context",
    [
        (
            "AuroraMesh documents retention options. "
            "Corporate accounts may retain archives for 47 days."
        ),
        (
            "Corporate accounts may retain archives for 47 days. "
            "AuroraMesh documents retention options."
        ),
    ],
)
def test_entity_in_neighboring_bounded_context_prevents_hard_conflict(
    context: str,
) -> None:
    claim = "AuroraMesh permits corporate accounts to retain archives for 47 days."
    aligned = "Corporate accounts may retain archives for 47 days."

    codes = {
        signal.code
        for signal in deterministic_signals(
            claim,
            aligned,
            bounded_evidence_context=context,
        )
    }

    assert "entity_mismatch" not in codes


def test_explicit_different_products_keep_entity_hard_conflict() -> None:
    signals = deterministic_signals(
        "AuroraMesh signs daily release manifests.",
        "BorealisGrid signs daily release manifests.",
    )

    assert ("entity_mismatch", "error") in {
        (signal.code, signal.severity) for signal in signals
    }


def test_claim_entity_absent_from_entire_bounded_context_keeps_conflict() -> None:
    claim = "AuroraMesh permits corporate accounts to retain archives for 47 days."
    aligned = "Corporate accounts may retain archives for 47 days."
    context = (
        "BorealisGrid documents retention options. "
        "Corporate accounts may retain archives for 47 days."
    )

    codes = {
        signal.code
        for signal in deterministic_signals(
            claim,
            aligned,
            bounded_evidence_context=context,
        )
    }

    assert "entity_mismatch" in codes


def test_adjacent_restricted_candidate_counts_as_bounded_context() -> None:
    hit = _evidence("Corporate accounts may retain archives for 47 days.")
    neighbor = _evidence(
        "AuroraMesh documents retention options.",
        locator="Operations > Overview",
        score=4.0,
    )
    context = bounded_evidence_text((hit, neighbor))

    codes = {
        signal.code
        for signal in deterministic_signals(
            "AuroraMesh permits corporate accounts to retain archives for 47 days.",
            hit.text,
            bounded_evidence_context=context,
        )
    }

    assert "entity_mismatch" not in codes


def test_live_judge_accepts_entailed_when_bounded_context_names_entity() -> None:
    hit_text = "Corporate accounts may retain archives for 47 days."
    context = (
        "AuroraMesh documents retention options. "
        "Corporate accounts may retain archives for 47 days."
    )
    judge = ClaimJudgeAgent(
        DeterministicFakeModel(lambda _task, _payload: _live_entailed(hit_text))
    )

    output = judge.judge(
        JudgeInput(
            claim=_claim(
                "AuroraMesh permits corporate accounts to retain archives for 47 days."
            ),
            evidence=(_evidence(hit_text, chunk_text=context),),
            source=_source(),
        )
    )

    assert output.verdict.relation is Relation.ENTAILED
    assert output.verdict.evidence_spans[0].text == hit_text


def test_live_judge_rejects_true_cross_entity_entailed_verdict() -> None:
    source_text = "BorealisGrid signs daily release manifests."
    judge = ClaimJudgeAgent(
        DeterministicFakeModel(lambda _task, _payload: _live_entailed(source_text))
    )

    with pytest.raises(DeterministicConflictError) as caught:
        judge.judge(
            JudgeInput(
                claim=_claim("AuroraMesh signs daily release manifests."),
                evidence=(_evidence(source_text),),
                source=_source(),
            )
        )

    assert caught.value.signal_codes == ("entity_mismatch",)


@pytest.mark.parametrize(
    ("claim", "source", "expected"),
    [
        (
            "AuroraMesh completes 73% of jobs.",
            "AuroraMesh completes 63% of jobs.",
            "numeric_mismatch",
        ),
        (
            "AuroraMesh shipped on 2026-05-11.",
            "AuroraMesh shipped on 2026-05-12.",
            "date_mismatch",
        ),
        (
            "AuroraMesh v4.6 signs builds.",
            "AuroraMesh v4.5 signs builds.",
            "version_mismatch",
        ),
        (
            "AuroraMesh does not discard journals.",
            "AuroraMesh discards journals.",
            "negation_mismatch",
        ),
    ],
)
def test_non_entity_hard_conflicts_are_unchanged(
    claim: str, source: str, expected: str
) -> None:
    assert expected in {signal.code for signal in deterministic_signals(claim, source)}


def test_deterministic_conflict_error_only_carries_allowlisted_codes() -> None:
    error = DeterministicConflictError({"numeric_mismatch", "entity_mismatch"})

    assert error.signal_codes == ("entity_mismatch", "numeric_mismatch")
    assert vars(error) == {"signal_codes": error.signal_codes}
    assert "AuroraMesh" not in repr(error)

    secret_marker = "credential-value-must-not-survive"
    with pytest.raises(ValueError) as caught:
        DeterministicConflictError({secret_marker})
    assert secret_marker not in str(caught.value)
    assert secret_marker not in repr(caught.value)


class _NoCallClient:
    prompt_version = "openai-compatible-v2"
    provider_id = "openai-compatible"
    telemetry_events: tuple[object, ...] = ()


def test_future_failure_artifact_records_only_safe_guard_codes(
    tmp_path: Path,
) -> None:
    output = tmp_path / "guard-failure"
    output.mkdir()
    dataset_path = ROOT / "eval_sets/v2/dev.jsonl"
    dataset = load_dataset(dataset_path)
    failure_path = eval_runner._write_live_failure_artifact(
        output=output,
        dataset=dataset,
        dataset_path=dataset_path,
        selected_split="dev",
        started=datetime(2026, 7, 21, tzinfo=UTC),
        model_client=_NoCallClient(),  # type: ignore[arg-type]
        model_id="offline-no-call",
        temperature=0.0,
        maximum_live_calls=0,
        calls_by_baseline={
            "single_agent_live": 0,
            "miner_judge_live": 0,
        },
        completed_live_predictions=0,
        failed_baseline="miner_judge_live",
        failed_case_id="offline_guard_fixture",
        failure_stage="baseline_case",
        error=DeterministicConflictError({"entity_mismatch"}),
    )

    raw_text = failure_path.read_text(encoding="utf-8")
    artifact = EvalFailureArtifact.model_validate_json(raw_text)
    assert artifact.artifact_version == "phase3e-live-failure-v3"
    assert artifact.failure_kind == "local_validation"
    assert artifact.error_type == "DeterministicConflictError"
    assert artifact.guard_signal_codes == ("entity_mismatch",)
    assert artifact.actual_model_calls == 0
    assert DETERMINISTIC_SIGNAL_POLICY_VERSION == "deterministic-signals-v2"

    forbidden = ("AuroraMesh", "BorealisGrid", "credential", "authorization")
    assert all(value.casefold() not in raw_text.casefold() for value in forbidden)

    raw = json.loads(raw_text)
    raw["guard_signal_codes"] = ["unknown_guard"]
    with pytest.raises(ValidationError):
        EvalFailureArtifact.model_validate(raw)


def test_future_failure_artifact_records_only_safe_miner_scope_code(
    tmp_path: Path,
) -> None:
    output = tmp_path / "miner-scope-failure"
    output.mkdir()
    dataset_path = ROOT / "eval_sets/v2/dev.jsonl"
    dataset = load_dataset(dataset_path)
    failure_path = eval_runner._write_live_failure_artifact(
        output=output,
        dataset=dataset,
        dataset_path=dataset_path,
        selected_split="dev",
        started=datetime(2026, 7, 21, tzinfo=UTC),
        model_client=_NoCallClient(),  # type: ignore[arg-type]
        model_id="offline-no-call",
        temperature=0.0,
        maximum_live_calls=0,
        calls_by_baseline={
            "single_agent_live": 0,
            "retrieval_judge_live": 0,
        },
        completed_live_predictions=0,
        failed_baseline="retrieval_judge_live",
        failed_case_id="offline_miner_scope_fixture",
        failure_stage="baseline_case",
        error=MinerScopeError("ambiguous_span"),
    )

    raw_text = failure_path.read_text(encoding="utf-8")
    artifact = EvalFailureArtifact.model_validate_json(raw_text)
    assert artifact.artifact_version == "phase3e2-live-failure-v4"
    assert artifact.failure_kind == "local_validation"
    assert artifact.error_type == "MinerScopeError"
    assert artifact.miner_scope_error_code == "ambiguous_span"
    assert artifact.guard_signal_codes == ()
    assert artifact.actual_model_calls == 0

    forbidden = ("model output", "claim payload", "credential", "authorization")
    assert all(value.casefold() not in raw_text.casefold() for value in forbidden)

    raw = json.loads(raw_text)
    raw["miner_scope_error_code"] = "unknown_scope"
    with pytest.raises(ValidationError):
        EvalFailureArtifact.model_validate(raw)
    raw = json.loads(raw_text)
    raw["raw_model_content"] = "must-not-persist"
    with pytest.raises(ValidationError, match="extra_forbidden"):
        EvalFailureArtifact.model_validate(raw)
