from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from evidencetrace.agents.judge import JudgeScopeError
from evidencetrace.checks.deterministic import DeterministicConflictError
from evidencetrace.eval import runner as eval_runner
from evidencetrace.eval.baselines import run_baseline
from evidencetrace.eval.dataset import load_dataset
from evidencetrace.eval.models import (
    EvalCase,
    EvalFailureArtifact,
    SourceFixture,
)
from evidencetrace.eval.stability import (
    JudgeSmokeArtifact,
    run_retrieval_judge_smoke,
)
from evidencetrace.model_client import DeterministicFakeModel
from evidencetrace.models import Relation

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ID = "generic_source"
SOURCE_URL = "offline://phase3e3/generic"


def _case(case_id: str, claim: str, relation: Relation) -> EvalCase:
    substantive = {
        Relation.ENTAILED,
        Relation.PARTIALLY_ENTAILED,
        Relation.CONTRADICTED,
    }
    return EvalCase(
        case_id=case_id,
        claim_text=claim,
        source_fixture="sources/generic.jsonl",
        source_id=SOURCE_ID,
        source_url=SOURCE_URL,
        gold_relation=relation,
        gold_evidence_span=(
            "Offline evidence placeholder." if relation in substantive else None
        ),
        claim_type="generic_claim",
        mutation_type="none",
        split="dev",
        provenance="offline Phase 3E.3 regression",
        annotation_status="provisional",
        annotation_notes="Offline validator regression fixture.",
    )


def _source(content: str) -> SourceFixture:
    return SourceFixture(
        source_id=SOURCE_ID,
        url=SOURCE_URL,
        content=content,
        provenance="offline Phase 3E.3 source fixture",
        content_hash="a" * 64,
    )


def _judge_response(
    _case_id: str,
    relation: Relation,
    evidence: str | None,
) -> dict[str, Any]:
    return {
        "relation": relation.value,
        "confidence": 0.8,
        "evidence_span": evidence,
        "reason": "Offline schema-valid semantic prediction.",
    }


@pytest.mark.parametrize(
    (
        "case_id",
        "claim",
        "source_text",
        "model_relation",
        "model_evidence",
    ),
    [
        (
            "subjective_best",
            "Pine Compiler is the best compiler available.",
            "Pine Compiler reduced median build time by 17%.",
            Relation.NOT_CHECKABLE,
            None,
        ),
        (
            "subjective_with_metric",
            "Pine Compiler is best because it reduced build time by 17%.",
            "Pine Compiler reduced median build time by 17%.",
            Relation.PARTIALLY_ENTAILED,
            "Pine Compiler reduced median build time by 17%.",
        ),
        (
            "legal_partial",
            "Cobalt supports durable retries across all regions.",
            "Cobalt supports durable retries in one region.",
            Relation.PARTIALLY_ENTAILED,
            "Cobalt supports durable retries in one region.",
        ),
    ],
)
def test_in_scope_semantic_relations_become_scored_predictions(
    case_id: str,
    claim: str,
    source_text: str,
    model_relation: Relation,
    model_evidence: str | None,
) -> None:
    case = _case(case_id, claim, model_relation)
    client = DeterministicFakeModel(
        lambda _task, _payload: _judge_response(
            case_id,
            model_relation,
            model_evidence,
        )
    )

    prediction = run_baseline(
        case,
        _source(source_text),
        "retrieval_judge_live",
        client=client,
    )

    assert prediction.predicted_relation is model_relation
    assert prediction.predicted_evidence_span == model_evidence
    assert prediction.model_calls == 1


def test_forged_evidence_span_remains_a_typed_fatal_scope_error() -> None:
    case_id = "forged_evidence"
    case = _case(
        case_id,
        "Cobalt completed 80% of benchmark tasks.",
        Relation.ENTAILED,
    )
    client = DeterministicFakeModel(
        lambda _task, _payload: _judge_response(
            case_id,
            Relation.ENTAILED,
            "Fabricated evidence outside the supplied candidates.",
        )
    )

    with pytest.raises(JudgeScopeError) as caught:
        run_baseline(
            case,
            _source("Cobalt completed 80% of benchmark tasks."),
            "retrieval_judge_live",
            client=client,
        )

    assert caught.value.code == "evidence_span_out_of_scope"
    assert vars(caught.value) == {"code": "evidence_span_out_of_scope"}


def test_true_fact_contradiction_guard_remains_fatal() -> None:
    case_id = "numeric_conflict"
    source_text = "Cobalt completed 62% of benchmark tasks."
    case = _case(
        case_id,
        "Cobalt completed 82% of benchmark tasks.",
        Relation.CONTRADICTED,
    )
    client = DeterministicFakeModel(
        lambda _task, _payload: _judge_response(
            case_id,
            Relation.ENTAILED,
            source_text,
        )
    )

    with pytest.raises(DeterministicConflictError) as caught:
        run_baseline(
            case,
            _source(source_text),
            "retrieval_judge_live",
            client=client,
        )

    assert caught.value.signal_codes == ("numeric_mismatch",)


def test_judge_scope_error_only_carries_an_allowlisted_code() -> None:
    error = JudgeScopeError("evidence_span_out_of_scope")

    assert vars(error) == {"code": "evidence_span_out_of_scope"}
    assert "claim" not in repr(error).casefold()

    marker = "credential-value-must-not-survive"
    with pytest.raises(ValueError) as caught:
        JudgeScopeError(marker)  # type: ignore[arg-type]
    assert marker not in str(caught.value)
    assert marker not in repr(caught.value)


class _NoCallClient:
    prompt_version = "openai-compatible-v2"
    provider_id = "openai-compatible"
    telemetry_events: tuple[object, ...] = ()


def test_failure_artifact_records_only_safe_judge_scope_code(
    tmp_path: Path,
) -> None:
    output = tmp_path / "judge-scope-failure"
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
        failed_case_id="offline_judge_scope_fixture",
        failure_stage="baseline_case",
        error=JudgeScopeError("evidence_span_out_of_scope"),
    )

    raw_text = failure_path.read_text(encoding="utf-8")
    artifact = EvalFailureArtifact.model_validate_json(raw_text)
    assert artifact.artifact_version == "phase3e3-live-failure-v5"
    assert artifact.failure_kind == "local_validation"
    assert artifact.error_type == "JudgeScopeError"
    assert artifact.judge_scope_error_code == "evidence_span_out_of_scope"
    assert artifact.actual_model_calls == 0

    forbidden = ("claim payload", "model output", "credential", "authorization")
    assert all(value.casefold() not in raw_text.casefold() for value in forbidden)

    raw = json.loads(raw_text)
    raw["judge_scope_error_code"] = "unknown_scope"
    with pytest.raises(ValidationError):
        EvalFailureArtifact.model_validate(raw)


def test_repeated_smoke_persists_relations_but_no_model_authored_text(
    tmp_path: Path,
) -> None:
    client = DeterministicFakeModel(
        lambda _task, _payload: {
            "relation": "not_checkable",
            "confidence": 0.91,
            "evidence_span": None,
            "reason": "private model-authored reason marker",
        }
    )

    path = run_retrieval_judge_smoke(
        ROOT / "eval_sets/v2/dev.jsonl",
        tmp_path / "smoke",
        case_id="v2_dev_017",
        client=client,
        frozen_provenance={
            "code_bundle_sha256": "a" * 64,
            "prompt_bundle_sha256": "b" * 64,
            "config_sha256": "c" * 64,
        },
    )

    raw_text = path.read_text(encoding="utf-8")
    artifact = JudgeSmokeArtifact.model_validate_json(raw_text)
    assert artifact.attempts_completed == 5
    assert artifact.all_canonical_predictions_valid is True
    assert artifact.resource_summary["calls"] == 5
    assert {
        prediction["predicted_relation"] for prediction in artifact.predictions
    } == {"not_checkable"}
    assert "private model-authored reason marker" not in raw_text
    assert "Pine Compiler is the best compiler available." not in raw_text
