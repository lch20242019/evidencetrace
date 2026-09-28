from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

import evidencetrace.eval.baselines as baselines
import evidencetrace.eval.runner as eval_runner
from evidencetrace.eval.baselines import run_baseline
from evidencetrace.eval.dataset import load_dataset
from evidencetrace.eval.errors import LOCAL_VALIDATION_CODES, LocalValidationError
from evidencetrace.eval.models import EvalCase, EvalFailureArtifact, SourceFixture
from evidencetrace.model_client import ModelCallTelemetry, ModelTelemetrySummary


def _source(*, available: bool = True) -> SourceFixture:
    content = "Aster 4.2 supports seven regions."
    return SourceFixture(
        source_id="aster_manual",
        url="https://synthetic.invalid/aster",
        content=content,
        provenance="fully synthetic Phase 4A fixture",
        content_hash=hashlib.sha256(content.encode()).hexdigest(),
        available=available,
    )


def _case(source: SourceFixture) -> EvalCase:
    return EvalCase(
        case_id="phase4a_synthetic_case",
        claim_text="Aster 4.2 supports seven regions.",
        source_fixture="sources.jsonl",
        source_id=source.source_id,
        source_url=source.url,
        gold_relation="entailed",
        gold_evidence_span=source.content,
        claim_type="versioned_capability",
        mutation_type="none",
        split="dev",
        provenance="fully synthetic Phase 4A fixture",
        annotation_status="provisional",
        annotation_notes="Synthetic post-schema branch coverage.",
    )


class SemanticClient:
    model_id = "synthetic-model"
    prompt_version = "synthetic-v1"
    provider_id = "synthetic"
    temperature = 0.0

    def __init__(
        self,
        *,
        relation: str,
        evidence_span: str | None,
        event_count: int = 1,
    ) -> None:
        self.relation = relation
        self.evidence_span = evidence_span
        self.event_count = event_count
        self.telemetry_events: tuple[ModelCallTelemetry, ...] = ()

    def complete_model(
        self,
        _task: str,
        _payload: dict[str, Any],
        schema: type[Any],
    ) -> Any:
        self.telemetry_events += tuple(
            ModelCallTelemetry(latency_ms=1.0, usage_status="missing")
            for _ in range(self.event_count)
        )
        return schema.model_validate(
            {
                "relation": self.relation,
                "confidence": 0.8,
                "reason": "Synthetic semantic decision.",
                "evidence_span": self.evidence_span,
            }
        )


def test_available_source_unavailable_relation_is_scored_as_model_quality() -> None:
    source = _source()
    prediction = run_baseline(
        _case(source),
        source,
        "single_agent_live",
        client=SemanticClient(relation="source_unavailable", evidence_span=None),
    )

    assert prediction.predicted_relation.value == "source_unavailable"
    assert prediction.model_calls == 1


def test_wrong_relation_with_grounded_span_is_a_scoreable_prediction() -> None:
    source = _source()
    prediction = run_baseline(
        _case(source),
        source,
        "single_agent_live",
        client=SemanticClient(
            relation="contradicted",
            evidence_span=source.content,
        ),
    )

    assert prediction.predicted_relation.value == "contradicted"
    assert prediction.predicted_evidence_span == source.content


@pytest.mark.parametrize(
    ("source", "client", "expected_code"),
    [
        (
            _source(),
            SemanticClient(
                relation="entailed",
                evidence_span="fabricated evidence marker",
            ),
            "evidence_span_not_in_source",
        ),
        (
            _source(available=False),
            SemanticClient(relation="not_in_source", evidence_span=None),
            "source_availability_conflict",
        ),
        (
            _source(),
            SemanticClient(
                relation="entailed",
                evidence_span="Aster 4.2 supports seven regions.",
                event_count=3,
            ),
            "model_call_count_out_of_bounds",
        ),
    ],
)
def test_post_schema_integrity_failures_are_typed_and_payload_free(
    source: SourceFixture,
    client: SemanticClient,
    expected_code: str,
) -> None:
    with pytest.raises(LocalValidationError) as caught:
        run_baseline(_case(source), source, "single_agent_live", client=client)

    assert caught.value.code == expected_code
    assert vars(caught.value) == {"code": expected_code}
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert "fabricated evidence marker" not in repr(caught.value)


def test_model_telemetry_invariant_has_a_typed_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source()
    monkeypatch.setattr(
        baselines,
        "summarize_model_telemetry",
        lambda _events: ModelTelemetrySummary(
            calls=0,
            call_latencies_ms=(),
            usage_status="not_applicable",
        ),
    )

    with pytest.raises(LocalValidationError) as caught:
        run_baseline(
            _case(source),
            source,
            "single_agent_live",
            client=SemanticClient(
                relation="entailed",
                evidence_span=source.content,
            ),
        )

    assert caught.value.code == "model_telemetry_inconsistent"


@pytest.mark.parametrize("error_type", [ValueError, AssertionError, KeyError])
def test_canonical_prediction_wraps_all_post_schema_exception_shapes(
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[Exception],
) -> None:
    marker = "private canonical payload"

    def fail_prediction(**_payload: Any) -> Any:
        raise error_type(marker)

    monkeypatch.setattr(baselines, "EvalPrediction", fail_prediction)

    with pytest.raises(LocalValidationError) as caught:
        baselines._canonical_prediction()

    assert caught.value.code == "canonical_prediction_invalid"
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert marker not in repr(caught.value)


def test_local_validation_code_allowlist_rejects_unknown_without_echo() -> None:
    marker = "claim-and-credential-marker"

    with pytest.raises(ValueError) as caught:
        LocalValidationError(marker)  # type: ignore[arg-type]

    assert marker not in str(caught.value)
    assert marker not in repr(caught.value)
    assert "evidence_span_not_in_source" in LOCAL_VALIDATION_CODES


class _NoCallClient:
    prompt_version = "synthetic-v1"
    provider_id = "synthetic"
    telemetry_events: tuple[object, ...] = ()


def _write_dataset(root: Path) -> Path:
    root.mkdir()
    source = _source()
    root.joinpath("sources.jsonl").write_text(
        source.model_dump_json() + "\n",
        encoding="utf-8",
    )
    dataset = root / "cases.jsonl"
    dataset.write_text(_case(source).model_dump_json() + "\n", encoding="utf-8")
    return dataset


def test_failure_artifact_records_only_typed_allowlisted_local_code(
    tmp_path: Path,
) -> None:
    dataset_path = _write_dataset(tmp_path / "dataset")
    output = tmp_path / "output"
    output.mkdir()
    error = LocalValidationError("evidence_span_not_in_source")
    failure_path = eval_runner._write_live_failure_artifact(
        output=output,
        dataset=load_dataset(dataset_path),
        dataset_path=dataset_path,
        selected_split="dev",
        started=datetime(2032, 4, 5, tzinfo=UTC),
        model_client=_NoCallClient(),  # type: ignore[arg-type]
        model_id="synthetic-no-call",
        temperature=0.0,
        maximum_live_calls=0,
        calls_by_baseline={
            "single_agent_live": 0,
            "retrieval_judge_live": 0,
            "adaptive_live": 0,
        },
        completed_live_predictions=0,
        failed_baseline="single_agent_live",
        failed_case_id="phase4a_synthetic_case",
        failure_stage="baseline_case",
        error=error,
    )

    serialized = failure_path.read_text(encoding="utf-8")
    artifact = EvalFailureArtifact.model_validate_json(serialized)
    assert artifact.artifact_version == "phase4a-local-validation-failure-v7"
    assert artifact.failure_kind == "local_validation"
    assert artifact.failure_code == "evidence_span_not_in_source"
    assert artifact.error_type == "LocalValidationError"
    assert error.__cause__ is None
    assert error.__context__ is None
    assert all(
        forbidden not in serialized.casefold()
        for forbidden in (
            "fabricated evidence marker",
            "authorization",
            "bearer",
            "raw_response",
            "request_headers",
        )
    )

    raw = json.loads(serialized)
    raw["failure_code"] = "unknown_local_code"
    with pytest.raises(ValidationError):
        EvalFailureArtifact.model_validate(raw)
