from __future__ import annotations

import hashlib
import json
import re
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from evidencetrace.eval.baselines import build_evidence_chunks
from evidencetrace.eval.full_document import (
    FULL_DOCUMENT_PROVIDER_CONTRACT_VERSION,
    FullDocumentBenchmarkError,
    FullDocumentRunArtifact,
    LoadedFullDocumentDataset,
    SingleAgentDocumentOutput,
    align_document_claims,
    compare_full_document_runs,
    compute_full_document_metrics,
    full_document_artifact_bytes,
    load_full_document_dataset,
    run_full_document_benchmark,
    safe_full_document_failure,
    write_full_document_outputs,
)
from evidencetrace.model_client import (
    DeterministicFakeModel,
    ModelCallTelemetry,
    ModelSchemaDiagnostic,
    ModelSchemaError,
    SchemaValidationIssue,
)
from evidencetrace.models import EffectiveConfig, Relation

ROOT = Path(__file__).parents[1]
PACK = ROOT / "eval_sets" / "full_document_v1"
STARTED = datetime(2034, 5, 6, 7, 8, 9, tzinfo=UTC)


@pytest.fixture(scope="module")
def dev_dataset() -> LoadedFullDocumentDataset:
    return load_full_document_dataset(PACK, split="dev")


def _subset(
    dataset: LoadedFullDocumentDataset,
    *document_ids: str,
) -> LoadedFullDocumentDataset:
    wanted = set(document_ids)
    return replace(
        dataset,
        documents=tuple(
            item for item in dataset.documents if item.document_id in wanted
        ),
        gold_claims=tuple(
            item for item in dataset.gold_claims if item.document_id in wanted
        ),
        document_text={
            key: value for key, value in dataset.document_text.items() if key in wanted
        },
    )


def _semantic(gold: Any) -> dict[str, Any]:
    return {
        "text": gold.text,
        "claim_type": gold.claim_type,
        "checkability": gold.checkability.value,
        "citation_urls": list(gold.citation_urls),
        "relation": gold.gold_relation.value,
        "confidence": 0.91,
        "reason": "Deterministic provisional benchmark response.",
        "evidence_span": gold.gold_evidence_span,
    }


def _perfect_handler(
    dataset: LoadedFullDocumentDataset,
    *,
    mutate: Any = None,
) -> Any:
    by_path = {
        document.path: dataset.gold_for(document.document_id)
        for document in dataset.documents
    }
    all_gold = dataset.gold_claims

    def find_gold(text: str) -> Any:
        return next(item for item in all_gold if item.text == text)

    def handler(task: str, payload: dict[str, Any]) -> dict[str, Any]:
        if task == "single_agent_document_verification":
            claims = [_semantic(item) for item in by_path[payload["document"]["path"]]]
            result: dict[str, Any] = {"claims": claims}
        elif task == "claim_mining":
            claims = [
                {
                    "text": re.sub(r"\s+", " ", item.text),
                    "claim_type": item.claim_type,
                    "checkability": item.checkability.value,
                }
                for item in by_path[payload["file"]]
                if payload["line_start"] <= item.line_start
                and item.line_end <= payload["line_end"]
            ]
            result = {"claims": claims}
        elif task in {
            "claim_judgement",
            "single_agent_full_source_verification",
        }:
            result = _semantic(find_gold(payload["claim"]["text"]))
            result = {
                key: result[key]
                for key in ("relation", "confidence", "reason", "evidence_span")
            }
        else:
            raise AssertionError(f"unexpected fake task: {task}")
        return mutate(task, payload, result) if mutate is not None else result

    return handler


def _run(
    dataset: LoadedFullDocumentDataset,
    baseline: str,
    *,
    mutate: Any = None,
    max_calls: int = 600,
) -> FullDocumentRunArtifact:
    return run_full_document_benchmark(
        dataset,
        baseline,  # type: ignore[arg-type]
        DeterministicFakeModel(_perfect_handler(dataset, mutate=mutate)),
        run_id=f"fixture-{baseline}",
        started_at=STARTED,
        max_calls=max_calls,
    )


class _RecoveringDocumentModel:
    def __init__(self, handler: Any) -> None:
        self.model_id = "recovering-document-fake-v1"
        self.prompt_version = "recovering-document-fake-v1"
        self.telemetry_events: tuple[ModelCallTelemetry, ...] = ()
        self.requests: list[dict[str, Any]] = []
        self._handler = handler

    def can_start_model_call(self) -> bool:
        return True

    def complete_model(
        self,
        task: str,
        payload: dict[str, Any],
        schema: type[BaseModel],
    ) -> BaseModel:
        self.requests.append(payload)
        first = len(self.requests) == 1
        self.telemetry_events += (
            ModelCallTelemetry(
                latency_ms=float(len(self.requests)),
                usage_status="reported",
                input_tokens=10,
                output_tokens=2,
                total_tokens=12,
                failure_kind="schema" if first else "none",
            ),
        )
        if first:
            raise ModelSchemaError(
                diagnostic=ModelSchemaDiagnostic(
                    schema_name="SingleAgentDocumentOutput",
                    failure_category="missing_field",
                    issues=(SchemaValidationIssue(error_type="missing"),),
                    safe_message="A required output field was missing.",
                    finish_reason="stop",
                    response_content_length=20,
                )
            ) from None
        return schema.model_validate(self._handler(task, payload))

    def complete(self, task: str, payload: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError("document benchmark must request a strict model")


def test_full_document_v1_structure_and_frozen_bundle() -> None:
    dataset = load_full_document_dataset(PACK)
    manifest = json.loads((PACK / "manifest.json").read_text(encoding="utf-8"))
    preregistration = json.loads(
        (PACK / "preregistration.json").read_text(encoding="utf-8")
    )

    assert len(dataset.documents) == 24
    assert len(dataset.gold_claims) == 120
    assert len(dataset.sources) == 72
    assert {item.split for item in dataset.documents} == {"dev", "test"}
    assert {
        stratum: sum(item.stratum == stratum for item in dataset.documents)
        for stratum in ("short", "medium", "long")
    } == {"short": 8, "medium": 8, "long": 8}
    assert {item.gold_relation for item in dataset.gold_claims} == set(Relation)
    assert manifest["blind_holdout"] is False
    assert manifest["public_benchmark_eligible"] is False
    assert manifest["rerunnable"] is True
    assert manifest["dataset_bundle_sha256"] == dataset.dataset_sha256
    freeze = preregistration["implementation_freeze"]
    assert freeze["code_bundle_sha256"] == (
        "2f63bd6ed61eaefda0cc09f3b671a7465b2f843c4f39bf7bef741a54047faf53"
    )
    assert freeze["prompt_schema_bundle_sha256"] == (
        "20070f810337e17dfc6ebff80fb73699f12f4758bbb73582b18295b8132121b0"
    )
    assert manifest["implementation_freeze"] == freeze


def test_dataset_contains_required_document_and_binding_shapes(
    dev_dataset: LoadedFullDocumentDataset,
) -> None:
    shared = dev_dataset.gold_for("fdv1_doc_001")
    multiple = dev_dataset.gold_for("fdv1_doc_002")
    cross_line = dev_dataset.gold_for("fdv1_doc_003")
    duplicate = dev_dataset.gold_for("fdv1_doc_004")

    assert shared[0].line_start == shared[1].line_start
    assert shared[0].citation_urls == shared[1].citation_urls
    assert any(len(item.citation_urls) == 2 for item in multiple)
    assert any(item.line_start != item.line_end for item in cross_line)
    assert len({item.text for item in duplicate}) < len(duplicate)
    assert any(not item.citation_urls for item in dev_dataset.gold_claims)


def test_source_strata_exercise_single_and_multi_chunk_retrieval(
    dev_dataset: LoadedFullDocumentDataset,
) -> None:
    chunk_counts = {
        document.stratum: len(
            build_evidence_chunks(
                dev_dataset.sources[
                    f"fdv1_s{int(document.document_id[-3:]):03d}_primary"
                ]
            )
        )
        for document in dev_dataset.documents[:3]
    }

    assert chunk_counts["short"] == 1
    assert 3 <= chunk_counts["medium"] <= 6
    assert 7 <= chunk_counts["long"] <= 10


def test_document_schema_contains_only_model_owned_semantic_fields() -> None:
    schema = SingleAgentDocumentOutput.model_json_schema()
    claim_schema = schema["$defs"]["LiveDocumentSemanticClaim"]

    assert schema["additionalProperties"] is False
    assert claim_schema["additionalProperties"] is False
    assert set(claim_schema["properties"]) == {
        "text",
        "claim_type",
        "checkability",
        "citation_urls",
        "relation",
        "confidence",
        "reason",
        "evidence_span",
    }
    assert set(claim_schema["required"]) == set(claim_schema["properties"])
    assert claim_schema["properties"]["reason"]["maxLength"] == 240
    for local_field in (
        "claim_id",
        "source_id",
        "locator",
        "line_start",
        "line_end",
        "offset_start",
        "offset_end",
    ):
        assert local_field not in claim_schema["properties"]


def test_single_agent_document_fake_run_is_end_to_end_perfect(
    dev_dataset: LoadedFullDocumentDataset,
) -> None:
    result = _run(dev_dataset, "single_agent_document_live")

    assert result.metrics["extraction"]["exact_f1"] == 1.0
    assert result.metrics["relation"]["fixed_taxonomy_macro_f1"] == 1.0
    assert result.metrics["end_to_end_claim_relation"]["f1"] == 1.0
    assert result.metrics["document_policy_decision_accuracy"] == 1.0
    assert result.metrics["evidence_span_f1"] == 1.0
    assert result.stage_telemetry["single_agent"].calls == 12
    assert result.stage_telemetry["miner"].calls == 0
    assert result.stage_telemetry["judge"].calls == 0
    assert result.schema_recovery.first_attempt_schema_failures == 0
    assert result.request_provenance.provider_contract_version == (
        FULL_DOCUMENT_PROVIDER_CONTRACT_VERSION
    )
    assert result.request_provenance.max_tokens_requested is None


def test_document_schema_failure_uses_one_existing_recovery_attempt(
    dev_dataset: LoadedFullDocumentDataset,
) -> None:
    subset = _subset(dev_dataset, "fdv1_doc_001")
    client = _RecoveringDocumentModel(_perfect_handler(subset))
    result = run_full_document_benchmark(
        subset,
        "single_agent_document_live",
        client,
        run_id="document-schema-recovery",
        started_at=STARTED,
    )

    assert len(client.requests) == 2
    assert "schema_recovery" not in client.requests[0]
    assert client.requests[1]["schema_recovery"]["attempt_index"] == 2
    assert result.stage_telemetry["single_agent"].calls == 2
    assert result.schema_recovery.first_attempt_schema_failures == 1
    assert result.schema_recovery.schema_retry_calls == 1
    assert result.schema_recovery.recovered_schema_failures == 1
    assert result.schema_recovery.unrecovered_schema_failures == 0
    assert len(result.schema_diagnostics) == 1
    assert result.schema_diagnostics[0].schema_name == "SingleAgentDocumentOutput"


def test_both_systems_complete_all_24_documents_with_fake_model() -> None:
    dataset = load_full_document_dataset(PACK)
    single = _run(dataset, "single_agent_document_live")
    multi = _run(dataset, "miner_adaptive_judge_live")

    assert len(single.documents) == len(multi.documents) == 24
    assert len(single.predictions) == len(multi.predictions) == 120
    assert single.metrics["end_to_end_claim_relation"]["f1"] == 1.0
    assert multi.metrics["end_to_end_claim_relation"]["f1"] == 1.0
    assert single.stage_telemetry["single_agent"].calls == 24
    assert multi.stage_telemetry["miner"].calls > 24


def test_miner_adaptive_judge_fake_run_uses_both_real_agents(
    dev_dataset: LoadedFullDocumentDataset,
) -> None:
    result = _run(dev_dataset, "miner_adaptive_judge_live")
    routes = {item.selected_route for item in result.predictions}

    assert result.metrics["extraction"]["exact_f1"] == 1.0
    assert result.metrics["relation"]["fixed_taxonomy_macro_f1"] == 1.0
    assert result.metrics["end_to_end_claim_relation"]["f1"] == 1.0
    assert result.metrics["miner_judge_handoff"]["failure_rate"] == 0.0
    assert result.stage_telemetry["miner"].calls > 12
    assert result.stage_telemetry["judge"].calls > 0
    assert {
        "deterministic_no_citation",
        "deterministic_source_unavailable",
        "deterministic_not_checkable",
        "full_context_single_agent",
        "retrieval_judge",
    } <= routes


def test_alignment_is_one_to_one_for_duplicate_claim_text(
    dev_dataset: LoadedFullDocumentDataset,
) -> None:
    subset = _subset(dev_dataset, "fdv1_doc_004")
    result = _run(subset, "single_agent_document_live")
    alignment = align_document_claims(subset.gold_claims, result.predictions)

    assert len(alignment) == 5
    assert len({item.gold_index for item in alignment}) == 5
    assert len({item.prediction_index for item in alignment}) == 5
    assert all(item.exact for item in alignment)


def test_missed_gold_claim_counts_as_relation_and_end_to_end_failure(
    dev_dataset: LoadedFullDocumentDataset,
) -> None:
    subset = _subset(dev_dataset, "fdv1_doc_001")
    result = _run(subset, "single_agent_document_live")
    predictions = result.predictions[:-1]
    document = result.documents[0].model_copy(
        update={"prediction_ids": tuple(item.prediction_id for item in predictions)}
    )
    metrics = compute_full_document_metrics(
        subset,
        predictions,
        (document,),
        config=EffectiveConfig(),
        stage_telemetry=result.stage_telemetry,
    )

    assert metrics["extraction"]["exact_recall"] == 0.8
    assert metrics["end_to_end_claim_relation"]["recall"] == 0.8
    assert metrics["relation"]["count"] == 5
    missing = sum(
        row.get("__missing__", 0)
        for row in metrics["relation"]["confusion_matrix"].values()
    )
    assert missing == 1


def test_combined_span_counts_as_atomicity_violation(
    dev_dataset: LoadedFullDocumentDataset,
) -> None:
    subset = _subset(dev_dataset, "fdv1_doc_001")
    result = _run(subset, "single_agent_document_live")
    first, second, *remaining = result.predictions
    combined_claim = first.claim.model_copy(
        update={
            "text": (
                subset.document_text["fdv1_doc_001"][
                    first.offset_start : second.offset_end
                ]
            ),
            "line_end": second.claim.line_end,
        }
    )
    combined = first.model_copy(
        update={
            "offset_end": second.offset_end,
            "claim": combined_claim,
        }
    )
    predictions = (combined, *remaining)
    document = result.documents[0].model_copy(
        update={"prediction_ids": tuple(item.prediction_id for item in predictions)}
    )
    metrics = compute_full_document_metrics(
        subset,
        predictions,
        (document,),
        config=EffectiveConfig(),
        stage_telemetry=result.stage_telemetry,
    )

    assert metrics["extraction"]["atomicity_violation_rate"] > 0.0
    assert metrics["extraction"]["exact_recall"] < 1.0


def test_quality_and_resource_comparison_is_explicit(
    dev_dataset: LoadedFullDocumentDataset,
) -> None:
    subset = _subset(dev_dataset, "fdv1_doc_001", "fdv1_doc_002")
    single = _run(subset, "single_agent_document_live")
    multi = _run(subset, "miner_adaptive_judge_live")
    delta = compare_full_document_runs(single, multi)

    assert delta["exact_claim_f1_delta"] == 0.0
    assert delta["relation_fixed_six_macro_f1_delta"] == 0.0
    assert delta["end_to_end_f1_delta"] == 0.0
    assert int(delta["model_calls_delta"]) > 0
    assert delta["total_tokens_delta"] is None


@pytest.mark.parametrize(
    ("mutation", "expected_stage", "expected_code"),
    [
        (
            lambda task, payload, result: (
                {"claims": [{**result["claims"][0], "text": "invented claim"}]}
                if task == "single_agent_document_verification"
                else result
            ),
            "single_agent",
            "claim_span_out_of_scope",
        ),
        (
            lambda task, payload, result: (
                {
                    "claims": [
                        {
                            **result["claims"][0],
                            "citation_urls": ["https://credential.invalid/private"],
                        }
                    ]
                }
                if task == "single_agent_document_verification"
                else result
            ),
            "single_agent",
            "citation_out_of_scope",
        ),
        (
            lambda task, payload, result: (
                {
                    "claims": [
                        {
                            **result["claims"][0],
                            "evidence_span": "invented source quotation",
                        }
                    ]
                }
                if task == "single_agent_document_verification"
                else result
            ),
            "single_agent",
            "evidence_span_out_of_scope",
        ),
    ],
)
def test_single_agent_scope_failures_are_payload_free(
    dev_dataset: LoadedFullDocumentDataset,
    mutation: Any,
    expected_stage: str,
    expected_code: str,
) -> None:
    subset = _subset(dev_dataset, "fdv1_doc_001")
    with pytest.raises(FullDocumentBenchmarkError) as caught:
        _run(subset, "single_agent_document_live", mutate=mutation)

    failure = safe_full_document_failure("single_agent_document_live", caught.value)
    encoded = failure.model_dump_json()
    assert failure.stage == expected_stage
    assert failure.failure_code == expected_code
    assert set(failure.model_dump(exclude_none=True)) == {
        "artifact_type",
        "artifact_version",
        "baseline",
        "stage",
        "exception_category",
        "failure_code",
    }
    assert "invented claim" not in encoded
    assert "credential.invalid" not in encoded
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_schema_and_judge_handoff_failures_are_safely_classified(
    dev_dataset: LoadedFullDocumentDataset,
) -> None:
    subset = _subset(dev_dataset, "fdv1_doc_002")

    def schema_mutation(task: str, payload: Any, result: Any) -> Any:
        return (
            {"claims": [{"text": "missing fields"}]}
            if task == ("single_agent_document_verification")
            else result
        )

    with pytest.raises(FullDocumentBenchmarkError) as schema_error:
        _run(subset, "single_agent_document_live", mutate=schema_mutation)
    schema_failure = safe_full_document_failure(
        "single_agent_document_live", schema_error.value
    )
    assert schema_failure.exception_category == "schema"
    assert schema_failure.failure_code == "model_response_invalid"

    def handoff_mutation(task: str, payload: Any, result: Any) -> Any:
        if task == "claim_judgement":
            return {
                **result,
                "relation": "entailed",
                "evidence_span": "invented source quotation",
            }
        return result

    with pytest.raises(FullDocumentBenchmarkError) as handoff_error:
        _run(subset, "miner_adaptive_judge_live", mutate=handoff_mutation)
    handoff_failure = safe_full_document_failure(
        "miner_adaptive_judge_live", handoff_error.value
    )
    assert handoff_failure.stage == "adaptive_judge"
    assert handoff_failure.exception_category == "scope"
    assert handoff_failure.failure_code == "judge_scope_violation"


def test_document_call_budget_stops_before_unbounded_loop(
    dev_dataset: LoadedFullDocumentDataset,
) -> None:
    subset = _subset(dev_dataset, "fdv1_doc_001", "fdv1_doc_002")

    with pytest.raises(FullDocumentBenchmarkError) as caught:
        _run(
            subset,
            "single_agent_document_live",
            max_calls=1,
        )

    failure = safe_full_document_failure("single_agent_document_live", caught.value)
    assert failure.exception_category == "budget"
    assert failure.failure_code == "call_budget_exhausted"


def test_artifact_and_sarif_are_byte_stable_and_bounded(
    dev_dataset: LoadedFullDocumentDataset,
    tmp_path: Path,
) -> None:
    subset = _subset(dev_dataset, "fdv1_doc_001", "fdv1_doc_002")
    first = _run(subset, "miner_adaptive_judge_live")
    second = _run(subset, "miner_adaptive_judge_live")

    assert full_document_artifact_bytes(first) == full_document_artifact_bytes(second)
    artifact_path, sarif_path = write_full_document_outputs(
        first,
        Path("results/run.json"),
        Path("results/run.sarif"),
        project_root=tmp_path,
    )
    assert hashlib.sha256(sarif_path.read_bytes()).hexdigest() == first.sarif_sha256
    assert artifact_path.read_bytes() == full_document_artifact_bytes(first)
    assert json.loads(sarif_path.read_text())["version"] == "2.1.0"
    with pytest.raises(FullDocumentBenchmarkError):
        write_full_document_outputs(
            first,
            Path("../escape.json"),
            Path("results/unused.sarif"),
            project_root=tmp_path,
        )


def test_artifacts_exclude_runtime_secret_material(
    dev_dataset: LoadedFullDocumentDataset,
) -> None:
    subset = _subset(dev_dataset, "fdv1_doc_001")
    artifact = _run(subset, "miner_adaptive_judge_live")
    encoded = full_document_artifact_bytes(artifact).decode()

    for forbidden in (
        "Authorization:",
        "Bearer ",
        "OPENAI_API_KEY",
        "request_headers",
        "raw_model_response",
        "chain_of_thought",
    ):
        assert forbidden not in encoded
