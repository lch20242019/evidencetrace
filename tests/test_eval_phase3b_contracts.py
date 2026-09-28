from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from evidencetrace.audit_models import MINER_DRAFT_CONTRACT_VERSION
from evidencetrace.eval.baselines import run_baseline
from evidencetrace.eval.errors import LocalValidationError
from evidencetrace.eval.metrics import (
    classification_metrics,
    compute_metrics,
    engineering_metrics,
    label_coverage_gate,
)
from evidencetrace.eval.models import EvalCase, SourceFixture
from evidencetrace.eval.runner import (
    DETERMINISTIC_BASELINES,
    LIVE_BASELINES,
    run_eval,
)


def _source() -> SourceFixture:
    content = "Orchid 2.4.0 ships offline mode. It supports five regions."
    return SourceFixture(
        source_id="orchid_docs",
        url="https://docs.example.test/orchid",
        content=content,
        provenance="newly written regression fixture",
        content_hash=hashlib.sha256(content.encode()).hexdigest(),
    )


def _case(*, split: str = "dev", annotation: str = "deterministic_gold") -> EvalCase:
    return EvalCase(
        case_id=f"orchid_{split}",
        claim_text="Orchid 2.4.0 ships offline mode.",
        source_fixture="sources.jsonl",
        source_id="orchid_docs",
        source_url="https://docs.example.test/orchid",
        gold_relation="entailed",
        gold_evidence_span="Orchid 2.4.0 ships offline mode.",
        claim_type="versioned_capability",
        mutation_type="none",
        split=split,
        provenance="newly written regression fixture",
        annotation_status=annotation,
        annotation_notes="Contract fixture.",
    )


class RecordingClient:
    model_id = "recording-live"
    prompt_version = "recording-v1"

    def __init__(self) -> None:
        self.tasks: list[str] = []
        self.payloads: list[dict[str, Any]] = []

    def complete(self, task: str, payload: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError("baselines must use schema-validated complete_model")

    def complete_model(
        self, task: str, payload: dict[str, Any], schema: type[Any]
    ) -> Any:
        self.tasks.append(task)
        self.payloads.append(payload)
        if task == "single_agent_full_source_verification":
            return schema.model_validate(
                {
                    "relation": "entailed",
                    "confidence": 0.8,
                    "reason": "The exact sentence supports the claim.",
                    "evidence_span": "Orchid 2.4.0 ships offline mode.",
                }
            )
        if task == "claim_mining":
            return schema.model_validate(
                {
                    "claims": [
                        {
                            "claim_id": "c_0001",
                            "text": payload["text"],
                            "file": payload["file"],
                            "line_start": payload["line_start"],
                            "line_end": payload["line_end"],
                            "claim_type": "versioned_capability",
                            "checkability": "checkable",
                            "citation_urls": payload["citation_urls"],
                        }
                    ],
                    "model_id": self.model_id,
                    "prompt_version": self.prompt_version,
                }
            )
        evidence = payload["evidence"][0]
        return schema.model_validate(
            {
                "relation": "entailed",
                "confidence": 0.8,
                "evidence_span": evidence["text"],
                "reason": "The retrieved sentence supports the claim.",
            }
        )


def test_baseline_names_and_live_execution_paths_are_explicit() -> None:
    assert DETERMINISTIC_BASELINES == (
        "lexical_rules",
        "lexical_full_source",
        "retrieval_judge_deterministic",
    )
    assert LIVE_BASELINES == (
        "single_agent_live",
        "retrieval_judge_live",
        "adaptive_live",
    )

    single_client = RecordingClient()
    single = run_baseline(_case(), _source(), "single_agent_live", client=single_client)
    assert single.model_calls == 1
    assert single_client.tasks == ["single_agent_full_source_verification"]
    assert single_client.payloads[0]["source"]["content"] == _source().content
    assert "evidence" not in single_client.payloads[0]

    isolated_client = RecordingClient()
    isolated = run_baseline(
        _case(), _source(), "retrieval_judge_live", client=isolated_client
    )
    assert isolated.model_calls == 1
    assert isolated_client.tasks == ["claim_judgement"]
    judge_claim = isolated_client.payloads[0]["claim"]
    assert judge_claim == {
        "text": "Orchid 2.4.0 ships offline mode.",
        "claim_type": "versioned_capability",
        "slots": {},
        "checkability": "checkable",
    }
    assert isolated_client.payloads[0]["source"] == {"status": "ok"}
    assert set(isolated_client.payloads[0]["evidence"][0]) == {"text", "score"}


def test_retrieval_judge_live_short_circuits_and_never_falls_back_to_miner() -> None:
    unavailable = _source().model_copy(update={"available": False})
    client = RecordingClient()

    prediction = run_baseline(
        _case(), unavailable, "retrieval_judge_live", client=client
    )

    assert prediction.predicted_relation.value == "source_unavailable"
    assert prediction.model_calls == 0
    assert client.tasks == []

    document_only = _case().model_copy(
        update={"claim_text": None, "document_path": "docs/orchid.md"}
    )
    with pytest.raises(ValueError, match="requires claim_text"):
        run_baseline(
            document_only,
            _source(),
            "retrieval_judge_live",
            client=RecordingClient(),
        )


@pytest.mark.parametrize("legacy", ["miner_judge_deterministic", "miner_judge_live"])
def test_legacy_pair_baseline_names_are_artifact_only(legacy: str) -> None:
    with pytest.raises(ValueError, match="artifact-only"):
        run_baseline(_case(), _source(), legacy)  # type: ignore[arg-type]


def test_deterministic_baselines_never_use_passed_client() -> None:
    client = RecordingClient()
    predictions = [
        run_baseline(_case(), _source(), baseline, client=client)
        for baseline in DETERMINISTIC_BASELINES
    ]
    assert client.tasks == []
    assert all(prediction.model_calls == 0 for prediction in predictions)


def test_single_agent_rejects_invented_evidence_span() -> None:
    client = RecordingClient()

    def bad_complete(task: str, payload: dict[str, Any], schema: type[Any]) -> Any:
        return schema.model_validate(
            {
                "relation": "entailed",
                "confidence": 0.9,
                "reason": "Invented quote.",
                "evidence_span": "This text is not in the source.",
            }
        )

    client.complete_model = bad_complete  # type: ignore[method-assign]
    with pytest.raises(LocalValidationError) as caught:
        run_baseline(_case(), _source(), "single_agent_live", client=client)
    assert caught.value.code == "evidence_span_not_in_source"


def test_pair_metrics_are_not_applicable_and_taxonomy_is_explicit() -> None:
    record = {
        "gold_relation": "entailed",
        "predicted_relation": "entailed",
        "gold_evidence_span": "real span",
        "predicted_evidence_span": "real span",
        "retrieved_texts": ["real span"],
        "confidence": 0.9,
    }
    metrics = compute_metrics([record])
    extraction = metrics["claim_extraction"]
    verification = metrics["verification"]

    assert extraction["status"] == "not_applicable"
    assert extraction["precision"] is None
    assert verification["observed_label_macro_f1"] == 1.0
    assert verification["fixed_taxonomy_macro_f1"] == pytest.approx(1 / 6)
    assert verification["weighted_f1"] == 1.0
    assert verification["balanced_accuracy"] == 1.0
    assert verification["label_support"]["entailed"] == 1
    assert metrics["retrieval"]["metric_role"] == "fixture_sanity"


def test_coverage_gate_and_false_block_denominator_are_auditable() -> None:
    verification = classification_metrics(
        ["entailed", "entailed"], ["entailed", "contradicted"]
    )
    coverage = label_coverage_gate(verification)
    engineering = engineering_metrics(
        [
            {
                "gold_relation": "entailed",
                "predicted_relation": "contradicted",
            },
            {
                "gold_relation": "contradicted",
                "predicted_relation": "contradicted",
            },
        ]
    )

    assert coverage["status"] == "insufficient_coverage"
    assert coverage["achieved"] is False
    assert engineering["false_block_numerator"] == 1
    assert engineering["false_block_denominator"] == 1
    assert engineering["severity_mapping"]["contradicted"] == "error_block"


def _write_dataset(path: Path) -> Path:
    path.mkdir()
    source_path = path / "sources.jsonl"
    source_path.write_text(_source().model_dump_json() + "\n", encoding="utf-8")
    cases = (
        _case(),
        _case(split="test", annotation="provisional").model_copy(
            update={
                "case_id": "orchid_test",
                "claim_text": "Orchid supports five regions.",
                "gold_evidence_span": "It supports five regions.",
                "claim_type": "numeric_measurement",
                "annotation_notes": "Diagnostic-only contract fixture.",
            }
        ),
    )
    dataset_path = path / "cases.jsonl"
    dataset_path.write_text(
        "".join(case.model_dump_json() + "\n" for case in cases),
        encoding="utf-8",
    )
    return dataset_path


def test_runner_marks_dev_provisional_test_diagnostic_and_live_clean_skip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    dataset = _write_dataset(tmp_path / "dataset")
    dev = run_eval(
        dataset,
        tmp_path / "dev",
        selected_split="dev",
        live_requested=True,
    )
    dev_metrics = json.loads(dev.joinpath("metrics.json").read_text())
    dev_manifest = json.loads(dev.joinpath("run_manifest.json").read_text())

    assert dev_metrics["benchmark_validity"] == "provisional"
    assert (
        dev_metrics["baselines"]["retrieval_judge_deterministic"]["headline_split"]
        == "dev"
    )
    assert dev_manifest["live_baseline_statuses"] == {
        "single_agent_live": "skipped_missing_credentials",
        "retrieval_judge_live": "skipped_missing_credentials",
        "adaptive_live": "skipped_missing_credentials",
    }
    assert set(dev_metrics["baselines"]) == set(DETERMINISTIC_BASELINES)
    assert dev_manifest["miner_contract_version"] == MINER_DRAFT_CONTRACT_VERSION

    diagnostic = run_eval(dataset, tmp_path / "diagnostic")
    diagnostic_metrics = json.loads(diagnostic.joinpath("metrics.json").read_text())
    assert diagnostic_metrics["benchmark_validity"] == "diagnostic_contaminated"
    assert all(
        analysis["stop_gate"]["status"] == "insufficient_coverage"
        for analysis in diagnostic_metrics["error_analysis"].values()
    )
