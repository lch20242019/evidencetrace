from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from pydantic import ValidationError

from evidencetrace.eval.metrics import LABELS, classification_metrics
from evidencetrace.eval.models import EvalCase, EvalRunManifest
from evidencetrace.eval.runner import (
    _benchmark_validity,
    _gate,
    _prepare_output_directory,
)
from evidencetrace.models import Relation


_SHA = hashlib.sha256(b"phase3d-validity-fixture").hexdigest()


def _case(
    case_id: str = "single_human_case",
    *,
    relation: Relation = Relation.ENTAILED,
    evidence: str | None = "The source supports this claim.",
) -> EvalCase:
    return EvalCase(
        case_id=case_id,
        claim_text=f"Unique claim for {case_id}.",
        source_fixture="sources.jsonl",
        source_id="phase3d_source",
        source_url="https://docs.example.test/phase3d",
        gold_relation=relation,
        gold_evidence_span=evidence,
        claim_type="factual_statement",
        mutation_type="none",
        split="test",
        provenance="single-human synthetic holdout regression fixture",
        annotation_status="single_human_review",
        annotation_notes="A single human reviewed this fixture.",
        source_sha256=_SHA,
        reviewer_record_sha256=_SHA,
        generation_code_sha256=_SHA,
    )


def test_single_human_annotation_requires_complete_lowercase_hashes() -> None:
    case = _case()
    assert case.annotation_status == "single_human_review"

    incomplete = case.model_dump()
    incomplete["reviewer_record_sha256"] = None
    with pytest.raises(
        ValidationError, match="single-human review requires complete provenance"
    ):
        EvalCase.model_validate(incomplete)

    payload = _case().model_dump()
    payload["source_sha256"] = _SHA.upper()
    with pytest.raises(ValidationError, match="lowercase SHA-256"):
        EvalCase.model_validate(payload)


def test_single_human_validity_is_distinct_from_human_reviewed_holdout() -> None:
    cases = tuple(_case(f"single_human_case_{index}") for index in range(6))

    assert (
        _benchmark_validity(cases, "test")
        == "single_human_synthetic_holdout"
    )


def test_all_six_labels_make_observed_and_fixed_macro_f1_identical() -> None:
    metrics = classification_metrics(LABELS, LABELS)

    assert all(metrics["label_support"][label] == 1 for label in LABELS)
    assert metrics["observed_label_macro_f1"] == 1.0
    assert metrics["fixed_taxonomy_macro_f1"] == 1.0


def test_single_human_internal_gate_requires_live_headline_and_all_labels() -> None:
    verification = classification_metrics(LABELS, LABELS)

    deterministic = _gate(
        verification,
        "single_human_synthetic_holdout",
        baseline="retrieval_judge_deterministic",
    )
    live = _gate(
        verification,
        "single_human_synthetic_holdout",
        baseline="retrieval_judge_live",
    )

    assert deterministic["status"] == "not_headline_baseline"
    assert deterministic["phase4_eligible"] is False
    assert live["status"] == "pass"
    assert live["phase4_eligible"] is True
    assert live["public_benchmark_eligible"] is False
    assert live["coverage"]["required_labels"] == list(LABELS)
    assert live["conditions"]["observed_label_macro_f1"]["threshold"] == 0.70
    assert live["technical_targets"]["relation_macro_f1"]["threshold"] == 0.75
    assert live["technical_targets"]["contradiction_recall"]["threshold"] == 0.80
    assert live["conditions"]["substantive_source_spans"]["achieved"] is True

    ungrounded = _gate(
        verification,
        "single_human_synthetic_holdout",
        baseline="retrieval_judge_live",
        substantive_spans_valid=False,
    )
    assert ungrounded["status"] == "metric_failure"
    assert ungrounded["phase4_eligible"] is False

    missing_label = classification_metrics(LABELS[:-1], LABELS[:-1])
    insufficient = _gate(
        missing_label,
        "single_human_synthetic_holdout",
        baseline="retrieval_judge_live",
    )
    assert insufficient["status"] == "insufficient_coverage"
    assert insufficient["phase4_eligible"] is False


def test_single_human_manifest_is_explicitly_not_public() -> None:
    manifest = EvalRunManifest(
        dataset_hash=_SHA,
        split_hash=_SHA,
        model_id="test-model",
        prompt_version="test-prompt",
        retrieval_config={"top_k": 5},
        python_version="3.11",
        git_commit="uncommitted",
        started_at="2026-07-14T00:00:00+00:00",
        finished_at="2026-07-14T00:00:01+00:00",
        selected_split="test",
        benchmark_status="single_human_review",
        benchmark_validity="single_human_synthetic_holdout",
        public_benchmark_eligible=False,
        baselines=("retrieval_judge_live",),
        live_baseline="executed",
    )

    assert manifest.benchmark_status == "single_human_review"
    assert manifest.benchmark_validity == "single_human_synthetic_holdout"
    assert manifest.public_benchmark_eligible is False

    invalid = manifest.model_dump()
    invalid["public_benchmark_eligible"] = True
    with pytest.raises(
        ValidationError, match="single-human synthetic holdout cannot be public"
    ):
        EvalRunManifest.model_validate(invalid)


def test_single_human_holdout_refuses_nonempty_output_directory(
    tmp_path: Path,
) -> None:
    output = tmp_path / "existing-run"
    output.mkdir()
    output.joinpath("metrics.json").write_text("already evaluated\n")

    with pytest.raises(
        ValueError, match="output directory must be new or empty"
    ):
        _prepare_output_directory(output, "single_human_synthetic_holdout")

    dev_output = tmp_path / "reusable-dev"
    dev_output.mkdir()
    dev_output.joinpath("metrics.json").write_text("dev artifact\n")
    _prepare_output_directory(dev_output, "provisional")
