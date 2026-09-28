from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from evidencetrace.cli import app
from evidencetrace.eval.dataset import DatasetValidationError, load_dataset, validate_dataset
from evidencetrace.eval.metrics import (
    classification_metrics,
    compute_metrics,
    confusion_matrix,
    extraction_metrics,
    retrieval_metrics,
)
from evidencetrace.eval.models import EvalCase, EvalMetricsArtifact, EvalPrediction, EvalResult, EvalRunManifest
from evidencetrace.eval.runner import run_eval
from evidencetrace.models import Relation

ROOT = Path(__file__).parents[1]
DATASET = ROOT / "eval_sets/core.jsonl"


def test_seed_dataset_has_required_shape_and_annotation_split() -> None:
    dataset = load_dataset(DATASET)

    assert len(dataset.cases) == 80
    assert len(dataset.dev_cases) == 40
    assert len(dataset.test_cases) == 40
    assert sum(case.annotation_status == "deterministic_gold" for case in dataset.cases) == 60
    assert sum(case.annotation_status == "human_reviewed" for case in dataset.cases) == 0
    assert sum(case.annotation_status == "provisional" for case in dataset.cases) == 20
    assert len({case.case_hash for case in dataset.cases}) == 80
    assert all(case.gold_evidence_span is None or case.gold_evidence_span in dataset.sources[case.source_id].content for case in dataset.cases)
    assert {case.gold_relation for case in dataset.cases} == set(Relation)


def test_seed_dataset_covers_requested_mutations_and_difficult_cases() -> None:
    dataset = load_dataset(DATASET)
    mutations = {case.mutation_type for case in dataset.cases if case.mutation_type != "none"}

    assert len([case for case in dataset.cases if case.mutation_type == "none"]) == 30
    assert len([case for case in dataset.cases if case.case_id.startswith("mutation_")]) == 30
    assert len([case for case in dataset.cases if case.case_id.startswith("difficult_")]) == 20
    assert {"percentage_swap", "date_swap", "version_swap", "entity_swap", "negation_flip"} <= mutations
    assert {"comparison_flip", "scope_expansion", "qualification_drop", "not_in_source", "source_unavailable"} <= mutations


def test_duplicate_claims_and_split_leakage_are_rejected() -> None:
    dataset = load_dataset(DATASET)
    first = dataset.cases[0]
    duplicate = first.model_copy(update={"case_id": "duplicate_case", "split": "test"})

    with pytest.raises(DatasetValidationError, match="duplicate claim"):
        validate_dataset((first, duplicate), sources=dataset.sources)


def test_frozen_case_hash_detects_modified_case(tmp_path: Path) -> None:
    eval_dir = tmp_path / "eval_sets"
    shutil.copytree(ROOT / "eval_sets", eval_dir)
    dataset_path = eval_dir / "core.jsonl"
    lines = dataset_path.read_text(encoding="utf-8").splitlines()
    lines[0] = lines[0].replace("Nimbus 1.3.0", "Nimbus 1.4.0")
    dataset_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(DatasetValidationError, match="hash mismatch|frozen case"):
        load_dataset(dataset_path)


def test_metric_functions_handle_zero_denominators() -> None:
    result = classification_metrics([], [])
    assert result["macro_f1"] == 0.0
    assert result["abstention_rate"] == 0.0
    assert retrieval_metrics([], {}) == {"recall_at_5": 0.0, "mrr": 0.0, "evidence_coverage": 0.0}
    assert extraction_metrics(predicted_counts=[], gold_counts=[], predicted_lines=[], gold_lines=[])["f1"] == 0.0
    beyond_five = retrieval_metrics([["miss"] * 5 + ["gold span"]], ["gold span"])
    assert beyond_five["recall_at_5"] == 0.0
    assert retrieval_metrics([[]], [None])["recall_at_5"] == 0.0
    atomicity = extraction_metrics(predicted_counts=[2], gold_counts=[1], predicted_lines=[1], gold_lines=[1])
    assert atomicity["atomicity_error_rate"] == 1.0


def test_confusion_matrix_has_all_relation_labels() -> None:
    matrix = confusion_matrix([Relation.ENTAILED], [Relation.CONTRADICTED])

    assert set(matrix) == {relation.value for relation in Relation}
    assert matrix[Relation.ENTAILED.value][Relation.CONTRADICTED.value] == 1
    assert matrix[Relation.NOT_IN_SOURCE.value][Relation.NOT_IN_SOURCE.value] == 0


def test_baseline_inputs_and_outputs_use_same_cases(tmp_path: Path) -> None:
    output = run_eval(DATASET, tmp_path / "run", selected_split="dev")
    records = [json.loads(line) for line in output.joinpath("eval_results.jsonl").read_text().splitlines()]
    grouped = {}
    for record in records:
        grouped.setdefault(record["baseline"], {})[record["case_id"]] = record

    assert set(grouped) == {
        "lexical_rules",
        "lexical_full_source",
        "retrieval_judge_deterministic",
    }
    assert all(set(items) == set(grouped["lexical_rules"]) for items in grouped.values())
    assert all(record["source_id"] for record in grouped["lexical_rules"].values())


def test_dev_tuning_reports_provisional_cases_separately(tmp_path: Path) -> None:
    output = run_eval(DATASET, tmp_path / "run", selected_split="dev")
    metrics = json.loads(output.joinpath("metrics.json").read_text())
    headline = metrics["baselines"]["retrieval_judge_deterministic"]["headline"]
    provisional = metrics["baselines"]["retrieval_judge_deterministic"]["provisional"]

    assert headline["verification"]["count"] == 40
    assert provisional["verification"]["count"] == 10
    assert metrics["provisional_excluded_from_headline"] is True
    analysis = metrics["error_analysis"]["retrieval_judge_deterministic"]
    assert analysis["retrieval_failure_count"] + analysis["judge_failure_count"] == analysis["failed_case_count"]
    report = output.joinpath("eval_report.md").read_text()
    assert "Primary headline confusion matrix" in report
    assert "Deterministic pair baselines" in report
    assert "Live model baselines" in report
    assert "Benchmark validity: provisional" in report

def test_deterministic_eval_repeats_predictions_and_reports_cost_as_null(tmp_path: Path) -> None:
    first = run_eval(DATASET, tmp_path / "first", selected_split="dev")
    second = run_eval(DATASET, tmp_path / "second", selected_split="dev")
    first_records = [json.loads(line) for line in first.joinpath("eval_results.jsonl").read_text().splitlines()]
    second_records = [json.loads(line) for line in second.joinpath("eval_results.jsonl").read_text().splitlines()]
    fields = ("case_id", "baseline", "predicted_relation", "predicted_evidence_span", "retrieved_texts")

    assert [[record[field] for field in fields] for record in first_records] == [[record[field] for field in fields] for record in second_records]
    assert all(record["cost_usd"] is None for record in first_records)


def test_artifact_schema_and_missing_api_key_clean_skip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    output = run_eval(
        DATASET, tmp_path / "run", selected_split="dev", live_requested=True
    )
    manifest = EvalRunManifest.model_validate_json(output.joinpath("run_manifest.json").read_text())
    prediction = EvalResult.model_validate_json(output.joinpath("eval_results.jsonl").read_text().splitlines()[0])
    metrics_artifact = EvalMetricsArtifact.model_validate_json(output.joinpath("metrics.json").read_text())

    assert manifest.live_baseline == "skipped_missing_credentials"
    assert metrics_artifact.provisional_excluded_from_headline is True
    assert prediction.case_id

    assert manifest.benchmark_status == "provisional"
    assert metrics_artifact.benchmark_status == "provisional"

def test_cli_eval_success_and_failure_paths(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "eval",
            str(DATASET),
            "--out",
            str(tmp_path / "run"),
            "--split",
            "dev",
        ],
    )

    assert result.exit_code == 0, result.stdout + result.stderr
    assert "eval_report.md" in result.stdout
    assert (tmp_path / "run/metrics.json").exists()

    missing = CliRunner().invoke(app, ["eval", str(tmp_path / "missing.jsonl"), "--out", str(tmp_path / "bad")])
    assert missing.exit_code != 0


def test_compute_metrics_contains_required_engineering_and_retrieval_fields() -> None:
    result = compute_metrics(
        [
            {
                "gold_relation": "entailed",
                "predicted_relation": "entailed",
                "gold_evidence_span": "real span",
                "predicted_evidence_span": "real span",
                "retrieved_texts": ["real span"],
                "gold_line": 1,
                "predicted_line": 1,
                "extraction_count": 1,
                "confidence": 0.9,
                "latency_ms": 1.0,
                "model_calls": 0,
                "cache_hit": False,
            }
        ]
    )

    assert result["claim_extraction"]["status"] == "not_applicable"
    assert result["claim_extraction"]["precision"] is None
    assert {"recall_at_5", "mrr", "evidence_coverage"} <= set(result["retrieval"])
    assert result["engineering"]["estimated_cost_usd"] is None
    assert result["source_span_token_f1"] == 1.0
