from __future__ import annotations

import json
import shutil
from pathlib import Path

from evidencetrace.eval.stability import (
    StabilityAggregateArtifact,
    generate_stability_aggregate,
)

ROOT = Path(__file__).resolve().parents[1]
COMPLETE_RUN = ROOT / "eval_runs/phase3e2_pair_dev_run_01"
FAILED_RUN = ROOT / "eval_runs/phase3e2_pair_dev_run_02"
FROZEN_PROVENANCE = {
    "code_bundle_sha256": "a" * 64,
    "prompt_bundle_sha256": "b" * 64,
    "config_sha256": "c" * 64,
}


def _copy_run(source: Path, target: Path) -> Path:
    shutil.copytree(source, target)
    return target


def test_three_complete_runs_produce_stability_and_unstable_case_report(
    tmp_path: Path,
) -> None:
    runs = tuple(
        _copy_run(COMPLETE_RUN, tmp_path / f"run_{number}")
        for number in range(1, 4)
    )
    result_path = runs[2] / "eval_results.jsonl"
    records = [
        json.loads(line)
        for line in result_path.read_text(encoding="utf-8").splitlines()
    ]
    for record in records:
        if (
            record["baseline"] == "retrieval_judge_live"
            and record["case_id"] == "v2_dev_017"
        ):
            record["predicted_relation"] = "not_checkable"
    result_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )

    aggregate_path = generate_stability_aggregate(
        runs,
        tmp_path / "aggregate",
        frozen_provenance=FROZEN_PROVENANCE,
    )

    raw_text = aggregate_path.read_text(encoding="utf-8")
    artifact = StabilityAggregateArtifact.model_validate_json(raw_text)
    assert artifact.operational_success is True
    assert artifact.three_of_three_complete is True
    assert artifact.complete_runs == 3
    assert artifact.resource_summary["calls"] == 93
    assert artifact.failure_summary == {
        "schema": 0,
        "transport": 0,
        "guard": 0,
        "local_validation": 0,
    }
    assert artifact.baseline_macro_f1_summary[
        "retrieval_judge_live"
    ]["std_population"] == 0.0
    assert len(artifact.retrieval_judge_vs_single_agent_deltas) == 3
    assert any(
        item["baseline"] == "retrieval_judge_live"
        and item["case_id"] == "v2_dev_017"
        and item["predictions"]
        == [
            "partially_entailed",
            "partially_entailed",
            "not_checkable",
        ]
        for item in artifact.unstable_cases
    )
    assert "The evidence shows a 17% reduction" not in raw_text
    assert "predicted_evidence_span" not in raw_text
    assert (tmp_path / "aggregate/aggregate.md").is_file()


def test_failed_run_is_counted_without_reconstructing_partial_metrics(
    tmp_path: Path,
) -> None:
    runs = (
        _copy_run(COMPLETE_RUN, tmp_path / "run_1"),
        _copy_run(COMPLETE_RUN, tmp_path / "run_2"),
        _copy_run(FAILED_RUN, tmp_path / "run_3"),
    )

    aggregate_path = generate_stability_aggregate(
        runs,
        tmp_path / "aggregate",
        frozen_provenance=FROZEN_PROVENANCE,
    )

    artifact = StabilityAggregateArtifact.model_validate_json(
        aggregate_path.read_text(encoding="utf-8")
    )
    assert artifact.operational_success is False
    assert artifact.three_of_three_complete is False
    assert artifact.complete_runs == 2
    assert artifact.runs[2]["status"] == "failed"
    assert artifact.runs[2]["error_type"] == "ValueError"
    assert artifact.failure_summary["local_validation"] == 1
    assert "baseline_metrics" not in artifact.runs[2]
