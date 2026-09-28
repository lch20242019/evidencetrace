from __future__ import annotations

import hashlib
import json
import runpy
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from evidencetrace.eval.dataset import (
    DatasetValidationError,
    compute_case_hash,
    load_dataset,
    validate_dataset,
)
from evidencetrace.models import Relation


ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "eval_sets" / "v2"
DATASET = V2 / "holdout_single_human.jsonl"
FREEZE = V2 / "holdout_single_human.freeze.json"
REVIEWER_SHA256 = (
    "4b4d3d8e25960d71e9b7cb151b1caec25fe9089dcd7bf1dd0a5352e13e470192"
)
GENERATOR = runpy.run_path(
    str(ROOT / "scripts" / "generate_v2_single_human_holdout.py")
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_single_human_holdout_is_complete_frozen_and_loadable() -> None:
    dataset = load_dataset(DATASET)
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))

    assert len(dataset.cases) == 60
    assert not dataset.dev_cases
    assert len(dataset.test_cases) == 60
    assert {case.case_id for case in dataset.cases} == {
        f"v2_holdout_{number:03d}" for number in range(1, 61)
    }
    assert all(
        case.annotation_status == "single_human_review" for case in dataset.cases
    )
    assert Counter(case.gold_relation.value for case in dataset.cases) == {
        "entailed": 10,
        "partially_entailed": 5,
        "contradicted": 14,
        "not_in_source": 11,
        "source_unavailable": 10,
        "not_checkable": 10,
    }
    assert {case.gold_relation for case in dataset.cases} == set(Relation)
    assert freeze["case_count"] == 60
    assert freeze["benchmark_validity"] == "single_human_synthetic_holdout"
    assert freeze["public_benchmark_eligible"] is False
    assert freeze["holdout_execution_status"] == "not_run"
    assert freeze["dataset_file_sha256"] == _sha256(DATASET)
    assert freeze["dataset_hash"] == dataset.raw_hash
    assert freeze["reviewer_record_sha256"] == REVIEWER_SHA256


def test_single_human_provenance_hashes_and_evidence_are_grounded() -> None:
    dataset = load_dataset(DATASET)
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    generator_hash = _sha256(ROOT / freeze["generation_code"])

    assert freeze["generation_code_sha256"] == generator_hash
    for case in dataset.cases:
        source = dataset.sources[case.source_id]
        assert case.source_sha256 == source.content_hash
        assert case.reviewer_record_sha256 == REVIEWER_SHA256
        assert case.generation_code_sha256 == generator_hash
        assert case.case_hash == compute_case_hash(case)
        if case.gold_evidence_span is not None:
            assert case.gold_evidence_span in source.content


def test_model_proposal_fields_cannot_enter_generated_gold() -> None:
    build_cases = GENERATOR["build_cases"]
    source_content = "The source supports the human annotation."
    source_hash = hashlib.sha256(source_content.encode()).hexdigest()
    source = {
        "source_id": "source_one",
        "url": "offline://test/source-one",
        "content": source_content,
        "provenance": "synthetic test fixture",
        "content_hash": source_hash,
        "available": True,
    }
    candidates: list[dict[str, Any]] = []
    annotations: dict[str, dict[str, str | None]] = {}
    for number in range(1, 61):
        case_id = f"v2_holdout_{number:03d}"
        claim = f"Human claim {number}."
        candidates.append(
            {
                "candidate_id": case_id,
                "claim": claim,
                "source_id": "source_one",
                "source_text": source_content,
                "source_sha256": source_hash,
                "proposed_relation": "contradicted",
                "model_notes": "must not be copied",
                "human_relation": "",
            }
        )
        annotations[case_id] = {
            "claim": claim,
            "relation": "entailed",
            "evidence": source_content,
            "notes": "Independent human note.",
        }

    cases = build_cases(
        candidates,
        annotations,
        {"source_one": source},
        reviewer_sha256="a" * 64,
        generation_code_sha256="b" * 64,
    )

    assert len(cases) == 60
    assert all(case["gold_relation"] == "entailed" for case in cases)
    assert all(
        case["annotation_notes"]
        == "Frozen single-human annotation; private reviewer notes are excluded."
        for case in cases
    )
    assert all("proposed_relation" not in case for case in cases)
    assert all("model_notes" not in case for case in cases)
    assert all("human_relation" not in case for case in cases)


def test_single_human_dataset_contains_no_private_identity_or_model_fields() -> None:
    rows = [
        json.loads(line)
        for line in DATASET.read_text(encoding="utf-8").splitlines()
    ]
    serialized = DATASET.read_text(encoding="utf-8")

    assert "EvidenceTrace Project Owner" not in serialized
    assert "evidencetrace-human-a-001" not in serialized
    assert "proposed_relation" not in serialized
    assert "model_note" not in serialized
    assert all(
        not (
            {"reviewer_name", "reviewer_id", "proposed_relation", "model_notes"}
            & row.keys()
        )
        for row in rows
    )


def test_physically_separate_split_is_valid_but_empty_dataset_is_not() -> None:
    dataset = load_dataset(DATASET)
    case = dataset.cases[0]

    assert validate_dataset((case,), sources=dataset.sources)[0].split == "test"
    wrong_source_hash = case.model_copy(
        update={"source_sha256": "0" * 64, "case_hash": None}
    )
    with pytest.raises(DatasetValidationError, match="source_sha256"):
        validate_dataset((wrong_source_hash,), sources=dataset.sources)
    with pytest.raises(DatasetValidationError, match="at least one case"):
        validate_dataset(())


def test_v2_frozen_input_and_output_hashes_do_not_drift() -> None:
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))

    assert freeze["candidate_input_sha256"] == (
        "e5f9777caf0e32df7d70ae279ec26548dc354aca8ed41ccdd0f60fd8681367ce"
    )
    assert freeze["source_fixture_sha256"] == (
        "2c81609ea654053ff6cc191a3a60a95b10a69082da5862e98dee9f6a20e0eed6"
    )
    assert freeze["dataset_file_sha256"] == (
        "0d19004b627f4ecabf9dd40fd28303b0539b89f36f3f7f5cec135c47a6582c75"
    )
    assert freeze["dataset_hash"] == (
        "b5c09264eae626a541a0a79454699e561b336cc1cb7c9606a7fe3dc6885233cd"
    )
