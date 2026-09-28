from __future__ import annotations

import json
from pathlib import Path

import pytest

from evidencetrace.eval.scifact_official import (
    SciFactOfficialProtocolError,
    compute_abstract_retrieval_metrics,
    compute_official_pipeline_metrics,
    validate_official_data,
    validate_pipeline_predictions,
)


def _data():
    corpus = [
        {"doc_id": 1, "title": "A", "abstract": ["s0", "s1", "s2"]},
        {"doc_id": 2, "title": "B", "abstract": ["t0", "t1"]},
        {"doc_id": 3, "title": "C", "abstract": ["u0"]},
    ]
    claims = [
        {
            "id": 10,
            "claim": "claim one",
            "evidence": {
                "1": [
                    {"label": "SUPPORT", "sentences": [0, 1]},
                    {"label": "SUPPORT", "sentences": [2]},
                ]
            },
            "cited_doc_ids": [1],
        },
        {
            "id": 11,
            "claim": "claim two",
            "evidence": {},
            "cited_doc_ids": [2],
        },
    ]
    return validate_official_data(corpus, claims)


def test_official_retrieval_metrics_keep_nei_and_evidence_denominators_separate():
    metrics = compute_abstract_retrieval_metrics(
        [
            {"claim_id": 10, "doc_ids": [2, 1]},
            {"claim_id": 11, "doc_ids": [2, 3]},
        ],
        _data(),
        top_k=2,
    )

    assert metrics["official_all_claims"] == {
        "claim_count": 2,
        "hit_one": 1.0,
        "hit_all": 1.0,
        "note": "Matches verisci/evaluate/abstract_retrieval.py, including NEI.",
    }
    assert metrics["evidence_claims_only"]["first_relevant_mrr_at_k"] == 0.5
    assert metrics["gold_document_recall_at_k"] == 1.0


def test_official_pipeline_metrics_enforce_complete_multi_sentence_rationale():
    metrics = compute_official_pipeline_metrics(
        [
            {
                "id": 10,
                "evidence": {
                    "1": {"label": "SUPPORT", "sentences": [0]},
                    "2": {"label": "CONTRADICT", "sentences": [0]},
                },
            },
            {"id": 11, "evidence": {}},
        ],
        _data(),
    )

    assert metrics["abstract_label_only"]["precision"] == 0.5
    assert metrics["abstract_label_only"]["recall"] == 1.0
    assert metrics["abstract_rationalized"]["f1"] == 0.0
    assert metrics["sentence_selection"]["f1"] == 0.0
    assert metrics["sentence_label"]["f1"] == 0.0


def test_official_pipeline_uses_three_sentence_default_at_abstract_level():
    metrics = compute_official_pipeline_metrics(
        [
            {
                "id": 10,
                "evidence": {
                    "1": {"label": "SUPPORT", "sentences": [0, 1, 0, 2]},
                },
            },
            {"id": 11, "evidence": {}},
        ],
        _data(),
    )

    assert metrics["abstract_rationalized"]["f1"] == 1.0
    assert metrics["sentence_selection"]["precision"] == 0.75


def test_official_pipeline_expands_abstract_cap_for_four_sentence_rationale():
    data = validate_official_data(
        [{"doc_id": 1, "title": "A", "abstract": ["s0", "s1", "s2", "s3"]}],
        [
            {
                "id": 10,
                "claim": "claim one",
                "evidence": {
                    "1": [{"label": "SUPPORT", "sentences": [0, 1, 2, 3]}]
                },
                "cited_doc_ids": [1],
            }
        ],
    )

    metrics = compute_official_pipeline_metrics(
        [
            {
                "id": 10,
                "evidence": {
                    "1": {"label": "SUPPORT", "sentences": [0, 1, 2, 3]}
                },
            }
        ],
        data,
    )

    assert metrics["abstract_rationalized"] == {
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
    }


def test_pipeline_metrics_match_frozen_leaderboard_fixture():
    fixture_dir = (
        Path(__file__).resolve().parents[2]
        / "env"
        / "reference"
        / "scifact-evaluator-66feffc5b2cc9e28e3ce3b8c9e824c3c642981eb"
        / "fixture"
    )
    if not fixture_dir.is_dir():
        pytest.skip("frozen SciFact leaderboard evaluator fixture is unavailable")

    def load_jsonl(path: Path) -> list[dict]:
        lines = path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines]

    gold_rows = load_jsonl(fixture_dir / "gold_small.jsonl")
    prediction_rows = load_jsonl(fixture_dir / "predictions_small.jsonl")
    expected = json.loads(
        (fixture_dir / "expected_metrics_small.json").read_text(encoding="utf-8")
    )

    document_ids: set[int] = set()
    maximum_sentence_index: dict[int, int] = {}
    for claim in gold_rows:
        document_ids.update(claim["cited_doc_ids"])
        for raw_doc_id, rationales in claim["evidence"].items():
            doc_id = int(raw_doc_id)
            document_ids.add(doc_id)
            for rationale in rationales:
                for sentence_index in rationale["sentences"]:
                    maximum_sentence_index[doc_id] = max(
                        maximum_sentence_index.get(doc_id, -1), sentence_index
                    )
    for prediction in prediction_rows:
        for raw_doc_id, evidence in prediction["evidence"].items():
            doc_id = int(raw_doc_id)
            document_ids.add(doc_id)
            for sentence_index in evidence["sentences"]:
                maximum_sentence_index[doc_id] = max(
                    maximum_sentence_index.get(doc_id, -1), sentence_index
                )

    corpus_rows = [
        {
            "doc_id": doc_id,
            "title": "",
            "abstract": [
                f"sentence {index}"
                for index in range(maximum_sentence_index.get(doc_id, 0) + 1)
            ],
        }
        for doc_id in sorted(document_ids)
    ]
    metrics = compute_official_pipeline_metrics(
        prediction_rows, validate_official_data(corpus_rows, gold_rows)
    )
    flattened = {
        f"{group_name}_{metric_name}": value
        for group_name, group in metrics.items()
        for metric_name, value in group.items()
    }

    assert flattened == pytest.approx(expected)


def test_missing_claim_prediction_is_rejected_before_official_scoring():
    with pytest.raises(
        SciFactOfficialProtocolError, match="prediction claim coverage is incomplete"
    ):
        validate_pipeline_predictions([{"id": 10, "evidence": {}}], _data())
