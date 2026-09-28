from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture()
def aligner():
    path = Path(__file__).parents[1] / "scripts/align_scifact_prediction_order.py"
    spec = importlib.util.spec_from_file_location("order_alignment_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _inputs(tmp_path, claims, predictions):
    claims_path = tmp_path / "claims.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    for path, rows in ((claims_path, claims), (predictions_path, predictions)):
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
    return claims_path, predictions_path


def test_alignment_changes_only_row_order_preserving_labels_citation_order_and_fields(
    aligner, tmp_path
):
    claims = [{"id": 2}, {"id": 1}, {"id": 3}]
    predictions = [
        {"id": 1, "evidence": {"10": {"label": "CONTRADICT", "sentences": [3, 1, 0]}}},
        {"id": 3, "evidence": {}, "extra_audit": {"text": "原文", "value": None}},
        {"id": 2, "evidence": {"20": {"label": "SUPPORT", "sentences": [4, 2]}}},
    ]
    claim_path, prediction_path = _inputs(tmp_path, claims, predictions)
    before = (claim_path.read_bytes(), prediction_path.read_bytes())
    result = aligner.align(claim_path, prediction_path, tmp_path / "aligned")
    ordered = [
        json.loads(line)
        for line in (result / "predictions_ordered.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert ordered == [predictions[2], predictions[0], predictions[1]]
    assert (claim_path.read_bytes(), prediction_path.read_bytes()) == before
    manifest = json.loads((result / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["original_prediction_ids"] == [1, 3, 2]
    assert manifest["target_claim_ids"] == [2, 1, 3]
    assert manifest["predictions_by_id_unchanged"] is True
    assert manifest["labels_and_citation_order_unchanged"] is True
    assert manifest["inputs"]["claims"]["sha256"] == aligner._sha(claim_path)
    assert manifest["inputs"]["predictions"]["sha256"] == aligner._sha(prediction_path)
    assert manifest["outputs_sha256"]["predictions_ordered.jsonl"] == aligner._sha(
        result / "predictions_ordered.jsonl"
    )
    with pytest.raises(aligner.PredictionOrderError, match="already exists"):
        aligner.align(claim_path, prediction_path, result)


@pytest.mark.parametrize(
    "claims,predictions",
    [
        ([{"id": 1}, {"id": 1}], [{"id": 1}]),
        ([{"id": 1}], [{"id": 1}, {"id": 1}]),
        ([{"id": 1}, {"id": 2}], [{"id": 1}]),
        ([{"id": 1}], [{"id": 1}, {"id": 2}]),
        ([{"id": 1}], [{"id": 2}]),
        ([{"id": True}], [{"id": 1}]),
        ([{"id": 1}], [{"id": "1"}]),
        ([], []),
    ],
)
def test_bad_identity_coverage_is_rejected_without_creating_output(
    aligner, tmp_path, claims, predictions
):
    claim_path, prediction_path = _inputs(tmp_path, claims, predictions)
    output = tmp_path / "aligned"
    with pytest.raises(aligner.PredictionOrderError):
        aligner.align(claim_path, prediction_path, output)
    assert not output.exists()


def test_duplicate_nested_json_fields_are_not_silently_discarded(aligner, tmp_path):
    claim_path, prediction_path = _inputs(tmp_path, [{"id": 1}], [])
    prediction_path.write_text(
        '{"id":1,"evidence":{"10":{"label":"SUPPORT","label":"CONTRADICT","sentences":[0]}}}\n',
        encoding="utf-8",
    )
    with pytest.raises(aligner.PredictionOrderError, match="duplicate JSON"):
        aligner.align(claim_path, prediction_path, tmp_path / "aligned")
