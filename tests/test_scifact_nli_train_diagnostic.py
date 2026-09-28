from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from evidencetrace.eval.scifact_official import (
    compute_official_pipeline_metrics,
    validate_official_data,
)

SCRIPT = (
    Path(__file__).parents[1] / "scripts" / "run_scifact_nli_train_diagnostic.py"
)


def _load_module():
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location(
        "scifact_nli_train_diagnostic", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def diagnostic():
    return _load_module()


def _claims():
    return [
        {
            "id": 10,
            "claim": "A supports B.",
            "evidence": {
                "1": [{"label": "SUPPORT", "sentences": [0]}],
            },
            "cited_doc_ids": [1],
        },
        {"id": 11, "claim": "No evidence.", "evidence": {}, "cited_doc_ids": []},
    ]


def test_contract_is_fixed_train_only_and_not_generalization(diagnostic):
    assert diagnostic.EVALUATION_MODE == "exploratory_not_generalization"
    assert diagnostic.SAMPLE_POLICY == "first_100_official_train_in_file_order"
    assert diagnostic.SAMPLE_COUNT == 100
    assert diagnostic.TOP_K_DOCS == 3
    assert not hasattr(
        diagnostic._parse_args(
            [
                "--corpus",
                "corpus.jsonl",
                "--claims-train",
                "claims_train.jsonl",
                "--retrieval-train",
                "retrieval_train.jsonl",
                "--retrieval-manifest",
                "manifest.json",
                "--model",
                diagnostic.MODEL_REVISION,
                "--official-evaluator-repo",
                "evaluator",
                "--out",
                "out",
            ]
        ),
        "split",
    )


def test_dev_named_inputs_and_outputs_are_rejected(diagnostic):
    with pytest.raises(diagnostic.NliTrainDiagnosticError, match="forbidden dev"):
        diagnostic._assert_not_dev(Path("claims_dev.jsonl"), "claims")
    with pytest.raises(diagnostic.NliTrainDiagnosticError, match="forbidden dev"):
        diagnostic._assert_not_dev(Path("my-dev-run"), "output")
    diagnostic._assert_not_dev(Path("deberta-model"), "model")


def test_selection_uses_lexical_top_n_threshold_and_relation_map(diagnostic):
    ranked = [
        {
            "pair_index": 0,
            "doc_id": 1,
            "sentence_index": 2,
            "lexical_score": 0.8,
            "nli_probabilities": [0.94, 0.01, 0.05],
        },
        {
            "pair_index": 1,
            "doc_id": 2,
            "sentence_index": 1,
            "lexical_score": 0.7,
            "nli_probabilities": [0.01, 0.01, 0.98],
        },
    ]
    empty, label, selected = diagnostic._select_prediction(
        claim_id=10,
        ranked_candidates=ranked,
        lexical_top_n=1,
        threshold=0.95,
    )
    assert empty == {"id": 10, "evidence": {}}
    assert label == "NEI"
    assert selected is None

    prediction, label, selected = diagnostic._select_prediction(
        claim_id=10,
        ranked_candidates=ranked,
        lexical_top_n=2,
        threshold=0.95,
    )
    assert label == "CONTRADICT"
    assert selected == ranked[1]
    assert prediction["evidence"] == {
        "2": {"label": "CONTRADICT", "sentences": [1]}
    }


def test_independent_official_style_metrics_match_project_implementation(diagnostic):
    corpus = [
        {"doc_id": 1, "title": "A", "abstract": ["a0", "a1"]},
        {"doc_id": 2, "title": "B", "abstract": ["b0"]},
    ]
    claims = _claims()
    predictions = [
        {"id": 10, "evidence": {"1": {"label": "SUPPORT", "sentences": [0]}}},
        {"id": 11, "evidence": {}},
    ]
    official_data = validate_official_data(corpus, claims)
    expected = compute_official_pipeline_metrics(predictions, official_data)
    assert diagnostic._official_style_metrics(claims, predictions) == expected


def test_strategy_reports_nonempty_distribution_and_metrics(diagnostic):
    claims = _claims()
    candidates = [
        {
            "pair_index": 0,
            "claim_id": 10,
            "doc_id": 1,
            "sentence_index": 0,
            "lexical_score": 1.0,
            "nli_probabilities": [0.98, 0.01, 0.01],
        },
        {
            "pair_index": 1,
            "claim_id": 11,
            "doc_id": 2,
            "sentence_index": 0,
            "lexical_score": 1.0,
            "nli_probabilities": [0.1, 0.8, 0.1],
        },
    ]
    result, predictions, cases = diagnostic._strategy_result(
        claims=claims,
        candidates=candidates,
        by_claim={10: [0], 11: [1]},
        retrieval_by_claim={10: [1, 2, 3], 11: [2, 1, 3]},
        lexical_top_n=1,
        threshold=0.95,
    )
    assert result["predicted_label_distribution"] == {"SUPPORT": 1, "NEI": 1}
    assert result["nonempty_prediction_rate"] == 0.5
    assert result["claim_label_accuracy"] == 1.0
    assert result["official_style_metrics"]["sentence_label"]["f1"] == 1.0
    assert predictions[1]["evidence"] == {}
    assert cases[0]["selected_pair_index"] == 0


def test_model_identity_hashes_snapshot_and_ignores_hf_cache(
    diagnostic, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    model = tmp_path / diagnostic.MODEL_REVISION
    model.mkdir()
    config = {"id2label": {"0": "entailment", "1": "neutral", "2": "contradiction"}}
    for name, value in {
        "config.json": json.dumps(config).encode(),
        "model.safetensors": b"weights",
        "spm.model": b"spm",
        "tokenizer.json": b"{}",
        "tokenizer_config.json": b"{}",
    }.items():
        (model / name).write_bytes(value)
    cache = model / ".cache"
    cache.mkdir()
    (cache / "ignored").write_bytes(b"ignored")
    identity = diagnostic._model_identity(model)
    assert identity["revision"] == diagnostic.MODEL_REVISION
    assert ".cache/ignored" not in identity["files"]
    assert len(identity["identity_sha256"]) == 64


def test_existing_output_directory_is_rejected_before_input_reads(
    diagnostic, tmp_path: Path
):
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(diagnostic.NliTrainDiagnosticError, match="already exists"):
        diagnostic.run(
            corpus_path=tmp_path / "missing-corpus",
            claims_path=tmp_path / "missing-claims",
            retrieval_path=tmp_path / "missing-retrieval",
            retrieval_manifest_path=tmp_path / "missing-manifest",
            model_dir=tmp_path / "missing-model",
            evaluator_repo=tmp_path / "missing-evaluator",
            output=output,
            device="cpu",
        )
