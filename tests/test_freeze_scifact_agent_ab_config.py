from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

SCRIPT = (
    Path(__file__).parents[1] / "scripts" / "freeze_scifact_agent_ab_config.py"
)


def _load_module():
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("freeze_scifact_agent_ab", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def freezer():
    return _load_module()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value) + "\n" for value in values), encoding="utf-8"
    )


def _metrics(value: float) -> tuple[dict[str, float], dict[str, Any]]:
    groups = (
        "abstract_label_only",
        "abstract_rationalized",
        "sentence_selection",
        "sentence_label",
    )
    flat = {
        f"{group}_{metric}": value
        for group in groups
        for metric in ("precision", "recall", "f1")
    }
    nested = {
        group: {metric: value for metric in ("precision", "recall", "f1")}
        for group in groups
    }
    return flat, nested


def _fixture(
    freezer,
    tmp_path: Path,
    *,
    nonempty: bool = True,
    constant_label: bool = False,
    bad_format: bool = False,
    score_value: float = 0.2,
    selected_count: int | None = None,
    score_split: str = "train",
) -> dict[str, Any]:
    claim_count = 20
    corpus = tmp_path / "corpus.jsonl"
    dev_claims = tmp_path / "claims_dev.jsonl"
    retrieval = tmp_path / "retrieval.jsonl"
    retrieval_manifest = tmp_path / "retrieval_manifest.json"
    corpus.write_text('{"doc_id":1}\n', encoding="utf-8")
    dev_claims.write_text('{"id":2}\n', encoding="utf-8")
    retrieval.write_text('{"claim_id":2}\n', encoding="utf-8")
    retrieval_manifest.write_text("{}\n", encoding="utf-8")

    freezer.OFFICIAL_TRAIN_CLAIM_COUNT = claim_count
    freezer.OFFICIAL_CORPUS_COUNT = 1
    freezer.OFFICIAL_CORPUS_SHA256 = _sha(corpus)
    freezer.OFFICIAL_TRAIN_CLAIMS_SHA256 = "a" * 64
    freezer.OFFICIAL_DEV_CLAIMS_SHA256 = _sha(dev_claims)

    train_dir = tmp_path / "train"
    predictions: dict[str, list[dict[str, Any]]] = {
        arm: [] for arm in freezer.ARMS
    }
    cases = []
    for index in range(claim_count):
        evidence = {}
        if nonempty and (constant_label or index % 2 == 0):
            evidence = {"1": {"label": "SUPPORT", "sentences": [0]}}
        arms = {}
        for arm in freezer.ARMS:
            valid = not (bad_format and index == 0 and arm == "single_self_review")
            arms[arm] = {"valid": valid, "final_evidence": evidence}
            predictions[arm].append({"id": index, "evidence": evidence})
        cases.append({"claim_id": index, "arms": arms})
    _write_jsonl(train_dir / "cases.jsonl", cases)
    for arm, values in predictions.items():
        _write_jsonl(train_dir / f"predictions_{arm}.jsonl", values)

    outputs = {
        path.name: _sha(path)
        for path in train_dir.iterdir()
        if path.is_file()
    }
    train_manifest = {
        "schema_version": "scifact-agent-ab-run-v1",
        "evaluation_status": {
            "reportable": False,
            "state": "ready_for_official_scoring",
            "gold_used_by_runner": False,
        },
        "configuration_lock": {
            "agent_protocol_version": "scifact-agent-decision-v2",
            "opaque_runner_configuration": "hash-bound",
        },
        "input_lock": {
            "split": "train",
            "corpus_sha256": freezer.OFFICIAL_CORPUS_SHA256,
            "claims_sha256": freezer.OFFICIAL_TRAIN_CLAIMS_SHA256,
            "retrieval_sha256": "b" * 64,
            "retrieval_manifest_sha256": "c" * 64,
        },
        "execution": {
            "completed": True,
            "source_claim_count": claim_count,
            "selected_claim_count": (
                claim_count if selected_count is None else selected_count
            ),
            "physical_call_count": claim_count * 3,
            "expected_physical_call_count": claim_count * 3,
        },
        "outputs_sha256": outputs,
    }
    train_manifest_path = train_dir / "run_manifest.json"
    _write_json(train_manifest_path, train_manifest)

    score_directories = {}
    for arm in freezer.ARMS:
        score_dir = tmp_path / f"score_{arm}"
        official, independent = _metrics(score_value)
        _write_json(score_dir / "official_metrics.json", official)
        _write_json(score_dir / "independent_metrics.json", independent)
        (score_dir / "official_stdout.txt").write_text("", encoding="utf-8")
        (score_dir / "official_stderr.txt").write_text("", encoding="utf-8")
        score_outputs = {
            path.name: _sha(path) for path in score_dir.iterdir() if path.is_file()
        }
        score_manifest = {
            "schema_version": freezer.SCORE_SCHEMA,
            "evaluation_status": {
                "reportable": True,
                "mode": "official_leaderboard_evaluator_reproduction",
            },
            "official_evaluator": freezer.OFFICIAL_EVALUATOR,
            "data": {
                "split": score_split,
                "claim_count": claim_count,
                "corpus_count": freezer.OFFICIAL_CORPUS_COUNT,
                "corpus_sha256": freezer.OFFICIAL_CORPUS_SHA256,
                "claims_sha256": freezer.OFFICIAL_TRAIN_CLAIMS_SHA256,
            },
            "predictions": {
                "sha256": outputs[f"predictions_{arm}.jsonl"],
                "exact_claim_coverage": True,
                "known_documents_and_valid_sentence_indices": True,
            },
            "validation": {
                "official_evaluator_cross_check": True,
                "absolute_tolerance": 1e-12,
            },
            "outputs_sha256": score_outputs,
        }
        _write_json(score_dir / "run_manifest.json", score_manifest)
        score_directories[arm] = score_dir

    return {
        "train_manifest": train_manifest_path,
        "scores": score_directories,
        "corpus": corpus,
        "dev_claims": dev_claims,
        "retrieval": retrieval,
        "retrieval_manifest": retrieval_manifest,
        "output": tmp_path / "frozen.json",
    }


def _freeze(freezer, values: dict[str, Any]) -> Path:
    return freezer.freeze(
        train_manifest_path=values["train_manifest"],
        score_directories=values["scores"],
        dev_corpus_path=values["corpus"],
        dev_claims_path=values["dev_claims"],
        dev_retrieval_path=values["retrieval"],
        dev_retrieval_manifest_path=values["retrieval_manifest"],
        output=values["output"],
    )


def test_freeze_records_passing_train_only_gate(freezer, tmp_path: Path) -> None:
    values = _fixture(freezer, tmp_path)

    result = _freeze(freezer, values)

    frozen = json.loads(result.read_text(encoding="utf-8"))
    assert frozen["schema_version"] == "scifact-agent-ab-frozen-config-v2"
    binding = frozen["train_admission_gate"]
    assert binding["status"] == "passed"
    assert frozen["train_admission_gate_sha256"] == freezer._sha256_json(binding)
    report_path = Path(binding["report"]["path"])
    assert binding["report"]["sha256"] == _sha(report_path)
    gate = json.loads(report_path.read_text(encoding="utf-8"))
    assert gate["scope"] == "complete_official_train_only"
    assert gate["agent_protocol_version"] == "scifact-agent-decision-v2"
    assert gate["dev_labels_or_metrics_inspected"] is False
    for arm in freezer.ARMS:
        assert gate["observed"][arm]["format_valid_rate"] == 1.0
        assert gate["observed"][arm]["nonempty_prediction_rate"] == 0.5
        assert gate["observed"][arm]["label_counts"] == {
            "SUPPORT": 10,
            "CONTRADICT": 0,
            "NEI": 10,
        }
        assert gate["official_train_scores"][arm]["official_metrics"][
            "abstract_rationalized_f1"
        ] == 0.2


def test_freeze_rejects_all_empty_outputs(freezer, tmp_path: Path) -> None:
    values = _fixture(freezer, tmp_path, nonempty=False)

    with pytest.raises(freezer.AgentABFreezeError, match="empty-output collapse"):
        _freeze(freezer, values)

    assert not values["output"].exists()


def test_freeze_rejects_constant_label_collapse(freezer, tmp_path: Path) -> None:
    values = _fixture(freezer, tmp_path, constant_label=True)

    with pytest.raises(freezer.AgentABFreezeError, match="constant-label collapse"):
        _freeze(freezer, values)


def test_freeze_rejects_excess_format_failures(freezer, tmp_path: Path) -> None:
    values = _fixture(freezer, tmp_path, bad_format=True)

    with pytest.raises(freezer.AgentABFreezeError, match="format-valid rate"):
        _freeze(freezer, values)


def test_freeze_rejects_non_string_prediction_label_without_crashing(
    freezer, tmp_path: Path
) -> None:
    values = _fixture(freezer, tmp_path)
    train_dir = values["train_manifest"].parent
    prediction_path = train_dir / "predictions_single_self_review.jsonl"
    predictions = [
        json.loads(line) for line in prediction_path.read_text().splitlines()
    ]
    predictions[0]["evidence"]["1"]["label"] = ["SUPPORT"]
    _write_jsonl(prediction_path, predictions)
    manifest = json.loads(values["train_manifest"].read_text(encoding="utf-8"))
    manifest["outputs_sha256"][prediction_path.name] = _sha(prediction_path)
    _write_json(values["train_manifest"], manifest)

    with pytest.raises(freezer.AgentABFreezeError, match="prediction schema"):
        _freeze(freezer, values)


def test_freeze_rejects_below_floor_official_train_quality(
    freezer, tmp_path: Path
) -> None:
    values = _fixture(freezer, tmp_path, score_value=0.0)

    with pytest.raises(freezer.AgentABFreezeError, match="official train"):
        _freeze(freezer, values)


def test_freeze_rejects_train_smoke_subset(freezer, tmp_path: Path) -> None:
    values = _fixture(freezer, tmp_path, selected_count=12)

    with pytest.raises(freezer.AgentABFreezeError, match="smoke subset"):
        _freeze(freezer, values)


def test_freeze_explicitly_rejects_collapsed_v1_agent_protocol(
    freezer, tmp_path: Path
) -> None:
    values = _fixture(freezer, tmp_path)
    manifest = json.loads(values["train_manifest"].read_text(encoding="utf-8"))
    manifest["configuration_lock"].pop("agent_protocol_version")
    _write_json(values["train_manifest"], manifest)

    with pytest.raises(freezer.AgentABFreezeError, match="decision-v2"):
        _freeze(freezer, values)


def test_freeze_rejects_dev_scores(freezer, tmp_path: Path) -> None:
    values = _fixture(freezer, tmp_path, score_split="dev")

    with pytest.raises(freezer.AgentABFreezeError, match="official train"):
        _freeze(freezer, values)


def test_freeze_rejects_score_for_different_predictions(
    freezer, tmp_path: Path
) -> None:
    values = _fixture(freezer, tmp_path)
    score_manifest_path = (
        values["scores"]["single_self_review"] / "run_manifest.json"
    )
    manifest = json.loads(score_manifest_path.read_text(encoding="utf-8"))
    manifest["predictions"]["sha256"] = "d" * 64
    _write_json(score_manifest_path, manifest)

    with pytest.raises(freezer.AgentABFreezeError, match="train predictions"):
        _freeze(freezer, values)
