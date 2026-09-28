from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "run_scifact_qwen_oracle_gate.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("scifact_qwen_oracle_gate", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE = _load_module()


def test_selects_shortest_valid_gold_rationale_with_frozen_ties() -> None:
    corpus = [
        {"doc_id": 10, "abstract": ["long first sentence", "x", "two words"]},
        {"doc_id": 9, "abstract": ["z"]},
    ]
    claims = [
        {
            "id": 7,
            "claim": "a claim",
            "evidence": {
                "10": [
                    {"sentences": [0, 1], "label": "SUPPORT"},
                    {"sentences": [2], "label": "SUPPORT"},
                ],
                "9": [{"sentences": [0], "label": "SUPPORT"}],
            },
        }
    ]

    selected = MODULE._select_oracle_cases(corpus, claims)

    assert selected == [
        {
            "claim_id": 7,
            "claim": "a claim",
            "gold_label": "SUPPORT",
            "gold_doc_id": 9,
            "gold_sentence_ids": [0],
            "gold_group_index": 0,
            "premise": "z",
        }
    ]


def test_rejects_conflicting_relation_labels() -> None:
    corpus = [{"doc_id": 1, "abstract": ["x"]}]
    claims = [
        {
            "id": 1,
            "claim": "claim",
            "evidence": {
                "1": [
                    {"sentences": [0], "label": "SUPPORT"},
                    {"sentences": [0], "label": "CONTRADICT"},
                ]
            },
        }
    ]

    with pytest.raises(MODULE.OracleGateError, match="conflicting labels"):
        MODULE._select_oracle_cases(corpus, claims)


def test_dev_paths_are_forbidden(tmp_path: Path) -> None:
    forbidden = tmp_path / "claims_dev.jsonl"

    with pytest.raises(MODULE.OracleGateError, match="forbidden dev token"):
        MODULE._assert_not_dev(forbidden, "claims")


def test_swapped_mapping_probabilities_are_averaged_by_relation() -> None:
    results = [
        {
            "mapping": {"A": "SUPPORT", "B": "CONTRADICT"},
            "probability_a": 0.8,
            "probability_b": 0.2,
        },
        {
            "mapping": {"A": "CONTRADICT", "B": "SUPPORT"},
            "probability_a": 0.3,
            "probability_b": 0.7,
        },
    ]

    averaged = MODULE._average_swapped_probabilities(results)

    assert averaged["SUPPORT"] == pytest.approx(0.75)
    assert averaged["CONTRADICT"] == pytest.approx(0.25)


def test_gate_requires_both_recalls_and_majority_gain() -> None:
    rows = [
        *[
            {"gold_label": "SUPPORT", "predicted_label": "SUPPORT"}
            for _ in range(6)
        ],
        *[
            {"gold_label": "SUPPORT", "predicted_label": "CONTRADICT"}
            for _ in range(4)
        ],
        *[
            {"gold_label": "CONTRADICT", "predicted_label": "CONTRADICT"}
            for _ in range(6)
        ],
        *[
            {"gold_label": "CONTRADICT", "predicted_label": "SUPPORT"}
            for _ in range(4)
        ],
    ]

    gate = MODULE._evaluate_gate(rows)

    assert gate["status"] == "go"
    assert gate["qwen_oracle"]["per_class"]["SUPPORT"]["recall"] == 0.6
    assert gate["qwen_oracle"]["per_class"]["CONTRADICT"]["recall"] == 0.6
    assert gate["fixed_train_majority_baseline"]["prediction"] == "SUPPORT"


def test_gate_is_no_go_when_one_class_recall_fails() -> None:
    rows = [
        *[
            {"gold_label": "SUPPORT", "predicted_label": "SUPPORT"}
            for _ in range(7)
        ],
        *[
            {"gold_label": "SUPPORT", "predicted_label": "CONTRADICT"}
            for _ in range(3)
        ],
        *[
            {"gold_label": "CONTRADICT", "predicted_label": "CONTRADICT"}
            for _ in range(5)
        ],
        *[
            {"gold_label": "CONTRADICT", "predicted_label": "SUPPORT"}
            for _ in range(5)
        ],
    ]

    gate = MODULE._evaluate_gate(rows)

    assert gate["status"] == "no_go"
    assert gate["checks"]["contradict_recall_at_least_0_60"] is False


def test_exclusive_writer_refuses_overwrite(tmp_path: Path) -> None:
    target = tmp_path / "artifact.json"
    MODULE._write_exclusive(target, b"first")

    with pytest.raises(FileExistsError):
        MODULE._write_exclusive(target, b"second")
    assert target.read_bytes() == b"first"


def test_protocol_has_fixed_swaps_and_forbids_generation() -> None:
    assert MODULE.MAPPINGS == (
        {"mapping_id": "support_a", "A": "SUPPORT", "B": "CONTRADICT"},
        {"mapping_id": "support_b", "A": "CONTRADICT", "B": "SUPPORT"},
    )
    assert MODULE.LETTER_IDS == {"A": 32, "B": 33}
    assert MODULE.MIN_CLASS_RECALL == 0.60
    assert MODULE.MIN_MACRO_F1_GAIN_OVER_MAJORITY == 0.05
