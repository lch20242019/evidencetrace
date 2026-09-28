from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from evidencetrace.agents.judge import ClaimJudgeAgent

SCRIPT = Path(__file__).parents[1] / "scripts" / "run_scifact_local_production_judge.py"


@pytest.fixture()
def runner():
    spec = importlib.util.spec_from_file_location(
        "scifact_local_judge_runner_test", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pilot_selection_uses_only_claim_ids_and_is_independent_of_row_order_and_gold(
    runner,
):
    rows = [
        {"claim_id": index, "gold": index % 2, "context": {"text": str(index)}}
        for index in range(100)
    ]
    selected = runner.select_pilot(rows)
    expected_ids = sorted(
        range(100),
        key=lambda claim_id: hashlib.sha256(
            f"{runner.SCHEMA}:{claim_id}".encode("ascii")
        ).hexdigest(),
    )[:30]
    assert len(selected) == runner.PILOT_SIZE == 30
    assert [row["claim_id"] for row in selected] == expected_ids
    altered = [
        {
            "claim_id": row["claim_id"],
            "gold": "changed",
            "context": {"text": "changed"},
            "model_result": "changed",
        }
        for row in reversed(rows)
    ]
    assert [row["claim_id"] for row in runner.select_pilot(altered)] == expected_ids


def test_synthetic_probe_provenance_and_expected_labels_never_enter_judge_payload(
    runner,
):
    probes = runner.synthetic_probes()
    assert len(probes) == len({row["claim_id"] for row in probes}) == 3
    assert {row["expected_relation_for_scoring_only"] for row in probes} == {
        "entailed",
        "contradicted",
        "not_in_source",
    }
    for row in probes:
        assert row["source"] == "synthetic_diagnostic"
        assert row["outer_fold"] == -1
        assert row["canonical_context_sha256"] == runner.digest(row["context"])
        assert "expected_relation_for_scoring_only" not in row["context"]
        data = runner.build_judge_input(row, row["context"]["documents"][0])
        payload = ClaimJudgeAgent._live_payload(data)
        assert "expected_relation_for_scoring_only" not in json.dumps(payload)
        assert all(item["score"] == 0.0 for item in payload["evidence"])


def _summary_fixture():
    corpus = [
        {
            "doc_id": doc_id,
            "title": f"Paper {doc_id}",
            "abstract": [f"Sentence {index}." for index in range(size)],
        }
        for doc_id, size in ((10, 2), (20, 1), (30, 2), (40, 1))
    ]
    claims = []
    records = []
    definitions = [
        (1, {}, 10, "error", None, [0], None, False, "model_schema_error"),
        (2, {}, 10, "unsupported_relation", "not_checkable", [0], None, False, None),
        (3, {}, None, "explicit_not_in_source", "not_in_source", [], None, True, None),
        (4, {}, 10, "explicit_not_in_source", "not_in_source", [0], None, True, None),
        (
            5,
            {"10": [{"label": "SUPPORT", "sentences": [0, 1]}]},
            10,
            "ok",
            "entailed",
            [0, 1],
            "SUPPORT",
            False,
            None,
        ),
        (
            6,
            {"20": [{"label": "SUPPORT", "sentences": [0]}]},
            20,
            "unsupported_relation",
            "partially_entailed",
            [0],
            None,
            False,
            None,
        ),
        (
            7,
            {"30": [{"label": "CONTRADICT", "sentences": [0, 1]}]},
            30,
            "ok",
            "contradicted",
            [0],
            "CONTRADICT",
            False,
            None,
        ),
        (
            8,
            {"40": [{"label": "SUPPORT", "sentences": [0]}]},
            40,
            "error",
            "entailed",
            [0],
            None,
            False,
            "deterministic_conflict",
        ),
    ]
    for (
        claim_id,
        gold,
        doc_id,
        status,
        relation,
        selected,
        label,
        pure_nei,
        error,
    ) in definitions:
        claims.append(
            {
                "id": claim_id,
                "claim": f"Claim {claim_id}.",
                "evidence": gold,
                "cited_doc_ids": [int(value) for value in gold],
            }
        )
        prediction = {
            "id": claim_id,
            "evidence": {}
            if label is None
            else {str(doc_id): {"label": label, "sentences": selected}},
        }
        outcome = {
            "doc_id": doc_id,
            "status": status,
            "relation": relation,
            "error": {"category": error} if error else None,
            "evidence_groups": [
                {"sentence_offsets": [{"sentence_index": value} for value in selected]}
            ]
            if selected
            else [],
        }
        records.append(
            {
                "claim_id": claim_id,
                "status": "ok"
                if status in {"ok", "explicit_not_in_source"}
                else status,
                "pure_nei_eligible": pure_nei,
                "prediction": prediction,
                "outcomes": [outcome],
            }
        )
    return records, claims, corpus


def test_summary_keeps_failure_denominators_without_crediting_empty_errors_as_nei(
    runner,
):
    records, claims, corpus = _summary_fixture()
    summary = runner.summarize(records, claims, corpus)
    assert summary["claim_count"] == 8
    assert summary["claim_status_counts"] == {
        "ok": 4,
        "error": 2,
        "unsupported_relation": 2,
    }
    assert summary["nei_claim_count"] == 4
    assert summary["explicit_nei_correct"] == 2
    assert summary["nonempty_nei_claim_count"] == 3
    assert summary["nonempty_nei_explicit_correct"] == 1
    assert summary["error_categories"] == {
        "model_schema_error": 1,
        "deterministic_conflict": 1,
    }


def test_complete_evidence_requires_full_rationale_and_valid_successful_label(
    runner,
):
    records, claims, corpus = _summary_fixture()
    summary = runner.summarize(records, claims, corpus)
    available = summary["complete_gold_evidence_available_docs"]
    assert {row["claim_id"] for row in available} == {5, 6, 8}
    assert summary["complete_gold_evidence_available_doc_count"] == 3
    assert summary["validated_label_correct_on_complete_evidence"] == 1
    assert {row["claim_id"] for row in available if row["validated_label_correct"]} == {
        5
    }


@pytest.mark.parametrize("experimental", [False, True])
def test_cli_keeps_candidate_prompt_opt_in_without_running_models(
    runner, monkeypatch, tmp_path, experimental
):
    calls = []

    def fake_run(output, *, execute, experimental_judge_scope=False):
        calls.append((output, execute, experimental_judge_scope))
        return {"status": "cli_wiring_only"}

    output = tmp_path / "train_cli_fixture"
    argv = [str(SCRIPT), "--out", str(output)]
    if experimental:
        argv.append("--experimental-judge-scope")
    monkeypatch.setattr(runner, "run", fake_run)
    monkeypatch.setattr(runner.sys, "argv", argv)

    assert runner.main() == 0
    assert calls == [(output, False, experimental)]
