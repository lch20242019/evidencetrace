from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "run_scifact_frozen_evidence_ab.py"


@pytest.fixture()
def runner():
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("scifact_frozen_ab_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _context_row(document_count=2):
    context = {
        "claim": "The treatment lowers marker levels.",
        "documents": [
            {
                "doc_id": 10 + index,
                "rank": index + 1,
                "title": f"Paper {index}",
                "sentences": [
                    {
                        "sentence_index": index,
                        "text": f"Treatment group {index} showed lower marker levels.",
                    }
                ],
            }
            for index in range(document_count)
        ],
    }
    encoded = json.dumps(
        context, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "claim_id": 7,
        "source": "oof",
        "outer_fold": 2,
        "canonical_context_sha256": hashlib.sha256(encoded).hexdigest(),
        "context": context,
        "top3_doc_ids": [10, 11, 12],
        "citation_sentence_indices": {
            str(document["doc_id"]): [
                row["sentence_index"] for row in document["sentences"]
            ]
            for document in context["documents"]
        },
    }


class _StubScorer:
    """No tokenizer, model, GPU, or external service is involved."""

    def __init__(self, probabilities_a=None):
        self.calls = []
        self.probabilities_a = probabilities_a

    def score(self, messages):
        index = len(self.calls)
        self.calls.append(messages)
        if self.probabilities_a is None:
            probability_a = 0.8 if index % 2 == 0 else 0.2
        else:
            probability_a = self.probabilities_a[index]
        encoded = json.dumps(messages, sort_keys=True).encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()
        return {
            "input_tokens": 10 + index,
            "probability_a": probability_a,
            "probability_b": 1.0 - probability_a,
            "logit_a": math.log(probability_a) if 0.0 < probability_a < 1.0 else 0.0,
            "logit_b": math.log(1.0 - probability_a)
            if 0.0 < probability_a < 1.0
            else 0.0,
            "rendered_prompt_sha256": digest,
            "input_token_ids_sha256": digest,
            "synchronized_inference_latency_ms": 2.0 + index,
            "end_to_end_latency_ms": 3.0 + index,
        }


def test_prompts_use_selected_evidence_and_distinct_review_roles_without_gold(runner):
    row = _context_row(1)
    document = row["context"]["documents"][0]
    context = {**row["context"], "gold_label": "PRIVATE_GOLD_SENTINEL"}
    mapping = {"mapping_id": "support_a", "A": "SUPPORT", "B": "CONTRADICT"}
    messages = {
        stage: runner._messages(
            context,
            document,
            stage,
            mapping,
            draft=None if stage == "initial" else "SUPPORT",
        )
        for stage in ("initial", "self_review", "independent_review")
    }
    rendered = {stage: json.dumps(value) for stage, value in messages.items()}
    assert len(set(rendered.values())) == 3
    for text in rendered.values():
        assert "PRIVATE_GOLD_SENTINEL" not in text
        assert row["context"]["claim"] in text
        assert document["sentences"][0]["text"] in text
    assert messages["self_review"] != messages["independent_review"]


def test_empty_selector_keeps_every_arm_nei_without_any_model_call(runner):
    scorer = _StubScorer()
    result = runner._run_case(_context_row(0), scorer, 0)
    assert scorer.calls == []
    assert result["document_count"] == 0
    assert result["calls"] == []
    assert result["physical_accounting"]["forward_calls"] == 0
    for arm in runner.ARMS:
        assert result["predictions"][arm] == {"id": 7, "evidence": {}}
        assert result["arm_accounting"][arm]["input_tokens"] == 0
        assert result["arm_accounting"][arm]["generated_output_tokens"] == 0


def test_two_documents_share_initial_calls_without_double_counting_physical_work(
    runner,
):
    scorer = _StubScorer()
    row = _context_row(2)
    result = runner._run_case(row, scorer, 0)
    calls = result["calls"]
    assert len(calls) == len(scorer.calls) == 12
    assert len({call["call_id"] for call in calls}) == 12
    assert result["physical_accounting"]["forward_calls"] == 12
    assert result["physical_accounting"]["input_tokens"] == sum(range(10, 22))
    assert result["physical_accounting"]["scored_next_token_positions"] == 12
    assert result["physical_accounting"]["generated_output_tokens"] == 0
    assert result["physical_accounting"]["external_api_cost_usd"] == 0.0
    assert result["physical_accounting"]["local_compute_cost_usd"] is None
    assert result["physical_accounting"]["total_cost_usd"] is None
    assert {call["canonical_context_sha256"] for call in calls} == {
        row["canonical_context_sha256"]
    }
    assert {call["doc_id"] for call in calls} == {10, 11}
    for index, call in enumerate(calls):
        assert call["messages_sha256"] == runner._digest(scorer.calls[index])
        assert call["input_tokens"] == 10 + index
        if call["stage"] == "initial":
            assert set(call["attributed_arms"]) == set(runner.ARMS)
        else:
            assert len(call["attributed_arms"]) == 1
    for arm, expected_calls in (
        ("single_one_pass", 4), ("single_self_review", 8), ("judge_challenger", 8)
    ):
        attributed = [call for call in calls if arm in call["attributed_arms"]]
        account = result["arm_accounting"][arm]
        assert account["forward_calls"] == expected_calls
        assert account["input_tokens"] == sum(
            call["input_tokens"] for call in attributed
        )
        assert account["gpu_inference_ms"] == sum(
            call["synchronized_inference_latency_ms"] for call in attributed
        )
        assert account["agent_path_ms"] == sum(
            call["end_to_end_latency_ms"] for call in attributed
        )
        assert result["predictions"][arm]["evidence"] == {
            "10": {"label": "SUPPORT", "sentences": [0]},
            "11": {"label": "SUPPORT", "sentences": [1]},
        }
    assert (
        sum(value["forward_calls"] for value in result["arm_accounting"].values()) == 20
    )


def test_swapped_mapping_averages_semantic_labels_and_both_reviews_share_draft(runner):
    scorer = _StubScorer([0.8, 0.1, 0.1, 0.9, 0.6, 0.2])
    result = runner._run_case(_context_row(1), scorer, 0)
    decisions = result["document_decisions"][0]["decisions"]
    assert decisions["initial"]["probabilities"]["SUPPORT"] == pytest.approx(0.85)
    assert decisions["self_review"]["label"] == "CONTRADICT"
    assert decisions["independent_review"]["label"] == "SUPPORT"
    assert decisions["independent_review"]["probabilities"]["SUPPORT"] == pytest.approx(
        0.70
    )
    reviews = [call for call in result["calls"] if call["stage"] != "initial"]
    assert {call["draft_label"] for call in reviews} == {"SUPPORT"}


def test_review_order_alternates_but_arm_predictions_and_budgets_stay_equal(runner):
    even = runner._run_case(_context_row(1), _StubScorer(), 0)
    odd = runner._run_case(_context_row(1), _StubScorer(), 1)
    assert even["branch_order"] == ["self_review", "independent_review"]
    assert odd["branch_order"] == ["independent_review", "self_review"]
    assert even["predictions"] == odd["predictions"]
    assert {value["forward_calls"] for value in even["arm_accounting"].values()} == {
        2,
        4,
    }
    assert {value["forward_calls"] for value in odd["arm_accounting"].values()} == {
        2,
        4,
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("probability_a", math.nan),
        ("probability_a", math.inf),
        ("probability_a", -0.1),
        ("probability_a", 1.1),
        ("probability_b", 0.7),
        ("input_tokens", 2049),
        ("input_tokens", True),
        ("input_tokens", 0),
        ("synchronized_inference_latency_ms", -1.0),
        ("end_to_end_latency_ms", 0.0),
        ("rendered_prompt_sha256", "not-a-hash"),
        ("input_token_ids_sha256", "z" * 64),
    ],
)
def test_invalid_probabilities_budgets_or_timing_fail_closed(runner, field, value):
    valid = _StubScorer().score([])
    valid[field] = value
    with pytest.raises(runner.FrozenEvidenceABError):
        runner._validate_score(valid)


def test_modified_context_is_rejected_before_model_call(runner):
    row = _context_row(1)
    row["context"]["claim"] = "Changed after freezing"
    scorer = _StubScorer()
    with pytest.raises(runner.FrozenEvidenceABError, match="context hash"):
        runner._run_case(row, scorer, 0)
    assert scorer.calls == []


def test_summary_reports_negative_challenger_effect_without_hiding_valid_comparison(
    runner, monkeypatch
):
    monkeypatch.setattr(runner, "EXPECTED_CLAIMS", 3)
    cases, claims = [], []
    # Same-role review gets all three correct; independent review misses one,
    # but predicts both labels and remains above the non-degeneracy thresholds.
    for index, (label, probabilities) in enumerate(
        (
            ("SUPPORT", [0.8, 0.2, 0.8, 0.2, 0.2, 0.8]),
            ("CONTRADICT", [0.2, 0.8, 0.2, 0.8, 0.2, 0.8]),
            ("SUPPORT", [0.8, 0.2, 0.8, 0.2, 0.8, 0.2]),
        )
    ):
        row = _context_row(1)
        row["claim_id"] = index + 1
        cases.append(runner._run_case(row, _StubScorer(probabilities), index))
        claims.append(
            {"id": index + 1, "evidence": {"10": [{"label": label, "sentences": [0]}]}}
        )
    summary = runner._summary(cases, claims)
    assert summary["challenger_minus_self_review_f1"][
        "abstract_rationalized"
    ] == pytest.approx(-1 / 3)
    assert summary["train_viability_gate"]["status"] == "pass"
    assert summary["train_viability_gate"]["positive_gain_required"] is False
    assert summary["train_viability_gate"]["dev_frozen"] is False
    assert summary["reportable_as_generalization"] is False
    assert summary["external_official_crosscheck_required"] is True
    assert summary["unique_physical_accounting"]["forward_calls"] == 18
    assert summary["accounting"]["single_self_review"]["forward_calls"] == 12
    assert summary["accounting"]["judge_challenger"]["forward_calls"] == 12
    assert summary["unique_physical_accounting"]["total_cost_usd"] is None


def test_prompt_budget_is_enforced_before_parent_model_inference(runner, monkeypatch):
    class Tokenizer:
        @staticmethod
        def apply_chat_template(*args, **kwargs):
            return "synthetic prompt"

        @staticmethod
        def encode(*args, **kwargs):
            return [0] * (runner.MAX_INPUT_TOKENS + 1)

    scorer = object.__new__(runner.BoundedQwenScorer)
    scorer.tokenizer = Tokenizer()
    monkeypatch.setattr(
        runner.oracle.QwenNextTokenScorer,
        "score",
        lambda *args, **kwargs: pytest.fail(
            "over-budget prompt must never reach model inference"
        ),
    )
    with pytest.raises(runner.FrozenEvidenceABError, match="no silent truncation"):
        scorer.score([])


def test_original_selector_citation_order_survives_prompt_sorting_and_official_cap(
    runner,
):
    row = _context_row(1)
    row["context"]["documents"][0]["sentences"] = [
        {"sentence_index": index, "text": f"Evidence sentence {index}."}
        for index in range(4)
    ]
    row["canonical_context_sha256"] = runner._digest(row["context"])
    row["citation_sentence_indices"] = {"10": [3, 1, 2, 0]}
    scorer = _StubScorer()
    result = runner._run_case(row, scorer, 0)
    gold = [{"id": 7, "evidence": {"10": [{"label": "SUPPORT", "sentences": [3]}]}}]
    for arm in runner.ARMS:
        prediction = result["predictions"][arm]
        assert prediction["evidence"]["10"]["sentences"] == [3, 1, 2, 0]
        assert (
            runner._quality(gold, [prediction])["metrics"]["abstract_rationalized"][
                "f1"
            ]
            == 1.0
        )
    for messages in scorer.calls:
        text = json.dumps(messages)
        assert text.index("Evidence sentence 0.") < text.index("Evidence sentence 3.")
        assert "citation_sentence_indices" not in text
        assert "[3, 1, 2, 0]" not in text


@pytest.mark.parametrize(
    "change",
    ["full_fit", "missing_citations", "foreign_sentence", "duplicate_sentence"],
)
def test_invalid_source_or_citation_metadata_is_rejected_before_inference(
    runner, change
):
    row = _context_row(1)
    if change == "full_fit":
        row["source"] = "full_train_fit"
    elif change == "missing_citations":
        row.pop("citation_sentence_indices")
    elif change == "foreign_sentence":
        row["citation_sentence_indices"] = {"10": [9]}
    else:
        row["citation_sentence_indices"] = {"10": [0, 0]}
    scorer = _StubScorer()
    with pytest.raises(runner.FrozenEvidenceABError):
        runner._run_case(row, scorer, 0)
    assert scorer.calls == []
