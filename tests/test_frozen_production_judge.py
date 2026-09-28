from __future__ import annotations

import hashlib
import importlib
import json

import httpx
import pytest

from evidencetrace.agents.judge import ClaimJudgeAgent
from evidencetrace.audit_models import EVIDENCE_EXACT_MAX_CHARS
from evidencetrace.model_client import OpenAICompatibleClient, SchemaRecoveryClient

PRIVATE_MARKER = "private-invalid-provider-response"
PRIVATE_KEY = "private-unit-test-api-key"
CLAIM_TEXT = "The intervention improved survival."
SUPPORT_TEXT = "The intervention improved survival."
CONTRADICT_TEXT = "The intervention did not improve survival."


@pytest.fixture()
def adapter():
    return importlib.import_module("evidencetrace.eval.frozen_judge")


def _row(document_count=1):
    documents = [
        {
            "doc_id": 10,
            "rank": 1,
            "title": "Intervention outcomes",
            "sentences": [
                {"sentence_index": 5, "text": SUPPORT_TEXT},
                {
                    "sentence_index": 9,
                    "text": "Follow-up used blinded outcome assessment.",
                },
            ],
        },
        {
            "doc_id": 20,
            "rank": 2,
            "title": "Replication outcomes",
            "sentences": [{"sentence_index": 3, "text": CONTRADICT_TEXT}],
        },
    ][:document_count]
    context = {"claim": CLAIM_TEXT, "documents": documents}
    encoded = json.dumps(
        context, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    citations = {"10": [9, 5], "20": [3]}
    return {
        "claim_id": 7,
        "source": "oof",
        "outer_fold": 2,
        "canonical_context_sha256": hashlib.sha256(encoded).hexdigest(),
        "context": context,
        "top3_doc_ids": [10, 20, 30],
        "citation_sentence_indices": {
            str(row["doc_id"]): citations[str(row["doc_id"])] for row in documents
        },
    }


def _refresh_context_hash(row):
    row["canonical_context_sha256"] = hashlib.sha256(
        json.dumps(
            row["context"], ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _semantic(relation="entailed", evidence_span=SUPPORT_TEXT, **extra):
    return {
        "relation": relation,
        "confidence": 0.85,
        "reason": "The supplied sentence establishes this relation.",
        "evidence_span": evidence_span,
        **extra,
    }


def _client(responses, *, status_code=200):
    captured = []
    remaining = iter(responses)

    def handler(request):
        captured.append(json.loads(request.content))
        content = next(remaining)
        return httpx.Response(
            status_code,
            request=request,
            json={
                "choices": [
                    {
                        "message": {"content": json.dumps(content)},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 31,
                    "completion_tokens": 13,
                    "total_tokens": 44,
                },
            },
        )

    client = OpenAICompatibleClient(
        model_id="frozen-production-judge-unit-model",
        api_key=PRIVATE_KEY,
        base_url="https://provider.invalid/v1",
        max_calls=8,
        transport=httpx.MockTransport(handler),
    )
    return client, captured


def _model_inputs(captured):
    return [
        json.loads(request["messages"][1]["content"])["input"] for request in captured
    ]


def test_judge_input_preserves_original_sentences_and_uses_neutral_scores(
    adapter,
):
    row = _row(1)
    result = adapter.build_judge_input(row, row["context"]["documents"][0])
    assert result.claim.text == CLAIM_TEXT
    assert result.claim.slots == {}
    assert [item.text for item in result.evidence] == [
        SUPPORT_TEXT,
        "Follow-up used blinded outcome assessment.",
    ]
    assert all(item.score == 0.0 for item in result.evidence)
    assert all(item.chunk.text == item.text for item in result.evidence)
    assert len({item.chunk.locator for item in result.evidence}) == 2
    assert all(
        item.chunk.source_id == result.source.source_id for item in result.evidence
    )
    assert result.source.status == "ok"
    payload = ClaimJudgeAgent._live_payload(result)
    assert payload["claim"]["checkability"] == "checkable"
    assert all(set(item) == {"text", "score"} for item in payload["evidence"])
    assert "gold" not in json.dumps(payload).casefold()
    assert "nli" not in json.dumps(payload).casefold()


def test_actual_judge_calls_once_per_document_and_maps_only_its_cited_sentence(
    adapter, monkeypatch
):
    client, captured = _client(
        [
            _semantic(evidence_span="improved survival"),
            _semantic("contradicted", CONTRADICT_TEXT),
        ]
    )
    judge_inputs = []
    actual_judge = ClaimJudgeAgent.judge

    def record_real_judge(self, input_data):
        judge_inputs.append(input_data)
        return actual_judge(self, input_data)

    monkeypatch.setattr(ClaimJudgeAgent, "judge", record_real_judge)
    row = _row(2)
    row["gold_label"] = PRIVATE_MARKER
    row["selector_score"] = 0.97
    row["nli_probability"] = 0.94
    result = adapter.judge_frozen_row(row, client)
    assert result["status"] == "ok"
    assert result["prediction"] == {
        "id": 7,
        "evidence": {
            "10": {"label": "SUPPORT", "sentences": [5]},
            "20": {"label": "CONTRADICT", "sentences": [3]},
        },
    }
    assert len(captured) == len(client.telemetry_events) == len(judge_inputs) == 2
    assert len({value.source.source_id for value in judge_inputs}) == 2
    assert all(event.failure_kind == "none" for event in client.telemetry_events)
    assert sum(event.input_tokens for event in client.telemetry_events) == 62
    assert sum(event.output_tokens for event in client.telemetry_events) == 26
    assert len(result["telemetry_delta"]["events"]) == 2
    assert result["judge_invocations"] == 2
    assert result["pure_nei_eligible"] is False
    for request, payload in zip(captured, _model_inputs(captured), strict=True):
        envelope = json.loads(request["messages"][1]["content"])
        assert envelope["task"] == "claim_judgement"
        assert envelope["output_contract"]["schema_name"] == "LiveJudgeSemanticOutput"
        assert all(item["score"] == 0.0 for item in payload["evidence"])
        encoded = json.dumps(payload)
        for forbidden in (
            PRIVATE_MARKER,
            "selector_score",
            "nli_probability",
            "gold_label",
            "claim_id",
            "source_id",
        ):
            assert forbidden not in encoded
    for outcome, payload, input_data in zip(
        result["outcomes"], _model_inputs(captured), judge_inputs, strict=True
    ):
        assert outcome["model_payload_sha256"] == adapter._digest(payload)
        assert outcome["judge_input_sha256"] == adapter._digest(
            input_data.model_dump(mode="json")
        )
        assert len(outcome["telemetry_delta"]["events"]) == 1
    assert PRIVATE_KEY not in json.dumps(result)


def test_empty_context_uses_actual_judge_without_a_model_call(adapter, monkeypatch):
    client, captured = _client([])
    invoked = []
    actual_judge = ClaimJudgeAgent.judge

    def record_real_judge(self, input_data):
        invoked.append(input_data)
        return actual_judge(self, input_data)

    monkeypatch.setattr(ClaimJudgeAgent, "judge", record_real_judge)
    result = adapter.judge_frozen_row(_row(0), client)
    assert len(invoked) == 1
    assert invoked[0].evidence == ()
    assert captured == []
    assert client.telemetry_events == ()
    assert result["status"] == "ok"
    assert result["prediction"] == {"id": 7, "evidence": {}}
    assert result["pure_nei_eligible"] is True


def test_explicit_not_in_source_is_the_only_nonempty_successful_nei(adapter):
    client, captured = _client([_semantic("not_in_source", None)])
    result = adapter.judge_frozen_row(_row(1), client)
    assert len(captured) == len(client.telemetry_events) == 1
    assert result["status"] == "ok"
    assert result["prediction"] == {"id": 7, "evidence": {}}
    assert result["pure_nei_eligible"] is True
    assert result["outcomes"][0]["explicit_not_in_source"] is True


@pytest.mark.parametrize(
    "relation,span",
    [
        ("partially_entailed", SUPPORT_TEXT),
        ("not_checkable", None),
    ],
)
def test_unsupported_relations_are_not_relabelled_as_correct_nei(
    adapter, relation, span
):
    client, captured = _client([_semantic(relation, span)])
    result = adapter.judge_frozen_row(_row(1), client)
    assert len(captured) == len(client.telemetry_events) == 1
    assert result["status"] == "unsupported_relation"
    assert result["pure_nei_eligible"] is False
    assert result["outcomes"][0]["explicit_not_in_source"] is False


@pytest.mark.parametrize(
    "semantic,category",
    [
        (
            _semantic(evidence_span="This span does not occur in selected evidence."),
            "judge_scope_error",
        ),
        (_semantic(evidence_span=None), "model_schema_error"),
        (_semantic(extra_private_field=PRIVATE_MARKER), "model_schema_error"),
        (_semantic("source_unavailable", None), "judge_scope_error"),
    ],
)
def test_scope_schema_and_missing_citation_failures_do_not_become_nei(
    adapter, semantic, category
):
    client, captured = _client([semantic])
    result = adapter.judge_frozen_row(_row(1), client)
    assert len(captured) == len(client.telemetry_events) == 1
    assert result["status"] == "error"
    assert result["pure_nei_eligible"] is False
    assert result["outcomes"][0]["explicit_not_in_source"] is False
    assert result["outcomes"][0]["error"]["category"] == category
    assert PRIVATE_MARKER not in json.dumps(result)
    assert PRIVATE_KEY not in json.dumps(result)


def test_transport_failure_is_recorded_once_without_retry_or_nei_credit(adapter):
    client, captured = _client([{"error": PRIVATE_MARKER}], status_code=503)
    result = adapter.judge_frozen_row(_row(1), client)
    assert result["status"] == "error"
    assert result["pure_nei_eligible"] is False
    assert len(captured) == len(client.telemetry_events) == 1
    assert client.telemetry_events[0].failure_kind == "transport"
    assert result["outcomes"][0]["error"]["category"] == "model_transport_error"
    assert PRIVATE_MARKER not in json.dumps(result)


def test_production_numeric_conflict_guard_remains_active(adapter):
    row = _row(1)
    row["context"]["claim"] = "The trial enrolled 50 participants."
    row["context"]["documents"][0]["sentences"][0]["text"] = (
        "The trial enrolled 40 participants."
    )
    _refresh_context_hash(row)
    client, captured = _client(
        [_semantic(evidence_span="The trial enrolled 40 participants.")]
    )
    result = adapter.judge_frozen_row(row, client)
    assert len(captured) == len(client.telemetry_events) == 1
    assert "numeric_mismatch" in {
        signal["code"] for signal in _model_inputs(captured)[0]["signals"]
    }
    assert result["status"] == "error"
    assert result["pure_nei_eligible"] is False
    assert result["outcomes"][0]["error"]["category"] == "deterministic_conflict"


def test_contiguous_sentence_span_maps_to_both_original_indices_in_selector_order(
    adapter,
):
    row = _row(1)
    first, last = row["context"]["documents"][0]["sentences"]
    middle = {
        "sentence_index": 6,
        "text": "Follow-up confirmed the survival improvement.",
    }
    row["context"]["documents"][0]["sentences"] = [first, middle, last]
    row["citation_sentence_indices"] = {"10": [9, 6, 5]}
    _refresh_context_hash(row)
    client, captured = _client([_semantic(evidence_span="survival.\nFollow-up")])
    result = adapter.judge_frozen_row(row, client)
    assert result["status"] == "ok"
    assert result["prediction"]["evidence"]["10"]["sentences"] == [6, 5]
    payload = _model_inputs(captured)[0]
    assert [item["text"] for item in payload["evidence"]] == [
        SUPPORT_TEXT + "\n" + middle["text"],
        last["text"],
    ]
    assert all(item["score"] == 0.0 for item in payload["evidence"])
    groups = result["outcomes"][0]["evidence_groups"]
    assert [offset["sentence_index"] for offset in groups[0]["sentence_offsets"]] == [
        5,
        6,
    ]
    assert groups[0]["sentence_offsets"][1]["char_start"] == len(SUPPORT_TEXT) + 1


def test_noncontiguous_selected_sentences_are_never_joined_for_model_span_matching(
    adapter,
):
    client, captured = _client([_semantic(evidence_span="survival.\nFollow-up")])
    result = adapter.judge_frozen_row(_row(1), client)
    assert len(_model_inputs(captured)[0]["evidence"]) == 2
    assert result["status"] == "error"
    assert result["outcomes"][0]["error"] == {
        "category": "judge_scope_error",
        "code": "evidence_span_out_of_scope",
    }
    assert result["pure_nei_eligible"] is False


def test_same_span_in_different_sentences_is_ambiguous_instead_of_guessing_first(
    adapter,
):
    row = _row(1)
    row["context"]["documents"][0]["sentences"][1]["text"] = SUPPORT_TEXT
    _refresh_context_hash(row)
    client, captured = _client([_semantic(evidence_span="improved survival")])
    result = adapter.judge_frozen_row(row, client)
    assert len(captured) == 1
    assert result["status"] == "error"
    assert result["pure_nei_eligible"] is False
    assert result["outcomes"][0]["error"] == {
        "category": "citation_error",
        "code": "ambiguous_citation",
    }


def test_repeated_span_within_same_sentence_still_maps_unambiguously(adapter):
    row = _row(1)
    row["context"]["documents"][0]["sentences"][0]["text"] = (
        SUPPORT_TEXT + " " + SUPPORT_TEXT
    )
    _refresh_context_hash(row)
    client, _ = _client([_semantic(evidence_span="improved survival")])
    result = adapter.judge_frozen_row(row, client)
    assert result["status"] == "ok"
    assert result["prediction"]["evidence"]["10"]["sentences"] == [5]


def test_one_failed_document_prevents_empty_prediction_from_receiving_nei_credit(
    adapter,
):
    client, captured = _client(
        [_semantic("not_in_source", None), _semantic(evidence_span="unseen evidence")]
    )
    result = adapter.judge_frozen_row(_row(2), client)
    assert len(captured) == 2
    assert result["status"] == "error"
    assert result["prediction"]["evidence"] == {}
    assert result["pure_nei_eligible"] is False
    assert result["operational_success"] is False
    assert [outcome["status"] for outcome in result["outcomes"]] == [
        "explicit_not_in_source",
        "error",
    ]


@pytest.mark.parametrize("client_kind", ["none", "recovery", "missing_telemetry"])
def test_missing_client_retry_wrapper_or_unaccounted_client_is_rejected(
    adapter, client_kind
):
    base, captured = _client([])
    client = (
        None
        if client_kind == "none"
        else SchemaRecoveryClient(base)
        if client_kind == "recovery"
        else object()
    )
    with pytest.raises(adapter.FrozenJudgeContractError):
        adapter.judge_frozen_row(_row(0), client)
    assert captured == []


def test_nonempty_success_without_any_recorded_provider_call_is_rejected(adapter):
    class SilentClient:
        model_id = "invalid-unaccounted-client"
        prompt_version = "invalid-unaccounted-client"
        telemetry_events = ()

        @staticmethod
        def complete_model(task, payload, schema):
            return schema.model_validate(_semantic())

    with pytest.raises(adapter.FrozenJudgeContractError):
        adapter.judge_frozen_row(_row(1), SilentClient())


@pytest.mark.parametrize("change", ["full_fit", "changed_text", "foreign_citation"])
def test_invalid_frozen_input_is_rejected_before_real_client_request(adapter, change):
    row = _row(1)
    if change == "full_fit":
        row["source"] = "full_train_fit"
    elif change == "changed_text":
        row["context"]["claim"] = "Changed after freezing."
    else:
        row["citation_sentence_indices"] = {"10": [99]}
    client, captured = _client([])
    with pytest.raises(adapter.FrozenJudgeContractError):
        adapter.judge_frozen_row(row, client)
    assert captured == []


def test_production_judge_recommendation_early_return_is_unsupported_not_nei(adapter):
    row = _row(1)
    row["context"]["claim"] = "The team should improve the evaluation."
    _refresh_context_hash(row)
    client, captured = _client([])
    result = adapter.judge_frozen_row(row, client)
    assert captured == []
    assert result["status"] == "unsupported_relation"
    assert result["pure_nei_eligible"] is False
    assert result["operational_success"] is False


def test_pydantic_input_failure_is_recorded_without_provider_call_or_nei_credit(
    adapter,
):
    row = _row(1)
    row["context"]["documents"][0]["sentences"][0]["text"] = "x" * (
        EVIDENCE_EXACT_MAX_CHARS + 1
    )
    _refresh_context_hash(row)
    client, captured = _client([])
    result = adapter.judge_frozen_row(row, client)
    assert captured == []
    assert result["status"] == "error"
    assert result["outcomes"][0]["error"]["category"] == "pydantic_validation_error"
    assert result["pure_nei_eligible"] is False
    assert result["judge_invocations"] == 0
    assert result["outcomes"][0]["judge_invoked"] is False


def test_synthetic_diagnostic_source_is_explicitly_preserved(adapter):
    row = _row(1)
    row["source"] = "synthetic_diagnostic"
    client, captured = _client([_semantic()])
    result = adapter.judge_frozen_row(row, client)
    assert len(captured) == 1
    assert result["status"] == "ok"
    assert result["selector_source"] == "synthetic_diagnostic"
