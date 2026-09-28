from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from evidencetrace.agents.judge import ClaimJudgeAgent, JudgeScopeError
from evidencetrace.audit_models import (
    JUDGE_SEMANTIC_CONTRACT_VERSION,
    JUDGE_VERDICT_OWNERSHIP_POLICY_VERSION,
    EvidenceChunk,
    JudgeInput,
    JudgeOutput,
    LiveJudgeSemanticOutput,
    RetrievedEvidence,
)
from evidencetrace.model_client import (
    DeterministicFakeModel,
    ModelSchemaError,
    OpenAICompatibleClient,
    SchemaRecoveryClient,
)
from evidencetrace.models import (
    AtomicClaim,
    Checkability,
    SourceMetadata,
)

SOURCE_ID = "ownership_source"
SOURCE_URL = "https://ownership.invalid/source"
PRIVATE_MARKER = "private-invalid-model-content"


def _input() -> JudgeInput:
    first = "星桥服务保留审计记录。"
    second = "星桥服务在版本8.2中支持离线校验。"

    def evidence(text: str, locator: str, start: int) -> RetrievedEvidence:
        chunk = EvidenceChunk(
            source_id=SOURCE_ID,
            url=SOURCE_URL,
            text=text,
            locator=locator,
            char_start=start,
            char_end=start + len(text),
        )
        return RetrievedEvidence(chunk=chunk, text=text, score=8.0)

    return JudgeInput(
        claim=AtomicClaim(
            claim_id="local_claim_001",
            text=second,
            file="docs/ownership.md",
            line_start=3,
            line_end=3,
            claim_type="versioned_capability",
            checkability=Checkability.CHECKABLE,
            citation_urls=(SOURCE_URL,),
        ),
        evidence=(
            evidence(first, "paragraph 1", 0),
            evidence(second, "paragraph 2", len(first) + 2),
        ),
        source=SourceMetadata(
            source_id=SOURCE_ID,
            url=SOURCE_URL,
            title="Ownership fixture",
            retrieved_at=datetime(2026, 7, 23, tzinfo=UTC),
            content_hash="f" * 64,
            mime_type="text/plain",
            status="ok",
        ),
    )


def _semantic(
    *,
    evidence_span: str = "星桥服务在版本8.2中支持离线校验。",
    **extra: Any,
) -> dict[str, Any]:
    return {
        "relation": "entailed",
        "confidence": 0.91,
        "reason": "The candidate evidence directly supports the claim.",
        "evidence_span": evidence_span,
        **extra,
    }


def test_live_judge_schema_contains_only_required_semantic_fields() -> None:
    schema = LiveJudgeSemanticOutput.model_json_schema()

    assert set(schema["properties"]) == {
        "relation",
        "confidence",
        "reason",
        "evidence_span",
    }
    assert set(schema["required"]) == set(schema["properties"])
    assert schema["additionalProperties"] is False
    for field in (
        "claim_id",
        "source_id",
        "source_ids",
        "locator",
        "judge_version",
        "model_id",
        "prompt_version",
        "corroboration",
    ):
        assert field not in schema["properties"]


def test_local_context_owns_all_canonical_identity_and_locator_fields() -> None:
    requests: list[tuple[str, dict[str, Any], type[Any]]] = []

    def respond(task: str, payload: dict[str, Any]) -> dict[str, Any]:
        requests.append((task, payload, LiveJudgeSemanticOutput))
        return _semantic()

    output = ClaimJudgeAgent(DeterministicFakeModel(respond)).judge(_input())

    assert output == JudgeOutput.model_validate(output.model_dump())
    assert output.verdict.claim_id == "local_claim_001"
    assert output.verdict.source_ids == (SOURCE_ID,)
    assert output.verdict.evidence_spans[0].source_id == SOURCE_ID
    assert output.verdict.evidence_spans[0].locator == "paragraph 2"
    assert output.verdict.judge_version == JUDGE_VERDICT_OWNERSHIP_POLICY_VERSION
    assert output.model_id == "deterministic-fake-v1"
    assert output.prompt_version == JUDGE_SEMANTIC_CONTRACT_VERSION

    task, payload, _schema = requests[0]
    assert task == "claim_judgement"
    assert set(payload["claim"]) == {
        "text",
        "claim_type",
        "slots",
        "checkability",
    }
    assert all(set(item) == {"text", "score"} for item in payload["evidence"])
    assert payload["source"] == {"status": "ok"}
    serialized = json.dumps(payload, ensure_ascii=False)
    for field in (
        "claim_id",
        "source_id",
        "locator",
        "judge_version",
        "model_id",
        "prompt_version",
    ):
        assert field not in serialized


@pytest.mark.parametrize(
    "field",
    [
        "claim_id",
        "source_id",
        "source_ids",
        "locator",
        "judge_version",
        "model_id",
        "prompt_version",
    ],
)
def test_model_cannot_add_or_override_locally_owned_field(field: str) -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        LiveJudgeSemanticOutput.model_validate(_semantic(**{field: "untrusted"}))


def test_forged_span_still_fails_closed_without_identity_payload() -> None:
    client = DeterministicFakeModel(
        lambda _task, _payload: _semantic(evidence_span="伪造且不在候选中的证据。")
    )

    with pytest.raises(JudgeScopeError) as raised:
        ClaimJudgeAgent(client).judge(_input())

    assert raised.value.code == "evidence_span_out_of_scope"
    assert vars(raised.value) == {"code": "evidence_span_out_of_scope"}


def _recovery_client(
    responses: list[dict[str, Any]],
    captured: list[dict[str, Any]],
) -> SchemaRecoveryClient:
    remaining = iter(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        content = json.dumps(next(remaining), ensure_ascii=False)
        return httpx.Response(
            200,
            request=request,
            json={
                "choices": [
                    {
                        "message": {"content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 17,
                    "completion_tokens": 7,
                    "total_tokens": 24,
                },
            },
        )

    base = OpenAICompatibleClient(
        model_id="offline-ownership-model",
        api_key="offline-placeholder",
        base_url="https://provider.invalid/v1",
        max_calls=2,
        transport=httpx.MockTransport(handler),
    )
    return SchemaRecoveryClient(base)


def test_extra_claim_id_triggers_one_schema_retry_then_local_assembly() -> None:
    captured: list[dict[str, Any]] = []
    client = _recovery_client(
        [
            _semantic(claim_id="model-claim"),
            _semantic(),
        ],
        captured,
    )

    output = ClaimJudgeAgent(client).judge(_input())

    assert output.verdict.claim_id == "local_claim_001"
    assert output.verdict.source_ids == (SOURCE_ID,)
    assert len(captured) == 2
    assert client.schema_recovery.first_attempt_schema_failures == 1
    assert client.schema_recovery.schema_retry_calls == 1
    assert client.schema_recovery.recovered_schema_failures == 1
    assert client.schema_recovery.unrecovered_schema_failures == 0
    first_user = json.loads(captured[0]["messages"][1]["content"])
    retry_user = json.loads(captured[1]["messages"][1]["content"])
    assert first_user["output_contract"]["schema_name"] == ("LiveJudgeSemanticOutput")
    assert retry_user["output_contract"] == first_user["output_contract"]
    assert retry_user["input"]["schema_recovery"]["attempt_index"] == 2
    assert "model-claim" not in captured[1]["messages"][1]["content"]


def test_second_extra_id_failure_stops_and_retains_no_model_content() -> None:
    captured: list[dict[str, Any]] = []
    client = _recovery_client(
        [
            _semantic(claim_id=PRIVATE_MARKER),
            _semantic(source_id=PRIVATE_MARKER),
        ],
        captured,
    )

    with pytest.raises(ModelSchemaError) as raised:
        ClaimJudgeAgent(client).judge(_input())

    assert len(captured) == 2
    assert client.schema_recovery.recovered_schema_failures == 0
    assert client.schema_recovery.unrecovered_schema_failures == 1
    safe = repr(raised.value) + str(raised.value)
    assert PRIVATE_MARKER not in safe
    assert "星桥服务" not in safe
    assert SOURCE_URL not in safe
