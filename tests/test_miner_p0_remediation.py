from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from evidencetrace.agents.miner import ClaimMinerAgent, MinerScopeError
from evidencetrace.audit_models import MinerInput
from evidencetrace.model_client import (
    DeterministicFakeModel,
    OpenAICompatibleClient,
    SchemaRecoveryClient,
)
from evidencetrace.product import (
    AgentName,
    DocumentRunStatus,
    ProductAuditPipeline,
    TraceState,
)

PRIVATE_RESPONSE = "private-provider-response-must-not-survive"
PRIVATE_KEY = "private-api-key-must-not-survive"


def _draft(text: str) -> dict[str, Any]:
    return {
        "claims": [
            {
                "text": text,
                "claim_type": "factual_statement",
                "checkability": "checkable",
            }
        ]
    }


def _chat_response(
    request: httpx.Request,
    content: str,
    *,
    finish_reason: str = "stop",
) -> httpx.Response:
    return httpx.Response(
        200,
        request=request,
        json={
            "choices": [
                {
                    "finish_reason": finish_reason,
                    "message": {"content": content},
                }
            ],
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 7,
                "total_tokens": 18,
            },
        },
    )


def _live_miner(
    responses: list[str],
    requests: list[dict[str, Any]],
) -> ClaimMinerAgent:
    remaining = iter(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return _chat_response(request, next(remaining))

    base = OpenAICompatibleClient(
        model_id="offline-miner",
        api_key=PRIVATE_KEY,
        base_url="https://provider.invalid/v1",
        max_calls=len(responses),
        transport=httpx.MockTransport(handler),
    )
    return ClaimMinerAgent(SchemaRecoveryClient(base))


def _miner_events(result: Any, state: TraceState) -> list[Any]:
    return [
        event
        for event in result.product.trace
        if event.agent is AgentName.MINER and event.state is state
    ]


@pytest.mark.parametrize(
    ("input_text", "draft_texts", "expected_code"),
    [
        (
            "Nimbus ships stable builds.",
            ["Nimbus reliably ships stable builds."],
            "non_source_span",
        ),
        (
            "Nimbus ships builds with signed metadata.",
            ["Nimbus ships signed metadata."],
            "non_source_span",
        ),
        (
            "Alpha remains ready. Beta remains ready.",
            ["Beta remains ready.", "Alpha remains ready."],
            "non_source_span",
        ),
        (
            "Nimbus ships stable builds.",
            ["Nimbus ships stable builds.", "stable builds."],
            "non_source_span",
        ),
        (
            "Nimbus did not ship v1.2.",
            ["Nimbus did ship v1.2."],
            "missing_protected_token",
        ),
        (
            "Firefox",
            ["Firefox"],
            "invalid_claim_fragment",
        ),
    ],
    ids=[
        "added-word",
        "non-contiguous",
        "out-of-order",
        "overlap",
        "missing-protected-token",
        "invalid-fragment",
    ],
)
def test_model_visible_contract_does_not_relax_local_scope_validation(
    input_text: str,
    draft_texts: list[str],
    expected_code: str,
) -> None:
    model = DeterministicFakeModel(
        lambda _task, _payload: {
            "claims": [
                {
                    "text": text,
                    "claim_type": "factual_statement",
                    "checkability": "checkable",
                }
                for text in draft_texts
            ]
        }
    )

    with pytest.raises(MinerScopeError) as raised:
        ClaimMinerAgent(model).mine(
            MinerInput(
                text=input_text,
                file="docs/contract.md",
                line_start=1,
                line_end=1,
            )
        )

    assert raised.value.code == expected_code


def test_empty_live_miner_claims_remain_valid_for_context_only_text() -> None:
    model = DeterministicFakeModel(lambda _task, _payload: {"claims": []})

    output = ClaimMinerAgent(model).mine(
        MinerInput(
            text="Browser support:",
            file="docs/contract.md",
            line_start=1,
            line_end=1,
        )
    )

    assert output.claims == ()


def test_scope_failure_is_typed_and_next_paragraph_still_runs(
    tmp_path: Path,
) -> None:
    document = tmp_path / "doc.md"
    document.write_text(
        "The first paragraph states Alpha.\n\nThe second paragraph states Beta.\n",
        encoding="utf-8",
    )
    calls = 0

    def handler(_task: str, payload: dict[str, Any]) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        text = (
            "Invented private claim."
            if calls == 1
            else "The second paragraph states Beta."
        )
        return _draft(text)

    pipeline = ProductAuditPipeline(project_root=tmp_path)
    pipeline.miner = ClaimMinerAgent(DeterministicFakeModel(handler))
    result = pipeline.run(document, persist=False)

    failures = _miner_events(result, TraceState.FAILED)
    completed = _miner_events(result, TraceState.COMPLETED)
    assert result.product.document_status is DocumentRunStatus.PARTIAL
    assert [claim.text for claim in result.audit.claims] == [
        "The second paragraph states Beta."
    ]
    assert len(failures) == len(completed) == 1
    assert failures[0].code == "miner_scope_non_source_span"
    assert failures[0].miner_dispatch_id == "miner_0001"
    assert failures[0].paragraph_id == "p_0001"
    assert completed[0].miner_dispatch_id == "miner_0002"
    assert completed[0].paragraph_id == "p_0002"
    safe_artifact = result.product.model_dump_json()
    assert "Invented private claim." not in safe_artifact


def test_schema_retry_recovers_once_and_records_safe_attempts(
    tmp_path: Path,
) -> None:
    document = tmp_path / "doc.md"
    claim = "The supported version is 4.2."
    document.write_text(claim + "\n", encoding="utf-8")
    requests: list[dict[str, Any]] = []
    pipeline = ProductAuditPipeline(project_root=tmp_path)
    pipeline.miner = _live_miner(
        [
            '{"unexpected":"' + PRIVATE_RESPONSE + '"}',
            json.dumps(_draft(claim)),
        ],
        requests,
    )

    result = pipeline.run(document, persist=False)
    completed = _miner_events(result, TraceState.COMPLETED)

    assert len(requests) == 2
    assert len(completed) == 1
    assert completed[0].provider_attempts == 2
    assert completed[0].schema_retry is True
    assert completed[0].finish_reason == "stop"
    assert result.audit.claims[0].text == claim
    assert PRIVATE_RESPONSE not in requests[1]["messages"][1]["content"]
    safe_artifact = result.product.model_dump_json()
    assert PRIVATE_RESPONSE not in safe_artifact
    assert PRIVATE_KEY not in safe_artifact


def test_second_schema_failure_stops_retry_but_not_later_paragraph(
    tmp_path: Path,
) -> None:
    document = tmp_path / "doc.md"
    second_claim = "The second paragraph ships version 5.1."
    document.write_text(
        "The first paragraph ships version 4.2.\n\n" + second_claim + "\n",
        encoding="utf-8",
    )
    requests: list[dict[str, Any]] = []
    pipeline = ProductAuditPipeline(project_root=tmp_path)
    pipeline.miner = _live_miner(
        [
            '{"unexpected":"' + PRIVATE_RESPONSE + '"}',
            '{"still_unexpected":true}',
            json.dumps(_draft(second_claim)),
        ],
        requests,
    )

    result = pipeline.run(document, persist=False)
    failures = _miner_events(result, TraceState.FAILED)
    completed = _miner_events(result, TraceState.COMPLETED)

    assert len(requests) == 3
    assert len(failures) == len(completed) == 1
    assert failures[0].code == "miner_schema_extra_field"
    assert failures[0].provider_attempts == 2
    assert failures[0].schema_retry is True
    assert completed[0].provider_attempts == 1
    assert completed[0].schema_retry is False
    assert [claim.text for claim in result.audit.claims] == [second_claim]
    safe_artifact = result.product.model_dump_json()
    assert PRIVATE_RESPONSE not in safe_artifact
    assert PRIVATE_KEY not in safe_artifact
