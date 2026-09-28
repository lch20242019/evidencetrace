from __future__ import annotations

import hashlib
import json
from typing import Any

import httpx
import pytest
from pydantic import BaseModel, ValidationError

from evidencetrace.eval.baselines import (
    maximum_expected_live_model_calls,
    run_baseline,
)
from evidencetrace.eval.metrics import engineering_metrics
from evidencetrace.eval.models import EvalCase, EvalPrediction, SourceFixture
from evidencetrace.model_client import (
    ModelCallBudgetExceeded,
    ModelCallTelemetry,
    ModelSchemaError,
    ModelTransportError,
    OpenAICompatibleClient,
)


class Output(BaseModel):
    value: int


def _response(
    *,
    content: dict[str, Any] | None = None,
    usage: object = ...,
    status: int = 200,
) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        request_json = json.loads(request.content)
        assert request_json["temperature"] == 0.25
        assert set(request_json) == {
            "model",
            "temperature",
            "response_format",
            "max_tokens",
            "messages",
        }
        assert request_json["response_format"] == {"type": "json_object"}
        assert request_json["max_tokens"] == 2048
        assert "thinking" not in request_json
        assert "reasoning_effort" not in request_json
        body: dict[str, Any] = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(content or {"value": 7}),
                    }
                }
            ]
        }
        if usage is not ...:
            body["usage"] = usage
        return httpx.Response(status, request=request, json=body)

    return httpx.MockTransport(handler)


def _adapter(*, transport: httpx.MockTransport) -> OpenAICompatibleClient:
    return OpenAICompatibleClient(
        model_id="telemetry-model",
        api_key="private-test-key",
        base_url="https://model.example/v1",
        temperature=0.25,
        transport=transport,
    )


def test_openai_usage_reported_missing_and_malformed_are_distinct() -> None:
    reported = _adapter(
        transport=_response(
            usage={
                "prompt_tokens": 11,
                "completion_tokens": 3,
                "total_tokens": 14,
            }
        )
    )
    assert reported.complete_model("task", {}, Output) == Output(value=7)
    assert reported.telemetry.usage_status == "reported"
    assert reported.telemetry.input_tokens == 11
    assert reported.telemetry.output_tokens == 3
    assert reported.telemetry.total_tokens == 14

    missing = _adapter(transport=_response())
    assert missing.complete_model("task", {}, Output) == Output(value=7)
    assert missing.telemetry.usage_status == "missing"
    assert missing.telemetry.total_tokens is None

    malformed = _adapter(
        transport=_response(
            usage={
                "prompt_tokens": "11",
                "completion_tokens": 3,
                "total_tokens": 14,
            }
        )
    )
    assert malformed.complete_model("task", {}, Output) == Output(value=7)
    assert malformed.telemetry.usage_status == "malformed"
    assert malformed.telemetry.input_tokens is None
    assert malformed.telemetry.total_tokens is None


def test_model_failures_are_classified_without_secret_material() -> None:
    schema_client = _adapter(
        transport=_response(
            content={"value": "not-an-int"},
            usage={
                "prompt_tokens": 4,
                "completion_tokens": 2,
                "total_tokens": 6,
            },
        )
    )
    with pytest.raises(ModelSchemaError):
        schema_client.complete_model("task", {}, Output)
    assert schema_client.telemetry.calls == 1
    assert schema_client.telemetry.schema_failures == 1
    assert schema_client.telemetry.transport_failures == 0

    transport_client = _adapter(transport=_response(status=503))
    with pytest.raises(ModelTransportError):
        transport_client.complete_model("task", {}, Output)
    assert transport_client.telemetry.calls == 1
    assert transport_client.telemetry.transport_failures == 1

    serialized = (
        schema_client.telemetry.model_dump_json()
        + transport_client.telemetry.model_dump_json()
        + repr(schema_client.telemetry_events)
    )
    assert "private-test-key" not in serialized
    assert "authorization" not in serialized.casefold()
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        ModelCallTelemetry(latency_ms=-1.0, usage_status="missing")


def test_hard_call_cap_blocks_before_post_and_503_is_not_retried() -> None:
    provider_posts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal provider_posts
        provider_posts += 1
        return httpx.Response(
            503,
            request=request,
            json={"error": {"message": "temporary provider failure"}},
        )

    client = OpenAICompatibleClient(
        model_id="telemetry-model",
        api_key="private-test-key",
        base_url="https://model.example/v1",
        temperature=0.0,
        max_calls=1,
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(ModelTransportError):
        client.complete_model("task", {}, Output)

    assert provider_posts == 1
    assert client.telemetry.calls == 1
    assert client.telemetry.transport_failures == 1
    assert client.automatic_retry_count == 0

    with pytest.raises(ModelCallBudgetExceeded, match="hard limit"):
        client.complete_model("task", {}, Output)

    assert provider_posts == 1
    assert client.telemetry.calls == 1
    assert client.telemetry.transport_failures == 1


def _source() -> SourceFixture:
    content = "Orchid 2.4.0 ships offline mode."
    return SourceFixture(
        source_id="orchid_docs",
        url="https://docs.example.test/orchid",
        content=content,
        provenance="new Phase 3D telemetry fixture",
        content_hash=hashlib.sha256(content.encode()).hexdigest(),
    )


def _case() -> EvalCase:
    return EvalCase(
        case_id="phase3d_telemetry",
        claim_text="Orchid 2.4.0 ships offline mode.",
        source_fixture="sources.jsonl",
        source_id="orchid_docs",
        source_url="https://docs.example.test/orchid",
        gold_relation="entailed",
        gold_evidence_span="Orchid 2.4.0 ships offline mode.",
        claim_type="versioned_capability",
        mutation_type="none",
        split="dev",
        provenance="new Phase 3D telemetry fixture",
        annotation_status="deterministic_gold",
        annotation_notes="Telemetry contract fixture.",
    )


class TelemetryClient:
    model_id = "same-live-model"
    prompt_version = "same-live-prompt"
    provider_id = "openai-compatible"
    temperature = 0.0

    def __init__(self) -> None:
        self.telemetry_events: tuple[ModelCallTelemetry, ...] = ()

    def complete(self, task: str, payload: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError("live baselines must use complete_model")

    def complete_model(
        self, task: str, payload: dict[str, Any], schema: type[Any]
    ) -> Any:
        self.telemetry_events += (
            ModelCallTelemetry(
                latency_ms=5.0,
                usage_status="reported",
                input_tokens=10,
                output_tokens=2,
                total_tokens=12,
            ),
        )
        if task == "single_agent_full_source_verification":
            return schema.model_validate(
                {
                    "relation": "entailed",
                    "confidence": 0.8,
                    "reason": "The exact source sentence supports the claim.",
                    "evidence_span": _source().content,
                }
            )
        if task == "claim_mining":
            return schema.model_validate(
                {
                    "claims": [
                        {
                            "claim_id": "c_0001",
                            "text": payload["text"],
                            "file": payload["file"],
                            "line_start": payload["line_start"],
                            "line_end": payload["line_end"],
                            "claim_type": "versioned_capability",
                            "checkability": "checkable",
                            "citation_urls": payload["citation_urls"],
                        }
                    ],
                    "model_id": self.model_id,
                    "prompt_version": self.prompt_version,
                }
            )
        evidence = payload["evidence"][0]
        return schema.model_validate(
            {
                "relation": "entailed",
                "confidence": 0.8,
                "evidence_span": evidence["text"],
                "reason": "The retrieved source sentence supports the claim.",
            }
        )


def test_live_baseline_telemetry_and_call_ceilings_are_explicit() -> None:
    client = TelemetryClient()
    single = run_baseline(_case(), _source(), "single_agent_live", client=client)
    isolated = run_baseline(_case(), _source(), "retrieval_judge_live", client=client)

    assert single.model_calls == 1
    assert single.model_call_latencies_ms == (5.0,)
    assert single.model_usage_status == "reported"
    assert (single.input_tokens, single.output_tokens, single.total_tokens) == (
        10,
        2,
        12,
    )
    assert isolated.model_calls == 1
    assert isolated.model_call_latencies_ms == (5.0,)
    assert isolated.total_tokens == 12
    assert single.model_provider == isolated.model_provider
    assert single.model_id == isolated.model_id
    assert single.model_temperature == isolated.model_temperature == 0.0
    assert maximum_expected_live_model_calls(60) == 180
    assert maximum_expected_live_model_calls(60, schema_retry_limit=1) == 360
    with pytest.raises(ValueError, match="negative"):
        maximum_expected_live_model_calls(-1)
    with pytest.raises(ValueError, match="zero or one"):
        maximum_expected_live_model_calls(60, schema_retry_limit=2)


def test_deterministic_baselines_stay_zero_call_and_cost_requires_price() -> None:
    client = TelemetryClient()
    predictions = [
        run_baseline(_case(), _source(), name, client=client)
        for name in (
            "lexical_rules",
            "lexical_full_source",
            "retrieval_judge_deterministic",
        )
    ]
    assert client.telemetry_events == ()
    assert all(prediction.model_calls == 0 for prediction in predictions)
    assert all(
        prediction.model_usage_status == "not_applicable" for prediction in predictions
    )
    assert all(prediction.model_call_latencies_ms == () for prediction in predictions)

    payload = predictions[0].model_dump()
    payload.update({"cost_usd": 0.01})
    with pytest.raises(ValidationError, match="pricing snapshot"):
        EvalPrediction.model_validate(payload)


def test_engineering_metrics_aggregate_provider_usage_and_latency() -> None:
    records = [
        {
            "gold_relation": "entailed",
            "predicted_relation": "entailed",
            "model_calls": 1,
            "model_call_latencies_ms": [5.0],
            "model_usage_status": "reported",
            "input_tokens": 10,
            "output_tokens": 2,
            "total_tokens": 12,
            "schema_failure_count": 0,
            "transport_failure_count": 0,
            "model_provider": "openai-compatible",
            "model_id": "same-live-model",
            "model_temperature": 0.0,
            "latency_ms": 7.0,
        },
        {
            "gold_relation": "entailed",
            "predicted_relation": "entailed",
            "model_calls": 2,
            "model_call_latencies_ms": [6.0, 8.0],
            "model_usage_status": "reported",
            "input_tokens": 20,
            "output_tokens": 4,
            "total_tokens": 24,
            "schema_failure_count": 1,
            "transport_failure_count": 0,
            "model_provider": "openai-compatible",
            "model_id": "same-live-model",
            "model_temperature": 0.0,
            "latency_ms": 16.0,
        },
    ]
    metrics = engineering_metrics(records)

    assert metrics["model_call_count"] == 3
    assert metrics["model_usage"] == {
        "status": "reported",
        "input_tokens": 30,
        "output_tokens": 6,
        "total_tokens": 36,
    }
    assert metrics["model_latency_ms_p50"] == 6.0
    assert metrics["model_latency_ms_p95"] == 8.0
    assert metrics["schema_failure_count"] == 1
    assert metrics["transport_failure_count"] == 0
    assert metrics["model_configuration"]["consistent"] is True
    assert metrics["cost_usd"] is None
    assert "price snapshot" in metrics["cost_explanation"]

    malformed = engineering_metrics(
        [
            {
                **records[0],
                "model_usage_status": "malformed",
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
            }
        ]
    )
    assert malformed["model_usage"]["status"] == "malformed"
    assert malformed["model_usage"]["total_tokens"] is None
