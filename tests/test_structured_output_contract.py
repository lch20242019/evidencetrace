from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import BaseModel, ValidationError

import evidencetrace.eval.runner as eval_runner
from evidencetrace.audit_models import (
    LiveJudgeSemanticOutput,
    LiveMinerDraftOutput,
)
from evidencetrace.eval.models import EvalFailureArtifact, SingleAgentLiveOutput
from evidencetrace.eval.runner import run_eval
from evidencetrace.model_client import (
    DEFAULT_MAX_TOKENS,
    MODEL_PROVIDER_CONTRACT_VERSION,
    ModelCallBudgetExceeded,
    ModelSchemaDiagnostic,
    ModelSchemaError,
    ModelTransportError,
    OpenAICompatibleClient,
    SchemaValidationIssue,
)

ROOT = Path(__file__).parents[1]
V2_DEV_DATASET = ROOT / "eval_sets/v2/dev.jsonl"
HISTORICAL_FAILURE = ROOT / "eval_runs/phase3d_v2_dev_live_rerun/failed_attempt.json"
_UNSET = object()
API_SECRET = "structured-output-api-secret"
RAW_CONTENT = "raw-model-content-must-not-persist"
REASONING_SECRET = "reasoning-content-must-not-persist"
CLAIM_SECRET = "claim-payload-must-not-persist"
SOURCE_SECRET = "source-payload-must-not-persist"
PROVIDER_FINISH_MARKER = "provider_private_marker"


_EXAMPLE_JSON = {
    "SingleAgentLiveOutput": (
        '{"relation":"entailed","confidence":0.9,'
        '"reason":"The source directly states the claim.",'
        '"evidence_span":"Nimbus shipped v1.3."}'
    ),
    "MinerOutput": (
        '{"claims":[{"claim_id":"c_0001",'
        '"text":"Nimbus shipped v1.3.","file":"docs/nimbus.md",'
        '"line_start":1,"line_end":1,"claim_type":'
        '"versioned_capability","checkability":"checkable",'
        '"citation_urls":["https://docs.example.test/nimbus"]}],'
        '"model_id":"example-miner","prompt_version":"miner-v1"}'
    ),
    "LiveMinerDraftOutput": (
        '{"claims":[{"text":"Nimbus shipped v1.3.",'
        '"claim_type":"versioned_capability",'
        '"checkability":"checkable"}]}'
    ),
    "LiveJudgeSemanticOutput": (
        '{"relation":"entailed","confidence":0.9,'
        '"reason":"The candidate evidence directly states the claim.",'
        '"evidence_span":"Nimbus shipped v1.3."}'
    ),
}


def _schema_example(schema: type[BaseModel]) -> dict[str, Any]:
    example = json.loads(_EXAMPLE_JSON[schema.__name__])
    schema.model_validate_json(json.dumps(example), strict=True)
    return example


def _mock_transport(
    content: object,
    *,
    finish_reason: object = "stop",
    reasoning_content: object = _UNSET,
    usage: object = _UNSET,
    inspect_request: Callable[[dict[str, Any]], None] | None = None,
    post_counter: list[int] | None = None,
) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if post_counter is not None:
            post_counter.append(1)
        request_body = json.loads(request.content)
        if inspect_request is not None:
            inspect_request(request_body)
        message: dict[str, object] = {"content": content}
        if reasoning_content is not _UNSET:
            message["reasoning_content"] = reasoning_content
        choice: dict[str, object] = {"message": message}
        if finish_reason is not _UNSET:
            choice["finish_reason"] = finish_reason
        body: dict[str, object] = {"choices": [choice]}
        if usage is not _UNSET:
            body["usage"] = usage
        return httpx.Response(200, request=request, json=body)

    return httpx.MockTransport(handler)


def _client(
    transport: httpx.MockTransport,
    *,
    max_calls: int | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> OpenAICompatibleClient:
    return OpenAICompatibleClient(
        model_id="deepseek-v4-flash",
        api_key=API_SECRET,
        base_url="https://provider.example/v1",
        temperature=0.0,
        max_calls=max_calls,
        max_tokens=max_tokens,
        transport=transport,
    )


@pytest.mark.parametrize(
    "schema",
    [SingleAgentLiveOutput, LiveMinerDraftOutput, LiveJudgeSemanticOutput],
    ids=["single-agent", "miner", "judge"],
)
def test_three_output_schemas_send_strict_json_contract(
    schema: type[BaseModel],
) -> None:
    example = _schema_example(schema)
    captured: list[dict[str, Any]] = []

    client = _client(
        _mock_transport(
            json.dumps(example, separators=(",", ":")),
            inspect_request=captured.append,
        )
    )
    result = client.complete_model(
        "contract_test",
        {"claim": CLAIM_SECRET, "source": SOURCE_SECRET},
        schema,
    )

    assert result == schema.model_validate_json(json.dumps(example), strict=True)
    assert client.prompt_version == MODEL_PROVIDER_CONTRACT_VERSION
    assert len(captured) == 1
    request = captured[0]
    assert request["response_format"] == {"type": "json_object"}
    assert request["max_tokens"] == DEFAULT_MAX_TOKENS
    assert request["temperature"] == 0.0
    assert "thinking" not in request
    assert "reasoning_effort" not in request

    system = request["messages"][0]["content"].casefold()
    assert "exactly one json object" in system
    assert "strictly matches" in system
    assert "extra fields" in system
    assert "code fence" in system

    user = json.loads(request["messages"][1]["content"])
    assert user["task"] == "contract_test"
    assert user["input"] == {
        "claim": CLAIM_SECRET,
        "source": SOURCE_SECRET,
    }
    contract = user["output_contract"]
    instructions = " ".join(contract["instructions"]).casefold()
    assert "exactly one json object" in instructions
    assert "strictly match output_schema" in instructions
    assert "do not add fields" in instructions
    assert "code fence" in instructions
    assert contract["schema_name"] == schema.__name__
    assert contract["output_schema"] == schema.model_json_schema()
    assert contract["output_example"] == example
    schema.model_validate_json(
        json.dumps(contract["output_example"]),
        strict=True,
    )


def test_live_miner_provider_request_exposes_scope_contract() -> None:
    captured: list[dict[str, Any]] = []
    example = _schema_example(LiveMinerDraftOutput)
    client = _client(
        _mock_transport(
            json.dumps(example),
            inspect_request=captured.append,
        )
    )

    client.complete_model(
        "claim_mining",
        {"text": "Nimbus shipped v1.3."},
        LiveMinerDraftOutput,
    )

    user = json.loads(captured[0]["messages"][1]["content"])
    output_schema = user["output_contract"]["output_schema"]
    draft_schema = output_schema["$defs"]["LiveMinedClaimDraft"]
    text_description = draft_schema["properties"]["text"]["description"].casefold()
    claims_description = output_schema["properties"]["claims"]["description"].casefold()

    for required in (
        "current input.text",
        "single, contiguous",
        "character-for-character",
        "case",
        "whitespace",
        "punctuation",
        "numbers",
        "versions",
        "identifiers",
        "do not paraphrase",
        "summarize",
        "add words",
        "delete words",
        "join non-contiguous fragments",
    ):
        assert required in text_description
    for required in (
        "current input.text",
        "source-text order",
        "without overlap",
        "empty array",
        "headings",
        "lead-ins",
        "isolated list labels",
        "atomic factual claim",
        "numbers",
        "versions",
        "dates",
        "negations",
        "comparison qualifiers",
    ):
        assert required in claims_description
    assert output_schema == LiveMinerDraftOutput.model_json_schema()
    assert output_schema["additionalProperties"] is False
    assert draft_schema["additionalProperties"] is False


@pytest.mark.parametrize("max_tokens", [0, True, 8193])
def test_max_tokens_has_a_validated_bound(max_tokens: int) -> None:
    with pytest.raises(ValueError, match="max_tokens"):
        _client(_mock_transport("{}"), max_tokens=max_tokens)


def test_custom_max_tokens_is_sent_without_changing_thinking_mode() -> None:
    captured: list[dict[str, Any]] = []
    example = _schema_example(SingleAgentLiveOutput)
    client = _client(
        _mock_transport(
            json.dumps(example),
            inspect_request=captured.append,
        ),
        max_tokens=512,
    )

    client.complete_model("task", {}, SingleAgentLiveOutput)

    assert captured[0]["max_tokens"] == 512
    assert "thinking" not in captured[0]
    assert client.thinking_mode == "provider_default"
    assert client.temperature == 0.0
    assert client.temperature_effective is None


def test_content_wins_and_reasoning_content_is_never_retained() -> None:
    example = _schema_example(SingleAgentLiveOutput)
    client = _client(
        _mock_transport(
            json.dumps(example),
            reasoning_content=REASONING_SECRET,
            usage={
                "prompt_tokens": 9,
                "completion_tokens": 4,
                "total_tokens": 13,
            },
        )
    )

    result = client.complete_model("task", {}, SingleAgentLiveOutput)

    assert result.relation.value == "entailed"
    serialized = client.telemetry.model_dump_json() + repr(client.telemetry_events)
    assert REASONING_SECRET not in serialized
    assert client.telemetry.total_tokens == 13


_BASE_SINGLE = {
    "relation": "entailed",
    "confidence": 0.8,
    "reason": "The source supports the claim.",
    "evidence_span": "Nimbus shipped v1.3.",
}


@pytest.mark.parametrize(
    ("payload", "category", "location", "error_type"),
    [
        (
            {
                "relation": "entailed",
                "confidence": 0.8,
                "evidence_span": "Nimbus shipped v1.3.",
            },
            "missing_field",
            ("reason",),
            "missing",
        ),
        (
            {**_BASE_SINGLE, "unexpected": RAW_CONTENT},
            "extra_field",
            ("unknown_field",),
            "extra_forbidden",
        ),
        (
            {**_BASE_SINGLE, "relation": "supported"},
            "enum",
            ("relation",),
            "enum",
        ),
        (
            {**_BASE_SINGLE, "confidence": RAW_CONTENT},
            "wrong_type",
            ("confidence",),
            "float_type",
        ),
    ],
    ids=["missing", "extra", "enum", "wrong-type"],
)
def test_pydantic_failures_produce_bounded_safe_diagnostics(
    payload: dict[str, Any],
    category: str,
    location: tuple[str, ...],
    error_type: str,
) -> None:
    client = _client(
        _mock_transport(
            json.dumps(payload),
            reasoning_content=REASONING_SECRET,
            usage={
                "prompt_tokens": 17,
                "completion_tokens": 5,
                "total_tokens": 22,
            },
        )
    )

    with pytest.raises(ModelSchemaError) as raised:
        client.complete_model(
            "task",
            {"claim": CLAIM_SECRET, "source": SOURCE_SECRET},
            SingleAgentLiveOutput,
        )

    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    diagnostic = raised.value.diagnostic
    assert diagnostic is not None
    assert diagnostic.schema_name == "SingleAgentLiveOutput"
    assert diagnostic.failure_category == category
    assert len(diagnostic.issues) <= 8
    assert diagnostic.issues[0].location == location
    assert diagnostic.issues[0].error_type == error_type
    assert len(diagnostic.safe_message) <= 160
    assert diagnostic.finish_reason == "stop"
    assert client.telemetry.calls == 1
    assert client.telemetry.total_tokens == 22
    assert client.telemetry.schema_failures == 1
    assert client.automatic_retry_count == 0

    serialized = diagnostic.model_dump_json() + repr(raised.value)
    for forbidden in (
        RAW_CONTENT,
        REASONING_SECRET,
        CLAIM_SECRET,
        SOURCE_SECRET,
        API_SECRET,
    ):
        assert forbidden not in serialized
    raw = diagnostic.model_dump()
    raw["raw_content"] = RAW_CONTENT
    with pytest.raises(ValidationError, match="extra_forbidden"):
        ModelSchemaDiagnostic.model_validate(raw)


@pytest.mark.parametrize(
    ("content", "finish_reason", "category", "content_length"),
    [
        (None, "stop", "empty_content", 0),
        ("", "stop", "empty_content", 0),
        ("not-json-" + RAW_CONTENT, "stop", "invalid_json", None),
        (
            "\x60\x60\x60json\n" + json.dumps(_BASE_SINGLE) + "\n\x60\x60\x60",
            "stop",
            "invalid_json",
            None,
        ),
        (json.dumps(_BASE_SINGLE), "length", "finish_reason", None),
    ],
    ids=["none", "empty", "invalid-json", "fenced", "length"],
)
def test_empty_invalid_fenced_and_truncated_content_fail_closed(
    content: object,
    finish_reason: str,
    category: str,
    content_length: int | None,
) -> None:
    posts: list[int] = []
    client = _client(
        _mock_transport(
            content,
            finish_reason=finish_reason,
            reasoning_content=REASONING_SECRET,
            usage={
                "prompt_tokens": 3,
                "completion_tokens": 2,
                "total_tokens": 5,
            },
            post_counter=posts,
        )
    )

    with pytest.raises(ModelSchemaError) as raised:
        client.complete_model("task", {}, SingleAgentLiveOutput)

    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    diagnostic = raised.value.diagnostic
    assert diagnostic is not None
    assert diagnostic.failure_category == category
    expected_length = (
        len(content)
        if isinstance(content, str) and content_length is None
        else content_length
    )
    assert diagnostic.response_content_length == expected_length
    assert diagnostic.finish_reason == finish_reason
    assert posts == [1]
    assert client.telemetry.calls == 1
    assert client.telemetry.total_tokens == 5
    assert client.telemetry.schema_failures == 1
    assert client.automatic_retry_count == 0
    serialized = diagnostic.model_dump_json() + repr(raised.value)
    assert RAW_CONTENT not in serialized
    assert REASONING_SECRET not in serialized


def test_schema_failure_consumes_budget_without_retry_or_repair_call() -> None:
    posts: list[int] = []
    client = _client(
        _mock_transport(
            "not-json",
            post_counter=posts,
        ),
        max_calls=1,
    )

    with pytest.raises(ModelSchemaError):
        client.complete_model("task", {}, SingleAgentLiveOutput)
    with pytest.raises(ModelCallBudgetExceeded, match="hard limit"):
        client.complete_model("task", {}, SingleAgentLiveOutput)

    assert posts == [1]
    assert client.telemetry.calls == 1
    assert client.telemetry.schema_failures == 1
    assert client.automatic_retry_count == 0


def test_runner_failure_artifact_contains_only_safe_schema_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider_payload = {
        **_BASE_SINGLE,
        "confidence": RAW_CONTENT,
        "reason": CLAIM_SECRET,
        "evidence_span": SOURCE_SECRET,
    }
    client = _client(
        _mock_transport(
            json.dumps(provider_payload),
            reasoning_content=REASONING_SECRET,
            usage={
                "prompt_tokens": 17,
                "completion_tokens": 5,
                "total_tokens": 22,
            },
        ),
        max_calls=54,
    )
    monkeypatch.setenv("OPENAI_API_KEY", API_SECRET)
    monkeypatch.setattr(
        eval_runner,
        "OpenAICompatibleClient",
        lambda **_kwargs: client,
    )
    output = tmp_path / "schema-failure"

    with pytest.raises(ModelSchemaError):
        run_eval(
            V2_DEV_DATASET,
            output,
            selected_split="dev",
            model_id="deepseek-v4-flash",
            live_requested=True,
        )

    assert {item.name for item in output.iterdir()} == {"failed_attempt.json"}
    raw_text = (output / "failed_attempt.json").read_text(encoding="utf-8")
    artifact = EvalFailureArtifact.model_validate_json(raw_text)
    assert artifact.artifact_version == "phase3g-schema-recovery-failure-v6"
    assert artifact.error_type == "ModelSchemaError"
    assert artifact.failure_kind == "schema"
    assert artifact.maximum_expected_live_model_calls == 108
    assert artifact.actual_model_calls == 2
    assert artifact.automatic_retry_count == 1
    assert artifact.completed_live_predictions == 0
    assert artifact.model_telemetry.total_tokens == 44
    assert artifact.model_telemetry.schema_failures == 2
    assert artifact.schema_recovery.first_attempt_schema_failures == 1
    assert artifact.schema_recovery.schema_retry_calls == 1
    assert artifact.schema_recovery.recovered_schema_failures == 0
    assert artifact.schema_recovery.unrecovered_schema_failures == 1
    assert artifact.schema_recovery.retry_total_tokens == 22
    assert [event.attempt_index for event in artifact.model_call_events] == [1, 2]
    assert artifact.schema_diagnostic is not None
    assert artifact.schema_diagnostic.schema_name == "SingleAgentLiveOutput"
    assert artifact.schema_diagnostic.failure_category == "wrong_type"
    assert artifact.schema_diagnostic.finish_reason == "stop"
    assert artifact.schema_diagnostic.response_content_length == len(
        json.dumps(provider_payload)
    )

    for forbidden in (
        API_SECRET,
        RAW_CONTENT,
        REASONING_SECRET,
        CLAIM_SECRET,
        SOURCE_SECRET,
        "authorization",
        "request_headers",
        "reasoning_content",
        "raw_content",
    ):
        assert forbidden.casefold() not in raw_text.casefold()


def test_historical_failure_artifact_remains_v1_without_invented_diagnostics() -> None:
    raw = HISTORICAL_FAILURE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == (
        "e6b7fda4daefe7d3f98911a238c7fdc492c71ab57a4a5d39406274d5f43a640c"
    )
    artifact = EvalFailureArtifact.model_validate_json(raw)

    assert artifact.artifact_version == "phase3d-live-failure-v1"
    assert artifact.schema_diagnostic is None


def _safe_diagnostic() -> ModelSchemaDiagnostic:
    return ModelSchemaDiagnostic(
        schema_name="SingleAgentLiveOutput",
        failure_category="invalid_json",
        issues=(SchemaValidationIssue(error_type="json_invalid"),),
        safe_message="Provider response content was not valid JSON.",
        finish_reason="stop",
        response_content_length=12,
    )


def test_non_json_provider_envelope_drops_decoder_exception_context() -> None:
    raw_response = "provider-envelope-" + RAW_CONTENT

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            request=request,
            content=raw_response.encode(),
            headers={"Content-Type": "application/json"},
        )

    client = _client(httpx.MockTransport(handler))
    with pytest.raises(ModelSchemaError) as raised:
        client.complete_model("task", {}, SingleAgentLiveOutput)

    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    diagnostic = raised.value.diagnostic
    assert diagnostic is not None
    assert diagnostic.failure_category == "invalid_envelope"
    assert diagnostic.response_content_length == len(raw_response)
    assert RAW_CONTENT not in diagnostic.model_dump_json()
    assert client.telemetry.schema_failures == 1
    assert client.automatic_retry_count == 0


def test_unknown_finish_reason_is_redacted_to_allowlisted_value() -> None:
    client = _client(
        _mock_transport(
            json.dumps({**_BASE_SINGLE, "confidence": RAW_CONTENT}),
            finish_reason=PROVIDER_FINISH_MARKER,
        )
    )

    with pytest.raises(ModelSchemaError) as raised:
        client.complete_model("task", {}, SingleAgentLiveOutput)

    diagnostic = raised.value.diagnostic
    assert diagnostic is not None
    assert diagnostic.finish_reason == "unknown"
    assert PROVIDER_FINISH_MARKER not in diagnostic.model_dump_json()


@pytest.mark.parametrize(
    ("field", "unsafe_value"),
    [
        ("schema_name", RAW_CONTENT),
        ("safe_message", RAW_CONTENT),
        ("finish_reason", PROVIDER_FINISH_MARKER),
    ],
)
def test_diagnostic_top_level_fields_reject_untrusted_text(
    field: str,
    unsafe_value: str,
) -> None:
    raw = _safe_diagnostic().model_dump()
    raw[field] = unsafe_value

    with pytest.raises(ValidationError):
        ModelSchemaDiagnostic.model_validate(raw)


@pytest.mark.parametrize("field", ["location", "error_type"])
def test_diagnostic_issue_fields_reject_untrusted_text(field: str) -> None:
    raw: dict[str, object] = {
        "location": (),
        "error_type": "json_invalid",
    }
    raw[field] = (RAW_CONTENT,) if field == "location" else RAW_CONTENT

    with pytest.raises(ValidationError):
        SchemaValidationIssue.model_validate(raw)


def test_diagnostic_revalidates_constructed_instances() -> None:
    unsafe = ModelSchemaDiagnostic.model_construct(
        schema_name=RAW_CONTENT,
        failure_category="invalid_json",
        issues=(),
        safe_message=RAW_CONTENT,
        finish_reason=PROVIDER_FINISH_MARKER,
        response_content_length=1,
    )

    with pytest.raises(ValidationError):
        ModelSchemaDiagnostic.model_validate(unsafe)


def test_model_schema_error_message_cannot_be_provider_controlled() -> None:
    with pytest.raises(TypeError):
        ModelSchemaError(RAW_CONTENT)  # type: ignore[misc]

    error = ModelSchemaError(diagnostic=_safe_diagnostic())
    assert str(error) == "model response did not satisfy the required JSON schema"
    assert RAW_CONTENT not in repr(error)


def test_transport_failure_drops_request_exception_context() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(RAW_CONTENT, request=request)

    client = _client(httpx.MockTransport(handler))
    with pytest.raises(ModelTransportError) as raised:
        client.complete_model("task", {}, SingleAgentLiveOutput)

    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    assert RAW_CONTENT not in repr(raised.value)
    assert API_SECRET not in repr(raised.value)
    assert client.telemetry.transport_failures == 1


def test_historical_v1_artifact_rejects_invented_schema_diagnostic() -> None:
    raw = json.loads(HISTORICAL_FAILURE.read_text(encoding="utf-8"))
    raw["schema_diagnostic"] = _safe_diagnostic().model_dump(mode="json")

    with pytest.raises(
        ValidationError,
        match="historical v1 artifacts cannot contain diagnostics",
    ):
        EvalFailureArtifact.model_validate(raw)
