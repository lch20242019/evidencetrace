from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import BaseModel, ConfigDict

import evidencetrace.eval.runner as eval_runner
from evidencetrace.agents.judge import JudgeScopeError
from evidencetrace.checks.deterministic import DeterministicConflictError
from evidencetrace.eval.baselines import run_baseline
from evidencetrace.eval.errors import LocalValidationError
from evidencetrace.eval.metrics import engineering_metrics
from evidencetrace.eval.models import (
    EvalCase,
    EvalFailureArtifact,
    EvalPrediction,
    SourceFixture,
)
from evidencetrace.model_client import (
    SCHEMA_RECOVERY_POLICY_VERSION,
    ModelCallBudgetExceeded,
    ModelCallTelemetry,
    ModelSchemaDiagnostic,
    ModelSchemaError,
    ModelTransportError,
    OpenAICompatibleClient,
    SchemaRecoveryClient,
    SchemaValidationIssue,
)

ROOT = Path(__file__).parents[1]
DEV_DATASET = ROOT / "eval_sets/phase3g_dev_zh/dev.jsonl"
RUN_03_FAILURE = ROOT / "eval_runs/phase3g_adaptive_zh_dev_run_03/failed_attempt.json"
READINESS_DIR = ROOT / "eval_runs/phase3g_schema_recovery_readiness"
RUN_03_FAILURE_SHA256 = (
    "2b8e926625cd87556fe6aab0a8036344678eeb337e3a470d28a60ca97b55f127"
)
RAW_FAILURE = "raw-provider-output-must-not-survive"
REQUEST_SECRET = "request-content-must-not-enter-safe-telemetry"
_SCHEMA_INVALID = object()
_TRANSPORT_FAILURE = object()


class _Output(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: int


Outcome = object | dict[str, Any] | Callable[[str, dict[str, Any]], dict[str, Any]]


class _SequenceModel:
    """Deterministic provider fake with one telemetry event per attempted call."""

    model_id = "offline-schema-sequence"
    prompt_version = "offline-schema-sequence-v1"
    provider_id = "offline"
    temperature = 0.0

    def __init__(
        self,
        outcomes: list[Outcome],
        *,
        maximum_calls: int | None = None,
    ) -> None:
        self.outcomes = list(outcomes)
        self.maximum_calls = maximum_calls
        self.telemetry_events: tuple[ModelCallTelemetry, ...] = ()
        self.requests: list[tuple[str, dict[str, Any], type[BaseModel]]] = []
        self._private_failed_content = RAW_FAILURE

    def can_start_model_call(self) -> bool:
        return (
            self.maximum_calls is None
            or len(self.telemetry_events) < self.maximum_calls
        )

    @staticmethod
    def _diagnostic() -> ModelSchemaDiagnostic:
        return ModelSchemaDiagnostic(
            schema_name="UnknownOutputSchema",
            failure_category="missing_field",
            issues=(SchemaValidationIssue(error_type="missing"),),
            safe_message="A required output field was missing.",
            finish_reason="stop",
            response_content_length=37,
        )

    def complete_model(
        self,
        task: str,
        payload: dict[str, Any],
        schema: type[BaseModel],
    ) -> BaseModel:
        if not self.can_start_model_call():
            raise ModelCallBudgetExceeded(
                "model call hard limit reached before provider request"
            )
        if not self.outcomes:
            raise AssertionError("deterministic sequence was exhausted")
        self.requests.append((task, payload, schema))
        outcome = self.outcomes.pop(0)
        call_number = len(self.telemetry_events) + 1
        failure_kind = (
            "schema"
            if outcome is _SCHEMA_INVALID
            else ("transport" if outcome is _TRANSPORT_FAILURE else "none")
        )
        self.telemetry_events += (
            ModelCallTelemetry(
                latency_ms=float(call_number),
                usage_status="reported",
                input_tokens=10 + call_number,
                output_tokens=2 + call_number,
                total_tokens=12 + 2 * call_number,
                failure_kind=failure_kind,
            ),
        )
        if outcome is _SCHEMA_INVALID:
            raise ModelSchemaError(diagnostic=self._diagnostic()) from None
        if outcome is _TRANSPORT_FAILURE:
            raise ModelTransportError("model request failed") from None
        value = outcome(task, payload) if callable(outcome) else outcome
        return schema.model_validate(value)

    def complete(self, task: str, payload: dict[str, Any]) -> dict[str, Any]:
        result = self.complete_model(task, payload, _Output)
        return result.model_dump(mode="json")


def _case(
    *,
    case_id: str = "phase3g_schema_recovery",
    claim: str = "云栈组件在版本7.4中启用了离线校验。",
    relation: str = "entailed",
) -> EvalCase:
    return EvalCase(
        case_id=case_id,
        claim_text=claim,
        source_fixture="sources/offline.jsonl",
        source_id="phase3g_schema_source",
        source_url="https://phase3g-schema.invalid/source",
        gold_relation=relation,
        gold_evidence_span=(
            "云栈组件在版本7.4中启用了离线校验。"
            if relation in {"entailed", "partially_entailed", "contradicted"}
            else None
        ),
        claim_type="versioned_capability",
        mutation_type="offline_recovery_fixture",
        split="dev",
        provenance="Offline Phase 3G-D regression fixture.",
        annotation_status="provisional",
        annotation_notes="No provider call.",
    )


def _source(
    content: str = "云栈组件在版本7.4中启用了离线校验。",
) -> SourceFixture:
    return SourceFixture(
        source_id="phase3g_schema_source",
        url="https://phase3g-schema.invalid/source",
        content=content,
        provenance="Offline Phase 3G-D regression fixture.",
        content_hash=hashlib.sha256(content.encode()).hexdigest(),
    )


def _valid_live_output(task: str, payload: dict[str, Any]) -> dict[str, Any]:
    if task == "single_agent_full_source_verification":
        source = payload["source"]
        if not source["available"]:
            return {
                "relation": "source_unavailable",
                "confidence": 0.0,
                "reason": "The supplied source is unavailable.",
                "evidence_span": None,
            }
        return {
            "relation": "entailed",
            "confidence": 0.9,
            "reason": "The source directly supports the claim.",
            "evidence_span": source["content"],
        }
    evidence = payload["evidence"][0]
    relation = (
        "contradicted"
        if any(signal["severity"] == "error" for signal in payload["signals"])
        else "entailed"
    )
    return {
        "relation": relation,
        "confidence": 0.9,
        "evidence_span": evidence["text"],
        "reason": "The candidate evidence directly supports the claim.",
    }


def test_first_attempt_success_uses_one_call_and_no_retry() -> None:
    wrapped = _SequenceModel([{"value": 7}])
    client = SchemaRecoveryClient(wrapped)

    assert client.complete_model("contract", {}, _Output).value == 7
    assert len(wrapped.requests) == 1
    assert client.telemetry.calls == 1
    assert client.schema_recovery.model_dump(mode="json") == {
        "policy_version": SCHEMA_RECOVERY_POLICY_VERSION,
        "retry_limit": 1,
        "logical_calls": 1,
        "first_attempt_schema_failures": 0,
        "schema_retry_calls": 0,
        "recovered_schema_failures": 0,
        "unrecovered_schema_failures": 0,
        "first_attempt_contract_success_rate": 1.0,
        "final_operational_success": True,
        "retry_input_tokens": 0,
        "retry_output_tokens": 0,
        "retry_total_tokens": 0,
        "retry_latency_ms": [],
        "retry_cache_status": "not_available",
    }


def test_schema_invalid_then_valid_retries_once_without_failed_content() -> None:
    wrapped = _SequenceModel([_SCHEMA_INVALID, {"value": 11}])
    client = SchemaRecoveryClient(wrapped)

    assert (
        client.complete_model(
            "contract",
            {"request": REQUEST_SECRET},
            _Output,
        ).value
        == 11
    )

    assert len(wrapped.requests) == 2
    first_task, first_payload, first_schema = wrapped.requests[0]
    retry_task, retry_payload, retry_schema = wrapped.requests[1]
    assert retry_task == first_task
    assert retry_schema is first_schema
    assert retry_payload["request"] == first_payload["request"]
    assert retry_payload["schema_recovery"] == {
        "policy_version": SCHEMA_RECOVERY_POLICY_VERSION,
        "attempt_index": 2,
        "reminder": (
            "Schema retry attempt 2 of 2. Return exactly one JSON object "
            "matching the same output_schema. Include every required field and "
            "no extra fields, Markdown, preamble, or trailing text."
        ),
    }
    assert RAW_FAILURE not in json.dumps(retry_payload)
    attempts = [
        (event.attempt_index, event.schema_retry) for event in client.telemetry_events
    ]
    assert attempts == [
        (1, False),
        (2, True),
    ]
    recovery = client.schema_recovery
    assert recovery.first_attempt_schema_failures == 1
    assert recovery.schema_retry_calls == 1
    assert recovery.recovered_schema_failures == 1
    assert recovery.unrecovered_schema_failures == 0
    assert recovery.first_attempt_contract_success_rate == 0.0
    assert recovery.final_operational_success is True
    assert recovery.retry_input_tokens == 12
    assert recovery.retry_output_tokens == 4
    assert recovery.retry_total_tokens == 16
    assert recovery.retry_latency_ms == (2.0,)


def test_two_schema_failures_fail_closed_and_do_not_leak(
    capsys: pytest.CaptureFixture[str],
) -> None:
    wrapped = _SequenceModel([_SCHEMA_INVALID, _SCHEMA_INVALID])
    client = SchemaRecoveryClient(wrapped)

    with pytest.raises(ModelSchemaError) as raised:
        client.complete_model(
            "contract",
            {"request": REQUEST_SECRET},
            _Output,
        )

    recovery = client.schema_recovery
    assert len(wrapped.requests) == 2
    assert recovery.schema_retry_calls == 1
    assert recovery.recovered_schema_failures == 0
    assert recovery.unrecovered_schema_failures == 1
    assert recovery.final_operational_success is False
    captured = capsys.readouterr()
    safe_state = (
        repr(raised.value)
        + recovery.model_dump_json()
        + repr(client.telemetry_events)
        + captured.out
        + captured.err
    )
    assert RAW_FAILURE not in safe_state
    assert REQUEST_SECRET not in safe_state


def test_transport_failure_is_not_retried() -> None:
    wrapped = _SequenceModel([_TRANSPORT_FAILURE, {"value": 7}])
    client = SchemaRecoveryClient(wrapped)

    with pytest.raises(ModelTransportError):
        client.complete_model("contract", {}, _Output)

    assert len(wrapped.requests) == 1
    assert client.schema_recovery.schema_retry_calls == 0
    assert client.schema_recovery.first_attempt_schema_failures == 0


def test_exhausted_pre_request_budget_blocks_schema_retry() -> None:
    wrapped = _SequenceModel(
        [_SCHEMA_INVALID, {"value": 7}],
        maximum_calls=1,
    )
    client = SchemaRecoveryClient(wrapped)

    with pytest.raises(ModelSchemaError):
        client.complete_model("contract", {}, _Output)

    assert len(wrapped.requests) == 1
    assert client.schema_recovery.first_attempt_schema_failures == 1
    assert client.schema_recovery.schema_retry_calls == 0
    assert client.schema_recovery.unrecovered_schema_failures == 1


@pytest.mark.parametrize(
    ("baseline", "content", "expected_route"),
    [
        (
            "single_agent_live",
            "云栈组件在版本7.4中启用了离线校验。",
            None,
        ),
        (
            "retrieval_judge_live",
            "云栈组件在版本7.4中启用了离线校验。",
            None,
        ),
        (
            "adaptive_live",
            ("维护说明记录了轮值流程。\n\n云栈组件在版本7.4中启用了离线校验。"),
            "retrieval_judge",
        ),
    ],
)
def test_single_judge_and_adaptive_share_schema_recovery(
    baseline: str,
    content: str,
    expected_route: str | None,
) -> None:
    wrapped = _SequenceModel([_SCHEMA_INVALID, _valid_live_output])
    client = SchemaRecoveryClient(wrapped)

    prediction = run_baseline(
        _case(),
        _source(content),
        baseline,  # type: ignore[arg-type]
        client=client,
    )

    assert prediction.predicted_relation.value == "entailed"
    assert prediction.model_calls == 2
    assert prediction.schema_failure_count == 1
    assert prediction.selected_route == expected_route
    recovery = client.schema_recovery
    assert recovery.first_attempt_schema_failures == 1
    assert recovery.schema_retry_calls == 1
    assert recovery.recovered_schema_failures == 1
    assert recovery.unrecovered_schema_failures == 0
    assert recovery.first_attempt_contract_success_rate == 0.0
    assert recovery.final_operational_success is True
    assert recovery.retry_input_tokens == 12
    assert recovery.retry_output_tokens == 4
    assert recovery.retry_total_tokens == 16
    assert recovery.retry_latency_ms == (2.0,)
    metrics = engineering_metrics(
        [
            {
                **prediction.model_dump(mode="json"),
                "gold_relation": "entailed",
            }
        ]
    )
    assert metrics["schema_failure_count"] == 1
    assert metrics["model_call_count"] == 2


def _judge_response(
    task: str,
    payload: dict[str, Any],
    *,
    evidence_text: str | None = None,
    relation: str = "entailed",
) -> dict[str, Any]:
    assert task == "claim_judgement"
    evidence = payload["evidence"][0]
    span = (
        evidence_text or evidence["text"]
        if relation in {"entailed", "partially_entailed", "contradicted"}
        else None
    )
    return {
        "relation": relation,
        "confidence": 0.8,
        "evidence_span": span,
        "reason": "Schema-valid deterministic response.",
    }


def _scope_failure_response(task: str, payload: dict[str, Any]) -> dict[str, Any]:
    return _judge_response(
        task,
        payload,
        evidence_text="不在候选上下文中的伪造证据。",
    )


def _guard_failure_response(task: str, payload: dict[str, Any]) -> dict[str, Any]:
    return _judge_response(task, payload)


def _single_local_failure_response(
    _task: str,
    _payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "relation": "entailed",
        "confidence": 0.8,
        "reason": "Schema-valid but out-of-scope evidence.",
        "evidence_span": "不在 source 中的证据。",
    }


@pytest.mark.parametrize("failure", ["scope", "guard", "single_local"])
def test_scope_guard_and_local_validation_do_not_retry(failure: str) -> None:
    if failure == "scope":
        response = _scope_failure_response
        baseline = "retrieval_judge_live"
        case = _case()
        source = _source()
        expected_error = JudgeScopeError
    elif failure == "guard":
        source_text = "云栈组件在版本7.3中启用了离线校验。"
        response = _guard_failure_response
        baseline = "retrieval_judge_live"
        case = _case()
        source = _source(source_text)
        expected_error = DeterministicConflictError
    else:
        response = _single_local_failure_response
        baseline = "single_agent_live"
        case = _case()
        source = _source()
        expected_error = LocalValidationError
    wrapped = _SequenceModel([response])
    client = SchemaRecoveryClient(wrapped)

    with pytest.raises(expected_error):
        run_baseline(
            case,
            source,
            baseline,  # type: ignore[arg-type]
            client=client,
        )

    assert len(wrapped.requests) == 1
    assert client.schema_recovery.schema_retry_calls == 0
    assert client.schema_recovery.first_attempt_schema_failures == 0


def test_schema_valid_semantic_disagreement_is_scored_without_retry() -> None:
    wrapped = _SequenceModel(
        [
            lambda task, payload: _judge_response(
                task,
                payload,
                relation="not_in_source",
            )
        ]
    )
    client = SchemaRecoveryClient(wrapped)

    prediction = run_baseline(
        _case(),
        _source(),
        "retrieval_judge_live",
        client=client,
    )

    assert prediction.predicted_relation.value == "not_in_source"
    assert prediction.model_calls == 1
    assert client.schema_recovery.schema_retry_calls == 0


def test_real_adapter_retry_keeps_schema_and_drops_first_response() -> None:
    requests: list[dict[str, Any]] = []
    responses = iter(
        [
            "not-json-" + RAW_FAILURE,
            json.dumps({"value": 19}),
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            request=request,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": next(responses)},
                    }
                ],
                "usage": {
                    "prompt_tokens": 8,
                    "completion_tokens": 3,
                    "total_tokens": 11,
                },
            },
        )

    base = OpenAICompatibleClient(
        model_id="offline-mock",
        api_key="offline-key",
        base_url="https://provider.invalid/v1",
        max_calls=2,
        transport=httpx.MockTransport(handler),
    )
    client = SchemaRecoveryClient(base)

    assert (
        client.complete_model(
            "contract",
            {"request": REQUEST_SECRET},
            _Output,
        ).value
        == 19
    )

    assert len(requests) == 2
    first = json.loads(requests[0]["messages"][1]["content"])
    second = json.loads(requests[1]["messages"][1]["content"])
    assert second["output_contract"] == first["output_contract"]
    assert second["input"]["request"] == first["input"]["request"]
    assert RAW_FAILURE not in requests[1]["messages"][1]["content"]
    assert client.schema_recovery.recovered_schema_failures == 1


def test_runner_writes_safe_unrecovered_schema_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    wrapped = _SequenceModel([_SCHEMA_INVALID, _SCHEMA_INVALID])
    monkeypatch.setenv("OPENAI_API_KEY", "offline-placeholder")
    monkeypatch.setattr(
        eval_runner,
        "OpenAICompatibleClient",
        lambda **_kwargs: wrapped,
    )
    output = tmp_path / "schema-failure"

    with pytest.raises(ModelSchemaError):
        eval_runner.run_eval(
            DEV_DATASET,
            output,
            selected_split="dev",
            model_id=wrapped.model_id,
            live_requested=True,
        )

    assert {path.name for path in output.iterdir()} == {"failed_attempt.json"}
    raw_text = output.joinpath("failed_attempt.json").read_text(encoding="utf-8")
    artifact = EvalFailureArtifact.model_validate_json(raw_text)
    assert artifact.artifact_version == "phase3g-schema-recovery-failure-v6"
    assert artifact.failed_baseline == "single_agent_live"
    assert artifact.failed_case_id == "phase3g_zh_dev_001"
    assert artifact.failure_kind == "schema"
    assert artifact.actual_model_calls == 2
    assert artifact.automatic_retry_count == 1
    assert artifact.schema_recovery.first_attempt_schema_failures == 1
    assert artifact.schema_recovery.schema_retry_calls == 1
    assert artifact.schema_recovery.recovered_schema_failures == 0
    assert artifact.schema_recovery.unrecovered_schema_failures == 1
    assert artifact.schema_recovery.final_operational_success is False
    assert [event.attempt_index for event in artifact.model_call_events] == [1, 2]
    first_case = json.loads(DEV_DATASET.read_text(encoding="utf-8").splitlines()[0])
    captured = capsys.readouterr()
    safe_text = raw_text + captured.out + captured.err
    for forbidden in (
        RAW_FAILURE,
        REQUEST_SECRET,
        first_case["claim_text"],
        "authorization",
        "offline-placeholder",
    ):
        assert forbidden.casefold() not in safe_text.casefold()


def test_successful_run_records_recovery_in_canonical_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapped = _SequenceModel([_SCHEMA_INVALID, *([_valid_live_output] * 500)])
    monkeypatch.setenv("OPENAI_API_KEY", "offline-placeholder")
    monkeypatch.setattr(
        eval_runner,
        "OpenAICompatibleClient",
        lambda **_kwargs: wrapped,
    )

    output = eval_runner.run_eval(
        DEV_DATASET,
        tmp_path / "recovered-run",
        selected_split="dev",
        model_id=wrapped.model_id,
        live_requested=True,
    )

    metrics = json.loads(output.joinpath("metrics.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        output.joinpath("run_manifest.json").read_text(encoding="utf-8")
    )
    recovery = metrics["live_execution"]["schema_recovery"]
    assert recovery["logical_calls"] == 168
    assert recovery["first_attempt_schema_failures"] == 1
    assert recovery["schema_retry_calls"] == 1
    assert recovery["recovered_schema_failures"] == 1
    assert recovery["unrecovered_schema_failures"] == 0
    assert recovery["first_attempt_contract_success_rate"] == pytest.approx(167 / 168)
    assert recovery["final_operational_success"] is True
    assert recovery["retry_total_tokens"] == 16
    assert recovery["retry_latency_ms"] == [2.0]
    assert recovery["retry_cache_status"] == "not_available"
    assert metrics["live_execution"]["actual_model_calls"] == 169
    assert manifest["automatic_retry_count"] == 1
    assert manifest["schema_retry_limit"] == 1


def test_historical_failure_and_prediction_contracts_remain_readable() -> None:
    raw = RUN_03_FAILURE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == RUN_03_FAILURE_SHA256
    historical_failure = json.loads(raw)
    assert historical_failure["schema_version"] == "phase3g-live-failure-v1"
    assert historical_failure["failure_code"] == "schema"
    assert historical_failure["retry_count"] == 0

    prediction = EvalPrediction.model_validate(
        {
            "case_id": "historical_prediction",
            "baseline": "single_agent_live",
            "predicted_relation": "not_in_source",
            "confidence": 0.0,
            "model_calls": 0,
            "latency_ms": 0.0,
        }
    )
    assert prediction.model_calls == 0


def test_offline_diagnosis_and_recovery_policy_are_bounded() -> None:
    diagnosis = json.loads(
        READINESS_DIR.joinpath("failure_diagnosis.json").read_text(encoding="utf-8")
    )
    policy = json.loads(
        READINESS_DIR.joinpath("recovery_policy.json").read_text(encoding="utf-8")
    )
    preregistration = json.loads(
        READINESS_DIR.joinpath("preregistration.json").read_text(encoding="utf-8")
    )

    assert diagnosis["model_calls"] == 0
    assert diagnosis["historical_run_reexecuted"] is False
    assert diagnosis["failure"]["classification"] == "unknown_schema_failure"
    assert diagnosis["failure"]["local_validator_bug"] is False
    retained = diagnosis["retained_contract_detail"]
    assert retained["finish_reason"] == "unknown_not_retained"
    assert retained["pydantic_error_code"] == "unknown_not_retained"
    for relative, expected in diagnosis["historical_artifact_sha256"].items():
        actual = hashlib.sha256(ROOT.joinpath(relative).read_bytes()).hexdigest()
        assert actual == expected

    assert policy["schema_retry_limit"] == 1
    assert policy["maximum_provider_calls_per_live_case"] == 2
    assert policy["retry_trigger"]["exception_type"] == "ModelSchemaError"
    assert policy["retry_request"]["includes_prior_model_output"] is False
    assert policy["cache"]["failed_response_cache_write"] is False
    assert policy["safety"]["raw_response_retained"] is False

    freeze = preregistration["freeze"]

    def bundle_hash_at_commit(paths: list[str], commit: str) -> str:
        entries = []
        for path in paths:
            raw = subprocess.run(
                ["git", "show", f"{commit}:{path}"],
                cwd=ROOT,
                check=True,
                capture_output=True,
            ).stdout
            entries.append(f"{hashlib.sha256(raw).hexdigest()}  {path}\n")
        return hashlib.sha256("".join(entries).encode()).hexdigest()

    freeze_commit = "060db3b066f19093a4b8cd97edfa4b47e73e519a"
    assert (
        bundle_hash_at_commit(freeze["code_paths"], freeze_commit)
        == freeze["code_sha256"]
    )
    assert (
        bundle_hash_at_commit(freeze["prompt_schema_paths"], freeze_commit)
        == freeze["prompt_schema_sha256"]
    )
    policy_hash = hashlib.sha256(
        READINESS_DIR.joinpath("recovery_policy.json").read_bytes()
    ).hexdigest()
    assert policy_hash == freeze["schema_recovery_policy_sha256"]
    combined = json.dumps(
        {
            "base_config_router_retrieval_sha256": freeze[
                "base_config_router_retrieval_sha256"
            ],
            "schema_recovery_policy_sha256": policy_hash,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    assert (
        hashlib.sha256(combined.encode()).hexdigest()
        == freeze["config_recovery_policy_sha256"]
    )
    conditions = preregistration["readiness_conditions"]
    assert conditions["operational_successful_runs_required"] == 3
    assert conditions["unrecovered_schema_failures_maximum"] == 0
    assert conditions["first_attempt_contract_success_rate_minimum"] == 0.99
    assert conditions["recovered_schema_failures_per_run_maximum"] == 2
