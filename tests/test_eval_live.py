from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

import evidencetrace.eval.runner as eval_runner
from evidencetrace.eval.models import EvalFailureArtifact
from evidencetrace.eval.runner import run_eval
from evidencetrace.model_client import ModelCallTelemetry

ROOT = Path(__file__).parents[1]
DATASET = ROOT / "eval_sets/core.jsonl"
V2_DEV_DATASET = ROOT / "eval_sets/v2/dev.jsonl"
FAILURE_API_KEY = "private-live-failure-key"
FAILURE_CONTENT_MARKER = "sensitive-request-response-content"
FIRST_JUDGE_FAILURE_MARKER = "first-judge-call-failure"
POST_PROCESSING_FAILURE_MARKER = "sensitive-post-processing-content"
CANONICAL_ARTIFACTS = {
    "eval_report.md",
    "eval_results.jsonl",
    "metrics.json",
    "run_manifest.json",
}


class FakeLiveClient:
    model_id = "fake-live-model"
    prompt_version = "fake-live-v1"
    provider_id = "openai-compatible"
    temperature = 0.0
    temperature_effective = None
    thinking_mode = "provider_default"
    automatic_retry_count = 0

    def __init__(self) -> None:
        self.telemetry_events: tuple[ModelCallTelemetry, ...] = ()

    def complete_model(
        self, task: str, payload: dict[str, Any], schema: type[Any]
    ) -> Any:
        self.telemetry_events += (
            ModelCallTelemetry(
                latency_ms=1.0,
                usage_status="reported",
                input_tokens=2,
                output_tokens=1,
                total_tokens=3,
            ),
        )
        if task == "single_agent_full_source_verification":
            source = payload["source"]
            return schema.model_validate(
                {
                    "relation": (
                        "entailed" if source["available"] else "source_unavailable"
                    ),
                    "confidence": 0.8 if source["available"] else 0.0,
                    "reason": "Deterministic schema-valid live baseline response.",
                    "evidence_span": source["content"] if source["available"] else None,
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
                            "claim_type": "factual_statement",
                            "checkability": "checkable",
                            "citation_urls": payload["citation_urls"],
                        }
                    ],
                    "model_id": self.model_id,
                    "prompt_version": self.prompt_version,
                }
            )
        evidence = payload["evidence"][0]
        conflict = any(
            signal["severity"] == "error" for signal in payload.get("signals", [])
        )
        relation = "contradicted" if conflict else "entailed"
        return schema.model_validate(
            {
                "relation": relation,
                "confidence": 0.8,
                "evidence_span": evidence["text"],
                "reason": "Deterministic fake live response.",
            }
        )


class MidRunFailureClient:
    model_id = "fake-live-model"
    prompt_version = "fake-live-v1"
    provider_id = "openai-compatible"
    temperature = 0.0
    temperature_effective = None
    thinking_mode = "provider_default"
    automatic_retry_count = 0

    def __init__(self) -> None:
        self.telemetry_events: tuple[ModelCallTelemetry, ...] = ()

    def complete_model(
        self, task: str, payload: dict[str, Any], schema: type[Any]
    ) -> Any:
        call_number = len(self.telemetry_events) + 1
        self.telemetry_events += (
            ModelCallTelemetry(
                latency_ms=float(call_number * 5),
                usage_status="reported",
                input_tokens=11,
                output_tokens=3,
                total_tokens=14,
            ),
        )
        if call_number == 2:
            raise ValueError(
                f"Authorization: Bearer {FAILURE_API_KEY}; {FAILURE_CONTENT_MARKER}"
            )
        assert task == "single_agent_full_source_verification"
        source = payload["source"]
        return schema.model_validate(
            {
                "relation": (
                    "entailed" if source["available"] else "source_unavailable"
                ),
                "confidence": 0.8 if source["available"] else 0.0,
                "reason": "Schema-valid response before a later local failure.",
                "evidence_span": source["content"] if source["available"] else None,
            }
        )


class FirstJudgeFailureClient(FakeLiveClient):
    def complete_model(
        self, task: str, payload: dict[str, Any], schema: type[Any]
    ) -> Any:
        if task == "claim_judgement":
            assert len(self.telemetry_events) == 18
            self.telemetry_events += (
                ModelCallTelemetry(
                    latency_ms=1.0,
                    usage_status="reported",
                    input_tokens=2,
                    output_tokens=1,
                    total_tokens=3,
                ),
            )
            raise ValueError(FIRST_JUDGE_FAILURE_MARKER)
        return super().complete_model(task, payload, schema)


def test_explicit_live_mode_uses_openai_compatible_client(
    tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret")
    monkeypatch.setattr(
        "evidencetrace.eval.runner.OpenAICompatibleClient",
        lambda **_kwargs: FakeLiveClient(),
    )

    output = run_eval(
        DATASET,
        tmp_path / "run",
        selected_split="dev",
        model_id="fake-live-model",
        live_requested=True,
    )
    manifest = json.loads(output.joinpath("run_manifest.json").read_text())
    metrics = json.loads(output.joinpath("metrics.json").read_text())
    records = [
        json.loads(line)
        for line in output.joinpath("eval_results.jsonl").read_text().splitlines()
    ]

    assert manifest["live_baseline"] == "executed"
    assert manifest["live_baseline_statuses"] == {
        "single_agent_live": "executed",
        "retrieval_judge_live": "executed",
        "adaptive_live": "executed",
    }
    assert manifest["model_temperature"] == 0.0
    assert manifest["model_temperature_actual_effective"] is None
    assert manifest["model_temperature_effective_status"] == "unknown"
    assert manifest["model_temperature_semantics"] == "requested_only"
    assert manifest["model_thinking_mode"] == "provider_default"
    assert manifest["automatic_retry_count"] == 0
    assert manifest["schema_recovery_policy_version"] == "schema-recovery-v1"
    assert manifest["schema_retry_limit"] == 1
    assert (
        manifest["retrieval_config"]["judge_verdict_ownership_policy_version"]
        == "local-verdict-ownership-v1"
    )

    live_execution = metrics["live_execution"]
    assert live_execution["temperature"] == 0.0
    assert live_execution["temperature_requested"] == 0.0
    assert live_execution["temperature_actual_effective"] is None
    assert live_execution["temperature_effective_status"] == "unknown"
    assert live_execution["temperature_semantics"] == "requested_only"
    assert live_execution["thinking_mode"] == "provider_default"
    assert live_execution["same_provider_model_temperature_thinking_mode"] is True
    assert live_execution["hard_call_limit_enforced"] is True
    assert live_execution["automatic_retry_count"] == 0
    assert live_execution["schema_recovery"]["retry_limit"] == 1
    assert live_execution["schema_recovery"]["schema_retry_calls"] == 0
    assert live_execution["schema_recovery"]["retry_cache_status"] == "not_available"

    assert sum(record["model_calls"] for record in records) > 0
    assert "test-secret" not in output.joinpath("run_manifest.json").read_text()
    assert {path.name for path in output.iterdir()} == CANONICAL_ARTIFACTS


def test_midrun_failure_writes_only_strict_secret_free_failure_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", FAILURE_API_KEY)
    client = MidRunFailureClient()
    monkeypatch.setattr(
        "evidencetrace.eval.runner.OpenAICompatibleClient",
        lambda **_kwargs: client,
    )
    output = tmp_path / "failed"

    with pytest.raises(ValueError, match=FAILURE_CONTENT_MARKER):
        run_eval(
            V2_DEV_DATASET,
            output,
            selected_split="dev",
            model_id=client.model_id,
            live_requested=True,
        )

    failure_path = output / "failed_attempt.json"
    artifact = EvalFailureArtifact.model_validate_json(
        failure_path.read_text(encoding="utf-8")
    )
    assert artifact.status == "failed"
    assert artifact.failure_kind == "unexpected"
    assert artifact.failure_code == "unexpected_exception"
    assert artifact.error_type == "ValueError"
    assert artifact.failed_baseline == "single_agent_live"
    assert artifact.failed_case_id == "v2_dev_002"
    assert artifact.model_temperature_requested == 0.0
    assert artifact.model_temperature_actual_effective is None
    assert artifact.model_temperature_effective_status == "unknown"
    assert artifact.model_temperature_semantics == "requested_only"
    assert artifact.model_thinking_mode == "provider_default"
    assert artifact.same_model_and_thinking_mode is True
    assert artifact.maximum_expected_live_model_calls == 108
    assert artifact.actual_model_calls == 2
    assert artifact.actual_model_calls_by_baseline == {
        "single_agent_live": 2,
        "retrieval_judge_live": 0,
        "adaptive_live": 0,
    }
    assert artifact.completed_live_predictions == 1
    assert artifact.calls_with_reported_usage == 2
    assert artifact.reported_input_token_subtotal == 22
    assert artifact.reported_output_token_subtotal == 6
    assert artifact.reported_total_token_subtotal == 28
    assert artifact.model_telemetry.total_tokens == 28
    assert artifact.model_telemetry.call_latencies_ms == (5.0, 10.0)
    assert artifact.model_latency_ms_p50 == 7.5
    assert artifact.model_latency_ms_p95 == 10.0
    assert artifact.canonical_artifacts_written is False
    assert artifact.hard_limit_enforced is True
    assert artifact.automatic_retry_count == 0
    assert artifact.schema_recovery.final_operational_success is False

    assert not any((output / name).exists() for name in CANONICAL_ARTIFACTS)
    serialized = "\n".join(
        path.read_text(encoding="utf-8") for path in output.iterdir() if path.is_file()
    ).casefold()
    for forbidden in (
        FAILURE_API_KEY,
        FAILURE_CONTENT_MARKER,
        "authorization",
        "bearer",
    ):
        assert forbidden.casefold() not in serialized

    raw = json.loads(failure_path.read_text(encoding="utf-8"))
    raw["request_headers"] = {"Authorization": f"Bearer {FAILURE_API_KEY}"}
    with pytest.raises(ValidationError):
        EvalFailureArtifact.model_validate(raw)


def test_live_post_processing_failure_preserves_all_safe_telemetry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", FAILURE_API_KEY)
    client = FakeLiveClient()
    monkeypatch.setattr(
        "evidencetrace.eval.runner.OpenAICompatibleClient",
        lambda **_kwargs: client,
    )
    original_report = eval_runner._baseline_report

    def fail_after_live_report(
        records: list[dict[str, Any]], *, headline_split: str
    ) -> dict[str, Any]:
        report = original_report(records, headline_split=headline_split)
        if records and records[0]["baseline"] == "single_agent_live":
            raise RuntimeError(
                "Authorization: Bearer "
                f"{FAILURE_API_KEY}; {POST_PROCESSING_FAILURE_MARKER}"
            )
        return report

    monkeypatch.setattr(eval_runner, "_baseline_report", fail_after_live_report)
    output = tmp_path / "post-processing-failed"

    with pytest.raises(RuntimeError, match=POST_PROCESSING_FAILURE_MARKER):
        run_eval(
            V2_DEV_DATASET,
            output,
            selected_split="dev",
            model_id=client.model_id,
            live_requested=True,
        )

    failure_path = output / "failed_attempt.json"
    artifact = EvalFailureArtifact.model_validate_json(
        failure_path.read_text(encoding="utf-8")
    )
    assert artifact.failure_stage == "baseline_post_processing"
    assert artifact.failure_kind == "unexpected"
    assert artifact.failure_code == "unexpected_exception"
    assert artifact.error_type == "RuntimeError"
    assert artifact.failed_baseline == "single_agent_live"
    assert artifact.failed_case_id == "v2_dev_018"
    assert artifact.maximum_expected_live_model_calls == 108
    assert artifact.actual_model_calls == 18
    assert artifact.actual_model_calls_by_baseline == {
        "single_agent_live": 18,
        "retrieval_judge_live": 0,
        "adaptive_live": 0,
    }
    assert artifact.completed_live_predictions == 18
    assert artifact.calls_with_reported_usage == 18
    assert artifact.reported_input_token_subtotal == 36
    assert artifact.reported_output_token_subtotal == 18
    assert artifact.reported_total_token_subtotal == 54
    assert artifact.model_telemetry.total_tokens == 54
    assert artifact.model_telemetry.call_latencies_ms == (1.0,) * 18
    assert len(artifact.model_call_events) == 18
    assert artifact.model_latency_ms_p50 == 1.0
    assert artifact.model_latency_ms_p95 == 1.0
    assert artifact.canonical_artifacts_written is False

    assert not any((output / name).exists() for name in CANONICAL_ARTIFACTS)
    serialized = "\n".join(
        path.read_text(encoding="utf-8") for path in output.iterdir() if path.is_file()
    ).casefold()
    for forbidden in (
        FAILURE_API_KEY,
        POST_PROCESSING_FAILURE_MARKER,
        "authorization",
        "bearer",
    ):
        assert forbidden.casefold() not in serialized


def test_failed_first_judge_call_is_allocated_after_completed_single_agent_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", FAILURE_API_KEY)
    client = FirstJudgeFailureClient()
    monkeypatch.setattr(
        "evidencetrace.eval.runner.OpenAICompatibleClient",
        lambda **_kwargs: client,
    )
    output = tmp_path / "first-judge-failed"

    with pytest.raises(ValueError, match=FIRST_JUDGE_FAILURE_MARKER):
        run_eval(
            V2_DEV_DATASET,
            output,
            selected_split="dev",
            model_id=client.model_id,
            live_requested=True,
        )

    artifact = EvalFailureArtifact.model_validate_json(
        (output / "failed_attempt.json").read_text(encoding="utf-8")
    )
    assert artifact.failure_stage == "baseline_case"
    assert artifact.failed_baseline == "retrieval_judge_live"
    assert artifact.failed_case_id == "v2_dev_001"
    assert artifact.actual_model_calls == 19
    assert artifact.actual_model_calls_by_baseline == {
        "single_agent_live": 18,
        "retrieval_judge_live": 1,
        "adaptive_live": 0,
    }
    assert artifact.completed_live_predictions == 18
    assert artifact.model_telemetry.calls == 19
    assert len(artifact.model_call_events) == 19
