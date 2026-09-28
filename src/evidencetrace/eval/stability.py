"""Aggregate independent dev runs without retaining model-authored text."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from evidencetrace.eval.baselines import run_baseline
from evidencetrace.eval.dataset import load_dataset
from evidencetrace.eval.models import (
    EvalFailureArtifact,
    EvalMetricsArtifact,
    EvalResult,
    EvalRunManifest,
)
from evidencetrace.model_client import ModelClient

SUCCESS_ARTIFACTS = (
    "eval_report.md",
    "eval_results.jsonl",
    "metrics.json",
    "run_manifest.json",
)
LIVE_BASELINES = (
    "single_agent_live",
    "retrieval_judge_live",
    "adaptive_live",
)
DELTA_FIELDS = {
    "observed_label_macro_f1": ("verification", "observed_label_macro_f1"),
    "fixed_taxonomy_macro_f1": ("verification", "fixed_taxonomy_macro_f1"),
    "contradiction_recall": ("verification", "contradiction_recall"),
    "entailed_precision": ("verification", "entailed_precision"),
    "partial_support_f1": ("verification", "partial_support_f1"),
    "high_confidence_error_rate": (
        "verification",
        "high_confidence_error_rate",
    ),
    "false_block_rate": ("engineering", "false_block_rate"),
    "evidence_span_token_f1": ("retrieval", "evidence_coverage"),
    "model_calls": ("engineering", "model_call_count"),
    "total_tokens": ("engineering", "model_usage", "total_tokens"),
    "model_latency_ms_p50": ("engineering", "model_latency_ms_p50"),
    "model_latency_ms_p95": ("engineering", "model_latency_ms_p95"),
}


class StabilityAggregateArtifact(BaseModel):
    """Secret-free aggregate over complete or failed evaluation attempts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_type: Literal["phase3e3_dev_stability_aggregate"] = (
        "phase3e3_dev_stability_aggregate"
    )
    schema_version: Literal[1] = 1
    created_at: str
    benchmark_validity: Literal["provisional"] = "provisional"
    public_benchmark_eligible: Literal[False] = False
    expected_runs: int = Field(ge=1)
    attempted_runs: int = Field(ge=0)
    complete_runs: int = Field(ge=0)
    operational_success: bool
    three_of_three_complete: bool
    frozen_provenance: dict[str, str]
    consistency_checks: dict[str, bool]
    runs: tuple[dict[str, Any], ...]
    baseline_macro_f1_summary: dict[str, dict[str, Any]]
    retrieval_judge_vs_single_agent_deltas: tuple[dict[str, Any], ...]
    per_case_relation_agreement: dict[str, Any]
    unstable_cases: tuple[dict[str, Any], ...]
    resource_summary: dict[str, Any]
    failure_summary: dict[str, int]
    input_artifact_sha256: dict[str, dict[str, str]]
    contains_sensitive_payloads: Literal[False] = False


class JudgeSmokeArtifact(BaseModel):
    """Safe summary of repeated single-case Retrieval-to-Judge calls."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_type: Literal["phase3e3_retrieval_judge_smoke"] = (
        "phase3e3_retrieval_judge_smoke"
    )
    schema_version: Literal[1] = 1
    created_at: str
    dataset_hash: str
    split_hash: str
    selected_split: Literal["dev"] = "dev"
    case_id: str
    gold_relation: str
    attempts_requested: int = Field(ge=1)
    attempts_completed: int = Field(ge=0)
    all_canonical_predictions_valid: bool
    model_provider: str
    model_id: str
    model_temperature_requested: float
    model_thinking_mode: Literal["provider_default"] = "provider_default"
    automatic_retry_count: Literal[0] = 0
    frozen_provenance: dict[str, str]
    predictions: tuple[dict[str, Any], ...]
    resource_summary: dict[str, Any]
    contains_sensitive_payloads: Literal[False] = False


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _percentile_95(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return float(ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))])


def _nested(payload: dict[str, Any], path: tuple[str, ...]) -> float | None:
    value: Any = payload
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _read_results(path: Path) -> tuple[EvalResult, ...]:
    return tuple(
        EvalResult.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _success_resources(
    results: tuple[EvalResult, ...],
) -> tuple[dict[str, Any], tuple[float, ...]]:
    calls = sum(item.model_calls for item in results)
    latencies = tuple(
        latency for item in results for latency in item.model_call_latencies_ms
    )
    usage_complete = all(
        item.model_calls == 0 or item.model_usage_status == "reported"
        for item in results
    )
    return (
        {
            "calls": calls,
            "usage_status": "reported" if usage_complete else "incomplete",
            "input_tokens": (
                sum(int(item.input_tokens or 0) for item in results)
                if usage_complete
                else None
            ),
            "output_tokens": (
                sum(int(item.output_tokens or 0) for item in results)
                if usage_complete
                else None
            ),
            "total_tokens": (
                sum(int(item.total_tokens or 0) for item in results)
                if usage_complete
                else None
            ),
            "model_latency_ms_p50": (float(median(latencies)) if latencies else None),
            "model_latency_ms_p95": _percentile_95(latencies),
        },
        latencies,
    )


def _run_signature(manifest: EvalRunManifest) -> dict[str, Any]:
    return {
        "dataset_hash": manifest.dataset_hash,
        "split_hash": manifest.split_hash,
        "git_commit": manifest.git_commit,
        "prompt_version": manifest.prompt_version,
        "miner_contract_version": manifest.miner_contract_version,
        "judge_output_validation_version": (manifest.judge_output_validation_version),
        "deterministic_signal_policy_version": (
            manifest.deterministic_signal_policy_version
        ),
        "model_id": manifest.model_id,
        "model_temperature": manifest.model_temperature,
        "model_thinking_mode": manifest.model_thinking_mode,
        "retrieval_config": manifest.retrieval_config,
    }


def _failure_signature(artifact: EvalFailureArtifact) -> dict[str, Any]:
    return {
        "dataset_hash": artifact.dataset_hash,
        "split_hash": artifact.split_hash,
        "git_commit": artifact.git_commit,
        "prompt_version": artifact.prompt_version,
        "miner_contract_version": artifact.miner_contract_version,
        "judge_output_validation_version": (artifact.judge_output_validation_version),
        "model_id": artifact.model_id,
        "model_temperature": artifact.model_temperature_requested,
        "model_thinking_mode": artifact.model_thinking_mode,
    }


def _same(signatures: Sequence[dict[str, Any]], key: str) -> bool:
    values = [signature.get(key) for signature in signatures]
    return bool(values) and all(value == values[0] for value in values)


def _macro_summary(
    successful: Sequence[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    names = {name for run in successful for name in run["baseline_metrics"]}
    summary: dict[str, dict[str, Any]] = {}
    for name in sorted(names):
        values = [
            float(
                run["baseline_metrics"][name]["verification"]["observed_label_macro_f1"]
            )
            for run in successful
            if name in run["baseline_metrics"]
        ]
        if not values:
            continue
        summary[name] = {
            "values": values,
            "mean": mean(values),
            "std_population": pstdev(values),
            "min": min(values),
            "max": max(values),
        }
    return summary


def _deltas(successful: Sequence[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for run in successful:
        metrics = run["baseline_metrics"]
        if not all(
            name in metrics for name in ("single_agent_live", "retrieval_judge_live")
        ):
            continue
        single = metrics["single_agent_live"]
        retrieval = metrics["retrieval_judge_live"]
        values: dict[str, float | None] = {}
        for label, path in DELTA_FIELDS.items():
            retrieval_value = _nested(retrieval, path)
            single_value = _nested(single, path)
            values[label] = (
                retrieval_value - single_value
                if retrieval_value is not None and single_value is not None
                else None
            )
        rows.append({"run": run["run"], "deltas": values})
    return tuple(rows)


def _agreement(
    result_sets: Sequence[tuple[EvalResult, ...]],
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    if not result_sets:
        return {}, ()
    report: dict[str, Any] = {}
    unstable: list[dict[str, Any]] = []
    for baseline in LIVE_BASELINES:
        by_run = [
            {item.case_id: item for item in results if item.baseline == baseline}
            for results in result_sets
        ]
        case_ids = sorted(set.intersection(*(set(items) for items in by_run)))
        cases: dict[str, Any] = {}
        agreed = 0
        for case_id in case_ids:
            predictions = tuple(
                items[case_id].predicted_relation.value for items in by_run
            )
            gold_values = {items[case_id].gold_relation.value for items in by_run}
            if len(gold_values) != 1:
                raise ValueError("gold relation changed across stability runs")
            agreement = len(set(predictions)) == 1
            agreed += int(agreement)
            case = {
                "gold": next(iter(gold_values)),
                "predictions": predictions,
                "agreement": agreement,
            }
            cases[case_id] = case
            if not agreement:
                unstable.append({"baseline": baseline, "case_id": case_id, **case})
        report[baseline] = {
            "agreed_cases": agreed,
            "total_cases": len(case_ids),
            "agreement_rate": agreed / len(case_ids) if case_ids else 0.0,
            "cases": cases,
        }
    return report, tuple(unstable)


def _render_markdown(artifact: StabilityAggregateArtifact) -> str:
    lines = [
        "# Phase 3E.3 dev stability aggregate",
        "",
        f"Operational success: **{artifact.complete_runs}/{artifact.expected_runs}**",
        "",
        "## Runs",
        "",
        "| Run | Status | Calls | Total tokens | Model p50 ms | Model p95 ms |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for run in artifact.runs:
        resources = run["resources"]
        lines.append(
            f"| {run['run']} | {run['status']} | {resources['calls']} | "
            f"{resources['total_tokens']} | "
            f"{resources['model_latency_ms_p50']} | "
            f"{resources['model_latency_ms_p95']} |"
        )
    lines.extend(
        [
            "",
            "## Macro-F1 stability",
            "",
            "| Baseline | Mean | Std (population) | Min | Max |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, values in artifact.baseline_macro_f1_summary.items():
        lines.append(
            f"| {name} | {values['mean']:.6f} | "
            f"{values['std_population']:.6f} | {values['min']:.6f} | "
            f"{values['max']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Unstable live predictions",
            "",
            "| Baseline | Case | Gold | Predictions |",
            "|---|---|---|---|",
        ]
    )
    for case in artifact.unstable_cases:
        predictions = ", ".join(case["predictions"])
        lines.append(
            f"| {case['baseline']} | {case['case_id']} | "
            f"{case['gold']} | {predictions} |"
        )
    if not artifact.unstable_cases:
        lines.append("| - | - | - | none |")
    return "\n".join(lines) + "\n"


def run_retrieval_judge_smoke(
    dataset_path: Path | str,
    output_directory: Path | str,
    *,
    case_id: str,
    client: ModelClient,
    attempts: int = 5,
    frozen_provenance: dict[str, str],
) -> Path:
    """Run repeated dev-only Judge calls and persist no model-authored text."""

    if attempts <= 0:
        raise ValueError("smoke attempts must be positive")
    dataset_path = Path(dataset_path)
    if dataset_path.name != "dev.jsonl":
        raise ValueError("Phase 3E.3 smoke accepts only a dev.jsonl dataset")
    dataset = load_dataset(dataset_path)
    case = next((item for item in dataset.cases if item.case_id == case_id), None)
    if case is None or case.split != "dev":
        raise ValueError("smoke case must exist in the selected dev dataset")
    source = dataset.sources[case.source_id]
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=False)
    predictions: list[dict[str, Any]] = []
    started = datetime.now(UTC)
    try:
        for attempt in range(1, attempts + 1):
            prediction = run_baseline(
                case,
                source,
                "retrieval_judge_live",
                client=client,
            )
            evidence = prediction.predicted_evidence_span
            predictions.append(
                {
                    "attempt": attempt,
                    "predicted_relation": prediction.predicted_relation.value,
                    "confidence": prediction.confidence,
                    "evidence_span_present": evidence is not None,
                    "evidence_span_sha256": (
                        hashlib.sha256(evidence.encode()).hexdigest()
                        if evidence is not None
                        else None
                    ),
                    "evidence_span_is_source_substring": (
                        evidence is None or evidence in source.content
                    ),
                    "model_calls": prediction.model_calls,
                    "model_usage_status": prediction.model_usage_status,
                    "input_tokens": prediction.input_tokens,
                    "output_tokens": prediction.output_tokens,
                    "total_tokens": prediction.total_tokens,
                    "model_call_latencies_ms": (prediction.model_call_latencies_ms),
                    "schema_failure_count": prediction.schema_failure_count,
                    "transport_failure_count": (prediction.transport_failure_count),
                }
            )
    except Exception as error:
        from evidencetrace.eval.runner import _write_live_failure_artifact

        events = getattr(client, "telemetry_events", ())
        call_count = len(events) if isinstance(events, tuple) else 0
        _write_live_failure_artifact(
            output=output,
            dataset=dataset,
            dataset_path=dataset_path,
            selected_split="dev",
            started=started,
            model_client=client,
            model_id=client.model_id,
            temperature=float(getattr(client, "temperature", 0.0)),
            maximum_live_calls=attempts,
            calls_by_baseline={
                "single_agent_live": 0,
                "retrieval_judge_live": call_count,
            },
            completed_live_predictions=len(predictions),
            failed_baseline="retrieval_judge_live",
            failed_case_id=case_id,
            failure_stage="baseline_case",
            error=error,
        )
        raise

    calls = sum(int(item["model_calls"]) for item in predictions)
    if calls > attempts:
        raise RuntimeError("smoke exceeded the declared provider-call ceiling")
    usage_complete = all(
        item["model_calls"] == 0 or item["model_usage_status"] == "reported"
        for item in predictions
    )
    latencies = tuple(
        float(latency)
        for item in predictions
        for latency in item["model_call_latencies_ms"]
    )
    artifact = JudgeSmokeArtifact(
        created_at=datetime.now(UTC).isoformat(),
        dataset_hash=dataset.raw_hash,
        split_hash=dataset.split_hash,
        case_id=case.case_id,
        gold_relation=case.gold_relation.value,
        attempts_requested=attempts,
        attempts_completed=len(predictions),
        all_canonical_predictions_valid=len(predictions) == attempts,
        model_provider=str(getattr(client, "provider_id", "custom")),
        model_id=client.model_id,
        model_temperature_requested=float(getattr(client, "temperature", 0.0)),
        frozen_provenance=frozen_provenance,
        predictions=tuple(predictions),
        resource_summary={
            "calls": calls,
            "usage_status": "reported" if usage_complete else "incomplete",
            "input_tokens": (
                sum(int(item["input_tokens"] or 0) for item in predictions)
                if usage_complete
                else None
            ),
            "output_tokens": (
                sum(int(item["output_tokens"] or 0) for item in predictions)
                if usage_complete
                else None
            ),
            "total_tokens": (
                sum(int(item["total_tokens"] or 0) for item in predictions)
                if usage_complete
                else None
            ),
            "model_latency_ms_p50": (float(median(latencies)) if latencies else None),
            "model_latency_ms_p95": _percentile_95(latencies),
            "schema_failures": sum(
                int(item["schema_failure_count"]) for item in predictions
            ),
            "transport_failures": sum(
                int(item["transport_failure_count"]) for item in predictions
            ),
            "guard_failures": 0,
            "local_validation_failures": 0,
        },
    )
    path = output / "smoke_attempt.json"
    path.write_text(artifact.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def generate_stability_aggregate(
    run_directories: Sequence[Path | str],
    output_directory: Path | str,
    *,
    frozen_provenance: dict[str, str],
    expected_runs: int = 3,
) -> Path:
    """Validate run artifacts and write a payload-free stability aggregate."""

    directories = tuple(Path(path) for path in run_directories)
    if len(directories) > expected_runs:
        raise ValueError("attempted run count exceeds the declared plan")
    summaries: list[dict[str, Any]] = []
    successful: list[dict[str, Any]] = []
    result_sets: list[tuple[EvalResult, ...]] = []
    signatures: list[dict[str, Any]] = []
    artifact_hashes: dict[str, dict[str, str]] = {}
    all_latencies: list[float] = []
    failures = {
        "schema": 0,
        "transport": 0,
        "guard": 0,
        "local_validation": 0,
    }

    for directory in directories:
        run_name = directory.name
        failed_path = directory / "failed_attempt.json"
        success_paths = tuple(directory / name for name in SUCCESS_ARTIFACTS)
        if all(path.is_file() for path in success_paths) and not failed_path.exists():
            manifest = EvalRunManifest.model_validate_json(
                (directory / "run_manifest.json").read_text(encoding="utf-8")
            )
            metrics = EvalMetricsArtifact.model_validate_json(
                (directory / "metrics.json").read_text(encoding="utf-8")
            )
            results = _read_results(directory / "eval_results.jsonl")
            resources, latencies = _success_resources(results)
            all_latencies.extend(latencies)
            run_failures = {
                "schema": sum(item.schema_failure_count for item in results),
                "transport": sum(item.transport_failure_count for item in results),
                "guard": 0,
                "local_validation": 0,
            }
            for key, value in run_failures.items():
                failures[key] += value
            summary = {
                "run": run_name,
                "status": "complete",
                "baseline_metrics": {
                    name: payload["headline"]
                    for name, payload in metrics.baselines.items()
                },
                "resources": resources,
                "failures": run_failures,
            }
            summaries.append(summary)
            successful.append(summary)
            result_sets.append(results)
            signatures.append(_run_signature(manifest))
            artifact_hashes[run_name] = {
                path.name: _sha256(path) for path in success_paths
            }
            continue
        if failed_path.is_file() and not any(path.exists() for path in success_paths):
            failure = EvalFailureArtifact.model_validate_json(
                failed_path.read_text(encoding="utf-8")
            )
            events = failure.model_call_events
            latencies = tuple(event.latency_ms for event in events)
            all_latencies.extend(latencies)
            run_failures = {
                "schema": failure.model_telemetry.schema_failures,
                "transport": failure.model_telemetry.transport_failures,
                "guard": int(failure.error_type == "DeterministicConflictError"),
                "local_validation": int(failure.failure_kind == "local_validation"),
            }
            for key, value in run_failures.items():
                failures[key] += value
            summaries.append(
                {
                    "run": run_name,
                    "status": "failed",
                    "failed_baseline": failure.failed_baseline,
                    "failed_case_id": failure.failed_case_id,
                    "error_type": failure.error_type,
                    "resources": {
                        "calls": failure.actual_model_calls,
                        "usage_status": failure.model_telemetry.usage_status,
                        "input_tokens": failure.model_telemetry.input_tokens,
                        "output_tokens": failure.model_telemetry.output_tokens,
                        "total_tokens": failure.model_telemetry.total_tokens,
                        "model_latency_ms_p50": (
                            float(median(latencies)) if latencies else None
                        ),
                        "model_latency_ms_p95": _percentile_95(latencies),
                    },
                    "failures": run_failures,
                }
            )
            signatures.append(_failure_signature(failure))
            artifact_hashes[run_name] = {failed_path.name: _sha256(failed_path)}
            continue
        raise ValueError(f"run directory has an invalid artifact set: {run_name}")

    consistency_keys = (
        "dataset_hash",
        "split_hash",
        "git_commit",
        "prompt_version",
        "miner_contract_version",
        "judge_output_validation_version",
        "model_id",
        "model_temperature",
        "model_thinking_mode",
    )
    consistency = {key: _same(signatures, key) for key in consistency_keys}
    success_signatures = [
        signature
        for signature, summary in zip(signatures, summaries, strict=True)
        if summary["status"] == "complete"
    ]
    consistency["deterministic_signal_policy_version"] = _same(
        success_signatures,
        "deterministic_signal_policy_version",
    )
    consistency["retrieval_config"] = _same(
        success_signatures,
        "retrieval_config",
    )
    consistency["same_frozen_version"] = all(consistency.values())

    agreement, unstable = _agreement(result_sets)
    complete_runs = len(successful)
    reported_usage = all(
        run["resources"]["usage_status"] == "reported" for run in summaries
    )
    resource_summary = {
        "calls": sum(run["resources"]["calls"] for run in summaries),
        "usage_status": "reported" if reported_usage else "incomplete",
        "input_tokens": (
            sum(int(run["resources"]["input_tokens"] or 0) for run in summaries)
            if reported_usage
            else None
        ),
        "output_tokens": (
            sum(int(run["resources"]["output_tokens"] or 0) for run in summaries)
            if reported_usage
            else None
        ),
        "total_tokens": (
            sum(int(run["resources"]["total_tokens"] or 0) for run in summaries)
            if reported_usage
            else None
        ),
        "model_latency_ms_p50": (
            float(median(all_latencies)) if all_latencies else None
        ),
        "model_latency_ms_p95": _percentile_95(all_latencies),
    }
    operational_success = (
        len(directories) == expected_runs
        and complete_runs == expected_runs
        and consistency["same_frozen_version"]
    )
    artifact = StabilityAggregateArtifact(
        created_at=datetime.now(UTC).isoformat(),
        expected_runs=expected_runs,
        attempted_runs=len(directories),
        complete_runs=complete_runs,
        operational_success=operational_success,
        three_of_three_complete=expected_runs == 3 and operational_success,
        frozen_provenance=frozen_provenance,
        consistency_checks=consistency,
        runs=tuple(summaries),
        baseline_macro_f1_summary=_macro_summary(successful),
        retrieval_judge_vs_single_agent_deltas=_deltas(successful),
        per_case_relation_agreement=agreement,
        unstable_cases=unstable,
        resource_summary=resource_summary,
        failure_summary=failures,
        input_artifact_sha256=artifact_hashes,
    )
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=False)
    json_path = output / "aggregate.json"
    json_path.write_text(
        artifact.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    output.joinpath("aggregate.md").write_text(
        _render_markdown(artifact),
        encoding="utf-8",
    )
    return json_path


__all__ = [
    "JudgeSmokeArtifact",
    "StabilityAggregateArtifact",
    "generate_stability_aggregate",
    "run_retrieval_judge_smoke",
]
