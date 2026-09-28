"""Evaluation runner with explicit deterministic/live and validity boundaries."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any

from pydantic import ValidationError

from evidencetrace.agents.judge import (
    JUDGE_OUTPUT_VALIDATION_VERSION,
    JudgeScopeError,
)
from evidencetrace.agents.miner import MinerScopeError
from evidencetrace.audit_models import (
    JUDGE_VERDICT_OWNERSHIP_POLICY_VERSION,
    MINER_DRAFT_CONTRACT_VERSION,
)
from evidencetrace.checks.deterministic import (
    CHECKABILITY_POLICY_VERSION,
    DETERMINISTIC_SIGNAL_POLICY_VERSION,
    DeterministicConflictError,
)
from evidencetrace.eval.baselines import (
    maximum_expected_live_model_calls,
    run_baseline,
)
from evidencetrace.eval.dataset import (
    DatasetValidationError,
    LoadedDataset,
    load_dataset,
)
from evidencetrace.eval.errors import (
    FailureCode,
    LocalValidationError,
    clear_exception_chain,
)
from evidencetrace.eval.metrics import LABELS, compute_metrics, label_coverage_gate
from evidencetrace.eval.models import (
    BaselineName,
    BenchmarkValidity,
    DeterministicBaselineName,
    EvalCase,
    EvalFailureArtifact,
    EvalMetricsArtifact,
    EvalPrediction,
    EvalResult,
    EvalRunManifest,
    LiveBaselineName,
    LiveBaselineStatus,
    LiveFailureStage,
    SourceFixture,
)
from evidencetrace.eval.router import (
    ADAPTIVE_ROUTER_POLICY_VERSION,
    DEFAULT_FULL_CONTEXT_BUDGET_BYTES,
)
from evidencetrace.model_client import (
    SCHEMA_RECOVERY_POLICY_VERSION,
    SCHEMA_RETRY_LIMIT,
    ModelCallBudgetExceeded,
    ModelCallTelemetry,
    ModelClient,
    ModelResponseError,
    ModelSchemaError,
    ModelTransportError,
    OpenAICompatibleClient,
    SchemaRecoveryClient,
    SchemaRecoverySummary,
    summarize_model_telemetry,
)
from evidencetrace.relation_policy import RELATION_DEFINITION_POLICY_VERSION
from evidencetrace.retrieval.rank import RETRIEVAL_POLICY_VERSION

DETERMINISTIC_BASELINES: tuple[DeterministicBaselineName, ...] = (
    "lexical_rules",
    "lexical_full_source",
    "retrieval_judge_deterministic",
)
LIVE_BASELINES: tuple[LiveBaselineName, ...] = (
    "single_agent_live",
    "retrieval_judge_live",
    "adaptive_live",
)
# Kept as the public deterministic baseline collection used by Phase 3A callers.
SUCCESS_ARTIFACT_NAMES = (
    "eval_report.md",
    "eval_results.jsonl",
    "metrics.json",
    "run_manifest.json",
)
BASELINES = DETERMINISTIC_BASELINES


def _git_commit(project_root: Path) -> str:
    try:
        completed = subprocess.run(
            ("git", "-C", str(project_root), "rev-parse", "HEAD"),
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return "uncommitted"
    value = completed.stdout.strip()
    return value if completed.returncode == 0 and value else "uncommitted"


def _record(case: EvalCase, prediction: EvalPrediction) -> dict[str, Any]:
    record = prediction.model_dump(mode="json")
    record.update(
        {
            "gold_relation": case.gold_relation.value,
            "gold_evidence_span": case.gold_evidence_span,
            "gold_line": case.claim_line,
            "claim_type": case.claim_type,
            "mutation_type": case.mutation_type,
            "split": case.split,
            "annotation_status": case.annotation_status,
            "case_hash": case.case_hash,
            "source_id": case.source_id,
            "source_length_stratum": case.source_length_stratum,
            "expected_chunk_count": case.expected_chunk_count,
            "expected_route": case.expected_route,
        }
    )
    if prediction.predicted_relation != case.gold_relation:
        span = case.gold_evidence_span
        record["error_stage"] = (
            "retrieval"
            if span
            and not any(
                span in text or text in span for text in prediction.retrieved_texts
            )
            and prediction.baseline not in {"lexical_full_source", "single_agent_live"}
            and prediction.selected_route != "full_context_single_agent"
            else "judge"
        )
    else:
        record["error_stage"] = "none"
    return EvalResult.model_validate(record).model_dump(mode="json")


def _subset_records(
    records: list[dict[str, Any]],
    *,
    split: str | None = None,
    annotation: str | None = None,
) -> list[dict[str, Any]]:
    return [
        record
        for record in records
        if (split is None or record["split"] == split)
        and (annotation is None or record["annotation_status"] == annotation)
    ]


def _baseline_report(
    records: list[dict[str, Any]], *, headline_split: str
) -> dict[str, Any]:
    if headline_split == "dev":
        # Dev is an explicitly provisional tuning surface. Its complete split
        # is shown so before/after remediation is comparable.
        headline = _subset_records(records, split="dev")
    else:
        headline = [
            record
            for record in records
            if record["split"] == "test"
            and record["annotation_status"]
            in {
                "deterministic_gold",
                "human_reviewed",
                "single_human_review",
            }
        ]
    provisional = [
        record for record in records if record["annotation_status"] == "provisional"
    ]
    human_reviewed = [
        record for record in records if record["annotation_status"] == "human_reviewed"
    ]
    single_human_review = [
        record
        for record in records
        if record["annotation_status"] == "single_human_review"
    ]
    return {
        "headline_split": headline_split,
        "headline": compute_metrics(headline),
        "all": compute_metrics(records),
        "provisional": compute_metrics(provisional),
        "human_reviewed": compute_metrics(human_reviewed),
        "single_human_review": compute_metrics(single_human_review),
        "dev": compute_metrics(_subset_records(records, split="dev")),
        "test": compute_metrics(_subset_records(records, split="test")),
        "source_length_strata": {
            stratum: compute_metrics(
                [
                    record
                    for record in records
                    if record.get("source_length_stratum") == stratum
                ]
            )
            for stratum in ("short", "medium", "long")
        },
    }


def _benchmark_validity(
    cases: tuple[EvalCase, ...], selected_split: str
) -> BenchmarkValidity:
    selected_test = [case for case in cases if case.split == "test"]
    if selected_split == "dev" or not selected_test:
        return "provisional"
    if all(case.annotation_status == "single_human_review" for case in selected_test):
        return "single_human_synthetic_holdout"
    if all(case.annotation_status == "human_reviewed" for case in selected_test):
        return "human_reviewed_holdout"
    return "diagnostic_contaminated"


def _gate(
    verification: dict[str, Any],
    validity: BenchmarkValidity,
    *,
    baseline: BaselineName,
    substantive_spans_valid: bool = True,
) -> dict[str, Any]:
    single_human_holdout = validity == "single_human_synthetic_holdout"
    coverage = (
        label_coverage_gate(
            verification,
            required_labels=LABELS,
            minimum_support=1,
        )
        if single_human_holdout
        else label_coverage_gate(verification)
    )
    internal_macro_condition = {
        "value": verification["observed_label_macro_f1"],
        "threshold": 0.70,
        "achieved": verification["observed_label_macro_f1"] >= 0.70,
    }
    conditions = (
        {
            "observed_label_macro_f1": internal_macro_condition,
            "substantive_source_spans": {
                "value": substantive_spans_valid,
                "required": True,
                "achieved": substantive_spans_valid,
            },
        }
        if single_human_holdout
        else {
            "observed_label_macro_f1": internal_macro_condition,
            "contradiction_recall": {
                "value": verification["contradiction_recall"],
                "threshold": 0.80,
                "achieved": verification["contradiction_recall"] >= 0.80,
            },
            "entailed_precision": {
                "value": verification["entailed_precision"],
                "threshold": 0.75,
                "achieved": verification["entailed_precision"] >= 0.75,
            },
        }
    )
    metrics_achieved = all(item["achieved"] for item in conditions.values())
    if not coverage["achieved"]:
        status = "insufficient_coverage"
    elif validity == "diagnostic_contaminated":
        status = "diagnostic_contaminated"
    elif single_human_holdout and baseline != "retrieval_judge_live":
        status = "not_headline_baseline"
    elif metrics_achieved:
        status = "pass"
    else:
        status = "metric_failure"
    phase4_eligible = status == "pass" and (
        validity == "human_reviewed_holdout"
        or (single_human_holdout and baseline == "retrieval_judge_live")
    )
    return {
        "status": status,
        "achieved": status == "pass",
        "metrics_achieved": metrics_achieved,
        "conditions": conditions,
        "coverage": coverage,
        "headline_baseline_required": (
            "retrieval_judge_live" if single_human_holdout else None
        ),
        "taxonomy_macro_consistent": (
            verification["observed_label_macro_f1"]
            == verification["fixed_taxonomy_macro_f1"]
            if coverage["achieved"] and single_human_holdout
            else None
        ),
        "technical_targets": {
            "relation_macro_f1": {
                "value": verification["observed_label_macro_f1"],
                "threshold": 0.75,
                "achieved": verification["observed_label_macro_f1"] >= 0.75,
            },
            "contradiction_recall": {
                "value": verification["contradiction_recall"],
                "threshold": 0.80,
                "achieved": verification["contradiction_recall"] >= 0.80,
            },
            "substantive_source_span": {
                "required": True,
                "achieved": substantive_spans_valid,
                "note": "The runner rejects non-substring substantive spans.",
            },
        },
        "partial_support_f1": verification["partial_support_f1"],
        "high_confidence_error_rate": verification["high_confidence_error_rate"],
        "phase4_eligible": phase4_eligible,
        "public_benchmark_eligible": validity == "human_reviewed_holdout",
        "note": (
            "The internal Phase 3 stop gate for a single-human synthetic "
            "holdout uses retrieval_judge_live macro-F1 >= 0.70. The unchanged "
            "v0.1 technical targets are reported separately."
            if single_human_holdout
            else (
                "Partial-support F1 and high-confidence error rate are exposed "
                "for comparison with the saved pre-remediation dev snapshot."
            )
        ),
    }


def _error_analysis(
    records: list[dict[str, Any]],
    metrics: dict[str, Any],
    validity: BenchmarkValidity,
    baseline: BaselineName,
) -> dict[str, Any]:
    failures = [
        record
        for record in records
        if record["gold_relation"] != record["predicted_relation"]
    ]
    counts: dict[str, int] = {}
    for record in failures:
        key = f"{record['gold_relation']}→{record['predicted_relation']}"
        counts[key] = counts.get(key, 0) + 1
    retrieval_failures = sum(
        record.get("error_stage") == "retrieval" for record in failures
    )
    judge_failures = sum(record.get("error_stage") == "judge" for record in failures)
    verification = metrics["headline"]["verification"]
    substantive_spans_valid = all(
        record["predicted_relation"]
        not in {"entailed", "partially_entailed", "contradicted"}
        or bool(record.get("predicted_evidence_span"))
        for record in records
    )
    return {
        "most_common_errors": [
            {"pattern": pattern, "count": count}
            for pattern, count in sorted(
                counts.items(), key=lambda item: (-item[1], item[0])
            )[:3]
        ],
        "failed_case_count": len(failures),
        "failed_cases": [
            {
                "case_id": record["case_id"],
                "gold": record["gold_relation"],
                "predicted": record["predicted_relation"],
                "error_stage": record.get("error_stage", "none"),
                "gold_evidence_span": record.get("gold_evidence_span"),
                "predicted_evidence_span": record.get("predicted_evidence_span"),
            }
            for record in failures[:10]
        ],
        "retrieval_failure_count": retrieval_failures,
        "judge_failure_count": judge_failures,
        "headline_macro_f1": verification["observed_label_macro_f1"],
        "stop_gate": _gate(
            verification,
            validity,
            baseline=baseline,
            substantive_spans_valid=substantive_spans_valid,
        ),
    }


def _summary_table(
    names: tuple[BaselineName, ...],
    reports: dict[str, dict[str, Any]],
    analyses: dict[str, dict[str, Any]],
) -> list[str]:
    lines = [
        "| Baseline | Obs macro-F1 | Fixed macro-F1 | Weighted-F1 | "
        "Balanced acc. | Contradiction recall | Entailed precision | "
        "Partial F1 | High-conf. error | Gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for baseline in names:
        verification = reports[baseline]["headline"]["verification"]
        gate = analyses[baseline]["stop_gate"]["status"]
        lines.append(
            f"| {baseline} | "
            f"{verification['observed_label_macro_f1']:.3f} | "
            f"{verification['fixed_taxonomy_macro_f1']:.3f} | "
            f"{verification['weighted_f1']:.3f} | "
            f"{verification['balanced_accuracy']:.3f} | "
            f"{verification['contradiction_recall']:.3f} | "
            f"{verification['entailed_precision']:.3f} | "
            f"{verification['partial_support_f1']:.3f} | "
            f"{verification['high_confidence_error_rate']:.3f} | {gate} |"
        )
    return lines


def _render_report(
    dataset: LoadedDataset,
    reports: dict[str, dict[str, Any]],
    analyses: dict[str, dict[str, Any]],
    manifest: EvalRunManifest,
) -> str:
    primary: BaselineName = (
        "retrieval_judge_live"
        if (
            manifest.benchmark_validity == "single_human_synthetic_holdout"
            and "retrieval_judge_live" in reports
        )
        else "retrieval_judge_deterministic"
    )
    primary_report = reports[primary]["headline"]
    primary_analysis = analyses[primary]
    executed_live = tuple(
        baseline
        for baseline in LIVE_BASELINES
        if manifest.live_baseline_statuses[baseline] == "executed"
    )
    lines = [
        "# EvidenceTrace evaluation report",
        "",
        f"Dataset: {dataset.path}",
        f"Cases in dataset: {len(dataset.cases)} "
        f"(dev {len(dataset.dev_cases)}, test {len(dataset.test_cases)})",
        f"Dataset hash: {dataset.raw_hash}",
        f"Git state: {manifest.git_commit}",
        f"Benchmark validity: {manifest.benchmark_validity}",
        f"Public benchmark eligible: {str(manifest.public_benchmark_eligible).lower()}",
        "",
        "Pair-level cases provide claim_text directly. Claim extraction "
        "precision, recall, F1, atomicity, and line mapping are therefore "
        "null / not_applicable; they require full Markdown gold claim spans.",
        "",
        "## Deterministic pair baselines",
        "",
    ]
    lines.extend(_summary_table(DETERMINISTIC_BASELINES, reports, analyses))
    lines.extend(["", "## Live model baselines", ""])
    if executed_live:
        lines.extend(_summary_table(executed_live, reports, analyses))
    else:
        lines.extend(
            [
                "| Baseline | Status |",
                "|---|---|",
                *[
                    f"| {name} | {manifest.live_baseline_statuses[name]} |"
                    for name in LIVE_BASELINES
                ],
            ]
        )
    if manifest.maximum_expected_live_model_calls:
        actual_live_calls = sum(
            int(reports[baseline]["all"]["engineering"]["model_call_count"])
            for baseline in executed_live
        )
        lines.extend(
            [
                "",
                "Live model call ceiling: "
                f"{manifest.maximum_expected_live_model_calls}; actual: "
                f"{actual_live_calls}. Both live baselines share provider "
                f"{manifest.model_provider}, model {manifest.model_id}, and "
                f"requested temperature {manifest.model_temperature}; actual "
                "effective temperature is unknown. Thinking mode is "
                f"{manifest.model_thinking_mode}; automatic retries: 0.",
            ]
        )
        for baseline in executed_live:
            telemetry = reports[baseline]["all"]["engineering"]
            usage = telemetry["model_usage"]
            lines.append(
                f"- {baseline}: calls {telemetry['model_call_count']}; "
                f"tokens input/output/total "
                f"{usage['input_tokens']}/{usage['output_tokens']}/"
                f"{usage['total_tokens']} ({usage['status']}); model latency "
                f"p50/p95 {telemetry['model_latency_ms_p50']}/"
                f"{telemetry['model_latency_ms_p95']} ms; schema/transport "
                f"failures {telemetry['schema_failure_count']}/"
                f"{telemetry['transport_failure_count']}; cost_usd "
                f"{telemetry['cost_usd']}."
            )

    verification = primary_report["verification"]
    gate = primary_analysis["stop_gate"]
    lines.extend(
        [
            "",
            "Deterministic code and real model results are intentionally shown "
            "in separate tables. Fake or lexical outputs are never presented "
            "as model performance.",
            "",
            "## Metric validity",
            "",
            f"Headline split: {reports[primary]['headline_split']}.",
            f"Internal Phase 3 gate status: {gate['status']}.",
            f"Coverage status: {gate['coverage']['status']} with minimum "
            f"support {gate['coverage']['minimum_support']} for "
            ", ".join(gate["coverage"]["required_labels"])
            + ".",
            "A diagnostic_contaminated result cannot pass a release gate even "
            "when numerical thresholds are met.",
            "",
            "Per-label support for the primary baseline:",
            "",
            "| Label | Support | Precision | Recall | F1 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for label, values in verification["per_label"].items():
        lines.append(
            f"| {label} | {values['support']} | "
            f"{values['precision']:.3f} | {values['recall']:.3f} | "
            f"{values['f1']:.3f} |"
        )
    matrix = verification["confusion_matrix"]
    labels = tuple(matrix)
    lines.extend(
        [
            "",
            "Primary headline confusion matrix:",
            "",
            "| Gold / predicted | " + " | ".join(labels) + " |",
            "|---|" + "---:|" * len(labels),
        ]
    )
    for actual in labels:
        lines.append(
            f"| {actual} | "
            + " | ".join(str(matrix[actual][guess]) for guess in labels)
            + " |"
        )
    engineering = primary_report["engineering"]
    lines.extend(
        [
            "",
            "Recall@5 is a fixture sanity metric only: repeated small sources "
            "do not constitute a production retrieval benchmark.",
            "",
            "False-block means predicted contradicted (error/block severity) "
            "among cases whose gold relation is not contradicted. "
            f"Numerator {engineering['false_block_numerator']}; denominator "
            f"{engineering['false_block_denominator']}; rate "
            f"{engineering['false_block_rate']:.3f}.",
            "",
            "## Primary baseline error analysis",
            "",
            f"Failures: {primary_analysis['failed_case_count']}; retrieval "
            f"{primary_analysis['retrieval_failure_count']}; Judge "
            f"{primary_analysis['judge_failure_count']}.",
            "",
            "Most common error patterns:",
            "",
        ]
    )
    for item in primary_analysis["most_common_errors"]:
        lines.append(f"- {item['pattern']}: {item['count']} cases")
    lines.extend(
        [
            "",
            "Failed cases (first 10):",
            "",
            "| Case | Gold | Predicted | Stage |",
            "|---|---|---|---|",
        ]
    )
    for item in primary_analysis["failed_cases"]:
        lines.append(
            f"| {item['case_id']} | {item['gold']} | "
            f"{item['predicted']} | {item['error_stage']} |"
        )
    if not primary_analysis["failed_cases"]:
        lines.append("| — | No failures | — | — |")
    lines.extend(
        [
            "",
            "## Reproducibility and release boundary",
            "",
            f"Model: {manifest.model_id}; prompt: {manifest.prompt_version}; "
            f"retrieval: {manifest.retrieval_config}.",
            "Cost remains null because no explicit model price snapshot was "
            "supplied; no cost is estimated or fabricated.",
            (
                "Internal Phase 3 gate passed; Phase 4 is eligible. "
                f"Public benchmark eligible: {manifest.public_benchmark_eligible}."
                if gate["phase4_eligible"]
                else "Phase 4 remains blocked until the frozen holdout's "
                "retrieval_judge_live baseline passes the internal gate."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _prepare_output_directory(
    output: Path, validity: BenchmarkValidity, *, live_requested: bool = False
) -> None:
    if (
        (validity == "single_human_synthetic_holdout" or live_requested)
        and output.exists()
        and any(output.iterdir())
    ):
        raise DatasetValidationError(
            "live evaluation output directory must be new or empty"
        )
    output.mkdir(parents=True, exist_ok=True)


def _model_events(client: ModelClient) -> tuple[ModelCallTelemetry, ...]:
    value = getattr(client, "telemetry_events", ())
    if not isinstance(value, tuple):
        return ()
    return tuple(event for event in value if isinstance(event, ModelCallTelemetry))


def _live_failure_kind(error: Exception) -> str:
    if isinstance(error, ModelCallBudgetExceeded):
        return "budget"
    if isinstance(error, ModelTransportError):
        return "transport"
    if isinstance(error, (ModelResponseError, ValidationError)):
        return "schema"
    if isinstance(
        error,
        (
            DeterministicConflictError,
            JudgeScopeError,
            LocalValidationError,
            MinerScopeError,
        ),
    ):
        return "local_validation"
    return "unexpected"


def _failure_code(error: Exception) -> FailureCode:
    if isinstance(error, ModelCallBudgetExceeded):
        return "call_budget_exhausted"
    if isinstance(error, ModelTransportError):
        return "model_transport_failure"
    if isinstance(error, (ModelResponseError, ValidationError)):
        return "model_schema_failure"
    if isinstance(error, DeterministicConflictError):
        return "deterministic_conflict"
    if isinstance(error, JudgeScopeError):
        return "judge_scope_violation"
    if isinstance(error, MinerScopeError):
        return "miner_scope_violation"
    if isinstance(error, LocalValidationError):
        return error.code
    return "unexpected_exception"


def _validated_baseline_prediction(
    case: EvalCase,
    source: SourceFixture,
    baseline: BaselineName,
    client: ModelClient | None,
) -> EvalPrediction:
    prediction = run_baseline(
        case,
        source,
        baseline,
        client=client,
    )
    if prediction.predicted_relation.value in {
        "entailed",
        "partially_entailed",
        "contradicted",
    } and (prediction.predicted_evidence_span is None):
        raise LocalValidationError("substantive_evidence_missing")
    if (
        prediction.predicted_evidence_span is not None
        and prediction.predicted_evidence_span not in source.content
    ):
        raise LocalValidationError("evidence_span_not_in_source")
    if baseline in DETERMINISTIC_BASELINES and prediction.model_calls:
        raise LocalValidationError("deterministic_baseline_model_call")
    return prediction


def _write_live_failure_artifact(
    *,
    output: Path,
    dataset: LoadedDataset,
    dataset_path: Path | str,
    selected_split: str,
    started: datetime,
    model_client: ModelClient,
    model_id: str,
    temperature: float,
    maximum_live_calls: int,
    calls_by_baseline: dict[LiveBaselineName, int],
    completed_live_predictions: int,
    failed_baseline: BaselineName,
    failed_case_id: str,
    failure_stage: LiveFailureStage,
    error: Exception,
) -> Path:
    clear_exception_chain(error)
    events = _model_events(model_client)
    telemetry = summarize_model_telemetry(events)
    recovery = getattr(
        model_client,
        "schema_recovery",
        SchemaRecoverySummary(retry_limit=0),
    )
    recovery = recovery.model_copy(update={"final_operational_success": False})
    reported = tuple(event for event in events if event.usage_status == "reported")
    latencies = sorted(event.latency_ms for event in events)
    p95 = (
        float(latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))])
        if latencies
        else None
    )
    if isinstance(error, DeterministicConflictError):
        artifact_version = "phase3e-live-failure-v3"
    elif isinstance(error, MinerScopeError):
        artifact_version = "phase3e2-live-failure-v4"
    elif isinstance(error, JudgeScopeError):
        artifact_version = "phase3e3-live-failure-v5"
    elif isinstance(error, LocalValidationError):
        artifact_version = "phase4a-local-validation-failure-v7"
    elif recovery.first_attempt_schema_failures:
        artifact_version = "phase3g-schema-recovery-failure-v6"
    else:
        artifact_version = "phase3d1-live-failure-v2"
    artifact = EvalFailureArtifact(
        artifact_version=artifact_version,
        dataset_hash=dataset.raw_hash,
        split_hash=dataset.split_hash,
        selected_split=selected_split,
        git_commit=_git_commit(Path(dataset_path).resolve().parent.parent),
        prompt_version=str(getattr(model_client, "prompt_version", "eval-v2")),
        miner_contract_version=MINER_DRAFT_CONTRACT_VERSION,
        judge_output_validation_version=JUDGE_OUTPUT_VALIDATION_VERSION,
        started_at=started.isoformat(),
        failed_at=datetime.now(UTC).isoformat(),
        model_provider=str(getattr(model_client, "provider_id", "openai-compatible")),
        model_id=model_id,
        model_temperature_requested=temperature,
        maximum_expected_live_model_calls=maximum_live_calls,
        actual_model_calls=telemetry.calls,
        actual_model_calls_by_baseline=calls_by_baseline,
        automatic_retry_count=recovery.schema_retry_calls,
        schema_recovery=recovery,
        completed_live_predictions=completed_live_predictions,
        failure_stage=failure_stage,
        failed_baseline=failed_baseline,
        failed_case_id=failed_case_id,
        failure_kind=_live_failure_kind(error),
        failure_code=_failure_code(error),
        error_type=error.__class__.__name__,
        schema_diagnostic=(
            error.diagnostic if isinstance(error, ModelSchemaError) else None
        ),
        guard_signal_codes=(
            error.signal_codes if isinstance(error, DeterministicConflictError) else ()
        ),
        miner_scope_error_code=(
            error.code if isinstance(error, MinerScopeError) else None
        ),
        judge_scope_error_code=(
            error.code if isinstance(error, JudgeScopeError) else None
        ),
        model_telemetry=telemetry,
        model_call_events=events,
        calls_with_reported_usage=len(reported),
        reported_input_token_subtotal=sum(
            int(event.input_tokens or 0) for event in reported
        ),
        reported_output_token_subtotal=sum(
            int(event.output_tokens or 0) for event in reported
        ),
        reported_total_token_subtotal=sum(
            int(event.total_tokens or 0) for event in reported
        ),
        model_latency_ms_p50=(float(median(latencies)) if latencies else None),
        model_latency_ms_p95=p95,
        cost_explanation=(
            "No explicit model price snapshot was supplied; cost is null."
        ),
    )
    path = output / "failed_attempt.json"
    temporary = output / "failed_attempt.json.tmp"
    temporary.write_text(artifact.model_dump_json(indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


@dataclass
class _LiveFailureState:
    dataset: LoadedDataset | None = None
    dataset_path: Path | str | None = None
    output: Path | None = None
    selected_split: str = "all"
    started: datetime | None = None
    model_client: ModelClient | None = None
    model_id: str = "deterministic-fake-v1"
    temperature: float = 0.0
    maximum_live_calls: int = 0
    calls_by_baseline: dict[LiveBaselineName, int] = field(
        default_factory=lambda: {
            "single_agent_live": 0,
            "retrieval_judge_live": 0,
            "adaptive_live": 0,
        }
    )
    completed_live_predictions: int = 0
    current_baseline: BaselineName = "lexical_rules"
    current_case_id: str = "not_started"
    failure_stage: LiveFailureStage = "run_post_processing"

    def persist(self, error: Exception) -> None:
        clear_exception_chain(error)
        if (
            self.model_client is None
            or self.dataset is None
            or self.dataset_path is None
            or self.output is None
            or self.started is None
        ):
            return

        failure_path = self.output / "failed_attempt.json"
        if failure_path.exists():
            return

        events = _model_events(self.model_client)
        calls = dict(self.calls_by_baseline)
        difference = len(events) - sum(calls.values())
        if difference:
            target: LiveBaselineName = (
                self.current_baseline
                if self.current_baseline in LIVE_BASELINES
                else "single_agent_live"
            )
            calls[target] = max(calls[target] + difference, 0)
        if sum(calls.values()) != len(events):
            calls = {
                "single_agent_live": len(events),
                "retrieval_judge_live": 0,
                "adaptive_live": 0,
            }

        for name in SUCCESS_ARTIFACT_NAMES:
            self.output.joinpath(name).unlink(missing_ok=True)
            self.output.joinpath(f"{name}.tmp").unlink(missing_ok=True)

        _write_live_failure_artifact(
            output=self.output,
            dataset=self.dataset,
            dataset_path=self.dataset_path,
            selected_split=self.selected_split,
            started=self.started,
            model_client=self.model_client,
            model_id=self.model_id,
            temperature=self.temperature,
            maximum_live_calls=self.maximum_live_calls,
            calls_by_baseline=calls,
            completed_live_predictions=self.completed_live_predictions,
            failed_baseline=self.current_baseline,
            failed_case_id=self.current_case_id,
            failure_stage=self.failure_stage,
            error=error,
        )


def _run_eval(
    dataset_path: Path | str,
    out: Path | str,
    *,
    selected_split: str = "all",
    model_id: str = "deterministic-fake-v1",
    live_requested: bool = False,
    temperature: float = 0.0,
    failure_state: _LiveFailureState,
) -> Path:
    """Run explicit baselines and write the four canonical eval artifacts."""

    dataset = load_dataset(dataset_path)
    if selected_split not in {"all", "dev", "test"}:
        raise DatasetValidationError("selected_split must be all, dev, or test")
    cases = (
        dataset.cases
        if selected_split == "all"
        else tuple(case for case in dataset.cases if case.split == selected_split)
    )
    if not cases:
        raise DatasetValidationError(f"selected split is empty: {selected_split}")
    validity = _benchmark_validity(cases, selected_split)
    output = Path(out)
    _prepare_output_directory(output, validity, live_requested=live_requested)
    started = datetime.now(UTC)
    headline_split = "dev" if selected_split == "dev" else "test"
    maximum_live_calls = (
        maximum_expected_live_model_calls(
            len(cases), schema_retry_limit=SCHEMA_RETRY_LIMIT
        )
        if live_requested
        else 0
    )
    failure_state.dataset = dataset
    failure_state.dataset_path = dataset_path
    failure_state.output = output
    failure_state.selected_split = selected_split
    failure_state.started = started
    failure_state.model_id = model_id
    failure_state.temperature = temperature
    failure_state.maximum_live_calls = maximum_live_calls
    failure_state.current_case_id = cases[0].case_id
    failure_state.current_baseline = DETERMINISTIC_BASELINES[0]
    failure_state.failure_stage = "baseline_case"

    live_statuses: dict[LiveBaselineName, LiveBaselineStatus] = {
        baseline: "not_requested" for baseline in LIVE_BASELINES
    }
    live_client: ModelClient | None = None
    skip_reason = None
    if live_requested:
        if not os.getenv("OPENAI_API_KEY"):
            live_statuses = {
                baseline: "skipped_missing_credentials" for baseline in LIVE_BASELINES
            }
            skip_reason = (
                "OPENAI_API_KEY is not set; live baselines were skipped "
                "and deterministic baselines still executed."
            )
        else:
            if model_id == "deterministic-fake-v1":
                raise DatasetValidationError(
                    "live baselines require an explicit model id"
                )
            live_client = SchemaRecoveryClient(
                OpenAICompatibleClient(
                    model_id=model_id,
                    temperature=temperature,
                    max_calls=maximum_live_calls,
                )
            )
            failure_state.model_client = live_client
            live_statuses = {baseline: "executed" for baseline in LIVE_BASELINES}

    executed: tuple[BaselineName, ...] = DETERMINISTIC_BASELINES + (
        LIVE_BASELINES if live_client is not None else ()
    )
    all_records: list[dict[str, Any]] = []
    reports: dict[str, dict[str, Any]] = {}
    analyses: dict[str, dict[str, Any]] = {}
    live_call_counts: dict[LiveBaselineName, int] = {
        baseline: 0 for baseline in LIVE_BASELINES
    }
    failure_state.calls_by_baseline = live_call_counts
    completed_live_predictions = 0
    for baseline in executed:
        records: list[dict[str, Any]] = []
        failure_state.current_baseline = baseline
        failure_state.failure_stage = "baseline_case"
        for case in cases:
            started_case = time.perf_counter()
            failure_state.current_case_id = case.case_id
            source = dataset.sources[case.source_id]
            is_live = live_client is not None and baseline in LIVE_BASELINES
            before_model_calls = len(_model_events(live_client)) if is_live else 0
            prediction = _validated_baseline_prediction(
                case,
                source,
                baseline,
                client=live_client if is_live else None,
            )
            if is_live:
                assert live_client is not None
                after_model_calls = len(_model_events(live_client))
                live_call_counts[baseline] += after_model_calls - before_model_calls
                completed_live_predictions += 1
                failure_state.completed_live_predictions = completed_live_predictions
            elapsed = (time.perf_counter() - started_case) * 1000.0
            prediction = prediction.model_copy(update={"latency_ms": elapsed})
            records.append(_record(case, prediction))
        failure_state.failure_stage = "baseline_post_processing"
        reports[baseline] = _baseline_report(records, headline_split=headline_split)
        analyses[baseline] = _error_analysis(
            records, reports[baseline], validity, baseline
        )
        all_records.extend(records)

    failure_state.failure_stage = "run_post_processing"
    reported_live_call_counts = {
        baseline: (
            int(reports[baseline]["all"]["engineering"]["model_call_count"])
            if baseline in reports
            else 0
        )
        for baseline in LIVE_BASELINES
    }
    if reported_live_call_counts != live_call_counts:
        raise RuntimeError("live prediction and provider call counts diverged")
    actual_live_calls = sum(live_call_counts.values())
    if live_client is not None and actual_live_calls != len(_model_events(live_client)):
        raise RuntimeError("live call accounting and provider telemetry diverged")
    if actual_live_calls > maximum_live_calls:
        raise RuntimeError("live model calls exceeded the declared ceiling")
    live_configurations = [
        reports[baseline]["all"]["engineering"]["model_configuration"]
        for baseline in LIVE_BASELINES
        if baseline in reports
    ]
    if any(not config["consistent"] for config in live_configurations):
        raise RuntimeError("a live baseline mixed model configurations")
    for config_field in ("provider", "model_id", "temperature"):
        observed = {
            config[config_field]
            for config in live_configurations
            if config[config_field] is not None
        }
        if len(observed) > 1:
            raise RuntimeError("live baselines used different model configurations")
    live_provider = "openai-compatible" if live_requested else None
    live_temperature = temperature if live_requested else None
    live_execution = {
        "requested": live_requested,
        "provider": live_provider,
        "model_id": model_id if live_requested else None,
        "temperature": live_temperature,
        "temperature_requested": live_temperature,
        "temperature_actual_effective": None,
        "temperature_effective_status": ("unknown" if live_requested else None),
        "temperature_semantics": (
            "requested_only" if live_requested else "not_applicable"
        ),
        "thinking_mode": ("provider_default" if live_requested else "not_applicable"),
        "same_provider_model_temperature": True,
        "same_provider_model_temperature_thinking_mode": True,
        "maximum_expected_model_calls": maximum_live_calls,
        "hard_call_limit_enforced": live_client is not None,
        "automatic_retry_count": (
            live_client.schema_recovery.schema_retry_calls
            if isinstance(live_client, SchemaRecoveryClient)
            else 0
        ),
        "schema_recovery": (
            live_client.schema_recovery.model_dump(mode="json")
            if isinstance(live_client, SchemaRecoveryClient)
            else SchemaRecoverySummary(retry_limit=0).model_dump(mode="json")
        ),
        "actual_model_calls": actual_live_calls,
        "actual_model_calls_by_baseline": live_call_counts,
        "baseline_telemetry": {
            baseline: {
                key: reports[baseline]["all"]["engineering"][key]
                for key in (
                    "model_call_count",
                    "model_usage",
                    "model_latency_ms_p50",
                    "model_latency_ms_p95",
                    "schema_failure_count",
                    "transport_failure_count",
                    "cost_usd",
                    "pricing_snapshot_id",
                    "cost_explanation",
                )
            }
            for baseline in LIVE_BASELINES
            if baseline in reports
        },
        "pricing_snapshot_id": None,
        "cost_usd": None,
        "cost_explanation": (
            "No explicit model price snapshot was supplied; cost is null."
        ),
    }

    finished = datetime.now(UTC)
    benchmark_status = (
        "single_human_review"
        if validity == "single_human_synthetic_holdout"
        else (
            "human_reviewed" if validity == "human_reviewed_holdout" else "provisional"
        )
    )
    public_benchmark_eligible = validity == "human_reviewed_holdout"
    headline_baseline: BaselineName = (
        "retrieval_judge_live"
        if (
            validity == "single_human_synthetic_holdout"
            and "retrieval_judge_live" in reports
        )
        else "retrieval_judge_deterministic"
    )
    aggregate_live_status: LiveBaselineStatus
    if all(status == "executed" for status in live_statuses.values()):
        aggregate_live_status = "executed"
    elif any(
        status == "skipped_missing_credentials" for status in live_statuses.values()
    ):
        aggregate_live_status = "skipped_missing_credentials"
    else:
        aggregate_live_status = "not_requested"
    failure_state.failure_stage = "artifact_build"
    manifest = EvalRunManifest(
        dataset_hash=dataset.raw_hash,
        split_hash=dataset.split_hash,
        model_id=model_id,
        model_provider=live_provider,
        model_temperature=live_temperature,
        model_temperature_actual_effective=None,
        model_temperature_effective_status=(
            "unknown" if live_requested else "not_applicable"
        ),
        model_temperature_semantics=(
            "requested_only" if live_requested else "not_applicable"
        ),
        model_thinking_mode=(
            "provider_default" if live_requested else "not_applicable"
        ),
        maximum_expected_live_model_calls=maximum_live_calls,
        pricing_snapshot_id=None,
        prompt_version=(
            str(getattr(live_client, "prompt_version", "eval-v2"))
            if live_client is not None
            else "eval-v2"
        ),
        miner_contract_version=MINER_DRAFT_CONTRACT_VERSION,
        judge_output_validation_version=JUDGE_OUTPUT_VALIDATION_VERSION,
        deterministic_signal_policy_version=(DETERMINISTIC_SIGNAL_POLICY_VERSION),
        retrieval_config={
            "retriever": "fts5-or-bm25-with-cjk-ngram-fallback",
            "retrieval_policy_version": RETRIEVAL_POLICY_VERSION,
            "router_policy_version": ADAPTIVE_ROUTER_POLICY_VERSION,
            "checkability_policy_version": CHECKABILITY_POLICY_VERSION,
            "relation_definition_policy_version": (RELATION_DEFINITION_POLICY_VERSION),
            "schema_recovery_policy_version": SCHEMA_RECOVERY_POLICY_VERSION,
            "judge_verdict_ownership_policy_version": (
                JUDGE_VERDICT_OWNERSHIP_POLICY_VERSION
            ),
            "full_context_budget_bytes": DEFAULT_FULL_CONTEXT_BUDGET_BYTES,
            "top_k": 5,
            "neighbor_window": 0,
            "metric_role": "fixture_sanity",
        },
        automatic_retry_count=(
            live_client.schema_recovery.schema_retry_calls
            if isinstance(live_client, SchemaRecoveryClient)
            else 0
        ),
        schema_recovery_policy_version=(
            SCHEMA_RECOVERY_POLICY_VERSION if live_requested else None
        ),
        schema_retry_limit=(SCHEMA_RETRY_LIMIT if live_requested else 0),
        python_version=platform.python_version(),
        git_commit=_git_commit(Path(dataset_path).resolve().parent.parent),
        started_at=started.isoformat(),
        finished_at=finished.isoformat(),
        selected_split=selected_split,
        benchmark_status=benchmark_status,
        benchmark_validity=validity,
        public_benchmark_eligible=public_benchmark_eligible,
        baselines=executed,
        live_baseline=aggregate_live_status,
        live_baseline_statuses=live_statuses,
        skip_reason=skip_reason,
    )
    failure_state.failure_stage = "artifact_write"
    output.joinpath("eval_results.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in all_records),
        encoding="utf-8",
    )
    metrics_payload = {
        "dataset": {
            "path": str(dataset.path),
            "hash": dataset.raw_hash,
            "split_hash": dataset.split_hash,
            "case_count": len(dataset.cases),
            "selected_case_count": len(cases),
            "dev_count": len(dataset.dev_cases),
            "test_count": len(dataset.test_cases),
            "evaluation_unit": "claim_source_pair",
            "annotation_counts": {
                status: sum(case.annotation_status == status for case in dataset.cases)
                for status in (
                    "deterministic_gold",
                    "human_reviewed",
                    "single_human_review",
                    "provisional",
                )
            },
        },
        "baselines": reports,
        "error_analysis": analyses,
        "headline_baseline": headline_baseline,
        "provisional_excluded_from_headline": True,
        "benchmark_status": benchmark_status,
        "benchmark_validity": validity,
        "public_benchmark_eligible": public_benchmark_eligible,
        "baseline_groups": {
            "deterministic": DETERMINISTIC_BASELINES,
            "live": LIVE_BASELINES if live_client is not None else (),
        },
        "live_baseline_statuses": live_statuses,
        "live_execution": live_execution,
    }
    metrics_artifact = EvalMetricsArtifact.model_validate(metrics_payload)
    output.joinpath("metrics.json").write_text(
        metrics_artifact.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    output.joinpath("run_manifest.json").write_text(
        manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    output.joinpath("eval_report.md").write_text(
        _render_report(dataset, reports, analyses, manifest), encoding="utf-8"
    )
    return output


def run_eval(
    dataset_path: Path | str,
    out: Path | str,
    *,
    selected_split: str = "all",
    model_id: str = "deterministic-fake-v1",
    live_requested: bool = False,
    temperature: float = 0.0,
) -> Path:
    """Run eval and preserve safe live telemetry for every failure stage."""

    failure_state = _LiveFailureState()
    try:
        return _run_eval(
            dataset_path,
            out,
            selected_split=selected_split,
            model_id=model_id,
            live_requested=live_requested,
            temperature=temperature,
            failure_state=failure_state,
        )
    except Exception as error:
        failure_state.persist(error)

        raise


__all__ = [
    "BASELINES",
    "DETERMINISTIC_BASELINES",
    "LIVE_BASELINES",
    "run_eval",
]
