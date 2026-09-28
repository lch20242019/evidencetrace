"""Recompute the resume metrics from EvidenceTrace JSONL artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path

LABELS = (
    "entailed",
    "partially_entailed",
    "contradicted",
    "not_in_source",
    "source_unavailable",
    "not_checkable",
)
SCIFACT_LABELS = ("SUPPORT", "CONTRADICT", "NOINFO")
ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if q == 50:
        return float(statistics.median(ordered))
    if q == 95:
        # Reproduce src/evidencetrace/eval/metrics.py::_p95 exactly.
        return float(ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))])
    raise ValueError(f"unsupported percentile: {q}")


def fixed_six_macro_f1(rows: list[dict]) -> float:
    scores = []
    for label in LABELS:
        true_positive = sum(
            row["gold_relation"] == label and row["predicted_relation"] == label
            for row in rows
        )
        false_positive = sum(
            row["gold_relation"] != label and row["predicted_relation"] == label
            for row in rows
        )
        false_negative = sum(
            row["gold_relation"] == label and row["predicted_relation"] != label
            for row in rows
        )
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        scores.append(
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
    return statistics.fmean(scores)


def fixed_scifact_macro_f1(rows: list[dict]) -> float:
    scores = []
    for label in SCIFACT_LABELS:
        true_positive = sum(
            row["official_label"] == label and row["predicted_label"] == label
            for row in rows
        )
        false_positive = sum(
            row["official_label"] != label and row["predicted_label"] == label
            for row in rows
        )
        false_negative = sum(
            row["official_label"] == label and row["predicted_label"] != label
            for row in rows
        )
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        scores.append(
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
    return statistics.fmean(scores)


def summarize(rows: list[dict]) -> dict:
    def integer(row: dict, key: str) -> int:
        value = row.get(key)
        return int(value) if value is not None else 0

    latencies = [
        float(row["latency_ms"]) for row in rows if row.get("latency_ms") is not None
    ]
    model_latencies = [
        float(value) for row in rows for value in row.get("model_call_latencies_ms", ())
    ]
    return {
        "cases": len(rows),
        "fixed_six_macro_f1": fixed_six_macro_f1(rows),
        "model_calls": sum(integer(row, "model_calls") for row in rows),
        "input_tokens": sum(integer(row, "input_tokens") for row in rows),
        "output_tokens": sum(integer(row, "output_tokens") for row in rows),
        "total_tokens": sum(integer(row, "total_tokens") for row in rows),
        "pipeline_latency_ms_p50": percentile(latencies, 50),
        "pipeline_latency_ms_p95": percentile(latencies, 95),
        "model_latency_ms_p95": (
            percentile(model_latencies, 95) if model_latencies else None
        ),
    }


def close(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-9)


def main() -> int:
    holdout_dir = ROOT / "eval_runs" / "phase3f_v3_zh_internal_gate"
    holdout_rows = read_jsonl(holdout_dir / "eval_results.jsonl")
    holdout_record = read_json(holdout_dir / "execution_record.json")
    holdout = {}
    expected_holdout_claims = {
        "single_agent_live": {
            "fixed_six_macro_f1": 0.7273598558686278,
            "total_tokens": 63175,
            "model_latency_ms_p95": 5217.567708808929,
        },
        "retrieval_judge_live": {
            "fixed_six_macro_f1": 0.6137566137566137,
            "total_tokens": 70658,
            "model_latency_ms_p95": 11690.203417092562,
        },
    }
    for baseline in ("single_agent_live", "retrieval_judge_live"):
        rows = [row for row in holdout_rows if row["baseline"] == baseline]
        holdout[baseline] = summarize(rows)
        expected = holdout_record["metrics"][baseline]["fixed_six_macro_f1"]
        if not close(holdout[baseline]["fixed_six_macro_f1"], expected):
            raise AssertionError(f"holdout macro-F1 mismatch: {baseline}")
        resources = holdout_record["resources"]["by_live_baseline"][baseline]
        for key in ("provider_calls", "input_tokens", "output_tokens", "total_tokens"):
            local_key = "model_calls" if key == "provider_calls" else key
            if holdout[baseline][local_key] != resources[key]:
                raise AssertionError(f"holdout resource mismatch: {baseline} {key}")
        for key, expected in expected_holdout_claims[baseline].items():
            actual = holdout[baseline][key]
            if isinstance(expected, float):
                matched = close(actual, expected)
            else:
                matched = actual == expected
            if not matched:
                raise AssertionError(f"holdout resume claim mismatch: {baseline} {key}")

    dev_runs = []
    for run_number in (1, 2, 3):
        run_dir = ROOT / "eval_runs" / f"phase3g_ownership_zh_dev_run_0{run_number}"
        rows = read_jsonl(run_dir / "eval_results.jsonl")
        if len(rows) != 432:
            raise AssertionError(
                f"unexpected row count in run {run_number}: {len(rows)}"
            )
        metrics = read_json(run_dir / "metrics.json")
        run_result = {"round": run_number, "raw_rows": len(rows), "baselines": {}}
        for baseline in ("single_agent_live", "adaptive_live"):
            selected = [row for row in rows if row["baseline"] == baseline]
            measured = summarize(selected)
            expected = metrics["baselines"][baseline]["headline"]["verification"]
            if not close(
                measured["fixed_six_macro_f1"],
                expected["fixed_taxonomy_macro_f1"],
            ):
                raise AssertionError(
                    f"dev macro-F1 mismatch: run {run_number} {baseline}"
                )
            run_result["baselines"][baseline] = measured
        dev_runs.append(run_result)

    means = {}
    for baseline in ("single_agent_live", "adaptive_live"):
        means[baseline] = {
            key: statistics.fmean(run["baselines"][baseline][key] for run in dev_runs)
            for key in (
                "fixed_six_macro_f1",
                "model_calls",
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "pipeline_latency_ms_p50",
                "pipeline_latency_ms_p95",
            )
        }

    single = means["single_agent_live"]
    adaptive = means["adaptive_live"]
    deltas = {
        "macro_f1_absolute": adaptive["fixed_six_macro_f1"]
        - single["fixed_six_macro_f1"],
        "model_call_reduction_fraction": 1.0
        - adaptive["model_calls"] / single["model_calls"],
        "input_token_reduction_fraction": 1.0
        - adaptive["input_tokens"] / single["input_tokens"],
        "total_token_reduction_fraction": 1.0
        - adaptive["total_tokens"] / single["total_tokens"],
        "pipeline_p50_reduction_fraction": 1.0
        - adaptive["pipeline_latency_ms_p50"] / single["pipeline_latency_ms_p50"],
        "pipeline_p95_reduction_fraction": 1.0
        - adaptive["pipeline_latency_ms_p95"] / single["pipeline_latency_ms_p95"],
    }

    expected_deltas = {
        "macro_f1_absolute": 0.053014339391150966,
        "model_call_reduction_fraction": 0.32407407407407407,
        "input_token_reduction_fraction": 0.3622465323927342,
        "total_token_reduction_fraction": 0.35611336692428786,
        "pipeline_p50_reduction_fraction": 0.12588101100180193,
        "pipeline_p95_reduction_fraction": 0.014295987023729584,
    }
    for key, expected in expected_deltas.items():
        if not close(deltas[key], expected):
            raise AssertionError(
                f"dev delta mismatch: {key} {deltas[key]} != {expected}"
            )

    aggregate = read_json(
        ROOT / "eval_runs" / "phase3g_ownership_zh_dev_stability" / "aggregate.json"
    )
    schema_recovery = aggregate["schema_recovery"]
    if not (
        schema_recovery["logical_calls"] == 504
        and schema_recovery["first_attempt_schema_failures"] == 4
        and schema_recovery["recovered_schema_failures"] == 4
        and schema_recovery["unrecovered_schema_failures"] == 0
    ):
        raise AssertionError("schema recovery totals do not match the resume claim")

    manifest = read_json(
        ROOT
        / "eval_runs"
        / "phase3g_ownership_zh_dev_stability"
        / "artifact_manifest.json"
    )
    artifact_matches = 0
    for relative_path, expected_hash in manifest["artifacts"].items():
        path = ROOT / Path(relative_path)
        if path.is_file() and sha256_file(path) == expected_hash:
            artifact_matches += 1
    if artifact_matches != manifest["artifact_count"]:
        raise AssertionError("artifact manifest verification failed")

    scifact_pack = ROOT / "eval_sets" / "scifact_public_dev"
    scifact_run = ROOT / "eval_runs" / "scifact_public_dev_local_nli"
    scifact_pack_manifest = read_json(scifact_pack / "manifest.json")
    for relative_path, expected_hash in scifact_pack_manifest["output_sha256"].items():
        if sha256_file(scifact_pack / relative_path) != expected_hash:
            raise AssertionError(f"SciFact pack hash mismatch: {relative_path}")

    scifact_run_manifest = read_json(scifact_run / "run_manifest.json")
    if scifact_run_manifest["evaluation_status"] != {
        "mode": "local_transformers_nli_reportable",
        "reportable": True,
    }:
        raise AssertionError("SciFact run is not reportable")
    if scifact_run_manifest["counts"] != {
        "cases": 300,
        "external_provider_calls": 0,
        "local_nli_batch_count": 297,
        "local_nli_pair_count": 8195,
        "local_nli_score_many_calls": 297,
        "local_nli_subset_count": 8195,
        "rows": 600,
        "rows_by_baseline": {
            "lexical_rules": 300,
            "retrieval_judge_scifact_nli": 300,
        },
    }:
        raise AssertionError("SciFact run counts do not match the resume claim")
    length_telemetry = scifact_run_manifest["local_nli"]["input_length_telemetry"]
    if length_telemetry != {
        "available": True,
        "max_input_tokens": 512,
        "max_observed_input_tokens": 486,
        "rejected_overlength_pair_count": 0,
    }:
        raise AssertionError("SciFact input-length telemetry mismatch")
    expected_results_hash = scifact_run_manifest["outputs"]["sha256"][
        "eval_results.jsonl"
    ]
    if sha256_file(scifact_run / "eval_results.jsonl") != expected_results_hash:
        raise AssertionError("SciFact result hash does not match its run manifest")

    scifact_metrics = read_json(scifact_run / "scored" / "metrics.json")
    scifact_rows = read_jsonl(scifact_run / "scored" / "results.jsonl")
    scifact_summary = {}
    expected_scifact = {
        "lexical_rules": (0.2714294488755517, 0.09042553191489362),
        "retrieval_judge_scifact_nli": (0.5860236750067259, 0.3601063829787233),
    }
    for baseline, (expected_macro, expected_evidence) in expected_scifact.items():
        selected = [row for row in scifact_rows if row["baseline"] == baseline]
        evidence_rows = [row for row in selected if row["evidence"]["eligible"]]
        measured_macro = fixed_scifact_macro_f1(selected)
        measured_evidence = statistics.fmean(
            row["evidence"]["best_sentence_set_f1"] for row in evidence_rows
        )
        measured_hit5 = statistics.fmean(
            float(row["retrieval"]["full_sentence_hit_at_5"]) for row in evidence_rows
        )
        if not close(measured_macro, expected_macro) or not close(
            measured_evidence, expected_evidence
        ):
            raise AssertionError(f"SciFact headline mismatch: {baseline}")
        if not close(measured_hit5, 0.8776595744680851):
            raise AssertionError(f"SciFact Hit@5 mismatch: {baseline}")
        scifact_summary[baseline] = {
            "cases": len(selected),
            "fixed_3way_macro_f1": measured_macro,
            "strict_evidence_sentence_set_f1": measured_evidence,
            "full_sentence_hit_at_5": measured_hit5,
        }
    effects = scifact_metrics["paired_bootstrap"]["effects"]
    expected_effects = {
        "fixed_3way_macro_f1": (
            0.3145942261311742,
            0.24407850504397605,
            0.380869631598457,
        ),
        "best_sentence_set_f1": (
            0.2696808510638297,
            0.2062482443820223,
            0.3320682919206125,
        ),
    }
    for metric, (delta, low, high) in expected_effects.items():
        effect = effects[metric]
        if not all(
            close(effect[key], expected)
            for key, expected in (
                ("delta", delta),
                ("ci95_low", low),
                ("ci95_high", high),
            )
        ):
            raise AssertionError(f"SciFact bootstrap mismatch: {metric}")

    result = {
        "status": "pass",
        "holdout": holdout,
        "dev_runs": dev_runs,
        "dev_three_run_means": means,
        "dev_adaptive_minus_single": deltas,
        "schema_recovery": {
            "logical_calls": schema_recovery["logical_calls"],
            "first_attempt_contract_success_rate": schema_recovery[
                "first_attempt_contract_success_rate"
            ],
            "recovered_schema_failures": schema_recovery["recovered_schema_failures"],
            "unrecovered_schema_failures": schema_recovery[
                "unrecovered_schema_failures"
            ],
        },
        "artifact_manifest": {
            "declared": manifest["artifact_count"],
            "matched": artifact_matches,
        },
        "scifact_public_dev_v2": {
            "pack_manifest_sha256": sha256_file(scifact_pack / "manifest.json"),
            "run_counts": scifact_run_manifest["counts"],
            "input_length_telemetry": length_telemetry,
            "baselines": scifact_summary,
            "paired_bootstrap_effects": {
                metric: effects[metric] for metric in expected_effects
            },
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
