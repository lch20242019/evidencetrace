from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from evidencetrace.eval.models import EvalFailureArtifact

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "eval_sets" / "v2"
RELATIONS = {
    "entailed",
    "partially_entailed",
    "contradicted",
    "not_in_source",
    "source_unavailable",
    "not_checkable",
}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256sum_bundle(relative_paths: list[str]) -> str:
    payload = "".join(
        f"{_sha256(ROOT / relative_path)}  {relative_path}\n"
        for relative_path in relative_paths
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _artifact_record_bundle(paths: list[Path]) -> str:
    records = [
        {
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(paths)
    ]
    payload = json.dumps(records, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def test_v2_holdout_candidate_contract_and_proposal_coverage() -> None:
    candidates = _jsonl(V2 / "holdout_candidates.jsonl")

    assert len(candidates) == 60
    assert {item["candidate_id"] for item in candidates} == {
        f"v2_holdout_{number:03d}" for number in range(1, 61)
    }
    assert Counter(item["proposed_relation"] for item in candidates) == {
        relation: 10 for relation in RELATIONS
    }
    for item in candidates:
        assert item["claim"]
        assert item["source_id"]
        assert item["source_text"] or not item["source_available"]
        assert item["candidate_origin"] == "synthetic_candidate"
        assert "synthetic_candidate" in item["source_provenance"]
        assert item["proposal_status"] == "model_proposal"
        assert item["proposed_relation"] in RELATIONS
        assert item["human_relation"] == ""
        assert item["human_evidence_span"] == ""
        assert item["reviewer_notes"] == ""
        assert item["annotation_status"] == "unreviewed_candidate"
        assert item["benchmark_validity"] == "provisional"
        assert item["evaluation_status"] == "not_run"
        assert "gold_relation" not in item
        assert "gold_evidence_span" not in item


def test_v2_candidate_sources_are_frozen_synthetic_fixtures() -> None:
    sources = {
        source["source_id"]: source
        for source in _jsonl(V2 / "sources" / "synthetic_sources.jsonl")
    }
    candidates = _jsonl(V2 / "holdout_candidates.jsonl")

    assert len(sources) == 12
    assert sum(source["available"] for source in sources.values()) == 10
    for source in sources.values():
        assert source["url"].startswith("offline://v2/synthetic/")
        assert source["provenance"].startswith("synthetic_candidate:")
        assert (
            hashlib.sha256(source["content"].encode("utf-8")).hexdigest()
            == source["content_hash"]
        )
    for candidate in candidates:
        source = sources[candidate["source_id"]]
        assert candidate["source_text"] == source["content"]
        assert candidate["source_available"] is source["available"]
        assert candidate["source_sha256"] == source["content_hash"]


def test_v2_dev_is_separate_and_contains_only_new_dev_cases() -> None:
    dev = _jsonl(V2 / "dev.jsonl")
    candidates = _jsonl(V2 / "holdout_candidates.jsonl")
    sources = {
        source["source_id"]: source
        for source in _jsonl(V2 / "sources" / "synthetic_sources.jsonl")
    }

    assert len(dev) == 18
    assert {case["gold_relation"] for case in dev} == RELATIONS
    assert all(case["case_id"].startswith("v2_dev_") for case in dev)
    assert all(case["split"] == "dev" for case in dev)
    assert all(case["annotation_status"] == "deterministic_gold" for case in dev)
    dev_claims = {case["claim_text"].casefold() for case in dev}
    candidate_claims = {item["claim"].casefold() for item in candidates}
    assert dev_claims.isdisjoint(candidate_claims)
    for case in dev:
        source = sources[case["source_id"]]
        evidence = case["gold_evidence_span"]
        if evidence is not None:
            assert evidence in source["content"]


def test_v2_manifest_hashes_and_holdout_state() -> None:
    manifest = json.loads((V2 / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["benchmark_validity"] == "single_human_synthetic_holdout"
    assert manifest["public_benchmark_eligible"] is False
    assert manifest["candidate_origin"] == "synthetic_candidate"
    assert manifest["evaluation_status"] == (
        "phase3d1_all_three_live_schema_smokes_passed_full_dev_holdout_not_run"
    )
    assert manifest["code_freeze_hash"] == (
        "eb5a01fa0065a5b6643c4c1d60cca746d1f626015e54ce0d4382343859120f62"
    )
    phase3d_freeze = manifest["phase3d_code_freeze"]
    assert phase3d_freeze["sha256"] == (
        "66284bd98b75f7ea1a7160bcf9d8662d2b37401f988378e92264a33df18e9df2"
    )
    assert phase3d_freeze["prompt_bundle_sha256"] == (
        "845b72bacf789fe3bb08e3a11b178e4dc6e96fb49b4867349c37c9ecb753619c"
    )
    phase3d1 = manifest["phase3d1_structured_output_freeze"]
    assert phase3d1["sha256"] == (
        "79fc7fa4ec8d886556baa8406c0d61253abbcc5010b1ae0a8cd27da26a9ca308"
    )
    assert phase3d1["prompt_bundle_sha256"] == (
        "43e4f32204726f608e0eed44d5c0f3e4779e01984f15d0ecdb14ad27add6326e"
    )
    assert manifest["latest_evaluation_status"] == ("diagnostic_consumed_guard_failure")
    assert phase3d1["base_commit"] == ("ab2eaaa5d2f083685842c7611b92d587af7442b0")
    assert phase3d1["real_model_api_calls"] == 0
    assert phase3d1["v2_holdout_executed"] is False
    assert phase3d1["phase4_blocked"] is True
    assert phase3d1["request_contract"] == {
        "response_format": {"type": "json_object"},
        "default_max_tokens": 2048,
        "maximum_configurable_max_tokens": 8192,
        "strict_pydantic_validation": True,
        "automatic_retry_count": 0,
        "guessing_or_repair": False,
    }
    phase3e = manifest["phase3e_context_aware_entity_guard"]
    assert phase3e["code_bundle_sha256"] == (
        "41c5bf4a9f5bdd9159dad26b37eeb3688efa09db3604dfb5bb5b3057d96083cf"
    )
    assert phase3e["prompt_bundle_sha256"] == (
        "6ccb4e34abc29819ef7fdd5d6b1648acbe8d86e96aab46168d4629dfdbd75e43"
    )
    config_hash = hashlib.sha256(
        json.dumps(
            phase3e["config"],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    assert config_hash == phase3e["config_sha256"]
    assert phase3e["deterministic_signal_policy_version"] == (
        "deterministic-signals-v2"
    )
    assert phase3e["structured_output_contract_version"] == ("openai-compatible-v2")
    assert phase3e["historical_failure_artifact_modified"] is False
    assert phase3e["real_model_api_calls"] == 0
    assert phase3e["v2_holdout_rerun"] is False
    assert phase3e["v3_created"] is False
    assert phase3e["phase4_blocked"] is True
    phase3e2 = manifest["phase3e2_miner_contract_pair_baseline"]
    assert phase3e2["code_bundle_sha256"] == (
        "309c570e4eb835384946bc1db210c2f8250e2b6a9b6630440839ef89425be64b"
    )
    assert phase3e2["prompt_bundle_sha256"] == (
        "01dde7fb745f2073d10f2ac1e0c3fe1497c51f51f190c4752a329ab7afb77c0a"
    )
    phase3e2_config_hash = hashlib.sha256(
        json.dumps(
            phase3e2["config"],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    assert phase3e2_config_hash == phase3e2["config_sha256"]
    assert phase3e2["miner_contract_version"] == "live-miner-draft-v2"
    assert phase3e2["deterministic_signal_policy_version"] == (
        "deterministic-signals-v2"
    )
    assert phase3e2["structured_output_contract_version"] == ("openai-compatible-v2")
    assert phase3e2["active_pair_baselines"] == {
        "deterministic": [
            "lexical_rules",
            "lexical_full_source",
            "retrieval_judge_deterministic",
        ],
        "live": ["single_agent_live", "retrieval_judge_live"],
    }
    assert phase3e2["maximum_live_calls_per_pair_case"] == 2
    assert phase3e2["pair_extraction_metrics"] == "not_applicable"
    assert phase3e2["real_model_api_calls"] == 0
    assert phase3e2["dev_live_executed"] is False
    assert phase3e2["consumed_v2_holdout_used_for_development"] is False
    assert phase3e2["v2_holdout_rerun"] is False
    assert phase3e2["v3_created"] is False
    assert phase3e2["phase4_blocked"] is True
    phase3e_failure = (
        ROOT / "eval_runs/phase3e_v2_dev_live_context_guard/failed_attempt.json"
    )
    assert _sha256(phase3e_failure) == (phase3e2["historical_failure_artifact_sha256"])
    assert phase3e2["historical_failure_artifact_modified"] is False
    phase3e3 = manifest["phase3e3_judge_validation_repair"]
    assert phase3e3["code_bundle_sha256"] == (
        "b05f9c35c72d06a3a45c6f08844db677fcceeaf868202e675b8cea0279b50c90"
    )
    assert phase3e3["prompt_bundle_sha256"] == (
        "8dc3d1f611c0673aad90c83678629eb947a5c5281ac19e8c16d25a72f885b5dd"
    )
    phase3e3_config_hash = hashlib.sha256(
        json.dumps(
            phase3e3["config"],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    assert phase3e3_config_hash == phase3e3["config_sha256"]
    assert phase3e3["judge_output_validation_version"] == ("judge-output-validation-v2")
    assert phase3e3["future_failure_artifact_version"] == ("phase3e3-live-failure-v5")
    assert phase3e3["integrity_failures_remain_fail_closed"] is True
    assert phase3e3["semantic_relation_errors_are_scored_predictions"] is True
    assert phase3e3["consumed_v2_holdout_loaded_parsed_or_evaluated"] is False
    assert phase3e3["status"] == "dev_stability_complete_3_of_3"
    assert phase3e3["real_model_api_calls"] == 98
    assert phase3e3["real_model_total_tokens"] == 117883
    assert phase3e3["operational_success"] == {
        "complete_runs": 3,
        "expected_runs": 3,
        "three_of_three_complete": True,
    }
    assert phase3e3["v2_holdout_rerun"] is False
    assert phase3e3["v3_created"] is False
    assert phase3e3["v3_holdout_creation_prerequisites_met"] is True
    assert phase3e3["phase4_blocked"] is True
    smoke = manifest["phase3d1_single_case_smoke"]
    assert smoke["frozen_git_commit"] == ("291314a59f949f676da1d9a10501409382567b9c")
    assert smoke["status"] == "contract_passed"
    assert smoke["actual_model_calls"] == 1
    assert smoke["automatic_retry_count"] == 0
    assert smoke["miner_judge_live_calls"] == 0
    assert smoke["full_dev_executed"] is False
    assert smoke["v2_holdout_executed"] is False
    assert smoke["quality_correct"] is True
    assert smoke["evidence_span_is_source_substring"] is True
    assert smoke["holdout_authorized"] is False
    assert smoke["phase4_blocked"] is True
    miner_judge_smoke = manifest["phase3d1_miner_judge_smoke"]
    assert miner_judge_smoke["frozen_git_commit"] == (
        "3fe6b764e813c06c10c965cbfc0d35648cdee7d7"
    )
    assert miner_judge_smoke["status"] == "contract_passed"
    assert miner_judge_smoke["validated_output_schemas"] == [
        "MinerOutput",
        "JudgeOutput",
    ]
    assert miner_judge_smoke["actual_model_calls"] == 2
    assert miner_judge_smoke["actual_model_calls_by_stage"] == {
        "miner": 1,
        "judge": 1,
    }
    assert miner_judge_smoke["automatic_retry_count"] == 0
    assert miner_judge_smoke["repair_model_calls"] == 0
    assert miner_judge_smoke["single_agent_live_calls"] == 0
    assert miner_judge_smoke["full_dev_executed"] is False
    assert miner_judge_smoke["v2_holdout_executed"] is False
    assert miner_judge_smoke["quality_correct"] is True
    assert miner_judge_smoke["holdout_authorized"] is False
    assert miner_judge_smoke["phase4_blocked"] is True
    assert manifest["phase3d1_contract_smoke_executed"] is True
    assert manifest["phase3d1_contract_smoke_succeeded"] is True
    assert manifest["phase3d1_all_live_schema_smokes_executed"] is True
    assert manifest["phase3d1_all_live_schema_smokes_succeeded"] is True
    assert manifest["holdout_executed"] is False
    assert manifest["live_dev_preflight_executed"] is True
    assert manifest["live_dev_preflight_succeeded"] is False
    attempt = manifest["live_dev_preflight_attempt"]
    assert attempt["provider"] == "openai-compatible"
    assert attempt["model_id"] == "deepseek-v4-flash"
    assert attempt["artifacts_written"] is False
    assert attempt["actual_model_calls"] is None
    assert (
        attempt["estimated_model_calls_lower_bound"]
        <= attempt["estimated_model_calls_upper_bound"]
    )
    rerun = manifest["live_dev_rerun"]
    assert rerun["model_id"] == "deepseek-v4-flash"
    assert rerun["thinking_mode"] == "provider_default"
    assert rerun["temperature_requested"] == 0.0
    assert rerun["temperature_actual_effective"] is None
    assert rerun["temperature_effective_status"] == "unknown"
    assert rerun["maximum_model_calls"] == 54
    assert rerun["hard_call_limit_enforced"] is True
    assert rerun["failed_requests_count_toward_limit"] is True
    assert rerun["automatic_retry_count"] == 0
    assert rerun["automatic_rerun_allowed"] is False
    assert rerun["holdout_authorized"] is False
    assert rerun["status"] == "failed_schema_no_automatic_rerun"
    assert rerun["outcome"] == "failed_schema_validation"
    assert rerun["failure_stage"] == "baseline_case"
    assert rerun["failed_baseline"] == "single_agent_live"
    assert rerun["failed_case_id"] == "v2_dev_001"
    assert rerun["error_type"] == "ModelSchemaError"
    assert rerun["actual_model_calls"] == 1
    assert rerun["actual_model_calls_by_baseline"] == {
        "single_agent_live": 1,
        "miner_judge_live": 0,
    }
    assert rerun["completed_live_predictions"] == 0
    assert rerun["usage_status"] == "reported"
    assert rerun["input_tokens"] == 259
    assert rerun["output_tokens"] == 391
    assert rerun["total_tokens"] == 650
    assert rerun["schema_failure_count"] == 1
    assert rerun["transport_failure_count"] == 0
    assert rerun["canonical_artifacts_written"] is False
    assert rerun["cost_usd"] is None
    assert rerun["authorization_consumed"] is True
    assert rerun["rerun_requires_new_authorization"] is True
    assert rerun["frozen_git_commit"] == "bcda80b9ea9c21d46bda05fa2160b88e19f60fa0"
    assert rerun["code_bundle_sha256"] == phase3d_freeze["sha256"]
    assert rerun["prompt_bundle_sha256"] == phase3d_freeze["prompt_bundle_sha256"]
    assert manifest["release_gate_eligible"] is False
    assert manifest["counts"]["holdout_candidates"] == 60
    assert manifest["counts"]["frozen_holdout_cases"] == 60
    assert manifest["counts"]["human_reviewed"] == 0
    assert manifest["counts"]["single_human_review"] == 60
    assert manifest["counts"]["double_human_review"] == 0
    assert manifest["counts"]["adjudicated_human_gold"] == 0
    assert manifest["annotation_policy"]["highest_completed_provenance"] == (
        "single_human_review"
    )
    assert (
        manifest["annotation_policy"][
            "second_annotator_required_for_internal_phase3_gate"
        ]
        is False
    )
    assert manifest["internal_phase3_gate"]["eligible"] is False
    assert manifest["internal_phase3_gate"]["headline_baseline"] == (
        "retrieval_judge_live"
    )
    assert manifest["model_proposal_distribution"] == {
        relation: 10 for relation in RELATIONS
    }
    for relative_path, metadata in manifest["files"].items():
        assert _sha256(V2 / relative_path) == metadata["sha256"]

    review_manifest = json.loads(
        (V2 / "review_manifest.json").read_text(encoding="utf-8")
    )
    assert review_manifest["human_annotation"]["status"] == (
        "single_human_review_complete"
    )
    comparison = review_manifest["human_model_comparison"]
    assert comparison["not_inter_human_agreement"] is True
    assert comparison["cohens_kappa_not_computed"] is True
    assert review_manifest["release_gate"]["eligible"] is False
    assert review_manifest["public_benchmark_eligible"] is False
    assert review_manifest["contains_human_labels"] is True
    assert (
        review_manifest["second_human_review_policy"][
            "required_for_internal_phase3_gate"
        ]
        is False
    )


def test_v2_live_dev_failure_artifact_is_frozen_and_secret_free() -> None:
    manifest = json.loads((V2 / "manifest.json").read_text(encoding="utf-8"))
    rerun = manifest["live_dev_rerun"]
    artifact_path = ROOT / rerun["failure_artifact"]
    assert _sha256(artifact_path) == rerun["failure_artifact_sha256"]
    assert {path.name for path in artifact_path.parent.iterdir()} == {
        "failed_attempt.json"
    }
    raw = artifact_path.read_text(encoding="utf-8")
    artifact = EvalFailureArtifact.model_validate_json(raw)
    assert artifact.git_commit == rerun["frozen_git_commit"]
    assert artifact.model_id == "deepseek-v4-flash"
    assert artifact.model_temperature_requested == 0.0
    assert artifact.model_temperature_actual_effective is None
    assert artifact.model_thinking_mode == "provider_default"
    assert artifact.maximum_expected_live_model_calls == 54
    assert artifact.actual_model_calls == 1
    assert artifact.actual_model_calls_by_baseline == {
        "single_agent_live": 1,
        "miner_judge_live": 0,
    }
    assert artifact.reported_total_token_subtotal == 650
    assert artifact.model_telemetry.total_tokens == 650
    assert artifact.model_telemetry.schema_failures == 1
    assert artifact.model_telemetry.transport_failures == 0
    assert artifact.failure_stage == "baseline_case"
    assert artifact.failed_baseline == "single_agent_live"
    assert artifact.failed_case_id == "v2_dev_001"
    assert artifact.automatic_retry_count == 0
    assert artifact.canonical_artifacts_written is False
    assert artifact.cost_usd is None
    assert artifact.contains_sensitive_payloads is False
    for forbidden in ("authorization", "bearer", "api_key", "request_headers"):
        assert forbidden not in raw.casefold()


def test_phase3e_live_dev_failure_artifact_is_frozen_and_secret_free() -> None:
    manifest = json.loads((V2 / "manifest.json").read_text(encoding="utf-8"))
    attempt = manifest["phase3e_full_dev_live_context_guard_attempt"]
    artifact_path = ROOT / attempt["failure_artifact"]

    assert _sha256(artifact_path) == attempt["failure_artifact_sha256"]
    assert {path.name for path in artifact_path.parent.iterdir()} == {
        "failed_attempt.json"
    }
    raw = artifact_path.read_text(encoding="utf-8")
    artifact = EvalFailureArtifact.model_validate_json(raw)

    assert artifact.git_commit == attempt["frozen_git_commit"]
    assert artifact.model_id == "deepseek-v4-flash"
    assert artifact.prompt_version == "openai-compatible-v2"
    assert artifact.model_temperature_requested == 0.0
    assert artifact.model_temperature_actual_effective is None
    assert artifact.model_temperature_effective_status == "unknown"
    assert artifact.model_thinking_mode == "provider_default"
    assert artifact.maximum_expected_live_model_calls == 54
    assert artifact.actual_model_calls == 44
    assert artifact.actual_model_calls_by_baseline == {
        "single_agent_live": 18,
        "miner_judge_live": 26,
    }
    assert attempt["actual_model_calls_by_stage_derived"] == {
        "single_agent": 18,
        "miner": 14,
        "judge": 12,
    }
    assert artifact.completed_live_predictions == 31
    assert artifact.failed_baseline == "miner_judge_live"
    assert artifact.failed_case_id == "v2_dev_014"
    assert artifact.failure_kind == "local_validation"
    assert artifact.error_type == "ValueError"
    assert artifact.guard_signal_codes == ()
    assert artifact.schema_diagnostic is None
    assert artifact.model_telemetry.schema_failures == 0
    assert artifact.model_telemetry.transport_failures == 0
    assert artifact.calls_with_reported_usage == 44
    assert artifact.reported_input_token_subtotal == 31207
    assert artifact.reported_output_token_subtotal == 22330
    assert artifact.reported_total_token_subtotal == 53537
    assert artifact.model_telemetry.total_tokens == 53537
    assert len(artifact.model_call_events) == 44
    assert artifact.model_latency_ms_p50 == pytest.approx(4422.921866411343)
    assert artifact.model_latency_ms_p95 == pytest.approx(8265.639363322407)
    assert artifact.automatic_retry_count == 0
    assert artifact.canonical_artifacts_written is False
    assert artifact.cost_usd is None
    assert artifact.contains_sensitive_payloads is False
    assert attempt["quality_metrics_available"] is False
    assert attempt["confusion_matrix_available"] is False
    assert attempt["deterministic_conflict_guard_failure_count"] == 0
    assert attempt["v2_holdout_rerun"] is False
    assert attempt["v3_created"] is False
    assert attempt["phase4_blocked"] is True

    top_level_keys = set(json.loads(raw))
    assert top_level_keys.isdisjoint(
        {
            "api_key",
            "authorization",
            "claim",
            "evidence",
            "headers",
            "prompt",
            "raw_content",
            "reasoning_content",
            "request_headers",
            "source",
        }
    )
    for forbidden in ("authorization", "bearer", "api_key", "request_headers"):
        assert forbidden not in raw.casefold()


def test_v2_review_packet_is_complete_and_does_not_anchor_labels() -> None:
    packet = (V2 / "REVIEW_PACKET.md").read_text(encoding="utf-8")
    reviewed_ids = re.findall(r"^### (v2_holdout_\d{3})$", packet, flags=re.MULTILINE)

    assert len(reviewed_ids) == 60
    assert len(set(reviewed_ids)) == 60
    assert "model-proposed relation is intentionally omitted" in packet
    assert "Human relation: ____________________" in packet
    assert "do not run as an evaluation" in packet
    assert "human_reviewed" in packet


def test_phase3d1_single_case_smoke_artifact_is_frozen_and_secret_free() -> None:
    manifest = json.loads((V2 / "manifest.json").read_text(encoding="utf-8"))
    provenance = manifest["phase3d1_single_case_smoke"]
    artifact_path = ROOT / provenance["artifact"]
    raw = artifact_path.read_text(encoding="utf-8")
    artifact = json.loads(raw)

    assert _sha256(artifact_path) == provenance["artifact_sha256"]
    assert {path.name for path in artifact_path.parent.iterdir()} == {
        "smoke_attempt.json"
    }
    assert artifact["artifact_type"] == "single_case_live_contract_smoke"
    assert artifact["artifact_version"] == "phase3d1-smoke-v1"
    assert artifact["git_commit"] == provenance["frozen_git_commit"]
    assert artifact["case_id"] == "v2_dev_001"
    assert artifact["baseline"] == "single_agent_live"
    assert artifact["model_id"] == "deepseek-v4-flash"
    assert artifact["thinking_mode"] == "provider_default"
    assert artifact["temperature_requested"] == 0.0
    assert artifact["temperature_actual_effective"] is None
    assert artifact["output_schema_name"] == "SingleAgentLiveOutput"
    assert artifact["prompt_version"] == "openai-compatible-v2"
    assert artifact["contract_status"] == "passed"
    assert artifact["quality_correct"] is True
    assert artifact["evidence_span_is_source_substring"] is True
    assert artifact["actual_model_calls"] == 1
    assert artifact["maximum_provider_requests"] == 1
    assert artifact["automatic_retry_count"] == 0
    assert artifact["repair_model_calls"] == 0
    assert artifact["miner_judge_live_calls"] == 0
    assert artifact["full_dev_executed"] is False
    assert artifact["v2_holdout_executed"] is False
    assert artifact["input_tokens"] == 553
    assert artifact["output_tokens"] == 177
    assert artifact["total_tokens"] == 730
    assert artifact["latency_ms"] == pytest.approx(2151.2728731613606)
    assert artifact["finish_reason"] == "stop"
    assert artifact["cost_usd"] is None
    assert artifact["schema_diagnostic"] is None
    assert artifact["raw_model_content_persisted"] is False
    assert artifact["reasoning_content_persisted"] is False
    assert artifact["model_reason_persisted"] is False
    assert artifact["prompt_claim_source_persisted"] is False
    assert artifact["request_or_response_headers_persisted"] is False
    assert artifact["credentials_persisted"] is False
    for forbidden_key in {
        "api_key",
        "authorization",
        "base_url",
        "claim",
        "evidence_span",
        "headers",
        "prompt",
        "raw_content",
        "reason",
        "reasoning_content",
        "source_content",
    }:
        assert forbidden_key not in artifact


def test_phase3d1_miner_judge_smoke_artifact_is_frozen_and_secret_free() -> None:
    manifest = json.loads((V2 / "manifest.json").read_text(encoding="utf-8"))
    provenance = manifest["phase3d1_miner_judge_smoke"]
    artifact_path = ROOT / provenance["artifact"]
    raw = artifact_path.read_text(encoding="utf-8")
    artifact = json.loads(raw)

    assert _sha256(artifact_path) == provenance["artifact_sha256"]
    assert {path.name for path in artifact_path.parent.iterdir()} == {
        "smoke_attempt.json"
    }
    assert artifact["artifact_type"] == "single_case_miner_judge_contract_smoke"
    assert artifact["artifact_version"] == "phase3d1-miner-judge-smoke-v1"
    assert artifact["git_commit"] == provenance["frozen_git_commit"]
    assert artifact["case_id"] == "v2_dev_001"
    assert artifact["baseline"] == "miner_judge_live"
    assert artifact["selected_case_count"] == 1
    assert artifact["selected_baseline_count"] == 1
    assert artifact["model_id"] == "deepseek-v4-flash"
    assert artifact["thinking_mode"] == "provider_default"
    assert artifact["temperature_requested"] == 0.0
    assert artifact["temperature_actual_effective"] is None
    assert artifact["prompt_version"] == "openai-compatible-v2"
    assert artifact["contract_status"] == "passed"
    assert artifact["quality_correct"] is True
    assert artifact["actual_model_calls"] == 2
    assert artifact["actual_model_calls_by_stage"] == {"judge": 1, "miner": 1}
    assert artifact["maximum_provider_requests"] == 2
    assert artifact["automatic_retry_count"] == 0
    assert artifact["repair_model_calls"] == 0
    assert artifact["single_agent_live_calls"] == 0
    assert artifact["full_dev_executed"] is False
    assert artifact["v2_holdout_executed"] is False
    assert artifact["schema_failure_count"] == 0
    assert artifact["transport_failure_count"] == 0

    miner = artifact["miner"]
    judge = artifact["judge"]
    assert miner["schema_name"] == "MinerOutput"
    assert judge["schema_name"] == "JudgeOutput"
    assert miner["status"] == judge["status"] == "passed"
    assert miner["schema_validated"] is judge["schema_validated"] is True
    assert miner["actual_calls"] == judge["actual_calls"] == 1
    assert miner["finish_reason"] == judge["finish_reason"] == "stop"
    assert all(miner["validation"].values())
    assert all(judge["validation"].values())
    assert artifact["input_tokens"] == (miner["input_tokens"] + judge["input_tokens"])
    assert artifact["output_tokens"] == (
        miner["output_tokens"] + judge["output_tokens"]
    )
    assert artifact["total_tokens"] == (miner["total_tokens"] + judge["total_tokens"])
    assert artifact["total_tokens"] == (
        artifact["input_tokens"] + artifact["output_tokens"]
    )
    assert artifact["cost_usd"] is None
    assert artifact["evidence_span_present"] is True
    assert artifact["evidence_span_is_candidate_substring"] is True

    cases = _jsonl(V2 / "dev.jsonl")
    case = next(item for item in cases if item["case_id"] == "v2_dev_001")
    sources = _jsonl(V2 / "sources" / "synthetic_sources.jsonl")
    source = next(item for item in sources if item["source_id"] == case["source_id"])
    assert (
        artifact["evidence_span_sha256"]
        == hashlib.sha256(case["gold_evidence_span"].encode("utf-8")).hexdigest()
    )
    for sensitive_text in (
        case["claim_text"],
        case["gold_evidence_span"],
        source["content"],
    ):
        assert sensitive_text not in raw

    assert artifact["raw_model_content_persisted"] is False
    assert artifact["reasoning_content_persisted"] is False
    assert artifact["miner_or_judge_output_persisted"] is False
    assert artifact["model_reason_or_relation_persisted"] is False
    assert artifact["prompt_claim_source_or_evidence_persisted"] is False
    assert artifact["request_or_response_headers_persisted"] is False
    assert artifact["credentials_persisted"] is False

    forbidden_keys = {
        "api_key",
        "authorization",
        "base_url",
        "claim",
        "evidence_span",
        "headers",
        "prompt",
        "raw_content",
        "reason",
        "reasoning_content",
        "relation",
        "source_content",
    }

    def collect_keys(value: object) -> set[str]:
        if isinstance(value, dict):
            return set(value) | {
                key for child in value.values() for key in collect_keys(child)
            }
        if isinstance(value, list):
            return {key for child in value for key in collect_keys(child)}
        return set()

    assert collect_keys(artifact).isdisjoint(forbidden_keys)


def test_phase3e3_live_dev_stability_artifacts_are_frozen_and_safe() -> None:
    manifest = json.loads((V2 / "manifest.json").read_text(encoding="utf-8"))
    phase3e3 = manifest["phase3e3_judge_validation_repair"]
    smoke_path = ROOT / phase3e3["smoke_artifact"]
    aggregate_path = ROOT / phase3e3["aggregate_artifact"]
    aggregate_report_path = ROOT / phase3e3["aggregate_report"]

    assert _sha256(smoke_path) == phase3e3["smoke_artifact_sha256"]
    assert _sha256(aggregate_path) == phase3e3["aggregate_artifact_sha256"]
    assert _sha256(aggregate_report_path) == (phase3e3["aggregate_report_sha256"])

    roots = [
        smoke_path.parent,
        *(ROOT / path for path in phase3e3["dev_run_directories"]),
        aggregate_path.parent,
    ]
    artifact_paths = [
        path for root in roots for path in root.iterdir() if path.is_file()
    ]
    assert len(artifact_paths) == 15
    assert (
        _artifact_record_bundle(artifact_paths) == (phase3e3["artifact_bundle_sha256"])
    )

    smoke_raw = smoke_path.read_text(encoding="utf-8")
    smoke = json.loads(smoke_raw)
    assert smoke["attempts_completed"] == 5
    assert smoke["all_canonical_predictions_valid"] is True
    assert smoke["resource_summary"]["calls"] == 5
    assert smoke["resource_summary"]["total_tokens"] == 8725
    assert smoke["contains_sensitive_payloads"] is False
    assert all(
        prediction["predicted_relation"] in RELATIONS
        for prediction in smoke["predictions"]
    )

    aggregate_raw = aggregate_path.read_text(encoding="utf-8")
    aggregate = json.loads(aggregate_raw)
    assert aggregate["operational_success"] is True
    assert aggregate["three_of_three_complete"] is True
    assert aggregate["attempted_runs"] == aggregate["complete_runs"] == 3
    assert all(aggregate["consistency_checks"].values())
    assert aggregate["resource_summary"]["calls"] == 93
    assert aggregate["resource_summary"]["total_tokens"] == 109158
    assert aggregate["failure_summary"] == {
        "schema": 0,
        "transport": 0,
        "guard": 0,
        "local_validation": 0,
    }
    assert aggregate["unstable_cases"] == [
        {
            "baseline": "retrieval_judge_live",
            "case_id": "v2_dev_017",
            "gold": "not_checkable",
            "predictions": [
                "not_in_source",
                "not_in_source",
                "partially_entailed",
            ],
            "agreement": False,
        }
    ]
    assert aggregate["contains_sensitive_payloads"] is False

    for run_name, hashes in aggregate["input_artifact_sha256"].items():
        run_directory = ROOT / "eval_runs" / run_name
        assert {
            filename: _sha256(run_directory / filename) for filename in hashes
        } == hashes

    for raw in (smoke_raw, aggregate_raw):
        lowered = raw.casefold()
        for forbidden in (
            "api_key",
            "authorization",
            "raw_model_content",
            "reasoning_content",
            "request_headers",
        ):
            assert forbidden not in lowered
