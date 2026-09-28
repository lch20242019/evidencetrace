from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from evidencetrace.eval.dataset import compute_case_hash, load_dataset

ROOT = Path(__file__).resolve().parents[1]
V3_ZH = ROOT / "eval_sets" / "v3_zh"
DATASET = V3_ZH / "holdout_single_human_zh.jsonl"
SOURCES = V3_ZH / "sources" / "source_snapshots_zh.jsonl"
CANDIDATES = V3_ZH / "holdout_candidates_zh.jsonl"
MANIFEST = V3_ZH / "manifest.json"
EXPECTED_REVIEWER_SHA256 = (
    "538e738fca9036df65091a985a7b9d590fb062f69f813fc3cb5a5890b245b4ac"
)
EXPECTED_PARENT_BUNDLE_SHA256 = (
    "21f0d790007f492b9fafdb022894981f1fc789e86072c55a899634ba93f2b8ef"
)
EXPECTED_CONSUMED_V2_SHA256 = (
    "0d19004b627f4ecabf9dd40fd28303b0539b89f36f3f7f5cec135c47a6582c75"
)
SUBSTANTIVE_RELATIONS = {"entailed", "partially_entailed", "contradicted"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def test_v3_zh_gold_is_complete_loadable_and_grounded() -> None:
    dataset = load_dataset(DATASET)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert len(dataset.cases) == 72
    assert not dataset.dev_cases
    assert len(dataset.test_cases) == 72
    assert Counter(case.gold_relation.value for case in dataset.cases) == {
        "entailed": 10,
        "partially_entailed": 9,
        "contradicted": 13,
        "not_in_source": 16,
        "source_unavailable": 12,
        "not_checkable": 12,
    }
    assert len(dataset.sources) == 24
    assert sum(source.available for source in dataset.sources.values()) == 20
    assert sum(not source.available for source in dataset.sources.values()) == 4
    for case in dataset.cases:
        source = dataset.sources[case.source_id]
        assert case.annotation_status == "single_human_review"
        assert case.reviewer_record_sha256 == EXPECTED_REVIEWER_SHA256
        assert case.source_sha256 == source.content_hash
        assert case.case_hash == compute_case_hash(case)
        if case.gold_relation.value in SUBSTANTIVE_RELATIONS:
            assert case.gold_evidence_span
            assert case.gold_evidence_span in source.content
        else:
            assert case.gold_evidence_span is None
    assert manifest["case_count"] == 72
    assert manifest["valid_case_count"] == 72
    assert manifest["blocked_case_count"] == 0


def test_v3_zh_candidates_sources_and_gold_exclude_answer_leakage() -> None:
    candidates = _jsonl(CANDIDATES)
    gold = _jsonl(DATASET)
    source_counts = Counter(item["source_id"] for item in candidates)
    forbidden_candidates = {
        "gold_relation",
        "gold_evidence",
        "model_output",
        "model_proposal",
        "proposed_relation",
        "target_relation",
    }
    forbidden_gold = {
        "reviewer_name",
        "reviewer_id",
        "reviewer_notes",
        "model_output",
        "model_proposal",
        "proposed_relation",
    }
    assert len(candidates) == 72
    assert len({item["candidate_id"] for item in candidates}) == 72
    assert set(source_counts.values()) == {3}
    assert all(set(item).isdisjoint(forbidden_candidates) for item in candidates)
    assert all(set(item).isdisjoint(forbidden_gold) for item in gold)
    assert "lch" not in DATASET.read_text(encoding="utf-8")


def test_v3_zh_manifest_hashes_bundle_and_internal_status() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["pack_id"] == "phase3f-v3-zh-single-human-gold"
    assert manifest["benchmark_provenance"] == {
        "annotation_language": "zh-CN",
        "benchmark_scope": "internal_chinese_canonical",
        "canonical_source_language": "zh-CN",
        "developer_agent_exposure_after_code_freeze": True,
        "evaluation_model": "deepseek-v4-flash",
        "evaluation_model_exposure": False,
        "human_labeled": True,
        "public_benchmark_eligible": False,
        "second_human_review": False,
        "source_origin": "machine_translated_derivative",
        "translation_fidelity_to_english_claimed": False,
        "translation_human_fidelity_review_required": False,
        "translation_model": "OpenAI Codex",
    }
    assert manifest["benchmark_status"] == {
        "annotation_status": "single_human_review_complete",
        "counts_as_human_review": True,
        "formal_internal_gate_eligible": True,
        "holdout_blind_to_evaluation_model": True,
        "holdout_execution_status": "not_run",
        "public_benchmark_eligible": False,
    }
    assert manifest["input_hashes"]["human_annotation_sha256"] == (
        EXPECTED_REVIEWER_SHA256
    )
    for relative, expected in manifest["files"].items():
        assert _sha256(ROOT / relative) == expected
    payload = "".join(
        f"{_sha256(ROOT / relative)}  {relative}\n"
        for relative in manifest["ordered_bundle_paths"]
    )
    assert (
        hashlib.sha256(payload.encode()).hexdigest()
        == (manifest["ordered_bundle_sha256"])
    )


def test_v3_zh_parent_and_consumed_v2_boundaries_remain_frozen() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    payload = "".join(
        f"{_sha256(ROOT / relative)}  {relative}\n"
        for relative in manifest["english_parent"]["ordered_bundle_paths"]
    )
    assert hashlib.sha256(payload.encode()).hexdigest() == (
        EXPECTED_PARENT_BUNDLE_SHA256
    )
    assert _sha256(ROOT / "eval_sets/v2/holdout_single_human.jsonl") == (
        EXPECTED_CONSUMED_V2_SHA256
    )
    assert manifest["english_parent"]["used_for_chinese_evidence_validation"] is False


def test_v3_zh_historical_blockers_are_all_human_resolved() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["historical_blocker_resolution"] == {
        "v3_holdout_candidate_018": "human_corrected_and_valid",
        "v3_holdout_candidate_039": "human_corrected_and_valid",
        "v3_holdout_candidate_047": "human_corrected_and_valid",
        "v3_holdout_candidate_060": "human_corrected_and_valid",
    }
    assert manifest["generation"]["real_model_calls"] == 0
    assert manifest["generation"]["production_code_modified"] is False
