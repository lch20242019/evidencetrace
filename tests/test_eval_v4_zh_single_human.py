from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from evidencetrace.eval.dataset import (
    compute_case_hash,
    load_dataset,
    validate_dataset,
)
from evidencetrace.models import Relation

ROOT = Path(__file__).parents[1]
PACK = ROOT / "eval_sets/v4_zh"
GOLD = PACK / "holdout_single_human_zh.jsonl"
FROZEN = PACK / "holdout_single_human_zh.frozen_hashes.json"
MANIFEST = PACK / "review_manifest.json"
RICH_SOURCES = PACK / "sources/source_snapshots_zh.jsonl"
CANDIDATES = PACK / "holdout_candidates_zh.jsonl"
SCRIPT = ROOT / "scripts/freeze_v4_zh_single_human.py"
REVIEWER_SHA256 = "46657539fd1d67b22fa15c9a34ee001a198b0952fb808a911d94bfa58750598a"
DISTRIBUTION = {
    "entailed": 11,
    "partially_entailed": 11,
    "contradicted": 11,
    "not_in_source": 15,
    "source_unavailable": 12,
    "not_checkable": 12,
}
ROUTES = {
    "full_context_single_agent": 17,
    "retrieval_judge": 31,
    "deterministic_not_checkable": 12,
    "deterministic_source_unavailable": 12,
}
BLIND_HASHES = {
    "source": "34f9128c38ae742a431b6be02ef44bc99887fad648b42e0bbe05c9bc61054d4f",
    "candidate": ("9803cc48dfa1a95911ae179a837d0448e06f6cef76a0b6762b14be1417f4dff3"),
    "packet": ("d143adf4c4e85aa197a5dd109791d1aadcfd5bd3276e7488468a83ce0f5b03bd"),
    "bundle": ("0e8dc6ccc0bf66302032cc94f74591e3ad4189ca04e09864dcb1449316be3d9d"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def bundle_hash(paths: list[str]) -> str:
    payload = "".join(f"{sha256(ROOT / path)}  {path}\n" for path in paths)
    return hashlib.sha256(payload.encode()).hexdigest()


def test_v4_zh_gold_loads_and_matches_frozen_cases() -> None:
    loaded = load_dataset(GOLD)
    frozen = json.loads(FROZEN.read_text())
    validated = validate_dataset(
        loaded.cases,
        sources=loaded.sources,
        frozen_hashes=frozen,
    )
    assert len(validated) == 72
    assert len(loaded.sources) == 24
    assert len(frozen) == 72
    assert all(case.annotation_status == "single_human_review" for case in validated)
    assert all(case.case_hash == compute_case_hash(case) for case in validated)


def test_v4_zh_gold_matches_blind_candidates_and_native_sources() -> None:
    loaded = load_dataset(GOLD)
    candidates = {str(row["candidate_id"]): row for row in rows(CANDIDATES)}
    sources = {str(row["source_id"]): row for row in rows(RICH_SOURCES)}
    assert {case.case_id for case in loaded.cases} == set(candidates)
    for case in loaded.cases:
        candidate = candidates[case.case_id]
        source = sources[case.source_id]
        fixture = loaded.sources[case.source_id]
        assert case.claim_text == candidate["claim"]
        assert case.source_id == candidate["source_id"]
        assert fixture.content == source["content"]
        assert fixture.url == source["url"]
        assert fixture.available == source["available"]
        assert source["original_language"] == "zh-CN"
        assert case.source_sha256 == fixture.content_hash
        assert case.source_length_stratum == source["source_length_stratum"]
        assert case.expected_chunk_count == source["expected_chunk_count"]


def test_v4_zh_labels_and_evidence_are_complete() -> None:
    loaded = load_dataset(GOLD)
    assert Counter(case.gold_relation.value for case in loaded.cases) == DISTRIBUTION
    substantive = {
        Relation.ENTAILED,
        Relation.PARTIALLY_ENTAILED,
        Relation.CONTRADICTED,
    }
    contiguous = 0
    empty = 0
    for case in loaded.cases:
        source = loaded.sources[case.source_id]
        if case.gold_relation in substantive:
            assert case.gold_evidence_span
            assert case.gold_evidence_span in source.content
            contiguous += 1
        else:
            assert case.gold_evidence_span is None
            empty += 1
        if source.available is False:
            assert case.gold_relation == Relation.SOURCE_UNAVAILABLE
            assert source.content == ""
    assert (contiguous, empty) == (33, 39)


def test_v4_zh_public_gold_excludes_private_reviewer_data() -> None:
    gold = rows(GOLD)
    assert all(row["reviewer_record_sha256"] == REVIEWER_SHA256 for row in gold)
    assert all(row["generation_code_sha256"] == sha256(SCRIPT) for row in gold)
    assert all(
        row["annotation_notes"]
        == (
            "Frozen single-human label; private identity and full notes remain "
            "outside the repository."
        )
        for row in gold
    )
    forbidden = {"reviewer_name", "reviewer_id", "reviewer_notes", "full_notes"}
    assert all(not (forbidden & row.keys()) for row in gold)


def test_v4_zh_review_manifest_status_and_provenance() -> None:
    manifest = json.loads(MANIFEST.read_text())
    assert manifest["status"] == ("single_human_review_complete_internal_gate_not_run")
    assert manifest["relation_distribution"] == DISTRIBUTION
    assert manifest["evidence_validation"] == {
        "contiguous_substring_count": 33,
        "required_empty_count": 39,
        "invalid_count": 0,
    }
    assert manifest["benchmark_status"] == {
        "annotation_status": "single_human_review_complete",
        "counts_as_human_review": True,
        "annotation_language": "zh-CN",
        "canonical_source_language": "zh-CN",
        "original_source_language": "zh-CN",
        "evaluation_model_exposure": False,
        "holdout_blind_to_evaluation_model": True,
        "second_human_review": False,
        "public_benchmark_eligible": False,
        "formal_internal_gate_eligible": True,
        "holdout_execution_status": "not_run",
    }
    provenance = manifest["private_reviewer_provenance"]
    assert provenance["reviewer_name_or_pseudonym_present"] is True
    assert provenance["reviewer_id_present"] is True
    assert provenance["independent_human_annotation_attestation_verified"] is True
    assert provenance["viewed_model_output"] is False
    assert provenance["identity_and_full_notes_published"] is False
    assert provenance["reviewer_record_sha256"] == REVIEWER_SHA256
    assert "reviewer_name" not in provenance
    assert "reviewer_id" not in provenance


def test_v4_zh_blind_freeze_routes_and_generation_are_preserved() -> None:
    manifest = json.loads(MANIFEST.read_text())
    inputs = manifest["input_hashes"]
    assert inputs["source_snapshots_sha256"] == BLIND_HASHES["source"]
    assert inputs["candidates_sha256"] == BLIND_HASHES["candidate"]
    assert inputs["blank_reviewer_packet_sha256"] == BLIND_HASHES["packet"]
    assert inputs["candidate_bundle_sha256"] == BLIND_HASHES["bundle"]
    assert sha256(RICH_SOURCES) == BLIND_HASHES["source"]
    assert sha256(CANDIDATES) == BLIND_HASHES["candidate"]
    assert sha256(PACK / "REVIEW_PACKET.md") == BLIND_HASHES["packet"]
    assert manifest["adaptive_expected_routes"] == ROUTES
    assert Counter(case.expected_route for case in load_dataset(GOLD).cases) == ROUTES
    assert manifest["generation"] == {
        "tool": "scripts/freeze_v4_zh_single_human.py",
        "tool_sha256": sha256(SCRIPT),
        "real_model_calls": 0,
        "holdout_executed": False,
        "production_code_modified": False,
        "intended_construction_strata_used": False,
    }


def test_v4_zh_gold_bundle_and_consumed_byte_hashes() -> None:
    manifest = json.loads(MANIFEST.read_text())
    for relative, expected in manifest["files"].items():
        assert sha256(ROOT / relative) == expected
    assert (
        bundle_hash(manifest["ordered_bundle_paths"])
        == (manifest["ordered_bundle_sha256"])
    )
    for relative, expected in manifest["consumed_byte_only_hashes"].items():
        assert sha256(ROOT / relative) == expected
