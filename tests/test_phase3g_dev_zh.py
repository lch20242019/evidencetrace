from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

from evidencetrace.checks.deterministic import is_high_confidence_subjective
from evidencetrace.eval.baselines import (
    build_evidence_chunks,
    estimate_full_context_size,
)
from evidencetrace.eval.dataset import compute_case_hash, load_dataset
from evidencetrace.eval.router import AdaptiveRouterConfig, select_adaptive_route
from evidencetrace.retrieval.rank import LexicalRetriever

ROOT = Path(__file__).parents[1]
PACK = ROOT / "eval_sets" / "phase3g_dev_zh"
DATASET = PACK / "dev.jsonl"
PHASE3G_B_FREEZE_COMMIT = "f5d5521803ffb9b058afafdd33e852541b86da5a"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bundle_hash(paths: list[str]) -> str:
    payload = "".join(f"{_sha256(ROOT / path)}  {path}\n" for path in paths)
    return hashlib.sha256(payload.encode()).hexdigest()


def _bundle_hash_at_commit(paths: list[str], commit: str) -> str:
    lines = []
    for path in paths:
        content = subprocess.run(
            ["git", "show", f"{commit}:{path}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        lines.append(f"{hashlib.sha256(content).hexdigest()}  {path}\n")
    return hashlib.sha256("".join(lines).encode()).hexdigest()


def test_phase3g_dev_structure_relations_strata_and_case_hashes() -> None:
    loaded = load_dataset(DATASET)
    cases = loaded.cases
    assert len(cases) == 72
    assert len(loaded.sources) == 24
    assert Counter(case.gold_relation.value for case in cases) == {
        "entailed": 12,
        "partially_entailed": 12,
        "contradicted": 12,
        "not_in_source": 12,
        "not_checkable": 12,
        "source_unavailable": 12,
    }
    assert Counter(case.source_length_stratum for case in cases) == {
        "short": 21,
        "medium": 21,
        "long": 18,
        "unavailable": 12,
    }
    assert set(Counter(case.source_id for case in cases).values()) == {3}
    assert all(case.split == "dev" for case in cases)
    assert all(case.annotation_status == "provisional" for case in cases)
    assert all(case.case_hash == compute_case_hash(case) for case in cases)


def test_expected_routes_match_frozen_policy_without_model_calls() -> None:
    loaded = load_dataset(DATASET)
    config = AdaptiveRouterConfig()
    for case in loaded.cases:
        source = loaded.sources[case.source_id]
        chunks = build_evidence_chunks(source)
        decision = select_adaptive_route(
            source_available=source.available,
            purely_subjective=is_high_confidence_subjective(str(case.claim_text)),
            chunk_count=len(chunks),
            estimated_context_size=estimate_full_context_size(case, source),
            config=config,
        )
        assert case.expected_chunk_count == len(chunks)
        assert case.expected_route == decision.selected_route
    assert Counter(case.expected_route for case in loaded.cases) == {
        "deterministic_source_unavailable": 12,
        "deterministic_not_checkable": 12,
        "full_context_single_agent": 17,
        "retrieval_judge": 31,
    }


def test_long_sources_have_real_distractors_and_gold_is_retrievable() -> None:
    loaded = load_dataset(DATASET)
    long_sources = {
        source.source_id: source
        for source in loaded.sources.values()
        if any(
            case.source_id == source.source_id and case.source_length_stratum == "long"
            for case in loaded.cases
        )
    }
    assert len(long_sources) == 6
    assert all(
        len(build_evidence_chunks(source)) == 7 for source in long_sources.values()
    )
    for case in loaded.cases:
        if not case.gold_evidence_span:
            continue
        source = loaded.sources[case.source_id]
        retrieved = LexicalRetriever(
            build_evidence_chunks(source), neighbor_window=0
        ).search(str(case.claim_text), top_k=5)
        assert any(
            case.gold_evidence_span in item.text or item.text in case.gold_evidence_span
            for item in retrieved
        ), case.case_id


def test_leakage_audit_and_freeze_hashes_are_self_consistent() -> None:
    manifest = json.loads(PACK.joinpath("manifest.json").read_text())
    audit = json.loads(PACK.joinpath("leakage_audit.json").read_text())
    assert manifest["status"] == "frozen_provisional_dev_not_run_live"
    assert manifest["annotation_contract"] == {
        "status": "deterministic_provisional_dev",
        "independent_holdout": False,
        "formal_gate_eligible": False,
        "pair_extraction_metrics": "not_applicable",
    }
    assert manifest["future_live_dev_preregistration"]["baselines"] == [
        "single_agent_live",
        "retrieval_judge_live",
        "adaptive_live",
    ]
    assert manifest["future_live_dev_preregistration"]["required_reporting"] == [
        "overall",
        "short",
        "medium",
        "long",
    ]
    assert audit["status"] == "passed"
    assert not audit["claim_similarity_flags"]
    assert not audit["source_similarity_flags"]
    assert not any(value for key, value in audit.items() if key.endswith("collisions"))
    for relative, expected in manifest["files"].items():
        assert _sha256(ROOT / relative) == expected
    assert (
        _bundle_hash(manifest["ordered_bundle_paths"])
        == (manifest["ordered_bundle_sha256"])
    )
    freeze = manifest["freeze_hashes"]
    assert (
        _bundle_hash_at_commit(freeze["code_paths"], PHASE3G_B_FREEZE_COMMIT)
        == freeze["code_sha256"]
    )
    assert (
        _bundle_hash_at_commit(
            freeze["prompt_schema_paths"],
            PHASE3G_B_FREEZE_COMMIT,
        )
        == freeze["prompt_schema_sha256"]
    )
    assert _sha256(PACK / "LEAKAGE_AUDIT.md") == (freeze["leakage_audit_sha256"])
    assert manifest["model_calls"] == 0
    assert manifest["v3_zh_rerun"] is False
    assert manifest["v4_created"] is False
