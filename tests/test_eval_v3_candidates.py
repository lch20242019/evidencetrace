from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / "eval_sets" / "v3"
CANDIDATES_PATH = V3 / "holdout_candidates.jsonl"
SOURCES_PATH = V3 / "sources" / "source_snapshots.jsonl"
CONSUMED_V2_HOLDOUT = ROOT / "eval_sets" / "v2" / "holdout_single_human.jsonl"
CONSUMED_V2_SHA256 = "0d19004b627f4ecabf9dd40fd28303b0539b89f36f3f7f5cec135c47a6582c75"

PRIOR_CASE_PATHS = (
    ROOT / "eval_sets" / "core.jsonl",
    ROOT / "eval_sets" / "v2" / "holdout_candidates.jsonl",
    ROOT / "eval_sets" / "v2" / "dev.jsonl",
)
PRIOR_SOURCE_PATHS = (
    ROOT / "eval_sets" / "sources" / "seed_sources.jsonl",
    ROOT / "eval_sets" / "v2" / "sources" / "synthetic_sources.jsonl",
)

CLAIM_TOKEN_THRESHOLD = 0.72
CLAIM_WORD_TRIGRAM_THRESHOLD = 0.35
CLAIM_CHAR_5GRAM_THRESHOLD = 0.60
SOURCE_TOKEN_THRESHOLD = 0.70
SOURCE_CHAR_5GRAM_THRESHOLD = 0.55

CANDIDATE_KEYS = {
    "candidate_id",
    "claim",
    "source_id",
    "source_snapshot_path",
    "candidate_origin",
    "annotation_status",
    "evaluation_status",
    "blind_status",
}
SOURCE_KEYS = {
    "source_id",
    "entity",
    "title",
    "url",
    "retrieved_at_utc",
    "published_at_utc",
    "available",
    "retrieval_status",
    "snapshot_kind",
    "content",
    "content_sha256",
    "provenance",
}
FORBIDDEN_CANDIDATE_KEYS = {
    "annotation_notes",
    "construction_stratum",
    "gold_evidence_span",
    "gold_relation",
    "human_evidence_span",
    "human_relation",
    "model_output",
    "proposal_status",
    "proposed_relation",
    "reviewer_notes",
    "target_relation",
}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalize(text: str) -> str:
    folded = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.findall(r"[a-z0-9]+", folded))


def _tokens(text: str) -> set[str]:
    return set(_normalize(text).split())


def _word_ngrams(text: str, size: int = 3) -> set[tuple[str, ...]]:
    words = _normalize(text).split()
    return {
        tuple(words[index : index + size])
        for index in range(max(0, len(words) - size + 1))
    }


def _char_ngrams(text: str, size: int = 5) -> set[str]:
    normalized = _normalize(text)
    return {
        normalized[index : index + size]
        for index in range(max(0, len(normalized) - size + 1))
    }


def _jaccard(left: set[Any], right: set[Any]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _similarity(left: str, right: str) -> tuple[float, float, float]:
    return (
        _jaccard(_tokens(left), _tokens(right)),
        _jaccard(_word_ngrams(left), _word_ngrams(right)),
        _jaccard(_char_ngrams(left), _char_ngrams(right)),
    )


def _claim_similarity_flag(scores: tuple[float, float, float]) -> bool:
    token, word_trigram, char_5gram = scores
    return token >= CLAIM_TOKEN_THRESHOLD and (
        word_trigram >= CLAIM_WORD_TRIGRAM_THRESHOLD
        or char_5gram >= CLAIM_CHAR_5GRAM_THRESHOLD
    )


def _source_similarity_flag(scores: tuple[float, float, float]) -> bool:
    token, _, char_5gram = scores
    return token >= SOURCE_TOKEN_THRESHOLD and (
        char_5gram >= SOURCE_CHAR_5GRAM_THRESHOLD
    )


def _prior_claims() -> list[tuple[str, str]]:
    claims: list[tuple[str, str]] = []
    for path in PRIOR_CASE_PATHS:
        for row in _jsonl(path):
            claim = row.get("claim_text", row.get("claim"))
            if isinstance(claim, str):
                identifier = str(row.get("case_id", row.get("candidate_id")))
                claims.append((f"{path.relative_to(ROOT)}:{identifier}", claim))
    return claims


def _prior_sources() -> list[tuple[str, dict[str, Any]]]:
    sources: list[tuple[str, dict[str, Any]]] = []
    for path in PRIOR_SOURCE_PATHS:
        for row in _jsonl(path):
            identifier = str(row["source_id"])
            sources.append((f"{path.relative_to(ROOT)}:{identifier}", row))
    return sources


def _entity_key(entity: str) -> str:
    normalized = _normalize(entity)
    words = [
        word
        for word in normalized.split()
        if not re.fullmatch(r"v?\d+(?:\d+)*", word)
        and word not in {"python", "unavailable", "release", "probe", "patch"}
    ]
    return " ".join(words)


def test_v3_blind_candidate_structure_and_source_distribution() -> None:
    candidates = _jsonl(CANDIDATES_PATH)
    sources = _jsonl(SOURCES_PATH)
    source_by_id = {source["source_id"]: source for source in sources}

    assert len(candidates) == 72
    assert len(sources) == 24
    assert len(source_by_id) == len(sources)
    assert {candidate["candidate_id"] for candidate in candidates} == {
        f"v3_holdout_candidate_{number:03d}" for number in range(1, 73)
    }
    assert all(set(candidate) == CANDIDATE_KEYS for candidate in candidates)
    assert all(
        set(candidate).isdisjoint(FORBIDDEN_CANDIDATE_KEYS) for candidate in candidates
    )
    assert all(candidate["claim"].strip() for candidate in candidates)
    assert all(
        candidate["candidate_origin"] == "phase3f_independent_blind_construction"
        for candidate in candidates
    )
    assert all(
        candidate["annotation_status"] == "unreviewed_blind_candidate"
        and candidate["evaluation_status"] == "not_run"
        and candidate["blind_status"] == "case_level_relation_not_recorded"
        for candidate in candidates
    )
    assert all(candidate["source_id"] in source_by_id for candidate in candidates)
    source_counts = Counter(candidate["source_id"] for candidate in candidates)
    assert set(source_counts.values()) == {3}
    assert max(source_counts.values()) <= 4

    raw = CANDIDATES_PATH.read_text(encoding="utf-8").casefold()
    for forbidden in (
        "gold_relation",
        "gold_evidence",
        "human_relation",
        "human_evidence",
        "model_proposal",
        "proposed_relation",
        "reviewer_notes",
        "target_relation",
    ):
        assert forbidden not in raw


def test_v3_source_provenance_availability_and_content_hashes() -> None:
    sources = _jsonl(SOURCES_PATH)
    assert all(set(source) == SOURCE_KEYS for source in sources)
    assert len({source["url"] for source in sources}) == len(sources)
    assert len({source["entity"].casefold() for source in sources}) == len(sources)
    assert sum(source["available"] for source in sources) == 20
    assert sum(not source["available"] for source in sources) == 4

    content_hashes: set[str] = set()
    for source in sources:
        assert re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z",
            source["retrieved_at_utc"],
        )
        assert source["url"].startswith("https://github.com/")
        assert source["provenance"]["repository"]
        assert source["provenance"]["release_tag"]
        if source["available"]:
            assert source["retrieval_status"] == "http_200"
            assert source["snapshot_kind"] == "official_release_note_excerpt"
            assert source["content"]
            assert source["published_at_utc"]
            expected_hash = hashlib.sha256(source["content"].encode()).hexdigest()
            assert source["content_sha256"] == expected_hash
            assert source["content_sha256"] not in content_hashes
            content_hashes.add(source["content_sha256"])
        else:
            assert source["retrieval_status"] == "http_404"
            assert source["snapshot_kind"] == (
                "official_repository_unavailable_url_probe"
            )
            assert source["content"] is None
            assert source["content_sha256"] is None
            assert source["published_at_utc"] is None
            assert source["provenance"]["api_url"] is None
            assert source["provenance"]["excerpt_policy"] == (
                "no content captured or inferred"
            )


def test_v3_internal_exact_normalized_and_similarity_leakage() -> None:
    candidates = _jsonl(CANDIDATES_PATH)
    sources = _jsonl(SOURCES_PATH)
    normalized_claims = [_normalize(candidate["claim"]) for candidate in candidates]
    assert len(set(normalized_claims)) == len(normalized_claims)

    cross_source_findings: list[tuple[str, str, tuple[float, float, float]]] = []
    for left, right in combinations(candidates, 2):
        if left["source_id"] == right["source_id"]:
            continue
        scores = _similarity(left["claim"], right["claim"])
        if _claim_similarity_flag(scores):
            cross_source_findings.append(
                (left["candidate_id"], right["candidate_id"], scores)
            )
    assert cross_source_findings == []

    available = [source for source in sources if source["available"]]
    source_findings: list[tuple[str, str, tuple[float, float, float]]] = []
    for left, right in combinations(available, 2):
        scores = _similarity(left["content"], right["content"])
        if _source_similarity_flag(scores):
            source_findings.append((left["source_id"], right["source_id"], scores))
    assert source_findings == []


def test_v3_has_no_exact_or_similarity_leakage_from_prior_allowed_corpora() -> None:
    candidates = _jsonl(CANDIDATES_PATH)
    sources = _jsonl(SOURCES_PATH)
    prior_claims = _prior_claims()
    prior_sources = _prior_sources()

    prior_claim_norms = {_normalize(claim) for _, claim in prior_claims}
    assert {_normalize(candidate["claim"]) for candidate in candidates}.isdisjoint(
        prior_claim_norms
    )

    prior_source_ids = {row["source_id"] for _, row in prior_sources}
    prior_urls = {row["url"] for _, row in prior_sources}
    prior_content_hashes = {
        hashlib.sha256(row["content"].encode()).hexdigest()
        for _, row in prior_sources
        if isinstance(row.get("content"), str) and row["content"]
    }
    assert {source["source_id"] for source in sources}.isdisjoint(prior_source_ids)
    assert {source["url"] for source in sources}.isdisjoint(prior_urls)
    assert {
        source["content_sha256"]
        for source in sources
        if source["content_sha256"] is not None
    }.isdisjoint(prior_content_hashes)

    claim_findings: list[tuple[str, str, tuple[float, float, float]]] = []
    for candidate in candidates:
        for prior_id, prior_claim in prior_claims:
            scores = _similarity(candidate["claim"], prior_claim)
            if _claim_similarity_flag(scores):
                claim_findings.append((candidate["candidate_id"], prior_id, scores))
    assert claim_findings == []

    source_findings: list[tuple[str, str, tuple[float, float, float]]] = []
    for source in sources:
        if not source["available"]:
            continue
        for prior_id, prior_source in prior_sources:
            prior_content = prior_source.get("content")
            if not isinstance(prior_content, str) or not prior_content:
                continue
            scores = _similarity(source["content"], prior_content)
            if _source_similarity_flag(scores):
                source_findings.append((source["source_id"], prior_id, scores))
    assert source_findings == []

    prior_haystacks = [_normalize(claim) for _, claim in prior_claims] + [
        _normalize(f"{row['source_id']} {row['url']}") for _, row in prior_sources
    ]
    entity_collisions = []
    for source in sources:
        entity = _entity_key(source["entity"])
        pattern = re.compile(rf"(?:^| )({re.escape(entity)})(?: |$)")
        if entity and any(pattern.search(text) for text in prior_haystacks):
            entity_collisions.append(source["source_id"])
    assert entity_collisions == []


def test_v3_review_packet_is_complete_and_blind() -> None:
    packet = (V3 / "REVIEW_PACKET.md").read_text(encoding="utf-8")
    guide = (V3 / "ANNOTATION_GUIDE.md").read_text(encoding="utf-8")
    worksheet_ids = re.findall(r"`(v3_holdout_candidate_\d{3})`", packet)
    assert Counter(worksheet_ids) == {
        f"v3_holdout_candidate_{number:03d}": 1 for number in range(1, 73)
    }
    packet_flat = " ".join(packet.split())
    assert "construction stratum mapping" in packet_flat
    assert "no case-level target relation" in packet_flat
    assert "Gold dataset: not created" in packet
    assert "Formal gate execution: not started" in packet
    for relation in (
        "entailed",
        "partially_entailed",
        "contradicted",
        "not_in_source",
        "source_unavailable",
        "not_checkable",
    ):
        assert f"`{relation}`" in guide


def test_consumed_v2_holdout_is_sha_only_and_not_an_audit_input() -> None:
    assert CONSUMED_V2_HOLDOUT not in PRIOR_CASE_PATHS
    assert CONSUMED_V2_HOLDOUT not in PRIOR_SOURCE_PATHS
    assert _sha256(CONSUMED_V2_HOLDOUT) == CONSUMED_V2_SHA256


def test_v3_manifest_hashes_and_formal_gate_preregistration() -> None:
    manifest = json.loads((V3 / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == (
        "blind_candidate_pack_frozen_gold_not_created_gate_not_run"
    )
    assert manifest["candidate_count"] == 72
    assert manifest["source_count"] == 24
    assert manifest["available_source_count"] == 20
    assert manifest["unavailable_source_count"] == 4
    assert manifest["maximum_cases_per_source"] == 3
    assert manifest["construction_stratification"] == {
        "aggregate_counts_only": True,
        "case_level_mapping_persisted": False,
        "entailed": 12,
        "partially_entailed": 12,
        "contradicted": 12,
        "not_in_source": 12,
        "source_unavailable": 12,
        "not_checkable": 12,
    }

    file_paths = {
        "holdout_candidates.jsonl": V3 / "holdout_candidates.jsonl",
        "sources/source_snapshots.jsonl": SOURCES_PATH,
        "REVIEW_PACKET.md": V3 / "REVIEW_PACKET.md",
        "ANNOTATION_GUIDE.md": V3 / "ANNOTATION_GUIDE.md",
        "LEAKAGE_AUDIT.md": V3 / "LEAKAGE_AUDIT.md",
        "tests/test_eval_v3_candidates.py": Path(__file__),
    }
    assert {name: _sha256(path) for name, path in file_paths.items()} == manifest[
        "files"
    ]
    bundle_payload = "".join(
        f"{_sha256(ROOT / relative_path)}  {relative_path}\n"
        for relative_path in manifest["ordered_bundle_paths"]
    )
    assert (
        hashlib.sha256(bundle_payload.encode()).hexdigest()
        == (manifest["ordered_bundle_sha256"])
    )

    gate = manifest["formal_gate_preregistration"]
    assert gate["status"] == "not_run"
    assert gate["execution_limit"] == 1
    assert gate["rerun_allowed"] is False
    assert gate["minimum_final_valid_case_count"] == 60
    assert gate["required_zero_failures"] == {
        "schema": 0,
        "transport": 0,
        "scope": 0,
        "guard": 0,
        "local_validation": 0,
    }
    assert gate["retrieval_judge_fixed_six_macro_f1_minimum"] == 0.70
    assert gate["v0_1_target_macro_f1_minimum"] == 0.75
    assert gate["contradiction_recall_minimum"] == 0.80
    assert gate["single_agent_delta_must_be_reported"] is True
    assert gate["pair_benchmark_complete_multi_agent_advantage_claim_allowed"] is False
    assert gate["automatic_retry_count"] == 0
    assert gate["automatic_repair"] is False

    assert manifest["v3_gold_dataset_created"] is False
    assert manifest["v3_formal_gate_executed"] is False
    assert manifest["phase3_gate_passed"] is False
    assert manifest["phase4_blocked"] is True
