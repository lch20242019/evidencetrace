from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

from evidencetrace.checks.deterministic import is_high_confidence_subjective
from evidencetrace.eval.baselines import (
    build_evidence_chunks,
    estimate_full_context_size,
)
from evidencetrace.eval.models import EvalCase, SourceFixture
from evidencetrace.eval.router import AdaptiveRouterConfig, select_adaptive_route
from evidencetrace.models import Relation

ROOT = Path(__file__).parents[1]
PACK = ROOT / "eval_sets" / "v4_zh"
SOURCES = PACK / "sources" / "source_snapshots_zh.jsonl"
CANDIDATES = PACK / "holdout_candidates_zh.jsonl"
CONSUMED_BYTE_ONLY_HASHES = {
    "eval_sets/v2/holdout_single_human.jsonl": (
        "0d19004b627f4ecabf9dd40fd28303b0539b89f36f3f7f5cec135c47a6582c75"
    ),
    "eval_sets/v3/holdout_candidates.jsonl": (
        "e159de6220254c8e56d2903f86192016b69d4a344b728dcc579f2e98f3c926ef"
    ),
    "eval_sets/v3/sources/source_snapshots.jsonl": (
        "bea8239cd83ebf9b9940ba1815fdfe7f44054df814f705e9edd9ad5cc42fbbce"
    ),
    "eval_sets/v3_zh/holdout_single_human_zh.jsonl": (
        "34518f925e82bd3a1d716bc42ab07ab6b6b3b4333408fab45bfb573d38e67302"
    ),
    "eval_sets/v3_zh/manifest.json": (
        "eb540f6e8e61483fd3e725b2d77817c6bebeacd4babe0d81a8b851a3939f19d5"
    ),
}
FORBIDDEN_CANDIDATE_KEYS = {
    "relation",
    "gold_relation",
    "gold_evidence",
    "gold_evidence_span",
    "intended_relation",
    "target",
    "proposal",
    "model_proposal",
    "expected_route",
    "notes",
}
CLAIM_THRESHOLDS = {
    "sequence": 0.82,
    "bigram": 0.68,
    "trigram": 0.58,
}
SOURCE_THRESHOLDS = {
    "sequence": 0.78,
    "bigram": 0.60,
    "trigram": 0.50,
}
STRUCTURED_RE = re.compile(
    r"(?:"
    r"\b20\d{2}(?:[-/]\d{1,2}(?:[-/]\d{1,2})?)?\b|"
    r"\bv?\d+(?:\.\d+){1,3}\b|"
    r"\b\d+(?:\.\d+)?(?:%|gb|gib|mb|ms|秒|分钟|小时|天|个月|年|倍|位|个)\b"
    r")",
    re.I,
)
NUMERIC_SLOT_RE = re.compile(r"\d+(?:\.\d+)*")


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bundle_hash(paths: list[str]) -> str:
    payload = "".join(f"{_sha256(ROOT / path)}  {path}\n" for path in paths)
    return hashlib.sha256(payload.encode()).hexdigest()


def _normalize(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return "".join(char for char in value if char.isalnum())


def _ngrams(value: str, size: int) -> set[str]:
    value = _normalize(value)
    return {
        value[index : index + size] for index in range(max(len(value) - size + 1, 0))
    }


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _similarity(left: str, right: str) -> dict[str, float]:
    return {
        "sequence": SequenceMatcher(
            None, _normalize(left), _normalize(right), autojunk=False
        ).ratio(),
        "bigram": _jaccard(_ngrams(left, 2), _ngrams(right, 2)),
        "trigram": _jaccard(_ngrams(left, 3), _ngrams(right, 3)),
    }


def _is_flagged(scores: dict[str, float], thresholds: dict[str, float]) -> bool:
    return any(scores[key] >= value for key, value in thresholds.items())


def _structured_fingerprint(value: str) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                unicodedata.normalize("NFKC", match.group(0)).casefold()
                for match in STRUCTURED_RE.finditer(value)
            }
        )
    )


def _one_numeric_slot_differs(left: str, right: str) -> bool:
    left_tokens = list(NUMERIC_SLOT_RE.finditer(left))
    right_tokens = list(NUMERIC_SLOT_RE.finditer(right))
    if len(left_tokens) != len(right_tokens):
        return False
    left_values = [match.group(0) for match in left_tokens]
    right_values = [match.group(0) for match in right_tokens]
    if sum(a != b for a, b in zip(left_values, right_values, strict=True)) != 1:
        return False
    left_skeleton = _normalize(NUMERIC_SLOT_RE.sub("SLOT", left))
    right_skeleton = _normalize(NUMERIC_SLOT_RE.sub("SLOT", right))
    return left_skeleton in right_skeleton or right_skeleton in left_skeleton


def _source_fixture(row: dict[str, object]) -> SourceFixture:
    content = str(row["content"])
    return SourceFixture(
        source_id=str(row["source_id"]),
        url=str(row["url"]),
        content=content,
        provenance=json.dumps(row["provenance"], ensure_ascii=False),
        content_hash=hashlib.sha256(content.encode()).hexdigest(),
        available=bool(row["available"]),
    )


def _routing_case(candidate: dict[str, object], source: SourceFixture) -> EvalCase:
    return EvalCase(
        case_id=str(candidate["candidate_id"]),
        claim_text=str(candidate["claim"]),
        source_fixture="sources/source_snapshots_zh.jsonl",
        source_id=source.source_id,
        source_url=source.url,
        gold_relation=Relation.NOT_IN_SOURCE,
        claim_type="blind_candidate",
        mutation_type="not_recorded",
        split="test",
        provenance="Phase 3H route validation only.",
        annotation_status="provisional",
        annotation_notes="No candidate gold relation exists.",
    )


def test_v4_zh_candidate_and_source_structure_is_blind() -> None:
    candidates = _jsonl(CANDIDATES)
    sources = _jsonl(SOURCES)
    assert len(candidates) == 72
    assert len(sources) == 24
    assert len({row["candidate_id"] for row in candidates}) == 72
    assert len({row["source_id"] for row in sources}) == 24
    assert Counter(row["source_id"] for row in candidates) == {
        row["source_id"]: 3 for row in sources
    }
    assert all(not (FORBIDDEN_CANDIDATE_KEYS & row.keys()) for row in candidates)
    assert all(
        row["annotation_status"] == "unreviewed_blind_candidate" for row in candidates
    )
    assert all(row["evaluation_status"] == "not_run" for row in candidates)
    assert all(
        row["blind_status"] == "case_level_relation_not_recorded" for row in candidates
    )
    serialized = CANDIDATES.read_text()
    for label in (
        '"entailed"',
        '"partially_entailed"',
        '"contradicted"',
        '"not_in_source"',
        '"source_unavailable"',
        '"not_checkable"',
    ):
        assert label not in serialized
    packet = (PACK / "REVIEW_PACKET.md").read_text()
    assert packet.count("- Relation:") == 72
    assert all(
        line == "- Relation:"
        for line in packet.splitlines()
        if line.startswith("- Relation:")
    )


def test_v4_zh_source_provenance_hashes_and_strata() -> None:
    sources = _jsonl(SOURCES)
    assert Counter(row["source_length_stratum"] for row in sources) == {
        "short": 7,
        "medium": 7,
        "long": 6,
        "unavailable": 4,
    }
    available = [row for row in sources if row["available"]]
    unavailable = [row for row in sources if not row["available"]]
    assert len(available) == 20
    assert len(unavailable) == 4
    assert len({row["url"] for row in sources}) == 24
    assert len({row["entity"] for row in sources}) == 24
    assert len({row["content_sha256"] for row in available}) == 20
    for row in available:
        assert row["original_language"] == "zh-CN"
        assert row["http_status"] == 200
        assert row["retrieval_status"] == "safe_fetch_http_200"
        assert (
            row["content_sha256"]
            == hashlib.sha256(str(row["content"]).encode()).hexdigest()
        )
        provenance = row["provenance"]
        assert re.fullmatch(r"[0-9a-f]{64}", provenance["upstream_body_sha256"])
        assert provenance["safe_fetch_retries"] == 0
        assert "no translation" in provenance["excerpt_policy"]
        chunks = build_evidence_chunks(_source_fixture(row))
        assert len(chunks) == row["expected_chunk_count"]
    for row in unavailable:
        assert row["original_language"] == "zh-CN"
        assert row["http_status"] == 404
        assert row["retrieval_status"] == "frozen_safe_fetch_http_404"
        assert row["content"] == ""
        assert row["content_sha256"] is None
        assert row["expected_chunk_count"] == 0
        assert row["provenance"]["safe_fetch_error_code"] == "http_404"
        assert "No source content was inferred" in row["provenance"]["excerpt_policy"]


def test_v4_zh_chunk_and_adaptive_route_distribution() -> None:
    source_rows = {str(row["source_id"]): row for row in _jsonl(SOURCES)}
    route_counts: Counter[str] = Counter()
    config = AdaptiveRouterConfig()
    for candidate in _jsonl(CANDIDATES):
        source = _source_fixture(source_rows[str(candidate["source_id"])])
        case = _routing_case(candidate, source)
        chunks = build_evidence_chunks(source)
        decision = select_adaptive_route(
            source_available=source.available,
            purely_subjective=is_high_confidence_subjective(str(candidate["claim"])),
            chunk_count=len(chunks),
            estimated_context_size=estimate_full_context_size(case, source),
            config=config,
        )
        route_counts[decision.selected_route] += 1
    assert route_counts == {
        "full_context_single_agent": 17,
        "retrieval_judge": 31,
        "deterministic_not_checkable": 12,
        "deterministic_source_unavailable": 12,
    }
    chunk_counts = Counter()
    for row in source_rows.values():
        chunk_counts[row["source_length_stratum"]] += len(
            build_evidence_chunks(_source_fixture(row))
        )
    assert chunk_counts == {
        "short": 7,
        "medium": 28,
        "long": 48,
        "unavailable": 0,
    }


def test_v4_zh_long_sources_have_genuine_technical_distractors() -> None:
    markers = {
        "v4zh_src_015": {"事务", "索引", "函数", "Catalog", "权限"},
        "v4zh_src_016": {"索引", "分词器", "Iceberg", "Paimon", "分区"},
        "v4zh_src_017": {"发布", "Modal", "Safari", "InputNumber", "内存"},
        "v4zh_src_018": {"API", "应用框架", "窗口", "音频", "视频"},
        "v4zh_src_019": {"应用", "键盘", "Video", "窗口", "运行时"},
        "v4zh_src_020": {"后台服务", "ArkWeb", "NativeFence", "FastBuffer", "升级"},
    }
    sources = {str(row["source_id"]): row for row in _jsonl(SOURCES)}
    assert set(markers) == {
        source_id
        for source_id, row in sources.items()
        if row["source_length_stratum"] == "long"
    }
    for source_id, expected_markers in markers.items():
        content = str(sources[source_id]["content"])
        chunks = build_evidence_chunks(_source_fixture(sources[source_id]))
        assert 7 <= len(chunks) <= 10
        assert all(marker in content for marker in expected_markers)
        assert len({chunk.text for chunk in chunks}) == len(chunks)


def test_v4_zh_reviewer_packet_contains_only_blind_fields() -> None:
    candidates = _jsonl(CANDIDATES)
    sources = {str(row["source_id"]): row for row in _jsonl(SOURCES)}
    packet = (PACK / "REVIEW_PACKET.md").read_text()
    allowed_prefixes = (
        "# EvidenceTrace v4_zh Blind Reviewer Packet",
        "## v4_zh_candidate_",
        "- Source ID:",
        "- Canonical source:",
        "- Claim:",
        "- Relation:",
        "- Evidence:",
        "- Notes:",
    )
    assert all(
        not line or line.startswith(allowed_prefixes) for line in packet.splitlines()
    )
    for candidate in candidates:
        assert packet.count(f"## {candidate['candidate_id']}") == 1
        assert f"- Source ID: `{candidate['source_id']}`" in packet
        source = sources[str(candidate["source_id"])]
        canonical = str(source["content"]) if source["available"] else "[不可用]"
        assert f"- Canonical source: {canonical}" in packet
        assert f"- Claim: {candidate['claim']}" in packet
    lowered = packet.casefold()
    for forbidden in (
        "intended relation",
        "gold relation",
        "gold evidence",
        "model proposal",
        "expected route",
        "construction stratum",
    ):
        assert forbidden not in lowered


def test_v4_zh_exact_and_similarity_leakage_boundaries() -> None:
    candidates = _jsonl(CANDIDATES)
    sources = _jsonl(SOURCES)
    core = _jsonl(ROOT / "eval_sets" / "core.jsonl")
    seed_sources = _jsonl(ROOT / "eval_sets" / "sources" / "seed_sources.jsonl")
    phase3g = _jsonl(ROOT / "eval_sets" / "phase3g_dev_zh" / "dev.jsonl")
    phase3g_sources = _jsonl(
        ROOT / "eval_sets" / "phase3g_dev_zh" / "sources" / "source_snapshots.jsonl"
    )
    new_claims = [
        (str(row["candidate_id"]), str(row["source_id"]), str(row["claim"]))
        for row in candidates
    ]
    prior_claims = [
        (f"core:{row['case_id']}", str(row["claim_text"])) for row in core
    ] + [(f"phase3g:{row['case_id']}", str(row["claim_text"])) for row in phase3g]
    new_sources = [
        (str(row["source_id"]), str(row["content"]))
        for row in sources
        if row["available"]
    ]
    prior_sources = [
        (f"seed:{row['source_id']}", str(row["content"])) for row in seed_sources
    ] + [
        (f"phase3g:{row['source_id']}", str(row["content"])) for row in phase3g_sources
    ]

    assert not (
        {row["candidate_id"] for row in candidates}
        & {row["case_id"] for row in [*core, *phase3g]}
    )
    assert not (
        {row["source_id"] for row in sources}
        & {row["source_id"] for row in [*seed_sources, *phase3g_sources]}
    )
    assert not (
        {row["url"] for row in sources}
        & {row["url"] for row in [*seed_sources, *phase3g_sources]}
    )
    assert not (
        {row["content_sha256"] for row in sources if row["content_sha256"]}
        & {row["content_hash"] for row in [*seed_sources, *phase3g_sources]}
    )
    assert not (
        {_normalize(text) for _, _, text in new_claims}
        & {_normalize(text) for _, text in prior_claims}
    )
    assert not (
        {
            _structured_fingerprint(text)
            for _, _, text in new_claims
            if _structured_fingerprint(text)
        }
        & {
            _structured_fingerprint(text)
            for _, text in prior_claims
            if _structured_fingerprint(text)
        }
    )

    for _, _, new_text in new_claims:
        assert all(
            not _is_flagged(_similarity(new_text, prior_text), CLAIM_THRESHOLDS)
            for _, prior_text in prior_claims
        )
    for _, new_text in new_sources:
        assert all(
            not _is_flagged(_similarity(new_text, prior_text), SOURCE_THRESHOLDS)
            for _, prior_text in prior_sources
        )

    internal_flags = []
    for index, (left_id, left_source, left_text) in enumerate(new_claims):
        for right_id, right_source, right_text in new_claims[index + 1 :]:
            if _is_flagged(_similarity(left_text, right_text), CLAIM_THRESHOLDS):
                internal_flags.append((left_id, right_id))
                assert left_source == right_source
                assert _one_numeric_slot_differs(left_text, right_text)
    assert len(internal_flags) == 1


def test_v4_zh_manifest_hashes_preregistration_and_consumed_boundary() -> None:
    manifest = json.loads((PACK / "manifest.json").read_text())
    assert manifest["status"] == "frozen_blind_candidates_awaiting_human_annotation"
    assert manifest["case_count"] == 72
    assert manifest["source_count"] == 24
    assert manifest["cases_per_source"] == 3
    assert manifest["case_level_construction_strata_recorded"] is False
    assert manifest["gold_dataset_status"] == "not_created"
    assert manifest["holdout_execution_status"] == "not_run"
    assert manifest["evaluation_model_exposure"] is False
    assert manifest["model_calls"] == 0
    assert manifest["production_code_modified"] is False
    assert manifest["phase4_status"] == "not_implemented"
    assert manifest["expected_routes_aggregate_only"] == {
        "full_context_single_agent": 17,
        "retrieval_judge": 31,
        "deterministic_not_checkable": 12,
        "deterministic_source_unavailable": 12,
    }
    prereg = manifest["formal_gate_preregistration"]
    assert prereg["minimum_valid_cases"] == 60
    assert prereg["target_valid_cases"] == 72
    assert prereg["all_final_valid_cases_must_complete"] is True
    assert prereg["final_operational_failures"] == 0
    assert prereg["unrecovered_schema_failures"] == 0
    assert prereg["first_attempt_contract_success_rate_minimum"] == 0.99
    assert prereg["recovered_schema_failures_maximum"] == 2
    assert prereg["adaptive_macro_f1_minimum"] == 0.75
    assert prereg["contradiction_recall_minimum"] == 0.80
    assert prereg["not_checkable_recall_minimum"] == 0.80
    assert prereg["adaptive_macro_f1_vs_single_agent_minimum_delta"] == -0.02
    assert prereg["formal_gate_execution_limit"] == 1
    for path, expected in manifest["files"].items():
        assert _sha256(ROOT / path) == expected
    assert (
        _bundle_hash(manifest["ordered_bundle_paths"])
        == manifest["ordered_bundle_sha256"]
    )
    for path, expected in CONSUMED_BYTE_ONLY_HASHES.items():
        assert _sha256(ROOT / path) == expected
        assert manifest["consumed_byte_hashes"][path] == expected


def test_v4_zh_freeze_boundary_and_leakage_report_are_explicit() -> None:
    manifest = json.loads((PACK / "manifest.json").read_text())
    assert manifest["freeze_boundary"] == {
        "code_sha256": (
            "88ace6e1b9675febb6071101aa04d5ee274b9c6986bdc6652156225d52c3d641"
        ),
        "prompt_schema_sha256": (
            "eb83e292e51dcdcf7b39679a6998cdfb4b727eb286b7768d36f3b0747438fff5"
        ),
        "config_recovery_ownership_sha256": (
            "d8a3ce764614e31c6151ad0a2859e143206a489af9cc55aac581ab4240929ed2"
        ),
        "ownership_policy_sha256": (
            "8432ead334bc0761b4bf78f1bccc19a81b1083741751275418245bff77b9cd8c"
        ),
    }
    audit = (PACK / "LEAKAGE_AUDIT.md").read_text()
    assert "passed with one reviewed internal single-slot pair" in audit
    assert "Prior ID collisions: `0`" in audit
    assert "Prior URL collisions: `0`" in audit
    assert "Prior entity occurrences: `0`" in audit
    assert "Prior normalized claim collisions: `0`" in audit
    assert "CJK bigram Jaccard: `0.230769`" in audit
    assert "v4_zh_candidate_015" in audit
    assert "v4_zh_candidate_047" in audit
    assert "No consumed gold record was loaded or parsed." in audit
