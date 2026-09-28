#!/usr/bin/env python3
"""Freeze the internal Chinese canonical v3 single-human benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REVIEW_ROOT = (
    Path.home() / "Projects" / "evidencetrace_review_records" / "v3_zh_experimental"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "eval_sets" / "v3_zh"
FREEZE_BASE_COMMIT = "0a8e5a9f0005f9ff64c44db9208e0a403930574a"
EXPECTED_REVIEWER_SHA256 = (
    "538e738fca9036df65091a985a7b9d590fb062f69f813fc3cb5a5890b245b4ac"
)
EXPECTED_CANDIDATE_SHA256 = (
    "ce5863c7120fbdae6653a09b21230c0caf3ac4d0cc72d7ec168413f0606ebaa4"
)
EXPECTED_SOURCE_SHA256 = (
    "6a3efe0ac3d9c8f0b5eec81997df2103cec8e98115cd827a98e91a0fb8edb1ae"
)
EXPECTED_CORRECTION_SHA256 = (
    "c86695a89139477b63e69d88826c5c68b0891deef2b0195aa54799d57d5d17a9"
)
EXPECTED_PRIOR_BLOCKED_SHA256 = (
    "91bfa4e4a6de294eb47ffa88b7eced1217bbce50007513dea2e02301203af272"
)
EXPECTED_CONSUMED_V2_SHA256 = (
    "0d19004b627f4ecabf9dd40fd28303b0539b89f36f3f7f5cec135c47a6582c75"
)
EXPECTED_PARENT_BUNDLE_SHA256 = (
    "21f0d790007f492b9fafdb022894981f1fc789e86072c55a899634ba93f2b8ef"
)
PARENT_FILES = {
    "eval_sets/v3/holdout_candidates.jsonl": (
        "e159de6220254c8e56d2903f86192016b69d4a344b728dcc579f2e98f3c926ef"
    ),
    "eval_sets/v3/sources/source_snapshots.jsonl": (
        "bea8239cd83ebf9b9940ba1815fdfe7f44054df814f705e9edd9ad5cc42fbbce"
    ),
    "eval_sets/v3/REVIEW_PACKET.md": (
        "8901bec92ebf04cf0883a88479d378add36b789cd1d95ccf3380ee49674cecf3"
    ),
    "eval_sets/v3/ANNOTATION_GUIDE.md": (
        "af05b224aba2bad3c4aa2f01273f0bdf0fec351259df95cacdc840bc090f83f2"
    ),
    "eval_sets/v3/LEAKAGE_AUDIT.md": (
        "6e3f0dd3901b80b36552ef7b49eb5caf9548988e6a82666f18a7d869ccd0a25a"
    ),
    "tests/test_eval_v3_candidates.py": (
        "c1d83d6314a2e19bd32841abb93e2237b5d0b8ae1bf74b5e2f898854d2ce5ec0"
    ),
}
HISTORICAL_RECORDS = {
    "phase3f_b_zh_corrections_required.md": (
        "8079a031e302fdefc24f0d4350804a9f4536f551b41ec7a987cbf1b4fcc53398"
    ),
    "phase3f_b_zh_validation_blocked_manifest.json": (
        "0608cbb07c637b22eca0e27873f9d9b469bfb90cc231cff0e71c803f9b25272e"
    ),
    "phase3f_b_zh_canonical_corrections_required.md": (
        "c86695a89139477b63e69d88826c5c68b0891deef2b0195aa54799d57d5d17a9"
    ),
    "phase3f_b_zh_canonical_validation_blocked_manifest.json": (
        "91bfa4e4a6de294eb47ffa88b7eced1217bbce50007513dea2e02301203af272"
    ),
}
RELATIONS = {
    "entailed",
    "partially_entailed",
    "contradicted",
    "not_in_source",
    "source_unavailable",
    "not_checkable",
}
ZH_RELATION_MAP = {
    "蕴含": "entailed",
    "部分蕴含": "partially_entailed",
    "矛盾": "contradicted",
    "原文未提及": "not_in_source",
    "来源不可用": "source_unavailable",
    "不可验证": "not_checkable",
}
SUBSTANTIVE_RELATIONS = {"entailed", "partially_entailed", "contradicted"}
FORBIDDEN_CANDIDATE_FIELDS = {
    "answer",
    "answer_note",
    "construction_stratum",
    "gold_evidence",
    "gold_relation",
    "model_output",
    "model_proposal",
    "proposal",
    "proposed_relation",
    "target_relation",
}
FORBIDDEN_SOURCE_FIELDS = {
    "answer",
    "answer_note",
    "candidate_target",
    "gold_relation",
    "model_output",
    "proposal",
    "target_relation",
}


class FreezeError(ValueError):
    """A Chinese canonical freeze input failed deterministic validation."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def require_hash(path: Path, expected: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise FreezeError(f"SHA-256 mismatch for {path}: {actual} != {expected}")


def verify_parent_bundle() -> dict[str, str]:
    observed: dict[str, str] = {}
    payload = ""
    for relative, expected in PARENT_FILES.items():
        path = REPO_ROOT / relative
        require_hash(path, expected)
        observed[relative] = expected
        payload += f"{expected}  {relative}\n"
    actual_bundle = sha256_text(payload)
    if actual_bundle != EXPECTED_PARENT_BUNDLE_SHA256:
        raise FreezeError(f"English parent bundle mismatch: {actual_bundle}")
    return observed


def parse_annotations(path: Path) -> dict[str, dict[str, str | None]]:
    annotations: dict[str, dict[str, str | None]] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.startswith("| `v3_holdout_candidate_"):
            continue
        parts = [part.strip() for part in line.split("|")]
        if len(parts) != 7:
            raise FreezeError(
                f"annotation row {line_number} must contain five data fields"
            )
        case_id = parts[1].strip("`")
        raw_relation, evidence, notes, reviewer = parts[2:6]
        relation = (
            raw_relation
            if raw_relation in RELATIONS
            else ZH_RELATION_MAP.get(raw_relation)
        )
        if relation is None:
            raise FreezeError(f"invalid relation for {case_id}: {raw_relation}")
        if case_id in annotations:
            raise FreezeError(f"duplicate annotation case: {case_id}")
        if not notes:
            raise FreezeError(f"blank reviewer rationale for {case_id}")
        if not reviewer:
            raise FreezeError(f"blank reviewer pseudonym for {case_id}")
        if relation in SUBSTANTIVE_RELATIONS and not evidence:
            raise FreezeError(f"missing substantive evidence for {case_id}")
        if relation not in SUBSTANTIVE_RELATIONS and evidence:
            raise FreezeError(f"evidence must be empty for {case_id}")
        annotations[case_id] = {
            "relation": relation,
            "evidence": evidence or None,
        }
    expected = {f"v3_holdout_candidate_{number:03d}" for number in range(1, 73)}
    if set(annotations) != expected:
        raise FreezeError("reviewer record must contain the exact 72 candidate IDs")
    return annotations


def validate_inputs(
    candidates: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    parent_candidates: list[dict[str, Any]],
    parent_sources: list[dict[str, Any]],
    annotations: dict[str, dict[str, str | None]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    if len(candidates) != 72 or len(sources) != 24:
        raise FreezeError("Chinese pack must contain 72 candidates and 24 sources")
    candidate_by_id = {item["candidate_id"]: item for item in candidates}
    source_by_id = {item["source_id"]: item for item in sources}
    parent_candidate_by_id = {item["candidate_id"]: item for item in parent_candidates}
    parent_source_by_id = {item["source_id"]: item for item in parent_sources}
    if len(candidate_by_id) != 72 or len(source_by_id) != 24:
        raise FreezeError("Chinese candidate/source IDs must be unique")
    if set(candidate_by_id) != set(parent_candidate_by_id):
        raise FreezeError("Chinese candidate IDs do not match English parent")
    if set(source_by_id) != set(parent_source_by_id):
        raise FreezeError("Chinese source IDs do not match English parent")
    if set(annotations) != set(candidate_by_id):
        raise FreezeError("annotation and candidate IDs do not match")
    counts = Counter(item["source_id"] for item in candidates)
    if set(counts.values()) != {3} or set(counts) != set(source_by_id):
        raise FreezeError("each Chinese source must map to exactly three candidates")

    for candidate_id, candidate in candidate_by_id.items():
        parent = parent_candidate_by_id[candidate_id]
        if candidate["source_id"] != parent["source_id"]:
            raise FreezeError(f"parent source mapping mismatch: {candidate_id}")
        forbidden = set(candidate) & FORBIDDEN_CANDIDATE_FIELDS
        if forbidden:
            raise FreezeError(f"forbidden candidate fields: {candidate_id} {forbidden}")
        if not candidate["claim"].strip():
            raise FreezeError(f"blank Chinese claim: {candidate_id}")

    for source_id, source in source_by_id.items():
        parent = parent_source_by_id[source_id]
        if source["url"] != parent["url"]:
            raise FreezeError(f"parent URL mismatch: {source_id}")
        if source["available"] != parent["available"]:
            raise FreezeError(f"parent availability mismatch: {source_id}")
        forbidden = set(source) & FORBIDDEN_SOURCE_FIELDS
        if forbidden:
            raise FreezeError(f"forbidden source fields: {source_id} {forbidden}")
        content = source.get("content")
        if not isinstance(content, str):
            raise FreezeError(f"Chinese source content must be text: {source_id}")
        if source["available"] and not content:
            raise FreezeError(f"available Chinese source is blank: {source_id}")
        if not source["available"] and content:
            raise FreezeError(f"unavailable Chinese source has content: {source_id}")

    for case_id, annotation in annotations.items():
        candidate = candidate_by_id[case_id]
        source = source_by_id[candidate["source_id"]]
        relation = str(annotation["relation"])
        evidence = annotation["evidence"]
        if evidence is not None and str(evidence) not in source["content"]:
            raise FreezeError(f"evidence is not a Chinese source substring: {case_id}")
        if relation == "source_unavailable" and source["available"]:
            raise FreezeError(
                f"source_unavailable points to available source: {case_id}"
            )
        if not source["available"] and relation != "source_unavailable":
            raise FreezeError(f"unavailable source has wrong relation: {case_id}")

    return candidate_by_id, source_by_id


def build_source_fixtures(
    source_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    fixtures: list[dict[str, Any]] = []
    for source_id in sorted(source_by_id):
        source = source_by_id[source_id]
        content = source["content"]
        fixtures.append(
            {
                "source_id": source_id,
                "url": source["url"],
                "content": content,
                "provenance": (
                    "Internal Chinese canonical machine-translated derivative; "
                    f"English parent source_id={source_id}; "
                    f"translation_origin={source['translation_origin']}."
                ),
                "content_hash": sha256_text(content),
                "available": source["available"],
            }
        )
    return fixtures


def canonical_case_hash(case: dict[str, Any]) -> str:
    payload = dict(case)
    payload.pop("case_hash", None)
    return sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def build_gold_cases(
    candidate_by_id: dict[str, dict[str, Any]],
    source_fixtures: list[dict[str, Any]],
    annotations: dict[str, dict[str, str | None]],
    *,
    reviewer_sha256: str,
    generation_code_sha256: str,
) -> list[dict[str, Any]]:
    fixture_by_id = {item["source_id"]: item for item in source_fixtures}
    cases: list[dict[str, Any]] = []
    for case_id in sorted(candidate_by_id):
        candidate = candidate_by_id[case_id]
        source = fixture_by_id[candidate["source_id"]]
        annotation = annotations[case_id]
        case: dict[str, Any] = {
            "case_id": case_id,
            "document_path": None,
            "claim_text": candidate["claim"],
            "claim_line": 1,
            "source_fixture": "sources/source_snapshots_zh.jsonl",
            "source_id": source["source_id"],
            "source_url": source["url"],
            "gold_relation": annotation["relation"],
            "gold_evidence_span": annotation["evidence"],
            "claim_type": "citation_pair",
            "mutation_type": "none",
            "split": "test",
            "provenance": (
                "Single human review of an internal Chinese canonical holdout."
            ),
            "annotation_status": "single_human_review",
            "annotation_notes": (
                "Frozen single-human annotation; private identity and notes excluded."
            ),
            "source_sha256": source["content_hash"],
            "reviewer_record_sha256": reviewer_sha256,
            "generation_code_sha256": generation_code_sha256,
        }
        case["case_hash"] = canonical_case_hash(case)
        cases.append(case)
    return cases


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def provenance_markdown(
    *,
    created_at: str,
    reviewer_sha256: str,
    parent_bundle_sha256: str,
) -> str:
    return f"""# EvidenceTrace v3_zh provenance

## Benchmark identity

- Scope: internal Chinese canonical benchmark
- Annotation language: `zh-CN`
- Canonical source language: `zh-CN`
- Source origin: `machine_translated_derivative`
- Translation model: `OpenAI Codex`
- Translation human fidelity review required: `false`
- Fidelity to English claimed: `false`
- English parent bundle SHA-256: `{parent_bundle_sha256}`
- Freeze created UTC: `{created_at}`

The Chinese source and claim text are canonical for this benchmark. The English
v3 pack supplies lineage, IDs, URLs, and availability provenance only. Chinese
evidence is validated solely against the corresponding Chinese canonical source.

## Blindness and eligibility

- Evaluation model: `deepseek-v4-flash`
- Evaluation model exposure before freeze: `false`
- Developer-agent exposure after code freeze: `true`
- Holdout blind to evaluation model: `true`
- Formal internal gate eligible: `true`
- Public benchmark eligible: `false`

The human maintainer explicitly confirmed that `deepseek-v4-flash` did not
participate in translation, annotation, review, or evaluation and did not read
this holdout before freeze. No model call or holdout execution occurred during
this freeze.

## Human annotation

- Human labeled: `true`
- Annotation status: `single_human_review_complete`
- Counts as human review: `true`
- Second human review: `false`
- Reviewer record SHA-256: `{reviewer_sha256}`

Reviewer identity and full notes remain in the private external record and are
not included in the public gold rows.

## Translation disclaimer

The benchmark does not claim that the Chinese text is an official translation
or a sentence-by-sentence faithful translation of the English parent. It
measures EvidenceTrace behavior on the frozen Chinese canonical text as written.
"""


def ordered_bundle_hash(paths: list[str]) -> str:
    payload = "".join(
        f"{sha256_file(REPO_ROOT / relative)}  {relative}\n" for relative in paths
    )
    return sha256_text(payload)


def freeze(
    *,
    reviewer_path: Path,
    candidates_path: Path,
    sources_path: Path,
    correction_path: Path,
    prior_blocked_path: Path,
    output_dir: Path,
    created_at: str,
) -> dict[str, Any]:
    require_hash(reviewer_path, EXPECTED_REVIEWER_SHA256)
    require_hash(candidates_path, EXPECTED_CANDIDATE_SHA256)
    require_hash(sources_path, EXPECTED_SOURCE_SHA256)
    require_hash(correction_path, EXPECTED_CORRECTION_SHA256)
    require_hash(prior_blocked_path, EXPECTED_PRIOR_BLOCKED_SHA256)
    for name, expected in HISTORICAL_RECORDS.items():
        require_hash(DEFAULT_REVIEW_ROOT / name, expected)
    require_hash(
        REPO_ROOT / "eval_sets/v2/holdout_single_human.jsonl",
        EXPECTED_CONSUMED_V2_SHA256,
    )
    parent_hashes = verify_parent_bundle()

    candidates = load_jsonl(candidates_path)
    sources = load_jsonl(sources_path)
    annotations = parse_annotations(reviewer_path)
    parent_candidates = load_jsonl(REPO_ROOT / "eval_sets/v3/holdout_candidates.jsonl")
    parent_sources = load_jsonl(
        REPO_ROOT / "eval_sets/v3/sources/source_snapshots.jsonl"
    )
    candidate_by_id, source_by_id = validate_inputs(
        candidates,
        sources,
        parent_candidates,
        parent_sources,
        annotations,
    )
    generation_code_sha256 = sha256_file(Path(__file__).resolve())
    source_fixtures = build_source_fixtures(source_by_id)
    gold_cases = build_gold_cases(
        candidate_by_id,
        source_fixtures,
        annotations,
        reviewer_sha256=EXPECTED_REVIEWER_SHA256,
        generation_code_sha256=generation_code_sha256,
    )
    if len(gold_cases) != 72:
        raise FreezeError("complete Chinese canonical freeze requires all 72 cases")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_candidates = output_dir / "holdout_candidates_zh.jsonl"
    output_sources = output_dir / "sources/source_snapshots_zh.jsonl"
    output_gold = output_dir / "holdout_single_human_zh.jsonl"
    output_frozen = output_dir / "holdout_single_human_zh.frozen_hashes.json"
    output_provenance = output_dir / "PROVENANCE.md"
    output_manifest = output_dir / "manifest.json"

    output_candidates.write_bytes(candidates_path.read_bytes())
    write_jsonl(output_sources, source_fixtures)
    write_jsonl(output_gold, gold_cases)
    output_frozen.write_text(
        json.dumps(
            {case["case_id"]: case["case_hash"] for case in gold_cases},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    output_provenance.write_text(
        provenance_markdown(
            created_at=created_at,
            reviewer_sha256=EXPECTED_REVIEWER_SHA256,
            parent_bundle_sha256=EXPECTED_PARENT_BUNDLE_SHA256,
        ),
        encoding="utf-8",
    )

    relative_files = [
        "eval_sets/v3_zh/holdout_candidates_zh.jsonl",
        "eval_sets/v3_zh/sources/source_snapshots_zh.jsonl",
        "eval_sets/v3_zh/holdout_single_human_zh.jsonl",
        "eval_sets/v3_zh/holdout_single_human_zh.frozen_hashes.json",
        "eval_sets/v3_zh/PROVENANCE.md",
        "scripts/freeze_v3_zh_single_human.py",
        "tests/test_eval_v3_zh_single_human.py",
    ]
    distribution = Counter(
        str(annotation["relation"]) for annotation in annotations.values()
    )
    manifest: dict[str, Any] = {
        "schema_version": "phase3f-b-zh-canonical-freeze-1",
        "pack_id": "phase3f-v3-zh-single-human-gold",
        "status": "single_human_review_complete_internal_gate_not_run",
        "created_at_utc": created_at,
        "freeze_base_commit": FREEZE_BASE_COMMIT,
        "checkpoint_name": "phase3f-v3-zh-single-human-gold",
        "case_count": 72,
        "valid_case_count": 72,
        "blocked_case_count": 0,
        "source_count": 24,
        "cases_per_source": 3,
        "relation_distribution": {
            relation: distribution[relation] for relation in sorted(RELATIONS)
        },
        "benchmark_provenance": {
            "annotation_language": "zh-CN",
            "canonical_source_language": "zh-CN",
            "source_origin": "machine_translated_derivative",
            "translation_model": "OpenAI Codex",
            "translation_human_fidelity_review_required": False,
            "translation_fidelity_to_english_claimed": False,
            "evaluation_model": "deepseek-v4-flash",
            "evaluation_model_exposure": False,
            "human_labeled": True,
            "second_human_review": False,
            "public_benchmark_eligible": False,
            "benchmark_scope": "internal_chinese_canonical",
            "developer_agent_exposure_after_code_freeze": True,
        },
        "benchmark_status": {
            "annotation_status": "single_human_review_complete",
            "counts_as_human_review": True,
            "holdout_blind_to_evaluation_model": True,
            "formal_internal_gate_eligible": True,
            "public_benchmark_eligible": False,
            "holdout_execution_status": "not_run",
        },
        "input_hashes": {
            "human_annotation_sha256": EXPECTED_REVIEWER_SHA256,
            "chinese_candidate_input_sha256": EXPECTED_CANDIDATE_SHA256,
            "chinese_source_input_sha256": EXPECTED_SOURCE_SHA256,
            "correction_record_sha256": EXPECTED_CORRECTION_SHA256,
            "prior_blocked_manifest_sha256": EXPECTED_PRIOR_BLOCKED_SHA256,
            "consumed_v2_sha256_only": EXPECTED_CONSUMED_V2_SHA256,
        },
        "english_parent": {
            "ordered_bundle_sha256": EXPECTED_PARENT_BUNDLE_SHA256,
            "ordered_bundle_paths": list(PARENT_FILES),
            "files": parent_hashes,
            "used_for": ["lineage", "ids", "urls", "availability"],
            "used_for_chinese_evidence_validation": False,
        },
        "historical_validation_records": HISTORICAL_RECORDS,
        "historical_blocker_resolution": {
            "v3_holdout_candidate_018": "human_corrected_and_valid",
            "v3_holdout_candidate_039": "human_corrected_and_valid",
            "v3_holdout_candidate_047": "human_corrected_and_valid",
            "v3_holdout_candidate_060": "human_corrected_and_valid",
        },
        "phase3e3_frozen_hashes": {
            "code_sha256": (
                "b05f9c35c72d06a3a45c6f08844db677fcceeaf868202e675b8cea0279b50c90"
            ),
            "prompt_schema_sha256": (
                "8dc3d1f611c0673aad90c83678629eb947a5c5281ac19e8c16d25a72f885b5dd"
            ),
            "config_sha256": (
                "af7bce4126a4b74e63d3db4a93d158efe4c37f3cbdb6c99c7efba0f327e7f1e4"
            ),
        },
        "generation": {
            "tool": "scripts/freeze_v3_zh_single_human.py",
            "tool_sha256": generation_code_sha256,
            "real_model_calls": 0,
            "production_code_modified": False,
        },
        "files": {
            relative: sha256_file(REPO_ROOT / relative) for relative in relative_files
        },
        "ordered_bundle_paths": relative_files,
        "ordered_bundle_sha256": ordered_bundle_hash(relative_files),
        "ordered_bundle_algorithm": (
            "SHA-256 of sha256sum-style lines in ordered_bundle_paths order"
        ),
        "remote_added": False,
        "push_performed": False,
    }
    output_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reviewer",
        type=Path,
        default=DEFAULT_REVIEW_ROOT / "v3_human_reviewer_zh.md",
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=DEFAULT_REVIEW_ROOT / "holdout_candidates_zh.jsonl",
    )
    parser.add_argument(
        "--sources",
        type=Path,
        default=DEFAULT_REVIEW_ROOT / "sources/source_snapshots_zh.jsonl",
    )
    parser.add_argument(
        "--correction-record",
        type=Path,
        default=DEFAULT_REVIEW_ROOT / "phase3f_b_zh_canonical_corrections_required.md",
    )
    parser.add_argument(
        "--prior-blocked-manifest",
        type=Path,
        default=DEFAULT_REVIEW_ROOT
        / "phase3f_b_zh_canonical_validation_blocked_manifest.json",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--created-at",
        default=(
            datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        ),
    )
    args = parser.parse_args()
    manifest = freeze(
        reviewer_path=args.reviewer,
        candidates_path=args.candidates,
        sources_path=args.sources,
        correction_path=args.correction_record,
        prior_blocked_path=args.prior_blocked_manifest,
        output_dir=args.out_dir,
        created_at=args.created_at,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
