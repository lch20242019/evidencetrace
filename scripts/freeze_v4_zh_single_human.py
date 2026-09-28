#!/usr/bin/env python3
"""Validate and freeze the v4_zh single-human internal holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evidencetrace.checks.deterministic import is_high_confidence_subjective
from evidencetrace.eval.baselines import (
    build_evidence_chunks,
    estimate_full_context_size,
)
from evidencetrace.eval.dataset import compute_case_hash
from evidencetrace.eval.models import EvalCase, SourceFixture
from evidencetrace.eval.router import AdaptiveRouterConfig, select_adaptive_route
from evidencetrace.models import Relation

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "eval_sets/v4_zh"
DEFAULT_REVIEWER = (
    Path.home()
    / "Projects"
    / "evidencetrace_review_records"
    / "v4_zh"
    / "human_reviewer_a.md"
)
FREEZE_BASE_COMMIT = "62bd1eee19ad5224ee50596d1790c76a6a33afb2"
REVIEWER_SHA256 = "46657539fd1d67b22fa15c9a34ee001a198b0952fb808a911d94bfa58750598a"
INPUT_HASHES = {
    "source_snapshots_sha256": (
        "34f9128c38ae742a431b6be02ef44bc99887fad648b42e0bbe05c9bc61054d4f"
    ),
    "candidates_sha256": (
        "9803cc48dfa1a95911ae179a837d0448e06f6cef76a0b6762b14be1417f4dff3"
    ),
    "blank_reviewer_packet_sha256": (
        "d143adf4c4e85aa197a5dd109791d1aadcfd5bd3276e7488468a83ce0f5b03bd"
    ),
    "annotation_guide_sha256": (
        "04d8a9b2479bba818853f9b7d9d05d9fa60ea0c55fffc77cdcae78e32a9a5d2e"
    ),
    "leakage_audit_sha256": (
        "303f0e1d24f3ca396bd4b52afa1e595b75538e3b67c55b11b2dcf6ae7af4938f"
    ),
    "blind_manifest_sha256": (
        "2f8e039f0810c7d9602aed8a7fdaf34365a6ff372ed34243bb8081617425b8fb"
    ),
    "candidate_bundle_sha256": (
        "0e8dc6ccc0bf66302032cc94f74591e3ad4189ca04e09864dcb1449316be3d9d"
    ),
    "external_human_record_sha256": REVIEWER_SHA256,
}
FREEZE_BOUNDARY = {
    "code_sha256": ("88ace6e1b9675febb6071101aa04d5ee274b9c6986bdc6652156225d52c3d641"),
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
CONSUMED_HASHES = {
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
BLIND_PATHS = (
    "eval_sets/v4_zh/sources/source_snapshots_zh.jsonl",
    "eval_sets/v4_zh/holdout_candidates_zh.jsonl",
    "eval_sets/v4_zh/REVIEW_PACKET.md",
    "eval_sets/v4_zh/ANNOTATION_GUIDE.md",
    "eval_sets/v4_zh/LEAKAGE_AUDIT.md",
)
BUNDLE_PATHS = (
    *BLIND_PATHS,
    "eval_sets/v4_zh/manifest.json",
    "eval_sets/v4_zh/sources/evaluation_source_fixtures_zh.jsonl",
    "eval_sets/v4_zh/holdout_single_human_zh.jsonl",
    "eval_sets/v4_zh/holdout_single_human_zh.frozen_hashes.json",
    "scripts/freeze_v4_zh_single_human.py",
    "tests/test_eval_v4_zh_candidates.py",
    "tests/test_eval_v4_zh_single_human.py",
)
RELATION_MAP = {
    "entailed": "entailed",
    "partially_entailed": "partially_entailed",
    "contradicted": "contradicted",
    "not_in_source": "not_in_source",
    "source_unavailable": "source_unavailable",
    "not_checkable": "not_checkable",
    "蕴含": "entailed",
    "部分蕴含": "partially_entailed",
    "矛盾": "contradicted",
    "原文未提及": "not_in_source",
    "来源不可用": "source_unavailable",
    "不可验证": "not_checkable",
}
SUBSTANTIVE = {"entailed", "partially_entailed", "contradicted"}
CASE_RE = re.compile(r"^## (v4_zh_candidate_\d{3})$", re.MULTILINE)


class FreezeError(ValueError):
    """The human record or a frozen input failed validation."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def require_hash(path: Path, expected: str) -> None:
    if sha256_file(path) != expected:
        raise FreezeError(f"hash_mismatch:{path.name}")


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def bundle_hash(paths: tuple[str, ...]) -> str:
    payload = "".join(
        f"{sha256_file(ROOT / relative)}  {relative}\n" for relative in paths
    )
    return sha256_text(payload)


def parse_utc(value: str) -> datetime:
    candidate = value
    compact_hour = re.fullmatch(
        r"(\d{4}-\d{2}-\d{2}T)(\d)(:\d{2}:\d{2}Z)",
        value,
    )
    if compact_hour:
        candidate = (
            f"{compact_hour.group(1)}0{compact_hour.group(2)}{compact_hour.group(3)}"
        )
    try:
        parsed = datetime.fromisoformat(candidate.replace("Z", "+00:00"))
    except ValueError as error:
        raise FreezeError("completion_utc_invalid") from error
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise FreezeError("completion_utc_invalid")
    if parsed > datetime.now(UTC):
        raise FreezeError("completion_utc_in_future")
    return parsed


def field(block: str, name: str) -> str | None:
    match = re.search(
        rf"^- {re.escape(name)}:[ \t]*(.*)$",
        block,
        re.MULTILINE,
    )
    return match.group(1).strip() if match else None


def metadata(text: str, name: str) -> str | None:
    match = re.search(
        rf"^- {re.escape(name)}:[ \t]*(.+?)[ \t]*$",
        text,
        re.MULTILINE | re.IGNORECASE,
    )
    return match.group(1).strip() if match else None


def parse_annotations(
    path: Path,
) -> tuple[dict[str, dict[str, str | None]], dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    private = {
        "reviewer_name": metadata(text, "reviewer_name"),
        "reviewer_id": metadata(text, "reviewer_id"),
        "completion_utc": metadata(text, "completion_utc"),
        "guide_hash": metadata(text, "annotation_guide_sha256"),
        "attestation": metadata(
            text,
            "independent_human_annotation_attestation",
        ),
        "viewed_model_output": metadata(text, "viewed_model_output"),
    }
    if any(not value for value in private.values()):
        raise FreezeError("reviewer_provenance_missing")
    parse_utc(str(private["completion_utc"]))
    if private["guide_hash"] != INPUT_HASHES["annotation_guide_sha256"]:
        raise FreezeError("annotation_guide_hash_mismatch")
    if str(private["attestation"]).casefold() in {"false", "no", "否", "0"}:
        raise FreezeError("independent_attestation_invalid")
    if str(private["viewed_model_output"]).casefold() not in {
        "false",
        "no",
        "否",
        "0",
    }:
        raise FreezeError("viewed_model_output_must_be_false")

    matches = list(CASE_RE.finditer(text))
    annotations: dict[str, dict[str, str | None]] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        block = text[match.end() : end]
        case_id = match.group(1)
        values = {
            name: field(block, name)
            for name in (
                "Source ID",
                "Canonical source",
                "Claim",
                "Relation",
                "Evidence",
                "Notes",
            )
        }
        if any(value is None for value in values.values()):
            raise FreezeError(f"annotation_field_missing:{case_id}")
        source_id = str(values["Source ID"])
        if source_id.startswith("`") and source_id.endswith("`"):
            source_id = source_id[1:-1]
        relation = RELATION_MAP.get(str(values["Relation"]))
        if relation is None:
            raise FreezeError(f"annotation_relation_invalid:{case_id}")
        evidence = str(values["Evidence"]) or None
        if relation in SUBSTANTIVE and evidence is None:
            raise FreezeError(f"annotation_evidence_missing:{case_id}")
        if relation not in SUBSTANTIVE and evidence is not None:
            raise FreezeError(f"annotation_evidence_must_be_empty:{case_id}")
        if not values["Notes"]:
            raise FreezeError(f"annotation_notes_missing:{case_id}")
        if case_id in annotations:
            raise FreezeError(f"annotation_duplicate:{case_id}")
        annotations[case_id] = {
            "source_id": source_id,
            "canonical_source": str(values["Canonical source"]),
            "claim": str(values["Claim"]),
            "relation": relation,
            "evidence": evidence,
        }
    if len(annotations) != 72:
        raise FreezeError("annotation_case_count_invalid")
    public_provenance = {
        "reviewer_name_or_pseudonym_present": True,
        "reviewer_id_present": True,
        "completion_utc": private["completion_utc"],
        "annotation_guide_sha256": private["guide_hash"],
        "independent_human_annotation_attestation_verified": True,
        "viewed_model_output": False,
    }
    return annotations, public_provenance


def validate_inputs(
    candidates: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    annotations: dict[str, dict[str, str | None]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    candidate_by_id = {str(row["candidate_id"]): row for row in candidates}
    source_by_id = {str(row["source_id"]): row for row in sources}
    if len(candidate_by_id) != 72 or len(source_by_id) != 24:
        raise FreezeError("candidate_or_source_count_invalid")
    if set(candidate_by_id) != set(annotations):
        raise FreezeError("annotation_candidate_id_mismatch")
    if Counter(row["source_id"] for row in candidates) != {
        source_id: 3 for source_id in source_by_id
    }:
        raise FreezeError("cases_per_source_invalid")

    for case_id, annotation in annotations.items():
        candidate = candidate_by_id[case_id]
        source_id = str(candidate["source_id"])
        source = source_by_id[source_id]
        expected_source = str(source["content"]) if source["available"] else "[不可用]"
        if annotation["source_id"] != source_id:
            raise FreezeError(f"annotation_source_id_mismatch:{case_id}")
        if annotation["claim"] != candidate["claim"]:
            raise FreezeError(f"annotation_claim_mismatch:{case_id}")
        if annotation["canonical_source"] != expected_source:
            raise FreezeError(f"annotation_source_mismatch:{case_id}")
        evidence = annotation["evidence"]
        if evidence is not None and str(evidence) not in str(source["content"]):
            raise FreezeError(f"annotation_evidence_not_substring:{case_id}")
        relation = annotation["relation"]
        if relation == "source_unavailable" and source["available"]:
            raise FreezeError(f"available_source_marked_unavailable:{case_id}")
        if not source["available"] and relation != "source_unavailable":
            raise FreezeError(f"unavailable_source_relation_invalid:{case_id}")

    for source_id, source in source_by_id.items():
        if source["original_language"] != "zh-CN":
            raise FreezeError(f"source_language_invalid:{source_id}")
        content = str(source["content"])
        if source["available"] and sha256_text(content) != source["content_sha256"]:
            raise FreezeError(f"source_content_hash_mismatch:{source_id}")
        if not source["available"] and content:
            raise FreezeError(f"unavailable_source_has_content:{source_id}")
    return candidate_by_id, source_by_id


def source_fixtures(
    source_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source_id in sorted(source_by_id):
        source = source_by_id[source_id]
        content = str(source["content"])
        provenance = json.dumps(
            {
                "canonical_source_language": "zh-CN",
                "derived_from": ("eval_sets/v4_zh/sources/source_snapshots_zh.jsonl"),
                "original_snapshot_sha256": INPUT_HASHES["source_snapshots_sha256"],
                "publisher": source["publisher"],
                "retrieved_at_utc": source["retrieved_at_utc"],
                "retrieval_status": source["retrieval_status"],
                "source_id": source_id,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        rows.append(
            SourceFixture(
                source_id=source_id,
                url=str(source["url"]),
                content=content,
                provenance=provenance,
                content_hash=sha256_text(content),
                available=bool(source["available"]),
            ).model_dump(mode="json")
        )
    return rows


def route_for(
    candidate: dict[str, Any],
    source: dict[str, Any],
    fixture: SourceFixture,
) -> str:
    case = EvalCase(
        case_id=str(candidate["candidate_id"]),
        claim_text=str(candidate["claim"]),
        source_fixture="sources/evaluation_source_fixtures_zh.jsonl",
        source_id=fixture.source_id,
        source_url=fixture.url,
        gold_relation=Relation.NOT_IN_SOURCE,
        claim_type="blind_candidate_route_freeze",
        mutation_type="not_recorded",
        split="test",
        provenance="Phase 3H-B deterministic route freeze.",
        annotation_status="provisional",
        annotation_notes="Route derived without using the human relation.",
    )
    chunks = build_evidence_chunks(fixture)
    if len(chunks) != source["expected_chunk_count"]:
        raise FreezeError(f"chunk_count_mismatch:{fixture.source_id}")
    return select_adaptive_route(
        source_available=fixture.available,
        purely_subjective=is_high_confidence_subjective(str(candidate["claim"])),
        chunk_count=len(chunks),
        estimated_context_size=estimate_full_context_size(case, fixture),
        config=AdaptiveRouterConfig(),
    ).selected_route


def gold_cases(
    candidate_by_id: dict[str, dict[str, Any]],
    source_by_id: dict[str, dict[str, Any]],
    fixture_rows: list[dict[str, Any]],
    annotations: dict[str, dict[str, str | None]],
    generation_hash: str,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    fixtures = {
        str(row["source_id"]): SourceFixture.model_validate(row) for row in fixture_rows
    }
    rows: list[dict[str, Any]] = []
    routes: Counter[str] = Counter()
    for case_id in sorted(candidate_by_id):
        candidate = candidate_by_id[case_id]
        source = source_by_id[str(candidate["source_id"])]
        fixture = fixtures[str(candidate["source_id"])]
        annotation = annotations[case_id]
        route = route_for(candidate, source, fixture)
        routes[route] += 1
        case = EvalCase(
            case_id=case_id,
            claim_text=str(candidate["claim"]),
            source_fixture="sources/evaluation_source_fixtures_zh.jsonl",
            source_id=fixture.source_id,
            source_url=fixture.url,
            gold_relation=Relation(str(annotation["relation"])),
            gold_evidence_span=(
                str(annotation["evidence"]) if annotation["evidence"] else None
            ),
            claim_type="citation_pair",
            mutation_type="human_annotated_blind_candidate",
            split="test",
            provenance=(
                "Single independent human review of a blind native Chinese "
                "official-source holdout candidate."
            ),
            annotation_status="single_human_review",
            annotation_notes=(
                "Frozen single-human label; private identity and full notes "
                "remain outside the repository."
            ),
            source_sha256=fixture.content_hash,
            reviewer_record_sha256=REVIEWER_SHA256,
            generation_code_sha256=generation_hash,
            source_length_stratum=str(source["source_length_stratum"]),
            expected_chunk_count=int(source["expected_chunk_count"]),
            expected_route=route,
        )
        rows.append(
            case.model_copy(update={"case_hash": compute_case_hash(case)}).model_dump(
                mode="json"
            )
        )
    return rows, routes


def freeze(reviewer: Path, frozen_at: str) -> dict[str, Any]:
    require_hash(reviewer, REVIEWER_SHA256)
    fixed = {
        PACK / "sources/source_snapshots_zh.jsonl": INPUT_HASHES[
            "source_snapshots_sha256"
        ],
        PACK / "holdout_candidates_zh.jsonl": INPUT_HASHES["candidates_sha256"],
        PACK / "REVIEW_PACKET.md": INPUT_HASHES["blank_reviewer_packet_sha256"],
        PACK / "ANNOTATION_GUIDE.md": INPUT_HASHES["annotation_guide_sha256"],
        PACK / "LEAKAGE_AUDIT.md": INPUT_HASHES["leakage_audit_sha256"],
        PACK / "manifest.json": INPUT_HASHES["blind_manifest_sha256"],
    }
    for path, expected in fixed.items():
        require_hash(path, expected)
    if bundle_hash(BLIND_PATHS) != INPUT_HASHES["candidate_bundle_sha256"]:
        raise FreezeError("candidate_bundle_hash_mismatch")
    for relative, expected in CONSUMED_HASHES.items():
        require_hash(ROOT / relative, expected)

    annotations, provenance = parse_annotations(reviewer)
    candidates, sources = validate_inputs(
        jsonl(PACK / "holdout_candidates_zh.jsonl"),
        jsonl(PACK / "sources/source_snapshots_zh.jsonl"),
        annotations,
    )
    generation_hash = sha256_file(Path(__file__).resolve())
    fixtures = source_fixtures(sources)
    cases, routes = gold_cases(
        candidates,
        sources,
        fixtures,
        annotations,
        generation_hash,
    )
    expected_routes = {
        "full_context_single_agent": 17,
        "retrieval_judge": 31,
        "deterministic_not_checkable": 12,
        "deterministic_source_unavailable": 12,
    }
    if routes != expected_routes:
        raise FreezeError("adaptive_route_distribution_changed")

    fixture_path = PACK / "sources/evaluation_source_fixtures_zh.jsonl"
    gold_path = PACK / "holdout_single_human_zh.jsonl"
    frozen_path = PACK / "holdout_single_human_zh.frozen_hashes.json"
    write_jsonl(fixture_path, fixtures)
    write_jsonl(gold_path, cases)
    frozen_path.write_text(
        json.dumps(
            {case["case_id"]: case["case_hash"] for case in cases},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if any(not (ROOT / relative).is_file() for relative in BUNDLE_PATHS):
        raise FreezeError("gold_bundle_input_missing")

    distribution = Counter(
        str(annotation["relation"]) for annotation in annotations.values()
    )
    evidence_count = sum(
        annotation["evidence"] is not None for annotation in annotations.values()
    )
    manifest = {
        "schema_version": "phase3h-b-v4-zh-single-human-freeze-v1",
        "pack_id": "phase3h-v4-zh-single-human-gold",
        "status": "single_human_review_complete_internal_gate_not_run",
        "checkpoint_name": "phase3h-v4-zh-single-human-gold",
        "freeze_base_commit": FREEZE_BASE_COMMIT,
        "frozen_at_utc": frozen_at,
        "case_count": 72,
        "valid_case_count": 72,
        "source_count": 24,
        "cases_per_source": 3,
        "relation_distribution": {
            relation: distribution[relation]
            for relation in sorted(set(RELATION_MAP.values()))
        },
        "evidence_validation": {
            "contiguous_substring_count": evidence_count,
            "required_empty_count": 72 - evidence_count,
            "invalid_count": 0,
        },
        "benchmark_status": {
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
        },
        "private_reviewer_provenance": {
            **provenance,
            "record_location": "retained_outside_repository",
            "reviewer_record_sha256": REVIEWER_SHA256,
            "identity_and_full_notes_published": False,
        },
        "input_hashes": INPUT_HASHES,
        "freeze_boundary": FREEZE_BOUNDARY,
        "consumed_byte_only_hashes": CONSUMED_HASHES,
        "adaptive_expected_routes": dict(sorted(routes.items())),
        "formal_gate_preregistration_source": "eval_sets/v4_zh/manifest.json",
        "generation": {
            "tool": "scripts/freeze_v4_zh_single_human.py",
            "tool_sha256": generation_hash,
            "real_model_calls": 0,
            "holdout_executed": False,
            "production_code_modified": False,
            "intended_construction_strata_used": False,
        },
        "files": {relative: sha256_file(ROOT / relative) for relative in BUNDLE_PATHS},
        "ordered_bundle_paths": list(BUNDLE_PATHS),
        "ordered_bundle_sha256": bundle_hash(BUNDLE_PATHS),
        "ordered_bundle_algorithm": (
            "SHA-256 of sha256sum-style lines in ordered path order"
        ),
        "remote_added": False,
        "push_performed": False,
    }
    (PACK / "review_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewer", type=Path, default=DEFAULT_REVIEWER)
    parser.add_argument(
        "--frozen-at",
        default=(
            datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        ),
    )
    args = parser.parse_args()
    result = freeze(args.reviewer, args.frozen_at)
    print(
        json.dumps(
            {
                "case_count": result["case_count"],
                "ordered_bundle_sha256": result["ordered_bundle_sha256"],
                "relation_distribution": result["relation_distribution"],
                "status": result["status"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
