#!/usr/bin/env python3
"""Freeze the v2 synthetic holdout from the private Reviewer A record.

The generator deliberately projects only candidate identity/source fields. Model
proposals and candidate annotation placeholders are never copied into gold data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
V2_DIR = REPO_ROOT / "eval_sets" / "v2"
DEFAULT_REVIEWER_RECORD = Path(
    "/data/yz_data/evidencetrace_review_records/human_reviewer_a.md"
)
EXPECTED_REVIEWER_SHA256 = (
    "4b4d3d8e25960d71e9b7cb151b1caec25fe9089dcd7bf1dd0a5352e13e470192"
)
FREEZE_BASE_COMMIT = "457e7b3242e057d15bf11568966d4adb11c861da"
RELATIONS = {
    "entailed",
    "partially_entailed",
    "contradicted",
    "not_in_source",
    "source_unavailable",
    "not_checkable",
}
SUBSTANTIVE_RELATIONS = {"entailed", "partially_entailed", "contradicted"}
PLACEHOLDER_RE = re.compile(r"^_+$")
CASE_RE = re.compile(
    r"^### (?P<case_id>v2_holdout_\d{3})\n\n"
    r"Claim: (?P<claim>[^\n]+)\n\n"
    r"- Human relation: (?P<relation>[^\n]+)\n"
    r"- Exact evidence span: (?P<evidence>[^\n]+)\n"
    r"- Reviewer notes: (?P<notes>[^\n]+)$",
    flags=re.MULTILINE,
)


class FreezeError(ValueError):
    """The private annotation or frozen candidate inputs are inconsistent."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _frontmatter(text: str) -> dict[str, str]:
    match = re.match(r"^---\n(?P<body>.*?)\n---\n", text, flags=re.DOTALL)
    if match is None:
        raise FreezeError("Reviewer A record has no frontmatter")
    result: dict[str, str] = {}
    for line in match.group("body").splitlines():
        key, separator, value = line.partition(":")
        if separator:
            result[key.strip()] = value.strip().strip('"')
    return result


def parse_reviewer_record(
    path: Path, expected_sha256: str
) -> dict[str, dict[str, str | None]]:
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise FreezeError(
            "Reviewer A SHA-256 mismatch: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    text = path.read_text(encoding="utf-8")
    metadata = _frontmatter(text)
    required_metadata = {
        "reviewer_type": "human",
        "annotation_status": "single_human_review_complete",
        "counts_as_human_review": "true",
    }
    for key, expected in required_metadata.items():
        if metadata.get(key) != expected:
            raise FreezeError(f"Reviewer A metadata {key} must be {expected!r}")

    annotations: dict[str, dict[str, str | None]] = {}
    for match in CASE_RE.finditer(text):
        item = {key: value.strip() for key, value in match.groupdict().items()}
        case_id = item.pop("case_id")
        relation = item["relation"]
        evidence = item["evidence"]
        if case_id in annotations:
            raise FreezeError(f"duplicate Reviewer A case: {case_id}")
        if relation not in RELATIONS:
            raise FreezeError(f"invalid Reviewer A relation for {case_id}: {relation}")
        if not item["notes"]:
            raise FreezeError(f"Reviewer A notes are blank for {case_id}")
        item["evidence"] = None if PLACEHOLDER_RE.fullmatch(evidence) else evidence
        if relation in SUBSTANTIVE_RELATIONS and item["evidence"] is None:
            raise FreezeError(f"substantive Reviewer A evidence is blank for {case_id}")
        annotations[case_id] = item

    expected_ids = {f"v2_holdout_{number:03d}" for number in range(1, 61)}
    if set(annotations) != expected_ids:
        missing = sorted(expected_ids - set(annotations))
        extra = sorted(set(annotations) - expected_ids)
        raise FreezeError(
            "Reviewer A must contain exactly 60 cases; "
            f"missing={missing}, extra={extra}"
        )
    for number in range(41, 51):
        case_id = f"v2_holdout_{number:03d}"
        if annotations[case_id]["relation"] != "source_unavailable":
            raise FreezeError(f"{case_id} must be source_unavailable")
    if {str(item["relation"]) for item in annotations.values()} != RELATIONS:
        raise FreezeError("Reviewer A holdout must have support for all six relations")
    return annotations


def load_sources(path: Path) -> dict[str, dict[str, Any]]:
    sources: dict[str, dict[str, Any]] = {}
    for source in load_jsonl(path):
        source_id = source["source_id"]
        if source_id in sources:
            raise FreezeError(f"duplicate source fixture: {source_id}")
        content_hash = hashlib.sha256(source["content"].encode("utf-8")).hexdigest()
        if content_hash != source["content_hash"]:
            raise FreezeError(f"source content hash mismatch: {source_id}")
        sources[source_id] = source
    return sources


def canonical_case_hash(case: dict[str, Any]) -> str:
    payload = dict(case)
    payload.pop("case_hash", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_cases(
    candidates: list[dict[str, Any]],
    annotations: dict[str, dict[str, str | None]],
    sources: dict[str, dict[str, Any]],
    *,
    reviewer_sha256: str,
    generation_code_sha256: str,
) -> list[dict[str, Any]]:
    if len(candidates) != 60:
        raise FreezeError(f"expected 60 candidates, got {len(candidates)}")
    cases: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for candidate in candidates:
        case_id = candidate["candidate_id"]
        if case_id in seen_ids:
            raise FreezeError(f"duplicate candidate: {case_id}")
        seen_ids.add(case_id)
        if case_id not in annotations:
            raise FreezeError(f"candidate has no Reviewer A annotation: {case_id}")
        source = sources.get(candidate["source_id"])
        if source is None:
            raise FreezeError(f"unknown source for {case_id}: {candidate['source_id']}")
        annotation = annotations[case_id]
        if candidate["claim"] != annotation["claim"]:
            raise FreezeError(
                f"claim mismatch between candidate and Reviewer A: {case_id}"
            )
        if candidate["source_sha256"] != source["content_hash"]:
            raise FreezeError(f"candidate source hash mismatch: {case_id}")
        if candidate["source_text"] != source["content"]:
            raise FreezeError(f"candidate source text mismatch: {case_id}")
        evidence = annotation["evidence"]
        if evidence is not None and evidence not in source["content"]:
            raise FreezeError(
                f"Reviewer A evidence is not a source substring: {case_id}"
            )

        case: dict[str, Any] = {
            "case_id": case_id,
            "document_path": None,
            "claim_text": candidate["claim"],
            "claim_line": 1,
            "source_fixture": "sources/synthetic_sources.jsonl",
            "source_id": source["source_id"],
            "source_url": source["url"],
            "gold_relation": annotation["relation"],
            "gold_evidence_span": evidence,
            "claim_type": "citation_pair",
            "mutation_type": "none",
            "split": "test",
            "provenance": (
                "single human review of an author-created synthetic holdout candidate"
            ),
            "annotation_status": "single_human_review",
            "annotation_notes": (
                "Frozen single-human annotation; private reviewer notes are excluded."
            ),
            "source_sha256": source["content_hash"],
            "reviewer_record_sha256": reviewer_sha256,
            "generation_code_sha256": generation_code_sha256,
        }
        case["case_hash"] = canonical_case_hash(case)
        cases.append(case)
    if seen_ids != set(annotations):
        raise FreezeError("candidate and Reviewer A case IDs do not match")
    return sorted(cases, key=lambda item: item["case_id"])


def combined_dataset_hash(
    dataset_path: Path, sources: dict[str, dict[str, Any]]
) -> str:
    digest = hashlib.sha256(dataset_path.read_bytes())
    source_fields = (
        "source_id",
        "url",
        "content",
        "provenance",
        "content_hash",
        "available",
    )
    for source_id in sorted(sources):
        payload = {key: sources[source_id][key] for key in source_fields}
        digest.update(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
    return digest.hexdigest()


def freeze(
    *,
    candidates_path: Path,
    sources_path: Path,
    reviewer_path: Path,
    output_path: Path,
    expected_reviewer_sha256: str = EXPECTED_REVIEWER_SHA256,
) -> tuple[Path, Path, Path]:
    generation_path = Path(__file__).resolve()
    generation_sha256 = sha256_file(generation_path)
    annotations = parse_reviewer_record(reviewer_path, expected_reviewer_sha256)
    sources = load_sources(sources_path)
    cases = build_cases(
        load_jsonl(candidates_path),
        annotations,
        sources,
        reviewer_sha256=expected_reviewer_sha256,
        generation_code_sha256=generation_sha256,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(
            json.dumps(case, ensure_ascii=False, separators=(",", ":")) + "\n"
            for case in cases
        ),
        encoding="utf-8",
    )

    frozen_path = output_path.with_name(output_path.stem + ".frozen_hashes.json")
    frozen_path.write_text(
        json.dumps(
            {case["case_id"]: case["case_hash"] for case in cases},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    freeze_path = output_path.with_name(output_path.stem + ".freeze.json")
    freeze_record = {
        "schema_version": "phase3d-single-human-freeze-1",
        "benchmark_validity": "single_human_synthetic_holdout",
        "public_benchmark_eligible": False,
        "annotation_status": "single_human_review",
        "case_count": len(cases),
        "freeze_base_commit": FREEZE_BASE_COMMIT,
        "holdout_execution_status": "not_run",
        "reviewer_record_sha256": expected_reviewer_sha256,
        "candidate_input_sha256": sha256_file(candidates_path),
        "source_fixture_sha256": sha256_file(sources_path),
        "generation_code": "scripts/generate_v2_single_human_holdout.py",
        "generation_code_sha256": generation_sha256,
        "dataset_file_sha256": sha256_file(output_path),
        "dataset_hash": combined_dataset_hash(output_path, sources),
        "frozen_hashes_file_sha256": sha256_file(frozen_path),
    }
    freeze_path.write_text(
        json.dumps(freeze_record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output_path, frozen_path, freeze_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates", type=Path, default=V2_DIR / "holdout_candidates.jsonl"
    )
    parser.add_argument(
        "--sources",
        type=Path,
        default=V2_DIR / "sources/synthetic_sources.jsonl",
    )
    parser.add_argument("--reviewer", type=Path, default=DEFAULT_REVIEWER_RECORD)
    parser.add_argument(
        "--out", type=Path, default=V2_DIR / "holdout_single_human.jsonl"
    )
    parser.add_argument("--expected-reviewer-sha256", default=EXPECTED_REVIEWER_SHA256)
    args = parser.parse_args()
    for path in freeze(
        candidates_path=args.candidates,
        sources_path=args.sources,
        reviewer_path=args.reviewer,
        output_path=args.out,
        expected_reviewer_sha256=args.expected_reviewer_sha256,
    ):
        print(path)


if __name__ == "__main__":
    main()
