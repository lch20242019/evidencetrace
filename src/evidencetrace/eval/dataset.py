"""Loading, hashing, and leakage checks for frozen eval sets."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from pydantic import ValidationError

from evidencetrace.eval.models import EvalCase, SourceFixture


class DatasetValidationError(ValueError):
    """The eval set is malformed, duplicated, leaked, or changed after freeze."""


@dataclass(frozen=True)
class LoadedDataset:
    path: Path
    cases: tuple[EvalCase, ...]
    sources: dict[str, SourceFixture]
    raw_hash: str
    split_hash: str

    @property
    def dev_cases(self) -> tuple[EvalCase, ...]:
        return tuple(case for case in self.cases if case.split == "dev")

    @property
    def test_cases(self) -> tuple[EvalCase, ...]:
        return tuple(case for case in self.cases if case.split == "test")


def canonical_case_payload(case: EvalCase) -> dict[str, Any]:
    """Return the frozen payload, excluding its derived hash field."""

    payload = case.model_dump(mode="json")
    payload.pop("case_hash", None)
    # Optional Phase 3D provenance must not alter historical frozen hashes.
    # Pop only absent fields; populated provenance remains part of new hashes.
    for field in (
        "source_sha256",
        "reviewer_record_sha256",
        "generation_code_sha256",
        "source_length_stratum",
        "expected_chunk_count",
        "expected_route",
    ):
        if payload.get(field) is None:
            payload.pop(field, None)
    return payload


def compute_case_hash(case: EvalCase) -> str:
    encoded = json.dumps(
        canonical_case_payload(case), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalize_claim(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def _safe_fixture_path(dataset_path: Path, fixture: str) -> Path:
    fixture_path = PurePosixPath(fixture)
    if fixture_path.is_absolute() or ".." in fixture_path.parts or "\\" in fixture:
        raise DatasetValidationError(f"source fixture escapes dataset root: {fixture}")
    base = (
        dataset_path.parent.parent
        if fixture_path.parts and fixture_path.parts[0] == dataset_path.parent.name
        else dataset_path.parent
    ).resolve()
    resolved = (base / Path(*fixture_path.parts)).resolve()
    try:
        resolved.relative_to(base)
    except ValueError as error:
        raise DatasetValidationError(
            f"source fixture escapes dataset root: {fixture}"
        ) from error
    return resolved


def _load_sources(
    dataset_path: Path, fixture_names: set[str]
) -> dict[str, SourceFixture]:
    sources: dict[str, SourceFixture] = {}
    for fixture in sorted(fixture_names):
        path = _safe_fixture_path(dataset_path, fixture)
        if not path.is_file():
            raise DatasetValidationError(f"source fixture does not exist: {fixture}")
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            try:
                source = SourceFixture.model_validate_json(line)
            except ValidationError as error:
                raise DatasetValidationError(
                    f"invalid source fixture {fixture}:{line_number}: {error}"
                ) from error
            actual_hash = hashlib.sha256(source.content.encode("utf-8")).hexdigest()
            if actual_hash != source.content_hash:
                raise DatasetValidationError(
                    f"source content hash mismatch for {source.source_id}"
                )
            if source.source_id in sources:
                raise DatasetValidationError(f"duplicate source_id: {source.source_id}")
            sources[source.source_id] = source
    return sources


def validate_dataset(
    cases: tuple[EvalCase, ...],
    *,
    sources: dict[str, SourceFixture] | None = None,
    frozen_hashes: dict[str, str] | None = None,
) -> tuple[EvalCase, ...]:
    """Validate IDs, duplicate claims, source provenance, and freezes.

    Dev and test may be stored in separate physical files, but an empty dataset
    is never valid. The ``Split`` schema still constrains every populated case.
    """

    if not cases:
        raise DatasetValidationError("dataset must contain at least one case")

    seen_ids: set[str] = set()
    normalized_claims: dict[str, str] = {}
    validated: list[EvalCase] = []
    for case in cases:
        if case.case_id in seen_ids:
            raise DatasetValidationError(f"duplicate case_id: {case.case_id}")
        seen_ids.add(case.case_id)
        claim = case.claim_text or case.document_path or ""
        normalized = _normalize_claim(claim)
        previous = normalized_claims.get(normalized)
        if previous is not None:
            raise DatasetValidationError(
                f"duplicate claim across cases: {previous} and {case.case_id}"
            )
        normalized_claims[normalized] = case.case_id
        expected_hash = compute_case_hash(case)
        if case.case_hash is not None and case.case_hash != expected_hash:
            raise DatasetValidationError(f"case hash mismatch: {case.case_id}")
        if frozen_hashes is not None:
            frozen = frozen_hashes.get(case.case_id)
            if frozen is None:
                raise DatasetValidationError(f"missing frozen hash: {case.case_id}")
            if frozen != expected_hash:
                raise DatasetValidationError(
                    f"frozen case was modified: {case.case_id}"
                )
        if sources is not None:
            source = sources.get(case.source_id)
            if source is None:
                raise DatasetValidationError(
                    f"case {case.case_id} references unknown source {case.source_id}"
                )
            if source.url != case.source_url:
                raise DatasetValidationError(
                    f"case {case.case_id} source_url does not match fixture"
                )
            if (
                case.source_sha256 is not None
                and case.source_sha256 != source.content_hash
            ):
                raise DatasetValidationError(
                    f"case {case.case_id} source_sha256 does not match fixture"
                )
            if (
                case.gold_evidence_span
                and case.gold_evidence_span not in source.content
            ):
                raise DatasetValidationError(
                    f"gold evidence span is not in source for {case.case_id}"
                )
        validated.append(case.model_copy(update={"case_hash": expected_hash}))
    return tuple(validated)


def dataset_hash(path: Path, sources: dict[str, SourceFixture] | None = None) -> str:
    digest = hashlib.sha256(path.read_bytes())
    for source_id in sorted(sources or {}):
        source = (sources or {})[source_id]
        payload = json.dumps(
            source.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        )
        digest.update(payload.encode("utf-8"))
    return digest.hexdigest()


def split_hash(cases: tuple[EvalCase, ...]) -> str:
    payload = [
        {
            "case_id": case.case_id,
            "split": case.split,
            "case_hash": compute_case_hash(case),
        }
        for case in sorted(cases, key=lambda item: item.case_id)
    ]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def load_dataset(path: Path | str) -> LoadedDataset:
    dataset_path = Path(path).resolve()
    if not dataset_path.is_file():
        raise DatasetValidationError(f"eval dataset does not exist: {path}")
    cases: list[EvalCase] = []
    for line_number, line in enumerate(
        dataset_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            cases.append(EvalCase.model_validate_json(line))
        except ValidationError as error:
            raise DatasetValidationError(
                f"invalid eval case {path}:{line_number}: {error}"
            ) from error
    fixtures = {case.source_fixture for case in cases}
    sources = _load_sources(dataset_path, fixtures)
    frozen_path = dataset_path.with_name(dataset_path.stem + ".frozen_hashes.json")
    frozen = None
    if frozen_path.exists():
        try:
            frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise DatasetValidationError(
                f"invalid frozen hash file: {frozen_path}"
            ) from error
        if not isinstance(frozen, dict):
            raise DatasetValidationError("frozen hash file must contain an object")
    validated = validate_dataset(tuple(cases), sources=sources, frozen_hashes=frozen)
    return LoadedDataset(
        path=dataset_path,
        cases=validated,
        sources=sources,
        raw_hash=dataset_hash(dataset_path, sources),
        split_hash=split_hash(validated),
    )


def frozen_hashes(cases: tuple[EvalCase, ...]) -> dict[str, str]:
    """Return the hash map suitable for a checked-in frozen-hash file."""

    return {case.case_id: compute_case_hash(case) for case in cases}


__all__ = [
    "DatasetValidationError",
    "LoadedDataset",
    "canonical_case_payload",
    "compute_case_hash",
    "dataset_hash",
    "frozen_hashes",
    "load_dataset",
    "split_hash",
    "validate_dataset",
]
