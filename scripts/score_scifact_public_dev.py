"""Score generic EvidenceTrace predictions on a frozen SciFact public-dev pack.

The generic runner stores one compatibility ``gold_evidence_span``.  SciFact can
annotate several alternative rationales, each containing one or more sentences.
This scorer therefore treats ``gold_rationales.jsonl`` as the authoritative
evidence gold and never promotes the runner's single-span metrics to headline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import Any

from evidencetrace.eval.dataset import LoadedDataset, load_dataset

SCORER_VERSION = "scifact-public-dev-scoring-v3"
LABELS = ("SUPPORT", "CONTRADICT", "NOINFO")
OFF_TAXONOMY = "OFF_TAXONOMY"
RELATION_TO_LABEL = {
    "entailed": "SUPPORT",
    "contradicted": "CONTRADICT",
    "not_in_source": "NOINFO",
}
RETRIEVAL_K = (1, 3, 5)
BOOTSTRAP_SEED = 20260903
BOOTSTRAP_SAMPLES = 10_000
BASELINE_NAME = "lexical_rules"
CANDIDATE_NAME = "retrieval_judge_scifact_nli"
REPORTABLE_RUNNER_VERSION = "scifact-public-dev-local-nli-runner-v2"
REPORTABLE_PACK_SCHEMA_VERSION = "scifact-public-dev-v2"
REPORTABLE_TASK = "scifact_public_dev_oracle_cited_abstract_pair"
REPORTABLE_TASK_PROFILE = {
    "version": "scifact-3way-prediction-profile-v1",
    "allowed_relations": ["entailed", "contradicted", "not_in_source"],
    "claim_checkability": "forced_checkable_by_scifact_task_contract",
    "partial_support_policy": (
        "insufficient full-claim coverage becomes not_in_source during "
        "prediction; no scorer label mapping"
    ),
    "source_unavailable_policy": "fail_closed",
}
REPORTABLE_EVALUATION_MODE = "local_transformers_nli_reportable"
REPORTABLE_ROW_MODEL_PROVIDER = "local_transformers_nli"
REPORTABLE_PROVENANCE_SCOPE = (
    "Consistency check against local hf_hub download metadata; this "
    "does not cryptographically authenticate the remote repository."
)
SENTENCE_INDEX_RELATIVE = "sources/sentence_index.jsonl"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
WEIGHT_FILE_RE = re.compile(
    r"^(?:model(?:-\d{5}-of-\d{5})?\.safetensors|"
    r"pytorch_model(?:-\d{5}-of-\d{5})?\.bin)$"
)
TELEMETRY_FIELDS = (
    "external_provider_calls",
    "local_nli_score_many_calls",
    "local_nli_pair_count",
    "local_nli_subset_count",
    "local_nli_batch_count",
)

BOUNDARIES = (
    "SciFact public dev is a development split, not a blind test or "
    "leaderboard result.",
    "The evaluation uses oracle cited abstracts; it does not measure "
    "full-corpus document retrieval.",
    "The pair runner bypasses Claim Miner.",
    "SciFact contains English scientific abstracts and does not represent "
    "Chinese technical documents or production traffic.",
    "The lexical baseline is a repository heuristic; the candidate is a pinned "
    "local three-label NLI cross-encoder, not an official SciFact leaderboard model.",
    "Generic runner single-gold-span evidence metrics are compatibility-only "
    "and are not headline metrics.",
    "Overlap Hit@K and overlap MRR are diagnostics only; strict retrieval "
    "metrics require full character coverage of official sentence spans.",
    "Exact prediction text that occurs more than once in a source is rejected "
    "as ambiguous rather than assigned to an arbitrary sentence.",
    "lexical_rules and retrieval_judge_scifact_nli consume the same "
    "LexicalRetriever chunks and top-5 context by construction; their retrieval "
    "metrics are identical, so that comparison isolates judgment policy.",
    "Structured predicted evidence is scored by exact source offsets and sentence "
    "locators; the legacy single-span field is compatibility-only.",
)


class ScoringError(ValueError):
    """The pack or generic predictions cannot be scored without ambiguity."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_json_sha256(payload: Any) -> str:
    encoded = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ScoringError(f"missing JSON input: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ScoringError(f"invalid JSON input: {path}") from error
    if not isinstance(payload, dict):
        raise ScoringError(f"JSON input must be an object: {path}")
    return payload


def _safe_manifest_path(pack: Path, value: str) -> Path:
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or "\\" in value:
        raise ScoringError(f"unsafe manifest output path: {value}")
    resolved = (pack / Path(*relative.parts)).resolve()
    try:
        resolved.relative_to(pack)
    except ValueError as error:
        raise ScoringError(f"manifest output escapes pack: {value}") from error
    return resolved


def _validate_manifest(pack: Path) -> tuple[dict[str, Any], dict[str, str]]:
    manifest = _read_json(pack / "manifest.json")
    output_hashes = manifest.get("output_sha256")
    if not isinstance(output_hashes, dict):
        raise ScoringError("manifest requires output_sha256")
    required = {"cases.jsonl", "gold_rationales.jsonl"}
    source_entries = {
        name
        for name in output_hashes
        if isinstance(name, str) and name.startswith("sources/")
    }
    if not required <= set(output_hashes) or not source_entries:
        raise ScoringError("manifest omits required cases/gold/source output hashes")
    verified: dict[str, str] = {}
    for raw_name, raw_hash in sorted(output_hashes.items()):
        if not isinstance(raw_name, str) or not isinstance(raw_hash, str):
            raise ScoringError("manifest output_sha256 entries must be strings")
        path = _safe_manifest_path(pack, raw_name)
        if not path.is_file():
            raise ScoringError(f"manifest output is missing: {raw_name}")
        actual = _sha256(path)
        if actual != raw_hash:
            raise ScoringError(f"manifest output hash mismatch: {raw_name}")
        verified[raw_name] = actual
    return manifest, verified


def _validate_run_manifest(
    results_path: Path, rows: list[dict[str, Any]]
) -> tuple[dict[str, Any], str]:
    manifest_path = results_path.parent / "run_manifest.json"
    manifest = _read_json(manifest_path)
    expected_hash = (
        manifest.get("outputs", {}).get("sha256", {}).get("eval_results.jsonl")
    )
    if expected_hash != _sha256(results_path):
        raise ScoringError("run manifest eval_results SHA-256 mismatch")
    baselines = manifest.get("baselines")
    counts = manifest.get("counts")
    if not isinstance(baselines, list) or not all(
        isinstance(item, str) for item in baselines
    ):
        raise ScoringError("run manifest requires a string baseline list")
    if not isinstance(counts, dict):
        raise ScoringError("run manifest requires counts")
    actual_by_baseline = Counter(row.get("baseline") for row in rows)
    actual_cases = {row.get("case_id") for row in rows}
    if set(baselines) != set(actual_by_baseline):
        raise ScoringError("run manifest baseline set mismatch")
    if (
        counts.get("cases") != len(actual_cases)
        or counts.get("rows") != len(rows)
        or counts.get("rows_by_baseline") != dict(actual_by_baseline)
    ):
        raise ScoringError("run manifest row/case counts mismatch")
    return manifest, _sha256(manifest_path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ScoringError(f"missing JSONL input: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ScoringError(f"invalid JSON at {path}:{line_number}") from error
        if not isinstance(row, dict):
            raise ScoringError(f"JSONL row must be an object at {path}:{line_number}")
        rows.append(row)
    if not rows:
        raise ScoringError(f"JSONL input is empty: {path}")
    return rows


def _require_text(row: dict[str, Any], field: str, *, context: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ScoringError(f"{context} requires non-empty {field}")
    return value


def _require_int(row: dict[str, Any], field: str, *, context: str) -> int:
    value = row.get(field)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ScoringError(f"{context} requires integer {field}")
    return value


def _load_sources(
    pack: Path, fixture_names: Iterable[str]
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    paths = sorted(
        {_safe_manifest_path(pack, name) for name in fixture_names},
        key=lambda item: item.as_posix(),
    )
    if not paths:
        raise ScoringError("dataset contains no source fixture paths")
    sources: dict[str, dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    for path in paths:
        hashes[path.relative_to(pack).as_posix()] = _sha256(path)
        for row in _read_jsonl(path):
            source_id = _require_text(row, "source_id", context=str(path))
            content = _require_text(row, "content", context=source_id)
            expected_hash = _require_text(row, "content_hash", context=source_id)
            actual_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            if expected_hash != actual_hash:
                raise ScoringError(f"source content hash mismatch: {source_id}")
            if source_id in sources:
                raise ScoringError(f"duplicate source_id: {source_id}")
            sources[source_id] = row
    return sources, hashes


def _load_gold(path: Path) -> dict[str, dict[str, Any]]:
    gold: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        case_id = _require_text(row, "case_id", context=str(path))
        claim_id = row.get("claim_id")
        if not isinstance(claim_id, (str, int)) or isinstance(claim_id, bool):
            raise ScoringError(f"{case_id} requires string/integer claim_id")
        raw_label = row.get("official_label", row.get("label"))
        if not isinstance(raw_label, str) or not raw_label.strip():
            raise ScoringError(f"{case_id} requires official_label or label")
        normalized_label = raw_label.strip()
        label = RELATION_TO_LABEL.get(normalized_label, normalized_label.upper())
        if label not in LABELS:
            raise ScoringError(f"unsupported official label for {case_id}: {label}")
        spans_raw = row.get("sentence_spans")
        if not isinstance(spans_raw, list) or not spans_raw:
            raise ScoringError(f"{case_id} requires sentence_spans")
        spans: list[dict[str, Any]] = []
        keys: set[str] = set()
        previous_end = -1
        for index, span in enumerate(spans_raw):
            if not isinstance(span, dict):
                raise ScoringError(f"{case_id} sentence span {index} must be an object")
            key_value = span.get("sentence_key", span.get("key"))
            if key_value is None and "doc_id" in span and "sentence_id" in span:
                key_value = f"{span['doc_id']}:{span['sentence_id']}"
            if not isinstance(key_value, (str, int)) or isinstance(key_value, bool):
                raise ScoringError(
                    f"{case_id} sentence span {index} lacks sentence_key"
                )
            key = str(key_value)
            start = _require_int(span, "char_start", context=f"{case_id}:{key}")
            end = _require_int(span, "char_end", context=f"{case_id}:{key}")
            if start < 0 or end <= start or start < previous_end:
                raise ScoringError(
                    f"invalid/overlapping sentence offsets for {case_id}:{key}"
                )
            if key in keys:
                raise ScoringError(f"duplicate sentence key for {case_id}:{key}")
            keys.add(key)
            previous_end = end
            canonical_span = {
                "sentence_key": key,
                "char_start": start,
                "char_end": end,
            }
            if "locator" in span:
                locator = span["locator"]
                if not isinstance(locator, str) or not locator.strip():
                    raise ScoringError(f"invalid locator for {case_id}:{key}")
                canonical_span["locator"] = locator
            if "text_sha256" in span:
                text_hash = span["text_sha256"]
                if not isinstance(text_hash, str) or len(text_hash) != 64:
                    raise ScoringError(f"invalid text_sha256 for {case_id}:{key}")
                canonical_span["text_sha256"] = text_hash
            spans.append(canonical_span)
        raw_sets = row.get("rationale_sets")
        if not isinstance(raw_sets, list):
            raise ScoringError(f"{case_id} rationale_sets must be a list")
        rationale_sets: list[tuple[str, ...]] = []
        for set_index, raw_set in enumerate(raw_sets):
            if isinstance(raw_set, dict):
                official = _require_text(
                    raw_set, "label", context=f"{case_id} rationale set {set_index}"
                ).upper()
                if official not in {"SUPPORT", "CONTRADICT"} or official != label:
                    raise ScoringError(
                        f"rationale label mismatch for {case_id} set {set_index}"
                    )
                raw_keys = raw_set.get("sentence_keys")
            else:
                raw_keys = raw_set
            if not isinstance(raw_keys, list) or not raw_keys:
                raise ScoringError(
                    f"{case_id} rationale set {set_index} must be non-empty"
                )
            rationale = tuple(str(value) for value in raw_keys)
            if len(set(rationale)) != len(rationale) or not set(rationale) <= keys:
                raise ScoringError(
                    f"invalid rationale keys for {case_id} set {set_index}"
                )
            rationale_sets.append(rationale)
        if label in {"SUPPORT", "CONTRADICT"} and not rationale_sets:
            raise ScoringError(f"substantive case lacks rationale sets: {case_id}")
        if label == "NOINFO" and rationale_sets:
            raise ScoringError(f"NOINFO case cannot contain rationale sets: {case_id}")
        if case_id in gold:
            raise ScoringError(f"duplicate gold case_id: {case_id}")
        gold[case_id] = {
            "case_id": case_id,
            "claim_id": str(claim_id),
            "official_label": label,
            "sentence_spans": spans,
            "rationale_sets": rationale_sets,
        }
    return gold


def _validate_sentence_index(
    pack: Path,
    pack_manifest: dict[str, Any],
    loaded: LoadedDataset,
    gold: dict[str, dict[str, Any]],
) -> Path:
    """Cross-check the label-free prediction index against authoritative gold."""

    mapping = pack_manifest.get("mapping")
    if not isinstance(mapping, dict) or (
        mapping.get("prediction_sentence_index") != SENTENCE_INDEX_RELATIVE
    ):
        raise ScoringError("v2 pack does not bind the prediction sentence index")
    index_path = _safe_manifest_path(pack, SENTENCE_INDEX_RELATIVE)
    rows = _read_jsonl(index_path)
    cases_by_id = {case.case_id: case for case in loaded.cases}
    by_case: dict[str, dict[str, Any]] = {}
    for row in rows:
        if set(row) != {"case_id", "source_id", "documents", "sentence_spans"}:
            raise ScoringError("sentence index contains non-structural fields")
        case_id = _require_text(row, "case_id", context="sentence index")
        if case_id not in cases_by_id or case_id in by_case:
            raise ScoringError("sentence index contains an invalid case identity")
        case = cases_by_id[case_id]
        source_id = _require_text(row, "source_id", context=case_id)
        if source_id != case.source_id:
            raise ScoringError(f"sentence index source binding mismatch: {case_id}")
        source = loaded.sources[source_id]

        raw_spans = row.get("sentence_spans")
        if not isinstance(raw_spans, list) or not raw_spans:
            raise ScoringError(f"sentence index requires sentence spans: {case_id}")
        canonical_spans: list[dict[str, Any]] = []
        identities: list[tuple[int, int]] = []
        for index, raw_span in enumerate(raw_spans):
            if not isinstance(raw_span, dict) or set(raw_span) != {
                "doc_id",
                "sentence_id",
                "char_start",
                "char_end",
                "locator",
                "text_sha256",
            }:
                raise ScoringError(f"sentence index span schema mismatch: {case_id}")
            doc_id = _require_int(raw_span, "doc_id", context=f"{case_id}:{index}")
            sentence_id = _require_int(
                raw_span, "sentence_id", context=f"{case_id}:{index}"
            )
            start = _require_int(raw_span, "char_start", context=f"{case_id}:{index}")
            end = _require_int(raw_span, "char_end", context=f"{case_id}:{index}")
            locator = _require_text(raw_span, "locator", context=f"{case_id}:{index}")
            text_hash = _require_text(
                raw_span, "text_sha256", context=f"{case_id}:{index}"
            )
            identity = (doc_id, sentence_id)
            expected_locator = f"scifact://document/{doc_id}/sentence/{sentence_id}"
            if (
                identity in identities
                or start < 0
                or end <= start
                or end > len(source.content)
                or locator != expected_locator
                or not SHA256_RE.fullmatch(text_hash)
                or hashlib.sha256(source.content[start:end].encode("utf-8")).hexdigest()
                != text_hash
            ):
                raise ScoringError(f"invalid sentence index span: {case_id}:{index}")
            identities.append(identity)
            canonical_spans.append(
                {
                    "sentence_key": f"{doc_id}:{sentence_id}",
                    "char_start": start,
                    "char_end": end,
                    "locator": locator,
                    "text_sha256": text_hash,
                }
            )
        if canonical_spans != gold[case_id]["sentence_spans"]:
            raise ScoringError(
                f"sentence index and gold sidecar spans disagree: {case_id}"
            )

        documents = row.get("documents")
        if not isinstance(documents, list) or not documents:
            raise ScoringError(f"sentence index requires documents: {case_id}")
        expected_doc_ids = list(dict.fromkeys(doc_id for doc_id, _ in identities))
        observed_doc_ids: list[int] = []
        for document in documents:
            if not isinstance(document, dict) or set(document) != {
                "doc_id",
                "title",
                "char_start",
                "char_end",
                "sentence_keys",
            }:
                raise ScoringError(
                    f"sentence index document schema mismatch: {case_id}"
                )
            doc_id = _require_int(document, "doc_id", context=case_id)
            positions = [
                item_index
                for item_index, identity in enumerate(identities)
                if identity[0] == doc_id
            ]
            expected_keys = [
                f"{identities[item_index][0]}:{identities[item_index][1]}"
                for item_index in positions
            ]
            if (
                not positions
                or doc_id in observed_doc_ids
                or not _require_text(document, "title", context=case_id)
                or document.get("sentence_keys") != expected_keys
                or _require_int(document, "char_start", context=case_id)
                != canonical_spans[positions[0]]["char_start"]
                or _require_int(document, "char_end", context=case_id)
                != canonical_spans[positions[-1]]["char_end"]
            ):
                raise ScoringError(f"invalid sentence index document: {case_id}")
            observed_doc_ids.append(doc_id)
        if observed_doc_ids != expected_doc_ids:
            raise ScoringError(f"sentence index document order mismatch: {case_id}")
        by_case[case_id] = row
    if set(by_case) != set(cases_by_id) or set(by_case) != set(gold):
        raise ScoringError("sentence index, cases, and gold sidecar IDs differ")
    return index_path


def _expected_run_input_hashes(
    pack: Path, loaded: LoadedDataset, sentence_index_path: Path
) -> dict[str, str]:
    cases_path = pack / "cases.jsonl"
    frozen_path = cases_path.with_name(f"{cases_path.stem}.frozen_hashes.json")
    if not frozen_path.is_file():
        raise ScoringError("reportable v2 pack requires frozen case hashes")
    paths = {pack / "manifest.json", cases_path, frozen_path, sentence_index_path}
    for fixture in {case.source_fixture for case in loaded.cases}:
        paths.add(_safe_manifest_path(pack, fixture))
    return {
        path.relative_to(pack).as_posix(): _sha256(path)
        for path in sorted(paths, key=lambda item: item.as_posix())
    }


def _evidence_identity(
    value: Any,
    *,
    expected_source_id: str,
    context: str,
    require_score: bool,
) -> tuple[int, str, str, int, int, str, float | None]:
    if not isinstance(value, dict):
        raise ScoringError(f"{context} item must be an object")
    rank = _require_int(value, "rank", context=context)
    source_id = _require_text(value, "source_id", context=context)
    text = _require_text(value, "text", context=context)
    locator = _require_text(value, "locator", context=context)
    start = _require_int(value, "char_start", context=context)
    end = _require_int(value, "char_end", context=context)
    if source_id != expected_source_id or not 1 <= rank <= 5:
        raise ScoringError(f"{context} has an invalid source identity or rank")
    score: float | None = None
    if require_score:
        raw_score = value.get("score")
        if (
            not isinstance(raw_score, (int, float))
            or isinstance(raw_score, bool)
            or not math.isfinite(float(raw_score))
            or float(raw_score) < 0.0
        ):
            raise ScoringError(f"{context} requires a finite non-negative score")
        score = float(raw_score)
    return rank, source_id, locator, start, end, text, score


def _validate_reportable_row_evidence(
    row: dict[str, Any], *, expected_source_id: str, predicted_relation: str
) -> tuple[tuple[int, str, str, int, int, str, float | None], ...]:
    retrieved = row.get("retrieved_evidence")
    predicted = row.get("predicted_evidence")
    context = f"{row.get('baseline')}/{row.get('case_id')}"
    if not isinstance(retrieved, list) or not isinstance(predicted, list):
        raise ScoringError(
            f"reportable row requires structured evidence lists: {context}"
        )
    if len(retrieved) > 5 or len(predicted) > 5:
        raise ScoringError(
            f"reportable row exceeds top-five evidence budget: {context}"
        )
    retrieved_identities = tuple(
        _evidence_identity(
            value,
            expected_source_id=expected_source_id,
            context=f"{context}.retrieved[{index}]",
            require_score=True,
        )
        for index, value in enumerate(retrieved)
    )
    if [identity[0] for identity in retrieved_identities] != list(
        range(1, len(retrieved_identities) + 1)
    ):
        raise ScoringError(f"retrieved evidence ranks are not contiguous: {context}")
    if len(set(retrieved_identities)) != len(retrieved_identities):
        raise ScoringError(f"retrieved evidence contains duplicates: {context}")

    predicted_identities = tuple(
        _evidence_identity(
            value,
            expected_source_id=expected_source_id,
            context=f"{context}.predicted[{index}]",
            require_score=False,
        )
        for index, value in enumerate(predicted)
    )
    predicted_core = {identity[:-1] for identity in predicted_identities}
    retrieved_core = {identity[:-1] for identity in retrieved_identities}
    if len(predicted_core) != len(predicted_identities):
        raise ScoringError(f"predicted evidence contains duplicates: {context}")
    if not predicted_core <= retrieved_core:
        raise ScoringError(f"predicted evidence was not retrieved: {context}")
    substantive = predicted_relation in {"entailed", "contradicted"}
    if substantive != bool(predicted_identities):
        raise ScoringError(
            f"predicted relation/evidence cardinality is inconsistent: {context}"
        )
    return retrieved_identities


def _validate_reportable_run(
    pack: Path,
    pack_manifest: dict[str, Any],
    run_manifest: dict[str, Any],
    rows: list[dict[str, Any]],
    loaded: LoadedDataset,
    sentence_index_path: Path,
) -> None:
    if pack_manifest.get("schema_version") != REPORTABLE_PACK_SCHEMA_VERSION:
        raise ScoringError("reportable scoring requires a v2 SciFact adapter pack")
    if run_manifest.get("runner_version") != REPORTABLE_RUNNER_VERSION:
        raise ScoringError("reportable scoring requires the pinned v2 runner")
    if run_manifest.get("task") != REPORTABLE_TASK:
        raise ScoringError("reportable run task mismatch")
    if run_manifest.get("baselines") != [BASELINE_NAME, CANDIDATE_NAME]:
        raise ScoringError("reportable run requires the exact paired baselines")
    if run_manifest.get("task_profile") != REPORTABLE_TASK_PROFILE:
        raise ScoringError("reportable run task profile mismatch")
    evaluation_status = run_manifest.get("evaluation_status")
    if not isinstance(evaluation_status, dict) or (
        evaluation_status.get("reportable") is not True
        or evaluation_status.get("mode") != REPORTABLE_EVALUATION_MODE
    ):
        raise ScoringError("run is not marked as reportable local Transformers NLI")

    inputs = run_manifest.get("inputs")
    expected_inputs = _expected_run_input_hashes(pack, loaded, sentence_index_path)
    if (
        not isinstance(inputs, dict)
        or inputs.get("pack_schema_version") != REPORTABLE_PACK_SCHEMA_VERSION
        or inputs.get("sha256") != expected_inputs
        or inputs.get("loaded_dataset_raw_sha256") != loaded.raw_hash
        or inputs.get("loaded_split_sha256") != loaded.split_hash
    ):
        raise ScoringError("run manifest is not bound to the scored v2 pack")

    local_nli = run_manifest.get("local_nli")
    if not isinstance(local_nli, dict):
        raise ScoringError("reportable run requires local NLI provenance")
    model_id = local_nli.get("upstream_model_id")
    revision = local_nli.get("revision")
    weights_hash = local_nli.get("weights_sha256")
    file_hashes = local_nli.get("model_file_sha256")
    batch_size = local_nli.get("batch_size")
    if (
        local_nli.get("offline_only") is not True
        or local_nli.get("absolute_local_path_recorded") is not False
        or local_nli.get("backend_class")
        != "evidencetrace.eval.scifact_judge.TransformersNliScorer"
        or not isinstance(model_id, str)
        or not model_id.strip()
        or not isinstance(revision, str)
        or not REVISION_RE.fullmatch(revision)
        or not isinstance(weights_hash, str)
        or not SHA256_RE.fullmatch(weights_hash)
        or not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size <= 0
        or not isinstance(file_hashes, dict)
        or not file_hashes
        or not all(
            isinstance(name, str)
            and isinstance(digest, str)
            and SHA256_RE.fullmatch(digest)
            for name, digest in file_hashes.items()
        )
        or not isinstance(local_nli.get("judge_config"), dict)
        or not _require_text(local_nli, "judge_policy_version", context="local_nli")
        or local_nli.get("provenance_binding_scope") != REPORTABLE_PROVENANCE_SCOPE
    ):
        raise ScoringError("invalid local NLI provenance")
    package_versions = local_nli.get("package_versions")
    if not isinstance(package_versions, dict) or any(
        not isinstance(package_versions.get(name), str)
        or not package_versions[name].strip()
        for name in ("torch", "transformers", "tokenizers")
    ):
        raise ScoringError("local NLI package provenance is incomplete")
    binding = {
        "upstream_model_id": model_id,
        "revision": revision,
        "weights_sha256": weights_hash,
    }
    if local_nli.get("identity_binding_sha256") != _canonical_json_sha256(binding):
        raise ScoringError("local NLI identity binding hash mismatch")
    weight_hashes = {
        name: digest
        for name, digest in file_hashes.items()
        if WEIGHT_FILE_RE.fullmatch(PurePosixPath(name).name)
    }
    if not weight_hashes:
        raise ScoringError("local NLI provenance contains no supported weights")
    observed_weights_hash = (
        next(iter(weight_hashes.values()))
        if len(weight_hashes) == 1
        else _canonical_json_sha256(weight_hashes)
    )
    if observed_weights_hash != weights_hash:
        raise ScoringError("local NLI file hashes do not bind the weights hash")

    hf_metadata = local_nli.get("hf_hub_download_metadata")
    hf_metadata_hashes = local_nli.get("hf_hub_metadata_sha256")
    if (
        not isinstance(hf_metadata, dict)
        or set(hf_metadata) != set(file_hashes)
        or not isinstance(hf_metadata_hashes, dict)
    ):
        raise ScoringError("local NLI Hugging Face metadata coverage is incomplete")
    expected_metadata_paths: set[str] = set()
    for filename, record in hf_metadata.items():
        if not isinstance(record, dict) or set(record) != {
            "revision",
            "etag",
            "downloaded_at",
            "metadata_path",
        }:
            raise ScoringError("invalid Hugging Face download metadata record")
        expected_metadata_path = (
            PurePosixPath(".cache")
            / "huggingface"
            / "download"
            / f"{filename}.metadata"
        ).as_posix()
        downloaded_at = record.get("downloaded_at")
        try:
            timestamp = float(downloaded_at)
        except (TypeError, ValueError):
            raise ScoringError("invalid Hugging Face metadata timestamp") from None
        if (
            record.get("revision") != revision
            or not math.isfinite(timestamp)
            or record.get("metadata_path") != expected_metadata_path
            or not isinstance(record.get("etag"), str)
            or not record["etag"].strip()
        ):
            raise ScoringError("Hugging Face metadata provenance mismatch")
        if filename in weight_hashes and record["etag"] != weight_hashes[filename]:
            raise ScoringError("Hugging Face weight etag does not match local hash")
        expected_metadata_paths.add(expected_metadata_path)
    if set(hf_metadata_hashes) != expected_metadata_paths or not all(
        isinstance(digest, str) and SHA256_RE.fullmatch(digest)
        for digest in hf_metadata_hashes.values()
    ):
        raise ScoringError("invalid Hugging Face metadata file hashes")

    length_telemetry = local_nli.get("input_length_telemetry")
    if not isinstance(length_telemetry, dict) or set(length_telemetry) != {
        "available",
        "max_input_tokens",
        "max_observed_input_tokens",
        "rejected_overlength_pair_count",
    }:
        raise ScoringError("reportable run lacks input-length telemetry")
    max_input_tokens = length_telemetry.get("max_input_tokens")
    max_observed_tokens = length_telemetry.get("max_observed_input_tokens")
    rejected_pairs = length_telemetry.get("rejected_overlength_pair_count")
    if (
        length_telemetry.get("available") is not True
        or not isinstance(max_input_tokens, int)
        or isinstance(max_input_tokens, bool)
        or max_input_tokens <= 0
        or not isinstance(max_observed_tokens, int)
        or isinstance(max_observed_tokens, bool)
        or max_observed_tokens < 0
        or max_observed_tokens > max_input_tokens
        or rejected_pairs != 0
    ):
        raise ScoringError("invalid reportable input-length telemetry")

    cases_by_id = {case.case_id: case for case in loaded.cases}
    telemetry = Counter({field: 0 for field in TELEMETRY_FIELDS})
    for row in rows:
        case_id = _require_text(row, "case_id", context="reportable row")
        baseline = _require_text(row, "baseline", context=case_id)
        relation = _require_text(row, "predicted_relation", context=case_id)
        if case_id not in cases_by_id:
            raise ScoringError(f"reportable row references unknown case: {case_id}")
        source_id = cases_by_id[case_id].source_id
        if row.get("source_id") != source_id:
            raise ScoringError(f"reportable row source binding mismatch: {case_id}")
        if row.get("evaluation_status") != evaluation_status:
            raise ScoringError(f"row evaluation status mismatch: {baseline}/{case_id}")
        retrieved = _validate_reportable_row_evidence(
            row, expected_source_id=source_id, predicted_relation=relation
        )
        for field in TELEMETRY_FIELDS:
            value = _require_int(row, field, context=f"{baseline}/{case_id}")
            if value < 0:
                raise ScoringError(
                    f"negative telemetry value: {baseline}/{case_id}/{field}"
                )
            telemetry[field] += value
        model_calls = _require_int(row, "model_calls", context=f"{baseline}/{case_id}")
        if (
            model_calls != row["local_nli_batch_count"]
            or row.get("model_call_semantics") != "local_nli_forward_batch"
            or row["external_provider_calls"] != 0
        ):
            raise ScoringError(
                f"row model-call accounting mismatch: {baseline}/{case_id}"
            )
        if baseline == BASELINE_NAME:
            if (
                any(row[field] != 0 for field in TELEMETRY_FIELDS[1:])
                or model_calls != 0
                or row.get("model_provider") is not None
                or row.get("model_id") is not None
                or row.get("model_revision") is not None
            ):
                raise ScoringError(f"lexical row contains model activity: {case_id}")
            continue
        if baseline != CANDIDATE_NAME:
            raise ScoringError(f"unexpected reportable baseline: {baseline}")
        expected_pairs = (1 << len(retrieved)) - 1 if retrieved else 0
        expected_batches = math.ceil(expected_pairs / batch_size)
        if (
            row.get("model_provider") != REPORTABLE_ROW_MODEL_PROVIDER
            or row.get("model_id") != model_id
            or row.get("model_revision") != revision
            or row["local_nli_score_many_calls"] != int(bool(retrieved))
            or row["local_nli_pair_count"] != expected_pairs
            or row["local_nli_subset_count"] != expected_pairs
            or row["local_nli_batch_count"] != expected_batches
        ):
            raise ScoringError(f"candidate NLI identity/telemetry mismatch: {case_id}")
    counts = run_manifest.get("counts")
    if not isinstance(counts, dict) or any(
        counts.get(field) != telemetry[field] for field in TELEMETRY_FIELDS
    ):
        raise ScoringError("run manifest telemetry totals disagree with result rows")


def _occurrences(content: str, text: str) -> list[int]:
    starts: list[int] = []
    offset = 0
    while text and (found := content.find(text, offset)) >= 0:
        starts.append(found)
        offset = found + 1
    return starts


def _map_exact_text(
    content: str, text: Any, sentence_spans: list[dict[str, Any]]
) -> tuple[set[str], set[str], str | None]:
    if not isinstance(text, str) or not text:
        return set(), set(), "missing_text"
    starts = _occurrences(content, text)
    if not starts:
        return set(), set(), "not_in_source"
    if len(starts) != 1:
        return set(), set(), "multiple_source_occurrences"
    start = starts[0]
    end = start + len(text)
    overlapped = {
        span["sentence_key"]
        for span in sentence_spans
        if start < span["char_end"] and end > span["char_start"]
    }
    if not overlapped:
        return set(), set(), "no_overlapping_sentence"
    fully_covered = {
        span["sentence_key"]
        for span in sentence_spans
        if start <= span["char_start"] and end >= span["char_end"]
    }
    return fully_covered, overlapped, None


def _map_structured_sentence(
    content: str,
    value: Any,
    sentence_spans: list[dict[str, Any]],
    *,
    expected_source_id: str | None = None,
) -> tuple[set[str], set[str], str | None]:
    """Map one prediction-owned exact sentence without text-occurrence ambiguity."""

    if not isinstance(value, dict):
        return set(), set(), "structured_item_not_object"
    if expected_source_id is not None and value.get("source_id") != expected_source_id:
        return set(), set(), "source_id_mismatch"
    text = value.get("text")
    if not isinstance(text, str) or not text:
        return set(), set(), "missing_text"
    start = value.get("char_start")
    end = value.get("char_end")
    if (
        not isinstance(start, int)
        or isinstance(start, bool)
        or not isinstance(end, int)
        or isinstance(end, bool)
        or start < 0
        or end <= start
        or end > len(content)
    ):
        return set(), set(), "invalid_offsets"
    if end - start != len(text) or content[start:end] != text:
        return set(), set(), "offset_text_mismatch"
    exact = [
        span
        for span in sentence_spans
        if span["char_start"] == start and span["char_end"] == end
    ]
    if len(exact) != 1:
        return set(), set(), "not_exact_indexed_sentence"
    locator = value.get("locator")
    expected_locator = exact[0].get("locator")
    if expected_locator is not None and (
        not isinstance(locator, str) or locator != expected_locator
    ):
        return set(), set(), "locator_mismatch"
    key = exact[0]["sentence_key"]
    return {key}, {key}, None


def _map_prediction_evidence(
    row: dict[str, Any],
    content: str,
    sentence_spans: list[dict[str, Any]],
    *,
    expected_source_id: str | None = None,
) -> tuple[set[str], set[str], str | None, bool]:
    """Prefer structured plural evidence and retain legacy single-span support."""

    if "predicted_evidence" not in row:
        raw_span = row.get("predicted_evidence_span")
        fully_covered, overlapped, invalid = _map_exact_text(
            content, raw_span, sentence_spans
        )
        return fully_covered, overlapped, invalid, bool(raw_span)
    values = row["predicted_evidence"]
    if not isinstance(values, list):
        raise ScoringError("predicted_evidence must be a list")
    if len(values) > 5:
        raise ScoringError("predicted_evidence exceeds the top-five budget")
    fully_covered: set[str] = set()
    overlapped: set[str] = set()
    seen_offsets: set[tuple[int, int]] = set()
    for index, value in enumerate(values):
        mapped, diagnostic, invalid = _map_structured_sentence(
            content,
            value,
            sentence_spans,
            expected_source_id=expected_source_id,
        )
        if invalid is not None:
            return set(), set(), f"item_{index}:{invalid}", bool(values)
        assert isinstance(value, dict)
        offset = (value["char_start"], value["char_end"])
        if offset in seen_offsets:
            return set(), set(), f"item_{index}:duplicate_offsets", True
        seen_offsets.add(offset)
        fully_covered.update(mapped)
        overlapped.update(diagnostic)
    return fully_covered, overlapped, None if values else "missing_text", bool(values)


def _set_f1(predicted: set[str], gold: Iterable[str]) -> float:
    target = set(gold)
    if not predicted and not target:
        return 1.0
    if not predicted or not target:
        return 0.0
    overlap = len(predicted & target)
    return 2.0 * overlap / (len(predicted) + len(target))


def _evidence_result(
    row: dict[str, Any], source: dict[str, Any], gold: dict[str, Any], pred_label: str
) -> dict[str, Any]:
    label = gold["official_label"]
    fully_covered, overlapped, invalid, has_evidence = _map_prediction_evidence(
        row,
        source["content"],
        gold["sentence_spans"],
        expected_source_id=source["source_id"],
    )
    if label == "NOINFO":
        return {
            "eligible": False,
            "abstained": not has_evidence,
            "invalid_reason": None,
            "fully_covered_sentence_keys": [],
            "overlapped_sentence_keys_diagnostic": [],
            "best_rationale_index": None,
            "best_sentence_set_f1": None,
            "complete_rationale_coverage": None,
            "complete_rationale_match": None,
            "joint_verdict_complete": None,
        }
    # A NOINFO prediction correctly carries no evidence.  It is wrong against a
    # substantive gold label, but it is an abstention rather than malformed
    # evidence.  Sentence-set F1 and the joint metric still score it as zero.
    if not has_evidence:
        invalid = None
    scores = [_set_f1(fully_covered, rationale) for rationale in gold["rationale_sets"]]
    best_index = max(range(len(scores)), key=lambda index: (scores[index], -index))
    complete = invalid is None and any(
        set(rationale) <= fully_covered for rationale in gold["rationale_sets"]
    )
    return {
        "eligible": True,
        "abstained": not has_evidence,
        "invalid_reason": invalid,
        "fully_covered_sentence_keys": sorted(fully_covered),
        "overlapped_sentence_keys_diagnostic": sorted(overlapped),
        "best_rationale_index": best_index,
        "best_sentence_set_f1": scores[best_index],
        "complete_rationale_coverage": complete,
        # Deprecated compatibility alias. Reports and new consumers use coverage.
        "complete_rationale_match": complete,
        "joint_verdict_complete": pred_label == label and complete,
    }


def _retrieval_result(
    row: dict[str, Any], source: dict[str, Any], gold: dict[str, Any]
) -> dict[str, Any]:
    if gold["official_label"] == "NOINFO":
        return {"eligible": False}
    mapped: list[dict[str, Any]] = []
    structured = row.get("retrieved_evidence")
    if structured is not None:
        if not isinstance(structured, list):
            raise ScoringError(
                f"retrieved_evidence must be a list for {gold['case_id']}"
            )
        if len(structured) > 5:
            raise ScoringError(
                f"retrieved_evidence exceeds top-five for {gold['case_id']}"
            )
        values: list[tuple[Any, int]] = []
        for index, value in enumerate(structured, 1):
            if not isinstance(value, dict) or value.get("rank") != index:
                raise ScoringError(
                    f"retrieved_evidence ranks must be contiguous for {gold['case_id']}"
                )
            values.append((value, index))
    else:
        texts = row.get("retrieved_texts", [])
        if not isinstance(texts, list):
            raise ScoringError(f"retrieved_texts must be a list for {gold['case_id']}")
        values = [(text, index) for index, text in enumerate(texts, 1)]
    for value, rank in values:
        if structured is not None:
            fully_covered, overlapped, invalid = _map_structured_sentence(
                source["content"],
                value,
                gold["sentence_spans"],
                expected_source_id=source["source_id"],
            )
        else:
            fully_covered, overlapped, invalid = _map_exact_text(
                source["content"], value, gold["sentence_spans"]
            )
        mapped.append(
            {
                "rank": rank,
                "fully_covered_sentence_keys": sorted(fully_covered),
                "overlapped_sentence_keys_diagnostic": sorted(overlapped),
                "invalid_reason": invalid,
            }
        )
    gold_union = set().union(*(set(item) for item in gold["rationale_sets"]))
    first_overlap_rank = next(
        (
            item["rank"]
            for item in mapped
            if set(item["overlapped_sentence_keys_diagnostic"]) & gold_union
        ),
        None,
    )
    first_full_sentence_rank = next(
        (
            item["rank"]
            for item in mapped
            if set(item["fully_covered_sentence_keys"]) & gold_union
        ),
        None,
    )
    output: dict[str, Any] = {
        "eligible": True,
        "mapped_items": mapped,
        "first_overlap_rank": first_overlap_rank,
        "overlap_reciprocal_rank": (
            0.0 if first_overlap_rank is None else 1.0 / first_overlap_rank
        ),
        "first_full_sentence_rank": first_full_sentence_rank,
        "full_sentence_reciprocal_rank": (
            0.0 if first_full_sentence_rank is None else 1.0 / first_full_sentence_rank
        ),
    }
    for k in RETRIEVAL_K:
        overlapped_selected = set().union(
            *(set(item["overlapped_sentence_keys_diagnostic"]) for item in mapped[:k])
        )
        fully_covered_selected = set().union(
            *(set(item["fully_covered_sentence_keys"]) for item in mapped[:k])
        )
        output[f"overlap_hit_at_{k}"] = bool(overlapped_selected & gold_union)
        output[f"full_sentence_hit_at_{k}"] = any(
            set(item["fully_covered_sentence_keys"]) & gold_union for item in mapped[:k]
        )
        output[f"complete_rationale_full_coverage_at_{k}"] = any(
            set(rationale) <= fully_covered_selected
            for rationale in gold["rationale_sets"]
        )
    return output


def _retrieval_comparison_identity(row: dict[str, Any]) -> tuple[str, tuple[Any, ...]]:
    """Return the representation actually consumed by retrieval scoring."""

    structured = row.get("retrieved_evidence")
    if isinstance(structured, list):
        return (
            "structured",
            tuple(
                (
                    value.get("rank"),
                    value.get("source_id"),
                    value.get("locator"),
                    value.get("char_start"),
                    value.get("char_end"),
                    value.get("text"),
                    value.get("score"),
                )
                for value in structured
            ),
        )
    texts = row.get("retrieved_texts", [])
    if not isinstance(texts, list):
        raise ScoringError("retrieved_texts must be a list")
    return "legacy", tuple(texts)


def _safe_divide(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def _verdict_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    confusion = {gold: {pred: 0 for pred in (*LABELS, OFF_TAXONOMY)} for gold in LABELS}
    correct = 0
    off = 0
    for row in rows:
        gold = row["official_label"]
        pred = row["predicted_label"]
        confusion[gold][pred] += 1
        correct += int(gold == pred)
        off += int(pred == OFF_TAXONOMY)
    per_label: dict[str, dict[str, float | int]] = {}
    for label in LABELS:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in LABELS if other != label)
        fn = sum(
            confusion[label][other]
            for other in (*LABELS, OFF_TAXONOMY)
            if other != label
        )
        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        per_label[label] = {
            "support": sum(confusion[label].values()),
            "precision": precision,
            "recall": recall,
            "f1": _safe_divide(2 * precision * recall, precision + recall),
        }
    return {
        "case_count": len(rows),
        "accuracy": _safe_divide(correct, len(rows)),
        "fixed_3way_macro_f1": sum(item["f1"] for item in per_label.values()) / 3,
        "per_label": per_label,
        "off_taxonomy_count": off,
        "off_taxonomy_rate": _safe_divide(off, len(rows)),
        "confusion": confusion,
    }


def _mean(values: list[float]) -> float:
    return _safe_divide(sum(values), len(values))


def _baseline_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    substantive = [row for row in rows if row["evidence"]["eligible"]]
    noinfo = [row for row in rows if row["official_label"] == "NOINFO"]
    retrieval = [row for row in rows if row["retrieval"]["eligible"]]
    evidence_metrics = {
        "case_count": len(substantive),
        "best_sentence_set_f1": _mean(
            [row["evidence"]["best_sentence_set_f1"] for row in substantive]
        ),
        "complete_rationale_coverage_rate": _mean(
            [
                float(row["evidence"]["complete_rationale_coverage"])
                for row in substantive
            ]
        ),
        "joint_verdict_complete_rate": _mean(
            [float(row["evidence"]["joint_verdict_complete"]) for row in substantive]
        ),
        "invalid_count": sum(
            row["evidence"]["invalid_reason"] is not None for row in substantive
        ),
        "substantive_abstention_count": sum(
            row["evidence"]["abstained"] for row in substantive
        ),
    }
    evidence_metrics["invalid_rate"] = _safe_divide(
        evidence_metrics["invalid_count"], len(substantive)
    )
    evidence_metrics["substantive_abstention_rate"] = _safe_divide(
        evidence_metrics["substantive_abstention_count"], len(substantive)
    )
    evidence_metrics["complete_rationale_match_rate"] = evidence_metrics[
        "complete_rationale_coverage_rate"
    ]
    evidence_metrics["complete_rationale_match_rate_role"] = (
        "deprecated_alias_for_complete_rationale_coverage_rate"
    )
    abstentions = sum(row["evidence"]["abstained"] for row in noinfo)
    evidence_metrics["noinfo_case_count"] = len(noinfo)
    evidence_metrics["noinfo_evidence_abstention_rate"] = _safe_divide(
        abstentions, len(noinfo)
    )
    retrieval_metrics: dict[str, Any] = {
        "case_count": len(retrieval),
        "invalid_mapped_item_count": sum(
            item["invalid_reason"] is not None
            for row in retrieval
            for item in row["retrieval"]["mapped_items"]
        ),
        "ambiguous_mapped_item_count": sum(
            item["invalid_reason"] == "multiple_source_occurrences"
            for row in retrieval
            for item in row["retrieval"]["mapped_items"]
        ),
        "overlap_mrr": _mean(
            [row["retrieval"]["overlap_reciprocal_rank"] for row in retrieval]
        ),
        "full_sentence_mrr": _mean(
            [row["retrieval"]["full_sentence_reciprocal_rank"] for row in retrieval]
        ),
    }
    for k in RETRIEVAL_K:
        retrieval_metrics[f"overlap_hit_at_{k}"] = _mean(
            [float(row["retrieval"][f"overlap_hit_at_{k}"]) for row in retrieval]
        )
        retrieval_metrics[f"full_sentence_hit_at_{k}"] = _mean(
            [float(row["retrieval"][f"full_sentence_hit_at_{k}"]) for row in retrieval]
        )
        retrieval_metrics[f"complete_rationale_full_coverage_at_{k}"] = _mean(
            [
                float(row["retrieval"][f"complete_rationale_full_coverage_at_{k}"])
                for row in retrieval
            ]
        )
    return {
        "verdict": _verdict_metrics(rows),
        "evidence": evidence_metrics,
        "retrieval": retrieval_metrics,
    }


def _percentile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        raise ScoringError("cannot compute percentile of an empty sample")
    position = (len(sorted_values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def _metric_triplet(rows: list[dict[str, Any]]) -> tuple[float, float, float]:
    metrics = _baseline_metrics(rows)
    return (
        metrics["verdict"]["fixed_3way_macro_f1"],
        metrics["evidence"]["best_sentence_set_f1"],
        metrics["retrieval"]["full_sentence_hit_at_5"],
    )


def _paired_bootstrap(
    by_baseline: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    if BASELINE_NAME not in by_baseline or CANDIDATE_NAME not in by_baseline:
        return {
            "available": False,
            "reason": f"requires {BASELINE_NAME} and {CANDIDATE_NAME}",
        }
    baseline_by_case = {row["case_id"]: row for row in by_baseline[BASELINE_NAME]}
    candidate_by_case = {row["case_id"]: row for row in by_baseline[CANDIDATE_NAME]}
    if set(baseline_by_case) != set(candidate_by_case):
        raise ScoringError("paired baselines do not contain identical case IDs")
    cases_by_claim: dict[str, list[str]] = defaultdict(list)
    for case_id, row in baseline_by_case.items():
        cases_by_claim[row["claim_id"]].append(case_id)
    claim_ids = sorted(cases_by_claim)
    if not claim_ids:
        raise ScoringError("paired bootstrap requires at least one claim")
    point_base = _metric_triplet(list(baseline_by_case.values()))
    point_candidate = _metric_triplet(list(candidate_by_case.values()))
    names = (
        "fixed_3way_macro_f1",
        "best_sentence_set_f1",
        "full_sentence_hit_at_5",
    )
    samples: dict[str, list[float]] = {name: [] for name in names}
    rng = random.Random(BOOTSTRAP_SEED)
    for _ in range(BOOTSTRAP_SAMPLES):
        chosen = [claim_ids[rng.randrange(len(claim_ids))] for _ in claim_ids]
        baseline_rows: list[dict[str, Any]] = []
        candidate_rows: list[dict[str, Any]] = []
        for claim_id in chosen:
            for case_id in cases_by_claim[claim_id]:
                baseline_rows.append(baseline_by_case[case_id])
                candidate_rows.append(candidate_by_case[case_id])
        before = _metric_triplet(baseline_rows)
        after = _metric_triplet(candidate_rows)
        for index, name in enumerate(names):
            samples[name].append(after[index] - before[index])
    effects: dict[str, Any] = {}
    for index, name in enumerate(names):
        values = sorted(samples[name])
        effects[name] = {
            "baseline": point_base[index],
            "candidate": point_candidate[index],
            "delta": point_candidate[index] - point_base[index],
            "ci95_low": _percentile(values, 0.025),
            "ci95_high": _percentile(values, 0.975),
            "probability_delta_le_zero": _mean([float(value <= 0) for value in values]),
        }
    return {
        "available": True,
        "comparison": f"{CANDIDATE_NAME}-minus-{BASELINE_NAME}",
        "unit": "original_scifact_claim_id_cluster",
        "samples": BOOTSTRAP_SAMPLES,
        "seed": BOOTSTRAP_SEED,
        "claim_count": len(claim_ids),
        "effects": effects,
    }


def _assert_output_available(output: Path) -> None:
    if output.exists():
        if not output.is_dir():
            raise ScoringError(f"output path exists and is not a directory: {output}")
        if any(output.iterdir()):
            raise ScoringError(f"output directory is not empty: {output}")


def _write_text_exclusive(path: Path, value: str) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(value)


def _write_json(path: Path, payload: Any) -> None:
    _write_text_exclusive(
        path,
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    _write_text_exclusive(
        path,
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for row in rows
        ),
    )


def _report(metrics: dict[str, Any]) -> str:
    denominators = metrics["denominators"]
    lines = [
        "# EvidenceTrace SciFact public-dev scoring",
        "",
        f"## Verdict (N={denominators['verdict']}) and evidence/retrieval "
        f"(N={denominators['evidence_and_retrieval']})",
        "",
        "| Baseline | Accuracy | Fixed 3-way macro-F1 | Strict evidence "
        "sentence-set F1 | Complete rationale coverage | Full-sentence Hit@5 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for baseline, payload in metrics["baselines"].items():
        verdict = payload["verdict"]
        if payload.get("verdict_only"):
            lines.append(
                f"| {baseline} | {verdict['accuracy']:.4f} | "
                f"{verdict['fixed_3way_macro_f1']:.4f} | n/a | n/a | n/a |"
            )
            continue
        evidence = payload["evidence"]
        retrieval = payload["retrieval"]
        lines.append(
            f"| {baseline} | {verdict['accuracy']:.4f} | "
            f"{verdict['fixed_3way_macro_f1']:.4f} | "
            f"{evidence['best_sentence_set_f1']:.4f} | "
            f"{evidence['complete_rationale_coverage_rate']:.4f} | "
            f"{retrieval['full_sentence_hit_at_5']:.4f} |"
        )
    lines.extend(["", "## Paired claim-level bootstrap", ""])
    bootstrap = metrics["paired_bootstrap"]
    if bootstrap["available"]:
        lines.extend(
            [
                f"Comparison: `{bootstrap['comparison']}`; "
                f"{bootstrap['samples']} samples; seed {bootstrap['seed']}.",
                "",
                "| Metric | Baseline | Candidate | Delta | 95% percentile CI |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for name, effect in bootstrap["effects"].items():
            lines.append(
                f"| {name} | {effect['baseline']:.4f} | {effect['candidate']:.4f} | "
                f"{effect['delta']:+.4f} | [{effect['ci95_low']:+.4f}, "
                f"{effect['ci95_high']:+.4f}] |"
            )
    else:
        lines.append(f"Unavailable: {bootstrap['reason']}.")
    lines.extend(["", "## Interpretation boundaries", ""])
    lines.extend(f"- {boundary}" for boundary in metrics["boundaries"])
    lines.extend(
        [
            "",
            "When absent, `lexical_full_source` was excluded; when present, it is "
            "a full-source control marked `not_same_context_budget`, not a "
            "like-for-like retrieval comparison.",
            "",
        ]
    )
    return "\n".join(lines)


def score(
    pack_dir: Path | str,
    eval_results: Path | str,
    out_dir: Path | str,
    *,
    allow_nonreportable: bool = False,
) -> Path:
    pack = Path(pack_dir).resolve()
    results_path = Path(eval_results).resolve()
    output = Path(out_dir).resolve()
    _assert_output_available(output)
    manifest, verified_manifest_hashes = _validate_manifest(pack)
    cases_path = pack / "cases.jsonl"
    try:
        loaded: LoadedDataset = load_dataset(cases_path)
    except (OSError, ValueError) as error:
        raise ScoringError(
            f"pack failed canonical dataset validation: {error}"
        ) from error
    gold_path = pack / "gold_rationales.jsonl"
    gold = _load_gold(gold_path)
    sources, source_hashes = _load_sources(
        pack, {case.source_fixture for case in loaded.cases}
    )
    if not set(source_hashes) <= set(verified_manifest_hashes):
        raise ScoringError("manifest omits one or more source fixture hashes")
    unmanifested_sources = set(source_hashes) - set(verified_manifest_hashes)
    if unmanifested_sources:
        names = ", ".join(sorted(unmanifested_sources))
        raise ScoringError(f"source fixtures missing manifest hashes: {names}")
    if set(sources) != set(loaded.sources):
        raise ScoringError(
            "source fixture loader disagrees with canonical dataset loader"
        )
    for source_id, source in sources.items():
        canonical = loaded.sources[source_id]
        if source["content"] != canonical.content or source["content_hash"] != (
            canonical.content_hash
        ):
            raise ScoringError(f"source fixture binding mismatch: {source_id}")
    cases_by_id = {case.case_id: case for case in loaded.cases}
    if set(gold) != set(cases_by_id):
        raise ScoringError(
            "cases.jsonl and rationale sidecar must contain identical IDs"
        )
    sentence_index_path: Path | None = None
    if not allow_nonreportable:
        sentence_index_path = _validate_sentence_index(pack, manifest, loaded, gold)
    generic_rows = _read_jsonl(results_path)
    run_manifest, run_manifest_hash = _validate_run_manifest(results_path, generic_rows)
    if not allow_nonreportable:
        assert sentence_index_path is not None
        _validate_reportable_run(
            pack,
            manifest,
            run_manifest,
            generic_rows,
            loaded,
            sentence_index_path,
        )

    derived: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    source_by_case: dict[str, str] = {}
    for row in generic_rows:
        case_id = _require_text(row, "case_id", context=str(results_path))
        baseline = _require_text(row, "baseline", context=case_id)
        if baseline == "majority_class":
            raise ScoringError("majority_class is a reserved scorer baseline")
        pair = (baseline, case_id)
        if pair in seen:
            raise ScoringError(f"duplicate baseline/case result: {baseline}/{case_id}")
        seen.add(pair)
        if case_id not in gold:
            raise ScoringError(f"prediction has no rationale sidecar row: {case_id}")
        source_id = _require_text(row, "source_id", context=case_id)
        if source_id not in sources:
            raise ScoringError(f"prediction references unknown source: {source_id}")
        if cases_by_id[case_id].source_id != source_id:
            raise ScoringError(
                f"result source_id disagrees with cases.jsonl for {case_id}"
            )
        prior_source_id = source_by_case.setdefault(case_id, source_id)
        if prior_source_id != source_id:
            raise ScoringError(f"baselines disagree on source_id for {case_id}")
        predicted_relation = _require_text(row, "predicted_relation", context=case_id)
        predicted_label = RELATION_TO_LABEL.get(predicted_relation, OFF_TAXONOMY)
        gold_row = gold[case_id]
        source_content = sources[source_id]["content"]
        content_length = len(source_content)
        if any(
            span["char_end"] > content_length for span in gold_row["sentence_spans"]
        ):
            raise ScoringError(f"sentence offsets exceed source content for {case_id}")
        for sentence_span in gold_row["sentence_spans"]:
            expected_text_hash = sentence_span.get("text_sha256")
            if expected_text_hash is not None:
                sentence_text = source_content[
                    sentence_span["char_start"] : sentence_span["char_end"]
                ]
                actual_text_hash = hashlib.sha256(
                    sentence_text.encode("utf-8")
                ).hexdigest()
                if actual_text_hash != expected_text_hash:
                    raise ScoringError(
                        f"sentence text hash mismatch for {case_id}:"
                        f"{sentence_span['sentence_key']}"
                    )
        evidence = _evidence_result(row, sources[source_id], gold_row, predicted_label)
        retrieval = _retrieval_result(row, sources[source_id], gold_row)
        derived.append(
            {
                "case_id": case_id,
                "claim_id": gold_row["claim_id"],
                "source_id": source_id,
                "baseline": baseline,
                "official_label": gold_row["official_label"],
                "predicted_relation": predicted_relation,
                "predicted_label": predicted_label,
                "verdict_correct": predicted_label == gold_row["official_label"],
                "off_taxonomy": predicted_label == OFF_TAXONOMY,
                "evidence": evidence,
                "retrieval": retrieval,
            }
        )

    by_baseline: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in derived:
        by_baseline[row["baseline"]].append(row)
    case_sets = {
        baseline: {row["case_id"] for row in rows}
        for baseline, rows in by_baseline.items()
    }
    if len({frozenset(value) for value in case_sets.values()}) != 1:
        raise ScoringError("all generic baselines must contain the same case IDs")
    expected_cases = next(iter(case_sets.values()))
    if expected_cases != set(gold):
        raise ScoringError(
            "generic results and rationale sidecar must contain identical case IDs"
        )

    same_retriever_verified_case_count = 0
    if BASELINE_NAME in by_baseline and CANDIDATE_NAME in by_baseline:
        base_retrieval = {
            row["case_id"]: row
            for row in generic_rows
            if row["baseline"] == BASELINE_NAME
        }
        candidate_retrieval = {
            row["case_id"]: row
            for row in generic_rows
            if row["baseline"] == CANDIDATE_NAME
        }
        for case_id in sorted(expected_cases):
            base_row = base_retrieval[case_id]
            candidate_row = candidate_retrieval[case_id]
            base_value = _retrieval_comparison_identity(base_row)
            candidate_value = _retrieval_comparison_identity(candidate_row)
            if base_value != candidate_value:
                raise ScoringError(
                    f"paired same-retriever invariant failed for {case_id}"
                )
        same_retriever_verified_case_count = len(expected_cases)

    label_counts = Counter(
        gold[case_id]["official_label"] for case_id in expected_cases
    )
    majority_label = max(
        LABELS, key=lambda label: (label_counts[label], -LABELS.index(label))
    )
    reference_rows = next(iter(by_baseline.values()))
    majority_rows: list[dict[str, Any]] = []
    for row in reference_rows:
        majority_rows.append(
            {
                "case_id": row["case_id"],
                "claim_id": row["claim_id"],
                "source_id": row["source_id"],
                "baseline": "majority_class",
                "official_label": row["official_label"],
                "predicted_relation": None,
                "predicted_label": majority_label,
                "verdict_correct": majority_label == row["official_label"],
                "off_taxonomy": False,
                "evidence": {"eligible": False, "reason": "verdict_only_baseline"},
                "retrieval": {"eligible": False, "reason": "verdict_only_baseline"},
            }
        )

    baseline_metrics = {
        baseline: _baseline_metrics(sorted(rows, key=lambda item: item["case_id"]))
        for baseline, rows in sorted(by_baseline.items())
    }
    baseline_metrics["majority_class"] = {
        "verdict": _verdict_metrics(majority_rows),
        "verdict_only": True,
        "majority_label": majority_label,
        "label_counts": {label: label_counts[label] for label in LABELS},
    }
    if "lexical_full_source" in baseline_metrics:
        baseline_metrics["lexical_full_source"]["context_budget_comparability"] = (
            "not_same_context_budget"
        )

    metrics = {
        "schema_version": SCORER_VERSION,
        "task": REPORTABLE_TASK,
        "evaluation_status": {
            "reportable": not allow_nonreportable,
            "mode": (
                REPORTABLE_EVALUATION_MODE
                if not allow_nonreportable
                else "nonreportable_explicit_opt_in"
            ),
        },
        "authoritative_evidence_gold": "gold_rationales.jsonl",
        "canonical_single_span_metrics_role": "compatibility_only_not_headline",
        "structured_plural_evidence_role": "authoritative_prediction_evidence",
        "inputs": {
            "pack": str(pack),
            "eval_results": str(results_path),
            "manifest_schema_version": manifest.get("schema_version"),
            "run_manifest_validation": (
                "reportable_v2_contract_verified"
                if not allow_nonreportable
                else "nonreportable_basic_integrity_only"
            ),
            "hashes": {
                "eval_results.jsonl": _sha256(results_path),
                "verified_manifest_outputs": verified_manifest_hashes,
                "source_fixtures": source_hashes,
                "run_manifest.json": run_manifest_hash,
            },
        },
        "config": {
            "labels": list(LABELS),
            "relation_to_label": RELATION_TO_LABEL,
            "off_taxonomy_policy": "count_as_wrong_and_report_separately",
            "rationale_semantics": "outer_OR_inner_AND",
            "complete_rationale_metric_semantics": (
                "coverage: one complete gold rationale must be a subset of the "
                "predicted sentence set; extra sentences are penalized separately "
                "by best_sentence_set_f1"
            ),
            "text_mapping": (
                "structured_exact_offsets_and_locator; legacy unique exact text "
                "fallback"
            ),
            "duplicate_text_policy": (
                "structured offsets disambiguate; legacy duplicate text is rejected"
            ),
            "strict_sentence_mapping": "exact indexed sentence offsets and locator",
            "overlap_metrics_role": "diagnostic_only",
            "retrieval_k": list(RETRIEVAL_K),
            "bootstrap_unit": "claim_id",
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "counts": {
            "claim_count": len({item["claim_id"] for item in gold.values()}),
            "case_count": len(gold),
            "source_count": len(sources),
            "generic_baseline_count": len(by_baseline),
            "same_retriever_verified_case_count": same_retriever_verified_case_count,
        },
        "denominators": {
            "verdict": len(gold),
            "evidence_and_retrieval": sum(
                item["official_label"] in {"SUPPORT", "CONTRADICT"}
                for item in gold.values()
            ),
            "noinfo_evidence_abstention": sum(
                item["official_label"] == "NOINFO" for item in gold.values()
            ),
        },
        "baselines": baseline_metrics,
        "paired_bootstrap": _paired_bootstrap(by_baseline),
        "comparability": {
            "lexical_full_source": "not_same_context_budget",
            f"{CANDIDATE_NAME}-vs-{BASELINE_NAME}": (
                "same_LexicalRetriever_chunks_and_top5_by_construction; "
                "retrieval_metrics_identical; local semantic judgment policy only"
            ),
        },
        "run_manifest": {
            "status": "verified",
            "runner_version": run_manifest.get("runner_version"),
        },
        "boundaries": [
            *(
                [
                    "This scoring artifact used the explicit non-reportable opt-in; "
                    "it must not be cited as a formal v2 result."
                ]
                if allow_nonreportable
                else []
            ),
            *BOUNDARIES,
        ],
    }

    output.mkdir(parents=True, exist_ok=True)
    all_results = sorted(
        [*derived, *majority_rows], key=lambda item: (item["baseline"], item["case_id"])
    )
    _write_json(output / "metrics.json", metrics)
    _write_jsonl(output / "results.jsonl", all_results)
    _write_text_exclusive(output / "report.md", _report(metrics))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--eval-results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--allow-nonreportable",
        action="store_true",
        help="Explicitly score legacy/test artifacts and mark output non-reportable.",
    )
    args = parser.parse_args()
    score(
        args.pack,
        args.eval_results,
        args.out,
        allow_nonreportable=args.allow_nonreportable,
    )


if __name__ == "__main__":
    main()
