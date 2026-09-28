"""Validate and summarize a frozen local SciFact agent A/B run.

The runner uses one physical initial generation as the shared draft for three
logical arms.  Consequently this script reports both counterfactual logical-arm
resource totals and non-duplicated physical-run totals.  Logical latency is an
active-path sum of synchronized generation stages, not an independently timed
wall-clock execution of each arm.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evidencetrace.eval.scifact_official import (
    ValidatedSciFact,
    compute_official_pipeline_metrics,
    validate_official_data,
    validate_pipeline_predictions,
)

ARMS = (
    "single_one_pass",
    "single_self_review",
    "judge_challenger",
)
MAIN_BASELINE = "single_self_review"
MAIN_CANDIDATE = "judge_challenger"
CALL_KINDS = (
    "shared_initial_judge",
    "self_review",
    "independent_challenger",
)
PREDICTION_FILES = {
    arm: f"predictions_{arm}.jsonl" for arm in ARMS
}
REQUIRED_RUNNER_OUTPUTS = {
    "contexts.jsonl",
    "cases.jsonl",
    "raw_telemetry.jsonl",
    "journal.jsonl",
    "run_header.json",
    *PREDICTION_FILES.values(),
}
OFFICIAL_GROUPS = (
    "sentence_selection",
    "sentence_label",
    "abstract_label_only",
    "abstract_rationalized",
)
OFFICIAL_STATISTICS = ("precision", "recall", "f1")
OFFICIAL_METRIC_NAMES = tuple(
    f"{group}_{statistic}"
    for group in OFFICIAL_GROUPS
    for statistic in OFFICIAL_STATISTICS
)
BOOTSTRAP_METRICS = (
    "abstract_rationalized_f1",
    "sentence_label_f1",
)
DEFAULT_BOOTSTRAP_SAMPLES = 10_000
DEFAULT_BOOTSTRAP_SEED = 20260904
OFFICIAL_EVALUATOR_COMMIT = "66feffc5b2cc9e28e3ce3b8c9e824c3c642981eb"
DRAFT_UNAVAILABLE = '{"status":"draft_unavailable"}'
INITIAL_DRAFT_PREFILL = '{"analysis":"'
FINAL_EVIDENCE_PREFILL = '{"evidence":{'
JSON_ROOT_STOPPING_POLICY = (
    "during_generation_first_complete_json_root_object_closure"
)
V2_AGENT_PROTOCOL = "scifact-agent-decision-v2"
V2_INITIAL_DRAFT_PREFILL = '{"label":"'
V2_FINAL_EVIDENCE_PREFILL = '{"label":"'
V2_JSON_ROOT_STOPPING_POLICY = (
    "during_generation_first_complete_type_matched_json_root_object_closure"
)
V2_COMMON_PARSER_POLICY = {
    "root_exact_keys": ["label", "citations"],
    "labels": ["SUPPORT", "CONTRADICT", "NEI"],
    "citation_shape": ["retrieval_rank", "sentence_indices"],
    "citation_container": "array",
    "max_citations": 3,
    "unique_citation_ranks": True,
    "retrieval_rank_scope": [1, 3],
    "rank_mapping": "one_based_rank_to_retrieved_top3_document",
    "sentence_scope": "listed_document_indices_only",
    "unique_sentence_indices": True,
    "nonempty_sentence_list_per_citation": True,
    "relation_requires_citations": True,
    "nei_requires_zero_citations": True,
    "duplicate_json_keys": "reject",
    "nonfinite_json_constants": "reject",
    "plain_json_whitespace": "JSON_standard_only",
    "allowed_transport_wrapper": "exact_lowercase_json_fence_with_newlines",
    "semantic_repairs": 0,
    "retries": 0,
}
V2_DRAFT_PARSER_POLICY = {
    **V2_COMMON_PARSER_POLICY,
    "version": "scifact-agent-draft-parser-v2",
    "invalid_draft_fallback": "fixed_draft_unavailable_sentinel",
    "one_pass_projection": "validated_label_and_ranked_citations_to_evidence",
}
V2_FINAL_PARSER_POLICY = {
    **V2_COMMON_PARSER_POLICY,
    "version": "scifact-agent-output-parser-v2",
    "invalid_final_fallback": "empty_evidence",
}
V2_PARSER_POLICIES = {
    "initial_draft": V2_DRAFT_PARSER_POLICY,
    "final_evidence": V2_FINAL_PARSER_POLICY,
}

OFFICIAL_TRAIN_CLAIM_COUNT = 809
OFFICIAL_CORPUS_COUNT = 5183
OFFICIAL_CORPUS_SHA256 = (
    "b8d6c89624cb2ed74dee8938effc4f5d8bd2086887880af8110d64be4ceade62"
)
OFFICIAL_TRAIN_CLAIMS_SHA256 = (
    "f4c8fa82d8bd0653a9cc8d61a6ea48c25eacea64e90af5dbf390ebb1b74372f0"
)
OFFICIAL_EVALUATOR = {
    "repository": "https://github.com/allenai/scifact-evaluator",
    "commit": OFFICIAL_EVALUATOR_COMMIT,
    "archive_sha256": (
        "16a743524ed0bbb83e56d862b7f04475c5fbdd22d21046aecad7583a1bc0b009"
    ),
    "evaluator_sha256": (
        "2554fed44c3f5592bdeed59ab0d3918a412b452ba1c77da73ba0f62e74c9987f"
    ),
}
V2_GATE_SCHEMA = "scifact-agent-ab-train-admission-v1"
V2_GATE_THRESHOLDS = {
    "minimum_format_valid_rate_all_arms": 0.99,
    "minimum_nonempty_prediction_rate_main_arms": 0.05,
    "minimum_distinct_final_labels_main_arms": 2,
    "minimum_official_train_abstract_rationalized_f1_main_arms": 0.05,
    "minimum_official_train_sentence_label_f1_main_arms": 0.05,
}


class AgentABSummaryError(RuntimeError):
    """Raised when an input is not a complete, hash-bound A/B run."""


class _DuplicateJSONKey(ValueError):
    pass


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise AgentABSummaryError(f"cannot hash input file: {path}") from exc
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _duplicate_guard(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJSONKey(key)
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _parse_initial_draft(
    raw: str, allowed_documents: Mapping[int, int]
) -> dict[str, Any]:
    transport_unwrapped = False
    payload = raw
    if raw.startswith("```json\n") and raw.endswith("\n```"):
        payload = raw[len("```json\n") : -len("\n```")]
        transport_unwrapped = True
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_duplicate_guard,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, _DuplicateJSONKey, ValueError):
        return {
            "valid": False,
            "failure_code": "invalid_draft_json",
            "transport_unwrapped": transport_unwrapped,
            "value": None,
        }
    if not isinstance(value, dict) or set(value) != {"analysis", "candidates"}:
        failure = "invalid_draft_root_schema"
    elif not isinstance(value["analysis"], str) or not value["analysis"].strip():
        failure = "invalid_draft_analysis"
    elif not isinstance(value["candidates"], list) or len(value["candidates"]) > 3:
        failure = "invalid_draft_candidates"
    else:
        ranked_doc_ids = list(allowed_documents)
        if len(ranked_doc_ids) != 3:
            raise AgentABSummaryError(
                "draft parser requires exactly three ranked documents"
            )
        seen_ranks: set[int] = set()
        seen_labels: set[str] = set()
        for candidate in value["candidates"]:
            if not isinstance(candidate, list) or len(candidate) != 3:
                failure = "invalid_draft_candidate_schema"
                break
            rank, label, sentences = candidate
            if (
                not isinstance(rank, int)
                or isinstance(rank, bool)
                or not 1 <= rank <= 3
            ):
                failure = "draft_rank_out_of_scope"
                break
            if rank in seen_ranks:
                failure = "duplicate_draft_rank"
                break
            seen_ranks.add(rank)
            if not isinstance(label, str) or label not in {
                "SUPPORT",
                "CONTRADICT",
                "NEI",
            }:
                failure = "invalid_draft_label"
                break
            seen_labels.add(label)
            if not isinstance(sentences, list):
                failure = "invalid_draft_sentences"
                break
            if any(
                not isinstance(index, int) or isinstance(index, bool)
                for index in sentences
            ):
                failure = "invalid_draft_sentence_index"
                break
            if len(set(sentences)) != len(sentences):
                failure = "duplicate_draft_sentence_index"
                break
            doc_id = ranked_doc_ids[rank - 1]
            if any(
                index < 0 or index >= allowed_documents[doc_id]
                for index in sentences
            ):
                failure = "draft_sentence_out_of_scope"
                break
            if label != "NEI" and not sentences:
                failure = "invalid_draft_sentences"
                break
        else:
            if "NEI" in seen_labels and len(seen_labels) > 1:
                return {
                    "valid": False,
                    "failure_code": "mixed_nei_and_relation_candidates",
                    "transport_unwrapped": transport_unwrapped,
                    "value": None,
                }
            return {
                "valid": True,
                "failure_code": None,
                "transport_unwrapped": transport_unwrapped,
                "value": value,
            }
    return {
        "valid": False,
        "failure_code": failure,
        "transport_unwrapped": transport_unwrapped,
        "value": None,
    }


def _parse_agent_output(
    raw: str, allowed_documents: Mapping[int, int]
) -> dict[str, Any]:
    transport_unwrapped = False
    payload = raw
    if raw.startswith("```json\n") and raw.endswith("\n```"):
        payload = raw[len("```json\n") : -len("\n```")]
        transport_unwrapped = True
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_duplicate_guard,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, _DuplicateJSONKey, ValueError):
        return {
            "valid": False,
            "failure_code": "invalid_json",
            "transport_unwrapped": transport_unwrapped,
            "value": None,
        }
    if not isinstance(value, dict) or set(value) != {"evidence"}:
        return {
            "valid": False,
            "failure_code": "invalid_root_schema",
            "transport_unwrapped": transport_unwrapped,
            "value": None,
        }
    evidence = value["evidence"]
    if not isinstance(evidence, dict) or len(evidence) > 3:
        return {
            "valid": False,
            "failure_code": "invalid_evidence_object",
            "transport_unwrapped": transport_unwrapped,
            "value": None,
        }
    for raw_doc_id, document in evidence.items():
        if not isinstance(raw_doc_id, str):
            failure = "invalid_document_id"
            break
        try:
            doc_id = int(raw_doc_id)
        except ValueError:
            failure = "invalid_document_id"
            break
        if str(doc_id) != raw_doc_id or doc_id not in allowed_documents:
            failure = "document_out_of_scope"
            break
        if not isinstance(document, dict) or set(document) != {"label", "sentences"}:
            failure = "invalid_document_schema"
            break
        label = document["label"]
        if not isinstance(label, str) or label not in {"SUPPORT", "CONTRADICT"}:
            failure = "invalid_label"
            break
        sentences = document["sentences"]
        if not isinstance(sentences, list) or not sentences:
            failure = "invalid_sentences"
            break
        if any(
            not isinstance(index, int) or isinstance(index, bool)
            for index in sentences
        ):
            failure = "invalid_sentence_index"
            break
        if len(set(sentences)) != len(sentences):
            failure = "duplicate_sentence_index"
            break
        if any(index < 0 or index >= allowed_documents[doc_id] for index in sentences):
            failure = "sentence_out_of_scope"
            break
    else:
        return {
            "valid": True,
            "failure_code": None,
            "transport_unwrapped": transport_unwrapped,
            "value": value,
        }
    return {
        "valid": False,
        "failure_code": failure,
        "transport_unwrapped": transport_unwrapped,
        "value": None,
    }


def _draft_to_evidence(
    draft: Mapping[str, Any], ranked_doc_ids: Sequence[int]
) -> dict[str, Any]:
    return {
        "evidence": {
            str(ranked_doc_ids[candidate[0] - 1]): {
                "label": candidate[1],
                "sentences": candidate[2],
            }
            for candidate in draft["candidates"]
            if candidate[1] != "NEI"
        }
    }


def _v2_invalid_parse(
    failure_code: str,
    transport_unwrapped: bool,
    detail: str | None = None,
) -> dict[str, Any]:
    return {
        "valid": False,
        "failure_code": failure_code,
        "transport_unwrapped": transport_unwrapped,
        "value": None,
        "projected_evidence": None,
        "detail": detail,
    }


def _parse_v2_decision(
    raw: str,
    allowed_documents: Mapping[int, int],
    *,
    draft: bool,
) -> dict[str, Any]:
    prefix = "draft_" if draft else ""
    transport_unwrapped = False
    payload = raw
    if raw.startswith("```json\n") and raw.endswith("\n```"):
        payload = raw[len("```json\n") : -len("\n```")]
        transport_unwrapped = True
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_duplicate_guard,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, _DuplicateJSONKey, ValueError) as exc:
        return _v2_invalid_parse(
            "invalid_draft_json" if draft else "invalid_json",
            transport_unwrapped,
            str(exc)[:300],
        )
    if not isinstance(value, dict) or set(value) != {"label", "citations"}:
        return _v2_invalid_parse(
            "invalid_draft_root_schema" if draft else "invalid_root_schema",
            transport_unwrapped,
        )
    label = value["label"]
    if not isinstance(label, str) or label not in {
        "SUPPORT",
        "CONTRADICT",
        "NEI",
    }:
        return _v2_invalid_parse(
            "invalid_draft_label" if draft else "invalid_label",
            transport_unwrapped,
        )
    citations = value["citations"]
    if not isinstance(citations, list) or len(citations) > 3:
        return _v2_invalid_parse(f"invalid_{prefix}citations", transport_unwrapped)
    if label == "NEI" and citations:
        return _v2_invalid_parse(f"{prefix}nei_with_citations", transport_unwrapped)
    if label != "NEI" and not citations:
        return _v2_invalid_parse(
            f"{prefix}relation_without_citations", transport_unwrapped
        )
    ranked_doc_ids = list(allowed_documents)
    if len(ranked_doc_ids) != 3:
        raise AgentABSummaryError(
            "decision parser requires exactly three ranked documents"
        )
    seen_ranks: set[int] = set()
    projected: dict[str, Any] = {}
    for citation in citations:
        if not isinstance(citation, list) or len(citation) != 2:
            return _v2_invalid_parse(
                f"invalid_{prefix}citation_schema", transport_unwrapped
            )
        rank, sentences = citation
        if type(rank) is not int or not 1 <= rank <= 3:
            return _v2_invalid_parse(
                f"{prefix}citation_rank_out_of_scope", transport_unwrapped
            )
        if rank in seen_ranks:
            return _v2_invalid_parse(
                f"duplicate_{prefix}citation_rank", transport_unwrapped
            )
        seen_ranks.add(rank)
        if not isinstance(sentences, list) or not sentences:
            return _v2_invalid_parse(
                f"invalid_{prefix}citation_sentences", transport_unwrapped
            )
        if any(type(index) is not int for index in sentences):
            return _v2_invalid_parse(
                f"invalid_{prefix}citation_sentence_index", transport_unwrapped
            )
        if len(set(sentences)) != len(sentences):
            return _v2_invalid_parse(
                f"duplicate_{prefix}citation_sentence_index",
                transport_unwrapped,
            )
        doc_id = ranked_doc_ids[rank - 1]
        if any(
            index < 0 or index >= allowed_documents[doc_id]
            for index in sentences
        ):
            return _v2_invalid_parse(
                f"{prefix}citation_sentence_out_of_scope", transport_unwrapped
            )
        projected[str(doc_id)] = {"label": label, "sentences": sentences}
    return {
        "valid": True,
        "failure_code": None,
        "transport_unwrapped": transport_unwrapped,
        "value": value,
        "projected_evidence": projected,
        "detail": None,
    }


def _parse_v2_initial_draft(
    raw: str, allowed_documents: Mapping[int, int]
) -> dict[str, Any]:
    return _parse_v2_decision(raw, allowed_documents, draft=True)


def _parse_v2_agent_output(
    raw: str, allowed_documents: Mapping[int, int]
) -> dict[str, Any]:
    return _parse_v2_decision(raw, allowed_documents, draft=False)


def _v2_draft_to_evidence(
    draft: Mapping[str, Any], ranked_doc_ids: Sequence[int]
) -> dict[str, Any]:
    return {
        "evidence": {
            str(ranked_doc_ids[citation[0] - 1]): {
                "label": draft["label"],
                "sentences": citation[1],
            }
            for citation in draft["citations"]
        }
    }


def _v2_json_root_closed(value: str) -> bool:
    """Mirror the v2 runner's type-matched JSON-root stopping predicate."""

    if not value or value[0] != "{":
        return False
    stack: list[str] = []
    in_string = False
    escaped = False
    pairs = {"}": "{", "]": "["}
    for character in value:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character in "[{":
            stack.append(character)
        elif character in "]}":
            if not stack or stack[-1] != pairs[character]:
                return False
            stack.pop()
            if not stack:
                return character == "}"
    return False


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AgentABSummaryError(f"invalid JSON file: {path}") from exc
    if not isinstance(value, dict):
        raise AgentABSummaryError(f"expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise AgentABSummaryError(
                        f"expected an object at {path}:{line_number}"
                    )
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise AgentABSummaryError(f"invalid JSONL file: {path}") from exc
    return rows


def _write_exclusive(path: Path, value: bytes) -> None:
    try:
        with path.open("xb") as handle:
            handle.write(value)
            handle.flush()
    except FileExistsError as exc:
        raise AgentABSummaryError(f"refusing to overwrite output: {path}") from exc


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AgentABSummaryError(f"{name} must be an object")
    return value


def _require_number(value: Any, name: str, *, nullable: bool = False) -> float | None:
    if value is None and nullable:
        return None
    if (
        not isinstance(value, int | float)
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise AgentABSummaryError(f"{name} must be a finite number")
    return float(value)


def _require_nonnegative_number(
    value: Any, name: str, *, nullable: bool = False
) -> float | None:
    number = _require_number(value, name, nullable=nullable)
    if number is not None and number < 0.0:
        raise AgentABSummaryError(f"{name} cannot be negative")
    return number


def _require_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise AgentABSummaryError(f"{name} must be an integer >= {minimum}")
    return value


def _require_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise AgentABSummaryError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _verified_owned_outputs(
    directory: Path, raw_hashes: Any, *, name: str
) -> dict[str, str]:
    hashes = _require_mapping(raw_hashes, f"{name} output hashes")
    verified: dict[str, str] = {}
    for filename, expected in hashes.items():
        if (
            not isinstance(filename, str)
            or not filename
            or Path(filename).name != filename
            or not isinstance(expected, str)
            or len(expected) != 64
            or any(character not in "0123456789abcdef" for character in expected)
        ):
            raise AgentABSummaryError(f"invalid {name} output hash entry")
        path = directory / filename
        if not path.is_file() or _sha256_file(path) != expected:
            raise AgentABSummaryError(f"{name} output hash mismatch: {filename}")
        verified[filename] = expected
    return verified


def _validate_v1_frozen_config_binding(
    raw_binding: Any,
    configuration_lock: Mapping[str, Any],
    input_lock: Mapping[str, Any],
) -> dict[str, Any]:
    binding = _require_mapping(raw_binding, "runner frozen-config binding")
    if set(binding) != {"path", "sha256"}:
        raise AgentABSummaryError("runner frozen-config binding is malformed")
    raw_path = binding.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise AgentABSummaryError("runner frozen-config path is invalid")
    frozen_path = Path(raw_path)
    expected_frozen_hash = _require_sha256(
        binding.get("sha256"), "runner frozen-config hash"
    )
    if not frozen_path.is_file() or _sha256_file(frozen_path) != expected_frozen_hash:
        raise AgentABSummaryError("runner frozen-config file/hash mismatch")
    frozen = _read_json(frozen_path)
    if frozen.get("schema_version") != "scifact-agent-ab-frozen-config-v1":
        raise AgentABSummaryError("frozen-config schema mismatch")
    if frozen.get("locked_configuration") != configuration_lock:
        raise AgentABSummaryError("frozen configuration differs from runner")
    if frozen.get("dev_inputs") != input_lock:
        raise AgentABSummaryError("frozen dev inputs differ from runner")
    selection = _require_mapping(frozen.get("selection"), "frozen selection policy")
    if selection != {
        "train_used_for_prompt_and_parser_smoke_only": True,
        "dev_tuning_permitted": False,
        "main_comparison": "judge_challenger_vs_single_self_review",
    }:
        raise AgentABSummaryError("frozen train/dev selection policy mismatch")

    train_binding = _require_mapping(
        frozen.get("train_run_manifest"), "frozen train-run binding"
    )
    if set(train_binding) != {"path", "sha256", "selected_claim_count"}:
        raise AgentABSummaryError("frozen train-run binding is malformed")
    raw_train_path = train_binding.get("path")
    if not isinstance(raw_train_path, str) or not raw_train_path:
        raise AgentABSummaryError("frozen train-run path is invalid")
    train_manifest_path = Path(raw_train_path)
    expected_train_hash = _require_sha256(
        train_binding.get("sha256"), "frozen train-run manifest hash"
    )
    if (
        not train_manifest_path.is_file()
        or _sha256_file(train_manifest_path) != expected_train_hash
    ):
        raise AgentABSummaryError("frozen train-run manifest/hash mismatch")
    train_manifest = _read_json(train_manifest_path)
    train_input_lock = _require_mapping(
        train_manifest.get("input_lock"), "frozen train-run input lock"
    )
    train_execution = _require_mapping(
        train_manifest.get("execution"), "frozen train-run execution"
    )
    if (
        train_manifest.get("schema_version") != "scifact-agent-ab-run-v1"
        or train_manifest.get("configuration_lock") != configuration_lock
        or train_input_lock.get("split") != "train"
        or train_execution.get("completed") is not True
        or train_execution.get("selected_claim_count")
        != train_binding.get("selected_claim_count")
    ):
        raise AgentABSummaryError("frozen train-run contract mismatch")
    train_outputs = _verified_owned_outputs(
        train_manifest_path.parent,
        train_manifest.get("outputs_sha256"),
        name="frozen train run",
    )
    if not train_outputs:
        raise AgentABSummaryError("frozen train run has no bound outputs")
    return {
        "path": str(frozen_path.resolve()),
        "sha256": expected_frozen_hash,
        "train_run_manifest": {
            "path": str(train_manifest_path.resolve()),
            "sha256": expected_train_hash,
            "verified_outputs_sha256": train_outputs,
        },
    }


def _flatten_metrics(metrics: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    flattened = {
        f"{group}_{statistic}": float(metrics[group][statistic])
        for group in OFFICIAL_GROUPS
        for statistic in OFFICIAL_STATISTICS
    }
    if set(flattened) != set(OFFICIAL_METRIC_NAMES):
        raise AgentABSummaryError("independent official metrics are incomplete")
    return flattened


def _validated_flat_metrics(value: Any, name: str) -> dict[str, float]:
    metrics = _require_mapping(value, name)
    if set(metrics) != set(OFFICIAL_METRIC_NAMES):
        raise AgentABSummaryError(f"{name} has an unexpected metric set")
    validated: dict[str, float] = {}
    for metric in OFFICIAL_METRIC_NAMES:
        number = _require_number(metrics[metric], f"{name}.{metric}")
        assert number is not None
        if not 0.0 <= number <= 1.0:
            raise AgentABSummaryError(f"{name}.{metric} is outside [0, 1]")
        validated[metric] = number
    return validated


def _flatten_independent_metrics(value: Any, name: str) -> dict[str, float]:
    nested = _require_mapping(value, name)
    if set(nested) != set(OFFICIAL_GROUPS):
        raise AgentABSummaryError(f"{name} has an unexpected group set")
    normalized: dict[str, dict[str, float]] = {}
    for group in OFFICIAL_GROUPS:
        raw_group = _require_mapping(nested[group], f"{name}.{group}")
        if set(raw_group) != set(OFFICIAL_STATISTICS):
            raise AgentABSummaryError(f"{name}.{group} is incomplete")
        normalized[group] = {}
        for statistic in OFFICIAL_STATISTICS:
            number = _require_number(
                raw_group[statistic], f"{name}.{group}.{statistic}"
            )
            assert number is not None
            normalized[group][statistic] = number
    return _flatten_metrics(normalized)


def _same_metrics(left: Mapping[str, float], right: Mapping[str, float]) -> bool:
    return set(left) == set(right) and all(
        abs(left[key] - right[key]) <= 1e-12 for key in left
    )


def _load_official_score(
    directory: Path,
    *,
    arm: str,
    prediction_path: Path,
    split: str,
    claim_count: int,
    corpus_count: int,
    corpus_sha256: str,
    claims_sha256: str,
) -> tuple[dict[str, float], dict[str, Any]]:
    manifest_path = directory / "run_manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != "scifact-official-pipeline-score-v1":
        raise AgentABSummaryError(f"{arm} score manifest schema mismatch")
    if manifest.get("evaluation_status") != {
        "reportable": True,
        "mode": "official_leaderboard_evaluator_reproduction",
    }:
        raise AgentABSummaryError(f"{arm} score is not a formal official run")
    evaluator = _require_mapping(
        manifest.get("official_evaluator"), f"{arm} official evaluator"
    )
    if evaluator.get("commit") != OFFICIAL_EVALUATOR_COMMIT:
        raise AgentABSummaryError(f"{arm} score used an unexpected evaluator commit")
    for field in ("archive_sha256", "evaluator_sha256"):
        value = evaluator.get(field)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise AgentABSummaryError(f"{arm} evaluator {field} is invalid")
    data = _require_mapping(manifest.get("data"), f"{arm} score data")
    if data != {
        "split": split,
        "claim_count": claim_count,
        "corpus_count": corpus_count,
        "corpus_sha256": corpus_sha256,
        "claims_sha256": claims_sha256,
    }:
        raise AgentABSummaryError(f"{arm} score data differs from runner inputs")
    prediction = _require_mapping(
        manifest.get("predictions"), f"{arm} score prediction binding"
    )
    if prediction != {
        "sha256": _sha256_file(prediction_path),
        "exact_claim_coverage": True,
        "known_documents_and_valid_sentence_indices": True,
    }:
        raise AgentABSummaryError(f"{arm} score prediction binding mismatch")
    validation = _require_mapping(
        manifest.get("validation"), f"{arm} score validation"
    )
    if validation.get("official_evaluator_cross_check") is not True or validation.get(
        "absolute_tolerance"
    ) != 1e-12:
        raise AgentABSummaryError(f"{arm} score lacks the official cross-check")
    output_hashes = _verified_owned_outputs(
        directory, manifest.get("outputs_sha256"), name=f"{arm} score"
    )
    required = {
        "official_metrics.json",
        "independent_metrics.json",
        "official_stdout.txt",
        "official_stderr.txt",
    }
    if not required.issubset(output_hashes):
        raise AgentABSummaryError(f"{arm} score outputs are incomplete")
    official = _validated_flat_metrics(
        _read_json(directory / "official_metrics.json"), f"{arm} official metrics"
    )
    independent = _flatten_independent_metrics(
        _read_json(directory / "independent_metrics.json"),
        f"{arm} independent metrics",
    )
    if not _same_metrics(official, independent):
        raise AgentABSummaryError(f"{arm} official and independent scores disagree")
    return official, {
        "directory": str(directory),
        "manifest_sha256": _sha256_file(manifest_path),
        "official_metrics_sha256": output_hashes["official_metrics.json"],
        "independent_metrics_sha256": output_hashes["independent_metrics.json"],
    }


def _valid_official_evidence_shape(value: Any) -> bool:
    if not isinstance(value, dict) or len(value) > 3:
        return False
    for raw_doc_id, document in value.items():
        label = document.get("label") if isinstance(document, dict) else None
        if (
            not isinstance(raw_doc_id, str)
            or not raw_doc_id.isascii()
            or not raw_doc_id.isdigit()
            or int(raw_doc_id) <= 0
            or not isinstance(document, dict)
            or set(document) != {"label", "sentences"}
            or not isinstance(label, str)
            or label not in {"SUPPORT", "CONTRADICT"}
        ):
            return False
        sentences = document.get("sentences")
        if (
            not isinstance(sentences, list)
            or not sentences
            or any(type(index) is not int or index < 0 for index in sentences)
            or len(sentences) != len(set(sentences))
        ):
            return False
    return True


def _recompute_train_gate_observed(
    train_directory: Path,
    train_outputs: Mapping[str, str],
) -> dict[str, dict[str, Any]]:
    required = {"cases.jsonl"} | set(PREDICTION_FILES.values())
    if not required.issubset(train_outputs):
        raise AgentABSummaryError("v2 frozen train run lacks gate inputs")
    cases = _read_jsonl(train_directory / "cases.jsonl")
    if len(cases) != OFFICIAL_TRAIN_CLAIM_COUNT:
        raise AgentABSummaryError("v2 train gate case coverage is not 809")
    case_ids = [case.get("claim_id") for case in cases]
    if (
        any(type(claim_id) is not int for claim_id in case_ids)
        or len(set(case_ids)) != OFFICIAL_TRAIN_CLAIM_COUNT
    ):
        raise AgentABSummaryError("v2 train gate case IDs are invalid")

    observed: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        predictions = _read_jsonl(train_directory / PREDICTION_FILES[arm])
        if len(predictions) != OFFICIAL_TRAIN_CLAIM_COUNT or [
            prediction.get("id") for prediction in predictions
        ] != case_ids:
            raise AgentABSummaryError(f"v2 train {arm} prediction coverage mismatch")
        valid_count = 0
        nonempty_count = 0
        label_counts = {"SUPPORT": 0, "CONTRADICT": 0, "NEI": 0}
        for case, prediction in zip(cases, predictions, strict=True):
            arms = _require_mapping(case.get("arms"), "v2 train case arms")
            if set(arms) != set(ARMS):
                raise AgentABSummaryError("v2 train case arm set mismatch")
            summary = _require_mapping(arms[arm], f"v2 train {arm} summary")
            evidence = prediction.get("evidence")
            if (
                case.get("schema_version") != "scifact-agent-case-v1"
                or set(prediction) != {"id", "evidence"}
                or not _valid_official_evidence_shape(evidence)
                or summary.get("final_evidence") != evidence
                or not isinstance(summary.get("valid"), bool)
            ):
                raise AgentABSummaryError(
                    f"v2 train {arm} case/prediction format mismatch"
                )
            valid_count += int(summary["valid"])
            nonempty_count += int(bool(evidence))
            if summary["valid"]:
                labels = {document["label"] for document in evidence.values()}
                if len(labels) > 1:
                    raise AgentABSummaryError(
                        f"v2 train {arm} mixes claim-level relation labels"
                    )
                label_counts[next(iter(labels), "NEI")] += 1
        observed[arm] = {
            "format_valid_count": valid_count,
            "nonempty_prediction_count": nonempty_count,
            "label_counts": label_counts,
            "claim_count": OFFICIAL_TRAIN_CLAIM_COUNT,
            "format_valid_rate": valid_count / OFFICIAL_TRAIN_CLAIM_COUNT,
            "nonempty_prediction_rate": (
                nonempty_count / OFFICIAL_TRAIN_CLAIM_COUNT
            ),
            "distinct_label_count": sum(
                count > 0 for count in label_counts.values()
            ),
        }
    return observed


def _validate_v2_train_score_report(
    *,
    arm: str,
    raw_report: Any,
    train_directory: Path,
    train_outputs: Mapping[str, str],
) -> dict[str, Any]:
    report = _require_mapping(raw_report, f"v2 gate {arm} score report")
    expected_keys = {
        "score_manifest_path",
        "score_manifest_sha256",
        "official_metrics_sha256",
        "independent_metrics_sha256",
        "official_metrics",
    }
    if set(report) != expected_keys:
        raise AgentABSummaryError(f"v2 gate {arm} score report schema mismatch")
    raw_manifest_path = report.get("score_manifest_path")
    if not isinstance(raw_manifest_path, str) or not raw_manifest_path:
        raise AgentABSummaryError(f"v2 gate {arm} score path is invalid")
    manifest_path = Path(raw_manifest_path)
    if manifest_path.name != "run_manifest.json" or not manifest_path.is_file():
        raise AgentABSummaryError(f"v2 gate {arm} score manifest is missing")
    expected_manifest_hash = _require_sha256(
        report.get("score_manifest_sha256"), f"v2 gate {arm} score manifest hash"
    )
    if _sha256_file(manifest_path) != expected_manifest_hash:
        raise AgentABSummaryError(f"v2 gate {arm} score manifest hash mismatch")
    score_manifest = _read_json(manifest_path)
    if score_manifest.get("official_evaluator") != OFFICIAL_EVALUATOR:
        raise AgentABSummaryError(f"v2 gate {arm} evaluator pin mismatch")
    prediction_path = train_directory / PREDICTION_FILES[arm]
    official, provenance = _load_official_score(
        manifest_path.parent,
        arm=f"train-{arm}",
        prediction_path=prediction_path,
        split="train",
        claim_count=OFFICIAL_TRAIN_CLAIM_COUNT,
        corpus_count=OFFICIAL_CORPUS_COUNT,
        corpus_sha256=OFFICIAL_CORPUS_SHA256,
        claims_sha256=OFFICIAL_TRAIN_CLAIMS_SHA256,
    )
    if (
        provenance["manifest_sha256"] != expected_manifest_hash
        or report.get("official_metrics_sha256")
        != provenance["official_metrics_sha256"]
        or report.get("independent_metrics_sha256")
        != provenance["independent_metrics_sha256"]
        or _validated_flat_metrics(
            report.get("official_metrics"), f"v2 gate {arm} reported metrics"
        )
        != official
        or _sha256_file(prediction_path)
        != train_outputs[PREDICTION_FILES[arm]]
    ):
        raise AgentABSummaryError(f"v2 gate {arm} score evidence mismatch")
    return {**provenance, "official_metrics": official}


def _validate_v2_frozen_config_binding(
    raw_binding: Any,
    configuration_lock: Mapping[str, Any],
    input_lock: Mapping[str, Any],
) -> dict[str, Any]:
    binding = _require_mapping(raw_binding, "runner frozen-config binding")
    if set(binding) != {"path", "sha256"}:
        raise AgentABSummaryError("runner frozen-config binding is malformed")
    raw_path = binding.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise AgentABSummaryError("runner frozen-config path is invalid")
    frozen_path = Path(raw_path)
    expected_frozen_hash = _require_sha256(
        binding.get("sha256"), "runner frozen-config hash"
    )
    if not frozen_path.is_file() or _sha256_file(frozen_path) != expected_frozen_hash:
        raise AgentABSummaryError("runner frozen-config file/hash mismatch")
    frozen = _read_json(frozen_path)
    expected_frozen_keys = {
        "schema_version",
        "created_at",
        "train_run_manifest",
        "locked_configuration",
        "dev_inputs",
        "selection",
        "train_admission_gate",
        "train_admission_gate_sha256",
    }
    if (
        set(frozen) != expected_frozen_keys
        or frozen.get("schema_version") != "scifact-agent-ab-frozen-config-v2"
        or not isinstance(frozen.get("created_at"), str)
        or not frozen["created_at"]
    ):
        raise AgentABSummaryError("v2 frozen-config schema mismatch")
    if frozen.get("locked_configuration") != configuration_lock:
        raise AgentABSummaryError("frozen configuration differs from runner")
    if frozen.get("dev_inputs") != input_lock:
        raise AgentABSummaryError("frozen dev inputs differ from runner")
    if frozen.get("selection") != {
        "train_used_for_viability_gate_only": True,
        "dev_tuning_permitted": False,
        "main_comparison": "judge_challenger_vs_single_self_review",
    }:
        raise AgentABSummaryError("v2 frozen train/dev selection policy mismatch")

    train_binding = _require_mapping(
        frozen.get("train_run_manifest"), "v2 frozen train-run binding"
    )
    if (
        set(train_binding) != {"path", "sha256", "selected_claim_count"}
        or train_binding.get("selected_claim_count") != OFFICIAL_TRAIN_CLAIM_COUNT
    ):
        raise AgentABSummaryError("v2 frozen train-run binding is malformed")
    raw_train_path = train_binding.get("path")
    if not isinstance(raw_train_path, str) or not raw_train_path:
        raise AgentABSummaryError("v2 frozen train-run path is invalid")
    train_manifest_path = Path(raw_train_path)
    expected_train_hash = _require_sha256(
        train_binding.get("sha256"), "v2 frozen train-run manifest hash"
    )
    if (
        not train_manifest_path.is_file()
        or _sha256_file(train_manifest_path) != expected_train_hash
    ):
        raise AgentABSummaryError("v2 frozen train-run manifest/hash mismatch")
    train_manifest = _read_json(train_manifest_path)
    train_inputs = _require_mapping(
        train_manifest.get("input_lock"), "v2 frozen train input lock"
    )
    train_execution = _require_mapping(
        train_manifest.get("execution"), "v2 frozen train execution"
    )
    if (
        train_manifest.get("schema_version") != "scifact-agent-ab-run-v1"
        or train_manifest.get("evaluation_status")
        != {
            "reportable": False,
            "state": "ready_for_official_scoring",
            "gold_used_by_runner": False,
        }
        or train_manifest.get("configuration_lock") != configuration_lock
        or train_inputs.get("split") != "train"
        or train_inputs.get("corpus_sha256") != OFFICIAL_CORPUS_SHA256
        or train_inputs.get("claims_sha256") != OFFICIAL_TRAIN_CLAIMS_SHA256
        or train_execution.get("completed") is not True
        or train_execution.get("source_claim_count") != OFFICIAL_TRAIN_CLAIM_COUNT
        or train_execution.get("selected_claim_count")
        != OFFICIAL_TRAIN_CLAIM_COUNT
        or train_execution.get("physical_call_count")
        != OFFICIAL_TRAIN_CLAIM_COUNT * 3
        or train_execution.get("expected_physical_call_count")
        != OFFICIAL_TRAIN_CLAIM_COUNT * 3
    ):
        raise AgentABSummaryError("v2 frozen train-run contract mismatch")
    train_outputs = _verified_owned_outputs(
        train_manifest_path.parent,
        train_manifest.get("outputs_sha256"),
        name="v2 frozen train run",
    )
    observed = _recompute_train_gate_observed(
        train_manifest_path.parent, train_outputs
    )

    gate = _require_mapping(
        frozen.get("train_admission_gate"), "v2 train admission gate binding"
    )
    if (
        set(gate) != {"schema_version", "status", "report"}
        or gate.get("schema_version") != V2_GATE_SCHEMA
        or gate.get("status") != "passed"
        or frozen.get("train_admission_gate_sha256")
        != _sha256_bytes(_canonical_json(gate).encode("utf-8"))
    ):
        raise AgentABSummaryError("v2 train admission gate binding is invalid")
    report_binding = _require_mapping(
        gate.get("report"), "v2 train admission report binding"
    )
    if set(report_binding) != {"path", "sha256"}:
        raise AgentABSummaryError("v2 train admission report binding is malformed")
    raw_report_path = report_binding.get("path")
    if not isinstance(raw_report_path, str) or not raw_report_path:
        raise AgentABSummaryError("v2 train admission report path is invalid")
    report_path = Path(raw_report_path)
    expected_report_hash = _require_sha256(
        report_binding.get("sha256"), "v2 train admission report hash"
    )
    if not report_path.is_file() or _sha256_file(report_path) != expected_report_hash:
        raise AgentABSummaryError("v2 train admission report file/hash mismatch")
    report = _read_json(report_path)
    expected_report_keys = {
        "schema_version",
        "status",
        "scope",
        "agent_protocol_version",
        "dev_labels_or_metrics_inspected",
        "thresholds_are_viability_floors_not_performance_claims",
        "implementation",
        "thresholds",
        "observed",
        "official_train_scores",
        "evidence_sha256",
    }
    if (
        set(report) != expected_report_keys
        or report.get("schema_version") != V2_GATE_SCHEMA
        or report.get("status") != "passed"
        or report.get("scope") != "complete_official_train_only"
        or report.get("agent_protocol_version") != V2_AGENT_PROTOCOL
        or report.get("dev_labels_or_metrics_inspected") is not False
        or report.get("thresholds_are_viability_floors_not_performance_claims")
        is not True
        or report.get("thresholds") != V2_GATE_THRESHOLDS
    ):
        raise AgentABSummaryError("v2 train admission report contract is invalid")
    implementation = _require_mapping(
        report.get("implementation"), "v2 gate implementation binding"
    )
    gate_script = Path(__file__).resolve().with_name(
        "freeze_scifact_agent_ab_config.py"
    )
    if (
        implementation
        != {
            "path": "scripts/freeze_scifact_agent_ab_config.py",
            "sha256": _sha256_file(gate_script),
        }
        or not gate_script.is_file()
    ):
        raise AgentABSummaryError("v2 gate implementation hash mismatch")
    evidence = _require_mapping(
        report.get("evidence_sha256"), "v2 gate evidence hashes"
    )
    expected_prediction_hashes = {
        arm: train_outputs[PREDICTION_FILES[arm]] for arm in ARMS
    }
    if evidence != {
        "train_manifest": expected_train_hash,
        "cases": train_outputs["cases.jsonl"],
        "predictions": expected_prediction_hashes,
    }:
        raise AgentABSummaryError("v2 gate evidence hashes mismatch")
    if report.get("observed") != observed:
        raise AgentABSummaryError("v2 gate observed rates are not reproducible")
    for arm in ARMS:
        if observed[arm]["format_valid_rate"] < 0.99:
            raise AgentABSummaryError("v2 train format viability floor was not met")
    for arm in (MAIN_BASELINE, MAIN_CANDIDATE):
        if observed[arm]["nonempty_prediction_rate"] < 0.05:
            raise AgentABSummaryError("v2 train non-empty viability floor was not met")
        if observed[arm]["distinct_label_count"] < 2:
            raise AgentABSummaryError("v2 train constant-label collapse detected")

    raw_scores = _require_mapping(
        report.get("official_train_scores"), "v2 gate official train scores"
    )
    if set(raw_scores) != set(ARMS):
        raise AgentABSummaryError("v2 gate official train score arms mismatch")
    train_scores = {
        arm: _validate_v2_train_score_report(
            arm=arm,
            raw_report=raw_scores[arm],
            train_directory=train_manifest_path.parent,
            train_outputs=train_outputs,
        )
        for arm in ARMS
    }
    for arm in (MAIN_BASELINE, MAIN_CANDIDATE):
        metrics = train_scores[arm]["official_metrics"]
        if (
            metrics["abstract_rationalized_f1"] < 0.05
            or metrics["sentence_label_f1"] < 0.05
        ):
            raise AgentABSummaryError("v2 official train quality floor was not met")
    return {
        "path": str(frozen_path.resolve()),
        "sha256": expected_frozen_hash,
        "schema_version": "scifact-agent-ab-frozen-config-v2",
        "train_admission_report": {
            "path": str(report_path.resolve()),
            "sha256": expected_report_hash,
        },
        "train_run_manifest": {
            "path": str(train_manifest_path.resolve()),
            "sha256": expected_train_hash,
            "verified_outputs_sha256": train_outputs,
        },
    }


def _validate_frozen_config_binding(
    raw_binding: Any,
    configuration_lock: Mapping[str, Any],
    input_lock: Mapping[str, Any],
) -> dict[str, Any]:
    protocol = configuration_lock.get("agent_protocol_version")
    if protocol is None:
        return _validate_v1_frozen_config_binding(
            raw_binding, configuration_lock, input_lock
        )
    if protocol == V2_AGENT_PROTOCOL:
        return _validate_v2_frozen_config_binding(
            raw_binding, configuration_lock, input_lock
        )
    raise AgentABSummaryError("unknown Agent A/B decision protocol")


def _load_retrieval(path: Path, claim_ids: Sequence[int]) -> dict[int, list[int]]:
    by_id: dict[int, list[int]] = {}
    for row in _read_jsonl(path):
        claim_id = row.get("claim_id")
        doc_ids = row.get("doc_ids")
        if (
            not isinstance(claim_id, int)
            or isinstance(claim_id, bool)
            or claim_id in by_id
            or not isinstance(doc_ids, list)
            or len(doc_ids) != 3
            or len(set(doc_ids)) != 3
            or any(
                not isinstance(doc_id, int) or isinstance(doc_id, bool)
                for doc_id in doc_ids
            )
        ):
            raise AgentABSummaryError("invalid official Top-3 retrieval row")
        by_id[claim_id] = doc_ids
    if set(by_id) != set(claim_ids):
        raise AgentABSummaryError("retrieval exact claim coverage mismatch")
    return by_id


def _validate_retrieval_binding(
    path: Path,
    *,
    split: str,
    corpus_path: Path,
    claims_path: Path,
    expected_manifest_sha256: Any,
) -> dict[str, Any]:
    manifest_path = path.parent / "run_manifest.json"
    if _sha256_file(manifest_path) != expected_manifest_sha256:
        raise AgentABSummaryError("retrieval manifest hash differs from runner lock")
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != "scifact-official-tfidf-run-v1":
        raise AgentABSummaryError("retrieval manifest schema mismatch")
    status = _require_mapping(manifest.get("evaluation_status"), "retrieval status")
    if status.get("reportable") is not True:
        raise AgentABSummaryError("retrieval run is not reportable")
    data = _require_mapping(manifest.get("data"), "retrieval data")
    configuration = _require_mapping(
        manifest.get("configuration"), "retrieval configuration"
    )
    file_hashes = _require_mapping(data.get("files_sha256"), "retrieval data hashes")
    output_hashes = _require_mapping(
        manifest.get("outputs_sha256"), "retrieval output hashes"
    )
    if (
        data.get("split") != split
        or configuration.get("top_k") != 3
        or file_hashes.get(corpus_path.name) != _sha256_file(corpus_path)
        or file_hashes.get(claims_path.name) != _sha256_file(claims_path)
        or output_hashes.get(path.name) != _sha256_file(path)
    ):
        raise AgentABSummaryError("retrieval provenance differs from runner inputs")
    return {
        "manifest_sha256": _sha256_file(manifest_path),
        "retrieval_sha256": _sha256_file(path),
    }


def _validate_contexts(
    contexts: Sequence[dict[str, Any]],
    cases: Sequence[dict[str, Any]],
    data: ValidatedSciFact,
    retrieval: Mapping[int, list[int]],
) -> dict[int, dict[int, int]]:
    if len(contexts) != len(cases) or len(contexts) != len(data.claims):
        raise AgentABSummaryError("context/case coverage mismatch")
    allowed_by_claim: dict[int, dict[int, int]] = {}
    for index, (context_row, case_row, claim) in enumerate(
        zip(contexts, cases, data.claims, strict=True)
    ):
        claim_id = claim["id"]
        if context_row.get("schema_version") != "scifact-agent-context-v1":
            raise AgentABSummaryError("context schema mismatch")
        if (
            context_row.get("claim_id") != claim_id
            or case_row.get("claim_id") != claim_id
        ):
            raise AgentABSummaryError("context/case claim order mismatch")
        if case_row.get("case_index") != index:
            raise AgentABSummaryError("case indices are not contiguous")
        if context_row.get("gold_fields_present") is not False:
            raise AgentABSummaryError("runner context reports gold fields")
        context = _require_mapping(context_row.get("context"), "canonical context")
        if set(context) != {"claim", "documents"} or context["claim"] != claim["claim"]:
            raise AgentABSummaryError("canonical context claim is invalid")
        documents = context.get("documents")
        if not isinstance(documents, list) or len(documents) != 3:
            raise AgentABSummaryError("canonical context is not Top-3")
        expected_ids = retrieval[claim_id]
        allowed_by_claim[claim_id] = {
            doc_id: len(data.corpus[doc_id]["abstract"]) for doc_id in expected_ids
        }
        for rank, (document, expected_doc_id) in enumerate(
            zip(documents, expected_ids, strict=True), 1
        ):
            doc = _require_mapping(document, "context document")
            if set(doc) != {"rank", "doc_id", "title", "sentences"}:
                raise AgentABSummaryError("context document has unexpected fields")
            corpus_doc = data.corpus[expected_doc_id]
            expected_sentences = [
                {"index": sentence_index, "text": sentence}
                for sentence_index, sentence in enumerate(corpus_doc["abstract"])
            ]
            if (
                doc["rank"] != rank
                or doc["doc_id"] != expected_doc_id
                or doc["title"] != corpus_doc["title"]
                or doc["sentences"] != expected_sentences
            ):
                raise AgentABSummaryError(
                    "context differs from frozen retrieval/corpus"
                )
        context_hash = _sha256_bytes(_canonical_json(context).encode("utf-8"))
        if (
            context_row.get("canonical_context_sha256") != context_hash
            or case_row.get("canonical_context_sha256") != context_hash
        ):
            raise AgentABSummaryError("canonical context hash mismatch")
    return allowed_by_claim


def _nullable_sum(values: Sequence[Any], name: str) -> float | None:
    normalized = [
        _require_nonnegative_number(value, name, nullable=True) for value in values
    ]
    if any(value is None for value in normalized):
        return None
    return float(sum(value for value in normalized if value is not None))


def _same_nullable_number(left: Any, right: float | None, name: str) -> None:
    normalized = _require_nonnegative_number(left, name, nullable=True)
    if normalized != right:
        raise AgentABSummaryError(f"{name} does not match raw telemetry")


def _validate_call(
    call: dict[str, Any],
    parser_policy_hashes: Mapping[str, str],
    stopping_policy: str,
    protocol: str | None,
) -> None:
    if call.get("schema_version") != "scifact-agent-physical-call-v1":
        raise AgentABSummaryError("physical-call schema mismatch")
    if call.get("call_kind") not in CALL_KINDS:
        raise AgentABSummaryError("unknown physical call kind")
    is_initial = call["call_kind"] == "shared_initial_judge"
    if protocol == V2_AGENT_PROTOCOL:
        expected_prefill = (
            V2_INITIAL_DRAFT_PREFILL if is_initial else V2_FINAL_EVIDENCE_PREFILL
        )
    else:
        expected_prefill = (
            INITIAL_DRAFT_PREFILL if is_initial else FINAL_EVIDENCE_PREFILL
        )
    expected_parse_schema = "initial_draft" if is_initial else "final_evidence"
    call_id = call.get("call_id")
    if not isinstance(call_id, str) or not call_id:
        raise AgentABSummaryError("physical call ID is invalid")
    claim_id = call.get("claim_id")
    context_sha256 = call.get("canonical_context_sha256")
    if not isinstance(claim_id, int) or isinstance(claim_id, bool):
        raise AgentABSummaryError("physical call claim ID is invalid")
    if (
        not isinstance(context_sha256, str)
        or len(context_sha256) != 64
        or any(character not in "0123456789abcdef" for character in context_sha256)
    ):
        raise AgentABSummaryError("physical call context hash is invalid")
    attribution = call.get("logical_attribution")
    expected_attribution = {
        "shared_initial_judge": list(ARMS),
        "self_review": ["single_self_review"],
        "independent_challenger": ["judge_challenger"],
    }[call["call_kind"]]
    if attribution != expected_attribution:
        raise AgentABSummaryError("physical call attribution mismatch")
    device = call.get("device")
    if not isinstance(device, str) or not device:
        raise AgentABSummaryError("physical call device is invalid")
    _require_sha256(call.get("messages_sha256"), "messages_sha256")
    if (
        call.get("assistant_prefill") != expected_prefill
        or call.get("assistant_prefill_sha256")
        != _sha256_bytes(expected_prefill.encode("utf-8"))
        or call.get("assistant_prefill_in_input_tokens") is not True
    ):
        raise AgentABSummaryError("physical call assistant prefill mismatch")
    if call.get("parse_schema") != expected_parse_schema:
        raise AgentABSummaryError("physical call parse schema mismatch")
    if call.get("parser_policy_sha256") != parser_policy_hashes[
        expected_parse_schema
    ]:
        raise AgentABSummaryError("physical call parser policy hash mismatch")
    if (
        call.get("stopping_policy") != stopping_policy
        or call.get("stopping_policy_sha256")
        != _sha256_bytes(stopping_policy.encode("utf-8"))
    ):
        raise AgentABSummaryError("physical call stopping policy mismatch")
    if not isinstance(call.get("json_root_stopping_triggered"), bool):
        raise AgentABSummaryError("JSON-root stopping marker must be boolean")
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        number = _require_nonnegative_number(call.get(key), key)
        assert number is not None
        if not number.is_integer():
            raise AgentABSummaryError(f"{key} must be an integer count")
    input_tokens = call.get("input_tokens")
    output_tokens = call.get("output_tokens")
    total_tokens = call.get("total_tokens")
    if total_tokens is not None and total_tokens != input_tokens + output_tokens:
        raise AgentABSummaryError("physical total token count is inconsistent")
    _require_nonnegative_number(call.get("synchronized_latency_ms"), "call latency")
    _require_nonnegative_number(call.get("end_to_end_latency_ms"), "call end-to-end")
    if (
        call.get("external_api_cost_usd") != 0.0
        or call.get("compute_cost_usd") is not None
    ):
        raise AgentABSummaryError("physical-call cost accounting mismatch")
    if protocol == V2_AGENT_PROTOCOL and call.get("generation_overrides") != {
        "repetition_penalty": 1.0
    }:
        raise AgentABSummaryError("v2 physical-call generation override mismatch")
    succeeded = call.get("generation_succeeded")
    if succeeded is not True or call.get("generation_error") is not None:
        raise AgentABSummaryError("completed runs require successful generations")
    rendered_prompt = call.get("rendered_prompt")
    generated_fragment = call.get("generated_fragment")
    raw = call.get("raw_output")
    raw_hash = call.get("raw_output_sha256")
    if not isinstance(rendered_prompt, str) or not rendered_prompt.endswith(
        expected_prefill
    ):
        raise AgentABSummaryError("rendered prompt does not contain the prefill")
    rendered_hash = _sha256_bytes(rendered_prompt.encode("utf-8"))
    if (
        call.get("rendered_prompt_sha256") != rendered_hash
        or call.get("prompt_sha256") != rendered_hash
        or call.get("prompt_character_count") != len(rendered_prompt)
    ):
        raise AgentABSummaryError("rendered prompt hash/count mismatch")
    if not isinstance(generated_fragment, str):
        raise AgentABSummaryError("generated fragment is missing")
    if call.get("generated_fragment_sha256") != _sha256_bytes(
        generated_fragment.encode("utf-8")
    ):
        raise AgentABSummaryError("generated fragment hash mismatch")
    if raw != expected_prefill + generated_fragment:
        raise AgentABSummaryError("raw output is not prefill plus generated fragment")
    if not isinstance(raw, str) or raw_hash != _sha256_bytes(raw.encode("utf-8")):
        raise AgentABSummaryError("successful raw output hash mismatch")
    if protocol == V2_AGENT_PROTOCOL and call.get(
        "json_root_stopping_triggered"
    ) is not _v2_json_root_closed(raw):
        raise AgentABSummaryError("v2 type-matched JSON stopping marker mismatch")
    parsed = _require_mapping(call.get("parse"), "physical-call parse result")
    if not isinstance(parsed.get("valid"), bool):
        raise AgentABSummaryError("parse.valid must be boolean")
    if parsed["valid"] and parsed.get("failure_code") is not None:
        raise AgentABSummaryError("valid parse cannot have a failure code")
    if not parsed["valid"] and not isinstance(parsed.get("failure_code"), str):
        raise AgentABSummaryError("invalid parse requires a failure code")


def _expected_branch_order(case_index: int) -> list[str]:
    return (
        ["single_self_review", "judge_challenger"]
        if case_index % 2 == 0
        else ["judge_challenger", "single_self_review"]
    )


def _validate_cases_and_telemetry(
    cases: Sequence[dict[str, Any]],
    telemetry: Sequence[dict[str, Any]],
    allowed_by_claim: Mapping[int, Mapping[int, int]],
    parser_policy_hashes: Mapping[str, str],
    stopping_policy: str,
    protocol: str | None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    by_call_id: dict[str, dict[str, Any]] = {}
    for call in telemetry:
        _validate_call(call, parser_policy_hashes, stopping_policy, protocol)
        call_id = call["call_id"]
        if call_id in by_call_id:
            raise AgentABSummaryError("duplicate physical call ID")
        by_call_id[call_id] = call
    if len(telemetry) != len(cases) * 3:
        raise AgentABSummaryError("physical-call coverage is not three per case")

    seen_call_ids: list[str] = []
    arm_rows: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
    for case in cases:
        if case.get("schema_version") != "scifact-agent-case-v1":
            raise AgentABSummaryError("case schema mismatch")
        claim_id = case.get("claim_id")
        if not isinstance(claim_id, int) or isinstance(claim_id, bool):
            raise AgentABSummaryError("case claim ID is invalid")
        case_index = _require_int(case.get("case_index"), "case index")
        if case.get("branch_execution_order") != _expected_branch_order(case_index):
            raise AgentABSummaryError("case branch order mismatch")
        call_ids = case.get("physical_call_ids")
        if (
            case.get("physical_call_count") != 3
            or not isinstance(call_ids, list)
            or len(call_ids) != 3
            or any(call_id not in by_call_id for call_id in call_ids)
        ):
            raise AgentABSummaryError("case physical calls are incomplete")
        seen_call_ids.extend(call_ids)
        calls = [by_call_id[call_id] for call_id in call_ids]
        if any(
            call["claim_id"] != claim_id
            or call["canonical_context_sha256"]
            != case.get("canonical_context_sha256")
            for call in calls
        ):
            raise AgentABSummaryError("case/call claim or context identity mismatch")
        allowed_documents = allowed_by_claim[claim_id]
        for call in calls:
            if protocol == V2_AGENT_PROTOCOL:
                reparsed = (
                    _parse_v2_initial_draft(call["raw_output"], allowed_documents)
                    if call["call_kind"] == "shared_initial_judge"
                    else _parse_v2_agent_output(
                        call["raw_output"], allowed_documents
                    )
                )
            else:
                reparsed = (
                    _parse_initial_draft(call["raw_output"], allowed_documents)
                    if call["call_kind"] == "shared_initial_judge"
                    else _parse_agent_output(call["raw_output"], allowed_documents)
                )
            stored_parse = call["parse"]
            compared_fields = [
                "valid",
                "failure_code",
                "transport_unwrapped",
                "value",
            ]
            if protocol == V2_AGENT_PROTOCOL:
                compared_fields.append("projected_evidence")
            for field in compared_fields:
                if stored_parse.get(field) != reparsed[field]:
                    raise AgentABSummaryError(
                        "raw output and stored parse result disagree"
                    )
        expected_kinds = ["shared_initial_judge"] + [
            "self_review" if arm == "single_self_review" else "independent_challenger"
            for arm in case["branch_execution_order"]
        ]
        if [call["call_kind"] for call in calls] != expected_kinds:
            raise AgentABSummaryError("case physical call order mismatch")
        prefix = f"{case_index:04d}:{claim_id}"
        expected_call_ids = [f"{prefix}:initial"] + [
            f"{prefix}:"
            + (
                "self_review"
                if arm == "single_self_review"
                else "independent_challenger"
            )
            for arm in case["branch_execution_order"]
        ]
        if call_ids != expected_call_ids:
            raise AgentABSummaryError("case physical call IDs are not canonical")
        by_kind = {call["call_kind"]: call for call in calls}
        initial_parse = by_kind["shared_initial_judge"]["parse"]
        expected_draft = (
            _canonical_json(initial_parse["value"])
            if initial_parse["valid"]
            else DRAFT_UNAVAILABLE
        )
        expected_draft_status = (
            "valid_initial_draft"
            if initial_parse["valid"]
            else "draft_unavailable"
        )
        if (
            case.get("draft_status") != expected_draft_status
            or case.get("draft_sha256")
            != _sha256_bytes(expected_draft.encode("utf-8"))
        ):
            raise AgentABSummaryError("case shared draft binding mismatch")
        expected_arm_calls = {
            "single_one_pass": [by_kind["shared_initial_judge"]],
            "single_self_review": [
                by_kind["shared_initial_judge"],
                by_kind["self_review"],
            ],
            "judge_challenger": [
                by_kind["shared_initial_judge"],
                by_kind["independent_challenger"],
            ],
        }
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            expected = _nullable_sum([call[key] for call in calls], f"physical_{key}")
            _same_nullable_number(
                case.get(f"physical_{key}"), expected, f"physical_{key}"
            )
        physical_latency = _nullable_sum(
            [call["synchronized_latency_ms"] for call in calls],
            "physical_inference_latency_ms",
        )
        _same_nullable_number(
            case.get("physical_inference_latency_ms"),
            physical_latency,
            "physical_inference_latency_ms",
        )
        physical_end_to_end = _nullable_sum(
            [call["end_to_end_latency_ms"] for call in calls],
            "physical_end_to_end_latency_ms",
        )
        _same_nullable_number(
            case.get("physical_end_to_end_latency_ms"),
            physical_end_to_end,
            "physical_end_to_end_latency_ms",
        )
        arms = _require_mapping(case.get("arms"), "case arm summaries")
        if set(arms) != set(ARMS):
            raise AgentABSummaryError("case arm set mismatch")
        for arm in ARMS:
            summary = _require_mapping(arms[arm], f"{arm} case summary")
            expected_calls = expected_arm_calls[arm]
            if summary.get("logical_call_count") != len(expected_calls) or summary.get(
                "physical_call_ids"
            ) != [call["call_id"] for call in expected_calls]:
                raise AgentABSummaryError(f"{arm} logical call attribution mismatch")
            for key in ("input_tokens", "output_tokens", "total_tokens"):
                expected = _nullable_sum(
                    [call[key] for call in expected_calls], f"{arm}.{key}"
                )
                _same_nullable_number(summary.get(key), expected, f"{arm}.{key}")
            expected_latency = _nullable_sum(
                [call["synchronized_latency_ms"] for call in expected_calls],
                f"{arm}.logical_inference_latency_ms",
            )
            _same_nullable_number(
                summary.get("logical_inference_latency_ms"),
                expected_latency,
                f"{arm}.logical_inference_latency_ms",
            )
            expected_end_to_end = _nullable_sum(
                [call["end_to_end_latency_ms"] for call in expected_calls],
                f"{arm}.logical_end_to_end_latency_ms",
            )
            _same_nullable_number(
                summary.get("logical_end_to_end_latency_ms"),
                expected_end_to_end,
                f"{arm}.logical_end_to_end_latency_ms",
            )
            final_call = expected_calls[-1]
            if arm == "single_one_pass" and initial_parse["valid"]:
                parsed = {
                    "valid": True,
                    "failure_code": None,
                    "transport_unwrapped": initial_parse[
                        "transport_unwrapped"
                    ],
                    "value": (
                        _v2_draft_to_evidence(
                            initial_parse["value"], list(allowed_documents)
                        )
                        if protocol == V2_AGENT_PROTOCOL
                        else _draft_to_evidence(
                            initial_parse["value"], list(allowed_documents)
                        )
                    ),
                }
            else:
                parsed = final_call["parse"]
            if (
                summary.get("valid") != parsed["valid"]
                or summary.get("failure_code") != parsed["failure_code"]
                or summary.get("transport_unwrapped")
                != parsed["transport_unwrapped"]
                or summary.get("invalid_final_fallback_applied") == parsed["valid"]
                or summary.get("external_api_cost_usd") != 0.0
                or summary.get("compute_cost_usd") is not None
            ):
                raise AgentABSummaryError(f"{arm} final status/cost mismatch")
            if not parsed["valid"]:
                expected_evidence = {}
            elif protocol == V2_AGENT_PROTOCOL and arm != "single_one_pass":
                expected_evidence = parsed["projected_evidence"]
            else:
                expected_evidence = parsed["value"]["evidence"]
            if summary.get("final_evidence") != expected_evidence:
                raise AgentABSummaryError(f"{arm} final evidence mismatch")
            arm_rows[arm].append(
                {"summary": summary, "logical_calls": expected_calls}
            )

    if (
        len(seen_call_ids) != len(set(seen_call_ids))
        or set(seen_call_ids) != set(by_call_id)
        or seen_call_ids != [call["call_id"] for call in telemetry]
    ):
        raise AgentABSummaryError("physical calls are not partitioned exactly by cases")

    shared_initial_calls = [
        call for call in telemetry if call["call_kind"] == "shared_initial_judge"
    ]
    if len(shared_initial_calls) != len(cases):
        raise AgentABSummaryError("shared initial call coverage mismatch")
    shared_end_to_end = [
        float(call["end_to_end_latency_ms"]) for call in shared_initial_calls
    ]
    physical = {
        "physical_call_count": len(telemetry),
        "input_tokens": _nullable_sum(
            [call["input_tokens"] for call in telemetry], "physical input tokens"
        ),
        "output_tokens": _nullable_sum(
            [call["output_tokens"] for call in telemetry], "physical output tokens"
        ),
        "total_tokens": _nullable_sum(
            [call["total_tokens"] for call in telemetry], "physical total tokens"
        ),
        "synchronized_inference_latency_ms_sum": _nullable_sum(
            [call["synchronized_latency_ms"] for call in telemetry],
            "physical latency",
        ),
        "active_end_to_end_latency_ms_sum": _nullable_sum(
            [call["end_to_end_latency_ms"] for call in telemetry],
            "physical end-to-end latency",
        ),
        "format_failure_calls": sum(
            bool(call["generation_succeeded"]) and not bool(call["parse"]["valid"])
            for call in telemetry
        ),
        "generation_failure_calls": sum(
            not bool(call["generation_succeeded"]) for call in telemetry
        ),
        "shared_initial_stage_latency": {
            "definition": "one_measured_physical_initial_call_shared_by_all_arms",
            "physical_call_count": len(shared_initial_calls),
            "percentile_method": "nearest_rank",
            "end_to_end_latency_ms_sum": sum(shared_end_to_end),
            "end_to_end_p50_ms": _nearest_rank(shared_end_to_end, 0.50),
            "end_to_end_p95_ms": _nearest_rank(shared_end_to_end, 0.95),
            "synchronized_inference_latency_ms_sum": _nullable_sum(
                [call["synchronized_latency_ms"] for call in shared_initial_calls],
                "shared initial inference latency",
            ),
        },
    }
    physical["format_failure_rate"] = physical["format_failure_calls"] / len(telemetry)
    physical["generation_failure_rate"] = (
        physical["generation_failure_calls"] / len(telemetry)
    )
    return {arm: {"rows": rows} for arm, rows in arm_rows.items()}, physical


def _nearest_rank(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    if not 0.0 < fraction <= 1.0:
        raise ValueError("nearest-rank fraction must be in (0, 1]")
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return float(ordered[index])


def _arm_efficiency(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    summaries = [
        _require_mapping(row.get("summary"), "internal arm summary") for row in rows
    ]
    logical_call_rows = [
        call
        for row in rows
        for call in row.get("logical_calls", [])
        if isinstance(call, dict)
    ]
    logical_calls = sum(
        _require_int(summary.get("logical_call_count"), "logical calls")
        for summary in summaries
    )
    if len(logical_call_rows) != logical_calls:
        raise AgentABSummaryError("internal logical-call accounting mismatch")
    inference_latencies = [
        _require_nonnegative_number(
            summary.get("logical_inference_latency_ms"),
            "logical latency",
            nullable=True,
        )
        for summary in summaries
    ]
    active_latencies = [
        _require_nonnegative_number(
            summary.get("logical_end_to_end_latency_ms"),
            "logical end-to-end latency",
            nullable=True,
        )
        for summary in summaries
    ]
    complete_latencies = all(value is not None for value in active_latencies)
    final_format_failures = sum(
        summary.get("failure_code") not in {None, "generation_failed"}
        for summary in summaries
    )
    final_generation_failures = sum(
        summary.get("failure_code") == "generation_failed" for summary in summaries
    )
    logical_format_failures = sum(
        bool(call["generation_succeeded"]) and not bool(call["parse"]["valid"])
        for call in logical_call_rows
    )
    logical_generation_failures = sum(
        not bool(call["generation_succeeded"]) for call in logical_call_rows
    )
    return {
        "case_count": len(summaries),
        "logical_call_count": logical_calls,
        "input_tokens": _nullable_sum(
            [summary.get("input_tokens") for summary in summaries],
            "arm input tokens",
        ),
        "output_tokens": _nullable_sum(
            [summary.get("output_tokens") for summary in summaries],
            "arm output tokens",
        ),
        "total_tokens": _nullable_sum(
            [summary.get("total_tokens") for summary in summaries],
            "arm total tokens",
        ),
        "active_path_latency": {
            "definition": "sum_of_end_to_end_generation_stages_with_shared_initial",
            "composed_from_shared_initial": True,
            "independent_wall_clock_measurement": False,
            "percentile_method": "nearest_rank",
            "complete_case_count": len(summaries) if complete_latencies else 0,
            "p50_ms": (
                _nearest_rank(
                    [
                        float(value)
                        for value in active_latencies
                        if value is not None
                    ],
                    0.50,
                )
                if complete_latencies
                else None
            ),
            "p95_ms": (
                _nearest_rank(
                    [
                        float(value)
                        for value in active_latencies
                        if value is not None
                    ],
                    0.95,
                )
                if complete_latencies
                else None
            ),
        },
        "synchronized_inference_latency_ms_sum": _nullable_sum(
            inference_latencies, "arm synchronized inference latency"
        ),
        "final_format_failure_count": final_format_failures,
        "final_format_failure_rate": final_format_failures / len(summaries),
        "final_generation_failure_count": final_generation_failures,
        "final_generation_failure_rate": final_generation_failures / len(summaries),
        "logical_format_failure_call_count": logical_format_failures,
        "logical_format_failure_call_rate": logical_format_failures / logical_calls,
        "logical_generation_failure_call_count": logical_generation_failures,
        "logical_generation_failure_call_rate": (
            logical_generation_failures / logical_calls
        ),
        "valid_final_count": sum(bool(summary.get("valid")) for summary in summaries),
        "valid_final_rate": sum(
            bool(summary.get("valid")) for summary in summaries
        )
        / len(summaries),
        "external_api_cost_usd": 0.0,
        "compute_cost_usd": None,
        "total_cost_usd": None,
    }


def _single_claim_counts(
    data: ValidatedSciFact,
    claim: dict[str, Any],
    prediction: dict[str, Any],
    metric_name: str,
) -> tuple[int, int, int]:
    group = metric_name.removesuffix("_f1")
    one = ValidatedSciFact(
        corpus=data.corpus,
        claims=(claim,),
        claim_ids=(claim["id"],),
    )
    metrics = compute_official_pipeline_metrics([prediction], one)[group]
    if group.startswith("abstract_"):
        relevant = len(claim["evidence"])
        retrieved = len(prediction["evidence"])
    else:
        relevant = sum(
            len(rationale["sentences"])
            for rationales in claim["evidence"].values()
            for rationale in rationales
        )
        retrieved = sum(
            len(document["sentences"])
            for document in prediction["evidence"].values()
        )
    if retrieved:
        correct = round(metrics["precision"] * retrieved)
    elif relevant:
        correct = round(metrics["recall"] * relevant)
    else:
        correct = 0
    expected_f1 = (
        2.0 * correct / (retrieved + relevant) if retrieved + relevant else 0.0
    )
    if abs(expected_f1 - metrics["f1"]) > 1e-12:
        raise AgentABSummaryError("could not recover additive official score counts")
    return correct, retrieved, relevant


def _bootstrap_delta(
    data: ValidatedSciFact,
    baseline_rows: Sequence[dict[str, Any]],
    candidate_rows: Sequence[dict[str, Any]],
    *,
    metric_name: str,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    baseline = [
        _single_claim_counts(data, claim, prediction, metric_name)
        for claim, prediction in zip(data.claims, baseline_rows, strict=True)
    ]
    candidate = [
        _single_claim_counts(data, claim, prediction, metric_name)
        for claim, prediction in zip(data.claims, candidate_rows, strict=True)
    ]
    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(samples):
        indices = [rng.randrange(len(data.claims)) for _ in data.claims]
        scores: list[float] = []
        for contributions in (baseline, candidate):
            correct = sum(contributions[index][0] for index in indices)
            retrieved = sum(contributions[index][1] for index in indices)
            relevant = sum(contributions[index][2] for index in indices)
            scores.append(
                2.0 * correct / (retrieved + relevant)
                if retrieved + relevant
                else 0.0
            )
        deltas.append(scores[1] - scores[0])
    return {
        "unit": "claim",
        "paired": True,
        "method": "percentile_nearest_rank",
        "samples": samples,
        "seed": seed,
        "delta_direction": f"{MAIN_CANDIDATE}_minus_{MAIN_BASELINE}",
        "ci95": [
            _nearest_rank(deltas, 0.025),
            _nearest_rank(deltas, 0.975),
        ],
    }


def summarize(
    *,
    runner_dir: Path,
    corpus_path: Path,
    claims_path: Path,
    retrieval_path: Path,
    score_directories: Mapping[str, Path],
    output: Path,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> Path:
    """Validate every input and exclusively materialize comparison artifacts."""

    if set(score_directories) != set(ARMS):
        raise AgentABSummaryError(
            "exactly three named official score directories required"
        )
    if bootstrap_samples <= 0:
        raise AgentABSummaryError("bootstrap sample count must be positive")
    run_root = runner_dir.resolve()
    corpus = corpus_path.resolve()
    claims = claims_path.resolve()
    retrieval = retrieval_path.resolve()
    destination = output.resolve()
    manifest_path = run_root / "run_manifest.json"
    runner_manifest = _read_json(manifest_path)
    if runner_manifest.get("schema_version") != "scifact-agent-ab-run-v1":
        raise AgentABSummaryError("runner manifest schema mismatch")
    status = _require_mapping(runner_manifest.get("evaluation_status"), "runner status")
    if status != {
        "reportable": True,
        "state": "ready_for_official_scoring",
        "gold_used_by_runner": False,
    }:
        raise AgentABSummaryError("runner is not a complete reportable gold-free run")
    execution = _require_mapping(runner_manifest.get("execution"), "runner execution")
    if execution.get("completed") is not True:
        raise AgentABSummaryError("runner did not complete")
    case_count = _require_int(
        execution.get("selected_claim_count"), "case count", minimum=1
    )
    expected_physical = case_count * 3
    if execution.get("physical_call_count") != expected_physical or execution.get(
        "expected_physical_call_count"
    ) != expected_physical:
        raise AgentABSummaryError("runner physical-call count mismatch")
    wall_time_ms = _require_nonnegative_number(
        execution.get("wall_time_ms"), "runner wall time", nullable=True
    )
    resume_used = execution.get("resume_used")
    if not isinstance(resume_used, bool):
        raise AgentABSummaryError("runner resume marker is invalid")
    expected_wall_scope = (
        "unavailable_across_resumed_processes"
        if resume_used
        else "run_entry_through_pre_materialization"
    )
    if execution.get("wall_time_scope") != expected_wall_scope:
        raise AgentABSummaryError("runner wall-time scope mismatch")
    if (wall_time_ms is None) != resume_used:
        raise AgentABSummaryError("runner wall-time availability mismatch")
    comparison_contract = _require_mapping(
        runner_manifest.get("comparison"), "runner comparison contract"
    )
    if (
        comparison_contract.get("main")
        != "judge_challenger_vs_single_self_review"
        or comparison_contract.get("reference_only") != "single_one_pass"
        or comparison_contract.get("shared_initial_call") is not True
        or comparison_contract.get("main_arms_logical_calls_per_case") != 2
        or comparison_contract.get("same_model_top_k_context_and_output_budget")
        is not True
        or comparison_contract.get(
            "branch_execution_alternated_by_case_index_parity"
        )
        is not True
    ):
        raise AgentABSummaryError("runner comparison contract mismatch")
    accounting = _require_mapping(
        runner_manifest.get("accounting"), "runner accounting"
    )
    if (
        accounting.get("physical_calls_count_shared_initial_once") is not True
        or accounting.get("logical_arm_totals_attribute_shared_initial_to_each_arm")
        is not True
        or accounting.get("external_api_cost_usd") != 0.0
        or accounting.get("compute_cost_usd") is not None
    ):
        raise AgentABSummaryError("runner shared-call/cost accounting mismatch")
    configuration_lock = _require_mapping(
        runner_manifest.get("configuration_lock"), "runner configuration lock"
    )
    protocol = configuration_lock.get("agent_protocol_version")
    if protocol not in {None, V2_AGENT_PROTOCOL}:
        raise AgentABSummaryError("unknown Agent A/B decision protocol")
    if protocol == V2_AGENT_PROTOCOL:
        scripts_directory = Path(__file__).resolve().parent
        v2_runner = scripts_directory / "compat" / "run_scifact_hf_agent_ab_v2.py"
        base_runner = scripts_directory / "compat" / "run_scifact_hf_agent_ab.py"
        dependencies = _require_mapping(
            configuration_lock.get("implementation_dependencies"),
            "v2 runner implementation dependencies",
        )
        if (
            configuration_lock.get("script_sha256") != _sha256_file(v2_runner)
            or dependencies
            != {
                "audited_runner_base": {
                    "path": str(base_runner.resolve()),
                    "sha256": _sha256_file(base_runner),
                }
            }
        ):
            raise AgentABSummaryError("v2 runner/base implementation hash mismatch")
    generation = _require_mapping(
        configuration_lock.get("generation"), "runner generation lock"
    )
    max_new_tokens = _require_int(
        generation.get("max_new_tokens_per_call"),
        "max new tokens per call",
        minimum=1,
    )
    initial_prefill = (
        V2_INITIAL_DRAFT_PREFILL
        if protocol == V2_AGENT_PROTOCOL
        else INITIAL_DRAFT_PREFILL
    )
    final_prefill = (
        V2_FINAL_EVIDENCE_PREFILL
        if protocol == V2_AGENT_PROTOCOL
        else FINAL_EVIDENCE_PREFILL
    )
    stopping_policy = (
        V2_JSON_ROOT_STOPPING_POLICY
        if protocol == V2_AGENT_PROTOCOL
        else JSON_ROOT_STOPPING_POLICY
    )
    expected_projection = (
        "validated_explicit_label_and_ranked_citations_to_evidence"
        if protocol == V2_AGENT_PROTOCOL
        else "validated_candidates_to_one_pass_evidence_preserving_order"
    )
    expected_prefills = {
        "shared_initial_judge": {
            "value": initial_prefill,
            "sha256": _sha256_bytes(initial_prefill.encode("utf-8")),
        },
        "self_review_and_challenger": {
            "value": final_prefill,
            "sha256": _sha256_bytes(final_prefill.encode("utf-8")),
        },
    }
    if (
        generation.get("calls_per_case_physical") != 3
        or generation.get("initial_call_shared") is not True
        or generation.get("logical_calls_single_one_pass") != 1
        or generation.get("logical_calls_single_self_review") != 2
        or generation.get("logical_calls_judge_challenger") != 2
        or generation.get("do_sample") is not False
        or generation.get("temperature") is not None
        or generation.get("top_p") is not None
        or generation.get("top_k") is not None
        or generation.get("num_beams") != 1
        or generation.get("batch_size") != 1
        or generation.get("assistant_prefills") != expected_prefills
        or generation.get("assistant_prefill_accounting")
        != "rendered_prompt_input_tokens"
        or generation.get("chat_template_mode") != "continue_final_message"
        or generation.get("stopping_policy") != stopping_policy
        or generation.get("stopping_policy_sha256")
        != _sha256_bytes(stopping_policy.encode("utf-8"))
        or generation.get("stopping_policy_posthoc_truncation") is not False
        or generation.get("application_response_cache") is not False
        or generation.get("retry_count") != 0
        or generation.get("initial_draft_projection") != expected_projection
    ):
        raise AgentABSummaryError("runner generation fairness lock mismatch")
    if protocol == V2_AGENT_PROTOCOL and (
        generation.get("repetition_penalty") != 1.0
        or generation.get("repetition_penalty_source")
        != "explicit_protocol_override_of_model_generation_config"
    ):
        raise AgentABSummaryError("v2 repetition-penalty lock mismatch")
    retrieval_policy = _require_mapping(
        configuration_lock.get("retrieval_policy"), "runner retrieval policy"
    )
    if retrieval_policy != {
        "system": "official_scifact_tfidf",
        "top_k": 3,
        "document_context": "title_and_all_indexed_abstract_sentences",
        "branch_order": "even_case_index_self_then_challenger_else_reverse",
    }:
        raise AgentABSummaryError("runner retrieval fairness lock mismatch")
    parser_policies = _require_mapping(
        configuration_lock.get("parser_policies"), "runner parser policies"
    )
    if set(parser_policies) != {"initial_draft", "final_evidence"}:
        raise AgentABSummaryError("runner parser schema set mismatch")
    parser_policies_hash = _sha256_bytes(
        _canonical_json(parser_policies).encode("utf-8")
    )
    if configuration_lock.get("parser_policies_sha256") != parser_policies_hash:
        raise AgentABSummaryError("runner parser-policy bundle hash mismatch")
    parser_policy_hashes = {
        name: _sha256_bytes(_canonical_json(value).encode("utf-8"))
        for name, value in parser_policies.items()
    }
    draft_parser = _require_mapping(
        parser_policies["initial_draft"], "runner initial-draft parser"
    )
    final_parser = _require_mapping(
        parser_policies["final_evidence"], "runner final-evidence parser"
    )
    if protocol == V2_AGENT_PROTOCOL:
        if parser_policies != V2_PARSER_POLICIES:
            raise AgentABSummaryError("v2 runner parser/failure policy mismatch")
    elif (
        draft_parser.get("version") != "scifact-agent-draft-parser-v1"
        or draft_parser.get("analysis_nonempty") is not True
        or draft_parser.get("analysis_max_words") is not None
        or draft_parser.get("candidate_shape")
        != ["retrieval_rank", "label", "sentence_indices"]
        or draft_parser.get("unique_candidate_ranks") is not True
        or draft_parser.get("retrieval_rank_scope") != [1, 3]
        or draft_parser.get("rank_mapping")
        != "one_based_rank_to_retrieved_top3_document"
        or draft_parser.get("labels") != ["SUPPORT", "CONTRADICT", "NEI"]
        or draft_parser.get("nei_candidate_sentence_indices")
        != "optional_but_strictly_scope_validated"
        or draft_parser.get("nei_mixed_with_non_nei") != "reject"
        or draft_parser.get("nei_projection")
        != "ignored_when_mapping_to_evidence"
        or draft_parser.get("semantic_repairs") != 0
        or draft_parser.get("invalid_draft_fallback")
        != "fixed_draft_unavailable_sentinel"
        or draft_parser.get("one_pass_projection")
        != "candidate_order_to_evidence_object_without_repair"
        or final_parser.get("version") != "scifact-agent-output-parser-v1"
        or final_parser.get("semantic_repairs") != 0
        or final_parser.get("retries") != 0
        or final_parser.get("invalid_final_fallback") != "empty_evidence"
    ):
        raise AgentABSummaryError("runner parser/failure policy mismatch")

    runner_hashes = _verified_owned_outputs(
        run_root, runner_manifest.get("outputs_sha256"), name="runner"
    )
    if not REQUIRED_RUNNER_OUTPUTS.issubset(runner_hashes):
        raise AgentABSummaryError("runner outputs are incomplete")
    input_lock = _require_mapping(
        runner_manifest.get("input_lock"), "runner input lock"
    )
    if (
        input_lock.get("corpus_sha256") != _sha256_file(corpus)
        or input_lock.get("claims_sha256") != _sha256_file(claims)
        or input_lock.get("retrieval_sha256") != _sha256_file(retrieval)
    ):
        raise AgentABSummaryError("runner input hashes do not match supplied data")
    split = input_lock.get("split")
    if split != "dev":
        raise AgentABSummaryError("reportable Agent A/B summary requires dev")
    frozen_provenance = _validate_frozen_config_binding(
        runner_manifest.get("frozen_config"), configuration_lock, input_lock
    )
    retrieval_provenance = _validate_retrieval_binding(
        retrieval,
        split=split,
        corpus_path=corpus,
        claims_path=claims,
        expected_manifest_sha256=input_lock.get("retrieval_manifest_sha256"),
    )

    corpus_rows = _read_jsonl(corpus)
    claim_rows = _read_jsonl(claims)
    data = validate_official_data(corpus_rows, claim_rows)
    if case_count != len(data.claims):
        raise AgentABSummaryError("runner does not cover the complete scored split")
    expected_ids_hash = _sha256_bytes(
        _canonical_json(list(data.claim_ids)).encode("utf-8")
    )
    if execution.get("selected_claim_ids_sha256") != expected_ids_hash:
        raise AgentABSummaryError("runner selected claim order hash mismatch")
    expected_header = {
        "schema_version": "scifact-agent-ab-incomplete-run-v1",
        "configuration_lock": configuration_lock,
        "input_lock": input_lock,
        "selected_claim_ids": list(data.claim_ids),
        "frozen_config_sha256": frozen_provenance["sha256"],
    }
    if _read_json(run_root / "run_header.json") != expected_header:
        raise AgentABSummaryError("runner header differs from frozen run contract")
    retrieval_by_id = _load_retrieval(retrieval, data.claim_ids)
    contexts = _read_jsonl(run_root / "contexts.jsonl")
    cases = _read_jsonl(run_root / "cases.jsonl")
    telemetry = _read_jsonl(run_root / "raw_telemetry.jsonl")
    allowed_by_claim = _validate_contexts(
        contexts, cases, data, retrieval_by_id
    )
    arm_rows, physical = _validate_cases_and_telemetry(
        cases,
        telemetry,
        allowed_by_claim,
        parser_policy_hashes,
        stopping_policy,
        protocol,
    )
    if any(
        call["output_tokens"] is not None
        and call["output_tokens"] > max_new_tokens
        for call in telemetry
    ):
        raise AgentABSummaryError(
            "observed output tokens exceed the frozen call budget"
        )
    observed_active_time = physical["active_end_to_end_latency_ms_sum"]
    manifest_active_time = _require_nonnegative_number(
        execution.get("active_physical_call_end_to_end_time_ms"),
        "manifest active physical time",
        nullable=True,
    )
    if manifest_active_time != observed_active_time:
        raise AgentABSummaryError("manifest active physical time mismatch")

    predictions: dict[str, list[dict[str, Any]]] = {}
    official_by_arm: dict[str, dict[str, float]] = {}
    score_provenance: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        prediction_path = run_root / PREDICTION_FILES[arm]
        rows = _read_jsonl(prediction_path)
        ordered = list(validate_pipeline_predictions(rows, data))
        if [row["id"] for row in ordered] != list(data.claim_ids):
            raise AgentABSummaryError(f"{arm} prediction order mismatch")
        for case, row in zip(cases, ordered, strict=True):
            if row["evidence"] != case["arms"][arm]["final_evidence"]:
                raise AgentABSummaryError(f"{arm} prediction differs from case record")
        predictions[arm] = ordered
        official, provenance = _load_official_score(
            score_directories[arm].resolve(),
            arm=arm,
            prediction_path=prediction_path,
            split=split,
            claim_count=len(data.claims),
            corpus_count=len(data.corpus),
            corpus_sha256=_sha256_file(corpus),
            claims_sha256=_sha256_file(claims),
        )
        independent = _flatten_metrics(compute_official_pipeline_metrics(ordered, data))
        if not _same_metrics(official, independent):
            raise AgentABSummaryError(f"{arm} score differs from current predictions")
        official_by_arm[arm] = official
        score_provenance[arm] = provenance

    deltas = {
        metric: official_by_arm[MAIN_CANDIDATE][metric]
        - official_by_arm[MAIN_BASELINE][metric]
        for metric in OFFICIAL_METRIC_NAMES
    }
    bootstrap = {}
    for metric in BOOTSTRAP_METRICS:
        result = _bootstrap_delta(
            data,
            predictions[MAIN_BASELINE],
            predictions[MAIN_CANDIDATE],
            metric_name=metric,
            samples=bootstrap_samples,
            seed=bootstrap_seed,
        )
        result["observed_delta"] = deltas[metric]
        bootstrap[metric] = result

    efficiency_by_arm = {
        arm: _arm_efficiency(arm_rows[arm]["rows"]) for arm in ARMS
    }
    comparison = {
        "schema_version": "scifact-agent-ab-comparison-v1",
        "evaluation_status": {
            "reportable": True,
            "official_scores_cross_checked": True,
            "complete_failure_denominator": True,
        },
        "dataset": {
            "split": split,
            "claim_count": len(data.claims),
            "corpus_count": len(data.corpus),
        },
        "comparison": {
            "main_baseline": MAIN_BASELINE,
            "main_candidate": MAIN_CANDIDATE,
            "delta_direction": f"{MAIN_CANDIDATE}_minus_{MAIN_BASELINE}",
            "reference_only": "single_one_pass",
            "single_one_pass_definition": (
                "deterministic_projection_of_valid_shared_initial_candidates"
            ),
            "shared_initial_call": True,
            "main_arms_logical_calls_per_case": 2,
            "physical_calls_per_case": 3,
            "max_new_tokens_per_physical_call": max_new_tokens,
            "main_arm_max_new_tokens_per_case": max_new_tokens * 2,
        },
        "official_quality": {
            "metrics_by_arm": official_by_arm,
            "main_delta_all_official_metrics": deltas,
            "paired_bootstrap": bootstrap,
        },
        "efficiency": {
            "logical_arms": efficiency_by_arm,
            "physical_run": physical,
            "entire_run_wall_time_ms": wall_time_ms,
            "entire_run_wall_time_seconds": (
                wall_time_ms / 1000.0 if wall_time_ms is not None else None
            ),
            "entire_run_wall_time_scope": execution["wall_time_scope"],
            "latency_interpretation": (
                "Arm p50/p95 are nearest-rank percentiles of per-case active-path "
                "sums. The same measured initial stage is attributed to both main "
                "arms; these are not independent wall-clock arm executions."
            ),
        },
        "cost": {
            "external_model_api_calls": 0,
            "external_api_spend_usd": 0.0,
            "compute_cost_usd": None,
            "total_cost_usd": None,
            "note": "Local electricity and hardware depreciation were not measured.",
        },
        "failure_accounting": {
            "invalid_outputs_remained_in_all_official_prediction_files": True,
            "invalid_final_prediction_fallback": "empty_evidence",
            "invalid_initial_draft_branch_input": DRAFT_UNAVAILABLE,
            "physical_format_failure_rate": physical["format_failure_rate"],
            "final_format_failure_rate_by_arm": {
                arm: efficiency_by_arm[arm]["final_format_failure_rate"] for arm in ARMS
            },
            "logical_format_failure_call_rate_by_arm": {
                arm: efficiency_by_arm[arm]["logical_format_failure_call_rate"]
                for arm in ARMS
            },
        },
    }

    try:
        destination.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise AgentABSummaryError(
            f"refusing to overwrite output directory: {destination}"
        ) from exc
    comparison_path = destination / "comparison.json"
    _write_exclusive(comparison_path, _json_bytes(comparison))
    summary_manifest = {
        "schema_version": "scifact-agent-ab-summary-manifest-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "inputs": {
            "runner": {
                "directory": str(run_root),
                "manifest_sha256": _sha256_file(manifest_path),
                "verified_outputs_sha256": runner_hashes,
                "frozen_config": frozen_provenance,
            },
            "data": {
                "corpus_sha256": _sha256_file(corpus),
                "claims_sha256": _sha256_file(claims),
                "retrieval_sha256": _sha256_file(retrieval),
                "retrieval_manifest_sha256": retrieval_provenance[
                    "manifest_sha256"
                ],
            },
            "official_scores": score_provenance,
        },
        "configuration": {
            "main_comparison": f"{MAIN_CANDIDATE}_vs_{MAIN_BASELINE}",
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": bootstrap_seed,
            "bootstrap_metric_seed_rule": "same_claim_resample_seed_for_each_metric",
            "percentile_method": "nearest_rank",
        },
        "validation": {
            "all_runner_output_hashes_verified": True,
            "exact_context_case_telemetry_prediction_coverage": True,
            "context_reconstructed_from_claims_retrieval_and_corpus": True,
            "all_official_score_hashes_verified": True,
            "official_and_independent_metrics_equal": True,
            "shared_initial_logical_and_physical_accounting_reconciled": True,
            "frozen_train_to_dev_contract_verified": True,
        },
        "outputs_sha256": {"comparison.json": _sha256_file(comparison_path)},
    }
    _write_exclusive(destination / "manifest.json", _json_bytes(summary_manifest))
    return destination


def _parse_score_directories(values: Sequence[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        arm, separator, raw_path = value.partition("=")
        if not separator or arm not in ARMS or not raw_path or arm in parsed:
            raise AgentABSummaryError("--score must be unique ARM=DIRECTORY entries")
        parsed[arm] = Path(raw_path)
    if set(parsed) != set(ARMS):
        raise AgentABSummaryError("--score is required once for every arm")
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner-dir", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--claims", type=Path, required=True)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument(
        "--score",
        action="append",
        required=True,
        metavar="ARM=DIRECTORY",
        help="repeat for single_one_pass, single_self_review, judge_challenger",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES
    )
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        result = summarize(
            runner_dir=args.runner_dir,
            corpus_path=args.corpus,
            claims_path=args.claims,
            retrieval_path=args.retrieval,
            score_directories=_parse_score_directories(args.score),
            output=args.out,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        )
    except AgentABSummaryError as exc:
        raise SystemExit(f"summary failed: {exc}") from exc
    print(result)


if __name__ == "__main__":
    main()
