"""Run the non-degenerate SciFact single-vs-role-separated Agent A/B protocol.

This v2 entrypoint deliberately reuses the audited data loading, local model,
journaling, and accounting implementation from ``run_scifact_hf_agent_ab.py``.
The v1 entrypoint remains unchanged because its SHA256 is part of the failed v3
run's immutable audit trail.  This file replaces only the decision protocol,
strict parsers, JSON stopping rule, freeze validation, and their configuration
locks.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Optional

_ENTRYPOINT = Path(__file__).resolve()
_BASE_PATH = _ENTRYPOINT.with_name("run_scifact_hf_agent_ab.py")
_SPEC = importlib.util.spec_from_file_location(
    "scifact_hf_agent_ab_protocol_v1_base", _BASE_PATH
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load protocol base: {_BASE_PATH}")
_base = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _base
_SPEC.loader.exec_module(_base)

AgentABError = _base.AgentABError
DuplicateJSONKey = _base.DuplicateJSONKey
TOP_K = _base.TOP_K
DEFAULT_MAX_NEW_TOKENS = _base.DEFAULT_MAX_NEW_TOKENS
DEFAULT_SEED = _base.DEFAULT_SEED
ARMS = _base.ARMS
JSON_ROOT_STOPPING_POLICY = (
    "during_generation_first_complete_type_matched_json_root_object_closure"
)
AGENT_PROTOCOL_VERSION = "scifact-agent-decision-v2"
REPETITION_PENALTY = 1.0
MAIN_ARMS = ("single_self_review", "judge_challenger")
EXPECTED_TRAIN_GATE_THRESHOLDS = {
    "minimum_format_valid_rate_all_arms": 0.99,
    "minimum_nonempty_prediction_rate_main_arms": 0.05,
    "minimum_distinct_final_labels_main_arms": 2,
    "minimum_official_train_abstract_rationalized_f1_main_arms": 0.05,
    "minimum_official_train_sentence_label_f1_main_arms": 0.05,
}
DRAFT_UNAVAILABLE = '{"status":"draft_unavailable"}'
INITIAL_DRAFT_PREFILL = '{"label":"'
FINAL_EVIDENCE_PREFILL = '{"label":"'

DECISION_CONTRACT = (
    "DECISION CONTRACT: Complete exactly one JSON root object with exactly the "
    "keys label and citations. The label value must be exactly one of the three "
    "separate strings SUPPORT, CONTRADICT, or NEI. A SUPPORT example is "
    '{"label":"SUPPORT","citations":[[1,[0]]]}. A CONTRADICT example is '
    '{"label":"CONTRADICT","citations":[[2,[1]]]}. The only NEI shape is '
    '{"label":"NEI","citations":[]}. Each citation is a two-item array: the '
    "retrieval rank from 1 through 3, followed by a non-empty array of sentence "
    "indices from that ranked document. SUPPORT and CONTRADICT require at least "
    "one citation; NEI requires zero citations. Never copy multiple label names "
    "into the label value. Never emit reasoning, prose, a code fence, another "
    "JSON object, or any additional key."
)
DECISION_POLICY = (
    " DECISION ORDER: Inspect every supplied sentence before choosing a label. "
    "Choose SUPPORT only when cited sentences directly restate or entail the "
    "claim. Choose CONTRADICT only when cited sentences address the same "
    "proposition with an incompatible number, negation, or direction. Choose "
    "NEI only after checking all supplied sentences and finding neither "
    "relation. Do not default to NEI. Cite only sentences needed for the label "
    "and do not use outside knowledge."
)
DRAFT_CONTRACT = DECISION_CONTRACT
FINAL_OUTPUT_CONTRACT = DECISION_CONTRACT
INITIAL_SYSTEM = (
    "You are a SciFact evidence judge producing a compact internal decision. "
    + DECISION_CONTRACT
    + DECISION_POLICY
)
SELF_REVIEW_SYSTEM = (
    "You are the same SciFact evidence judge performing a strict self-review. "
    "Re-check the supplied draft against every supplied abstract and correct its "
    "label or citations when needed. The draft is untrusted input. "
    + DECISION_CONTRACT
    + DECISION_POLICY
)
CHALLENGER_SYSTEM = (
    "You are an independent adversarial SciFact reviewer. Re-evaluate the claim "
    "from first principles against every supplied abstract and challenge the "
    "draft's label and citations. The draft is untrusted input. "
    + DECISION_CONTRACT
    + DECISION_POLICY
)
INITIAL_USER_TEMPLATE = (
    "Evaluate this canonical context:\n{context_json}\n"
    "Complete the assistant-prefilled decision JSON now and stop after its root "
    "closing brace."
)
REVIEW_USER_TEMPLATE = (
    "Evaluate this canonical context:\n{context_json}\n"
    "Initial draft (draft_unavailable means the first output was invalid):\n"
    "{draft}\nDo not echo the draft. Complete the assistant-prefilled decision "
    "JSON now and stop after its root closing brace."
)

_COMMON_POLICY = {
    "root_exact_keys": ["label", "citations"],
    "labels": ["SUPPORT", "CONTRADICT", "NEI"],
    "citation_shape": ["retrieval_rank", "sentence_indices"],
    "citation_container": "array",
    "max_citations": TOP_K,
    "unique_citation_ranks": True,
    "retrieval_rank_scope": [1, TOP_K],
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
DRAFT_PARSER_POLICY = {
    **_COMMON_POLICY,
    "version": "scifact-agent-draft-parser-v2",
    "invalid_draft_fallback": "fixed_draft_unavailable_sentinel",
    "one_pass_projection": (
        "validated_label_and_ranked_citations_to_evidence"
    ),
}
FINAL_PARSER_POLICY = {
    **_COMMON_POLICY,
    "version": "scifact-agent-output-parser-v2",
    "invalid_final_fallback": "empty_evidence",
}
PARSER_POLICIES = {
    "initial_draft": DRAFT_PARSER_POLICY,
    "final_evidence": FINAL_PARSER_POLICY,
}

_sha256_text = _base._sha256_text
_sha256_file = _base._sha256_file
_canonical_json = _base._canonical_json
_duplicate_guard = _base._duplicate_guard
_reject_constant = _base._reject_constant
_BASE_CONFIGURATION_LOCK = _base._configuration_lock


class LocalGenerator(_base.LocalGenerator):
    """Use an explicitly locked neutral repetition penalty for classification."""

    def __init__(self, model_path: Path, max_new_tokens: int, seed: int) -> None:
        super().__init__(model_path, max_new_tokens, seed)
        self.model.generation_config.repetition_penalty = REPETITION_PENALTY

    def generate(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        # Re-assert immediately before every call so neither model defaults nor
        # an earlier caller can silently restore Qwen's packaged value (1.1).
        self.model.generation_config.repetition_penalty = REPETITION_PENALTY
        record = super().generate(*args, **kwargs)
        record["generation_overrides"] = {
            "repetition_penalty": REPETITION_PENALTY
        }
        return record


def _json_root_closed(value: str) -> bool:
    """Return true only after a type-matched root JSON object has closed."""

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


def _invalid_parse(
    failure_code: str,
    transport_unwrapped: bool,
    detail: Optional[str] = None,  # noqa: UP045
) -> dict[str, Any]:
    return {
        "valid": False,
        "failure_code": failure_code,
        "transport_unwrapped": transport_unwrapped,
        "value": None,
        "projected_evidence": None,
        "detail": detail,
    }


def _parse_decision(
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
    except (json.JSONDecodeError, DuplicateJSONKey, ValueError) as exc:
        return _invalid_parse(
            "invalid_draft_json" if draft else "invalid_json",
            transport_unwrapped,
            str(exc)[:300],
        )
    if not isinstance(value, dict) or set(value) != {"label", "citations"}:
        return _invalid_parse(
            "invalid_draft_root_schema" if draft else "invalid_root_schema",
            transport_unwrapped,
        )
    label = value["label"]
    if not isinstance(label, str) or label not in {
        "SUPPORT",
        "CONTRADICT",
        "NEI",
    }:
        return _invalid_parse(
            "invalid_draft_label" if draft else "invalid_label",
            transport_unwrapped,
        )
    citations = value["citations"]
    if not isinstance(citations, list) or len(citations) > TOP_K:
        return _invalid_parse(
            f"invalid_{prefix}citations", transport_unwrapped
        )
    if label == "NEI" and citations:
        return _invalid_parse(f"{prefix}nei_with_citations", transport_unwrapped)
    if label != "NEI" and not citations:
        return _invalid_parse(
            f"{prefix}relation_without_citations", transport_unwrapped
        )
    ranked_doc_ids = list(allowed_documents)
    if len(ranked_doc_ids) != TOP_K:
        raise AgentABError("decision parser requires exactly three ranked documents")
    seen_ranks: set[int] = set()
    projected: dict[str, Any] = {}
    for citation in citations:
        if not isinstance(citation, list) or len(citation) != 2:
            return _invalid_parse(
                f"invalid_{prefix}citation_schema", transport_unwrapped
            )
        rank, sentences = citation
        if type(rank) is not int or not 1 <= rank <= TOP_K:
            return _invalid_parse(
                f"{prefix}citation_rank_out_of_scope", transport_unwrapped
            )
        if rank in seen_ranks:
            return _invalid_parse(
                f"duplicate_{prefix}citation_rank", transport_unwrapped
            )
        seen_ranks.add(rank)
        if not isinstance(sentences, list) or not sentences:
            return _invalid_parse(
                f"invalid_{prefix}citation_sentences", transport_unwrapped
            )
        if any(type(index) is not int for index in sentences):
            return _invalid_parse(
                f"invalid_{prefix}citation_sentence_index", transport_unwrapped
            )
        if len(set(sentences)) != len(sentences):
            return _invalid_parse(
                f"duplicate_{prefix}citation_sentence_index",
                transport_unwrapped,
            )
        doc_id = ranked_doc_ids[rank - 1]
        if any(
            index < 0 or index >= allowed_documents[doc_id]
            for index in sentences
        ):
            return _invalid_parse(
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


def parse_initial_draft(
    raw: str, allowed_documents: Mapping[int, int]
) -> dict[str, Any]:
    return _parse_decision(raw, allowed_documents, draft=True)


def parse_agent_output(
    raw: str, allowed_documents: Mapping[int, int]
) -> dict[str, Any]:
    return _parse_decision(raw, allowed_documents, draft=False)


def _draft_to_evidence(
    draft: Mapping[str, Any], ranked_doc_ids: Sequence[int]
) -> dict[str, Any]:
    evidence = {
        str(ranked_doc_ids[citation[0] - 1]): {
            "label": draft["label"],
            "sentences": citation[1],
        }
        for citation in draft["citations"]
    }
    return {
        "evidence": evidence,
    }


def _arm_summary(
    calls: list[dict[str, Any]], parsed: dict[str, Any]
) -> dict[str, Any]:
    if parsed["valid"]:
        evidence = parsed.get("projected_evidence")
        if evidence is None:
            evidence = parsed["value"]["evidence"]
    else:
        evidence = {}
    return {
        "valid": parsed["valid"],
        "failure_code": parsed["failure_code"],
        "transport_unwrapped": parsed["transport_unwrapped"],
        "final_evidence": evidence,
        "invalid_final_fallback_applied": not parsed["valid"],
        "logical_call_count": len(calls),
        "physical_call_ids": [call["call_id"] for call in calls],
        "input_tokens": _base._sum_nullable(calls, "input_tokens"),
        "output_tokens": _base._sum_nullable(calls, "output_tokens"),
        "total_tokens": _base._sum_nullable(calls, "total_tokens"),
        "logical_inference_latency_ms": _base._sum_nullable(
            calls, "synchronized_latency_ms"
        ),
        "logical_end_to_end_latency_ms": _base._sum_nullable(
            calls, "end_to_end_latency_ms"
        ),
        "external_api_cost_usd": 0.0,
        "compute_cost_usd": None,
    }


def _configuration_lock(
    _script_path: Path,
    model_lock: dict[str, Any],
    max_new_tokens: int,
    seed: int,
) -> dict[str, Any]:
    lock = _BASE_CONFIGURATION_LOCK(
        _ENTRYPOINT, model_lock, max_new_tokens, seed
    )
    lock["agent_protocol_version"] = AGENT_PROTOCOL_VERSION
    lock["implementation_dependencies"] = {
        "audited_runner_base": {
            "path": str(_BASE_PATH),
            "sha256": _sha256_file(_BASE_PATH),
        }
    }
    lock["generation"]["initial_draft_projection"] = (
        "validated_explicit_label_and_ranked_citations_to_evidence"
    )
    lock["generation"]["repetition_penalty"] = REPETITION_PENALTY
    lock["generation"]["repetition_penalty_source"] = (
        "explicit_protocol_override_of_model_generation_config"
    )
    return lock


def write_frozen_config(*_args: Any, **_kwargs: Any) -> Path:
    raise AgentABError(
        "protocol v2 disables the smoke-only freeze path; use "
        "freeze_scifact_agent_ab_config.py with full train scores"
    )


def _validate_frozen_config(
    path: Path,
    configuration_lock: dict[str, Any],
    input_lock: dict[str, Any],
) -> dict[str, Any]:
    frozen = _base._read_json(path)
    if frozen.get("schema_version") != "scifact-agent-ab-frozen-config-v2":
        raise AgentABError("protocol v2 requires a quality-gated frozen config")
    if frozen.get("locked_configuration") != configuration_lock:
        raise AgentABError("runtime configuration differs from frozen config")
    if frozen.get("dev_inputs") != input_lock:
        raise AgentABError("dev inputs differ from frozen config")
    gate = frozen.get("train_admission_gate")
    if (
        not isinstance(gate, dict)
        or set(gate) != {"schema_version", "status", "report"}
        or gate.get("schema_version")
        != "scifact-agent-ab-train-admission-v1"
        or gate.get("status") != "passed"
    ):
        raise AgentABError("frozen config has no passed train admission gate")
    if frozen.get("train_admission_gate_sha256") != _sha256_text(
        _canonical_json(gate)
    ):
        raise AgentABError("frozen train admission gate binding hash is invalid")
    report_binding = gate.get("report")
    if not isinstance(report_binding, dict) or set(report_binding) != {
        "path",
        "sha256",
    }:
        raise AgentABError("frozen config has no train gate report binding")
    report_path = Path(str(report_binding.get("path", "")))
    if not report_path.is_file() or _sha256_file(report_path) != report_binding.get(
        "sha256"
    ):
        raise AgentABError("frozen train gate report binding is invalid")
    report = _base._read_json(report_path)
    if (
        report.get("schema_version")
        != "scifact-agent-ab-train-admission-v1"
        or report.get("status") != "passed"
        or report.get("scope") != "complete_official_train_only"
        or report.get("agent_protocol_version") != AGENT_PROTOCOL_VERSION
        or report.get("dev_labels_or_metrics_inspected") is not False
    ):
        raise AgentABError("bound train admission gate contract is invalid")
    if report.get("thresholds") != EXPECTED_TRAIN_GATE_THRESHOLDS:
        raise AgentABError("bound train admission gate thresholds are invalid")
    observed = report.get("observed")
    if not isinstance(observed, dict) or set(observed) != set(ARMS):
        raise AgentABError("bound train admission label observations are invalid")
    expected_observed_keys = {
        "format_valid_count",
        "nonempty_prediction_count",
        "label_counts",
        "claim_count",
        "format_valid_rate",
        "nonempty_prediction_rate",
        "distinct_label_count",
    }
    for arm, arm_observed in observed.items():
        if not isinstance(arm_observed, dict) or set(arm_observed) != (
            expected_observed_keys
        ):
            raise AgentABError("bound train admission arm observation is invalid")
        label_counts = arm_observed.get("label_counts")
        if not isinstance(label_counts, dict) or set(label_counts) != {
            "SUPPORT",
            "CONTRADICT",
            "NEI",
        }:
            raise AgentABError("bound train admission label counts are invalid")
        integer_fields = [
            arm_observed.get("format_valid_count"),
            arm_observed.get("nonempty_prediction_count"),
            arm_observed.get("claim_count"),
            arm_observed.get("distinct_label_count"),
            *label_counts.values(),
        ]
        if any(type(value) is not int or value < 0 for value in integer_fields):
            raise AgentABError("bound train admission counts are invalid")
        claim_count = arm_observed["claim_count"]
        if (
            claim_count != 809
            or sum(label_counts.values())
            != arm_observed["format_valid_count"]
            or arm_observed["distinct_label_count"]
            != sum(count > 0 for count in label_counts.values())
            or arm_observed["format_valid_rate"]
            != arm_observed["format_valid_count"] / claim_count
            or arm_observed["nonempty_prediction_rate"]
            != arm_observed["nonempty_prediction_count"] / claim_count
            or arm_observed["format_valid_rate"]
            < EXPECTED_TRAIN_GATE_THRESHOLDS[
                "minimum_format_valid_rate_all_arms"
            ]
        ):
            raise AgentABError("bound train admission rates are inconsistent")
        if arm in MAIN_ARMS and (
            arm_observed["distinct_label_count"]
            < EXPECTED_TRAIN_GATE_THRESHOLDS[
                "minimum_distinct_final_labels_main_arms"
            ]
            or arm_observed["nonempty_prediction_rate"]
            < EXPECTED_TRAIN_GATE_THRESHOLDS[
                "minimum_nonempty_prediction_rate_main_arms"
            ]
        ):
            raise AgentABError("bound train admission main arm collapsed")
    train_binding = frozen.get("train_run_manifest")
    if (
        not isinstance(train_binding, dict)
        or train_binding.get("selected_claim_count") != 809
    ):
        raise AgentABError("frozen config lacks train binding")
    train_path = Path(str(train_binding.get("path", "")))
    if not train_path.is_file() or _sha256_file(train_path) != train_binding.get(
        "sha256"
    ):
        raise AgentABError("frozen train run binding is invalid")
    _base._verify_owned_outputs(train_path)
    evidence_hashes = report.get("evidence_sha256")
    if (
        not isinstance(evidence_hashes, dict)
        or evidence_hashes.get("train_manifest") != train_binding.get("sha256")
    ):
        raise AgentABError("train gate evidence differs from frozen train run")
    implementation = report.get("implementation")
    if not isinstance(implementation, dict) or set(implementation) != {
        "path",
        "sha256",
    }:
        raise AgentABError("train gate implementation binding is missing")
    gate_script = _ENTRYPOINT.parents[2] / str(implementation["path"])
    if (
        not gate_script.is_file()
        or _sha256_file(gate_script) != implementation["sha256"]
    ):
        raise AgentABError("train gate implementation binding is invalid")
    return frozen


def _install_protocol() -> None:
    replacements = {
        "DRAFT_UNAVAILABLE": DRAFT_UNAVAILABLE,
        "INITIAL_DRAFT_PREFILL": INITIAL_DRAFT_PREFILL,
        "FINAL_EVIDENCE_PREFILL": FINAL_EVIDENCE_PREFILL,
        "JSON_ROOT_STOPPING_POLICY": JSON_ROOT_STOPPING_POLICY,
        "DRAFT_CONTRACT": DRAFT_CONTRACT,
        "FINAL_OUTPUT_CONTRACT": FINAL_OUTPUT_CONTRACT,
        "DECISION_POLICY": DECISION_POLICY,
        "INITIAL_SYSTEM": INITIAL_SYSTEM,
        "SELF_REVIEW_SYSTEM": SELF_REVIEW_SYSTEM,
        "CHALLENGER_SYSTEM": CHALLENGER_SYSTEM,
        "INITIAL_USER_TEMPLATE": INITIAL_USER_TEMPLATE,
        "REVIEW_USER_TEMPLATE": REVIEW_USER_TEMPLATE,
        "DRAFT_PARSER_POLICY": DRAFT_PARSER_POLICY,
        "FINAL_PARSER_POLICY": FINAL_PARSER_POLICY,
        "PARSER_POLICIES": PARSER_POLICIES,
        "_json_root_closed": _json_root_closed,
        "parse_initial_draft": parse_initial_draft,
        "parse_agent_output": parse_agent_output,
        "_draft_to_evidence": _draft_to_evidence,
        "_arm_summary": _arm_summary,
        "LocalGenerator": LocalGenerator,
        "_configuration_lock": _configuration_lock,
        "write_frozen_config": write_frozen_config,
        "_validate_frozen_config": _validate_frozen_config,
    }
    for name, value in replacements.items():
        setattr(_base, name, value)


_install_protocol()


def run(args: Any) -> Path:
    return _base.run(args)


def _parse_args() -> Any:
    return _base._parse_args()


def main() -> None:
    try:
        print(run(_parse_args()))
    except AgentABError as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    main()
