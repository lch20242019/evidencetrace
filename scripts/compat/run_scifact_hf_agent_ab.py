"""Run a frozen, gold-free SciFact single-vs-multi-agent comparison locally.

This file intentionally targets Python 3.9 so the CUDA environment used by the
frozen SciFact reproduction can execute it without importing EvidenceTrace.
The official scorer is a separate, Python 3.11 step.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

TOP_K = 3
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_SEED = 20260904
ARMS = (
    "single_one_pass",
    "single_self_review",
    "judge_challenger",
)
DRAFT_UNAVAILABLE = '{"status":"draft_unavailable"}'
INITIAL_DRAFT_PREFILL = '{"analysis":"'
FINAL_EVIDENCE_PREFILL = '{"evidence":{'
JSON_ROOT_STOPPING_POLICY = (
    "during_generation_first_complete_json_root_object_closure"
)

DRAFT_CONTRACT = (
    "DRAFT CONTRACT: Complete exactly one JSON root object with exactly the keys "
    'analysis and candidates. The complete schema is {"analysis":"non-empty '
    'reasoning","candidates":[[1,"SUPPORT|CONTRADICT|NEI",[0]]]}. Analysis '
    "must be a non-empty JSON string. Every candidate must be a three-item array: "
    "retrieval rank 1 through 3, one allowed label, and a sentence-index array. "
    "Rank maps to the supplied documents in retrieval order. A NEI candidate may "
    "list valid inspected sentence indices but is ignored in the evidence output; "
    "never mix NEI and non-NEI candidates. Never emit prose outside the single "
    "JSON object."
)
FINAL_OUTPUT_CONTRACT = (
    "OUTPUT CONTRACT: Complete exactly one JSON root object whose only root key "
    'is evidence. The complete and only allowed schema is {"evidence":'
    '{"DOC_ID":{"label":"SUPPORT|CONTRADICT","sentences":[0]}}}. Replace '
    "DOC_ID, the label, and sentence indices with values from the supplied "
    "context. The evidence value is always an object, never a list. For NEI, "
    "the evidence object must contain zero members. Never output claim, "
    "documents, text, found_documents, evidence_schema, analysis, explanation, "
    "commentary, a second JSON object, or any other key or prose. End immediately "
    "after the single root object's closing brace."
)
DECISION_POLICY = (
    " DECISION ORDER: First inspect every supplied sentence against the claim. "
    "If a sentence directly restates or entails the claim, select SUPPORT and "
    "cite it. If a sentence addresses the same proposition but has an "
    "incompatible number, negation, or direction, select CONTRADICT and cite it. "
    "Only after checking all supplied sentences, if neither relation exists, "
    "select NEI. Do not default to NEI. Perform this decision process silently."
)
INITIAL_SYSTEM = (
    "You are a SciFact evidence judge producing an internal structured draft. "
    + DRAFT_CONTRACT
    + DECISION_POLICY
    + " Cite only sentences needed to establish the claim. Do not use outside "
    "knowledge."
)
SELF_REVIEW_SYSTEM = (
    "You are the same SciFact evidence judge performing a strict self-review. "
    "Re-check the draft against the supplied abstracts, correct unsupported "
    "labels or sentence citations. "
    + FINAL_OUTPUT_CONTRACT
    + DECISION_POLICY
    + " The draft is input only and never changes the output schema. Do not use "
    "outside knowledge and do not defer to the draft."
)
CHALLENGER_SYSTEM = (
    "You are an independent adversarial SciFact reviewer. Re-evaluate the draft "
    "from first principles against the supplied abstracts, challenge unsupported "
    "labels or sentence citations. "
    + FINAL_OUTPUT_CONTRACT
    + DECISION_POLICY
    + " The draft is input only and never changes the output schema. Do not use "
    "outside knowledge and do not defer to the draft."
)
INITIAL_USER_TEMPLATE = (
    "Evaluate this canonical context:\n{context_json}\n"
    "Complete the assistant-prefilled internal draft JSON now, then stop after its "
    "root closing brace."
)
REVIEW_USER_TEMPLATE = (
    "Evaluate this canonical context:\n{context_json}\n"
    "Initial draft (the fixed draft_unavailable sentinel means no valid draft was "
    "available):\n{draft}\nDo not echo the draft or emit alternatives. Complete "
    "the assistant-prefilled evidence JSON now, then stop after its root closing "
    "brace."
)

FINAL_PARSER_POLICY = {
    "version": "scifact-agent-output-parser-v1",
    "root_exact_keys": ["evidence"],
    "document_exact_keys": ["label", "sentences"],
    "labels": ["SUPPORT", "CONTRADICT"],
    "max_documents": TOP_K,
    "document_scope": "retrieved_top3_only",
    "sentence_scope": "listed_document_indices_only",
    "unique_sentence_indices": True,
    "nonempty_sentence_list_per_evidence_document": True,
    "duplicate_json_keys": "reject",
    "nonfinite_json_constants": "reject",
    "plain_json_whitespace": "JSON_standard_only",
    "allowed_transport_wrapper": "exact_lowercase_json_fence_with_newlines",
    "semantic_repairs": 0,
    "retries": 0,
    "invalid_final_fallback": "empty_evidence",
}
DRAFT_PARSER_POLICY = {
    "version": "scifact-agent-draft-parser-v1",
    "root_exact_keys": ["analysis", "candidates"],
    "analysis_type": "string",
    "analysis_nonempty": True,
    "analysis_max_words": None,
    "candidate_shape": ["retrieval_rank", "label", "sentence_indices"],
    "candidate_container": "array",
    "max_candidates": TOP_K,
    "unique_candidate_ranks": True,
    "retrieval_rank_scope": [1, TOP_K],
    "rank_mapping": "one_based_rank_to_retrieved_top3_document",
    "labels": ["SUPPORT", "CONTRADICT", "NEI"],
    "nei_candidate_sentence_indices": "optional_but_strictly_scope_validated",
    "nei_mixed_with_non_nei": "reject",
    "nei_projection": "ignored_when_mapping_to_evidence",
    "sentence_scope": "listed_document_indices_only",
    "unique_sentence_indices": True,
    "nonempty_sentence_list_per_relation_candidate": True,
    "duplicate_json_keys": "reject",
    "nonfinite_json_constants": "reject",
    "semantic_repairs": 0,
    "invalid_draft_fallback": "fixed_draft_unavailable_sentinel",
    "one_pass_projection": "candidate_order_to_evidence_object_without_repair",
}
PARSER_POLICIES = {
    "initial_draft": DRAFT_PARSER_POLICY,
    "final_evidence": FINAL_PARSER_POLICY,
}


class AgentABError(RuntimeError):
    """Raised when a run violates its frozen protocol."""


class DuplicateJSONKey(ValueError):
    """Raised for JSON objects whose keys are not unique."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _pretty_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(
        (_canonical_json(row) + "\n").encode("utf-8") for row in rows
    )


def _write_exclusive(path: Path, value: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def _write_atomic_owned(path: Path, value: bytes) -> None:
    """Atomically materialize a final file inside an incomplete owned run."""

    temporary = path.with_name(path.name + ".materializing")
    with temporary.open("wb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AgentABError(f"invalid JSON file: {path}") from exc
    if not isinstance(value, dict):
        raise AgentABError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise AgentABError(f"expected object at {path}:{line_number}")
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise AgentABError(f"invalid JSONL file: {path}") from exc
    return rows


def _load_corpus(path: Path) -> dict[int, dict[str, Any]]:
    corpus = {}
    for row in _read_jsonl(path):
        doc_id = row.get("doc_id")
        title = row.get("title")
        abstract = row.get("abstract")
        if type(doc_id) is not int or doc_id in corpus:
            raise AgentABError("corpus doc_id must be a unique integer")
        if not isinstance(title, str) or not isinstance(abstract, list):
            raise AgentABError("invalid corpus title or abstract")
        if any(not isinstance(sentence, str) for sentence in abstract):
            raise AgentABError("corpus abstract sentences must be strings")
        corpus[doc_id] = {"title": title, "abstract": abstract}
    if not corpus:
        raise AgentABError("corpus is empty")
    return corpus


def _load_claims(path: Path) -> list[dict[str, Any]]:
    """Load only controller fields; gold evidence and cited IDs are never retained."""

    claims = []
    seen = set()
    for row in _read_jsonl(path):
        claim_id = row.get("id")
        claim = row.get("claim")
        if type(claim_id) is not int or claim_id in seen:
            raise AgentABError("claim id must be a unique integer")
        if not isinstance(claim, str) or not claim:
            raise AgentABError("claim text must be non-empty")
        seen.add(claim_id)
        claims.append({"id": claim_id, "claim": claim})
    if not claims:
        raise AgentABError("claims are empty")
    return claims


def _load_retrieval(
    path: Path,
    claims: list[dict[str, Any]],
    corpus: dict[int, dict[str, Any]],
) -> dict[int, list[int]]:
    by_id = {}
    for row in _read_jsonl(path):
        claim_id = row.get("claim_id")
        doc_ids = row.get("doc_ids")
        if type(claim_id) is not int or claim_id in by_id:
            raise AgentABError("retrieval claim_id must be unique")
        if (
            not isinstance(doc_ids, list)
            or len(doc_ids) != TOP_K
            or len(set(doc_ids)) != TOP_K
            or any(type(doc_id) is not int for doc_id in doc_ids)
        ):
            raise AgentABError("retrieval must contain exactly three unique doc IDs")
        if any(doc_id not in corpus for doc_id in doc_ids):
            raise AgentABError("retrieval contains an unknown document")
        by_id[claim_id] = doc_ids
    claim_ids = {row["id"] for row in claims}
    if set(by_id) != claim_ids:
        raise AgentABError("retrieval must cover exactly every claim")
    return by_id


def _validate_retrieval_manifest(
    retrieval_path: Path,
    corpus_path: Path,
    claims_path: Path,
    split: str,
) -> tuple[Path, str]:
    manifest_path = retrieval_path.parent / "run_manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != "scifact-official-tfidf-run-v1":
        raise AgentABError("retrieval is not a frozen official TF-IDF run")
    data = manifest.get("data")
    configuration = manifest.get("configuration")
    outputs = manifest.get("outputs_sha256")
    if not isinstance(data, dict) or data.get("split") != split:
        raise AgentABError("retrieval manifest split mismatch")
    if not isinstance(configuration, dict) or configuration.get("top_k") != TOP_K:
        raise AgentABError("retrieval manifest is not Top-3")
    retrieval_hash = _sha256_file(retrieval_path)
    if not isinstance(outputs, dict) or outputs.get(
        retrieval_path.name
    ) != retrieval_hash:
        raise AgentABError("retrieval file hash is not bound by its manifest")
    file_hashes = data.get("files_sha256")
    if not isinstance(file_hashes, dict):
        raise AgentABError("retrieval manifest lacks data hashes")
    if file_hashes.get(corpus_path.name) != _sha256_file(corpus_path):
        raise AgentABError("corpus hash differs from retrieval run")
    if file_hashes.get(claims_path.name) != _sha256_file(claims_path):
        raise AgentABError("claims hash differs from retrieval run")
    return manifest_path, _sha256_file(manifest_path)


def _validate_model_provenance(
    model_path: Path, provenance_path: Path
) -> dict[str, Any]:
    provenance = _read_json(provenance_path)
    if provenance.get("schema_version") != "local-huggingface-model-snapshot-v1":
        raise AgentABError("unsupported model provenance schema")
    hashes = provenance.get("files_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise AgentABError("model provenance has no file hashes")
    required = {"config.json", "model.safetensors", "tokenizer_config.json"}
    if not required.issubset(hashes):
        raise AgentABError("model provenance omits required artifacts")
    verified = {}
    model_root = model_path.resolve()
    for raw_name, expected in sorted(hashes.items()):
        if not isinstance(raw_name, str) or not isinstance(expected, str):
            raise AgentABError("invalid model provenance file entry")
        relative = Path(raw_name)
        candidate = (model_root / relative).resolve()
        try:
            candidate.relative_to(model_root)
        except ValueError as exc:
            raise AgentABError("model provenance path escapes snapshot") from exc
        if relative.is_absolute() or not candidate.is_file():
            raise AgentABError(f"model artifact is missing: {raw_name}")
        actual = _sha256_file(candidate)
        if actual != expected:
            raise AgentABError(f"model artifact hash mismatch: {raw_name}")
        verified[raw_name] = actual
    return {
        "repository": provenance.get("repository"),
        "revision": provenance.get("revision"),
        "provenance_sha256": _sha256_file(provenance_path),
        "files_sha256": verified,
    }


def _prompt_hashes() -> dict[str, str]:
    prompts = {
        "draft_contract": DRAFT_CONTRACT,
        "final_output_contract": FINAL_OUTPUT_CONTRACT,
        "decision_policy": DECISION_POLICY,
        "initial_system": INITIAL_SYSTEM,
        "self_review_system": SELF_REVIEW_SYSTEM,
        "challenger_system": CHALLENGER_SYSTEM,
        "initial_user_template": INITIAL_USER_TEMPLATE,
        "review_user_template": REVIEW_USER_TEMPLATE,
        "draft_unavailable_sentinel": DRAFT_UNAVAILABLE,
        "initial_draft_prefill": INITIAL_DRAFT_PREFILL,
        "final_evidence_prefill": FINAL_EVIDENCE_PREFILL,
    }
    return {name: _sha256_text(value) for name, value in prompts.items()}


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise AgentABError(f"required package is missing: {name}") from exc


def _configuration_lock(
    script_path: Path,
    model_lock: dict[str, Any],
    max_new_tokens: int,
    seed: int,
) -> dict[str, Any]:
    return {
        "script_sha256": _sha256_file(script_path),
        "model": model_lock,
        "prompts_sha256": _prompt_hashes(),
        "generation": {
            "max_new_tokens_per_call": max_new_tokens,
            "calls_per_case_physical": 3,
            "initial_call_shared": True,
            "logical_calls_single_one_pass": 1,
            "logical_calls_single_self_review": 2,
            "logical_calls_judge_challenger": 2,
            "do_sample": False,
            "temperature": None,
            "top_p": None,
            "top_k": None,
            "num_beams": 1,
            "batch_size": 1,
            "dtype": "torch.bfloat16",
            "attention_implementation": "eager",
            "model_mode": "eval+inference_mode",
            "local_files_only": True,
            "assistant_prefills": {
                "shared_initial_judge": {
                    "value": INITIAL_DRAFT_PREFILL,
                    "sha256": _sha256_text(INITIAL_DRAFT_PREFILL),
                },
                "self_review_and_challenger": {
                    "value": FINAL_EVIDENCE_PREFILL,
                    "sha256": _sha256_text(FINAL_EVIDENCE_PREFILL),
                },
            },
            "assistant_prefill_accounting": "rendered_prompt_input_tokens",
            "chat_template_mode": "continue_final_message",
            "stopping_policy": JSON_ROOT_STOPPING_POLICY,
            "stopping_policy_sha256": _sha256_text(JSON_ROOT_STOPPING_POLICY),
            "stopping_policy_posthoc_truncation": False,
            "application_response_cache": False,
            "model_kv_cache": True,
            "retry_count": 0,
            "seed": seed,
            "initial_draft_projection": (
                "validated_candidates_to_one_pass_evidence_preserving_order"
            ),
        },
        "retrieval_policy": {
            "system": "official_scifact_tfidf",
            "top_k": TOP_K,
            "document_context": "title_and_all_indexed_abstract_sentences",
            "branch_order": "even_case_index_self_then_challenger_else_reverse",
        },
        "parser_policies": PARSER_POLICIES,
        "parser_policies_sha256": _sha256_text(
            _canonical_json(PARSER_POLICIES)
        ),
        "runtime": {
            "python": platform.python_version(),
            "torch": _package_version("torch"),
            "transformers": _package_version("transformers"),
        },
    }


def _input_lock(
    corpus_path: Path,
    claims_path: Path,
    retrieval_path: Path,
    retrieval_manifest_path: Path,
    split: str,
) -> dict[str, Any]:
    return {
        "split": split,
        "corpus_sha256": _sha256_file(corpus_path),
        "claims_sha256": _sha256_file(claims_path),
        "retrieval_sha256": _sha256_file(retrieval_path),
        "retrieval_manifest_sha256": _sha256_file(retrieval_manifest_path),
    }


def _build_context(
    claim: dict[str, Any],
    doc_ids: list[int],
    corpus: dict[int, dict[str, Any]],
) -> tuple[dict[str, Any], str, str]:
    documents = []
    for rank, doc_id in enumerate(doc_ids, 1):
        document = corpus[doc_id]
        documents.append(
            {
                "rank": rank,
                "doc_id": doc_id,
                "title": document["title"],
                "sentences": [
                    {"index": index, "text": sentence}
                    for index, sentence in enumerate(document["abstract"])
                ],
            }
        )
    context = {"claim": claim["claim"], "documents": documents}
    canonical = _canonical_json(context)
    return context, canonical, _sha256_text(canonical)


def _chat_messages(
    system: str, user: str, assistant_prefill: str
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant_prefill},
    ]


def _json_root_closed(value: str) -> bool:
    """Return true once a root JSON object has closed, respecting JSON strings."""

    if not value or value[0] != "{":
        return False
    depth = 0
    in_string = False
    escaped = False
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
            depth += 1
        elif character in "]}":
            depth -= 1
            if depth == 0:
                return True
            if depth < 0:
                return False
    return False


def _duplicate_guard(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateJSONKey(key)
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def parse_initial_draft(
    raw: str,
    allowed_documents: dict[int, int],
) -> dict[str, Any]:
    """Strictly parse the internal analysis/candidate draft schema."""

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
        return {
            "valid": False,
            "failure_code": "invalid_draft_json",
            "transport_unwrapped": transport_unwrapped,
            "value": None,
            "detail": str(exc)[:300],
        }
    if not isinstance(value, dict) or set(value) != {"analysis", "candidates"}:
        failure = "invalid_draft_root_schema"
    elif not isinstance(value["analysis"], str) or not value["analysis"].strip():
        failure = "invalid_draft_analysis"
    elif not isinstance(value["candidates"], list) or len(
        value["candidates"]
    ) > TOP_K:
        failure = "invalid_draft_candidates"
    else:
        ranked_doc_ids = list(allowed_documents)
        if len(ranked_doc_ids) != TOP_K:
            raise AgentABError("draft parser requires exactly three ranked documents")
        seen_ranks = set()
        seen_labels = set()
        for candidate in value["candidates"]:
            if not isinstance(candidate, list) or len(candidate) != 3:
                failure = "invalid_draft_candidate_schema"
                break
            rank, label, sentences = candidate
            if type(rank) is not int or not 1 <= rank <= TOP_K:
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
            if any(type(index) is not int for index in sentences):
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
            if label == "NEI":
                continue
            if not sentences:
                failure = "invalid_draft_sentences"
                break
        else:
            if "NEI" in seen_labels and len(seen_labels) > 1:
                return {
                    "valid": False,
                    "failure_code": "mixed_nei_and_relation_candidates",
                    "transport_unwrapped": transport_unwrapped,
                    "value": None,
                    "detail": None,
                }
            return {
                "valid": True,
                "failure_code": None,
                "transport_unwrapped": transport_unwrapped,
                "value": value,
                "detail": None,
            }
    return {
        "valid": False,
        "failure_code": failure,
        "transport_unwrapped": transport_unwrapped,
        "value": None,
        "detail": None,
    }


def _draft_to_evidence(
    draft: dict[str, Any], ranked_doc_ids: list[int]
) -> dict[str, Any]:
    """Project a validated candidate array without semantic changes or repair."""

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


def parse_agent_output(
    raw: str,
    allowed_documents: dict[int, int],
) -> dict[str, Any]:
    """Parse without semantic repair; only the exact JSON fence is transport."""

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
        return {
            "valid": False,
            "failure_code": "invalid_json",
            "transport_unwrapped": transport_unwrapped,
            "value": None,
            "detail": str(exc)[:300],
        }
    if not isinstance(value, dict) or set(value) != {"evidence"}:
        return {
            "valid": False,
            "failure_code": "invalid_root_schema",
            "transport_unwrapped": transport_unwrapped,
            "value": None,
            "detail": None,
        }
    evidence = value["evidence"]
    if not isinstance(evidence, dict) or len(evidence) > TOP_K:
        return {
            "valid": False,
            "failure_code": "invalid_evidence_object",
            "transport_unwrapped": transport_unwrapped,
            "value": None,
            "detail": None,
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
        if document["label"] not in {"SUPPORT", "CONTRADICT"}:
            failure = "invalid_label"
            break
        sentences = document["sentences"]
        if not isinstance(sentences, list) or not sentences:
            failure = "invalid_sentences"
            break
        if any(type(index) is not int for index in sentences):
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
            "detail": None,
        }
    return {
        "valid": False,
        "failure_code": failure,
        "transport_unwrapped": transport_unwrapped,
        "value": None,
        "detail": None,
    }


class LocalGenerator:
    """Offline, single-example, deterministic Hugging Face generation."""

    def __init__(self, model_path: Path, max_new_tokens: int, seed: int) -> None:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            StoppingCriteria,
            StoppingCriteriaList,
        )

        if not torch.cuda.is_available():
            raise AgentABError("CUDA is required by the frozen local runtime")
        if not torch.cuda.is_bf16_supported():
            raise AgentABError("the CUDA device does not support bfloat16")
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.torch = torch
        self.device = torch.device("cuda")
        self.max_new_tokens = max_new_tokens
        self.stopping_criteria_type = StoppingCriteria
        self.stopping_criteria_list_type = StoppingCriteriaList
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), local_files_only=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        ).to(self.device)
        self.model.eval()
        self.device_name = torch.cuda.get_device_name(self.device)
        self.max_position_embeddings = int(
            getattr(self.model.config, "max_position_embeddings", 0)
        )

    def generate(
        self,
        call_id: str,
        call_kind: str,
        messages: list[dict[str, str]],
        logical_attribution: list[str],
        assistant_prefill: str,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        messages_serialized = _canonical_json(messages)
        record = {
            "schema_version": "scifact-agent-physical-call-v1",
            "call_id": call_id,
            "call_kind": call_kind,
            "logical_attribution": logical_attribution,
            "device": self.device_name,
            "messages_sha256": _sha256_text(messages_serialized),
            "rendered_prompt": None,
            "rendered_prompt_sha256": None,
            "prompt_sha256": None,
            "prompt_character_count": None,
            "assistant_prefill": assistant_prefill,
            "assistant_prefill_sha256": _sha256_text(assistant_prefill),
            "assistant_prefill_in_input_tokens": True,
            "generated_fragment": None,
            "generated_fragment_sha256": None,
            "raw_output": None,
            "raw_output_sha256": None,
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "synchronized_latency_ms": None,
            "end_to_end_latency_ms": None,
            "generation_succeeded": False,
            "generation_error": None,
            "json_root_stopping_triggered": False,
            "stopping_policy": JSON_ROOT_STOPPING_POLICY,
            "stopping_policy_sha256": _sha256_text(
                JSON_ROOT_STOPPING_POLICY
            ),
            "external_api_cost_usd": 0.0,
            "compute_cost_usd": None,
        }
        try:
            rendered_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                continue_final_message=True,
            )
            if not rendered_prompt.endswith(assistant_prefill):
                raise AgentABError("chat template did not preserve assistant prefill")
            record.update(
                {
                    "rendered_prompt": rendered_prompt,
                    "rendered_prompt_sha256": _sha256_text(rendered_prompt),
                    "prompt_sha256": _sha256_text(rendered_prompt),
                    "prompt_character_count": len(rendered_prompt),
                }
            )
            encoded = self.tokenizer(
                rendered_prompt,
                add_special_tokens=False,
                return_tensors="pt",
            )
            encoded = {name: tensor.to(self.device) for name, tensor in encoded.items()}
            if "attention_mask" not in encoded:
                encoded["attention_mask"] = self.torch.ones_like(encoded["input_ids"])
            input_tokens = int(encoded["attention_mask"].sum().item())
            if (
                self.max_position_embeddings > 0
                and input_tokens + self.max_new_tokens > self.max_position_embeddings
            ):
                raise AgentABError("prompt plus output budget exceeds model context")
            record["input_tokens"] = input_tokens

            tokenizer = self.tokenizer
            prompt_length = int(encoded["input_ids"].shape[1])
            stopping_base = self.stopping_criteria_type

            class JSONRootClosureStoppingCriteria(stopping_base):
                def __init__(self) -> None:
                    self.triggered = False

                def __call__(
                    self, input_ids: Any, scores: Any, **kwargs: Any
                ) -> bool:
                    del scores, kwargs
                    fragment = tokenizer.decode(
                        input_ids[0, prompt_length:],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    self.triggered = _json_root_closed(
                        assistant_prefill + fragment
                    )
                    return self.triggered

            root_stopper = JSONRootClosureStoppingCriteria()
            self.torch.cuda.synchronize(self.device)
            generation_started = time.perf_counter()
            with self.torch.inference_mode():
                generated = self.model.generate(
                    **encoded,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    num_beams=1,
                    use_cache=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    stopping_criteria=self.stopping_criteria_list_type(
                        [root_stopper]
                    ),
                )
            self.torch.cuda.synchronize(self.device)
            latency_ms = (time.perf_counter() - generation_started) * 1000.0
            output_ids = generated[0, encoded["input_ids"].shape[1] :]
            output_tokens = int(output_ids.shape[0])
            generated_fragment = self.tokenizer.decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            raw = assistant_prefill + generated_fragment
            record.update(
                {
                    "raw_output": raw,
                    "raw_output_sha256": _sha256_text(raw),
                    "generated_fragment": generated_fragment,
                    "generated_fragment_sha256": _sha256_text(
                        generated_fragment
                    ),
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                    "synchronized_latency_ms": latency_ms,
                    "end_to_end_latency_ms": (
                        time.perf_counter() - started
                    )
                    * 1000.0,
                    "generation_succeeded": True,
                    "json_root_stopping_triggered": root_stopper.triggered,
                }
            )
        except Exception:
            if self.torch.cuda.is_available():
                with suppress(Exception):
                    self.torch.cuda.synchronize(self.device)
            # Hardware/runtime failures are not model predictions. Abort and keep
            # the run incomplete so --resume can retry the uncommitted case.
            raise
        return record


def _attach_parse(
    call: dict[str, Any], allowed_documents: dict[int, int], schema: str
) -> dict[str, Any]:
    if not call["generation_succeeded"]:
        parsed = {
            "valid": False,
            "failure_code": "generation_failed",
            "transport_unwrapped": False,
            "value": None,
            "detail": None,
        }
    else:
        if schema == "initial_draft":
            parsed = parse_initial_draft(call["raw_output"], allowed_documents)
        elif schema == "final_evidence":
            parsed = parse_agent_output(call["raw_output"], allowed_documents)
        else:
            raise AgentABError("unknown output parser schema")
    call["parse_schema"] = schema
    policy = (
        DRAFT_PARSER_POLICY if schema == "initial_draft" else FINAL_PARSER_POLICY
    )
    call["parser_policy_sha256"] = _sha256_text(_canonical_json(policy))
    call["parse"] = parsed
    return parsed


def _sum_nullable(calls: list[dict[str, Any]], key: str) -> Any:
    values = [call[key] for call in calls]
    if any(value is None for value in values):
        return None
    return sum(values)


def _arm_summary(
    calls: list[dict[str, Any]], parsed: dict[str, Any]
) -> dict[str, Any]:
    evidence = parsed["value"]["evidence"] if parsed["valid"] else {}
    return {
        "valid": parsed["valid"],
        "failure_code": parsed["failure_code"],
        "transport_unwrapped": parsed["transport_unwrapped"],
        "final_evidence": evidence,
        "invalid_final_fallback_applied": not parsed["valid"],
        "logical_call_count": len(calls),
        "physical_call_ids": [call["call_id"] for call in calls],
        "input_tokens": _sum_nullable(calls, "input_tokens"),
        "output_tokens": _sum_nullable(calls, "output_tokens"),
        "total_tokens": _sum_nullable(calls, "total_tokens"),
        "logical_inference_latency_ms": _sum_nullable(
            calls, "synchronized_latency_ms"
        ),
        "logical_end_to_end_latency_ms": _sum_nullable(
            calls, "end_to_end_latency_ms"
        ),
        "external_api_cost_usd": 0.0,
        "compute_cost_usd": None,
    }


def _run_case(
    generator: LocalGenerator,
    case_index: int,
    claim: dict[str, Any],
    doc_ids: list[int],
    corpus: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    context, context_json, context_sha256 = _build_context(claim, doc_ids, corpus)
    allowed = {doc_id: len(corpus[doc_id]["abstract"]) for doc_id in doc_ids}
    prefix = "{:04d}:{}".format(case_index, claim["id"])
    initial = generator.generate(
        prefix + ":initial",
        "shared_initial_judge",
        _chat_messages(
            INITIAL_SYSTEM,
            INITIAL_USER_TEMPLATE.format(context_json=context_json),
            INITIAL_DRAFT_PREFILL,
        ),
        list(ARMS),
        INITIAL_DRAFT_PREFILL,
    )
    initial["claim_id"] = claim["id"]
    initial["canonical_context_sha256"] = context_sha256
    initial_parse = _attach_parse(initial, allowed, "initial_draft")
    if initial_parse["valid"]:
        draft = _canonical_json(initial_parse["value"])
        draft_status = "valid_initial_draft"
        one_pass_parse = {
            "valid": True,
            "failure_code": None,
            "transport_unwrapped": initial_parse["transport_unwrapped"],
            "value": _draft_to_evidence(initial_parse["value"], doc_ids),
            "detail": None,
        }
    else:
        draft = DRAFT_UNAVAILABLE
        draft_status = "draft_unavailable"
        one_pass_parse = dict(initial_parse)

    branch_order = (
        ["single_self_review", "judge_challenger"]
        if case_index % 2 == 0
        else ["judge_challenger", "single_self_review"]
    )
    branches = {}
    for arm in branch_order:
        if arm == "single_self_review":
            system = SELF_REVIEW_SYSTEM
            kind = "self_review"
        else:
            system = CHALLENGER_SYSTEM
            kind = "independent_challenger"
        call = generator.generate(
            prefix + ":" + kind,
            kind,
            _chat_messages(
                system,
                REVIEW_USER_TEMPLATE.format(
                    context_json=context_json, draft=draft
                ),
                FINAL_EVIDENCE_PREFILL,
            ),
            [arm],
            FINAL_EVIDENCE_PREFILL,
        )
        call["claim_id"] = claim["id"]
        call["canonical_context_sha256"] = context_sha256
        parsed = _attach_parse(call, allowed, "final_evidence")
        branches[arm] = {"call": call, "parse": parsed}

    self_call = branches["single_self_review"]["call"]
    challenger_call = branches["judge_challenger"]["call"]
    arms = {
        "single_one_pass": _arm_summary([initial], one_pass_parse),
        "single_self_review": _arm_summary(
            [initial, self_call], branches["single_self_review"]["parse"]
        ),
        "judge_challenger": _arm_summary(
            [initial, challenger_call], branches["judge_challenger"]["parse"]
        ),
    }
    physical_calls = [initial] + [branches[arm]["call"] for arm in branch_order]
    context_record = {
        "schema_version": "scifact-agent-context-v1",
        "claim_id": claim["id"],
        "context": context,
        "canonical_context_sha256": context_sha256,
        "gold_fields_present": False,
    }
    case_record = {
        "schema_version": "scifact-agent-case-v1",
        "case_index": case_index,
        "claim_id": claim["id"],
        "canonical_context_sha256": context_sha256,
        "branch_execution_order": branch_order,
        "draft_status": draft_status,
        "draft_sha256": _sha256_text(draft),
        "physical_call_count": 3,
        "physical_call_ids": [call["call_id"] for call in physical_calls],
        "physical_input_tokens": _sum_nullable(physical_calls, "input_tokens"),
        "physical_output_tokens": _sum_nullable(physical_calls, "output_tokens"),
        "physical_total_tokens": _sum_nullable(physical_calls, "total_tokens"),
        "physical_inference_latency_ms": _sum_nullable(
            physical_calls, "synchronized_latency_ms"
        ),
        "physical_end_to_end_latency_ms": _sum_nullable(
            physical_calls, "end_to_end_latency_ms"
        ),
        "arms": arms,
    }
    predictions = {
        arm: {"id": claim["id"], "evidence": arms[arm]["final_evidence"]}
        for arm in ARMS
    }
    return {
        "case_index": case_index,
        "claim_id": claim["id"],
        "context": context_record,
        "case": case_record,
        "predictions": predictions,
        "telemetry": physical_calls,
    }


def _journal_envelope(
    record: dict[str, Any], previous_sha256: Optional[str]  # noqa: UP045
) -> dict[str, Any]:
    return {
        "schema_version": "scifact-agent-case-journal-v1",
        "previous_record_sha256": previous_sha256,
        "record_sha256": _sha256_text(_canonical_json(record)),
        "record": record,
    }


def _validate_journal_record(
    envelope: dict[str, Any],
    expected_previous: Optional[str],  # noqa: UP045
    expected_index: int,
    claim: dict[str, Any],
    doc_ids: list[int],
    corpus: dict[int, dict[str, Any]],
) -> str:
    if envelope.get("schema_version") != "scifact-agent-case-journal-v1":
        raise AgentABError("journal schema mismatch")
    if envelope.get("previous_record_sha256") != expected_previous:
        raise AgentABError("journal hash chain is broken")
    record = envelope.get("record")
    if not isinstance(record, dict):
        raise AgentABError("journal record is missing")
    actual_hash = _sha256_text(_canonical_json(record))
    if envelope.get("record_sha256") != actual_hash:
        raise AgentABError("journal record hash mismatch")
    if (
        record.get("case_index") != expected_index
        or record.get("claim_id") != claim["id"]
    ):
        raise AgentABError("journal is not the selected claim prefix")
    expected_context, context_json, expected_context_hash = _build_context(
        claim, doc_ids, corpus
    )
    context_record = record.get("context")
    case_record = record.get("case")
    predictions = record.get("predictions")
    telemetry = record.get("telemetry")
    if (
        not isinstance(context_record, dict)
        or context_record.get("context") != expected_context
    ):
        raise AgentABError("journal context differs from frozen input")
    if context_record.get("canonical_context_sha256") != expected_context_hash:
        raise AgentABError("journal context hash mismatch")
    if not isinstance(case_record, dict) or case_record.get("claim_id") != claim["id"]:
        raise AgentABError("journal case summary mismatch")
    expected_order = (
        ["single_self_review", "judge_challenger"]
        if expected_index % 2 == 0
        else ["judge_challenger", "single_self_review"]
    )
    if case_record.get("branch_execution_order") != expected_order:
        raise AgentABError("journal branch order mismatch")
    expected_kinds = ["shared_initial_judge"] + [
        "self_review" if arm == "single_self_review" else "independent_challenger"
        for arm in expected_order
    ]
    if not isinstance(telemetry, list) or [
        call.get("call_kind") for call in telemetry if isinstance(call, dict)
    ] != expected_kinds:
        raise AgentABError("journal physical call order mismatch")
    prefix = f"{expected_index:04d}:{claim['id']}"
    expected_call_ids = [prefix + ":initial"] + [
        prefix
        + (
            ":self_review"
            if arm == "single_self_review"
            else ":independent_challenger"
        )
        for arm in expected_order
    ]
    if [call.get("call_id") for call in telemetry] != expected_call_ids:
        raise AgentABError("journal physical call IDs are not canonical")
    allowed = {doc_id: len(corpus[doc_id]["abstract"]) for doc_id in doc_ids}
    reparsed = []
    for call in telemetry:
        parse_schema = (
            "initial_draft"
            if call.get("call_kind") == "shared_initial_judge"
            else "final_evidence"
        )
        expected_prefill = (
            INITIAL_DRAFT_PREFILL
            if parse_schema == "initial_draft"
            else FINAL_EVIDENCE_PREFILL
        )
        if (
            call.get("claim_id") != claim["id"]
            or call.get("canonical_context_sha256") != expected_context_hash
            or call.get("generation_succeeded") is not True
        ):
            raise AgentABError("journal call identity or status is invalid")
        raw_output = call.get("raw_output")
        if not isinstance(raw_output, str):
            raise AgentABError("journal call has no raw output")
        if call.get("raw_output_sha256") != _sha256_text(raw_output):
            raise AgentABError("journal raw output hash mismatch")
        rendered_prompt = call.get("rendered_prompt")
        if not isinstance(rendered_prompt, str):
            raise AgentABError("journal call has no rendered prompt")
        rendered_hash = _sha256_text(rendered_prompt)
        if (
            call.get("rendered_prompt_sha256") != rendered_hash
            or call.get("prompt_sha256") != rendered_hash
            or not rendered_prompt.endswith(expected_prefill)
        ):
            raise AgentABError("journal rendered prompt hash mismatch")
        generated_fragment = call.get("generated_fragment")
        if (
            not isinstance(generated_fragment, str)
            or call.get("generated_fragment_sha256")
            != _sha256_text(generated_fragment)
            or raw_output != expected_prefill + generated_fragment
            or call.get("assistant_prefill") != expected_prefill
            or call.get("assistant_prefill_sha256")
            != _sha256_text(expected_prefill)
        ):
            raise AgentABError("journal prefill composition is invalid")
        parsed = (
            parse_initial_draft(raw_output, allowed)
            if parse_schema == "initial_draft"
            else parse_agent_output(raw_output, allowed)
        )
        parser_policy = (
            DRAFT_PARSER_POLICY
            if parse_schema == "initial_draft"
            else FINAL_PARSER_POLICY
        )
        if (
            call.get("parse_schema") != parse_schema
            or call.get("parser_policy_sha256")
            != _sha256_text(_canonical_json(parser_policy))
            or call.get("parse") != parsed
            or call.get("stopping_policy") != JSON_ROOT_STOPPING_POLICY
            or call.get("stopping_policy_sha256")
            != _sha256_text(JSON_ROOT_STOPPING_POLICY)
        ):
            raise AgentABError("journal parse record is not reproducible")
        reparsed.append(parsed)
    draft = (
        _canonical_json(reparsed[0]["value"])
        if reparsed[0]["valid"]
        else DRAFT_UNAVAILABLE
    )
    expected_messages = [
        _chat_messages(
            INITIAL_SYSTEM,
            INITIAL_USER_TEMPLATE.format(context_json=context_json),
            INITIAL_DRAFT_PREFILL,
        )
    ]
    for arm in expected_order:
        system = (
            SELF_REVIEW_SYSTEM
            if arm == "single_self_review"
            else CHALLENGER_SYSTEM
        )
        expected_messages.append(
            _chat_messages(
                system,
                REVIEW_USER_TEMPLATE.format(
                    context_json=context_json, draft=draft
                ),
                FINAL_EVIDENCE_PREFILL,
            )
        )
    expected_message_hashes = [
        _sha256_text(_canonical_json(messages)) for messages in expected_messages
    ]
    if [call.get("messages_sha256") for call in telemetry] != expected_message_hashes:
        raise AgentABError("journal message hashes are not reproducible")
    if not isinstance(predictions, dict) or set(predictions) != set(ARMS):
        raise AgentABError("journal predictions are incomplete")
    arms = case_record.get("arms")
    if not isinstance(arms, dict) or set(arms) != set(ARMS):
        raise AgentABError("journal arm summaries are incomplete")
    for arm in ARMS:
        expected_prediction = {
            "id": claim["id"],
            "evidence": arms[arm].get("final_evidence"),
        }
        if predictions[arm] != expected_prediction:
            raise AgentABError("journal prediction and arm summary disagree")
    return actual_hash


def _read_and_validate_journal(
    journal_path: Path,
    selected_claims: list[dict[str, Any]],
    retrieval: dict[int, list[int]],
    corpus: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    if not journal_path.exists():
        return []
    records = []
    previous = None
    try:
        with journal_path.open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if index >= len(selected_claims):
                    raise AgentABError("journal contains excess cases")
                if not line.endswith("\n"):
                    raise AgentABError("journal has an incomplete trailing write")
                envelope = json.loads(line)
                if not isinstance(envelope, dict):
                    raise AgentABError("journal line is not an object")
                claim = selected_claims[index]
                previous = _validate_journal_record(
                    envelope,
                    previous,
                    index,
                    claim,
                    retrieval[claim["id"]],
                    corpus,
                )
                records.append(envelope["record"])
    except (OSError, json.JSONDecodeError) as exc:
        raise AgentABError("journal cannot be safely resumed") from exc
    return records


def _verify_owned_outputs(manifest_path: Path) -> None:
    manifest = _read_json(manifest_path)
    outputs = manifest.get("outputs_sha256")
    if not isinstance(outputs, dict):
        raise AgentABError("train manifest has no output hashes")
    for name, expected in outputs.items():
        path = manifest_path.parent / name
        if not path.is_file() or _sha256_file(path) != expected:
            raise AgentABError(f"train run output hash mismatch: {name}")


def write_frozen_config(
    train_manifest_path: Path,
    destination: Path,
    configuration_lock: dict[str, Any],
    dev_inputs: dict[str, Any],
) -> Path:
    train_manifest = _read_json(train_manifest_path)
    if train_manifest.get("schema_version") != "scifact-agent-ab-run-v1":
        raise AgentABError("freeze source is not an Agent A/B run")
    if train_manifest.get("input_lock", {}).get("split") != "train":
        raise AgentABError("freeze source must be a train run")
    if train_manifest.get("configuration_lock") != configuration_lock:
        raise AgentABError("current configuration differs from train smoke")
    if train_manifest.get("execution", {}).get("completed") is not True:
        raise AgentABError("train smoke did not complete")
    _verify_owned_outputs(train_manifest_path)
    value = {
        "schema_version": "scifact-agent-ab-frozen-config-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
        "train_run_manifest": {
            "path": str(train_manifest_path.resolve()),
            "sha256": _sha256_file(train_manifest_path),
            "selected_claim_count": train_manifest["execution"][
                "selected_claim_count"
            ],
        },
        "locked_configuration": configuration_lock,
        "dev_inputs": dev_inputs,
        "selection": {
            "train_used_for_prompt_and_parser_smoke_only": True,
            "dev_tuning_permitted": False,
            "main_comparison": "judge_challenger_vs_single_self_review",
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_exclusive(destination, _pretty_json_bytes(value))
    return destination


def _validate_frozen_config(
    path: Path,
    configuration_lock: dict[str, Any],
    input_lock: dict[str, Any],
) -> dict[str, Any]:
    frozen = _read_json(path)
    if frozen.get("schema_version") != "scifact-agent-ab-frozen-config-v1":
        raise AgentABError("invalid frozen config schema")
    if frozen.get("locked_configuration") != configuration_lock:
        raise AgentABError("runtime configuration differs from frozen config")
    if frozen.get("dev_inputs") != input_lock:
        raise AgentABError("dev inputs differ from frozen config")
    train_binding = frozen.get("train_run_manifest")
    if not isinstance(train_binding, dict):
        raise AgentABError("frozen config lacks train binding")
    train_path = Path(str(train_binding.get("path", "")))
    if not train_path.is_file() or _sha256_file(train_path) != train_binding.get(
        "sha256"
    ):
        raise AgentABError("frozen train run binding is invalid")
    _verify_owned_outputs(train_path)
    return frozen


def _run_contract(
    configuration_lock: dict[str, Any],
    input_lock: dict[str, Any],
    selected_claims: list[dict[str, Any]],
    frozen_config_sha256: Optional[str],  # noqa: UP045
) -> dict[str, Any]:
    return {
        "schema_version": "scifact-agent-ab-incomplete-run-v1",
        "configuration_lock": configuration_lock,
        "input_lock": input_lock,
        "selected_claim_ids": [claim["id"] for claim in selected_claims],
        "frozen_config_sha256": frozen_config_sha256,
    }


def _prepare_or_resume(
    output: Path, run_contract: dict[str, Any], resume: bool
) -> tuple[Path, Path]:
    header_path = output / "run_header.json"
    journal_path = output / "journal.jsonl"
    if resume:
        if not output.is_dir() or not header_path.is_file():
            raise AgentABError("resume target is not an incomplete owned run")
        if (output / "run_manifest.json").exists():
            raise AgentABError("completed runs cannot be resumed")
        if _read_json(header_path) != run_contract:
            raise AgentABError("resume parameters differ from run header")
    else:
        output.mkdir(parents=True, exist_ok=False)
        _write_exclusive(header_path, _pretty_json_bytes(run_contract))
        _write_exclusive(journal_path, b"")
    return header_path, journal_path


def _append_journal(handle: Any, envelope: dict[str, Any]) -> None:
    handle.write((_canonical_json(envelope) + "\n").encode("utf-8"))
    handle.flush()
    os.fsync(handle.fileno())


def _materialize_final_outputs(
    output: Path,
    records: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> Path:
    rows_by_name = {
        "contexts.jsonl": [record["context"] for record in records],
        "cases.jsonl": [record["case"] for record in records],
        "raw_telemetry.jsonl": [
            call for record in records for call in record["telemetry"]
        ],
        "predictions_single_one_pass.jsonl": [
            record["predictions"]["single_one_pass"] for record in records
        ],
        "predictions_single_self_review.jsonl": [
            record["predictions"]["single_self_review"] for record in records
        ],
        "predictions_judge_challenger.jsonl": [
            record["predictions"]["judge_challenger"] for record in records
        ],
    }
    for name, rows in rows_by_name.items():
        _write_atomic_owned(output / name, _jsonl_bytes(rows))
    output_hashes = {
        name: _sha256_file(output / name) for name in sorted(rows_by_name)
    }
    output_hashes["journal.jsonl"] = _sha256_file(output / "journal.jsonl")
    output_hashes["run_header.json"] = _sha256_file(output / "run_header.json")
    manifest["outputs_sha256"] = output_hashes
    manifest_path = output / "run_manifest.json"
    _write_exclusive(manifest_path, _pretty_json_bytes(manifest))
    return manifest_path


def run(args: argparse.Namespace) -> Path:
    invocation_started = time.perf_counter()
    script_path = Path(__file__).resolve()
    corpus_path = args.corpus.resolve()
    claims_path = args.claims.resolve()
    retrieval_path = args.retrieval.resolve()
    model_path = args.model.resolve()
    provenance_path = args.model_provenance.resolve()
    for path in (
        script_path,
        corpus_path,
        claims_path,
        retrieval_path,
        model_path,
        provenance_path,
    ):
        if not path.exists():
            raise AgentABError(f"required path is missing: {path}")
    if args.max_new_tokens <= 0:
        raise AgentABError("max-new-tokens must be positive")
    if args.limit is not None and args.limit <= 0:
        raise AgentABError("limit must be positive")
    if args.split == "dev" and args.limit is not None:
        raise AgentABError("formal dev must run the complete split")

    corpus = _load_corpus(corpus_path)
    claims = _load_claims(claims_path)
    retrieval = _load_retrieval(retrieval_path, claims, corpus)
    if args.split == "dev" and len(claims) != 300:
        raise AgentABError("formal SciFact dev must contain 300 claims")
    retrieval_manifest_path, _ = _validate_retrieval_manifest(
        retrieval_path, corpus_path, claims_path, args.split
    )
    model_lock = _validate_model_provenance(model_path, provenance_path)
    configuration_lock = _configuration_lock(
        script_path, model_lock, args.max_new_tokens, args.seed
    )
    input_lock = _input_lock(
        corpus_path,
        claims_path,
        retrieval_path,
        retrieval_manifest_path,
        args.split,
    )

    if args.freeze_from_train_run is not None or args.write_frozen_config is not None:
        if args.freeze_from_train_run is None or args.write_frozen_config is None:
            raise AgentABError("freeze mode requires both freeze arguments")
        if args.split != "dev" or args.limit is not None or args.resume:
            raise AgentABError("freeze mode requires full dev inputs and no resume")
        return write_frozen_config(
            args.freeze_from_train_run.resolve(),
            args.write_frozen_config.resolve(),
            configuration_lock,
            input_lock,
        )

    if args.out is None:
        raise AgentABError("normal run mode requires --out")
    if args.split == "dev" and args.frozen_config is None:
        raise AgentABError("formal dev requires --frozen-config")
    if args.split == "train" and args.frozen_config is not None:
        raise AgentABError("train smoke must not use a dev frozen config")
    selected_claims = claims if args.limit is None else claims[: args.limit]
    frozen_hash = None
    if args.frozen_config is not None:
        _validate_frozen_config(
            args.frozen_config.resolve(), configuration_lock, input_lock
        )
        frozen_hash = _sha256_file(args.frozen_config.resolve())
    run_contract = _run_contract(
        configuration_lock, input_lock, selected_claims, frozen_hash
    )
    output = args.out.resolve()
    _, journal_path = _prepare_or_resume(output, run_contract, args.resume)
    records = _read_and_validate_journal(
        journal_path, selected_claims, retrieval, corpus
    )
    generator = None
    model_load_time_ms = None
    if len(records) == len(selected_claims):
        print("journal already contains every selected case; materializing outputs")
    else:
        model_load_started = time.perf_counter()
        generator = LocalGenerator(model_path, args.max_new_tokens, args.seed)
        model_load_time_ms = (time.perf_counter() - model_load_started) * 1000.0
        previous_sha = None
        if records:
            previous_sha = _sha256_text(_canonical_json(records[-1]))
        with journal_path.open("ab") as journal:
            for index in range(len(records), len(selected_claims)):
                claim = selected_claims[index]
                record = _run_case(
                    generator,
                    index,
                    claim,
                    retrieval[claim["id"]],
                    corpus,
                )
                envelope = _journal_envelope(record, previous_sha)
                _append_journal(journal, envelope)
                previous_sha = envelope["record_sha256"]
                records.append(record)
                print(
                    _canonical_json(
                        {
                            "event": "case_committed",
                            "completed": len(records),
                            "total": len(selected_claims),
                            "claim_id": claim["id"],
                        }
                    ),
                    flush=True,
                )

    # Re-read the fsynced source of truth before producing reportable files.
    records = _read_and_validate_journal(
        journal_path, selected_claims, retrieval, corpus
    )
    if len(records) != len(selected_claims):
        raise AgentABError("run is incomplete and cannot be materialized")
    physical_calls = [call for record in records for call in record["telemetry"]]
    wall_time_ms = (
        None
        if args.resume
        else (time.perf_counter() - invocation_started) * 1000.0
    )
    manifest = {
        "schema_version": "scifact-agent-ab-run-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
        "evaluation_status": {
            "reportable": args.split == "dev",
            "state": "ready_for_official_scoring",
            "gold_used_by_runner": False,
        },
        "configuration_lock": configuration_lock,
        "input_lock": input_lock,
        "frozen_config": (
            None
            if args.frozen_config is None
            else {
                "path": str(args.frozen_config.resolve()),
                "sha256": frozen_hash,
            }
        ),
        "execution": {
            "completed": True,
            "resume_used": args.resume,
            "source_claim_count": len(claims),
            "selected_claim_count": len(selected_claims),
            "selected_claim_ids_sha256": _sha256_text(
                _canonical_json([claim["id"] for claim in selected_claims])
            ),
            "physical_call_count": len(physical_calls),
            "expected_physical_call_count": len(selected_claims) * 3,
            "journal_fsync_per_case": True,
            "wall_time_ms": wall_time_ms,
            "wall_time_scope": (
                "run_entry_through_pre_materialization"
                if not args.resume
                else "unavailable_across_resumed_processes"
            ),
            "model_load_time_ms": model_load_time_ms,
            "model_load_time_scope": (
                "this_process_only" if model_load_time_ms is not None else None
            ),
            "active_physical_call_end_to_end_time_ms": _sum_nullable(
                physical_calls, "end_to_end_latency_ms"
            ),
        },
        "comparison": {
            "main": "judge_challenger_vs_single_self_review",
            "reference_only": "single_one_pass",
            "shared_initial_call": True,
            "main_arms_logical_calls_per_case": 2,
            "same_model_top_k_context_and_output_budget": True,
            "branch_execution_alternated_by_case_index_parity": True,
        },
        "accounting": {
            "physical_calls_count_shared_initial_once": True,
            "logical_arm_totals_attribute_shared_initial_to_each_arm": True,
            "external_api_cost_usd": 0.0,
            "compute_cost_usd": None,
            "compute_cost_note": (
                "Local electricity and hardware cost were not measured."
            ),
        },
        "validation": {
            "exact_selected_claim_coverage": True,
            "contexts_contain_only_claim_and_retrieved_documents": True,
            "gold_evidence_and_cited_doc_ids_excluded": True,
            "three_physical_calls_per_case": True,
            "no_retries": True,
            "no_application_cache": True,
            "invalid_outputs_remain_in_denominator": True,
        },
        "environment_observed": {
            "python": sys.version,
            "platform": platform.platform(),
            "cuda_device": (
                (
                    physical_calls[0].get("device")
                    if physical_calls
                    else None
                )
                if generator is None
                else generator.device_name
            ),
        },
    }
    return _materialize_final_outputs(output, records, manifest)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--claims", type=Path, required=True)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-provenance", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "dev"), required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--frozen-config", type=Path)
    parser.add_argument("--freeze-from-train-run", type=Path)
    parser.add_argument("--write-frozen-config", type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    try:
        print(run(_parse_args()))
    except AgentABError as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    main()
