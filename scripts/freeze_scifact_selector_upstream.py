"""Freeze verified train OOF evidence contexts for a later fair Agent comparison.

The full-train selector is bound as a deployable upstream artifact, but is never
applied back to train here. Contexts come exclusively from the passed selector's
hybrid outer-fold predictions. This module performs no model inference.
"""

from __future__ import annotations

# ruff: noqa: E501, UP017, UP045
import argparse
import hashlib
import importlib.util
import json
import math
import re
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

SCHEMA_VERSION = "scifact-selector-upstream-freeze-v1"
PINNED_SELECTOR_MANIFEST_SHA256 = "44d9ae59a4017209a6196e2c29d66bc031919c9bf5fcc6204a94d03316819c31"
PINNED_SELECTOR_SOURCE_SHA256 = "056833466028590172edf036962fa356291c098c2c29517c293d9d757197cae3"
SELECTOR_PATH = Path(__file__).resolve().with_name("calibrate_scifact_sentence_selector_v2.py")
DEFAULT_SELECTOR_RUN = SELECTOR_PATH.parent.parent / "eval_runs" / "scifact_sentence_selector_train_dual_threshold_v1"
CONTEXTS_NAME = "contexts_train_oof.jsonl"
DISALLOWED_SPLIT = re.compile(r"(^|[_.-])(dev|test)([_.-]|$)", re.IGNORECASE)
EXPECTED_GATE_CHECKS = frozenset({
    "official_f1_delta_vs_lexical_at_least_0_020",
    "official_f1_delta_vs_nli_at_least_0_020",
    "official_cluster_ci_lower_vs_lexical_above_zero",
    "official_cluster_ci_lower_vs_nli_above_zero",
    "at_least_3_of_4_outer_folds_noninferior_to_fold_max_baseline",
    "all_505_relation_complete_rationale_recall_at_least_0_50",
    "all_396_retrievable_complete_rationale_recall_at_least_0_65",
    "all_304_nei_correct_empty_at_least_0_70",
    "oof_claim_coverage_100_percent", "cross_fold_claim_overlap_zero",
    "cross_fold_top3_doc_overlap_zero", "feature_whitelist_violations_zero",
    "bootstrap_protocol_and_keys_exact", "hybrid_all_inner_feasible",
})


class UpstreamFreezeError(RuntimeError):
    """The supplied artifacts do not satisfy the fixed upstream contract."""


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _jsonl_bytes(rows: Sequence[dict[str, Any]]) -> bytes:
    return b"".join(_canonical(row) + b"\n" for row in rows)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _assert_train_path(path: Path, role: str) -> None:
    if DISALLOWED_SPLIT.search(path.name):
        raise UpstreamFreezeError(f"{role} must not name a dev/test artifact")


def _load_selector() -> Any:
    if _sha(SELECTOR_PATH) != PINNED_SELECTOR_SOURCE_SHA256:
        raise UpstreamFreezeError("pinned v2 selector source changed")
    spec = importlib.util.spec_from_file_location("_readonly_selector_for_upstream_freeze", SELECTOR_PATH)
    if spec is None or spec.loader is None:
        raise UpstreamFreezeError("cannot load pinned selector module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


selector_v2 = _load_selector()
base = selector_v2.base


def _bind(bindings: dict[str, Any], role: str, path: Path, expected: str) -> None:
    resolved = path.resolve()
    if not resolved.is_file() or _sha(resolved) != expected:
        raise UpstreamFreezeError(f"source SHA256 mismatch or missing file: {role}")
    bindings[role] = {"path": str(resolved), "sha256": expected, "bytes": resolved.stat().st_size}


def _recheck_bindings(bindings: Mapping[str, Any]) -> None:
    for role, item in bindings.items():
        path = Path(item["path"])
        if not path.is_file() or path.stat().st_size != item["bytes"] or _sha(path) != item["sha256"]:
            raise UpstreamFreezeError(f"frozen source binding changed: {role}")


def _validate_serialized_model(model: Mapping[str, Any], audits: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if model.get("schema_version") != selector_v2.SCHEMA_VERSION or model.get("source_split") != "train" or model.get("reportable_as_generalization") is not False:
        raise UpstreamFreezeError("selector model provenance is invalid")
    fitted = model["model"]
    if fitted["feature_names"] != list(base.FEATURE_NAMES):
        raise UpstreamFreezeError("selector feature order differs from pinned 16 features")
    scaler = fitted["standard_scaler"]
    lr = fitted["logistic_regression"]
    arrays = [scaler["mean"], scaler["variance"], scaler["scale"], lr["coefficients"]]
    if any(len(values) != 16 or any(not math.isfinite(value) for value in values) for values in arrays) or any(value <= 0 for value in scaler["scale"]) or lr["classes"] != [0, 1] or not math.isfinite(lr["intercept"]):
        raise UpstreamFreezeError("invalid serialized scaler or logistic regression")
    tfidf = fitted["tfidf"]
    if _canonical_sha({key: value for key, value in tfidf.items() if key != "state_sha256"}) != tfidf["state_sha256"]:
        raise UpstreamFreezeError("aggregate TF-IDF state hash mismatch")
    for analyzer in ("word", "char_wb"):
        state = tfidf[analyzer]
        values = {name: state[name] for name in ("parameters", "vocabulary", "idf_by_index")}
        count = state["vocabulary_size"]
        ngram = (1, 2) if analyzer == "word" else (3, 5)
        if (
            _canonical_sha(values) != state["state_sha256"]
            or state["parameters"] != base._tfidf_parameters(analyzer, ngram)
            or len(state["vocabulary"]) != count or len(state["idf_by_index"]) != count
            or set(state["vocabulary"].values()) != set(range(count))
            or any(not math.isfinite(value) or value <= 0 for value in state["idf_by_index"])
        ):
            raise UpstreamFreezeError(f"serialized {analyzer} TF-IDF state is invalid")
    choices = []
    for fold in audits:
        arm = fold["strategies"]["hybrid"]
        tuning = arm["inner_tuning"]
        if tuning["status"] != "feasible" or tuning["selected"] is None or arm["diagnostic_fallback_used"] or not all(selector_v2._constraint_checks(tuning["selected"]["metrics"]).values()):
            raise UpstreamFreezeError("hybrid has an infeasible inner selection")
        choices.append(tuning["selected"])
    if len(choices) != base.OUTER_FOLDS:
        raise UpstreamFreezeError("four feasible outer-fold choices are required")
    parameters = {name: [item[name] for item in choices] for name in ("accept_threshold", "expansion_threshold", "max_sentences")}
    counts = Counter(parameters["max_sentences"])
    expected = {
        "accept_threshold": float(statistics.median(parameters["accept_threshold"])),
        "expansion_threshold": float(statistics.median(parameters["expansion_threshold"])),
        "max_sentences": min(value for value, count in counts.items() if count == max(counts.values())),
    }
    selection = model["selection"]
    if any(selection[name] != value for name, value in expected.items()) or selection["outer_fold_parameters"] != parameters or selection["post_gate_full_train_retuning"] is not False:
        raise UpstreamFreezeError("selector parameters differ from frozen fold aggregation")
    if not 0 <= expected["expansion_threshold"] <= expected["accept_threshold"] <= 1 or expected["max_sentences"] not in base.MAX_SENTENCES:
        raise UpstreamFreezeError("selector thresholds/cap are invalid")
    return expected


def _build_contexts(
    *, claims: Sequence[dict[str, Any]], corpus: Sequence[dict[str, Any]],
    retrieval: Sequence[dict[str, Any]], frozen: Sequence[dict[str, Any]],
    cases: Sequence[dict[str, Any]], outer_folds: Sequence[dict[str, Any]],
    audits: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    claims_by_id = {int(row["id"]): row for row in claims}
    docs_by_id = {int(row["doc_id"]): row for row in corpus}
    retrieval_by_id = {int(row["claim_id"]): list(row["doc_ids"]) for row in retrieval}
    claim_ids = set(claims_by_id)
    if len(claims) != base.SOURCE_CLAIM_COUNT or len(claim_ids) != len(claims) or set(retrieval_by_id) != claim_ids or len(retrieval_by_id) != len(retrieval) or sum(bool(row.get("evidence")) for row in claims) != base.RELATION_CLAIM_COUNT:
        raise UpstreamFreezeError("train claim/retrieval coverage or validation-only relation count differs")
    if len(corpus) != base.SOURCE_CORPUS_COUNT or len(docs_by_id) != len(corpus):
        raise UpstreamFreezeError("corpus document coverage differs")
    for doc_ids in retrieval_by_id.values():
        if len(doc_ids) != base.TOP_K_DOCS or len(set(doc_ids)) != base.TOP_K_DOCS or not set(doc_ids) <= set(docs_by_id):
            raise UpstreamFreezeError("retrieval must contain exactly three known unique documents")
    isolation = base._validate_fold_isolation(outer_folds, claim_ids)
    if not isolation["claim_coverage_complete"] or isolation["cross_fold_claim_overlap_count"] or isolation["cross_fold_doc_overlap_count"]:
        raise UpstreamFreezeError("outer OOF folds overlap in claims or Top-3 documents")
    audits_by_fold = {int(row["outer_fold"]): row for row in audits}
    folds_by_id = {int(row["fold"]): row for row in outer_folds}
    if len(audits_by_fold) != base.OUTER_FOLDS or len(folds_by_id) != base.OUTER_FOLDS or set(audits_by_fold) != set(folds_by_id):
        raise UpstreamFreezeError("outer fold audit coverage is incomplete")
    claim_fold: dict[int, int] = {}
    pair_by_claim: dict[int, list[int]] = {claim_id: [] for claim_id in claim_ids}
    seen_keys = set()
    for index, pair in enumerate(frozen):
        claim_id, doc_id, sentence_index = int(pair["claim_id"]), int(pair["doc_id"]), int(pair["sentence_index"])
        if pair["pair_index"] != index or claim_id not in claim_ids or doc_id not in retrieval_by_id[claim_id]:
            raise UpstreamFreezeError("frozen pair indexing or train membership invalid")
        doc = docs_by_id[doc_id]
        key = (claim_id, doc_id, sentence_index)
        if key in seen_keys or sentence_index < 0 or sentence_index >= len(doc["abstract"]) or pair["claim"] != claims_by_id[claim_id]["claim"] or pair["sentence"] != doc["abstract"][sentence_index] or pair["doc_rank"] != retrieval_by_id[claim_id].index(doc_id) + 1:
            raise UpstreamFreezeError("frozen pair differs from raw claim/document/sentence")
        seen_keys.add(key)
        pair_by_claim[claim_id].append(index)
    expected_keys = {(claim_id, doc_id, index) for claim_id, doc_ids in retrieval_by_id.items() for doc_id in doc_ids for index in range(len(docs_by_id[doc_id]["abstract"]))}
    if seen_keys != expected_keys or len(frozen) != base.CANDIDATE_PAIR_COUNT:
        raise UpstreamFreezeError("frozen candidate pair coverage is incomplete")
    for fold_id, fold in folds_by_id.items():
        validation_ids = set(fold["claim_ids"])
        expected_train = sorted(claim_ids - validation_ids)
        validation_docs = {doc_id for claim_id in validation_ids for doc_id in retrieval_by_id[claim_id]}
        train_docs = {doc_id for claim_id in expected_train for doc_id in retrieval_by_id[claim_id]}
        if validation_docs & train_docs or set(fold["doc_ids"]) != validation_docs:
            raise UpstreamFreezeError("raw Top-3 documents contaminate outer training fold")
        arm = audits_by_fold[fold_id]["strategies"]["hybrid"]
        binding = arm["outer_model"]["tfidf"]
        if binding["fit_claim_ids"] != expected_train or binding["fit_claim_ids_sha256"] != _canonical_sha(expected_train) or binding["fit_pair_count"] != sum(len(pair_by_claim[value]) for value in expected_train):
            raise UpstreamFreezeError("OOF transformer training IDs or candidate count are invalid")
        inner_folds = audits_by_fold[fold_id]["inner_folds"]
        inner_isolation = base._validate_fold_isolation(inner_folds, set(expected_train))
        if not inner_isolation["claim_coverage_complete"] or inner_isolation["cross_fold_claim_overlap_count"] or inner_isolation["cross_fold_doc_overlap_count"]:
            raise UpstreamFreezeError("inner tuning fold isolation is invalid")
        inner_models = {int(item["inner_fold"]): item for item in arm["inner_tuning"]["inner_models"]}
        if len(inner_models) != base.INNER_FOLDS or len(inner_folds) != base.INNER_FOLDS:
            raise UpstreamFreezeError("inner tuning model coverage is incomplete")
        for inner in inner_folds:
            inner_validation = set(inner["claim_ids"])
            inner_train = sorted(set(expected_train) - inner_validation)
            inner_binding = inner_models[int(inner["fold"])]["model"]["tfidf"]
            inner_validation_docs = {doc_id for claim_id in inner_validation for doc_id in retrieval_by_id[claim_id]}
            inner_train_docs = {doc_id for claim_id in inner_train for doc_id in retrieval_by_id[claim_id]}
            if inner_validation_docs & inner_train_docs or inner_binding["fit_claim_ids"] != inner_train or inner_binding["fit_claim_ids_sha256"] != _canonical_sha(inner_train) or inner_binding["fit_pair_count"] != sum(len(pair_by_claim[value]) for value in inner_train):
                raise UpstreamFreezeError("inner training fold claim/document isolation is invalid")
        for claim_id in validation_ids:
            if claim_id in claim_fold:
                raise UpstreamFreezeError("duplicate OOF claim membership")
            claim_fold[claim_id] = fold_id
    hybrid_cases = [row for row in cases if row.get("strategy") == "hybrid"]
    case_by_id = {int(row["claim_id"]): row for row in hybrid_cases}
    if len(hybrid_cases) != len(claims) or set(case_by_id) != claim_ids:
        raise UpstreamFreezeError("hybrid OOF prediction coverage is not exact")
    contexts = []
    for claim in claims:
        claim_id = int(claim["id"])
        row = case_by_id[claim_id]
        fold_id = claim_fold[claim_id]
        arm = audits_by_fold[fold_id]["strategies"]["hybrid"]
        chosen = arm["inner_tuning"]["selected"]
        parameters = {name: chosen[name] for name in ("accept_threshold", "expansion_threshold", "max_sentences")}
        if row.get("source", "oof") != "oof" or row["outer_fold"] != fold_id or row["inner_selection_status"] != "feasible" or row["diagnostic_fallback_used"] or row["prediction_configuration_source"] != "feasible_inner_selection" or row["selection_parameters"] != parameters:
            raise UpstreamFreezeError("claim prediction is not the corresponding feasible outer-fold selection")
        selected = row["selected_pair_indices"]
        if not isinstance(selected, list) or any(type(index) is not int for index in selected) or len(set(selected)) != len(selected) or len(selected) > parameters["max_sentences"] or not set(selected) <= set(pair_by_claim[claim_id]):
            raise UpstreamFreezeError("OOF selection contains illegal, duplicate, foreign or over-budget pairs")
        selected_by_doc: dict[int, list[int]] = {}
        for index in selected:
            pair = frozen[index]
            selected_by_doc.setdefault(int(pair["doc_id"]), []).append(int(pair["sentence_index"]))
        documents = []
        for rank, doc_id in enumerate(retrieval_by_id[claim_id], 1):
            if doc_id not in selected_by_doc:
                continue
            doc = docs_by_id[doc_id]
            if not isinstance(doc["title"], str):
                raise UpstreamFreezeError("document title must be text")
            documents.append({
                "doc_id": doc_id, "rank": rank, "title": doc["title"],
                "sentences": [{"sentence_index": index, "text": doc["abstract"][index]} for index in sorted(selected_by_doc[doc_id])],
            })
        context = {"claim": claim["claim"], "documents": documents}
        contexts.append({
            "claim_id": claim_id, "source": "oof", "outer_fold": fold_id,
            "canonical_context_sha256": _canonical_sha(context), "context": context,
            "top3_doc_ids": retrieval_by_id[claim_id],
            "citation_sentence_indices": {
                str(doc_id): selected_by_doc[doc_id]
                for doc_id in retrieval_by_id[claim_id] if doc_id in selected_by_doc
            },
        })
    return contexts


def _validate_selector_run(selector_run: Path) -> dict[str, Any]:
    root = selector_run.resolve()
    _assert_train_path(root, "selector run")
    bindings: dict[str, Any] = {}
    module_path = Path(__file__).resolve()
    _bind(bindings, "upstream_freeze_implementation", module_path, _sha(module_path))
    _bind(bindings, "selector_manifest", root / "run_manifest.json", PINNED_SELECTOR_MANIFEST_SHA256)
    manifest = _read_json(root / "run_manifest.json")
    status = manifest["evaluation_status"]
    if manifest["schema_version"] != selector_v2.SCHEMA_VERSION or status["source_split"] != "train" or status["dev_read_or_scored"] is not False or status["reportable_as_generalization"] is not False or status["acceptance"] != "pass" or status["full_train_model_emitted"] is not True:
        raise UpstreamFreezeError("selector run must be a passed train-only run with exported model")
    outputs = manifest["outputs_sha256"]
    required = {"selector_model.json", "acceptance_gate.json", "frozen_features.jsonl", "nested_oof_cases.jsonl", "nested_oof_audit.json", "outer_folds.json", "configuration.json", "preregistration.json"}
    if not required <= set(outputs):
        raise UpstreamFreezeError("selector run is missing required bound outputs")
    for name, expected in outputs.items():
        path = (root / name).resolve()
        if path.parent != root or path.name != name:
            raise UpstreamFreezeError("selector output path escapes the run directory")
        _bind(bindings, "selector_output/" + name, path, expected)
    for role, expected in (("implementation", PINNED_SELECTOR_SOURCE_SHA256), ("readonly_base_implementation", selector_v2.BASE_SOURCE_SHA256)):
        item = manifest[role]
        if item["sha256"] != expected:
            raise UpstreamFreezeError("selector implementation identity differs from pinned code")
        _bind(bindings, role, Path(item["path"]), expected)
    inputs = manifest["inputs"]
    for name, item in inputs.items():
        _assert_train_path(Path(item["path"]), name)
        _bind(bindings, "input/" + name, Path(item["path"]), item["sha256"])
    corpus, claims, retrieval = base._validate_inputs(
        corpus_path=Path(inputs["corpus"]["path"]), claims_path=Path(inputs["claims_train"]["path"]),
        retrieval_path=Path(inputs["tfidf_top3_train"]["path"]), retrieval_manifest_path=Path(inputs["tfidf_manifest"]["path"]),
    )
    nli = manifest["model"]
    if nli["identity_sha256"] != base.PINNED_MODEL_IDENTITY_SHA256 or _canonical_sha(nli["files"]) != nli["identity_sha256"]:
        raise UpstreamFreezeError("pinned NLI model identity is invalid")
    nli_root = Path(nli["path"]).resolve()
    for name, item in nli["files"].items():
        path = (nli_root / name).resolve()
        if nli_root not in path.parents:
            raise UpstreamFreezeError("NLI file escapes the model snapshot")
        _bind(bindings, "nli_model/" + name, path, item["sha256"])
    cache = manifest["feature_cache"]
    cache_root = Path(cache["path"]).resolve()
    _, verified_cache_hashes = selector_v2._validate_cache(cache_root, {name: item["sha256"] for name, item in inputs.items()}, nli)
    if cache["manifest_sha256"] != selector_v2.CACHE_MANIFEST_SHA256 or cache["all_source_files_sha256"] != verified_cache_hashes:
        raise UpstreamFreezeError("selector cache source binding differs from verified cache")
    for name, expected in verified_cache_hashes.items():
        _bind(bindings, "nli_cache/" + name, cache_root / name, expected)
    prereg = _read_json(root / "preregistration.json")
    if manifest["preregistration"]["sha256"] != outputs["preregistration.json"] or manifest["preregistration"]["protocol_sha256"] != _canonical_sha(prereg["protocol"]) or prereg["protocol_sha256"] != _canonical_sha(prereg["protocol"]) or _read_json(root / "configuration.json") != prereg["protocol"]:
        raise UpstreamFreezeError("preregistration or protocol binding differs")
    for name in ("implementation", "readonly_base_implementation", "inputs", "model", "feature_cache"):
        if prereg[name] != manifest[name]:
            raise UpstreamFreezeError("preregistered source bindings differ from run manifest")
    gate = _read_json(root / "acceptance_gate.json")
    if gate["status"] != "pass" or gate["full_train_model_permitted"] is not True or set(gate["checks"]) != EXPECTED_GATE_CHECKS or not all(value is True for value in gate["checks"].values()):
        raise UpstreamFreezeError("all 14 selector acceptance checks must pass")
    audits = _read_json(root / "nested_oof_audit.json")
    model = _read_json(root / "selector_model.json")
    selection = _validate_serialized_model(model, audits)
    if model["model"]["tfidf"]["fit_claim_ids"] != sorted(int(row["id"]) for row in claims) or model["model"]["tfidf"]["fit_pair_count"] != base.CANDIDATE_PAIR_COUNT:
        raise UpstreamFreezeError("deployable selector is not bound to the complete train corpus")
    contexts = _build_contexts(
        claims=claims, corpus=corpus, retrieval=retrieval,
        frozen=_read_jsonl(root / "frozen_features.jsonl"), cases=_read_jsonl(root / "nested_oof_cases.jsonl"),
        outer_folds=_read_json(root / "outer_folds.json"), audits=audits,
    )
    _recheck_bindings(bindings)
    return {
        "selector_run": {"path": str(root), "manifest_sha256": PINNED_SELECTOR_MANIFEST_SHA256},
        "bindings": bindings, "contexts": contexts, "selection": selection,
        "validation": {
            "all_14_selector_gate_checks_pass": True, "all_source_hashes_verified": True,
            "all_contexts_from_hybrid_outer_fold_predictions": True,
            "no_full_train_model_applied_to_train": True, "outer_and_inner_claim_document_isolation_verified": True,
            "claim_count": len(contexts), "validation_only_relation_count": base.RELATION_CLAIM_COUNT,
            "new_nli_calls": 0, "new_selector_inference_calls": 0,
        },
    }


def _freeze_document(bundle: Mapping[str, Any], contexts_sha256: str, created_at: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION, "created_at": created_at,
        "source_split": "train", "dev_read_or_scored": False,
        "evaluation_mode": "repeated_train_oof_contexts_not_generalization",
        "reportable_as_generalization": False,
        "selector_run": bundle["selector_run"], "bindings": bundle["bindings"],
        "deployable_selector_selection": bundle["selection"],
        "context_contract": {
            "source": "hybrid_outer_fold_predictions_only",
            "payload_keys": ["claim", "documents"],
            "document_order": "ascending frozen Top-3 retrieval rank",
            "sentence_order": "ascending original sentence index",
            "gold_or_diagnostic_fields_in_payload": False,
            "top3_doc_ids_is_metadata_only": True,
            "citation_sentence_indices_is_metadata_only": True,
            "score_order_preserved_for_official_citations": True,
            "citation_order": "per-document order inherited from the original hybrid OOF selected_pair_indices; prompt sentence display remains original-index order",
            "canonical_json": "UTF-8; ensure_ascii=False; sort_keys=True; separators=(',', ':')",
            "empty_selection_payload_documents": [],
            "future_agent_arms_must_share_identical_context_bytes": True,
        },
        "validation": bundle["validation"],
        "outputs_sha256": {CONTEXTS_NAME: contexts_sha256},
    }


def freeze(selector_run: Path, output: Path) -> Path:
    destination = output.resolve()
    _assert_train_path(destination, "freeze output")
    if destination.exists():
        raise UpstreamFreezeError("exclusive freeze output already exists")
    bundle = _validate_selector_run(selector_run)
    context_bytes = _jsonl_bytes(bundle["contexts"])
    context_hash = hashlib.sha256(context_bytes).hexdigest()
    document = _freeze_document(bundle, context_hash, datetime.now(timezone.utc).isoformat())
    destination.mkdir(parents=True, exist_ok=False)
    with (destination / CONTEXTS_NAME).open("xb") as handle:
        handle.write(context_bytes)
    _recheck_bindings(bundle["bindings"])
    with (destination / "freeze.json").open("xb") as handle:
        handle.write(_json_bytes(document))
    _recheck_bindings(bundle["bindings"])
    if _sha(destination / CONTEXTS_NAME) != context_hash:
        raise UpstreamFreezeError("context artifact changed during freeze")
    return destination


def load_frozen(path: Path) -> dict[str, Any]:
    root = path.resolve() if path.is_dir() else path.resolve().parent
    document_path = root / "freeze.json"
    if not path.is_dir() and path.name != "freeze.json":
        raise UpstreamFreezeError("load_frozen expects the freeze directory or freeze.json")
    _assert_train_path(root, "frozen context input")
    document = _read_json(document_path)
    initial_document_hash = _sha(document_path)
    if document.get("schema_version") != SCHEMA_VERSION or document.get("source_split") != "train" or document.get("dev_read_or_scored") is not False:
        raise UpstreamFreezeError("freeze must use the train OOF context schema")
    bundle = _validate_selector_run(Path(document["selector_run"]["path"]))
    context_path = root / CONTEXTS_NAME
    context_hash = _sha(context_path)
    expected = _freeze_document(bundle, context_hash, document["created_at"])
    if document != expected:
        raise UpstreamFreezeError("freeze metadata or source bindings differ from verified upstream")
    contexts = _read_jsonl(context_path)
    if contexts != bundle["contexts"] or _jsonl_bytes(contexts) != context_path.read_bytes():
        raise UpstreamFreezeError("frozen contexts differ from verified raw OOF reconstruction")
    _recheck_bindings(bundle["bindings"])
    if _sha(document_path) != initial_document_hash or _sha(context_path) != context_hash:
        raise UpstreamFreezeError("freeze artifacts changed during load")
    return {"freeze": document, "contexts": contexts}


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selector-run", type=Path, default=DEFAULT_SELECTOR_RUN)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main() -> None:
    args = _parse_args()
    destination = freeze(args.selector_run, args.out)
    print(destination)


if __name__ == "__main__":
    main()
