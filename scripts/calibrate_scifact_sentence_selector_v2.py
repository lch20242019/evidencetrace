"""Train-only, nested-OOF calibration of document admission and evidence expansion.

Only the decision rule and its constrained tuning objective change relative to
the pinned v1 implementation. Static NLI features are reused from an explicitly
identified prior train run; all TF-IDF, scaler and classifier fitting remains
inside the original document-disjoint folds. Repeated train results are not
generalization estimates. No dev/test arguments or inference fallback exist.
"""

from __future__ import annotations

# The supported project environment is Python 3.9.
# ruff: noqa: E501, UP017, UP045
import argparse
import importlib.metadata
import importlib.util
import json
import math
import statistics
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

BASE_SOURCE_SHA256 = "9d11dcc3575af86c33055b941cfaefb29b2d5ca287e09a15662a725a6ffe848a"
CACHE_MANIFEST_SHA256 = "c506f3c49cb75f244862531ddf35c3f902488b64cf0468bcf07f9da5d278471d"
CACHE_FEATURES_SHA256 = "afa9e7a99ac91a6c8f3084026e47eed2c402669b4452b1164e313e723f708c5f"
BASE_PATH = Path(__file__).resolve().with_name("calibrate_scifact_sentence_selector.py")


def _load_base() -> Any:
    import hashlib

    if hashlib.sha256(BASE_PATH.read_bytes()).hexdigest() != BASE_SOURCE_SHA256:
        raise RuntimeError("v1 selector source differs from the read-only pinned version")
    spec = importlib.util.spec_from_file_location("_scifact_selector_v2_readonly_base", BASE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load pinned selector base")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _load_base()
SelectorCalibrationError = base.SelectorCalibrationError
SCHEMA_VERSION = "scifact-sentence-selector-calibration-v2"
EVALUATION_MODE = "repeated_train_only_not_generalization"
THRESHOLDS = base.THRESHOLDS
MAX_SENTENCES = base.MAX_SENTENCES
STRATEGIES = base.STRATEGIES
GRID_SIZE = len(THRESHOLDS) * (len(THRESHOLDS) + 1) // 2 * len(MAX_SENTENCES)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent


def _validate_runtime_environment() -> dict[str, str]:
    expected = {"scikit-learn": "1.4.2", "numpy": "1.26.4", "scipy": "1.13.0"}
    if sys.version_info[:2] != (3, 9):
        raise SelectorCalibrationError("calibration requires the project's pinned Python 3.9 environment")
    actual = {name: importlib.metadata.version(name) for name in expected}
    if actual != expected:
        raise SelectorCalibrationError(f"calibration requires pinned scikit-learn/numpy/scipy versions: {expected}")
    return {"python": sys.version, **actual}


def _protocol() -> dict[str, Any]:
    protocol = base._protocol()
    protocol.update(schema_version=SCHEMA_VERSION, evaluation_mode=EVALUATION_MODE)
    protocol["previous_train_results_observed"] = True
    protocol["scope_of_change"] = [
        "per-document admission threshold plus within-document sentence expansion threshold",
        "constrained tuning using official-compatible pooled sentence-selection F1",
    ]
    protocol["features"]["nli_inference_passes"] = 0
    protocol["features"]["nli_feature_source"] = {
        "mode": "reuse_pinned_prior_train_static_features",
        "manifest_sha256": CACHE_MANIFEST_SHA256,
        "frozen_features_sha256": CACHE_FEATURES_SHA256,
        "reused_pair_count": base.CANDIDATE_PAIR_COUNT,
        "original_inference_passes": 1,
        "all_fold_fit_transforms_and_classifiers_refit": True,
    }
    protocol["selection"] = {
        "admission_scope": "each document independently within the frozen Top-3 candidates",
        "admission": "document maximum sentence score >= accept_threshold",
        "expansion": "only admitted documents; sentence score >= expansion_threshold",
        "threshold_order": "expansion_threshold <= accept_threshold",
        "sentence_cap_scope": "claim-global after expansion from admitted documents",
        "ranking": "score descending, doc_rank, sentence_index, doc_id, pair_index ascending",
        "no_gold_used_at_selection": True,
        "equal_thresholds": "exactly equivalent to v1 single-threshold selection",
    }
    protocol["inner_selection"] = {
        "accept_thresholds": list(THRESHOLDS),
        "expansion_thresholds": list(THRESHOLDS),
        "max_sentences": list(MAX_SENTENCES),
        "grid_size": GRID_SIZE,
        "same_grid_for_all_strategies": True,
        "constraints": {
            "all_relation_complete_rationale_recall": base.MIN_ALL_RELATION_ANY_GOLD_RECALL,
            "retrievable_relation_complete_rationale_recall": base.MIN_RETRIEVABLE_ANY_GOLD_RECALL,
            "nei_correct_empty_rate": base.MIN_NEI_CORRECT_EMPTY,
        },
        "rule": "retain configurations satisfying all three constraints; maximize official-compatible pooled sentence-selection F1",
        "tie_breaking_in_order": [
            "higher all-relation complete-rationale recall",
            "smaller max_sentences",
            "higher accept_threshold",
            "higher expansion_threshold",
        ],
        "custom_f1_used_in_selection": False,
        "infeasible_policy": {
            "status": "infeasible",
            "selected": None,
            "diagnostic_unconstrained_best": "actual whole-grid official-F1-maximizing configuration with identical tie-breaks",
            "outer_predictions": "real diagnostic fallback, tagged in every fold and case; never fabricated empty predictions",
            "hybrid": "any outer fold with infeasible inner selection forces final rejection and forbids model export",
            "baselines": "tagged unconstrained fallback remains the comparison; no claim of constraint compliance",
        },
    }
    protocol["acceptance"]["hybrid_all_inner_feasible"] = True
    protocol["final_refit"].pop("threshold")
    protocol["final_refit"].update({
        "accept_threshold": "median of four feasible outer-fold inner selections",
        "expansion_threshold": "median of four feasible outer-fold inner selections",
        "threshold_order_verified": True,
    })
    protocol["fail_closed_structural_checks"].extend([
        "pinned v1 source and v2 source unchanged during run",
        "pinned cache manifest plus every declared output hash verified before and after run",
        "cache static features reconstructed exactly from raw candidates and cached NLI probabilities",
        "cache targets, connected components and outer folds reconstructed exactly from train",
    ])
    return protocol


def _select_pairs(
    claim_ids: Sequence[int],
    by_claim: Mapping[int, Sequence[int]],
    scores: Mapping[int, float],
    frozen: Sequence[dict[str, Any]],
    *,
    accept_threshold: float,
    expansion_threshold: float,
    max_sentences: int,
) -> dict[int, list[int]]:
    if (
        not math.isfinite(accept_threshold)
        or not math.isfinite(expansion_threshold)
        or not 0.0 <= expansion_threshold <= accept_threshold <= 1.0
        or type(max_sentences) is not int
        or max_sentences not in MAX_SENTENCES
    ):
        raise SelectorCalibrationError("invalid dual thresholds or sentence cap")
    selected: dict[int, list[int]] = {}
    for claim_id in claim_ids:
        available = [index for index in by_claim[claim_id] if index in scores]
        if any(not math.isfinite(scores[index]) or not 0.0 <= scores[index] <= 1.0 for index in available):
            raise SelectorCalibrationError("non-finite or out-of-range selection score")
        admitted_docs = {
            int(frozen[index]["doc_id"])
            for index in available
            if scores[index] >= accept_threshold
        }
        eligible = [
            index for index in available
            if int(frozen[index]["doc_id"]) in admitted_docs
            and scores[index] >= expansion_threshold
        ]
        eligible.sort(key=lambda index: (
            -scores[index], int(frozen[index]["doc_rank"]),
            int(frozen[index]["sentence_index"]), int(frozen[index]["doc_id"]), index,
        ))
        selected[claim_id] = eligible[:max_sentences]
    return selected


def _constraint_checks(metrics: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "all_relation_complete_rationale_recall": float(metrics["all_relation_complete_rationale_recall"]) >= base.MIN_ALL_RELATION_ANY_GOLD_RECALL,
        "retrievable_relation_complete_rationale_recall": float(metrics["retrievable_relation_complete_rationale_recall"]) >= base.MIN_RETRIEVABLE_ANY_GOLD_RECALL,
        "nei_correct_empty_rate": float(metrics["nei_correct_empty_rate"]) >= base.MIN_NEI_CORRECT_EMPTY,
    }


def _configuration_key(
    metrics: Mapping[str, Any], accept_threshold: float,
    expansion_threshold: float, maximum: int,
) -> tuple[Any, ...]:
    return (
        float(metrics["official_compatible_pooled_sentence_selection"]["f1"]),
        float(metrics["all_relation_complete_rationale_recall"]),
        -maximum, accept_threshold, expansion_threshold,
    )


def _sweep_selection(
    *, claim_ids: Sequence[int], by_claim: Mapping[int, Sequence[int]],
    scores: Mapping[int, float], frozen: Sequence[dict[str, Any]],
    claims_by_id: Mapping[int, dict[str, Any]], retrievable_relation_ids: set[int],
) -> dict[str, Any]:
    sweep: list[dict[str, Any]] = []
    best: Optional[dict[str, Any]] = None
    unconstrained: Optional[dict[str, Any]] = None
    failures = Counter({name: 0 for name in _protocol()["inner_selection"]["constraints"]})
    metrics_cache: dict[tuple[tuple[int, ...], ...], dict[str, Any]] = {}
    for accept_threshold in THRESHOLDS:
        for expansion_threshold in THRESHOLDS:
            if expansion_threshold > accept_threshold:
                continue
            for maximum in MAX_SENTENCES:
                selected = _select_pairs(
                    claim_ids, by_claim, scores, frozen,
                    accept_threshold=accept_threshold,
                    expansion_threshold=expansion_threshold, max_sentences=maximum,
                )
                prediction_key = tuple(tuple(selected[claim_id]) for claim_id in claim_ids)
                metrics = metrics_cache.get(prediction_key)
                if metrics is None:
                    computed = base._selector_metrics(
                        claim_ids, claims_by_id, frozen, selected, retrievable_relation_ids,
                    )
                    metrics = {key: value for key, value in computed.items() if key != "per_claim"}
                    metrics_cache[prediction_key] = metrics
                checks = _constraint_checks(metrics)
                failures.update(name for name, passed in checks.items() if not passed)
                row = {
                    "accept_threshold": accept_threshold,
                    "expansion_threshold": expansion_threshold,
                    "max_sentences": maximum, "metrics": metrics,
                    "constraint_checks": checks, "feasible": all(checks.values()),
                }
                sweep.append(row)
                key = _configuration_key(metrics, accept_threshold, expansion_threshold, maximum)
                if unconstrained is None or key > _configuration_key(
                    unconstrained["metrics"], unconstrained["accept_threshold"],
                    unconstrained["expansion_threshold"], unconstrained["max_sentences"],
                ):
                    unconstrained = row
                if row["feasible"] and (best is None or key > _configuration_key(
                    best["metrics"], best["accept_threshold"],
                    best["expansion_threshold"], best["max_sentences"],
                )):
                    best = row
    if len(sweep) != GRID_SIZE or unconstrained is None:
        raise SelectorCalibrationError("dual-threshold grid is incomplete")
    return {
        "status": "feasible" if best is not None else "infeasible",
        "selected": best,
        "diagnostic_unconstrained_best": unconstrained,
        "grid_size": len(sweep), "feasible_configuration_count": sum(row["feasible"] for row in sweep),
        "failed_constraint_counts": dict(failures),
        "failure_reason": None if best is not None else "no grid configuration satisfies all three constraints",
        "unique_prediction_sets_evaluated": len(metrics_cache), "sweep": sweep,
    }


def _fit_and_score(
    *, strategy: str, frozen: Sequence[dict[str, Any]],
    targets: Sequence[dict[str, Any]], train_ids: Sequence[int],
    validation_ids: Sequence[int], by_claim: Mapping[int, Sequence[int]],
    transformer_cache: dict[tuple[int, ...], Mapping[str, Any]],
) -> tuple[dict[int, float], Optional[dict[str, Any]]]:
    train_pairs = [index for claim_id in train_ids for index in by_claim[claim_id]]
    validation_pairs = [index for claim_id in validation_ids for index in by_claim[claim_id]]
    fitted = None
    model = None
    transformer = None
    if strategy in {"hybrid", "lexical_only"}:
        cache_key = tuple(train_pairs)
        transformer = transformer_cache.get(cache_key)
        if transformer is None:
            transformer = base._fit_fold_transformer(frozen, train_pairs)
            transformer_cache[cache_key] = transformer
        if list(transformer["fit_claim_ids"]) != list(train_ids):
            raise SelectorCalibrationError("transformer fit claim IDs differ from expected training IDs")
    if strategy == "hybrid":
        scaler, classifier, model, transformer = base._fit_hybrid(
            frozen, targets, train_pairs, transformer=transformer, include_vectorizer_state=False,
        )
        fitted = (scaler, classifier)
    elif strategy == "lexical_only":
        assert transformer is not None
        model = {"tfidf": base._fold_transformer_binding(transformer)}
    scores = base._score_pairs(
        strategy, frozen, validation_pairs, transformer=transformer, fitted=fitted,
    )
    return scores, model


def _tune_selection(
    *, strategy: str, inner_folds: Sequence[dict[str, Any]],
    outer_train_claim_ids: Sequence[int], frozen: Sequence[dict[str, Any]],
    targets: Sequence[dict[str, Any]], by_claim: Mapping[int, Sequence[int]],
    claims_by_id: Mapping[int, dict[str, Any]], retrievable_relation_ids: set[int],
    transformer_cache: Optional[dict[tuple[int, ...], Mapping[str, Any]]] = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cache = transformer_cache if transformer_cache is not None else {}
    outer_set = set(outer_train_claim_ids)
    covered: set[int] = set()
    flat_scores: dict[int, float] = {}
    fold_models: list[dict[str, Any]] = []
    for inner in inner_folds:
        validation_ids = list(inner["claim_ids"])
        if covered & set(validation_ids):
            raise SelectorCalibrationError("duplicate inner OOF claim coverage")
        train_ids = sorted(outer_set - set(validation_ids))
        scores, model = _fit_and_score(
            strategy=strategy, frozen=frozen, targets=targets,
            train_ids=train_ids, validation_ids=validation_ids,
            by_claim=by_claim, transformer_cache=cache,
        )
        expected_pairs = {index for claim_id in validation_ids for index in by_claim[claim_id]}
        if set(scores) != expected_pairs:
            raise SelectorCalibrationError("inner OOF score pair coverage is incomplete")
        flat_scores.update(scores)
        covered.update(validation_ids)
        fold_models.append({
            "inner_fold": int(inner["fold"]), "train_claim_count": len(train_ids),
            "validation_claim_count": len(validation_ids), "model": model,
        })
    if covered != outer_set:
        raise SelectorCalibrationError("inner OOF score claim coverage is incomplete")
    tuning = _sweep_selection(
        claim_ids=outer_train_claim_ids, by_claim=by_claim, scores=flat_scores,
        frozen=frozen, claims_by_id=claims_by_id,
        retrievable_relation_ids=retrievable_relation_ids,
    )
    tuning["inner_models"] = fold_models
    return tuning, tuning["sweep"]


def _run_nested_oof(
    *, claims: Sequence[dict[str, Any]], frozen: Sequence[dict[str, Any]],
    targets: Sequence[dict[str, Any]], by_claim: Mapping[int, Sequence[int]],
    components: Sequence[dict[str, Any]], outer_folds: Sequence[dict[str, Any]],
    retrievable_relation_ids: set[int],
) -> tuple[dict[str, Any], dict[str, dict[int, list[int]]], list[dict[str, Any]], list[dict[str, Any]]]:
    claims_by_id = {int(row["id"]): row for row in claims}
    all_claim_ids = sorted(claims_by_id)
    component_by_id = {row["component_id"]: row for row in components}
    selected_by_strategy: dict[str, dict[int, list[int]]] = {strategy: {} for strategy in STRATEGIES}
    case_provenance: dict[str, dict[int, dict[str, Any]]] = {strategy: {} for strategy in STRATEGIES}
    audit_folds: list[dict[str, Any]] = []
    for outer in outer_folds:
        cache: dict[tuple[int, ...], Mapping[str, Any]] = {}
        validation_ids = list(outer["claim_ids"])
        train_ids = sorted(set(all_claim_ids) - set(validation_ids))
        train_component_ids = sorted(set(component_by_id) - set(outer["component_ids"]))
        inner_folds = base._assign_component_folds(
            [component_by_id[value] for value in train_component_ids], base.INNER_FOLDS,
        )
        isolation = base._validate_fold_isolation(inner_folds, set(train_ids))
        if not isolation["claim_coverage_complete"] or isolation["cross_fold_claim_overlap_count"] or isolation["cross_fold_doc_overlap_count"]:
            raise SelectorCalibrationError("inner component isolation failed")
        audit = {
            "outer_fold": int(outer["fold"]), "train_claim_count": len(train_ids),
            "validation_claim_count": len(validation_ids), "inner_folds": inner_folds,
            "inner_isolation": isolation, "strategies": {},
        }
        for strategy in STRATEGIES:
            print("outer fold {}/{} strategy {}: fit inner folds and tune {} configurations".format(
                int(outer["fold"]) + 1, base.OUTER_FOLDS, strategy, GRID_SIZE,
            ), flush=True)
            tuning, _ = _tune_selection(
                strategy=strategy, inner_folds=inner_folds, outer_train_claim_ids=train_ids,
                frozen=frozen, targets=targets, by_claim=by_claim, claims_by_id=claims_by_id,
                retrievable_relation_ids=retrievable_relation_ids, transformer_cache=cache,
            )
            scores, model = _fit_and_score(
                strategy=strategy, frozen=frozen, targets=targets,
                train_ids=train_ids, validation_ids=validation_ids,
                by_claim=by_claim, transformer_cache=cache,
            )
            expected_pairs = {index for claim_id in validation_ids for index in by_claim[claim_id]}
            if set(scores) != expected_pairs or set(selected_by_strategy[strategy]) & set(validation_ids):
                raise SelectorCalibrationError("outer OOF scores or claim coverage invalid")
            fallback = tuning["selected"] is None
            chosen = tuning["diagnostic_unconstrained_best"] if fallback else tuning["selected"]
            selection_parameters = {key: chosen[key] for key in (
                "accept_threshold", "expansion_threshold", "max_sentences",
            )}
            selected = _select_pairs(validation_ids, by_claim, scores, frozen, **selection_parameters)
            selected_by_strategy[strategy].update(selected)
            provenance = {
                "outer_fold": int(outer["fold"]),
                "inner_selection_status": tuning["status"],
                "diagnostic_fallback_used": fallback,
                "prediction_configuration_source": "diagnostic_unconstrained_best" if fallback else "feasible_inner_selection",
                "selection_parameters": selection_parameters,
            }
            for claim_id in validation_ids:
                case_provenance[strategy][claim_id] = provenance
            fold_metrics = base._selector_metrics(
                validation_ids, claims_by_id, frozen, selected, retrievable_relation_ids,
            )
            audit["strategies"][strategy] = {
                "inner_tuning": tuning, "outer_model": model, **provenance,
                "outer_validation_metrics": {key: value for key, value in fold_metrics.items() if key != "per_claim"},
            }
            print("outer fold {} strategy {}: inner={}, diagnostic_fallback={}, official_F1={:.6f}".format(
                int(outer["fold"]) + 1, strategy, tuning["status"], fallback,
                fold_metrics["official_compatible_pooled_sentence_selection"]["f1"],
            ), flush=True)
        audit_folds.append(audit)
        cache.clear()
    metrics: dict[str, Any] = {}
    cases: list[dict[str, Any]] = []
    for strategy in STRATEGIES:
        if set(selected_by_strategy[strategy]) != set(all_claim_ids):
            raise SelectorCalibrationError("outer OOF coverage is incomplete")
        result = base._selector_metrics(
            all_claim_ids, claims_by_id, frozen, selected_by_strategy[strategy], retrievable_relation_ids,
        )
        metrics[strategy] = {key: value for key, value in result.items() if key != "per_claim"}
        metrics[strategy]["diagnostic_fallback_claim_count"] = sum(
            row["diagnostic_fallback_used"] for row in case_provenance[strategy].values()
        )
        for claim_id in all_claim_ids:
            cases.append({
                "strategy": strategy, "claim_id": claim_id,
                "selected_pair_indices": selected_by_strategy[strategy][claim_id],
                **result["per_claim"][str(claim_id)], **case_provenance[strategy][claim_id],
            })
        rows = [row for row in cases if row["strategy"] == strategy]
        if len(rows) != len(all_claim_ids) or len({row["claim_id"] for row in rows}) != len(all_claim_ids) or sum(bool(row["is_relation"]) for row in rows) != base.RELATION_CLAIM_COUNT:
            raise SelectorCalibrationError("per-strategy OOF case coverage is invalid")
    return metrics, selected_by_strategy, audit_folds, cases


def _gate(
    *, metrics: dict[str, Any], audit_folds: Sequence[dict[str, Any]],
    cases: Sequence[dict[str, Any]], bootstrap: dict[str, Any],
    isolation: dict[str, Any], feature_whitelist_violations: int,
) -> dict[str, Any]:
    gate = base._gate(
        metrics=metrics, audit_folds=audit_folds, cases=cases, bootstrap=bootstrap,
        isolation=isolation, feature_whitelist_violations=feature_whitelist_violations,
    )
    feasible = len(audit_folds) == base.OUTER_FOLDS and all(
        fold["strategies"]["hybrid"]["inner_tuning"]["status"] == "feasible"
        and fold["strategies"]["hybrid"]["inner_tuning"]["selected"] is not None
        and all(_constraint_checks(fold["strategies"]["hybrid"]["inner_tuning"]["selected"]["metrics"]).values())
        and not fold["strategies"]["hybrid"]["diagnostic_fallback_used"]
        for fold in audit_folds
    )
    gate["schema_version"] = SCHEMA_VERSION
    gate["checks"]["hybrid_all_inner_feasible"] = feasible
    gate["status"] = "pass" if all(gate["checks"].values()) else "reject"
    gate["full_train_model_permitted"] = gate["status"] == "pass"
    gate["infeasible_outer_folds_by_strategy"] = {
        strategy: [int(fold["outer_fold"]) for fold in audit_folds
                   if fold["strategies"][strategy]["inner_tuning"]["status"] == "infeasible"]
        for strategy in STRATEGIES
    }
    gate["comparison_policy"] = "real tagged unconstrained outer predictions used when baseline inner constraints infeasible"
    return gate


def _full_train_model(
    *, claims: Sequence[dict[str, Any]], frozen: Sequence[dict[str, Any]],
    targets: Sequence[dict[str, Any]], by_claim: Mapping[int, Sequence[int]],
    audit_folds: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    choices = [fold["strategies"]["hybrid"]["inner_tuning"] for fold in audit_folds]
    if len(choices) != base.OUTER_FOLDS or any(
        item["status"] != "feasible" or item["selected"] is None
        or not all(_constraint_checks(item["selected"]["metrics"]).values()) for item in choices
    ):
        raise SelectorCalibrationError("final parameters require all four feasible inner selections")
    parameters = {name: [item["selected"][name] for item in choices]
                  for name in ("accept_threshold", "expansion_threshold", "max_sentences")}
    accept = float(statistics.median(parameters["accept_threshold"]))
    expansion = float(statistics.median(parameters["expansion_threshold"]))
    counts = Counter(parameters["max_sentences"])
    maximum = min(value for value, count in counts.items() if count == max(counts.values()))
    if not 0.0 <= expansion <= accept <= 1.0:
        raise SelectorCalibrationError("final median thresholds violate ordering")
    _, _, model, transformer = base._fit_hybrid(
        frozen, targets, list(range(len(frozen))), include_vectorizer_state=True,
    )
    expected_ids = sorted(int(row["id"]) for row in claims)
    if list(transformer["fit_claim_ids"]) != expected_ids or set(by_claim) != set(expected_ids):
        raise SelectorCalibrationError("full model fit claim coverage invalid")
    return {
        "schema_version": SCHEMA_VERSION,
        "training_scope": "all_809_official_train_claims_after_nested_oof_gate_pass",
        "source_split": "train", "reportable_as_generalization": False,
        "selection": {
            **_protocol()["selection"], "accept_threshold": accept,
            "expansion_threshold": expansion, "max_sentences": maximum,
            "outer_fold_parameters": parameters,
            "threshold_rule": "separate medians of four feasible outer-fold inner selections",
            "max_sentences_rule": "mode of four outer-fold choices; ties choose smaller",
            "post_gate_full_train_retuning": False,
        }, "model": model,
    }


def _validate_cache(
    cache_dir: Path, input_hashes: Mapping[str, str], model_identity: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    root = cache_dir.resolve()
    base._assert_not_dev(root, "feature cache")
    manifest_path = root / "run_manifest.json"
    if base._sha256(manifest_path) != CACHE_MANIFEST_SHA256:
        raise SelectorCalibrationError("feature cache manifest identity mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != base.SCHEMA_VERSION
        or manifest.get("implementation", {}).get("sha256") != BASE_SOURCE_SHA256
        or manifest.get("evaluation_status", {}).get("source_split") != "train"
        or manifest.get("evaluation_status", {}).get("dev_read_or_scored") is not False
        or manifest.get("evaluation_status", {}).get("reportable_as_generalization") is not False
        or manifest.get("inference", {}).get("inference_pass_count") != 1
        or manifest.get("inference", {}).get("pair_count") != base.CANDIDATE_PAIR_COUNT
    ):
        raise SelectorCalibrationError("cache is not the pinned train-only NLI run")
    cached_input_hashes = {name: item["sha256"] for name, item in manifest["inputs"].items()}
    if cached_input_hashes != dict(input_hashes):
        raise SelectorCalibrationError("cache train input bindings differ from current inputs")
    for key in ("revision", "identity_sha256", "files", "id2label"):
        if manifest["model"][key] != model_identity[key]:
            raise SelectorCalibrationError("cache model binding differs from the pinned local model")
    declared = manifest.get("outputs_sha256", {})
    required = {"frozen_features.jsonl", "training_targets.jsonl", "connected_components.json", "outer_folds.json", "preregistration.json"}
    if not required.issubset(declared) or declared["frozen_features.jsonl"] != CACHE_FEATURES_SHA256:
        raise SelectorCalibrationError("cache output bindings are incomplete or changed")
    hashes = {"run_manifest.json": CACHE_MANIFEST_SHA256}
    for name, expected in declared.items():
        path = (root / name).resolve()
        if path.parent != root or path.name != name or not path.is_file():
            raise SelectorCalibrationError("cache output path is missing or escapes its directory")
        actual = base._sha256(path)
        if actual != expected:
            raise SelectorCalibrationError(f"cache output SHA256 mismatch: {name}")
        hashes[name] = actual
    return manifest, hashes


def _reconstruct_frozen_features(
    cache_dir: Path, candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    cached = base._read_jsonl(cache_dir / "frozen_features.jsonl")
    probabilities = [[row["static_features"][name] for name in (
        "nli_entailment", "nli_neutral", "nli_contradiction",
    )] for row in cached]
    frozen = base._freeze_features(candidates, probabilities)
    if frozen != cached or base._sha256_bytes(base._jsonl_bytes(frozen)) != CACHE_FEATURES_SHA256:
        raise SelectorCalibrationError("cached features do not exactly reconstruct from raw train candidates")
    return frozen


def _verify_unchanged(
    *, script_path: Path, script_sha256: str, input_paths: Mapping[str, Path],
    input_hashes: Mapping[str, str], model_dir: Path, model_identity: Mapping[str, Any],
    cache_dir: Path, cache_hashes: Mapping[str, str], preregistration_path: Path,
    preregistration_sha256: str,
) -> None:
    if base._sha256(script_path) != script_sha256 or base._sha256(BASE_PATH) != BASE_SOURCE_SHA256:
        raise SelectorCalibrationError("implementation changed during run")
    if {name: base._sha256(path) for name, path in input_paths.items()} != dict(input_hashes):
        raise SelectorCalibrationError("train inputs changed during run")
    if base._model_identity(model_dir) != dict(model_identity):
        raise SelectorCalibrationError("local model identity changed during run")
    if {name: base._sha256(cache_dir / name) for name in cache_hashes} != dict(cache_hashes):
        raise SelectorCalibrationError("cache artifacts changed during run")
    if base._sha256(preregistration_path) != preregistration_sha256:
        raise SelectorCalibrationError("preregistration changed during run")


def run(
    *, corpus_path: Path, claims_path: Path, retrieval_path: Path,
    retrieval_manifest_path: Path, model_dir: Path, cache_dir: Path, output: Path,
    bootstrap_samples: int = base.BOOTSTRAP_SAMPLES,
) -> Path:
    started = time.perf_counter()
    destination = output.resolve()
    base._assert_not_dev(destination, "output")
    if destination.exists():
        raise SelectorCalibrationError("output directory already exists (exclusive run)")
    if bootstrap_samples != base.BOOTSTRAP_SAMPLES:
        raise SelectorCalibrationError("formal bootstrap sample count must equal 10000")
    environment = _validate_runtime_environment()
    script_path = Path(__file__).resolve()
    script_sha256 = base._sha256(script_path)
    if base._sha256(BASE_PATH) != BASE_SOURCE_SHA256:
        raise SelectorCalibrationError("pinned read-only base implementation changed")
    input_paths = {
        "corpus": corpus_path.resolve(), "claims_train": claims_path.resolve(),
        "tfidf_top3_train": retrieval_path.resolve(), "tfidf_manifest": retrieval_manifest_path.resolve(),
    }
    corpus, claims, retrieval = base._validate_inputs(
        corpus_path=input_paths["corpus"], claims_path=input_paths["claims_train"],
        retrieval_path=input_paths["tfidf_top3_train"], retrieval_manifest_path=input_paths["tfidf_manifest"],
    )
    input_hashes = {name: base._sha256(path) for name, path in input_paths.items()}
    model_identity = base._model_identity(model_dir)
    cache_dir = cache_dir.resolve()
    cache_manifest, cache_hashes = _validate_cache(cache_dir, input_hashes, model_identity)
    protocol = _protocol()
    protocol_sha256 = base._sha256_bytes(base._canonical_bytes(protocol))
    source_bindings = {
        "implementation": {"path": str(script_path), "sha256": script_sha256},
        "readonly_base_implementation": {"path": str(BASE_PATH), "sha256": BASE_SOURCE_SHA256},
        "inputs": {name: {"path": str(path), "sha256": input_hashes[name]} for name, path in input_paths.items()},
        "model": model_identity,
        "feature_cache": {
            "path": str(cache_dir), "manifest_sha256": CACHE_MANIFEST_SHA256,
            "all_source_files_sha256": cache_hashes,
            "original_inference": cache_manifest["inference"],
            "source_is_prior_train_run": True,
        },
    }
    preregistration = {
        "schema_version": SCHEMA_VERSION, "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_oof": True, "new_nli_inference_performed": False,
        "previous_train_results_observed": True, **source_bindings,
        "protocol": protocol, "protocol_sha256": protocol_sha256,
        "crash_policy": "retain incomplete exclusive output directory",
    }
    destination.mkdir(parents=True, exist_ok=False)
    preregistration_path = destination / "preregistration.json"
    base._write_exclusive(preregistration_path, base._json_bytes(preregistration))
    preregistration_sha256 = base._sha256(preregistration_path)
    candidates, by_claim, retrieval_by_claim = base._build_candidates(corpus, claims, retrieval)
    claim_ids = [int(row["id"]) for row in claims]
    candidate_validation = base._validate_candidate_keys(candidates, by_claim, set(claim_ids))
    frozen = _reconstruct_frozen_features(cache_dir, candidates)
    targets = base._build_targets(claims, frozen)
    if targets != base._read_jsonl(cache_dir / "training_targets.jsonl"):
        raise SelectorCalibrationError("cache targets differ from freshly reconstructed train targets")
    retrievable_ids = {int(row["claim_id"]) for row in targets if row["is_gold_sentence"]}
    components = base._connected_components(
        claim_ids, retrieval_by_claim, {int(row["id"]): str(row["claim"]) for row in claims},
    )
    if (
        len(frozen) != base.CANDIDATE_PAIR_COUNT
        or sum(bool(row["is_gold_sentence"]) for row in targets) != base.POSITIVE_PAIR_COUNT
        or len(retrievable_ids) != base.RETRIEVABLE_RELATION_CLAIM_COUNT
        or len(components) != base.CONNECTED_COMPONENT_COUNT
        or max(int(row["claim_count"]) for row in components) != base.LARGEST_COMPONENT_CLAIM_COUNT
    ):
        raise SelectorCalibrationError("fixed train candidate/target/component counts changed")
    outer_folds = base._assign_component_folds(components, base.OUTER_FOLDS)
    for name, value in (("connected_components.json", components), ("outer_folds.json", outer_folds)):
        if json.loads((cache_dir / name).read_text(encoding="utf-8")) != value:
            raise SelectorCalibrationError(f"reconstructed train split differs from cache: {name}")
    isolation = base._validate_fold_isolation(outer_folds, set(claim_ids))
    if not isolation["claim_coverage_complete"] or isolation["cross_fold_claim_overlap_count"] or isolation["cross_fold_doc_overlap_count"]:
        raise SelectorCalibrationError("outer fold isolation failed")
    fold_statistics = base._fold_statistics(outer_folds, {int(row["id"]): row for row in claims}, retrievable_ids)
    print(f"Verified pinned cache: reused {len(frozen)} NLI pairs; new NLI calls=0; starting train-only OOF", flush=True)
    metrics, selections, audit_folds, cases = _run_nested_oof(
        claims=claims, frozen=frozen, targets=targets, by_claim=by_claim,
        components=components, outer_folds=outer_folds, retrievable_relation_ids=retrievable_ids,
    )
    print("OOF complete; paired 10000-sample component bootstrap for both metrics and baselines", flush=True)
    bootstrap = base._bootstrap_comparisons(components=components, cases=cases, samples=bootstrap_samples)
    violations = sum(tuple(row["static_features"]) != base.STATIC_FEATURE_NAMES for row in frozen) + int(len(base.FEATURE_NAMES) != 16)
    gate = _gate(metrics=metrics, audit_folds=audit_folds, cases=cases, bootstrap=bootstrap, isolation=isolation, feature_whitelist_violations=violations)
    official_metrics = {strategy: base._official_style_metrics(claims, frozen, selections[strategy]) for strategy in STRATEGIES}
    verification = dict(
        script_path=script_path, script_sha256=script_sha256,
        input_paths=input_paths, input_hashes=input_hashes, model_dir=model_dir,
        model_identity=model_identity, cache_dir=cache_dir, cache_hashes=cache_hashes,
        preregistration_path=preregistration_path, preregistration_sha256=preregistration_sha256,
    )
    _verify_unchanged(**verification)
    final_model = None
    if gate["full_train_model_permitted"]:
        final_model = _full_train_model(claims=claims, frozen=frozen, targets=targets, by_claim=by_claim, audit_folds=audit_folds)
    _verify_unchanged(**verification)
    json_outputs = {
        "configuration.json": protocol, "connected_components.json": components,
        "outer_folds.json": outer_folds, "outer_fold_statistics.json": fold_statistics,
        "nested_oof_audit.json": audit_folds,
        "oof_metrics.json": {
            "evaluation_mode": EVALUATION_MODE, "reportable_as_generalization": False,
            "custom_metric_is_not_official_sentence_selection_f1": True,
            "gate_metric": "official_compatible_pooled_sentence_selection.f1",
            "selector_metrics": metrics,
            "independent_official_style_metrics_reported_separately": official_metrics,
            "official_evaluator_invoked": False,
            "infeasible_fallbacks": gate["infeasible_outer_folds_by_strategy"],
        },
        "cluster_bootstrap.json": bootstrap, "acceptance_gate.json": gate,
    }
    if final_model is not None:
        json_outputs["selector_model.json"] = final_model
    for name, value in json_outputs.items():
        base._write_exclusive(destination / name, base._json_bytes(value))
    for name, rows in (("frozen_features.jsonl", frozen), ("training_targets.jsonl", targets), ("nested_oof_cases.jsonl", cases)):
        base._write_exclusive(destination / name, base._jsonl_bytes(rows))
    _verify_unchanged(**verification)
    manifest = {
        "schema_version": SCHEMA_VERSION, "created_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_status": {
            "mode": EVALUATION_MODE, "source_split": "train", "dev_read_or_scored": False,
            "reportable_as_generalization": False, "previous_train_results_observed": True,
            "acceptance": gate["status"], "full_train_model_emitted": final_model is not None,
        }, **source_bindings,
        "preregistration": {"path": str(preregistration_path), "sha256": preregistration_sha256, "protocol_sha256": protocol_sha256},
        "inference": {"inference_pass_count": 0, "new_nli_model_calls": 0, "reused_pair_count": len(frozen), "source_manifest_sha256": CACHE_MANIFEST_SHA256},
        "counts": {"claims": len(claims), "relation_claims": base.RELATION_CLAIM_COUNT,
                   "nei_claims": base.NEI_CLAIM_COUNT, "retrievable_relation_claims": len(retrievable_ids),
                   "candidate_sentence_pairs": len(frozen), "connected_components": len(components)},
        "validation": {**isolation, **candidate_validation, "feature_whitelist_violations": violations,
                       "all_bound_sources_unchanged_during_run": True, "cache_fully_verified": True,
                       "raw_feature_reconstruction_exact": True, "train_targets_and_folds_reconstructed_exactly": True},
        "environment": environment,
        "elapsed_seconds": time.perf_counter() - started,
        "outputs_sha256": {path.name: base._sha256(path) for path in sorted(destination.iterdir()) if path.is_file()},
    }
    base._write_exclusive(destination / "run_manifest.json", base._json_bytes(manifest))
    print("Completed train-only calibration: {}".format(gate["status"]), flush=True)
    return destination


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    raw = WORKSPACE_ROOT / "env" / "datasets" / "scifact" / "raw" / "data"
    retrieval = PROJECT_ROOT / "eval_runs" / "scifact_official_tfidf_train"
    parser.add_argument("--corpus", type=Path, default=raw / "corpus.jsonl")
    parser.add_argument("--claims-train", type=Path, default=raw / "claims_train.jsonl")
    parser.add_argument("--retrieval-train", type=Path, default=retrieval / "abstract_retrieval.jsonl")
    parser.add_argument("--retrieval-manifest", type=Path, default=retrieval / "run_manifest.json")
    parser.add_argument("--model", type=Path, default=WORKSPACE_ROOT / "env" / "models" / "MoritzLaurer--DeBERTa-v3-base-mnli-fever-anli" / base.MODEL_REVISION)
    parser.add_argument("--feature-cache", type=Path, default=PROJECT_ROOT / "eval_runs" / "scifact_sentence_selector_train_oof_v3")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main() -> None:
    import faulthandler

    faulthandler.enable()
    if sys.platform == "win32":
        import ctypes

        # Process-local reporting policy: keep native failures in stderr and the
        # event log instead of waiting for a GUI error dialog in a batch run.
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.SetErrorMode.argtypes = [ctypes.c_uint]
        kernel32.SetErrorMode.restype = ctypes.c_uint
        kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)
    args = _parse_args()
    destination = run(
        corpus_path=args.corpus, claims_path=args.claims_train,
        retrieval_path=args.retrieval_train, retrieval_manifest_path=args.retrieval_manifest,
        model_dir=args.model, cache_dir=args.feature_cache, output=args.out,
    )
    print(destination, flush=True)
    gate = json.loads((destination / "acceptance_gate.json").read_text(encoding="utf-8"))
    if gate["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
