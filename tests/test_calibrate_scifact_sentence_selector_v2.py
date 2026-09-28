from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = (
    Path(__file__).parents[1] / "scripts" / "calibrate_scifact_sentence_selector_v2.py"
)
ORIGINAL_SHA256 = "9d11dcc3575af86c33055b941cfaefb29b2d5ca287e09a15662a725a6ffe848a"


@pytest.fixture()
def selector():
    # Each test owns its base module instance; patched constants never alter
    # the original selector or another test's imported module.
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("scifact_selector_v2_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _selection_fixture():
    rows = [
        (1, 10, 1, 0, 0.85),
        (1, 10, 1, 1, 0.60),
        (1, 10, 1, 2, 0.45),
        (1, 20, 2, 0, 0.70),
        (1, 20, 2, 1, 0.55),
        (1, 30, 3, 0, 0.40),
        (2, 40, 1, 0, 0.75),
        (2, 40, 1, 1, 0.65),
    ]
    frozen = [
        {
            "pair_index": index,
            "claim_id": claim_id,
            "doc_id": doc_id,
            "doc_rank": rank,
            "sentence_index": sentence,
        }
        for index, (claim_id, doc_id, rank, sentence, _) in enumerate(rows)
    ]
    by_claim = {1: list(range(6)), 2: [6, 7]}
    scores = {index: row[-1] for index, row in enumerate(rows)}
    return frozen, by_claim, scores


def test_equal_thresholds_reproduce_original_selection_including_ties(selector):
    frozen, by_claim, scores = _selection_fixture()
    # Ties span documents and sentence positions, and equality must be included.
    scores.update({0: 0.60, 1: 0.60, 3: 0.60, 4: 0.60})
    for threshold in (0.0, 0.60, 0.75, 1.0):
        for maximum in (1, 2, 3, 4):
            observed = selector._select_pairs(
                [1, 2],
                by_claim,
                scores,
                frozen,
                accept_threshold=threshold,
                expansion_threshold=threshold,
                max_sentences=maximum,
            )
            expected = selector.base._select_pairs(
                [1, 2],
                by_claim,
                scores,
                frozen,
                threshold=threshold,
                max_sentences=maximum,
            )
            assert observed == expected
    assert selector._select_pairs(
        [1],
        by_claim,
        scores,
        frozen,
        accept_threshold=0.60,
        expansion_threshold=0.60,
        max_sentences=4,
    )[1] == [0, 1, 3, 4]


def test_document_admission_prevents_other_document_expansion(selector):
    frozen, by_claim, scores = _selection_fixture()
    result = selector._select_pairs(
        [1, 2],
        by_claim,
        scores,
        frozen,
        accept_threshold=0.80,
        expansion_threshold=0.40,
        max_sentences=4,
    )
    assert result == {1: [0, 1, 2], 2: []}
    assert all(frozen[index]["doc_id"] == 10 for index in result[1])


def test_expansion_changes_evidence_without_changing_empty_claims(selector):
    frozen, by_claim, scores = _selection_fixture()
    results = [
        selector._select_pairs(
            [1, 2],
            by_claim,
            scores,
            frozen,
            accept_threshold=0.80,
            expansion_threshold=threshold,
            max_sentences=4,
        )
        for threshold in (0.80, 0.60, 0.40, 0.0)
    ]
    assert [len(result[1]) for result in results] == [1, 2, 3, 3]
    assert all(
        {claim: bool(rows) for claim, rows in result.items()} == {1: True, 2: False}
        for result in results
    )


def test_every_selection_is_legal_unique_and_globally_capped(selector):
    frozen, by_claim, scores = _selection_fixture()
    for acceptance in (0.0, 0.50, 0.80, 1.0):
        for expansion in (0.0, acceptance / 2, acceptance):
            result = selector._select_pairs(
                [1, 2],
                by_claim,
                scores,
                frozen,
                accept_threshold=acceptance,
                expansion_threshold=expansion,
                max_sentences=4,
            )
            for claim_id, selected in result.items():
                assert len(selected) <= 4
                assert len(set(selected)) == len(selected)
                assert set(selected) <= set(by_claim[claim_id])
                for index in selected:
                    doc_id = frozen[index]["doc_id"]
                    assert any(
                        scores[candidate] >= acceptance
                        for candidate in by_claim[claim_id]
                        if frozen[candidate]["doc_id"] == doc_id
                    )
                    assert scores[index] >= expansion


@pytest.mark.parametrize(
    "acceptance,expansion,maximum",
    [
        (-0.1, 0.0, 4),
        (1.1, 0.0, 4),
        (0.5, -0.1, 4),
        (0.5, 0.6, 4),
        (math.nan, 0.0, 4),
        (0.5, math.inf, 4),
        (0.5, 0.2, 0),
        (0.5, 0.2, 5),
    ],
)
def test_invalid_selection_configuration_fails_closed(
    selector, acceptance, expansion, maximum
):
    frozen, by_claim, scores = _selection_fixture()
    with pytest.raises(selector.SelectorCalibrationError):
        selector._select_pairs(
            [1, 2],
            by_claim,
            scores,
            frozen,
            accept_threshold=acceptance,
            expansion_threshold=expansion,
            max_sentences=maximum,
        )


@pytest.mark.parametrize("bad_score", [math.nan, math.inf, -math.inf, -0.01, 1.01])
def test_nonfinite_or_out_of_range_scores_cannot_enter_selection(selector, bad_score):
    frozen, by_claim, scores = _selection_fixture()
    scores[2] = bad_score
    with pytest.raises(selector.SelectorCalibrationError):
        selector._select_pairs(
            [1, 2],
            by_claim,
            scores,
            frozen,
            accept_threshold=0.80,
            expansion_threshold=0.40,
            max_sentences=4,
        )


def _metric(
    official=0.3, custom=0.2, *, all_recall=0.50, reachable_recall=0.65, nei=0.70
):
    return {
        "official_compatible_pooled_sentence_selection": {"f1": official},
        "custom_best_rationale_macro_f1": {"value": custom},
        "all_relation_complete_rationale_recall": all_recall,
        "retrievable_relation_complete_rationale_recall": reachable_recall,
        "nei_correct_empty_rate": nei,
    }


def test_three_constraints_are_inclusive_and_independently_required(selector):
    assert all(selector._constraint_checks(_metric()).values())
    for parameter, value in (
        ("all_recall", 0.499),
        ("reachable_recall", 0.649),
        ("nei", 0.699),
    ):
        checks = selector._constraint_checks(_metric(**{parameter: value}))
        assert sum(not passed for passed in checks.values()) == 1


def _controlled_sweep(selector, monkeypatch, outcome):
    # Isolate policy choice from the separately tested sentence selection.
    configurations = []

    def selection(*args, **kwargs):
        configurations.append(kwargs)
        return {1: [len(configurations) - 1]}

    def metrics(claim_ids, claims_by_id, frozen, selected, reachable):
        return outcome(configurations[selected[1][0]])

    monkeypatch.setattr(selector, "_select_pairs", selection)
    monkeypatch.setattr(selector.base, "_selector_metrics", metrics)
    return selector._sweep_selection(
        claim_ids=[1],
        by_claim={1: [0]},
        scores={0: 0.8},
        frozen=[{}],
        claims_by_id={1: {}},
        retrievable_relation_ids={1},
    )


def test_sweep_prioritizes_feasibility_then_official_f1_not_custom_f1(
    selector, monkeypatch
):
    def outcome(config):
        if config["accept_threshold"] == 0.50 and config["expansion_threshold"] == 0.20:
            return _metric(official=0.30, custom=0.10)
        if config["accept_threshold"] == 0.40:
            return _metric(official=0.20, custom=0.99)
        return _metric(official=0.95, custom=1.0, reachable_recall=0.64)

    result = _controlled_sweep(selector, monkeypatch, outcome)
    assert result["status"] == "feasible"
    chosen = result["selected"]
    assert chosen["accept_threshold"] == 0.50
    assert chosen["expansion_threshold"] == 0.20
    assert chosen["max_sentences"] == 1
    assert (
        chosen["metrics"]["official_compatible_pooled_sentence_selection"]["f1"] == 0.30
    )
    assert len(result["sweep"]) == 264


def test_infeasible_sweep_keeps_real_unconstrained_best_and_no_selected_config(
    selector, monkeypatch
):
    def outcome(config):
        return _metric(
            official=0.8 if config["accept_threshold"] == 0.60 else 0.1,
            all_recall=0.49,
        )

    result = _controlled_sweep(selector, monkeypatch, outcome)
    assert result["status"] == "infeasible"
    assert result["selected"] is None
    diagnostic = result["diagnostic_unconstrained_best"]
    assert diagnostic["accept_threshold"] == 0.60
    assert (
        diagnostic["metrics"]["official_compatible_pooled_sentence_selection"]["f1"]
        == 0.8
    )
    assert diagnostic in result["sweep"]
    assert sum(result["failed_constraint_counts"].values()) >= 264


def _outer_choices():
    return [
        {
            "strategies": {
                "hybrid": {
                    "inner_tuning": {
                        "status": "feasible",
                        "selected": {
                            "accept_threshold": acceptance,
                            "expansion_threshold": expansion,
                            "max_sentences": maximum,
                            "metrics": _metric(),
                        },
                    },
                    "diagnostic_fallback_used": False,
                }
            },
            "outer_fold": index,
        }
        for index, (acceptance, expansion, maximum) in enumerate(
            (
                (0.40, 0.10, 1),
                (0.60, 0.20, 2),
                (0.80, 0.40, 2),
                (0.90, 0.50, 1),
            )
        )
    ]


def test_full_refit_aggregates_two_medians_and_smallest_tied_mode(
    selector, monkeypatch
):
    fit_calls = []

    def fit(*args, **kwargs):
        fit_calls.append(kwargs)
        return None, None, {"fitted": True}, {"fit_claim_ids": [1]}

    monkeypatch.setattr(selector.base, "_fit_hybrid", fit)
    result = selector._full_train_model(
        claims=[{"id": 1}],
        frozen=[{"claim_id": 1}],
        targets=[{}],
        by_claim={1: [0]},
        audit_folds=_outer_choices(),
    )
    selection = result["selection"]
    assert selection["accept_threshold"] == pytest.approx(0.70)
    assert selection["expansion_threshold"] == pytest.approx(0.30)
    assert selection["max_sentences"] == 1
    assert selection["expansion_threshold"] <= selection["accept_threshold"]
    assert selection["post_gate_full_train_retuning"] is False
    assert fit_calls == [{"include_vectorizer_state": True}]


def test_any_infeasible_inner_fold_blocks_full_refit(selector, monkeypatch):
    audits = _outer_choices()
    audits[2]["strategies"]["hybrid"]["inner_tuning"].update(
        status="infeasible", selected=None
    )
    monkeypatch.setattr(
        selector.base,
        "_fit_hybrid",
        lambda *args, **kwargs: pytest.fail(
            "infeasible tuning must prevent model fitting"
        ),
    )
    with pytest.raises(selector.SelectorCalibrationError, match=r"feasible|infeasible"):
        selector._full_train_model(
            claims=[],
            frozen=[{"claim_id": 1}],
            targets=[{}],
            by_claim={},
            audit_folds=audits,
        )


def test_acceptance_gate_rejects_even_strong_metrics_when_inner_tuning_failed(selector):
    metrics = {}
    audits = _outer_choices()
    for strategy, score in (
        ("hybrid", 0.70),
        ("lexical_only", 0.40),
        ("nli_only", 0.50),
    ):
        metrics[strategy] = {
            **_metric(official=score, custom=score),
            "relation_claim_count": 505,
            "retrievable_relation_claim_count": 396,
            "nei_claim_count": 304,
        }
        for audit in audits:
            row = audit["strategies"].setdefault(strategy, {})
            row["outer_validation_metrics"] = metrics[strategy]
            row.setdefault("inner_tuning", {"status": "feasible"})
    cases = [
        {"strategy": strategy, "claim_id": claim_id, "is_relation": claim_id < 505}
        for strategy in selector.base.STRATEGIES
        for claim_id in range(809)
    ]
    bootstrap = {
        "samples": 10_000,
        "component_count": 228,
        "metrics": {
            name: {
                comparison: {"confidence_interval": {"lower": 0.10}}
                for comparison in ("hybrid_vs_lexical_only", "hybrid_vs_nli_only")
            }
            for name in (
                "custom_best_rationale_macro",
                "official_compatible_pooled_sentence_selection",
            )
        },
    }
    arguments = {
        "metrics": metrics,
        "audit_folds": audits,
        "cases": cases,
        "bootstrap": bootstrap,
        "isolation": {
            "claim_coverage_complete": True,
            "cross_fold_claim_overlap_count": 0,
            "cross_fold_doc_overlap_count": 0,
        },
        "feature_whitelist_violations": 0,
    }
    assert selector._gate(**arguments)["status"] == "pass"
    audits[1]["strategies"]["hybrid"]["inner_tuning"].update(
        status="infeasible", selected=None
    )
    gate = selector._gate(**arguments)
    assert gate["status"] == "reject"
    assert gate["full_train_model_permitted"] is False


def _synthetic_data():
    corpus, claims, retrieval = [], [], []
    for component in range(4):
        doc_ids = [100 + component, 200 + component, 300 + component]
        for offset, doc_id in enumerate(doc_ids):
            sentence = (
                f"marker{component} evidence"
                if offset == 0
                else f"unrelated material {doc_id}"
            )
            corpus.append(
                {"doc_id": doc_id, "title": str(doc_id), "abstract": [sentence]}
            )
        relation_id, nei_id = component * 2 + 1, component * 2 + 2
        claims.extend(
            [
                {
                    "id": relation_id,
                    "claim": f"marker{component} evidence",
                    "evidence": {
                        str(doc_ids[0]): [{"label": "SUPPORT", "sentences": [0]}]
                    },
                    "cited_doc_ids": [doc_ids[0]],
                },
                {
                    "id": nei_id,
                    "claim": f"absent{component} finding",
                    "evidence": {},
                    "cited_doc_ids": [],
                },
            ]
        )
        retrieval.extend(
            {"claim_id": value, "doc_ids": doc_ids} for value in (relation_id, nei_id)
        )
    return corpus, claims, retrieval


def test_original_implementation_is_unchanged_and_loaded_independently(selector):
    assert (
        selector.base._sha256(
            SCRIPT.with_name("calibrate_scifact_sentence_selector.py")
        )
        == ORIGINAL_SHA256
    )
    assert selector.base._select_pairs.__module__ != selector._select_pairs.__module__
    assert "threshold" in selector.base._select_pairs.__annotations__


def _synthetic_run_inputs(selector, tmp_path, monkeypatch):
    corpus, claims, retrieval = _synthetic_data()
    for name, value in (
        ("SOURCE_CORPUS_COUNT", 12),
        ("SOURCE_CLAIM_COUNT", 8),
        ("RELATION_CLAIM_COUNT", 4),
        ("NEI_CLAIM_COUNT", 4),
        ("RETRIEVABLE_RELATION_CLAIM_COUNT", 4),
        ("CANDIDATE_PAIR_COUNT", 24),
        ("POSITIVE_PAIR_COUNT", 4),
        ("CONNECTED_COMPONENT_COUNT", 4),
        ("LARGEST_COMPONENT_CLAIM_COUNT", 2),
    ):
        monkeypatch.setattr(selector.base, name, value)
    paths = {}
    for key, name, rows in (
        ("corpus", "corpus.jsonl", corpus),
        ("claims_train", "claims_train.jsonl", claims),
        ("tfidf_top3_train", "retrieval_train.jsonl", retrieval),
        ("tfidf_manifest", "retrieval_manifest.json", [{}]),
    ):
        path = tmp_path / name
        path.write_bytes(selector.base._jsonl_bytes(rows))
        paths[key] = path
    input_hashes = {key: selector.base._sha256(path) for key, path in paths.items()}
    candidates, _, retrieval_by_claim = selector.base._build_candidates(
        corpus, claims, retrieval
    )
    # Uniform NLI probabilities deliberately make the NLI-only constraints
    # impossible: it cannot admit relation claims while rejecting NEI claims.
    # Hybrid and lexical methods can still discriminate the synthetic text.
    frozen = selector.base._freeze_features(
        candidates, [[0.96, 0.02, 0.02] for _ in candidates]
    )
    targets = selector.base._build_targets(claims, frozen)
    components = selector.base._connected_components(
        [row["id"] for row in claims],
        retrieval_by_claim,
        {row["id"]: row["claim"] for row in claims},
    )
    folds = selector.base._assign_component_folds(components, 4)
    cache_dir = tmp_path / "prior-train-cache"
    cache_dir.mkdir()
    for name, value in (
        ("connected_components.json", components),
        ("outer_folds.json", folds),
        ("preregistration.json", {"synthetic": True}),
    ):
        (cache_dir / name).write_bytes(selector.base._json_bytes(value))
    for name, rows in (
        ("frozen_features.jsonl", frozen),
        ("training_targets.jsonl", targets),
    ):
        (cache_dir / name).write_bytes(selector.base._jsonl_bytes(rows))
    cache_outputs = {
        path.name: selector.base._sha256(path) for path in cache_dir.iterdir()
    }
    model_identity = {
        "revision": "synthetic-model",
        "identity_sha256": "synthetic-pinned-identity",
        "files": {},
        "id2label": {"0": "entailment", "1": "neutral", "2": "contradiction"},
    }
    manifest = {
        "schema_version": selector.base.SCHEMA_VERSION,
        "implementation": {"sha256": ORIGINAL_SHA256},
        "evaluation_status": {
            "source_split": "train",
            "dev_read_or_scored": False,
            "reportable_as_generalization": False,
        },
        "inference": {"inference_pass_count": 1, "pair_count": 24},
        "inputs": {key: {"sha256": digest} for key, digest in input_hashes.items()},
        "model": model_identity,
        "outputs_sha256": cache_outputs,
    }
    manifest_path = cache_dir / "run_manifest.json"
    manifest_path.write_bytes(selector.base._json_bytes(manifest))
    monkeypatch.setattr(
        selector, "CACHE_MANIFEST_SHA256", selector.base._sha256(manifest_path)
    )
    monkeypatch.setattr(
        selector, "CACHE_FEATURES_SHA256", cache_outputs["frozen_features.jsonl"]
    )
    monkeypatch.setattr(selector, "_validate_runtime_environment", lambda: {})
    monkeypatch.setattr(selector.base, "_model_identity", lambda path: model_identity)
    monkeypatch.setattr(
        selector.base, "_validate_inputs", lambda **kwargs: (corpus, claims, retrieval)
    )
    monkeypatch.setattr(
        selector.base,
        "_infer_probabilities",
        lambda *args, **kwargs: pytest.fail("new NLI inference is forbidden"),
    )
    arguments = {
        "corpus_path": paths["corpus"],
        "claims_path": paths["claims_train"],
        "retrieval_path": paths["tfidf_top3_train"],
        "retrieval_manifest_path": paths["tfidf_manifest"],
        "model_dir": tmp_path / "synthetic-model",
        "cache_dir": cache_dir,
        "output": tmp_path / "new-train-result",
    }
    return arguments, input_hashes, model_identity


def test_rejected_synthetic_run_uses_real_diagnostic_baseline_and_hashes_all_outputs(
    selector, tmp_path, monkeypatch
):
    arguments, input_hashes, _ = _synthetic_run_inputs(selector, tmp_path, monkeypatch)
    output = arguments["output"]
    actual_oof = selector._run_nested_oof

    def checked_oof(**kwargs):
        prereg = json.loads(
            (output / "preregistration.json").read_text(encoding="utf-8")
        )
        assert prereg["written_before_oof"] is True
        assert prereg["new_nli_inference_performed"] is False
        assert prereg["previous_train_results_observed"] is True
        assert prereg["implementation"]["sha256"] == selector.base._sha256(SCRIPT)
        assert prereg["readonly_base_implementation"]["sha256"] == ORIGINAL_SHA256
        assert {
            key: value["sha256"] for key, value in prereg["inputs"].items()
        } == input_hashes
        assert (
            prereg["feature_cache"]["manifest_sha256"] == selector.CACHE_MANIFEST_SHA256
        )
        assert (
            prereg["feature_cache"]["all_source_files_sha256"]["frozen_features.jsonl"]
            == selector.CACHE_FEATURES_SHA256
        )
        return actual_oof(**kwargs)

    monkeypatch.setattr(selector, "_run_nested_oof", checked_oof)
    monkeypatch.setattr(
        selector,
        "_full_train_model",
        lambda **kwargs: pytest.fail("rejected calibration must not export a model"),
    )
    selector.run(**arguments)
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    gate = json.loads((output / "acceptance_gate.json").read_text(encoding="utf-8"))
    prereg = json.loads((output / "preregistration.json").read_text(encoding="utf-8"))
    configuration = json.loads(
        (output / "configuration.json").read_text(encoding="utf-8")
    )
    audits = json.loads((output / "nested_oof_audit.json").read_text(encoding="utf-8"))
    cases = selector.base._read_jsonl(output / "nested_oof_cases.jsonl")
    metrics = json.loads((output / "oof_metrics.json").read_text(encoding="utf-8"))[
        "selector_metrics"
    ]
    assert gate["status"] == "reject"
    assert not (output / "selector_model.json").exists()
    assert manifest["evaluation_status"]["dev_read_or_scored"] is False
    assert manifest["evaluation_status"]["reportable_as_generalization"] is False
    assert manifest["inference"]["new_nli_model_calls"] == 0
    assert manifest["inference"]["reused_pair_count"] == 24
    assert configuration == prereg["protocol"]
    assert manifest["preregistration"]["sha256"] == selector.base._sha256(
        output / "preregistration.json"
    )
    assert manifest["preregistration"][
        "protocol_sha256"
    ] == selector.base._sha256_bytes(selector.base._canonical_bytes(configuration))
    assert manifest["validation"]["all_bound_sources_unchanged_during_run"] is True
    assert manifest["validation"]["raw_feature_reconstruction_exact"] is True
    assert set(manifest["outputs_sha256"]) == {
        path.name for path in output.iterdir() if path.name != "run_manifest.json"
    }
    assert all(
        selector.base._sha256(output / name) == digest
        for name, digest in manifest["outputs_sha256"].items()
    )
    assert len(cases) == 24
    for strategy in selector.STRATEGIES:
        strategy_cases = [row for row in cases if row["strategy"] == strategy]
        assert {row["claim_id"] for row in strategy_cases} == set(range(1, 9))
        assert len(strategy_cases) == 8
    baseline_cases = [row for row in cases if row["strategy"] == "nli_only"]
    assert all(row["diagnostic_fallback_used"] for row in baseline_cases)
    assert all(row["inner_selection_status"] == "infeasible" for row in baseline_cases)
    assert all(
        row["prediction_configuration_source"] == "diagnostic_unconstrained_best"
        for row in baseline_cases
    )
    assert all(row["selected_pair_indices"] for row in baseline_cases)
    assert (
        metrics["nli_only"]["official_compatible_pooled_sentence_selection"]["f1"] > 0
    )
    for audit in audits:
        assert audit["inner_isolation"]["cross_fold_doc_overlap_count"] == 0
        tuning = audit["strategies"]["nli_only"]["inner_tuning"]
        assert tuning["selected"] is None
        assert tuning["diagnostic_unconstrained_best"] in tuning["sweep"]
        assert audit["strategies"]["nli_only"]["selection_parameters"] == {
            key: tuning["diagnostic_unconstrained_best"][key]
            for key in ("accept_threshold", "expansion_threshold", "max_sentences")
        }


def test_cache_rejects_changed_declared_file_even_if_feature_file_is_unchanged(
    selector, tmp_path, monkeypatch
):
    arguments, input_hashes, model_identity = _synthetic_run_inputs(
        selector, tmp_path, monkeypatch
    )
    selector._validate_cache(arguments["cache_dir"], input_hashes, model_identity)
    (arguments["cache_dir"] / "training_targets.jsonl").write_text(
        "{}\n", encoding="utf-8"
    )
    with pytest.raises(selector.SelectorCalibrationError, match="SHA256 mismatch"):
        selector._validate_cache(arguments["cache_dir"], input_hashes, model_identity)


def test_cache_binds_current_train_inputs_and_model_identity(
    selector, tmp_path, monkeypatch
):
    arguments, input_hashes, model_identity = _synthetic_run_inputs(
        selector, tmp_path, monkeypatch
    )
    with pytest.raises(selector.SelectorCalibrationError, match="input bindings"):
        selector._validate_cache(
            arguments["cache_dir"],
            {**input_hashes, "claims_train": "changed"},
            model_identity,
        )
    with pytest.raises(selector.SelectorCalibrationError, match="model binding"):
        selector._validate_cache(
            arguments["cache_dir"],
            input_hashes,
            {**model_identity, "identity_sha256": "changed"},
        )


def test_train_contract_preserves_exclusive_output_and_rejects_other_split(
    selector, tmp_path, monkeypatch
):
    args = selector._parse_args(["--out", str(tmp_path / "result")])
    assert not hasattr(args, "claims_dev")
    assert not hasattr(args, "split")
    assert not hasattr(args, "device")
    assert not hasattr(args, "bootstrap_samples")
    assert args.feature_cache.name == "scifact_sentence_selector_train_oof_v3"
    arguments, _, _ = _synthetic_run_inputs(selector, tmp_path, monkeypatch)
    arguments["output"].mkdir()
    with pytest.raises(selector.SelectorCalibrationError, match="already exists"):
        selector.run(**arguments)
    arguments["output"] = tmp_path / "forbidden-dev-result"
    with pytest.raises(selector.SelectorCalibrationError, match="forbidden dev"):
        selector.run(**arguments)


@pytest.mark.parametrize("python_version", [(3, 10), (3, 9)])
def test_incompatible_runtime_is_rejected_before_inputs_or_fitting(
    selector, tmp_path, monkeypatch, python_version
):
    monkeypatch.setattr(
        selector,
        "sys",
        SimpleNamespace(version_info=python_version, version="synthetic"),
    )
    monkeypatch.setattr(
        selector,
        "importlib",
        SimpleNamespace(metadata=SimpleNamespace(version=lambda package: "unapproved")),
    )
    monkeypatch.setattr(
        selector.base,
        "_validate_inputs",
        lambda **kwargs: pytest.fail("runtime rejection must precede any dataset read"),
    )
    with pytest.raises(selector.SelectorCalibrationError, match="pinned"):
        selector.run(
            corpus_path=tmp_path / "missing-corpus",
            claims_path=tmp_path / "missing-claims",
            retrieval_path=tmp_path / "missing-retrieval",
            retrieval_manifest_path=tmp_path / "missing-manifest",
            model_dir=tmp_path / "missing-model",
            cache_dir=tmp_path / "missing-cache",
            output=tmp_path / "result",
        )
