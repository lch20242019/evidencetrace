from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from evidencetrace.eval.scifact_official import (
    compute_official_pipeline_metrics,
    validate_official_data,
)

SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "calibrate_scifact_sentence_selector.py"
)


def _load_module():
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("scifact_sentence_selector", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def selector():
    return _load_module()


def _synthetic_data():
    corpus = []
    claims = []
    retrieval = []
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
        relation_id = component * 2 + 1
        nei_id = component * 2 + 2
        claims.extend(
            [
                {
                    "id": relation_id,
                    "claim": f"marker{component} evidence",
                    "evidence": {
                        str(doc_ids[0]): [
                            {"label": "SUPPORT", "sentences": [0]}
                        ]
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
            [
                {"claim_id": relation_id, "doc_ids": doc_ids},
                {"claim_id": nei_id, "doc_ids": doc_ids},
            ]
        )
    return corpus, claims, retrieval


def _frozen(selector, corpus, claims, retrieval):
    candidates, by_claim, retrieval_by_claim = selector._build_candidates(
        corpus, claims, retrieval
    )
    probabilities = []
    for row in candidates:
        marker = row["claim"].split()[0]
        if marker in row["sentence"]:
            probabilities.append([0.96, 0.02, 0.02])
        else:
            probabilities.append([0.05, 0.90, 0.05])
    frozen = selector._freeze_features(candidates, probabilities)
    targets = selector._build_targets(claims, frozen)
    return frozen, targets, by_claim, retrieval_by_claim


def test_contract_has_no_dev_or_split_argument(selector):
    args = selector._parse_args(
        [
            "--corpus",
            "corpus.jsonl",
            "--claims-train",
            "claims_train.jsonl",
            "--retrieval-train",
            "retrieval_train.jsonl",
            "--retrieval-manifest",
            "manifest.json",
            "--model",
            selector.MODEL_REVISION,
            "--out",
            "train-output",
        ]
    )
    assert selector.SOURCE_CLAIM_COUNT == 809
    assert selector.RELATION_CLAIM_COUNT == 505
    assert selector.NEI_CLAIM_COUNT == 304
    assert selector.RETRIEVABLE_RELATION_CLAIM_COUNT == 396
    assert selector.OUTER_FOLDS == 4
    assert selector.INNER_FOLDS == 3
    assert selector.MAX_SENTENCES == (1, 2, 3, 4)
    assert selector.FEATURE_NAMES[:2] == (
        "word_tfidf_cosine",
        "char_tfidf_cosine",
    )
    assert not hasattr(args, "split")
    assert not hasattr(args, "claims_dev")
    assert not hasattr(args, "bootstrap_samples")
    assert selector.EVALUATION_MODE.endswith("not_generalization")
    assert selector.OFFICIAL_TFIDF_TRAIN_RETRIEVAL_SHA256 == (
        "ce1b6a9bb8e2b02214dc1fa1ae62ea62a2a13660192e67bbbf609e0509edf7a1"
    )
    assert selector.OFFICIAL_TFIDF_TRAIN_MANIFEST_SHA256 == (
        "76d1707c8b10d1c705579cdad36b090929d26ef0c1dec6ba556fffc48137f2a2"
    )
    assert selector.PINNED_MODEL_IDENTITY_SHA256 == (
        "9194298eb4045bb0a48daa2c01ecadfa981fe0300fa87c5975c18f5ec972a5e2"
    )


def test_dev_named_input_and_output_are_rejected(selector):
    with pytest.raises(selector.SelectorCalibrationError, match="forbidden dev"):
        selector._assert_not_dev(Path("claims_dev.jsonl"), "claims")
    with pytest.raises(selector.SelectorCalibrationError, match="forbidden dev"):
        selector._assert_not_dev(Path("selector-dev-run"), "output")
    selector._assert_not_dev(Path("claims_train.jsonl"), "claims")


def test_model_identity_rejects_same_revision_name_with_changed_files(
    selector, tmp_path: Path
):
    model = tmp_path / selector.MODEL_REVISION
    model.mkdir()
    config = {"id2label": {"0": "entailment", "1": "neutral", "2": "contradiction"}}
    for name, content in {
        "config.json": json.dumps(config).encode(),
        "model.safetensors": b"changed weights",
        "spm.model": b"spm",
        "tokenizer.json": b"{}",
        "tokenizer_config.json": b"{}",
    }.items():
        (model / name).write_bytes(content)
    with pytest.raises(selector.SelectorCalibrationError, match="identity mismatch"):
        selector._model_identity(model)


def test_retrieval_manifest_requires_pinned_source_and_configuration(
    selector, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    corpus_path = tmp_path / "corpus.jsonl"
    claims_path = tmp_path / "claims_train.jsonl"
    retrieval_path = tmp_path / "retrieval_train.jsonl"
    manifest_path = tmp_path / "manifest.json"
    corpus_rows = [
        {"doc_id": value, "title": str(value), "abstract": ["sentence"]}
        for value in (1, 2, 3)
    ]
    claim_rows = [
        {
            "id": 10,
            "claim": "claim",
            "evidence": {"1": [{"label": "SUPPORT", "sentences": [0]}]},
        }
    ]
    retrieval_rows = [{"claim_id": 10, "doc_ids": [1, 2, 3]}]

    def write_jsonl(path, rows):
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )

    write_jsonl(corpus_path, corpus_rows)
    write_jsonl(claims_path, claim_rows)
    write_jsonl(retrieval_path, retrieval_rows)
    monkeypatch.setattr(selector, "SOURCE_CORPUS_COUNT", 3)
    monkeypatch.setattr(selector, "SOURCE_CLAIM_COUNT", 1)
    monkeypatch.setattr(selector, "RELATION_CLAIM_COUNT", 1)
    monkeypatch.setattr(selector, "NEI_CLAIM_COUNT", 0)
    monkeypatch.setattr(
        selector, "OFFICIAL_CORPUS_SHA256", selector._sha256(corpus_path)
    )
    monkeypatch.setattr(
        selector, "OFFICIAL_TRAIN_CLAIMS_SHA256", selector._sha256(claims_path)
    )
    monkeypatch.setattr(
        selector,
        "OFFICIAL_TFIDF_TRAIN_RETRIEVAL_SHA256",
        selector._sha256(retrieval_path),
    )
    manifest = {
        "schema_version": "scifact-official-tfidf-run-v1",
        "official_source": {
            "commit": selector.OFFICIAL_SCIFACT_COMMIT,
            "archive_sha256": selector.OFFICIAL_SCIFACT_ARCHIVE_SHA256,
        },
        "data": {
            "split": "train",
            "claim_count": 1,
            "files_sha256": {
                "corpus.jsonl": selector._sha256(corpus_path),
                "claims_train.jsonl": selector._sha256(claims_path),
            },
        },
        "configuration": {
            "top_k": 3,
            "min_gram": 2,
            "max_gram": 2,
            "stop_words": "english",
            "document_text": "title + ' '.join(abstract)",
        },
        "outputs_sha256": {
            "retrieval_train.jsonl": selector._sha256(retrieval_path)
        },
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(
        selector,
        "OFFICIAL_TFIDF_TRAIN_MANIFEST_SHA256",
        selector._sha256(manifest_path),
    )
    with pytest.raises(selector.SelectorCalibrationError, match="not bound"):
        selector._validate_inputs(
            corpus_path=corpus_path,
            claims_path=claims_path,
            retrieval_path=retrieval_path,
            retrieval_manifest_path=manifest_path,
        )


def test_features_are_whitelisted_and_targets_are_separate(selector):
    corpus, claims, retrieval = _synthetic_data()
    frozen, targets, _, _ = _frozen(selector, corpus, claims, retrieval)
    assert len(selector.FEATURE_NAMES) == 16
    assert all(
        tuple(row["static_features"]) == selector.STATIC_FEATURE_NAMES
        for row in frozen
    )
    assert all("is_gold_sentence" not in row for row in frozen)
    assert sum(row["is_gold_sentence"] for row in targets) == 4
    assert all(
        set(row) == {"pair_index", "claim_id", "is_gold_sentence"}
        for row in targets
    )


def test_numeric_and_negation_features_follow_fixed_rules(selector):
    features = selector._lexical_features(
        "No effect at 1.00 and 2e1 percent.",
        "The effect was not seen at 1 and 20 percent.",
    )
    assert features["numeric_token_set_match"] == 1.0
    assert features["negation_presence_match"] == 1.0
    assert selector._lexical_features("No value", "A value")[
        "negation_presence_match"
    ] == 0.0
    assert selector._lexical_features("plain text", "other words")[
        "numeric_token_set_match"
    ] == 1.0


def test_candidate_keys_are_unique_and_cover_every_claim(selector):
    corpus, claims, retrieval = _synthetic_data()
    candidates, by_claim, _ = selector._build_candidates(corpus, claims, retrieval)
    validation = selector._validate_candidate_keys(
        candidates, by_claim, {row["id"] for row in claims}
    )
    assert all(validation.values())
    candidates[-1]["pair_index"] = 0
    with pytest.raises(selector.SelectorCalibrationError, match="uniqueness/coverage"):
        selector._validate_candidate_keys(
            candidates, by_claim, {row["id"] for row in claims}
        )


def test_claim_sample_weight_is_one_with_fixed_class_allocation(selector):
    targets = [
        {"claim_id": 1, "is_gold_sentence": True},
        {"claim_id": 1, "is_gold_sentence": False},
        {"claim_id": 1, "is_gold_sentence": False},
        {"claim_id": 2, "is_gold_sentence": False},
        {"claim_id": 2, "is_gold_sentence": False},
    ]
    weights = selector._claim_sample_weights(targets, list(range(5)))
    assert weights == pytest.approx([0.5, 0.25, 0.25, 0.5, 0.5])
    assert sum(weights[:3]) == pytest.approx(1.0)
    assert sum(weights[3:]) == pytest.approx(1.0)


def test_shared_top3_documents_form_components_and_never_cross_folds(selector):
    corpus, claims, retrieval = _synthetic_data()
    _, _, _, retrieval_by_claim = _frozen(selector, corpus, claims, retrieval)
    claim_ids = [row["id"] for row in claims]
    claim_texts = {row["id"]: row["claim"] for row in claims}
    components = selector._connected_components(
        claim_ids, retrieval_by_claim, claim_texts
    )
    assert len(components) == 4
    assert sorted(row["claim_count"] for row in components) == [2, 2, 2, 2]
    folds = selector._assign_component_folds(components, 4)
    isolation = selector._validate_fold_isolation(folds, set(claim_ids))
    assert isolation == {
        "claim_coverage_count": 8,
        "claim_coverage_complete": True,
        "cross_fold_claim_overlap_count": 0,
        "cross_fold_doc_overlap_count": 0,
    }
    with pytest.raises(selector.SelectorCalibrationError, match="at least 5"):
        selector._assign_component_folds(components, 5)


def test_duplicate_normalized_claim_text_is_grouped_without_shared_doc(selector):
    components = selector._connected_components(
        [1, 2],
        {1: [10, 11, 12], 2: [20, 21, 22]},
        {1: " Same Claim ", 2: "same claim"},
    )
    assert len(components) == 1
    assert components[0]["claim_ids"] == [1, 2]


def test_fold_tfidf_fits_train_text_only_and_serializes_replay_state(selector):
    corpus, claims, retrieval = _synthetic_data()
    frozen, _, by_claim, _ = _frozen(selector, corpus, claims, retrieval)
    train_ids = [1, 2, 3, 4, 5, 6]
    train_pairs = [index for claim_id in train_ids for index in by_claim[claim_id]]
    held_out_pairs = [index for claim_id in [7, 8] for index in by_claim[claim_id]]
    transformer = selector._fit_fold_transformer(frozen, train_pairs)
    state = selector._serialize_fold_transformer(transformer)
    assert state["fit_claim_ids"] == train_ids
    assert "marker3" not in state["word"]["vocabulary"]
    assert len(state["word"]["state_sha256"]) == 64
    rows = selector._fold_feature_rows(frozen, held_out_pairs, transformer)
    assert all(tuple(row) == selector.FEATURE_NAMES for row in rows)


def test_best_rationale_metric_is_outer_or_and_inner_set_f1(selector):
    claim = {
        "id": 1,
        "claim": "x",
        "evidence": {
            "10": [
                {"label": "SUPPORT", "sentences": [0, 1]},
                {"label": "SUPPORT", "sentences": [2]},
            ]
        },
    }
    frozen = [
        {"doc_id": 10, "sentence_index": 0},
        {"doc_id": 10, "sentence_index": 2},
    ]
    metrics = selector._selector_metrics(
        [1],
        {1: claim},
        frozen,
        {1: [1]},
        {1},
    )
    assert metrics["custom_best_rationale_macro_f1"]["value"] == 1.0
    assert metrics["all_relation_complete_rationale_recall"] == 1.0
    assert "best_rationale" in metrics["custom_best_rationale_macro_f1"]["name"]
    assert "outer-OR" in metrics["custom_best_rationale_macro_f1"]["semantics"]
    assert metrics["official_compatible_pooled_sentence_selection"] == {
        "external_official_evaluator_invoked": False,
        "label_free": True,
        "sufficient_counts": {
            "relevant": 3,
            "retrieved": 1,
            "correct_selection": 1,
        },
        "precision": 1.0,
        "recall": pytest.approx(1 / 3),
        "f1": pytest.approx(0.5),
    }
    case = metrics["per_claim"]["1"]
    assert case["contains_complete_gold_rationale"] is True
    assert case["complete_gold_document_count"] == 1
    assert metrics["gold_document_complete_rationale"] == {
        "gold_document_count": 1,
        "complete_gold_document_count": 1,
        "recall": 1.0,
    }


def test_label_free_pooled_counts_match_official_sentence_selection(selector):
    corpus = [
        {"doc_id": 10, "title": "A", "abstract": ["a", "b"]},
        {"doc_id": 20, "title": "B", "abstract": ["c"]},
    ]
    claims = [
        {
            "id": 1,
            "claim": "a",
            "evidence": {"10": [{"label": "SUPPORT", "sentences": [0]}]},
            "cited_doc_ids": [10],
        },
        {"id": 2, "claim": "unknown", "evidence": {}, "cited_doc_ids": []},
    ]
    frozen = [
        {"doc_id": 10, "sentence_index": 0},
        {"doc_id": 20, "sentence_index": 0},
    ]
    selected = {1: [0], 2: [1]}
    metrics = selector._selector_metrics(
        [1, 2], {row["id"]: row for row in claims}, frozen, selected, {1}
    )
    predictions = [
        {"id": 1, "evidence": {"10": {"label": "SUPPORT", "sentences": [0]}}},
        {"id": 2, "evidence": {"20": {"label": "SUPPORT", "sentences": [0]}}},
    ]
    official = compute_official_pipeline_metrics(
        predictions, validate_official_data(corpus, claims)
    )
    observed = metrics["official_compatible_pooled_sentence_selection"]
    assert observed["precision"] == official["sentence_selection"]["precision"]
    assert observed["recall"] == official["sentence_selection"]["recall"]
    assert observed["f1"] == official["sentence_selection"]["f1"]


def test_standardized_l2_logistic_model_has_auditable_coefficients(selector):
    corpus, claims, retrieval = _synthetic_data()
    frozen, targets, by_claim, _ = _frozen(selector, corpus, claims, retrieval)
    pair_indices = [index for claim in claims for index in by_claim[claim["id"]]]
    scaler, classifier, model, transformer = selector._fit_hybrid(
        frozen, targets, pair_indices, include_vectorizer_state=True
    )
    scores = selector._score_pairs(
        "hybrid",
        frozen,
        pair_indices,
        transformer=transformer,
        fitted=(scaler, classifier),
    )
    assert len(scores) == len(frozen)
    assert model["feature_names"] == list(selector.FEATURE_NAMES)
    assert model["logistic_regression"]["penalty"] == "l2"
    assert len(model["standard_scaler"]["mean"]) == len(selector.FEATURE_NAMES)
    assert len(model["logistic_regression"]["coefficients"]) == len(
        selector.FEATURE_NAMES
    )
    assert "vocabulary" in model["tfidf"]["word"]
    assert "idf_by_index" in model["tfidf"]["char_wb"]


def test_nested_oof_has_complete_coverage_and_train_only_inner_models(
    selector, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(selector, "RELATION_CLAIM_COUNT", 4)
    corpus, claims, retrieval = _synthetic_data()
    frozen, targets, by_claim, retrieval_by_claim = _frozen(
        selector, corpus, claims, retrieval
    )
    claim_ids = [row["id"] for row in claims]
    components = selector._connected_components(
        claim_ids,
        retrieval_by_claim,
        {row["id"]: row["claim"] for row in claims},
    )
    outer = selector._assign_component_folds(components, 4)
    metrics, selections, audits, cases = selector._run_nested_oof(
        claims=claims,
        frozen=frozen,
        targets=targets,
        by_claim=by_claim,
        components=components,
        outer_folds=outer,
        retrievable_relation_ids={1, 3, 5, 7},
    )
    assert set(metrics) == set(selector.STRATEGIES)
    assert all(set(rows) == set(claim_ids) for rows in selections.values())
    assert len(audits) == 4
    assert len(cases) == 3 * len(claims)
    assert all(
        audit["inner_isolation"]["cross_fold_doc_overlap_count"] == 0
        for audit in audits
    )
    assert all(
        audit["strategies"]["hybrid"]["outer_model"] is not None
        for audit in audits
    )
    outer_hashes = []
    for audit in audits:
        hybrid_hash = audit["strategies"]["hybrid"]["outer_model"]["tfidf"][
            "state_sha256"
        ]
        lexical_hash = audit["strategies"]["lexical_only"]["outer_model"][
            "tfidf"
        ]["state_sha256"]
        assert hybrid_hash == lexical_hash
        outer_hashes.append(hybrid_hash)
        assert "vocabulary" not in audit["strategies"]["hybrid"]["outer_model"][
            "tfidf"
        ]["word"]
    assert len(set(outer_hashes)) == 4


def test_cluster_bootstrap_is_paired_component_level_and_deterministic(selector):
    components = [
        {"claim_ids": [1, 2]},
        {"claim_ids": [3]},
        {"claim_ids": [4]},
    ]
    hybrid = {
        claim_id: {
            "is_relation": True,
            "best_rationale_set_f1": value,
            "official_sentence_selection_sufficient_counts": {
                "relevant": 1,
                "retrieved": 1,
                "correct_selection": 1,
            },
        }
        for claim_id, value in {1: 1.0, 2: 0.8, 3: 0.7, 4: 0.0}.items()
    }
    baseline = {
        claim_id: {
            "is_relation": True,
            "best_rationale_set_f1": value,
            "official_sentence_selection_sufficient_counts": {
                "relevant": 1,
                "retrieved": 1,
                "correct_selection": 0,
            },
        }
        for claim_id, value in {1: 0.5, 2: 0.4, 3: 0.3, 4: 0.0}.items()
    }
    first = selector._paired_component_bootstrap(
        components=components,
        hybrid_cases=hybrid,
        baseline_cases=baseline,
        metric="official_compatible_pooled_sentence_selection",
        samples=10_000,
        seed=7,
    )
    second = selector._paired_component_bootstrap(
        components=components,
        hybrid_cases=hybrid,
        baseline_cases=baseline,
        metric="official_compatible_pooled_sentence_selection",
        samples=10_000,
        seed=7,
    )
    assert first == second
    assert first["all_components_in_sampling_frame"] is True
    assert first["component_count"] == 3
    assert first["confidence_interval"]["lower"] > 0
    with pytest.raises(selector.SelectorCalibrationError, match="exactly 10000"):
        selector._paired_component_bootstrap(
            components=components,
            hybrid_cases=hybrid,
            baseline_cases=baseline,
            metric="official_compatible_pooled_sentence_selection",
            samples=1,
        )


def _gate_fixture(selector, *, hybrid_value, baseline_value, ci_lower):
    def metric(value):
        return {
            "custom_best_rationale_macro_f1": {"value": value},
            "official_compatible_pooled_sentence_selection": {"f1": value},
            "relation_claim_count": 505,
            "retrievable_relation_claim_count": 396,
            "nei_claim_count": 304,
            "all_relation_complete_rationale_recall": 0.60,
            "retrievable_relation_complete_rationale_recall": 0.70,
            "nei_correct_empty_rate": 0.80,
        }

    metrics = {
        "hybrid": metric(hybrid_value),
        "lexical_only": metric(baseline_value),
        "nli_only": metric(baseline_value - 0.01),
    }
    folds = []
    for _index in range(4):
        strategy_rows = {}
        for strategy, value in (
            ("hybrid", hybrid_value),
            ("lexical_only", baseline_value),
            ("nli_only", baseline_value - 0.01),
        ):
            strategy_rows[strategy] = {
                "outer_validation_metrics": {
                    "official_compatible_pooled_sentence_selection": {
                        "f1": value
                    }
                }
            }
        folds.append(
            {
                "strategies": strategy_rows
            }
        )
    cases = []
    for strategy in selector.STRATEGIES:
        cases.extend(
            {
                "strategy": strategy,
                "claim_id": claim_id,
                "is_relation": claim_id < 505,
            }
            for claim_id in range(809)
        )
    comparison = {"confidence_interval": {"lower": ci_lower}}
    bootstrap = {
        "samples": 10_000,
        "component_count": 228,
        "metrics": {
            metric_name: {
                "hybrid_vs_lexical_only": comparison,
                "hybrid_vs_nli_only": comparison,
            }
            for metric_name in (
                "custom_best_rationale_macro",
                "official_compatible_pooled_sentence_selection",
            )
        },
    }
    isolation = {
        "claim_coverage_complete": True,
        "cross_fold_claim_overlap_count": 0,
        "cross_fold_doc_overlap_count": 0,
    }
    return metrics, folds, cases, bootstrap, isolation


def test_gate_requires_effect_size_ci_and_every_integrity_check(selector):
    values = _gate_fixture(
        selector, hybrid_value=0.55, baseline_value=0.52, ci_lower=0.005
    )
    passed = selector._gate(
        metrics=values[0],
        audit_folds=values[1],
        cases=values[2],
        bootstrap=values[3],
        isolation=values[4],
        feature_whitelist_violations=0,
    )
    assert passed["status"] == "pass"
    assert passed["full_train_model_permitted"] is True
    assert all(passed["checks"].values())

    failed_values = _gate_fixture(
        selector, hybrid_value=0.53, baseline_value=0.52, ci_lower=-0.001
    )
    rejected = selector._gate(
        metrics=failed_values[0],
        audit_folds=failed_values[1],
        cases=failed_values[2],
        bootstrap=failed_values[3],
        isolation=failed_values[4],
        feature_whitelist_violations=0,
    )
    assert rejected["status"] == "reject"
    assert rejected["full_train_model_permitted"] is False


def test_full_refit_uses_outer_choices_without_full_train_retuning(
    selector, monkeypatch: pytest.MonkeyPatch
):
    audits = []
    for threshold, maximum in zip(
        [0.2, 0.4, 0.6, 0.8], [1, 2, 2, 1], strict=True
    ):
        audits.append(
            {
                "strategies": {
                    "hybrid": {
                        "inner_tuning": {
                            "selected": {
                                "threshold": threshold,
                                "max_sentences": maximum,
                            }
                        }
                    }
                }
            }
        )

    calls = []

    def fake_fit(*args, **kwargs):
        calls.append(kwargs)
        return object(), object(), {"coefficients": "fitted"}, {"fit_claim_ids": [1]}

    monkeypatch.setattr(selector, "_fit_hybrid", fake_fit)
    model = selector._full_train_model(
        claims=[],
        frozen=[{"claim_id": 1}],
        targets=[{}],
        by_claim={},
        audit_folds=audits,
    )
    assert model["selection"]["threshold"] == pytest.approx(0.5)
    assert model["selection"]["max_sentences"] == 1
    assert model["selection"]["post_gate_full_train_retuning"] is False
    assert calls == [{"include_vectorizer_state": True}]


def test_rejected_synthetic_run_is_hashed_and_emits_no_full_model(
    selector, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    corpus, claims, retrieval = _synthetic_data()
    inputs = {}
    for name in (
        "corpus.jsonl",
        "claims_train.jsonl",
        "retrieval_train.jsonl",
        "manifest.json",
    ):
        path = tmp_path / name
        path.write_text("{}\n", encoding="utf-8")
        inputs[name] = path

    monkeypatch.setattr(selector, "SOURCE_CLAIM_COUNT", 8)
    monkeypatch.setattr(selector, "RELATION_CLAIM_COUNT", 4)
    monkeypatch.setattr(selector, "NEI_CLAIM_COUNT", 4)
    monkeypatch.setattr(selector, "RETRIEVABLE_RELATION_CLAIM_COUNT", 4)
    monkeypatch.setattr(selector, "CANDIDATE_PAIR_COUNT", 24)
    monkeypatch.setattr(selector, "POSITIVE_PAIR_COUNT", 4)
    monkeypatch.setattr(selector, "CONNECTED_COMPONENT_COUNT", 4)
    monkeypatch.setattr(selector, "LARGEST_COMPONENT_CLAIM_COUNT", 2)
    monkeypatch.setattr(
        selector,
        "_validate_inputs",
        lambda **kwargs: (corpus, claims, retrieval),
    )
    monkeypatch.setattr(
        selector,
        "_model_identity",
        lambda path: {"path": str(path), "revision": selector.MODEL_REVISION},
    )
    inference_calls = 0
    output = tmp_path / "train-result"

    def infer(premises, hypotheses, **kwargs):
        nonlocal inference_calls
        assert (output / "preregistration.json").is_file()
        inference_calls += 1
        values = []
        for premise, hypothesis in zip(premises, hypotheses, strict=True):
            marker = hypothesis.split()[0]
            values.append(
                [0.96, 0.02, 0.02]
                if marker in premise
                else [0.05, 0.90, 0.05]
            )
        return values, {
            "inference_pass_count": 1,
            "pair_count": len(values),
            "elapsed_seconds": 0.01,
        }

    monkeypatch.setattr(selector, "_infer_probabilities", infer)
    monkeypatch.setattr(
        selector.importlib.metadata, "version", lambda package: "synthetic"
    )
    monkeypatch.setattr(
        selector,
        "_full_train_model",
        lambda **kwargs: pytest.fail("full model must not fit after gate rejection"),
    )
    selector.run(
        corpus_path=inputs["corpus.jsonl"],
        claims_path=inputs["claims_train.jsonl"],
        retrieval_path=inputs["retrieval_train.jsonl"],
        retrieval_manifest_path=inputs["manifest.json"],
        model_dir=tmp_path / selector.MODEL_REVISION,
        output=output,
        device="cpu",
    )
    assert inference_calls == 1
    gate = json.loads((output / "acceptance_gate.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    preregistration = json.loads(
        (output / "preregistration.json").read_text(encoding="utf-8")
    )
    configuration = json.loads(
        (output / "configuration.json").read_text(encoding="utf-8")
    )
    bootstrap = json.loads(
        (output / "cluster_bootstrap.json").read_text(encoding="utf-8")
    )
    assert gate["status"] == "reject"
    assert not (output / "selector_model.json").exists()
    assert manifest["evaluation_status"]["dev_read_or_scored"] is False
    assert manifest["validation"]["nli_inference_pass_count"] == 1
    assert preregistration["written_before_nli_or_oof"] is True
    assert configuration == preregistration["protocol"]
    assert manifest["preregistration"]["sha256"] == selector._sha256(
        output / "preregistration.json"
    )
    assert (
        manifest["preregistration"]["protocol_sha256"]
        == preregistration["protocol_sha256"]
    )
    assert manifest["implementation"]["unchanged_during_run"] is True
    assert set(bootstrap["metrics"]) == {
        "custom_best_rationale_macro",
        "official_compatible_pooled_sentence_selection",
    }
    assert all(
        set(comparisons) == {"hybrid_vs_lexical_only", "hybrid_vs_nli_only"}
        for comparisons in bootstrap["metrics"].values()
    )
    assert set(manifest["outputs_sha256"]) == {
        path.name for path in output.iterdir() if path.name != "run_manifest.json"
    }
    for name, digest in manifest["outputs_sha256"].items():
        assert selector._sha256(output / name) == digest


def test_crash_after_preregistration_retains_incomplete_directory(
    selector, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    corpus, claims, retrieval = _synthetic_data()
    paths = []
    for name in (
        "corpus.jsonl",
        "claims_train.jsonl",
        "retrieval_train.jsonl",
        "manifest.json",
    ):
        path = tmp_path / name
        path.write_text("{}\n", encoding="utf-8")
        paths.append(path)
    for name, value in (
        ("SOURCE_CLAIM_COUNT", 8),
        ("RELATION_CLAIM_COUNT", 4),
        ("NEI_CLAIM_COUNT", 4),
        ("RETRIEVABLE_RELATION_CLAIM_COUNT", 4),
        ("CANDIDATE_PAIR_COUNT", 24),
        ("POSITIVE_PAIR_COUNT", 4),
        ("CONNECTED_COMPONENT_COUNT", 4),
        ("LARGEST_COMPONENT_CLAIM_COUNT", 2),
    ):
        monkeypatch.setattr(selector, name, value)
    monkeypatch.setattr(
        selector, "_validate_inputs", lambda **kwargs: (corpus, claims, retrieval)
    )
    monkeypatch.setattr(
        selector,
        "_model_identity",
        lambda path: {"path": str(path), "identity_sha256": "pinned"},
    )

    def crash(*args, **kwargs):
        raise RuntimeError("synthetic crash")

    monkeypatch.setattr(selector, "_infer_probabilities", crash)
    output = tmp_path / "crash-train-run"
    with pytest.raises(RuntimeError, match="synthetic crash"):
        selector.run(
            corpus_path=paths[0],
            claims_path=paths[1],
            retrieval_path=paths[2],
            retrieval_manifest_path=paths[3],
            model_dir=tmp_path / selector.MODEL_REVISION,
            output=output,
            device="cpu",
        )
    assert (output / "preregistration.json").is_file()
    assert not (output / "run_manifest.json").exists()


def test_existing_output_is_rejected_before_any_input_read(selector, tmp_path: Path):
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(selector.SelectorCalibrationError, match="already exists"):
        selector.run(
            corpus_path=tmp_path / "missing-corpus",
            claims_path=tmp_path / "missing-claims",
            retrieval_path=tmp_path / "missing-retrieval",
            retrieval_manifest_path=tmp_path / "missing-manifest",
            model_dir=tmp_path / "missing-model",
            output=output,
            device="cpu",
        )
