from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "freeze_scifact_selector_upstream.py"


@pytest.fixture()
def freezer():
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location(
        "scifact_upstream_freeze_test", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _synthetic_context_inputs(
    freezer, monkeypatch, primary_sentences=2, first_selection=None
):
    corpus, claims, retrieval = [], [], []
    for claim_id in range(1, 5):
        doc_ids = [claim_id * 10 + offset for offset in range(3)]
        for offset, doc_id in enumerate(doc_ids):
            abstract = [f"OOF selected sentence for {claim_id}."]
            if offset == 0:
                abstract.append(f"Gold-only sentence for {claim_id}.")
                abstract.extend(
                    f"Additional sentence {index} for {claim_id}."
                    for index in range(2, primary_sentences)
                )
            corpus.append(
                {"doc_id": doc_id, "title": f"Paper {doc_id}", "abstract": abstract}
            )
        claims.append(
            {
                "id": claim_id,
                "claim": f"Claim text {claim_id}.",
                "evidence": {}
                if claim_id == 2
                else {str(doc_ids[0]): [{"label": "SUPPORT", "sentences": [1]}]},
                "annotation_note": "PRIVATE_GOLD_SENTINEL",
            }
        )
        retrieval.append({"claim_id": claim_id, "doc_ids": doc_ids})
    for name, value in (
        ("SOURCE_CLAIM_COUNT", 4),
        ("SOURCE_CORPUS_COUNT", 12),
        ("RELATION_CLAIM_COUNT", 3),
        ("CANDIDATE_PAIR_COUNT", 4 * (primary_sentences + 2)),
    ):
        monkeypatch.setattr(freezer.base, name, value)
    frozen, by_claim, retrieval_by_claim = freezer.base._build_candidates(
        corpus, claims, retrieval
    )
    components = freezer.base._connected_components(
        [row["id"] for row in claims],
        retrieval_by_claim,
        {row["id"]: row["claim"] for row in claims},
    )
    outer_folds = freezer.base._assign_component_folds(components, 4)
    all_ids = {row["id"] for row in claims}
    audits, cases = [], []

    def binding(claim_ids):
        return {
            "fit_claim_ids": claim_ids,
            "fit_claim_ids_sha256": freezer._canonical_sha(claim_ids),
            "fit_pair_count": sum(len(by_claim[value]) for value in claim_ids),
        }

    for fold in outer_folds:
        train_ids = sorted(all_ids - set(fold["claim_ids"]))
        inner_folds = freezer.base._assign_component_folds(
            [
                row
                for row in components
                if row["component_id"] not in fold["component_ids"]
            ],
            3,
        )
        choice = {
            "accept_threshold": 0.8,
            "expansion_threshold": 0.4,
            "max_sentences": primary_sentences,
        }
        inner_models = [
            {
                "inner_fold": inner["fold"],
                "model": {
                    "tfidf": binding(sorted(set(train_ids) - set(inner["claim_ids"])))
                },
            }
            for inner in inner_folds
        ]
        audits.append(
            {
                "outer_fold": fold["fold"],
                "inner_folds": inner_folds,
                "strategies": {
                    "hybrid": {
                        "outer_model": {"tfidf": binding(train_ids)},
                        "inner_tuning": {
                            "selected": choice,
                            "status": "feasible",
                            "inner_models": inner_models,
                        },
                        "diagnostic_fallback_used": False,
                    }
                },
            }
        )
        for claim_id in fold["claim_ids"]:
            selected = [] if claim_id == 2 else [by_claim[claim_id][0]]
            if claim_id == 1 and first_selection is not None:
                selected = [by_claim[claim_id][index] for index in first_selection]
            cases.append(
                {
                    "strategy": "hybrid",
                    "source": "oof",
                    "claim_id": claim_id,
                    "outer_fold": fold["fold"],
                    "inner_selection_status": "feasible",
                    "diagnostic_fallback_used": False,
                    "prediction_configuration_source": "feasible_inner_selection",
                    "selection_parameters": choice,
                    "selected_pair_indices": selected,
                    "is_relation": claim_id != 2,
                    "gold_label": "PRIVATE_GOLD_SENTINEL",
                }
            )
    return {
        "claims": claims,
        "corpus": corpus,
        "retrieval": retrieval,
        "frozen": frozen,
        "cases": cases,
        "outer_folds": outer_folds,
        "audits": audits,
    }


def test_contexts_reproduce_only_oof_selection_and_never_gold_or_fullfit_predictions(
    freezer, monkeypatch
):
    values = _synthetic_context_inputs(freezer, monkeypatch)
    monkeypatch.setattr(
        freezer.selector_v2,
        "_full_train_model",
        lambda **kwargs: pytest.fail(
            "full-train selector must never be fit or applied here"
        ),
    )
    contexts = freezer._build_contexts(**values)
    assert len(contexts) == 4
    by_id = {row["claim_id"]: row for row in contexts}
    assert by_id[2]["context"]["documents"] == []
    for claim_id in (1, 3, 4):
        row = by_id[claim_id]
        assert row["source"] == "oof"
        assert row["canonical_context_sha256"] == freezer._canonical_sha(row["context"])
        assert row["context"]["documents"][0]["sentences"] == [
            {"sentence_index": 0, "text": f"OOF selected sentence for {claim_id}."}
        ]
        assert set(row["context"]) == {"claim", "documents"}
    rendered = json.dumps(contexts)
    assert "PRIVATE_GOLD_SENTINEL" not in rendered
    assert "Gold-only sentence" not in rendered
    assert "is_relation" not in rendered
    assert "gold_label" not in rendered


@pytest.mark.parametrize(
    "change", ["full_fit", "wrong_fold", "fallback", "foreign_pair", "duplicate_pair"]
)
def test_contexts_reject_non_oof_or_invalid_sentence_provenance(
    freezer, monkeypatch, change
):
    values = _synthetic_context_inputs(freezer, monkeypatch)
    case = next(row for row in values["cases"] if row["claim_id"] == 1)
    if change == "full_fit":
        case["source"] = "full_train_fit"
    elif change == "wrong_fold":
        case["outer_fold"] = (case["outer_fold"] + 1) % 4
    elif change == "fallback":
        case["diagnostic_fallback_used"] = True
    elif change == "foreign_pair":
        case["selected_pair_indices"] = [
            next(row["pair_index"] for row in values["frozen"] if row["claim_id"] == 3)
        ]
    else:
        case["selected_pair_indices"] *= 2
    with pytest.raises(freezer.UpstreamFreezeError):
        freezer._build_contexts(**values)


def test_full_train_transformer_disguised_as_outer_model_is_rejected(
    freezer, monkeypatch
):
    values = _synthetic_context_inputs(freezer, monkeypatch)
    binding = values["audits"][0]["strategies"]["hybrid"]["outer_model"]["tfidf"]
    binding["fit_claim_ids"] = [1, 2, 3, 4]
    binding["fit_claim_ids_sha256"] = freezer._canonical_sha([1, 2, 3, 4])
    binding["fit_pair_count"] = 16
    with pytest.raises(freezer.UpstreamFreezeError, match="OOF transformer"):
        freezer._build_contexts(**values)


def _synthetic_verified_bundle(freezer, tmp_path, monkeypatch):
    values = _synthetic_context_inputs(freezer, monkeypatch)
    contexts = freezer._build_contexts(**values)
    source = tmp_path / "upstream-train-source.json"
    source.write_text('{"synthetic":true}\n', encoding="utf-8")
    bundle = {
        "selector_run": {
            "path": str(tmp_path / "prior-train-run"),
            "manifest_sha256": freezer._sha(source),
        },
        "bindings": {
            "synthetic_source": {
                "path": str(source),
                "sha256": freezer._sha(source),
                "bytes": source.stat().st_size,
            }
        },
        "contexts": contexts,
        # Deployment parameters intentionally differ from the OOF fold choices.
        "selection": {
            "accept_threshold": 0.7,
            "expansion_threshold": 0.55,
            "max_sentences": 4,
        },
        "validation": {"claim_count": 4, "no_full_train_model_applied_to_train": True},
    }
    monkeypatch.setattr(freezer, "_validate_selector_run", lambda path: bundle)
    return bundle, source


def test_freeze_roundtrip_binds_shared_contexts_and_separate_deployment_parameters(
    freezer, tmp_path, monkeypatch
):
    bundle, _ = _synthetic_verified_bundle(freezer, tmp_path, monkeypatch)
    output = tmp_path / "frozen-train-contexts"
    freezer.freeze(tmp_path / "prior-train-run", output)
    loaded = freezer.load_frozen(output)
    assert loaded["contexts"] == bundle["contexts"]
    document = loaded["freeze"]
    assert document["deployable_selector_selection"] == bundle["selection"]
    assert (
        document["context_contract"][
            "future_agent_arms_must_share_identical_context_bytes"
        ]
        is True
    )
    assert document["context_contract"]["gold_or_diagnostic_fields_in_payload"] is False
    assert document["outputs_sha256"][freezer.CONTEXTS_NAME] == freezer._sha(
        output / freezer.CONTEXTS_NAME
    )
    assert document["dev_read_or_scored"] is False
    assert document["reportable_as_generalization"] is False
    assert freezer.load_frozen(output / "freeze.json") == loaded
    with pytest.raises(freezer.UpstreamFreezeError, match="already exists"):
        freezer.freeze(tmp_path / "prior-train-run", output)


@pytest.mark.parametrize("target", ["contexts", "freeze_metadata", "source"])
def test_freeze_loader_rejects_modified_context_metadata_or_bound_source(
    freezer, tmp_path, monkeypatch, target
):
    _, source = _synthetic_verified_bundle(freezer, tmp_path, monkeypatch)
    output = tmp_path / "frozen-train-contexts"
    freezer.freeze(tmp_path / "prior-train-run", output)
    if target == "contexts":
        path = output / freezer.CONTEXTS_NAME
        rows = freezer._read_jsonl(path)
        rows[0]["context"]["claim"] = "Changed claim"
        path.write_bytes(freezer._jsonl_bytes(rows))
    elif target == "freeze_metadata":
        path = output / "freeze.json"
        document = freezer._read_json(path)
        document["deployable_selector_selection"]["max_sentences"] = 1
        path.write_bytes(freezer._json_bytes(document))
    else:
        source.write_text("changed", encoding="utf-8")
    with pytest.raises(freezer.UpstreamFreezeError):
        freezer.load_frozen(output)


def test_unknown_selector_manifest_cannot_self_declare_trusted_identity(
    freezer, tmp_path
):
    source = tmp_path / "unknown-train-run"
    source.mkdir()
    (source / "run_manifest.json").write_text(
        json.dumps({"self_declared_sha256": freezer.PINNED_SELECTOR_MANIFEST_SHA256}),
        encoding="utf-8",
    )
    with pytest.raises(freezer.UpstreamFreezeError, match="selector_manifest"):
        freezer._validate_selector_run(source)


def test_citation_metadata_preserves_score_order_while_prompt_sentences_are_sorted(
    freezer, monkeypatch
):
    values = _synthetic_context_inputs(
        freezer, monkeypatch, primary_sentences=4, first_selection=[3, 1, 0, 2]
    )
    contexts = freezer._build_contexts(**values)
    row = next(value for value in contexts if value["claim_id"] == 1)
    assert row["citation_sentence_indices"] == {"10": [3, 1, 0, 2]}
    assert [
        value["sentence_index"] for value in row["context"]["documents"][0]["sentences"]
    ] == [0, 1, 2, 3]
    assert "citation_sentence_indices" not in row["context"]
    assert row["canonical_context_sha256"] == freezer._canonical_sha(row["context"])


def test_redeclared_output_hash_cannot_authorize_changed_citation_metadata(
    freezer, tmp_path, monkeypatch
):
    _synthetic_verified_bundle(freezer, tmp_path, monkeypatch)
    output = tmp_path / "frozen-train-contexts"
    freezer.freeze(tmp_path / "prior-train-run", output)
    context_path = output / freezer.CONTEXTS_NAME
    rows = freezer._read_jsonl(context_path)
    row = next(value for value in rows if value["claim_id"] == 1)
    row["citation_sentence_indices"] = {"10": [1]}
    context_path.write_bytes(freezer._jsonl_bytes(rows))
    document_path = output / "freeze.json"
    document = freezer._read_json(document_path)
    document["outputs_sha256"][freezer.CONTEXTS_NAME] = freezer._sha(context_path)
    document_path.write_bytes(freezer._json_bytes(document))
    with pytest.raises(freezer.UpstreamFreezeError, match="raw OOF reconstruction"):
        freezer.load_frozen(output)
