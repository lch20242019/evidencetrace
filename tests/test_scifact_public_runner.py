from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from evidencetrace.eval.models import EvalCase, SourceFixture
from evidencetrace.eval.scifact_judge import NliScores
from evidencetrace.models import Relation
from scripts.run_scifact_public_dev import (
    BASELINES,
    RETRIEVAL_TOP_K,
    SCIFACT_ALLOWED_RELATIONS,
    SENTENCE_INDEX_RELATIVE,
    SciFactRunnerError,
    _model_identity,
    run_pack,
)

REPO = Path(__file__).resolve().parents[1]
PUBLIC_PACK = REPO / "eval_sets" / "scifact_public_dev"
UPSTREAM_MODEL_ID = "cross-encoder/nli-MiniLM2-L6-H768"
REVISION = "b95119ce93d3e065de6214e38cd4a97b0f2f2c6d"
NEUTRAL = NliScores(entailment=0.05, contradiction=0.05, neutral=0.90)


class FakeScorer:
    model_id = UPSTREAM_MODEL_ID
    revision = REVISION
    batch_size = 16

    def __init__(self, *, exact_entailment: bool = True) -> None:
        self.exact_entailment = exact_entailment
        self.calls: list[tuple[tuple[str, str], ...]] = []

    def score_many(self, pairs: Sequence[tuple[str, str]]) -> Sequence[NliScores]:
        captured = tuple(pairs)
        self.calls.append(captured)
        return tuple(
            NliScores(entailment=0.92, contradiction=0.03, neutral=0.05)
            if self.exact_entailment and premise == hypothesis
            else NEUTRAL
            for premise, hypothesis in captured
        )


def _jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _small_pack(
    tmp_path: Path,
    *,
    pairs: tuple[tuple[str, str], ...] | None = None,
) -> Path:
    pack = tmp_path / "pack"
    values = pairs or (
        (
            "Alpha treatment reduces fever in controlled trials.",
            "Alpha treatment reduces fever in controlled trials.",
        ),
        (
            "Beta therapy improves sleep quality in adult patients.",
            "Beta therapy improves sleep quality in adult patients.",
        ),
    )
    sources: list[SourceFixture] = []
    cases: list[EvalCase] = []
    indexes: list[dict] = []
    for index, (claim, text) in enumerate(values, 1):
        source_id = f"unit_source_{index}"
        case_id = f"unit_case_{index}"
        url = f"https://example.test/{source_id}"
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        sources.append(
            SourceFixture(
                source_id=source_id,
                url=url,
                content=text,
                provenance="unit fixture",
                content_hash=content_hash,
            )
        )
        cases.append(
            EvalCase(
                case_id=case_id,
                claim_text=claim,
                source_fixture="sources/source_snapshots.jsonl",
                source_id=source_id,
                source_url=url,
                gold_relation=Relation.ENTAILED,
                gold_evidence_span=text,
                claim_type="unit_scientific_claim",
                mutation_type="unit_authored_claim",
                split="dev",
                provenance="unit fixture",
                annotation_status="human_reviewed",
                annotation_notes="Test-only deterministic fixture.",
                source_sha256=content_hash,
            )
        )
        doc_id = 7000 + index
        locator = f"scifact://document/{doc_id}/sentence/0"
        span = {
            "doc_id": doc_id,
            "sentence_id": 0,
            "char_start": 0,
            "char_end": len(text),
            "locator": locator,
            "text_sha256": content_hash,
        }
        indexes.append(
            {
                "case_id": case_id,
                "source_id": source_id,
                "documents": [
                    {
                        "doc_id": doc_id,
                        "title": f"Unit title {index}",
                        "char_start": 0,
                        "char_end": len(text),
                        "sentence_keys": [f"{doc_id}:0"],
                    }
                ],
                "sentence_spans": [span],
            }
        )
    _write_jsonl(
        pack / "sources" / "source_snapshots.jsonl",
        [source.model_dump(mode="json") for source in sources],
    )
    _write_jsonl(
        pack / "cases.jsonl",
        [case.model_dump(mode="json") for case in cases],
    )
    index_path = pack / SENTENCE_INDEX_RELATIVE
    _write_jsonl(index_path, indexes)
    # Deliberately invalid gold: a successful run proves the runner never reads it.
    (pack / "gold_rationales.jsonl").write_text("not json\n", encoding="utf-8")
    manifest = {
        "schema_version": "scifact-public-dev-v2",
        "mapping": {"prediction_sentence_index": SENTENCE_INDEX_RELATIVE},
        "output_sha256": {
            SENTENCE_INDEX_RELATIVE: hashlib.sha256(index_path.read_bytes()).hexdigest()
        },
    }
    (pack / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    return pack


def _model(tmp_path: Path) -> tuple[Path, str]:
    model = tmp_path / "unit_model"
    model.mkdir(exist_ok=True)
    weights = model / "model.safetensors"
    weights.write_bytes(b"unit test weights")
    return model, hashlib.sha256(weights.read_bytes()).hexdigest()


def _run(
    pack: Path,
    output: Path,
    model: Path,
    weights_sha256: str,
    *,
    scorer: FakeScorer | None = None,
    expected_case_count: int = 2,
) -> Path:
    return run_pack(
        pack,
        output,
        nli_model_path=model,
        nli_upstream_model_id=UPSTREAM_MODEL_ID,
        nli_revision=REVISION,
        nli_weights_sha256=weights_sha256,
        nli_batch_size=16,
        scorer=scorer or FakeScorer(),
        expected_case_count=expected_case_count,
    )


def test_small_pack_is_three_way_structured_and_provenance_bound(
    tmp_path: Path,
) -> None:
    pack = _small_pack(tmp_path)
    model, weights_sha256 = _model(tmp_path)
    scorer = FakeScorer()
    output = _run(
        pack,
        tmp_path / "run",
        model,
        weights_sha256,
        scorer=scorer,
    )
    rows = _jsonl(output / "eval_results.jsonl")
    assert len(rows) == 2 * len(BASELINES) == 4
    assert {row["baseline"] for row in rows} == set(BASELINES)
    assert {row["predicted_relation"] for row in rows} <= {
        relation.value for relation in SCIFACT_ALLOWED_RELATIONS
    }
    assert all(len(row["retrieved_evidence"]) <= RETRIEVAL_TOP_K for row in rows)
    assert all(
        row["evaluation_status"]
        == {"reportable": False, "mode": "test_double_nonreportable"}
        for row in rows
    )
    assert all(row["model_provider"] != "local_transformers_nli" for row in rows)
    for row in rows:
        assert row["retrieved_texts"] == [
            item["text"] for item in row["retrieved_evidence"]
        ]
        assert [item["rank"] for item in row["retrieved_evidence"]] == list(
            range(1, len(row["retrieved_evidence"]) + 1)
        )
        if row["predicted_evidence"]:
            assert (
                row["predicted_evidence_span"] == row["predicted_evidence"][0]["text"]
            )
        else:
            assert row["predicted_evidence_span"] is None
    by_case: dict[str, list[dict]] = {}
    for row in rows:
        by_case.setdefault(row["case_id"], []).append(row)
    assert all(
        values[0]["retrieved_evidence"] == values[1]["retrieved_evidence"]
        for values in by_case.values()
    )

    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["baselines"] == list(BASELINES)
    assert manifest["evaluation_status"] == {
        "reportable": False,
        "mode": "test_double_nonreportable",
    }
    assert manifest["task_profile"]["allowed_relations"] == [
        relation.value for relation in SCIFACT_ALLOWED_RELATIONS
    ]
    assert manifest["counts"] == {
        "cases": 2,
        "rows": 4,
        "rows_by_baseline": {baseline: 2 for baseline in BASELINES},
        "external_provider_calls": 0,
        "local_nli_score_many_calls": 2,
        "local_nli_pair_count": 2,
        "local_nli_subset_count": 2,
        "local_nli_batch_count": 2,
    }
    assert manifest["local_nli"]["offline_only"] is True
    assert manifest["local_nli"]["upstream_model_id"] == UPSTREAM_MODEL_ID
    assert manifest["local_nli"]["revision"] == REVISION
    assert manifest["local_nli"]["weights_sha256"] == weights_sha256
    assert manifest["local_nli"]["absolute_local_path_recorded"] is False
    assert manifest["local_nli"]["hf_hub_download_metadata"] == {}
    assert manifest["local_nli"]["hf_hub_metadata_sha256"] == {}
    assert manifest["local_nli"]["input_length_telemetry"] == {
        "available": False,
        "max_input_tokens": None,
        "max_observed_input_tokens": None,
        "rejected_overlength_pair_count": None,
    }
    assert manifest["inputs"]["sha256"][SENTENCE_INDEX_RELATIVE]
    assert "gold_rationales.jsonl" not in manifest["inputs"]["sha256"]
    assert len(scorer.calls) == 2


def test_partial_coverage_and_recommendation_never_escape_profile(
    tmp_path: Path,
) -> None:
    pack = _small_pack(
        tmp_path,
        pairs=(
            (
                "Alpha treatment reduces fever rapidly in children.",
                "Alpha treatment reduces fever in controlled trials.",
            ),
            (
                "Clinicians should use this best therapy.",
                "Clinicians should use this best therapy.",
            ),
        ),
    )
    model, weights_sha256 = _model(tmp_path)
    output = _run(pack, tmp_path / "run", model, weights_sha256)
    rows = _jsonl(output / "eval_results.jsonl")
    lexical = {
        row["case_id"]: row for row in rows if row["baseline"] == "lexical_rules"
    }
    assert lexical["unit_case_1"]["predicted_relation"] == "not_in_source"
    assert {row["predicted_relation"] for row in rows}.isdisjoint(
        {"partially_entailed", "not_checkable"}
    )


def test_existing_output_fails_closed_without_overwriting(tmp_path: Path) -> None:
    pack = _small_pack(tmp_path)
    model, weights_sha256 = _model(tmp_path)
    output = tmp_path / "occupied"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("do not replace", encoding="utf-8")

    with pytest.raises(SciFactRunnerError, match="refusing to overwrite"):
        _run(pack, output, model, weights_sha256)
    assert sentinel.read_text(encoding="utf-8") == "do not replace"
    assert not (output / "eval_results.jsonl").exists()


def test_wrong_weights_hash_and_mutable_revision_fail_closed(tmp_path: Path) -> None:
    pack = _small_pack(tmp_path)
    model, weights_sha256 = _model(tmp_path)
    with pytest.raises(SciFactRunnerError, match="weights SHA-256 mismatch"):
        _run(pack, tmp_path / "bad_hash", model, "0" * 64)
    with pytest.raises(SciFactRunnerError, match="commit revision"):
        run_pack(
            pack,
            tmp_path / "bad_revision",
            nli_model_path=model,
            nli_upstream_model_id=UPSTREAM_MODEL_ID,
            nli_revision="main",
            nli_weights_sha256=weights_sha256,
            scorer=FakeScorer(),
            expected_case_count=2,
        )


def test_reportable_identity_requires_matching_hf_download_metadata(
    tmp_path: Path,
) -> None:
    model, weights_sha256 = _model(tmp_path)
    metadata_dir = model / ".cache" / "huggingface" / "download"
    metadata_dir.mkdir(parents=True)
    metadata = metadata_dir / "model.safetensors.metadata"
    metadata.write_text(
        f"{REVISION}\n{weights_sha256}\n1788441211.1681798\n",
        encoding="utf-8",
    )
    identity = _model_identity(
        model,
        upstream_model_id=UPSTREAM_MODEL_ID,
        revision=REVISION,
        weights_sha256=weights_sha256,
        batch_size=16,
        device="cpu",
        require_hf_metadata=True,
    )
    relative = ".cache/huggingface/download/model.safetensors.metadata"
    assert identity.hf_metadata["model.safetensors"]["revision"] == REVISION
    assert identity.hf_metadata["model.safetensors"]["etag"] == weights_sha256
    assert (
        identity.hf_metadata_sha256[relative]
        == hashlib.sha256(metadata.read_bytes()).hexdigest()
    )

    metadata.write_text(
        f"{'0' * 40}\n{weights_sha256}\n1788441211.1681798\n",
        encoding="utf-8",
    )
    with pytest.raises(SciFactRunnerError, match="metadata revision mismatch"):
        _model_identity(
            model,
            upstream_model_id=UPSTREAM_MODEL_ID,
            revision=REVISION,
            weights_sha256=weights_sha256,
            batch_size=16,
            device="cpu",
            require_hf_metadata=True,
        )
    metadata.write_text(
        f"{REVISION}\n{'0' * 64}\n1788441211.1681798\n",
        encoding="utf-8",
    )
    with pytest.raises(SciFactRunnerError, match="weight etag"):
        _model_identity(
            model,
            upstream_model_id=UPSTREAM_MODEL_ID,
            revision=REVISION,
            weights_sha256=weights_sha256,
            batch_size=16,
            device="cpu",
            require_hf_metadata=True,
        )


def test_sentence_index_rejects_non_structural_or_unindexed_content(
    tmp_path: Path,
) -> None:
    pack = _small_pack(tmp_path)
    model, weights_sha256 = _model(tmp_path)
    path = pack / SENTENCE_INDEX_RELATIVE
    rows = _jsonl(path)
    rows[0]["official_label"] = "SUPPORT"
    _write_jsonl(path, rows)
    with pytest.raises(SciFactRunnerError, match="non-structural"):
        _run(pack, tmp_path / "run", model, weights_sha256)


@pytest.mark.skipif(
    not (PUBLIC_PACK / SENTENCE_INDEX_RELATIVE).is_file(),
    reason="v2 frozen SciFact public-dev pack is unavailable",
)
def test_real_public_pack_runs_all_cases_with_local_nli_accounting(
    tmp_path: Path,
) -> None:
    model, weights_sha256 = _model(tmp_path)
    output = _run(
        PUBLIC_PACK,
        tmp_path / "public_run",
        model,
        weights_sha256,
        scorer=FakeScorer(exact_entailment=False),
        expected_case_count=300,
    )
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    rows = _jsonl(output / "eval_results.jsonl")
    assert manifest["counts"]["cases"] == 300
    assert manifest["counts"]["rows"] == 600
    assert manifest["counts"]["external_provider_calls"] == 0
    assert manifest["counts"]["local_nli_score_many_calls"] == 297
    assert manifest["counts"]["local_nli_pair_count"] == 8195
    assert manifest["counts"]["local_nli_subset_count"] == 8195
    assert manifest["counts"]["local_nli_batch_count"] == 550
    assert len(rows) == 600
    assert {row["baseline"] for row in rows} == set(BASELINES)
    assert {row["predicted_relation"] for row in rows} <= {
        relation.value for relation in SCIFACT_ALLOWED_RELATIONS
    }
