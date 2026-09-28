from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

from evidencetrace.eval.dataset import compute_case_hash
from evidencetrace.eval.models import EvalCase
from evidencetrace.eval.scifact_judge import NliScores

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_PACK = ROOT / "eval_sets" / "scifact_public_dev"
PUBLIC_RESULTS = (
    ROOT / "eval_runs" / "scifact_public_dev_local_nli" / "eval_results.jsonl"
)
SCRIPT = ROOT / "scripts" / "score_scifact_public_dev.py"
SPEC = importlib.util.spec_from_file_location("score_scifact_public_dev", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SCORER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCORER)
RUNNER_SCRIPT = ROOT / "scripts" / "run_scifact_public_dev.py"
RUNNER_SPEC = importlib.util.spec_from_file_location(
    "run_scifact_public_dev_for_scoring_test", RUNNER_SCRIPT
)
assert RUNNER_SPEC is not None and RUNNER_SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(RUNNER_SPEC)
sys.modules[RUNNER_SPEC.name] = RUNNER
RUNNER_SPEC.loader.exec_module(RUNNER)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_results(path: Path, rows: list[dict[str, object]]) -> None:
    _write_jsonl(path, rows)
    counts: dict[str, int] = {}
    for row in rows:
        baseline = str(row["baseline"])
        counts[baseline] = counts.get(baseline, 0) + 1
    manifest = {
        "runner_version": "unit-runner-v1",
        "baselines": list(counts),
        "counts": {
            "cases": len({str(row["case_id"]) for row in rows}),
            "rows": len(rows),
            "rows_by_baseline": counts,
        },
        "outputs": {
            "sha256": {
                "eval_results.jsonl": hashlib.sha256(path.read_bytes()).hexdigest()
            }
        },
    }
    (path.parent / "run_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
    )


def _source(source_id: str, content: str) -> dict[str, object]:
    return {
        "source_id": source_id,
        "url": f"https://example.test/{source_id}",
        "content": content,
        "provenance": "unit fixture",
        "content_hash": hashlib.sha256(content.encode()).hexdigest(),
        "available": True,
    }


def _prediction(
    case_id: str,
    baseline: str,
    relation: str,
    source_id: str,
    evidence: str | None,
    retrieved: list[str],
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "baseline": baseline,
        "predicted_relation": relation,
        "predicted_evidence_span": evidence,
        "retrieved_texts": retrieved,
        "source_id": source_id,
    }


def _sentence_span(
    content: str, text: str, *, doc_id: int, sentence_id: int
) -> dict[str, object]:
    start = content.index(text)
    return {
        "doc_id": doc_id,
        "sentence_id": sentence_id,
        "char_start": start,
        "char_end": start + len(text),
        "locator": f"scifact://document/{doc_id}/sentence/{sentence_id}",
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def _structured_sentence(
    content: str,
    text: str,
    *,
    doc_id: int,
    sentence_id: int,
    rank: int,
    source_id: str = "doc_a",
) -> dict[str, object]:
    start = content.index(text)
    return {
        "rank": rank,
        "source_id": source_id,
        "text": text,
        "locator": f"scifact://document/{doc_id}/sentence/{sentence_id}",
        "char_start": start,
        "char_end": start + len(text),
    }


def _rationale(
    label: str, *, doc_id: int, sentence_ids: list[int]
) -> dict[str, object]:
    return {
        "doc_id": doc_id,
        "label": label,
        "sentence_ids": sentence_ids,
        "sentence_keys": [f"{doc_id}:{sentence_id}" for sentence_id in sentence_ids],
    }


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    pack = tmp_path / "pack"
    alpha = "Alpha supports the claim."
    neutral = "Neutral bridge."
    omega = "Omega also supports it."
    content_a = "\n".join((alpha, neutral, omega))
    content_b = "This abstract contains no supporting information."
    _write_jsonl(
        pack / "sources" / "corpus.jsonl",
        [_source("doc_a", content_a), _source("doc_b", content_b)],
    )
    gold = [
        {
            "case_id": "case_alt",
            "claim_id": 10,
            "official_split": "dev",
            "label": "entailed",
            "cited_doc_ids": [101],
            "sentence_spans": [
                _sentence_span(content_a, alpha, doc_id=101, sentence_id=0),
                _sentence_span(content_a, neutral, doc_id=101, sentence_id=1),
                _sentence_span(content_a, omega, doc_id=101, sentence_id=2),
            ],
            "rationale_sets": [
                _rationale("SUPPORT", doc_id=101, sentence_ids=[0]),
                _rationale("SUPPORT", doc_id=101, sentence_ids=[2]),
            ],
        },
        {
            "case_id": "case_noncontiguous",
            "claim_id": 10,
            "official_split": "dev",
            "label": "contradicted",
            "cited_doc_ids": [101],
            "sentence_spans": [
                _sentence_span(content_a, alpha, doc_id=101, sentence_id=0),
                _sentence_span(content_a, neutral, doc_id=101, sentence_id=1),
                _sentence_span(content_a, omega, doc_id=101, sentence_id=2),
            ],
            "rationale_sets": [
                _rationale("CONTRADICT", doc_id=101, sentence_ids=[0, 2])
            ],
        },
        {
            "case_id": "case_noinfo",
            "claim_id": 11,
            "official_split": "dev",
            "label": "not_in_source",
            "cited_doc_ids": [202],
            "sentence_spans": [
                _sentence_span(content_b, content_b, doc_id=202, sentence_id=0)
            ],
            "rationale_sets": [],
        },
    ]
    _write_jsonl(pack / "gold_rationales.jsonl", gold)
    cases: list[dict[str, object]] = []
    for case_id, claim, source_id, relation, evidence in (
        ("case_alt", "alternative claim", "doc_a", "entailed", alpha),
        (
            "case_noncontiguous",
            "noncontiguous claim",
            "doc_a",
            "contradicted",
            alpha,
        ),
        ("case_noinfo", "unknown claim", "doc_b", "not_in_source", None),
    ):
        case = EvalCase(
            case_id=case_id,
            claim_text=claim,
            source_fixture="sources/corpus.jsonl",
            source_id=source_id,
            source_url=f"https://example.test/{source_id}",
            gold_relation=relation,
            gold_evidence_span=evidence,
            claim_type="scientific_claim",
            mutation_type="official",
            split="dev",
            provenance="unit fixture",
            annotation_status="human_reviewed",
            annotation_notes="unit fixture",
        )
        case = case.model_copy(update={"case_hash": compute_case_hash(case)})
        cases.append(case.model_dump(mode="json"))
    cases_path = pack / "cases.jsonl"
    _write_jsonl(cases_path, cases)
    frozen_path = pack / "cases.frozen_hashes.json"
    frozen_path.write_text(
        json.dumps({row["case_id"]: row["case_hash"] for row in cases}, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    sentence_index_path = pack / "sources" / "sentence_index.jsonl"
    _write_jsonl(
        sentence_index_path,
        [
            {
                "case_id": row["case_id"],
                "source_id": ("doc_b" if row["case_id"] == "case_noinfo" else "doc_a"),
                "documents": [
                    {
                        "doc_id": row["cited_doc_ids"][0],
                        "title": f"Unit document {row['cited_doc_ids'][0]}",
                        "char_start": row["sentence_spans"][0]["char_start"],
                        "char_end": row["sentence_spans"][-1]["char_end"],
                        "sentence_keys": [
                            f"{span['doc_id']}:{span['sentence_id']}"
                            for span in row["sentence_spans"]
                        ],
                    }
                ],
                "sentence_spans": row["sentence_spans"],
            }
            for row in gold
        ],
    )
    manifest_paths = (
        cases_path,
        frozen_path,
        pack / "gold_rationales.jsonl",
        pack / "sources" / "corpus.jsonl",
        sentence_index_path,
    )
    manifest = {
        "schema_version": "scifact-public-dev-v2",
        "mapping": {"prediction_sentence_index": "sources/sentence_index.jsonl"},
        "output_sha256": {
            path.relative_to(pack).as_posix(): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in manifest_paths
        },
    }
    (pack / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
    )
    rows: list[dict[str, object]] = []
    for baseline in (
        "lexical_rules",
        "retrieval_judge_scifact_nli",
        "lexical_full_source",
    ):
        rows.extend(
            [
                _prediction("case_alt", baseline, "entailed", "doc_a", omega, [omega]),
                _prediction(
                    "case_noncontiguous",
                    baseline,
                    "contradicted",
                    "doc_a",
                    content_a,
                    [alpha, neutral, omega],
                ),
                _prediction(
                    "case_noinfo",
                    baseline,
                    "not_in_source",
                    "doc_b",
                    None,
                    [],
                ),
            ]
        )
    results = tmp_path / "eval_results.jsonl"
    _write_results(results, rows)
    return pack, results, tmp_path / "scored"


def test_alternative_rationale_is_not_penalized(tmp_path: Path) -> None:
    pack, results, output = _fixture(tmp_path)
    SCORER.score(pack, results, output, allow_nonreportable=True)
    rows = [
        json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()
    ]
    row = next(
        item
        for item in rows
        if item["baseline"] == "lexical_rules" and item["case_id"] == "case_alt"
    )
    assert row["evidence"]["fully_covered_sentence_keys"] == ["101:2"]
    assert row["evidence"]["best_rationale_index"] == 1
    assert row["evidence"]["best_sentence_set_f1"] == 1.0
    assert row["evidence"]["complete_rationale_coverage"] is True


def test_noncontiguous_rationale_and_ranked_retrieval(tmp_path: Path) -> None:
    pack, results, output = _fixture(tmp_path)
    SCORER.score(pack, results, output, allow_nonreportable=True)
    rows = [
        json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()
    ]
    row = next(
        item
        for item in rows
        if item["baseline"] == "lexical_rules"
        and item["case_id"] == "case_noncontiguous"
    )
    assert row["evidence"]["fully_covered_sentence_keys"] == [
        "101:0",
        "101:1",
        "101:2",
    ]
    assert row["evidence"]["best_sentence_set_f1"] == 0.8
    assert row["evidence"]["complete_rationale_coverage"] is True
    assert row["retrieval"]["overlap_hit_at_1"] is True
    assert row["retrieval"]["full_sentence_hit_at_1"] is True
    assert row["retrieval"]["complete_rationale_full_coverage_at_1"] is False
    assert row["retrieval"]["complete_rationale_full_coverage_at_3"] is True
    assert row["retrieval"]["full_sentence_reciprocal_rank"] == 1.0


def test_structured_plural_evidence_does_not_include_bridge_sentence(
    tmp_path: Path,
) -> None:
    pack, results, output = _fixture(tmp_path)
    rows = [
        json.loads(line) for line in results.read_text(encoding="utf-8").splitlines()
    ]
    content = "Alpha supports the claim.\nNeutral bridge.\nOmega also supports it."
    alpha = "Alpha supports the claim."
    omega = "Omega also supports it."
    for row in rows:
        if row["baseline"] == "lexical_rules" and row["case_id"] == (
            "case_noncontiguous"
        ):
            row["predicted_evidence"] = [
                _structured_sentence(content, alpha, doc_id=101, sentence_id=0, rank=1),
                _structured_sentence(content, omega, doc_id=101, sentence_id=2, rank=3),
            ]
    _write_results(results, rows)

    SCORER.score(pack, results, output, allow_nonreportable=True)
    scored = [
        json.loads(line)
        for line in (output / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    row = next(
        item
        for item in scored
        if item["baseline"] == "lexical_rules"
        and item["case_id"] == "case_noncontiguous"
    )
    assert row["evidence"]["fully_covered_sentence_keys"] == ["101:0", "101:2"]
    assert row["evidence"]["best_sentence_set_f1"] == 1.0
    assert row["evidence"]["complete_rationale_coverage"] is True


def test_structured_offsets_disambiguate_repeated_sentence_text() -> None:
    content = "Echo.\nEcho."
    first = {
        "sentence_key": "7:0",
        "char_start": 0,
        "char_end": 5,
        "locator": "scifact://document/7/sentence/0",
    }
    second = {
        "sentence_key": "8:0",
        "char_start": 6,
        "char_end": 11,
        "locator": "scifact://document/8/sentence/0",
    }
    mapped, overlap, invalid = SCORER._map_structured_sentence(
        content,
        {
            "text": "Echo.",
            "char_start": 6,
            "char_end": 11,
            "locator": "scifact://document/8/sentence/0",
        },
        [first, second],
    )

    assert mapped == overlap == {"8:0"}
    assert invalid is None
    _, _, legacy_invalid = SCORER._map_exact_text(content, "Echo.", [first, second])
    assert legacy_invalid == "multiple_source_occurrences"


def test_metrics_artifacts_majority_and_bootstrap(tmp_path: Path) -> None:
    pack, results, output = _fixture(tmp_path)
    returned = SCORER.score(pack, results, output, allow_nonreportable=True)
    assert returned == output.resolve()
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert {path.name for path in output.iterdir()} == {
        "metrics.json",
        "results.jsonl",
        "report.md",
    }
    assert metrics["baselines"]["majority_class"]["verdict_only"] is True
    assert metrics["baselines"]["majority_class"]["majority_label"] == "SUPPORT"
    assert (
        metrics["baselines"]["lexical_full_source"]["context_budget_comparability"]
        == "not_same_context_budget"
    )
    bootstrap = metrics["paired_bootstrap"]
    assert bootstrap["available"] is True
    assert bootstrap["samples"] == 10_000
    assert bootstrap["seed"] == 20260903
    assert set(bootstrap["effects"]) == {
        "fixed_3way_macro_f1",
        "best_sentence_set_f1",
        "full_sentence_hit_at_5",
    }
    assert metrics["canonical_single_span_metrics_role"] == (
        "compatibility_only_not_headline"
    )
    assert metrics["evaluation_status"] == {
        "reportable": False,
        "mode": "nonreportable_explicit_opt_in",
    }


def test_off_taxonomy_and_ambiguous_span_are_explicit(tmp_path: Path) -> None:
    pack, results, output = _fixture(tmp_path)
    rows = [
        json.loads(line) for line in results.read_text(encoding="utf-8").splitlines()
    ]
    for row in rows:
        if row["baseline"] == "lexical_rules" and row["case_id"] == "case_alt":
            row["predicted_relation"] = "partially_entailed"
            row["predicted_evidence_span"] = "supports"
    _write_results(results, rows)
    SCORER.score(pack, results, output, allow_nonreportable=True)
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    verdict = metrics["baselines"]["lexical_rules"]["verdict"]
    assert verdict["off_taxonomy_count"] == 1
    assert verdict["confusion"]["SUPPORT"]["OFF_TAXONOMY"] == 1
    scored = [
        json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()
    ]
    row = next(
        item
        for item in scored
        if item["baseline"] == "lexical_rules" and item["case_id"] == "case_alt"
    )
    assert row["evidence"]["invalid_reason"] == "multiple_source_occurrences"
    assert row["evidence"]["best_sentence_set_f1"] == 0.0


def test_noinfo_without_evidence_is_abstention_not_malformed_span(
    tmp_path: Path,
) -> None:
    pack, results, output = _fixture(tmp_path)
    rows = [
        json.loads(line) for line in results.read_text(encoding="utf-8").splitlines()
    ]
    for row in rows:
        if row["baseline"] == "lexical_rules" and row["case_id"] == "case_alt":
            row["predicted_relation"] = "not_in_source"
            row["predicted_evidence_span"] = None
    _write_results(results, rows)

    SCORER.score(pack, results, output, allow_nonreportable=True)
    scored = [
        json.loads(line)
        for line in (output / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    row = next(
        item
        for item in scored
        if item["baseline"] == "lexical_rules" and item["case_id"] == "case_alt"
    )
    assert row["evidence"]["abstained"] is True
    assert row["evidence"]["invalid_reason"] is None
    assert row["evidence"]["best_sentence_set_f1"] == 0.0


def test_two_baseline_path_and_partial_overlap_are_strict(tmp_path: Path) -> None:
    pack, results, output = _fixture(tmp_path)
    rows = [
        json.loads(line) for line in results.read_text(encoding="utf-8").splitlines()
    ]
    rows = [row for row in rows if row["baseline"] != "lexical_full_source"]
    for row in rows:
        if row["case_id"] == "case_alt":
            row["predicted_evidence_span"] = "Omega also"
            row["retrieved_texts"] = ["Omega also"]
    _write_results(results, rows)

    SCORER.score(pack, results, output, allow_nonreportable=True)
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert set(metrics["baselines"]) == {
        "lexical_rules",
        "retrieval_judge_scifact_nli",
        "majority_class",
    }
    assert metrics["paired_bootstrap"]["available"] is True
    scored = [
        json.loads(line)
        for line in (output / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    row = next(
        item
        for item in scored
        if item["baseline"] == "lexical_rules" and item["case_id"] == "case_alt"
    )
    assert row["evidence"]["fully_covered_sentence_keys"] == []
    assert row["evidence"]["overlapped_sentence_keys_diagnostic"] == ["101:2"]
    assert row["evidence"]["best_sentence_set_f1"] == 0.0
    assert row["evidence"]["complete_rationale_coverage"] is False
    assert row["retrieval"]["overlap_hit_at_1"] is True
    assert row["retrieval"]["full_sentence_hit_at_1"] is False
    assert row["retrieval"]["complete_rationale_full_coverage_at_1"] is False


def test_manifest_hash_tampering_fails_before_output(tmp_path: Path) -> None:
    pack, results, output = _fixture(tmp_path)
    with (pack / "gold_rationales.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("\n")

    with pytest.raises(SCORER.ScoringError, match="manifest output hash mismatch"):
        SCORER.score(pack, results, output, allow_nonreportable=True)
    assert not output.exists()


def test_result_case_source_binding_mismatch_fails(tmp_path: Path) -> None:
    pack, results, output = _fixture(tmp_path)
    rows = [
        json.loads(line) for line in results.read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["source_id"] = "doc_b"
    _write_results(results, rows)

    with pytest.raises(SCORER.ScoringError, match=r"disagrees with cases\.jsonl"):
        SCORER.score(pack, results, output, allow_nonreportable=True)
    assert not output.exists()


def test_nonempty_output_is_preserved(tmp_path: Path) -> None:
    pack, results, output = _fixture(tmp_path)
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("do not overwrite", encoding="utf-8")

    with pytest.raises(SCORER.ScoringError, match="not empty"):
        SCORER.score(pack, results, output, allow_nonreportable=True)
    assert sentinel.read_text(encoding="utf-8") == "do not overwrite"
    assert list(output.iterdir()) == [sentinel]


def test_runner_to_scorer_small_pack_e2e(tmp_path: Path) -> None:
    pack, _, output = _fixture(tmp_path)
    run_dir = tmp_path / "real-runner-output"
    model = tmp_path / "unit-nli"
    model.mkdir()
    weights = model / "model.safetensors"
    weights.write_bytes(b"unit-test-only-weights")

    class NeutralScorer:
        model_id = "cross-encoder/nli-MiniLM2-L6-H768"
        revision = "b95119ce93d3e065de6214e38cd4a97b0f2f2c6d"
        batch_size = 16

        def score_many(self, pairs: Sequence[tuple[str, str]]) -> Sequence[NliScores]:
            return tuple(
                NliScores(entailment=0.05, contradiction=0.05, neutral=0.90)
                for _ in pairs
            )

    RUNNER.run_pack(
        pack,
        run_dir,
        nli_model_path=model,
        nli_upstream_model_id=NeutralScorer.model_id,
        nli_revision=NeutralScorer.revision,
        nli_weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
        scorer=NeutralScorer(),
        expected_case_count=3,
    )

    SCORER.score(
        pack,
        run_dir / "eval_results.jsonl",
        output,
        allow_nonreportable=True,
    )
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert (
        metrics["inputs"]["run_manifest_validation"]
        == "nonreportable_basic_integrity_only"
    )
    assert metrics["counts"]["same_retriever_verified_case_count"] == 3
    assert metrics["run_manifest"]["runner_version"] == RUNNER.RUNNER_VERSION


@pytest.mark.skipif(
    not PUBLIC_RESULTS.is_file(),
    reason="reportable SciFact local-NLI artifact is unavailable",
)
def test_real_reportable_artifacts_pass_default_strict_score(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The contract validation is independent of bootstrap sample count; keeping
    # this small makes the complete 300-case regression test fast.
    monkeypatch.setattr(SCORER, "BOOTSTRAP_SAMPLES", 32)
    output = tmp_path / "strict-score"

    SCORER.score(PUBLIC_PACK, PUBLIC_RESULTS, output)

    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["evaluation_status"] == {
        "mode": "local_transformers_nli_reportable",
        "reportable": True,
    }
    assert metrics["inputs"]["run_manifest_validation"] == (
        "reportable_v2_contract_verified"
    )
    assert metrics["counts"]["case_count"] == 300
    assert metrics["counts"]["same_retriever_verified_case_count"] == 300
