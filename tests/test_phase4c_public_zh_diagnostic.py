from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from evidencetrace.eval.baselines import build_evidence_chunks
from evidencetrace.eval.dataset import frozen_hashes, load_dataset
from evidencetrace.models import Relation
from evidencetrace.retrieval.rank import (
    EVIDENCE_EXACT_MAX_BYTES,
    EVIDENCE_EXACT_MAX_CHARS,
    FULL_CONTEXT_FALLBACK_SCORE,
    LexicalRetriever,
)


ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "eval_sets" / "phase4c_public_zh_diagnostic"
DATASET = PACK / "holdout_public_zh.jsonl"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _literal_rank(results: tuple[object, ...], span: str) -> int | None:
    for index, result in enumerate(results, 1):
        text = str(getattr(result, "text"))
        if span in text or text in span:
            return index
    return None


def test_phase4c_pack_is_frozen_and_structurally_valid() -> None:
    loaded = load_dataset(DATASET)
    assert len(loaded.cases) == 32
    assert len(loaded.dev_cases) == 16
    assert len(loaded.test_cases) == 16
    assert len(loaded.sources) == 4
    assert all(source.available for source in loaded.sources.values())
    assert all(case.annotation_status == "deterministic_gold" for case in loaded.cases)
    assert Relation.SOURCE_UNAVAILABLE not in {case.gold_relation for case in loaded.cases}

    expected = json.loads(
        (PACK / "holdout_public_zh.frozen_hashes.json").read_text(encoding="utf-8")
    )
    assert set(expected) == {case.case_id for case in loaded.cases}
    assert expected == frozen_hashes(loaded.cases)
    assert all(
        case.gold_evidence_span is None
        or case.gold_evidence_span in loaded.sources[case.source_id].content
        for case in loaded.cases
    )


def test_phase4c_test_split_has_preregistered_substantive_coverage() -> None:
    loaded = load_dataset(DATASET)
    support = Counter(case.gold_relation for case in loaded.test_cases)
    assert support[Relation.ENTAILED] == 4
    assert support[Relation.PARTIALLY_ENTAILED] == 3
    assert support[Relation.CONTRADICTED] == 4
    assert all(
        support[relation] >= 3
        for relation in (
            Relation.ENTAILED,
            Relation.PARTIALLY_ENTAILED,
            Relation.CONTRADICTED,
        )
    )


def test_phase4c_manifest_hashes_and_leakage_audit_are_consistent() -> None:
    manifest = json.loads((PACK / "manifest.json").read_text(encoding="utf-8"))
    audit = json.loads((PACK / "leakage_audit.json").read_text(encoding="utf-8"))
    assert manifest["benchmark_validity"] == "diagnostic_contaminated"
    assert manifest["public_benchmark_eligible"] is False
    assert manifest["case_count"] == 32
    assert manifest["source_count"] == 4
    assert manifest["files"]["holdout_public_zh.jsonl"] == _sha256(DATASET)
    assert manifest["files"]["sources/source_snapshots.jsonl"] == _sha256(
        PACK / "sources" / "source_snapshots.jsonl"
    )
    assert manifest["files"]["holdout_public_zh.frozen_hashes.json"] == _sha256(
        PACK / "holdout_public_zh.frozen_hashes.json"
    )
    assert audit["result"] == "pass"
    assert audit["selection_used_labels"] is True
    assert audit["selection_used_predictions_or_metrics"] is False
    assert audit["old_case_id_overlap"] == []
    assert audit["old_claim_overlap"] == []
    assert audit["old_source_url_overlap"] == []
    assert audit["old_source_hash_overlap"] == []


def test_phase4c_cjk_literal_top1_and_bounded_empty_fallback() -> None:
    loaded = load_dataset(DATASET)
    substantive_test = [
        case for case in loaded.test_cases if case.gold_evidence_span is not None
    ]
    assert len(substantive_test) == 11
    for case in substantive_test:
        source = loaded.sources[case.source_id]
        chunks = build_evidence_chunks(source)
        results = LexicalRetriever(chunks, neighbor_window=0).search(
            case.claim_text or "",
            top_k=5,
            fallback_to_full_context=False,
        )
        assert _literal_rank(results, case.gold_evidence_span or "") == 1

    for source in loaded.sources.values():
        chunks = build_evidence_chunks(source)
        assert len(chunks) > 1
        retriever = LexicalRetriever(chunks, neighbor_window=0)
        assert (
            retriever.search(
                "量子纠缠熵", top_k=5, fallback_to_full_context=False
            )
            == ()
        )
        fallback = retriever.search(
            "量子纠缠熵", top_k=5, fallback_to_full_context=True
        )
        assert len(fallback) == 1
        assert fallback[0].score == FULL_CONTEXT_FALLBACK_SCORE
        assert fallback[0].text == source.content
        assert fallback[0].chunk.char_start == 0
        assert fallback[0].chunk.char_end == len(source.content)
        assert len(fallback[0].text) <= EVIDENCE_EXACT_MAX_CHARS
        assert len(fallback[0].text.encode("utf-8")) <= EVIDENCE_EXACT_MAX_BYTES
