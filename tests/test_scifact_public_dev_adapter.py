from __future__ import annotations

import hashlib
import json
from itertools import pairwise
from pathlib import Path

from evidencetrace.eval.dataset import frozen_hashes, load_dataset
from scripts.build_scifact_public_dev import SENTENCE_INDEX_RELATIVE, build_pack

REPO = Path(__file__).resolve().parents[1]
RAW = REPO.parent / "env" / "datasets" / "scifact" / "raw" / "data"
PACK = REPO / "eval_sets" / "scifact_public_dev"


def _jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_pack_loads_as_complete_public_dev() -> None:
    loaded = load_dataset(PACK / "cases.jsonl")
    assert len(loaded.cases) == 300
    assert len(loaded.sources) == 300
    assert len({case.case_id for case in loaded.cases}) == 300
    assert len({case.source_id for case in loaded.cases}) == 300
    assert all(case.split == "dev" for case in loaded.cases)
    assert all(case.annotation_status == "human_reviewed" for case in loaded.cases)
    assert {case.gold_relation.value for case in loaded.cases} == {
        "entailed",
        "contradicted",
        "not_in_source",
    }


def test_labels_freezes_and_one_source_per_claim() -> None:
    loaded = load_dataset(PACK / "cases.jsonl")
    counts: dict[str, int] = {}
    for case in loaded.cases:
        counts[case.gold_relation.value] = counts.get(case.gold_relation.value, 0) + 1
        source = loaded.sources[case.source_id]
        assert case.source_sha256 == source.content_hash
        assert (
            case.gold_evidence_span is None or case.gold_evidence_span in source.content
        )
    assert counts == {"not_in_source": 112, "entailed": 124, "contradicted": 64}
    expected = json.loads(
        (PACK / "cases.frozen_hashes.json").read_text(encoding="utf-8")
    )
    assert expected == frozen_hashes(loaded.cases)


def test_sidecar_preserves_all_rationale_alternatives_and_exact_spans() -> None:
    loaded = load_dataset(PACK / "cases.jsonl")
    rows = _jsonl(PACK / "gold_rationales.jsonl")
    assert len(rows) == 300
    assert sum(len(row["rationale_sets"]) for row in rows) == 338
    assert sum(len(row["rationale_sets"]) > 1 for row in rows) == 83
    assert (
        sum(len({item["doc_id"] for item in row["rationale_sets"]}) > 1 for row in rows)
        == 10
    )
    noncontiguous = 0
    for row in rows:
        case = next(item for item in loaded.cases if item.case_id == row["case_id"])
        assert row["label"] == case.gold_relation.value
        source = loaded.sources[case.source_id].content
        keys = set()
        for span in row["sentence_spans"]:
            text = source[span["char_start"] : span["char_end"]]
            assert (
                hashlib.sha256(text.encode("utf-8")).hexdigest() == span["text_sha256"]
            )
            assert span["locator"] == (
                f"scifact://document/{span['doc_id']}/sentence/{span['sentence_id']}"
            )
            keys.add(f"{span['doc_id']}:{span['sentence_id']}")
        for rationale in row["rationale_sets"]:
            assert rationale["sentence_keys"] == [
                f"{rationale['doc_id']}:{sentence_id}"
                for sentence_id in rationale["sentence_ids"]
            ]
            assert set(rationale["sentence_keys"]) <= keys
            ids = rationale["sentence_ids"]
            noncontiguous += len(ids) > 1 and any(
                right - left != 1 for left, right in pairwise(ids)
            )
        if row["rationale_sets"]:
            first = row["rationale_sets"][0]
            span = next(
                item
                for item in row["sentence_spans"]
                if item["doc_id"] == first["doc_id"]
                and item["sentence_id"] == first["sentence_ids"][0]
            )
            assert (
                case.gold_evidence_span == source[span["char_start"] : span["char_end"]]
            )
        else:
            assert case.gold_evidence_span is None
    assert noncontiguous == 9


def test_source_format_and_original_cited_order() -> None:
    loaded = load_dataset(PACK / "cases.jsonl")
    index_by_case = {
        row["case_id"]: row for row in _jsonl(PACK / SENTENCE_INDEX_RELATIVE)
    }
    duplicate_rows = []
    for row in _jsonl(PACK / "gold_rationales.jsonl"):
        source_id = f"scifact_dev_source_{row['claim_id']:06d}"
        source = loaded.sources[source_id].content
        index = index_by_case[row["case_id"]]
        assert index["source_id"] == source_id
        assert [item["doc_id"] for item in index["documents"]] == row["cited_doc_ids"]
        assert index["sentence_spans"] == row["sentence_spans"]
        assert row["cited_doc_ids"] == list(dict.fromkeys(row["raw_cited_doc_ids"]))
        assert len(
            {
                f"{span['doc_id']}:{span['sentence_id']}"
                for span in row["sentence_spans"]
            }
        ) == len(row["sentence_spans"])
        if row["raw_cited_doc_ids"] != row["cited_doc_ids"]:
            duplicate_rows.append(row)
        assert "[DOC " not in source
        assert "[SENT " not in source
        assert "Title: " not in source
        covered = [False] * len(source)
        for span in index["sentence_spans"]:
            for offset in range(span["char_start"], span["char_end"]):
                covered[offset] = True
        assert all(
            is_covered or character == "\n"
            for is_covered, character in zip(covered, source, strict=True)
        )
        for document in index["documents"]:
            assert document["title"].strip()
            document_spans = [
                span
                for span in index["sentence_spans"]
                if span["doc_id"] == document["doc_id"]
            ]
            assert document["char_start"] == document_spans[0]["char_start"]
            assert document["char_end"] == document_spans[-1]["char_end"]
            assert document["sentence_keys"] == [
                f"{span['doc_id']}:{span['sentence_id']}" for span in document_spans
            ]
    assert [
        (row["claim_id"], row["raw_cited_doc_ids"], row["cited_doc_ids"])
        for row in duplicate_rows
    ] == [(1245, [7662395, 7662395], [7662395])]


def test_manifest_hashes_and_limitations() -> None:
    manifest = json.loads((PACK / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["raw_counts"] == {
        "corpus": 5183,
        "dev": 300,
        "test": 300,
        "train": 809,
    }
    assert manifest["adapter_counts"]["labels"] == {
        "contradicted": 64,
        "entailed": 124,
        "not_in_source": 112,
    }
    assert manifest["adapter_counts"]["multi_evidence_document_claims"] == 10
    assert manifest["adapter_counts"]["multi_rationale_claims"] == 83
    assert manifest["adapter_counts"]["noncontiguous_rationales"] == 9
    assert manifest["adapter_counts"]["duplicate_cited_doc_claim_count"] == 1
    assert manifest["adapter_counts"]["duplicate_cited_doc_claims"] == [
        {
            "claim_id": 1245,
            "raw_cited_doc_ids": [7662395, 7662395],
            "distinct_cited_doc_ids": [7662395],
        }
    ]
    assert manifest["schema_version"] == "scifact-public-dev-v2"
    assert manifest["mapping"]["prediction_sentence_index"] == (SENTENCE_INDEX_RELATIVE)
    for relative, digest in manifest["output_sha256"].items():
        assert _sha256(PACK / relative) == digest
    assert "manifest.json" not in manifest["output_sha256"]
    assert manifest["license_notice"]["copied_verbatim"] is True
    assert manifest["license_notice"]["raw_sha256"] == _sha256(
        RAW.parent / "LICENSE.md"
    )
    assert (PACK / "SCIFACT-LICENSE.md").read_bytes() == (
        RAW.parent / "LICENSE.md"
    ).read_bytes()
    assert "not a blind holdout" in " ".join(manifest["limitations"])
    assert "oracle corpus" in " ".join(manifest["limitations"])


def test_builder_is_deterministic_and_self_validates(tmp_path: Path) -> None:
    rebuilt = tmp_path / "pack"
    manifest = build_pack(RAW, rebuilt)
    load_dataset(rebuilt / "cases.jsonl")
    for relative, digest in manifest["output_sha256"].items():
        assert _sha256(rebuilt / relative) == digest
        assert digest == _sha256(PACK / relative)
