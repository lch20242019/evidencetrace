"""Build a frozen one-claim/one-source adapter for the official SciFact dev set.

The adapter deliberately preserves the official claim boundary.  Every source
contains all distinct abstracts named by ``cited_doc_ids`` in their original
order (the raw list is retained separately for audit);
evidence documents and rationale alternatives are never expanded into extra
evaluation cases.  ``cases.jsonl`` keeps one legacy-compatible evidence span,
while ``gold_rationales.jsonl`` is the authoritative evidence annotation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Iterable
from itertools import pairwise
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from evidencetrace.eval.dataset import (  # noqa: E402
    compute_case_hash,
    frozen_hashes,
    load_dataset,
)
from evidencetrace.eval.models import EvalCase, SourceFixture  # noqa: E402
from evidencetrace.models import Relation  # noqa: E402

DEFAULT_PACK = REPO_ROOT / "eval_sets" / "scifact_public_dev"
CASES_NAME = "cases.jsonl"
FROZEN_NAME = "cases.frozen_hashes.json"
SOURCES_RELATIVE = "sources/source_snapshots.jsonl"
SENTENCE_INDEX_RELATIVE = "sources/sentence_index.jsonl"
RATIONALES_NAME = "gold_rationales.jsonl"
OFFICIAL_REPOSITORY_URL = "https://github.com/allenai/scifact"
OFFICIAL_DATA_URL = (
    "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
)
OFFICIAL_LICENSE_URL = "https://github.com/allenai/scifact/blob/master/LICENSE.md"
EXPECTED_COUNTS = {"train": 809, "dev": 300, "test": 300, "corpus": 5183}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _json_line(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _write_jsonl(path: Path, rows: Iterable[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(_json_line(row))
            handle.write("\n")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"required SciFact file does not exist: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected JSON object at {path}:{line_number}")
        rows.append(value)
    return rows


def _source_length_stratum(content: str) -> str:
    size = len(content.encode("utf-8"))
    return "short" if size < 1200 else "medium" if size < 3000 else "long"


def _stable_unique(values: Iterable[int]) -> list[int]:
    """Remove exact duplicate IDs without changing first-occurrence order."""

    return list(dict.fromkeys(values))


def _render_source(
    cited_doc_ids: list[int], corpus: dict[int, dict[str, Any]]
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    """Render semantic abstract text plus out-of-band sentence locators.

    Document IDs, sentence IDs, and titles are deliberately excluded from the
    text searched by the baselines. They remain available in the sidecar so a
    character span can still be traced back to the official SciFact record.
    """

    parts: list[str] = []
    spans: list[dict[str, Any]] = []
    documents: list[dict[str, Any]] = []
    cursor = 0

    def append(value: str) -> None:
        nonlocal cursor
        parts.append(value)
        cursor += len(value)

    for doc_id in cited_doc_ids:
        document = corpus.get(doc_id)
        if document is None:
            raise ValueError(f"cited doc {doc_id} is absent from corpus.jsonl")
        title = document.get("title")
        abstract = document.get("abstract")
        if not isinstance(title, str) or not title.strip():
            raise ValueError(f"doc {doc_id} has an invalid title")
        if not isinstance(abstract, list) or not all(
            isinstance(sentence, str) and sentence for sentence in abstract
        ):
            raise ValueError(f"doc {doc_id} has an invalid abstract")
        document_start = cursor + (1 if parts else 0)
        sentence_keys: list[str] = []
        for sentence_id, sentence in enumerate(abstract):
            if parts:
                append("\n")
            char_start = cursor
            append(sentence)
            char_end = cursor
            locator = f"scifact://document/{doc_id}/sentence/{sentence_id}"
            sentence_key = f"{doc_id}:{sentence_id}"
            sentence_keys.append(sentence_key)
            spans.append(
                {
                    "doc_id": doc_id,
                    "sentence_id": sentence_id,
                    "char_start": char_start,
                    "char_end": char_end,
                    "locator": locator,
                    "text_sha256": _sha256_text(sentence),
                }
            )
        documents.append(
            {
                "doc_id": doc_id,
                "title": title,
                "char_start": document_start,
                "char_end": cursor,
                "sentence_keys": sentence_keys,
            }
        )
    return "".join(parts), spans, documents


def _sorted_rationale_sets(
    evidence: dict[str, Any],
    *,
    corpus: dict[int, dict[str, Any]],
    cited_doc_ids: list[int],
) -> tuple[list[dict[str, Any]], set[str]]:
    cited = set(cited_doc_ids)
    rationale_sets: list[dict[str, Any]] = []
    labels: set[str] = set()
    for raw_doc_id, raw_rationales in evidence.items():
        doc_id = int(raw_doc_id)
        if doc_id not in cited:
            raise ValueError(f"evidence doc {doc_id} is absent from cited_doc_ids")
        if not isinstance(raw_rationales, list) or not raw_rationales:
            raise ValueError(f"evidence doc {doc_id} has no rationale")
        abstract = corpus[doc_id]["abstract"]
        for rationale in raw_rationales:
            label = str(rationale.get("label", ""))
            if label not in {"SUPPORT", "CONTRADICT"}:
                raise ValueError(f"unsupported SciFact rationale label: {label!r}")
            sentence_ids = [int(value) for value in rationale.get("sentences", [])]
            if not sentence_ids or sentence_ids != sorted(set(sentence_ids)):
                raise ValueError(
                    f"rationale sentence IDs must be non-empty, unique, and sorted: "
                    f"claim evidence doc {doc_id}"
                )
            if any(value < 0 or value >= len(abstract) for value in sentence_ids):
                raise ValueError(
                    f"rationale sentence ID out of bounds for doc {doc_id}"
                )
            labels.add(label)
            rationale_sets.append(
                {
                    "doc_id": doc_id,
                    "label": label,
                    "sentence_ids": sentence_ids,
                    "sentence_keys": [f"{doc_id}:{value}" for value in sentence_ids],
                }
            )
    rationale_sets.sort(key=lambda item: (item["doc_id"], item["sentence_ids"]))
    return rationale_sets, labels


def _relation(evidence: dict[str, Any], labels: set[str]) -> Relation:
    if not evidence:
        if labels:
            raise ValueError("empty SciFact evidence unexpectedly has labels")
        return Relation.NOT_IN_SOURCE
    if len(labels) != 1:
        raise ValueError(f"SciFact claim has inconsistent rationale labels: {labels}")
    label = next(iter(labels))
    return Relation.ENTAILED if label == "SUPPORT" else Relation.CONTRADICTED


def _build_rows(
    dev_rows: list[dict[str, Any]], corpus: dict[int, dict[str, Any]]
) -> tuple[
    list[EvalCase],
    list[SourceFixture],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    cases: list[EvalCase] = []
    sources: list[SourceFixture] = []
    sidecar: list[dict[str, Any]] = []
    sentence_index: list[dict[str, Any]] = []
    label_counts: Counter[str] = Counter()
    raw_cited_counts: Counter[int] = Counter()
    distinct_cited_counts: Counter[int] = Counter()
    duplicate_cited_doc_claims: list[dict[str, Any]] = []
    multi_evidence_docs = 0
    multi_rationale_claims = 0
    noncontiguous_rationales = 0

    for raw in dev_rows:
        claim_id = int(raw["id"])
        claim = str(raw["claim"])
        raw_cited_doc_ids = [int(value) for value in raw.get("cited_doc_ids", [])]
        if not raw_cited_doc_ids:
            raise ValueError(f"claim {claim_id} must have cited_doc_ids")
        cited_doc_ids = _stable_unique(raw_cited_doc_ids)
        if cited_doc_ids != raw_cited_doc_ids:
            duplicate_cited_doc_claims.append(
                {
                    "claim_id": claim_id,
                    "raw_cited_doc_ids": raw_cited_doc_ids,
                    "distinct_cited_doc_ids": cited_doc_ids,
                }
            )
        evidence = raw.get("evidence")
        if not isinstance(evidence, dict):
            raise ValueError(f"claim {claim_id} has invalid evidence")

        content, sentence_spans, documents = _render_source(cited_doc_ids, corpus)
        rationale_sets, labels = _sorted_rationale_sets(
            evidence, corpus=corpus, cited_doc_ids=cited_doc_ids
        )
        relation = _relation(evidence, labels)
        label_counts[relation.value] += 1
        raw_cited_counts[len(raw_cited_doc_ids)] += 1
        distinct_cited_counts[len(cited_doc_ids)] += 1
        multi_evidence_docs += len(evidence) > 1
        multi_rationale_claims += len(rationale_sets) > 1
        noncontiguous_rationales += sum(
            len(item["sentence_ids"]) > 1
            and any(right - left != 1 for left, right in pairwise(item["sentence_ids"]))
            for item in rationale_sets
        )

        gold_evidence_span = None
        if rationale_sets:
            first = rationale_sets[0]
            gold_evidence_span = corpus[first["doc_id"]]["abstract"][
                first["sentence_ids"][0]
            ]

        case_id = f"scifact_dev_{claim_id:06d}"
        source_id = f"scifact_dev_source_{claim_id:06d}"
        content_hash = _sha256_text(content)
        source = SourceFixture(
            source_id=source_id,
            url=OFFICIAL_REPOSITORY_URL,
            content=content,
            provenance=(
                "Official SciFact corpus abstracts, concatenated in the original "
                f"cited_doc_ids order for public dev claim {claim_id}."
            ),
            content_hash=content_hash,
            available=True,
        )
        case = EvalCase(
            case_id=case_id,
            document_path=None,
            claim_text=claim,
            claim_line=1,
            source_fixture=SOURCES_RELATIVE,
            source_id=source_id,
            source_url=OFFICIAL_REPOSITORY_URL,
            gold_relation=relation,
            gold_evidence_span=gold_evidence_span,
            claim_type="official_scifact_scientific_claim",
            mutation_type="official_human_authored_claim",
            split="dev",
            provenance=(
                "Official SciFact public dev annotation adapted without changing "
                "the claim boundary; cited abstracts are provided as an oracle corpus."
            ),
            annotation_status="human_reviewed",
            annotation_notes=(
                "Public SciFact dev, not a blind holdout. The official human-reviewed "
                "verdict and all rationale alternatives are preserved in "
                "gold_rationales.jsonl; gold_evidence_span is compatibility-only."
            ),
            source_sha256=content_hash,
            source_length_stratum=_source_length_stratum(content),
        )
        case = case.model_copy(update={"case_hash": compute_case_hash(case)})
        cases.append(case)
        sources.append(source)
        sidecar.append(
            {
                "case_id": case_id,
                "claim_id": claim_id,
                "official_split": "dev",
                "label": relation.value,
                "raw_cited_doc_ids": raw_cited_doc_ids,
                "cited_doc_ids": cited_doc_ids,
                "documents": documents,
                "sentence_spans": sentence_spans,
                "rationale_sets": rationale_sets,
            }
        )
        sentence_index.append(
            {
                "case_id": case_id,
                "source_id": source_id,
                "documents": documents,
                "sentence_spans": sentence_spans,
            }
        )

    stats = {
        "labels": dict(sorted(label_counts.items())),
        "raw_cited_doc_entries_per_claim": {
            str(key): value for key, value in sorted(raw_cited_counts.items())
        },
        "distinct_cited_docs_per_claim": {
            str(key): value for key, value in sorted(distinct_cited_counts.items())
        },
        "duplicate_cited_doc_claim_count": len(duplicate_cited_doc_claims),
        "duplicate_cited_doc_claims": duplicate_cited_doc_claims,
        "multi_evidence_document_claims": multi_evidence_docs,
        "multi_rationale_claims": multi_rationale_claims,
        "noncontiguous_rationales": noncontiguous_rationales,
        "total_rationale_sets": sum(len(row["rationale_sets"]) for row in sidecar),
    }
    return cases, sources, sidecar, sentence_index, stats


def _readme() -> str:
    return """# SciFact public dev adapter

This pack adapts the complete official SciFact development split into the
EvidenceTrace one-claim/one-source contract. It contains 300 public development
claims and is **not a blind holdout**.

Each case keeps the original claim intact and concatenates every distinct
abstract in the claim's `cited_doc_ids`, in first-occurrence order. The raw list
is also retained in `gold_rationales.jsonl`: official dev claim 1245 repeats
document 7662395, so the rendered source de-duplicates that exact ID to avoid
duplicate sentence identities. This is an **oracle cited abstract** setting:
document retrieval from the full 5,183-abstract corpus is not evaluated by this
pack. The semantic source contains only official abstract sentences separated
by newlines. Synthetic document, sentence, and title prefixes are not inserted
into searchable text. Identity and title metadata remain out-of-band:

```text
documents: [{doc_id, title, char_start, char_end, sentence_keys}]
sentence_spans: [{doc_id, sentence_id, char_start, char_end, locator, text_sha256}]
```

Official `SUPPORT`, `CONTRADICT`, and empty-evidence annotations map to
`entailed`, `contradicted`, and `not_in_source`. SciFact does not provide
EvidenceTrace's `partially_entailed` or `not_checkable` classes. For NOINFO,
`not_in_source` means no annotated evidence in the cited abstracts; it does not
mean that the claim is uncheckable in the world.

`cases.jsonl` stores one compatibility evidence sentence. The authoritative
gold annotation is `gold_rationales.jsonl`: each `rationale_sets` item is one
alternative evidence set (outer OR), while every sentence inside that set is
jointly required (inner AND). Non-contiguous rationale sentences remain
separate sentence keys. Its top-level `label` is the mapped EvidenceTrace
relation; every rationale set also retains the official SciFact label.

Files:

- `cases.jsonl`: EvidenceTrace cases, all `split=dev` and
  `annotation_status=human_reviewed`.
- `sources/source_snapshots.jsonl`: one source fixture per claim.
- `sources/sentence_index.jsonl`: label-free document/sentence identities,
  locators, and exact source offsets for prediction-time chunk construction.
- `gold_rationales.jsonl`: all official rationale alternatives and exact source
  sentence offsets.
- `cases.frozen_hashes.json`: case freeze hashes consumed by `load_dataset`.
- `manifest.json`: raw/output hashes, counts, mapping, and limitations.
- `SCIFACT-LICENSE.md`: upstream licensing notice.

Rebuild from an extracted official release:

```powershell
python scripts/build_scifact_public_dev.py --raw-dir <official-data-directory>
```

Run the two same-retriever deterministic paths and the strict custom scorer:

```powershell
python scripts/run_scifact_public_dev.py `
  --pack eval_sets/scifact_public_dev `
  --out eval_runs/scifact_public_dev_deterministic
python scripts/score_scifact_public_dev.py `
  --pack eval_sets/scifact_public_dev `
  --eval-results eval_runs/scifact_public_dev_deterministic/eval_results.jsonl `
  --out eval_runs/scifact_public_dev_deterministic/scored
```

These are EvidenceTrace diagnostic metrics on public dev data, not the official
SciFact leaderboard protocol. Both deterministic paths share the same lexical
retriever, chunks, and top-5 budget, so their retrieval scores are identical by
construction; the comparison isolates their judgment rules only.
"""


def build_pack(raw_dir: Path, output_pack: Path = DEFAULT_PACK) -> dict[str, Any]:
    raw_dir = raw_dir.resolve()
    output_pack = output_pack.resolve()
    upstream_license_path = raw_dir.parent / "LICENSE.md"
    if not upstream_license_path.is_file():
        raise FileNotFoundError(
            "official SciFact LICENSE.md must exist next to the data directory: "
            f"{upstream_license_path}"
        )
    upstream_license = upstream_license_path.read_bytes()
    if not upstream_license:
        raise ValueError(f"official SciFact license is empty: {upstream_license_path}")
    raw_paths = {
        "claims_train.jsonl": raw_dir / "claims_train.jsonl",
        "claims_dev.jsonl": raw_dir / "claims_dev.jsonl",
        "claims_test.jsonl": raw_dir / "claims_test.jsonl",
        "corpus.jsonl": raw_dir / "corpus.jsonl",
    }
    raw_rows = {name: _read_jsonl(path) for name, path in raw_paths.items()}
    observed_counts = {
        "train": len(raw_rows["claims_train.jsonl"]),
        "dev": len(raw_rows["claims_dev.jsonl"]),
        "test": len(raw_rows["claims_test.jsonl"]),
        "corpus": len(raw_rows["corpus.jsonl"]),
    }
    if observed_counts != EXPECTED_COUNTS:
        raise ValueError(
            f"unexpected SciFact release counts: {observed_counts}; "
            f"expected {EXPECTED_COUNTS}"
        )
    if any(set(row) != {"id", "claim"} for row in raw_rows["claims_test.jsonl"]):
        raise ValueError("official test is expected to contain only hidden-gold claims")

    corpus: dict[int, dict[str, Any]] = {}
    for row in raw_rows["corpus.jsonl"]:
        doc_id = int(row["doc_id"])
        if doc_id in corpus:
            raise ValueError(f"duplicate corpus doc_id: {doc_id}")
        corpus[doc_id] = row

    cases, sources, sidecar, sentence_index, dev_stats = _build_rows(
        raw_rows["claims_dev.jsonl"], corpus
    )
    if len(cases) != EXPECTED_COUNTS["dev"] or len(sources) != len(cases):
        raise AssertionError(
            "adapter must emit exactly one case and source per dev claim"
        )

    cases_path = output_pack / CASES_NAME
    sources_path = output_pack / SOURCES_RELATIVE
    sentence_index_path = output_pack / SENTENCE_INDEX_RELATIVE
    frozen_path = output_pack / FROZEN_NAME
    rationales_path = output_pack / RATIONALES_NAME
    readme_path = output_pack / "README.md"
    license_path = output_pack / "SCIFACT-LICENSE.md"
    manifest_path = output_pack / "manifest.json"

    _write_jsonl(cases_path, (case.model_dump(mode="json") for case in cases))
    _write_jsonl(sources_path, (source.model_dump(mode="json") for source in sources))
    _write_jsonl(sentence_index_path, sentence_index)
    _write_jsonl(rationales_path, sidecar)
    _write_json(frozen_path, frozen_hashes(tuple(cases)))
    readme_path.parent.mkdir(parents=True, exist_ok=True)
    readme_path.write_text(_readme(), encoding="utf-8", newline="\n")
    license_path.write_bytes(upstream_license)

    loaded = load_dataset(cases_path)
    if len(loaded.cases) != 300 or len(loaded.sources) != 300:
        raise AssertionError(
            "load_dataset self-check did not recover 300 cases/sources"
        )
    if any(case.split != "dev" for case in loaded.cases):
        raise AssertionError("all SciFact adapter cases must remain in dev")

    output_paths = {
        CASES_NAME: cases_path,
        SOURCES_RELATIVE: sources_path,
        SENTENCE_INDEX_RELATIVE: sentence_index_path,
        FROZEN_NAME: frozen_path,
        RATIONALES_NAME: rationales_path,
        "README.md": readme_path,
        "SCIFACT-LICENSE.md": license_path,
    }
    manifest_core = {
        "schema_version": "scifact-public-dev-v2",
        "status": "frozen_public_dev_non_blind",
        "dataset": "SciFact official release",
        "official_split": "dev",
        "official_urls": {
            "repository": OFFICIAL_REPOSITORY_URL,
            "data": OFFICIAL_DATA_URL,
            "license": OFFICIAL_LICENSE_URL,
        },
        "licenses": {
            "claims_and_evidence_annotations": "CC BY 4.0",
            "corpus_abstracts": "ODC-By 1.0",
            "upstream_code": "Apache-2.0",
        },
        "raw_counts": observed_counts,
        "raw_sha256": {
            name: _sha256_file(path) for name, path in sorted(raw_paths.items())
        },
        "license_notice": {
            "raw_path": "../LICENSE.md",
            "raw_sha256": _sha256_file(upstream_license_path),
            "output_path": "SCIFACT-LICENSE.md",
            "copied_verbatim": True,
        },
        "adapter_counts": {
            "cases": len(cases),
            "source_fixtures": len(sources),
            **dev_stats,
        },
        "mapping": {
            "empty_evidence": Relation.NOT_IN_SOURCE.value,
            "SUPPORT": Relation.ENTAILED.value,
            "CONTRADICT": Relation.CONTRADICTED.value,
            "sidecar_label": (
                "mapped EvidenceTrace relation; each rationale set retains its "
                "official SUPPORT or CONTRADICT label"
            ),
            "source_policy": (
                "one source per original claim; concatenate distinct cited "
                "abstract sentences in first-occurrence cited_doc_ids order; "
                "exclude synthetic document/sentence/title prefixes from semantic "
                "source text; preserve document titles, sentence locators, exact "
                "character spans, and the raw cited list out-of-band"
            ),
            "compatibility_gold_evidence_span": (
                "first sentence of the first rationale sorted by (doc_id, sentence_ids)"
            ),
            "authoritative_gold": RATIONALES_NAME,
            "prediction_sentence_index": SENTENCE_INDEX_RELATIVE,
        },
        "load_dataset_self_check": {
            "cases": len(loaded.cases),
            "sources": len(loaded.sources),
            "raw_hash": loaded.raw_hash,
            "split_hash": loaded.split_hash,
        },
        "limitations": [
            (
                "Public official dev annotations were available during adapter "
                "construction; this is not a blind holdout."
            ),
            (
                "Cited abstracts are supplied as an oracle corpus; retrieval from "
                "the full 5,183-document corpus is not measured."
            ),
            (
                "SciFact is English scientific-domain data and does not validate "
                "CJK retrieval."
            ),
            "SciFact has no partially_entailed or not_checkable gold classes.",
            (
                "not_in_source means no annotated evidence in cited abstracts, not "
                "global uncheckability."
            ),
            (
                "cases.jsonl keeps one compatibility span; gold_rationales.jsonl is "
                "authoritative for alternative/multi-sentence rationales."
            ),
            (
                "Official dev claim 1245 repeats cited doc 7662395; the raw list is "
                "retained while the rendered source uses stable exact-ID "
                "de-duplication."
            ),
        ],
    }
    manifest_core_sha256 = _sha256_text(_json_line(manifest_core))
    manifest = {
        **manifest_core,
        "manifest_core_sha256": manifest_core_sha256,
        "output_sha256": {
            name: _sha256_file(path) for name, path in sorted(output_paths.items())
        },
        "output_hash_scope": (
            "All generated outputs except manifest.json; the manifest cannot embed "
            "its own final file hash. manifest_core_sha256 covers its metadata "
            "before output_sha256 is attached."
        ),
    }
    _write_json(manifest_path, manifest)
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        required=True,
        type=Path,
        help="Directory containing official corpus.jsonl and claims_*.jsonl files.",
    )
    parser.add_argument(
        "--output-pack",
        type=Path,
        default=DEFAULT_PACK,
        help=f"Output pack directory (default: {DEFAULT_PACK}).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = build_pack(args.raw_dir, args.output_pack)
    print(
        json.dumps(
            {
                "output_pack": str(args.output_pack.resolve()),
                "cases": manifest["adapter_counts"]["cases"],
                "manifest_sha256": _sha256_file(
                    args.output_pack.resolve() / "manifest.json"
                ),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
