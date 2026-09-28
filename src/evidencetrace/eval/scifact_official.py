"""Strict adapters for the public SciFact evaluation protocol.

The implementation mirrors the frozen AllenAI leaderboard evaluator commit
``66feffc5b2cc9e28e3ce3b8c9e824c3c642981eb`` while adding input-completeness
checks that the reference scorer omits.  It intentionally keeps the official
pooled sentence/abstract metrics separate from project-specific claim metrics.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

OFFICIAL_MIN_ABSTRACT_SENTENCES = 3
OFFICIAL_LABELS = frozenset({"SUPPORT", "CONTRADICT"})


class SciFactOfficialProtocolError(ValueError):
    """Raised when an input cannot be scored under the frozen protocol."""


@dataclass(frozen=True, slots=True)
class ValidatedSciFact:
    """Normalized official corpus and claim rows."""

    corpus: dict[int, dict[str, Any]]
    claims: tuple[dict[str, Any], ...]
    claim_ids: tuple[int, ...]


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _f1(counts: Mapping[str, int], correct_key: str) -> dict[str, float]:
    precision = _ratio(counts[correct_key], counts["retrieved"])
    recall = _ratio(counts[correct_key], counts["relevant"])
    harmonic = (
        2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": harmonic}


def validate_official_data(
    corpus_rows: Sequence[dict[str, Any]], claim_rows: Sequence[dict[str, Any]]
) -> ValidatedSciFact:
    """Validate and normalize official SciFact corpus/claim records."""

    corpus: dict[int, dict[str, Any]] = {}
    for row in corpus_rows:
        doc_id = row.get("doc_id")
        abstract = row.get("abstract")
        if not isinstance(doc_id, int) or doc_id in corpus:
            raise SciFactOfficialProtocolError("corpus doc_id must be a unique integer")
        if not isinstance(row.get("title"), str) or not isinstance(abstract, list):
            raise SciFactOfficialProtocolError("invalid corpus row")
        if any(not isinstance(sentence, str) for sentence in abstract):
            raise SciFactOfficialProtocolError("abstract sentences must be strings")
        corpus[doc_id] = row

    claims: list[dict[str, Any]] = []
    seen_claim_ids: set[int] = set()
    for row in claim_rows:
        claim_id = row.get("id")
        evidence = row.get("evidence")
        cited = row.get("cited_doc_ids")
        if not isinstance(claim_id, int) or claim_id in seen_claim_ids:
            raise SciFactOfficialProtocolError("claim id must be a unique integer")
        if not isinstance(row.get("claim"), str) or not isinstance(evidence, dict):
            raise SciFactOfficialProtocolError("invalid claim row")
        if not isinstance(cited, list) or any(
            not isinstance(item, int) for item in cited
        ):
            raise SciFactOfficialProtocolError("cited_doc_ids must be integer IDs")
        if any(item not in corpus for item in cited):
            raise SciFactOfficialProtocolError("claim cites a missing corpus document")
        for raw_doc_id, rationale_sets in evidence.items():
            try:
                doc_id = int(raw_doc_id)
            except (TypeError, ValueError) as exc:
                raise SciFactOfficialProtocolError(
                    "invalid evidence document ID"
                ) from exc
            if doc_id not in corpus or not isinstance(rationale_sets, list):
                raise SciFactOfficialProtocolError("invalid evidence document")
            abstract_size = len(corpus[doc_id]["abstract"])
            for rationale in rationale_sets:
                if not isinstance(rationale, dict):
                    raise SciFactOfficialProtocolError("invalid rationale")
                label = rationale.get("label")
                sentences = rationale.get("sentences")
                if label not in OFFICIAL_LABELS or not isinstance(sentences, list):
                    raise SciFactOfficialProtocolError("invalid rationale fields")
                if any(
                    not isinstance(index, int) or not 0 <= index < abstract_size
                    for index in sentences
                ):
                    raise SciFactOfficialProtocolError(
                        "rationale sentence is out of range"
                    )
        seen_claim_ids.add(claim_id)
        claims.append(row)
    return ValidatedSciFact(
        corpus=corpus,
        claims=tuple(claims),
        claim_ids=tuple(row["id"] for row in claims),
    )


def validate_abstract_retrieval(
    rows: Sequence[dict[str, Any]],
    data: ValidatedSciFact,
    *,
    top_k: int,
) -> tuple[dict[str, Any], ...]:
    """Require one ordered, unique Top-K document list per official claim."""

    if top_k <= 0:
        raise SciFactOfficialProtocolError("top_k must be positive")
    by_id: dict[int, dict[str, Any]] = {}
    for row in rows:
        claim_id = row.get("claim_id")
        doc_ids = row.get("doc_ids")
        if not isinstance(claim_id, int) or claim_id in by_id:
            raise SciFactOfficialProtocolError("retrieval claim_id must be unique")
        if (
            not isinstance(doc_ids, list)
            or len(doc_ids) != top_k
            or len(set(doc_ids)) != top_k
            or any(not isinstance(doc_id, int) for doc_id in doc_ids)
        ):
            raise SciFactOfficialProtocolError(
                "retrieval must contain unique Top-K IDs"
            )
        if any(doc_id not in data.corpus for doc_id in doc_ids):
            raise SciFactOfficialProtocolError("retrieval contains an unknown document")
        by_id[claim_id] = row
    if set(by_id) != set(data.claim_ids):
        raise SciFactOfficialProtocolError("retrieval claim coverage is incomplete")
    return tuple(by_id[claim_id] for claim_id in data.claim_ids)


def compute_abstract_retrieval_metrics(
    rows: Sequence[dict[str, Any]],
    data: ValidatedSciFact,
    *,
    top_k: int,
) -> dict[str, Any]:
    """Compute the official retrieval scores plus explicit evidence-only diagnostics."""

    ordered = validate_abstract_retrieval(rows, data, top_k=top_k)
    official_hit_one = 0
    official_hit_all = 0
    evidence_hit_one = 0
    evidence_hit_all = 0
    evidence_claims = 0
    gold_document_count = 0
    retrieved_gold_document_count = 0
    first_relevant_reciprocal_ranks: list[float] = []

    for claim, retrieval in zip(data.claims, ordered, strict=True):
        gold = {int(doc_id) for doc_id in claim["evidence"]}
        predicted = tuple(retrieval["doc_ids"])
        predicted_set = set(predicted)
        any_hit = bool(predicted_set & gold)
        all_hit = predicted_set.issuperset(gold)
        official_hit_one += int(any_hit or not gold)
        official_hit_all += int(all_hit)
        if not gold:
            continue
        evidence_claims += 1
        evidence_hit_one += int(any_hit)
        evidence_hit_all += int(all_hit)
        gold_document_count += len(gold)
        retrieved_gold_document_count += len(predicted_set & gold)
        first_rank = next(
            (rank for rank, doc_id in enumerate(predicted, 1) if doc_id in gold), None
        )
        first_relevant_reciprocal_ranks.append(
            1.0 / first_rank if first_rank is not None else 0.0
        )

    count = len(data.claims)
    return {
        "official_all_claims": {
            "claim_count": count,
            "hit_one": _ratio(official_hit_one, count),
            "hit_all": _ratio(official_hit_all, count),
            "note": "Matches verisci/evaluate/abstract_retrieval.py, including NEI.",
        },
        "evidence_claims_only": {
            "claim_count": evidence_claims,
            "hit_one": _ratio(evidence_hit_one, evidence_claims),
            "hit_all": _ratio(evidence_hit_all, evidence_claims),
            "first_relevant_mrr_at_k": statistics.fmean(
                first_relevant_reciprocal_ranks
            ),
        },
        "gold_document_recall_at_k": _ratio(
            retrieved_gold_document_count, gold_document_count
        ),
        "top_k": top_k,
    }


def official_raw_rank_statistics(
    corpus_doc_ids_in_order: Sequence[int],
    full_rankings: Sequence[Sequence[int]],
    data: ValidatedSciFact,
) -> dict[str, float | int]:
    """Recompute the raw 0-based ranks mislabeled as reciprocal ranks upstream."""

    corpus_ids = set(corpus_doc_ids_in_order)
    if corpus_ids != set(data.corpus) or len(corpus_ids) != len(
        corpus_doc_ids_in_order
    ):
        raise SciFactOfficialProtocolError("invalid corpus order")
    if len(full_rankings) != len(data.claims):
        raise SciFactOfficialProtocolError("full ranking coverage is incomplete")
    ranks: list[int] = []
    for claim, ranking in zip(data.claims, full_rankings, strict=True):
        position = {doc_id: rank for rank, doc_id in enumerate(ranking)}
        if len(position) != len(corpus_ids) or set(position) != corpus_ids:
            raise SciFactOfficialProtocolError(
                "full ranking is not a corpus permutation"
            )
        ranks.extend(position[int(doc_id)] for doc_id in claim["evidence"])
    if not ranks:
        return {"count": 0, "median_rank": 0.0, "mean_rank": 0.0}
    return {
        "count": len(ranks),
        "median_rank": float(statistics.median(ranks)),
        "mean_rank": statistics.fmean(ranks),
        "min_rank": min(ranks),
        "max_rank": max(ranks),
        "note": "These are 0-based raw ranks, not reciprocal ranks or MRR.",
    }


def _gold_doc_label(rationale_sets: Sequence[dict[str, Any]]) -> str:
    labels = {rationale["label"] for rationale in rationale_sets}
    if len(labels) != 1:
        raise SciFactOfficialProtocolError("gold document has inconsistent labels")
    return next(iter(labels))


def _gold_rationale_sets(
    rationale_sets: Sequence[dict[str, Any]],
) -> tuple[frozenset[int], ...]:
    return tuple(frozenset(rationale["sentences"]) for rationale in rationale_sets)


def _contains_complete_rationale(
    predicted: set[int], gold_sets: Iterable[frozenset[int]]
) -> bool:
    return any(gold.issubset(predicted) for gold in gold_sets)


def _correct_rationale_sentence_count(
    predicted: set[int], gold_sets: tuple[frozenset[int], ...]
) -> int:
    correct = 0
    for sentence in predicted:
        containing = [gold for gold in gold_sets if sentence in gold]
        if len(containing) > 1:
            raise SciFactOfficialProtocolError(
                "one sentence occurs in multiple gold rationale alternatives"
            )
        if containing and containing[0].issubset(predicted):
            correct += 1
    return correct


def validate_pipeline_predictions(
    prediction_rows: Sequence[dict[str, Any]], data: ValidatedSciFact
) -> tuple[dict[str, Any], ...]:
    """Enforce exact claim coverage before invoking the permissive official scorer."""

    by_id: dict[int, dict[str, Any]] = {}
    for row in prediction_rows:
        claim_id = row.get("id")
        evidence = row.get("evidence")
        if not isinstance(claim_id, int) or claim_id in by_id:
            raise SciFactOfficialProtocolError("prediction claim ID must be unique")
        if not isinstance(evidence, dict):
            raise SciFactOfficialProtocolError("prediction evidence must be an object")
        normalized_docs: set[int] = set()
        for raw_doc_id, prediction in evidence.items():
            try:
                doc_id = int(raw_doc_id)
            except (TypeError, ValueError) as exc:
                raise SciFactOfficialProtocolError(
                    "prediction document ID is invalid"
                ) from exc
            if doc_id in normalized_docs or doc_id not in data.corpus:
                raise SciFactOfficialProtocolError(
                    "prediction document ID is duplicate or unknown"
                )
            normalized_docs.add(doc_id)
            if not isinstance(prediction, dict):
                raise SciFactOfficialProtocolError("invalid predicted document")
            label = prediction.get("label")
            sentences = prediction.get("sentences")
            if label not in OFFICIAL_LABELS or not isinstance(sentences, list):
                raise SciFactOfficialProtocolError(
                    "invalid predicted label or sentences"
                )
            abstract_size = len(data.corpus[doc_id]["abstract"])
            if any(
                not isinstance(index, int) or not 0 <= index < abstract_size
                for index in sentences
            ):
                raise SciFactOfficialProtocolError(
                    "predicted sentence index is out of range"
                )
        by_id[claim_id] = row
    if set(by_id) != set(data.claim_ids):
        raise SciFactOfficialProtocolError("prediction claim coverage is incomplete")
    return tuple(by_id[claim_id] for claim_id in data.claim_ids)


def compute_official_pipeline_metrics(
    prediction_rows: Sequence[dict[str, Any]], data: ValidatedSciFact
) -> dict[str, dict[str, float]]:
    """Reproduce the official pooled sentence/abstract P/R/F1 protocol."""

    ordered = validate_pipeline_predictions(prediction_rows, data)
    abstract_counts: Counter[str] = Counter()
    sentence_counts: Counter[str] = Counter()

    for claim, row in zip(data.claims, ordered, strict=True):
        gold_evidence = claim["evidence"]
        abstract_counts["relevant"] += len(gold_evidence)
        sentence_counts["relevant"] += sum(
            len(rationale["sentences"])
            for rationale_sets in gold_evidence.values()
            for rationale in rationale_sets
        )
        for raw_doc_id, prediction in row["evidence"].items():
            doc_id = int(raw_doc_id)
            predicted_sentences = prediction["sentences"]
            abstract_counts["retrieved"] += 1
            sentence_counts["retrieved"] += len(predicted_sentences)
            gold_sets_raw = gold_evidence.get(str(doc_id))
            if gold_sets_raw is None:
                gold_sets_raw = gold_evidence.get(doc_id)
            if not gold_sets_raw:
                continue
            gold_label = _gold_doc_label(gold_sets_raw)
            gold_sets = _gold_rationale_sets(gold_sets_raw)
            label_correct = prediction["label"] == gold_label
            abstract_counts["correct_label_only"] += int(label_correct)
            shortest_rationale = min(len(gold_set) for gold_set in gold_sets)
            max_abstract_sentences = max(
                OFFICIAL_MIN_ABSTRACT_SENTENCES, shortest_rationale
            )
            abstract_sentences = set(
                predicted_sentences[:max_abstract_sentences]
            )
            rationalized = label_correct and _contains_complete_rationale(
                abstract_sentences, gold_sets
            )
            abstract_counts["correct_rationalized"] += int(rationalized)
            correct_sentences = _correct_rationale_sentence_count(
                set(predicted_sentences), gold_sets
            )
            sentence_counts["correct_selection"] += correct_sentences
            sentence_counts["correct_label"] += int(label_correct) * correct_sentences

    metrics = {
        "sentence_selection": _f1(sentence_counts, "correct_selection"),
        "sentence_label": _f1(sentence_counts, "correct_label"),
        "abstract_label_only": _f1(abstract_counts, "correct_label_only"),
        "abstract_rationalized": _f1(abstract_counts, "correct_rationalized"),
    }
    if any(
        not math.isfinite(value)
        for group in metrics.values()
        for value in group.values()
    ):
        raise SciFactOfficialProtocolError("non-finite official metric")
    return metrics
