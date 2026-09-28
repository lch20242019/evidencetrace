"""Deterministic metric calculations with explicit zero-denominator behavior."""

from __future__ import annotations

import re
from collections.abc import Iterable
from statistics import median
from typing import Any

from evidencetrace.models import Relation

LABELS = tuple(item.value for item in Relation)
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[._-][A-Za-z0-9]+)*")


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def confusion_matrix(
    gold: Iterable[str | Relation], predicted: Iterable[str | Relation]
) -> dict[str, dict[str, int]]:
    matrix = {actual: {guess: 0 for guess in LABELS} for actual in LABELS}
    for actual, guess in zip(gold, predicted, strict=False):
        actual_value = actual.value if isinstance(actual, Relation) else str(actual)
        guess_value = guess.value if isinstance(guess, Relation) else str(guess)
        matrix.setdefault(actual_value, {}).setdefault(guess_value, 0)
        matrix[actual_value][guess_value] += 1
    return matrix


def classification_metrics(
    gold: Iterable[str | Relation],
    predicted: Iterable[str | Relation],
    *,
    confidences: Iterable[float] | None = None,
) -> dict[str, Any]:
    gold_values = [
        item.value if isinstance(item, Relation) else str(item) for item in gold
    ]
    predicted_values = [
        item.value if isinstance(item, Relation) else str(item) for item in predicted
    ]
    matrix = confusion_matrix(gold_values, predicted_values)
    per_label: dict[str, dict[str, float | int]] = {}
    observed_f1_values: list[float] = []
    observed_recalls: list[float] = []
    weighted_f1_total = 0.0
    total_support = 0
    for label in LABELS:
        tp = matrix[label][label]
        fp = sum(matrix[actual].get(label, 0) for actual in matrix if actual != label)
        fn = sum(
            matrix[label].get(guess, 0) for guess in matrix[label] if guess != label
        )
        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        f1 = safe_divide(2 * precision * recall, precision + recall)
        support = tp + fn
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        if support:
            observed_f1_values.append(f1)
            observed_recalls.append(recall)
            weighted_f1_total += f1 * support
            total_support += support
    confidence_values = list(confidences or ())
    incorrect_high = sum(
        1
        for actual, guess, confidence in zip(
            gold_values, predicted_values, confidence_values, strict=False
        )
        if confidence >= 0.8 and actual != guess
    )
    high_count = sum(1 for confidence in confidence_values if confidence >= 0.8)
    abstention = sum(
        guess in {Relation.NOT_IN_SOURCE.value, Relation.SOURCE_UNAVAILABLE.value}
        for guess in predicted_values
    )
    contradiction_recall = per_label[Relation.CONTRADICTED.value]["recall"]
    entailed_precision = per_label[Relation.ENTAILED.value]["precision"]
    observed_macro_f1 = safe_divide(sum(observed_f1_values), len(observed_f1_values))
    return {
        "count": len(gold_values),
        # macro_f1 remains an alias for old artifact readers. New reports name
        # the denominator explicitly.
        "macro_f1": observed_macro_f1,
        "observed_label_macro_f1": observed_macro_f1,
        "fixed_taxonomy_macro_f1": safe_divide(
            sum(float(per_label[label]["f1"]) for label in LABELS), len(LABELS)
        ),
        "weighted_f1": safe_divide(weighted_f1_total, total_support),
        "balanced_accuracy": safe_divide(sum(observed_recalls), len(observed_recalls)),
        "label_support": {label: int(per_label[label]["support"]) for label in LABELS},
        "per_label": per_label,
        "contradiction_recall": contradiction_recall,
        "entailed_precision": entailed_precision,
        "partial_support_f1": per_label[Relation.PARTIALLY_ENTAILED.value]["f1"],
        "high_confidence_error_rate": safe_divide(incorrect_high, high_count),
        "abstention_rate": safe_divide(abstention, len(predicted_values)),
        "confusion_matrix": matrix,
    }


def _token_set(value: str | None) -> set[str]:
    return {token.casefold() for token in TOKEN_RE.findall(value or "")}


def token_f1(predicted: str | None, gold: str | None) -> float:
    predicted_tokens = _token_set(predicted)
    gold_tokens = _token_set(gold)
    if not predicted_tokens and not gold_tokens:
        return 1.0
    if not predicted_tokens or not gold_tokens:
        return 0.0
    overlap = len(predicted_tokens & gold_tokens)
    precision = safe_divide(overlap, len(predicted_tokens))
    recall = safe_divide(overlap, len(gold_tokens))
    return safe_divide(2 * precision * recall, precision + recall)


def extraction_metrics(
    *,
    predicted_counts: Iterable[int],
    gold_counts: Iterable[int],
    predicted_lines: Iterable[int | None],
    gold_lines: Iterable[int],
) -> dict[str, float]:
    predicted = list(predicted_counts)
    gold = list(gold_counts)
    lines = list(predicted_lines)
    gold_line_values = list(gold_lines)
    true_positive = sum(
        min(left, right) for left, right in zip(predicted, gold, strict=False)
    )
    precision = safe_divide(true_positive, sum(predicted))
    recall = safe_divide(true_positive, sum(gold))
    return {
        "precision": precision,
        "recall": recall,
        "f1": safe_divide(2 * precision * recall, precision + recall),
        "atomicity_error_rate": safe_divide(
            sum(left != right for left, right in zip(predicted, gold, strict=False)),
            len(gold),
        ),
        "line_mapping_accuracy": safe_divide(
            sum(
                left == right
                for left, right in zip(lines, gold_line_values, strict=False)
            ),
            len(gold_line_values),
        ),
    }


def retrieval_metrics(
    retrieved_texts: Iterable[Iterable[str]], gold_spans: Iterable[str | None]
) -> dict[str, float]:
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    coverage: list[float] = []
    for candidates, gold in zip(retrieved_texts, gold_spans, strict=False):
        values = list(candidates)[:5]
        if not gold:
            continue
        ranks = [
            index + 1
            for index, text in enumerate(values)
            if gold in text or text in gold
        ]
        recalls.append(1.0 if ranks else 0.0)
        reciprocal_ranks.append(safe_divide(1.0, ranks[0]) if ranks else 0.0)
        coverage.append(max((token_f1(text, gold) for text in values), default=0.0))
    return {
        "recall_at_5": safe_divide(sum(recalls), len(recalls)),
        "mrr": safe_divide(sum(reciprocal_ranks), len(reciprocal_ranks)),
        "evidence_coverage": safe_divide(sum(coverage), len(coverage)),
    }


def _p95(values: list[float]) -> float | None:
    if not values:
        return None
    return float(values[min(len(values) - 1, int(len(values) * 0.95))])


def _model_usage(values: list[dict[str, Any]], model_calls: int) -> dict[str, Any]:
    call_records = [item for item in values if int(item.get("model_calls", 0))]
    if model_calls == 0:
        return {
            "status": "not_applicable",
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }
    statuses = {str(item.get("model_usage_status", "missing")) for item in call_records}
    if "malformed" in statuses:
        status = "malformed"
    elif statuses == {"reported"}:
        status = "reported"
    else:
        status = "missing"
    token_fields = ("input_tokens", "output_tokens", "total_tokens")
    complete = status == "reported" and all(
        item.get(field) is not None for item in call_records for field in token_fields
    )
    return {
        "status": status,
        **{
            field: (
                sum(int(item[field]) for item in call_records) if complete else None
            )
            for field in token_fields
        },
    }


def _model_configuration(values: list[dict[str, Any]]) -> dict[str, Any]:
    call_records = [item for item in values if int(item.get("model_calls", 0))]
    fields = ("model_provider", "model_id", "model_temperature")
    observed = {
        field: {item.get(field) for item in call_records if item.get(field) is not None}
        for field in fields
    }
    return {
        "provider": (
            next(iter(observed["model_provider"]))
            if len(observed["model_provider"]) == 1
            else None
        ),
        "model_id": (
            next(iter(observed["model_id"])) if len(observed["model_id"]) == 1 else None
        ),
        "temperature": (
            next(iter(observed["model_temperature"]))
            if len(observed["model_temperature"]) == 1
            else None
        ),
        "consistent": all(len(items) <= 1 for items in observed.values()),
    }


def engineering_metrics(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = list(records)
    latencies = sorted(float(item.get("latency_ms", 0.0)) for item in values)
    model_latencies = sorted(
        float(latency)
        for item in values
        for latency in item.get("model_call_latencies_ms", ())
    )
    model_calls = sum(int(item.get("model_calls", 0)) for item in values)
    usage = _model_usage(values, model_calls)
    non_blocking = [
        item
        for item in values
        if item.get("gold_relation") != Relation.CONTRADICTED.value
    ]
    incorrect_blocks = sum(
        item.get("predicted_relation") == Relation.CONTRADICTED.value
        for item in non_blocking
    )
    priced_records = [item for item in values if int(item.get("model_calls", 0))]
    pricing_ids = {
        str(item["pricing_snapshot_id"])
        for item in priced_records
        if item.get("pricing_snapshot_id")
    }
    cost_available = (
        bool(priced_records)
        and len(pricing_ids) == 1
        and all(item.get("cost_usd") is not None for item in priced_records)
    )
    cost_usd = (
        sum(float(item["cost_usd"]) for item in priced_records)
        if cost_available
        else None
    )
    return {
        "latency_ms_p50": float(median(latencies)) if latencies else 0.0,
        "latency_ms_p95": _p95(latencies) if latencies else 0.0,
        "model_latency_ms_p50": (
            float(median(model_latencies)) if model_latencies else None
        ),
        "model_latency_ms_p95": _p95(model_latencies),
        "model_tool_call_count": model_calls,
        "model_call_count": model_calls,
        "model_usage": usage,
        "schema_failure_count": sum(
            int(item.get("schema_failure_count", 0)) for item in values
        ),
        "transport_failure_count": sum(
            int(item.get("transport_failure_count", 0)) for item in values
        ),
        "model_configuration": _model_configuration(values),
        "cache_hit_rate": safe_divide(
            sum(bool(item.get("cache_hit", False)) for item in values), len(values)
        ),
        "estimated_cost_usd": cost_usd,
        "cost_usd": cost_usd,
        "pricing_snapshot_id": next(iter(pricing_ids)) if cost_available else None,
        "cost_explanation": (
            "Calculated only from the explicit pricing snapshot recorded above."
            if cost_available
            else "No explicit model price snapshot was supplied; cost is null."
        ),
        "false_block_rate": safe_divide(incorrect_blocks, len(non_blocking)),
        "false_block_numerator": incorrect_blocks,
        "false_block_denominator": len(non_blocking),
        "false_block_definition": (
            "Predicted contradicted (blocking/error) among cases whose gold "
            "relation is not contradicted."
        ),
        "severity_mapping": {
            "contradicted": "error_block",
            "partially_entailed": "warning",
            "not_in_source": "warning",
            "source_unavailable": "notice",
            "not_checkable": "notice",
            "entailed": "pass",
        },
    }


def label_coverage_gate(
    verification: dict[str, Any],
    *,
    required_labels: tuple[str, ...] = (
        Relation.ENTAILED.value,
        Relation.PARTIALLY_ENTAILED.value,
        Relation.CONTRADICTED.value,
    ),
    minimum_support: int = 3,
) -> dict[str, Any]:
    """Require meaningful support for the release-critical semantic labels."""

    support = verification.get("label_support", {})
    insufficient = {
        label: int(support.get(label, 0))
        for label in required_labels
        if int(support.get(label, 0)) < minimum_support
    }
    return {
        "status": "insufficient_coverage" if insufficient else "sufficient",
        "achieved": not insufficient,
        "required_labels": list(required_labels),
        "minimum_support": minimum_support,
        "support": {label: int(support.get(label, 0)) for label in required_labels},
        "insufficient_labels": insufficient,
    }


def _pair_extraction_not_applicable() -> dict[str, Any]:
    return {
        "status": "not_applicable",
        "reason": (
            "Claim-source pair cases provide claim_text directly and do not "
            "contain gold document-level claim spans or counts."
        ),
        "precision": None,
        "recall": None,
        "f1": None,
        "atomicity_error_rate": None,
        "line_mapping_accuracy": None,
    }


def compute_metrics(
    records: Iterable[dict[str, Any]],
    *,
    evaluation_unit: str = "claim_source_pair",
) -> dict[str, Any]:
    values = list(records)
    span_records = [item for item in values if item.get("gold_evidence_span")]
    retrieval = retrieval_metrics(
        (item.get("retrieved_texts", ()) for item in values),
        (item.get("gold_evidence_span") for item in values),
    )
    retrieval.update(
        {
            "metric_role": "fixture_sanity",
            "benchmark_limitation": (
                "Repeated small source fixtures make Recall@5 a pipeline sanity "
                "check, not a production retrieval benchmark."
            ),
        }
    )
    return {
        "claim_extraction": (
            _pair_extraction_not_applicable()
            if evaluation_unit == "claim_source_pair"
            else extraction_metrics(
                predicted_counts=(
                    int(item.get("extraction_count", 0)) for item in values
                ),
                gold_counts=(int(item.get("gold_count", 0)) for item in values),
                predicted_lines=(item.get("predicted_line") for item in values),
                gold_lines=(int(item.get("gold_line", 1)) for item in values),
            )
        ),
        "verification": classification_metrics(
            (item["gold_relation"] for item in values),
            (item["predicted_relation"] for item in values),
            confidences=(float(item.get("confidence", 0.0)) for item in values),
        ),
        "retrieval": retrieval,
        "engineering": engineering_metrics(values),
        "source_span_token_f1": safe_divide(
            sum(
                token_f1(
                    item.get("predicted_evidence_span"),
                    item.get("gold_evidence_span"),
                )
                for item in span_records
            ),
            len(span_records),
        ),
    }


__all__ = [
    "LABELS",
    "classification_metrics",
    "compute_metrics",
    "confusion_matrix",
    "engineering_metrics",
    "extraction_metrics",
    "label_coverage_gate",
    "retrieval_metrics",
    "safe_divide",
    "token_f1",
]
