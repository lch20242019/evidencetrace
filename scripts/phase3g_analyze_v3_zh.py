"""Offline diagnostics for the consumed v3_zh formal gate.

This script never imports a model client or evaluation runner. It verifies the
frozen inputs before reading them and writes only new postmortem artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

RELATIONS = (
    "entailed",
    "partially_entailed",
    "contradicted",
    "not_in_source",
    "source_unavailable",
    "not_checkable",
)
SUBSTANTIVE_RELATIONS = frozenset({"entailed", "partially_entailed", "contradicted"})
FORMAL_ARTIFACT_HASHES = {
    "eval_runs/phase3f_v3_zh_internal_gate/eval_report.md": (
        "335ad7b4e434773db4c4581d6195970ac81b41dd8ed2f7b5cfe6bfb348bfd32c"
    ),
    "eval_runs/phase3f_v3_zh_internal_gate/eval_results.jsonl": (
        "1e95c9f0306b401477ad7b3acc15af9f9757fd9facc893c4710e7c5cd69c2690"
    ),
    "eval_runs/phase3f_v3_zh_internal_gate/metrics.json": (
        "7983c7aca138eb007d6f1aa1bddea49d21a9371ce7dacfcda7472c62c297a25f"
    ),
    "eval_runs/phase3f_v3_zh_internal_gate/run_manifest.json": (
        "e07a5999cf8932dc28ff69f9fd93e4a511fb357e5e6e1fe31648d3cfb33c6069"
    ),
    "eval_runs/phase3f_v3_zh_internal_gate/execution_record.json": (
        "74540003d8200410038028dc4a898aee422d00f1683a03ddfd465c8f01febf4a"
    ),
    "eval_sets/v3_zh/holdout_single_human_zh.jsonl": (
        "34518f925e82bd3a1d716bc42ab07ab6b6b3b4333408fab45bfb573d38e67302"
    ),
    "eval_sets/v3_zh/manifest.json": (
        "eb540f6e8e61483fd3e725b2d77817c6bebeacd4babe0d81a8b851a3939f19d5"
    ),
}
ERROR_CATEGORIES = (
    "retrieval_miss",
    "retrieval_partial_context",
    "judge_error_with_sufficient_evidence",
    "deterministic_router_error",
    "evidence_span_selection_error",
    "annotation_or_translation_ambiguity",
    "unsupported_other",
    "unknown",
)
AMBIGUITY_FLAGS = {
    "v3_holdout_candidate_053": (
        "The Chinese source states support for a .venv path containing a "
        "centralized project environment, while gold is not_in_source."
    ),
    "v3_holdout_candidate_058": (
        "The Chinese source directly names Nushell 0.114.1, while gold is "
        "not_in_source."
    ),
    "v3_holdout_candidate_061": (
        "The conjunction contains support plus an explicit conflict, making "
        "partially_entailed versus contradicted a relation-boundary ambiguity."
    ),
}
DATE_RE = re.compile(
    r"20\d{2}(?:[-/]\d{1,2}[-/]\d{1,2}|年\s*\d{1,2}\s*月\s*\d{1,2}\s*日)"
)
VERSION_RE = re.compile(r"\bv?\d+(?:\.\d+){1,3}\b", re.I)
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?%?")
LATIN_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9])[A-Za-z][A-Za-z0-9]*(?:[-_.][A-Za-z0-9]+)*"
)
NEGATION_TERMS = (
    "没有",
    "不再",
    "不能",
    "不会",
    "并非",
    "从不",
    "禁止",
    "删除",
    "弃用",
    "丢弃",
    "未",
    "无",
    "不",
)
ENTITY_STOPWORDS = frozenset(
    {
        "add",
        "bundle",
        "compile",
        "core",
        "date",
        "file",
        "files",
        "help",
        "news",
        "path",
        "release",
        "run",
        "source",
        "system",
        "version",
    }
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_formal_artifacts(root: Path) -> dict[str, str]:
    """Fail before analysis if any consumed-gate input has changed."""

    actual = {relative: _sha256(root / relative) for relative in FORMAL_ARTIFACT_HASHES}
    mismatches = {
        relative: {"expected": FORMAL_ARTIFACT_HASHES[relative], "actual": digest}
        for relative, digest in actual.items()
        if digest != FORMAL_ARTIFACT_HASHES[relative]
    }
    if mismatches:
        raise ValueError("formal artifact hash mismatch")
    return actual


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _paragraph_count(content: str) -> int:
    if not content.strip():
        return 0
    return len([part for part in re.split(r"\n\s*\n", content) if part.strip()])


def _compact_chars(text: str) -> str:
    return "".join(character.casefold() for character in text if character.isalnum())


def _char_ngrams(text: str, size: int = 2) -> set[str]:
    compact = _compact_chars(text)
    if len(compact) < size:
        return {compact} if compact else set()
    return {compact[index : index + size] for index in range(len(compact) - size + 1)}


def _partial_context(gold: str, candidates: list[str]) -> bool:
    gold_ngrams = _char_ngrams(gold)
    if not gold_ngrams:
        return False
    return any(
        len(gold_ngrams & _char_ngrams(candidate)) / len(gold_ngrams) >= 0.5
        for candidate in candidates
    )


def _truncated_evidence(gold: str, candidates: list[str]) -> bool:
    compact_gold = _compact_chars(gold)
    return any(
        4 <= len(compact_candidate) < len(compact_gold)
        and compact_candidate in compact_gold
        for candidate in candidates
        if (compact_candidate := _compact_chars(candidate))
    )


def _gold_rank(gold: str | None, candidates: list[str]) -> int | None:
    if gold is None:
        return None
    for index, candidate in enumerate(candidates, start=1):
        if gold in candidate:
            return index
    return None


def _normalize_date(value: str) -> str:
    numbers = re.findall(r"\d+", value)
    return "-".join([numbers[0], numbers[1].zfill(2), numbers[2].zfill(2)])


def _feature_values(text: str) -> dict[str, list[str]]:
    dates = {_normalize_date(match.group(0)) for match in DATE_RE.finditer(text)}
    without_dates = DATE_RE.sub(" ", text)
    versions = {
        match.group(0).casefold().removeprefix("v")
        for match in VERSION_RE.finditer(without_dates)
    }
    without_slots = VERSION_RE.sub(" ", without_dates)
    numbers = {match.group(0) for match in NUMBER_RE.finditer(without_slots)}
    entities = {
        match.group(0).casefold()
        for match in LATIN_TOKEN_RE.finditer(text)
        if match.group(0).casefold() not in ENTITY_STOPWORDS
        and not VERSION_RE.fullmatch(match.group(0))
        and not match.group(0).startswith("--")
    }
    negations = {term for term in NEGATION_TERMS if term in text}
    return {
        "numbers": sorted(numbers),
        "dates": sorted(dates),
        "versions": sorted(versions),
        "entities": sorted(entities),
        "negations": sorted(negations),
    }


def _feature_context(claim: str, source: str, contexts: list[str]) -> dict[str, Any]:
    context = "\n".join(contexts)
    claim_values = _feature_values(claim)
    source_values = _feature_values(source)
    context_values = _feature_values(context)
    return {
        name: {
            "claim_values": claim_values[name],
            "source_values": source_values[name],
            "context_values": context_values[name],
            "all_claim_values_in_context": bool(claim_values[name])
            and set(claim_values[name]) <= set(context_values[name]),
            "all_source_values_in_context": bool(source_values[name])
            and set(source_values[name]) <= set(context_values[name]),
        }
        for name in ("numbers", "dates", "versions", "entities", "negations")
    }


def _quadrant(single_correct: bool, retrieval_correct: bool) -> str:
    if single_correct and retrieval_correct:
        return "both_correct"
    if single_correct:
        return "single_agent_only_correct"
    if retrieval_correct:
        return "retrieval_judge_only_correct"
    return "both_wrong"


def _error_category(
    *,
    case_id: str,
    correct: bool,
    gold_evidence: str | None,
    exact_hit: bool | None,
    partial_context: bool | None,
    model_calls: int,
    source_available: bool,
) -> tuple[str | None, str | None]:
    if correct:
        return None, None
    if case_id in AMBIGUITY_FLAGS:
        return "annotation_or_translation_ambiguity", AMBIGUITY_FLAGS[case_id]
    if gold_evidence is not None and not exact_hit:
        if partial_context:
            return (
                "retrieval_partial_context",
                "Gold evidence was not exact but at least half of its character "
                "bigrams reached the Judge context.",
            )
        return (
            "retrieval_miss",
            "Gold evidence did not occur in any retrieved candidate.",
        )
    if model_calls == 0 and source_available:
        return (
            "deterministic_router_error",
            "The wrong relation was emitted before a Judge model call.",
        )
    if model_calls > 0:
        return (
            "judge_error_with_sufficient_evidence",
            "The Judge received the available source context but returned the "
            "wrong relation.",
        )
    return "unsupported_other", "No narrower category is supported by the artifact."


def _relation_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for relation in RELATIONS:
        selected = [case for case in cases if case["gold_relation"] == relation]
        single_correct = sum(case["single_agent"]["correct"] for case in selected)
        retrieval_correct = sum(case["retrieval_judge"]["correct"] for case in selected)
        single_confusions = Counter(
            case["single_agent"]["prediction"]
            for case in selected
            if not case["single_agent"]["correct"]
        )
        retrieval_confusions = Counter(
            case["retrieval_judge"]["prediction"]
            for case in selected
            if not case["retrieval_judge"]["correct"]
        )
        summary[relation] = {
            "count": len(selected),
            "single_agent_correct": single_correct,
            "single_agent_accuracy": single_correct / len(selected),
            "retrieval_judge_correct": retrieval_correct,
            "retrieval_judge_accuracy": retrieval_correct / len(selected),
            "single_agent_wrong_predictions": dict(sorted(single_confusions.items())),
            "retrieval_judge_wrong_predictions": dict(
                sorted(retrieval_confusions.items())
            ),
        }
    return summary


def _length_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    bins = {
        "available_le_50": lambda case: (
            case["source_available"] and case["source_characters"] <= 50
        ),
        "available_51_75": lambda case: (
            case["source_available"] and 51 <= case["source_characters"] <= 75
        ),
        "available_76_127": lambda case: (
            case["source_available"] and 76 <= case["source_characters"] <= 127
        ),
        "source_unavailable": lambda case: not case["source_available"],
    }
    result: dict[str, Any] = {}
    for name, predicate in bins.items():
        selected = [case for case in cases if predicate(case)]
        result[name] = {
            "count": len(selected),
            "single_agent_correct": sum(
                case["single_agent"]["correct"] for case in selected
            ),
            "retrieval_judge_correct": sum(
                case["retrieval_judge"]["correct"] for case in selected
            ),
        }
    return result


def _feature_summary(details: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for feature in ("numbers", "dates", "versions", "entities", "negations"):
        claim_relevant = [
            item for item in details if item["feature_context"][feature]["claim_values"]
        ]
        source_relevant = [
            item
            for item in details
            if item["feature_context"][feature]["source_values"]
        ]
        relevant = [
            item
            for item in details
            if item["feature_context"][feature]["claim_values"]
            or item["feature_context"][feature]["source_values"]
        ]
        summary[feature] = {
            "relevant_case_count": len(relevant),
            "context_received_count": sum(
                item["context_received"] for item in relevant
            ),
            "claim_feature_case_count": len(claim_relevant),
            "all_claim_values_in_context_count": sum(
                item["feature_context"][feature]["all_claim_values_in_context"]
                for item in claim_relevant
            ),
            "source_feature_case_count": len(source_relevant),
            "all_source_values_in_context_count": sum(
                item["feature_context"][feature]["all_source_values_in_context"]
                for item in source_relevant
            ),
        }
    return summary


def build_analysis(root: Path) -> dict[str, Any]:
    """Build a deterministic diagnostic view without changing formal inputs."""

    hashes = verify_formal_artifacts(root)
    run_dir = root / "eval_runs/phase3f_v3_zh_internal_gate"
    rows = _jsonl(run_dir / "eval_results.jsonl")
    gold_rows = _jsonl(root / "eval_sets/v3_zh/holdout_single_human_zh.jsonl")
    sources = _jsonl(root / "eval_sets/v3_zh/sources/source_snapshots_zh.jsonl")
    execution = json.loads(
        (run_dir / "execution_record.json").read_text(encoding="utf-8")
    )
    if execution["status"] != "consumed_metric_failure":
        raise ValueError("unexpected consumed gate status")
    gold_by_id = {row["case_id"]: row for row in gold_rows}
    source_by_id = {row["source_id"]: row for row in sources}
    live_by_case: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        if row["baseline"] in {"single_agent_live", "retrieval_judge_live"}:
            live_by_case.setdefault(row["case_id"], {})[row["baseline"]] = row
    if len(gold_by_id) != 72 or len(live_by_case) != 72 or len(rows) != 360:
        raise ValueError("consumed gate result cardinality mismatch")

    cases: list[dict[str, Any]] = []
    for gold in gold_rows:
        case_id = gold["case_id"]
        source = source_by_id[gold["source_id"]]
        single = live_by_case[case_id]["single_agent_live"]
        retrieval = live_by_case[case_id]["retrieval_judge_live"]
        if single["gold_relation"] != gold["gold_relation"]:
            raise ValueError("single-agent gold relation mismatch")
        if retrieval["gold_relation"] != gold["gold_relation"]:
            raise ValueError("retrieval-judge gold relation mismatch")
        candidates = retrieval["retrieved_texts"]
        gold_evidence = gold["gold_evidence_span"]
        rank = _gold_rank(gold_evidence, candidates)
        exact_hit = rank is not None if gold_evidence is not None else None
        partial = (
            _partial_context(gold_evidence, candidates)
            if gold_evidence is not None and rank is None
            else False
            if gold_evidence is not None
            else None
        )
        truncated = (
            _truncated_evidence(gold_evidence, candidates)
            if gold_evidence is not None and rank is None
            else False
            if gold_evidence is not None
            else None
        )
        single_correct = single["predicted_relation"] == gold["gold_relation"]
        retrieval_correct = retrieval["predicted_relation"] == gold["gold_relation"]
        category, rationale = _error_category(
            case_id=case_id,
            correct=retrieval_correct,
            gold_evidence=gold_evidence,
            exact_hit=exact_hit,
            partial_context=partial,
            model_calls=retrieval["model_calls"],
            source_available=source["available"],
        )
        cases.append(
            {
                "case_id": case_id,
                "source_id": gold["source_id"],
                "claim_text": gold["claim_text"],
                "gold_relation": gold["gold_relation"],
                "gold_evidence_span": gold_evidence,
                "single_agent": {
                    "prediction": single["predicted_relation"],
                    "correct": single_correct,
                    "confidence": single["confidence"],
                    "predicted_evidence_span": single["predicted_evidence_span"],
                    "confusion_category": (
                        f"{gold['gold_relation']}→{single['predicted_relation']}"
                    ),
                },
                "retrieval_judge": {
                    "prediction": retrieval["predicted_relation"],
                    "correct": retrieval_correct,
                    "confidence": retrieval["confidence"],
                    "predicted_evidence_span": retrieval["predicted_evidence_span"],
                    "model_calls": retrieval["model_calls"],
                    "confusion_category": (
                        f"{gold['gold_relation']}→{retrieval['predicted_relation']}"
                    ),
                    "error_category": category,
                    "error_rationale": rationale,
                },
                "quadrant": _quadrant(single_correct, retrieval_correct),
                "retrieval": {
                    "stored_retrieval_rank": retrieval["retrieval_rank"],
                    "gold_evidence_rank": rank,
                    "candidate_count": len(candidates),
                    "context_contains_exact_gold_evidence": exact_hit,
                    "partial_context_without_exact_evidence": partial,
                    "truncated_evidence_without_exact_match": truncated,
                },
                "source_available": source["available"],
                "source_characters": len(source["content"]),
                "source_paragraph_count": _paragraph_count(source["content"]),
                "source_nonempty_line_count": len(
                    [line for line in source["content"].splitlines() if line.strip()]
                ),
            }
        )

    quadrant_counts = Counter(case["quadrant"] for case in cases)
    substantive = [
        case for case in cases if case["gold_relation"] in SUBSTANTIVE_RELATIONS
    ]
    ranks = [
        case["retrieval"]["gold_evidence_rank"]
        for case in substantive
        if case["retrieval"]["gold_evidence_rank"] is not None
    ]
    errors = [case for case in cases if not case["retrieval_judge"]["correct"]]
    category_counts = Counter(
        case["retrieval_judge"]["error_category"] for case in errors
    )
    contradicted_details = []
    for case in cases:
        if case["gold_relation"] != "contradicted":
            continue
        source = source_by_id[case["source_id"]]
        retrieval = live_by_case[case["case_id"]]["retrieval_judge_live"]
        contradicted_details.append(
            {
                "case_id": case["case_id"],
                "context_received": bool(retrieval["retrieved_texts"]),
                "candidate_count": len(retrieval["retrieved_texts"]),
                "feature_context": _feature_context(
                    case["claim_text"],
                    source["content"],
                    retrieval["retrieved_texts"],
                ),
            }
        )
    available_lengths = sorted(
        len(source["content"]) for source in sources if source["available"]
    )
    return {
        "schema_version": "phase3g-v3-zh-postmortem-v1",
        "analysis_status": "offline_diagnostic_only",
        "formal_gate_status": "consumed_metric_failure",
        "formal_gate_rerun": False,
        "model_calls": 0,
        "metric_boundary": (
            "These are post-hoc diagnostics over a consumed holdout, not new "
            "holdout metrics and not a replacement formal gate."
        ),
        "formal_artifact_sha256": hashes,
        "methods": {
            "exact_retrieval_hit": (
                "Literal gold-evidence substring occurrence in a retrieved "
                "candidate; rank is one-indexed."
            ),
            "partial_context": (
                "No exact hit and at least 50% of normalized gold-evidence "
                "character bigrams occur in one candidate."
            ),
            "truncated_evidence": (
                "No exact hit and a normalized candidate of at least four "
                "characters is a proper substring of gold evidence."
            ),
            "source_characters": "Python Unicode code-point count.",
            "source_paragraphs": "Non-empty blocks separated by blank lines.",
            "feature_context": (
                "Deterministic lexical audit with Chinese date/negation support; "
                "entity values are Latin identifier tokens and aliases are not "
                "resolved."
            ),
            "ambiguity_flags": AMBIGUITY_FLAGS,
        },
        "case_count": len(cases),
        "case_comparisons": cases,
        "quadrants": {
            name: {
                "count": quadrant_counts[name],
                "case_ids": [
                    case["case_id"] for case in cases if case["quadrant"] == name
                ],
            }
            for name in (
                "both_correct",
                "single_agent_only_correct",
                "retrieval_judge_only_correct",
                "both_wrong",
            )
        },
        "relations": _relation_summary(cases),
        "retrieval_diagnostics": {
            "gold_evidence_case_count": len(substantive),
            "recall_at_1": sum(rank <= 1 for rank in ranks) / len(substantive),
            "recall_at_3": sum(rank <= 3 for rank in ranks) / len(substantive),
            "recall_at_5": sum(rank <= 5 for rank in ranks) / len(substantive),
            "mrr": sum(1 / rank for rank in ranks) / len(substantive),
            "exact_substring_coverage": len(ranks) / len(substantive),
            "exact_hit_count": len(ranks),
            "complete_miss_count": sum(
                not case["retrieval"]["context_contains_exact_gold_evidence"]
                and not case["retrieval"]["partial_context_without_exact_evidence"]
                for case in substantive
            ),
            "partial_context_count": sum(
                case["retrieval"]["partial_context_without_exact_evidence"]
                for case in substantive
            ),
            "truncated_evidence_count": sum(
                case["retrieval"]["truncated_evidence_without_exact_match"]
                for case in substantive
            ),
            "maximum_candidate_count": max(
                case["retrieval"]["candidate_count"] for case in cases
            ),
            "benchmark_limitation": (
                "Repeated short source fixtures make these consumed-set "
                "diagnostics unsuitable as a general retrieval benchmark."
            ),
        },
        "retrieval_judge_errors": {
            "count": len(errors),
            "categories": {
                category: {
                    "count": category_counts[category],
                    "proportion": category_counts[category] / len(errors),
                    "case_ids": [
                        case["case_id"]
                        for case in errors
                        if case["retrieval_judge"]["error_category"] == category
                    ],
                }
                for category in ERROR_CATEGORIES
            },
        },
        "contradicted_feature_context": {
            "case_count": len(contradicted_details),
            "context_received_count": sum(
                item["context_received"] for item in contradicted_details
            ),
            "feature_summary": _feature_summary(contradicted_details),
            "cases": contradicted_details,
        },
        "source_profile": {
            "source_count": len(sources),
            "available_source_count": len(available_lengths),
            "unavailable_source_count": len(sources) - len(available_lengths),
            "available_characters_min": min(available_lengths),
            "available_characters_median": median(available_lengths),
            "available_characters_max": max(available_lengths),
            "length_strata": _length_summary(cases),
        },
    }


def _ratio(value: float) -> str:
    return f"{value:.6f}"


def render_markdown(analysis: dict[str, Any]) -> str:
    """Render the structured analysis without model-authored reasons."""

    retrieval = analysis["retrieval_diagnostics"]
    errors = analysis["retrieval_judge_errors"]
    lines = [
        "# v3_zh consumed-gate offline error analysis",
        "",
        "**Status:** offline diagnostic only. The formal gate remains",
        "`consumed_metric_failure`; v3_zh was not rerun and model calls were 0.",
        "",
        analysis["metric_boundary"],
        "",
        "## Four-way comparison",
        "",
        "| Quadrant | Cases |",
        "|---|---:|",
    ]
    for name, value in analysis["quadrants"].items():
        lines.append(f"| `{name}` | {value['count']} |")
    lines.extend(
        [
            "",
            "## Relation accuracy",
            "",
            "| Gold relation | Cases | Single correct / accuracy | "
            "Retrieval-Judge correct / accuracy |",
            "|---|---:|---:|---:|",
        ]
    )
    for relation, value in analysis["relations"].items():
        lines.append(
            f"| `{relation}` | {value['count']} | "
            f"{value['single_agent_correct']} / "
            f"{_ratio(value['single_agent_accuracy'])} | "
            f"{value['retrieval_judge_correct']} / "
            f"{_ratio(value['retrieval_judge_accuracy'])} |"
        )
    lines.extend(
        [
            "",
            "## Retrieval diagnostics",
            "",
            f"- Gold-evidence cases: {retrieval['gold_evidence_case_count']}",
            f"- Recall@1 / @3 / @5: {_ratio(retrieval['recall_at_1'])} / "
            f"{_ratio(retrieval['recall_at_3'])} / "
            f"{_ratio(retrieval['recall_at_5'])}",
            f"- MRR: {_ratio(retrieval['mrr'])}",
            "- Exact substring coverage: "
            f"{retrieval['exact_hit_count']}/"
            f"{retrieval['gold_evidence_case_count']} "
            f"({_ratio(retrieval['exact_substring_coverage'])})",
            f"- Complete misses: {retrieval['complete_miss_count']}; partial "
            f"context: {retrieval['partial_context_count']}; truncated "
            f"evidence: {retrieval['truncated_evidence_count']}",
            "",
            retrieval["benchmark_limitation"],
            "",
            "## Retrieval-Judge relation errors",
            "",
            "| Category | Count | Proportion | Representative case IDs |",
            "|---|---:|---:|---|",
        ]
    )
    for category, value in errors["categories"].items():
        representatives = ", ".join(value["case_ids"][:5]) or "none"
        lines.append(
            f"| `{category}` | {value['count']} | "
            f"{_ratio(value['proportion'])} | {representatives} |"
        )
    lines.extend(
        [
            "",
            "## Contradicted-case feature context",
            "",
            "The entity audit is lexical: it tracks Latin identifier tokens "
            "and does not resolve aliases.",
            "",
            "| Feature | Relevant | Context received | Claim feature cases / "
            "all entered | Source feature cases / all entered |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    feature_summary = analysis["contradicted_feature_context"]["feature_summary"]
    for feature, value in feature_summary.items():
        lines.append(
            f"| `{feature}` | {value['relevant_case_count']} | "
            f"{value['context_received_count']} | "
            f"{value['claim_feature_case_count']} / "
            f"{value['all_claim_values_in_context_count']} | "
            f"{value['source_feature_case_count']} / "
            f"{value['all_source_values_in_context_count']} |"
        )
    lines.extend(
        [
            "",
            "## Per-case comparison",
            "",
            "| Case | Source | Gold | Single prediction / correct | "
            "Retrieval-Judge prediction / correct | Candidates | Gold rank | "
            "Exact evidence | Quadrant | Error category |",
            "|---|---|---|---|---|---:|---:|---|---|---|",
        ]
    )
    for case in analysis["case_comparisons"]:
        rank = case["retrieval"]["gold_evidence_rank"]
        exact = case["retrieval"]["context_contains_exact_gold_evidence"]
        lines.append(
            f"| {case['case_id']} | {case['source_id']} | "
            f"`{case['gold_relation']}` | "
            f"`{case['single_agent']['prediction']}` / "
            f"{case['single_agent']['correct']} | "
            f"`{case['retrieval_judge']['prediction']}` / "
            f"{case['retrieval_judge']['correct']} | "
            f"{case['retrieval']['candidate_count']} | "
            f"{rank if rank is not None else 'n/a'} | "
            f"{exact if exact is not None else 'n/a'} | "
            f"`{case['quadrant']}` | "
            f"`{case['retrieval_judge']['error_category'] or 'none'}` |"
        )
    lines.extend(
        [
            "",
            "Gold and predicted evidence, confidence, retrieval diagnostics, "
            "source size, paragraph/line counts, confusion categories, and "
            "contradicted-case feature context are preserved in "
            "`error_analysis.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def write_analysis(root: Path, json_path: Path, markdown_path: Path) -> None:
    """Write only the two new postmortem artifacts."""

    protected = {root / relative for relative in FORMAL_ARTIFACT_HASHES}
    resolved_outputs = {json_path.resolve(), markdown_path.resolve()}
    if protected & resolved_outputs:
        raise ValueError("analysis output cannot replace a formal artifact")
    analysis = build_analysis(root)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(analysis), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("eval_runs/phase3f_v3_zh_internal_gate/error_analysis.json"),
    )
    parser.add_argument(
        "--markdown-out",
        type=Path,
        default=Path("eval_runs/phase3f_v3_zh_internal_gate/error_analysis.md"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    json_path = args.json_out if args.json_out.is_absolute() else root / args.json_out
    markdown_path = (
        args.markdown_out
        if args.markdown_out.is_absolute()
        else root / args.markdown_out
    )
    write_analysis(root, json_path, markdown_path)


if __name__ == "__main__":
    main()
