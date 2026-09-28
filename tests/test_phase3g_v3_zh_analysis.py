from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.phase3g_analyze_v3_zh import (
    ERROR_CATEGORIES,
    FORMAL_ARTIFACT_HASHES,
    build_analysis,
    render_markdown,
    verify_formal_artifacts,
    write_analysis,
)

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_phase3g_summary_matches_consumed_gate() -> None:
    analysis = build_analysis(ROOT)

    assert analysis["analysis_status"] == "offline_diagnostic_only"
    assert analysis["formal_gate_status"] == "consumed_metric_failure"
    assert analysis["formal_gate_rerun"] is False
    assert analysis["model_calls"] == 0
    assert analysis["case_count"] == 72
    assert {name: value["count"] for name, value in analysis["quadrants"].items()} == {
        "both_correct": 46,
        "single_agent_only_correct": 10,
        "retrieval_judge_only_correct": 1,
        "both_wrong": 15,
    }
    retrieval = analysis["retrieval_diagnostics"]
    assert retrieval["gold_evidence_case_count"] == 32
    assert retrieval["recall_at_1"] == 22 / 32
    assert retrieval["recall_at_3"] == 22 / 32
    assert retrieval["recall_at_5"] == 22 / 32
    assert retrieval["mrr"] == 22 / 32
    assert retrieval["partial_context_count"] == 0
    assert retrieval["truncated_evidence_count"] == 0


def test_every_case_and_retrieval_error_is_accounted_for() -> None:
    analysis = build_analysis(ROOT)
    cases = analysis["case_comparisons"]

    assert len(cases) == len({case["case_id"] for case in cases}) == 72
    assert all(case["source_id"].startswith("v3_src_") for case in cases)
    assert all(case["source_characters"] >= 0 for case in cases)
    assert all(case["source_paragraph_count"] >= 0 for case in cases)
    assert all(case["retrieval"]["candidate_count"] <= 1 for case in cases)
    errors = [case for case in cases if not case["retrieval_judge"]["correct"]]
    assert len(errors) == 25
    assert all(
        case["retrieval_judge"]["error_category"] in ERROR_CATEGORIES for case in errors
    )
    assert sum(
        value["count"]
        for value in analysis["retrieval_judge_errors"]["categories"].values()
    ) == len(errors)
    assert {
        category: value["count"]
        for category, value in analysis["retrieval_judge_errors"]["categories"].items()
        if value["count"]
    } == {
        "retrieval_miss": 10,
        "judge_error_with_sufficient_evidence": 8,
        "deterministic_router_error": 4,
        "annotation_or_translation_ambiguity": 3,
    }


def test_relation_and_contradiction_diagnostics_are_complete() -> None:
    analysis = build_analysis(ROOT)

    assert {
        relation: (
            value["count"],
            value["single_agent_correct"],
            value["retrieval_judge_correct"],
        )
        for relation, value in analysis["relations"].items()
    } == {
        "entailed": (10, 10, 4),
        "partially_entailed": (9, 8, 7),
        "contradicted": (13, 13, 10),
        "not_in_source": (16, 13, 14),
        "source_unavailable": (12, 12, 12),
        "not_checkable": (12, 0, 0),
    }
    contradiction = analysis["contradicted_feature_context"]
    assert contradiction["case_count"] == 13
    assert contradiction["context_received_count"] == 10
    assert {
        feature: value["relevant_case_count"]
        for feature, value in contradiction["feature_summary"].items()
    } == {
        "numbers": 3,
        "dates": 0,
        "versions": 12,
        "entities": 13,
        "negations": 9,
    }
    assert (
        contradiction["feature_summary"]["versions"]["claim_feature_case_count"] == 12
    )
    assert (
        contradiction["feature_summary"]["versions"][
            "all_claim_values_in_context_count"
        ]
        == 2
    )
    assert all(
        set(case["feature_context"])
        == {"numbers", "dates", "versions", "entities", "negations"}
        for case in contradiction["cases"]
    )
    assert analysis["source_profile"]["available_characters_max"] == 127


def test_writer_preserves_formal_artifacts_and_emits_safe_outputs(
    tmp_path: Path,
) -> None:
    before = verify_formal_artifacts(ROOT)
    json_path = tmp_path / "error_analysis.json"
    markdown_path = tmp_path / "error_analysis.md"

    write_analysis(ROOT, json_path, markdown_path)

    assert verify_formal_artifacts(ROOT) == before
    assert before == {
        relative: _sha256(ROOT / relative) for relative in FORMAL_ARTIFACT_HASHES
    }
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["model_calls"] == 0
    markdown = markdown_path.read_text(encoding="utf-8")
    assert markdown == render_markdown(payload)
    assert "new holdout metrics" in markdown
    assert "OPENAI_API_KEY" not in json_path.read_text(encoding="utf-8")
