from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from evidencetrace.eval.scifact_official import (
    compute_official_pipeline_metrics,
    validate_official_data,
)
from scripts.summarize_scifact_agent_ab import (
    ARMS,
    FINAL_EVIDENCE_PREFILL,
    INITIAL_DRAFT_PREFILL,
    JSON_ROOT_STOPPING_POLICY,
    OFFICIAL_CORPUS_COUNT,
    OFFICIAL_CORPUS_SHA256,
    OFFICIAL_EVALUATOR,
    OFFICIAL_EVALUATOR_COMMIT,
    OFFICIAL_TRAIN_CLAIM_COUNT,
    OFFICIAL_TRAIN_CLAIMS_SHA256,
    V2_AGENT_PROTOCOL,
    V2_FINAL_EVIDENCE_PREFILL,
    V2_INITIAL_DRAFT_PREFILL,
    V2_JSON_ROOT_STOPPING_POLICY,
    V2_PARSER_POLICIES,
    AgentABSummaryError,
    _draft_to_evidence,
    _parse_initial_draft,
    _parse_v2_agent_output,
    _parse_v2_initial_draft,
    _sha256_bytes,
    _v2_json_root_closed,
    summarize,
)

PARSER_POLICIES = {
    "initial_draft": {
        "version": "scifact-agent-draft-parser-v1",
        "analysis_nonempty": True,
        "analysis_max_words": None,
        "candidate_shape": ["retrieval_rank", "label", "sentence_indices"],
        "unique_candidate_ranks": True,
        "retrieval_rank_scope": [1, 3],
        "rank_mapping": "one_based_rank_to_retrieved_top3_document",
        "labels": ["SUPPORT", "CONTRADICT", "NEI"],
        "nei_candidate_sentence_indices": (
            "optional_but_strictly_scope_validated"
        ),
        "nei_mixed_with_non_nei": "reject",
        "nei_projection": "ignored_when_mapping_to_evidence",
        "semantic_repairs": 0,
        "invalid_draft_fallback": "fixed_draft_unavailable_sentinel",
        "one_pass_projection": "candidate_order_to_evidence_object_without_repair",
    },
    "final_evidence": {
        "version": "scifact-agent-output-parser-v1",
        "semantic_repairs": 0,
        "retries": 0,
        "invalid_final_fallback": "empty_evidence",
    },
}


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(_canonical(row) + "\n" for row in rows), encoding="utf-8"
    )


def _flatten(metrics: dict[str, dict[str, float]]) -> dict[str, float]:
    return {
        f"{group}_{statistic}": value
        for group, values in metrics.items()
        for statistic, value in values.items()
    }


def _valid_parse(evidence: dict[str, Any]) -> dict[str, Any]:
    return {
        "valid": True,
        "failure_code": None,
        "transport_unwrapped": False,
        "value": {"evidence": evidence},
        "detail": None,
    }


def _invalid_parse() -> dict[str, Any]:
    return {
        "valid": False,
        "failure_code": "invalid_json",
        "transport_unwrapped": False,
        "value": None,
        "detail": None,
    }


def _valid_draft(candidates: list[list[Any]]) -> dict[str, Any]:
    return {
        "valid": True,
        "failure_code": None,
        "transport_unwrapped": False,
        "value": {"analysis": "checked", "candidates": candidates},
        "detail": None,
    }


def _call(
    *,
    case_index: int,
    claim_id: int,
    context_sha256: str,
    kind: str,
    raw: str,
    parsed: dict[str, Any],
    input_tokens: int,
    output_tokens: int,
    synchronized_latency_ms: float,
    end_to_end_latency_ms: float,
) -> dict[str, Any]:
    is_initial = kind == "shared_initial_judge"
    assistant_prefill = (
        INITIAL_DRAFT_PREFILL if is_initial else FINAL_EVIDENCE_PREFILL
    )
    parse_schema = "initial_draft" if is_initial else "final_evidence"
    assert raw.startswith(assistant_prefill)
    suffix = {
        "shared_initial_judge": "initial",
        "self_review": "self_review",
        "independent_challenger": "independent_challenger",
    }[kind]
    attribution = {
        "shared_initial_judge": list(ARMS),
        "self_review": ["single_self_review"],
        "independent_challenger": ["judge_challenger"],
    }[kind]
    rendered_prompt = f"prompt:{claim_id}:{kind}:{assistant_prefill}"
    generated_fragment = raw[len(assistant_prefill) :]
    return {
        "schema_version": "scifact-agent-physical-call-v1",
        "call_id": f"{case_index:04d}:{claim_id}:{suffix}",
        "call_kind": kind,
        "logical_attribution": attribution,
        "claim_id": claim_id,
        "canonical_context_sha256": context_sha256,
        "device": "fixture-gpu",
        "messages_sha256": _sha_text(f"messages:{claim_id}:{kind}"),
        "rendered_prompt": rendered_prompt,
        "rendered_prompt_sha256": _sha_text(rendered_prompt),
        "prompt_sha256": _sha_text(rendered_prompt),
        "prompt_character_count": len(rendered_prompt),
        "assistant_prefill": assistant_prefill,
        "assistant_prefill_sha256": _sha_text(assistant_prefill),
        "assistant_prefill_in_input_tokens": True,
        "generated_fragment": generated_fragment,
        "generated_fragment_sha256": _sha_text(generated_fragment),
        "raw_output": raw,
        "raw_output_sha256": _sha_text(raw),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "synchronized_latency_ms": synchronized_latency_ms,
        "end_to_end_latency_ms": end_to_end_latency_ms,
        "generation_succeeded": True,
        "generation_error": None,
        "json_root_stopping_triggered": parsed["valid"],
        "stopping_policy": JSON_ROOT_STOPPING_POLICY,
        "stopping_policy_sha256": _sha_text(JSON_ROOT_STOPPING_POLICY),
        "external_api_cost_usd": 0.0,
        "compute_cost_usd": None,
        "parse_schema": parse_schema,
        "parser_policy_sha256": _sha_text(_canonical(PARSER_POLICIES[parse_schema])),
        "parse": parsed,
    }


def _arm(calls: list[dict[str, Any]], parsed: dict[str, Any]) -> dict[str, Any]:
    return {
        "valid": parsed["valid"],
        "failure_code": parsed["failure_code"],
        "transport_unwrapped": parsed["transport_unwrapped"],
        "final_evidence": parsed["value"]["evidence"] if parsed["valid"] else {},
        "invalid_final_fallback_applied": not parsed["valid"],
        "logical_call_count": len(calls),
        "physical_call_ids": [call["call_id"] for call in calls],
        "input_tokens": sum(call["input_tokens"] for call in calls),
        "output_tokens": sum(call["output_tokens"] for call in calls),
        "total_tokens": sum(call["total_tokens"] for call in calls),
        "logical_inference_latency_ms": sum(
            call["synchronized_latency_ms"] for call in calls
        ),
        "logical_end_to_end_latency_ms": sum(
            call["end_to_end_latency_ms"] for call in calls
        ),
        "external_api_cost_usd": 0.0,
        "compute_cost_usd": None,
    }


def _v2_call(
    *,
    case_index: int,
    claim_id: int,
    context_sha256: str,
    kind: str,
    raw: str,
    parsed: dict[str, Any],
    input_tokens: int,
    output_tokens: int,
    synchronized_latency_ms: float,
    end_to_end_latency_ms: float,
) -> dict[str, Any]:
    assistant_prefill = (
        V2_INITIAL_DRAFT_PREFILL
        if kind == "shared_initial_judge"
        else V2_FINAL_EVIDENCE_PREFILL
    )
    assert raw.startswith(assistant_prefill)
    parse_schema = (
        "initial_draft" if kind == "shared_initial_judge" else "final_evidence"
    )
    suffix = {
        "shared_initial_judge": "initial",
        "self_review": "self_review",
        "independent_challenger": "independent_challenger",
    }[kind]
    attribution = {
        "shared_initial_judge": list(ARMS),
        "self_review": ["single_self_review"],
        "independent_challenger": ["judge_challenger"],
    }[kind]
    rendered_prompt = f"v2-prompt:{claim_id}:{kind}:{assistant_prefill}"
    generated_fragment = raw[len(assistant_prefill) :]
    return {
        "schema_version": "scifact-agent-physical-call-v1",
        "call_id": f"{case_index:04d}:{claim_id}:{suffix}",
        "call_kind": kind,
        "logical_attribution": attribution,
        "claim_id": claim_id,
        "canonical_context_sha256": context_sha256,
        "device": "fixture-gpu",
        "messages_sha256": _sha_text(f"v2-messages:{claim_id}:{kind}"),
        "rendered_prompt": rendered_prompt,
        "rendered_prompt_sha256": _sha_text(rendered_prompt),
        "prompt_sha256": _sha_text(rendered_prompt),
        "prompt_character_count": len(rendered_prompt),
        "assistant_prefill": assistant_prefill,
        "assistant_prefill_sha256": _sha_text(assistant_prefill),
        "assistant_prefill_in_input_tokens": True,
        "generated_fragment": generated_fragment,
        "generated_fragment_sha256": _sha_text(generated_fragment),
        "raw_output": raw,
        "raw_output_sha256": _sha_text(raw),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "synchronized_latency_ms": synchronized_latency_ms,
        "end_to_end_latency_ms": end_to_end_latency_ms,
        "generation_succeeded": True,
        "generation_error": None,
        "json_root_stopping_triggered": _v2_json_root_closed(raw),
        "stopping_policy": V2_JSON_ROOT_STOPPING_POLICY,
        "stopping_policy_sha256": _sha_text(V2_JSON_ROOT_STOPPING_POLICY),
        "external_api_cost_usd": 0.0,
        "compute_cost_usd": None,
        "generation_overrides": {"repetition_penalty": 1.0},
        "parse_schema": parse_schema,
        "parser_policy_sha256": _sha_text(
            _canonical(V2_PARSER_POLICIES[parse_schema])
        ),
        "parse": parsed,
    }


def _v2_arm(calls: list[dict[str, Any]], parsed: dict[str, Any]) -> dict[str, Any]:
    evidence = parsed["projected_evidence"] if parsed["valid"] else {}
    return {
        "valid": parsed["valid"],
        "failure_code": parsed["failure_code"],
        "transport_unwrapped": parsed["transport_unwrapped"],
        "final_evidence": evidence,
        "invalid_final_fallback_applied": not parsed["valid"],
        "logical_call_count": len(calls),
        "physical_call_ids": [call["call_id"] for call in calls],
        "input_tokens": sum(call["input_tokens"] for call in calls),
        "output_tokens": sum(call["output_tokens"] for call in calls),
        "total_tokens": sum(call["total_tokens"] for call in calls),
        "logical_inference_latency_ms": sum(
            call["synchronized_latency_ms"] for call in calls
        ),
        "logical_end_to_end_latency_ms": sum(
            call["end_to_end_latency_ms"] for call in calls
        ),
        "external_api_cost_usd": 0.0,
        "compute_cost_usd": None,
    }


def _score_directory(
    directory: Path,
    *,
    predictions_path: Path,
    predictions: list[dict[str, Any]],
    data: Any,
    corpus_path: Path,
    claims_path: Path,
) -> None:
    directory.mkdir()
    nested = compute_official_pipeline_metrics(predictions, data)
    official = _flatten(nested)
    _write_json(directory / "official_metrics.json", official)
    _write_json(directory / "independent_metrics.json", nested)
    (directory / "official_stdout.txt").write_text("fixture\n", encoding="utf-8")
    (directory / "official_stderr.txt").write_text("", encoding="utf-8")
    outputs = {
        name: _sha(directory / name)
        for name in (
            "official_metrics.json",
            "independent_metrics.json",
            "official_stdout.txt",
            "official_stderr.txt",
        )
    }
    _write_json(
        directory / "run_manifest.json",
        {
            "schema_version": "scifact-official-pipeline-score-v1",
            "evaluation_status": {
                "reportable": True,
                "mode": "official_leaderboard_evaluator_reproduction",
            },
            "official_evaluator": {
                "repository": "https://github.com/allenai/scifact-evaluator",
                "commit": OFFICIAL_EVALUATOR_COMMIT,
                "archive_sha256": "a" * 64,
                "evaluator_sha256": "b" * 64,
            },
            "data": {
                "split": "dev",
                "claim_count": len(data.claims),
                "corpus_count": len(data.corpus),
                "corpus_sha256": _sha(corpus_path),
                "claims_sha256": _sha(claims_path),
            },
            "predictions": {
                "sha256": _sha(predictions_path),
                "exact_claim_coverage": True,
                "known_documents_and_valid_sentence_indices": True,
            },
            "validation": {
                "official_evaluator_cross_check": True,
                "absolute_tolerance": 1e-12,
            },
            "outputs_sha256": outputs,
        },
    )


def _fixture(tmp_path: Path) -> dict[str, Any]:
    data_dir = tmp_path / "data"
    retrieval_dir = tmp_path / "retrieval"
    runner_dir = tmp_path / "runner"
    data_dir.mkdir()
    retrieval_dir.mkdir()
    runner_dir.mkdir()
    corpus_rows = [
        {"doc_id": 1, "title": "One", "abstract": ["support sentence"]},
        {"doc_id": 2, "title": "Two", "abstract": ["other sentence"]},
        {"doc_id": 3, "title": "Three", "abstract": ["third sentence"]},
    ]
    claim_rows = [
        {
            "id": 10,
            "claim": "supported claim",
            "evidence": {
                "1": [{"label": "SUPPORT", "sentences": [0]}],
            },
            "cited_doc_ids": [1],
        },
        {
            "id": 11,
            "claim": "not enough evidence",
            "evidence": {},
            "cited_doc_ids": [2],
        },
    ]
    retrieval_rows = [
        {"claim_id": row["id"], "doc_ids": [1, 2, 3]} for row in claim_rows
    ]
    corpus_path = data_dir / "corpus.jsonl"
    claims_path = data_dir / "claims_dev.jsonl"
    retrieval_path = retrieval_dir / "abstract_retrieval.jsonl"
    _write_jsonl(corpus_path, corpus_rows)
    _write_jsonl(claims_path, claim_rows)
    _write_jsonl(retrieval_path, retrieval_rows)
    _write_json(
        retrieval_dir / "run_manifest.json",
        {
            "schema_version": "scifact-official-tfidf-run-v1",
            "evaluation_status": {"reportable": True},
            "data": {
                "split": "dev",
                "files_sha256": {
                    corpus_path.name: _sha(corpus_path),
                    claims_path.name: _sha(claims_path),
                },
            },
            "configuration": {"top_k": 3},
            "outputs_sha256": {retrieval_path.name: _sha(retrieval_path)},
        },
    )

    contexts: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    telemetry: list[dict[str, Any]] = []
    predictions = {arm: [] for arm in ARMS}
    for case_index, claim in enumerate(claim_rows):
        context = {
            "claim": claim["claim"],
            "documents": [
                {
                    "rank": rank,
                    "doc_id": document["doc_id"],
                    "title": document["title"],
                    "sentences": [
                        {"index": index, "text": sentence}
                        for index, sentence in enumerate(document["abstract"])
                    ],
                }
                for rank, document in enumerate(corpus_rows, 1)
            ],
        }
        context_hash = _sha_text(_canonical(context))
        contexts.append(
            {
                "schema_version": "scifact-agent-context-v1",
                "claim_id": claim["id"],
                "context": context,
                "canonical_context_sha256": context_hash,
                "gold_fields_present": False,
            }
        )
        empty = _valid_parse({})
        self_parse = (
            _valid_parse({"1": {"label": "SUPPORT", "sentences": [0]}})
            if case_index == 0
            else empty
        )
        initial_draft = _valid_draft(
            [[1, "SUPPORT", [0]]] if case_index == 0 else [[2, "NEI", []]]
        )
        initial_raw = _canonical(initial_draft["value"])
        one_pass_parse = self_parse if case_index == 0 else empty
        challenger_parse = _invalid_parse() if case_index == 0 else empty
        initial = _call(
            case_index=case_index,
            claim_id=claim["id"],
            context_sha256=context_hash,
            kind="shared_initial_judge",
            raw=initial_raw,
            parsed=initial_draft,
            input_tokens=10,
            output_tokens=2,
            synchronized_latency_ms=1.0,
            end_to_end_latency_ms=1.5,
        )
        self_call = _call(
            case_index=case_index,
            claim_id=claim["id"],
            context_sha256=context_hash,
            kind="self_review",
            raw=_canonical(self_parse["value"]),
            parsed=self_parse,
            input_tokens=12,
            output_tokens=3,
            synchronized_latency_ms=2.0,
            end_to_end_latency_ms=2.5,
        )
        challenger_call = _call(
            case_index=case_index,
            claim_id=claim["id"],
            context_sha256=context_hash,
            kind="independent_challenger",
            raw=(
                '{"evidence":{not-json'
                if case_index == 0
                else '{"evidence":{}}'
            ),
            parsed=challenger_parse,
            input_tokens=14,
            output_tokens=4,
            synchronized_latency_ms=3.0,
            end_to_end_latency_ms=3.5,
        )
        branch_order = (
            ["single_self_review", "judge_challenger"]
            if case_index % 2 == 0
            else ["judge_challenger", "single_self_review"]
        )
        by_arm_call = {
            "single_self_review": self_call,
            "judge_challenger": challenger_call,
        }
        physical_calls = [initial] + [by_arm_call[arm] for arm in branch_order]
        arm_values = {
            "single_one_pass": _arm([initial], one_pass_parse),
            "single_self_review": _arm([initial, self_call], self_parse),
            "judge_challenger": _arm(
                [initial, challenger_call], challenger_parse
            ),
        }
        cases.append(
            {
                "schema_version": "scifact-agent-case-v1",
                "case_index": case_index,
                "claim_id": claim["id"],
                "canonical_context_sha256": context_hash,
                "branch_execution_order": branch_order,
                "draft_status": "valid_initial_draft",
                "draft_sha256": _sha_text(initial_raw),
                "physical_call_count": 3,
                "physical_call_ids": [call["call_id"] for call in physical_calls],
                "physical_input_tokens": 36,
                "physical_output_tokens": 9,
                "physical_total_tokens": 45,
                "physical_inference_latency_ms": 6.0,
                "physical_end_to_end_latency_ms": 7.5,
                "arms": arm_values,
            }
        )
        telemetry.extend(physical_calls)
        for arm in ARMS:
            predictions[arm].append(
                {"id": claim["id"], "evidence": arm_values[arm]["final_evidence"]}
            )

    _write_jsonl(runner_dir / "contexts.jsonl", contexts)
    _write_jsonl(runner_dir / "cases.jsonl", cases)
    _write_jsonl(runner_dir / "raw_telemetry.jsonl", telemetry)
    for arm in ARMS:
        _write_jsonl(runner_dir / f"predictions_{arm}.jsonl", predictions[arm])
    (runner_dir / "journal.jsonl").write_text("fixture\n", encoding="utf-8")
    retrieval_manifest = retrieval_dir / "run_manifest.json"
    configuration_lock = {
        "generation": {
            "max_new_tokens_per_call": 256,
            "calls_per_case_physical": 3,
            "initial_call_shared": True,
            "logical_calls_single_one_pass": 1,
            "logical_calls_single_self_review": 2,
            "logical_calls_judge_challenger": 2,
            "do_sample": False,
            "temperature": None,
            "top_p": None,
            "top_k": None,
            "num_beams": 1,
            "batch_size": 1,
            "assistant_prefills": {
                "shared_initial_judge": {
                    "value": INITIAL_DRAFT_PREFILL,
                    "sha256": _sha_text(INITIAL_DRAFT_PREFILL),
                },
                "self_review_and_challenger": {
                    "value": FINAL_EVIDENCE_PREFILL,
                    "sha256": _sha_text(FINAL_EVIDENCE_PREFILL),
                },
            },
            "assistant_prefill_accounting": "rendered_prompt_input_tokens",
            "chat_template_mode": "continue_final_message",
            "stopping_policy": JSON_ROOT_STOPPING_POLICY,
            "stopping_policy_sha256": _sha_text(JSON_ROOT_STOPPING_POLICY),
            "stopping_policy_posthoc_truncation": False,
            "application_response_cache": False,
            "retry_count": 0,
            "initial_draft_projection": (
                "validated_candidates_to_one_pass_evidence_preserving_order"
            ),
        },
        "retrieval_policy": {
            "system": "official_scifact_tfidf",
            "top_k": 3,
            "document_context": "title_and_all_indexed_abstract_sentences",
            "branch_order": "even_case_index_self_then_challenger_else_reverse",
        },
        "parser_policies": PARSER_POLICIES,
        "parser_policies_sha256": _sha_text(_canonical(PARSER_POLICIES)),
    }
    input_lock = {
        "split": "dev",
        "corpus_sha256": _sha(corpus_path),
        "claims_sha256": _sha(claims_path),
        "retrieval_sha256": _sha(retrieval_path),
        "retrieval_manifest_sha256": _sha(retrieval_manifest),
    }
    train_dir = tmp_path / "train_runner"
    train_dir.mkdir()
    (train_dir / "completed.txt").write_text("fixture\n", encoding="utf-8")
    train_manifest_path = train_dir / "run_manifest.json"
    _write_json(
        train_manifest_path,
        {
            "schema_version": "scifact-agent-ab-run-v1",
            "configuration_lock": configuration_lock,
            "input_lock": {"split": "train"},
            "execution": {"completed": True, "selected_claim_count": 2},
            "outputs_sha256": {
                "completed.txt": _sha(train_dir / "completed.txt")
            },
        },
    )
    frozen_path = tmp_path / "frozen_agent_ab.json"
    _write_json(
        frozen_path,
        {
            "schema_version": "scifact-agent-ab-frozen-config-v1",
            "train_run_manifest": {
                "path": str(train_manifest_path.resolve()),
                "sha256": _sha(train_manifest_path),
                "selected_claim_count": 2,
            },
            "locked_configuration": configuration_lock,
            "dev_inputs": input_lock,
            "selection": {
                "train_used_for_prompt_and_parser_smoke_only": True,
                "dev_tuning_permitted": False,
                "main_comparison": "judge_challenger_vs_single_self_review",
            },
        },
    )
    _write_json(
        runner_dir / "run_header.json",
        {
            "schema_version": "scifact-agent-ab-incomplete-run-v1",
            "configuration_lock": configuration_lock,
            "input_lock": input_lock,
            "selected_claim_ids": [10, 11],
            "frozen_config_sha256": _sha(frozen_path),
        },
    )
    output_names = [
        "contexts.jsonl",
        "cases.jsonl",
        "raw_telemetry.jsonl",
        "journal.jsonl",
        "run_header.json",
        *(f"predictions_{arm}.jsonl" for arm in ARMS),
    ]
    runner_outputs = {name: _sha(runner_dir / name) for name in output_names}
    _write_json(
        runner_dir / "run_manifest.json",
        {
            "schema_version": "scifact-agent-ab-run-v1",
            "evaluation_status": {
                "reportable": True,
                "state": "ready_for_official_scoring",
                "gold_used_by_runner": False,
            },
            "configuration_lock": configuration_lock,
            "input_lock": input_lock,
            "frozen_config": {
                "path": str(frozen_path.resolve()),
                "sha256": _sha(frozen_path),
            },
            "execution": {
                "completed": True,
                "resume_used": False,
                "selected_claim_count": 2,
                "selected_claim_ids_sha256": _sha_text(_canonical([10, 11])),
                "physical_call_count": 6,
                "expected_physical_call_count": 6,
                "wall_time_ms": 1000.0,
                "wall_time_scope": "run_entry_through_pre_materialization",
                "active_physical_call_end_to_end_time_ms": 15.0,
            },
            "comparison": {
                "main": "judge_challenger_vs_single_self_review",
                "reference_only": "single_one_pass",
                "shared_initial_call": True,
                "main_arms_logical_calls_per_case": 2,
                "same_model_top_k_context_and_output_budget": True,
                "branch_execution_alternated_by_case_index_parity": True,
            },
            "accounting": {
                "physical_calls_count_shared_initial_once": True,
                "logical_arm_totals_attribute_shared_initial_to_each_arm": True,
                "external_api_cost_usd": 0.0,
                "compute_cost_usd": None,
            },
            "outputs_sha256": runner_outputs,
        },
    )
    data = validate_official_data(corpus_rows, claim_rows)
    scores = {}
    for arm in ARMS:
        score_dir = tmp_path / f"score_{arm}"
        _score_directory(
            score_dir,
            predictions_path=runner_dir / f"predictions_{arm}.jsonl",
            predictions=predictions[arm],
            data=data,
            corpus_path=corpus_path,
            claims_path=claims_path,
        )
        scores[arm] = score_dir
    return {
        "runner": runner_dir,
        "corpus": corpus_path,
        "claims": claims_path,
        "retrieval": retrieval_path,
        "scores": scores,
    }


def _upgrade_fixture_to_v2(
    fixture: dict[str, Any], tmp_path: Path
) -> dict[str, Any]:
    runner_dir = fixture["runner"]
    runner_manifest_path = runner_dir / "run_manifest.json"
    runner_manifest = json.loads(runner_manifest_path.read_text(encoding="utf-8"))
    configuration = runner_manifest["configuration_lock"]
    scripts = Path(__file__).parents[1] / "scripts"
    v2_runner = scripts / "compat" / "run_scifact_hf_agent_ab_v2.py"
    base_runner = scripts / "compat" / "run_scifact_hf_agent_ab.py"
    configuration["script_sha256"] = _sha(v2_runner)
    configuration["agent_protocol_version"] = V2_AGENT_PROTOCOL
    configuration["implementation_dependencies"] = {
        "audited_runner_base": {
            "path": str(base_runner.resolve()),
            "sha256": _sha(base_runner),
        }
    }
    generation = configuration["generation"]
    generation["assistant_prefills"] = {
        "shared_initial_judge": {
            "value": V2_INITIAL_DRAFT_PREFILL,
            "sha256": _sha_text(V2_INITIAL_DRAFT_PREFILL),
        },
        "self_review_and_challenger": {
            "value": V2_FINAL_EVIDENCE_PREFILL,
            "sha256": _sha_text(V2_FINAL_EVIDENCE_PREFILL),
        },
    }
    generation["stopping_policy"] = V2_JSON_ROOT_STOPPING_POLICY
    generation["stopping_policy_sha256"] = _sha_text(
        V2_JSON_ROOT_STOPPING_POLICY
    )
    generation["initial_draft_projection"] = (
        "validated_explicit_label_and_ranked_citations_to_evidence"
    )
    generation["repetition_penalty"] = 1.0
    generation["repetition_penalty_source"] = (
        "explicit_protocol_override_of_model_generation_config"
    )
    configuration["parser_policies"] = V2_PARSER_POLICIES
    configuration["parser_policies_sha256"] = _sha_text(
        _canonical(V2_PARSER_POLICIES)
    )

    contexts = [
        json.loads(line)
        for line in (runner_dir / "contexts.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    cases = []
    telemetry = []
    predictions = {arm: [] for arm in ARMS}
    allowed = {1: 1, 2: 1, 3: 1}
    for case_index, context in enumerate(contexts):
        claim_id = context["claim_id"]
        context_hash = context["canonical_context_sha256"]
        if case_index == 0:
            initial_raw = '{"label":"SUPPORT","citations":[[1,[0]]]}'
            self_raw = '{"label":"SUPPORT","citations":[[1,[0]]]}'
            challenger_raw = '{"label":"MAYBE","citations":[]}'
        else:
            initial_raw = '{"label":"NEI","citations":[]}'
            self_raw = '{"label":"NEI","citations":[]}'
            challenger_raw = '{"label":"NEI","citations":[]}'
        initial_parse = _parse_v2_initial_draft(initial_raw, allowed)
        self_parse = _parse_v2_agent_output(self_raw, allowed)
        challenger_parse = _parse_v2_agent_output(challenger_raw, allowed)
        initial = _v2_call(
            case_index=case_index,
            claim_id=claim_id,
            context_sha256=context_hash,
            kind="shared_initial_judge",
            raw=initial_raw,
            parsed=initial_parse,
            input_tokens=10,
            output_tokens=2,
            synchronized_latency_ms=1.0,
            end_to_end_latency_ms=1.5,
        )
        self_call = _v2_call(
            case_index=case_index,
            claim_id=claim_id,
            context_sha256=context_hash,
            kind="self_review",
            raw=self_raw,
            parsed=self_parse,
            input_tokens=12,
            output_tokens=3,
            synchronized_latency_ms=2.0,
            end_to_end_latency_ms=2.5,
        )
        challenger_call = _v2_call(
            case_index=case_index,
            claim_id=claim_id,
            context_sha256=context_hash,
            kind="independent_challenger",
            raw=challenger_raw,
            parsed=challenger_parse,
            input_tokens=14,
            output_tokens=4,
            synchronized_latency_ms=3.0,
            end_to_end_latency_ms=3.5,
        )
        branch_order = (
            ["single_self_review", "judge_challenger"]
            if case_index % 2 == 0
            else ["judge_challenger", "single_self_review"]
        )
        branches = {
            "single_self_review": self_call,
            "judge_challenger": challenger_call,
        }
        physical_calls = [initial] + [branches[arm] for arm in branch_order]
        arm_values = {
            "single_one_pass": _v2_arm([initial], initial_parse),
            "single_self_review": _v2_arm([initial, self_call], self_parse),
            "judge_challenger": _v2_arm(
                [initial, challenger_call], challenger_parse
            ),
        }
        canonical_draft = _canonical(initial_parse["value"])
        cases.append(
            {
                "schema_version": "scifact-agent-case-v1",
                "case_index": case_index,
                "claim_id": claim_id,
                "canonical_context_sha256": context_hash,
                "branch_execution_order": branch_order,
                "draft_status": "valid_initial_draft",
                "draft_sha256": _sha_text(canonical_draft),
                "physical_call_count": 3,
                "physical_call_ids": [
                    call["call_id"] for call in physical_calls
                ],
                "physical_input_tokens": 36,
                "physical_output_tokens": 9,
                "physical_total_tokens": 45,
                "physical_inference_latency_ms": 6.0,
                "physical_end_to_end_latency_ms": 7.5,
                "arms": arm_values,
            }
        )
        telemetry.extend(physical_calls)
        for arm in ARMS:
            predictions[arm].append(
                {
                    "id": claim_id,
                    "evidence": arm_values[arm]["final_evidence"],
                }
            )
    _write_jsonl(runner_dir / "cases.jsonl", cases)
    _write_jsonl(runner_dir / "raw_telemetry.jsonl", telemetry)
    for arm in ARMS:
        _write_jsonl(runner_dir / f"predictions_{arm}.jsonl", predictions[arm])

    train_dir = tmp_path / "train_runner_v2"
    train_dir.mkdir()
    train_cases = [
        {
            "schema_version": "scifact-agent-case-v1",
            "claim_id": claim_id,
            "arms": {
                arm: {
                    "valid": True,
                    "final_evidence": (
                        {"1": {"label": "SUPPORT", "sentences": [0]}}
                        if claim_id % 2 == 0
                        else {}
                    ),
                }
                for arm in ARMS
            },
        }
        for claim_id in range(OFFICIAL_TRAIN_CLAIM_COUNT)
    ]
    _write_jsonl(train_dir / "cases.jsonl", train_cases)
    for arm in ARMS:
        _write_jsonl(
            train_dir / f"predictions_{arm}.jsonl",
            [
                {
                    "id": claim_id,
                    "evidence": (
                        {"1": {"label": "SUPPORT", "sentences": [0]}}
                        if claim_id % 2 == 0
                        else {}
                    ),
                }
                for claim_id in range(OFFICIAL_TRAIN_CLAIM_COUNT)
            ],
        )
    train_outputs = {
        path.name: _sha(path) for path in train_dir.iterdir() if path.is_file()
    }
    train_manifest_path = train_dir / "run_manifest.json"
    _write_json(
        train_manifest_path,
        {
            "schema_version": "scifact-agent-ab-run-v1",
            "evaluation_status": {
                "reportable": False,
                "state": "ready_for_official_scoring",
                "gold_used_by_runner": False,
            },
            "configuration_lock": configuration,
            "input_lock": {
                "split": "train",
                "corpus_sha256": OFFICIAL_CORPUS_SHA256,
                "claims_sha256": OFFICIAL_TRAIN_CLAIMS_SHA256,
                "retrieval_sha256": "a" * 64,
                "retrieval_manifest_sha256": "b" * 64,
            },
            "execution": {
                "completed": True,
                "source_claim_count": OFFICIAL_TRAIN_CLAIM_COUNT,
                "selected_claim_count": OFFICIAL_TRAIN_CLAIM_COUNT,
                "physical_call_count": OFFICIAL_TRAIN_CLAIM_COUNT * 3,
                "expected_physical_call_count": OFFICIAL_TRAIN_CLAIM_COUNT * 3,
            },
            "outputs_sha256": train_outputs,
        },
    )

    score_reports = {}
    nested_metrics = {
        group: {statistic: 0.2 for statistic in ("precision", "recall", "f1")}
        for group in (
            "sentence_selection",
            "sentence_label",
            "abstract_label_only",
            "abstract_rationalized",
        )
    }
    flat_metrics = _flatten(nested_metrics)
    for arm in ARMS:
        score_dir = tmp_path / f"train_score_v2_{arm}"
        score_dir.mkdir()
        _write_json(score_dir / "official_metrics.json", flat_metrics)
        _write_json(score_dir / "independent_metrics.json", nested_metrics)
        (score_dir / "official_stdout.txt").write_text("", encoding="utf-8")
        (score_dir / "official_stderr.txt").write_text("", encoding="utf-8")
        score_outputs = {
            path.name: _sha(path)
            for path in score_dir.iterdir()
            if path.is_file()
        }
        score_manifest_path = score_dir / "run_manifest.json"
        _write_json(
            score_manifest_path,
            {
                "schema_version": "scifact-official-pipeline-score-v1",
                "evaluation_status": {
                    "reportable": True,
                    "mode": "official_leaderboard_evaluator_reproduction",
                },
                "official_evaluator": OFFICIAL_EVALUATOR,
                "data": {
                    "split": "train",
                    "claim_count": OFFICIAL_TRAIN_CLAIM_COUNT,
                    "corpus_count": OFFICIAL_CORPUS_COUNT,
                    "corpus_sha256": OFFICIAL_CORPUS_SHA256,
                    "claims_sha256": OFFICIAL_TRAIN_CLAIMS_SHA256,
                },
                "predictions": {
                    "sha256": train_outputs[f"predictions_{arm}.jsonl"],
                    "exact_claim_coverage": True,
                    "known_documents_and_valid_sentence_indices": True,
                },
                "validation": {
                    "official_evaluator_cross_check": True,
                    "absolute_tolerance": 1e-12,
                },
                "outputs_sha256": score_outputs,
            },
        )
        score_reports[arm] = {
            "score_manifest_path": str(score_manifest_path.resolve()),
            "score_manifest_sha256": _sha(score_manifest_path),
            "official_metrics_sha256": score_outputs["official_metrics.json"],
            "independent_metrics_sha256": score_outputs[
                "independent_metrics.json"
            ],
            "official_metrics": flat_metrics,
        }

    observed = {
        arm: {
            "format_valid_count": OFFICIAL_TRAIN_CLAIM_COUNT,
            "nonempty_prediction_count": 405,
            "label_counts": {
                "SUPPORT": 405,
                "CONTRADICT": 0,
                "NEI": 404,
            },
            "claim_count": OFFICIAL_TRAIN_CLAIM_COUNT,
            "format_valid_rate": 1.0,
            "nonempty_prediction_rate": 405 / OFFICIAL_TRAIN_CLAIM_COUNT,
            "distinct_label_count": 2,
        }
        for arm in ARMS
    }
    gate_script = scripts / "freeze_scifact_agent_ab_config.py"
    gate_report = {
        "schema_version": "scifact-agent-ab-train-admission-v1",
        "status": "passed",
        "scope": "complete_official_train_only",
        "agent_protocol_version": V2_AGENT_PROTOCOL,
        "dev_labels_or_metrics_inspected": False,
        "thresholds_are_viability_floors_not_performance_claims": True,
        "implementation": {
            "path": "scripts/freeze_scifact_agent_ab_config.py",
            "sha256": _sha(gate_script),
        },
        "thresholds": {
            "minimum_format_valid_rate_all_arms": 0.99,
            "minimum_nonempty_prediction_rate_main_arms": 0.05,
            "minimum_distinct_final_labels_main_arms": 2,
            "minimum_official_train_abstract_rationalized_f1_main_arms": 0.05,
            "minimum_official_train_sentence_label_f1_main_arms": 0.05,
        },
        "observed": observed,
        "official_train_scores": score_reports,
        "evidence_sha256": {
            "train_manifest": _sha(train_manifest_path),
            "cases": train_outputs["cases.jsonl"],
            "predictions": {
                arm: train_outputs[f"predictions_{arm}.jsonl"] for arm in ARMS
            },
        },
    }
    gate_report_path = tmp_path / "v2.train_admission_gate.json"
    _write_json(gate_report_path, gate_report)
    gate_binding = {
        "schema_version": "scifact-agent-ab-train-admission-v1",
        "status": "passed",
        "report": {
            "path": str(gate_report_path.resolve()),
            "sha256": _sha(gate_report_path),
        },
    }
    input_lock = runner_manifest["input_lock"]
    frozen_path = tmp_path / "frozen_agent_ab_v2.json"
    _write_json(
        frozen_path,
        {
            "schema_version": "scifact-agent-ab-frozen-config-v2",
            "created_at": "2026-09-04T00:00:00+00:00",
            "train_run_manifest": {
                "path": str(train_manifest_path.resolve()),
                "sha256": _sha(train_manifest_path),
                "selected_claim_count": OFFICIAL_TRAIN_CLAIM_COUNT,
            },
            "locked_configuration": configuration,
            "dev_inputs": input_lock,
            "selection": {
                "train_used_for_viability_gate_only": True,
                "dev_tuning_permitted": False,
                "main_comparison": "judge_challenger_vs_single_self_review",
            },
            "train_admission_gate": gate_binding,
            "train_admission_gate_sha256": _sha256_bytes(
                _canonical(gate_binding).encode("utf-8")
            ),
        },
    )
    _write_json(
        runner_dir / "run_header.json",
        {
            "schema_version": "scifact-agent-ab-incomplete-run-v1",
            "configuration_lock": configuration,
            "input_lock": input_lock,
            "selected_claim_ids": [10, 11],
            "frozen_config_sha256": _sha(frozen_path),
        },
    )
    output_names = [
        "contexts.jsonl",
        "cases.jsonl",
        "raw_telemetry.jsonl",
        "journal.jsonl",
        "run_header.json",
        *(f"predictions_{arm}.jsonl" for arm in ARMS),
    ]
    runner_manifest["configuration_lock"] = configuration
    runner_manifest["frozen_config"] = {
        "path": str(frozen_path.resolve()),
        "sha256": _sha(frozen_path),
    }
    runner_manifest["outputs_sha256"] = {
        name: _sha(runner_dir / name) for name in output_names
    }
    _write_json(runner_manifest_path, runner_manifest)
    fixture["v2_gate_report"] = gate_report_path
    fixture["v2_frozen"] = frozen_path
    fixture["v2_train_manifest"] = train_manifest_path
    return fixture


def _rebind_v2_frozen_to_runner(fixture: dict[str, Any]) -> None:
    runner_dir = fixture["runner"]
    frozen_path = fixture["v2_frozen"]
    header_path = runner_dir / "run_header.json"
    header = json.loads(header_path.read_text(encoding="utf-8"))
    header["frozen_config_sha256"] = _sha(frozen_path)
    _write_json(header_path, header)
    manifest_path = runner_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["frozen_config"]["sha256"] = _sha(frozen_path)
    manifest["outputs_sha256"]["run_header.json"] = _sha(header_path)
    _write_json(manifest_path, manifest)


def _rebind_v2_gate_report(fixture: dict[str, Any]) -> None:
    report_path = fixture["v2_gate_report"]
    frozen_path = fixture["v2_frozen"]
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    frozen["train_admission_gate"]["report"]["sha256"] = _sha(report_path)
    frozen["train_admission_gate_sha256"] = _sha256_bytes(
        _canonical(frozen["train_admission_gate"]).encode("utf-8")
    )
    _write_json(frozen_path, frozen)
    _rebind_v2_frozen_to_runner(fixture)


def test_summary_reconciles_shared_calls_quality_efficiency_and_cost(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "summary"

    summarize(
        runner_dir=fixture["runner"],
        corpus_path=fixture["corpus"],
        claims_path=fixture["claims"],
        retrieval_path=fixture["retrieval"],
        score_directories=fixture["scores"],
        output=output,
        bootstrap_samples=100,
        bootstrap_seed=7,
    )

    result = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
    assert result["official_quality"]["main_delta_all_official_metrics"][
        "abstract_rationalized_f1"
    ] == -1.0
    bootstrap = result["official_quality"]["paired_bootstrap"]
    assert bootstrap["abstract_rationalized_f1"]["samples"] == 100
    assert bootstrap["abstract_rationalized_f1"]["seed"] == 7
    efficiency = result["efficiency"]
    assert efficiency["logical_arms"]["single_one_pass"]["total_tokens"] == 24
    assert efficiency["logical_arms"]["single_self_review"]["total_tokens"] == 54
    assert efficiency["logical_arms"]["judge_challenger"]["total_tokens"] == 60
    assert efficiency["logical_arms"]["single_self_review"][
        "logical_call_count"
    ] == 4
    assert efficiency["physical_run"]["physical_call_count"] == 6
    assert efficiency["physical_run"]["total_tokens"] == 90
    assert efficiency["physical_run"]["shared_initial_stage_latency"][
        "end_to_end_p95_ms"
    ] == 1.5
    assert efficiency["logical_arms"]["single_self_review"][
        "active_path_latency"
    ]["p95_ms"] == 4.0
    assert efficiency["logical_arms"]["judge_challenger"][
        "active_path_latency"
    ]["p95_ms"] == 5.0
    assert efficiency["entire_run_wall_time_seconds"] == 1.0
    assert result["failure_accounting"]["physical_format_failure_rate"] == 1 / 6
    assert result["failure_accounting"]["final_format_failure_rate_by_arm"][
        "judge_challenger"
    ] == 0.5
    assert result["cost"] == {
        "external_model_api_calls": 0,
        "external_api_spend_usd": 0.0,
        "compute_cost_usd": None,
        "total_cost_usd": None,
        "note": "Local electricity and hardware depreciation were not measured.",
    }
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["outputs_sha256"]["comparison.json"] == _sha(
        output / "comparison.json"
    )


def test_summary_accepts_strict_quality_gated_decision_v2(
    tmp_path: Path,
) -> None:
    fixture = _upgrade_fixture_to_v2(_fixture(tmp_path), tmp_path)
    output = tmp_path / "summary_v2"

    summarize(
        runner_dir=fixture["runner"],
        corpus_path=fixture["corpus"],
        claims_path=fixture["claims"],
        retrieval_path=fixture["retrieval"],
        score_directories=fixture["scores"],
        output=output,
        bootstrap_samples=10,
    )

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    frozen = manifest["inputs"]["runner"]["frozen_config"]
    assert frozen["schema_version"] == "scifact-agent-ab-frozen-config-v2"
    assert frozen["train_admission_report"]["sha256"] == _sha(
        fixture["v2_gate_report"]
    )


def test_v2_summary_rejects_missing_per_call_generation_override(
    tmp_path: Path,
) -> None:
    fixture = _upgrade_fixture_to_v2(_fixture(tmp_path), tmp_path)
    telemetry_path = fixture["runner"] / "raw_telemetry.jsonl"
    telemetry = [
        json.loads(line)
        for line in telemetry_path.read_text(encoding="utf-8").splitlines()
    ]
    telemetry[0].pop("generation_overrides")
    _write_jsonl(telemetry_path, telemetry)
    manifest_path = fixture["runner"] / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["outputs_sha256"]["raw_telemetry.jsonl"] = _sha(telemetry_path)
    _write_json(manifest_path, manifest)

    with pytest.raises(AgentABSummaryError, match="generation override"):
        summarize(
            runner_dir=fixture["runner"],
            corpus_path=fixture["corpus"],
            claims_path=fixture["claims"],
            retrieval_path=fixture["retrieval"],
            score_directories=fixture["scores"],
            output=tmp_path / "summary_v2",
            bootstrap_samples=10,
        )


def test_v2_summary_rejects_forged_type_matched_stopper_marker(
    tmp_path: Path,
) -> None:
    fixture = _upgrade_fixture_to_v2(_fixture(tmp_path), tmp_path)
    telemetry_path = fixture["runner"] / "raw_telemetry.jsonl"
    telemetry = [
        json.loads(line)
        for line in telemetry_path.read_text(encoding="utf-8").splitlines()
    ]
    telemetry[0]["json_root_stopping_triggered"] = False
    _write_jsonl(telemetry_path, telemetry)
    manifest_path = fixture["runner"] / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["outputs_sha256"]["raw_telemetry.jsonl"] = _sha(telemetry_path)
    _write_json(manifest_path, manifest)

    with pytest.raises(AgentABSummaryError, match="type-matched JSON stopping"):
        summarize(
            runner_dir=fixture["runner"],
            corpus_path=fixture["corpus"],
            claims_path=fixture["claims"],
            retrieval_path=fixture["retrieval"],
            score_directories=fixture["scores"],
            output=tmp_path / "summary_v2",
            bootstrap_samples=10,
        )


def test_v2_summary_rejects_rehashed_gate_report_contract(
    tmp_path: Path,
) -> None:
    fixture = _upgrade_fixture_to_v2(_fixture(tmp_path), tmp_path)
    report_path = fixture["v2_gate_report"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["scope"] = "smoke_subset"
    _write_json(report_path, report)
    _rebind_v2_gate_report(fixture)

    with pytest.raises(AgentABSummaryError, match="report contract"):
        summarize(
            runner_dir=fixture["runner"],
            corpus_path=fixture["corpus"],
            claims_path=fixture["claims"],
            retrieval_path=fixture["retrieval"],
            score_directories=fixture["scores"],
            output=tmp_path / "summary_v2",
            bootstrap_samples=10,
        )


def test_v2_summary_rejects_rehashed_gate_implementation(
    tmp_path: Path,
) -> None:
    fixture = _upgrade_fixture_to_v2(_fixture(tmp_path), tmp_path)
    report_path = fixture["v2_gate_report"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["implementation"]["sha256"] = "0" * 64
    _write_json(report_path, report)
    _rebind_v2_gate_report(fixture)

    with pytest.raises(AgentABSummaryError, match="implementation hash"):
        summarize(
            runner_dir=fixture["runner"],
            corpus_path=fixture["corpus"],
            claims_path=fixture["claims"],
            retrieval_path=fixture["retrieval"],
            score_directories=fixture["scores"],
            output=tmp_path / "summary_v2",
            bootstrap_samples=10,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("repetition_penalty", 1.1, "repetition-penalty"),
        (
            "repetition_penalty_source",
            "model_default",
            "repetition-penalty",
        ),
    ],
)
def test_v2_summary_rejects_generation_lock_drift(
    tmp_path: Path, field: str, value: Any, message: str
) -> None:
    fixture = _upgrade_fixture_to_v2(_fixture(tmp_path), tmp_path)
    manifest_path = fixture["runner"] / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["configuration_lock"]["generation"][field] = value
    _write_json(manifest_path, manifest)

    with pytest.raises(AgentABSummaryError, match=message):
        summarize(
            runner_dir=fixture["runner"],
            corpus_path=fixture["corpus"],
            claims_path=fixture["claims"],
            retrieval_path=fixture["retrieval"],
            score_directories=fixture["scores"],
            output=tmp_path / "summary_v2",
            bootstrap_samples=10,
        )


def test_v2_summary_rejects_base_runner_hash_drift(tmp_path: Path) -> None:
    fixture = _upgrade_fixture_to_v2(_fixture(tmp_path), tmp_path)
    manifest_path = fixture["runner"] / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["configuration_lock"]["implementation_dependencies"][
        "audited_runner_base"
    ]["sha256"] = "0" * 64
    _write_json(manifest_path, manifest)

    with pytest.raises(AgentABSummaryError, match="implementation hash"):
        summarize(
            runner_dir=fixture["runner"],
            corpus_path=fixture["corpus"],
            claims_path=fixture["claims"],
            retrieval_path=fixture["retrieval"],
            score_directories=fixture["scores"],
            output=tmp_path / "summary_v2",
            bootstrap_samples=10,
        )


def test_initial_draft_rank_nei_and_projection_contract() -> None:
    allowed = {101: 1, 202: 2, 303: 1}
    parsed = _parse_initial_draft(
        '{"analysis":"checked","candidates":[[2,"CONTRADICT",[1]]]}',
        allowed,
    )
    assert parsed["valid"] is True
    assert _draft_to_evidence(parsed["value"], list(allowed)) == {
        "evidence": {"202": {"label": "CONTRADICT", "sentences": [1]}}
    }
    nei = _parse_initial_draft(
        '{"analysis":"checked","candidates":[[1,"NEI",[]]]}', allowed
    )
    assert nei["valid"] is True
    assert _draft_to_evidence(nei["value"], list(allowed)) == {"evidence": {}}
    mixed = _parse_initial_draft(
        '{"analysis":"checked","candidates":'
        '[[1,"NEI",[]],[2,"SUPPORT",[0]]]}',
        allowed,
    )
    assert mixed["failure_code"] == "mixed_nei_and_relation_candidates"
    unhashable_label = _parse_initial_draft(
        '{"analysis":"checked","candidates":[[1,["SUPPORT"],[0]]]}',
        allowed,
    )
    assert unhashable_label["failure_code"] == "invalid_draft_label"


@pytest.mark.parametrize(
    ("raw", "expected_failure"),
    [
        ('{"label":["SUPPORT"],"citations":[]}', "invalid_label"),
        ('{"label":"SUPPORT","citations":[]}', "relation_without_citations"),
        (
            '{"label":"NEI","citations":[[1,[0]]]}',
            "nei_with_citations",
        ),
        (
            '{"label":"SUPPORT","citations":[[true,[0]]]}',
            "citation_rank_out_of_scope",
        ),
        (
            '{"label":"SUPPORT","citations":[[1,[0]],[1,[0]]]}',
            "duplicate_citation_rank",
        ),
    ],
)
def test_v2_summary_parser_independently_rejects_invalid_decisions(
    raw: str, expected_failure: str
) -> None:
    parsed = _parse_v2_agent_output(raw, {1: 1, 2: 1, 3: 1})

    assert parsed["valid"] is False
    assert parsed["failure_code"] == expected_failure
    assert parsed["projected_evidence"] is None


@pytest.mark.parametrize(
    ("raw", "closed"),
    [
        ('{"label":"NEI","citations":[]}', True),
        ('{"label":"NEI","citations":[1}}', False),
        ('{"label":"NEI","citations":[]', False),
        ('["label","NEI"]', False),
    ],
)
def test_v2_summary_stopper_requires_matched_container_types(
    raw: str, closed: bool
) -> None:
    assert _v2_json_root_closed(raw) is closed


def test_summary_rejects_a_tampered_runner_output(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    cases_path = fixture["runner"] / "cases.jsonl"
    cases_path.write_text(
        cases_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )

    with pytest.raises(AgentABSummaryError, match="runner output hash mismatch"):
        summarize(
            runner_dir=fixture["runner"],
            corpus_path=fixture["corpus"],
            claims_path=fixture["claims"],
            retrieval_path=fixture["retrieval"],
            score_directories=fixture["scores"],
            output=tmp_path / "summary",
            bootstrap_samples=10,
        )


def test_summary_rejects_rehashed_inconsistent_telemetry(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    telemetry_path = fixture["runner"] / "raw_telemetry.jsonl"
    telemetry = [
        json.loads(line)
        for line in telemetry_path.read_text(encoding="utf-8").splitlines()
    ]
    telemetry[0]["generated_fragment_sha256"] = "0" * 64
    _write_jsonl(telemetry_path, telemetry)
    manifest_path = fixture["runner"] / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["outputs_sha256"]["raw_telemetry.jsonl"] = _sha(telemetry_path)
    _write_json(manifest_path, manifest)

    with pytest.raises(AgentABSummaryError, match="generated fragment hash mismatch"):
        summarize(
            runner_dir=fixture["runner"],
            corpus_path=fixture["corpus"],
            claims_path=fixture["claims"],
            retrieval_path=fixture["retrieval"],
            score_directories=fixture["scores"],
            output=tmp_path / "summary",
            bootstrap_samples=10,
        )


def test_summary_refuses_to_overwrite_a_completed_output(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "summary"
    output.mkdir()

    with pytest.raises(AgentABSummaryError, match="refusing to overwrite"):
        summarize(
            runner_dir=fixture["runner"],
            corpus_path=fixture["corpus"],
            claims_path=fixture["claims"],
            retrieval_path=fixture["retrieval"],
            score_directories=fixture["scores"],
            output=output,
            bootstrap_samples=10,
        )
