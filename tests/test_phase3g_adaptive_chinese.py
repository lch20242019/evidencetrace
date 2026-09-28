# ruff: noqa: RUF001

from __future__ import annotations

import hashlib
from typing import Any

import pytest
from pydantic import ValidationError

from evidencetrace.audit_models import EvidenceChunk
from evidencetrace.cache import cache_key
from evidencetrace.checks.deterministic import (
    is_high_confidence_subjective,
)
from evidencetrace.eval.baselines import (
    build_evidence_chunks,
    run_baseline,
)
from evidencetrace.eval.models import EvalCase, EvalPrediction, SourceFixture
from evidencetrace.eval.router import AdaptiveRouterConfig
from evidencetrace.model_client import ModelCallTelemetry
from evidencetrace.retrieval.rank import LexicalRetriever


def _source(
    content: str,
    *,
    source_id: str = "qingluan_manual",
    available: bool = True,
) -> SourceFixture:
    return SourceFixture(
        source_id=source_id,
        url=f"https://phase3g-dev.invalid/{source_id}",
        content=content,
        provenance="independent Phase 3G Chinese test fixture",
        content_hash=hashlib.sha256(content.encode()).hexdigest(),
        available=available,
    )


def _case(
    claim: str,
    source: SourceFixture,
    *,
    relation: str = "entailed",
    evidence: str | None = None,
) -> EvalCase:
    substantive = relation in {
        "entailed",
        "partially_entailed",
        "contradicted",
    }
    return EvalCase(
        case_id="phase3g_zh_case",
        claim_text=claim,
        source_fixture="sources.jsonl",
        source_id=source.source_id,
        source_url=source.url,
        gold_relation=relation,
        gold_evidence_span=(evidence or source.content) if substantive else None,
        claim_type="factual_statement",
        mutation_type="none",
        split="dev",
        provenance="independent Phase 3G Chinese test fixture",
        annotation_status="provisional",
        annotation_notes="Deterministic development contract.",
    )


def _chunk(text: str, index: int) -> EvidenceChunk:
    return EvidenceChunk(
        source_id="cjk_source",
        url="https://phase3g-dev.invalid/cjk",
        text=text,
        locator=f"chunk-{index}",
        char_start=index * 100,
        char_end=index * 100 + len(text),
    )


class RecordingClient:
    model_id = "offline-fake"
    prompt_version = "offline-fake-v1"
    provider_id = "offline"
    temperature = 0.0

    def __init__(self) -> None:
        self.tasks: list[str] = []
        self.telemetry_events: tuple[ModelCallTelemetry, ...] = ()

    def complete_model(
        self, task: str, payload: dict[str, Any], schema: type[Any]
    ) -> Any:
        self.tasks.append(task)
        self.telemetry_events += (
            ModelCallTelemetry(latency_ms=1.0, usage_status="missing"),
        )
        if task == "single_agent_full_source_verification":
            return schema.model_validate(
                {
                    "relation": "entailed",
                    "confidence": 0.8,
                    "reason": "Offline full-context fixture result.",
                    "evidence_span": payload["source"]["content"],
                }
            )
        evidence = payload["evidence"][0]
        return schema.model_validate(
            {
                "relation": "entailed",
                "confidence": 0.8,
                "evidence_span": evidence["text"],
                "reason": "Offline retrieval fixture result.",
            }
        )


def test_unspaced_chinese_uses_ngram_fallback_and_ranks_matching_chunk() -> None:
    chunks = (
        _chunk("苍衡引擎记录磁盘清理策略。", 0),
        _chunk("苍衡引擎支持离线批次校验。", 1),
    )
    results = LexicalRetriever(chunks, neighbor_window=0).search(
        "离线批次校验", top_k=2
    )
    assert results
    assert results[0].text == chunks[1].text


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("星澜 CoreX 采用冷启动模式", "CoreX"),
        ("版本 v7.41.6 于 2031-09-17 启用", "v7.41.6"),
        ("版本 v7.41.6 于 2031 年 9 月 17 日启用", "2031-09-17"),
    ],
)
def test_mixed_chinese_and_structured_slots_are_retained(
    query: str, expected: str
) -> None:
    chunks = (
        _chunk("星澜 CoreX 采用热启动模式。", 0),
        _chunk("星澜 CoreX 的版本 v7.41.6 于 2031-09-17 启用冷启动。", 1),
    )
    result = LexicalRetriever(chunks, neighbor_window=0).search(query, top_k=1)
    assert result and expected in result[0].text


def test_cjk_features_remain_active_with_multiple_latin_tokens() -> None:
    """A Chinese query must not lose its CJK signal beside product/version terms."""

    chunks = (
        _chunk("星澄 CoreX v7.41.6 采用热启动模式。", 0),
        _chunk("星澄 CoreX v7.41.6 采用冷启动模式。", 1),
    )

    result = LexicalRetriever(chunks, neighbor_window=0).search(
        "星澄 CoreX v7.41.6 采用冷启动模式", top_k=1
    )

    assert result
    assert result[0].text == chunks[1].text


def test_one_nonempty_chunk_survives_lexical_mismatch() -> None:
    only = _chunk("澄镜节点仅记录校验日志。", 0)
    result = LexicalRetriever((only,), neighbor_window=0).search(
        "完全无关的查询词", top_k=5
    )
    assert len(result) == 1
    assert result[0].chunk == only


def test_english_ranking_regression_empty_and_unavailable_inputs() -> None:
    chunks = (
        _chunk("Orchid records audit logs.", 0),
        _chunk("Orchid supports offline validation.", 1),
    )
    result = LexicalRetriever(chunks, neighbor_window=0).search(
        "offline validation", top_k=1
    )
    assert result[0].text == chunks[1].text
    assert LexicalRetriever((), neighbor_window=0).search("离线校验") == ()
    unavailable = _source("", available=False)
    prediction = run_baseline(
        _case("来源不可用。", unavailable, relation="source_unavailable"),
        unavailable,
        "retrieval_judge_deterministic",
    )
    assert prediction.predicted_relation.value == "source_unavailable"


@pytest.mark.parametrize(
    "text",
    [
        "苍衡是最好用的调度器。",
        "这套方案最值得推荐。",
        "操作者应该选择更容易使用的界面。",
        "霁光工具是最理想的选择。",
    ],
)
def test_chinese_subjective_claims_are_high_confidence_not_checkable(
    text: str,
) -> None:
    assert is_high_confidence_subjective(text)


@pytest.mark.parametrize(
    "text",
    [
        "在 JadeBench 指标中，苍衡吞吐量最高，为 947 QPS。",
        "手册把霁光列为平台兼容性排名第一。",
        "版本 v7.41.6 应该在 2031-09-17 启用。",
        "该平台容量上限为 347 GB，值得推荐。",
        "青衡应该支持 Linux 平台。",
    ],
)
def test_objective_slots_prevent_subjective_short_circuit(text: str) -> None:
    assert not is_high_confidence_subjective(text)


def test_adaptive_routes_unavailable_and_subjective_without_calls() -> None:
    client = RecordingClient()
    unavailable = _source("", available=False)
    unavailable_result = run_baseline(
        _case("无法取得来源。", unavailable, relation="source_unavailable"),
        unavailable,
        "adaptive_live",
        client=client,
    )
    subjective_source = _source("霁光面板提供三种布局。")
    subjective_result = run_baseline(
        _case("霁光面板是最理想的选择。", subjective_source, relation="not_checkable"),
        subjective_source,
        "adaptive_live",
        client=client,
    )
    assert unavailable_result.selected_route == "deterministic_source_unavailable"
    assert subjective_result.selected_route == "deterministic_not_checkable"
    assert unavailable_result.model_calls == subjective_result.model_calls == 0
    assert client.tasks == []


def test_adaptive_uses_full_context_for_one_safe_chunk() -> None:
    source = _source("青鸾模块启用本地校验。")
    client = RecordingClient()
    result = run_baseline(
        _case("青鸾模块启用本地校验。", source),
        source,
        "adaptive_live",
        client=client,
    )
    assert result.selected_route == "full_context_single_agent"
    assert result.route_reason_code == "single_chunk_within_context_budget"
    assert result.chunk_count == 1
    assert result.estimated_context_size
    assert result.model_calls == 1
    assert client.tasks == ["single_agent_full_source_verification"]


@pytest.mark.parametrize(
    ("content", "budget", "reason"),
    [
        (
            "青鸾模块记录审计日志。\n\n青鸾模块启用本地校验。",
            32_768,
            "multi_chunk_source",
        ),
        ("青鸾模块启用本地校验。", 1, "context_budget_exceeded"),
    ],
)
def test_adaptive_routes_multi_chunk_or_over_budget_to_judge(
    content: str, budget: int, reason: str
) -> None:
    source = _source(content)
    client = RecordingClient()
    result = run_baseline(
        _case("青鸾模块启用本地校验。", source, evidence="青鸾模块启用本地校验。"),
        source,
        "adaptive_live",
        client=client,
        router_config=AdaptiveRouterConfig(full_context_budget_bytes=budget),
    )
    assert result.selected_route == "retrieval_judge"
    assert result.route_reason_code == reason
    assert result.model_calls == 1
    assert client.tasks == ["claim_judgement"]


def test_chinese_chunking_and_route_telemetry_are_stable() -> None:
    source = _source("第一段记录审计。\n\n第二段说明离线校验。")
    chunks = build_evidence_chunks(source)
    assert [chunk.text for chunk in chunks] == [
        "第一段记录审计。",
        "第二段说明离线校验。",
    ]
    payload = EvalPrediction(
        case_id="legacy",
        baseline="single_agent_live",
        predicted_relation="not_in_source",
        confidence=0.0,
        model_calls=0,
        latency_ms=0.0,
    ).model_dump()
    payload["selected_route"] = "retrieval_judge"
    with pytest.raises(ValidationError, match="reserved"):
        EvalPrediction.model_validate(payload)


def test_policy_versions_invalidate_cache_and_historical_prediction_loads() -> None:
    base = {
        "source_hash": "source",
        "claim_hash": "claim",
        "model_id": "model",
        "prompt_version": "prompt",
        "retrieval_config_version": "retrieval-config",
    }
    current = cache_key(**base)
    assert current != cache_key(**base, retrieval_policy_version="older")
    assert current != cache_key(**base, router_policy_version="older")
    assert current != cache_key(**base, checkability_policy_version="older")
    assert current != cache_key(
        **base,
        schema_recovery_policy_version="schema-recovery-disabled-v0",
    )
    assert current != cache_key(
        **base,
        judge_verdict_ownership_policy_version="model-owned-verdict-v0",
    )
    historical = {
        "case_id": "historical",
        "baseline": "retrieval_judge_live",
        "predicted_relation": "not_in_source",
        "confidence": 0.0,
        "model_calls": 0,
        "latency_ms": 0.0,
    }
    assert EvalPrediction.model_validate(historical).selected_route is None
