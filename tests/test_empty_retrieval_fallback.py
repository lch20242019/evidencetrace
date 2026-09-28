from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import httpx

from evidencetrace.agents.scout import (
    RETRIEVAL_FALLBACK_REASON,
    DiscoveryTools,
)
from evidencetrace.audit_models import EvidenceChunk
from evidencetrace.eval.baselines import run_baseline
from evidencetrace.eval.models import EvalCase, SourceFixture
from evidencetrace.eval.router import AdaptiveRouterConfig
from evidencetrace.model_client import ModelCallTelemetry
from evidencetrace.models import AtomicClaim, Checkability, Relation
from evidencetrace.product import AgentName, ProductAuditPipeline, TraceState
from evidencetrace.retrieval.fetch import SafeFetcher
from evidencetrace.retrieval.rank import (
    FULL_CONTEXT_FALLBACK_SCORE,
    LexicalRetriever,
)


def _chunk(text: str, start: int, *, locator: str) -> EvidenceChunk:
    return EvidenceChunk(
        source_id="source",
        url="https://fallback.example.test/source",
        text=text,
        locator=locator,
        char_start=start,
        char_end=start + len(text),
    )


def test_full_context_fallback_is_exact_and_zero_scored() -> None:
    chunks = (
        _chunk("Orchid records telemetry.", 0, locator="paragraph 1"),
        _chunk("Orchid stores it offline.", 27, locator="paragraph 2"),
    )

    result = LexicalRetriever(chunks, neighbor_window=0).search(
        "Zeta quantum ledger",
        top_k=5,
        fallback_to_full_context=True,
    )

    assert len(result) == 1
    assert result[0].score == FULL_CONTEXT_FALLBACK_SCORE == 0.0
    assert result[0].text == "Orchid records telemetry.\n\nOrchid stores it offline."
    assert result[0].text == result[0].chunk.text
    assert result[0].chunk.char_start == 0
    assert result[0].chunk.char_end == len(result[0].text)
    assert "full-source fallback" in result[0].chunk.locator


def test_full_context_fallback_rejects_discontinuous_chunks() -> None:
    chunks = (
        _chunk("Orchid records telemetry.", 0, locator="paragraph 1"),
        _chunk("Orchid stores it offline.", 100, locator="paragraph 2"),
    )

    assert (
        LexicalRetriever(chunks, neighbor_window=0).search(
            "Zeta quantum ledger",
            top_k=5,
            fallback_to_full_context=True,
        )
        == ()
    )


def test_cjk_features_are_unionable_with_multiple_latin_tokens() -> None:
    chunks = (
        _chunk("星澄 CoreX v7.41.6 采用热启动模式。", 0, locator="chunk-1"),
        _chunk("星澄 CoreX v7.41.6 采用冷启动模式。", 100, locator="chunk-2"),
    )

    result = LexicalRetriever(chunks, neighbor_window=0).search(
        "星澄 CoreX v7.41.6 采用冷启动模式",
        top_k=1,
    )

    assert result
    assert result[0].text == chunks[1].text


def _source(content: str) -> SourceFixture:
    return SourceFixture(
        source_id="fallback_source",
        url="https://fallback.example.test/source",
        content=content,
        provenance="offline fallback fixture",
        content_hash=hashlib.sha256(content.encode()).hexdigest(),
    )


def _case(claim: str, source: SourceFixture) -> EvalCase:
    return EvalCase(
        case_id="fallback_case",
        claim_text=claim,
        source_fixture="sources.jsonl",
        source_id=source.source_id,
        source_url=source.url,
        gold_relation=Relation.NOT_IN_SOURCE,
        claim_type="factual_statement",
        mutation_type="none",
        split="dev",
        provenance="offline fallback fixture",
        annotation_status="provisional",
        annotation_notes="Unit-test-only route contract.",
    )


class _RecordingModel:
    model_id = "offline-fallback-model"
    prompt_version = "offline-fallback-v1"
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
                    "relation": "not_in_source",
                    "confidence": 0.0,
                    "reason": "The bounded full source does not support the claim.",
                    "evidence_span": None,
                }
            )
        return schema.model_validate(
            {
                "relation": "not_in_source",
                "confidence": 0.0,
                "reason": "No lexical candidate was available.",
                "evidence_span": None,
            }
        )


def test_adaptive_empty_retrieval_uses_full_context_only_within_budget() -> None:
    source = _source("Orchid records telemetry.\n\nOrchid stores it offline.")
    client = _RecordingModel()

    result = run_baseline(
        _case("Zeta supports quantum ledger.", source),
        source,
        "adaptive_live",
        client=client,
    )

    assert result.selected_route == "full_context_single_agent"
    assert result.route_reason_code == "empty_retrieval_full_context_fallback"
    assert result.model_calls == 1
    assert client.tasks == ["single_agent_full_source_verification"]


def test_adaptive_empty_retrieval_does_not_bypass_context_budget() -> None:
    source = _source("Orchid records telemetry and stores it offline.")
    client = _RecordingModel()

    result = run_baseline(
        _case("Zeta supports quantum ledger.", source),
        source,
        "adaptive_live",
        client=client,
        router_config=AdaptiveRouterConfig(full_context_budget_bytes=1),
    )

    assert result.selected_route == "retrieval_judge"
    assert result.route_reason_code == "context_budget_exceeded"
    assert result.model_calls == 1
    assert client.tasks == ["claim_judgement"]


def _claim() -> AtomicClaim:
    return AtomicClaim(
        claim_id="c_fallback",
        text="Zeta supports quantum ledger.",
        file="doc.md",
        line_start=1,
        line_end=1,
        claim_type="factual_statement",
        checkability=Checkability.CHECKABLE,
    )


def _mock_fetcher(content: str) -> SafeFetcher:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            request=request,
            headers={"content-type": "text/html; charset=utf-8"},
            text=f"<main><p>{content}</p></main>",
        )

    return SafeFetcher(
        transport=httpx.MockTransport(handler),
        resolve_dns=False,
        retries=0,
    )


def test_discovery_tools_exposes_empty_retrieval_fallback() -> None:
    url = "https://fallback.example.test/source"
    result = DiscoveryTools(
        None,
        _mock_fetcher("Orchid records glacier telemetry."),
    ).resolve_urls(_claim(), (url,))

    assert result.status == "ok"
    assert result.retrieval_fallback is True
    assert result.fallback_reason == RETRIEVAL_FALLBACK_REASON
    assert len(result.evidence) == 1
    assert result.evidence[0].score == 0.0
    assert result.evidence[0].text == "Orchid records glacier telemetry."


def test_product_pipeline_records_discovery_fallback_trace(tmp_path: Path) -> None:
    url = "https://fallback.example.test/source"
    document = tmp_path / "doc.md"
    document.write_text(
        f"Zeta supports quantum ledger [source]({url}).\n",
        encoding="utf-8",
    )
    pipeline = ProductAuditPipeline(
        project_root=tmp_path,
        fetcher=_mock_fetcher("Orchid records glacier telemetry."),
    )

    result = pipeline.run(document, persist=False)

    assert any(
        event.state is TraceState.FALLBACK
        and event.agent is AgentName.CONTROLLER
        and event.action == "retrieval_empty_full_context_fallback"
        and event.claim_id == "c_0001"
        for event in result.product.trace
    )
    assert any(
        event.state is TraceState.DISPATCHED
        and event.agent is AgentName.JUDGE
        and event.claim_id == "c_0001"
        for event in result.product.trace
    )
