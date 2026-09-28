"""Bounded Scout plans plus Controller-owned search and safe fetch."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator

from evidencetrace.audit_models import RetrievedEvidence
from evidencetrace.model_client import ModelClient
from evidencetrace.models import AtomicClaim, SourceMetadata
from evidencetrace.retrieval.extract import extract_html
from evidencetrace.retrieval.fetch import FetchError, SafeFetcher
from evidencetrace.retrieval.rank import LexicalRetriever

SCOUT_CONTRACT_VERSION = "evidence-scout-v1"
RETRIEVAL_FALLBACK_REASON = "empty_retrieval_full_context"
MAX_QUERIES_PER_CLAIM = 2
MAX_CANDIDATE_URLS_PER_CLAIM = 5
MAX_FETCHES_PER_CLAIM = 3
_TOKEN_RE = re.compile(
    r"[A-Za-z0-9]+(?:[._-][A-Za-z0-9]+)*"
    r"|[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+"
)
_GENERIC = {
    "benchmark",
    "documentation",
    "docs",
    "evidence",
    "manual",
    "notes",
    "official",
    "release",
    "source",
    "specification",
    "官方",
    "发布",
    "文档",
    "来源",
    "规范",
    "说明",
}


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SearchResult(_Model):
    url: str = Field(min_length=1, max_length=2048)
    snippet: str = Field(default="", max_length=2000)


class SearchResponse(_Model):
    status: Literal["ok", "discovery_unavailable"]
    results: tuple[SearchResult, ...] = Field(default=(), max_length=5)


class SearchClient(Protocol):
    def search(self, query: str, *, max_results: int) -> SearchResponse: ...


class SearchPlan(_Model):
    queries: tuple[str, ...] = Field(min_length=1, max_length=2)

    @field_validator("queries")
    @classmethod
    def valid_queries(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        values = tuple(item.strip() for item in values)
        if any(not item or len(item) > 500 for item in values):
            raise ValueError("query must be non-empty and bounded")
        if len(values) != len(set(values)):
            raise ValueError("queries must be unique")
        return values


class ScoutPlanError(RuntimeError):
    pass


class TavilySearchClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        timeout: float = 15.0,
        endpoint: str = "https://api.tavily.com/search",
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._api_key = api_key if api_key is not None else os.getenv("TAVILY_API_KEY")
        self.timeout, self.endpoint, self.transport = timeout, endpoint, transport

    def search(self, query: str, *, max_results: int) -> SearchResponse:
        if not self._api_key:
            return SearchResponse(status="discovery_unavailable")
        limit = max(1, min(max_results, 5))
        try:
            with httpx.Client(
                timeout=self.timeout, transport=self.transport, trust_env=False
            ) as client:
                response = client.post(
                    self.endpoint,
                    json={
                        "api_key": self._api_key,
                        "query": query,
                        "max_results": limit,
                        "search_depth": "basic",
                        "include_answer": False,
                        "include_raw_content": False,
                    },
                )
                response.raise_for_status()
                raw = response.json()
        except (httpx.HTTPError, ValueError):
            return SearchResponse(status="discovery_unavailable")
        if not isinstance(raw, dict) or not isinstance(raw.get("results"), list):
            return SearchResponse(status="discovery_unavailable")
        results = []
        for item in raw["results"][:limit]:
            if not isinstance(item, dict) or not isinstance(item.get("url"), str):
                continue
            snippet = item.get("content", "")
            try:
                results.append(
                    SearchResult(
                        url=item["url"],
                        snippet=snippet if isinstance(snippet, str) else "",
                    )
                )
            except ValueError:
                continue
        return SearchResponse(status="ok", results=tuple(results))


class FixtureSearchClient:
    def __init__(self, values: dict[str, tuple[SearchResult, ...]]) -> None:
        self.values = dict(values)
        self.calls: list[str] = []

    @classmethod
    def from_manifest(cls, path: Path) -> FixtureSearchClient:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            {
                item["query"]: tuple(
                    SearchResult.model_validate(value)
                    for value in item.get("results", [])
                )
                for item in raw.get("search_results", [])
                if isinstance(item.get("query"), str)
            }
        )

    def search(self, query: str, *, max_results: int) -> SearchResponse:
        self.calls.append(query)
        return SearchResponse(
            status="ok", results=self.values.get(query, ())[:max_results]
        )


@dataclass(frozen=True)
class DiscoveryCandidate:
    source: SourceMetadata
    evidence: tuple[RetrievedEvidence, ...] = ()


@dataclass(frozen=True)
class DiscoveryResult:
    status: Literal[
        "ok",
        "discovery_unavailable",
        "source_error",
        "evidence_not_found",
    ]
    source: SourceMetadata | None = None
    evidence: tuple[RetrievedEvidence, ...] = ()
    sources: tuple[SourceMetadata, ...] = ()
    query_count: int = 0
    fetch_count: int = 0
    candidates: tuple[DiscoveryCandidate, ...] = ()
    # ``True`` means the lexical retriever found no positive-scoring hit and a
    # bounded full-source candidate was used.  Keep this provenance explicit so
    # callers do not present the zero score as a normal retrieval result.
    retrieval_fallback: bool = False
    fallback_reason: str | None = None


def _terms(value: str) -> set[str]:
    return {item.casefold() for item in _TOKEN_RE.findall(value)}


def validate_search_plan(plan: SearchPlan, claim: AtomicClaim) -> SearchPlan:
    if claim.text not in plan.queries:
        raise ScoutPlanError("original claim query is required")
    allowed = _terms(claim.text) | _GENERIC
    if any(not _terms(query) <= allowed for query in plan.queries):
        raise ScoutPlanError("query introduced new lexical provenance")
    return plan


class EvidenceScoutAgent:
    def __init__(self, client: ModelClient | None = None) -> None:
        self.client = client

    def plan(self, claim: AtomicClaim) -> SearchPlan:
        if self.client is None:
            return SearchPlan(queries=(claim.text,))
        output = self.client.complete_model(
            "evidence_discovery_plan",
            {
                "contract_version": SCOUT_CONTRACT_VERSION,
                "claim": claim.text,
                "maximum_queries": 2,
                "required_original_query": claim.text,
            },
            SearchPlan,
        )
        return validate_search_plan(SearchPlan.model_validate(output), claim)


class DiscoveryTools:
    def __init__(
        self, search_client: SearchClient | None, fetcher: SafeFetcher
    ) -> None:
        self.search_client, self.fetcher = search_client, fetcher

    def discover(self, claim: AtomicClaim, plan: SearchPlan) -> DiscoveryResult:
        plan = validate_search_plan(plan, claim)
        if self.search_client is None:
            return DiscoveryResult(
                "discovery_unavailable", query_count=len(plan.queries)
            )
        candidates: list[str] = []
        unavailable = False
        for query in plan.queries:
            response = self.search_client.search(query, max_results=5)
            unavailable |= response.status == "discovery_unavailable"
            for result in response.results:
                if result.url not in candidates and len(candidates) < 5:
                    candidates.append(result.url)
        if not candidates and unavailable:
            return DiscoveryResult(
                "discovery_unavailable", query_count=len(plan.queries)
            )
        return self.resolve_urls(
            claim, tuple(candidates), query_count=len(plan.queries)
        )

    def resolve_urls(
        self,
        claim: AtomicClaim,
        urls: tuple[str, ...],
        *,
        query_count: int = 0,
    ) -> DiscoveryResult:
        candidates = tuple(dict.fromkeys(urls))[:5]
        sources: list[SourceMetadata] = []
        candidate_evidence: list[DiscoveryCandidate] = []
        best_source: SourceMetadata | None = None
        best_evidence: tuple[RetrievedEvidence, ...] = ()
        fetch_count = 0
        for url in candidates[:3]:
            fetch_count += 1
            try:
                fetched = self.fetcher.fetch(url)
                source_id = (
                    "s_" + hashlib.sha256(fetched.final_url.encode()).hexdigest()[:12]
                )
                source, chunks = extract_html(fetched, source_id=source_id)
            except (FetchError, ValueError):
                continue
            sources.append(source)
            retriever = LexicalRetriever(chunks)
            evidence = retriever.search(claim.text, top_k=5)
            if not evidence:
                fallback = retriever.full_context_fallback()
                if fallback is not None:
                    evidence = (fallback,)
            candidate_evidence.append(DiscoveryCandidate(source, evidence))
            if best_source is None or (
                evidence
                and (not best_evidence or evidence[0].score > best_evidence[0].score)
            ):
                best_source, best_evidence = source, evidence
        status: Literal["ok", "source_error", "evidence_not_found"]
        if best_source is None:
            status = "source_error"
        else:
            status = "ok" if best_evidence else "evidence_not_found"
        best_fallback = bool(
            best_evidence
            and best_evidence[0].score == 0.0
            and best_source is not None
        )
        return DiscoveryResult(
            status,
            best_source,
            best_evidence,
            tuple(sources),
            query_count,
            fetch_count,
            tuple(candidate_evidence),
            retrieval_fallback=best_fallback,
            fallback_reason=(RETRIEVAL_FALLBACK_REASON if best_fallback else None),
        )


__all__ = [
    "MAX_CANDIDATE_URLS_PER_CLAIM",
    "MAX_FETCHES_PER_CLAIM",
    "MAX_QUERIES_PER_CLAIM",
    "RETRIEVAL_FALLBACK_REASON",
    "DiscoveryCandidate",
    "DiscoveryResult",
    "DiscoveryTools",
    "EvidenceScoutAgent",
    "FixtureSearchClient",
    "ScoutPlanError",
    "SearchClient",
    "SearchPlan",
    "SearchResponse",
    "SearchResult",
    "TavilySearchClient",
    "validate_search_plan",
]
