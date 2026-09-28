"""Deterministic routing policy for adaptive claim-source verification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# Bumped with the empty-retrieval fallback route.  The router version is part
# of cache keys and run manifests, so old adaptive predictions must not be
# silently reused under the new policy.
ADAPTIVE_ROUTER_POLICY_VERSION = "adaptive-router-v2"
DEFAULT_FULL_CONTEXT_BUDGET_BYTES = 32_768

AdaptiveRoute = Literal[
    "deterministic_source_unavailable",
    "deterministic_not_checkable",
    "full_context_single_agent",
    "retrieval_judge",
]
AdaptiveRouteReason = Literal[
    "source_unavailable",
    "subjective_without_fact_slots",
    "single_chunk_within_context_budget",
    "multi_chunk_source",
    "context_budget_exceeded",
    "empty_available_source",
    "empty_retrieval_full_context_fallback",
]


@dataclass(frozen=True)
class AdaptiveRouterConfig:
    full_context_budget_bytes: int = DEFAULT_FULL_CONTEXT_BUDGET_BYTES
    policy_version: str = ADAPTIVE_ROUTER_POLICY_VERSION

    def __post_init__(self) -> None:
        if self.full_context_budget_bytes <= 0:
            raise ValueError("full-context budget must be positive")
        if not self.policy_version.strip():
            raise ValueError("router policy version must not be blank")


@dataclass(frozen=True)
class AdaptiveRouteDecision:
    selected_route: AdaptiveRoute
    route_reason_code: AdaptiveRouteReason
    chunk_count: int
    estimated_context_size: int


def select_adaptive_route(
    *,
    source_available: bool,
    purely_subjective: bool,
    chunk_count: int,
    estimated_context_size: int,
    config: AdaptiveRouterConfig,
) -> AdaptiveRouteDecision:
    if chunk_count < 0 or estimated_context_size < 0:
        raise ValueError("router measurements cannot be negative")
    if not source_available:
        route: AdaptiveRoute = "deterministic_source_unavailable"
        reason: AdaptiveRouteReason = "source_unavailable"
    elif purely_subjective:
        route = "deterministic_not_checkable"
        reason = "subjective_without_fact_slots"
    elif chunk_count == 0:
        route = "retrieval_judge"
        reason = "empty_available_source"
    elif chunk_count > 1:
        route = "retrieval_judge"
        reason = "multi_chunk_source"
    elif estimated_context_size > config.full_context_budget_bytes:
        route = "retrieval_judge"
        reason = "context_budget_exceeded"
    else:
        route = "full_context_single_agent"
        reason = "single_chunk_within_context_budget"
    return AdaptiveRouteDecision(
        selected_route=route,
        route_reason_code=reason,
        chunk_count=chunk_count,
        estimated_context_size=estimated_context_size,
    )


__all__ = [
    "ADAPTIVE_ROUTER_POLICY_VERSION",
    "DEFAULT_FULL_CONTEXT_BUDGET_BYTES",
    "AdaptiveRoute",
    "AdaptiveRouteDecision",
    "AdaptiveRouteReason",
    "AdaptiveRouterConfig",
    "select_adaptive_route",
]
