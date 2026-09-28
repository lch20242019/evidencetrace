"""Two-call planning Agent with no tool or budget authority."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from evidencetrace.model_client import ModelClient

COORDINATOR_CONTRACT_VERSION = "audit-coordinator-v1"
COORDINATOR_CALL_LIMIT = 2


class PlanStage(StrEnum):
    INITIAL = "initial"
    REVIEW = "review"


class PlanAction(StrEnum):
    VERIFY_CITATION = "verify_citation"
    DISCOVER = "discover"
    SKIP_NOT_CHECKABLE = "skip_not_checkable"
    REQUEST_HUMAN = "request_human"
    ACCEPT = "accept"
    CHALLENGE = "challenge"
    COUNTER_SEARCH = "counter_search"


INITIAL_ACTIONS = frozenset(
    {
        PlanAction.VERIFY_CITATION,
        PlanAction.DISCOVER,
        PlanAction.SKIP_NOT_CHECKABLE,
        PlanAction.REQUEST_HUMAN,
    }
)
REVIEW_ACTIONS = frozenset(
    {
        PlanAction.ACCEPT,
        PlanAction.CHALLENGE,
        PlanAction.COUNTER_SEARCH,
        PlanAction.REQUEST_HUMAN,
    }
)


class PlanTask(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    claim_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
    action: PlanAction


class ExecutionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    stage: PlanStage
    tasks: tuple[PlanTask, ...]

    @model_validator(mode="after")
    def valid_actions(self) -> ExecutionPlan:
        allowed = INITIAL_ACTIONS if self.stage is PlanStage.INITIAL else REVIEW_ACTIONS
        if any(task.action not in allowed for task in self.tasks):
            raise ValueError("action is invalid for plan stage")
        return self


class CoordinatorUnavailableError(RuntimeError):
    pass


class AuditCoordinatorAgent:
    def __init__(self, client: ModelClient | None = None) -> None:
        self.client = client
        self.calls = 0

    def plan(
        self,
        stage: PlanStage,
        claims: list[dict[str, Any]],
        *,
        discover_enabled: bool,
        discovery_limit: int,
    ) -> ExecutionPlan:
        if self.calls >= COORDINATOR_CALL_LIMIT:
            raise CoordinatorUnavailableError("coordinator call limit exhausted")
        self.calls += 1
        if self.client is None:
            raise CoordinatorUnavailableError("coordinator model is unavailable")
        allowed = INITIAL_ACTIONS if stage is PlanStage.INITIAL else REVIEW_ACTIONS
        return ExecutionPlan.model_validate(
            self.client.complete_model(
                f"audit_coordination_{stage.value}",
                {
                    "contract_version": COORDINATOR_CONTRACT_VERSION,
                    "stage": stage.value,
                    "allowed_actions": sorted(item.value for item in allowed),
                    "discover_enabled": discover_enabled,
                    "discovery_limit": discovery_limit,
                    "claims": claims,
                },
                ExecutionPlan,
            )
        )


__all__ = [
    "COORDINATOR_CALL_LIMIT",
    "COORDINATOR_CONTRACT_VERSION",
    "INITIAL_ACTIONS",
    "REVIEW_ACTIONS",
    "AuditCoordinatorAgent",
    "CoordinatorUnavailableError",
    "ExecutionPlan",
    "PlanAction",
    "PlanStage",
    "PlanTask",
]
