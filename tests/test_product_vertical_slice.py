from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import httpx
import pytest

from evidencetrace.agents.challenger import ChallengerAgent
from evidencetrace.agents.coordinator import (
    AuditCoordinatorAgent,
    CoordinatorUnavailableError,
    ExecutionPlan,
    PlanAction,
    PlanStage,
    PlanTask,
)
from evidencetrace.agents.judge import ClaimJudgeAgent
from evidencetrace.agents.scout import (
    DiscoveryTools,
    FixtureSearchClient,
    ScoutPlanError,
    SearchPlan,
    TavilySearchClient,
    validate_search_plan,
)
from evidencetrace.model_client import (
    DeterministicFakeModel,
    ModelCallBudgetExceeded,
    ModelResponseError,
)
from evidencetrace.models import AtomicClaim, Checkability, Relation, Verdict
from evidencetrace.pipeline import fixture_transport
from evidencetrace.product import (
    AgentName,
    ClaimRunStatus,
    DocumentRunStatus,
    ProductAuditPipeline,
    ProductPipelineError,
    TraceState,
    write_suggestion_patch,
)
from evidencetrace.render import render_audit_markdown
from evidencetrace.retrieval.fetch import SafeFetcher
from evidencetrace.sarif import build_sarif

ROOT = Path(__file__).parents[1]
OWNERSHIP_URL = "https://fixtures.evidencetrace.invalid/nimbus/ownership"
BENCHMARK_URL = "https://fixtures.evidencetrace.invalid/nimbus/benchmark-2026"


def _copy_examples(tmp_path: Path) -> Path:
    shutil.copytree(ROOT / "examples", tmp_path / "examples")
    return tmp_path / "examples/evidence/source_manifest.json"


def _pipeline(tmp_path: Path, *, search: bool = False) -> ProductAuditPipeline:
    manifest = tmp_path / "examples/evidence/source_manifest.json"
    return ProductAuditPipeline(
        project_root=tmp_path,
        fetcher=SafeFetcher(
            transport=fixture_transport(manifest),
            resolve_dns=False,
            retries=0,
        ),
        search_client=(FixtureSearchClient.from_manifest(manifest) if search else None),
    )


def _claim(text: str = "Orion ships version 4.2 on 2028-03-04.") -> AtomicClaim:
    return AtomicClaim(
        claim_id="claim_fixture",
        text=text,
        file="doc.md",
        line_start=1,
        line_end=1,
        claim_type="versioned_capability",
        checkability=Checkability.CHECKABLE,
    )


def test_product_demo_exercises_conditional_agent_handoffs(
    tmp_path: Path,
) -> None:
    _copy_examples(tmp_path)
    result = _pipeline(tmp_path, search=True).run(
        tmp_path / "examples/product-demo/README.md",
        discover=True,
        persist=False,
    )

    claims = result.product.claim_runs
    assert [item.initial_action for item in claims] == [
        PlanAction.VERIFY_CITATION,
        PlanAction.DISCOVER,
        PlanAction.VERIFY_CITATION,
        PlanAction.SKIP_NOT_CHECKABLE,
    ]
    dispatched = {
        (event.agent, event.claim_id)
        for event in result.product.trace
        if event.state is TraceState.DISPATCHED
    }
    assert (AgentName.JUDGE, "c_0001") in dispatched
    assert (AgentName.SCOUT, "c_0002") in dispatched
    assert (AgentName.JUDGE, "c_0002") in dispatched
    assert (AgentName.CHALLENGER, "c_0003") in dispatched
    assert (AgentName.JUDGE, "c_0004") not in dispatched
    assert (AgentName.SCOUT, "c_0004") not in dispatched
    assert result.product.budget["coordinator_calls"] == 2
    assert result.product.budget["coordinator_call_limit"] == 2
    assert result.product.budget["search_queries"] == 1
    assert result.product.budget["fetches"] == 3
    assert "Remaining budget:" in result.terminal


def test_claim_failure_isolated_from_other_claims(tmp_path: Path) -> None:
    _copy_examples(tmp_path)
    document = tmp_path / "doc.md"
    document.write_text(
        "The Orion team owns Nimbus "
        f"[in the registry]({OWNERSHIP_URL}).\n\n"
        "Nimbus completed 82% of tasks "
        f"[in the benchmark report]({BENCHMARK_URL}).\n",
        encoding="utf-8",
    )
    pipeline = _pipeline(tmp_path)
    deterministic = ClaimJudgeAgent()

    class OneFailureJudge:
        def judge(self, input_data: Any) -> Any:
            if "Orion" in input_data.claim.text:
                raise RuntimeError("synthetic-private-payload")
            return deterministic.judge(input_data)

    pipeline.judge = OneFailureJudge()  # type: ignore[assignment]
    result = pipeline.run(document, persist=False)

    assert result.product.document_status is DocumentRunStatus.PARTIAL
    assert result.product.claim_runs[0].status is ClaimRunStatus.AGENT_ERROR
    assert result.product.claim_runs[0].error_code == "judge_error"
    assert result.product.claim_runs[1].status is ClaimRunStatus.COMPLETED
    safe_product = result.product.model_dump_json()
    assert "synthetic-private-payload" not in safe_product
    assert OWNERSHIP_URL not in safe_product


def test_model_call_ceiling_becomes_claim_budget_exhaustion(
    tmp_path: Path,
) -> None:
    _copy_examples(tmp_path)
    document = tmp_path / "doc.md"
    document.write_text(
        f"The Orion team owns Nimbus [in the registry]({OWNERSHIP_URL}).\n",
        encoding="utf-8",
    )
    pipeline = _pipeline(tmp_path)

    class ExhaustedJudge:
        def judge(self, _input_data: Any) -> Any:
            raise ModelCallBudgetExceeded("synthetic budget")

    pipeline.judge = ExhaustedJudge()  # type: ignore[assignment]
    result = pipeline.run(document, persist=False)

    assert result.product.claim_runs[0].status is ClaimRunStatus.BUDGET_EXHAUSTED
    assert result.product.claim_runs[0].error_code == "budget_exhausted"
    assert any(
        event.agent is AgentName.JUDGE and event.code == "budget_exhausted"
        for event in result.product.trace
    )


def test_scout_failure_is_safe_and_other_taxonomy_stays_distinct(
    tmp_path: Path,
) -> None:
    _copy_examples(tmp_path)
    document = tmp_path / "doc.md"
    document.write_text("The cache migration starts on 2026-08-04.\n", encoding="utf-8")
    pipeline = _pipeline(tmp_path, search=True)

    class FailedScout:
        def plan(self, _claim: AtomicClaim) -> SearchPlan:
            raise RuntimeError("private-scout-response")

    pipeline.scout = FailedScout()  # type: ignore[assignment]
    result = pipeline.run(document, discover=True, persist=False)

    assert result.product.claim_runs[0].status is ClaimRunStatus.AGENT_ERROR
    assert result.product.claim_runs[0].error_code == "scout_error"
    assert "private-scout-response" not in result.product.model_dump_json()


def test_rogue_coordinator_plan_is_rejected_and_falls_back(
    tmp_path: Path,
) -> None:
    document = tmp_path / "doc.md"
    document.write_text(
        "Teams should prefer Nimbus for every project.\n", encoding="utf-8"
    )
    pipeline = ProductAuditPipeline(project_root=tmp_path)

    class RogueCoordinator:
        calls = 0

        def plan(self, stage: PlanStage, *_args: Any, **_kwargs: Any) -> ExecutionPlan:
            self.calls += 1
            return ExecutionPlan(
                stage=stage,
                tasks=(
                    PlanTask(
                        claim_id="unknown_claim",
                        action=(
                            PlanAction.VERIFY_CITATION
                            if stage is PlanStage.INITIAL
                            else PlanAction.ACCEPT
                        ),
                    ),
                ),
            )

    pipeline.coordinator = RogueCoordinator()  # type: ignore[assignment]
    result = pipeline.run(document, persist=False)
    failures = [
        event
        for event in result.product.trace
        if event.code == "coordinator_plan_rejected"
    ]

    assert len(failures) == 2
    assert (
        sum(event.state is TraceState.FALLBACK for event in result.product.trace) == 2
    )
    assert result.product.claim_runs[0].initial_action is PlanAction.SKIP_NOT_CHECKABLE
    assert result.product.document_status is DocumentRunStatus.COMPLETE


def test_controller_enforces_citation_first_before_discovery(
    tmp_path: Path,
) -> None:
    _copy_examples(tmp_path)
    document = tmp_path / "doc.md"
    document.write_text(
        "The Orion team owns and maintains the Nimbus runtime "
        f"[in the ownership registry]({OWNERSHIP_URL}).\n",
        encoding="utf-8",
    )
    pipeline = _pipeline(tmp_path, search=True)

    class CitationBypassCoordinator:
        calls = 0

        def plan(self, stage: PlanStage, *_args: Any, **_kwargs: Any) -> ExecutionPlan:
            self.calls += 1
            return ExecutionPlan(
                stage=stage,
                tasks=(
                    PlanTask(
                        claim_id="c_0001",
                        action=(
                            PlanAction.DISCOVER
                            if stage is PlanStage.INITIAL
                            else PlanAction.ACCEPT
                        ),
                    ),
                ),
            )

    pipeline.coordinator = CitationBypassCoordinator()  # type: ignore[assignment]
    result = pipeline.run(document, discover=True, persist=False)

    assert result.product.claim_runs[0].initial_action is PlanAction.VERIFY_CITATION
    assert all(event.agent is not AgentName.SCOUT for event in result.product.trace)
    assert any(
        event.code == "coordinator_plan_rejected" for event in result.product.trace
    )


def test_coordinator_has_exactly_two_typed_planning_calls() -> None:
    def handler(_task: str, payload: dict[str, Any]) -> dict[str, Any]:
        action = (
            PlanAction.VERIFY_CITATION
            if payload["stage"] == PlanStage.INITIAL
            else PlanAction.ACCEPT
        )
        return {
            "stage": payload["stage"],
            "tasks": [{"claim_id": "claim_1", "action": action}],
        }

    coordinator = AuditCoordinatorAgent(DeterministicFakeModel(handler))
    initial = coordinator.plan(
        PlanStage.INITIAL,
        [{"claim_id": "claim_1"}],
        discover_enabled=False,
        discovery_limit=0,
    )
    review = coordinator.plan(
        PlanStage.REVIEW,
        [{"claim_id": "claim_1"}],
        discover_enabled=False,
        discovery_limit=0,
    )

    assert initial.tasks[0].action is PlanAction.VERIFY_CITATION
    assert review.tasks[0].action is PlanAction.ACCEPT
    with pytest.raises(CoordinatorUnavailableError):
        coordinator.plan(
            PlanStage.REVIEW,
            [{"claim_id": "claim_1"}],
            discover_enabled=False,
            discovery_limit=0,
        )


@pytest.mark.parametrize(
    "queries",
    [
        ("Orion ships version 4.2 on 2028-03-04. official",),
        (
            "Orion ships version 4.2 on 2028-03-04.",
            "Orion ships version 9.9 on 2028-03-04.",
        ),
    ],
)
def test_search_query_provenance_rejects_missing_or_new_claim_terms(
    queries: tuple[str, ...],
) -> None:
    with pytest.raises(ScoutPlanError):
        validate_search_plan(SearchPlan(queries=queries), _claim())


def test_search_snippet_is_never_used_as_evidence(tmp_path: Path) -> None:
    manifest = _copy_examples(tmp_path)
    claim = _claim(
        "The cache migration starts on 2026-08-04 with a two-hour maintenance window."
    )
    search = FixtureSearchClient.from_manifest(manifest)
    result = DiscoveryTools(
        search,
        SafeFetcher(
            transport=fixture_transport(manifest),
            resolve_dns=False,
            retries=0,
        ),
    ).discover(claim, SearchPlan(queries=(claim.text,)))

    assert result.status == "ok"
    assert result.evidence
    assert all("UNTRUSTED-SNIPPET-ONLY" not in item.text for item in result.evidence)
    assert search.calls == [claim.text]


def test_tavily_without_key_is_an_offline_unavailable_result() -> None:
    def forbidden_request(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("missing credentials must prevent network access")

    response = TavilySearchClient(
        api_key="",
        transport=httpx.MockTransport(forbidden_request),
    ).search("bounded fixture query", max_results=5)

    assert response.status == "discovery_unavailable"
    assert response.results == ()


@pytest.mark.parametrize(
    ("discover", "reason_code"),
    [
        (False, "human_review_requested"),
        (True, "discovery_unavailable"),
    ],
)
def test_unresolved_claim_uses_canonical_operational_outcome_without_relation(
    tmp_path: Path,
    discover: bool,
    reason_code: str,
) -> None:
    _copy_examples(tmp_path)
    document = tmp_path / "doc.md"
    document.write_text(
        "Nimbus ships version 4.2 on 2028-03-04.\n",
        encoding="utf-8",
    )
    result = _pipeline(tmp_path).run(
        document,
        discover=discover,
        persist=False,
    )
    outcome = result.audit.claim_operational_outcomes[0]
    markdown = render_audit_markdown(result.audit)
    sarif_results = build_sarif(result.audit)["runs"][0]["results"]
    patch = write_suggestion_patch(
        document,
        result.audit,
        Path("suggestions.diff"),
        project_root=tmp_path,
    )

    assert result.audit.verdicts == ()
    assert outcome.status == "needs_human"
    assert outcome.reason_code == reason_code
    assert result.product.claim_runs[0].error_code == reason_code
    assert "Relation: not_assigned" in result.terminal
    assert reason_code in result.terminal
    assert "No factual relation was assigned." in markdown
    assert [item["ruleId"] for item in sarif_results] == ["ET2002"]
    assert "relation" not in sarif_results[0]["properties"]
    assert patch.read_text(encoding="utf-8") == ""


def test_suggestion_patch_is_deterministic_and_never_applied(
    tmp_path: Path,
) -> None:
    _copy_examples(tmp_path)
    source = tmp_path / "examples/product-demo/README.md"
    before = source.read_bytes()
    result = _pipeline(tmp_path, search=True).run(source, discover=True, persist=False)

    output = write_suggestion_patch(
        source,
        result.audit,
        Path("suggestions.diff"),
        project_root=tmp_path,
    )
    first = output.read_bytes()
    write_suggestion_patch(
        source,
        result.audit,
        Path("suggestions.diff"),
        project_root=tmp_path,
    )

    assert source.read_bytes() == before
    assert output.read_bytes() == first
    assert b"EvidenceTrace candidate correction" in first
    assert b"bounded evidence" in first
    assert b"--- a/examples/product-demo/README.md" in first
    with pytest.raises(ProductPipelineError):
        write_suggestion_patch(
            source,
            result.audit,
            source,
            project_root=tmp_path,
        )


def test_challenger_never_owns_search_actions() -> None:
    schema = DeterministicFakeModel(
        lambda _task, _payload: {
            "action": "counter_search",
            "revised_relation": None,
            "confidence": None,
            "reason": "Synthetic invalid action.",
            "evidence_span": None,
        }
    )
    challenger = ChallengerAgent(schema)

    with pytest.raises(ModelResponseError):
        challenger.challenge(
            _claim(),
            Verdict(
                claim_id="claim_fixture",
                relation=Relation.NOT_IN_SOURCE,
                confidence=0.2,
                reason="Synthetic verdict.",
                judge_version="synthetic-v1",
            ),
            (),
            (),
        )
