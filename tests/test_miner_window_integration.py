from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

import evidencetrace.product as product_module
from evidencetrace.agents.miner import ClaimMinerAgent, MinerScopeError
from evidencetrace.audit_models import (
    DEFAULT_MINER_MAX_WINDOW_CHARS,
    MINER_COVERAGE_POLICY_VERSION,
    MINER_DRAFT_CONTRACT_VERSION,
    MINER_WINDOW_POLICY_VERSION,
    MinerInput,
)
from evidencetrace.model_client import (
    DeterministicFakeModel,
    OpenAICompatibleClient,
    SchemaRecoveryClient,
)
from evidencetrace.product import (
    DocumentRunStatus,
    ProductAuditPipeline,
    ProductRunArtifact,
)
from evidencetrace.render import render_audit_markdown
from evidencetrace.sarif import build_sarif

PRIVATE_CONTENT = "private-invalid-model-content"
PRIVATE_KEY = "private-integration-key"


def _draft(text: str, *, checkability: str = "not_checkable") -> dict[str, Any]:
    return {
        "claims": [
            {
                "text": text,
                "claim_type": "factual_statement",
                "checkability": checkability,
            }
        ]
    }


def _fake_pipeline(
    tmp_path: Path,
    handler: Any,
    *,
    max_window_chars: int = 64,
    miner_window_limit: int | None = None,
) -> ProductAuditPipeline:
    pipeline = ProductAuditPipeline(
        project_root=tmp_path,
        max_window_chars=max_window_chars,
        miner_window_limit=miner_window_limit,
    )
    pipeline.miner = ClaimMinerAgent(DeterministicFakeModel(handler))
    return pipeline


def _live_miner(
    contents: list[str],
    requests: list[dict[str, Any]],
) -> ClaimMinerAgent:
    remaining = iter(contents)

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            request=request,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": next(remaining)},
                    }
                ],
                "usage": {
                    "prompt_tokens": 8,
                    "completion_tokens": 5,
                    "total_tokens": 13,
                },
            },
        )

    base = OpenAICompatibleClient(
        model_id="offline-window-miner",
        api_key=PRIVATE_KEY,
        base_url="https://provider.invalid/v1",
        max_calls=len(contents),
        transport=httpx.MockTransport(handler),
    )
    return ClaimMinerAgent(SchemaRecoveryClient(base))


def test_all_paragraph_windows_are_planned_before_first_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = tmp_path / "doc.md"
    document.write_text(
        "First bounded statement remains stable.\n\n"
        "Second bounded statement remains stable.\n",
        encoding="utf-8",
    )
    planned: list[str] = []
    original = product_module.plan_miner_windows

    def recording_plan(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        planned.append(result.paragraph_id)
        return result

    def handler(_task: str, payload: dict[str, Any]) -> dict[str, Any]:
        assert planned == ["p_0001", "p_0002"]
        return _draft(payload["text"])

    monkeypatch.setattr(product_module, "plan_miner_windows", recording_plan)
    result = _fake_pipeline(tmp_path, handler).run(document, persist=False)

    assert result.product.budget["miner_windows_planned"] == 2
    assert result.product.budget["miner_windows_dispatched"] == 2
    assert [item.status for item in result.product.paragraph_mining_outcomes] == [
        "complete",
        "complete",
    ]


def test_exact_window_inputs_aggregate_claims_in_source_order_and_map_lines(
    tmp_path: Path,
) -> None:
    document = tmp_path / "doc.md"
    document.write_text(
        "# Operations\n\n"
        "The first bounded statement\n"
        "remains stable; The second bounded statement remains stable.\n",
        encoding="utf-8",
    )
    payloads: list[dict[str, Any]] = []

    def handler(_task: str, payload: dict[str, Any]) -> dict[str, Any]:
        payloads.append(payload)
        return _draft(payload["text"])

    result = _fake_pipeline(tmp_path, handler, max_window_chars=55).run(
        document, persist=False
    )
    paragraph = result.product.paragraph_mining_outcomes[0]

    assert [payload["text"] for payload in payloads] == [
        "The first bounded statement remains stable;",
        "The second bounded statement remains stable.",
    ]
    assert all(
        set(payload)
        == {
            "text",
            "heading_path",
            "citation_urls",
            "miner_contract_version",
        }
        for payload in payloads
    )
    assert [claim.text for claim in result.audit.claims] == [
        payload["text"] for payload in payloads
    ]
    assert [claim.line_start for claim in result.audit.claims] == [3, 4]
    assert [claim.line_end for claim in result.audit.claims] == [4, 4]
    assert paragraph.accepted_claim_ids == ("c_0001", "c_0002")
    assert [window.window_id for window in paragraph.windows] == [
        "p_0001_w_0001",
        "p_0001_w_0002",
    ]


def test_window_citation_and_locator_ownership_stay_local(
    tmp_path: Path,
) -> None:
    document = tmp_path / "doc.md"
    source_url = "https://docs.example.test/bounded"
    document.write_text(
        f"A bounded statement is recorded [in the guide]({source_url}).\n",
        encoding="utf-8",
    )
    payloads: list[dict[str, Any]] = []

    def handler(_task: str, payload: dict[str, Any]) -> dict[str, Any]:
        payloads.append(payload)
        return _draft(payload["text"])

    result = _fake_pipeline(tmp_path, handler, max_window_chars=96).run(
        document, persist=False
    )
    claim = result.audit.claims[0]
    window = result.product.paragraph_mining_outcomes[0].windows[0]

    assert set(payloads[0]) == {
        "text",
        "heading_path",
        "citation_urls",
        "miner_contract_version",
    }
    assert payloads[0]["citation_urls"] == (source_url,)
    assert claim.file == "doc.md"
    assert claim.line_start == claim.line_end == 1
    assert claim.citation_urls == (source_url,)
    assert window.citation_ids == ("cit_0001",)
    assert not any(
        field in payloads[0]
        for field in ("paragraph_id", "window_id", "file", "line_start", "line_end")
    )


def test_unrecovered_schema_window_preserves_later_window_claim(
    tmp_path: Path,
) -> None:
    document = tmp_path / "doc.md"
    second = "Second bounded statement remains stable."
    document.write_text(
        "First bounded statement remains stable; " + second + "\n",
        encoding="utf-8",
    )
    requests: list[dict[str, Any]] = []
    pipeline = ProductAuditPipeline(
        project_root=tmp_path,
        max_window_chars=44,
    )
    pipeline.miner = _live_miner(
        [
            '{"unexpected":"' + PRIVATE_CONTENT + '"}',
            '{"still_unexpected":true}',
            json.dumps(_draft(second)),
        ],
        requests,
    )

    result = pipeline.run(document, persist=False)
    paragraph = result.product.paragraph_mining_outcomes[0]

    assert len(requests) == 3
    assert [window.status for window in paragraph.windows] == [
        "agent_error",
        "complete",
    ]
    assert paragraph.windows[0].provider_attempts == 2
    assert paragraph.windows[0].schema_retry is True
    assert paragraph.windows[1].provider_attempts == 1
    assert paragraph.status == "partial"
    assert [claim.text for claim in result.audit.claims] == [second]
    assert result.product.budget["miner_provider_attempts"] == 3
    assert PRIVATE_CONTENT not in result.product.model_dump_json()
    assert PRIVATE_KEY not in result.product.model_dump_json()


def test_scope_failed_window_preserves_other_window_claim(
    tmp_path: Path,
) -> None:
    document = tmp_path / "doc.md"
    document.write_text(
        "First bounded statement remains stable; "
        "Second bounded statement remains stable.\n",
        encoding="utf-8",
    )
    calls = 0

    def handler(_task: str, payload: dict[str, Any]) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return _draft(
            "Invented statement outside the source." if calls == 1 else payload["text"]
        )

    result = _fake_pipeline(tmp_path, handler, max_window_chars=44).run(
        document, persist=False
    )
    paragraph = result.product.paragraph_mining_outcomes[0]

    assert [window.status for window in paragraph.windows] == [
        "agent_error",
        "complete",
    ]
    assert paragraph.windows[0].failure_codes == ("miner_scope_non_source_span",)
    assert [claim.text for claim in result.audit.claims] == [
        "Second bounded statement remains stable."
    ]
    assert paragraph.status == "partial"
    canonical = result.audit.paragraph_mining_outcomes[0]
    assert canonical.status == "partial"
    assert canonical.accepted_claim_ids == ("c_0001",)
    assert canonical.reason_codes == ("miner_scope_non_source_span",)
    assert canonical.file == "doc.md"
    assert "windows" not in canonical.model_dump()
    assert "Verdict: not_checkable" in result.terminal
    assert "Claim extraction incomplete; human review required" in result.terminal


def test_same_window_missing_hard_qualifier_withholds_exact_draft(
    tmp_path: Path,
) -> None:
    document = tmp_path / "doc.md"
    document.write_text(
        "Only the primary service handles requests.\n", encoding="utf-8"
    )
    result = _fake_pipeline(
        tmp_path,
        lambda _task, _payload: _draft("the primary service handles requests."),
    ).run(document, persist=False)
    paragraph = result.product.paragraph_mining_outcomes[0]
    window = paragraph.windows[0]

    assert window.status == "needs_human"
    assert window.failure_codes == ("miner_scope_missing_protected_token",)
    assert window.withheld_exact_draft_count == 1
    assert window.protected_occurrences_total == 1
    assert window.protected_occurrences_covered == 0
    assert paragraph.status == "needs_human"
    assert result.audit.claims == ()
    assert result.audit.paragraph_mining_outcomes[0].status == "needs_human"
    assert "No checkable cited claims were found." not in result.terminal
    assert "Claim extraction incomplete; human review required" in result.terminal


def test_uncovered_qualifier_in_other_window_does_not_delete_complete_claim(
    tmp_path: Path,
) -> None:
    document = tmp_path / "doc.md"
    first = "Primary service remains stable;"
    document.write_text(first + " Secondary service ships v2.4.\n", encoding="utf-8")

    def handler(_task: str, payload: dict[str, Any]) -> dict[str, Any]:
        return {"claims": []} if "v2.4" in payload["text"] else _draft(payload["text"])

    result = _fake_pipeline(tmp_path, handler, max_window_chars=40).run(
        document, persist=False
    )
    paragraph = result.product.paragraph_mining_outcomes[0]

    assert paragraph.windows[0].status == "complete"
    assert paragraph.windows[1].failure_codes == (
        "miner_scope_missing_protected_token",
    )
    assert [claim.text for claim in result.audit.claims] == [first]
    assert paragraph.status == "partial"


def test_protected_coverage_counts_duplicate_occurrences() -> None:
    source = "Version v2.4 remains active while v2.4 remains archived."
    model = DeterministicFakeModel(
        lambda _task, _payload: _draft(
            "Version v2.4 remains active", checkability="checkable"
        )
    )

    with pytest.raises(MinerScopeError) as raised:
        ClaimMinerAgent(model).mine(
            MinerInput(
                text=source,
                file="docs/versions.md",
                line_start=1,
                line_end=1,
            )
        )

    assert raised.value.code == "missing_protected_token"
    assert raised.value.protected_occurrences_total == 2
    assert raised.value.protected_occurrences_covered == 1
    assert raised.value.withheld_exact_draft_count == 1


def test_schema_valid_intentional_empty_window_is_complete(
    tmp_path: Path,
) -> None:
    document = tmp_path / "doc.md"
    document.write_text("Browser compatibility details:\n", encoding="utf-8")
    calls = 0

    def handler(_task: str, _payload: dict[str, Any]) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {"claims": []}

    result = _fake_pipeline(tmp_path, handler).run(document, persist=False)
    paragraph = result.product.paragraph_mining_outcomes[0]

    assert calls == 1
    assert paragraph.status == "complete"
    assert paragraph.windows[0].status == "complete"
    assert paragraph.windows[0].accepted_claim_ids == ()
    assert result.product.document_status is DocumentRunStatus.COMPLETE


def test_schema_valid_empty_identifier_window_uses_canonical_needs_human_path(
    tmp_path: Path,
) -> None:
    document = tmp_path / "doc.md"
    identifier = "worker.retry_v2"
    document.write_text(
        f"The `{identifier}` mode remains documented.\n",
        encoding="utf-8",
    )
    requests: list[dict[str, Any]] = []
    pipeline = ProductAuditPipeline(project_root=tmp_path, max_window_chars=64)
    pipeline.miner = _live_miner([json.dumps({"claims": []})], requests)
    result = pipeline.run(document, persist=False)
    paragraph = result.product.paragraph_mining_outcomes[0]
    window = paragraph.windows[0]
    canonical = result.audit.paragraph_mining_outcomes[0]
    markdown = render_audit_markdown(result.audit)
    sarif = build_sarif(result.audit)
    sarif_results = sarif["runs"][0]["results"]

    assert len(requests) == 1
    assert result.product.budget["miner_windows_dispatched"] == 1
    assert result.product.budget["miner_provider_attempts"] == 1
    assert result.audit.claims == ()
    assert result.audit.verdicts == ()
    assert result.product.claim_runs == ()
    assert all(event.claim_id is None for event in result.product.trace)
    assert window.status == "needs_human"
    assert window.failure_codes == ("miner_coverage_empty_identifier_window",)
    assert window.provider_attempts == 1
    assert window.schema_retry is False
    assert paragraph.status == "needs_human"
    assert paragraph.reason_codes == ("miner_coverage_empty_identifier_window",)
    assert canonical.reason_codes == ("miner_coverage_empty_identifier_window",)
    assert "Claim extraction incomplete; human review required" in result.terminal
    assert "miner_coverage_empty_identifier_window" in result.terminal
    assert "miner_coverage_empty_identifier_window" in markdown
    assert [item["ruleId"] for item in sarif_results] == ["ET2001"]
    assert "relation" not in sarif_results[0]["properties"]
    assert identifier not in result.product.model_dump_json()
    assert identifier not in markdown


def test_identifier_empty_window_makes_other_valid_claim_partial(
    tmp_path: Path,
) -> None:
    document = tmp_path / "doc.md"
    first = "The primary service remains stable;"
    document.write_text(
        first + " the `worker.retry_v2` mode remains documented.\n",
        encoding="utf-8",
    )

    def handler(_task: str, payload: dict[str, Any]) -> dict[str, Any]:
        return (
            {"claims": []}
            if "worker.retry_v2" in payload["text"]
            else _draft(payload["text"])
        )

    result = _fake_pipeline(tmp_path, handler, max_window_chars=64).run(
        document, persist=False
    )
    paragraph = result.product.paragraph_mining_outcomes[0]

    assert [claim.text for claim in result.audit.claims] == [first]
    assert [window.status for window in paragraph.windows] == [
        "complete",
        "needs_human",
    ]
    assert paragraph.status == "partial"
    assert paragraph.reason_codes == ("miner_coverage_empty_identifier_window",)
    assert (
        len(
            [
                item
                for item in build_sarif(result.audit)["runs"][0]["results"]
                if item["ruleId"] == "ET2001"
            ]
        )
        == 1
    )


@pytest.mark.parametrize(
    "source",
    [
        "Plain worker.retry_v2 text remains visible.",
        "The [worker.retry_v2](https://example.test/guide) label remains visible.",
        "The `natural language phrase` remains visible.",
        "The ` ` span remains blank.",
        "The `https://example.test/guide` URL is omitted.",
    ],
)
def test_non_identifier_empty_windows_remain_complete(
    tmp_path: Path,
    source: str,
) -> None:
    document = tmp_path / "doc.md"
    document.write_text(source + "\n", encoding="utf-8")
    result = _fake_pipeline(tmp_path, lambda _task, _payload: {"claims": []}).run(
        document, persist=False
    )
    paragraph = result.product.paragraph_mining_outcomes[0]

    assert paragraph.status == "complete"
    assert paragraph.windows[0].status == "complete"
    assert paragraph.reason_codes == ()
    assert all(
        item["ruleId"] != "ET2001"
        for item in build_sarif(result.audit)["runs"][0]["results"]
    )


def test_nonempty_claims_do_not_use_empty_identifier_signal(
    tmp_path: Path,
) -> None:
    document = tmp_path / "doc.md"
    document.write_text(
        "The primary service remains stable while `worker.retry_v2` is documented.\n",
        encoding="utf-8",
    )
    result = _fake_pipeline(
        tmp_path,
        lambda _task, _payload: _draft("The primary service remains stable"),
        max_window_chars=96,
    ).run(document, persist=False)
    paragraph = result.product.paragraph_mining_outcomes[0]

    assert [claim.text for claim in result.audit.claims] == [
        "The primary service remains stable"
    ]
    assert paragraph.status == "complete"
    assert paragraph.windows[0].failure_codes == ()


def test_exact_claim_containing_identifier_remains_complete(
    tmp_path: Path,
) -> None:
    document = tmp_path / "doc.md"
    document.write_text(
        "The `worker.retry_v2` mode remains documented.\n",
        encoding="utf-8",
    )
    result = _fake_pipeline(
        tmp_path, lambda _task, payload: _draft(payload["text"])
    ).run(document, persist=False)
    paragraph = result.product.paragraph_mining_outcomes[0]

    assert [claim.text for claim in result.audit.claims] == [
        "The worker.retry_v2 mode remains documented."
    ]
    assert paragraph.status == "complete"
    assert paragraph.windows[0].failure_codes == ()


def test_multiple_identifier_anchors_emit_one_window_reason(
    tmp_path: Path,
) -> None:
    document = tmp_path / "doc.md"
    document.write_text(
        "The `worker.retry_v2` and `queue.mode_v3` settings remain documented.\n",
        encoding="utf-8",
    )
    result = _fake_pipeline(
        tmp_path,
        lambda _task, _payload: {"claims": []},
        max_window_chars=96,
    ).run(document, persist=False)
    window = result.product.paragraph_mining_outcomes[0].windows[0]

    assert window.failure_codes == ("miner_coverage_empty_identifier_window",)
    assert result.audit.claims == ()


def test_multiple_empty_identifier_windows_emit_one_public_paragraph_issue(
    tmp_path: Path,
) -> None:
    document = tmp_path / "doc.md"
    document.write_text(
        "Use `worker.retry_v2` for one mode; use `queue.mode_v3` for another mode.\n",
        encoding="utf-8",
    )
    result = _fake_pipeline(
        tmp_path,
        lambda _task, _payload: {"claims": []},
        max_window_chars=48,
    ).run(document, persist=False)
    paragraph = result.product.paragraph_mining_outcomes[0]
    et2001 = [
        item
        for item in build_sarif(result.audit)["runs"][0]["results"]
        if item["ruleId"] == "ET2001"
    ]

    assert len(paragraph.windows) == 2
    assert paragraph.reason_codes == ("miner_coverage_empty_identifier_window",)
    assert len(result.audit.paragraph_mining_outcomes) == 1
    assert result.terminal.count("Claim extraction incomplete") == 1
    assert (
        render_audit_markdown(result.audit).count(
            "Mining issues requiring human review"
        )
        == 1
    )
    assert len(et2001) == 1


@pytest.mark.parametrize(
    ("source", "response", "expected_code"),
    [
        (
            "Only the `worker.retry_v2` mode remains active.",
            {"claims": []},
            "miner_scope_missing_protected_token",
        ),
        (
            "The `worker.retry_v2` mode remains active.",
            _draft("Invented content remains active."),
            "miner_scope_non_source_span",
        ),
        (
            "The `worker.retry_v2` mode is available.",
            _draft("The worker mode is"),
            "miner_scope_invalid_claim_fragment",
        ),
        (
            "Read `more details` before proceeding.",
            {"claims": []},
            "miner_scope_missing_protected_token",
        ),
    ],
)
def test_existing_scope_failures_precede_empty_identifier_signal(
    tmp_path: Path,
    source: str,
    response: dict[str, Any],
    expected_code: str,
) -> None:
    document = tmp_path / "doc.md"
    document.write_text(source + "\n", encoding="utf-8")
    result = _fake_pipeline(tmp_path, lambda _task, _payload: response).run(
        document, persist=False
    )
    window = result.product.paragraph_mining_outcomes[0].windows[0]

    assert window.failure_codes == (expected_code,)
    assert "miner_coverage_empty_identifier_window" not in window.failure_codes
    assert result.audit.claims == ()


def test_schema_failure_precedes_empty_identifier_signal(
    tmp_path: Path,
) -> None:
    document = tmp_path / "doc.md"
    document.write_text(
        "The `worker.retry_v2` mode remains documented.\n",
        encoding="utf-8",
    )
    requests: list[dict[str, Any]] = []
    pipeline = ProductAuditPipeline(project_root=tmp_path)
    pipeline.miner = _live_miner(
        ['{"unexpected":true}', '{"still_unexpected":true}'],
        requests,
    )

    result = pipeline.run(document, persist=False)
    window = result.product.paragraph_mining_outcomes[0].windows[0]

    assert len(requests) == 2
    assert window.provider_attempts == 2
    assert window.schema_retry is True
    assert len(window.failure_codes) == 1
    assert window.failure_codes[0].startswith("miner_schema_")
    assert "miner_coverage_empty_identifier_window" not in window.failure_codes


def test_budget_failure_precedes_empty_identifier_signal_without_call(
    tmp_path: Path,
) -> None:
    document = tmp_path / "doc.md"
    document.write_text(
        "The `worker.retry_v2` mode remains documented.\n",
        encoding="utf-8",
    )
    calls = 0

    def forbidden(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        raise AssertionError("budget-exhausted window must not call the Miner")

    result = _fake_pipeline(
        tmp_path,
        forbidden,
        miner_window_limit=0,
    ).run(document, persist=False)
    window = result.product.paragraph_mining_outcomes[0].windows[0]

    assert calls == 0
    assert window.failure_codes == ("budget_exhausted",)
    assert window.provider_attempts == 0
    assert result.product.budget["miner_windows_dispatched"] == 0


def test_oversized_window_is_located_needs_human_without_model_call(
    tmp_path: Path,
) -> None:
    document = tmp_path / "doc.md"
    document.write_text(
        "A continuous factual statement deliberately has no safe splitting "
        "boundary before its final word.\n",
        encoding="utf-8",
    )
    calls = 0

    def forbidden(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        raise AssertionError("oversized window must not call the model")

    result = _fake_pipeline(tmp_path, forbidden, max_window_chars=24).run(
        document, persist=False
    )
    paragraph = result.product.paragraph_mining_outcomes[0]

    assert calls == 0
    assert paragraph.status == "needs_human"
    assert paragraph.windows[0].failure_codes == ("miner_window_fragment_oversized",)
    assert paragraph.windows[0].provider_attempts == 0
    assert paragraph.windows[0].source_line_start == 1
    assert result.product.budget["miner_windows_dispatched"] == 0


def test_window_budget_stops_dispatch_without_conflating_claim_count(
    tmp_path: Path,
) -> None:
    document = tmp_path / "doc.md"
    document.write_text(
        "First bounded statement is stable; "
        "Second bounded statement is stable; "
        "Third bounded statement is stable.\n",
        encoding="utf-8",
    )
    calls = 0

    def handler(_task: str, payload: dict[str, Any]) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return _draft(payload["text"])

    result = _fake_pipeline(
        tmp_path,
        handler,
        max_window_chars=38,
        miner_window_limit=1,
    ).run(document, persist=False)
    paragraph = result.product.paragraph_mining_outcomes[0]
    budget = result.product.budget

    assert calls == 1
    assert [window.status for window in paragraph.windows] == [
        "complete",
        "budget_exhausted",
        "budget_exhausted",
    ]
    assert budget["claim_limit"] == 30
    assert budget["miner_window_limit"] == 1
    assert budget["miner_windows_planned"] == 3
    assert budget["miner_windows_dispatched"] == 1
    assert budget["miner_provider_attempt_limit"] == 2
    assert budget["miner_provider_attempts"] == 1
    assert len(result.audit.claims) == 1


def test_one_window_can_emit_multiple_claims_under_independent_claim_budget(
    tmp_path: Path,
) -> None:
    document = tmp_path / "doc.md"
    text = "Alpha remains stable and Beta remains ready"
    document.write_text(text + "\n", encoding="utf-8")

    def handler(_task: str, _payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "claims": [
                {
                    "text": "Alpha remains stable",
                    "claim_type": "factual_statement",
                    "checkability": "not_checkable",
                },
                {
                    "text": "Beta remains ready",
                    "claim_type": "factual_statement",
                    "checkability": "not_checkable",
                },
            ]
        }

    result = _fake_pipeline(
        tmp_path, handler, max_window_chars=64, miner_window_limit=1
    ).run(document, persist=False)

    assert result.product.budget["miner_windows_planned"] == 1
    assert result.product.budget["miner_windows_dispatched"] == 1
    assert [claim.claim_id for claim in result.audit.claims] == [
        "c_0001",
        "c_0002",
    ]


def test_scope_failure_is_not_schema_retried(
    tmp_path: Path,
) -> None:
    document = tmp_path / "doc.md"
    document.write_text("Bounded statement remains stable.\n", encoding="utf-8")
    requests: list[dict[str, Any]] = []
    pipeline = ProductAuditPipeline(project_root=tmp_path)
    pipeline.miner = _live_miner(
        [
            json.dumps(_draft("Invented statement remains stable.")),
            json.dumps(_draft("Bounded statement remains stable.")),
        ],
        requests,
    )

    result = pipeline.run(document, persist=False)
    window = result.product.paragraph_mining_outcomes[0].windows[0]

    assert len(requests) == 1
    assert window.failure_codes == ("miner_scope_non_source_span",)
    assert window.provider_attempts == 1
    assert window.schema_retry is False


def test_safe_outcomes_and_v2_provenance_preserve_v1_reader(
    tmp_path: Path,
) -> None:
    document = tmp_path / "doc.md"
    document.write_text("Bounded statement remains stable.\n", encoding="utf-8")
    result = _fake_pipeline(
        tmp_path, lambda _task, payload: _draft(payload["text"])
    ).run(document, persist=False)
    serialized = result.product.model_dump_json()
    provenance = result.product.miner_provenance

    assert result.product.artifact_version == "bounded-multi-agent-product-v2"
    assert provenance is not None
    assert provenance.window_policy_version == MINER_WINDOW_POLICY_VERSION
    assert provenance.coverage_policy_version == MINER_COVERAGE_POLICY_VERSION
    assert provenance.miner_contract_version == MINER_DRAFT_CONTRACT_VERSION
    assert provenance.max_window_chars == 64
    assert provenance.miner_window_limit == 30
    assert provenance.miner_provider_attempt_limit == 60
    assert provenance.model_max_tokens == 2048
    assert MINER_COVERAGE_POLICY_VERSION == "miner-coverage-policy-v2"
    assert MINER_WINDOW_POLICY_VERSION == "miner-window-policy-v1"
    assert MINER_DRAFT_CONTRACT_VERSION == "live-miner-draft-v3"
    assert "Bounded statement remains stable." not in serialized
    for forbidden in ("prompt", "response", "authorization", PRIVATE_KEY):
        assert forbidden not in serialized.casefold()

    historical = ProductRunArtifact.model_validate(
        {
            "artifact_version": "bounded-multi-agent-product-v1",
            "document_status": "complete",
            "claim_runs": [],
            "trace": [],
            "budget": {},
            "live_agents_enabled": False,
            "discover_enabled": False,
        }
    )
    assert historical.paragraph_mining_outcomes == ()
    assert historical.miner_provenance is None

    historical_v2_payload = result.product.model_dump(mode="json")
    historical_v2_payload["miner_provenance"]["coverage_policy_version"] = (
        "miner-coverage-policy-v1"
    )
    for paragraph in historical_v2_payload["paragraph_mining_outcomes"]:
        paragraph["coverage_policy_version"] = "miner-coverage-policy-v1"
        for window in paragraph["windows"]:
            window["coverage_policy_version"] = "miner-coverage-policy-v1"
    historical_v2 = ProductRunArtifact.model_validate(historical_v2_payload)
    assert (
        historical_v2.miner_provenance is not None
        and historical_v2.miner_provenance.coverage_policy_version
        == "miner-coverage-policy-v1"
    )
    assert DEFAULT_MINER_MAX_WINDOW_CHARS == 384
