from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from evidencetrace.models import (
    AtomicClaim,
    AuditArtifact,
    Checkability,
    EffectiveConfig,
    ParagraphMiningAuditOutcome,
    Relation,
    RunMetadata,
    Verdict,
)
from evidencetrace.render import render_audit_markdown, render_terminal
from evidencetrace.sarif import build_sarif, sarif_bytes

PRIVATE_SOURCE = "private source body"
PRIVATE_RESPONSE = "Authorization: Bearer private-response"


def _run() -> RunMetadata:
    return RunMetadata(
        run_id="miner-public-outcome-fixture",
        started_at=datetime(2034, 1, 2, tzinfo=UTC),
        finished_at=datetime(2034, 1, 2, tzinfo=UTC),
        tool_version="0.0.0",
    )


def _claim() -> AtomicClaim:
    return AtomicClaim(
        claim_id="c_0001",
        text="A bounded synthetic statement remains stable.",
        file="docs/guide.md",
        line_start=8,
        line_end=8,
        claim_type="factual_statement",
        checkability=Checkability.NOT_CHECKABLE,
    )


def _verdict() -> Verdict:
    return Verdict(
        claim_id="c_0001",
        relation=Relation.NOT_CHECKABLE,
        confidence=0.8,
        reason="No objective source check applies.",
        judge_version="fixture-judge-v1",
    )


def _outcome(
    *,
    paragraph_id: str = "p_0001",
    file: str = "docs/guide.md",
    line_start: int = 7,
    line_end: int = 9,
    status: str = "needs_human",
    window_count: int = 2,
    completed_window_count: int = 0,
    accepted_claim_ids: tuple[str, ...] = (),
    reason_codes: tuple[str, ...] = ("miner_scope_non_source_span",),
    protected_total: int = 0,
    protected_covered: int = 0,
) -> ParagraphMiningAuditOutcome:
    return ParagraphMiningAuditOutcome.model_validate(
        {
            "paragraph_id": paragraph_id,
            "file": file,
            "line_start": line_start,
            "line_end": line_end,
            "status": status,
            "window_count": window_count,
            "completed_window_count": completed_window_count,
            "accepted_claim_ids": accepted_claim_ids,
            "reason_codes": reason_codes,
            "protected_occurrences_total": protected_total,
            "protected_occurrences_covered": protected_covered,
        }
    )


def _audit(
    *outcomes: ParagraphMiningAuditOutcome,
    include_claim: bool = False,
) -> AuditArtifact:
    return AuditArtifact(
        run=_run(),
        effective_config=EffectiveConfig(),
        claims=(_claim(),) if include_claim else (),
        verdicts=(_verdict(),) if include_claim else (),
        paragraph_mining_outcomes=outcomes,
    )


def _sarif_results(audit: AuditArtifact) -> list[dict[str, object]]:
    return build_sarif(audit)["runs"][0]["results"]


def test_canonical_audit_orders_safe_outcomes_and_reads_historical_payload() -> None:
    second = _outcome(
        paragraph_id="p_0002",
        file="zeta.md",
        line_start=20,
        line_end=21,
    )
    first = _outcome(
        paragraph_id="p_0001",
        file="docs/alpha.md",
        line_start=3,
        line_end=4,
    )
    audit = _audit(second, first)
    payload = audit.model_dump(mode="json")

    assert [item.paragraph_id for item in audit.paragraph_mining_outcomes] == [
        "p_0001",
        "p_0002",
    ]
    assert payload["paragraph_mining_outcomes"][0] == {
        "paragraph_id": "p_0001",
        "file": "docs/alpha.md",
        "line_start": 3,
        "line_end": 4,
        "status": "needs_human",
        "window_count": 2,
        "completed_window_count": 0,
        "accepted_claim_ids": [],
        "reason_codes": ["miner_scope_non_source_span"],
        "protected_occurrences_total": 0,
        "protected_occurrences_covered": 0,
    }

    payload.pop("paragraph_mining_outcomes")
    historical = AuditArtifact.model_validate(payload)
    assert historical.paragraph_mining_outcomes == ()
    assert "Claim extraction incomplete" not in render_terminal(historical)
    assert "Mining issues requiring human review" not in render_audit_markdown(
        historical
    )


def test_canonical_outcome_rejects_unknown_claim_and_duplicate_paragraph() -> None:
    unknown_claim = _outcome(
        status="partial",
        completed_window_count=1,
        accepted_claim_ids=("c_missing",),
    )
    with pytest.raises(ValidationError, match="unknown claim"):
        _audit(unknown_claim)

    duplicate = _outcome()
    with pytest.raises(ValidationError, match="must be unique"):
        _audit(duplicate, duplicate)


def test_complete_and_intentional_empty_paragraphs_have_no_public_finding() -> None:
    complete = _outcome(
        status="complete",
        window_count=1,
        completed_window_count=1,
        reason_codes=(),
    )
    audit = _audit(complete)

    assert "Claim extraction incomplete" not in render_terminal(audit)
    assert "Mining issues requiring human review" not in render_audit_markdown(audit)
    assert "No checkable cited claims were found." in render_terminal(audit)
    assert _sarif_results(audit) == []
    assert all(
        rule["id"] != "ET2001"
        for rule in build_sarif(audit)["runs"][0]["tool"]["driver"]["rules"]
    )


def test_zero_claim_incomplete_paragraph_is_visible_in_all_outputs() -> None:
    outcome = _outcome(
        line_start=12,
        line_end=14,
        window_count=3,
        completed_window_count=1,
        reason_codes=(
            "miner_scope_non_source_span",
            "miner_window_fragment_oversized",
        ),
    )
    audit = _audit(outcome)
    terminal = render_terminal(audit)
    markdown = render_audit_markdown(audit)
    result = _sarif_results(audit)[0]

    assert "No checkable cited claims were found." not in terminal
    assert "No completed claim verdicts were produced." in terminal
    assert "Claim extraction incomplete; human review required" in terminal
    assert "MINING REVIEW docs/guide.md:12-14" in terminal
    assert "Windows: 1/3 complete" in terminal
    assert "Accepted claims: 0" in terminal
    assert "incomplete_windows=2" in terminal
    assert markdown.count("`docs/guide.md:12-14`") == 1
    assert "its facts were not fully audited" in markdown
    assert result["ruleId"] == "ET2001"
    assert result["level"] == "warning"
    assert result["kind"] == "review"
    assert "not supported" not in result["message"]["text"]
    assert result["properties"]["reasonCodes"] == [
        "miner_scope_non_source_span",
        "miner_window_fragment_oversized",
    ]


def test_partial_paragraph_keeps_claim_verdict_and_one_mining_issue() -> None:
    outcome = _outcome(
        status="partial",
        completed_window_count=1,
        accepted_claim_ids=("c_0001",),
        reason_codes=("miner_schema_missing_field",),
    )
    audit = _audit(outcome, include_claim=True)
    terminal = render_terminal(audit)
    markdown = render_audit_markdown(audit)
    results = _sarif_results(audit)

    assert "Verdict: not_checkable" in terminal
    assert terminal.count("Claim extraction incomplete") == 1
    assert "`not_checkable`" in markdown
    assert markdown.count("Mining issues requiring human review") == 1
    assert [result["ruleId"] for result in results] == ["ET2001", "ET1005"]
    assert sum(result["ruleId"] == "ET2001" for result in results) == 1


@pytest.mark.parametrize(
    "reason_code",
    ["miner_window_fragment_oversized", "budget_exhausted"],
)
def test_allowlisted_planning_reasons_remain_distinct_from_relations(
    reason_code: str,
) -> None:
    audit = _audit(_outcome(reason_codes=(reason_code,)))
    result = _sarif_results(audit)[0]

    assert result["ruleId"] == "ET2001"
    assert result["properties"]["reasonCodes"] == [reason_code]
    assert "relation" not in result["properties"]


def test_mining_sarif_location_is_relative_and_uses_trusted_line_range() -> None:
    audit = _audit(
        _outcome(
            file="docs/space name.md",
            line_start=31,
            line_end=34,
        )
    )
    result = _sarif_results(audit)[0]
    location = result["locations"][0]["physicalLocation"]

    assert location["artifactLocation"]["uri"] == "docs/space%20name.md"
    assert location["region"] == {
        "startLine": 31,
        "startColumn": 1,
        "endLine": 34,
    }
    assert result["ruleId"] != "ET1002"
    assert result["message"]["text"] == (
        "Claim extraction incomplete; human review required."
    )


def test_rendering_is_byte_stable_ordered_and_contains_no_private_payload() -> None:
    outcomes = (
        _outcome(
            paragraph_id="p_0002",
            file="zeta.md",
            line_start=20,
            line_end=20,
            reason_codes=("budget_exhausted",),
        ),
        _outcome(
            paragraph_id="p_0001",
            file="docs/alpha.md",
            line_start=3,
            line_end=4,
            reason_codes=("miner_transport_error",),
        ),
    )
    audit = _audit(*outcomes)
    first = (
        audit.model_dump_json(),
        render_terminal(audit),
        render_audit_markdown(audit),
        sarif_bytes(audit),
    )
    second = (
        audit.model_dump_json(),
        render_terminal(audit),
        render_audit_markdown(audit),
        sarif_bytes(audit),
    )
    combined = "\n".join(
        value.decode() if isinstance(value, bytes) else value for value in first
    )

    assert first == second
    assert first[1].index("docs/alpha.md:3-4") < first[1].index("zeta.md:20")
    assert first[2].count("docs/alpha.md:3-4") == 1
    assert len(json.loads(first[3])["runs"][0]["results"]) == 2
    assert PRIVATE_SOURCE not in combined
    assert PRIVATE_RESPONSE not in combined
    assert "authorization" not in combined.casefold()


@pytest.mark.parametrize(
    "unsafe_field",
    ["source_text", "prompt", "response", "reasoning", "authorization"],
)
def test_canonical_outcome_forbids_private_or_model_owned_fields(
    unsafe_field: str,
) -> None:
    payload = _outcome().model_dump()
    payload[unsafe_field] = PRIVATE_RESPONSE

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ParagraphMiningAuditOutcome.model_validate(payload)
