"""Render terminal and Markdown views from the canonical AuditArtifact."""

from __future__ import annotations

from evidencetrace.models import (
    AuditArtifact,
    ClaimOperationalOutcome,
    ParagraphMiningAuditOutcome,
)
from evidencetrace.policy import relation_severity


def _mining_issues(
    audit: AuditArtifact,
) -> tuple[ParagraphMiningAuditOutcome, ...]:
    return tuple(
        outcome
        for outcome in audit.paragraph_mining_outcomes
        if outcome.status != "complete"
    )


def _source_location(outcome: ParagraphMiningAuditOutcome) -> str:
    lines = (
        str(outcome.line_start)
        if outcome.line_start == outcome.line_end
        else f"{outcome.line_start}-{outcome.line_end}"
    )
    return f"{outcome.file}:{lines}"


def _claim_issue_location(outcome: ClaimOperationalOutcome) -> str:
    lines = (
        str(outcome.line_start)
        if outcome.line_start == outcome.line_end
        else f"{outcome.line_start}-{outcome.line_end}"
    )
    return f"{outcome.file}:{lines}"


def render_audit_markdown(audit: AuditArtifact) -> str:
    mining_issues = _mining_issues(audit)
    claim_issues = audit.claim_operational_outcomes
    lines = [
        "# EvidenceTrace audit",
        "",
        f"Run: `{audit.run.run_id}`",
        "",
        "| File | Lines | Relation | Severity | Claim |",
        "|---|---:|---|---|---|",
    ]
    claims = {claim.claim_id: claim for claim in audit.claims}
    for verdict in audit.verdicts:
        claim = claims.get(verdict.claim_id)
        if claim is None:
            continue
        severity = relation_severity(
            verdict.relation,
            verdict.corroboration,
            audit.effective_config.policy,
        )
        lines.append(
            f"| `{claim.file}` | {claim.line_start}-{claim.line_end} | "
            f"`{verdict.relation.value}` | `{severity.value}` | {claim.text} |"
        )
        for span in verdict.evidence_spans:
            lines.extend(["", f"> Evidence ({span.locator}): `{span.text}`"])
    if not audit.verdicts:
        lines.extend(
            [
                "",
                (
                    "No completed claim verdicts were produced."
                    if mining_issues or claim_issues
                    else "No checkable cited claims were found."
                ),
            ]
        )
    if claim_issues:
        lines.extend(["", "## Claim checks requiring human review", ""])
        for claim_outcome in claim_issues:
            lines.append(
                f"- `{_claim_issue_location(claim_outcome)}`: claim "
                f"`{claim_outcome.claim_id}`; status `{claim_outcome.status}`; "
                f"reason `{claim_outcome.reason_code}`. No factual relation "
                "was assigned."
            )
    if mining_issues:
        lines.extend(["", "## Mining issues requiring human review", ""])
        for mining_outcome in mining_issues:
            reasons = ", ".join(mining_outcome.reason_codes)
            lines.append(
                f"- `{_source_location(mining_outcome)}`: status "
                f"`{mining_outcome.status}`; completed windows "
                f"`{mining_outcome.completed_window_count}/"
                f"{mining_outcome.window_count}`; accepted claims "
                f"`{len(mining_outcome.accepted_claim_ids)}`; reasons "
                f"`{reasons}`. "
                "Claim extraction was incomplete for this paragraph; its facts "
                "were not fully audited."
            )
    return "\n".join(lines) + "\n"


def render_terminal(audit: AuditArtifact) -> str:
    claims = {claim.claim_id: claim for claim in audit.claims}
    mining_issues = _mining_issues(audit)
    claim_issues = audit.claim_operational_outcomes
    lines = [f"EvidenceTrace CI: {', '.join(doc.path for doc in audit.documents)}", ""]
    if audit.paragraph_mining_outcomes:
        complete = sum(
            outcome.status == "complete" for outcome in audit.paragraph_mining_outcomes
        )
        partial = sum(
            outcome.status == "partial" for outcome in audit.paragraph_mining_outcomes
        )
        needs_human = sum(
            outcome.status == "needs_human"
            for outcome in audit.paragraph_mining_outcomes
        )
        incomplete_windows = sum(
            outcome.window_count - outcome.completed_window_count
            for outcome in audit.paragraph_mining_outcomes
        )
        oversized = sum(
            "miner_window_fragment_oversized" in outcome.reason_codes
            for outcome in mining_issues
        )
        budget_exhausted = sum(
            "budget_exhausted" in outcome.reason_codes for outcome in mining_issues
        )
        lines.extend(
            [
                f"Checked claims: {len(audit.verdicts)}",
                "Mining paragraphs: "
                f"complete={complete}, partial={partial}, "
                f"needs_human={needs_human}",
                "Mining gaps: "
                f"incomplete_windows={incomplete_windows}, "
                f"oversized_paragraphs={oversized}, "
                f"budget_exhausted_paragraphs={budget_exhausted}",
                "",
            ]
        )
    for verdict in audit.verdicts:
        claim = claims.get(verdict.claim_id)
        if claim is None:
            continue
        severity = relation_severity(
            verdict.relation,
            verdict.corroboration,
            audit.effective_config.policy,
        )
        lines.append(f"{severity.value.upper()} {claim.file}:{claim.line_start}")
        lines.append(f"Claim: {claim.text}")
        lines.append(f"Verdict: {verdict.relation.value}")
        if verdict.evidence_spans:
            lines.append(f"Source span: {verdict.evidence_spans[0].text}")
        lines.append(f"Reason: {verdict.reason}")
        lines.append("")
    if not audit.verdicts:
        lines.append(
            "No completed claim verdicts were produced."
            if mining_issues or claim_issues
            else "No checkable cited claims were found."
        )
        if mining_issues or claim_issues:
            lines.append("")
    for claim_outcome in claim_issues:
        lines.extend(
            [
                f"CLAIM REVIEW {_claim_issue_location(claim_outcome)}",
                "Claim audit incomplete; human review required",
                f"Claim ID: {claim_outcome.claim_id}",
                f"Status: {claim_outcome.status}",
                f"Reason: {claim_outcome.reason_code}",
                "Relation: not_assigned",
                "",
            ]
        )
    for mining_outcome in mining_issues:
        lines.extend(
            [
                f"MINING REVIEW {_source_location(mining_outcome)}",
                "Claim extraction incomplete; human review required",
                f"Status: {mining_outcome.status}",
                "Windows: "
                f"{mining_outcome.completed_window_count}/"
                f"{mining_outcome.window_count} complete",
                f"Accepted claims: {len(mining_outcome.accepted_claim_ids)}",
                f"Reasons: {', '.join(mining_outcome.reason_codes)}",
                "",
            ]
        )
    return "\n".join(lines)


__all__ = ["render_audit_markdown", "render_terminal"]
