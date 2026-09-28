"""Deterministic SARIF 2.1.0 rendering from canonical audit artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote

from evidencetrace import __version__
from evidencetrace.models import (
    AtomicClaim,
    AuditArtifact,
    ClaimOperationalOutcome,
    ParagraphMiningAuditOutcome,
    PolicyDecision,
    Relation,
    Severity,
    SourceSpan,
    Verdict,
)
from evidencetrace.policy import decide_policy

SarifErrorCode = Literal[
    "duplicate_claim_id",
    "invalid_output_path",
    "invalid_policy_finding",
    "verdict_claim_missing",
    "write_failed",
]
_SARIF_ERROR_CODES: frozenset[str] = frozenset(
    {
        "duplicate_claim_id",
        "invalid_output_path",
        "invalid_policy_finding",
        "verdict_claim_missing",
        "write_failed",
    }
)
_RULES: dict[Relation, tuple[str, str, str]] = {
    Relation.CONTRADICTED: (
        "ET1001",
        "Contradicted claim",
        "The cited source conflicts with the claim.",
    ),
    Relation.NOT_IN_SOURCE: (
        "ET1002",
        "Unsupported claim",
        "The cited source does not support the claim.",
    ),
    Relation.PARTIALLY_ENTAILED: (
        "ET1003",
        "Partially supported claim",
        "The cited source supports only part of the claim.",
    ),
    Relation.SOURCE_UNAVAILABLE: (
        "ET1004",
        "Source unavailable",
        "The cited source could not be checked.",
    ),
    Relation.NOT_CHECKABLE: (
        "ET1005",
        "Claim not checkable",
        "The claim has no deterministic cited-source check.",
    ),
    Relation.ENTAILED: (
        "ET1006",
        "Supported claim",
        "The cited source supports the claim.",
    ),
}
_SARIF_LEVEL = {
    Severity.ERROR: "error",
    Severity.WARNING: "warning",
    Severity.NOTICE: "note",
    Severity.PASS: "none",
}
_MINING_RULE_ID = "ET2001"
_MINING_RULE = {
    "id": _MINING_RULE_ID,
    "name": "MinerNeedsHuman",
    "shortDescription": {"text": "Claim extraction incomplete"},
    "fullDescription": {
        "text": (
            "Claim extraction did not complete for a source paragraph and "
            "requires human review."
        )
    },
    "properties": {
        "category": "miner-needs-human",
        "tags": ["claim-extraction", "human-review"],
    },
}
_CLAIM_OPERATIONAL_RULE_ID = "ET2002"
_CLAIM_OPERATIONAL_RULE = {
    "id": _CLAIM_OPERATIONAL_RULE_ID,
    "name": "ClaimNeedsHuman",
    "shortDescription": {"text": "Claim audit incomplete"},
    "fullDescription": {
        "text": (
            "A claim could not receive a factual relation and requires human review."
        )
    },
    "properties": {
        "category": "claim-needs-human",
        "tags": ["claim-audit", "human-review"],
    },
}


class SarifError(RuntimeError):
    """Payload-free SARIF generation or output-boundary failure."""

    def __init__(self, code: SarifErrorCode) -> None:
        if code not in _SARIF_ERROR_CODES:
            raise ValueError("unsupported SARIF error code")
        super().__init__("SARIF operation failed")
        self.code = code


@dataclass(frozen=True)
class AuditPolicyFinding:
    claim: AtomicClaim
    verdict: Verdict
    decision: PolicyDecision


def _claim_span(claim: AtomicClaim) -> SourceSpan:
    return SourceSpan(
        file=claim.file,
        line_start=claim.line_start,
        line_end=claim.line_end,
        column_start=1,
        column_end=1,
        offset_start=0,
        offset_end=0,
    )


def audit_policy_findings(audit: AuditArtifact) -> tuple[AuditPolicyFinding, ...]:
    """Apply the existing policy once and retain safe claim locations."""

    claims: dict[str, AtomicClaim] = {}
    for audit_claim in audit.claims:
        if audit_claim.claim_id in claims:
            raise SarifError("duplicate_claim_id")
        claims[audit_claim.claim_id] = audit_claim
    documents = {document.path: document for document in audit.documents}
    findings: list[AuditPolicyFinding] = []
    for verdict in audit.verdicts:
        claim = claims.get(verdict.claim_id)
        if claim is None:
            raise SarifError("verdict_claim_missing")
        decision = decide_policy(
            verdict.relation,
            verdict.corroboration,
            policy=audit.effective_config.policy,
            source=_claim_span(claim),
            claim_type=claim.claim_type,
            document=documents.get(claim.file),
        )
        findings.append(
            AuditPolicyFinding(
                claim=claim,
                verdict=verdict,
                decision=decision,
            )
        )
    return tuple(findings)


def _rule_descriptors(
    *,
    include_mining: bool = False,
    include_claim_operational: bool = False,
) -> list[dict[str, Any]]:
    descriptors = []
    for relation, (rule_id, name, description) in sorted(
        _RULES.items(),
        key=lambda item: item[1][0],
    ):
        descriptors.append(
            {
                "id": rule_id,
                "name": name.replace(" ", ""),
                "shortDescription": {"text": name},
                "fullDescription": {"text": description},
                "properties": {
                    "relation": relation.value,
                    "tags": ["citation", "factual-consistency"],
                },
            }
        )
    if include_mining:
        descriptors.append(_MINING_RULE)
    if include_claim_operational:
        descriptors.append(_CLAIM_OPERATIONAL_RULE)
    descriptors.sort(key=lambda descriptor: str(descriptor["id"]))
    return descriptors


def _result(finding: AuditPolicyFinding) -> dict[str, Any]:
    claim = finding.claim
    verdict = finding.verdict
    severity = finding.decision.effective_severity
    if severity is None or severity is Severity.PASS:
        raise SarifError("invalid_policy_finding")
    rule_id = _RULES[verdict.relation][0]
    fingerprint_payload = "\0".join(
        (
            claim.file,
            str(claim.line_start),
            claim.claim_id,
            verdict.relation.value,
        )
    )
    return {
        "ruleId": rule_id,
        "level": _SARIF_LEVEL[severity],
        "kind": "fail",
        "message": {
            "text": (
                "EvidenceTrace policy result: "
                f"{verdict.relation.value.replace('_', ' ')}."
            )
        },
        "locations": [
            {
                "physicalLocation": {
                    "artifactLocation": {
                        "uri": quote(claim.file, safe="/-._~"),
                    },
                    "region": {
                        "startLine": claim.line_start,
                        "startColumn": 1,
                        "endLine": claim.line_end,
                    },
                }
            }
        ],
        "partialFingerprints": {
            "evidencetraceFinding/v1": hashlib.sha256(
                fingerprint_payload.encode()
            ).hexdigest()
        },
        "properties": {
            "claimId": claim.claim_id,
            "corroboration": verdict.corroboration.value,
            "policySeverity": severity.value,
            "relation": verdict.relation.value,
        },
    }


def _mining_result(outcome: ParagraphMiningAuditOutcome) -> dict[str, Any]:
    fingerprint_payload = "\0".join(
        (
            outcome.file,
            str(outcome.line_start),
            outcome.paragraph_id,
            outcome.status,
            ",".join(outcome.reason_codes),
        )
    )
    return {
        "ruleId": _MINING_RULE_ID,
        "level": "warning",
        "kind": "review",
        "message": {"text": "Claim extraction incomplete; human review required."},
        "locations": [
            {
                "physicalLocation": {
                    "artifactLocation": {
                        "uri": quote(outcome.file, safe="/-._~"),
                    },
                    "region": {
                        "startLine": outcome.line_start,
                        "startColumn": 1,
                        "endLine": outcome.line_end,
                    },
                }
            }
        ],
        "partialFingerprints": {
            "evidencetraceMiningFinding/v1": hashlib.sha256(
                fingerprint_payload.encode()
            ).hexdigest()
        },
        "properties": {
            "acceptedClaimCount": len(outcome.accepted_claim_ids),
            "completedWindowCount": outcome.completed_window_count,
            "paragraphId": outcome.paragraph_id,
            "protectedOccurrencesCovered": outcome.protected_occurrences_covered,
            "protectedOccurrencesTotal": outcome.protected_occurrences_total,
            "reasonCodes": list(outcome.reason_codes),
            "status": outcome.status,
            "windowCount": outcome.window_count,
        },
    }


def _claim_operational_result(
    outcome: ClaimOperationalOutcome,
) -> dict[str, Any]:
    fingerprint_payload = "\0".join(
        (
            outcome.file,
            str(outcome.line_start),
            outcome.claim_id,
            outcome.status,
            outcome.reason_code,
        )
    )
    return {
        "ruleId": _CLAIM_OPERATIONAL_RULE_ID,
        "level": "warning",
        "kind": "review",
        "message": {
            "text": (
                "Claim audit incomplete; human review required. "
                f"Reason: {outcome.reason_code}."
            )
        },
        "locations": [
            {
                "physicalLocation": {
                    "artifactLocation": {
                        "uri": quote(outcome.file, safe="/-._~"),
                    },
                    "region": {
                        "startLine": outcome.line_start,
                        "startColumn": 1,
                        "endLine": outcome.line_end,
                    },
                }
            }
        ],
        "partialFingerprints": {
            "evidencetraceClaimOperationalFinding/v1": hashlib.sha256(
                fingerprint_payload.encode()
            ).hexdigest()
        },
        "properties": {
            "claimId": outcome.claim_id,
            "reasonCode": outcome.reason_code,
            "status": outcome.status,
        },
    }


def _self_use_run_properties(audit: AuditArtifact) -> dict[str, Any] | None:
    if getattr(audit, "self_use_schema_version", None) != 1:
        return None
    execution = getattr(audit, "execution", None)
    file_state = getattr(audit, "file_state", None)
    decisions = tuple(getattr(audit, "repair_decisions", ()))
    resolutions = tuple(getattr(audit, "evidence_resolutions", ()))
    if execution is None or file_state is None:
        return None

    def value(item: Any) -> Any:
        return getattr(item, "value", item)

    return {
        "selfUseSchemaVersion": 1,
        "targetId": getattr(audit, "target_id", ""),
        "mode": getattr(audit, "mode", ""),
        "status": getattr(audit, "status", ""),
        "documentStatus": value(execution.document_status),
        "fileState": value(file_state.status),
        "beforeSha256": file_state.before_sha256,
        "afterSha256": file_state.after_sha256,
        "repairCounts": {
            "eligible": sum(
                value(item.eligibility_status) == "eligible"
                for item in decisions
            ),
            "approved": sum(
                value(item.review_status) == "approved" for item in decisions
            ),
            "applied": sum(
                value(item.apply_status) == "applied" for item in decisions
            ),
        },
        "needsHumanEvidenceCount": sum(
            value(item.status) == "needs_human" for item in resolutions
        ),
    }


def build_sarif(audit: AuditArtifact) -> dict[str, Any]:
    """Build stable SARIF without request, source, or model-authored content."""

    findings = [
        finding
        for finding in audit_policy_findings(audit)
        if finding.decision.effective_severity not in {None, Severity.PASS}
    ]
    mining_findings = tuple(
        outcome
        for outcome in audit.paragraph_mining_outcomes
        if outcome.status != "complete"
    )
    claim_operational_findings = audit.claim_operational_outcomes
    results = [
        (
            (
                finding.claim.file,
                finding.claim.line_start,
                finding.claim.line_end,
                _RULES[finding.verdict.relation][0],
                finding.claim.claim_id,
            ),
            _result(finding),
        )
        for finding in findings
    ]
    results.extend(
        (
            (
                outcome.file,
                outcome.line_start,
                outcome.line_end,
                _MINING_RULE_ID,
                outcome.paragraph_id,
            ),
            _mining_result(outcome),
        )
        for outcome in mining_findings
    )
    results.extend(
        (
            (
                outcome.file,
                outcome.line_start,
                outcome.line_end,
                _CLAIM_OPERATIONAL_RULE_ID,
                outcome.claim_id,
            ),
            _claim_operational_result(outcome),
        )
        for outcome in claim_operational_findings
    )
    results.sort(key=lambda item: item[0])
    run: dict[str, Any] = {
        "tool": {
            "driver": {
                "name": "EvidenceTrace",
                "semanticVersion": __version__,
                "rules": _rule_descriptors(
                    include_mining=bool(mining_findings),
                    include_claim_operational=bool(claim_operational_findings),
                ),
            }
        },
        "results": [result for _, result in results],
    }
    self_use_properties = _self_use_run_properties(audit)
    if self_use_properties is not None:
        run["properties"] = self_use_properties
    return {
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "version": "2.1.0",
        "runs": [run],
    }


def sarif_bytes(audit: AuditArtifact) -> bytes:
    """Serialize one audit deterministically with a trailing newline."""

    return (
        json.dumps(
            build_sarif(audit),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode()


def _bounded_output_path(output: Path, project_root: Path) -> Path:
    root = project_root.resolve()
    candidate = output if output.is_absolute() else root / output
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        raise SarifError("invalid_output_path") from None
    if resolved == root or (resolved.exists() and not resolved.is_file()):
        raise SarifError("invalid_output_path")
    return resolved


def write_sarif(
    audit: AuditArtifact,
    output: Path,
    *,
    project_root: Path,
) -> Path:
    """Atomically replace one project-contained SARIF output."""

    destination = _bounded_output_path(output, project_root)
    temporary_path: Path | None = None
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination = _bounded_output_path(destination, project_root)
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=".evidencetrace-sarif.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(sarif_bytes(audit))
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, destination)
    except SarifError:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    except OSError:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise SarifError("write_failed") from None
    return destination


__all__ = [
    "AuditPolicyFinding",
    "SarifError",
    "audit_policy_findings",
    "build_sarif",
    "sarif_bytes",
    "write_sarif",
]
