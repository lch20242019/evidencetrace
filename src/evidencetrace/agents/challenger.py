"""One-shot high-risk verdict review without tools."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from evidencetrace.audit_models import CheckSignal, RetrievedEvidence
from evidencetrace.model_client import MODEL_REASON_MAX_CHARS, ModelClient
from evidencetrace.models import AtomicClaim, EvidenceSpan, Relation, Verdict

CHALLENGER_CONTRACT_VERSION = "claim-challenger-v1"
_RISK_RE = re.compile(
    r"\b(?:v?\d+(?:\.\d+){1,3}|20\d{2}[-/]\d{1,2}[-/]\d{1,2}|"
    r"\d+(?:[.,]\d+)?%?|more|less|higher|lower|faster|slower|best|"
    r"worst|maximum|minimum|before|after|than)\b",
    re.I,
)
_HARD_SIGNALS = {
    "numeric_mismatch",
    "date_mismatch",
    "version_mismatch",
    "negation_mismatch",
    "entity_mismatch",
}
_SUBSTANTIVE = {
    Relation.ENTAILED,
    Relation.PARTIALLY_ENTAILED,
    Relation.CONTRADICTED,
}


class ChallengeAction(StrEnum):
    UPHOLD = "uphold"
    REVISE = "revise"
    ABSTAIN = "abstain"


class LiveChallengeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    action: ChallengeAction
    revised_relation: Relation | None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    reason: str = Field(max_length=MODEL_REASON_MAX_CHARS)
    evidence_span: str | None

    @field_validator("reason")
    @classmethod
    def non_blank_reason(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("reason must not be blank")
        return value

    @model_validator(mode="after")
    def consistent_revision(self) -> LiveChallengeOutput:
        revision = self.action is ChallengeAction.REVISE
        if revision != (
            self.revised_relation is not None and self.confidence is not None
        ):
            raise ValueError("revision fields do not match action")
        if (
            revision
            and self.revised_relation in _SUBSTANTIVE
            and not self.evidence_span
        ):
            raise ValueError("substantive revision requires evidence")
        if not revision and self.evidence_span is not None:
            raise ValueError("non-revision cannot include evidence")
        return self


@dataclass(frozen=True)
class ChallengeResult:
    action: ChallengeAction
    verdict: Verdict
    needs_human: bool = False


class ChallengerScopeError(RuntimeError):
    pass


def is_high_risk_claim(
    claim: AtomicClaim,
    verdict: Verdict,
    signals: tuple[CheckSignal, ...],
) -> bool:
    return (
        verdict.relation is Relation.CONTRADICTED
        or verdict.confidence < 0.65
        or bool(_RISK_RE.search(claim.text))
        or any(item.code in _HARD_SIGNALS for item in signals)
    )


class ChallengerAgent:
    def __init__(self, client: ModelClient | None = None) -> None:
        self.client = client

    def challenge(
        self,
        claim: AtomicClaim,
        verdict: Verdict,
        signals: tuple[CheckSignal, ...],
        evidence: tuple[RetrievedEvidence, ...],
    ) -> ChallengeResult:
        if self.client is None:
            abstain = (
                verdict.confidence < 0.5
                and verdict.relation is not Relation.CONTRADICTED
            )
            return ChallengeResult(
                ChallengeAction.ABSTAIN if abstain else ChallengeAction.UPHOLD,
                verdict,
                abstain,
            )
        output = LiveChallengeOutput.model_validate(
            self.client.complete_model(
                "claim_challenge",
                {
                    "contract_version": CHALLENGER_CONTRACT_VERSION,
                    "claim": {"text": claim.text, "claim_type": claim.claim_type},
                    "judge_verdict": {
                        "relation": verdict.relation.value,
                        "confidence": verdict.confidence,
                    },
                    "signals": [item.model_dump(mode="json") for item in signals],
                    "evidence": [item.text for item in evidence],
                    "allowed_actions": [item.value for item in ChallengeAction],
                },
                LiveChallengeOutput,
            )
        )
        if output.action is ChallengeAction.UPHOLD:
            return ChallengeResult(output.action, verdict)
        if output.action is ChallengeAction.ABSTAIN:
            return ChallengeResult(output.action, verdict, True)
        matched = next(
            (
                item
                for item in evidence
                if output.evidence_span is not None
                and output.evidence_span in item.text
            ),
            None,
        )
        if output.evidence_span is not None and matched is None:
            raise ChallengerScopeError("challenger evidence escaped scope")
        spans = (
            (
                EvidenceSpan(
                    source_id=matched.chunk.source_id,
                    text=output.evidence_span,
                    locator=matched.chunk.locator,
                ),
            )
            if matched is not None and output.evidence_span is not None
            else ()
        )
        assert output.revised_relation is not None and output.confidence is not None
        revised = Verdict(
            claim_id=claim.claim_id,
            relation=output.revised_relation,
            corroboration=verdict.corroboration,
            confidence=output.confidence,
            source_ids=tuple(
                dict.fromkeys(verdict.source_ids + tuple(x.source_id for x in spans))
            ),
            evidence_spans=spans,
            reason=output.reason,
            judge_version=CHALLENGER_CONTRACT_VERSION,
        )
        return ChallengeResult(output.action, revised)


__all__ = [
    "CHALLENGER_CONTRACT_VERSION",
    "ChallengeAction",
    "ChallengeResult",
    "ChallengerAgent",
    "ChallengerScopeError",
    "LiveChallengeOutput",
    "is_high_risk_claim",
]
