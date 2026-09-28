"""Source-grounded Claim Judge with no fetch or filesystem permissions."""

from __future__ import annotations

import re
from typing import Literal

from evidencetrace.audit_models import (
    JUDGE_SEMANTIC_CONTRACT_VERSION,
    JUDGE_VERDICT_OWNERSHIP_POLICY_VERSION,
    JudgeInput,
    JudgeOutput,
    LiveJudgeSemanticOutput,
)
from evidencetrace.checks.deterministic import (
    CAUTION_CODES,
    HARD_CONFLICT_CODES,
    DeterministicConflictError,
    bounded_evidence_text,
    deterministic_signals,
    is_recommendation,
    validate_evidence_span,
)
from evidencetrace.model_client import ModelClient
from evidencetrace.models import CorroborationStatus, EvidenceSpan, Relation, Verdict
from evidencetrace.relation_policy import RELATION_DEFINITIONS_BILINGUAL

STOPWORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "of",
    "to",
    "in",
    "on",
    "and",
    "for",
    "with",
}
SUBSTANTIVE_RELATIONS = {
    Relation.ENTAILED,
    Relation.PARTIALLY_ENTAILED,
    Relation.CONTRADICTED,
}
SLOT_MISMATCH_CODES = frozenset(
    {"numeric_mismatch", "date_mismatch", "version_mismatch"}
)
PARTIAL_OVERLAP_THRESHOLD = 0.4
ENTAILED_OVERLAP_THRESHOLD = 0.75
LEXICAL_ENTAILMENT_CONFIDENCE_CAP = 0.74
CAUTION_CONFIDENCE_CAP = 0.69
JUDGE_OUTPUT_VALIDATION_VERSION = "judge-output-validation-v3"

JudgeScopeCode = Literal[
    "claim_id_mismatch",
    "source_id_out_of_scope",
    "evidence_span_out_of_scope",
    "source_availability_mismatch",
    "missing_substantive_evidence",
]
_JUDGE_SCOPE_CODES: frozenset[str] = frozenset(
    {
        "claim_id_mismatch",
        "source_id_out_of_scope",
        "evidence_span_out_of_scope",
        "source_availability_mismatch",
        "missing_substantive_evidence",
    }
)


class JudgeScopeError(ValueError):
    """Payload-free rejection of a live Judge integrity violation."""

    def __init__(self, code: JudgeScopeCode) -> None:
        if code not in _JUDGE_SCOPE_CODES:
            raise ValueError("unsupported Judge scope code")
        super().__init__("live Judge output violated bounded input scope")
        self.code = code


def _words(value: str) -> set[str]:
    return {
        token.casefold()
        for token in re.findall(
            r"[A-Za-z0-9]+(?:[._-][A-Za-z0-9]+)*",
            value,
        )
        if token.casefold() not in STOPWORDS
    }


class ClaimJudgeAgent:
    """Receives one claim, one source, candidate spans, and deterministic signals."""

    def __init__(self, client: ModelClient | None = None) -> None:
        self.client = client

    def judge(self, input_data: JudgeInput) -> JudgeOutput:
        claim = input_data.claim
        if claim.checkability.value == "not_checkable" or is_recommendation(claim.text):
            return JudgeOutput(
                verdict=Verdict(
                    claim_id=claim.claim_id,
                    relation=Relation.NOT_CHECKABLE,
                    confidence=0.95,
                    reason="The text is an opinion or recommendation.",
                    judge_version="deterministic-judge-v1",
                )
            )
        if input_data.source.status == "unavailable":
            return JudgeOutput(
                verdict=Verdict(
                    claim_id=claim.claim_id,
                    relation=Relation.SOURCE_UNAVAILABLE,
                    confidence=0.0,
                    source_ids=(input_data.source.source_id,),
                    reason="The cited source is unavailable.",
                    judge_version="deterministic-judge-v1",
                )
            )
        if not input_data.evidence:
            return JudgeOutput(
                verdict=Verdict(
                    claim_id=claim.claim_id,
                    relation=Relation.NOT_IN_SOURCE,
                    confidence=0.0,
                    source_ids=(input_data.source.source_id,),
                    reason="No candidate evidence span was retrieved; abstaining.",
                    judge_version="deterministic-judge-v1",
                )
            )
        input_data = self._with_derived_signals(input_data)
        if self.client is not None:
            semantic = LiveJudgeSemanticOutput.model_validate(
                self.client.complete_model(
                    "claim_judgement",
                    self._live_payload(input_data),
                    LiveJudgeSemanticOutput,
                )
            )
            output = self._assemble_live_output(semantic, input_data)
            return self._validate_output(output, input_data)
        best = input_data.evidence[0]
        span_error = validate_evidence_span(best.text, best.chunk.text)
        if span_error is not None:
            return JudgeOutput(
                verdict=Verdict(
                    claim_id=claim.claim_id,
                    relation=Relation.NOT_IN_SOURCE,
                    confidence=0.0,
                    source_ids=(input_data.source.source_id,),
                    reason=span_error.detail,
                    judge_version="deterministic-judge-v1",
                )
            )
        claim_words = _words(claim.text)
        overlap = len(claim_words & _words(best.text)) / max(len(claim_words), 1)
        signal_codes = {signal.code for signal in input_data.signals}
        slot_mismatch = bool(signal_codes & SLOT_MISMATCH_CODES)
        entity_mismatch = "entity_mismatch" in signal_codes
        negation_mismatch = "negation_mismatch" in signal_codes
        conjunction_mismatch = "conjunction_mismatch" in signal_codes
        qualifier_mismatch = "qualifier_mismatch" in signal_codes
        qualifier_conflict = any(
            signal.code == "qualifier_mismatch" and signal.severity == "error"
            for signal in input_data.signals
        )
        if slot_mismatch and overlap >= PARTIAL_OVERLAP_THRESHOLD:
            relation = Relation.CONTRADICTED
            reason = "A deterministic numeric, date, or version slot conflicts."
        elif entity_mismatch and overlap >= PARTIAL_OVERLAP_THRESHOLD:
            relation = Relation.CONTRADICTED
            reason = "Aligned predicate/context refers to a different entity."
        elif negation_mismatch and overlap >= 0.5:
            relation = Relation.CONTRADICTED
            reason = "Aligned claim and source text differ in negation."
        elif qualifier_conflict and overlap >= PARTIAL_OVERLAP_THRESHOLD:
            relation = Relation.CONTRADICTED
            reason = "Aligned claim and source use opposing scope qualifiers."
        elif (conjunction_mismatch or qualifier_mismatch) and overlap >= 0.4:
            relation = Relation.PARTIALLY_ENTAILED
            reason = "The source omits part of the claim or uses a narrower scope."
        elif overlap >= ENTAILED_OVERLAP_THRESHOLD:
            relation = Relation.ENTAILED
            reason = "The candidate source span covers the claim's lexical content."
        elif overlap >= PARTIAL_OVERLAP_THRESHOLD:
            relation = Relation.PARTIALLY_ENTAILED
            reason = (
                "The source span overlaps the claim but does not cover all content."
            )
        else:
            relation = Relation.NOT_IN_SOURCE
            reason = "The related source span does not contain enough claim content."
        if relation is Relation.NOT_IN_SOURCE:
            return JudgeOutput(
                verdict=Verdict(
                    claim_id=claim.claim_id,
                    relation=relation,
                    corroboration=CorroborationStatus.CITED_ONLY,
                    confidence=0.0,
                    source_ids=(input_data.source.source_id,),
                    reason=reason + " Abstaining.",
                    judge_version="deterministic-judge-v1",
                )
            )
        confidence = min(max(best.score / 10.0, 0.1), 0.99)
        if relation is Relation.ENTAILED:
            confidence = min(confidence, LEXICAL_ENTAILMENT_CONFIDENCE_CAP)
        if signal_codes & CAUTION_CODES:
            confidence = min(confidence, CAUTION_CONFIDENCE_CAP)
        verdict = Verdict(
            claim_id=claim.claim_id,
            relation=relation,
            corroboration=CorroborationStatus.CITED_ONLY,
            confidence=confidence,
            source_ids=(input_data.source.source_id,),
            evidence_spans=(
                EvidenceSpan(
                    source_id=input_data.source.source_id,
                    text=best.text,
                    locator=best.chunk.locator,
                ),
            ),
            reason=reason,
            judge_version="deterministic-judge-v1",
        )
        return JudgeOutput(verdict=verdict)

    @staticmethod
    def _live_payload(input_data: JudgeInput) -> dict[str, object]:
        claim = input_data.claim
        return {
            "claim": {
                "text": claim.text,
                "claim_type": claim.claim_type,
                "slots": claim.slots,
                "checkability": claim.checkability.value,
            },
            "evidence": [
                {"text": item.text, "score": item.score} for item in input_data.evidence
            ],
            "signals": [
                signal.model_dump(mode="json") for signal in input_data.signals
            ],
            "source": {"status": input_data.source.status},
            "relation_definitions": RELATION_DEFINITIONS_BILINGUAL,
        }

    def _assemble_live_output(
        self,
        semantic: LiveJudgeSemanticOutput,
        input_data: JudgeInput,
    ) -> JudgeOutput:
        spans: tuple[EvidenceSpan, ...] = ()
        if semantic.evidence_span is not None:
            matched = next(
                (
                    item
                    for item in input_data.evidence
                    if semantic.evidence_span in item.text
                ),
                None,
            )
            if matched is None:
                raise JudgeScopeError("evidence_span_out_of_scope")
            spans = (
                EvidenceSpan(
                    source_id=input_data.source.source_id,
                    text=semantic.evidence_span,
                    locator=matched.chunk.locator,
                ),
            )
        verdict = Verdict(
            claim_id=input_data.claim.claim_id,
            relation=semantic.relation,
            corroboration=CorroborationStatus.CITED_ONLY,
            confidence=semantic.confidence,
            source_ids=(input_data.source.source_id,),
            evidence_spans=spans,
            reason=semantic.reason,
            judge_version=JUDGE_VERDICT_OWNERSHIP_POLICY_VERSION,
        )
        return JudgeOutput(
            verdict=verdict,
            model_id=self.client.model_id if self.client is not None else "local",
            prompt_version=JUDGE_SEMANTIC_CONTRACT_VERSION,
        )

    @staticmethod
    def _with_derived_signals(input_data: JudgeInput) -> JudgeInput:
        """Ensure live and deterministic Judges see the same rule constraints."""

        best = input_data.evidence[0]
        derived = deterministic_signals(
            input_data.claim.text,
            best.text,
            bounded_evidence_context=bounded_evidence_text(input_data.evidence),
        )
        existing_codes = {signal.code for signal in input_data.signals}
        merged = input_data.signals + tuple(
            signal for signal in derived if signal.code not in existing_codes
        )
        if merged == input_data.signals:
            return input_data
        return input_data.model_copy(update={"signals": merged})

    @staticmethod
    def _validate_output(output: JudgeOutput, input_data: JudgeInput) -> JudgeOutput:
        verdict = output.verdict
        if verdict.claim_id != input_data.claim.claim_id:
            raise JudgeScopeError("claim_id_mismatch")
        if not set(verdict.source_ids) <= {input_data.source.source_id}:
            raise JudgeScopeError("source_id_out_of_scope")
        for span in verdict.evidence_spans:
            real_span = any(
                span.source_id == item.chunk.source_id
                and span.locator == item.chunk.locator
                and span.text in item.text
                for item in input_data.evidence
            )
            if not real_span:
                raise JudgeScopeError("evidence_span_out_of_scope")
        if (
            verdict.relation is Relation.SOURCE_UNAVAILABLE
            and input_data.source.status != "unavailable"
        ):
            raise JudgeScopeError("source_availability_mismatch")
        signal_codes = {signal.code for signal in input_data.signals}
        blocking_codes = signal_codes & HARD_CONFLICT_CODES
        if any(
            signal.code == "qualifier_mismatch" and signal.severity == "error"
            for signal in input_data.signals
        ):
            blocking_codes.add("qualifier_mismatch")
        if blocking_codes and verdict.relation is Relation.ENTAILED:
            raise DeterministicConflictError(blocking_codes)
        if verdict.relation in SUBSTANTIVE_RELATIONS and not verdict.evidence_spans:
            raise JudgeScopeError("missing_substantive_evidence")
        if signal_codes & CAUTION_CODES and verdict.confidence > CAUTION_CONFIDENCE_CAP:
            verdict = verdict.model_copy(update={"confidence": CAUTION_CONFIDENCE_CAP})
            return output.model_copy(update={"verdict": verdict})
        return output


__all__ = [
    "JUDGE_OUTPUT_VALIDATION_VERSION",
    "ClaimJudgeAgent",
    "DeterministicConflictError",
    "JudgeScopeCode",
    "JudgeScopeError",
]
