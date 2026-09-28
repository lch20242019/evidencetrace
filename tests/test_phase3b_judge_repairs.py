from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from evidencetrace.agents.judge import ClaimJudgeAgent
from evidencetrace.agents.miner import ClaimMinerAgent
from evidencetrace.audit_models import (
    EvidenceChunk,
    JudgeInput,
    MinerInput,
    RetrievedEvidence,
)
from evidencetrace.checks.deterministic import deterministic_signals
from evidencetrace.model_client import DeterministicFakeModel
from evidencetrace.models import (
    AtomicClaim,
    Checkability,
    Relation,
    SourceMetadata,
)

SOURCE_ID = "s_cobalt"
SOURCE_URL = "https://manual.example.test/cobalt"
LOCATOR = "Reference > Behavior"


def make_claim(text: str) -> AtomicClaim:
    return AtomicClaim(
        claim_id="c_generic",
        text=text,
        file="docs/cobalt.md",
        line_start=8,
        line_end=8,
        claim_type="factual_statement",
        checkability=Checkability.CHECKABLE,
        citation_urls=(SOURCE_URL,),
    )


def make_evidence(text: str, *, score: float = 9.8) -> RetrievedEvidence:
    chunk = EvidenceChunk(
        source_id=SOURCE_ID,
        url=SOURCE_URL,
        text=text,
        heading_path=("Reference", "Behavior"),
        locator=LOCATOR,
        char_start=0,
        char_end=len(text),
    )
    return RetrievedEvidence(chunk=chunk, text=text, score=score)


def make_source() -> SourceMetadata:
    return SourceMetadata(
        source_id=SOURCE_ID,
        url=SOURCE_URL,
        title="Cobalt reference",
        retrieved_at=datetime(2026, 7, 10, tzinfo=UTC),
        content_hash="fixture-content-hash",
        mime_type="text/plain",
        status="ok",
    )


def live_entailed(text: str, *, confidence: float = 0.96) -> dict[str, Any]:
    return {
        "relation": "entailed",
        "confidence": confidence,
        "evidence_span": text,
        "reason": "The model selected the supplied source text.",
    }


@pytest.mark.parametrize(
    "text",
    [
        "We recommend Cobalt for new deployments.",
        "The team recommends Cobalt for new deployments.",
        "The team recommended Cobalt for new deployments.",
        "The team is recommending Cobalt for new deployments.",
        "Operators should prefer Cobalt for new deployments.",
    ],
)
def test_recommendation_morphology_is_not_checkable(text: str) -> None:
    output = ClaimMinerAgent().mine(
        MinerInput(
            text=text,
            file="docs/advice.md",
            line_start=3,
            line_end=3,
            citation_urls=(SOURCE_URL,),
        )
    )

    assert output.claims[0].checkability is Checkability.NOT_CHECKABLE


def test_judge_independently_blocks_checkable_recommendation() -> None:
    called = False

    def handler(_task: str, _payload: dict[str, Any]) -> dict[str, Any]:
        nonlocal called
        called = True
        raise AssertionError("recommendations must bypass the live Judge")

    verdict = (
        ClaimJudgeAgent(DeterministicFakeModel(handler))
        .judge(
            JudgeInput(
                claim=make_claim("Operators should prefer Cobalt."),
                evidence=(make_evidence("Operators prefer Cobalt."),),
                source=make_source(),
            )
        )
        .verdict
    )

    assert verdict.relation is Relation.NOT_CHECKABLE
    assert called is False


def test_negation_conflict_has_one_error_contract_and_is_enforced() -> None:
    claim_text = "Cobalt does not retain audit records."
    source_text = "Cobalt retains audit records."
    signals = deterministic_signals(claim_text, source_text)

    assert [(signal.code, signal.severity) for signal in signals] == [
        ("negation_mismatch", "error")
    ]
    verdict = (
        ClaimJudgeAgent()
        .judge(
            JudgeInput(
                claim=make_claim(claim_text),
                evidence=(make_evidence(source_text),),
                source=make_source(),
            )
        )
        .verdict
    )
    assert verdict.relation is Relation.CONTRADICTED


def test_unaligned_negation_does_not_create_a_hard_conflict() -> None:
    codes = {
        signal.code
        for signal in deterministic_signals(
            "Cobalt does not retain audit records.",
            "Quartz compresses image archives.",
        )
    }
    assert "negation_mismatch" not in codes
    assert "entity_mismatch" not in codes


def test_entity_mismatch_requires_aligned_predicate_context() -> None:
    aligned = deterministic_signals(
        "Cobalt signs release archives.",
        "Quartz signs release archives.",
    )
    unrelated = deterministic_signals(
        "Cobalt signs release archives.",
        "Quartz schedules nightly backups.",
    )

    assert [(item.code, item.severity) for item in aligned] == [
        ("entity_mismatch", "error")
    ]
    assert "entity_mismatch" not in {item.code for item in unrelated}
    verdict = (
        ClaimJudgeAgent()
        .judge(
            JudgeInput(
                claim=make_claim("Cobalt signs release archives."),
                evidence=(make_evidence("Quartz signs release archives."),),
                source=make_source(),
            )
        )
        .verdict
    )
    assert verdict.relation is Relation.CONTRADICTED


def test_pronoun_source_does_not_invent_an_entity_conflict() -> None:
    codes = {
        signal.code
        for signal in deterministic_signals(
            "Cobalt supports durable jobs in one region.",
            "It supports durable jobs in one region.",
        )
    }
    assert "entity_mismatch" not in codes


def test_unrelated_source_clause_qualifier_does_not_downgrade_claim() -> None:
    signals = deterministic_signals(
        "Cobalt 2.0 launched on 2026-02-03.",
        "Cobalt 2.0 launched on 2026-02-03 and serves only premium accounts.",
    )
    assert "qualifier_mismatch" not in {signal.code for signal in signals}


@pytest.mark.parametrize(
    ("claim_text", "source_text"),
    [
        ("Cobalt serves only enterprise accounts.", "Cobalt serves all accounts."),
        ("The interval is narrow for Cobalt.", "The interval is wide for Cobalt."),
    ],
)
def test_opposing_qualifiers_are_contradictions(
    claim_text: str, source_text: str
) -> None:
    signals = deterministic_signals(claim_text, source_text)
    assert ("qualifier_mismatch", "error") in {
        (signal.code, signal.severity) for signal in signals
    }
    verdict = (
        ClaimJudgeAgent()
        .judge(
            JudgeInput(
                claim=make_claim(claim_text),
                evidence=(make_evidence(source_text),),
                source=make_source(),
            )
        )
        .verdict
    )
    assert verdict.relation is Relation.CONTRADICTED


@pytest.mark.parametrize(
    ("claim_text", "source_text"),
    [
        (
            "Cobalt serves only enterprise accounts.",
            "Cobalt serves enterprise accounts.",
        ),
        ("Cobalt operates in all regions.", "Cobalt operates in regions."),
        ("Cobalt covers every account.", "Cobalt covers an account."),
        ("At least 20 samples passed validation.", "20 samples passed validation."),
        ("At most 4 tiers are enabled.", "4 tiers are enabled."),
        (
            "Cobalt passed 8 out of 10 benchmarks.",
            "Cobalt passed 8 benchmarks.",
        ),
        (
            "Cobalt is available in EU regions.",
            "Cobalt is available in regions.",
        ),
        (
            "Cobalt supports premium tiers.",
            "Cobalt supports standard tiers.",
        ),
    ],
)
def test_scope_and_denominator_changes_emit_qualifier_signal(
    claim_text: str, source_text: str
) -> None:
    assert "qualifier_mismatch" in {
        item.code for item in deterministic_signals(claim_text, source_text)
    }


def test_scope_and_missing_conjunct_prefer_partial_support() -> None:
    scoped = (
        ClaimJudgeAgent()
        .judge(
            JudgeInput(
                claim=make_claim("Cobalt supports all enterprise accounts."),
                evidence=(make_evidence("Cobalt supports enterprise accounts."),),
                source=make_source(),
            )
        )
        .verdict
    )
    compound = (
        ClaimJudgeAgent()
        .judge(
            JudgeInput(
                claim=make_claim("Cobalt signs releases and encrypts archives."),
                evidence=(make_evidence("Cobalt signs releases."),),
                source=make_source(),
            )
        )
        .verdict
    )

    assert scoped.relation is Relation.PARTIALLY_ENTAILED
    assert compound.relation is Relation.PARTIALLY_ENTAILED


def test_key_slot_conflict_wins_over_missing_conjunct() -> None:
    verdict = (
        ClaimJudgeAgent()
        .judge(
            JudgeInput(
                claim=make_claim("Cobalt signs 82% of releases and encrypts archives."),
                evidence=(make_evidence("Cobalt signs 62% of releases."),),
                source=make_source(),
            )
        )
        .verdict
    )

    assert verdict.relation is Relation.CONTRADICTED


def test_lexical_entailment_and_cautions_limit_confidence() -> None:
    lexical = (
        ClaimJudgeAgent()
        .judge(
            JudgeInput(
                claim=make_claim("Cobalt signs release archives."),
                evidence=(make_evidence("Cobalt signs release archives."),),
                source=make_source(),
            )
        )
        .verdict
    )
    comparison = (
        ClaimJudgeAgent()
        .judge(
            JudgeInput(
                claim=make_claim("Cobalt is faster than Quartz."),
                evidence=(make_evidence("Cobalt is faster than Quartz."),),
                source=make_source(),
            )
        )
        .verdict
    )

    assert lexical.relation is Relation.ENTAILED
    assert lexical.confidence <= 0.74
    assert comparison.relation is Relation.ENTAILED
    assert comparison.confidence <= 0.69


@pytest.mark.parametrize(
    ("claim_text", "source_text", "expected_code"),
    [
        (
            "Cobalt completes 82% of tasks.",
            "Cobalt completes 62% of tasks.",
            "numeric_mismatch",
        ),
        (
            "Cobalt shipped on 2026-03-18.",
            "Cobalt shipped on 2026-03-19.",
            "date_mismatch",
        ),
        ("Cobalt v3.2 signs builds.", "Cobalt v3.1 signs builds.", "version_mismatch"),
        (
            "Cobalt signs release archives.",
            "Quartz signs release archives.",
            "entity_mismatch",
        ),
        ("Cobalt does not retain logs.", "Cobalt retains logs.", "negation_mismatch"),
    ],
)
def test_live_judge_cannot_bypass_derived_hard_conflicts(
    claim_text: str, source_text: str, expected_code: str
) -> None:
    seen_codes: set[str] = set()

    def handler(_task: str, payload: dict[str, Any]) -> dict[str, Any]:
        seen_codes.update(signal["code"] for signal in payload["signals"])
        return live_entailed(source_text)

    with pytest.raises(ValueError, match="deterministic mismatch"):
        ClaimJudgeAgent(DeterministicFakeModel(handler)).judge(
            JudgeInput(
                claim=make_claim(claim_text),
                evidence=(make_evidence(source_text),),
                source=make_source(),
            )
        )

    assert expected_code in seen_codes


def test_live_judge_caution_signal_caps_confidence_with_real_span() -> None:
    text = "Cobalt is faster than Quartz."
    output = ClaimJudgeAgent(
        DeterministicFakeModel(
            lambda _task, _payload: live_entailed(text, confidence=0.96)
        )
    ).judge(
        JudgeInput(
            claim=make_claim(text),
            evidence=(make_evidence(text),),
            source=make_source(),
        )
    )

    assert output.verdict.relation is Relation.ENTAILED
    assert output.verdict.confidence == 0.69
    assert output.verdict.evidence_spans[0].text == text
