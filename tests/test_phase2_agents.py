from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from evidencetrace.agents.judge import ClaimJudgeAgent, JudgeScopeError
from evidencetrace.agents.miner import ClaimMinerAgent, MinerScopeError
from evidencetrace.audit_models import (
    MINER_DRAFT_CONTRACT_VERSION,
    CheckSignal,
    EvidenceChunk,
    JudgeInput,
    LiveMinerDraftOutput,
    MinerInput,
    MinerOutput,
    RetrievedEvidence,
)
from evidencetrace.checks.deterministic import deterministic_signals
from evidencetrace.markdown import parse_markdown_file
from evidencetrace.model_client import DeterministicFakeModel, ModelResponseError
from evidencetrace.models import (
    AtomicClaim,
    Checkability,
    Relation,
    SourceMetadata,
)

SOURCE_URL = "https://docs.example.test/nimbus"


def make_source(
    *,
    source_id: str = "s_nimbus",
    url: str = SOURCE_URL,
    status: str = "ok",
) -> SourceMetadata:
    return SourceMetadata(
        source_id=source_id,
        url=url,
        title="Nimbus documentation",
        retrieved_at=datetime(2026, 7, 10, tzinfo=UTC),
        content_hash="content-hash",
        mime_type="text/html",
        status=status,
    )


def make_claim(
    text: str = "Nimbus supports durable runs starting in v1.3.",
    *,
    claim_id: str = "c_0001",
    checkability: Checkability = Checkability.CHECKABLE,
    line_start: int = 12,
    line_end: int = 12,
    citation_urls: tuple[str, ...] = (SOURCE_URL,),
) -> AtomicClaim:
    return AtomicClaim(
        claim_id=claim_id,
        text=text,
        file="docs/nimbus.md",
        line_start=line_start,
        line_end=line_end,
        claim_type="versioned_capability",
        checkability=checkability,
        citation_urls=citation_urls,
    )


def make_evidence(
    text: str = "Nimbus supports durable runs starting in v1.3.",
    *,
    source_id: str = "s_nimbus",
    url: str = SOURCE_URL,
    locator: str = "Guide > Durable runs",
    score: float = 8.0,
) -> RetrievedEvidence:
    chunk = EvidenceChunk(
        source_id=source_id,
        url=url,
        text=text,
        heading_path=("Guide", "Durable runs"),
        locator=locator,
        char_start=0,
        char_end=len(text),
    )
    return RetrievedEvidence(chunk=chunk, text=text, score=score)


def judge_response(
    *,
    relation: str = "entailed",
    evidence_text: str = "supports durable runs",
    **extra: Any,
) -> dict[str, Any]:
    return {
        "relation": relation,
        "confidence": 0.82,
        "reason": "The source directly states the atomic claim.",
        "evidence_span": evidence_text,
        **extra,
    }


def test_agent_inputs_forbid_broad_context_and_tool_fields() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        MinerInput(
            text="Nimbus shipped v1.3.",
            file="docs/nimbus.md",
            line_start=4,
            line_end=4,
            full_document="unrelated report context",  # type: ignore[call-arg]
        )

    with pytest.raises(ValidationError, match="extra_forbidden"):
        JudgeInput(
            claim=make_claim(),
            evidence=(make_evidence(),),
            source=make_source(),
            search_web=True,  # type: ignore[call-arg]
        )


def test_phase2_schemas_reject_invalid_ranges_sources_and_spans() -> None:
    with pytest.raises(ValidationError, match="line_end"):
        MinerInput(
            text="Nimbus shipped.",
            file="docs/nimbus.md",
            line_start=5,
            line_end=4,
        )
    with pytest.raises(ValidationError, match="char_end"):
        EvidenceChunk(
            source_id="s_nimbus",
            url=SOURCE_URL,
            text="real text",
            locator="Guide",
            char_start=9,
            char_end=2,
        )
    with pytest.raises(ValidationError, match="source substring"):
        RetrievedEvidence(
            chunk=make_evidence().chunk,
            text="model-authored quotation",
            score=1.0,
        )
    with pytest.raises(ValidationError, match="source_id"):
        JudgeInput(
            claim=make_claim(),
            evidence=(make_evidence(source_id="s_other"),),
            source=make_source(),
        )


def test_deterministic_miner_preserves_protected_facts_and_line_mapping() -> None:
    output = ClaimMinerAgent().mine(
        MinerInput(
            text=(
                "Nimbus did not ship v1.2 on 2025-03-18 at 82% availability.\n"
                "Nimbus shipped v1.3 on 2025-04-18."
            ),
            file="docs/nimbus.md",
            line_start=20,
            line_end=21,
            heading_path=("Release status",),
            citation_urls=(SOURCE_URL,),
        )
    )

    assert len(output.claims) == 2
    assert [(claim.line_start, claim.line_end) for claim in output.claims] == [
        (20, 20),
        (21, 21),
    ]
    assert "not" in output.claims[0].text
    assert "v1.2" in output.claims[0].text
    assert "2025-03-18" in output.claims[0].text
    assert "82%" in output.claims[0].text
    assert all(claim.citation_urls == (SOURCE_URL,) for claim in output.claims)


def test_miner_filters_incomplete_link_removal_residue() -> None:
    output = ClaimMinerAgent().mine(
        MinerInput(
            text=(
                "Nimbus became generally available on 2025-03-18. The launch record is"
            ),
            file="examples/bad-agent-comparison.md",
            line_start=23,
            line_end=23,
            citation_urls=(SOURCE_URL,),
        )
    )

    assert [claim.text for claim in output.claims] == [
        "Nimbus became generally available on 2025-03-18."
    ]


def test_bad_fixture_mines_exactly_eight_complete_claims() -> None:
    project_root = Path(__file__).parents[1]
    document = parse_markdown_file(
        project_root / "examples" / "bad-agent-comparison.md",
        repo_root=project_root,
    )
    citations = {item.citation_id: item for item in document.citations}
    claims = []
    for paragraph in document.paragraphs:
        if not paragraph.citation_ids:
            continue
        urls = tuple(
            url
            for citation_id in paragraph.citation_ids
            for url in citations[citation_id].resolved_urls
        )
        claims.extend(
            ClaimMinerAgent()
            .mine(
                MinerInput(
                    text=paragraph.plain_text,
                    file=document.path,
                    line_start=paragraph.source.line_start,
                    line_end=paragraph.source.line_end,
                    heading_path=paragraph.heading_path,
                    citation_urls=urls,
                )
            )
            .claims
        )

    assert len(claims) == 8
    assert [claim.line_start for claim in claims] == [9, 12, 15, 20, 23, 26, 29, 32]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("claim_id", "model_claim"),
        ("file", "docs/other.md"),
        ("line_start", 9),
        ("line_end", 10),
        ("citation_urls", ["https://invented.example/source"]),
        ("slots", {"version": "v9.9"}),
    ],
)
def test_live_miner_draft_rejects_model_authored_metadata(
    field: str, value: object
) -> None:
    claim_payload: dict[str, object] = {
        "text": "Nimbus shipped v1.2.",
        "claim_type": "versioned_capability",
        "checkability": "checkable",
        field: value,
    }
    model = DeterministicFakeModel(lambda _task, _payload: {"claims": [claim_payload]})

    with pytest.raises(ModelResponseError, match="schema validation"):
        ClaimMinerAgent(model).mine(
            MinerInput(
                text="Nimbus shipped v1.2.",
                file="docs/nimbus.md",
                line_start=10,
                line_end=10,
                citation_urls=(SOURCE_URL,),
            )
        )


@pytest.mark.parametrize(
    ("text", "code"),
    [
        ("Orion shipped v1.2.", "non_source_span"),
        ("The launch record is", "invalid_claim_fragment"),
    ],
)
def test_live_miner_rejects_non_source_or_incomplete_claims(
    text: str, code: str
) -> None:
    model = DeterministicFakeModel(
        lambda _task, _payload: {
            "claims": [
                {
                    "text": text,
                    "claim_type": "versioned_capability",
                    "checkability": "checkable",
                }
            ]
        }
    )

    with pytest.raises(MinerScopeError) as raised:
        ClaimMinerAgent(model).mine(
            MinerInput(
                text="Nimbus shipped v1.2.",
                file="docs/nimbus.md",
                line_start=10,
                line_end=10,
                citation_urls=(SOURCE_URL,),
            )
        )
    assert raised.value.code == code


def test_fake_miner_cannot_drop_number_version_date_or_negation() -> None:
    model = DeterministicFakeModel(
        lambda _task, _payload: {
            "claims": [
                {
                    "text": "Nimbus did ship availability.",
                    "claim_type": "factual_statement",
                    "checkability": "checkable",
                }
            ]
        }
    )
    input_data = MinerInput(
        text="Nimbus did not ship v1.2 on 2025-03-18 at 82% availability.",
        file="docs/nimbus.md",
        line_start=10,
        line_end=10,
        citation_urls=(SOURCE_URL,),
    )

    with pytest.raises(MinerScopeError) as raised:
        ClaimMinerAgent(model).mine(input_data)
    assert raised.value.code == "missing_protected_token"


def test_live_miner_cannot_drop_a_qualifier() -> None:
    model = DeterministicFakeModel(
        lambda _task, _payload: {
            "claims": [
                {
                    "text": "Cobalt supports enterprise regions.",
                    "claim_type": "factual_statement",
                    "checkability": "checkable",
                }
            ]
        }
    )

    with pytest.raises(MinerScopeError) as raised:
        ClaimMinerAgent(model).mine(
            MinerInput(
                text="Only Cobalt supports enterprise regions.",
                file="docs/cobalt.md",
                line_start=4,
                line_end=4,
            )
        )

    assert raised.value.code == "missing_protected_token"


def test_fake_miner_preserves_decimal_percentage_without_false_drop() -> None:
    text = "Nimbus had a 1.8% error rate under a 64-client load."
    model = DeterministicFakeModel(
        lambda _task, _payload: {
            "claims": [
                {
                    "text": text,
                    "claim_type": "numeric_measurement",
                    "checkability": "checkable",
                }
            ]
        }
    )

    output = ClaimMinerAgent(model).mine(
        MinerInput(
            text=text,
            file="docs/nimbus.md",
            line_start=10,
            line_end=10,
            citation_urls=(SOURCE_URL,),
        )
    )

    assert output.claims[0].text == text


def test_fake_miner_cannot_drop_protected_fact_via_empty_claims() -> None:
    model = DeterministicFakeModel(lambda _task, _payload: {"claims": []})

    with pytest.raises(MinerScopeError) as raised:
        ClaimMinerAgent(model).mine(
            MinerInput(
                text="Nimbus had a 1.8% error rate.",
                file="docs/nimbus.md",
                line_start=10,
                line_end=10,
                citation_urls=(SOURCE_URL,),
            )
        )
    assert raised.value.code == "missing_protected_token"


def test_fake_miner_receives_only_paragraph_contract() -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def handler(task: str, payload: dict[str, Any]) -> dict[str, Any]:
        calls.append((task, payload))
        return {
            "claims": [
                {
                    "text": "Nimbus shipped v1.3.",
                    "claim_type": "versioned_capability",
                    "checkability": "checkable",
                }
            ]
        }

    output = ClaimMinerAgent(DeterministicFakeModel(handler)).mine(
        MinerInput(
            text="Nimbus shipped v1.3.",
            file="docs/nimbus.md",
            line_start=7,
            line_end=7,
            heading_path=("Releases",),
            citation_urls=(SOURCE_URL,),
        )
    )

    assert len(output.claims) == 1
    claim = output.claims[0]
    assert claim.claim_id == "c_0001"
    assert claim.file == "docs/nimbus.md"
    assert claim.line_start == claim.line_end == 7
    assert claim.citation_urls == (SOURCE_URL,)
    assert claim.slots == {}
    assert MINER_DRAFT_CONTRACT_VERSION == "live-miner-draft-v3"
    assert output.prompt_version == MINER_DRAFT_CONTRACT_VERSION
    assert calls[0][0] == "claim_mining"
    assert set(calls[0][1]) == {
        "text",
        "file",
        "line_start",
        "line_end",
        "heading_path",
        "citation_urls",
        "miner_contract_version",
    }
    assert calls[0][1]["miner_contract_version"] == MINER_DRAFT_CONTRACT_VERSION


def test_live_miner_maps_multiple_lines_and_order_locally() -> None:
    paragraph = "Background context.\nAurora ships v3.4.\nAurora supports 12 zones."
    model = DeterministicFakeModel(
        lambda _task, _payload: {
            "claims": [
                {
                    "text": "Aurora ships v3.4.",
                    "claim_type": "versioned_capability",
                    "checkability": "checkable",
                },
                {
                    "text": "Aurora supports 12 zones.",
                    "claim_type": "numeric_measurement",
                    "checkability": "checkable",
                },
            ]
        }
    )

    output = ClaimMinerAgent(model).mine(
        MinerInput(
            text=paragraph,
            file="docs/aurora.md",
            line_start=40,
            line_end=42,
            citation_urls=("https://docs.example.test/aurora",),
        )
    )

    assert [claim.claim_id for claim in output.claims] == ["c_0001", "c_0002"]
    assert [claim.line_start for claim in output.claims] == [41, 42]
    assert [claim.line_end for claim in output.claims] == [41, 42]
    assert all(claim.slots == {} for claim in output.claims)


def test_live_miner_duplicate_spans_are_monotonic_or_fail_ambiguous() -> None:
    text = "Beacon is active.\nBeacon is active."
    draft = {
        "text": "Beacon is active.",
        "claim_type": "factual_statement",
        "checkability": "checkable",
    }
    input_data = MinerInput(
        text=text,
        file="docs/beacon.md",
        line_start=8,
        line_end=9,
    )
    ordered = ClaimMinerAgent(
        DeterministicFakeModel(lambda _task, _payload: {"claims": [draft, draft]})
    ).mine(input_data)

    assert [claim.line_start for claim in ordered.claims] == [8, 9]

    ambiguous = ClaimMinerAgent(
        DeterministicFakeModel(lambda _task, _payload: {"claims": [draft]})
    )
    with pytest.raises(MinerScopeError) as raised:
        ambiguous.mine(input_data)
    assert raised.value.code == "ambiguous_span"


def test_live_miner_many_repeated_spans_fail_closed_without_path_explosion() -> None:
    text = "\n".join(["Quartz is ready."] * 25)
    draft = {
        "text": "Quartz is ready.",
        "claim_type": "factual_statement",
        "checkability": "checkable",
    }
    model = DeterministicFakeModel(lambda _task, _payload: {"claims": [draft] * 26})

    with pytest.raises(MinerScopeError) as raised:
        ClaimMinerAgent(model).mine(
            MinerInput(
                text=text,
                file="docs/quartz.md",
                line_start=1,
                line_end=25,
            )
        )

    assert raised.value.code == "non_source_span"


def test_miner_scope_matching_does_not_depend_on_python_recursion_depth() -> None:
    line_count = 1100
    paragraph = "\n".join(f"Node{index} remains ready." for index in range(line_count))

    output = ClaimMinerAgent().mine(
        MinerInput(
            text=paragraph,
            file="docs/nodes.md",
            line_start=1,
            line_end=line_count,
        )
    )

    assert len(output.claims) == line_count
    assert output.claims[-1].claim_id == "c_1100"
    assert output.claims[-1].line_start == line_count


def test_miner_scope_error_contains_only_allowlisted_code() -> None:
    secret = "private-claim-payload"
    error = MinerScopeError("non_source_span")

    assert error.code == "non_source_span"
    assert secret not in repr(error)
    assert secret not in str(error)
    assert error.args == ("live Miner output violated paragraph scope",)
    with pytest.raises(ValueError, match="unsupported Miner scope code"):
        MinerScopeError("unknown")  # type: ignore[arg-type]


def test_live_draft_and_public_miner_schemas_keep_separate_contracts() -> None:
    draft = {
        "claims": [
            {
                "text": "Cinder ships v2.1.",
                "claim_type": "versioned_capability",
                "checkability": "checkable",
            }
        ]
    }
    assert LiveMinerDraftOutput.model_validate(draft).claims[0].text == (
        "Cinder ships v2.1."
    )
    with pytest.raises(ValidationError, match="missing"):
        LiveMinerDraftOutput.model_validate({})
    for forbidden in ("model_id", "prompt_version"):
        with pytest.raises(ValidationError, match="extra_forbidden"):
            LiveMinerDraftOutput.model_validate({**draft, forbidden: "model-value"})

    legacy = MinerOutput(claims=(make_claim("Nimbus shipped v1.3."),))
    restored = MinerOutput.model_validate_json(legacy.model_dump_json())
    assert restored == legacy
    assert restored.claims[0].slots == {}


def test_miner_model_output_always_passes_schema_validation() -> None:
    duplicate = make_claim(
        "Nimbus shipped v1.3.",
        line_start=7,
        line_end=7,
    ).model_dump(mode="json")
    model = DeterministicFakeModel(
        lambda _task, _payload: {"claims": [duplicate, duplicate]}
    )

    with pytest.raises(ModelResponseError, match="schema validation"):
        ClaimMinerAgent(model).mine(
            MinerInput(
                text="Nimbus shipped v1.3.",
                file="docs/nimbus.md",
                line_start=7,
                line_end=7,
                citation_urls=(SOURCE_URL,),
            )
        )


def test_fake_judge_receives_narrow_input_and_can_select_real_subspan() -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def handler(task: str, payload: dict[str, Any]) -> dict[str, Any]:
        calls.append((task, payload))
        return judge_response()

    output = ClaimJudgeAgent(DeterministicFakeModel(handler)).judge(
        JudgeInput(
            claim=make_claim(),
            evidence=(make_evidence(),),
            source=make_source(),
        )
    )

    assert output.verdict.relation is Relation.ENTAILED
    assert output.verdict.evidence_spans[0].text == "supports durable runs"
    assert output.verdict.claim_id == "c_0001"
    assert output.verdict.source_ids == ("s_nimbus",)
    assert output.verdict.evidence_spans[0].source_id == "s_nimbus"
    assert output.verdict.evidence_spans[0].locator == "Guide > Durable runs"
    assert calls[0][0] == "claim_judgement"
    assert set(calls[0][1]) == {
        "claim",
        "evidence",
        "source",
        "signals",
        "relation_definitions",
    }
    assert set(calls[0][1]["claim"]) == {
        "text",
        "claim_type",
        "slots",
        "checkability",
    }
    assert set(calls[0][1]["evidence"][0]) == {"text", "score"}
    assert calls[0][1]["source"] == {"status": "ok"}


def test_fake_judge_cannot_invent_evidence_span() -> None:
    model = DeterministicFakeModel(
        lambda _task, _payload: judge_response(
            evidence_text="a quotation the source never contained"
        )
    )

    with pytest.raises(JudgeScopeError) as caught:
        ClaimJudgeAgent(model).judge(
            JudgeInput(
                claim=make_claim(),
                evidence=(make_evidence(),),
                source=make_source(),
            )
        )

    assert caught.value.code == "evidence_span_out_of_scope"


@pytest.mark.parametrize("field", ["claim_id", "source_id", "locator", "judge_version"])
def test_fake_judge_cannot_return_locally_owned_fields(field: str) -> None:
    model = DeterministicFakeModel(
        lambda _task, _payload: judge_response(**{field: "model-controlled"})
    )

    with pytest.raises(ModelResponseError, match="schema validation"):
        ClaimJudgeAgent(model).judge(
            JudgeInput(
                claim=make_claim(),
                evidence=(make_evidence(),),
                source=make_source(),
            )
        )


@pytest.mark.parametrize(
    "response",
    [
        {
            "relation": "supported",
            "confidence": 0.8,
            "reason": "Invalid label.",
            "evidence_span": "supports durable runs",
        },
        {
            "relation": "entailed",
            "confidence": 0.8,
            "reason": "Missing required span.",
        },
    ],
)
def test_judge_model_output_always_passes_schema_validation(
    response: dict[str, Any],
) -> None:
    model = DeterministicFakeModel(lambda _task, _payload: response)

    with pytest.raises(ModelResponseError, match="schema validation"):
        ClaimJudgeAgent(model).judge(
            JudgeInput(
                claim=make_claim(),
                evidence=(make_evidence(),),
                source=make_source(),
            )
        )


def test_judge_abstains_without_evidence_or_with_low_overlap() -> None:
    no_evidence = (
        ClaimJudgeAgent()
        .judge(JudgeInput(claim=make_claim(), source=make_source()))
        .verdict
    )
    unrelated = (
        ClaimJudgeAgent()
        .judge(
            JudgeInput(
                claim=make_claim(
                    "Nimbus guarantees exactly-once execution across regions."
                ),
                evidence=(
                    make_evidence(
                        "Nimbus retries failed tasks with exponential backoff "
                        "in a single region."
                    ),
                ),
                source=make_source(),
            )
        )
        .verdict
    )

    assert no_evidence.relation is Relation.NOT_IN_SOURCE
    assert no_evidence.evidence_spans == ()
    assert unrelated.relation is Relation.NOT_IN_SOURCE
    assert unrelated.evidence_spans == ()
    assert "Abstaining" in unrelated.reason


def test_judge_uses_all_six_contract_labels() -> None:
    assert {relation.value for relation in Relation} == {
        "entailed",
        "partially_entailed",
        "contradicted",
        "not_in_source",
        "source_unavailable",
        "not_checkable",
    }

    exact = (
        ClaimJudgeAgent()
        .judge(
            JudgeInput(
                claim=make_claim(),
                evidence=(make_evidence(),),
                source=make_source(),
            )
        )
        .verdict
    )
    partial_claim = make_claim(
        "Nimbus supports durable retries across multiple regions."
    )
    partial_text = "Nimbus supports durable retries in one region."
    partial = (
        ClaimJudgeAgent()
        .judge(
            JudgeInput(
                claim=partial_claim,
                evidence=(make_evidence(partial_text),),
                source=make_source(),
            )
        )
        .verdict
    )
    mismatch_claim = make_claim("Nimbus completed 82% of locked tasks.")
    mismatch_text = "Nimbus completed 62% of the locked tasks."
    contradicted = (
        ClaimJudgeAgent()
        .judge(
            JudgeInput(
                claim=mismatch_claim,
                evidence=(make_evidence(mismatch_text),),
                source=make_source(),
                signals=deterministic_signals(
                    mismatch_claim.text,
                    mismatch_text,
                ),
            )
        )
        .verdict
    )
    unavailable = (
        ClaimJudgeAgent()
        .judge(
            JudgeInput(
                claim=make_claim(),
                source=make_source(status="unavailable"),
            )
        )
        .verdict
    )
    not_checkable = (
        ClaimJudgeAgent()
        .judge(
            JudgeInput(
                claim=make_claim(
                    "We should prefer Nimbus.",
                    checkability=Checkability.NOT_CHECKABLE,
                ),
                source=make_source(),
            )
        )
        .verdict
    )

    assert exact.relation is Relation.ENTAILED
    assert partial.relation is Relation.PARTIALLY_ENTAILED
    assert contradicted.relation is Relation.CONTRADICTED
    assert unavailable.relation is Relation.SOURCE_UNAVAILABLE
    assert not_checkable.relation is Relation.NOT_CHECKABLE


def test_unavailable_and_not_checkable_inputs_bypass_model() -> None:
    def fail_if_called(_task: str, _payload: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError("model must not be called for deterministic abstention")

    judge = ClaimJudgeAgent(DeterministicFakeModel(fail_if_called))
    unavailable = judge.judge(
        JudgeInput(
            claim=make_claim(),
            source=make_source(status="unavailable"),
        )
    )
    not_checkable = judge.judge(
        JudgeInput(
            claim=make_claim(
                "We recommend Nimbus.",
                checkability=Checkability.NOT_CHECKABLE,
            ),
            source=make_source(),
        )
    )

    assert unavailable.verdict.relation is Relation.SOURCE_UNAVAILABLE
    assert not_checkable.verdict.relation is Relation.NOT_CHECKABLE


def test_judge_rejects_entailed_when_deterministic_slots_mismatch() -> None:
    model = DeterministicFakeModel(
        lambda _task, _payload: judge_response(
            evidence_text="Nimbus completed 62% of the locked tasks."
        )
    )
    claim = make_claim("Nimbus completed 82% of the locked tasks.")
    source_text = "Nimbus completed 62% of the locked tasks."
    signals = deterministic_signals(claim.text, source_text)

    with pytest.raises(ValueError, match="deterministic mismatch"):
        ClaimJudgeAgent(model).judge(
            JudgeInput(
                claim=claim,
                evidence=(make_evidence(source_text),),
                source=make_source(),
                signals=signals,
            )
        )


def test_check_signal_detail_and_miner_claim_ids_are_schema_validated() -> None:
    with pytest.raises(ValidationError, match="detail"):
        CheckSignal(code="comparison_reminder", detail=" ")

    duplicate = make_claim().model_copy(update={"claim_id": "c_duplicate"})
    with pytest.raises(ValidationError, match="unique"):
        MinerOutput(claims=(duplicate, duplicate))
