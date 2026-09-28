from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from evidencetrace.models import (
    AuditArtifact,
    CitationKind,
    CitationOccurrence,
    CorroborationStatus,
    EffectiveConfig,
    EvidenceSpan,
    MarkdownParagraph,
    ParagraphKind,
    ParsedDocument,
    PolicyConfig,
    Relation,
    RunMetadata,
    RunPaths,
    SourceSpan,
    Verdict,
)


def source_span(
    *,
    file: str = "docs/example.md",
    start: int = 0,
    end: int = 12,
) -> SourceSpan:
    return SourceSpan(
        file=file,
        line_start=1,
        line_end=1,
        column_start=1,
        column_end=max(end - start, 1),
        offset_start=start,
        offset_end=end,
    )


def run_metadata() -> RunMetadata:
    return RunMetadata(
        run_id="20260710T120000Z-abcdef0",
        started_at=datetime(2026, 7, 10, 12, tzinfo=UTC),
        git_sha="ABCDEF0",
        tool_version="0.0.0",
    )


def test_contracts_forbid_extra_fields_and_are_frozen() -> None:
    span = source_span()

    with pytest.raises(ValidationError, match="extra_forbidden"):
        SourceSpan(**span.model_dump(), unexpected=True)
    with pytest.raises(ValidationError, match="frozen_instance"):
        span.line_end = 2  # type: ignore[misc]


@pytest.mark.parametrize(
    "updates",
    [
        {"line_start": 2, "line_end": 1},
        {"column_start": 5, "column_end": 4},
        {"offset_start": 10, "offset_end": 9},
        {"file": "/absolute.md"},
        {"file": "docs/../secret.md"},
        {"file": r"docs\windows.md"},
    ],
)
def test_source_span_rejects_invalid_locations(updates: dict[str, object]) -> None:
    values = source_span().model_dump()
    values.update(updates)

    with pytest.raises(ValidationError):
        SourceSpan.model_validate(values)


def test_parsed_document_validates_hash_and_child_paths() -> None:
    span = source_span()
    paragraph = MarkdownParagraph(
        paragraph_id="p_001",
        kind=ParagraphKind.PARAGRAPH,
        source=span,
        raw_text="A cited fact.",
        plain_text="A cited fact.",
    )
    citation = CitationOccurrence(
        citation_id="cit_001",
        kind=CitationKind.BARE_URL,
        raw="https://example.com",
        destination="https://example.com",
        resolved_urls=("https://example.com",),
        source=span,
        resolved=True,
    )
    document = ParsedDocument(
        path="docs/example.md",
        content_sha256="A" * 64,
        line_count=1,
        paragraphs=(paragraph,),
        citations=(citation,),
    )

    assert document.content_sha256 == "a" * 64
    with pytest.raises(ValidationError, match="64-character"):
        ParsedDocument.model_validate(
            {**document.model_dump(), "content_sha256": "short"}
        )
    with pytest.raises(ValidationError, match="document path"):
        ParsedDocument(
            path="README.md",
            content_sha256="a" * 64,
            line_count=1,
            paragraphs=(paragraph,),
        )


def test_run_metadata_requires_safe_id_timezone_and_git_hash() -> None:
    metadata = run_metadata()

    assert metadata.git_sha == "abcdef0"
    for update in (
        {"run_id": "../escape"},
        {"started_at": datetime(2026, 7, 10, 12)},
        {"git_sha": "not-a-sha"},
    ):
        values = metadata.model_dump()
        values.update(update)
        with pytest.raises(ValidationError):
            RunMetadata.model_validate(values)


def test_run_metadata_rejects_finish_before_start() -> None:
    with pytest.raises(ValidationError, match="finished_at cannot precede"):
        RunMetadata(
            run_id="run-1",
            started_at=datetime(2026, 7, 10, 12, tzinfo=UTC),
            finished_at=datetime(2026, 7, 10, 11, tzinfo=UTC),
            tool_version="0.0.0",
        )


def test_run_paths_require_the_canonical_filename() -> None:
    paths = RunPaths(run_dir=Path("runs/run-1"), audit_json=Path("runs/run-1/audit.json"))

    assert paths.audit_json == paths.run_dir / "audit.json"
    with pytest.raises(ValidationError, match="audit_json"):
        RunPaths(run_dir=Path("runs/run-1"), audit_json=Path("runs/other.json"))


@pytest.mark.parametrize(
    "relation",
    [Relation.ENTAILED, Relation.PARTIALLY_ENTAILED, Relation.CONTRADICTED],
)
def test_substantive_verdict_requires_evidence_span(relation: Relation) -> None:
    with pytest.raises(ValidationError, match="requires an evidence span"):
        Verdict(
            claim_id="c_001",
            relation=relation,
            confidence=0.8,
            source_ids=("s_001",),
            reason="A reason.",
            judge_version="judge-v1",
        )


def test_verdict_requires_evidence_source_to_be_declared() -> None:
    with pytest.raises(ValidationError, match="must appear in source_ids"):
        Verdict(
            claim_id="c_001",
            relation=Relation.CONTRADICTED,
            corroboration=CorroborationStatus.NOT_REQUESTED,
            confidence=0.8,
            source_ids=("s_001",),
            evidence_spans=(
                EvidenceSpan(source_id="s_002", text="Different fact", locator="line 2"),
            ),
            reason="The cited value differs.",
            judge_version="judge-v1",
        )


def test_non_substantive_verdict_can_abstain_without_span() -> None:
    verdict = Verdict(
        claim_id="c_001",
        relation=Relation.NOT_IN_SOURCE,
        confidence=0.2,
        reason="No sufficient source text was found.",
        judge_version="judge-v1",
    )

    assert verdict.evidence_spans == ()


def test_policy_config_rejects_overlapping_label_groups() -> None:
    with pytest.raises(ValidationError, match="must be disjoint"):
        PolicyConfig(fail_on=("contradicted",), warn_on=("contradicted",))


def test_effective_config_and_audit_artifact_json_round_trip() -> None:
    audit = AuditArtifact(run=run_metadata(), effective_config=EffectiveConfig())
    payload = audit.model_dump_json()
    decoded = json.loads(payload)

    assert decoded["schema_version"] == 1
    assert decoded["claims"] == []
    assert decoded["verdicts"] == []
    assert AuditArtifact.model_validate_json(payload) == audit
