from __future__ import annotations

import hashlib
from pathlib import Path

import httpx

from evidencetrace.agents.judge import ClaimJudgeAgent
from evidencetrace.audit_models import (
    EVIDENCE_CONTEXT_MAX_BYTES,
    EVIDENCE_CONTEXT_MAX_CHARS,
    EVIDENCE_EXACT_MAX_BYTES,
    EVIDENCE_EXACT_MAX_CHARS,
    EVIDENCE_RESOLUTION_CONTEXT_MAX_BYTES,
    EVIDENCE_RESOLUTION_CONTEXT_MAX_CHARS,
    EVIDENCE_RESOLUTION_EXACT_MAX_BYTES,
    EVIDENCE_RESOLUTION_EXACT_MAX_CHARS,
    EVIDENCE_RESOLUTION_MAX_OPTIONS,
    JudgeInput,
    JudgeOutput,
)
from evidencetrace.local_evidence import (
    LocalReferenceIndex,
    LocalReferenceInput,
    parse_plain_text_with_source_map,
)
from evidencetrace.models import Relation
from evidencetrace.product import (
    AgentName,
    ClaimRunStatus,
    EvidenceOrigin,
    EvidenceResolutionReason,
    EvidenceResolutionStatus,
    ProductAuditPipeline,
    TraceState,
)
from evidencetrace.retrieval.fetch import SafeFetcher


def _reference(path: Path, safe_id: str) -> LocalReferenceInput:
    return LocalReferenceInput(
        path=path,
        reference_id=safe_id,
        reference_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


class _RecordingJudge:
    def __init__(self) -> None:
        self.inputs: list[JudgeInput] = []
        self.delegate = ClaimJudgeAgent()

    def judge(self, input_data: JudgeInput) -> JudgeOutput:
        self.inputs.append(input_data)
        return self.delegate.judge(input_data)


def test_local_reference_only_reaches_judge_and_challenger(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.md"
    target.write_text("Nimbus shipped version 1.2.3.\n", encoding="utf-8")
    reference = tmp_path / "reference.md"
    reference.write_text("Nimbus shipped version 1.2.4.\n", encoding="utf-8")
    index = LocalReferenceIndex((_reference(reference, "reference.md"),))

    result = ProductAuditPipeline(
        project_root=tmp_path,
        reference_index=index,
        enforce_human_evidence_gates=True,
    ).run(target, persist=False)

    assert result.product.claim_runs[0].status is ClaimRunStatus.COMPLETED
    assert result.audit.verdicts[0].relation is Relation.CONTRADICTED
    assert (
        result.evidence_resolutions[0].selected_origin
        is EvidenceOrigin.REFERENCE
    )
    dispatched = {
        event.agent
        for event in result.product.trace
        if event.state is TraceState.DISPATCHED
    }
    assert AgentName.SCOUT not in dispatched
    assert AgentName.JUDGE in dispatched
    assert AgentName.CHALLENGER in dispatched


def test_canonical_selected_option_exactly_matches_full_bounded_judge_payload(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.md"
    target.write_text("Nimbus shipped version 1.2.3.\n", encoding="utf-8")
    reference = tmp_path / "reference.md"
    reference.write_text(
        "Nimbus shipped version 1.2.4.\n\n"
        "The Nimbus release notes confirm shipped version 1.2.4.\n\n"
        "Nimbus version 1.2.4 is the shipped release.\n",
        encoding="utf-8",
    )
    pipeline = ProductAuditPipeline(
        project_root=tmp_path,
        reference_index=LocalReferenceIndex(
            (_reference(reference, "reference.md"),)
        ),
        enforce_human_evidence_gates=True,
    )
    recorder = _RecordingJudge()
    pipeline.judge = recorder  # type: ignore[assignment]

    result = pipeline.run(target, persist=False)

    resolution = result.evidence_resolutions[0]
    selected = next(
        option
        for option in resolution.options
        if option.option_id == resolution.selected_option_id
    )
    assert recorder.inputs
    assert len(recorder.inputs[-1].evidence) >= 2
    assert selected.evidence == recorder.inputs[-1].evidence
    assert all(
        len(item.text) <= EVIDENCE_EXACT_MAX_CHARS
        and len(item.text.encode("utf-8")) <= EVIDENCE_EXACT_MAX_BYTES
        and len(item.chunk.text) <= EVIDENCE_CONTEXT_MAX_CHARS
        and len(item.chunk.text.encode("utf-8")) <= EVIDENCE_CONTEXT_MAX_BYTES
        for item in selected.evidence
    )


def test_same_source_scalar_conflicts_require_selection_of_one_exact_group(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.md"
    target.write_text("Nimbus shipped version 1.2.3.\n", encoding="utf-8")
    reference = tmp_path / "reference.md"
    reference.write_text(
        "Nimbus shipped version 1.2.4.\n\n"
        "Nimbus shipped version 1.2.5.\n",
        encoding="utf-8",
    )

    class PickNewer:
        def choose_evidence(self, _claim, reason, options):  # type: ignore[no-untyped-def]
            assert reason is EvidenceResolutionReason.CONFLICT
            return next(
                option.option_id
                for option in options
                if any("1.2.5" in item.text for item in option.evidence)
            )

    pipeline = ProductAuditPipeline(
        project_root=tmp_path,
        reference_index=LocalReferenceIndex(
            (_reference(reference, "reference.md"),)
        ),
        evidence_resolver=PickNewer(),
        enforce_human_evidence_gates=True,
    )
    recorder = _RecordingJudge()
    pipeline.judge = recorder  # type: ignore[assignment]

    result = pipeline.run(target, persist=False)

    resolution = result.evidence_resolutions[0]
    selected = next(
        option
        for option in resolution.options
        if option.option_id == resolution.selected_option_id
    )
    assert resolution.status is EvidenceResolutionStatus.HUMAN_RESOLVED_CONFLICT
    assert len(resolution.options) == 2
    assert all("1.2.5" in item.text for item in selected.evidence)
    assert all("1.2.4" not in item.text for item in selected.evidence)
    assert selected.evidence == recorder.inputs[-1].evidence


def test_resolution_bounds_all_persisted_reference_options(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.md"
    target.write_text("Nimbus shipped version 1.2.3.\n", encoding="utf-8")
    references = []
    for index in range(EVIDENCE_RESOLUTION_MAX_OPTIONS + 3):
        reference = tmp_path / f"reference-{index}.md"
        reference.write_text(
            "Nimbus shipped version 1.2.4.\n",
            encoding="utf-8",
        )
        references.append(_reference(reference, f"reference-{index}.md"))

    result = ProductAuditPipeline(
        project_root=tmp_path,
        reference_index=LocalReferenceIndex(tuple(references)),
        enforce_human_evidence_gates=True,
    ).run(target, persist=False)

    resolution = result.evidence_resolutions[0]
    evidence = tuple(
        item for option in resolution.options for item in option.evidence
    )
    assert len(resolution.options) <= EVIDENCE_RESOLUTION_MAX_OPTIONS
    assert sum(len(item.text) for item in evidence) <= (
        EVIDENCE_RESOLUTION_EXACT_MAX_CHARS
    )
    assert sum(len(item.text.encode("utf-8")) for item in evidence) <= (
        EVIDENCE_RESOLUTION_EXACT_MAX_BYTES
    )
    assert sum(len(item.chunk.text) for item in evidence) <= (
        EVIDENCE_RESOLUTION_CONTEXT_MAX_CHARS
    )
    assert sum(len(item.chunk.text.encode("utf-8")) for item in evidence) <= (
        EVIDENCE_RESOLUTION_CONTEXT_MAX_BYTES
    )


def test_long_plain_webpage_persists_only_bounded_exact_evidence(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.md"
    url = "https://evidence.example/release"
    target.write_text(
        f"Nimbus shipped version 1.2.3 [release]({url}).\n",
        encoding="utf-8",
    )
    relevant = "Nimbus shipped version 1.2.4."
    private_tail = "FULL-WEBPAGE-TAIL-MUST-NOT-PERSIST"
    page = (
        ("prefix-padding " * 2_000)
        + relevant
        + (" suffix-padding " * 2_000)
        + private_tail
    )

    def fetched(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            request=request,
            headers={"content-type": "text/plain; charset=utf-8"},
            text=page,
        )

    pipeline = ProductAuditPipeline(
        project_root=tmp_path,
        fetcher=SafeFetcher(
            transport=httpx.MockTransport(fetched),
            resolve_dns=False,
            retries=0,
        ),
        enforce_human_evidence_gates=True,
    )
    recorder = _RecordingJudge()
    pipeline.judge = recorder  # type: ignore[assignment]

    result = pipeline.run(target, persist=False)

    resolution = result.evidence_resolutions[0]
    selected = next(
        option
        for option in resolution.options
        if option.option_id == resolution.selected_option_id
    )
    assert selected.source.content_hash == hashlib.sha256(page.encode()).hexdigest()
    assert selected.evidence == recorder.inputs[-1].evidence
    assert relevant in selected.evidence[0].text
    assert private_tail not in resolution.model_dump_json()
    assert all(
        item.chunk.source_id == selected.source.source_id
        and item.chunk.url == selected.source.url
        and item.text in item.chunk.text
        and len(item.text) <= EVIDENCE_EXACT_MAX_CHARS
        and len(item.text.encode("utf-8")) <= EVIDENCE_EXACT_MAX_BYTES
        and len(item.chunk.text) <= EVIDENCE_CONTEXT_MAX_CHARS
        and len(item.chunk.text.encode("utf-8")) <= EVIDENCE_CONTEXT_MAX_BYTES
        for item in selected.evidence
    )


def test_conflicting_local_references_need_bounded_human_selection(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.md"
    target.write_text("Nimbus shipped version 1.2.3.\n", encoding="utf-8")
    first = tmp_path / "first.md"
    second = tmp_path / "second.txt"
    first.write_text("Nimbus shipped version 1.2.4.\n", encoding="utf-8")
    second.write_text("Nimbus shipped version 1.2.5.\n", encoding="utf-8")
    index = LocalReferenceIndex(
        (
            _reference(first, "first.md"),
            _reference(second, "second.txt"),
        )
    )

    unresolved = ProductAuditPipeline(
        project_root=tmp_path,
        reference_index=index,
        enforce_human_evidence_gates=True,
    ).run(target, persist=False)

    assert unresolved.product.claim_runs[0].status is ClaimRunStatus.NEEDS_HUMAN
    assert unresolved.audit.verdicts == ()
    assert (
        unresolved.evidence_resolutions[0].reason
        is EvidenceResolutionReason.CONFLICT
    )
    assert (
        unresolved.evidence_resolutions[0].status
        is EvidenceResolutionStatus.NEEDS_HUMAN
    )

    class PickSecond:
        def choose_evidence(self, _claim, reason, options):  # type: ignore[no-untyped-def]
            assert reason is EvidenceResolutionReason.CONFLICT
            return next(
                option.option_id
                for option in options
                if option.reference_id == "second.txt"
            )

    selected = ProductAuditPipeline(
        project_root=tmp_path,
        reference_index=index,
        evidence_resolver=PickSecond(),
        enforce_human_evidence_gates=True,
    ).run(target, persist=False)

    assert selected.audit.verdicts[0].relation is Relation.CONTRADICTED
    resolution = selected.evidence_resolutions[0]
    assert resolution.status is EvidenceResolutionStatus.HUMAN_RESOLVED_CONFLICT
    assert resolution.reference_id == "second.txt"


def test_plain_text_target_uses_safe_display_path_and_url_protection(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.txt"
    text = (
        "# This remains plain text\n"
        "Nimbus shipped version 1.2.3 at "
        "https://example.test/releases/1.2.3.\n"
    )
    target.write_text(text, encoding="utf-8")
    parsed = parse_plain_text_with_source_map(text, "target-safe.txt")

    result = ProductAuditPipeline(project_root=tmp_path).run(
        target,
        persist=False,
        parsed_document=parsed,
    )

    assert result.audit.documents[0].path == "target-safe.txt"
    assert result.audit.documents[0].paragraphs[0].raw_text == text.rstrip("\n")
    assert parsed.protected_spans
    protected = parsed.protected_spans[0]
    assert text[protected.char_start : protected.char_end].startswith("https://")


def test_reference_grouping_treats_unit_mismatch_as_conflict(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.md"
    first = tmp_path / "first.md"
    second = tmp_path / "second.md"
    target.write_text("Nimbus latency is 100 ms.\n", encoding="utf-8")
    first.write_text("Nimbus latency is 120 ms.\n", encoding="utf-8")
    second.write_text("Nimbus latency is 120 seconds.\n", encoding="utf-8")
    index = LocalReferenceIndex(
        (
            _reference(first, "first.md"),
            _reference(second, "second.md"),
        )
    )

    result = ProductAuditPipeline(
        project_root=tmp_path,
        reference_index=index,
        enforce_human_evidence_gates=True,
    ).run(target, persist=False)

    assert result.product.claim_runs[0].status is ClaimRunStatus.NEEDS_HUMAN
    assert (
        result.evidence_resolutions[0].reason
        is EvidenceResolutionReason.CONFLICT
    )
