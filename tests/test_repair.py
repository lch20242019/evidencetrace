from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import pytest

from evidencetrace.agents.challenger import ChallengeAction
from evidencetrace.models import Relation
from evidencetrace.repair import (
    ApplicationStatus,
    BeforeFileWrite,
    ChallengerGate,
    DecisionStatus,
    ExactEvidence,
    FileApplyResult,
    FileDigest,
    RepairCandidate,
    RepairProposal,
    RepairReason,
    ScalarKind,
    apply_repair_plan,
    derive_unique_scalar_pair,
    plan_repairs,
    scalar_outcome_signature,
    sha256_bytes,
)


def _proposal(
    content: bytes,
    *,
    claim: str,
    target_scalar: str,
    evidence: str,
    evidence_scalar: str,
    proposal_id: str = "p1",
    path: str = "doc.md",
    target_sha256: str | None = None,
    references: tuple[FileDigest, ...] = (),
    final_relation: Relation = Relation.CONTRADICTED,
    challenger: ChallengerGate | None = None,
    evidence_exact: bool = True,
) -> RepairProposal:
    text = content.decode()
    claim_offset = text.index(claim)
    line_start = text[:claim_offset].count("\n") + 1
    line_end = line_start + claim.count("\n")
    return RepairProposal(
        proposal_id=proposal_id,
        target=FileDigest(path, target_sha256 or sha256_bytes(content)),
        claim_text=claim,
        line_start=line_start,
        line_end=line_end,
        target_scalar=target_scalar,
        evidence=ExactEvidence(
            source_id="source-1",
            text=evidence,
            scalar=evidence_scalar,
            exact=evidence_exact,
        ),
        final_relation=final_relation,
        challenger=challenger or ChallengerGate(True, ChallengeAction.UPHOLD),
        references=references,
    )


def _candidate(content: bytes, proposal: RepairProposal) -> RepairCandidate:
    plan = plan_repairs({proposal.target.path: content}, (proposal,))
    decision = plan.decisions[0]
    assert decision.status is DecisionStatus.ELIGIBLE, decision
    assert decision.candidate is not None
    return decision.candidate


@pytest.mark.parametrize(
    ("claim", "target", "evidence", "evidence_scalar", "kind", "replacement"),
    [
        (
            "Count is 10 items.",
            "10",
            "Count is 12 items.",
            "12",
            ScalarKind.INTEGER,
            "12",
        ),
        (
            "Latency is 1.50 ms.",
            "1.50",
            "Latency is 2.25 ms.",
            "2.25",
            ScalarKind.DECIMAL,
            "2.25",
        ),
        (
            "Success is 10 %.",
            "10 %",
            "Success is 12.5%.",
            "12.5%",
            ScalarKind.PERCENTAGE,
            "12.5 %",
        ),
        (
            "Released 2024-01-02.",
            "2024-01-02",
            "Released 2025/03/04.",
            "2025/03/04",
            ScalarKind.DATE,
            "2025-03-04",
        ),
        (
            "Released January 02, 2024.",
            "January 02, 2024",
            "Released March 4, 2025.",
            "March 4, 2025",
            ScalarKind.DATE,
            "March 04, 2025",
        ),
        (
            "Released 02 January 2024.",
            "02 January 2024",
            "Released 4 March 2025.",
            "4 March 2025",
            ScalarKind.DATE,
            "04 March 2025",
        ),
        (
            "Requires 1.2.3-alpha.1+build.5.",
            "1.2.3-alpha.1+build.5",
            "Requires 2.0.0-rc.1+build.7.",
            "2.0.0-rc.1+build.7",
            ScalarKind.SEMVER,
            "2.0.0-rc.1+build.7",
        ),
    ],
)
def test_plans_supported_scalar_types(
    claim: str,
    target: str,
    evidence: str,
    evidence_scalar: str,
    kind: ScalarKind,
    replacement: str,
) -> None:
    content = f"{claim}\n".encode()
    candidate = _candidate(
        content,
        _proposal(
            content,
            claim=claim,
            target_scalar=target,
            evidence=evidence,
            evidence_scalar=evidence_scalar,
        ),
    )

    assert candidate.scalar_kind is kind
    assert candidate.replacement_scalar == replacement
    assert f"-{claim}" in candidate.suggested_diff
    assert replacement in candidate.suggested_diff


@pytest.mark.parametrize(
    "claim",
    [
        "Count is 10.",
        "- Count is 10.",
        "| count | 10 |",
        "Use `10` workers.",
    ],
)
def test_body_list_table_and_inline_code_are_allowed(claim: str) -> None:
    content = f"{claim}\n".encode()
    proposal = _proposal(
        content,
        claim=claim,
        target_scalar="10",
        evidence=("Use 12 workers." if "workers" in claim else "The count is 12."),
        evidence_scalar="12",
    )

    assert (
        plan_repairs({"doc.md": content}, (proposal,)).decisions[0].status
        is DecisionStatus.ELIGIBLE
    )


@pytest.mark.parametrize(
    ("document", "claim", "line_start"),
    [
        ("```\nCount is 10.\n```\n", "Count is 10.", 2),
        ("See https://example.test/?count=10 now.\n", "count=10", 1),
        ("[source](ref-10)\n", "[source](ref-10)", 1),
        ("[source]: ref-10\n", "[source]: ref-10", 1),
        ("---\nyear: 2024\n---\nBody.\n", "year: 2024", 2),
        ('<span data-count="10">value</span>\n', 'data-count="10"', 1),
    ],
)
def test_protected_markdown_locations_are_rejected(
    document: str, claim: str, line_start: int
) -> None:
    content = document.encode()
    target_scalar = "2024" if "2024" in claim else "10"
    proposal = RepairProposal(
        proposal_id="protected",
        target=FileDigest("doc.md", sha256_bytes(content)),
        claim_text=claim,
        line_start=line_start,
        line_end=line_start,
        target_scalar=target_scalar,
        evidence=ExactEvidence("source", "The value is 12.", "12"),
        final_relation=Relation.CONTRADICTED,
        challenger=ChallengerGate(True, ChallengeAction.UPHOLD),
    )

    decision = plan_repairs({"doc.md": content}, (proposal,)).decisions[0]

    assert decision.status is DecisionStatus.NEEDS_HUMAN
    assert decision.reason is RepairReason.PROTECTED_LOCATION


def test_units_must_match_and_are_never_converted() -> None:
    content = b"Latency is 100 ms.\n"
    proposal = _proposal(
        content,
        claim="Latency is 100 ms.",
        target_scalar="100",
        evidence="Latency is 1 seconds.",
        evidence_scalar="1",
    )

    decision = plan_repairs({"doc.md": content}, (proposal,)).decisions[0]

    assert decision.status is DecisionStatus.NEEDS_HUMAN
    assert decision.reason is RepairReason.UNIT_MISMATCH


def test_unique_scalar_pair_is_strict_and_evidence_derived() -> None:
    assert derive_unique_scalar_pair("Latency is 100 ms.", "Latency is 120 ms.") == (
        "100",
        "120",
    )
    assert derive_unique_scalar_pair("Use `10` workers.", "Use 12 workers.") == (
        "10",
        "12",
    )
    assert (
        derive_unique_scalar_pair("Values are 100 and 101 ms.", "The value is 120 ms.")
        is None
    )
    assert (
        derive_unique_scalar_pair("Latency is 100 ms.", "Latency is 1 seconds.") is None
    )
    assert (
        derive_unique_scalar_pair("Released 2024-01-02.", "Released March 4, 2025.")
        is None
    )
    assert scalar_outcome_signature("Latency is 100 ms.", "Latency is 100 ms.") == (
        ScalarKind.INTEGER,
        "100",
        "suffix:ms",
    )
    assert (
        scalar_outcome_signature(
            "Latency is 100 ms.",
            "Latency is 100 seconds.",
        )
        is None
    )


@pytest.mark.parametrize(
    ("content", "claim", "target", "evidence", "evidence_scalar", "reason"),
    [
        (
            b"Count is 10; Count is 10.\n",
            "Count is 10",
            "10",
            "Count is 12.",
            "12",
            RepairReason.CLAIM_AMBIGUOUS,
        ),
        (
            b"Values 10 and 10 differ.\n",
            "Values 10 and 10 differ.",
            "10",
            "Values 12 and 13 differ.",
            "12",
            RepairReason.SCALAR_AMBIGUOUS,
        ),
        (
            b"Count is 10.\n",
            "Count is 10.",
            "10",
            "Both 12 and 12 were measured.",
            "12",
            RepairReason.EVIDENCE_SCALAR_AMBIGUOUS,
        ),
        (
            b"Released 07/28/2026.\n",
            "Released 07/28/2026.",
            "07/28/2026",
            "Released 08/01/2026.",
            "08/01/2026",
            RepairReason.UNSUPPORTED_SCALAR,
        ),
    ],
)
def test_ambiguity_and_non_allowlisted_dates_need_human(
    content: bytes,
    claim: str,
    target: str,
    evidence: str,
    evidence_scalar: str,
    reason: RepairReason,
) -> None:
    proposal = _proposal(
        content,
        claim=claim,
        target_scalar=target,
        evidence=evidence,
        evidence_scalar=evidence_scalar,
    )

    decision = plan_repairs({"doc.md": content}, (proposal,)).decisions[0]

    assert decision.status is DecisionStatus.NEEDS_HUMAN
    assert decision.reason is reason


def test_date_format_families_cannot_cross() -> None:
    content = b"Released 2024-01-02.\n"
    proposal = _proposal(
        content,
        claim="Released 2024-01-02.",
        target_scalar="2024-01-02",
        evidence="Released March 4, 2025.",
        evidence_scalar="March 4, 2025",
    )

    decision = plan_repairs({"doc.md": content}, (proposal,)).decisions[0]

    assert decision.reason is RepairReason.DATE_FAMILY_MISMATCH


def test_typed_judge_challenger_and_exact_evidence_gates() -> None:
    content = b"Count is 10.\n"
    base = _proposal(
        content,
        claim="Count is 10.",
        target_scalar="10",
        evidence="Count is 12.",
        evidence_scalar="12",
    )
    safe_revise = replace(
        base,
        proposal_id="safe-revise",
        challenger=ChallengerGate(
            completed=True,
            action=ChallengeAction.REVISE,
            revised_relation=Relation.CONTRADICTED,
            revised_evidence_text=base.evidence.text,
            revised_evidence_scalar=base.evidence.scalar,
        ),
    )
    proposals = (
        replace(base, proposal_id="entailed", final_relation=Relation.ENTAILED),
        replace(
            base,
            proposal_id="abstain",
            challenger=ChallengerGate(True, ChallengeAction.ABSTAIN),
        ),
        replace(
            safe_revise,
            proposal_id="unsafe-revise",
            challenger=replace(
                safe_revise.challenger,
                revised_evidence_scalar="13",
            ),
        ),
        replace(
            base,
            proposal_id="not-exact",
            evidence=replace(base.evidence, exact=False),
        ),
        safe_revise,
    )

    decisions = plan_repairs({"doc.md": content}, proposals).decisions

    assert [item.reason for item in decisions] == [
        RepairReason.FINAL_NOT_CONTRADICTED,
        RepairReason.CHALLENGER_ABSTAINED,
        RepairReason.CHALLENGER_REVISION_UNSAFE,
        RepairReason.EVIDENCE_NOT_EXACT,
        RepairReason.ELIGIBLE,
    ]


def test_all_overlapping_proposals_are_rejected() -> None:
    content = b"Count is 10.\n"
    first = _proposal(
        content,
        proposal_id="first",
        claim="Count is 10.",
        target_scalar="10",
        evidence="Count is 12.",
        evidence_scalar="12",
    )
    second = replace(
        first,
        proposal_id="second",
        evidence=ExactEvidence("source-2", "Count is 13.", "13"),
    )

    plan = plan_repairs({"doc.md": content}, (first, second))

    assert plan.suggested_diff == ""
    assert all(
        decision.status is DecisionStatus.NEEDS_HUMAN
        and decision.reason is RepairReason.OVERLAP
        for decision in plan.decisions
    )


def test_target_and_reference_sha_are_planning_gates() -> None:
    content = b"Count is 10.\n"
    reference = b"Count is 12.\n"
    digest = FileDigest("reference.txt", sha256_bytes(reference))
    stale_target = _proposal(
        content,
        proposal_id="stale-target",
        claim="Count is 10.",
        target_scalar="10",
        evidence="Count is 12.",
        evidence_scalar="12",
        target_sha256="0" * 64,
    )
    stale_reference = _proposal(
        content,
        proposal_id="stale-reference",
        claim="Count is 10.",
        target_scalar="10",
        evidence="Count is 12.",
        evidence_scalar="12",
        references=(digest,),
    )

    decisions = plan_repairs(
        {"doc.md": content},
        (stale_target, stale_reference),
        reference_contents={"reference.txt": b"changed"},
    ).decisions

    assert decisions[0].status is DecisionStatus.STALE_TARGET
    assert decisions[1].status is DecisionStatus.STALE_REFERENCE


def test_apply_generates_three_diffs_and_preserves_unicode_offsets(
    tmp_path: Path,
) -> None:
    content = "中文 Count is 10 items.\n".encode()
    target = tmp_path / "doc.md"
    target.write_bytes(content)
    proposal = _proposal(
        content,
        claim="Count is 10 items.",
        target_scalar="10",
        evidence="Count is 12 items.",
        evidence_scalar="12",
    )
    plan = plan_repairs({"doc.md": content}, (proposal,))

    result = apply_repair_plan(tmp_path, plan, {"p1"})

    assert target.read_bytes() == "中文 Count is 12 items.\n".encode()
    assert result.decisions[0].status is ApplicationStatus.APPLIED
    assert result.suggested_diff == plan.suggested_diff
    assert "-中文 Count is 10 items." in result.applied_diff
    assert "+中文 Count is 12 items." in result.applied_diff
    assert "-中文 Count is 12 items." in result.reverse_diff
    assert "+中文 Count is 10 items." in result.reverse_diff
    assert result.files[0].before_sha256 == sha256_bytes(content)
    assert result.files[0].after_sha256 == sha256_bytes(target.read_bytes())


def test_before_file_write_receives_complete_preview_before_replacement(
    tmp_path: Path,
) -> None:
    content = b"Count is 10.\n"
    expected = b"Count is 12.\n"
    target = tmp_path / "doc.md"
    target.write_bytes(content)
    proposal = _proposal(
        content,
        claim="Count is 10.",
        target_scalar="10",
        evidence="Count is 12.",
        evidence_scalar="12",
    )
    plan = plan_repairs({"doc.md": content}, (proposal,))
    previews: list[FileApplyResult] = []

    def observe(preview: FileApplyResult) -> None:
        assert target.read_bytes() == content
        assert preview.status is ApplicationStatus.APPLIED
        assert preview.before_sha256 == sha256_bytes(content)
        assert preview.after_sha256 == sha256_bytes(expected)
        assert preview.applied_diff
        assert preview.reverse_diff
        previews.append(preview)

    callback: BeforeFileWrite = observe
    result = apply_repair_plan(
        tmp_path,
        plan,
        {"p1"},
        before_file_write=callback,
    )

    assert target.read_bytes() == expected
    assert previews == [result.files[0]]


def test_before_file_write_failure_isolated_to_one_file(tmp_path: Path) -> None:
    first_content = b"First count is 10.\n"
    second_content = b"Second count is 20.\n"
    first = tmp_path / "first.md"
    second = tmp_path / "second.md"
    first.write_bytes(first_content)
    second.write_bytes(second_content)
    first_proposal = _proposal(
        first_content,
        proposal_id="first",
        path="first.md",
        claim="First count is 10.",
        target_scalar="10",
        evidence="First count is 12.",
        evidence_scalar="12",
    )
    second_proposal = _proposal(
        second_content,
        proposal_id="second",
        path="second.md",
        claim="Second count is 20.",
        target_scalar="20",
        evidence="Second count is 22.",
        evidence_scalar="22",
    )
    plan = plan_repairs(
        {
            "first.md": first_content,
            "second.md": second_content,
        },
        (first_proposal, second_proposal),
    )
    seen: list[str] = []

    def fail_first(preview: FileApplyResult) -> None:
        seen.append(preview.target_path)
        assert preview.reverse_diff
        if preview.target_path == "first.md":
            assert first.read_bytes() == first_content
            raise RuntimeError("reject first write")

    result = apply_repair_plan(
        tmp_path,
        plan,
        {"first", "second"},
        before_file_write=fail_first,
    )

    statuses = {decision.proposal_id: decision.status for decision in result.decisions}
    assert statuses == {
        "first": ApplicationStatus.APPLY_ERROR,
        "second": ApplicationStatus.APPLIED,
    }
    assert seen == ["first.md", "second.md"]
    assert first.read_bytes() == first_content
    assert second.read_bytes() == b"Second count is 22.\n"
    assert "first.md" not in result.applied_diff
    assert "--- a/second.md" in result.applied_diff


def test_runtime_paths_apply_external_files_but_diffs_keep_safe_ids(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    target = external / "actual-document.md"
    reference_path = external / "actual-reference.txt"
    content = b"Count is 10.\n"
    reference = b"Count is 12.\n"
    target.write_bytes(content)
    reference_path.write_bytes(reference)
    reference_digest = FileDigest("references/source.txt", sha256_bytes(reference))
    proposal = _proposal(
        content,
        path="targets/document.md",
        claim="Count is 10.",
        target_scalar="10",
        evidence="Count is 12.",
        evidence_scalar="12",
        references=(reference_digest,),
    )
    plan = plan_repairs(
        {"targets/document.md": content},
        (proposal,),
        reference_contents={"references/source.txt": reference},
    )

    result = apply_repair_plan(
        root,
        plan,
        {"p1"},
        runtime_paths={
            "targets/document.md": target,
            "references/source.txt": reference_path,
        },
    )

    assert target.read_bytes() == b"Count is 12.\n"
    assert result.decisions[0].status is ApplicationStatus.APPLIED
    assert "--- a/targets/document.md" in result.applied_diff
    assert str(external) not in result.applied_diff


def test_runtime_paths_reject_symlinks_before_writing(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    actual = tmp_path / "actual.md"
    link = tmp_path / "linked.md"
    content = b"Count is 10.\n"
    actual.write_bytes(content)
    link.symlink_to(actual)
    proposal = _proposal(
        content,
        claim="Count is 10.",
        target_scalar="10",
        evidence="Count is 12.",
        evidence_scalar="12",
    )
    plan = plan_repairs({"doc.md": content}, (proposal,))

    result = apply_repair_plan(root, plan, {"p1"}, runtime_paths={"doc.md": link})

    assert actual.read_bytes() == content
    assert link.is_symlink()
    assert result.decisions[0].status is ApplicationStatus.APPLY_ERROR
    assert result.files[0].status is ApplicationStatus.APPLY_ERROR


def test_parent_swap_after_preview_cannot_redirect_atomic_replace(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    live = tmp_path / "live"
    moved = tmp_path / "moved-live"
    outside = tmp_path / "outside"
    root.mkdir()
    live.mkdir()
    outside.mkdir()
    content = b"Count is 10.\n"
    target = live / "doc.md"
    target.write_bytes(content)
    proposal = _proposal(
        content,
        path="targets/document.md",
        claim="Count is 10.",
        target_scalar="10",
        evidence="Count is 12.",
        evidence_scalar="12",
    )
    plan = plan_repairs({"targets/document.md": content}, (proposal,))

    def swap_parent(preview: FileApplyResult) -> None:
        assert preview.reverse_diff
        live.rename(moved)
        os.link(moved / "doc.md", outside / "doc.md")
        live.symlink_to(outside, target_is_directory=True)

    result = apply_repair_plan(
        root,
        plan,
        {"p1"},
        runtime_paths={"targets/document.md": target},
        before_file_write=swap_parent,
    )

    assert result.decisions[0].status is ApplicationStatus.APPLY_ERROR
    assert result.files[0].status is ApplicationStatus.APPLY_ERROR
    assert result.applied_diff == ""
    assert live.is_symlink()
    assert (moved / "doc.md").read_bytes() == content
    assert (outside / "doc.md").read_bytes() == content
    assert not tuple(moved.glob(".evidencetrace-*.tmp"))
    assert not tuple(outside.glob(".evidencetrace-*.tmp"))


def test_runtime_path_collision_rejects_every_alias_id(
    tmp_path: Path,
) -> None:
    content = b"Count is 10.\n"
    target = tmp_path / "actual.md"
    target.write_bytes(content)
    first = _proposal(
        content,
        proposal_id="first",
        path="targets/first.md",
        claim="Count is 10.",
        target_scalar="10",
        evidence="Count is 12.",
        evidence_scalar="12",
    )
    second = _proposal(
        content,
        proposal_id="second",
        path="targets/second.md",
        claim="Count is 10.",
        target_scalar="10",
        evidence="Count is 13.",
        evidence_scalar="13",
    )
    plan = plan_repairs(
        {
            "targets/first.md": content,
            "targets/second.md": content,
        },
        (first, second),
    )

    result = apply_repair_plan(
        tmp_path,
        plan,
        {"first", "second"},
        runtime_paths={
            "targets/first.md": target,
            "targets/second.md": target,
        },
    )

    assert target.read_bytes() == content
    assert result.applied_diff == ""
    assert all(
        decision.status is ApplicationStatus.APPLY_ERROR
        for decision in result.decisions
    )


def test_missing_reference_isolated_from_unrelated_target(
    tmp_path: Path,
) -> None:
    first_content = b"First count is 10.\n"
    second_content = b"Second count is 20.\n"
    reference_content = b"First count is 12.\n"
    first = tmp_path / "first.md"
    second = tmp_path / "second.md"
    reference = tmp_path / "reference.txt"
    first.write_bytes(first_content)
    second.write_bytes(second_content)
    reference.write_bytes(reference_content)
    reference_digest = FileDigest(
        "references/source.txt",
        sha256_bytes(reference_content),
    )
    first_proposal = _proposal(
        first_content,
        proposal_id="first",
        path="targets/first.md",
        claim="First count is 10.",
        target_scalar="10",
        evidence="First count is 12.",
        evidence_scalar="12",
        references=(reference_digest,),
    )
    second_proposal = _proposal(
        second_content,
        proposal_id="second",
        path="targets/second.md",
        claim="Second count is 20.",
        target_scalar="20",
        evidence="Second count is 22.",
        evidence_scalar="22",
    )
    plan = plan_repairs(
        {
            "targets/first.md": first_content,
            "targets/second.md": second_content,
        },
        (first_proposal, second_proposal),
        reference_contents={"references/source.txt": reference_content},
    )
    reference.unlink()

    result = apply_repair_plan(
        tmp_path,
        plan,
        {"first", "second"},
        runtime_paths={
            "targets/first.md": first,
            "targets/second.md": second,
            "references/source.txt": reference,
        },
    )

    statuses = {decision.proposal_id: decision.status for decision in result.decisions}
    assert statuses == {
        "first": ApplicationStatus.STALE_REFERENCE,
        "second": ApplicationStatus.APPLIED,
    }
    assert first.read_bytes() == first_content
    assert second.read_bytes() == b"Second count is 22.\n"
    assert "targets/first.md" not in result.applied_diff
    assert "--- a/targets/second.md" in result.applied_diff


def test_apply_rechecks_stale_target_and_reference(tmp_path: Path) -> None:
    content = b"Count is 10.\n"
    reference = b"Count is 12.\n"
    (tmp_path / "doc.md").write_bytes(content)
    (tmp_path / "reference.txt").write_bytes(reference)
    reference_digest = FileDigest("reference.txt", sha256_bytes(reference))
    proposal = _proposal(
        content,
        claim="Count is 10.",
        target_scalar="10",
        evidence="Count is 12.",
        evidence_scalar="12",
        references=(reference_digest,),
    )
    plan = plan_repairs(
        {"doc.md": content},
        (proposal,),
        reference_contents={"reference.txt": reference},
    )

    (tmp_path / "reference.txt").write_bytes(b"Count is 13.\n")
    stale_reference = apply_repair_plan(tmp_path, plan, {"p1"})
    assert stale_reference.decisions[0].status is ApplicationStatus.STALE_REFERENCE
    assert (tmp_path / "doc.md").read_bytes() == content

    (tmp_path / "reference.txt").write_bytes(reference)
    (tmp_path / "doc.md").write_bytes(b"Count is 11.\n")
    stale_target = apply_repair_plan(tmp_path, plan, {"p1"})
    assert stale_target.decisions[0].status is ApplicationStatus.STALE_TARGET
    assert (tmp_path / "doc.md").read_bytes() == b"Count is 11.\n"


def test_atomic_replace_failure_keeps_original_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = b"Count is 10.\n"
    target = tmp_path / "doc.md"
    target.write_bytes(content)
    proposal = _proposal(
        content,
        claim="Count is 10.",
        target_scalar="10",
        evidence="Count is 12.",
        evidence_scalar="12",
    )
    plan = plan_repairs({"doc.md": content}, (proposal,))

    def fail_replace(
        source: str,
        destination: str,
        *,
        src_dir_fd: int,
        dst_dir_fd: int,
    ) -> None:
        raise OSError(f"blocked {source} -> {destination}")

    monkeypatch.setattr("evidencetrace.repair.os.replace", fail_replace)
    result = apply_repair_plan(tmp_path, plan, {"p1"})

    assert target.read_bytes() == content
    assert result.decisions[0].status is ApplicationStatus.APPLY_ERROR
    assert str(tmp_path) not in result.decisions[0].detail
    assert str(tmp_path) not in result.files[0].detail
    assert result.applied_diff == ""
    assert not tuple(tmp_path.glob(".evidencetrace-*.tmp"))
