from __future__ import annotations

import json
import os
import stat
from collections.abc import Collection, Mapping
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

import evidencetrace.self_use as self_use_module
from evidencetrace.agents.scout import SearchResponse, SearchResult
from evidencetrace.product import (
    EvidenceOrigin,
    EvidenceResolutionStatus,
    ProductAuditPipeline,
    ProductRunResult,
)
from evidencetrace.repair import (
    ApplyResult,
    BeforeFileWrite,
    DecisionStatus,
    RepairPlan,
    apply_repair_plan,
)
from evidencetrace.retrieval.fetch import SafeFetcher
from evidencetrace.self_use import (
    InteractiveSession,
    RepairApplyStatus,
    RepairReviewStatus,
    SelfUseAuditArtifact,
    SelfUseError,
    SelfUseRunner,
    atomic_write_text,
    prepare_self_use_run,
)


def _fixture_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    first = tmp_path / "docs" / "nimbus.md"
    second = tmp_path / "docs" / "cirrus.txt"
    reference = tmp_path / "facts.md"
    first.parent.mkdir()
    first.write_text("Nimbus shipped version 1.2.3.\n", encoding="utf-8")
    second.write_text("Cirrus shipped version 2.0.0.\n", encoding="utf-8")
    reference.write_text(
        "Nimbus shipped version 1.2.4.\n\n"
        "Cirrus shipped version 2.0.1.\n",
        encoding="utf-8",
    )
    return first, second, reference


def test_check_batch_is_read_only_and_writes_canonical_per_target_artifacts(
    tmp_path: Path,
) -> None:
    first, second, reference = _fixture_files(tmp_path)
    before = (first.read_bytes(), second.read_bytes(), reference.read_bytes())
    run_root = tmp_path / "artifacts"

    result = SelfUseRunner(project_root=tmp_path).run(
        (first, second),
        references=(reference,),
        mode="check",
        output_dir=run_root,
        run_id="offline-check",
        started_at=datetime(2035, 1, 2, tzinfo=UTC),
    )

    assert result.exit_code == 0
    assert result.manifest.status == "complete"
    assert (first.read_bytes(), second.read_bytes(), reference.read_bytes()) == before
    assert (run_root / "batch-manifest.json").is_file()
    assert len(result.targets) == 2
    for target in result.targets:
        artifact = SelfUseAuditArtifact.model_validate_json(
            target.paths.audit_json.read_text(encoding="utf-8")
        )
        assert artifact.target_id == target.target_id
        assert artifact.repair_decisions
        assert all(
            decision.eligibility_status is DecisionStatus.ELIGIBLE
            for decision in artifact.repair_decisions
        )
        assert target.paths.suggested_diff.read_text(encoding="utf-8")
        assert target.paths.applied_diff.read_bytes() == b""
        assert target.paths.reverse_diff.read_bytes() == b""
        assert target.paths.sarif.read_bytes()
        sarif_run = json.loads(
            target.paths.sarif.read_text(encoding="utf-8")
        )["runs"][0]
        assert sarif_run["properties"]["mode"] == artifact.mode
        assert sarif_run["properties"]["status"] == artifact.status
        assert (
            sarif_run["properties"]["fileState"]
            == artifact.file_state.status
        )
    persisted = "\n".join(
        path.read_text(encoding="utf-8")
        for path in run_root.rglob("*")
        if path.is_file()
    )
    assert str(tmp_path.resolve()) not in persisted


def test_txt_target_with_spaces_uses_safe_source_identity(tmp_path: Path) -> None:
    target = tmp_path / "my notes.txt"
    reference = tmp_path / "facts.txt"
    target.write_text("Nimbus shipped version 1.2.3.\n", encoding="utf-8")
    reference.write_text("Nimbus shipped version 1.2.4.\n", encoding="utf-8")

    result = SelfUseRunner(project_root=tmp_path).run(
        (target,),
        references=(reference,),
        mode="check",
        output_dir=tmp_path / "space-artifacts",
        run_id="space-check",
    )

    assert result.manifest.status == "complete"
    assert result.targets[0].artifact.documents[0].path.startswith("target-")
    assert result.manifest.targets[0].display_path == "my notes.txt"


def test_single_legacy_check_can_use_markdown_reference(tmp_path: Path) -> None:
    target = tmp_path / "README"
    reference = tmp_path / "facts.md"
    target.write_text("Nimbus shipped version 1.2.3.\n", encoding="utf-8")
    reference.write_text("Nimbus shipped version 1.2.4.\n", encoding="utf-8")

    prepared = prepare_self_use_run(
        (target,),
        references=(reference,),
        mode="check",
        project_root=tmp_path,
        output_dir=tmp_path / "legacy-artifacts",
        run_id="legacy-reference-check",
    )
    result = SelfUseRunner(project_root=tmp_path).run_prepared(prepared)

    assert result.manifest.status == "complete"
    assert result.targets[0].artifact.repair_decisions


def test_canonical_artifact_rejects_status_or_challenger_chain_tampering(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.md"
    reference = tmp_path / "facts.md"
    target.write_text("Nimbus shipped version 1.2.3.\n", encoding="utf-8")
    reference.write_text("Nimbus shipped version 1.2.4.\n", encoding="utf-8")
    artifact = SelfUseRunner(project_root=tmp_path).run(
        (target,),
        references=(reference,),
        mode="check",
        output_dir=tmp_path / "canonical-artifacts",
        run_id="canonical-check",
    ).targets[0].artifact
    payload = artifact.model_dump(mode="python")

    wrong_status = deepcopy(payload)
    wrong_status["status"] = "partial"
    with pytest.raises(ValidationError):
        SelfUseAuditArtifact.model_validate(wrong_status)

    missing_challenger = deepcopy(payload)
    missing_challenger["execution"]["claim_runs"][0]["challenger_action"] = None
    with pytest.raises(ValidationError):
        SelfUseAuditArtifact.model_validate(missing_challenger)

    applied_check = deepcopy(payload)
    applied_check["repair_decisions"][0]["review_status"] = "approved"
    applied_check["repair_decisions"][0]["apply_status"] = "applied"
    applied_check["file_state"]["after_sha256"] = "0" * 64
    applied_check["file_state"]["status"] = "applied"
    with pytest.raises(ValidationError):
        SelfUseAuditArtifact.model_validate(applied_check)

    incomplete_execution = deepcopy(payload)
    incomplete_execution["repair_decisions"] = []
    incomplete_execution["execution"]["claim_runs"][0]["status"] = "agent_error"
    incomplete_execution["execution"]["claim_runs"][0][
        "challenger_action"
    ] = None
    with pytest.raises(ValidationError):
        SelfUseAuditArtifact.model_validate(incomplete_execution)

    forged_span = deepcopy(payload)
    replacement = forged_span["repair_decisions"][0]["replacement"]
    forged_span["verdicts"][0]["evidence_spans"][0]["text"] = (
        f"forged evidence {replacement}"
    )
    forged_span["verdicts"][0]["evidence_spans"][0]["locator"] = "forged"
    with pytest.raises(ValidationError):
        SelfUseAuditArtifact.model_validate(forged_span)


def test_target_audit_failure_is_isolated_and_mixed_batch_exits_two(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, second, reference = _fixture_files(tmp_path)
    original = ProductAuditPipeline.run

    def flaky_run(
        pipeline: ProductAuditPipeline,
        markdown_path: Path,
        **kwargs: Any,
    ) -> ProductRunResult:
        if markdown_path == second:
            raise RuntimeError("isolated probe")
        return original(pipeline, markdown_path, **kwargs)

    monkeypatch.setattr(ProductAuditPipeline, "run", flaky_run)
    result = SelfUseRunner(project_root=tmp_path).run(
        (first, second),
        references=(reference,),
        mode="check",
        output_dir=tmp_path / "isolated-artifacts",
        run_id="isolated-check",
    )

    assert result.exit_code == 2
    assert result.manifest.status == "partial"
    assert [item.status for item in result.targets] == ["complete", "fatal"]
    assert all(item.paths.audit_json.is_file() for item in result.targets)
    assert result.targets[1].artifact.execution.document_status.value == "failed"


def test_all_target_audit_failures_are_fatal_exit_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "target.md"
    target.write_text("Nimbus shipped version 1.2.3.\n", encoding="utf-8")

    def failed_run(
        _pipeline: ProductAuditPipeline,
        _markdown_path: Path,
        **_kwargs: Any,
    ) -> ProductRunResult:
        raise RuntimeError("isolated probe")

    monkeypatch.setattr(ProductAuditPipeline, "run", failed_run)
    result = SelfUseRunner(project_root=tmp_path).run(
        (target,),
        mode="check",
        output_dir=tmp_path / "fatal-artifacts",
        run_id="fatal-check",
    )

    assert result.exit_code == 1
    assert result.manifest.status == "fatal"
    assert result.targets[0].paths.audit_json.is_file()


def test_fix_applies_only_yes_after_second_confirmation_and_reverse_restores(
    tmp_path: Path,
) -> None:
    first, second, reference = _fixture_files(tmp_path)
    first_before = first.read_bytes()
    responses = iter(("yes", "no", "yes"))
    session = InteractiveSession(read=lambda _prompt: next(responses))

    result = SelfUseRunner(
        project_root=tmp_path,
        interactive=session,
    ).run(
        (first, second),
        references=(reference,),
        mode="fix",
        output_dir=tmp_path / "fix-artifacts",
        run_id="offline-fix",
        started_at=datetime(2035, 1, 2, tzinfo=UTC),
    )

    assert result.exit_code == 2
    assert first.read_text(encoding="utf-8") == (
        "Nimbus shipped version 1.2.4.\n"
    )
    assert second.read_text(encoding="utf-8") == (
        "Cirrus shipped version 2.0.0.\n"
    )
    first_decision = result.targets[0].artifact.repair_decisions[0]
    second_decision = result.targets[1].artifact.repair_decisions[0]
    assert first_decision.review_status is RepairReviewStatus.APPROVED
    assert first_decision.apply_status is RepairApplyStatus.APPLIED
    assert second_decision.review_status is RepairReviewStatus.REJECTED
    assert second_decision.apply_status is RepairApplyStatus.NOT_APPLIED
    reverse = result.targets[0].paths.reverse_diff.read_text(encoding="utf-8")
    assert reverse
    restored = first.read_text(encoding="utf-8").replace("1.2.4", "1.2.3")
    assert restored.encode("utf-8") == first_before
    manifest = json.loads(
        (result.run_root / "batch-manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "partial"


def test_post_apply_sha_mismatch_never_records_a_false_applied_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "target.md"
    reference = tmp_path / "facts.md"
    target.write_text("Nimbus shipped version 1.2.3.\n", encoding="utf-8")
    reference.write_text("Nimbus shipped version 1.2.4.\n", encoding="utf-8")

    def raced_apply(
        root: Path,
        plan: RepairPlan,
        approved_proposal_ids: Collection[str],
        *,
        runtime_paths: Mapping[str, Path] | None = None,
        before_file_write: BeforeFileWrite | None = None,
    ) -> ApplyResult:
        result = apply_repair_plan(
            root,
            plan,
            approved_proposal_ids,
            runtime_paths=runtime_paths,
            before_file_write=before_file_write,
        )
        target.write_text("Nimbus raced to version 9.9.9.\n", encoding="utf-8")
        return result

    monkeypatch.setattr("evidencetrace.self_use.apply_repair_plan", raced_apply)
    responses = iter(("yes", "yes"))
    result = SelfUseRunner(
        project_root=tmp_path,
        interactive=InteractiveSession(read=lambda _prompt: next(responses)),
    ).run(
        (target,),
        references=(reference,),
        mode="fix",
        output_dir=tmp_path / "raced-artifacts",
        run_id="raced-fix",
    )

    decision = result.targets[0].artifact.repair_decisions[0]
    assert result.exit_code == 2
    assert decision.apply_status is RepairApplyStatus.APPLY_ERROR
    assert result.targets[0].artifact.file_state.status == "apply_error"
    assert result.targets[0].paths.applied_diff.read_bytes() == b""
    assert result.targets[0].paths.reverse_diff.read_bytes() == b""


def test_unplanned_target_change_without_repairs_is_canonical_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "target.md"
    target.write_text("A short note without a scalar claim.\n", encoding="utf-8")

    def raced_empty_apply(
        root: Path,
        plan: RepairPlan,
        approved_proposal_ids: Collection[str],
        *,
        runtime_paths: Mapping[str, Path] | None = None,
        before_file_write: BeforeFileWrite | None = None,
    ) -> ApplyResult:
        result = apply_repair_plan(
            root,
            plan,
            approved_proposal_ids,
            runtime_paths=runtime_paths,
            before_file_write=before_file_write,
        )
        target.write_text("Changed concurrently without a repair.\n", encoding="utf-8")
        return result

    monkeypatch.setattr(
        "evidencetrace.self_use.apply_repair_plan",
        raced_empty_apply,
    )
    result = SelfUseRunner(
        project_root=tmp_path,
        interactive=InteractiveSession(read=lambda _prompt: "no"),
    ).run(
        (target,),
        mode="fix",
        output_dir=tmp_path / "empty-race-artifacts",
        run_id="empty-race-fix",
    )

    artifact = result.targets[0].artifact
    assert result.exit_code == 2
    assert artifact.status == "partial"
    assert artifact.file_state.status == "partial"
    assert not artifact.repair_decisions


def test_atomic_artifact_write_rejects_symlinked_parent_without_escape(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(outside, target_is_directory=True)

    with pytest.raises(SelfUseError, match="artifact_path_changed"):
        atomic_write_text(
            linked / "nested" / "audit.json",
            '{"safe":true}\n',
        )

    assert not (outside / "nested").exists()


def test_atomic_artifact_write_rejects_ancestor_rebind_to_same_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ancestor = tmp_path / "canonical"
    parent = ancestor / "run"
    parent.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    moved = outside / "moved"
    original = self_use_module._pin_or_create_artifact_directory

    def rebind(path: Path) -> self_use_module._PinnedArtifactDirectory:
        pinned = original(path)
        ancestor.rename(moved)
        ancestor.symlink_to(moved, target_is_directory=True)
        return pinned

    monkeypatch.setattr(
        self_use_module,
        "_pin_or_create_artifact_directory",
        rebind,
    )

    with pytest.raises(SelfUseError, match="artifact_path_changed"):
        atomic_write_text(parent / "audit.json", '{"safe":true}\n')

    assert not (moved / "run" / "audit.json").exists()


def test_atomic_artifact_parent_move_during_temp_open_leaves_no_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ancestor = tmp_path / "canonical"
    parent = ancestor / "run"
    parent.mkdir(parents=True)
    moved = tmp_path / "moved"
    original_fsync = os.fsync
    raced = False

    def racing_fsync(descriptor: int) -> None:
        nonlocal raced
        if not raced and stat.S_ISREG(os.fstat(descriptor).st_mode):
            raced = True
            ancestor.rename(moved)
            parent.mkdir(parents=True)
        original_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", racing_fsync)

    with pytest.raises(SelfUseError, match="artifact_path_changed"):
        atomic_write_text(parent / "audit.json", '{"safe":true}\n')

    assert raced
    assert not (parent / "audit.json").exists()
    assert not tuple((moved / "run").iterdir())


def test_atomic_artifact_capability_failure_precedes_directory_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "not-created" / "audit.json"
    monkeypatch.setattr(os, "O_NOFOLLOW", 0)

    with pytest.raises(SelfUseError, match="artifact_atomic_unsupported"):
        atomic_write_text(destination, '{"safe":true}\n')

    assert not destination.parent.exists()


def test_atomic_artifact_post_replace_failure_is_commit_uncertain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "audit.json"
    destination.write_text('{"old":true}\n', encoding="utf-8")
    original_fsync = os.fsync

    def fail_directory_fsync(descriptor: int) -> None:
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError("simulated directory fsync failure")
        original_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", fail_directory_fsync)

    with pytest.raises(SelfUseError, match="artifact_commit_uncertain"):
        atomic_write_text(destination, '{"new":true}\n')

    assert destination.read_text(encoding="utf-8") == '{"new":true}\n'


def test_write_ahead_canonical_and_reverse_survive_final_artifact_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "target.md"
    reference = tmp_path / "facts.md"
    run_root = tmp_path / "recovery-artifacts"
    target.write_text("Nimbus shipped version 1.2.3.\n", encoding="utf-8")
    reference.write_text("Nimbus shipped version 1.2.4.\n", encoding="utf-8")
    audit_writes = 0

    def fail_second_audit_write(path: Path, content: str) -> None:
        nonlocal audit_writes
        if path.name == "audit.json":
            audit_writes += 1
            if audit_writes == 2:
                raise OSError("simulated final artifact failure")
        atomic_write_text(path, content)

    monkeypatch.setattr(
        "evidencetrace.self_use.atomic_write_text",
        fail_second_audit_write,
    )
    responses = iter(("yes", "yes"))

    with pytest.raises(OSError, match="simulated final artifact failure"):
        SelfUseRunner(
            project_root=tmp_path,
            interactive=InteractiveSession(read=lambda _prompt: next(responses)),
        ).run(
            (target,),
            references=(reference,),
            mode="fix",
            output_dir=run_root,
            run_id="recovery-fix",
        )

    assert target.read_text(encoding="utf-8") == (
        "Nimbus shipped version 1.2.4.\n"
    )
    audit_path = next((run_root / "targets").glob("*/audit.json"))
    preliminary = SelfUseAuditArtifact.model_validate_json(
        audit_path.read_text(encoding="utf-8")
    )
    assert preliminary.file_state.before_sha256 == preliminary.file_state.after_sha256
    reverse_path = audit_path.with_name("reverse.diff")
    assert "1.2.3" in reverse_path.read_text(encoding="utf-8")


def test_citation_only_check_produces_a_scalar_candidate_from_fetched_html(
    tmp_path: Path,
) -> None:
    target = tmp_path / "citation.md"
    target.write_text(
        "Nimbus shipped version 1.2.3 "
        "[release](https://evidence.example/release).\n",
        encoding="utf-8",
    )

    def fetched(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            request=request,
            headers={"content-type": "text/html; charset=utf-8"},
            text="<main><p>Nimbus shipped version 1.2.4.</p></main>",
        )

    result = SelfUseRunner(
        project_root=tmp_path,
        fetcher=SafeFetcher(
            transport=httpx.MockTransport(fetched),
            resolve_dns=False,
            retries=0,
        ),
    ).run(
        (target,),
        mode="check",
        output_dir=tmp_path / "citation-artifacts",
        run_id="citation-check",
    )

    assert result.exit_code == 0
    decision = result.targets[0].artifact.repair_decisions[0]
    assert decision.eligibility_status is DecisionStatus.ELIGIBLE
    assert decision.original == "1.2.3"
    assert decision.replacement == "1.2.4"
    assert target.read_text(encoding="utf-8").startswith(
        "Nimbus shipped version 1.2.3"
    )


def test_tavily_only_requires_trust_then_apply_and_never_uses_snippet(
    tmp_path: Path,
) -> None:
    target = tmp_path / "discovery.md"
    target.write_text("Nimbus shipped version 1.2.3.\n", encoding="utf-8")
    url = "https://evidence.example/release"

    class Search:
        def search(self, _query: str, *, max_results: int) -> SearchResponse:
            assert max_results == 5
            return SearchResponse(
                status="ok",
                results=(
                    SearchResult(
                        url=url,
                        snippet="Untrusted snippet claims version 9.9.9.",
                    ),
                ),
            )

    def fetched(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            request=request,
            headers={"content-type": "text/html; charset=utf-8"},
            text="<main><p>Nimbus shipped version 1.2.4.</p></main>",
        )

    responses = iter(("yes", "yes", "yes"))
    result = SelfUseRunner(
        project_root=tmp_path,
        search_client=Search(),
        fetcher=SafeFetcher(
            transport=httpx.MockTransport(fetched),
            resolve_dns=False,
            retries=0,
        ),
        interactive=InteractiveSession(read=lambda _prompt: next(responses)),
    ).run(
        (target,),
        mode="fix",
        discover=True,
        output_dir=tmp_path / "discovery-artifacts",
        run_id="tavily-fix",
    )

    assert target.read_text(encoding="utf-8") == (
        "Nimbus shipped version 1.2.4.\n"
    )
    resolution = result.targets[0].artifact.evidence_resolutions[0]
    assert resolution.status is EvidenceResolutionStatus.HUMAN_CORROBORATED
    assert resolution.selected_origin is EvidenceOrigin.TAVILY
    assert all(
        "9.9.9" not in evidence.text
        for option in resolution.options
        for evidence in option.evidence
    )


def test_conflicting_citation_sources_remain_needs_human_in_check(
    tmp_path: Path,
) -> None:
    target = tmp_path / "citations.md"
    target.write_text(
        "Nimbus shipped version 1.2.3 "
        "[first](https://evidence.example/first) "
        "[second](https://evidence.example/second).\n",
        encoding="utf-8",
    )

    def fetched(request: httpx.Request) -> httpx.Response:
        version = "1.2.4" if request.url.path.endswith("first") else "1.2.5"
        return httpx.Response(
            200,
            request=request,
            headers={"content-type": "text/html; charset=utf-8"},
            text=f"<main><p>Nimbus shipped version {version}.</p></main>",
        )

    result = SelfUseRunner(
        project_root=tmp_path,
        fetcher=SafeFetcher(
            transport=httpx.MockTransport(fetched),
            resolve_dns=False,
            retries=0,
        ),
    ).run(
        (target,),
        mode="check",
        output_dir=tmp_path / "citation-conflict-artifacts",
        run_id="citation-conflict-check",
    )

    assert result.exit_code == 2
    assert target.read_text(encoding="utf-8").startswith(
        "Nimbus shipped version 1.2.3"
    )
    resolution = result.targets[0].artifact.evidence_resolutions[0]
    assert resolution.status is EvidenceResolutionStatus.NEEDS_HUMAN
    assert len(resolution.options) == 2
    assert result.targets[0].paths.suggested_diff.read_bytes() == b""


def test_conflict_selection_is_bounded_to_existing_reference_option(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.md"
    first = tmp_path / "first.md"
    second = tmp_path / "second.md"
    target.write_text("Nimbus shipped version 1.2.3.\n", encoding="utf-8")
    first.write_text("Nimbus shipped version 1.2.4.\n", encoding="utf-8")
    second.write_text("Nimbus shipped version 1.2.5.\n", encoding="utf-8")
    responses = iter(("99", "2", "yes", "yes"))

    result = SelfUseRunner(
        project_root=tmp_path,
        interactive=InteractiveSession(read=lambda _prompt: next(responses)),
    ).run(
        (target,),
        references=(first, second),
        mode="fix",
        output_dir=tmp_path / "conflict-artifacts",
        run_id="conflict-fix",
    )

    assert target.read_text(encoding="utf-8") == (
        "Nimbus shipped version 1.2.5.\n"
    )
    resolution = result.targets[0].artifact.evidence_resolutions[0]
    assert resolution.status is EvidenceResolutionStatus.HUMAN_RESOLVED_CONFLICT
    assert resolution.selected_option_id == resolution.options[1].option_id


def test_stale_target_isolated_from_other_approved_file(
    tmp_path: Path,
) -> None:
    first, second, reference = _fixture_files(tmp_path)
    prompts = 0

    def answer(_prompt: str) -> str:
        nonlocal prompts
        prompts += 1
        if prompts == 3:
            first.write_text(
                "Nimbus changed concurrently to version 1.2.3.\n",
                encoding="utf-8",
            )
        return "yes"

    result = SelfUseRunner(
        project_root=tmp_path,
        interactive=InteractiveSession(read=answer),
    ).run(
        (first, second),
        references=(reference,),
        mode="fix",
        output_dir=tmp_path / "stale-artifacts",
        run_id="stale-fix",
    )

    first_decision = result.targets[0].artifact.repair_decisions[0]
    second_decision = result.targets[1].artifact.repair_decisions[0]
    assert first_decision.apply_status is RepairApplyStatus.STALE_TARGET
    assert second_decision.apply_status is RepairApplyStatus.APPLIED
    assert "changed concurrently" in first.read_text(encoding="utf-8")
    assert "2.0.1" in second.read_text(encoding="utf-8")


def test_stale_reference_invalidates_only_its_dependent_candidate(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.md"
    reference = tmp_path / "facts.md"
    target.write_text("Nimbus shipped version 1.2.3.\n", encoding="utf-8")
    reference.write_text("Nimbus shipped version 1.2.4.\n", encoding="utf-8")
    prompts = 0

    def answer(_prompt: str) -> str:
        nonlocal prompts
        prompts += 1
        if prompts == 2:
            reference.write_text(
                "Nimbus changed concurrently to version 1.2.5.\n",
                encoding="utf-8",
            )
        return "yes"

    result = SelfUseRunner(
        project_root=tmp_path,
        interactive=InteractiveSession(read=answer),
    ).run(
        (target,),
        references=(reference,),
        mode="fix",
        output_dir=tmp_path / "stale-reference-artifacts",
        run_id="stale-reference-fix",
    )

    decision = result.targets[0].artifact.repair_decisions[0]
    assert decision.review_status is RepairReviewStatus.APPROVED
    assert decision.apply_status is RepairApplyStatus.STALE_REFERENCE
    assert target.read_text(encoding="utf-8") == (
        "Nimbus shipped version 1.2.3.\n"
    )


def test_second_confirmation_no_keeps_approved_candidate_unapplied(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.md"
    reference = tmp_path / "facts.md"
    target.write_text("Nimbus shipped version 1.2.3.\n", encoding="utf-8")
    reference.write_text("Nimbus shipped version 1.2.4.\n", encoding="utf-8")
    responses = iter(("yes", "no"))

    result = SelfUseRunner(
        project_root=tmp_path,
        interactive=InteractiveSession(read=lambda _prompt: next(responses)),
    ).run(
        (target,),
        references=(reference,),
        mode="fix",
        output_dir=tmp_path / "confirm-no-artifacts",
        run_id="confirm-no-fix",
    )

    decision = result.targets[0].artifact.repair_decisions[0]
    assert decision.review_status is RepairReviewStatus.APPROVED
    assert decision.apply_status is RepairApplyStatus.NOT_APPLIED
    assert target.read_text(encoding="utf-8") == (
        "Nimbus shipped version 1.2.3.\n"
    )
    assert result.targets[0].paths.suggested_diff.read_bytes()
    assert result.targets[0].paths.applied_diff.read_bytes() == b""
    assert result.targets[0].paths.reverse_diff.read_bytes() == b""


def test_quit_stops_later_reviews_but_can_apply_prior_approval(
    tmp_path: Path,
) -> None:
    first, second, reference = _fixture_files(tmp_path)
    responses = iter(("yes", "quit", "yes"))

    result = SelfUseRunner(
        project_root=tmp_path,
        interactive=InteractiveSession(read=lambda _prompt: next(responses)),
    ).run(
        (first, second),
        references=(reference,),
        mode="fix",
        output_dir=tmp_path / "quit-artifacts",
        run_id="quit-fix",
    )

    first_decision = result.targets[0].artifact.repair_decisions[0]
    second_decision = result.targets[1].artifact.repair_decisions[0]
    assert first_decision.apply_status is RepairApplyStatus.APPLIED
    assert second_decision.review_status is RepairReviewStatus.UNREVIEWED
    assert second_decision.apply_status is RepairApplyStatus.NOT_APPLIED
    assert "1.2.4" in first.read_text(encoding="utf-8")
    assert "2.0.0" in second.read_text(encoding="utf-8")


def test_external_input_artifacts_and_terminal_never_persist_private_paths(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    private = tmp_path / "private"
    root.mkdir()
    private.mkdir()
    target = private / "target.md"
    reference = private / "facts.md"
    target.write_text("Nimbus shipped version 1.2.3.\n", encoding="utf-8")
    reference.write_text("Nimbus shipped version 1.2.4.\n", encoding="utf-8")

    result = SelfUseRunner(project_root=root).run(
        (target,),
        references=(reference,),
        mode="check",
        output_dir=root / "run",
        run_id="external-check",
    )

    persisted = result.terminal + "\n" + "\n".join(
        path.read_text(encoding="utf-8")
        for path in result.run_root.rglob("*")
        if path.is_file()
    )
    assert str(private.resolve()) not in persisted
    assert str(target.resolve()) not in persisted
    assert str(reference.resolve()) not in persisted
    assert "external/target-" in persisted
