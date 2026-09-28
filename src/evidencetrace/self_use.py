"""Deterministic multi-target orchestration and interactive repair artifacts."""

from __future__ import annotations

import os
import re
import stat
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from evidencetrace import __version__
from evidencetrace.agents.challenger import (
    CHALLENGER_CONTRACT_VERSION,
    ChallengeAction,
)
from evidencetrace.agents.scout import SearchClient
from evidencetrace.local_evidence import (
    LocalReferenceIndex,
    LocalReferenceInput,
    parse_plain_text_with_source_map,
)
from evidencetrace.markdown import parse_markdown_with_source_map
from evidencetrace.model_client import ModelClient
from evidencetrace.models import (
    AtomicClaim,
    AuditArtifact,
    EffectiveConfig,
    ParsedDocumentWithSourceMap,
    Relation,
    RunMetadata,
)
from evidencetrace.product import (
    AgentName,
    ClaimRunStatus,
    DocumentRunStatus,
    EvidenceOption,
    EvidenceOrigin,
    EvidenceResolution,
    EvidenceResolutionReason,
    EvidenceResolutionStatus,
    HumanEvidenceResolver,
    ProductAuditPipeline,
    ProductRunArtifact,
    ProductRunResult,
    TraceEvent,
    TraceState,
)
from evidencetrace.render import render_audit_markdown, render_terminal
from evidencetrace.repair import (
    ApplicationStatus,
    ApplyResult,
    ChallengerGate,
    DecisionStatus,
    ExactEvidence,
    FileApplyResult,
    FileDigest,
    RepairPlan,
    RepairProposal,
    RepairReason,
    apply_repair_plan,
    derive_unique_scalar_pair,
    plan_repairs,
    sha256_bytes,
)
from evidencetrace.retrieval.fetch import SafeFetcher
from evidencetrace.sarif import sarif_bytes
from evidencetrace.self_use_inputs import (
    InputFileRecord,
    OutputPathRequest,
    SelfUseInputContract,
    preflight_self_use_inputs,
    validate_output_paths,
)

_TARGET_ID_RE = re.compile(r"^target-[a-z0-9-]+-[0-9a-f]{64}$")
_REFERENCE_ID_RE = re.compile(r"^reference-[a-z0-9-]+-[0-9a-f]{64}$")
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class SelfUseError(RuntimeError):
    """A bounded self-use failure safe to expose without private path content."""

    def __init__(self, code: str) -> None:
        super().__init__(f"self-use operation failed ({code})")
        self.code = code


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _safe_target_id(value: str) -> str:
    if not _TARGET_ID_RE.fullmatch(value):
        raise ValueError("target ID must be a preflight-generated safe ID")
    return value


def _safe_reference_id(value: str) -> str:
    if not _REFERENCE_ID_RE.fullmatch(value):
        raise ValueError("reference ID must be a preflight-generated safe ID")
    return value


def _safe_relative_display(value: str) -> str:
    path = Path(value)
    if (
        not value
        or "\\" in value
        or path.is_absolute()
        or ".." in path.parts
    ):
        raise ValueError("display path must be privacy-safe and relative")
    return value


class RepairReviewStatus(StrEnum):
    APPROVED = "approved"
    REJECTED = "rejected"
    UNREVIEWED = "unreviewed"


class RepairApplyStatus(StrEnum):
    NOT_APPLIED = "not_applied"
    APPLIED = "applied"
    STALE_TARGET = "stale_target"
    STALE_REFERENCE = "stale_reference"
    APPLY_ERROR = "apply_error"


class RepairDecisionRecord(_Model):
    proposal_id: str
    claim_id: str
    line_start: int = Field(ge=1)
    original: str | None = None
    replacement: str | None = None
    evidence_source_id: str | None = None
    eligibility_status: DecisionStatus
    eligibility_reason: RepairReason
    review_status: RepairReviewStatus
    apply_status: RepairApplyStatus = RepairApplyStatus.NOT_APPLIED

    @field_validator("proposal_id")
    @classmethod
    def validate_proposal_id(cls, value: str) -> str:
        target_id, separator, claim_id = value.partition(":")
        if (
            not separator
            or not claim_id
            or not _TARGET_ID_RE.fullmatch(target_id)
        ):
            raise ValueError("proposal ID must bind a safe target and claim")
        return value

    @model_validator(mode="after")
    def validate_decision_state(self) -> RepairDecisionRecord:
        eligible = self.eligibility_status is DecisionStatus.ELIGIBLE
        scalar_fields = (
            self.original,
            self.replacement,
            self.evidence_source_id,
        )
        if eligible != all(value is not None for value in scalar_fields):
            raise ValueError("eligible repairs require a complete scalar decision")
        if eligible != (self.eligibility_reason is RepairReason.ELIGIBLE):
            raise ValueError("repair eligibility status and reason disagree")
        if (
            self.apply_status is not RepairApplyStatus.NOT_APPLIED
            and (
                self.review_status is not RepairReviewStatus.APPROVED
                or not eligible
            )
        ):
            raise ValueError("only approved repairs can be applied")
        return self


class TargetFileState(_Model):
    before_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    after_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: Literal["unchanged", "applied", "partial", "apply_error"]

    @model_validator(mode="after")
    def status_matches_hashes(self) -> TargetFileState:
        changed = self.before_sha256 != self.after_sha256
        if self.status == "unchanged" and changed:
            raise ValueError("unchanged file state cannot change SHA-256")
        if self.status == "applied" and not changed:
            raise ValueError("applied file state must change SHA-256")
        return self


class SelfUseAuditArtifact(AuditArtifact):
    """Top-level canonical audit extension for one self-use target."""

    self_use_schema_version: Literal[1] = 1
    target_id: str
    mode: Literal["check", "fix"]
    status: Literal["complete", "partial", "fatal"]
    execution: ProductRunArtifact
    evidence_resolutions: tuple[EvidenceResolution, ...] = ()
    repair_decisions: tuple[RepairDecisionRecord, ...] = ()
    file_state: TargetFileState

    @field_validator("target_id")
    @classmethod
    def validate_target_id(cls, value: str) -> str:
        return _safe_target_id(value)

    @model_validator(mode="after")
    def validate_self_use_references(self) -> SelfUseAuditArtifact:
        claims = {claim.claim_id: claim for claim in self.claims}
        claim_ids = set(claims)
        verdicts = {verdict.claim_id: verdict for verdict in self.verdicts}
        claim_runs = {
            claim_run.claim_id: claim_run
            for claim_run in self.execution.claim_runs
        }
        if len(claims) != len(self.claims):
            raise ValueError("canonical claim IDs must be unique")
        if len(verdicts) != len(self.verdicts):
            raise ValueError("canonical verdict claim IDs must be unique")
        if len(claim_runs) != len(self.execution.claim_runs):
            raise ValueError("execution claim runs must be unique")
        source_ids = {source.source_id for source in self.sources}
        if len(source_ids) != len(self.sources):
            raise ValueError("canonical source IDs must be unique")
        if set(claim_runs) != claim_ids:
            raise ValueError("execution claim runs must match canonical claims")
        if (
            self.execution.document_status is DocumentRunStatus.COMPLETE
            and any(
                claim_run.status is not ClaimRunStatus.COMPLETED
                for claim_run in self.execution.claim_runs
            )
        ):
            raise ValueError(
                "complete execution requires every claim run to complete"
            )
        if any(
            item.claim_id not in claim_ids for item in self.evidence_resolutions
        ):
            raise ValueError("evidence resolution references an unknown claim")
        if any(
            item.claim_id not in claim_ids for item in self.repair_decisions
        ):
            raise ValueError("repair decision references an unknown claim")
        if any(
            option.source.source_id not in source_ids
            for resolution in self.evidence_resolutions
            for option in resolution.options
        ):
            raise ValueError("evidence option references an unknown source")
        selected_sources = {
            item.claim_id: item.selected_source_id
            for item in self.evidence_resolutions
        }
        resolutions = {
            item.claim_id: item for item in self.evidence_resolutions
        }
        if len(resolutions) != len(self.evidence_resolutions):
            raise ValueError("evidence resolutions must be unique per claim")
        if any(
            item.eligibility_status is DecisionStatus.ELIGIBLE
            and selected_sources.get(item.claim_id)
            != item.evidence_source_id
            for item in self.repair_decisions
        ):
            raise ValueError("repair evidence must match selected evidence")
        proposal_ids = [item.proposal_id for item in self.repair_decisions]
        if len(set(proposal_ids)) != len(proposal_ids):
            raise ValueError("repair proposal IDs must be unique")
        decision_claim_ids = [
            item.claim_id for item in self.repair_decisions
        ]
        if len(set(decision_claim_ids)) != len(decision_claim_ids):
            raise ValueError("repair decisions must be unique per claim")
        for decision in self.repair_decisions:
            if decision.eligibility_status is not DecisionStatus.ELIGIBLE:
                continue
            if (
                decision.original is None
                or decision.replacement is None
                or decision.evidence_source_id is None
            ):
                raise ValueError("eligible repair scalar state is incomplete")
            original = decision.original
            replacement = decision.replacement
            evidence_source_id = decision.evidence_source_id
            claim = claims[decision.claim_id]
            verdict = verdicts.get(decision.claim_id)
            claim_run = claim_runs[decision.claim_id]
            resolution = resolutions.get(decision.claim_id)
            selected_option = (
                next(
                    (
                        option
                        for option in resolution.options
                        if option.option_id == resolution.selected_option_id
                    ),
                    None,
                )
                if resolution is not None
                else None
            )
            if (
                original not in claim.text
                or verdict is None
                or verdict.relation is not Relation.CONTRADICTED
                or claim_run.status is not ClaimRunStatus.COMPLETED
                or claim_run.challenger_action
                not in {ChallengeAction.UPHOLD, ChallengeAction.REVISE}
                or resolution is None
                or resolution.status is EvidenceResolutionStatus.NEEDS_HUMAN
                or selected_option is None
                or not any(
                    replacement in evidence.text
                    for evidence in selected_option.evidence
                )
                or not any(
                    span.source_id == evidence_source_id
                    and span.source_id == evidence.chunk.source_id
                    and span.locator == evidence.chunk.locator
                    and span.text in evidence.text
                    and replacement in span.text
                    for span in verdict.evidence_spans
                    for evidence in selected_option.evidence
                )
            ):
                raise ValueError(
                    "eligible repair lacks its canonical evidence/verdict/"
                    "Challenger chain"
                )
            if (
                claim_run.challenger_action is ChallengeAction.REVISE
                and verdict.judge_version != CHALLENGER_CONTRACT_VERSION
            ):
                raise ValueError(
                    "revised repair must carry the Challenger contract verdict"
                )
        applied = any(
            item.apply_status is RepairApplyStatus.APPLIED
            for item in self.repair_decisions
        )
        changed = (
            self.file_state.before_sha256 != self.file_state.after_sha256
        )
        if applied and not changed:
            raise ValueError("an applied repair must change the target SHA-256")
        if self.file_state.status == "applied" and not applied:
            raise ValueError("applied file state requires an applied repair")
        if self.mode == "check" and (
            applied
            or any(
                item.review_status is not RepairReviewStatus.UNREVIEWED
                or item.apply_status is not RepairApplyStatus.NOT_APPLIED
                for item in self.repair_decisions
            )
        ):
            raise ValueError("check mode cannot review, apply, or change a target")
        unresolved = (
            self.execution.document_status is DocumentRunStatus.PARTIAL
            or any(
                item.status is EvidenceResolutionStatus.NEEDS_HUMAN
                for item in self.evidence_resolutions
            )
            or any(
                item.eligibility_status is not DecisionStatus.ELIGIBLE
                for item in self.repair_decisions
            )
            or self.file_state.status in {"partial", "apply_error"}
        )
        if self.mode == "fix":
            unresolved = unresolved or any(
                item.review_status is not RepairReviewStatus.APPROVED
                or item.apply_status is not RepairApplyStatus.APPLIED
                for item in self.repair_decisions
            )
        expected_status: Literal["complete", "partial", "fatal"]
        if self.execution.document_status is DocumentRunStatus.FAILED:
            expected_status = "fatal"
        elif unresolved:
            expected_status = "partial"
        else:
            expected_status = "complete"
        if self.status != expected_status:
            raise ValueError("target status must be derived from canonical outcome")
        return self


class BatchTargetRecord(_Model):
    target_id: str
    display_path: str
    status: Literal["complete", "partial", "fatal"]
    audit_path: str
    before_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    after_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("target_id")
    @classmethod
    def validate_target_id(cls, value: str) -> str:
        return _safe_target_id(value)

    @field_validator("display_path", "audit_path")
    @classmethod
    def validate_relative_paths(cls, value: str) -> str:
        return _safe_relative_display(value)

    @model_validator(mode="after")
    def audit_path_matches_target(self) -> BatchTargetRecord:
        if self.audit_path != f"targets/{self.target_id}/audit.json":
            raise ValueError("manifest audit path must match its target ID")
        return self


class BatchReferenceRecord(_Model):
    reference_id: str
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("reference_id")
    @classmethod
    def validate_reference_id(cls, value: str) -> str:
        return _safe_reference_id(value)


class BatchManifest(_Model):
    schema_version: Literal[1] = 1
    run_id: str
    mode: Literal["check", "fix"]
    status: Literal["complete", "partial", "fatal"]
    started_at: datetime
    finished_at: datetime
    tool_version: str = __version__
    targets: tuple[BatchTargetRecord, ...] = Field(
        min_length=1,
        max_length=20,
    )
    references: tuple[BatchReferenceRecord, ...] = Field(
        default=(),
        max_length=50,
    )

    @field_validator("run_id")
    @classmethod
    def validate_run_id(cls, value: str) -> str:
        if not _RUN_ID_RE.fullmatch(value):
            raise ValueError("run ID is not safe")
        return value

    @model_validator(mode="after")
    def validate_batch_state(self) -> BatchManifest:
        if self.finished_at < self.started_at:
            raise ValueError("batch finish cannot precede its start")
        target_ids = [item.target_id for item in self.targets]
        reference_ids = [item.reference_id for item in self.references]
        if len(set(target_ids)) != len(target_ids):
            raise ValueError("batch target IDs must be unique")
        if len(set(reference_ids)) != len(reference_ids):
            raise ValueError("batch reference IDs must be unique")
        if self.targets and self.status != manifest_status(
            tuple(item.status for item in self.targets)
        ):
            raise ValueError("batch status must aggregate target statuses")
        return self

@dataclass(frozen=True, slots=True)
class TargetArtifactPaths:
    """Runtime-only canonical destinations for one target."""

    target_id: str
    directory: Path
    audit_json: Path
    audit_markdown: Path
    sarif: Path
    suggested_diff: Path
    applied_diff: Path
    reverse_diff: Path


@dataclass(frozen=True, slots=True)
class PreparedSelfUseRun:
    """Fully validated, read-only run contract created before providers."""

    inputs: SelfUseInputContract
    mode: Literal["check", "fix"]
    discover: bool
    changed_from: str | None
    run_id: str
    started_at: datetime
    run_root: Path
    manifest_path: Path
    targets: tuple[TargetArtifactPaths, ...]
    explicit_sarif: Path | None = None
    explicit_suggested_diff: Path | None = None


@dataclass(frozen=True, slots=True)
class SelfUseTargetResult:
    target_id: str
    status: Literal["complete", "partial", "fatal"]
    artifact: SelfUseAuditArtifact
    paths: TargetArtifactPaths
    terminal: str


@dataclass(frozen=True, slots=True)
class SelfUseRunResult:
    manifest: BatchManifest
    run_root: Path
    targets: tuple[SelfUseTargetResult, ...]
    terminal: str

    @property
    def exit_code(self) -> Literal[0, 1, 2]:
        if self.manifest.status == "complete":
            return 0
        if self.manifest.status == "fatal":
            return 1
        return 2


@dataclass(frozen=True, slots=True)
class _ProposalContext:
    claim_id: str
    proposal: RepairProposal


@dataclass(frozen=True, slots=True)
class _SkippedRepair:
    proposal_id: str
    claim_id: str
    line_start: int
    reason: RepairReason
    evidence_source_id: str | None = None


@dataclass(frozen=True, slots=True)
class _AuditedTarget:
    record: InputFileRecord
    result: ProductRunResult
    plan: RepairPlan
    proposals: tuple[_ProposalContext, ...]
    skipped: tuple[_SkippedRepair, ...]
    fatal: bool = False


class InteractiveSession(HumanEvidenceResolver):
    """Bounded TTY decisions; it never accepts free-form replacement values."""

    def __init__(
        self,
        *,
        read: Callable[[str], str] = input,
        write: Callable[[str], None] = print,
        max_invalid_attempts: int = 3,
    ) -> None:
        if max_invalid_attempts < 1:
            raise ValueError("max_invalid_attempts must be positive")
        self._read = read
        self._write = write
        self.max_invalid_attempts = max_invalid_attempts
        self.quit_requested = False

    def _bounded_read(
        self,
        prompt: str,
        *,
        allowed: frozenset[str],
    ) -> str | None:
        for _ in range(self.max_invalid_attempts):
            try:
                value = self._read(prompt).strip().casefold()
            except (EOFError, KeyboardInterrupt):
                self.quit_requested = True
                return None
            if value in allowed:
                return value
            self._write("Invalid choice; no free-form value was accepted.")
        return None

    def _show_option(self, index: int, option: EvidenceOption) -> None:
        evidence = option.evidence[0] if option.evidence else None
        self._write(
            f"[{index}] origin={option.origin.value} "
            f"source={option.source.source_id}"
        )
        if evidence is not None:
            self._write(f"    locator={evidence.chunk.locator}")
            self._write(f"    evidence={evidence.text}")

    def choose_evidence(
        self,
        claim: AtomicClaim,
        reason: EvidenceResolutionReason,
        options: tuple[EvidenceOption, ...],
    ) -> str | None:
        if self.quit_requested or not options:
            return None
        self._write(f"Evidence resolution for {claim.file}:{claim.line_start}")
        self._write(f"Claim: {claim.text}")
        for index, option in enumerate(options, start=1):
            self._show_option(index, option)
        if reason is EvidenceResolutionReason.TAVILY_ONLY:
            tavily_options = tuple(
                (index, option)
                for index, option in enumerate(options, start=1)
                if option.origin is EvidenceOrigin.TAVILY
            )
            if not tavily_options:
                return None
            if len(tavily_options) > 1:
                allowed = "/".join(str(index) for index, _ in tavily_options)
                value = self._bounded_read(
                    f"Select fetched Tavily evidence [{allowed}], skip, or quit: ",
                    allowed=frozenset(
                        {
                            *(str(index) for index, _ in tavily_options),
                            "skip",
                            "s",
                            "no",
                            "n",
                            "quit",
                            "q",
                        }
                    ),
                )
                if value in {"quit", "q"}:
                    self.quit_requested = True
                    return None
                if value in {"skip", "s", "no", "n"}:
                    return None
                try:
                    selected = int(value or "")
                except ValueError:
                    return None
                return next(
                    (
                        option.option_id
                        for index, option in tavily_options
                        if index == selected
                    ),
                    None,
                )
            value = self._bounded_read(
                "Trust this fetched Tavily evidence? [yes/no/quit] ",
                allowed=frozenset({"yes", "y", "no", "n", "quit", "q"}),
            )
            if value in {"quit", "q"}:
                self.quit_requested = True
                return None
            if value not in {"yes", "y"}:
                return None
            return tavily_options[0][1].option_id
        value = self._bounded_read(
            f"Select evidence [1-{len(options)}], skip, or quit: ",
            allowed=frozenset(
                {
                    *(str(index) for index in range(1, len(options) + 1)),
                    "skip",
                    "s",
                    "no",
                    "n",
                    "quit",
                    "q",
                }
            ),
        )
        if value in {"quit", "q"}:
            self.quit_requested = True
            return None
        if value in {"skip", "s", "no", "n"}:
            return None
        try:
            selected = int(value or "")
        except ValueError:
            return None
        if not 1 <= selected <= len(options):
            return None
        return options[selected - 1].option_id

    def review_repair(
        self,
        *,
        file: str,
        line: int,
        claim: str,
        original: str,
        replacement: str,
        evidence: str,
        source: str,
        final_outcome: str = "contradicted",
    ) -> RepairReviewStatus:
        if self.quit_requested:
            return RepairReviewStatus.UNREVIEWED
        self._write(f"Repair candidate {file}:{line}")
        self._write(f"Claim: {claim}")
        self._write(f"Scalar: {original} -> {replacement}")
        self._write(f"Evidence ({source}): {evidence}")
        self._write(f"Final outcome: {final_outcome}")
        value = self._bounded_read(
            "Apply this repair? [yes/no/quit] ",
            allowed=frozenset({"yes", "y", "no", "n", "quit", "q"}),
        )
        if value in {"quit", "q"}:
            self.quit_requested = True
            return RepairReviewStatus.UNREVIEWED
        if value in {"yes", "y"}:
            return RepairReviewStatus.APPROVED
        if value is None:
            return RepairReviewStatus.UNREVIEWED
        return RepairReviewStatus.REJECTED

    def confirm_apply(self, approved_count: int) -> bool:
        if approved_count <= 0:
            return False
        value = self._bounded_read(
            f"Write {approved_count} approved repair(s)? [yes/no] ",
            allowed=frozenset({"yes", "y", "no", "n"}),
        )
        return value in {"yes", "y"}


def _artifact_directory_flags() -> int:
    directory = getattr(os, "O_DIRECTORY", None)
    nofollow = getattr(os, "O_NOFOLLOW", None)
    required_dir_fd = {os.open, os.mkdir, os.stat, os.unlink}
    if (
        not isinstance(directory, int)
        or directory == 0
        or not isinstance(nofollow, int)
        or nofollow == 0
        or not required_dir_fd <= os.supports_dir_fd
        or os.stat not in os.supports_follow_symlinks
        or os.rename not in os.supports_dir_fd
    ):
        raise SelfUseError("artifact_atomic_unsupported")
    return (
        os.O_RDONLY
        | directory
        | nofollow
        | getattr(os, "O_CLOEXEC", 0)
    )


@dataclass(frozen=True, slots=True)
class _PinnedArtifactDirectory:
    """A root-to-parent descriptor chain that rejects namespace rebinding."""

    path: Path
    descriptors: tuple[int, ...]
    components: tuple[str, ...]

    @property
    def parent_fd(self) -> int:
        return self.descriptors[-1]

    def validate(self) -> None:
        if len(self.descriptors) != len(self.components) + 1:
            raise SelfUseError("artifact_path_changed")
        for parent_fd, component, child_fd in zip(
            self.descriptors[:-1],
            self.components,
            self.descriptors[1:],
            strict=True,
        ):
            try:
                entry = os.stat(
                    component,
                    dir_fd=parent_fd,
                    follow_symlinks=False,
                )
                opened = os.fstat(child_fd)
            except OSError:
                raise SelfUseError("artifact_path_changed") from None
            if (
                not stat.S_ISDIR(entry.st_mode)
                or not stat.S_ISDIR(opened.st_mode)
                or (entry.st_dev, entry.st_ino)
                != (opened.st_dev, opened.st_ino)
            ):
                raise SelfUseError("artifact_path_changed")

    def close(self) -> None:
        for descriptor in reversed(self.descriptors):
            with suppress(OSError):
                os.close(descriptor)


def _pin_or_create_artifact_directory(
    path: Path,
) -> _PinnedArtifactDirectory:
    """Create and pin every absolute path component without following links."""

    if not path.is_absolute() or not path.anchor:
        raise SelfUseError("artifact_path_changed")
    flags = _artifact_directory_flags()
    descriptors: list[int] = []
    components: list[str] = []
    try:
        descriptors.append(os.open(path.anchor, flags))
        for part in path.parts[1:]:
            if part in {"", ".", ".."}:
                raise SelfUseError("artifact_path_changed")
            pinned = _PinnedArtifactDirectory(
                path=path,
                descriptors=tuple(descriptors),
                components=tuple(components),
            )
            pinned.validate()
            try:
                next_fd = os.open(part, flags, dir_fd=descriptors[-1])
            except FileNotFoundError:
                pinned.validate()
                with suppress(FileExistsError):
                    os.mkdir(part, mode=0o700, dir_fd=descriptors[-1])
                next_fd = os.open(part, flags, dir_fd=descriptors[-1])
            descriptors.append(next_fd)
            components.append(part)
            _PinnedArtifactDirectory(
                path=path,
                descriptors=tuple(descriptors),
                components=tuple(components),
            ).validate()
        result = _PinnedArtifactDirectory(
            path=path,
            descriptors=tuple(descriptors),
            components=tuple(components),
        )
        result.validate()
        return result
    except SelfUseError:
        for descriptor in reversed(descriptors):
            with suppress(OSError):
                os.close(descriptor)
        raise
    except (NotImplementedError, OSError, TypeError):
        for descriptor in reversed(descriptors):
            with suppress(OSError):
                os.close(descriptor)
        raise SelfUseError("artifact_path_changed") from None


def atomic_write_bytes(path: Path, content: bytes) -> None:
    """Atomically write through a validated root-to-parent descriptor chain."""

    if not path.is_absolute() or path.name in {"", ".", ".."}:
        raise SelfUseError("artifact_path_changed")
    _artifact_directory_flags()
    try:
        temporary_name = f".et-{os.urandom(16).hex()}.tmp"
    except OSError:
        raise SelfUseError("artifact_entropy_failed") from None
    pinned = _pin_or_create_artifact_directory(path.parent)
    parent_fd = pinned.parent_fd
    temporary_exists = False
    descriptor: int | None = None
    committed = False
    try:
        pinned.validate()
        try:
            existing = os.stat(
                path.name,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            existing = None
        if existing is not None and not stat.S_ISREG(existing.st_mode):
            raise SelfUseError("artifact_path_changed")
        pinned.validate()
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0)
        )
        descriptor = os.open(
            temporary_name,
            flags,
            0o600,
            dir_fd=parent_fd,
        )
        temporary_exists = True
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        pinned.validate()
        os.replace(
            temporary_name,
            path.name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        temporary_exists = False
        committed = True
        try:
            os.fsync(parent_fd)
            pinned.validate()
        except (NotImplementedError, OSError, SelfUseError, TypeError):
            raise SelfUseError("artifact_commit_uncertain") from None
    except SelfUseError:
        raise
    except (NotImplementedError, OSError, TypeError):
        code = "artifact_commit_uncertain" if committed else "artifact_write_failed"
        raise SelfUseError(code) from None
    finally:
        if descriptor is not None:
            with suppress(OSError):
                os.close(descriptor)
        if temporary_exists:
            with suppress(OSError):
                os.unlink(temporary_name, dir_fd=parent_fd)
        pinned.close()


def atomic_write_text(path: Path, content: str) -> None:
    atomic_write_bytes(path, content.encode("utf-8"))


def write_json_artifact(path: Path, value: BaseModel) -> None:
    """Write only a validated typed model as a canonical JSON artifact."""

    atomic_write_text(path, value.model_dump_json(indent=2) + "\n")


def read_self_use_audit(path: Path) -> SelfUseAuditArtifact:
    """Read and schema-validate one canonical self-use audit."""

    try:
        content = path.read_text(encoding="utf-8")
        return SelfUseAuditArtifact.model_validate_json(content)
    except Exception:
        raise SelfUseError("canonical_audit_read_failed") from None


def new_run_id(now: datetime | None = None) -> str:
    instant = now or datetime.now(UTC)
    entropy = os.urandom(4).hex()
    return f"{instant.strftime('%Y%m%dT%H%M%SZ')}-{entropy}"


def manifest_status(statuses: Sequence[str]) -> Literal["complete", "partial", "fatal"]:
    if statuses and all(status == "fatal" for status in statuses):
        return "fatal"
    if any(status != "complete" for status in statuses):
        return "partial"
    return "complete"


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _validate_output_parent(path: Path) -> None:
    current = path.parent
    while not current.exists():
        if current == current.parent:
            raise SelfUseError("output_parent_invalid")
        current = current.parent
    if not current.is_dir():
        raise SelfUseError("output_parent_invalid")


def prepare_self_use_run(
    targets: Sequence[Path | str],
    *,
    references: Sequence[Path | str] = (),
    mode: Literal["check", "fix"] = "check",
    project_root: Path | str,
    discover: bool = False,
    changed_from: str | None = None,
    output_dir: Path | None = None,
    sarif: Path | None = None,
    suggest_patch: Path | None = None,
    run_id: str | None = None,
    started_at: datetime | None = None,
) -> PreparedSelfUseRun:
    """Resolve every input and output before a model or network call."""

    target_values = tuple(targets)
    if len(target_values) > 1 and (sarif is not None or suggest_patch is not None):
        raise SelfUseError("ambiguous_multi_target_output")
    instant = started_at or datetime.now(UTC)
    if instant.tzinfo is None or instant.utcoffset() is None:
        raise SelfUseError("started_at_not_timezone_aware")
    identifier = run_id or new_run_id(instant)
    if not _RUN_ID_RE.fullmatch(identifier):
        raise SelfUseError("invalid_run_id")
    inputs = preflight_self_use_inputs(
        target_values,
        references=references,
        command=mode,
        project_root=project_root,
    )
    legacy_targets = tuple(
        record for record in inputs.targets if record.kind == "legacy"
    )
    if legacy_targets and not (mode == "check" and len(inputs.targets) == 1):
        raise SelfUseError("unsupported_self_use_target")
    if changed_from is not None:
        changed_target = inputs.targets[0]
        if (
            len(inputs.targets) != 1
            or changed_target.kind not in {"markdown", "legacy"}
            or changed_target.display_path.startswith("external/")
        ):
            raise SelfUseError("changed_from_requires_internal_markdown")

    requested_root = (
        output_dir
        if output_dir is not None
        else inputs.project_root / ".evidencetrace" / "runs" / identifier
    )
    requests = [
        OutputPathRequest(
            "run_root",
            requested_root,
            origin="explicit" if output_dir is not None else "derived",
        ),
        OutputPathRequest(
            "batch_manifest",
            requested_root / "batch-manifest.json",
            origin="derived",
        ),
    ]
    for record in inputs.targets:
        target_root = requested_root / "targets" / record.safe_id
        requests.extend(
            (
                OutputPathRequest(
                    f"{record.safe_id}:directory",
                    target_root,
                    origin="derived",
                ),
                OutputPathRequest(
                    f"{record.safe_id}:audit_json",
                    target_root / "audit.json",
                    origin="derived",
                ),
                OutputPathRequest(
                    f"{record.safe_id}:audit_markdown",
                    target_root / "audit.md",
                    origin="derived",
                ),
                OutputPathRequest(
                    f"{record.safe_id}:sarif",
                    target_root / "results.sarif",
                    origin="derived",
                ),
                OutputPathRequest(
                    f"{record.safe_id}:suggested_diff",
                    target_root / "suggested.diff",
                    origin="derived",
                ),
                OutputPathRequest(
                    f"{record.safe_id}:applied_diff",
                    target_root / "applied.diff",
                    origin="derived",
                ),
                OutputPathRequest(
                    f"{record.safe_id}:reverse_diff",
                    target_root / "reverse.diff",
                    origin="derived",
                ),
            )
        )
    if sarif is not None:
        if sarif.is_symlink():
            raise SelfUseError("explicit_output_is_symlink")
        requests.append(OutputPathRequest("explicit_sarif", sarif))
    if suggest_patch is not None:
        if suggest_patch.is_symlink():
            raise SelfUseError("explicit_output_is_symlink")
        requests.append(OutputPathRequest("explicit_suggested_diff", suggest_patch))

    outputs = validate_output_paths(inputs, requests)
    output_by_name = {item.name: item.path for item in outputs}
    run_root = output_by_name["run_root"]
    if any(
        output.origin == "derived"
        and output.name != "run_root"
        and not _within(output.path, run_root)
        for output in outputs
    ):
        raise SelfUseError("derived_output_escaped_run_root")
    if run_root.exists() and not run_root.is_dir():
        raise SelfUseError("output_root_not_directory")
    for output in outputs:
        _validate_output_parent(output.path)
        if (
            output.origin == "derived"
            and output.name not in {"run_root"}
            and not output.name.endswith(":directory")
            and (output.path.exists() or output.path.is_symlink())
        ):
            raise SelfUseError("derived_output_exists")
    for name in ("explicit_sarif", "explicit_suggested_diff"):
        explicit = output_by_name.get(name)
        if explicit is not None and not _within(explicit, inputs.project_root):
            raise SelfUseError("explicit_output_escaped_project")

    layouts = tuple(
        TargetArtifactPaths(
            target_id=record.safe_id,
            directory=output_by_name[f"{record.safe_id}:directory"],
            audit_json=output_by_name[f"{record.safe_id}:audit_json"],
            audit_markdown=output_by_name[f"{record.safe_id}:audit_markdown"],
            sarif=output_by_name[f"{record.safe_id}:sarif"],
            suggested_diff=output_by_name[f"{record.safe_id}:suggested_diff"],
            applied_diff=output_by_name[f"{record.safe_id}:applied_diff"],
            reverse_diff=output_by_name[f"{record.safe_id}:reverse_diff"],
        )
        for record in inputs.targets
    )
    return PreparedSelfUseRun(
        inputs=replace(inputs, outputs=outputs),
        mode=mode,
        discover=discover,
        changed_from=changed_from,
        run_id=identifier,
        started_at=instant,
        run_root=run_root,
        manifest_path=output_by_name["batch_manifest"],
        targets=layouts,
        explicit_sarif=output_by_name.get("explicit_sarif"),
        explicit_suggested_diff=output_by_name.get("explicit_suggested_diff"),
    )


def _read_snapshot(record: InputFileRecord) -> bytes:
    try:
        if record.path.is_symlink() or not record.path.is_file():
            raise OSError
        content = record.path.read_bytes()
    except OSError:
        raise SelfUseError("input_changed_after_preflight") from None
    if sha256_bytes(content) != record.content_sha256:
        raise SelfUseError("input_changed_after_preflight")
    return content


def _revalidate_outputs_before_apply(prepared: PreparedSelfUseRun) -> None:
    requests = tuple(
        OutputPathRequest(
            output.name,
            output.path,
            origin=output.origin,
        )
        for output in prepared.inputs.outputs
    )
    validate_output_paths(prepared.inputs, requests)
    for output in prepared.inputs.outputs:
        _validate_output_parent(output.path)
        if (
            output.origin == "derived"
            and output.name != "run_root"
            and not output.name.endswith(":directory")
            and (output.path.exists() or output.path.is_symlink())
        ):
            raise SelfUseError("derived_output_changed_during_run")


def _challenger_gate(
    action: ChallengeAction | None,
    *,
    final_relation: Relation,
    evidence_text: str,
    evidence_scalar: str,
) -> ChallengerGate:
    if action is None:
        return ChallengerGate(completed=False, action=None)
    if action is ChallengeAction.REVISE:
        return ChallengerGate(
            completed=True,
            action=action,
            revised_relation=final_relation,
            revised_evidence_text=evidence_text,
            revised_evidence_scalar=evidence_scalar,
        )
    return ChallengerGate(completed=True, action=action)


def _repair_inputs(
    record: InputFileRecord,
    result: ProductRunResult,
    target_content: bytes,
) -> tuple[tuple[_ProposalContext, ...], tuple[_SkippedRepair, ...]]:
    source_lines = target_content.decode("utf-8").splitlines(keepends=True)
    claims = {claim.claim_id: claim for claim in result.audit.claims}
    resolutions = {
        resolution.claim_id: resolution
        for resolution in result.evidence_resolutions
    }
    claim_runs = {
        claim_run.claim_id: claim_run for claim_run in result.product.claim_runs
    }
    proposals: list[_ProposalContext] = []
    skipped: list[_SkippedRepair] = []
    for verdict in result.audit.verdicts:
        if verdict.relation is not Relation.CONTRADICTED:
            continue
        claim = claims.get(verdict.claim_id)
        if claim is None:
            continue
        proposal_id = f"{record.safe_id}:{claim.claim_id}"
        resolution = resolutions.get(claim.claim_id)
        selected_source = (
            resolution.selected_source_id if resolution is not None else None
        )
        spans = tuple(
            span
            for span in verdict.evidence_spans
            if selected_source is not None and span.source_id == selected_source
        )
        scalar_spans = tuple(
            (span, pair)
            for span in spans
            if (pair := derive_unique_scalar_pair(claim.text, span.text))
            is not None
        )
        if len(spans) != 1:
            skipped.append(
                _SkippedRepair(
                    proposal_id,
                    claim.claim_id,
                    claim.line_start,
                    RepairReason.EVIDENCE_NOT_EXACT,
                    selected_source,
                )
            )
            continue
        if len(scalar_spans) != 1:
            skipped.append(
                _SkippedRepair(
                    proposal_id,
                    claim.claim_id,
                    claim.line_start,
                    RepairReason.SCALAR_AMBIGUOUS,
                    selected_source,
                )
            )
            continue
        span, (target_scalar, evidence_scalar) = scalar_spans[0]
        source_claim_text = "".join(
            source_lines[claim.line_start - 1 : claim.line_end]
        ).rstrip("\r\n")
        if not source_claim_text or target_scalar not in source_claim_text:
            skipped.append(
                _SkippedRepair(
                    proposal_id,
                    claim.claim_id,
                    claim.line_start,
                    RepairReason.CLAIM_NOT_FOUND,
                    selected_source,
                )
            )
            continue
        references: tuple[FileDigest, ...] = ()
        if (
            resolution is not None
            and resolution.reference_id is not None
            and resolution.reference_sha256 is not None
        ):
            references = (
                FileDigest(
                    resolution.reference_id,
                    resolution.reference_sha256,
                ),
            )
        claim_run = claim_runs.get(claim.claim_id)
        gate = _challenger_gate(
            claim_run.challenger_action if claim_run is not None else None,
            final_relation=verdict.relation,
            evidence_text=span.text,
            evidence_scalar=evidence_scalar,
        )
        if (
            claim_run is not None
            and claim_run.challenger_action is ChallengeAction.REVISE
            and verdict.judge_version != CHALLENGER_CONTRACT_VERSION
        ):
            gate = ChallengerGate(completed=False, action=None)
        if claim_run is None or claim_run.status is not ClaimRunStatus.COMPLETED:
            gate = ChallengerGate(completed=False, action=None)
        proposal = RepairProposal(
            proposal_id=proposal_id,
            target=FileDigest(record.safe_id, record.content_sha256),
            claim_text=source_claim_text,
            line_start=claim.line_start,
            line_end=claim.line_end,
            target_scalar=target_scalar,
            evidence=ExactEvidence(
                source_id=span.source_id,
                text=span.text,
                scalar=evidence_scalar,
            ),
            final_relation=verdict.relation,
            challenger=gate,
            references=references,
        )
        proposals.append(_ProposalContext(claim.claim_id, proposal))
    return tuple(proposals), tuple(skipped)


def _empty_apply(plan: RepairPlan) -> ApplyResult:
    return ApplyResult(
        decisions=(),
        files=(),
        suggested_diff=plan.suggested_diff,
        applied_diff="",
        reverse_diff="",
    )


def _failed_product_result(
    *,
    parsed: ParsedDocumentWithSourceMap,
    run_id: str,
    started_at: datetime,
    discover: bool,
    live_agents_enabled: bool,
) -> ProductRunResult:
    """Create a path-safe canonical result for one isolated target failure."""

    audit = AuditArtifact(
        run=RunMetadata(
            run_id=run_id,
            started_at=started_at,
            finished_at=started_at,
            tool_version=__version__,
        ),
        effective_config=EffectiveConfig(),
        documents=(parsed.document,),
    )
    product = ProductRunArtifact(
        artifact_version="bounded-multi-agent-product-v1",
        document_status=DocumentRunStatus.FAILED,
        claim_runs=(),
        trace=(
            TraceEvent(
                sequence=1,
                state=TraceState.FAILED,
                agent=AgentName.CONTROLLER,
                action="target_audit",
            ),
        ),
        budget={
            "claim_limit": 0,
            "miner_calls": 0,
            "miner_window_limit": 0,
            "miner_windows_planned": 0,
            "miner_windows_dispatched": 0,
            "miner_provider_attempt_limit": 0,
            "miner_provider_attempts": 0,
            "coordinator_call_limit": 0,
            "coordinator_calls": 0,
            "search_query_limit": 0,
            "search_queries": 0,
            "search_queries_remaining": 0,
            "fetch_limit": 0,
            "fetches": 0,
            "fetches_remaining": 0,
            "judge_calls": 0,
            "challenger_calls": 0,
        },
        live_agents_enabled=live_agents_enabled,
        discover_enabled=discover,
    )
    return ProductRunResult(
        audit=audit,
        product=product,
        audit_path=None,
        terminal="Target audit failed safely.",
    )


def _apply_status(status: ApplicationStatus | None) -> RepairApplyStatus:
    if status is ApplicationStatus.APPLIED:
        return RepairApplyStatus.APPLIED
    if status is ApplicationStatus.STALE_TARGET:
        return RepairApplyStatus.STALE_TARGET
    if status is ApplicationStatus.STALE_REFERENCE:
        return RepairApplyStatus.STALE_REFERENCE
    if status is ApplicationStatus.APPLY_ERROR:
        return RepairApplyStatus.APPLY_ERROR
    return RepairApplyStatus.NOT_APPLIED


def _decision_records(
    audited: _AuditedTarget,
    reviews: Mapping[str, RepairReviewStatus],
    application: Mapping[str, ApplicationStatus],
) -> tuple[RepairDecisionRecord, ...]:
    contexts = {item.proposal.proposal_id: item for item in audited.proposals}
    records: list[RepairDecisionRecord] = []
    for decision in audited.plan.decisions:
        context = contexts[decision.proposal_id]
        candidate = decision.candidate
        records.append(
            RepairDecisionRecord(
                proposal_id=decision.proposal_id,
                claim_id=context.claim_id,
                line_start=context.proposal.line_start,
                original=(
                    candidate.original_scalar if candidate is not None else None
                ),
                replacement=(
                    candidate.replacement_scalar if candidate is not None else None
                ),
                evidence_source_id=(
                    candidate.evidence.source_id if candidate is not None else None
                ),
                eligibility_status=decision.status,
                eligibility_reason=decision.reason,
                review_status=reviews.get(
                    decision.proposal_id,
                    RepairReviewStatus.UNREVIEWED,
                ),
                apply_status=_apply_status(application.get(decision.proposal_id)),
            )
        )
    records.extend(
        RepairDecisionRecord(
            proposal_id=item.proposal_id,
            claim_id=item.claim_id,
            line_start=item.line_start,
            eligibility_status=DecisionStatus.NEEDS_HUMAN,
            eligibility_reason=item.reason,
            review_status=RepairReviewStatus.UNREVIEWED,
            evidence_source_id=None,
        )
        for item in audited.skipped
    )
    return tuple(records)


def _target_is_partial(
    mode: Literal["check", "fix"],
    result: ProductRunResult,
    decisions: tuple[RepairDecisionRecord, ...],
) -> bool:
    if result.product.document_status is DocumentRunStatus.PARTIAL:
        return True
    if any(
        resolution.status is EvidenceResolutionStatus.NEEDS_HUMAN
        for resolution in result.evidence_resolutions
    ):
        return True
    if any(
        decision.eligibility_status is not DecisionStatus.ELIGIBLE
        for decision in decisions
    ):
        return True
    if mode == "fix":
        return any(
            decision.review_status is not RepairReviewStatus.APPROVED
            or decision.apply_status is not RepairApplyStatus.APPLIED
            for decision in decisions
        )
    return False


def _file_state_status(
    *,
    before_sha256: str,
    after_sha256: str,
    partial: bool,
    decisions: tuple[RepairDecisionRecord, ...],
) -> Literal["unchanged", "applied", "partial", "apply_error"]:
    if any(
        decision.apply_status is RepairApplyStatus.APPLY_ERROR
        for decision in decisions
    ):
        return "apply_error"
    if before_sha256 == after_sha256:
        return "partial" if partial else "unchanged"
    return "partial" if partial else "applied"


def _canonical_target_artifact(
    audited: _AuditedTarget,
    *,
    mode: Literal["check", "fix"],
    status: Literal["complete", "partial", "fatal"],
    decisions: tuple[RepairDecisionRecord, ...],
    after_sha256: str,
    partial: bool,
    finished_at: datetime,
) -> SelfUseAuditArtifact:
    record = audited.record
    base = audited.result.audit.model_copy(
        update={
            "run": audited.result.audit.run.model_copy(
                update={"finished_at": finished_at}
            )
        }
    )
    return SelfUseAuditArtifact.model_validate(
        {
            **base.model_dump(mode="python"),
            "target_id": record.safe_id,
            "mode": mode,
            "status": status,
            "execution": audited.result.product,
            "evidence_resolutions": audited.result.evidence_resolutions,
            "repair_decisions": decisions,
            "file_state": TargetFileState(
                before_sha256=record.content_sha256,
                after_sha256=after_sha256,
                status=_file_state_status(
                    before_sha256=record.content_sha256,
                    after_sha256=after_sha256,
                    partial=partial,
                    decisions=decisions,
                ),
            ),
        }
    )


def _write_target_bundle(
    paths: TargetArtifactPaths,
    artifact: SelfUseAuditArtifact,
    *,
    suggested_diff: str,
    applied_diff: str,
    reverse_diff: str,
) -> None:
    """Atomically replace every derivative of one canonical target outcome."""

    atomic_write_text(
        paths.audit_json,
        artifact.model_dump_json(indent=2) + "\n",
    )
    atomic_write_text(paths.audit_markdown, render_self_use_markdown(artifact))
    atomic_write_bytes(paths.sarif, sarif_bytes(artifact))
    atomic_write_text(paths.suggested_diff, suggested_diff)
    atomic_write_text(paths.applied_diff, applied_diff)
    atomic_write_text(paths.reverse_diff, reverse_diff)


def _markdown_cell(value: str | None) -> str:
    return (
        (value or "—")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("|", r"\|")
        .replace("\n", " ")
    )


def render_self_use_markdown(artifact: SelfUseAuditArtifact) -> str:
    """Render the human view solely from one canonical self-use artifact."""

    content = render_audit_markdown(artifact)
    lines = [
        "",
        "## Self-use outcome",
        "",
        f"- Mode: `{artifact.mode}`",
        f"- Status: `{artifact.status}`",
        f"- File state: `{artifact.file_state.status}`",
        "",
        "## Evidence routing",
        "",
        "| Claim | Reason | Status | Selected source |",
        "|---|---|---|---|",
    ]
    for resolution in artifact.evidence_resolutions:
        lines.append(
            f"| `{_markdown_cell(resolution.claim_id)}` | "
            f"`{resolution.reason.value}` | `{resolution.status.value}` | "
            f"`{_markdown_cell(resolution.selected_source_id)}` |"
        )
    if not artifact.evidence_resolutions:
        lines.append("| — | — | — | No bounded evidence resolution. |")
    if any(
        resolution.options for resolution in artifact.evidence_resolutions
    ):
        lines.extend(
            [
                "",
                "### Bounded evidence options",
                "",
                "| Claim | Origin | Source / locator | Exact excerpt |",
                "|---|---|---|---|",
            ]
        )
        for resolution in artifact.evidence_resolutions:
            for option in resolution.options:
                evidence = option.evidence[0] if option.evidence else None
                locator = (
                    evidence.chunk.locator if evidence is not None else "—"
                )
                excerpt = (
                    evidence.text if evidence is not None else "No exact span."
                )
                lines.append(
                    f"| `{_markdown_cell(resolution.claim_id)}` | "
                    f"`{option.origin.value}` | "
                    f"`{_markdown_cell(option.source.source_id)}` / "
                    f"`{_markdown_cell(locator)}` | "
                    f"{_markdown_cell(excerpt)} |"
                )
    lines.extend(
        [
            "",
        "## Repair decisions",
        "",
        "| Claim | Eligibility | Review | Apply | Scalar |",
        "|---|---|---|---|---|",
        ]
    )
    for decision in artifact.repair_decisions:
        scalar = (
            f"{_markdown_cell(decision.original)} → "
            f"{_markdown_cell(decision.replacement)}"
        )
        lines.append(
            f"| `{_markdown_cell(decision.claim_id)}` | "
            f"`{decision.eligibility_status.value}/"
            f"{decision.eligibility_reason.value}` | "
            f"`{decision.review_status.value}` | "
            f"`{decision.apply_status.value}` | {scalar} |"
        )
    if not artifact.repair_decisions:
        lines.append("| — | — | — | — | No scalar repair decision. |")
    return content.rstrip("\n") + "\n" + "\n".join(lines) + "\n"


def render_self_use_terminal(artifact: SelfUseAuditArtifact) -> str:
    """Render terminal output from the same canonical outcome as artifacts."""

    eligible = sum(
        item.eligibility_status is DecisionStatus.ELIGIBLE
        for item in artifact.repair_decisions
    )
    approved = sum(
        item.review_status is RepairReviewStatus.APPROVED
        for item in artifact.repair_decisions
    )
    applied = sum(
        item.apply_status is RepairApplyStatus.APPLIED
        for item in artifact.repair_decisions
    )
    agents = (
        AgentName.MINER,
        AgentName.COORDINATOR,
        AgentName.SCOUT,
        AgentName.JUDGE,
        AgentName.CHALLENGER,
    )
    counts = {
        agent: sum(
            event.agent is agent and event.state is TraceState.DISPATCHED
            for event in artifact.execution.trace
        )
        for agent in agents
    }
    claim_runs = ", ".join(
        f"{item.claim_id}={item.status.value}"
        for item in artifact.execution.claim_runs
    )
    resolutions = ", ".join(
        f"{item.claim_id}={item.status.value}"
        for item in artifact.evidence_resolutions
    )
    budget = artifact.execution.budget
    return "\n".join(
        (
            render_terminal(artifact),
            f"Self-use target: {artifact.target_id}",
            f"Self-use mode: {artifact.mode}",
            f"Self-use status: {artifact.status}",
            "Claim runs: " + (claim_runs or "none"),
            "Evidence routing: " + (resolutions or "none"),
            "Agents run: "
            + ", ".join(
                f"{agent.value}={counts[agent]}" for agent in agents
            ),
            (
                "Remaining budget: "
                f"search_queries={budget['search_queries_remaining']}, "
                f"fetches={budget['fetches_remaining']}"
            ),
            (
                "Repair decisions: "
                f"eligible={eligible}, approved={approved}, applied={applied}"
            ),
        )
    )


class SelfUseRunner:
    """Sequential batch controller around the existing single-document pipeline."""

    def __init__(
        self,
        *,
        project_root: Path,
        model: ModelClient | None = None,
        fetcher: SafeFetcher | None = None,
        search_client: SearchClient | None = None,
        interactive: InteractiveSession | None = None,
    ) -> None:
        self.project_root = project_root.resolve()
        self.model = model
        self.fetcher = fetcher
        self.search_client = search_client
        self.interactive = interactive

    def run(
        self,
        targets: Sequence[Path | str],
        *,
        references: Sequence[Path | str] = (),
        mode: Literal["check", "fix"] = "check",
        discover: bool = False,
        changed_from: str | None = None,
        output_dir: Path | None = None,
        sarif: Path | None = None,
        suggest_patch: Path | None = None,
        run_id: str | None = None,
        started_at: datetime | None = None,
    ) -> SelfUseRunResult:
        prepared = prepare_self_use_run(
            targets,
            references=references,
            mode=mode,
            project_root=self.project_root,
            discover=discover,
            changed_from=changed_from,
            output_dir=output_dir,
            sarif=sarif,
            suggest_patch=suggest_patch,
            run_id=run_id,
            started_at=started_at,
        )
        return self.run_prepared(prepared)

    def run_prepared(self, prepared: PreparedSelfUseRun) -> SelfUseRunResult:
        if prepared.inputs.project_root != self.project_root:
            raise SelfUseError("prepared_root_mismatch")
        if prepared.mode == "fix" and self.interactive is None:
            raise SelfUseError("interactive_tty_required")

        target_snapshots = {
            record.safe_id: _read_snapshot(record)
            for record in prepared.inputs.targets
        }
        reference_snapshots = {
            record.safe_id: _read_snapshot(record)
            for record in prepared.inputs.references
        }
        try:
            reference_index = LocalReferenceIndex(
                LocalReferenceInput(
                    path=record.path,
                    reference_id=record.safe_id,
                    reference_sha256=record.content_sha256,
                )
                for record in prepared.inputs.references
            )
            parsed_targets = {
                record.safe_id: (
                    parse_plain_text_with_source_map(
                        target_snapshots[record.safe_id].decode("utf-8"),
                        record.safe_id,
                    )
                    if record.kind == "text"
                    else parse_markdown_with_source_map(
                        target_snapshots[record.safe_id].decode("utf-8"),
                        record.display_path,
                    )
                )
                for record in prepared.inputs.targets
                if record.kind in {"legacy", "markdown", "text"}
            }
        except Exception:
            raise SelfUseError("source_initialization_failed") from None
        if len(parsed_targets) != len(prepared.inputs.targets):
            raise SelfUseError("unsupported_self_use_target")

        shared_fetcher = self.fetcher or SafeFetcher()
        audited_targets: list[_AuditedTarget] = []
        for index, record in enumerate(prepared.inputs.targets, start=1):
            target_run_id = f"{prepared.run_id}-t{index:02d}"
            try:
                result = ProductAuditPipeline(
                    project_root=self.project_root,
                    model=self.model,
                    fetcher=shared_fetcher,
                    search_client=(
                        self.search_client if prepared.discover else None
                    ),
                    reference_index=reference_index,
                    evidence_resolver=(
                        self.interactive if prepared.mode == "fix" else None
                    ),
                    enforce_human_evidence_gates=True,
                ).run(
                    record.path,
                    discover=prepared.discover,
                    changed_from=prepared.changed_from,
                    run_id=target_run_id,
                    started_at=prepared.started_at,
                    persist=False,
                    parsed_document=parsed_targets[record.safe_id],
                )
            except Exception:
                result = _failed_product_result(
                    parsed=parsed_targets[record.safe_id],
                    run_id=target_run_id,
                    started_at=prepared.started_at,
                    discover=prepared.discover,
                    live_agents_enabled=self.model is not None,
                )
                audited_targets.append(
                    _AuditedTarget(
                        record=record,
                        result=result,
                        plan=RepairPlan(decisions=(), suggested_diff=""),
                        proposals=(),
                        skipped=(),
                        fatal=True,
                    )
                )
                continue
            proposals, skipped = _repair_inputs(
                record,
                result,
                target_snapshots[record.safe_id],
            )
            plan = plan_repairs(
                {record.safe_id: target_snapshots[record.safe_id]},
                tuple(item.proposal for item in proposals),
                reference_contents=reference_snapshots,
            )
            audited_targets.append(
                _AuditedTarget(
                    record=record,
                    result=result,
                    plan=plan,
                    proposals=proposals,
                    skipped=skipped,
                )
            )

        combined_plan = RepairPlan(
            decisions=tuple(
                decision
                for target in audited_targets
                for decision in target.plan.decisions
            ),
            suggested_diff="".join(
                target.plan.suggested_diff for target in audited_targets
            ),
        )
        reviews: dict[str, RepairReviewStatus] = {
            decision.proposal_id: RepairReviewStatus.UNREVIEWED
            for decision in combined_plan.decisions
        }
        approved: set[str] = set()
        if prepared.mode == "fix":
            assert self.interactive is not None
            for candidate in combined_plan.candidates:
                review = self.interactive.review_repair(
                    file=candidate.target.path,
                    line=candidate.line_start,
                    claim=candidate.claim_text,
                    original=candidate.original_scalar,
                    replacement=candidate.replacement_scalar,
                    evidence=candidate.evidence.text,
                    source=candidate.evidence.source_id,
                    final_outcome="contradicted / challenger=completed",
                )
                reviews[candidate.proposal_id] = review
                if review is RepairReviewStatus.APPROVED:
                    approved.add(candidate.proposal_id)
                if self.interactive.quit_requested:
                    break
            if approved and not self.interactive.confirm_apply(len(approved)):
                approved.clear()

        _revalidate_outputs_before_apply(prepared)
        if prepared.mode == "fix":
            preliminary_finished = max(datetime.now(UTC), prepared.started_at)
            preliminary_targets: list[BatchTargetRecord] = []
            preliminary_artifacts: list[SelfUseAuditArtifact] = []
            for audited, paths in zip(
                audited_targets, prepared.targets, strict=True
            ):
                decisions = _decision_records(audited, reviews, {})
                partial = audited.fatal or _target_is_partial(
                    prepared.mode,
                    audited.result,
                    decisions,
                )
                status: Literal["complete", "partial", "fatal"] = (
                    "fatal"
                    if audited.fatal
                    else ("partial" if partial else "complete")
                )
                artifact = _canonical_target_artifact(
                    audited,
                    mode=prepared.mode,
                    status=status,
                    decisions=decisions,
                    after_sha256=audited.record.content_sha256,
                    partial=partial,
                    finished_at=preliminary_finished,
                )
                _write_target_bundle(
                    paths,
                    artifact,
                    suggested_diff=audited.plan.suggested_diff,
                    applied_diff="",
                    reverse_diff="",
                )
                preliminary_artifacts.append(artifact)
                preliminary_targets.append(
                    BatchTargetRecord(
                        target_id=audited.record.safe_id,
                        display_path=audited.record.display_path,
                        status=status,
                        audit_path=(
                            f"targets/{audited.record.safe_id}/audit.json"
                        ),
                        before_sha256=audited.record.content_sha256,
                        after_sha256=audited.record.content_sha256,
                    )
                )
            preliminary_manifest = BatchManifest(
                run_id=prepared.run_id,
                mode=prepared.mode,
                status=manifest_status(
                    tuple(item.status for item in preliminary_targets)
                ),
                started_at=prepared.started_at,
                finished_at=preliminary_finished,
                targets=tuple(preliminary_targets),
                references=tuple(
                    BatchReferenceRecord(
                        reference_id=record.safe_id,
                        content_sha256=record.content_sha256,
                    )
                    for record in prepared.inputs.references
                ),
            )
            write_json_artifact(prepared.manifest_path, preliminary_manifest)
            if prepared.explicit_sarif is not None:
                atomic_write_bytes(
                    prepared.explicit_sarif,
                    sarif_bytes(preliminary_artifacts[0]),
                )
            if prepared.explicit_suggested_diff is not None:
                atomic_write_text(
                    prepared.explicit_suggested_diff,
                    audited_targets[0].plan.suggested_diff,
                )
        runtime_paths = {
            record.safe_id: record.path
            for record in (
                *prepared.inputs.targets,
                *prepared.inputs.references,
            )
        }
        artifact_paths = {
            paths.target_id: paths for paths in prepared.targets
        }

        def persist_reverse_before_source_write(
            preview: FileApplyResult,
        ) -> None:
            paths = artifact_paths.get(preview.target_path)
            if (
                paths is None
                or preview.status is not ApplicationStatus.APPLIED
                or not preview.reverse_diff
            ):
                raise SelfUseError("invalid_apply_preview")
            atomic_write_text(paths.reverse_diff, preview.reverse_diff)

        apply_result = (
            apply_repair_plan(
                self.project_root,
                combined_plan,
                approved,
                runtime_paths=runtime_paths,
                before_file_write=persist_reverse_before_source_write,
            )
            if prepared.mode == "fix"
            else _empty_apply(combined_plan)
        )
        application = {
            decision.proposal_id: decision.status
            for decision in apply_result.decisions
        }
        file_apply = {
            result.target_path: result for result in apply_result.files
        }

        finished_at = max(datetime.now(UTC), prepared.started_at)
        target_results: list[SelfUseTargetResult] = []
        manifest_targets: list[BatchTargetRecord] = []
        for audited, paths in zip(
            audited_targets, prepared.targets, strict=True
        ):
            record = audited.record
            try:
                if (
                    record.path.is_symlink()
                    or not record.path.is_file()
                    or record.path.resolve(strict=True) != record.path
                ):
                    raise OSError
                after_bytes = record.path.read_bytes()
            except OSError:
                raise SelfUseError("target_read_after_apply_failed") from None
            after_sha256 = sha256_bytes(after_bytes)
            applied = file_apply.get(record.safe_id)
            unexpected_target_change = (
                after_sha256 != record.content_sha256
                and (
                    applied is None
                    or applied.status is not ApplicationStatus.APPLIED
                )
            )
            if (
                applied is not None
                and applied.status is ApplicationStatus.APPLIED
                and applied.after_sha256 != after_sha256
            ):
                for proposal_id in applied.proposal_ids:
                    application[proposal_id] = ApplicationStatus.APPLY_ERROR
                applied = replace(
                    applied,
                    status=ApplicationStatus.APPLY_ERROR,
                    after_sha256=after_sha256,
                    applied_diff="",
                    reverse_diff="",
                )
                file_apply[record.safe_id] = applied
            decisions = _decision_records(audited, reviews, application)
            partial = audited.fatal or _target_is_partial(
                prepared.mode, audited.result, decisions
            )
            if unexpected_target_change:
                partial = True
            target_status: Literal["complete", "partial", "fatal"] = (
                "fatal"
                if audited.fatal
                else ("partial" if partial else "complete")
            )
            artifact = _canonical_target_artifact(
                audited,
                mode=prepared.mode,
                status=target_status,
                decisions=decisions,
                after_sha256=after_sha256,
                partial=partial,
                finished_at=finished_at,
            )
            _write_target_bundle(
                paths,
                artifact,
                suggested_diff=audited.plan.suggested_diff,
                applied_diff=applied.applied_diff if applied is not None else "",
                reverse_diff=applied.reverse_diff if applied is not None else "",
            )
            terminal = render_self_use_terminal(artifact)
            target_results.append(
                SelfUseTargetResult(
                    target_id=record.safe_id,
                    status=target_status,
                    artifact=artifact,
                    paths=paths,
                    terminal=terminal,
                )
            )
            manifest_targets.append(
                BatchTargetRecord(
                    target_id=record.safe_id,
                    display_path=record.display_path,
                    status=target_status,
                    audit_path=f"targets/{record.safe_id}/audit.json",
                    before_sha256=record.content_sha256,
                    after_sha256=after_sha256,
                )
            )

        status = manifest_status(tuple(item.status for item in target_results))
        manifest = BatchManifest(
            run_id=prepared.run_id,
            mode=prepared.mode,
            status=status,
            started_at=prepared.started_at,
            finished_at=finished_at,
            targets=tuple(manifest_targets),
            references=tuple(
                BatchReferenceRecord(
                    reference_id=record.safe_id,
                    content_sha256=record.content_sha256,
                )
                for record in prepared.inputs.references
            ),
        )
        write_json_artifact(prepared.manifest_path, manifest)
        if prepared.explicit_sarif is not None:
            atomic_write_bytes(
                prepared.explicit_sarif,
                sarif_bytes(target_results[0].artifact),
            )
        if prepared.explicit_suggested_diff is not None:
            atomic_write_text(
                prepared.explicit_suggested_diff,
                audited_targets[0].plan.suggested_diff,
            )
        terminal = "\n\n".join(item.terminal for item in target_results)
        terminal += (
            f"\n\nBatch status: {manifest.status}\n"
            f"Run ID: {manifest.run_id}"
        )
        return SelfUseRunResult(
            manifest=manifest,
            run_root=prepared.run_root,
            targets=tuple(target_results),
            terminal=terminal,
        )


__all__ = [
    "BatchManifest",
    "BatchReferenceRecord",
    "BatchTargetRecord",
    "InteractiveSession",
    "PreparedSelfUseRun",
    "RepairApplyStatus",
    "RepairDecisionRecord",
    "RepairReviewStatus",
    "SelfUseAuditArtifact",
    "SelfUseError",
    "SelfUseRunResult",
    "SelfUseRunner",
    "SelfUseTargetResult",
    "TargetArtifactPaths",
    "TargetFileState",
    "atomic_write_bytes",
    "atomic_write_text",
    "manifest_status",
    "new_run_id",
    "prepare_self_use_run",
    "read_self_use_audit",
    "render_self_use_markdown",
    "render_self_use_terminal",
    "write_json_artifact",
]
