"""Typed contracts for the deterministic EvidenceTrace core.

The models in this module deliberately describe data, not Phase 2 behaviour.
They are shared by the Markdown parser, Git diff mapper, policy functions, and
canonical artifact writer.
"""

from __future__ import annotations

import re
from datetime import datetime
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveInt,
    field_validator,
    model_validator,
)

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-fA-F]{7,64}$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SLUG_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_DOMAIN_RE = re.compile(
    r"^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)*"
    r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$"
)

ParagraphMiningReasonCode = Literal[
    "budget_exhausted",
    "miner_coverage_empty_identifier_window",
    "miner_coverage_incomplete",
    "miner_error",
    "miner_local_validation_error",
    "miner_model_response_error",
    "miner_schema_constraint",
    "miner_schema_empty_content",
    "miner_schema_enum",
    "miner_schema_extra_field",
    "miner_schema_finish_reason",
    "miner_schema_invalid_envelope",
    "miner_schema_invalid_json",
    "miner_schema_missing_field",
    "miner_schema_wrong_type",
    "miner_scope_ambiguous_span",
    "miner_scope_invalid_claim_fragment",
    "miner_scope_missing_protected_token",
    "miner_scope_non_source_span",
    "miner_transport_error",
    "miner_unknown_error",
    "miner_window_fragment_oversized",
]

ClaimOperationalReasonCode = Literal[
    "budget_exhausted",
    "challenger_error",
    "discovery_unavailable",
    "human_review_requested",
    "judge_error",
    "scout_error",
]
ClaimOperationalStatus = Literal[
    "agent_error",
    "budget_exhausted",
    "needs_human",
]

ModelTextSourceKind = Literal[
    "plain",
    "link_label",
    "inline_code",
    "inline_code_identifier",
]


def _repo_path(value: str) -> str:
    """Validate a stable, project-relative POSIX path."""

    if not value or value != value.strip():
        raise ValueError("path must be non-empty and have no surrounding whitespace")
    if "\\" in value:
        raise ValueError("path must use POSIX separators")
    path = PurePosixPath(value)
    if path.is_absolute() or value in {".", ".."} or ".." in path.parts:
        raise ValueError("path must be project-relative and cannot contain '..'")
    return value


def _non_empty(value: str, *, field_name: str = "value") -> str:
    if not value.strip():
        raise ValueError(f"{field_name} must not be blank")
    return value


def _identifier(value: str) -> str:
    if not _IDENTIFIER_RE.fullmatch(value):
        raise ValueError("identifier contains unsupported characters")
    return value


class ContractModel(BaseModel):
    """Base settings for all serialized deterministic contracts."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class SourceSpan(ContractModel):
    """A non-negative source slice with human-readable one-based locations.

    Lines and columns are one-based and inclusive. Offsets index the original
    Python string and use the conventional half-open interval.
    """

    file: str
    line_start: PositiveInt
    line_end: PositiveInt
    column_start: PositiveInt
    column_end: PositiveInt
    offset_start: NonNegativeInt
    offset_end: NonNegativeInt

    _validate_file = field_validator("file")(_repo_path)

    @model_validator(mode="after")
    def validate_order(self) -> Self:
        if self.line_end < self.line_start:
            raise ValueError("line_end cannot precede line_start")
        if self.line_end == self.line_start and self.column_end < self.column_start:
            raise ValueError("column_end cannot precede column_start on one line")
        if self.offset_end < self.offset_start:
            raise ValueError("offset_end cannot precede offset_start")
        return self


class ParagraphKind(str, Enum):
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    BLOCKQUOTE = "blockquote"


class CitationKind(str, Enum):
    INLINE_LINK = "inline_link"
    REFERENCE_LINK = "reference_link"
    FOOTNOTE = "footnote"
    AUTOLINK = "autolink"
    BARE_URL = "bare_url"


class TargetKind(str, Enum):
    WEB = "web"
    RELATIVE = "relative"
    ANCHOR = "anchor"
    EMAIL = "email"
    OTHER = "other"


class DefinitionKind(str, Enum):
    REFERENCE = "reference"
    FOOTNOTE = "footnote"


class SuppressionScope(str, Enum):
    LINE = "line"
    PARAGRAPH = "paragraph"


class DiagnosticSeverity(str, Enum):
    NOTICE = "notice"
    WARNING = "warning"
    ERROR = "error"


class MarkdownParagraph(ContractModel):
    paragraph_id: str
    kind: ParagraphKind
    source: SourceSpan
    raw_text: str
    plain_text: str
    heading_path: tuple[str, ...] = ()
    citation_ids: tuple[str, ...] = ()
    suppression_ids: tuple[str, ...] = ()

    _validate_id = field_validator("paragraph_id")(_identifier)


class ModelTextSourceSegment(ContractModel):
    """Trusted line provenance for one contiguous model-visible text range."""

    model_start: NonNegativeInt
    model_end: PositiveInt
    source_line_start: PositiveInt
    source_line_end: PositiveInt
    split_protected: bool = False
    kind: ModelTextSourceKind = "plain"

    @model_validator(mode="after")
    def validate_ranges(self) -> Self:
        if self.model_end <= self.model_start:
            raise ValueError("model text segment must be non-empty")
        if self.source_line_end < self.source_line_start:
            raise ValueError("source line range must be ordered")
        return self


class TrustedModelVisibleParagraph(ContractModel):
    """Local-only model text and line map; never part of AuditArtifact."""

    paragraph_id: str
    text: str
    source: SourceSpan
    segments: tuple[ModelTextSourceSegment, ...] = ()
    heading_path: tuple[str, ...] = ()
    citation_ids: tuple[str, ...] = ()
    citation_urls: tuple[str, ...] = ()

    _validate_id = field_validator("paragraph_id")(_identifier)

    @model_validator(mode="after")
    def validate_source_map(self) -> Self:
        if not self.text:
            if self.segments:
                raise ValueError("empty model text cannot have source segments")
            return self
        if not self.segments:
            raise ValueError("non-empty model text requires source segments")
        cursor = 0
        for segment in self.segments:
            if segment.model_start != cursor:
                raise ValueError("model text segments must be contiguous and ordered")
            if (
                segment.source_line_start < self.source.line_start
                or segment.source_line_end > self.source.line_end
            ):
                raise ValueError("model text segment escaped paragraph source lines")
            cursor = segment.model_end
        if cursor != len(self.text):
            raise ValueError("model text segments must cover the complete text")
        return self

    def source_line_range(self, start: int, end: int) -> tuple[int, int]:
        """Return the conservative source lines for a non-empty text slice."""

        if not 0 <= start < end <= len(self.text):
            raise ValueError("model-visible slice must be non-empty and in bounds")
        matches = tuple(
            segment
            for segment in self.segments
            if start < segment.model_end and segment.model_start < end
        )
        if not matches:
            raise ValueError("model-visible slice has no trusted source mapping")
        return (
            min(segment.source_line_start for segment in matches),
            max(segment.source_line_end for segment in matches),
        )

    def split_protected_ranges(self) -> tuple[tuple[int, int], ...]:
        """Return merged model-text ranges that must not be split internally."""

        merged: list[tuple[int, int]] = []
        for segment in self.segments:
            if not segment.split_protected:
                continue
            if merged and merged[-1][1] == segment.model_start:
                merged[-1] = (merged[-1][0], segment.model_end)
            else:
                merged.append((segment.model_start, segment.model_end))
        return tuple(merged)

    def inline_code_identifier_ranges(self) -> tuple[tuple[int, int], ...]:
        """Return parser-owned model ranges for narrow inline-code identifiers."""

        merged: list[tuple[int, int]] = []
        for segment in self.segments:
            if segment.kind != "inline_code_identifier":
                continue
            if merged and merged[-1][1] == segment.model_start:
                merged[-1] = (merged[-1][0], segment.model_end)
            else:
                merged.append((segment.model_start, segment.model_end))
        return tuple(merged)


class CitationOccurrence(ContractModel):
    citation_id: str
    kind: CitationKind
    raw: str
    label: str | None = None
    reference_key: str | None = None
    destination: str | None = None
    target_kind: TargetKind | None = None
    resolved_urls: tuple[str, ...] = ()
    definition_id: str | None = None
    source: SourceSpan
    resolved: bool

    _validate_id = field_validator("citation_id")(_identifier)

    @field_validator("definition_id")
    @classmethod
    def validate_definition_id(cls, value: str | None) -> str | None:
        return _identifier(value) if value is not None else None

    @field_validator("resolved_urls")
    @classmethod
    def validate_resolved_urls(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        for value in values:
            _non_empty(value, field_name="resolved URL")
        return values


class CitationDefinition(ContractModel):
    definition_id: str
    kind: DefinitionKind
    normalized_key: str
    raw_text: str
    destinations: tuple[str, ...] = ()
    source: SourceSpan

    _validate_id = field_validator("definition_id")(_identifier)

    @field_validator("normalized_key")
    @classmethod
    def validate_key(cls, value: str) -> str:
        return _non_empty(value, field_name="normalized_key")

    @field_validator("destinations")
    @classmethod
    def validate_destinations(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        for value in values:
            _non_empty(value, field_name="destination")
        return values


class SuppressionDirective(ContractModel):
    suppression_id: str
    action: Literal["ignore"] = "ignore"
    reason: str
    scope: SuppressionScope
    directive_source: SourceSpan
    target_source: SourceSpan

    _validate_id = field_validator("suppression_id")(_identifier)

    @field_validator("reason")
    @classmethod
    def validate_reason(cls, value: str) -> str:
        return _non_empty(value, field_name="suppression reason")

    @model_validator(mode="after")
    def validate_files(self) -> Self:
        if self.directive_source.file != self.target_source.file:
            raise ValueError("suppression directive and target must be in one file")
        return self


class ParseDiagnostic(ContractModel):
    code: str
    message: str
    severity: DiagnosticSeverity
    source: SourceSpan | None = None

    @field_validator("code")
    @classmethod
    def validate_code(cls, value: str) -> str:
        if not _SLUG_RE.fullmatch(value):
            raise ValueError("diagnostic code must be a lowercase snake_case slug")
        return value

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        return _non_empty(value, field_name="diagnostic message")


class ParsedDocument(ContractModel):
    schema_version: Literal[1] = 1
    path: str
    content_sha256: str
    line_count: NonNegativeInt
    paragraphs: tuple[MarkdownParagraph, ...] = ()
    citations: tuple[CitationOccurrence, ...] = ()
    definitions: tuple[CitationDefinition, ...] = ()
    suppressions: tuple[SuppressionDirective, ...] = ()
    diagnostics: tuple[ParseDiagnostic, ...] = ()

    _validate_path = field_validator("path")(_repo_path)

    @field_validator("content_sha256")
    @classmethod
    def validate_content_hash(cls, value: str) -> str:
        if not _SHA256_RE.fullmatch(value):
            raise ValueError("content_sha256 must be a 64-character hexadecimal hash")
        return value.lower()

    @model_validator(mode="after")
    def validate_child_files(self) -> Self:
        spans = (
            [paragraph.source for paragraph in self.paragraphs]
            + [citation.source for citation in self.citations]
            + [definition.source for definition in self.definitions]
            + [suppression.directive_source for suppression in self.suppressions]
            + [suppression.target_source for suppression in self.suppressions]
            + [diagnostic.source for diagnostic in self.diagnostics if diagnostic.source]
        )
        if any(span.file != self.path for span in spans):
            raise ValueError("all document source spans must use the document path")
        return self


class ParsedDocumentWithSourceMap(ContractModel):
    """Parser result plus local trusted maps kept outside public artifacts."""

    document: ParsedDocument
    model_visible_paragraphs: tuple[TrustedModelVisibleParagraph, ...] = ()

    @model_validator(mode="after")
    def validate_paragraph_alignment(self) -> Self:
        public = self.document.paragraphs
        trusted = self.model_visible_paragraphs
        citations = {
            citation.citation_id: citation for citation in self.document.citations
        }
        if len(public) != len(trusted):
            raise ValueError("public and trusted paragraph counts must match")
        for paragraph, model_visible in zip(public, trusted, strict=True):
            if any(
                citation_id not in citations for citation_id in paragraph.citation_ids
            ):
                raise ValueError("trusted paragraph references an unknown citation")
            citation_urls = tuple(
                url
                for citation_id in paragraph.citation_ids
                for url in citations[citation_id].resolved_urls
            )
            if (
                paragraph.paragraph_id != model_visible.paragraph_id
                or paragraph.plain_text != model_visible.text
                or paragraph.source != model_visible.source
                or paragraph.heading_path != model_visible.heading_path
                or paragraph.citation_ids != model_visible.citation_ids
                or citation_urls != model_visible.citation_urls
            ):
                raise ValueError("trusted model text does not match parsed paragraph")
        return self


class LineRange(ContractModel):
    """A one-based inclusive line range."""

    start: PositiveInt
    end: PositiveInt

    @model_validator(mode="after")
    def validate_order(self) -> Self:
        if self.end < self.start:
            raise ValueError("line range end cannot precede start")
        return self


class DiffHunk(ContractModel):
    old_start: NonNegativeInt
    old_count: NonNegativeInt
    new_start: NonNegativeInt
    new_count: NonNegativeInt
    added_ranges: tuple[LineRange, ...] = ()
    deletion_anchor: NonNegativeInt | None = None

    @model_validator(mode="after")
    def validate_empty_new_side(self) -> Self:
        if self.new_count == 0 and self.added_ranges:
            raise ValueError("a deletion-only hunk cannot contain added ranges")
        return self


class ChangeStatus(str, Enum):
    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"
    RENAMED = "renamed"
    COPIED = "copied"


class FileChange(ContractModel):
    status: ChangeStatus
    old_path: str | None = None
    new_path: str | None = None
    hunks: tuple[DiffHunk, ...] = ()

    @field_validator("old_path", "new_path")
    @classmethod
    def validate_optional_path(cls, value: str | None) -> str | None:
        return _repo_path(value) if value is not None else None

    @model_validator(mode="after")
    def validate_paths_for_status(self) -> Self:
        if self.old_path is None and self.new_path is None:
            raise ValueError("a file change requires an old or new path")
        if self.status in {ChangeStatus.RENAMED, ChangeStatus.COPIED} and (
            self.old_path is None or self.new_path is None
        ):
            raise ValueError("renamed and copied files require old_path and new_path")
        if self.status == ChangeStatus.ADDED and self.new_path is None:
            raise ValueError("an added file requires new_path")
        if self.status == ChangeStatus.DELETED and self.old_path is None:
            raise ValueError("a deleted file requires old_path")
        return self


class GitDiff(ContractModel):
    base_revision: str | None = None
    files: tuple[FileChange, ...] = ()
    diagnostics: tuple[ParseDiagnostic, ...] = ()

    @field_validator("base_revision")
    @classmethod
    def validate_base_revision(cls, value: str | None) -> str | None:
        return _non_empty(value, field_name="base_revision") if value is not None else None


class ChangeSelection(ContractModel):
    path: str
    changed_ranges: tuple[LineRange, ...] = ()
    paragraph_ids: tuple[str, ...] = ()
    changed_citation_ids: tuple[str, ...] = ()
    changed_definition_ids: tuple[str, ...] = ()
    changed_suppression_ids: tuple[str, ...] = ()

    _validate_path = field_validator("path")(_repo_path)


class Relation(str, Enum):
    ENTAILED = "entailed"
    PARTIALLY_ENTAILED = "partially_entailed"
    CONTRADICTED = "contradicted"
    NOT_IN_SOURCE = "not_in_source"
    SOURCE_UNAVAILABLE = "source_unavailable"
    NOT_CHECKABLE = "not_checkable"


class CorroborationStatus(str, Enum):
    CITED_ONLY = "cited_only"
    CORROBORATED = "corroborated"
    DISPUTED = "disputed"
    NO_EVIDENCE = "no_evidence"
    NOT_REQUESTED = "not_requested"


class Severity(str, Enum):
    PASS = "pass"
    NOTICE = "notice"
    WARNING = "warning"
    ERROR = "error"


class Checkability(str, Enum):
    CHECKABLE = "checkable"
    NOT_CHECKABLE = "not_checkable"


class PolicyLabel(str, Enum):
    ENTAILED = "entailed"
    PARTIALLY_ENTAILED = "partially_entailed"
    CONTRADICTED = "contradicted"
    NOT_IN_SOURCE = "not_in_source"
    SOURCE_UNAVAILABLE = "source_unavailable"
    NOT_CHECKABLE = "not_checkable"
    CITED_ONLY = "cited_only"
    CORROBORATED = "corroborated"
    DISPUTED = "disputed"
    NO_EVIDENCE = "no_evidence"
    NOT_REQUESTED = "not_requested"


class AtomicClaim(ContractModel):
    claim_id: str
    text: str
    file: str
    line_start: PositiveInt
    line_end: PositiveInt
    claim_type: str
    slots: dict[str, str] = Field(default_factory=dict)
    checkability: Checkability
    citation_urls: tuple[str, ...] = ()

    _validate_id = field_validator("claim_id")(_identifier)
    _validate_file = field_validator("file")(_repo_path)

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        return _non_empty(value, field_name="claim text")

    @field_validator("claim_type")
    @classmethod
    def validate_claim_type(cls, value: str) -> str:
        if not _SLUG_RE.fullmatch(value):
            raise ValueError("claim_type must be a lowercase snake_case slug")
        return value

    @model_validator(mode="after")
    def validate_lines(self) -> Self:
        if self.line_end < self.line_start:
            raise ValueError("claim line_end cannot precede line_start")
        return self


class EvidenceSpan(ContractModel):
    source_id: str
    text: str
    locator: str

    _validate_id = field_validator("source_id")(_identifier)

    @field_validator("text", "locator")
    @classmethod
    def validate_non_empty_fields(cls, value: str) -> str:
        return _non_empty(value)


class Verdict(ContractModel):
    claim_id: str
    relation: Relation
    corroboration: CorroborationStatus = CorroborationStatus.NOT_REQUESTED
    confidence: float = Field(ge=0.0, le=1.0)
    source_ids: tuple[str, ...] = ()
    evidence_spans: tuple[EvidenceSpan, ...] = ()
    reason: str
    judge_version: str

    _validate_claim_id = field_validator("claim_id")(_identifier)

    @field_validator("source_ids")
    @classmethod
    def validate_source_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        for value in values:
            _identifier(value)
        if len(values) != len(set(values)):
            raise ValueError("source_ids must be unique")
        return values

    @field_validator("reason", "judge_version")
    @classmethod
    def validate_non_empty_fields(cls, value: str) -> str:
        return _non_empty(value)

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        substantive = {
            Relation.ENTAILED,
            Relation.PARTIALLY_ENTAILED,
            Relation.CONTRADICTED,
        }
        if self.relation in substantive and not self.evidence_spans:
            raise ValueError(f"{self.relation.value} requires an evidence span")
        source_ids = set(self.source_ids)
        missing = {
            span.source_id for span in self.evidence_spans if span.source_id not in source_ids
        }
        if missing:
            raise ValueError("every evidence span source_id must appear in source_ids")
        return self


class PathsConfig(ContractModel):
    include: tuple[str, ...] = ("**/*.md",)
    exclude: tuple[str, ...] = ()

    @field_validator("include", "exclude")
    @classmethod
    def validate_patterns(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        for value in values:
            if not value or value != value.strip():
                raise ValueError("path patterns must be non-empty and trimmed")
            if value.startswith("/") or "\\" in value or ".." in PurePosixPath(value).parts:
                raise ValueError("path patterns must be project-relative POSIX globs")
        return values


class PolicyConfig(ContractModel):
    fail_on: tuple[PolicyLabel, ...] = (PolicyLabel.CONTRADICTED,)
    warn_on: tuple[PolicyLabel, ...] = (
        PolicyLabel.PARTIALLY_ENTAILED,
        PolicyLabel.NOT_IN_SOURCE,
        PolicyLabel.DISPUTED,
        PolicyLabel.NO_EVIDENCE,
    )
    notice_on: tuple[PolicyLabel, ...] = (
        PolicyLabel.SOURCE_UNAVAILABLE,
        PolicyLabel.NOT_CHECKABLE,
    )
    trusted_domains: tuple[str, ...] = ()
    prefer_primary_sources: bool = True
    require_source_span: bool = True
    ignore_claim_types: tuple[str, ...] = ()
    fail_threshold: Severity = Severity.ERROR

    @field_validator("trusted_domains")
    @classmethod
    def validate_domains(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized: list[str] = []
        for value in values:
            domain = value.strip().lower().rstrip(".")
            if not _DOMAIN_RE.fullmatch(domain):
                raise ValueError("trusted_domains entries must be bare DNS names")
            if domain not in normalized:
                normalized.append(domain)
        return tuple(normalized)

    @field_validator("ignore_claim_types")
    @classmethod
    def validate_ignored_types(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not _SLUG_RE.fullmatch(value) for value in values):
            raise ValueError("ignored claim types must be lowercase snake_case slugs")
        return tuple(dict.fromkeys(values))

    @model_validator(mode="after")
    def validate_label_sets(self) -> Self:
        groups = [set(self.fail_on), set(self.warn_on), set(self.notice_on)]
        if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
            raise ValueError("fail_on, warn_on, and notice_on must be disjoint")
        if self.fail_threshold == Severity.PASS:
            raise ValueError("fail_threshold cannot be pass")
        return self


class BudgetConfig(ContractModel):
    max_changed_claims: PositiveInt = 30
    max_fetches: NonNegativeInt = 20
    max_searches: NonNegativeInt = 6
    max_cost_usd: float = Field(default=0.50, ge=0.0)
    timeout_seconds: PositiveInt = 180


class CacheConfig(ContractModel):
    ttl_days: NonNegativeInt = 30
    path: str = ".evidencetrace/cache.sqlite3"

    _validate_path = field_validator("path")(_repo_path)


class ModelConfig(ContractModel):
    provider: str = "openai-compatible"
    name: str | None = None

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, value: str) -> str:
        return _non_empty(value, field_name="model provider")

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str | None) -> str | None:
        return _non_empty(value, field_name="model name") if value is not None else None


class EffectiveConfig(ContractModel):
    version: Literal[1] = 1
    paths: PathsConfig = Field(default_factory=PathsConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)


class RunMetadata(ContractModel):
    run_id: str
    started_at: datetime
    git_sha: str | None = None
    tool_version: str
    finished_at: datetime | None = None

    @field_validator("run_id")
    @classmethod
    def validate_run_id(cls, value: str) -> str:
        if not _RUN_ID_RE.fullmatch(value) or value in {".", ".."}:
            raise ValueError("run_id must be a safe directory component")
        return value

    @field_validator("started_at", "finished_at")
    @classmethod
    def validate_timezone(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("run timestamps must be timezone-aware")
        return value

    @field_validator("git_sha")
    @classmethod
    def validate_git_sha(cls, value: str | None) -> str | None:
        if value is not None and not _GIT_SHA_RE.fullmatch(value):
            raise ValueError("git_sha must be a 7-64 character hexadecimal hash")
        return value.lower() if value is not None else None

    @field_validator("tool_version")
    @classmethod
    def validate_tool_version(cls, value: str) -> str:
        return _non_empty(value, field_name="tool_version")

    @model_validator(mode="after")
    def validate_finished_at(self) -> Self:
        if self.finished_at is not None and self.finished_at < self.started_at:
            raise ValueError("finished_at cannot precede started_at")
        return self


class RunPaths(ContractModel):
    run_dir: Path
    audit_json: Path

    @model_validator(mode="after")
    def validate_canonical_audit_path(self) -> Self:
        if self.audit_json != self.run_dir / "audit.json":
            raise ValueError("audit_json must be <run_dir>/audit.json")
        return self


class SuppressionKind(str, Enum):
    DIRECTIVE = "directive"
    CLAIM_TYPE = "claim_type"


class SuppressionDecision(ContractModel):
    kind: SuppressionKind
    reason: str
    suppression_id: str | None = None
    source: SourceSpan | None = None

    @field_validator("reason")
    @classmethod
    def validate_reason(cls, value: str) -> str:
        return _non_empty(value, field_name="suppression decision reason")


class PolicyDecision(ContractModel):
    relation: Relation
    corroboration: CorroborationStatus
    raw_severity: Severity
    effective_severity: Severity | None
    suppression: SuppressionDecision | None = None

    @model_validator(mode="after")
    def validate_suppressed_severity(self) -> Self:
        if (self.suppression is None) != (self.effective_severity is not None):
            raise ValueError(
                "effective_severity must be absent exactly when a suppression applies"
            )
        return self


class SourceMetadata(ContractModel):
    """Safe, non-secret metadata for a fetched evidence source."""

    source_id: str
    url: str = Field(max_length=2_048)
    title: str = Field(default="", max_length=512)
    retrieved_at: datetime
    content_hash: str
    mime_type: str
    status: Literal["ok", "unavailable"] = "ok"

    _validate_id = field_validator("source_id")(_identifier)

    @field_validator("url", "content_hash", "mime_type")
    @classmethod
    def validate_text_fields(cls, value: str) -> str:
        return _non_empty(value)

    @field_validator("url")
    @classmethod
    def validate_url_bytes(cls, value: str) -> str:
        if len(value.encode("utf-8")) > 8_192:
            raise ValueError("source URL exceeds the hard size limit")
        return value

    @field_validator("title")
    @classmethod
    def validate_title_bytes(cls, value: str) -> str:
        if len(value.encode("utf-8")) > 2_048:
            raise ValueError("source title exceeds the hard size limit")
        return value


class ParagraphMiningAuditOutcome(ContractModel):
    """Safe paragraph-level extraction state persisted in canonical audits."""

    paragraph_id: str
    file: str
    line_start: PositiveInt
    line_end: PositiveInt
    status: Literal["complete", "partial", "needs_human"]
    window_count: NonNegativeInt
    completed_window_count: NonNegativeInt
    accepted_claim_ids: tuple[str, ...] = ()
    reason_codes: tuple[ParagraphMiningReasonCode, ...] = ()
    protected_occurrences_total: NonNegativeInt = 0
    protected_occurrences_covered: NonNegativeInt = 0

    _validate_id = field_validator("paragraph_id")(_identifier)
    _validate_file = field_validator("file")(_repo_path)

    @field_validator("accepted_claim_ids")
    @classmethod
    def accepted_claim_ids_are_unique(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        for value in values:
            _identifier(value)
        if len(values) != len(set(values)):
            raise ValueError("accepted claim IDs must be unique")
        return values

    @field_validator("reason_codes")
    @classmethod
    def reason_codes_are_unique(
        cls, values: tuple[ParagraphMiningReasonCode, ...]
    ) -> tuple[ParagraphMiningReasonCode, ...]:
        if len(values) != len(set(values)):
            raise ValueError("paragraph mining reason codes must be unique")
        return values

    @model_validator(mode="after")
    def outcome_is_consistent(self) -> Self:
        if self.line_end < self.line_start:
            raise ValueError("paragraph mining lines must be ordered")
        if self.completed_window_count > self.window_count:
            raise ValueError("completed window count cannot exceed window count")
        if self.protected_occurrences_covered > self.protected_occurrences_total:
            raise ValueError("protected coverage cannot exceed its total")
        if self.status == "complete":
            if self.completed_window_count != self.window_count:
                raise ValueError("complete paragraph requires all windows to complete")
            if self.reason_codes:
                raise ValueError("complete paragraph cannot carry reason codes")
            if self.protected_occurrences_covered != self.protected_occurrences_total:
                raise ValueError("complete paragraph requires full protected coverage")
        elif not self.reason_codes:
            raise ValueError("incomplete paragraph requires an allowlisted reason")
        if self.status == "partial" and not self.accepted_claim_ids:
            raise ValueError("partial paragraph requires an accepted claim")
        return self


class ClaimOperationalOutcome(ContractModel):
    """Safe claim-level status when no factual relation was completed."""

    claim_id: str
    file: str
    line_start: PositiveInt
    line_end: PositiveInt
    status: ClaimOperationalStatus
    reason_code: ClaimOperationalReasonCode

    _validate_claim_id = field_validator("claim_id")(_identifier)
    _validate_file = field_validator("file")(_repo_path)

    @model_validator(mode="after")
    def status_matches_reason(self) -> Self:
        allowed = {
            "agent_error": {"challenger_error", "judge_error", "scout_error"},
            "budget_exhausted": {"budget_exhausted"},
            "needs_human": {
                "discovery_unavailable",
                "human_review_requested",
            },
        }
        if self.reason_code not in allowed[self.status]:
            raise ValueError("claim operational status and reason do not match")
        if self.line_end < self.line_start:
            raise ValueError("claim operational lines must be ordered")
        return self


class AuditArtifact(ContractModel):
    schema_version: Literal[1] = 1
    run: RunMetadata
    effective_config: EffectiveConfig
    documents: tuple[ParsedDocument, ...] = ()
    change_selections: tuple[ChangeSelection, ...] = ()
    claims: tuple[AtomicClaim, ...] = ()
    verdicts: tuple[Verdict, ...] = ()
    sources: tuple[SourceMetadata, ...] = ()
    diagnostics: tuple[ParseDiagnostic, ...] = ()
    paragraph_mining_outcomes: tuple[ParagraphMiningAuditOutcome, ...] = ()
    claim_operational_outcomes: tuple[ClaimOperationalOutcome, ...] = ()

    @field_validator("paragraph_mining_outcomes")
    @classmethod
    def mining_outcomes_are_unique_and_ordered(
        cls, values: tuple[ParagraphMiningAuditOutcome, ...]
    ) -> tuple[ParagraphMiningAuditOutcome, ...]:
        keys = [(value.file, value.paragraph_id) for value in values]
        if len(keys) != len(set(keys)):
            raise ValueError("paragraph mining outcomes must be unique")
        return tuple(
            sorted(
                values,
                key=lambda value: (
                    value.file,
                    value.line_start,
                    value.line_end,
                    value.paragraph_id,
                ),
            )
        )

    @model_validator(mode="after")
    def mining_claim_references_are_local(self) -> Self:
        claims = {claim.claim_id: claim for claim in self.claims}
        accepted_by: dict[str, tuple[str, str]] = {}
        for outcome in self.paragraph_mining_outcomes:
            for claim_id in outcome.accepted_claim_ids:
                claim = claims.get(claim_id)
                if claim is None:
                    raise ValueError(
                        "paragraph mining outcome references unknown claim"
                    )
                if claim.file != outcome.file or not (
                    outcome.line_start <= claim.line_start <= claim.line_end
                    and outcome.line_start <= claim.line_end <= outcome.line_end
                ):
                    raise ValueError(
                        "accepted claim must remain inside its paragraph location"
                    )
                if claim_id in accepted_by:
                    raise ValueError(
                        "accepted claim cannot belong to multiple paragraphs"
                    )
                accepted_by[claim_id] = (outcome.file, outcome.paragraph_id)
        return self

    @field_validator("claim_operational_outcomes")
    @classmethod
    def operational_outcomes_are_unique_and_ordered(
        cls, values: tuple[ClaimOperationalOutcome, ...]
    ) -> tuple[ClaimOperationalOutcome, ...]:
        claim_ids = [value.claim_id for value in values]
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("claim operational outcomes must be unique")
        return tuple(
            sorted(
                values,
                key=lambda value: (
                    value.file,
                    value.line_start,
                    value.line_end,
                    value.claim_id,
                ),
            )
        )

    @model_validator(mode="after")
    def operational_claim_references_are_local(self) -> Self:
        claims = {claim.claim_id: claim for claim in self.claims}
        verdict_ids = {verdict.claim_id for verdict in self.verdicts}
        for outcome in self.claim_operational_outcomes:
            claim = claims.get(outcome.claim_id)
            if claim is None:
                raise ValueError("claim operational outcome references unknown claim")
            if outcome.claim_id in verdict_ids:
                raise ValueError(
                    "operational claim cannot also carry a completed verdict"
                )
            if (
                claim.file != outcome.file
                or claim.line_start != outcome.line_start
                or claim.line_end != outcome.line_end
            ):
                raise ValueError(
                    "claim operational outcome must use its trusted claim location"
                )
        return self


__all__ = [
    "AtomicClaim",
    "AuditArtifact",
    "BudgetConfig",
    "CacheConfig",
    "ChangeSelection",
    "ChangeStatus",
    "Checkability",
    "CitationDefinition",
    "CitationKind",
    "CitationOccurrence",
    "ClaimOperationalOutcome",
    "ClaimOperationalReasonCode",
    "ClaimOperationalStatus",
    "ContractModel",
    "CorroborationStatus",
    "DefinitionKind",
    "DiagnosticSeverity",
    "DiffHunk",
    "EffectiveConfig",
    "EvidenceSpan",
    "FileChange",
    "GitDiff",
    "LineRange",
    "MarkdownParagraph",
    "ModelConfig",
    "ModelTextSourceKind",
    "ModelTextSourceSegment",
    "ParagraphKind",
    "ParagraphMiningAuditOutcome",
    "ParagraphMiningReasonCode",
    "ParseDiagnostic",
    "ParsedDocument",
    "ParsedDocumentWithSourceMap",
    "PathsConfig",
    "PolicyConfig",
    "PolicyDecision",
    "PolicyLabel",
    "Relation",
    "RunMetadata",
    "RunPaths",
    "Severity",
    "SourceMetadata",
    "SourceSpan",
    "SuppressionDecision",
    "SuppressionDirective",
    "SuppressionKind",
    "SuppressionScope",
    "TargetKind",
    "TrustedModelVisibleParagraph",
    "Verdict",
]
