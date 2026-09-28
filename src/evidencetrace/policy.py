"""Deterministic Phase 1 policy and suppression decisions."""

from __future__ import annotations

import re
from collections.abc import Sequence
from functools import lru_cache

from evidencetrace.models import (
    CorroborationStatus,
    MarkdownParagraph,
    ParsedDocument,
    PathsConfig,
    PolicyConfig,
    PolicyDecision,
    PolicyLabel,
    Relation,
    Severity,
    SourceSpan,
    SuppressionDecision,
    SuppressionKind,
)

_SEVERITY_RANK = {
    Severity.PASS: 0,
    Severity.NOTICE: 1,
    Severity.WARNING: 2,
    Severity.ERROR: 3,
}


def _label_value(label: Relation | CorroborationStatus | PolicyLabel) -> str:
    return label.value


def _severity_for_label(
    label: Relation | CorroborationStatus,
    policy: PolicyConfig,
) -> Severity:
    value = _label_value(label)
    if value in {_label_value(item) for item in policy.fail_on}:
        return Severity.ERROR
    if value in {_label_value(item) for item in policy.warn_on}:
        return Severity.WARNING
    if value in {_label_value(item) for item in policy.notice_on}:
        return Severity.NOTICE
    return Severity.PASS


def relation_severity(
    relation: Relation,
    corroboration: CorroborationStatus = CorroborationStatus.NOT_REQUESTED,
    policy: PolicyConfig | None = None,
) -> Severity:
    """Map both verdict dimensions and return the more severe result."""

    effective_policy = policy or PolicyConfig()
    candidates = (
        _severity_for_label(relation, effective_policy),
        _severity_for_label(corroboration, effective_policy),
    )
    return max(candidates, key=_SEVERITY_RANK.__getitem__)


@lru_cache(maxsize=256)
def _compile_glob(pattern: str) -> re.Pattern[str]:
    """Compile a small, anchored POSIX glob with true ``**`` semantics."""

    pieces = ["^"]
    index = 0
    while index < len(pattern):
        char = pattern[index]
        if char == "*":
            if index + 1 < len(pattern) and pattern[index + 1] == "*":
                index += 2
                while index < len(pattern) and pattern[index] == "*":
                    index += 1
                if index < len(pattern) and pattern[index] == "/":
                    pieces.append("(?:.*/)?")
                    index += 1
                else:
                    pieces.append(".*")
                continue
            pieces.append("[^/]*")
        elif char == "?":
            pieces.append("[^/]")
        elif char == "[":
            closing = pattern.find("]", index + 1)
            if closing == -1:
                pieces.append(r"\[")
            else:
                content = pattern[index + 1 : closing]
                negate = content.startswith(("!", "^"))
                if negate:
                    content = content[1:]
                escaped = re.escape(content).replace(r"\-", "-")
                pieces.append(f"[{'^' if negate else ''}{escaped}]")
                index = closing
        else:
            pieces.append(re.escape(char))
        index += 1
    pieces.append("$")
    return re.compile("".join(pieces))


def path_matches(path: str, pattern: str) -> bool:
    """Match a repository-relative path using deterministic POSIX semantics."""

    normalized = path.replace("\\", "/")
    return _compile_glob(pattern).fullmatch(normalized) is not None


def path_is_in_scope(path: str, config: PathsConfig) -> bool:
    """Apply includes first and let excludes take precedence."""

    included = not config.include or any(path_matches(path, item) for item in config.include)
    excluded = any(path_matches(path, item) for item in config.exclude)
    return included and not excluded


def _spans_overlap(left: SourceSpan, right: SourceSpan) -> bool:
    if left.file != right.file:
        return False
    if left.offset_start == left.offset_end or right.offset_start == right.offset_end:
        return left.line_start <= right.line_end and right.line_start <= left.line_end
    return left.offset_start < right.offset_end and right.offset_start < left.offset_end


def _paragraph_for_span(
    source: SourceSpan,
    document: ParsedDocument,
) -> MarkdownParagraph | None:
    return next(
        (item for item in document.paragraphs if _spans_overlap(source, item.source)),
        None,
    )


def suppression_for(
    source: SourceSpan | None,
    claim_type: str | None,
    document: ParsedDocument | None,
    policy: PolicyConfig,
) -> SuppressionDecision | None:
    """Find an auditable directive or claim-type suppression.

    Explicit document directives take precedence over configuration-level
    claim-type ignores because they carry a user-authored reason and location.
    """

    if source is not None and document is not None:
        paragraph = _paragraph_for_span(source, document)
        candidate_ids = set(paragraph.suppression_ids) if paragraph else set()
        for directive in document.suppressions:
            attached = not candidate_ids or directive.suppression_id in candidate_ids
            if attached and _spans_overlap(source, directive.target_source):
                return SuppressionDecision(
                    kind=SuppressionKind.DIRECTIVE,
                    reason=directive.reason,
                    suppression_id=directive.suppression_id,
                    source=directive.directive_source,
                )
    if claim_type is not None and claim_type in policy.ignore_claim_types:
        return SuppressionDecision(
            kind=SuppressionKind.CLAIM_TYPE,
            reason=f"claim type ignored by policy: {claim_type}",
        )
    return None


def decide_policy(
    relation: Relation,
    corroboration: CorroborationStatus = CorroborationStatus.NOT_REQUESTED,
    *,
    policy: PolicyConfig | None = None,
    source: SourceSpan | None = None,
    claim_type: str | None = None,
    document: ParsedDocument | None = None,
) -> PolicyDecision:
    """Create one deterministic, suppression-aware policy decision."""

    effective_policy = policy or PolicyConfig()
    raw = relation_severity(relation, corroboration, effective_policy)
    suppression = suppression_for(source, claim_type, document, effective_policy)
    return PolicyDecision(
        relation=relation,
        corroboration=corroboration,
        raw_severity=raw,
        effective_severity=None if suppression else raw,
        suppression=suppression,
    )


def should_fail(
    decisions: Sequence[PolicyDecision],
    policy: PolicyConfig | None = None,
) -> bool:
    """Return whether any unsuppressed decision reaches the fail threshold."""

    effective_policy = policy or PolicyConfig()
    threshold = _SEVERITY_RANK[effective_policy.fail_threshold]
    return any(
        decision.effective_severity is not None
        and _SEVERITY_RANK[decision.effective_severity] >= threshold
        for decision in decisions
    )


def exit_code(
    decisions: Sequence[PolicyDecision],
    policy: PolicyConfig | None = None,
) -> int:
    """Return the Phase 1 policy exit code (zero or one)."""

    return 1 if should_fail(decisions, policy) else 0


__all__ = [
    "decide_policy",
    "exit_code",
    "path_is_in_scope",
    "path_matches",
    "relation_severity",
    "should_fail",
    "suppression_for",
]
