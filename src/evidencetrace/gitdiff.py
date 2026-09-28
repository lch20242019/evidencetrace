"""Deterministic Git diff parsing and Markdown source selection.

This module deliberately stops at Phase 1 boundaries: it identifies lines that
actually changed on the new side of a unified diff and maps those lines to
already-parsed Markdown objects.  It does not grow context around a change.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from evidencetrace.models import (
    ChangeSelection,
    ChangeStatus,
    DiagnosticSeverity,
    DiffHunk,
    FileChange,
    GitDiff,
    LineRange,
    ParseDiagnostic,
    ParsedDocument,
    SourceSpan,
)

_HUNK_HEADER_RE = re.compile(
    r"^@@ -(?P<old_start>\d+)(?:,(?P<old_count>\d+))? "
    r"\+(?P<new_start>\d+)(?:,(?P<new_count>\d+))? @@(?: .*)?$"
)
_FULL_SHA_RE = re.compile(r"^[0-9a-fA-F]{40}(?:[0-9a-fA-F]{24})?$")
_GIT_ESCAPES = {
    "a": b"\a",
    "b": b"\b",
    "f": b"\f",
    "n": b"\n",
    "r": b"\r",
    "t": b"\t",
    "v": b"\v",
    "\\": b"\\",
    '"': b'"',
}


class GitDiffError(RuntimeError):
    """Raised when Git cannot resolve or collect the requested diff."""


@dataclass
class _HunkState:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    old_cursor: int = field(init=False)
    new_cursor: int = field(init=False)
    added_lines: list[int] = field(default_factory=list)
    saw_deletion: bool = False

    def __post_init__(self) -> None:
        self.old_cursor = self.old_start
        self.new_cursor = self.new_start

    def consume(self, line: str) -> None:
        """Consume one unified-diff body line."""

        if line.startswith("+"):
            # A valid added line is always one-based.  The max is defensive for
            # malformed third-party patches and never changes valid Git output.
            self.added_lines.append(max(1, self.new_cursor))
            self.new_cursor += 1
        elif line.startswith("-"):
            self.saw_deletion = True
            self.old_cursor += 1
        elif line.startswith(" "):
            self.old_cursor += 1
            self.new_cursor += 1

    def freeze(self) -> DiffHunk:
        added_ranges = _collapse_line_numbers(self.added_lines)
        deletion_anchor = None
        if self.saw_deletion and not added_ranges:
            # With a zero-length new range, Git's new_start points at the
            # surviving line immediately before the deletion (or zero when the
            # deletion is at the beginning).  Clamp zero so source locations
            # remain human-readable and one-based.
            deletion_anchor = max(1, self.new_start)
        return DiffHunk(
            old_start=self.old_start,
            old_count=self.old_count,
            new_start=self.new_start,
            new_count=self.new_count,
            added_ranges=added_ranges,
            deletion_anchor=deletion_anchor,
        )


@dataclass
class _FileState:
    old_path: str | None = None
    new_path: str | None = None
    status: ChangeStatus = ChangeStatus.MODIFIED
    hunks: list[DiffHunk] = field(default_factory=list)
    binary: bool = False


def parse_unified_diff(
    patch: str, *, base_revision: str | None = None
) -> GitDiff:
    """Parse a Git-style unified diff into deterministic typed contracts.

    Only true ``+`` body lines are recorded as additions.  Metadata headers and
    the ``No newline at end of file`` marker never affect line accounting.
    Combined and binary diffs are reported explicitly instead of being guessed.
    """

    files: list[FileChange] = []
    diagnostics: list[ParseDiagnostic] = []
    current_file: _FileState | None = None
    current_hunk: _HunkState | None = None
    skipping_combined = False

    def finish_hunk() -> None:
        nonlocal current_hunk
        if current_hunk is not None and current_file is not None:
            current_file.hunks.append(current_hunk.freeze())
        current_hunk = None

    def finish_file() -> None:
        nonlocal current_file
        finish_hunk()
        if current_file is None:
            return

        status = _final_status(current_file)
        old_path = current_file.old_path
        new_path = current_file.new_path
        if status == ChangeStatus.ADDED:
            old_path = None
        elif status == ChangeStatus.DELETED:
            new_path = None

        display_path = new_path or old_path or "<unknown>"
        if current_file.binary:
            diagnostics.append(
                ParseDiagnostic(
                    code="binary_diff",
                    message=(
                        f"Binary diff for {display_path!r} has no parseable text hunks."
                    ),
                    severity=DiagnosticSeverity.NOTICE,
                )
            )

        if old_path is None and new_path is None:
            diagnostics.append(
                ParseDiagnostic(
                    code="unparsed_diff_path",
                    message=(
                        "A diff file section had no parseable "
                        "project-relative path."
                    ),
                    severity=DiagnosticSeverity.WARNING,
                )
            )
            current_file = None
            return

        try:
            files.append(
                FileChange(
                    status=status,
                    old_path=old_path,
                    new_path=new_path,
                    hunks=tuple(current_file.hunks),
                )
            )
        except ValueError as exc:
            diagnostics.append(
                ParseDiagnostic(
                    code="invalid_diff_path",
                    message=f"Could not represent diff path {display_path!r}: {exc}",
                    severity=DiagnosticSeverity.WARNING,
                )
            )
        current_file = None

    for line in patch.splitlines():
        if line.startswith(("diff --cc ", "diff --combined ")):
            finish_file()
            combined_path = _decode_git_path(line.split(" ", 2)[-1])
            diagnostics.append(
                ParseDiagnostic(
                    code="combined_diff",
                    message=(
                        f"Combined diff for {combined_path!r} is unsupported; "
                        "inspect each parent separately."
                    ),
                    severity=DiagnosticSeverity.WARNING,
                )
            )
            skipping_combined = True
            continue

        if line.startswith("diff --git "):
            finish_file()
            old_path, new_path = _parse_diff_git_header(line)
            current_file = _FileState(old_path=old_path, new_path=new_path)
            skipping_combined = False
            continue

        if skipping_combined:
            continue

        if current_hunk is not None:
            if line.startswith("@@ "):
                finish_hunk()
                current_hunk = _parse_hunk_header(line, diagnostics)
            elif line == r"\ No newline at end of file":
                continue
            elif line.startswith(("+", "-", " ")):
                current_hunk.consume(line)
            # Blank separators and unfamiliar metadata do not move cursors.
            continue

        if line.startswith("--- "):
            if current_file is None:
                current_file = _FileState()
            current_file.old_path = _parse_marker_path(line[4:], "a/")
            continue

        if line.startswith("+++ "):
            if current_file is None:
                current_file = _FileState()
            current_file.new_path = _parse_marker_path(line[4:], "b/")
            continue

        if line.startswith("@@ "):
            if current_file is None:
                current_file = _FileState()
            current_hunk = _parse_hunk_header(line, diagnostics)
            continue

        if current_file is None:
            continue

        if line.startswith("new file mode "):
            current_file.status = ChangeStatus.ADDED
            current_file.old_path = None
        elif line.startswith("deleted file mode "):
            current_file.status = ChangeStatus.DELETED
            current_file.new_path = None
        elif line.startswith("rename from "):
            current_file.status = ChangeStatus.RENAMED
            current_file.old_path = _decode_git_path(line[len("rename from ") :])
        elif line.startswith("rename to "):
            current_file.status = ChangeStatus.RENAMED
            current_file.new_path = _decode_git_path(line[len("rename to ") :])
        elif line.startswith("copy from "):
            current_file.status = ChangeStatus.COPIED
            current_file.old_path = _decode_git_path(line[len("copy from ") :])
        elif line.startswith("copy to "):
            current_file.status = ChangeStatus.COPIED
            current_file.new_path = _decode_git_path(line[len("copy to ") :])
        elif line.startswith("Binary files ") or line == "GIT binary patch":
            current_file.binary = True
        elif line.startswith("@@@"):
            diagnostics.append(
                ParseDiagnostic(
                    code="combined_diff",
                    message=(
                        "A combined hunk is unsupported; "
                        "inspect each parent separately."
                    ),
                    severity=DiagnosticSeverity.WARNING,
                )
            )

    finish_file()
    return GitDiff(
        base_revision=base_revision,
        files=tuple(files),
        diagnostics=tuple(diagnostics),
    )


def collect_git_diff(
    repo: str | Path,
    changed_from: str,
    *,
    paths: Sequence[str | Path] = (),
) -> GitDiff:
    """Resolve ``changed_from`` and collect its zero-context working-tree diff.

    Git is invoked with an argv sequence and never through a shell.  The base
    ref is resolved to an immutable commit SHA before it is passed to ``diff``.
    """

    repo_path = Path(repo).resolve()
    revision = _run_git(
        repo_path,
        ("rev-parse", "--verify", "--end-of-options", f"{changed_from}^{{commit}}"),
        operation=f"resolve revision {changed_from!r}",
    ).strip()
    if not _FULL_SHA_RE.fullmatch(revision):
        raise GitDiffError(f"Git returned an invalid commit SHA: {revision!r}")

    diff_args = [
        "diff",
        "--no-ext-diff",
        "--no-color",
        "--find-renames",
        "--unified=0",
        revision,
        "--",
    ]
    diff_args.extend(str(path) for path in paths)
    patch = _run_git(
        repo_path,
        tuple(diff_args),
        operation=f"collect diff from {revision}",
    )
    return parse_unified_diff(patch, base_revision=revision.lower())


def select_changed_paragraphs(
    document: ParsedDocument, change: FileChange
) -> ChangeSelection:
    """Select parsed objects whose own source spans intersect changed lines.

    This intentionally performs no neighboring-paragraph or heading expansion.
    A citation or definition is selected only when its own source span changes,
    not merely because its containing paragraph was selected.  Pure-deletion
    anchors remain hunk metadata and do not select adjacent new-side content.
    """

    target_path = change.new_path
    if target_path is None:
        if change.old_path != document.path:
            raise ValueError("deleted file change does not match parsed document path")
        # A deleted file has no new-side document to map.  Returning an empty
        # selection is safer than applying new-side anchors to old-side lines.
        return ChangeSelection(path=document.path)
    if target_path != document.path:
        raise ValueError(
            f"file change targets {target_path!r}, not document {document.path!r}"
        )

    changed_ranges = _changed_ranges(change)
    paragraph_ids = tuple(
        paragraph.paragraph_id
        for paragraph in document.paragraphs
        if _span_intersects_ranges(paragraph.source, changed_ranges)
    )
    citation_ids = tuple(
        citation.citation_id
        for citation in document.citations
        if _span_intersects_ranges(citation.source, changed_ranges)
    )
    definition_ids = tuple(
        definition.definition_id
        for definition in document.definitions
        if _span_intersects_ranges(definition.source, changed_ranges)
    )
    suppression_ids = tuple(
        suppression.suppression_id
        for suppression in document.suppressions
        if _span_intersects_ranges(suppression.directive_source, changed_ranges)
        or _span_intersects_ranges(suppression.target_source, changed_ranges)
    )
    return ChangeSelection(
        path=document.path,
        changed_ranges=changed_ranges,
        paragraph_ids=paragraph_ids,
        changed_citation_ids=citation_ids,
        changed_definition_ids=definition_ids,
        changed_suppression_ids=suppression_ids,
    )


def _run_git(repo: Path, args: Sequence[str], *, operation: str) -> str:
    completed = subprocess.run(
        ("git", "-C", str(repo), *args),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="surrogateescape",
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "unknown error"
        raise GitDiffError(f"Could not {operation}: {detail}")
    return completed.stdout


def _parse_hunk_header(
    line: str, diagnostics: list[ParseDiagnostic]
) -> _HunkState | None:
    match = _HUNK_HEADER_RE.match(line)
    if match is None:
        diagnostics.append(
            ParseDiagnostic(
                code="malformed_hunk_header",
                message=f"Could not parse unified diff hunk header: {line!r}",
                severity=DiagnosticSeverity.WARNING,
            )
        )
        return None
    return _HunkState(
        old_start=int(match.group("old_start")),
        old_count=int(match.group("old_count") or "1"),
        new_start=int(match.group("new_start")),
        new_count=int(match.group("new_count") or "1"),
    )


def _final_status(state: _FileState) -> ChangeStatus:
    if state.status in {
        ChangeStatus.ADDED,
        ChangeStatus.DELETED,
        ChangeStatus.RENAMED,
        ChangeStatus.COPIED,
    }:
        return state.status
    if state.old_path is None and state.new_path is not None:
        return ChangeStatus.ADDED
    if state.new_path is None and state.old_path is not None:
        return ChangeStatus.DELETED
    return ChangeStatus.MODIFIED


def _collapse_line_numbers(lines: Sequence[int]) -> tuple[LineRange, ...]:
    if not lines:
        return ()
    ranges: list[LineRange] = []
    start = previous = lines[0]
    for line in lines[1:]:
        if line == previous + 1:
            previous = line
            continue
        ranges.append(LineRange(start=start, end=previous))
        start = previous = line
    ranges.append(LineRange(start=start, end=previous))
    return tuple(ranges)


def _changed_ranges(change: FileChange) -> tuple[LineRange, ...]:
    ranges: list[LineRange] = []
    for hunk in change.hunks:
        ranges.extend(hunk.added_ranges)
    return _merge_ranges(ranges)


def _merge_ranges(ranges: Sequence[LineRange]) -> tuple[LineRange, ...]:
    if not ranges:
        return ()
    ordered = sorted(ranges, key=lambda item: (item.start, item.end))
    merged: list[LineRange] = [ordered[0]]
    for current in ordered[1:]:
        previous = merged[-1]
        if current.start <= previous.end + 1:
            merged[-1] = LineRange(
                start=previous.start, end=max(previous.end, current.end)
            )
        else:
            merged.append(current)
    return tuple(merged)


def _span_intersects_ranges(
    span: SourceSpan, ranges: Sequence[LineRange]
) -> bool:
    return any(
        changed.start <= span.line_end and span.line_start <= changed.end
        for changed in ranges
    )


def _parse_marker_path(value: str, prefix: str) -> str | None:
    decoded = _decode_git_path(value)
    if decoded == "/dev/null":
        return None
    return decoded[len(prefix) :] if decoded.startswith(prefix) else decoded


def _parse_diff_git_header(line: str) -> tuple[str | None, str | None]:
    value = line[len("diff --git ") :]
    first, remainder = _consume_header_token(value)
    if first is not None and remainder:
        second, trailing = _consume_header_token(remainder.lstrip())
        if second is not None and not trailing.strip():
            return _strip_side_prefix(first, "a/"), _strip_side_prefix(second, "b/")

    # Unquoted Git headers containing spaces are intrinsically ambiguous.  For
    # the common unchanged-path case, choose the delimiter whose two sides are
    # identical.  The unambiguous ---/+++ or rename metadata overrides this
    # fallback as soon as it is encountered.
    if value.startswith("a/"):
        positions = [match.start() for match in re.finditer(r" b/", value)]
        for position in positions:
            old_path = value[2:position]
            new_path = value[position + 3 :]
            if old_path == new_path:
                return old_path, new_path
        if positions:
            position = positions[0]
            return value[2:position], value[position + 3 :]
    return None, None


def _consume_header_token(value: str) -> tuple[str | None, str]:
    if not value:
        return None, ""
    if value[0] != '"':
        boundary = value.find(" ")
        if boundary < 0:
            return _decode_git_path(value), ""
        return _decode_git_path(value[:boundary]), value[boundary + 1 :]

    escaped = False
    for index in range(1, len(value)):
        character = value[index]
        if character == '"' and not escaped:
            return _decode_git_path(value[: index + 1]), value[index + 1 :]
        if character == "\\" and not escaped:
            escaped = True
        else:
            escaped = False
    return None, value


def _strip_side_prefix(path: str, prefix: str) -> str:
    return path[len(prefix) :] if path.startswith(prefix) else path


def _decode_git_path(value: str) -> str:
    value = value.strip()
    if len(value) < 2 or not (value.startswith('"') and value.endswith('"')):
        return value

    payload = value[1:-1]
    decoded = bytearray()
    index = 0
    while index < len(payload):
        character = payload[index]
        if character != "\\":
            decoded.extend(character.encode("utf-8"))
            index += 1
            continue

        index += 1
        if index >= len(payload):
            decoded.extend(b"\\")
            break
        escaped = payload[index]
        if escaped in "01234567":
            end = index + 1
            while end < min(index + 3, len(payload)) and payload[end] in "01234567":
                end += 1
            decoded.append(int(payload[index:end], 8))
            index = end
            continue
        replacement = _GIT_ESCAPES.get(escaped)
        if replacement is None:
            decoded.extend(escaped.encode("utf-8"))
        else:
            decoded.extend(replacement)
        index += 1
    return decoded.decode("utf-8", errors="surrogateescape")


__all__ = [
    "GitDiffError",
    "collect_git_diff",
    "parse_unified_diff",
    "select_changed_paragraphs",
]
