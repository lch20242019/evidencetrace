from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from evidencetrace.gitdiff import (
    GitDiffError,
    collect_git_diff,
    parse_unified_diff,
    select_changed_paragraphs,
)
from evidencetrace.models import (
    ChangeStatus,
    CitationDefinition,
    CitationKind,
    CitationOccurrence,
    DefinitionKind,
    DiffHunk,
    FileChange,
    LineRange,
    MarkdownParagraph,
    ParagraphKind,
    ParsedDocument,
    SourceSpan,
    SuppressionDirective,
    SuppressionScope,
    TargetKind,
)


def test_parse_unified_diff_supports_file_statuses_and_multiple_hunks() -> None:
    patch = r'''diff --git a/docs/guide.md b/docs/guide.md
index 1111111..2222222 100644
--- a/docs/guide.md
+++ b/docs/guide.md
@@ -2 +2,2 @@
-old
+new
+next
@@ -8 +9 @@
-old ending
+new ending
\ No newline at end of file
diff --git "a/docs/\303\274ber view.md" "b/docs/\303\274ber view.md"
new file mode 100644
index 0000000..3333333
--- /dev/null
+++ "b/docs/\303\274ber view.md"
@@ -0,0 +1 @@
+hello
diff --git a/docs/remove.md b/docs/remove.md
deleted file mode 100644
index 4444444..0000000
--- a/docs/remove.md
+++ /dev/null
@@ -1,2 +0,0 @@
-one
-two
diff --git a/docs/old name.md b/docs/new name.md
similarity index 100%
rename from docs/old name.md
rename to docs/new name.md
'''

    result = parse_unified_diff(patch, base_revision="a" * 40)

    assert result.base_revision == "a" * 40
    assert [change.status for change in result.files] == [
        ChangeStatus.MODIFIED,
        ChangeStatus.ADDED,
        ChangeStatus.DELETED,
        ChangeStatus.RENAMED,
    ]
    modified, added, deleted, renamed = result.files
    assert modified.old_path == modified.new_path == "docs/guide.md"
    assert len(modified.hunks) == 2
    assert modified.hunks[0].added_ranges == (LineRange(start=2, end=3),)
    assert modified.hunks[1].added_ranges == (LineRange(start=9, end=9),)
    assert added.old_path is None
    assert added.new_path == "docs/über view.md"
    assert deleted.old_path == "docs/remove.md"
    assert deleted.new_path is None
    assert deleted.hunks[0].deletion_anchor == 1
    assert renamed.old_path == "docs/old name.md"
    assert renamed.new_path == "docs/new name.md"
    assert result.diagnostics == ()


def test_parse_unified_diff_reports_combined_and_binary_content() -> None:
    patch = """diff --cc docs/merge.md
index 1111111,2222222..3333333
--- a/docs/merge.md
+++ b/docs/merge.md
@@@ -1,1 -1,1 +1,1 @@@
diff --git a/assets/blob.bin b/assets/blob.bin
index 1111111..2222222 100644
Binary files a/assets/blob.bin and b/assets/blob.bin differ
"""

    result = parse_unified_diff(patch)

    assert len(result.files) == 1
    assert result.files[0].old_path == "assets/blob.bin"
    assert result.files[0].new_path == "assets/blob.bin"
    assert {diagnostic.code for diagnostic in result.diagnostics} == {
        "binary_diff",
        "combined_diff",
    }


def test_pure_deletion_uses_a_new_side_anchor() -> None:
    patch = """diff --git a/docs/guide.md b/docs/guide.md
--- a/docs/guide.md
+++ b/docs/guide.md
@@ -4 +3,0 @@
-removed
"""

    result = parse_unified_diff(patch)

    hunk = result.files[0].hunks[0]
    assert hunk.added_ranges == ()
    assert hunk.deletion_anchor == 3


def test_select_changed_objects_requires_direct_source_intersection() -> None:
    path = "docs/guide.md"
    selected_citation = CitationOccurrence(
        citation_id="cit_selected",
        kind=CitationKind.INLINE_LINK,
        raw="[source](https://example.test)",
        label="source",
        destination="https://example.test",
        target_kind=TargetKind.WEB,
        resolved_urls=("https://example.test",),
        source=_span(path, 3),
        resolved=True,
    )
    same_paragraph_citation = CitationOccurrence(
        citation_id="cit_not_selected",
        kind=CitationKind.INLINE_LINK,
        raw="[other](https://other.test)",
        label="other",
        destination="https://other.test",
        target_kind=TargetKind.WEB,
        resolved_urls=("https://other.test",),
        source=_span(path, 4),
        resolved=True,
    )
    document = ParsedDocument(
        path=path,
        content_sha256="0" * 64,
        line_count=10,
        paragraphs=(
            MarkdownParagraph(
                paragraph_id="p_selected",
                kind=ParagraphKind.PARAGRAPH,
                source=_span(path, 2, 5),
                raw_text="selected",
                plain_text="selected",
                citation_ids=("cit_selected", "cit_not_selected"),
                suppression_ids=("sup_selected",),
            ),
            MarkdownParagraph(
                paragraph_id="p_unchanged",
                kind=ParagraphKind.PARAGRAPH,
                source=_span(path, 6, 7),
                raw_text="unchanged",
                plain_text="unchanged",
            ),
            MarkdownParagraph(
                paragraph_id="p_definition",
                kind=ParagraphKind.PARAGRAPH,
                source=_span(path, 8),
                raw_text="definition",
                plain_text="definition",
            ),
        ),
        citations=(selected_citation, same_paragraph_citation),
        definitions=(
            CitationDefinition(
                definition_id="def_selected",
                kind=DefinitionKind.REFERENCE,
                normalized_key="source",
                raw_text="[source]: https://example.test",
                destinations=("https://example.test",),
                source=_span(path, 8),
            ),
            CitationDefinition(
                definition_id="def_not_selected",
                kind=DefinitionKind.REFERENCE,
                normalized_key="other",
                raw_text="[other]: https://other.test",
                destinations=("https://other.test",),
                source=_span(path, 9),
            ),
        ),
        suppressions=(
            SuppressionDirective(
                suppression_id="sup_selected",
                reason="fixture",
                scope=SuppressionScope.PARAGRAPH,
                directive_source=_span(path, 1),
                target_source=_span(path, 2, 5),
            ),
            SuppressionDirective(
                suppression_id="sup_not_selected",
                reason="fixture",
                scope=SuppressionScope.PARAGRAPH,
                directive_source=_span(path, 6),
                target_source=_span(path, 7),
            ),
        ),
    )
    change = FileChange(
        status=ChangeStatus.MODIFIED,
        old_path=path,
        new_path=path,
        hunks=(
            DiffHunk(
                old_start=3,
                old_count=1,
                new_start=3,
                new_count=1,
                added_ranges=(LineRange(start=3, end=3),),
            ),
            DiffHunk(
                old_start=8,
                old_count=1,
                new_start=8,
                new_count=1,
                added_ranges=(LineRange(start=8, end=8),),
            ),
        ),
    )

    selection = select_changed_paragraphs(document, change)

    assert selection.changed_ranges == (
        LineRange(start=3, end=3),
        LineRange(start=8, end=8),
    )
    assert selection.paragraph_ids == ("p_selected", "p_definition")
    assert selection.changed_citation_ids == ("cit_selected",)
    assert selection.changed_definition_ids == ("def_selected",)
    assert selection.changed_suppression_ids == ("sup_selected",)


def test_pure_deletion_anchor_does_not_expand_selection_context() -> None:
    path = "docs/guide.md"
    document = ParsedDocument(
        path=path,
        content_sha256="0" * 64,
        line_count=5,
        paragraphs=(
            MarkdownParagraph(
                paragraph_id="p_anchor",
                kind=ParagraphKind.PARAGRAPH,
                source=_span(path, 3),
                raw_text="surviving paragraph",
                plain_text="surviving paragraph",
            ),
        ),
    )
    change = FileChange(
        status=ChangeStatus.MODIFIED,
        old_path=path,
        new_path=path,
        hunks=(
            DiffHunk(
                old_start=4,
                old_count=1,
                new_start=3,
                new_count=0,
                deletion_anchor=3,
            ),
        ),
    )

    selection = select_changed_paragraphs(document, change)

    assert change.hunks[0].deletion_anchor == 3
    assert selection.changed_ranges == ()
    assert selection.paragraph_ids == ()


def test_collect_git_diff_resolves_sha_and_handles_real_repository(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    docs = repo / "docs"
    docs.mkdir(parents=True)
    _git(repo, "init", "--quiet")
    _git(repo, "config", "user.name", "EvidenceTrace Test")
    _git(repo, "config", "user.email", "test@example.invalid")

    (docs / "guide.md").write_text(
        "".join(f"line {number}\n" for number in range(1, 13)), encoding="utf-8"
    )
    (docs / "space ü.md").write_text("original\n", encoding="utf-8")
    (docs / "remove.md").write_text("remove me\n", encoding="utf-8")
    (docs / "old name.md").write_text("rename payload\n" * 5, encoding="utf-8")
    _git(repo, "add", "docs")
    _git(repo, "commit", "--quiet", "-m", "base")
    base_sha = _git(repo, "rev-parse", "HEAD").strip()

    guide_lines = (docs / "guide.md").read_text(encoding="utf-8").splitlines()
    guide_lines[1] = "changed line 2"
    guide_lines[10] = "changed line 11"
    (docs / "guide.md").write_text("\n".join(guide_lines) + "\n", encoding="utf-8")
    (docs / "space ü.md").write_text("changed\n", encoding="utf-8")
    (docs / "remove.md").unlink()
    (docs / "old name.md").rename(docs / "new name.md")
    (docs / "added.md").write_text("new\n", encoding="utf-8")
    _git(repo, "add", "--all")

    result = collect_git_diff(repo, "HEAD")

    assert result.base_revision == base_sha
    by_new_or_old_path = {
        change.new_path or change.old_path: change for change in result.files
    }
    assert by_new_or_old_path["docs/added.md"].status == ChangeStatus.ADDED
    assert by_new_or_old_path["docs/remove.md"].status == ChangeStatus.DELETED
    assert by_new_or_old_path["docs/new name.md"].status == ChangeStatus.RENAMED
    assert by_new_or_old_path["docs/space ü.md"].status == ChangeStatus.MODIFIED
    assert len(by_new_or_old_path["docs/guide.md"].hunks) == 2


def test_collect_git_diff_rejects_an_unknown_revision(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "--quiet")

    with pytest.raises(GitDiffError, match="resolve revision"):
        collect_git_diff(repo, "missing; echo not-a-shell")


def _span(path: str, start: int, end: int | None = None) -> SourceSpan:
    final_line = end if end is not None else start
    return SourceSpan(
        file=path,
        line_start=start,
        line_end=final_line,
        column_start=1,
        column_end=1,
        offset_start=start - 1,
        offset_end=final_line,
    )


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ("git", "-C", str(repo), *args),
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout
