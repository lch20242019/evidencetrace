from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from evidencetrace.audit_models import (
    EVIDENCE_EXACT_MAX_BYTES,
    EVIDENCE_EXACT_MAX_CHARS,
    EVIDENCE_HEADING_MAX_BYTES,
    EVIDENCE_HEADING_MAX_CHARS,
)
from evidencetrace.local_evidence import (
    LOCAL_EVIDENCE_MAX_TOP_K,
    LocalEvidenceError,
    LocalReferenceIndex,
    LocalReferenceInput,
    parse_plain_text_with_source_map,
)
from evidencetrace.models import AtomicClaim, Checkability


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _claim(text: str = "Python 3.13 was released in October 2024.") -> AtomicClaim:
    return AtomicClaim(
        claim_id="c_0001",
        text=text,
        file="target.md",
        line_start=1,
        line_end=1,
        claim_type="versioned_release",
        checkability=Checkability.CHECKABLE,
    )


def _reference(path: Path, reference_id: str) -> LocalReferenceInput:
    return LocalReferenceInput(
        path=path,
        reference_id=reference_id,
        reference_sha256=_sha256(path.read_bytes()),
    )


def test_markdown_reference_returns_bounded_exact_source_and_line(
    tmp_path: Path,
) -> None:
    text = (
        "# Release record\n\n"
        "Python **3.13** was released in October 2024.\n\n"
        "An unrelated paragraph follows.\n"
    )
    path = tmp_path / "reference.md"
    path.write_text(text, encoding="utf-8")

    index = LocalReferenceIndex((_reference(path, "ref_0001.md"),))
    result = index.lookup(_claim())[0]
    evidence = result.evidence[0]

    assert result.reference_id == "ref_0001.md"
    assert result.reference_sha256 == _sha256(text.encode())
    assert evidence.text in text
    assert evidence.chunk.text in text
    assert evidence.text == "Python **3.13** was released in October 2024."
    assert evidence.chunk.heading_path == ("Release record",)
    assert evidence.chunk.locator == "ref_0001.md:3-3"
    assert len(evidence.chunk.text) <= 2_000
    assert result.source.content_hash == result.reference_sha256
    assert str(tmp_path) not in result.model_dump_json()
    assert path.read_text(encoding="utf-8") == text


def test_plain_text_parser_preserves_literal_syntax_lines_and_url_bytes() -> None:
    text = (
        "# This is text, not a heading\n"
        "**Version 2.0** [docs](https://example.test/版本).\n"
        "\n"
        "second paragraph\n"
    )

    parsed = parse_plain_text_with_source_map(text, "ref_0002.txt")

    assert len(parsed.document.paragraphs) == 2
    first = parsed.document.paragraphs[0]
    trusted = parsed.model_visible_paragraphs[0]
    assert first.raw_text == (
        "# This is text, not a heading\n"
        "**Version 2.0** [docs](https://example.test/版本)."
    )
    assert first.plain_text == first.raw_text
    assert first.heading_path == ()
    assert trusted.text == first.raw_text
    assert trusted.heading_path == ()
    assert (first.source.line_start, first.source.line_end) == (1, 2)
    assert parsed.document.paragraphs[1].source.line_start == 4

    citation = parsed.document.citations[0]
    protected = parsed.protected_spans[0]
    assert citation.raw == "https://example.test/版本"
    assert citation.source.line_start == 2
    assert text[protected.char_start : protected.char_end] == citation.raw
    encoded = text.encode("utf-8")
    assert encoded[protected.byte_start : protected.byte_end].decode() == citation.raw
    assert protected.byte_end - protected.byte_start > (
        protected.char_end - protected.char_start
    )
    assert trusted.split_protected_ranges() == (
        (
            citation.source.offset_start - first.source.offset_start,
            citation.source.offset_end - first.source.offset_start,
        ),
    )


def test_markdown_and_txt_do_not_share_markup_semantics(tmp_path: Path) -> None:
    text = (
        "# Heading\n\n"
        "Use **version 2.0** from [the guide](https://example.test/guide).\n"
    )
    markdown = tmp_path / "reference.md"
    plain = tmp_path / "reference.txt"
    markdown.write_text(text, encoding="utf-8")
    plain.write_text(text, encoding="utf-8")

    index = LocalReferenceIndex(
        (
            _reference(markdown, "markdown_ref.md"),
            _reference(plain, "plain_ref.txt"),
        )
    )
    results = {
        result.reference_id: result
        for result in index.lookup(_claim("Use version 2.0 from the guide."))
    }

    markdown_text = results["markdown_ref.md"].evidence[0].text
    plain_text = results["plain_ref.txt"].evidence[0].text
    assert markdown_text == (
        "Use **version 2.0** from [the guide](https://example.test/guide)."
    )
    assert plain_text == markdown_text
    assert results["markdown_ref.md"].evidence[0].chunk.heading_path == ("Heading",)
    assert results["plain_ref.txt"].evidence[0].chunk.heading_path == ()
    heading_results = index.lookup(_claim("# Heading"))
    assert [item.reference_id for item in heading_results] == ["plain_ref.txt"]
    assert heading_results[0].evidence[0].text == "# Heading"
    plain_summary = next(
        item for item in index.references if item.reference_id == "plain_ref.txt"
    )
    assert len(plain_summary.protected_spans) == 1
    assert all(
        paragraph.heading_path == ()
        for paragraph in parse_plain_text_with_source_map(
            text, "plain_ref.txt"
        ).document.paragraphs
    )


def test_lookup_searches_multiple_references_and_caps_each_source(
    tmp_path: Path,
) -> None:
    paths = []
    for number, date in enumerate(("October 7, 2024", "October 8, 2024"), 1):
        path = tmp_path / f"ref-{number}.txt"
        path.write_text(
            "\n\n".join(
                f"Python 3.13 release record {copy}: {date}." for copy in range(8)
            ),
            encoding="utf-8",
        )
        paths.append(path)
    index = LocalReferenceIndex(
        tuple(
            _reference(path, f"ref_{number:04d}.txt")
            for number, path in enumerate(paths, 1)
        )
    )

    results = index.lookup(_claim(), top_k=999)

    assert [item.reference_id for item in results] == [
        "ref_0001.txt",
        "ref_0002.txt",
    ]
    assert all(1 <= len(item.evidence) <= LOCAL_EVIDENCE_MAX_TOP_K for item in results)
    assert all(
        evidence.text in path.read_text(encoding="utf-8")
        for result, path in zip(results, paths, strict=True)
        for evidence in result.evidence
    )
    assert index.lookup(_claim(), top_k=0) == ()


def test_long_local_paragraph_returns_only_bounded_exact_source_spans(
    tmp_path: Path,
) -> None:
    relevant = "Nimbus shipped version 1.2.4."
    private_tail = "FULL-REFERENCE-TAIL-MUST-NOT-PERSIST"
    text = (
        ("填充内容 " * 2_000)
        + relevant
        + (" trailing " * 2_000)
        + private_tail
    )
    path = tmp_path / "long.txt"
    path.write_text(text, encoding="utf-8")

    result = LocalReferenceIndex((_reference(path, "long.txt"),)).lookup(
        _claim("Nimbus shipped version 1.2.3.")
    )[0]

    assert result.evidence
    assert relevant in result.evidence[0].text
    assert private_tail not in result.model_dump_json()
    assert all(
        item.chunk.text == text[item.chunk.char_start : item.chunk.char_end]
        and item.text in item.chunk.text
        and len(item.text) <= EVIDENCE_EXACT_MAX_CHARS
        and len(item.text.encode("utf-8")) <= EVIDENCE_EXACT_MAX_BYTES
        for item in result.evidence
    )


def test_long_markdown_heading_cannot_bypass_evidence_metadata_caps(
    tmp_path: Path,
) -> None:
    heading = "Private-" + ("H" * 10_000)
    text = f"# {heading}\n\nNimbus shipped version 1.2.4.\n"
    path = tmp_path / "heading.md"
    path.write_text(text, encoding="utf-8")

    result = LocalReferenceIndex((_reference(path, "heading.md"),)).lookup(
        _claim("Nimbus shipped version 1.2.3.")
    )[0]
    heading_path = result.evidence[0].chunk.heading_path

    assert heading not in result.model_dump_json()
    assert len(" > ".join(heading_path)) <= EVIDENCE_HEADING_MAX_CHARS
    assert (
        len(" > ".join(heading_path).encode("utf-8"))
        <= EVIDENCE_HEADING_MAX_BYTES
    )


def test_serializable_metadata_contains_no_private_path_or_full_file(
    tmp_path: Path,
) -> None:
    private_marker = "private full reference body marker"
    path = tmp_path / "private.txt"
    path.write_text(
        f"{private_marker}\n\nPython 3.13 was released in October 2024.\n",
        encoding="utf-8",
    )
    reference = _reference(path, "safe_ref.txt")

    index = LocalReferenceIndex((reference,))
    metadata = json.dumps(
        [item.model_dump(mode="json") for item in index.references],
        sort_keys=True,
    )
    result = index.lookup(_claim())[0].model_dump_json()

    assert str(tmp_path) not in repr(reference)
    assert str(tmp_path) not in metadata
    assert str(tmp_path) not in result
    assert private_marker not in metadata
    assert private_marker not in result
    assert index.references[0].source.url == "local-reference://safe_ref.txt"


def test_bad_utf8_and_hash_change_fail_with_safe_messages(tmp_path: Path) -> None:
    path = tmp_path / "invalid.txt"
    path.write_bytes(b"\xff\xfe")
    invalid = _reference(path, "invalid_ref.txt")

    with pytest.raises(LocalEvidenceError, match="not valid UTF-8") as caught:
        LocalReferenceIndex((invalid,))
    assert str(tmp_path) not in str(caught.value)

    path.write_text("valid now", encoding="utf-8")
    with pytest.raises(LocalEvidenceError, match="SHA-256 changed") as caught:
        LocalReferenceIndex((invalid,))
    assert str(tmp_path) not in str(caught.value)
