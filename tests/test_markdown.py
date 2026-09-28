from __future__ import annotations

from pathlib import Path

import pytest

from evidencetrace.markdown import (
    parse_markdown,
    parse_markdown_file,
    parse_markdown_with_source_map,
)
from evidencetrace.models import (
    CitationKind,
    ModelTextSourceSegment,
    ParagraphKind,
    ParsedDocument,
    SuppressionScope,
    TargetKind,
)


def parse(text: str) -> ParsedDocument:
    return parse_markdown(text, "docs/fixture.md")


def test_empty_document_has_no_objects() -> None:
    document = parse("")

    assert document.line_count == 0
    assert document.paragraphs == ()
    assert document.citations == ()


def test_plain_paragraph_has_exact_source_span() -> None:
    document = parse("A plain paragraph.\n")
    paragraph = document.paragraphs[0]

    assert paragraph.source.line_start == paragraph.source.line_end == 1
    assert paragraph.raw_text == "A plain paragraph."
    assert document.content_sha256
    assert (
        "A plain paragraph."[
            paragraph.source.offset_start : paragraph.source.offset_end
        ]
        == "A plain paragraph."
    )


def test_heading_context_is_not_a_paragraph() -> None:
    document = parse("# Runtime\n\n## Limits\n\nThe limit is 10.\n")

    assert len(document.paragraphs) == 1
    assert document.paragraphs[0].heading_path == ("Runtime", "Limits")


def test_inline_link_and_title_are_citations() -> None:
    document = parse('[docs](https://example.test/docs "API docs")')
    citation = document.citations[0]

    assert citation.kind is CitationKind.INLINE_LINK
    assert citation.destination == "https://example.test/docs"
    assert citation.target_kind is TargetKind.WEB
    assert citation.source.offset_start == 0
    assert citation.raw.startswith("[docs](")


def test_inline_link_with_nested_parentheses() -> None:
    document = parse("[release](https://example.test/a_(b)/c)")

    assert document.citations[0].destination == "https://example.test/a_(b)/c"


def test_reference_full_collapsed_and_shortcut_forms() -> None:
    document = parse(
        "[full][API] [API][] [API]\n\n[API]: https://example.test/reference\n"
    )

    assert [citation.kind for citation in document.citations] == [
        CitationKind.REFERENCE_LINK,
        CitationKind.REFERENCE_LINK,
        CitationKind.REFERENCE_LINK,
    ]
    assert all(
        citation.destination == "https://example.test/reference"
        for citation in document.citations
    )
    assert len(document.definitions) == 1


def test_reference_key_normalization_and_first_definition_wins() -> None:
    document = parse(
        "[x][  MiXeD   Key ]\n\n"
        "[mixed key]: https://example.test/first\n"
        "[MIXED KEY]: https://example.test/second\n"
    )

    assert document.citations[0].destination == "https://example.test/first"
    assert document.citations[0].resolved is True
    assert len(document.definitions) == 2
    assert any(item.code == "duplicate_definition" for item in document.diagnostics)


def test_unresolved_explicit_reference_is_retained_with_diagnostic() -> None:
    document = parse("[missing][no-such-definition]")

    assert document.citations[0].resolved is False
    assert document.citations[0].kind is CitationKind.REFERENCE_LINK
    assert any(item.code == "unresolved_reference" for item in document.diagnostics)


def test_basic_and_multiline_footnotes_resolve_urls() -> None:
    document = parse(
        "A note[^one].\n\n"
        "[^one]: See https://example.test/one and\n"
        "    https://example.test/two\n"
    )

    citation = document.citations[0]
    assert citation.kind is CitationKind.FOOTNOTE
    assert citation.resolved_urls == (
        "https://example.test/one",
        "https://example.test/two",
    )
    assert document.definitions[0].kind.value == "footnote"


def test_unresolved_footnote_is_retained() -> None:
    document = parse("A note[^missing].")

    assert document.citations[0].kind is CitationKind.FOOTNOTE
    assert document.citations[0].resolved is False
    assert any(item.code == "unresolved_footnote" for item in document.diagnostics)


def test_autolink_https_and_email() -> None:
    document = parse("<https://example.test> and <person@example.test>")

    assert [citation.kind for citation in document.citations] == [
        CitationKind.AUTOLINK,
        CitationKind.AUTOLINK,
    ]
    assert document.citations[1].target_kind is TargetKind.EMAIL
    assert document.citations[1].destination == "mailto:person@example.test"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("See https://example.test/a.", "https://example.test/a"),
        ("See https://example.test/a,b;", "https://example.test/a,b"),
        ("See https://example.test/a_(b).", "https://example.test/a_(b)"),
    ],
)
def test_bare_url_trims_terminal_punctuation(text: str, expected: str) -> None:
    document = parse(text)

    assert document.citations[0].kind is CitationKind.BARE_URL
    assert document.citations[0].destination == expected


def test_multiple_citations_keep_source_order_and_duplicate_occurrences() -> None:
    document = parse(
        "https://example.test/a and [b](https://example.test/b) and "
        "https://example.test/a"
    )

    assert [citation.destination for citation in document.citations] == [
        "https://example.test/a",
        "https://example.test/b",
        "https://example.test/a",
    ]
    assert len({citation.citation_id for citation in document.citations}) == 3


def test_multiline_paragraph_preserves_line_range() -> None:
    document = parse("first line\nsecond line with https://example.test/x\n")
    paragraph = document.paragraphs[0]

    assert (paragraph.source.line_start, paragraph.source.line_end) == (1, 2)
    assert document.citations[0].source.line_start == 2


def test_list_items_are_paragraphs() -> None:
    document = parse("- one\n- [two](https://example.test/two)\n")

    assert [paragraph.kind for paragraph in document.paragraphs] == [
        ParagraphKind.LIST_ITEM,
        ParagraphKind.LIST_ITEM,
    ]
    assert document.citations[0].source.line_start == 2


def test_blockquote_is_a_paragraph() -> None:
    document = parse("> quoted https://example.test/quote\n")

    assert document.paragraphs[0].kind is ParagraphKind.BLOCKQUOTE
    assert document.citations[0].source.line_start == 1


def test_fenced_and_indented_code_are_ignored() -> None:
    document = parse(
        "```text\nhttps://example.test/fenced\n```\n\n"
        "    https://example.test/indented\n"
    )

    assert document.citations == ()
    assert document.paragraphs == ()


def test_inline_code_urls_are_ignored() -> None:
    document = parse("Use `https://example.test/inline` in the shell.")

    assert document.citations == ()
    assert document.paragraphs[0].plain_text == "Use in the shell."


def test_inline_code_facts_remain_in_miner_plain_text() -> None:
    document = parse(
        "Use version `0.0.0` before the first release `0.1.0`, "
        "and dispatch `pull_request_target` only for trusted workflows."
    )

    assert document.citations == ()
    assert document.paragraphs[0].plain_text == (
        "Use version 0.0.0 before the first release 0.1.0, "
        "and dispatch pull_request_target only for trusted workflows."
    )


def test_model_visible_source_map_preserves_public_plain_text_contract() -> None:
    text = (
        "# Runtime\n\n"
        "The [stable channel](https://example.test/releases) uses `worker.v2`.\n"
    )

    bundle = parse_markdown_with_source_map(text, "docs/fixture.md")

    assert bundle.document == parse(text)
    assert bundle.model_visible_paragraphs[0].text == (
        bundle.document.paragraphs[0].plain_text
    )
    assert "model_visible_paragraphs" not in bundle.document.model_dump()


def test_inline_code_identifier_metadata_is_parser_owned_and_narrow() -> None:
    text = (
        "Use `pull_request_target` and `worker.retry_v2`. "
        "Keep `more details`, ` `, "
        "[pull_request_target](https://example.test/guide), "
        "and plain pull_request_target. Ignore `https://example.test/x`."
    )

    bundle = parse_markdown_with_source_map(text, "docs/fixture.md")
    paragraph = bundle.model_visible_paragraphs[0]
    identifier_ranges = paragraph.inline_code_identifier_ranges()

    assert paragraph.text == (
        "Use pull_request_target and worker.retry_v2. "
        "Keep more details, , pull_request_target, "
        "and plain pull_request_target. Ignore"
    )
    assert paragraph.text == bundle.document.paragraphs[0].plain_text
    assert tuple(paragraph.text[start:end] for start, end in identifier_ranges) == (
        "pull_request_target",
        "worker.retry_v2",
    )
    assert all(
        paragraph.source_line_range(start, end) == (1, 1)
        for start, end in identifier_ranges
    )
    assert paragraph.paragraph_id == bundle.document.paragraphs[0].paragraph_id
    assert (
        sum(segment.kind == "inline_code_identifier" for segment in paragraph.segments)
        == 2
    )
    assert any(segment.kind == "inline_code" for segment in paragraph.segments)
    assert any(segment.kind == "link_label" for segment in paragraph.segments)


def test_historical_model_text_segment_without_kind_defaults_to_plain() -> None:
    segment = ModelTextSourceSegment.model_validate(
        {
            "model_start": 0,
            "model_end": 4,
            "source_line_start": 1,
            "source_line_end": 1,
            "split_protected": False,
        }
    )

    assert segment.kind == "plain"


def test_single_line_model_text_has_trusted_line_mapping() -> None:
    bundle = parse_markdown_with_source_map(
        "Alpha   ships\tversion 2.1.", "docs/fixture.md"
    )
    model_visible = bundle.model_visible_paragraphs[0]

    assert model_visible.text == "Alpha ships version 2.1."
    assert model_visible.source_line_range(0, len(model_visible.text)) == (1, 1)
    assert tuple(
        (segment.model_start, segment.model_end) for segment in model_visible.segments
    ) == ((0, len(model_visible.text)),)


def test_multiline_softbreak_has_conservative_line_mapping() -> None:
    bundle = parse_markdown_with_source_map(
        "Alpha ships version 2.1.\nBeta waits until 2026-08-01.",
        "docs/fixture.md",
    )
    model_visible = bundle.model_visible_paragraphs[0]
    separator = model_visible.text.index(" Beta")

    assert model_visible.text == (
        "Alpha ships version 2.1. Beta waits until 2026-08-01."
    )
    assert model_visible.source_line_range(0, separator) == (1, 1)
    assert model_visible.source_line_range(separator, separator + 1) == (1, 2)
    assert model_visible.source_line_range(separator + 1, len(model_visible.text)) == (
        2,
        2,
    )


def test_multiline_inline_code_is_retained_with_conservative_mapping() -> None:
    bundle = parse_markdown_with_source_map(
        "Use `worker.v2;\nretry_01` for bounded jobs.",
        "docs/fixture.md",
    )
    model_visible = bundle.model_visible_paragraphs[0]
    code_start = model_visible.text.index("worker.v2")
    code_end = code_start + len("worker.v2; retry_01")

    assert model_visible.text == "Use worker.v2; retry_01 for bounded jobs."
    assert model_visible.source_line_range(code_start, code_end) == (1, 2)
    assert model_visible.split_protected_ranges() == ((code_start, code_end),)


def test_multiline_link_keeps_label_and_removes_destination_from_model_text() -> None:
    bundle = parse_markdown_with_source_map(
        "The [stable\nchannel](https://example.test/releases) ships.",
        "docs/fixture.md",
    )
    model_visible = bundle.model_visible_paragraphs[0]
    channel_start = model_visible.text.index("channel")

    assert model_visible.text == "The stable channel ships."
    assert "example.test" not in model_visible.text
    assert model_visible.source_line_range(4, len("The stable")) == (1, 1)
    assert model_visible.source_line_range(
        channel_start, channel_start + len("channel")
    ) == (2, 2)
    assert model_visible.citation_ids == ("cit_0001",)
    assert model_visible.citation_urls == ("https://example.test/releases",)


def test_removed_markdown_constructs_leave_in_bounds_source_map() -> None:
    bundle = parse_markdown_with_source_map(
        "Read https://example.test/x and ![diagram](image.png) "
        "<span>now</span> note[^n].",
        "docs/fixture.md",
    )
    model_visible = bundle.model_visible_paragraphs[0]

    assert model_visible.text == "Read and now note."
    assert "diagram" not in model_visible.text
    assert "[^n]" not in model_visible.text
    assert all(
        model_visible.source.line_start
        <= segment.source_line_start
        <= segment.source_line_end
        <= model_visible.source.line_end
        for segment in model_visible.segments
    )


def test_escaped_link_syntax_is_not_a_citation() -> None:
    document = parse(r"\[not a link](https://example.test/escaped)")

    assert document.citations == ()


def test_image_destination_is_not_a_citation() -> None:
    document = parse("![diagram](https://example.test/image.png)")

    assert document.citations == ()


def test_relative_anchor_and_mailto_targets_are_classified() -> None:
    document = parse(
        "[relative](docs/source.md) [anchor](#section) "
        "[mail](mailto:person@example.test)"
    )

    assert [citation.target_kind for citation in document.citations] == [
        TargetKind.RELATIVE,
        TargetKind.ANCHOR,
        TargetKind.EMAIL,
    ]


def test_urls_in_html_comments_are_ignored() -> None:
    document = parse("<!-- https://example.test/comment -->\nVisible text.")

    assert document.citations == ()


def test_standalone_suppression_binds_next_paragraph() -> None:
    document = parse(
        '<!-- evidencetrace: ignore reason="team recommendation" -->\n'
        "\nThe recommendation is local.\n"
    )

    assert len(document.suppressions) == 1
    assert document.suppressions[0].scope is SuppressionScope.PARAGRAPH
    assert document.paragraphs[0].suppression_ids == ("sup_0001",)


def test_inline_suppression_is_line_scoped() -> None:
    document = parse(
        'Recommendation text. <!-- evidencetrace: ignore reason="local choice" -->\n'
    )

    assert document.suppressions[0].scope is SuppressionScope.LINE
    assert document.suppressions[0].target_source.line_start == 1


def test_suppression_without_reason_is_diagnostic_and_inactive() -> None:
    document = parse("<!-- evidencetrace: ignore -->\nClaim text.\n")

    assert document.suppressions == ()
    assert any(
        item.code == "invalid_suppression_reason" for item in document.diagnostics
    )


def test_suppression_does_not_cross_a_heading() -> None:
    document = parse(
        '<!-- evidencetrace: ignore reason="too narrow" -->\n'
        "\n## A heading\n\nClaim text.\n"
    )

    assert document.suppressions == ()
    assert any(item.code == "orphan_suppression" for item in document.diagnostics)


def test_crlf_unicode_and_no_final_newline_keep_source_offsets() -> None:
    text = "# Café\r\n\r\nClaim é https://example.test/x"
    document = parse(text)
    paragraph = document.paragraphs[0]
    citation = document.citations[0]

    assert document.line_count == 3
    assert paragraph.source.line_start == paragraph.source.line_end == 3
    assert (
        text[citation.source.offset_start : citation.source.offset_end] == citation.raw
    )
    assert citation.source.column_start == len("Claim é ") + 1


def test_repository_fixtures_parse_without_network() -> None:
    root = Path(__file__).parents[1]
    bad = parse_markdown_file(root / "examples/bad-agent-comparison.md")
    simple = parse_markdown_file(root / "examples/simple-work-document.md")
    complex_doc = parse_markdown_file(root / "examples/complex-work-document.md")

    assert len(bad.citations) == 8
    assert len(simple.suppressions) == 1
    assert len(complex_doc.definitions) == 3


@pytest.mark.parametrize(
    ("text", "expected_kinds"),
    [
        pytest.param(
            "[s](https://example.test/a)", (CitationKind.INLINE_LINK,), id="inline"
        ),
        pytest.param(
            '[s](https://example.test/a "title")',
            (CitationKind.INLINE_LINK,),
            id="inline-title",
        ),
        pytest.param(
            "[s](<https://example.test/a>)",
            (CitationKind.INLINE_LINK,),
            id="inline-angle",
        ),
        pytest.param("[s](../a)", (CitationKind.INLINE_LINK,), id="relative"),
        pytest.param("[s](#a)", (CitationKind.INLINE_LINK,), id="anchor"),
        pytest.param(
            "[s](mailto:a@example.test)", (CitationKind.INLINE_LINK,), id="mailto"
        ),
        pytest.param(
            "[s][r]\n\n[r]: https://example.test/r\n",
            (CitationKind.REFERENCE_LINK,),
            id="reference-full",
        ),
        pytest.param(
            "[r][]\n\n[r]: https://example.test/r\n",
            (CitationKind.REFERENCE_LINK,),
            id="reference-collapsed",
        ),
        pytest.param(
            "[r]\n\n[r]: https://example.test/r\n",
            (CitationKind.REFERENCE_LINK,),
            id="reference-shortcut",
        ),
        pytest.param(
            "[s][missing]", (CitationKind.REFERENCE_LINK,), id="reference-unresolved"
        ),
        pytest.param(
            "A[^n]\n\n[^n]: https://example.test/n\n",
            (CitationKind.FOOTNOTE,),
            id="footnote",
        ),
        pytest.param("A[^n]", (CitationKind.FOOTNOTE,), id="footnote-unresolved"),
        pytest.param(
            "<https://example.test/a>", (CitationKind.AUTOLINK,), id="autolink-web"
        ),
        pytest.param("<a@example.test>", (CitationKind.AUTOLINK,), id="autolink-email"),
        pytest.param(
            "https://example.test/a.", (CitationKind.BARE_URL,), id="bare-period"
        ),
        pytest.param(
            "(https://example.test/a).",
            (CitationKind.BARE_URL,),
            id="bare-closing-paren",
        ),
        pytest.param(
            "https://example.test/a_(b).",
            (CitationKind.BARE_URL,),
            id="bare-balanced-paren",
        ),
        pytest.param(
            "https://example.test/a and <https://example.test/b>",
            (CitationKind.BARE_URL, CitationKind.AUTOLINK),
            id="source-order",
        ),
        pytest.param("`https://example.test/a`", (), id="exclude-inline-code"),
        pytest.param("```\nhttps://example.test/a\n```\n", (), id="exclude-fence"),
        pytest.param("![x](https://example.test/a)", (), id="exclude-image"),
        pytest.param(
            "<!-- https://example.test/a -->\nText.", (), id="exclude-comment"
        ),
        pytest.param(r"\[x](https://example.test/a)", (), id="exclude-escaped-link"),
        pytest.param(
            "[x](https://example.test/a)",
            (CitationKind.INLINE_LINK,),
            id="no-bare-duplicate",
        ),
    ],
)
def test_parameterized_parser_contract(
    text: str, expected_kinds: tuple[CitationKind, ...]
) -> None:
    document = parse(text)

    assert tuple(citation.kind for citation in document.citations) == expected_kinds
