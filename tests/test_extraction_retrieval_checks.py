from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest

from evidencetrace.agents.judge import ClaimJudgeAgent
from evidencetrace.agents.scout import DiscoveryTools
from evidencetrace.audit_models import (
    EVIDENCE_COLLECTION_CONTEXT_MAX_BYTES,
    EVIDENCE_COLLECTION_CONTEXT_MAX_CHARS,
    EVIDENCE_COLLECTION_EXACT_MAX_BYTES,
    EVIDENCE_COLLECTION_EXACT_MAX_CHARS,
    EVIDENCE_CONTEXT_MAX_BYTES,
    EVIDENCE_CONTEXT_MAX_CHARS,
    EVIDENCE_EXACT_MAX_BYTES,
    EVIDENCE_EXACT_MAX_CHARS,
    EVIDENCE_HEADING_MAX_BYTES,
    EVIDENCE_HEADING_MAX_CHARS,
    EVIDENCE_LOCATOR_MAX_BYTES,
    EVIDENCE_LOCATOR_MAX_CHARS,
    EvidenceChunk,
    JudgeInput,
)
from evidencetrace.checks.deterministic import (
    deterministic_signals,
    validate_evidence_span,
)
from evidencetrace.model_client import DeterministicFakeModel
from evidencetrace.models import AtomicClaim, Checkability
from evidencetrace.retrieval.extract import extract_html
from evidencetrace.retrieval.fetch import FetchedSource, SafeFetcher
from evidencetrace.retrieval.rank import LexicalRetriever

RETRIEVED_AT = datetime(2026, 7, 10, 12, tzinfo=UTC)


def fetched(content: str, *, mime_type: str = "text/html") -> FetchedSource:
    return FetchedSource(
        url="https://example.com/start",
        final_url="https://example.com/releases",
        content=content,
        content_type=mime_type,
        status_code=200,
        retrieved_at=RETRIEVED_AT,
    )


def chunk(
    text: str,
    *,
    index: int,
    heading: tuple[str, ...] = ("Release notes",),
) -> EvidenceChunk:
    start = index * 1_000
    return EvidenceChunk(
        source_id="s_release",
        url="https://example.com/releases",
        text=text,
        heading_path=heading,
        locator=f"Release notes > paragraph {index + 1}",
        char_start=start,
        char_end=start + len(text),
    )


def signal_codes(claim: str, evidence: str) -> set[str]:
    return {item.code for item in deterministic_signals(claim, evidence)}


def test_html_extraction_removes_noise_and_keeps_heading_paths() -> None:
    html = """
    <html>
      <head><title>Nimbus release notes</title><style>.x { color: red }</style></head>
      <body>
        <header><p>Marketing header</p></header>
        <nav><img src="logo.png"><p>Navigation link</p></nav>
        <main>
          <h1>Nimbus</h1>
          <p>Durable <strong>execution</strong> starts in v1.3.0.</p>
          <div class="cookie-banner"><p>Accept every cookie.</p></div>
          <h2>Benchmarks</h2>
          <p>The measured success rate was 82% on 2026-07-10.</p>
          <p hidden>Hidden content</p>
          <p aria-hidden="true">Screen-reader noise</p>
          <script>window.secret = "not evidence";</script>
        </main>
        <footer><p>Repeated footer</p></footer>
      </body>
    </html>
    """

    metadata, chunks = extract_html(fetched(html), source_id="s_release")

    assert metadata.url == "https://example.com/releases"
    assert metadata.title == "Nimbus release notes"
    assert metadata.retrieved_at == RETRIEVED_AT
    assert [item.text for item in chunks] == [
        "Durable execution starts in v1.3.0.",
        "The measured success rate was 82% on 2026-07-10.",
    ]
    assert chunks[0].heading_path == ("Nimbus",)
    assert chunks[1].heading_path == ("Nimbus", "Benchmarks")
    assert "Nimbus > paragraph 1" in chunks[0].locator
    assert "Nimbus > Benchmarks > paragraph 1" in chunks[1].locator
    combined = "\n\n".join(item.text for item in chunks)
    for item in chunks:
        assert combined[item.char_start : item.char_end] == item.text
    assert not any(
        word in combined
        for word in ("Navigation", "cookie", "secret", "footer", "Hidden")
    )


@pytest.mark.parametrize("mime_type", ["text/plain", "text/html"])
def test_long_single_source_block_is_split_and_neighbor_context_stays_bounded(
    mime_type: str,
) -> None:
    relevant = "Nimbus shipped version 1.2.4."
    body = ("填充内容 " * 1_200) + relevant + (" trailing " * 1_200)
    content = body if mime_type == "text/plain" else f"<main><p>{body}</p></main>"
    _source, chunks = extract_html(fetched(content, mime_type=mime_type), source_id="s")
    extracted = body if mime_type == "text/plain" else " ".join(body.split())

    assert len(chunks) > 3
    assert all(
        extracted[item.char_start : item.char_end] == item.text for item in chunks
    )
    assert all(len(item.text) <= EVIDENCE_EXACT_MAX_CHARS for item in chunks)
    assert all(
        len(item.text.encode("utf-8")) <= EVIDENCE_EXACT_MAX_BYTES
        for item in chunks
    )
    result = LexicalRetriever(chunks, neighbor_window=10).search(relevant, top_k=1)[0]
    assert relevant in result.text
    assert result.text in result.chunk.text
    assert len(result.text) <= EVIDENCE_EXACT_MAX_CHARS
    assert len(result.text.encode("utf-8")) <= EVIDENCE_EXACT_MAX_BYTES
    assert len(result.chunk.text) <= EVIDENCE_CONTEXT_MAX_CHARS
    assert len(result.chunk.text.encode("utf-8")) <= EVIDENCE_CONTEXT_MAX_BYTES


def test_plain_text_evidence_preserves_raw_hashed_source_offsets() -> None:
    raw = "alpha   Nimbus\nshipped version 1.2.4.\n"
    source, chunks = extract_html(
        fetched(raw, mime_type="text/plain"),
        source_id="s",
    )

    assert chunks[0].text == raw.rstrip()
    assert raw[chunks[0].char_start : chunks[0].char_end] == chunks[0].text
    assert source.content_hash


def test_untrusted_title_heading_and_locator_metadata_are_bounded() -> None:
    oversized = "Private-heading-" + ("H" * 10_000)
    source, chunks = extract_html(
        fetched(
            f"<title>{oversized}</title><h1>{oversized}</h1>"
            "<p>Nimbus shipped version 1.2.4.</p>"
        ),
        source_id="s",
    )

    assert oversized not in source.model_dump_json()
    assert len(source.title) <= EVIDENCE_HEADING_MAX_CHARS
    assert len(source.title.encode("utf-8")) <= EVIDENCE_HEADING_MAX_BYTES
    assert all(
        len(" > ".join(item.heading_path)) <= EVIDENCE_HEADING_MAX_CHARS
        and len(" > ".join(item.heading_path).encode("utf-8"))
        <= EVIDENCE_HEADING_MAX_BYTES
        and len(item.locator) <= EVIDENCE_LOCATOR_MAX_CHARS
        and len(item.locator.encode("utf-8")) <= EVIDENCE_LOCATOR_MAX_BYTES
        for item in chunks
    )


def test_retrieval_caps_the_aggregate_exact_and_context_payload() -> None:
    body = (
        "Nimbus shipped version 1.2.4. " + ("supporting detail " * 140)
    ) * 12
    _source, chunks = extract_html(
        fetched(body, mime_type="text/plain"),
        source_id="s",
    )

    evidence = LexicalRetriever(chunks, neighbor_window=10).search(
        "Nimbus shipped version 1.2.4",
        top_k=5,
    )

    assert evidence
    assert sum(len(item.text) for item in evidence) <= (
        EVIDENCE_COLLECTION_EXACT_MAX_CHARS
    )
    assert sum(len(item.text.encode("utf-8")) for item in evidence) <= (
        EVIDENCE_COLLECTION_EXACT_MAX_BYTES
    )
    assert sum(len(item.chunk.text) for item in evidence) <= (
        EVIDENCE_COLLECTION_CONTEXT_MAX_CHARS
    )
    assert sum(len(item.chunk.text.encode("utf-8")) for item in evidence) <= (
        EVIDENCE_COLLECTION_CONTEXT_MAX_BYTES
    )


def test_html_fallback_does_not_reintroduce_script_or_style_text() -> None:
    html = """
    <html><body>
      Loose evidence text.
      <script>script must not be evidence</script>
      <style>style must not be evidence</style>
    </body></html>
    """

    _, chunks = extract_html(fetched(html), source_id="s_release")

    assert tuple(item.text for item in chunks) == ("Loose evidence text.",)


def test_noise_substring_wrapper_preserves_bounded_judge_evidence() -> None:
    url = "https://evidence.example.test/releases/atlas-2-4"
    source_text = "Atlas 2.4 was released on May 6, 2031."
    html = (
        "<html><head><title>Atlas release</title></head><body>"
        '<div id="unavailable-shell"><main>'
        f"<p>{source_text}</p>"
        "</main></div>"
        "<nav><p>Untrusted navigation text.</p></nav>"
        "</body></html>"
    )

    def source(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            request=request,
            headers={"content-type": "text/html; charset=utf-8"},
            text=html,
        )

    claim = AtomicClaim(
        claim_id="c_atlas_release",
        text="Atlas 2.4 was released on May 7, 2031.",
        file="docs/atlas.md",
        line_start=4,
        line_end=4,
        claim_type="version_release",
        checkability=Checkability.CHECKABLE,
        citation_urls=(url,),
    )
    result = DiscoveryTools(
        None,
        SafeFetcher(
            transport=httpx.MockTransport(source),
            resolve_dns=False,
            retries=0,
        ),
    ).resolve_urls(claim, (url,))
    captured: list[dict[str, Any]] = []

    def judge_response(task: str, payload: dict[str, Any]) -> dict[str, Any]:
        assert task == "claim_judgement"
        captured.append(payload)
        return {
            "relation": "not_in_source",
            "confidence": 0.0,
            "reason": "The bounded evidence is insufficient.",
            "evidence_span": None,
        }

    assert result.status == "ok"
    assert result.source is not None
    assert result.evidence
    assert all(item.text in item.chunk.text for item in result.evidence)
    assert all(
        "Untrusted navigation text." not in item.text for item in result.evidence
    )

    ClaimJudgeAgent(DeterministicFakeModel(judge_response)).judge(
        JudgeInput(
            claim=claim,
            source=result.source,
            evidence=result.evidence,
        )
    )

    assert len(captured) == 1
    assert [item["text"] for item in captured[0]["evidence"]] == [
        item.text for item in result.evidence
    ]
    assert source_text in captured[0]["evidence"][0]["text"]


def test_plain_text_source_becomes_one_retrievable_chunk() -> None:
    metadata, chunks = extract_html(
        fetched("Nimbus shipped version 1.3.0.", mime_type="text/plain"),
        source_id="s_release",
    )

    assert metadata.mime_type == "text/plain"
    assert chunks[0].heading_path == ()
    assert chunks[0].text == "Nimbus shipped version 1.3.0."


@pytest.mark.parametrize(
    ("query", "expected_fragment"),
    [
        ("Nimbus success rate was 82%", "82%"),
        ("Nimbus launched on 2026-07-10", "2026-07-10"),
        ("Nimbus durable execution arrived in v1.3.0", "v1.3.0"),
    ],
)
def test_retrieval_boosts_exact_numbers_dates_and_versions(
    query: str, expected_fragment: str
) -> None:
    chunks = (
        chunk(
            "Nimbus success was 62%; it launched on 2025-07-10 in v1.2.0.",
            index=0,
        ),
        chunk(
            "Nimbus success was 82%; it launched on 2026-07-10 in v1.3.0.",
            index=1,
        ),
        chunk("Orion has unrelated release notes.", index=2),
    )

    result = LexicalRetriever(chunks, neighbor_window=0).search(query, top_k=2)

    assert result
    assert expected_fragment in result[0].text


def test_retrieval_boosts_entity_tokens_and_omits_zero_matches() -> None:
    chunks = (
        chunk("Nimbus reports median latency.", index=0),
        chunk("Orion reports median latency.", index=1),
        chunk("Completely unrelated prose.", index=2),
    )

    results = LexicalRetriever(chunks, neighbor_window=0).search(
        "Orion latency", top_k=5
    )

    assert results[0].text == "Orion reports median latency."
    assert all("unrelated" not in item.text for item in results)


def test_retrieval_includes_bounded_neighbor_context() -> None:
    _source, chunks = extract_html(
        fetched(
            "<p>Previous scope qualifier.</p>"
            "<p>The exact benchmark result was 82%.</p>"
            "<p>Following methodology qualifier.</p>"
        ),
        source_id="s_release",
    )

    result = LexicalRetriever(chunks).search("benchmark 82%", top_k=1)[0]

    assert result.text == "The exact benchmark result was 82%."
    assert "Previous scope qualifier." in result.chunk.text
    assert "Following methodology qualifier." in result.chunk.text
    assert result.text in result.chunk.text
    assert result.chunk.char_end - result.chunk.char_start == len(result.chunk.text)


def test_minimal_bm25_fallback_is_ranked_and_filters_misses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def no_fts(*_args: object, **_kwargs: object) -> sqlite3.Connection:
        raise sqlite3.OperationalError("fts5 unavailable")

    monkeypatch.setattr(sqlite3, "connect", no_fts)
    chunks = (
        chunk("Nimbus shipped version 1.2.0.", index=0),
        chunk("Nimbus shipped version 1.3.0 with durable execution.", index=1),
        chunk("Unrelated paragraph.", index=2),
    )

    retriever = LexicalRetriever(chunks, neighbor_window=0)
    results = retriever.search("Nimbus durable execution v1.3.0", top_k=5)

    assert retriever.fts5 is False
    assert results[0].text == chunks[1].text
    assert all(item.text != chunks[2].text for item in results)


@pytest.mark.parametrize(
    ("claim", "source", "expected"),
    [
        ("Success was 82%.", "Success was 62%.", "numeric_mismatch"),
        ("Latency was 1,000 ms.", "Latency was 900 ms.", "numeric_mismatch"),
        ("Released on 2026-07-10.", "Released on 2026-07-11.", "date_mismatch"),
        ("Available in v1.3.0.", "Available in version 1.2.0.", "version_mismatch"),
    ],
)
def test_deterministic_slot_mismatches(claim: str, source: str, expected: str) -> None:
    assert expected in signal_codes(claim, source)


def test_date_and_version_checks_do_not_emit_duplicate_numeric_mismatch() -> None:
    date_codes = signal_codes("Released 2026-07-10.", "Released 2026-07-11.")
    version_codes = signal_codes("Uses v1.3.0.", "Uses v1.2.0.")

    assert date_codes == {"date_mismatch"}
    assert version_codes == {"version_mismatch"}


def test_equivalent_numeric_and_version_spellings_do_not_mismatch() -> None:
    assert signal_codes("Processed 1,000 items.", "Processed 1000 items.") == set()
    assert signal_codes("Uses v1.3.0.", "Uses version 1.3.0.") == set()


def test_negation_is_a_hard_conflict_and_comparison_remains_a_reminder() -> None:
    signals = deterministic_signals(
        "Nimbus is not faster than Orion.",
        "Nimbus is faster than Orion.",
    )
    by_code = {item.code: item for item in signals}

    assert by_code["negation_mismatch"].severity == "error"
    assert by_code["comparison_reminder"].severity == "notice"


def test_span_validation_requires_nonempty_exact_source_substring() -> None:
    source = "The measured success rate was exactly 82%."

    assert validate_evidence_span("success rate was exactly 82%", source) is None
    assert validate_evidence_span("Success rate was exactly 82%", source) is not None
    assert validate_evidence_span("", source) is not None
