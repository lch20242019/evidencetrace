from __future__ import annotations

import pytest

from evidencetrace.agents.miner import (
    MINER_WINDOW_POLICY_VERSION,
    ClaimMinerAgent,
    plan_miner_windows,
)
from evidencetrace.audit_models import MINER_DRAFT_CONTRACT_VERSION
from evidencetrace.markdown import parse_markdown_with_source_map


def _model_visible(text: str):
    bundle = parse_markdown_with_source_map(text, "docs/window-fixture.md")
    return bundle.model_visible_paragraphs[0]


def test_windows_are_exact_ordered_non_overlapping_and_locally_owned() -> None:
    paragraph = _model_visible(
        "# Safety\n\n"
        "Requests reject unsigned metadata before processing; "
        "Operators record each bounded decision in an append-only trace; "
        "Workers preserve version v2.4 through 2026-08-01; "
        "Reviewers compare lower latency against a named baseline "
        "[guide](https://example.test/guide)."
    )

    plan = plan_miner_windows(paragraph, max_window_chars=100)

    assert plan.status == "ready"
    assert len(plan.windows) == 4
    assert plan.oversized_fragments == ()
    for index, window in enumerate(plan.windows):
        assert paragraph.text[window.start : window.end] == window.text
        assert window.window_id == f"{paragraph.paragraph_id}_w_{index + 1:04d}"
        assert window.paragraph_id == paragraph.paragraph_id
        assert window.source_file == paragraph.source.file
        assert window.heading_path == ("Safety",)
        assert window.citation_ids == ("cit_0001",)
        assert window.citation_urls == ("https://example.test/guide",)
        assert (
            paragraph.source.line_start
            <= window.source_line_start
            <= window.source_line_end
            <= paragraph.source.line_end
        )
    assert all(
        left.end <= right.start
        for left, right in zip(plan.windows, plan.windows[1:], strict=False)
    )


def test_generic_long_paragraph_plans_multiple_bounded_windows() -> None:
    paragraph = _model_visible(
        "Unsigned metadata is rejected before processing begins; "
        "Each bounded decision is recorded in an append-only trace; "
        "Version v2.4 remains pinned through 2026-08-01; "
        "Lower latency is compared only against the named baseline."
    )

    plan = plan_miner_windows(paragraph, max_window_chars=72)

    assert plan.status == "ready"
    assert len(plan.windows) == 4
    assert max(len(window.text) for window in plan.windows) <= 72


def test_identifier_version_date_and_inline_code_are_not_cut_internally() -> None:
    paragraph = _model_visible(
        "Keep `worker.v2; retry_01` intact; retain version v2.4 through 2026-08-01."
    )

    plan = plan_miner_windows(paragraph, max_window_chars=80)

    assert [window.text for window in plan.windows] == [
        "Keep worker.v2; retry_01 intact;",
        "retain version v2.4 through 2026-08-01.",
    ]
    assert "worker.v2; retry_01" in plan.windows[0].text


def test_citation_label_is_not_cut_internally() -> None:
    paragraph = _model_visible(
        "Use the [stable; supported channel](https://example.test/channel); "
        "record the selected version."
    )

    plan = plan_miner_windows(paragraph, max_window_chars=80)

    assert [window.text for window in plan.windows] == [
        "Use the stable; supported channel;",
        "record the selected version.",
    ]


def test_window_identifier_count_comes_only_from_trusted_inline_code() -> None:
    paragraph = _model_visible(
        "Use `worker.retry_v2` for jobs; "
        "mention [worker.retry_v2](https://example.test/guide) and "
        "plain worker.retry_v2 elsewhere."
    )

    plan = plan_miner_windows(paragraph, max_window_chars=80)

    assert [window.inline_code_identifier_count for window in plan.windows] == [1, 0]
    assert tuple(
        paragraph.text[start:end]
        for start, end in paragraph.inline_code_identifier_ranges()
    ) == ("worker.retry_v2",)


def test_unsplittable_oversized_fragment_is_typed_and_never_dispatched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def forbidden_mine(*args: object, **kwargs: object) -> None:
        calls.append("mine")
        raise AssertionError("window planning must not dispatch the Miner")

    monkeypatch.setattr(ClaimMinerAgent, "mine", forbidden_mine)
    paragraph = _model_visible(
        "A continuous factual fragment deliberately contains no safe sentence "
        "or semicolon boundary before its final word"
    )

    plan = plan_miner_windows(paragraph, max_window_chars=40)

    assert calls == []
    assert plan.status == "oversized"
    assert plan.windows == ()
    assert len(plan.oversized_fragments) == 1
    outcome = plan.oversized_fragments[0]
    assert outcome.code == "miner_window_fragment_oversized"
    assert outcome.fragment_length == len(paragraph.text)
    assert outcome.fragment_length > outcome.max_window_chars
    assert outcome.heading_path == ()
    assert outcome.citation_ids == ()
    assert not hasattr(outcome, "text")


def test_oversized_plan_retains_other_precomputed_safe_windows() -> None:
    paragraph = _model_visible(
        "A short complete fact; "
        "This continuous factual fragment deliberately contains no further "
        "safe boundary before its final word"
    )

    plan = plan_miner_windows(paragraph, max_window_chars=40)

    assert plan.status == "oversized"
    assert [window.text for window in plan.windows] == ["A short complete fact;"]
    assert len(plan.oversized_fragments) == 1
    assert plan.windows[0].end <= plan.oversized_fragments[0].start


def test_empty_model_visible_paragraph_has_intentional_empty_plan() -> None:
    paragraph = _model_visible("https://example.test/source")

    plan = plan_miner_windows(paragraph, max_window_chars=80)

    assert paragraph.text == ""
    assert paragraph.segments == ()
    assert plan.status == "intentional_empty"
    assert plan.windows == ()
    assert plan.oversized_fragments == ()


def test_window_plan_is_byte_stable_and_versions_are_independent() -> None:
    text = "The first bounded fact remains stable; the second fact remains stable."
    paragraph = _model_visible(text)

    first = plan_miner_windows(paragraph, max_window_chars=48)
    second = plan_miner_windows(_model_visible(text), max_window_chars=48)

    assert first.model_dump_json() == second.model_dump_json()
    assert first.policy_version == MINER_WINDOW_POLICY_VERSION
    assert MINER_WINDOW_POLICY_VERSION == "miner-window-policy-v1"
    assert MINER_DRAFT_CONTRACT_VERSION == "live-miner-draft-v3"


@pytest.mark.parametrize("bound", [0, -1, True, 1.5])
def test_window_plan_requires_explicit_positive_integer_bound(
    bound: object,
) -> None:
    paragraph = _model_visible("One stable fact.")

    with pytest.raises(ValueError, match="positive integer"):
        plan_miner_windows(paragraph, max_window_chars=bound)  # type: ignore[arg-type]
