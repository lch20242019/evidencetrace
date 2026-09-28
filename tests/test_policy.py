from __future__ import annotations

from datetime import UTC, datetime

import pytest

from evidencetrace.config import ConfigError, load_config, loads_config
from evidencetrace.models import (
    CorroborationStatus,
    EffectiveConfig,
    MarkdownParagraph,
    ParagraphKind,
    ParsedDocument,
    PathsConfig,
    PolicyConfig,
    Relation,
    Severity,
    SourceSpan,
    SuppressionDirective,
    SuppressionKind,
    SuppressionScope,
)
from evidencetrace.policy import (
    decide_policy,
    exit_code,
    path_is_in_scope,
    path_matches,
    relation_severity,
    should_fail,
    suppression_for,
)


def span(start: int = 0, end: int = 20, *, line: int = 2) -> SourceSpan:
    return SourceSpan(
        file="docs/example.md",
        line_start=line,
        line_end=line,
        column_start=1,
        column_end=max(end - start, 1),
        offset_start=start,
        offset_end=end,
    )


def suppressed_document() -> ParsedDocument:
    paragraph_source = span(50, 100, line=4)
    directive_source = span(20, 49, line=3)
    directive = SuppressionDirective(
        suppression_id="sup_001",
        reason="team recommendation, not a factual claim",
        scope=SuppressionScope.PARAGRAPH,
        directive_source=directive_source,
        target_source=paragraph_source,
    )
    paragraph = MarkdownParagraph(
        paragraph_id="p_001",
        kind=ParagraphKind.PARAGRAPH,
        source=paragraph_source,
        raw_text="We should prefer this tool.",
        plain_text="We should prefer this tool.",
        suppression_ids=(directive.suppression_id,),
    )
    return ParsedDocument(
        path="docs/example.md",
        content_sha256="a" * 64,
        line_count=4,
        paragraphs=(paragraph,),
        suppressions=(directive,),
    )


@pytest.mark.parametrize(
    ("relation", "expected"),
    [
        (Relation.CONTRADICTED, Severity.ERROR),
        (Relation.PARTIALLY_ENTAILED, Severity.WARNING),
        (Relation.NOT_IN_SOURCE, Severity.WARNING),
        (Relation.SOURCE_UNAVAILABLE, Severity.NOTICE),
        (Relation.NOT_CHECKABLE, Severity.NOTICE),
        (Relation.ENTAILED, Severity.PASS),
    ],
)
def test_default_relation_mapping(relation: Relation, expected: Severity) -> None:
    assert relation_severity(relation) == expected


@pytest.mark.parametrize(
    ("corroboration", "expected"),
    [
        (CorroborationStatus.DISPUTED, Severity.WARNING),
        (CorroborationStatus.NO_EVIDENCE, Severity.WARNING),
        (CorroborationStatus.CITED_ONLY, Severity.PASS),
        (CorroborationStatus.CORROBORATED, Severity.PASS),
        (CorroborationStatus.NOT_REQUESTED, Severity.PASS),
    ],
)
def test_corroboration_mapping_uses_more_severe_dimension(
    corroboration: CorroborationStatus,
    expected: Severity,
) -> None:
    assert relation_severity(Relation.ENTAILED, corroboration) == expected


def test_custom_policy_mapping_is_respected() -> None:
    policy = PolicyConfig(
        fail_on=("contradicted", "not_in_source"),
        warn_on=(),
        notice_on=(),
    )

    assert relation_severity(Relation.NOT_IN_SOURCE, policy=policy) == Severity.ERROR
    assert relation_severity(Relation.SOURCE_UNAVAILABLE, policy=policy) == Severity.PASS


@pytest.mark.parametrize(
    ("path", "pattern", "matches"),
    [
        ("README.md", "**/*.md", True),
        ("docs/one.md", "docs/**/*.md", True),
        ("docs/deep/two.md", "docs/**/*.md", True),
        ("docs/deep/two.txt", "docs/**/*.md", False),
        ("DOCS/one.md", "docs/**/*.md", False),
        ("docs/a.md", "docs/?.md", True),
    ],
)
def test_posix_glob_matching(path: str, pattern: str, matches: bool) -> None:
    assert path_matches(path, pattern) is matches


def test_path_exclude_takes_precedence() -> None:
    config = PathsConfig(
        include=("README.md", "docs/**/*.md"),
        exclude=("docs/archive/**", "CHANGELOG.md"),
    )

    assert path_is_in_scope("README.md", config)
    assert path_is_in_scope("docs/design.md", config)
    assert not path_is_in_scope("docs/archive/old.md", config)
    assert not path_is_in_scope("src/module.py", config)


def test_explicit_directive_suppression_is_auditable() -> None:
    document = suppressed_document()
    claim_source = span(55, 70, line=4)

    suppression = suppression_for(claim_source, None, document, PolicyConfig())
    decision = decide_policy(
        Relation.CONTRADICTED,
        source=claim_source,
        document=document,
    )

    assert suppression is not None
    assert suppression.kind == SuppressionKind.DIRECTIVE
    assert suppression.reason == "team recommendation, not a factual claim"
    assert decision.raw_severity == Severity.ERROR
    assert decision.effective_severity is None
    assert decision.suppression == suppression
    assert not should_fail((decision,))


def test_claim_type_ignore_applies_without_document_directive() -> None:
    policy = PolicyConfig(ignore_claim_types=("recommendation",))

    decision = decide_policy(
        Relation.CONTRADICTED,
        policy=policy,
        claim_type="recommendation",
    )

    assert decision.suppression is not None
    assert decision.suppression.kind == SuppressionKind.CLAIM_TYPE
    assert decision.effective_severity is None


def test_fail_threshold_is_applied_after_suppression() -> None:
    policy = PolicyConfig(fail_threshold=Severity.WARNING)
    warning = decide_policy(Relation.NOT_IN_SOURCE, policy=policy)
    pass_decision = decide_policy(Relation.ENTAILED, policy=policy)

    assert should_fail((warning, pass_decision), policy)
    assert exit_code((warning,), policy) == 1
    assert exit_code((pass_decision,), policy) == 0


def test_loads_complete_plan_config() -> None:
    config = loads_config(
        """
version = 1

[paths]
include = ["README.md", "docs/**/*.md"]
exclude = ["docs/archive/**", "CHANGELOG.md"]

[policy]
fail_on = ["contradicted"]
warn_on = ["partially_entailed", "not_in_source", "no_evidence"]
trusted_domains = ["Docs.Python.org.", "github.com"]
prefer_primary_sources = true
require_source_span = true

[budget]
max_changed_claims = 30
max_fetches = 20
max_searches = 6
max_cost_usd = 0.50
timeout_seconds = 180

[cache]
ttl_days = 30
path = ".evidencetrace/cache.sqlite3"

[model]
provider = "openai-compatible"
name = "model-id"
"""
    )

    assert config.policy.trusted_domains == ("docs.python.org", "github.com")
    assert config.budget.max_cost_usd == 0.5
    assert config.model.name == "model-id"


def test_implicit_missing_config_uses_defaults(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    assert load_config() == EffectiveConfig()


def test_explicit_missing_or_invalid_config_is_an_error(tmp_path) -> None:
    with pytest.raises(ConfigError, match="does not exist"):
        load_config(tmp_path / "missing.toml")

    invalid = tmp_path / "invalid.toml"
    invalid.write_text("unknown = true\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid EvidenceTrace configuration"):
        load_config(invalid)


def test_run_timestamp_example_uses_aware_datetime() -> None:
    # Guard a common caller mistake while keeping policy tests independent of artifacts.
    timestamp = datetime(2026, 7, 10, tzinfo=UTC)

    assert timestamp.utcoffset() is not None
