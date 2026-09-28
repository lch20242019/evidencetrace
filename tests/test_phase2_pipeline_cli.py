from __future__ import annotations

import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

from evidencetrace.cli import app
from evidencetrace.models import (
    AtomicClaim,
    AuditArtifact,
    Checkability,
    EffectiveConfig,
    EvidenceSpan,
    Relation,
    RunMetadata,
    Verdict,
)
from evidencetrace.pipeline import CitationAuditPipeline, fixture_transport
from evidencetrace.product import DocumentRunStatus
from evidencetrace.render import render_audit_markdown
from evidencetrace.retrieval.fetch import SafeFetcher

ROOT = Path(__file__).parents[1]


def copy_examples(tmp_path: Path) -> None:
    shutil.copytree(ROOT / "examples", tmp_path / "examples")


def offline_pipeline(tmp_path: Path) -> CitationAuditPipeline:
    manifest = tmp_path / "examples/evidence/source_manifest.json"
    return CitationAuditPipeline(
        project_root=tmp_path,
        fetcher=SafeFetcher(
            transport=fixture_transport(manifest),
            resolve_dns=False,
            retries=0,
        ),
    )


def test_bad_fixture_end_to_end_writes_canonical_and_derived_artifacts(
    tmp_path: Path,
) -> None:
    copy_examples(tmp_path)
    result = offline_pipeline(tmp_path).run(
        tmp_path / "examples/bad-agent-comparison.md",
        run_id="bad-fixture-run",
    )
    audit, audit_path = result.audit, result.audit_path

    assert audit_path is not None
    decoded = AuditArtifact.model_validate_json(audit_path.read_text(encoding="utf-8"))
    assert decoded == audit
    assert audit_path.name == "audit.json"
    assert audit_path.with_name("audit.md").read_text(
        encoding="utf-8"
    ) == render_audit_markdown(audit)
    assert "Verdict: contradicted" in result.terminal
    assert Relation.SOURCE_UNAVAILABLE in {item.relation for item in audit.verdicts}
    assert len(audit.claims) == 10
    assert all(
        span.text
        for verdict in audit.verdicts
        if verdict.relation
        in {Relation.ENTAILED, Relation.PARTIALLY_ENTAILED, Relation.CONTRADICTED}
        for span in verdict.evidence_spans
    )


def test_simple_and_complex_fixtures_run_offline(tmp_path: Path) -> None:
    copy_examples(tmp_path)
    pipeline = offline_pipeline(tmp_path)

    simple = pipeline.run(
        tmp_path / "examples/simple-work-document.md", run_id="simple-run"
    ).audit
    complex_doc = pipeline.run(
        tmp_path / "examples/complex-work-document.md", run_id="complex-run"
    ).audit

    assert simple.documents[0].path == "examples/simple-work-document.md"
    assert complex_doc.documents[0].path == "examples/complex-work-document.md"
    assert simple.verdicts
    assert complex_doc.verdicts


def test_cli_demo_succeeds_without_api_key(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    copy_examples(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    first = CliRunner().invoke(app, ["demo"])
    second = CliRunner().invoke(app, ["demo"])

    assert first.exit_code == second.exit_code == 0
    assert first.stdout == second.stdout
    assert "Verdict: contradicted" in first.stdout
    assert "Verdict: not_checkable" in first.stdout
    assert "scout=1" in first.stdout
    assert "challenger=3" in first.stdout
    assert "c_0001=needs_human" in first.stdout
    assert "c_0004=completed" in first.stdout
    assert not (tmp_path / ".evidencetrace").exists()


def test_cli_check_without_credentials_uses_deterministic_fallback(
    tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    document = tmp_path / "doc.md"
    document.write_text(
        "Teams should prefer Nimbus for every project.", encoding="utf-8"
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("EVIDENCETRACE_MODEL", raising=False)
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(app, ["check", "doc.md"])

    assert result.exit_code == 0, result.stderr
    assert "live_model=missing_provider_configuration" in result.stdout
    assert "Verdict: not_checkable" in result.stdout


def test_cli_check_help_describes_self_use_safety_contract() -> None:
    result = CliRunner().invoke(app, ["check", "--help"])

    assert result.exit_code == 0
    assert "citation-first" in result.stdout
    assert "Explicitly authorize bounded Scout discovery" in result.stdout
    assert "missing search key" in result.stdout
    assert "discovery_unavailable" in result.stdout
    assert "without applying, staging, or committing" in result.stdout


def test_cli_check_returns_policy_failure_for_contradicted_result(
    tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    document = tmp_path / "doc.md"
    document.write_text("Claim [source](https://example.test).\n", encoding="utf-8")
    audit = AuditArtifact(
        run=RunMetadata(
            run_id="contradicted-cli-fixture",
            started_at=datetime(2033, 2, 3, tzinfo=UTC),
            tool_version="0.0.0",
        ),
        effective_config=EffectiveConfig(),
        claims=(
            AtomicClaim(
                claim_id="claim_cli",
                text="Synthetic claim.",
                file="doc.md",
                line_start=1,
                line_end=1,
                claim_type="synthetic_fact",
                checkability=Checkability.CHECKABLE,
            ),
            AtomicClaim(
                claim_id="claim_unsupported",
                text="Another synthetic claim.",
                file="doc.md",
                line_start=1,
                line_end=1,
                claim_type="synthetic_fact",
                checkability=Checkability.CHECKABLE,
            ),
        ),
        verdicts=(
            Verdict(
                claim_id="claim_cli",
                relation=Relation.CONTRADICTED,
                confidence=0.8,
                source_ids=("source_cli",),
                evidence_spans=(
                    EvidenceSpan(
                        source_id="source_cli",
                        text="Synthetic evidence.",
                        locator="line 1",
                    ),
                ),
                reason="Synthetic contradiction.",
                judge_version="synthetic-v1",
            ),
            Verdict(
                claim_id="claim_unsupported",
                relation=Relation.NOT_IN_SOURCE,
                confidence=0.7,
                reason="Synthetic unsupported result.",
                judge_version="synthetic-v1",
            ),
        ),
    )

    class FakePipeline:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def run(self, _path: Path, **_kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(
                audit=audit,
                audit_path=tmp_path / "audit.json",
                terminal="ERROR doc.md:1\nVerdict: contradicted",
                product=SimpleNamespace(document_status=DocumentRunStatus.COMPLETE),
            )

    monkeypatch.setattr("evidencetrace.product.ProductAuditPipeline", FakePipeline)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("EVIDENCETRACE_MODEL", "test-model")
    monkeypatch.chdir(tmp_path)
    before = document.read_bytes()

    result = CliRunner().invoke(
        app,
        ["check", "doc.md", "--suggest-patch", "suggested.diff"],
    )

    assert result.exit_code == 1, result.stderr
    assert "Verdict: contradicted" in result.stdout
    assert document.read_bytes() == before
    patch = tmp_path / "suggested.diff"
    patch_text = patch.read_text(encoding="utf-8")
    assert "EvidenceTrace candidate correction" in patch_text
    assert "Synthetic evidence." in patch_text
    assert "claim_unsupported" not in patch_text
    subprocess.run(
        ["git", "apply", "--check", patch.name],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
