from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from evidencetrace.cli import app
from evidencetrace.models import (
    AtomicClaim,
    AuditArtifact,
    Checkability,
    EffectiveConfig,
    EvidenceSpan,
    PolicyConfig,
    Relation,
    RunMetadata,
    SourceMetadata,
    Verdict,
)
from evidencetrace.pipeline import CitationAuditPipeline, fixture_transport
from evidencetrace.product import DocumentRunStatus
from evidencetrace.retrieval.fetch import SafeFetcher
from evidencetrace.sarif import SarifError, build_sarif, sarif_bytes, write_sarif

ROOT = Path(__file__).parents[1]
CLAIM_SECRET = "credential-like claim marker sk-local-claim"
SOURCE_SECRET = "private canonical source body marker"
REASON_SECRET = "Authorization: Bearer private-model-reason"


def _audit(
    rows: list[tuple[str, Relation, str, int]],
    *,
    policy: PolicyConfig | None = None,
) -> AuditArtifact:
    claims = []
    verdicts = []
    for claim_id, relation, file, line in rows:
        claims.append(
            AtomicClaim(
                claim_id=claim_id,
                text=f"{CLAIM_SECRET} {claim_id}",
                file=file,
                line_start=line,
                line_end=line,
                claim_type="synthetic_fact",
                checkability=Checkability.CHECKABLE,
                citation_urls=("https://synthetic.invalid/source",),
            )
        )
        substantive = relation in {
            Relation.ENTAILED,
            Relation.PARTIALLY_ENTAILED,
            Relation.CONTRADICTED,
        }
        verdicts.append(
            Verdict(
                claim_id=claim_id,
                relation=relation,
                confidence=0.8,
                source_ids=("synthetic_source",),
                evidence_spans=(
                    (
                        EvidenceSpan(
                            source_id="synthetic_source",
                            text=SOURCE_SECRET,
                            locator="synthetic locator",
                        ),
                    )
                    if substantive
                    else ()
                ),
                reason=REASON_SECRET,
                judge_version="synthetic-judge-v1",
            )
        )
    return AuditArtifact(
        run=RunMetadata(
            run_id="phase4a-sarif-fixture",
            started_at=datetime(2033, 1, 2, tzinfo=UTC),
            tool_version="0.0.0",
        ),
        effective_config=EffectiveConfig(policy=policy or PolicyConfig()),
        claims=tuple(claims),
        verdicts=tuple(verdicts),
        sources=(
            SourceMetadata(
                source_id="synthetic_source",
                url="https://synthetic.invalid/source",
                title=SOURCE_SECRET,
                retrieved_at=datetime(2033, 1, 2, tzinfo=UTC),
                content_hash="a" * 64,
                mime_type="text/plain",
            ),
        ),
    )


def test_sarif_21_structure_severity_location_and_privacy(tmp_path: Path) -> None:
    audit = _audit(
        [
            ("claim_warning", Relation.NOT_IN_SOURCE, 'docs/quoted "β".md', 19),
            ("claim_error", Relation.CONTRADICTED, "README.md", 7),
        ]
    )

    payload = build_sarif(audit)
    serialized = sarif_bytes(audit).decode()
    run = payload["runs"][0]
    rules = run["tool"]["driver"]["rules"]
    results = run["results"]

    assert payload["version"] == "2.1.0"
    assert payload["$schema"].endswith("/sarif-2.1.0.json")
    assert run["tool"]["driver"]["name"] == "EvidenceTrace"
    assert [rule["id"] for rule in rules] == [
        "ET1001",
        "ET1002",
        "ET1003",
        "ET1004",
        "ET1005",
        "ET1006",
    ]
    assert [(item["ruleId"], item["level"]) for item in results] == [
        ("ET1001", "error"),
        ("ET1002", "warning"),
    ]
    warning_location = results[1]["locations"][0]["physicalLocation"]
    assert warning_location["artifactLocation"]["uri"] == (
        "docs/quoted%20%22%CE%B2%22.md"
    )
    assert warning_location["region"] == {
        "startLine": 19,
        "startColumn": 1,
        "endLine": 19,
    }
    assert str(tmp_path.resolve()) not in serialized
    assert CLAIM_SECRET not in serialized
    assert SOURCE_SECRET not in serialized
    assert REASON_SECRET not in serialized
    assert "authorization" not in serialized.casefold()


def test_empty_findings_still_produce_valid_sarif() -> None:
    payload = build_sarif(
        _audit([("claim_pass", Relation.ENTAILED, "docs/pass.md", 3)])
    )

    assert payload["version"] == "2.1.0"
    assert payload["runs"][0]["results"] == []
    assert len(payload["runs"][0]["tool"]["driver"]["rules"]) == 6


def test_multiple_findings_have_stable_order_and_byte_output(tmp_path: Path) -> None:
    audit = _audit(
        [
            ("claim_z", Relation.NOT_CHECKABLE, "zeta.md", 4),
            ("claim_b", Relation.NOT_IN_SOURCE, "docs/a.md", 20),
            ("claim_a", Relation.CONTRADICTED, "docs/a.md", 2),
        ]
    )

    first = sarif_bytes(audit)
    second = sarif_bytes(audit)
    assert first == second
    results = json.loads(first)["runs"][0]["results"]
    assert [
        item["locations"][0]["physicalLocation"]["artifactLocation"]["uri"]
        for item in results
    ] == ["docs/a.md", "docs/a.md", "zeta.md"]
    assert [item["ruleId"] for item in results] == [
        "ET1001",
        "ET1002",
        "ET1005",
    ]

    output = write_sarif(audit, Path("reports/audit.sarif"), project_root=tmp_path)
    assert output.read_bytes() == first
    write_sarif(audit, Path("reports/audit.sarif"), project_root=tmp_path)
    assert output.read_bytes() == first
    assert not list(output.parent.glob(".evidencetrace-sarif.*.tmp"))


def test_sarif_reuses_custom_policy_severity() -> None:
    audit = _audit(
        [("claim_custom", Relation.NOT_IN_SOURCE, "docs/custom.md", 6)],
        policy=PolicyConfig(
            fail_on=("contradicted", "not_in_source"),
            warn_on=(),
            notice_on=(),
        ),
    )

    result = build_sarif(audit)["runs"][0]["results"][0]

    assert result["ruleId"] == "ET1002"
    assert result["level"] == "error"
    assert result["properties"]["policySeverity"] == "error"


@pytest.mark.parametrize(
    "output",
    [
        Path("../escaped.sarif"),
        Path("/tmp/evidencetrace-outside.sarif"),
        Path("."),
    ],
)
def test_sarif_rejects_output_traversal_and_invalid_paths(
    tmp_path: Path,
    output: Path,
) -> None:
    audit = _audit([("claim_error", Relation.CONTRADICTED, "docs/error.md", 5)])

    with pytest.raises(SarifError) as caught:
        write_sarif(audit, output, project_root=tmp_path)

    assert caught.value.code == "invalid_output_path"
    assert not (tmp_path.parent / "escaped.sarif").exists()


def test_sarif_reports_unwritable_parent_as_typed_error(tmp_path: Path) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")

    with pytest.raises(SarifError) as caught:
        write_sarif(
            _audit([("claim_error", Relation.CONTRADICTED, "docs/error.md", 5)]),
            Path("blocker/result.sarif"),
            project_root=tmp_path,
        )

    assert caught.value.code == "write_failed"


@pytest.mark.parametrize(
    ("relation", "expected_exit"),
    [
        (Relation.ENTAILED, 0),
        (Relation.CONTRADICTED, 1),
    ],
)
def test_cli_writes_sarif_before_returning_policy_exit_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relation: Relation,
    expected_exit: int,
) -> None:
    document = tmp_path / "doc.md"
    document.write_text("Synthetic cited claim.", encoding="utf-8")
    audit = _audit([("claim_cli", relation, "doc.md", 1)])

    class FakePipeline:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def run(self, _path: Path, **_kwargs: object) -> SimpleNamespace:
            audit_path = tmp_path / "audit.json"
            audit_path.write_text(audit.model_dump_json() + "\n", encoding="utf-8")
            canonical = AuditArtifact.model_validate_json(
                audit_path.read_text(encoding="utf-8")
            )
            return SimpleNamespace(
                audit=canonical,
                audit_path=audit_path,
                terminal="Synthetic terminal",
                product=SimpleNamespace(document_status=DocumentRunStatus.COMPLETE),
            )

    monkeypatch.setattr("evidencetrace.product.ProductAuditPipeline", FakePipeline)
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-test-key")
    monkeypatch.setenv("EVIDENCETRACE_MODEL", "synthetic-model")
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(
        app,
        ["check", "doc.md", "--sarif", "reports/evidencetrace.sarif"],
    )

    assert result.exit_code == expected_exit
    assert "Synthetic terminal" in result.stdout
    sarif_path = tmp_path / "reports/evidencetrace.sarif"
    assert sarif_path.is_file()
    assert json.loads(sarif_path.read_text())["version"] == "2.1.0"


def test_cli_invalid_sarif_path_is_fatal_exit_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = tmp_path / "doc.md"
    document.write_text("Synthetic cited claim.", encoding="utf-8")
    audit = _audit([("claim_cli", Relation.ENTAILED, "doc.md", 1)])

    class FakePipeline:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def run(self, _path: Path, **_kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(
                audit=audit,
                audit_path=tmp_path / "audit.json",
                terminal="terminal",
                product=SimpleNamespace(document_status=DocumentRunStatus.COMPLETE),
            )

    monkeypatch.setattr("evidencetrace.product.ProductAuditPipeline", FakePipeline)
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-test-key")
    monkeypatch.setenv("EVIDENCETRACE_MODEL", "synthetic-model")
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(
        app,
        ["check", "doc.md", "--sarif", "../escaped.sarif"],
    )

    assert result.exit_code == 1
    assert "invalid_output_path" in result.stderr
    assert not (tmp_path.parent / "escaped.sarif").exists()


@pytest.mark.parametrize(
    "fixture",
    [
        "bad-agent-comparison.md",
        "simple-work-document.md",
        "complex-work-document.md",
    ],
)
def test_bad_simple_and_complex_fixtures_generate_sarif(
    tmp_path: Path,
    fixture: str,
) -> None:
    shutil.copytree(ROOT / "examples", tmp_path / "examples")
    manifest = tmp_path / "examples/evidence/source_manifest.json"
    pipeline = CitationAuditPipeline(
        project_root=tmp_path,
        fetcher=SafeFetcher(
            transport=fixture_transport(manifest),
            resolve_dns=False,
            retries=0,
        ),
    )
    audit = pipeline.run(
        tmp_path / "examples" / fixture,
        run_id=f"phase4a-{fixture.removesuffix('.md')}",
    ).audit

    output = write_sarif(
        audit,
        Path(f"reports/{fixture}.sarif"),
        project_root=tmp_path,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["version"] == "2.1.0"
    assert payload["runs"][0]["tool"]["driver"]["name"] == "EvidenceTrace"
    assert all(
        not item["locations"][0]["physicalLocation"]["artifactLocation"][
            "uri"
        ].startswith("/")
        for item in payload["runs"][0]["results"]
    )


def test_github_workflow_uploads_sarif_before_propagating_failure() -> None:
    workflow = ROOT.joinpath(".github/workflows/evidencetrace.yml").read_text(
        encoding="utf-8"
    )

    assert "security-events: write" in workflow
    assert "contents: read" in workflow
    assert "pull-requests: write" not in workflow
    assert "pull_request_target:" not in workflow
    assert "continue-on-error: true" in workflow
    assert "OPENAI_API_KEY:\n        required: false" in workflow
    assert "TAVILY_API_KEY" in workflow
    assert "--discover" in workflow
    assert '--sarif "$SARIF_PATH"' in workflow
    assert "GITHUB_STEP_SUMMARY" in workflow
    assert "always() && steps.check.outputs.sarif_created == 'true'" in workflow
    assert "github/codeql-action/upload-sarif@v3" in workflow
    assert workflow.index("- name: Run EvidenceTrace") < workflow.index(
        "- name: Upload SARIF"
    )
    assert workflow.index("- name: Upload SARIF") < workflow.index(
        "- name: Preserve EvidenceTrace exit status"
    )
