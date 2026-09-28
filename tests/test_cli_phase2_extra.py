from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from evidencetrace.cli import app
from evidencetrace.model_client import (
    OpenAICompatibleClient,
    SchemaRecoveryClient,
)
from evidencetrace.models import AuditArtifact, EffectiveConfig, RunMetadata
from evidencetrace.product import (
    PRODUCT_MAX_PROVIDER_ATTEMPTS,
    DocumentRunStatus,
)


def test_cli_check_success_path_uses_configured_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = tmp_path / "doc.md"
    document.write_text(
        "Nimbus shipped v1.3 [source](https://example.test/release).",
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret")
    monkeypatch.setenv("EVIDENCETRACE_MODEL", "test-model")
    audit = AuditArtifact(
        run=RunMetadata(
            run_id="configured-cli-fixture",
            started_at=datetime(2033, 2, 3, tzinfo=UTC),
            tool_version="0.0.0",
        ),
        effective_config=EffectiveConfig(),
    )
    seen: dict[str, object] = {}

    class FakePipeline:
        def __init__(self, **kwargs: object) -> None:
            seen.update(kwargs)

        def run(self, _path: Path, **kwargs: object) -> SimpleNamespace:
            seen.update(kwargs)
            return SimpleNamespace(
                audit=audit,
                audit_path=tmp_path / "audit.json",
                terminal="Verdict: entailed",
                product=SimpleNamespace(document_status=DocumentRunStatus.COMPLETE),
            )

    monkeypatch.setattr("evidencetrace.product.ProductAuditPipeline", FakePipeline)
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(
        app, ["check", str(document), "--changed-from", "HEAD~1"]
    )

    assert result.exit_code == 0, result.stderr
    assert "Verdict: entailed" in result.stdout
    assert "test-secret" not in result.stdout + result.stderr
    model = seen["model"]
    assert isinstance(model, SchemaRecoveryClient)
    assert isinstance(model.wrapped, OpenAICompatibleClient)
    assert model.wrapped.max_calls == PRODUCT_MAX_PROVIDER_ATTEMPTS
    assert seen["changed_from"] == "HEAD~1"
