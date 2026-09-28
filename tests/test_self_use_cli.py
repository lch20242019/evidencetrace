from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from evidencetrace.cli import app


def test_check_and_fix_help_expose_bounded_self_use_contract() -> None:
    check_help = CliRunner().invoke(app, ["check", "--help"])
    fix_help = CliRunner().invoke(app, ["fix", "--help"])

    assert check_help.exit_code == 0
    assert "{targets}..." in check_help.stdout
    assert "--reference" in check_help.stdout
    assert "--output-dir" in check_help.stdout
    assert "always read-only" in check_help.stdout
    assert fix_help.exit_code == 0
    assert "{targets}..." in fix_help.stdout
    assert "Tavily-only exact evidence" in fix_help.stdout
    assert "--yes" not in fix_help.stdout


def test_non_tty_fix_is_zero_write_and_does_not_initialize_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "target.md"
    target.write_text("Nimbus shipped version 1.2.3.\n", encoding="utf-8")
    before = target.read_bytes()
    constructed = 0

    class ForbiddenClient:
        def __init__(self, **_kwargs: object) -> None:
            nonlocal constructed
            constructed += 1

    monkeypatch.setattr(
        "evidencetrace.model_client.OpenAICompatibleClient",
        ForbiddenClient,
    )
    monkeypatch.setenv("OPENAI_API_KEY", "not-used")
    monkeypatch.setenv("EVIDENCETRACE_MODEL", "not-used")
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(app, ["fix", "target.md"])

    assert result.exit_code == 2
    assert "requires an interactive TTY" in result.stderr
    assert constructed == 0
    assert target.read_bytes() == before
    assert not (tmp_path / ".evidencetrace").exists()


def test_cli_tty_fix_runs_review_and_second_confirmation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "target.md"
    reference = tmp_path / "facts.md"
    target.write_text("Nimbus shipped version 1.2.3.\n", encoding="utf-8")
    reference.write_text("Nimbus shipped version 1.2.4.\n", encoding="utf-8")
    monkeypatch.setattr("evidencetrace.cli._has_interactive_tty", lambda: True)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("EVIDENCETRACE_MODEL", raising=False)
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "fix",
            "target.md",
            "--reference",
            "facts.md",
            "--output-dir",
            "run",
        ],
        input="yes\nyes\n",
    )

    assert result.exit_code == 0, result.stderr
    assert "Apply this repair?" in result.stdout
    assert "Write 1 approved repair" in result.stdout
    assert target.read_text(encoding="utf-8") == (
        "Nimbus shipped version 1.2.4.\n"
    )
    assert (tmp_path / "run" / "batch-manifest.json").is_file()


def test_cli_fix_failure_reports_privacy_safe_recovery_location(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "target.md"
    target.write_text("Nimbus shipped version 1.2.3.\n", encoding="utf-8")

    def fail_after_prepare(*_args: object, **_kwargs: object) -> None:
        raise OSError("simulated failure")

    monkeypatch.setattr("evidencetrace.cli._has_interactive_tty", lambda: True)
    monkeypatch.setattr(
        "evidencetrace.self_use.SelfUseRunner.run_prepared",
        fail_after_prepare,
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("EVIDENCETRACE_MODEL", raising=False)
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(app, ["fix", "target.md"])

    assert result.exit_code == 1
    assert "Recovery artifacts, if created" in result.stderr
    assert ".evidencetrace/runs/" in result.stderr
    assert str(tmp_path.resolve()) not in result.stderr


def test_output_collision_rejects_before_model_initialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "target.md"
    target.write_text("Nimbus shipped version 1.2.3.\n", encoding="utf-8")
    constructed = 0

    class ForbiddenClient:
        def __init__(self, **_kwargs: object) -> None:
            nonlocal constructed
            constructed += 1

    monkeypatch.setattr(
        "evidencetrace.model_client.OpenAICompatibleClient",
        ForbiddenClient,
    )
    monkeypatch.setenv("OPENAI_API_KEY", "not-used")
    monkeypatch.setenv("EVIDENCETRACE_MODEL", "not-used")
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(
        app,
        ["check", "target.md", "--output-dir", "target.md"],
    )

    assert result.exit_code == 1
    assert "output_input_collision" in result.stderr
    assert constructed == 0


def test_legacy_derived_output_collision_rejects_before_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "target.md"
    target.write_text("Nimbus shipped version 1.2.3.\n", encoding="utf-8")
    existing = tmp_path / ".evidencetrace" / "runs" / "occupied"
    existing.mkdir(parents=True)
    constructed = 0

    class ForbiddenClient:
        def __init__(self, **_kwargs: object) -> None:
            nonlocal constructed
            constructed += 1

    monkeypatch.setattr(
        "evidencetrace.model_client.OpenAICompatibleClient",
        ForbiddenClient,
    )
    monkeypatch.setattr("evidencetrace.self_use.new_run_id", lambda: "occupied")
    monkeypatch.setenv("OPENAI_API_KEY", "not-used")
    monkeypatch.setenv("EVIDENCETRACE_MODEL", "not-used")
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(app, ["check", "target.md"])

    assert result.exit_code == 1
    assert "invalid_output_path" in result.stderr
    assert constructed == 0


def test_external_changed_from_is_rejected_before_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    outside = tmp_path / "private"
    root.mkdir()
    outside.mkdir()
    target = outside / "target.md"
    target.write_text("Nimbus shipped version 1.2.3.\n", encoding="utf-8")
    constructed = 0

    class ForbiddenClient:
        def __init__(self, **_kwargs: object) -> None:
            nonlocal constructed
            constructed += 1

    monkeypatch.setattr(
        "evidencetrace.model_client.OpenAICompatibleClient",
        ForbiddenClient,
    )
    monkeypatch.setenv("OPENAI_API_KEY", "not-used")
    monkeypatch.setenv("EVIDENCETRACE_MODEL", "not-used")
    monkeypatch.chdir(root)

    result = CliRunner().invoke(
        app,
        ["check", str(target), "--changed-from", "HEAD~1"],
    )

    assert result.exit_code == 1
    assert "changed_from_requires_internal_markdown" in result.stderr
    assert constructed == 0


def test_multi_target_explicit_file_output_is_rejected_before_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = tmp_path / "first.md"
    second = tmp_path / "second.txt"
    first.write_text("Nimbus shipped 1.2.3.\n", encoding="utf-8")
    second.write_text("Cirrus shipped 2.0.0.\n", encoding="utf-8")
    constructed = 0

    class ForbiddenClient:
        def __init__(self, **_kwargs: object) -> None:
            nonlocal constructed
            constructed += 1

    monkeypatch.setattr(
        "evidencetrace.model_client.OpenAICompatibleClient",
        ForbiddenClient,
    )
    monkeypatch.setenv("OPENAI_API_KEY", "not-used")
    monkeypatch.setenv("EVIDENCETRACE_MODEL", "not-used")
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(
        app,
        ["check", "first.md", "second.txt", "--sarif", "result.sarif"],
    )

    assert result.exit_code == 1
    assert "ambiguous_multi_target_output" in result.stderr
    assert constructed == 0


def test_cli_check_batch_with_shared_reference_is_read_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = tmp_path / "first.md"
    second = tmp_path / "second.txt"
    reference = tmp_path / "facts.md"
    first.write_text("Nimbus shipped version 1.2.3.\n", encoding="utf-8")
    second.write_text("Cirrus shipped version 2.0.0.\n", encoding="utf-8")
    reference.write_text(
        "Nimbus shipped version 1.2.4.\n\n"
        "Cirrus shipped version 2.0.1.\n",
        encoding="utf-8",
    )
    before = (first.read_bytes(), second.read_bytes(), reference.read_bytes())
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("EVIDENCETRACE_MODEL", raising=False)
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "check",
            "first.md",
            "second.txt",
            "--reference",
            "facts.md",
            "--output-dir",
            "run",
        ],
    )

    assert result.exit_code == 0, result.stderr
    assert "Batch status: complete" in result.stdout
    assert (first.read_bytes(), second.read_bytes(), reference.read_bytes()) == before
    assert (tmp_path / "run" / "batch-manifest.json").is_file()
