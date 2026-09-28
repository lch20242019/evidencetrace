from __future__ import annotations

from typer.testing import CliRunner

from evidencetrace import __version__
from evidencetrace.cli import app


def test_version_is_exposed_by_package_and_cli() -> None:
    result = CliRunner().invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == __version__


def test_phase_two_cli_advertises_check_and_demo() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "check" in result.stdout.lower()
    assert "demo" in result.stdout.lower()

