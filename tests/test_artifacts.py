from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from evidencetrace.artifacts import ArtifactError, ArtifactManager
from evidencetrace.models import (
    AuditArtifact,
    EffectiveConfig,
    RunMetadata,
    RunPaths,
)


def metadata(run_id: str = "20260710T120000Z-abcdef0") -> RunMetadata:
    return RunMetadata(
        run_id=run_id,
        started_at=datetime(2026, 7, 10, 12, tzinfo=UTC),
        git_sha="abcdef0",
        tool_version="0.0.0",
    )


def artifact(run: RunMetadata | None = None) -> AuditArtifact:
    return AuditArtifact(
        run=run or metadata(),
        effective_config=EffectiveConfig(),
    )


def test_audit_round_trip_writes_only_canonical_json(tmp_path: Path) -> None:
    manager = ArtifactManager(tmp_path)
    run = metadata()
    paths = manager.create_run(run)

    written = manager.write_audit(paths, artifact(run))

    assert written == paths.audit_json
    assert manager.read_audit(written) == artifact(run)
    assert [path.name for path in paths.run_dir.iterdir()] == ["audit.json"]


def test_existing_run_is_never_reused(tmp_path: Path) -> None:
    manager = ArtifactManager(tmp_path)
    manager.create_run(metadata())

    with pytest.raises(ArtifactError, match="run already exists"):
        manager.create_run(metadata())


def test_existing_audit_is_never_overwritten(tmp_path: Path) -> None:
    manager = ArtifactManager(tmp_path)
    paths = manager.create_run(metadata())
    manager.write_audit(paths, artifact())
    original = paths.audit_json.read_bytes()

    with pytest.raises(ArtifactError, match="refusing to overwrite"):
        manager.write_audit(paths, artifact())

    assert paths.audit_json.read_bytes() == original


@pytest.mark.parametrize("run_id", ["../escape", "/absolute", ".", "..", "a/b"])
def test_run_id_rejects_path_traversal(run_id: str) -> None:
    with pytest.raises(ValidationError):
        metadata(run_id)


def test_runs_root_must_be_project_relative(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="project-relative"):
        ArtifactManager(tmp_path, runs_path=Path("../runs"))


def test_writer_rejects_paths_outside_managed_root(tmp_path: Path) -> None:
    manager = ArtifactManager(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    paths = RunPaths(run_dir=outside, audit_json=outside / "audit.json")

    with pytest.raises(ArtifactError, match="escapes"):
        manager.write_audit(paths, artifact())


def test_read_rejects_noncanonical_filename(tmp_path: Path) -> None:
    manager = ArtifactManager(tmp_path)
    run_dir = manager.create_run(metadata()).run_dir
    other = run_dir / "other.json"
    other.write_text("{}", encoding="utf-8")

    with pytest.raises(ArtifactError, match="named audit.json"):
        manager.read_audit(other)

