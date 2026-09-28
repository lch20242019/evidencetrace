"""Canonical Phase 1 run-artifact management.

Only canonical ``audit.json`` is managed here. Human-readable output and
SARIF are deterministic derivatives owned by their dedicated renderers.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from evidencetrace.models import AuditArtifact, RunMetadata, RunPaths


class ArtifactError(RuntimeError):
    """Raised when an artifact operation would violate the run contract."""


class ArtifactManager:
    """Create isolated run directories and atomically persist ``audit.json``."""

    def __init__(
        self,
        project_root: Path,
        *,
        runs_path: Path = Path(".evidencetrace/runs"),
    ) -> None:
        if runs_path.is_absolute() or ".." in runs_path.parts:
            raise ValueError("runs_path must be a project-relative path")
        self.project_root = project_root.resolve()
        self.runs_root = (self.project_root / runs_path).resolve()
        self._require_inside_runs(self.runs_root)

    def create_run(self, metadata: RunMetadata) -> RunPaths:
        """Create a new run directory without overwriting an earlier run."""

        run_dir = self.runs_root / metadata.run_id
        self._require_inside_runs(run_dir)
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError as error:
            raise ArtifactError(f"run already exists: {metadata.run_id}") from error
        return RunPaths(run_dir=run_dir, audit_json=run_dir / "audit.json")

    def write_audit(self, paths: RunPaths, audit: AuditArtifact) -> Path:
        """Validate and atomically write the canonical artifact exactly once."""

        audit_path = paths.audit_json
        self._require_canonical_path(paths)
        if audit_path.exists():
            raise ArtifactError(
                f"refusing to overwrite canonical artifact: {audit_path}"
            )

        payload = audit.model_dump_json(indent=2) + "\n"
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=paths.run_dir,
                prefix=".audit.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary.write(payload)
                temporary.flush()
                os.fsync(temporary.fileno())
                temporary_path = Path(temporary.name)
            os.replace(temporary_path, audit_path)
        except Exception:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise
        return audit_path

    def read_audit(self, path: Path) -> AuditArtifact:
        """Read and schema-validate a canonical artifact managed by this root."""

        resolved = path.resolve()
        self._require_inside_runs(resolved)
        if resolved.name != "audit.json":
            raise ArtifactError("canonical artifact must be named audit.json")
        return AuditArtifact.model_validate_json(resolved.read_text(encoding="utf-8"))

    def _require_canonical_path(self, paths: RunPaths) -> None:
        run_dir = paths.run_dir.resolve()
        audit_path = paths.audit_json.resolve()
        self._require_inside_runs(run_dir)
        self._require_inside_runs(audit_path)
        if audit_path != run_dir / "audit.json":
            raise ArtifactError("audit_json must be <run_dir>/audit.json")
        if not run_dir.is_dir():
            raise ArtifactError(f"run directory does not exist: {run_dir}")

    def _require_inside_runs(self, path: Path) -> None:
        try:
            path.resolve().relative_to(self.runs_root)
        except ValueError as error:
            raise ArtifactError(f"path escapes the managed run root: {path}") from error


__all__ = ["ArtifactError", "ArtifactManager"]
