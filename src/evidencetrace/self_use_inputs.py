"""Deterministic, read-only input preflight for the self-use CLI."""

from __future__ import annotations

import codecs
import hashlib
import os
import re
import stat
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, TypeAlias

MAX_TARGETS = 20
MAX_REFERENCES = 50
MAX_TOTAL_INPUT_BYTES = 5 * 1024 * 1024

InputCommand: TypeAlias = Literal["check", "fix"]
InputKind: TypeAlias = Literal["markdown", "text", "legacy"]
InputRole: TypeAlias = Literal["target", "reference"]
OutputOrigin: TypeAlias = Literal["explicit", "derived"]
InputContractErrorCode: TypeAlias = Literal[
    "invalid_command",
    "invalid_project_root",
    "target_required",
    "target_count_exceeded",
    "reference_count_exceeded",
    "input_path_invalid",
    "input_missing",
    "input_unreadable",
    "input_is_directory",
    "input_is_symlink",
    "input_not_regular",
    "unsupported_target_type",
    "unsupported_reference_type",
    "duplicate_target",
    "duplicate_reference",
    "target_reference_collision",
    "input_size_exceeded",
    "input_not_utf8",
    "input_changed_during_preflight",
    "output_request_invalid",
    "duplicate_output_name",
    "output_path_invalid",
    "output_input_collision",
    "output_collision",
]

_ALLOWED_SUFFIXES = frozenset({".md", ".markdown", ".txt"})
_READ_CHUNK_BYTES = 64 * 1024
_SAFE_NAME_RE = re.compile(r"[^a-z0-9]+")


class InputContractError(ValueError):
    """A path-free, typed rejection raised before any external call."""

    def __init__(
        self,
        code: InputContractErrorCode,
        *,
        role: InputRole | None = None,
        index: int | None = None,
        output_name: str | None = None,
    ) -> None:
        super().__init__(f"self-use input contract rejected ({code})")
        self.code = code
        self.role = role
        self.index = index
        self.output_name = output_name


@dataclass(frozen=True, slots=True)
class InputFileRecord:
    """Canonical in-memory identity plus privacy-safe persisted metadata."""

    path: Path
    display_path: str
    safe_id: str
    content_sha256: str
    size_bytes: int
    kind: InputKind


@dataclass(frozen=True, slots=True)
class OutputPathRequest:
    """One explicit or caller-derived output to canonicalize and validate."""

    name: str
    path: Path
    origin: OutputOrigin = "explicit"


@dataclass(frozen=True, slots=True)
class OutputPathRecord:
    """A canonical output path validated against every input and output."""

    name: str
    path: Path
    origin: OutputOrigin


@dataclass(frozen=True, slots=True)
class SelfUseInputContract:
    """Complete immutable preflight result; constructing it writes nothing."""

    project_root: Path
    command: InputCommand
    targets: tuple[InputFileRecord, ...]
    references: tuple[InputFileRecord, ...]
    total_bytes: int
    outputs: tuple[OutputPathRecord, ...] = ()

    @property
    def is_multi_target(self) -> bool:
        return len(self.targets) > 1


@dataclass(frozen=True, slots=True)
class _InputCandidate:
    path: Path
    role: InputRole
    index: int
    kind: InputKind
    device: int
    inode: int
    size_bytes: int
    modified_ns: int
    changed_ns: int


def _error(
    code: InputContractErrorCode,
    *,
    role: InputRole | None = None,
    index: int | None = None,
    output_name: str | None = None,
) -> InputContractError:
    return InputContractError(
        code,
        role=role,
        index=index,
        output_name=output_name,
    )


def _project_root(value: Path | str) -> Path:
    try:
        root = Path(value).resolve(strict=True)
    except (OSError, RuntimeError, TypeError, ValueError):
        raise _error("invalid_project_root") from None
    if not root.is_dir():
        raise _error("invalid_project_root")
    return root


def _input_kind(path: Path) -> InputKind:
    suffix = path.suffix.casefold()
    if suffix in {".md", ".markdown"}:
        return "markdown"
    if suffix == ".txt":
        return "text"
    return "legacy"


def _candidate(
    value: Path | str,
    *,
    root: Path,
    role: InputRole,
    index: int,
) -> _InputCandidate:
    try:
        requested = Path(value)
    except (TypeError, ValueError):
        raise _error("input_path_invalid", role=role, index=index) from None
    candidate = requested if requested.is_absolute() else root / requested
    try:
        requested_stat = candidate.lstat()
    except FileNotFoundError:
        raise _error("input_missing", role=role, index=index) from None
    except OSError:
        raise _error("input_unreadable", role=role, index=index) from None
    if stat.S_ISLNK(requested_stat.st_mode):
        raise _error("input_is_symlink", role=role, index=index)
    if _has_symlink_component(candidate):
        raise _error("input_is_symlink", role=role, index=index)
    if stat.S_ISDIR(requested_stat.st_mode):
        raise _error("input_is_directory", role=role, index=index)
    if not stat.S_ISREG(requested_stat.st_mode):
        raise _error("input_not_regular", role=role, index=index)
    try:
        canonical = candidate.resolve(strict=True)
        canonical_stat = canonical.stat(follow_symlinks=False)
    except FileNotFoundError:
        raise _error(
            "input_changed_during_preflight",
            role=role,
            index=index,
        ) from None
    except OSError:
        raise _error("input_unreadable", role=role, index=index) from None
    if stat.S_ISLNK(canonical_stat.st_mode):
        raise _error("input_is_symlink", role=role, index=index)
    if not stat.S_ISREG(canonical_stat.st_mode):
        raise _error("input_not_regular", role=role, index=index)
    return _InputCandidate(
        path=canonical,
        role=role,
        index=index,
        kind=_input_kind(candidate),
        device=canonical_stat.st_dev,
        inode=canonical_stat.st_ino,
        size_bytes=canonical_stat.st_size,
        modified_ns=canonical_stat.st_mtime_ns,
        changed_ns=canonical_stat.st_ctime_ns,
    )


def _same_candidate(left: _InputCandidate, right: _InputCandidate) -> bool:
    return left.path == right.path or (
        left.device == right.device and left.inode == right.inode
    )


def _require_unique(
    candidates: tuple[_InputCandidate, ...],
    code: Literal["duplicate_target", "duplicate_reference"],
) -> None:
    for index, candidate in enumerate(candidates):
        if any(
            _same_candidate(candidate, earlier) for earlier in candidates[:index]
        ):
            raise _error(code, role=candidate.role, index=candidate.index)


def _safe_id(
    candidate: _InputCandidate,
    *,
    root: Path,
) -> tuple[str, str]:
    try:
        relative = candidate.path.relative_to(root)
        identity = relative.as_posix()
        display_path: str | None = identity
    except ValueError:
        identity = candidate.path.as_posix()
        display_path = None
    normalized = unicodedata.normalize("NFKD", candidate.path.name)
    ascii_name = normalized.encode("ascii", "ignore").decode("ascii").casefold()
    slug = _SAFE_NAME_RE.sub("-", ascii_name).strip("-")[:40] or "file"
    digest = hashlib.sha256(
        identity.encode("utf-8", errors="surrogatepass")
    ).hexdigest()
    prefix = "target" if candidate.role == "target" else "reference"
    safe_id = f"{prefix}-{slug}-{digest}"
    return safe_id, display_path or f"external/{safe_id}"


def _hash_candidate(
    candidate: _InputCandidate,
    *,
    bytes_already_read: int,
) -> tuple[str, int]:
    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(candidate.path, flags)
    except OSError:
        raise _error(
            "input_changed_during_preflight",
            role=candidate.role,
            index=candidate.index,
        ) from None
    digest = hashlib.sha256()
    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    size_bytes = 0
    try:
        opened_stat = os.fstat(descriptor)
        if not stat.S_ISREG(opened_stat.st_mode):
            raise _error(
                "input_not_regular",
                role=candidate.role,
                index=candidate.index,
            )
        if (
            opened_stat.st_dev != candidate.device
            or opened_stat.st_ino != candidate.inode
        ):
            raise _error(
                "input_changed_during_preflight",
                role=candidate.role,
                index=candidate.index,
            )
        while chunk := os.read(descriptor, _READ_CHUNK_BYTES):
            size_bytes += len(chunk)
            if bytes_already_read + size_bytes > MAX_TOTAL_INPUT_BYTES:
                raise _error(
                    "input_size_exceeded",
                    role=candidate.role,
                    index=candidate.index,
                )
            digest.update(chunk)
            try:
                decoder.decode(chunk)
            except UnicodeDecodeError:
                raise _error(
                    "input_not_utf8",
                    role=candidate.role,
                    index=candidate.index,
                ) from None
        try:
            decoder.decode(b"", final=True)
        except UnicodeDecodeError:
            raise _error(
                "input_not_utf8",
                role=candidate.role,
                index=candidate.index,
            ) from None
        finished_stat = os.fstat(descriptor)
    except OSError:
        raise _error(
            "input_unreadable",
            role=candidate.role,
            index=candidate.index,
        ) from None
    finally:
        os.close(descriptor)
    if (
        size_bytes != candidate.size_bytes
        or finished_stat.st_size != size_bytes
        or finished_stat.st_mtime_ns != candidate.modified_ns
        or finished_stat.st_ctime_ns != candidate.changed_ns
    ):
        raise _error(
            "input_changed_during_preflight",
            role=candidate.role,
            index=candidate.index,
        )
    return digest.hexdigest(), size_bytes


def _records(
    candidates: tuple[_InputCandidate, ...],
    *,
    root: Path,
    bytes_already_read: int,
) -> tuple[tuple[InputFileRecord, ...], int]:
    records: list[InputFileRecord] = []
    total = bytes_already_read
    for candidate in candidates:
        content_sha256, size_bytes = _hash_candidate(
            candidate,
            bytes_already_read=total,
        )
        safe_id, display_path = _safe_id(candidate, root=root)
        records.append(
            InputFileRecord(
                path=candidate.path,
                display_path=display_path,
                safe_id=safe_id,
                content_sha256=content_sha256,
                size_bytes=size_bytes,
                kind=candidate.kind,
            )
        )
        total += size_bytes
    return tuple(records), total


def _same_path(left: Path, right: Path) -> bool:
    if left == right:
        return True
    try:
        return os.path.samefile(left, right)
    except OSError:
        return False


def _has_symlink_component(path: Path) -> bool:
    """Reject an existing symlink anywhere in an output's lexical path."""

    current = path
    while True:
        try:
            if current.is_symlink():
                return True
        except OSError:
            return True
        if current == current.parent:
            return False
        current = current.parent


def validate_output_paths(
    contract: SelfUseInputContract,
    requests: Iterable[OutputPathRequest],
) -> tuple[OutputPathRecord, ...]:
    """Canonicalize outputs and reject all input/output alias collisions."""

    records: list[OutputPathRecord] = []
    names: set[str] = set()
    input_paths = tuple(
        item.path for item in (*contract.targets, *contract.references)
    )
    for request in requests:
        if (
            not isinstance(request, OutputPathRequest)
            or not isinstance(request.name, str)
            or not request.name
            or request.origin not in {"explicit", "derived"}
        ):
            raise _error("output_request_invalid")
        if request.name in names:
            raise _error("duplicate_output_name", output_name=request.name)
        names.add(request.name)
        try:
            requested = Path(request.path)
            candidate = (
                requested
                if requested.is_absolute()
                else contract.project_root / requested
            )
            has_symlink_component = _has_symlink_component(candidate)
            canonical = candidate.resolve(strict=False)
        except (OSError, RuntimeError, TypeError, ValueError):
            raise _error(
                "output_path_invalid",
                output_name=request.name,
            ) from None
        if any(_same_path(canonical, path) for path in input_paths):
            raise _error(
                "output_input_collision",
                output_name=request.name,
            )
        if has_symlink_component:
            raise _error(
                "output_path_invalid",
                output_name=request.name,
            )
        if any(_same_path(canonical, item.path) for item in records):
            raise _error("output_collision", output_name=request.name)
        records.append(
            OutputPathRecord(
                name=request.name,
                path=canonical,
                origin=request.origin,
            )
        )
    return tuple(records)


def preflight_self_use_inputs(
    targets: Sequence[Path | str],
    *,
    references: Sequence[Path | str] = (),
    command: InputCommand = "check",
    project_root: Path | str,
    output_paths: Iterable[OutputPathRequest] = (),
) -> SelfUseInputContract:
    """Validate and hash every input without writing or calling providers."""

    if command not in {"check", "fix"}:
        raise _error("invalid_command")
    target_values = tuple(targets)
    reference_values = tuple(references)
    if not target_values:
        raise _error("target_required")
    if len(target_values) > MAX_TARGETS:
        raise _error("target_count_exceeded")
    if len(reference_values) > MAX_REFERENCES:
        raise _error("reference_count_exceeded")
    root = _project_root(project_root)
    target_candidates = tuple(
        _candidate(value, root=root, role="target", index=index)
        for index, value in enumerate(target_values)
    )
    reference_candidates = tuple(
        _candidate(value, root=root, role="reference", index=index)
        for index, value in enumerate(reference_values)
    )
    strict_targets = command == "fix" or len(target_candidates) > 1
    for candidate in target_candidates:
        if strict_targets and candidate.kind == "legacy":
            raise _error(
                "unsupported_target_type",
                role="target",
                index=candidate.index,
            )
    for candidate in reference_candidates:
        if candidate.kind == "legacy":
            raise _error(
                "unsupported_reference_type",
                role="reference",
                index=candidate.index,
            )
    _require_unique(target_candidates, "duplicate_target")
    _require_unique(reference_candidates, "duplicate_reference")
    for target in target_candidates:
        if any(
            _same_candidate(target, reference)
            for reference in reference_candidates
        ):
            raise _error(
                "target_reference_collision",
                role="target",
                index=target.index,
            )
    all_candidates = (*target_candidates, *reference_candidates)
    if sum(item.size_bytes for item in all_candidates) > MAX_TOTAL_INPUT_BYTES:
        raise _error("input_size_exceeded")
    target_records, total = _records(
        target_candidates,
        root=root,
        bytes_already_read=0,
    )
    reference_records, total = _records(
        reference_candidates,
        root=root,
        bytes_already_read=total,
    )
    contract = SelfUseInputContract(
        project_root=root,
        command=command,
        targets=target_records,
        references=reference_records,
        total_bytes=total,
    )
    return replace(
        contract,
        outputs=validate_output_paths(contract, output_paths),
    )


__all__ = [
    "MAX_REFERENCES",
    "MAX_TARGETS",
    "MAX_TOTAL_INPUT_BYTES",
    "InputCommand",
    "InputContractError",
    "InputContractErrorCode",
    "InputFileRecord",
    "InputKind",
    "OutputOrigin",
    "OutputPathRecord",
    "OutputPathRequest",
    "SelfUseInputContract",
    "preflight_self_use_inputs",
    "validate_output_paths",
]
