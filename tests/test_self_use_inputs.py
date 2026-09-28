from __future__ import annotations

import hashlib
import os
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from evidencetrace.self_use_inputs import (
    MAX_REFERENCES,
    MAX_TARGETS,
    MAX_TOTAL_INPUT_BYTES,
    InputContractError,
    OutputPathRequest,
    preflight_self_use_inputs,
    validate_output_paths,
)


def _assert_code(error: pytest.ExceptionInfo[InputContractError], code: str) -> None:
    assert error.value.code == code


def test_batch_preflight_records_immutable_safe_metadata_without_writes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    first = root / "docs" / "first.md"
    first.parent.mkdir()
    first.write_text("Version 1.2.3\n", encoding="utf-8")
    second = root / "notes.txt"
    second.write_text("Launched 2026-07-28\n", encoding="utf-8")
    reference = root / "facts.markdown"
    reference.write_text("Version 1.2.4\n", encoding="utf-8")
    run_root = root / "artifacts" / "run-1"

    contract = preflight_self_use_inputs(
        [first, second],
        references=[reference],
        command="check",
        project_root=root,
        output_paths=(
            OutputPathRequest("run_root", run_root),
            OutputPathRequest(
                "first_audit",
                run_root / "targets" / "first" / "audit.json",
                origin="derived",
            ),
        ),
    )

    assert contract.is_multi_target is True
    assert contract.total_bytes == sum(
        path.stat().st_size for path in (first, second, reference)
    )
    assert [item.kind for item in contract.targets] == ["markdown", "text"]
    assert contract.references[0].kind == "markdown"
    assert contract.targets[0].display_path == "docs/first.md"
    assert contract.targets[1].display_path == "notes.txt"
    assert contract.targets[0].content_sha256 == hashlib.sha256(
        first.read_bytes()
    ).hexdigest()
    assert all(item.path.is_absolute() for item in contract.targets)
    assert all("/" not in item.safe_id for item in contract.targets)
    assert contract.outputs[0].path == run_root.resolve()
    assert contract.outputs[1].origin == "derived"
    assert not run_root.exists()
    with pytest.raises(FrozenInstanceError):
        contract.total_bytes = 0  # type: ignore[misc]


def test_output_path_rejects_existing_symlink_component(tmp_path: Path) -> None:
    target = tmp_path / "target.md"
    outside = tmp_path / "outside"
    linked = tmp_path / "run"
    target.write_text("Version 1.2.3\n", encoding="utf-8")
    outside.mkdir()
    linked.symlink_to(outside, target_is_directory=True)
    contract = preflight_self_use_inputs(
        [target],
        command="check",
        project_root=tmp_path,
    )

    with pytest.raises(InputContractError) as caught:
        validate_output_paths(
            contract,
            (
                OutputPathRequest(
                    "audit",
                    linked / "audit.json",
                    origin="derived",
                ),
            ),
        )

    _assert_code(caught, "output_path_invalid")


@pytest.mark.parametrize("name", ["README", "changes.diff", "architecture.adoc"])
def test_single_target_legacy_check_keeps_extension_compatibility(
    tmp_path: Path,
    name: str,
) -> None:
    target = tmp_path / name
    target.write_text("legacy check input\n", encoding="utf-8")

    contract = preflight_self_use_inputs(
        [target],
        command="check",
        project_root=tmp_path,
    )

    assert contract.targets[0].kind == "legacy"


def test_multi_target_and_fix_restrict_target_extensions(tmp_path: Path) -> None:
    legacy = tmp_path / "changes.diff"
    markdown = tmp_path / "doc.md"
    legacy.write_text("diff --git a/a b/a\n", encoding="utf-8")
    markdown.write_text("A fact.\n", encoding="utf-8")

    with pytest.raises(InputContractError) as multi_error:
        preflight_self_use_inputs(
            [legacy, markdown],
            command="check",
            project_root=tmp_path,
        )
    _assert_code(multi_error, "unsupported_target_type")

    with pytest.raises(InputContractError) as fix_error:
        preflight_self_use_inputs(
            [legacy],
            command="fix",
            project_root=tmp_path,
        )
    _assert_code(fix_error, "unsupported_target_type")


def test_references_always_restrict_extensions(tmp_path: Path) -> None:
    target = tmp_path / "README"
    reference = tmp_path / "facts.pdf"
    target.write_text("A fact.\n", encoding="utf-8")
    reference.write_text("Not really a PDF.\n", encoding="utf-8")

    with pytest.raises(InputContractError) as caught:
        preflight_self_use_inputs(
            [target],
            references=[reference],
            command="check",
            project_root=tmp_path,
        )

    _assert_code(caught, "unsupported_reference_type")


@pytest.mark.parametrize(
    ("targets", "references", "expected"),
    [
        ([], [], "target_required"),
        ([Path("missing")] * (MAX_TARGETS + 1), [], "target_count_exceeded"),
        (
            [Path("missing")],
            [Path("missing")] * (MAX_REFERENCES + 1),
            "reference_count_exceeded",
        ),
    ],
)
def test_count_limits_are_checked_before_filesystem_reads(
    tmp_path: Path,
    targets: list[Path],
    references: list[Path],
    expected: str,
) -> None:
    with pytest.raises(InputContractError) as caught:
        preflight_self_use_inputs(
            targets,
            references=references,
            command="check",
            project_root=tmp_path,
        )

    _assert_code(caught, expected)


def test_combined_input_limit_allows_exactly_five_mib(tmp_path: Path) -> None:
    target = tmp_path / "exact.md"
    target.write_bytes(b"x" * MAX_TOTAL_INPUT_BYTES)

    contract = preflight_self_use_inputs(
        [target],
        command="check",
        project_root=tmp_path,
    )

    assert contract.total_bytes == MAX_TOTAL_INPUT_BYTES


def test_combined_input_limit_rejects_one_extra_byte(tmp_path: Path) -> None:
    target = tmp_path / "too-large.md"
    target.write_bytes(b"x" * (MAX_TOTAL_INPUT_BYTES + 1))

    with pytest.raises(InputContractError) as caught:
        preflight_self_use_inputs(
            [target],
            command="check",
            project_root=tmp_path,
        )

    _assert_code(caught, "input_size_exceeded")


def test_preflight_rejects_non_utf8_before_returning_metadata(tmp_path: Path) -> None:
    target = tmp_path / "binary.txt"
    target.write_bytes(b"\xff\xfe")

    with pytest.raises(InputContractError) as caught:
        preflight_self_use_inputs(
            [target],
            command="check",
            project_root=tmp_path,
        )

    _assert_code(caught, "input_not_utf8")


def test_preflight_distinguishes_missing_directory_symlink_and_fifo(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "directory.md"
    directory.mkdir()
    real = tmp_path / "real.md"
    real.write_text("A fact.\n", encoding="utf-8")
    symlink = tmp_path / "link.md"
    symlink.symlink_to(real)
    fifo = tmp_path / "pipe.txt"
    os.mkfifo(fifo)

    cases = (
        (tmp_path / "missing.md", "input_missing"),
        (directory, "input_is_directory"),
        (symlink, "input_is_symlink"),
        (fifo, "input_not_regular"),
    )
    for path, expected in cases:
        with pytest.raises(InputContractError) as caught:
            preflight_self_use_inputs(
                [path],
                command="check",
                project_root=tmp_path,
            )
        _assert_code(caught, expected)


def test_preflight_rejects_symlinked_input_ancestor(tmp_path: Path) -> None:
    real_directory = tmp_path / "real"
    real_directory.mkdir()
    target = real_directory / "target.md"
    target.write_text("A fact.\n", encoding="utf-8")
    alias = tmp_path / "alias"
    alias.symlink_to(real_directory, target_is_directory=True)

    with pytest.raises(InputContractError) as caught:
        preflight_self_use_inputs(
            [alias / "target.md"],
            command="check",
            project_root=tmp_path,
        )

    _assert_code(caught, "input_is_symlink")


def test_duplicate_targets_use_canonical_paths_and_inode_identity(
    tmp_path: Path,
) -> None:
    target = tmp_path / "doc.md"
    target.write_text("A fact.\n", encoding="utf-8")
    subdirectory = tmp_path / "subdirectory"
    subdirectory.mkdir()

    with pytest.raises(InputContractError) as canonical_error:
        preflight_self_use_inputs(
            [target, subdirectory / ".." / "doc.md"],
            command="check",
            project_root=tmp_path,
        )
    _assert_code(canonical_error, "duplicate_target")

    hardlink = tmp_path / "hardlink.md"
    os.link(target, hardlink)
    with pytest.raises(InputContractError) as inode_error:
        preflight_self_use_inputs(
            [target, hardlink],
            command="check",
            project_root=tmp_path,
        )
    _assert_code(inode_error, "duplicate_target")


def test_duplicate_references_and_target_reference_collisions_are_distinct(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.md"
    reference = tmp_path / "reference.md"
    target.write_text("Target.\n", encoding="utf-8")
    reference.write_text("Reference.\n", encoding="utf-8")

    with pytest.raises(InputContractError) as duplicate:
        preflight_self_use_inputs(
            [target],
            references=[reference, reference],
            command="check",
            project_root=tmp_path,
        )
    _assert_code(duplicate, "duplicate_reference")

    with pytest.raises(InputContractError) as cross_role:
        preflight_self_use_inputs(
            [target],
            references=[target],
            command="check",
            project_root=tmp_path,
        )
    _assert_code(cross_role, "target_reference_collision")


def test_external_display_paths_and_ids_do_not_leak_absolute_parents(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    first_parent = tmp_path / "private-one"
    second_parent = tmp_path / "private-two"
    first_parent.mkdir()
    second_parent.mkdir()
    first = first_parent / "same-name.md"
    second = second_parent / "same-name.md"
    first.write_text("First.\n", encoding="utf-8")
    second.write_text("Second.\n", encoding="utf-8")

    contract = preflight_self_use_inputs(
        [first, second],
        command="check",
        project_root=root,
    )

    assert contract.targets[0].safe_id != contract.targets[1].safe_id
    for item in contract.targets:
        assert item.display_path == f"external/{item.safe_id}"
        assert not Path(item.display_path).is_absolute()
        assert str(first_parent.resolve()) not in item.display_path
        assert str(second_parent.resolve()) not in item.display_path


def test_outputs_reject_direct_symlink_hardlink_and_output_alias_collisions(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.md"
    target.write_text("A fact.\n", encoding="utf-8")
    contract = preflight_self_use_inputs(
        [target],
        command="check",
        project_root=tmp_path,
    )

    with pytest.raises(InputContractError) as direct:
        validate_output_paths(
            contract,
            [OutputPathRequest("sarif", target)],
        )
    _assert_code(direct, "output_input_collision")

    symlink = tmp_path / "target-alias.sarif"
    symlink.symlink_to(target)
    with pytest.raises(InputContractError) as symlink_error:
        validate_output_paths(
            contract,
            [OutputPathRequest("sarif", symlink)],
        )
    _assert_code(symlink_error, "output_input_collision")

    hardlink = tmp_path / "target-hardlink.sarif"
    os.link(target, hardlink)
    with pytest.raises(InputContractError) as hardlink_error:
        validate_output_paths(
            contract,
            [OutputPathRequest("sarif", hardlink)],
        )
    _assert_code(hardlink_error, "output_input_collision")

    reports = tmp_path / "reports"
    with pytest.raises(InputContractError) as outputs:
        validate_output_paths(
            contract,
            (
                OutputPathRequest("explicit", reports / "result.json"),
                OutputPathRequest(
                    "derived",
                    reports / ".." / "reports" / "result.json",
                    origin="derived",
                ),
            ),
        )
    _assert_code(outputs, "output_collision")
    assert not reports.exists()


def test_preflight_rejects_duplicate_output_names(tmp_path: Path) -> None:
    target = tmp_path / "target.md"
    target.write_text("A fact.\n", encoding="utf-8")

    with pytest.raises(InputContractError) as caught:
        preflight_self_use_inputs(
            [target],
            command="check",
            project_root=tmp_path,
            output_paths=(
                OutputPathRequest("audit", Path("one.json")),
                OutputPathRequest("audit", Path("two.json"), origin="derived"),
            ),
        )

    _assert_code(caught, "duplicate_output_name")
    assert caught.value.output_name == "audit"
    assert not (tmp_path / "one.json").exists()
    assert not (tmp_path / "two.json").exists()


def test_error_text_never_contains_external_absolute_input_path(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    missing = tmp_path / "private" / "missing.md"

    with pytest.raises(InputContractError) as caught:
        preflight_self_use_inputs(
            [missing],
            command="check",
            project_root=root,
        )

    _assert_code(caught, "input_missing")
    assert str(missing) not in str(caught.value)
