from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture()
def runtime(tmp_path, monkeypatch):
    script = Path(__file__).parents[1] / "scripts/check_scifact_gpu_runtime.py"
    spec = importlib.util.spec_from_file_location("scifact_gpu_runtime", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / "eval_runs").mkdir()
    monkeypatch.setattr(module, "PROJECT", tmp_path.resolve())
    return module


@pytest.mark.parametrize("name", [
    "../train_escape", "C:/train", "train/dev", "trial_dev_train",
    "train_test", "training", "train.", "train:stream", "train\\escape",
])
def test_output_rejects_non_train_or_escaping_paths(runtime, name):
    with pytest.raises(ValueError, match="train-only basename"):
        runtime._output_paths(name)


def test_output_reserves_independent_audit_and_rejects_reuse(runtime):
    output, audit, native_log = runtime._output_paths("selector_train_v1")
    assert audit.parent == output.parent == native_log.parent
    assert audit.name == "selector_train_v1.runtime.json"
    audit.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="must all be new"):
        runtime._output_paths("selector_train_v1")


def test_wrong_interpreter_is_rejected_before_package_import(runtime, monkeypatch):
    monkeypatch.setattr(runtime, "sys", SimpleNamespace(
        platform="win32", executable=str(runtime.PROJECT / "wrong-python.exe"),
    ))
    with pytest.raises(RuntimeError, match="wrong interpreter"):
        runtime._verify_environment()


def test_cpu_torch_package_is_rejected_before_native_import(runtime, monkeypatch):
    monkeypatch.setattr(runtime, "sys", SimpleNamespace(
        platform="win32", executable=str(runtime.PYTHON), version_info=(3, 9, 2),
    ))
    monkeypatch.setattr(runtime.importlib.metadata, "version", lambda name: (
        "2.2.0+cpu" if name == "torch" else runtime.EXPECTED_VERSIONS[name]
    ))
    with pytest.raises(RuntimeError, match=r"package version mismatch.*2\.2\.0\+cpu"):
        runtime._verify_environment()


def test_failed_preflight_persists_stage_without_creating_run(runtime, monkeypatch):
    def reject():
        raise RuntimeError("wrong interpreter for test")

    monkeypatch.setattr(runtime, "_verify_environment", reject)
    # This failure is exercised before any Windows or ML native calls.
    assert runtime.run("selector_train_failure") == 1
    report = json.loads(
        (runtime.PROJECT / "eval_runs/selector_train_failure.runtime.json")
        .read_text(encoding="utf-8")
    )
    assert report["status"] == "failed"
    assert report["stage"] == "verify_interpreter_and_packages"
    assert report["error"]["message"] == "wrong interpreter for test"
    assert not (runtime.PROJECT / "eval_runs/selector_train_failure").exists()


def test_model_snapshot_tampering_is_rejected(runtime, monkeypatch, tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("changed", encoding="utf-8")
    monkeypatch.setattr(runtime, "MODEL", model)
    with pytest.raises(RuntimeError, match="snapshot identity mismatch"):
        runtime._model_identity()


def test_preflight_module_does_not_import_ml_dependencies(runtime):
    # Importing the checker is stdlib-only, so wrong-interpreter checks remain safe.
    assert not any(
        name in runtime.__dict__ for name in ("torch", "numpy", "sklearn", "scipy")
    )
    assert runtime.sys is sys
