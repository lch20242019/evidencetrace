"""Preflight the fixed Windows GPU environment; never read claim labels or run OOF.

This checks one controlled project entry point, not the system-wide MSVC runtime.
Only the adjacent PowerShell entry point starts the unchanged train selector.
"""

from __future__ import annotations

# This entry point must work in the pinned Python 3.9 environment.
# ruff: noqa: UP017, UP045
import argparse
import ctypes
import faulthandler
import hashlib
import importlib.metadata
import json
import os
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
PYTHON = WORKSPACE / "env/scifact-verisci-py39-gpu/Scripts/python.exe"
SELECTOR = PROJECT / "scripts/calibrate_scifact_sentence_selector.py"
SELECTOR_SHA256 = "9d11dcc3575af86c33055b941cfaefb29b2d5ca287e09a15662a725a6ffe848a"
MODEL = WORKSPACE / (
    "env/models/MoritzLaurer--DeBERTa-v3-base-mnli-fever-anli/"
    "6f5cf0a2b59cabb106aca4c287eed12e357e90eb"
)
MODEL_IDENTITY_SHA256 = (
    "9194298eb4045bb0a48daa2c01ecadfa981fe0300fa87c5975c18f5ec972a5e2"
)
EXPECTED_VERSIONS = {
    "torch": "2.2.0+cu118",
    "transformers": "4.49.0",
    "scikit-learn": "1.4.2",
    "numpy": "1.26.4",
    "scipy": "1.13.0",
    "tokenizers": "0.21.0",
    "safetensors": "0.5.3",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _output_paths(name: str) -> tuple[Path, Path, Path]:
    if (
        not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name)
        or not re.search(r"(^|[_.-])train([_.-]|$)", name, re.IGNORECASE)
        or re.search(r"(^|[_.-])(dev|test)([_.-]|$)", name, re.IGNORECASE)
        or name.endswith(".")
    ):
        raise ValueError("output must be a train-only basename, not a path")
    root = PROJECT / "eval_runs"
    if not root.is_dir() or root.resolve() != root:
        raise ValueError("eval_runs must be an existing project-local directory")
    output = root / name
    audit = root / (name + ".runtime.json")
    native_log = root / (name + ".runtime.native.log")
    if any(path.exists() for path in (output, audit, native_log)):
        raise ValueError("output and runtime audit paths must all be new")
    return output, audit, native_log


def _verify_environment() -> dict[str, Any]:
    if sys.platform != "win32":
        raise RuntimeError("the controlled runtime requires Windows")
    if Path(sys.executable).resolve() != PYTHON.resolve():
        raise RuntimeError("wrong interpreter: " + sys.executable)
    if sys.version_info[:2] != (3, 9):
        raise RuntimeError("the controlled runtime requires Python 3.9")
    versions = {
        name: importlib.metadata.version(name) for name in EXPECTED_VERSIONS
    }
    mismatches = {
        name: {"expected": version, "actual": versions[name]}
        for name, version in EXPECTED_VERSIONS.items()
        if versions[name] != version
    }
    if mismatches:
        raise RuntimeError("package version mismatch: " + json.dumps(mismatches))
    return {
        "executable": str(Path(sys.executable).resolve()),
        "python": sys.version,
        "prefix": sys.prefix,
        "base_prefix": sys.base_prefix,
        "packages": versions,
    }


def _native_error_mode() -> int:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.SetErrorMode.argtypes = [ctypes.c_uint]
    kernel32.SetErrorMode.restype = ctypes.c_uint
    # SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX.
    return int(kernel32.SetErrorMode(0x8003))


def _loaded_native_modules() -> list[str]:
    """Read loaded module paths from this process; do not infer DLL resolution."""
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    enum = kernel32.K32EnumProcessModules
    enum.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.HMODULE),
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    enum.restype = wintypes.BOOL
    filename = kernel32.GetModuleFileNameW
    filename.argtypes = [wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD]
    filename.restype = wintypes.DWORD
    capacity = 256
    while True:
        modules = (wintypes.HMODULE * capacity)()
        needed = wintypes.DWORD()
        if not enum(
            kernel32.GetCurrentProcess(), modules, ctypes.sizeof(modules),
            ctypes.byref(needed),
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        if needed.value <= ctypes.sizeof(modules):
            break
        capacity = needed.value // ctypes.sizeof(wintypes.HMODULE) + 16
    paths = []
    for module in modules[: needed.value // ctypes.sizeof(wintypes.HMODULE)]:
        buffer = ctypes.create_unicode_buffer(32768)
        size = filename(module, buffer, len(buffer))
        if not size or size == len(buffer):
            raise ctypes.WinError(ctypes.get_last_error())
        if buffer.value.lower().endswith((".dll", ".pyd")):
            paths.append(buffer.value)
    return sorted(set(paths), key=str.casefold)


def _model_identity() -> dict[str, Any]:
    files = {
        path.relative_to(MODEL).as_posix(): {
            "sha256": _sha256(path), "bytes": path.stat().st_size,
        }
        for path in sorted(MODEL.rglob("*"))
        if path.is_file() and ".cache" not in path.parts
    }
    identity = hashlib.sha256(
        json.dumps(files, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        .encode("utf-8")
    ).hexdigest()
    if identity != MODEL_IDENTITY_SHA256:
        raise RuntimeError("pinned DeBERTa snapshot identity mismatch")
    return {"path": str(MODEL), "identity_sha256": identity, "files": files}


def _save(path: Path, report: dict[str, Any], *, exclusive: bool = False) -> None:
    with path.open("x" if exclusive else "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def run(name: str) -> int:
    output, audit, native_log = _output_paths(name)
    started = time.perf_counter()
    report: dict[str, Any] = {
        "schema_version": "scifact-gpu-runtime-preflight-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "scope": "controlled_project_gpu_environment_not_system_runtime_repair",
        "output_directory": str(output),
        "native_log": str(native_log),
        "actual_executable": sys.executable,
        "stages": [],
        "loaded_native_modules": {},
    }
    _save(audit, report, exclusive=True)

    def checkpoint(stage: str, *, modules: bool = False) -> None:
        report["stage"] = stage
        report["elapsed_seconds"] = time.perf_counter() - started
        report["stages"].append({
            "stage": stage, "elapsed_seconds": report["elapsed_seconds"],
        })
        if modules:
            report["loaded_native_modules"][stage] = _loaded_native_modules()
        _save(audit, report)
        print("[runtime] " + stage, flush=True)

    with native_log.open("x", encoding="utf-8") as fault_file:
        try:
            checkpoint("verify_interpreter_and_packages")
            report["environment"] = _verify_environment()
            report["previous_error_mode"] = _native_error_mode()
            faulthandler.enable(file=fault_file, all_threads=True)
            checkpoint("verify_source_and_model", modules=True)
            source_hash = _sha256(SELECTOR)
            if source_hash != SELECTOR_SHA256:
                raise RuntimeError("selector source differs from the audited version")
            report["sources"] = {
                str(path): _sha256(path)
                for path in (SELECTOR, Path(__file__).resolve(),
                             PROJECT / "scripts/run_scifact_sentence_selector.ps1")
            }
            report["model"] = _model_identity()
            checkpoint("import_torch_and_transformers")
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            report["torch_cuda"] = {
                "available": torch.cuda.is_available(),
                "build": torch.version.cuda,
                "torch_module": torch.__file__,
            }
            if not report["torch_cuda"]["available"] or torch.version.cuda != "11.8":
                raise RuntimeError("CUDA 11.8 build and an available GPU are required")
            report["torch_cuda"]["device_name"] = torch.cuda.get_device_name(0)
            checkpoint("load_tokenizer", modules=True)
            tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
            checkpoint("tokenize_synthetic_batch", modules=True)
            premise = "A controlled experiment measured cell growth after treatment."
            hypothesis = "Cell growth was measured in the experiment."
            encoded = tokenizer(
                [premise] * 8, [hypothesis] * 8, padding=True, truncation=True,
                max_length=384, return_tensors="pt",
            )
            checkpoint("load_model_float16_cuda", modules=True)
            model = AutoModelForSequenceClassification.from_pretrained(
                str(MODEL), local_files_only=True, torch_dtype=torch.float16,
            ).eval().to("cuda")
            checkpoint("synthetic_forward_batch8", modules=True)
            torch.cuda.synchronize()
            forward_started = time.perf_counter()
            with torch.inference_mode():
                logits = model(**{
                    key: value.to("cuda") for key, value in encoded.items()
                }).logits.float()
                probabilities = torch.softmax(logits, dim=-1)
            torch.cuda.synchronize()
            finite = bool(torch.isfinite(probabilities).all().item())
            report["synthetic_forward"] = {
                "batch_size": 8, "max_length": 384, "dtype": "torch.float16",
                "input_shape": list(encoded["input_ids"].shape),
                "output_shape": list(probabilities.shape), "finite": finite,
                "elapsed_seconds": time.perf_counter() - forward_started,
                "premise": premise, "hypothesis": hypothesis,
                "evaluation_data_used": False,
            }
            if list(probabilities.shape) != [8, 3] or not finite:
                raise RuntimeError(
                    "synthetic NLI forward returned invalid probabilities"
                )
            checkpoint("import_oof_dependencies", modules=True)
            import numpy
            import scipy
            import sklearn

            report["numerical_module_paths"] = {
                "numpy": numpy.__file__, "scipy": scipy.__file__,
                "sklearn": sklearn.__file__,
            }
            report["status"] = "passed"
            checkpoint("preflight_passed", modules=True)
            return 0
        except BaseException as exc:
            report["status"] = "failed"
            report["error"] = {"type": type(exc).__name__, "message": str(exc)}
            report["elapsed_seconds"] = time.perf_counter() - started
            _save(audit, report)
            traceback.print_exc()
            return 1
        finally:
            faulthandler.disable()


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", required=True,
                        help="new train-only basename under project eval_runs")
    args = parser.parse_args(argv)
    return run(args.output_directory)


if __name__ == "__main__":
    raise SystemExit(main())
