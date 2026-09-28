"""Run frozen official VeriSci inference with strict provenance checks."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evidencetrace.eval.scifact_official import (
    validate_abstract_retrieval,
    validate_official_data,
    validate_pipeline_predictions,
)

OFFICIAL_COMMIT = "68b98a56d93e0f9da0d2aab4e6c3294699a0f72e"
OFFICIAL_THRESHOLD = 0.5
OFFICIAL_TOP_K = 3
COMPATIBILITY_MICRO_BATCH_SIZE = 8
STRATEGY_FLAGS = {
    "flex": "--output-flex",
    "k2": "--output-k2",
    "k3": "--output-k3",
    "k4": "--output-k4",
    "k5": "--output-k5",
}


class OfficialVeriSciRunError(RuntimeError):
    """Raised when official inference or its reproducibility gates fail."""


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise OfficialVeriSciRunError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise OfficialVeriSciRunError(
                    f"expected object at {path}:{line_number}"
                )
            rows.append(value)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _write_exclusive(path: Path, value: str | bytes) -> None:
    mode = "xb" if isinstance(value, bytes) else "x"
    kwargs = {} if isinstance(value, bytes) else {"encoding": "utf-8", "newline": "\n"}
    with path.open(mode, **kwargs) as handle:
        handle.write(value)


def _verify_hash(path: Path, expected: str, description: str) -> str:
    normalized = expected.lower()
    if len(normalized) != 64 or any(ch not in "0123456789abcdef" for ch in normalized):
        raise OfficialVeriSciRunError(f"{description} hash must be SHA-256")
    observed = _sha256(path)
    if observed != normalized:
        raise OfficialVeriSciRunError(f"{description} SHA-256 mismatch")
    return observed


def _verify_retrieval_manifest(
    manifest_path: Path,
    retrieval_path: Path,
    *,
    split: str,
    corpus_path: Path,
    claims_path: Path,
) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    source = manifest.get("official_source", {})
    data = manifest.get("data", {})
    configuration = manifest.get("configuration", {})
    outputs = manifest.get("outputs_sha256", {})
    if source.get("commit") != OFFICIAL_COMMIT:
        raise OfficialVeriSciRunError("retrieval source commit is not frozen")
    if data.get("split") != split:
        raise OfficialVeriSciRunError("retrieval split mismatch")
    if configuration.get("top_k") != OFFICIAL_TOP_K:
        raise OfficialVeriSciRunError("VeriSci requires official TF-IDF Top-3")
    if data.get("files_sha256", {}).get(corpus_path.name) != _sha256(corpus_path):
        raise OfficialVeriSciRunError("retrieval corpus hash mismatch")
    if data.get("files_sha256", {}).get(claims_path.name) != _sha256(claims_path):
        raise OfficialVeriSciRunError("retrieval claims hash mismatch")
    if outputs.get(retrieval_path.name) != _sha256(retrieval_path):
        raise OfficialVeriSciRunError("retrieval output hash mismatch")
    return manifest


def _run_stage(
    name: str,
    command: list[str],
    *,
    cwd: Path,
    destination: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    elapsed = time.perf_counter() - started
    _write_exclusive(destination / f"{name}.stdout.txt", completed.stdout)
    _write_exclusive(destination / f"{name}.stderr.txt", completed.stderr)
    if completed.returncode:
        raise OfficialVeriSciRunError(
            f"{name} failed with exit code {completed.returncode}"
        )
    return {"stage": name, "elapsed_seconds": elapsed, "exit_code": 0}


def _validate_rationales(
    rows: list[dict[str, Any]],
    retrieval_rows: tuple[dict[str, Any], ...],
    corpus: dict[int, dict[str, Any]],
) -> None:
    if len(rows) != len(retrieval_rows):
        raise OfficialVeriSciRunError("rationale claim coverage is incomplete")
    for row, retrieval in zip(rows, retrieval_rows, strict=True):
        if row.get("claim_id") != retrieval["claim_id"]:
            raise OfficialVeriSciRunError("rationale claim order mismatch")
        evidence = row.get("evidence")
        if not isinstance(evidence, dict):
            raise OfficialVeriSciRunError("rationale evidence must be an object")
        expected_docs = {str(doc_id) for doc_id in retrieval["doc_ids"]}
        if set(evidence) != expected_docs:
            raise OfficialVeriSciRunError("rationale documents differ from Top-3")
        for raw_doc_id, sentence_ids in evidence.items():
            if not isinstance(sentence_ids, list):
                raise OfficialVeriSciRunError("rationale sentence IDs must be a list")
            abstract_size = len(corpus[int(raw_doc_id)]["abstract"])
            if any(
                not isinstance(index, int) or not 0 <= index < abstract_size
                for index in sentence_ids
            ):
                raise OfficialVeriSciRunError("rationale sentence is out of range")


def _validate_labels(
    rows: list[dict[str, Any]], rationale_rows: list[dict[str, Any]]
) -> None:
    if len(rows) != len(rationale_rows):
        raise OfficialVeriSciRunError("label claim coverage is incomplete")
    allowed = {"SUPPORT", "CONTRADICT", "NOT_ENOUGH_INFO"}
    for row, rationale in zip(rows, rationale_rows, strict=True):
        if row.get("claim_id") != rationale["claim_id"]:
            raise OfficialVeriSciRunError("label claim order mismatch")
        labels = row.get("labels")
        if not isinstance(labels, dict) or set(labels) != set(rationale["evidence"]):
            raise OfficialVeriSciRunError("label documents differ from rationales")
        if any(
            not isinstance(value, dict) or value.get("label") not in allowed
            for value in labels.values()
        ):
            raise OfficialVeriSciRunError("invalid official label prediction")


def _probe_environment(python: Path, *, cwd: Path) -> dict[str, Any]:
    code = (
        "import json,platform,torch,transformers;"
        "print(json.dumps({'python':platform.python_version(),"
        "'torch':torch.__version__,'transformers':transformers.__version__,"
        "'cuda_available':torch.cuda.is_available(),"
        "'cuda_runtime':torch.version.cuda,"
        "'device':torch.cuda.get_device_name(0) "
        "if torch.cuda.is_available() else None}))"
    )
    completed = subprocess.run(
        [str(python), "-I", "-c", code],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode:
        raise OfficialVeriSciRunError("could not probe inference environment")
    value = json.loads(completed.stdout)
    if not isinstance(value, dict):
        raise OfficialVeriSciRunError("invalid inference environment probe")
    return value


def _verify_dev_freeze(
    path: Path,
    *,
    strategies: tuple[str, ...],
    source_archive_sha256: str,
    rationale_archive_sha256: str,
    label_archive_sha256: str,
    launcher_sha256: str,
) -> dict[str, Any]:
    frozen = _read_json(path)
    if frozen.get("schema_version") != "scifact-official-verisci-config-freeze-v1":
        raise OfficialVeriSciRunError("unexpected VeriSci frozen-config schema")
    boundary = frozen.get("freeze_boundary")
    if not isinstance(boundary, dict) or boundary != {
        "selection_split": "official_train_only",
        "dev_inputs_supported_by_this_tool": False,
        "configuration_frozen_before_formal_dev": True,
    }:
        raise OfficialVeriSciRunError("invalid train-only freeze boundary")
    allowed = frozen.get("allowed_dev_strategies")
    if not isinstance(allowed, list) or tuple(allowed) != strategies:
        raise OfficialVeriSciRunError("dev strategies differ from the frozen selection")
    if frozen.get("selected_strategy") not in allowed or "flex" not in allowed:
        raise OfficialVeriSciRunError("frozen strategy selection is inconsistent")

    train_run = frozen.get("train_run")
    if not isinstance(train_run, dict):
        raise OfficialVeriSciRunError("frozen train-run provenance is missing")
    train_manifest_path = Path(str(train_run.get("manifest_path", ""))).resolve()
    expected_train_hash = train_run.get("manifest_sha256")
    if (
        not train_manifest_path.is_file()
        or _sha256(train_manifest_path) != expected_train_hash
    ):
        raise OfficialVeriSciRunError("frozen train-run manifest hash mismatch")
    train_manifest = _read_json(train_manifest_path)
    if train_manifest.get("data", {}).get("split") != "train":
        raise OfficialVeriSciRunError("frozen configuration is not based on train")
    if train_manifest.get("official_source", {}).get("archive_sha256") != (
        source_archive_sha256
    ):
        raise OfficialVeriSciRunError("dev official source differs from train")
    models = train_manifest.get("models", {})
    if models.get("rationale", {}).get("archive_sha256") != rationale_archive_sha256:
        raise OfficialVeriSciRunError("dev rationale model differs from train")
    if models.get("label", {}).get("archive_sha256") != label_archive_sha256:
        raise OfficialVeriSciRunError("dev label model differs from train")
    adapter = train_manifest.get("compatibility_adapter", {})
    if adapter.get("sha256") != launcher_sha256:
        raise OfficialVeriSciRunError("dev compatibility adapter differs from train")
    if adapter.get("model_forward_micro_batch_size") != COMPATIBILITY_MICRO_BATCH_SIZE:
        raise OfficialVeriSciRunError("dev micro-batch differs from train")
    return frozen


def run(
    *,
    official_repo: Path,
    official_archive: Path,
    official_archive_sha256: str,
    raw_dir: Path,
    split: str,
    retrieval_path: Path,
    retrieval_manifest_path: Path,
    rationale_model: Path,
    rationale_archive: Path,
    rationale_archive_sha256: str,
    label_model: Path,
    label_archive: Path,
    label_archive_sha256: str,
    inference_python: Path,
    compatibility_launcher: Path,
    frozen_config_path: Path | None,
    strategies: tuple[str, ...],
    output: Path,
) -> Path:
    """Run unchanged official component scripts and validate every intermediate."""

    if split not in {"train", "dev"}:
        raise OfficialVeriSciRunError("only train/dev have public gold")
    if not strategies or len(set(strategies)) != len(strategies):
        raise OfficialVeriSciRunError("strategies must be non-empty and unique")
    if any(strategy not in STRATEGY_FLAGS for strategy in strategies):
        raise OfficialVeriSciRunError("unknown rationale strategy")

    repo = official_repo.resolve()
    raw = raw_dir.resolve()
    retrieval = retrieval_path.resolve()
    retrieval_manifest = retrieval_manifest_path.resolve()
    rationale_dir = rationale_model.resolve()
    label_dir = label_model.resolve()
    python = inference_python.resolve()
    launcher = compatibility_launcher.resolve()
    frozen_config = frozen_config_path.resolve() if frozen_config_path else None
    destination = output.resolve()
    corpus_path = raw / "corpus.jsonl"
    claims_path = raw / f"claims_{split}.jsonl"
    rationale_script = repo / "verisci/inference/rationale_selection/transformer.py"
    label_script = repo / "verisci/inference/label_prediction/transformer.py"
    merge_script = repo / "verisci/inference/merge_predictions.py"
    required = (
        repo,
        corpus_path,
        claims_path,
        retrieval,
        retrieval_manifest,
        rationale_dir,
        label_dir,
        python,
        launcher,
        rationale_script,
        label_script,
        merge_script,
    )
    if any(not path.exists() for path in required):
        raise OfficialVeriSciRunError(
            "official source, data, model, or input is missing"
        )

    source_archive_hash = _verify_hash(
        official_archive.resolve(), official_archive_sha256, "official source archive"
    )
    rationale_archive_hash = _verify_hash(
        rationale_archive.resolve(), rationale_archive_sha256, "rationale model archive"
    )
    label_archive_hash = _verify_hash(
        label_archive.resolve(), label_archive_sha256, "label model archive"
    )
    if split == "train" and frozen_config is not None:
        raise OfficialVeriSciRunError("train calibration must precede the freeze")
    if split == "dev" and frozen_config is None:
        raise OfficialVeriSciRunError("formal dev requires a train-only frozen config")
    frozen: dict[str, Any] | None = None
    if frozen_config is not None:
        frozen = _verify_dev_freeze(
            frozen_config,
            strategies=strategies,
            source_archive_sha256=source_archive_hash,
            rationale_archive_sha256=rationale_archive_hash,
            label_archive_sha256=label_archive_hash,
            launcher_sha256=_sha256(launcher),
        )
    retrieval_run = _verify_retrieval_manifest(
        retrieval_manifest,
        retrieval,
        split=split,
        corpus_path=corpus_path,
        claims_path=claims_path,
    )
    data = validate_official_data(_read_jsonl(corpus_path), _read_jsonl(claims_path))
    ordered_retrieval = validate_abstract_retrieval(
        _read_jsonl(retrieval), data, top_k=OFFICIAL_TOP_K
    )
    environment = _probe_environment(python, cwd=repo)
    if not environment.get("cuda_available"):
        raise OfficialVeriSciRunError(
            "formal VeriSci run requires the frozen CUDA path"
        )

    destination.mkdir(parents=True, exist_ok=False)
    rationale_paths = {
        strategy: destination / f"rationale_selection_{strategy}.jsonl"
        for strategy in strategies
    }
    rationale_command = [
        str(python),
        "-I",
        str(launcher),
        str(COMPATIBILITY_MICRO_BATCH_SIZE),
        str(rationale_script),
        "--corpus",
        str(corpus_path),
        "--dataset",
        str(claims_path),
        "--threshold",
        str(OFFICIAL_THRESHOLD),
        "--abstract-retrieval",
        str(retrieval),
        "--model",
        str(rationale_dir),
    ]
    for strategy, path in rationale_paths.items():
        rationale_command.extend([STRATEGY_FLAGS[strategy], str(path)])
    timings = [
        _run_stage(
            "rationale_selection",
            rationale_command,
            cwd=repo,
            destination=destination,
        )
    ]

    predictions: dict[str, str] = {}
    for strategy, rationale_path in rationale_paths.items():
        rationale_rows = _read_jsonl(rationale_path)
        _validate_rationales(rationale_rows, ordered_retrieval, data.corpus)
        label_path = destination / f"label_prediction_{strategy}.jsonl"
        timings.append(
            _run_stage(
                f"label_prediction_{strategy}",
                [
                    str(python),
                    "-I",
                    str(launcher),
                    str(COMPATIBILITY_MICRO_BATCH_SIZE),
                    str(label_script),
                    "--corpus",
                    str(corpus_path),
                    "--dataset",
                    str(claims_path),
                    "--model",
                    str(label_dir),
                    "--rationale-selection",
                    str(rationale_path),
                    "--mode",
                    "claim_and_rationale",
                    "--output",
                    str(label_path),
                ],
                cwd=repo,
                destination=destination,
            )
        )
        _validate_labels(_read_jsonl(label_path), rationale_rows)
        merged_path = destination / f"merged_predictions_{strategy}.jsonl"
        timings.append(
            _run_stage(
                f"merge_predictions_{strategy}",
                [
                    str(python),
                    "-I",
                    str(merge_script),
                    "--rationale-file",
                    str(rationale_path),
                    "--label-file",
                    str(label_path),
                    "--result-file",
                    str(merged_path),
                ],
                cwd=repo,
                destination=destination,
            )
        )
        validate_pipeline_predictions(_read_jsonl(merged_path), data)
        predictions[strategy] = merged_path.name

    output_hashes = {
        path.name: _sha256(path)
        for path in sorted(destination.iterdir())
        if path.is_file()
    }
    manifest = {
        "schema_version": "scifact-official-verisci-run-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "evaluation_status": {
            "reportable": True,
            "mode": "official_source_slow_tokenizer_compatibility_reproduction",
            "official_source_modified": False,
        },
        "official_source": {
            "repository": "https://github.com/allenai/scifact",
            "commit": OFFICIAL_COMMIT,
            "archive_sha256": source_archive_hash,
            "scripts_sha256": {
                str(path.relative_to(repo)).replace("\\", "/"): _sha256(path)
                for path in (rationale_script, label_script, merge_script)
            },
        },
        "data": {
            "split": split,
            "claim_count": len(data.claims),
            "corpus_count": len(data.corpus),
            "corpus_sha256": _sha256(corpus_path),
            "claims_sha256": _sha256(claims_path),
        },
        "retrieval": {
            "path": str(retrieval),
            "sha256": _sha256(retrieval),
            "manifest_sha256": _sha256(retrieval_manifest),
            "schema_version": retrieval_run.get("schema_version"),
            "top_k": OFFICIAL_TOP_K,
        },
        "models": {
            "rationale": {
                "name": "rationale_roberta_large_scifact",
                "archive_sha256": rationale_archive_hash,
                "config_sha256": _sha256(rationale_dir / "config.json"),
            },
            "label": {
                "name": "label_roberta_large_fever_scifact",
                "archive_sha256": label_archive_hash,
                "config_sha256": _sha256(label_dir / "config.json"),
            },
        },
        "configuration": {
            "rationale_strategies": list(strategies),
            "official_flex_threshold": OFFICIAL_THRESHOLD,
            "label_mode": "claim_and_rationale",
            "predictions": predictions,
        },
        "environment": environment,
        "frozen_config": (
            {
                "path": str(frozen_config),
                "sha256": _sha256(frozen_config),
                "selected_strategy": frozen["selected_strategy"],
                "allowed_dev_strategies": frozen["allowed_dev_strategies"],
            }
            if frozen_config is not None and frozen is not None
            else None
        ),
        "compatibility_adapter": {
            "path": str(launcher),
            "sha256": _sha256(launcher),
            "change": (
                "Force AutoTokenizer use_fast=False to restore the upstream "
                "Transformers 2.7 tokenizer class and split model forward calls "
                "into micro-batches of 8 after a measured whole-batch CUDA OOM; "
                "official scripts are unmodified."
            ),
            "model_forward_micro_batch_size": COMPATIBILITY_MICRO_BATCH_SIZE,
        },
        "orchestrator": {
            "python": sys.version,
            "evidencetrace": importlib.metadata.version("evidencetrace"),
        },
        "timings": timings,
        "outputs_sha256": output_hashes,
    }
    _write_exclusive(destination / "run_manifest.json", _json_bytes(manifest))
    return destination


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--official-archive", type=Path, required=True)
    parser.add_argument("--official-archive-sha256", required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "dev"), required=True)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--retrieval-manifest", type=Path, required=True)
    parser.add_argument("--rationale-model", type=Path, required=True)
    parser.add_argument("--rationale-archive", type=Path, required=True)
    parser.add_argument("--rationale-archive-sha256", required=True)
    parser.add_argument("--label-model", type=Path, required=True)
    parser.add_argument("--label-archive", type=Path, required=True)
    parser.add_argument("--label-archive-sha256", required=True)
    parser.add_argument("--inference-python", type=Path, required=True)
    parser.add_argument("--compatibility-launcher", type=Path, required=True)
    parser.add_argument("--frozen-config", type=Path)
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=tuple(STRATEGY_FLAGS),
        required=True,
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    print(
        run(
            official_repo=args.official_repo,
            official_archive=args.official_archive,
            official_archive_sha256=args.official_archive_sha256,
            raw_dir=args.raw_dir,
            split=args.split,
            retrieval_path=args.retrieval,
            retrieval_manifest_path=args.retrieval_manifest,
            rationale_model=args.rationale_model,
            rationale_archive=args.rationale_archive,
            rationale_archive_sha256=args.rationale_archive_sha256,
            label_model=args.label_model,
            label_archive=args.label_archive,
            label_archive_sha256=args.label_archive_sha256,
            inference_python=args.inference_python,
            compatibility_launcher=args.compatibility_launcher,
            frozen_config_path=args.frozen_config,
            strategies=tuple(args.strategies),
            output=args.out,
        )
    )


if __name__ == "__main__":
    main()
