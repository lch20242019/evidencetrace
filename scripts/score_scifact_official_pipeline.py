"""Score complete SciFact predictions with the frozen official evaluator."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evidencetrace.eval.scifact_official import (
    compute_official_pipeline_metrics,
    validate_official_data,
)

EVALUATOR_COMMIT = "66feffc5b2cc9e28e3ce3b8c9e824c3c642981eb"
SHA256_RE = re.compile(r"[0-9a-f]{64}")


class OfficialPipelineScoreError(RuntimeError):
    """Raised when provenance, inputs, or the official cross-check is invalid."""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise OfficialPipelineScoreError(
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


def _flatten_metrics(metrics: dict[str, dict[str, float]]) -> dict[str, float]:
    return {
        f"{group}_{metric}": value
        for group, values in metrics.items()
        for metric, value in values.items()
    }


def _same_metrics(
    official: dict[str, Any], independent: dict[str, float], *, tolerance: float = 1e-12
) -> bool:
    if set(official) != set(independent):
        return False
    return all(
        isinstance(official[key], int | float)
        and abs(float(official[key]) - independent[key]) <= tolerance
        for key in independent
    )


def score(
    *,
    evaluator_repo: Path,
    evaluator_archive: Path,
    evaluator_archive_sha256: str,
    corpus_path: Path,
    claims_path: Path,
    predictions_path: Path,
    split: str,
    output: Path,
) -> Path:
    """Validate, invoke, and independently cross-check the official evaluator."""

    repo = evaluator_repo.resolve()
    archive = evaluator_archive.resolve()
    corpus = corpus_path.resolve()
    claims = claims_path.resolve()
    predictions = predictions_path.resolve()
    destination = output.resolve()
    expected_archive_hash = evaluator_archive_sha256.lower()
    if not SHA256_RE.fullmatch(expected_archive_hash):
        raise OfficialPipelineScoreError("evaluator archive hash must be SHA-256")
    if _sha256(archive) != expected_archive_hash:
        raise OfficialPipelineScoreError("official evaluator archive hash mismatch")
    if split not in {"train", "dev"}:
        raise OfficialPipelineScoreError("only train/dev have public gold")

    evaluator = repo / "evaluator" / "eval.py"
    if any(not path.is_file() for path in (evaluator, corpus, claims, predictions)):
        raise OfficialPipelineScoreError("evaluator, data, or predictions are missing")

    data = validate_official_data(_read_jsonl(corpus), _read_jsonl(claims))
    prediction_rows = _read_jsonl(predictions)
    independent_nested = compute_official_pipeline_metrics(prediction_rows, data)
    independent = _flatten_metrics(independent_nested)

    destination.mkdir(parents=True, exist_ok=False)
    official_metrics_path = destination / "official_metrics.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(evaluator),
            "--labels_file",
            str(claims),
            "--preds_file",
            str(predictions),
            "--metrics_output_file",
            str(official_metrics_path),
            "--verbose",
        ],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    _write_exclusive(destination / "official_stdout.txt", completed.stdout)
    _write_exclusive(destination / "official_stderr.txt", completed.stderr)
    if completed.returncode:
        raise OfficialPipelineScoreError(
            f"official evaluator failed with exit code {completed.returncode}"
        )
    official = json.loads(official_metrics_path.read_text(encoding="utf-8"))
    if not isinstance(official, dict) or not _same_metrics(official, independent):
        raise OfficialPipelineScoreError(
            "official evaluator and independent implementation disagree"
        )

    independent_path = destination / "independent_metrics.json"
    _write_exclusive(independent_path, _json_bytes(independent_nested))
    output_hashes = {
        path.name: _sha256(path)
        for path in sorted(destination.iterdir())
        if path.is_file()
    }
    manifest = {
        "schema_version": "scifact-official-pipeline-score-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "evaluation_status": {
            "reportable": True,
            "mode": "official_leaderboard_evaluator_reproduction",
        },
        "official_evaluator": {
            "repository": "https://github.com/allenai/scifact-evaluator",
            "commit": EVALUATOR_COMMIT,
            "archive_sha256": expected_archive_hash,
            "evaluator_sha256": _sha256(evaluator),
        },
        "data": {
            "split": split,
            "claim_count": len(data.claims),
            "corpus_count": len(data.corpus),
            "corpus_sha256": _sha256(corpus),
            "claims_sha256": _sha256(claims),
        },
        "predictions": {
            "sha256": _sha256(predictions),
            "exact_claim_coverage": True,
            "known_documents_and_valid_sentence_indices": True,
        },
        "validation": {
            "official_evaluator_cross_check": True,
            "absolute_tolerance": 1e-12,
        },
        "outputs_sha256": output_hashes,
    }
    _write_exclusive(destination / "run_manifest.json", _json_bytes(manifest))
    return destination


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluator-repo", type=Path, required=True)
    parser.add_argument("--evaluator-archive", type=Path, required=True)
    parser.add_argument("--evaluator-archive-sha256", required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--claims", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "dev"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    print(
        score(
            evaluator_repo=args.evaluator_repo,
            evaluator_archive=args.evaluator_archive,
            evaluator_archive_sha256=args.evaluator_archive_sha256,
            corpus_path=args.corpus,
            claims_path=args.claims,
            predictions_path=args.predictions,
            split=args.split,
            output=args.out,
        )
    )


if __name__ == "__main__":
    main()
