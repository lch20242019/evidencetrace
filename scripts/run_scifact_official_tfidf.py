"""Run the frozen AllenAI SciFact TF-IDF retriever and audit its outputs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evidencetrace.eval.scifact_official import (
    compute_abstract_retrieval_metrics,
    validate_official_data,
)

OFFICIAL_COMMIT = "68b98a56d93e0f9da0d2aab4e6c3294699a0f72e"
OFFICIAL_TOP_K = 3
OFFICIAL_MIN_GRAM = 1
OFFICIAL_MAX_GRAM = 2
SHA256_RE = re.compile(r"[0-9a-f]{64}")


class OfficialTfidfRunError(RuntimeError):
    """Raised when the official run or its provenance is invalid."""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise OfficialTfidfRunError(f"expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_exclusive(path: Path, value: str | bytes) -> None:
    mode = "xb" if isinstance(value, bytes) else "x"
    kwargs = {} if isinstance(value, bytes) else {"encoding": "utf-8", "newline": "\n"}
    with path.open(mode, **kwargs) as handle:
        handle.write(value)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _prepare_output(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=False)


def _run(command: list[str], *, cwd: Path) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode:
        raise OfficialTfidfRunError(
            f"official command failed ({completed.returncode}): {completed.stderr}"
        )
    return completed.stdout


def _parse_official_retrieval_stdout(value: str) -> dict[str, float]:
    found: dict[str, float] = {}
    for line in value.splitlines():
        if line.startswith("Hit one:"):
            found["hit_one"] = float(line.partition(":")[2].strip())
        elif line.startswith("Hit all:"):
            found["hit_all"] = float(line.partition(":")[2].strip())
    if set(found) != {"hit_one", "hit_all"}:
        raise OfficialTfidfRunError("could not parse official retrieval evaluator")
    return found


def run(
    *,
    official_repo: Path,
    official_archive: Path,
    official_archive_sha256: str,
    raw_dir: Path,
    split: str,
    output: Path,
) -> Path:
    """Execute the immutable official script and write a strict run manifest."""

    repo = official_repo.resolve()
    archive = official_archive.resolve()
    raw = raw_dir.resolve()
    destination = output.resolve()
    expected_archive_hash = official_archive_sha256.lower()
    if not SHA256_RE.fullmatch(expected_archive_hash):
        raise OfficialTfidfRunError("official archive hash must be SHA-256")
    if _sha256(archive) != expected_archive_hash:
        raise OfficialTfidfRunError("official source archive hash mismatch")
    if split not in {"train", "dev"}:
        raise OfficialTfidfRunError("only train/dev have public gold")

    tfidf_script = repo / "verisci" / "inference" / "abstract_retrieval" / "tfidf.py"
    evaluator_script = repo / "verisci" / "evaluate" / "abstract_retrieval.py"
    corpus_path = raw / "corpus.jsonl"
    claims_path = raw / f"claims_{split}.jsonl"
    required = (tfidf_script, evaluator_script, corpus_path, claims_path)
    if any(not path.is_file() for path in required):
        raise OfficialTfidfRunError("official source or data file is missing")
    _prepare_output(destination)
    retrieval_path = destination / "abstract_retrieval.jsonl"

    command = [
        sys.executable,
        str(tfidf_script),
        "--corpus",
        str(corpus_path),
        "--dataset",
        str(claims_path),
        "--k",
        str(OFFICIAL_TOP_K),
        "--min-gram",
        str(OFFICIAL_MIN_GRAM),
        "--max-gram",
        str(OFFICIAL_MAX_GRAM),
        "--output",
        str(retrieval_path),
    ]
    tfidf_stdout = _run(command, cwd=repo)
    evaluator_stdout = _run(
        [
            sys.executable,
            str(evaluator_script),
            "--dataset",
            str(claims_path),
            "--abstract-retrieval",
            str(retrieval_path),
        ],
        cwd=repo,
    )

    data = validate_official_data(_read_jsonl(corpus_path), _read_jsonl(claims_path))
    retrieval_rows = _read_jsonl(retrieval_path)
    metrics = compute_abstract_retrieval_metrics(
        retrieval_rows, data, top_k=OFFICIAL_TOP_K
    )
    parsed = _parse_official_retrieval_stdout(evaluator_stdout)
    independently_rounded = {
        key: round(metrics["official_all_claims"][key], 4)
        for key in ("hit_one", "hit_all")
    }
    if parsed != independently_rounded:
        raise OfficialTfidfRunError(
            "official and independent retrieval scores disagree"
        )

    _write_exclusive(destination / "official_tfidf_stdout.txt", tfidf_stdout)
    _write_exclusive(destination / "official_evaluator_stdout.txt", evaluator_stdout)
    metrics_path = destination / "metrics.json"
    _write_exclusive(metrics_path, _json_bytes(metrics))
    output_hashes = {
        path.name: _sha256(path)
        for path in sorted(destination.iterdir())
        if path.is_file()
    }
    manifest = {
        "schema_version": "scifact-official-tfidf-run-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "evaluation_status": {
            "reportable": True,
            "mode": "official_source_compatibility_reproduction",
        },
        "official_source": {
            "repository": "https://github.com/allenai/scifact",
            "commit": OFFICIAL_COMMIT,
            "archive_sha256": expected_archive_hash,
            "files_sha256": {
                str(path.relative_to(repo)).replace("\\", "/"): _sha256(path)
                for path in (tfidf_script, evaluator_script)
            },
        },
        "data": {
            "split": split,
            "claim_count": len(data.claims),
            "corpus_count": len(data.corpus),
            "files_sha256": {
                corpus_path.name: _sha256(corpus_path),
                claims_path.name: _sha256(claims_path),
            },
        },
        "configuration": {
            "top_k": OFFICIAL_TOP_K,
            "min_gram": OFFICIAL_MIN_GRAM,
            "max_gram": OFFICIAL_MAX_GRAM,
            "stop_words": "english",
            "document_text": "title + ' '.join(abstract)",
        },
        "environment": {
            "python": sys.version,
            "numpy": importlib.metadata.version("numpy"),
            "scipy": importlib.metadata.version("scipy"),
            "scikit_learn": importlib.metadata.version("scikit-learn"),
            "jsonlines": importlib.metadata.version("jsonlines"),
        },
        "validation": {
            "exact_claim_coverage": True,
            "top_k_unique_known_documents": True,
            "official_evaluator_cross_check": True,
            "official_stdout_scores": parsed,
            "warning": (
                "Official tfidf.py labels 0-based raw ranks as reciprocal ranks; "
                "those stdout values are retained but are not reported as MRR."
            ),
        },
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
            output=args.out,
        )
    )


if __name__ == "__main__":
    main()
