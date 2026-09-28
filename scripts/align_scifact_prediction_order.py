"""Align existing SciFact predictions to gold-file ID order without changing values."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class PredictionOrderError(ValueError):
    """Prediction and claim identities cannot be aligned losslessly."""


def _sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PredictionOrderError("duplicate JSON object key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise PredictionOrderError(f"non-finite JSON constant: {value}")


def _read_rows(path: Path) -> tuple[list[dict[str, Any]], list[int]]:
    rows = [
        json.loads(
            line, object_pairs_hook=_unique_object, parse_constant=_reject_constant
        )
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or any(
        not isinstance(row, dict) or type(row.get("id")) is not int for row in rows
    ):
        raise PredictionOrderError(
            "nonempty JSONL objects with integer claim IDs required"
        )
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise PredictionOrderError("claim IDs must be unique in each input")
    return rows, ids


def align(claims: Path, predictions: Path, output: Path) -> Path:
    claims, predictions, destination = (
        claims.resolve(),
        predictions.resolve(),
        output.resolve(),
    )
    if destination.exists():
        raise PredictionOrderError("exclusive output directory already exists")
    input_hashes = {"claims": _sha(claims), "predictions": _sha(predictions)}
    _, target_ids = _read_rows(claims)
    rows, original_ids = _read_rows(predictions)
    if set(original_ids) != set(target_ids):
        raise PredictionOrderError(
            "prediction and claim ID sets differ; no rows are filled or dropped"
        )
    by_id = {row["id"]: row for row in rows}
    ordered = [by_id[claim_id] for claim_id in target_ids]
    unchanged = {row["id"]: row for row in ordered} == by_id
    if not unchanged:
        raise PredictionOrderError("prediction content changed during alignment")
    encoded = b"".join(
        (
            json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode("utf-8")
        for row in ordered
    )
    if input_hashes != {"claims": _sha(claims), "predictions": _sha(predictions)}:
        raise PredictionOrderError("input changed while being read")
    destination.mkdir(parents=True, exist_ok=False)
    output_path = destination / "predictions_ordered.jsonl"
    with output_path.open("xb") as handle:
        handle.write(encoded)
    if input_hashes != {"claims": _sha(claims), "predictions": _sha(predictions)}:
        raise PredictionOrderError("input changed during alignment")
    manifest = {
        "schema_version": "scifact-prediction-order-alignment-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "operation": "ID-order normalization only; no model calls or prediction edits",
        "inputs": {
            "claims": {"path": str(claims), "sha256": input_hashes["claims"]},
            "predictions": {
                "path": str(predictions),
                "sha256": input_hashes["predictions"],
            },
        },
        "claim_count": len(target_ids),
        "original_prediction_ids": original_ids,
        "target_claim_ids": target_ids,
        "predictions_by_id_unchanged": unchanged,
        "labels_and_citation_order_unchanged": True,
        "outputs_sha256": {output_path.name: _sha(output_path)},
    }
    with (destination / "manifest.json").open(
        "x", encoding="utf-8", newline="\n"
    ) as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claims", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(align(args.claims, args.predictions, args.out))


if __name__ == "__main__":
    main()
