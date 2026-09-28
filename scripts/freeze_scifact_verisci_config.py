"""Freeze the VeriSci strategy selected on official SciFact train only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

STRATEGY_ORDER = ("flex", "k2", "k3", "k4", "k5")
SELECTION_METRICS = (
    "abstract_rationalized_f1",
    "sentence_label_f1",
    "sentence_selection_f1",
)
OFFICIAL_METRIC_GROUPS = (
    "abstract_label_only",
    "abstract_rationalized",
    "sentence_selection",
    "sentence_label",
)
OFFICIAL_METRIC_NAMES = frozenset(
    f"{group}_{metric}"
    for group in OFFICIAL_METRIC_GROUPS
    for metric in ("precision", "recall", "f1")
)

VERISCI_RUN_SCHEMA = "scifact-official-verisci-run-v1"
SCORE_RUN_SCHEMA = "scifact-official-pipeline-score-v1"
OFFICIAL_SCIFACT_COMMIT = "68b98a56d93e0f9da0d2aab4e6c3294699a0f72e"
OFFICIAL_SCIFACT_ARCHIVE_SHA256 = (
    "aba85f44b80e4014a93b905de8f91929104d1bc4886950e2ec33be243355f75b"
)
OFFICIAL_SOURCE_SCRIPT_HASHES = {
    "verisci/inference/rationale_selection/transformer.py": (
        "14c800eb054bde42715c07bb86542da8357fab100b11268329dfd134edb933b9"
    ),
    "verisci/inference/label_prediction/transformer.py": (
        "5dfa3f059268bc44512ffaf2889766db2ebd0f429602aebc8962d46c80933d63"
    ),
    "verisci/inference/merge_predictions.py": (
        "76dcfc66794b5ca2699fb0966b28a36fe21ab9cda320b4d06f548107da83258d"
    ),
}
OFFICIAL_MODELS = {
    "rationale": {
        "name": "rationale_roberta_large_scifact",
        "archive_sha256": (
            "f103cf8d63c8724915f410606e5d85f25fedf49ce1a1651a1ec502b17a715178"
        ),
        "config_sha256": (
            "82ff8d0621331694afc330ad52fabcc906c2cbae90f52044a90b85beb52827e4"
        ),
    },
    "label": {
        "name": "label_roberta_large_fever_scifact",
        "archive_sha256": (
            "1196f65610195977ff6b3abd058c5871200a3509ed15381b1e6cd6ce897da325"
        ),
        "config_sha256": (
            "87329d5d46509e560cd0e5e01938ee499f6644686bd483c878875bc6a9ce1214"
        ),
    },
}
COMPATIBILITY_ADAPTER_SHA256 = (
    "1405a08699eeb338146fd83a3657b77e634e1111e03f7c342a4302b813da2f3c"
)
COMPATIBILITY_MICRO_BATCH_SIZE = 8
OFFICIAL_EVALUATOR = {
    "repository": "https://github.com/allenai/scifact-evaluator",
    "commit": "66feffc5b2cc9e28e3ce3b8c9e824c3c642981eb",
    "archive_sha256": (
        "16a743524ed0bbb83e56d862b7f04475c5fbdd22d21046aecad7583a1bc0b009"
    ),
    "evaluator_sha256": (
        "2554fed44c3f5592bdeed59ab0d3918a412b452ba1c77da73ba0f62e74c9987f"
    ),
}


class VeriSciFreezeError(RuntimeError):
    """Raised when a train artifact cannot be admitted to the freeze."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VeriSciFreezeError(f"could not read JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise VeriSciFreezeError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise VeriSciFreezeError(f"could not hash file: {path}") from exc
    return digest.hexdigest()


def _require_sha256(value: Any, description: str) -> str:
    if not isinstance(value, str):
        raise VeriSciFreezeError(f"{description} must be a SHA-256 string")
    normalized = value.lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise VeriSciFreezeError(f"{description} must be a SHA-256 string")
    return normalized


def _require_mapping(value: Any, description: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise VeriSciFreezeError(f"{description} must be an object")
    return value


def _require_positive_int(value: Any, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise VeriSciFreezeError(f"{description} must be a positive integer")
    return value


def _require_metric(value: Any, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise VeriSciFreezeError(f"{description} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise VeriSciFreezeError(f"{description} must be finite and in [0, 1]")
    return result


def _verified_outputs(
    directory: Path, raw_hashes: Any, description: str
) -> dict[str, str]:
    hashes = _require_mapping(raw_hashes, f"{description} output hashes")
    if not hashes:
        raise VeriSciFreezeError(f"{description} output hashes cannot be empty")
    verified: dict[str, str] = {}
    for relative_name, expected_value in hashes.items():
        relative_path = Path(relative_name)
        if (
            relative_path.is_absolute()
            or len(relative_path.parts) != 1
            or relative_name in {"", ".", ".."}
        ):
            raise VeriSciFreezeError(
                f"{description} output must be a direct child: {relative_name!r}"
            )
        expected = _require_sha256(
            expected_value, f"{description} output {relative_name} hash"
        )
        output_path = directory / relative_path
        if not output_path.is_file() or _sha256(output_path) != expected:
            raise VeriSciFreezeError(
                f"{description} output hash mismatch: {relative_name}"
            )
        verified[relative_name] = expected
    return verified


def _flatten_independent_metrics(value: Any) -> dict[str, float]:
    groups = _require_mapping(value, "independent metrics")
    if set(groups) != set(OFFICIAL_METRIC_GROUPS):
        raise VeriSciFreezeError("independent metric groups are incomplete or unknown")
    flattened: dict[str, float] = {}
    for group_name in OFFICIAL_METRIC_GROUPS:
        group = _require_mapping(groups[group_name], f"{group_name} metrics")
        if set(group) != {"precision", "recall", "f1"}:
            raise VeriSciFreezeError(
                f"{group_name} metrics are incomplete or contain unknown keys"
            )
        for metric_name in ("precision", "recall", "f1"):
            key = f"{group_name}_{metric_name}"
            flattened[key] = _require_metric(group[metric_name], key)
    return flattened


def _official_metrics(value: Any) -> dict[str, float]:
    metrics = _require_mapping(value, "official metrics")
    if set(metrics) != set(OFFICIAL_METRIC_NAMES):
        raise VeriSciFreezeError(
            "official metrics are incomplete or contain unknown keys"
        )
    return {
        key: _require_metric(metrics[key], key) for key in sorted(OFFICIAL_METRIC_NAMES)
    }


def _same_metrics(left: dict[str, float], right: dict[str, float]) -> bool:
    return set(left) == set(right) and all(
        abs(left[key] - right[key]) <= 1e-12 for key in left
    )


def _verify_train_run(manifest_path: Path) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != VERISCI_RUN_SCHEMA:
        raise VeriSciFreezeError("unexpected VeriSci train run schema")
    status = _require_mapping(manifest.get("evaluation_status"), "evaluation status")
    if status.get("reportable") is not True:
        raise VeriSciFreezeError("VeriSci train run is not reportable")
    if status.get("official_source_modified") is not False:
        raise VeriSciFreezeError("VeriSci train run did not preserve official source")

    source = _require_mapping(manifest.get("official_source"), "official source")
    if source.get("commit") != OFFICIAL_SCIFACT_COMMIT:
        raise VeriSciFreezeError(
            "VeriSci source commit is not the pinned official commit"
        )
    if (
        _require_sha256(source.get("archive_sha256"), "official source archive hash")
        != OFFICIAL_SCIFACT_ARCHIVE_SHA256
    ):
        raise VeriSciFreezeError("VeriSci source archive is not the pinned archive")
    script_hashes = _require_mapping(
        source.get("scripts_sha256"), "official source script hashes"
    )
    normalized_script_hashes = {
        name: _require_sha256(value, f"official source script {name} hash")
        for name, value in script_hashes.items()
    }
    if normalized_script_hashes != OFFICIAL_SOURCE_SCRIPT_HASHES:
        raise VeriSciFreezeError("official VeriSci script hashes do not match the pin")

    models = _require_mapping(manifest.get("models"), "official models")
    normalized_models: dict[str, dict[str, str]] = {}
    if set(models) != set(OFFICIAL_MODELS):
        raise VeriSciFreezeError("official model roles are incomplete or unknown")
    for role in OFFICIAL_MODELS:
        model = _require_mapping(models[role], f"{role} model")
        normalized_models[role] = {
            "name": model.get("name"),
            "archive_sha256": _require_sha256(
                model.get("archive_sha256"), f"{role} model archive hash"
            ),
            "config_sha256": _require_sha256(
                model.get("config_sha256"), f"{role} model config hash"
            ),
        }
    if normalized_models != OFFICIAL_MODELS:
        raise VeriSciFreezeError("official model provenance does not match the pin")

    adapter = _require_mapping(
        manifest.get("compatibility_adapter"), "compatibility adapter"
    )
    adapter_hash = _require_sha256(adapter.get("sha256"), "compatibility adapter hash")
    if adapter_hash != COMPATIBILITY_ADAPTER_SHA256:
        raise VeriSciFreezeError("compatibility adapter does not match the pin")
    if adapter.get("model_forward_micro_batch_size") != COMPATIBILITY_MICRO_BATCH_SIZE:
        raise VeriSciFreezeError("compatibility micro-batch size must remain 8")
    raw_adapter_path = adapter.get("path")
    if not isinstance(raw_adapter_path, str) or not raw_adapter_path:
        raise VeriSciFreezeError("compatibility adapter path is missing")
    adapter_path = Path(raw_adapter_path).resolve()
    if not adapter_path.is_file() or _sha256(adapter_path) != adapter_hash:
        raise VeriSciFreezeError("compatibility adapter file hash mismatch")

    data = _require_mapping(manifest.get("data"), "train data")
    if data.get("split") != "train":
        raise VeriSciFreezeError("strategy selection accepts official train only")
    _require_positive_int(data.get("claim_count"), "train claim count")
    _require_positive_int(data.get("corpus_count"), "train corpus count")
    _require_sha256(data.get("corpus_sha256"), "train corpus hash")
    _require_sha256(data.get("claims_sha256"), "train claims hash")

    retrieval = _require_mapping(manifest.get("retrieval"), "retrieval provenance")
    if retrieval.get("top_k") != 3:
        raise VeriSciFreezeError("official VeriSci configuration requires TF-IDF Top-3")
    _require_sha256(retrieval.get("sha256"), "retrieval output hash")
    _require_sha256(retrieval.get("manifest_sha256"), "retrieval manifest hash")

    configuration = _require_mapping(manifest.get("configuration"), "configuration")
    raw_strategies = configuration.get("rationale_strategies")
    if (
        not isinstance(raw_strategies, list)
        or len(raw_strategies) != len(STRATEGY_ORDER)
        or set(raw_strategies) != set(STRATEGY_ORDER)
    ):
        raise VeriSciFreezeError("train run must contain all five rationale strategies")
    if configuration.get("official_flex_threshold") != 0.5:
        raise VeriSciFreezeError("official flex threshold must remain 0.5")
    if configuration.get("label_mode") != "claim_and_rationale":
        raise VeriSciFreezeError("unexpected VeriSci label mode")
    predictions = _require_mapping(configuration.get("predictions"), "predictions")
    if set(predictions) != set(STRATEGY_ORDER):
        raise VeriSciFreezeError(
            "train prediction map must contain all five strategies"
        )

    train_directory = manifest_path.parent
    output_hashes = _verified_outputs(
        train_directory, manifest.get("outputs_sha256"), "VeriSci train run"
    )
    prediction_hashes: dict[str, str] = {}
    prediction_paths: dict[str, Path] = {}
    for strategy in STRATEGY_ORDER:
        filename = predictions[strategy]
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise VeriSciFreezeError(f"invalid prediction filename for {strategy}")
        if filename not in output_hashes:
            raise VeriSciFreezeError(f"prediction is not hash-bound for {strategy}")
        prediction_hashes[strategy] = output_hashes[filename]
        prediction_paths[strategy] = (train_directory / filename).resolve()

    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256(manifest_path),
        "data": data,
        "models": normalized_models,
        "compatibility_adapter": {
            "path": adapter_path,
            "sha256": adapter_hash,
            "model_forward_micro_batch_size": COMPATIBILITY_MICRO_BATCH_SIZE,
        },
        "prediction_hashes": prediction_hashes,
        "prediction_paths": prediction_paths,
    }


def _verify_score(
    *,
    strategy: str,
    score_directory: Path,
    train: dict[str, Any],
) -> dict[str, Any]:
    manifest_path = score_directory / "run_manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != SCORE_RUN_SCHEMA:
        raise VeriSciFreezeError(f"unexpected score schema for {strategy}")
    status = _require_mapping(
        manifest.get("evaluation_status"), f"{strategy} score status"
    )
    if status != {
        "reportable": True,
        "mode": "official_leaderboard_evaluator_reproduction",
    }:
        raise VeriSciFreezeError(f"{strategy} score is not a formal official run")

    evaluator = _require_mapping(
        manifest.get("official_evaluator"), f"{strategy} official evaluator"
    )
    normalized_evaluator = {
        "repository": evaluator.get("repository"),
        "commit": evaluator.get("commit"),
        "archive_sha256": _require_sha256(
            evaluator.get("archive_sha256"), f"{strategy} evaluator archive hash"
        ),
        "evaluator_sha256": _require_sha256(
            evaluator.get("evaluator_sha256"), f"{strategy} evaluator script hash"
        ),
    }
    if normalized_evaluator != OFFICIAL_EVALUATOR:
        raise VeriSciFreezeError(
            f"{strategy} did not use the pinned official evaluator"
        )

    train_data = train["data"]
    data = _require_mapping(manifest.get("data"), f"{strategy} score data")
    expected_data = {
        "split": "train",
        "claim_count": train_data["claim_count"],
        "corpus_count": train_data["corpus_count"],
        "corpus_sha256": train_data["corpus_sha256"],
        "claims_sha256": train_data["claims_sha256"],
    }
    if data != expected_data:
        raise VeriSciFreezeError(f"{strategy} score data differs from the train run")

    predictions = _require_mapping(
        manifest.get("predictions"), f"{strategy} scored predictions"
    )
    if predictions != {
        "sha256": train["prediction_hashes"][strategy],
        "exact_claim_coverage": True,
        "known_documents_and_valid_sentence_indices": True,
    }:
        raise VeriSciFreezeError(
            f"{strategy} score is not bound to its complete train prediction"
        )

    validation = _require_mapping(
        manifest.get("validation"), f"{strategy} score validation"
    )
    if validation.get("official_evaluator_cross_check") is not True or (
        validation.get("absolute_tolerance") != 1e-12
    ):
        raise VeriSciFreezeError(f"{strategy} score lacks the official cross-check")

    output_hashes = _verified_outputs(
        score_directory, manifest.get("outputs_sha256"), f"{strategy} score"
    )
    required_outputs = {
        "official_metrics.json",
        "independent_metrics.json",
        "official_stdout.txt",
        "official_stderr.txt",
    }
    if not required_outputs.issubset(output_hashes):
        raise VeriSciFreezeError(f"{strategy} score outputs are incomplete")

    official = _official_metrics(_read_json(score_directory / "official_metrics.json"))
    independent = _flatten_independent_metrics(
        _read_json(score_directory / "independent_metrics.json")
    )
    if not _same_metrics(official, independent):
        raise VeriSciFreezeError(
            f"{strategy} official and independent metrics do not agree"
        )

    return {
        "score_directory": str(score_directory),
        "score_manifest_path": str(manifest_path),
        "score_manifest_sha256": _sha256(manifest_path),
        "prediction_path": str(train["prediction_paths"][strategy]),
        "prediction_sha256": train["prediction_hashes"][strategy],
        "official_metrics_sha256": output_hashes["official_metrics.json"],
        "independent_metrics_sha256": output_hashes["independent_metrics.json"],
        "official_metrics": official,
    }


def _parse_score_arguments(values: list[str]) -> dict[str, Path]:
    scores: dict[str, Path] = {}
    for value in values:
        strategy, separator, raw_directory = value.partition("=")
        if not separator or not raw_directory:
            raise VeriSciFreezeError("--score must use STRATEGY=DIR")
        if strategy not in STRATEGY_ORDER:
            raise VeriSciFreezeError(f"unknown strategy in --score: {strategy}")
        if strategy in scores:
            raise VeriSciFreezeError(f"duplicate --score strategy: {strategy}")
        scores[strategy] = Path(raw_directory).resolve()
    if set(scores) != set(STRATEGY_ORDER):
        missing = ", ".join(
            strategy for strategy in STRATEGY_ORDER if strategy not in scores
        )
        raise VeriSciFreezeError(
            f"all five --score entries are required; missing: {missing}"
        )
    if len(set(scores.values())) != len(STRATEGY_ORDER):
        raise VeriSciFreezeError("each strategy must use a distinct score directory")
    return scores


def freeze(
    *,
    train_run_manifest: Path,
    score_directories: dict[str, Path],
    output: Path,
) -> Path:
    """Validate train-only artifacts, select one strategy, and write once."""

    destination = output.resolve()
    if destination.exists():
        raise VeriSciFreezeError(f"refusing to overwrite frozen config: {destination}")
    manifest_path = train_run_manifest.resolve()
    if destination == manifest_path:
        raise VeriSciFreezeError("frozen config cannot replace the train manifest")

    train = _verify_train_run(manifest_path)
    candidates = {
        strategy: _verify_score(
            strategy=strategy,
            score_directory=score_directories[strategy],
            train=train,
        )
        for strategy in STRATEGY_ORDER
    }
    selected = max(
        STRATEGY_ORDER,
        key=lambda strategy: (
            *(
                candidates[strategy]["official_metrics"][key]
                for key in SELECTION_METRICS
            ),
            -STRATEGY_ORDER.index(strategy),
        ),
    )
    allowed_dev_strategies = ["flex"]
    if selected != "flex":
        allowed_dev_strategies.append(selected)

    script_path = Path(__file__).resolve()
    frozen = {
        "schema_version": "scifact-official-verisci-config-freeze-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "freeze_boundary": {
            "selection_split": "official_train_only",
            "dev_inputs_supported_by_this_tool": False,
            "configuration_frozen_before_formal_dev": True,
        },
        "selection_rule": {
            "direction": "descending",
            "primary": SELECTION_METRICS[0],
            "secondary": SELECTION_METRICS[1],
            "tertiary": SELECTION_METRICS[2],
            "complete_tie_order": list(STRATEGY_ORDER),
        },
        "train_run": {
            "manifest_path": str(manifest_path),
            "manifest_sha256": train["manifest_sha256"],
            "schema_version": VERISCI_RUN_SCHEMA,
            "data": {
                key: train["data"][key]
                for key in (
                    "split",
                    "claim_count",
                    "corpus_count",
                    "corpus_sha256",
                    "claims_sha256",
                )
            },
            "official_source": {
                "commit": OFFICIAL_SCIFACT_COMMIT,
                "archive_sha256": OFFICIAL_SCIFACT_ARCHIVE_SHA256,
                "scripts_sha256": OFFICIAL_SOURCE_SCRIPT_HASHES,
            },
            "models": train["models"],
            "compatibility_adapter": {
                "path": str(train["compatibility_adapter"]["path"]),
                "sha256": train["compatibility_adapter"]["sha256"],
                "model_forward_micro_batch_size": train["compatibility_adapter"][
                    "model_forward_micro_batch_size"
                ],
            },
        },
        "official_evaluator": OFFICIAL_EVALUATOR,
        "candidates": candidates,
        "selected_strategy": selected,
        "allowed_dev_strategies": allowed_dev_strategies,
        "freeze_tool": {
            "path": str(script_path),
            "sha256": _sha256(script_path),
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(frozen, ensure_ascii=False, indent=2) + "\n"
    try:
        with destination.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
    except FileExistsError as exc:
        raise VeriSciFreezeError(
            f"refusing to overwrite frozen config: {destination}"
        ) from exc
    return destination


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-run-manifest", type=Path, required=True)
    parser.add_argument(
        "--score",
        action="append",
        required=True,
        metavar="STRATEGY=DIR",
        help="repeat once for each of flex, k2, k3, k4, and k5",
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        score_directories = _parse_score_arguments(args.score)
        frozen_path = freeze(
            train_run_manifest=args.train_run_manifest,
            score_directories=score_directories,
            output=args.out,
        )
    except VeriSciFreezeError as exc:
        raise SystemExit(f"freeze failed: {exc}") from exc
    print(frozen_path)


if __name__ == "__main__":
    main()
