"""Freeze Agent A/B configuration only after a train-only viability gate passes.

This is deliberately separate from the model runner.  It consumes an immutable,
complete official SciFact train run and official-score artifacts for all arms.  It
never reads dev labels or dev metrics; dev files are hash-bound only.

The thresholds below are collapse guards, not publishable performance targets.
They prevent an operationally valid but degenerate run (for example, every
prediction being ``{"evidence": {}}``) from being frozen for formal dev use.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ARMS = (
    "single_one_pass",
    "single_self_review",
    "judge_challenger",
)
MAIN_ARMS = ("single_self_review", "judge_challenger")

RUN_SCHEMA = "scifact-agent-ab-run-v1"
FREEZE_SCHEMA = "scifact-agent-ab-frozen-config-v2"
SCORE_SCHEMA = "scifact-official-pipeline-score-v1"
GATE_SCHEMA = "scifact-agent-ab-train-admission-v1"
REQUIRED_AGENT_PROTOCOL = "scifact-agent-decision-v2"

OFFICIAL_TRAIN_CLAIM_COUNT = 809
OFFICIAL_CORPUS_COUNT = 5183
OFFICIAL_CORPUS_SHA256 = (
    "b8d6c89624cb2ed74dee8938effc4f5d8bd2086887880af8110d64be4ceade62"
)
OFFICIAL_TRAIN_CLAIMS_SHA256 = (
    "f4c8fa82d8bd0653a9cc8d61a6ea48c25eacea64e90af5dbf390ebb1b74372f0"
)
OFFICIAL_DEV_CLAIMS_SHA256 = (
    "86f0435d08fdb65d1aa41d1472684f57e6e71930626497bdf4d7a9ec1a632217"
)
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

# These are deliberately permissive viability floors.  Passing them means only
# "not collapsed and scoreable", not "competitive".
MIN_FORMAT_VALID_RATE = 0.99
MIN_NONEMPTY_PREDICTION_RATE = 0.05
MIN_DISTINCT_FINAL_LABELS = 2
MIN_ABSTRACT_RATIONALIZED_F1 = 0.05
MIN_SENTENCE_LABEL_F1 = 0.05

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


class AgentABFreezeError(RuntimeError):
    """Raised when train artifacts cannot be admitted to the formal freeze."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AgentABFreezeError(f"could not read JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise AgentABFreezeError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise AgentABFreezeError(f"blank JSONL row: {path}:{line_number}")
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise AgentABFreezeError(
                        f"expected JSON object: {path}:{line_number}"
                    )
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise AgentABFreezeError(f"could not read JSONL: {path}") from exc
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise AgentABFreezeError(f"could not hash file: {path}") from exc
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _write_exclusive(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(value)
    except FileExistsError as exc:
        raise AgentABFreezeError(
            f"refusing to overwrite frozen config: {path}"
        ) from exc


def _require_mapping(value: Any, description: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise AgentABFreezeError(f"{description} must be an object")
    return value


def _require_sha256(value: Any, description: str) -> str:
    if not isinstance(value, str):
        raise AgentABFreezeError(f"{description} must be a SHA-256 string")
    normalized = value.lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise AgentABFreezeError(f"{description} must be a SHA-256 string")
    return normalized


def _require_rate(value: Any, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise AgentABFreezeError(f"{description} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise AgentABFreezeError(f"{description} must be finite and in [0, 1]")
    return result


def _verified_outputs(manifest_path: Path) -> dict[str, str]:
    manifest = _read_json(manifest_path)
    raw_hashes = _require_mapping(manifest.get("outputs_sha256"), "output hashes")
    if not raw_hashes:
        raise AgentABFreezeError("output hashes cannot be empty")
    verified: dict[str, str] = {}
    for name, raw_expected in raw_hashes.items():
        if not name or Path(name).name != name:
            raise AgentABFreezeError(f"output must be a direct child: {name!r}")
        expected = _require_sha256(raw_expected, f"{name} output hash")
        path = manifest_path.parent / name
        if not path.is_file() or _sha256(path) != expected:
            raise AgentABFreezeError(f"output hash mismatch: {name}")
        verified[name] = expected
    return verified


def _verify_official_train_identity(
    manifest: dict[str, Any],
) -> tuple[int, dict[str, Any]]:
    inputs = _require_mapping(manifest.get("input_lock"), "train input lock")
    if inputs.get("split") != "train":
        raise AgentABFreezeError("freeze source must use official train")
    if inputs.get("corpus_sha256") != OFFICIAL_CORPUS_SHA256:
        raise AgentABFreezeError(
            "freeze source corpus is not the pinned official corpus"
        )
    if inputs.get("claims_sha256") != OFFICIAL_TRAIN_CLAIMS_SHA256:
        raise AgentABFreezeError("freeze source claims are not pinned official train")

    execution = _require_mapping(manifest.get("execution"), "train execution")
    source_count = execution.get("source_claim_count")
    selected_count = execution.get("selected_claim_count")
    if (
        execution.get("completed") is not True
        or isinstance(source_count, bool)
        or not isinstance(source_count, int)
        or isinstance(selected_count, bool)
        or not isinstance(selected_count, int)
    ):
        raise AgentABFreezeError("train run is incomplete or has invalid claim counts")
    if source_count != OFFICIAL_TRAIN_CLAIM_COUNT or selected_count != source_count:
        raise AgentABFreezeError(
            "freeze requires complete official train coverage; "
            "a smoke subset is insufficient"
        )
    if execution.get("physical_call_count") != selected_count * 3 or (
        execution.get("expected_physical_call_count") != selected_count * 3
    ):
        raise AgentABFreezeError("train run physical-call coverage is incomplete")
    return selected_count, inputs


def _verify_train_predictions(
    manifest_path: Path,
    manifest: dict[str, Any],
    claim_count: int,
    outputs: dict[str, str],
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    required = {"cases.jsonl"} | {
        f"predictions_{arm}.jsonl" for arm in ARMS
    }
    if not required.issubset(outputs):
        missing = ", ".join(sorted(required - outputs.keys()))
        raise AgentABFreezeError(f"train run lacks gate inputs: {missing}")

    cases = _read_jsonl(manifest_path.parent / "cases.jsonl")
    if len(cases) != claim_count:
        raise AgentABFreezeError("case count differs from complete official train")
    case_ids = [case.get("claim_id") for case in cases]
    if any(
        isinstance(claim_id, bool) or not isinstance(claim_id, int)
        for claim_id in case_ids
    ):
        raise AgentABFreezeError("cases contain invalid claim IDs")
    if len(set(case_ids)) != claim_count:
        raise AgentABFreezeError("case claim IDs are missing or duplicated")

    observed: dict[str, dict[str, Any]] = {
        arm: {
            "format_valid_count": 0,
            "nonempty_prediction_count": 0,
            "label_counts": {"SUPPORT": 0, "CONTRADICT": 0, "NEI": 0},
        }
        for arm in ARMS
    }

    def valid_evidence_schema(value: Any) -> bool:
        if not isinstance(value, dict) or len(value) > 3:
            return False
        for raw_doc_id, raw_entry in value.items():
            label = raw_entry.get("label") if isinstance(raw_entry, dict) else None
            if (
                not isinstance(raw_doc_id, str)
                or not raw_doc_id.isascii()
                or not raw_doc_id.isdigit()
                or int(raw_doc_id) <= 0
                or not isinstance(raw_entry, dict)
                or set(raw_entry) != {"label", "sentences"}
                or not isinstance(label, str)
                or label not in {"SUPPORT", "CONTRADICT"}
            ):
                return False
            sentences = raw_entry.get("sentences")
            if (
                not isinstance(sentences, list)
                or not sentences
                or any(
                    isinstance(index, bool) or not isinstance(index, int) or index < 0
                    for index in sentences
                )
                or len(set(sentences)) != len(sentences)
            ):
                return False
        return True

    prediction_hashes: dict[str, str] = {}
    for arm in ARMS:
        filename = f"predictions_{arm}.jsonl"
        prediction_hashes[arm] = outputs[filename]
        predictions = _read_jsonl(manifest_path.parent / filename)
        if len(predictions) != claim_count:
            raise AgentABFreezeError(f"{arm} prediction count is incomplete")
        if [prediction.get("id") for prediction in predictions] != case_ids:
            raise AgentABFreezeError(f"{arm} prediction IDs differ from cases")

        for case, prediction in zip(cases, predictions, strict=True):
            arms = _require_mapping(case.get("arms"), "case arms")
            if set(arms) != set(ARMS):
                raise AgentABFreezeError("case arms are incomplete or unknown")
            summary = _require_mapping(arms[arm], f"{arm} case summary")
            evidence = prediction.get("evidence")
            if set(prediction) != {"id", "evidence"} or not valid_evidence_schema(
                evidence
            ):
                raise AgentABFreezeError(f"{arm} prediction schema is invalid")
            if summary.get("final_evidence") != evidence:
                raise AgentABFreezeError(f"{arm} case/prediction evidence mismatch")
            if summary.get("valid") is True and valid_evidence_schema(evidence):
                observed[arm]["format_valid_count"] += 1
                labels = {
                    document["label"] for document in evidence.values()
                }
                if len(labels) > 1:
                    raise AgentABFreezeError(
                        f"{arm} prediction mixes claim-level relation labels"
                    )
                final_label = next(iter(labels), "NEI")
                observed[arm]["label_counts"][final_label] += 1
            if evidence:
                observed[arm]["nonempty_prediction_count"] += 1

    for arm, values in observed.items():
        values["claim_count"] = claim_count
        values["format_valid_rate"] = values["format_valid_count"] / claim_count
        values["nonempty_prediction_rate"] = (
            values["nonempty_prediction_count"] / claim_count
        )
        values["distinct_label_count"] = sum(
            count > 0 for count in values["label_counts"].values()
        )
        if values["format_valid_rate"] < MIN_FORMAT_VALID_RATE:
            raise AgentABFreezeError(
                f"{arm} format-valid rate {values['format_valid_rate']:.6f} is below "
                f"{MIN_FORMAT_VALID_RATE:.2f}"
            )

    for arm in MAIN_ARMS:
        rate = observed[arm]["nonempty_prediction_rate"]
        if rate < MIN_NONEMPTY_PREDICTION_RATE:
            raise AgentABFreezeError(
                f"{arm} non-empty prediction rate {rate:.6f} is below "
                f"{MIN_NONEMPTY_PREDICTION_RATE:.2f}; empty-output collapse detected"
            )
        if observed[arm]["distinct_label_count"] < MIN_DISTINCT_FINAL_LABELS:
            raise AgentABFreezeError(
                f"{arm} emitted fewer than {MIN_DISTINCT_FINAL_LABELS} final labels; "
                "constant-label collapse detected"
            )
    return observed, prediction_hashes


def _flatten_independent_metrics(value: Any) -> dict[str, float]:
    groups = _require_mapping(value, "independent metrics")
    if set(groups) != set(OFFICIAL_METRIC_GROUPS):
        raise AgentABFreezeError("independent metric groups are incomplete or unknown")
    flattened: dict[str, float] = {}
    for group_name in OFFICIAL_METRIC_GROUPS:
        group = _require_mapping(groups[group_name], f"{group_name} metrics")
        if set(group) != {"precision", "recall", "f1"}:
            raise AgentABFreezeError(f"{group_name} metrics are incomplete or unknown")
        for metric_name in ("precision", "recall", "f1"):
            key = f"{group_name}_{metric_name}"
            flattened[key] = _require_rate(group[metric_name], key)
    return flattened


def _verify_score(
    *,
    arm: str,
    score_directory: Path,
    train_inputs: dict[str, Any],
    claim_count: int,
    prediction_sha256: str,
) -> dict[str, Any]:
    manifest_path = score_directory / "run_manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != SCORE_SCHEMA:
        raise AgentABFreezeError(f"unexpected official score schema for {arm}")
    if manifest.get("evaluation_status") != {
        "reportable": True,
        "mode": "official_leaderboard_evaluator_reproduction",
    }:
        raise AgentABFreezeError(f"{arm} score is not an official reproduction")
    if manifest.get("official_evaluator") != OFFICIAL_EVALUATOR:
        raise AgentABFreezeError(f"{arm} score did not use the pinned evaluator")

    data = _require_mapping(manifest.get("data"), f"{arm} score data")
    if data != {
        "split": "train",
        "claim_count": claim_count,
        "corpus_count": OFFICIAL_CORPUS_COUNT,
        "corpus_sha256": train_inputs["corpus_sha256"],
        "claims_sha256": train_inputs["claims_sha256"],
    }:
        raise AgentABFreezeError(f"{arm} score is not bound to complete official train")
    if manifest.get("predictions") != {
        "sha256": prediction_sha256,
        "exact_claim_coverage": True,
        "known_documents_and_valid_sentence_indices": True,
    }:
        raise AgentABFreezeError(f"{arm} score is not bound to its train predictions")
    if manifest.get("validation") != {
        "official_evaluator_cross_check": True,
        "absolute_tolerance": 1e-12,
    }:
        raise AgentABFreezeError(f"{arm} score lacks the independent cross-check")

    outputs = _verified_outputs(manifest_path)
    required = {
        "official_metrics.json",
        "independent_metrics.json",
        "official_stdout.txt",
        "official_stderr.txt",
    }
    if not required.issubset(outputs):
        raise AgentABFreezeError(f"{arm} score outputs are incomplete")
    official_raw = _read_json(score_directory / "official_metrics.json")
    if set(official_raw) != set(OFFICIAL_METRIC_NAMES):
        raise AgentABFreezeError(f"{arm} official metrics are incomplete or unknown")
    official = {
        name: _require_rate(official_raw[name], f"{arm} {name}")
        for name in sorted(OFFICIAL_METRIC_NAMES)
    }
    independent = _flatten_independent_metrics(
        _read_json(score_directory / "independent_metrics.json")
    )
    if any(abs(official[key] - independent[key]) > 1e-12 for key in official):
        raise AgentABFreezeError(f"{arm} official and independent metrics disagree")

    return {
        "score_manifest_path": str(manifest_path.resolve()),
        "score_manifest_sha256": _sha256(manifest_path),
        "official_metrics_sha256": outputs["official_metrics.json"],
        "independent_metrics_sha256": outputs["independent_metrics.json"],
        "official_metrics": official,
    }


def _enforce_quality_gate(scores: dict[str, dict[str, Any]]) -> None:
    thresholds = {
        "abstract_rationalized_f1": MIN_ABSTRACT_RATIONALIZED_F1,
        "sentence_label_f1": MIN_SENTENCE_LABEL_F1,
    }
    for arm in MAIN_ARMS:
        metrics = scores[arm]["official_metrics"]
        for metric, minimum in thresholds.items():
            value = metrics[metric]
            if value < minimum:
                raise AgentABFreezeError(
                    f"{arm} official train {metric} {value:.6f} is below "
                    f"the viability floor {minimum:.2f}"
                )


def _dev_input_lock(
    *, corpus: Path, claims: Path, retrieval: Path, retrieval_manifest: Path
) -> dict[str, Any]:
    for path in (corpus, claims, retrieval, retrieval_manifest):
        if not path.is_file():
            raise AgentABFreezeError(f"dev hash-binding input is missing: {path}")
    result = {
        "split": "dev",
        "corpus_sha256": _sha256(corpus),
        "claims_sha256": _sha256(claims),
        "retrieval_sha256": _sha256(retrieval),
        "retrieval_manifest_sha256": _sha256(retrieval_manifest),
    }
    if result["corpus_sha256"] != OFFICIAL_CORPUS_SHA256:
        raise AgentABFreezeError("dev corpus is not the pinned official corpus")
    if result["claims_sha256"] != OFFICIAL_DEV_CLAIMS_SHA256:
        raise AgentABFreezeError("dev claims are not the pinned official dev split")
    return result


def _parse_score_arguments(values: list[str]) -> dict[str, Path]:
    scores: dict[str, Path] = {}
    for value in values:
        arm, separator, raw_directory = value.partition("=")
        if not separator or not raw_directory:
            raise AgentABFreezeError("--score must use ARM=DIR")
        if arm not in ARMS:
            raise AgentABFreezeError(f"unknown arm in --score: {arm}")
        if arm in scores:
            raise AgentABFreezeError(f"duplicate --score arm: {arm}")
        scores[arm] = Path(raw_directory).resolve()
    if set(scores) != set(ARMS):
        missing = ", ".join(arm for arm in ARMS if arm not in scores)
        raise AgentABFreezeError(
            f"all three --score entries are required; missing: {missing}"
        )
    if len(set(scores.values())) != len(ARMS):
        raise AgentABFreezeError("each arm must use a distinct score directory")
    return scores


def freeze(
    *,
    train_manifest_path: Path,
    score_directories: dict[str, Path],
    dev_corpus_path: Path,
    dev_claims_path: Path,
    dev_retrieval_path: Path,
    dev_retrieval_manifest_path: Path,
    output: Path,
) -> Path:
    """Validate train-only viability evidence and write a dev configuration once."""

    manifest_path = train_manifest_path.resolve()
    destination = output.resolve()
    gate_report_path = destination.with_name(
        destination.stem + ".train_admission_gate.json"
    )
    if destination.exists():
        raise AgentABFreezeError(f"refusing to overwrite frozen config: {destination}")
    if gate_report_path.exists():
        raise AgentABFreezeError(
            f"refusing to overwrite train admission report: {gate_report_path}"
        )
    train = _read_json(manifest_path)
    if train.get("schema_version") != RUN_SCHEMA:
        raise AgentABFreezeError("freeze source is not an Agent A/B run")
    if train.get("evaluation_status") != {
        "reportable": False,
        "state": "ready_for_official_scoring",
        "gold_used_by_runner": False,
    }:
        raise AgentABFreezeError(
            "train run status or gold-isolation contract is invalid"
        )
    configuration = _require_mapping(
        train.get("configuration_lock"), "train configuration lock"
    )
    if configuration.get("agent_protocol_version") != REQUIRED_AGENT_PROTOCOL:
        raise AgentABFreezeError(
            "freeze requires scifact-agent-decision-v2; the collapsed v1 protocol "
            "is explicitly rejected"
        )
    claim_count, train_inputs = _verify_official_train_identity(train)
    outputs = _verified_outputs(manifest_path)
    observed, prediction_hashes = _verify_train_predictions(
        manifest_path, train, claim_count, outputs
    )
    if set(score_directories) != set(ARMS):
        raise AgentABFreezeError(
            "official train scores are required for all three arms"
        )
    scores = {
        arm: _verify_score(
            arm=arm,
            score_directory=score_directories[arm].resolve(),
            train_inputs=train_inputs,
            claim_count=claim_count,
            prediction_sha256=prediction_hashes[arm],
        )
        for arm in ARMS
    }
    _enforce_quality_gate(scores)
    dev_inputs = _dev_input_lock(
        corpus=dev_corpus_path.resolve(),
        claims=dev_claims_path.resolve(),
        retrieval=dev_retrieval_path.resolve(),
        retrieval_manifest=dev_retrieval_manifest_path.resolve(),
    )

    gate_report = {
        "schema_version": GATE_SCHEMA,
        "status": "passed",
        "scope": "complete_official_train_only",
        "agent_protocol_version": REQUIRED_AGENT_PROTOCOL,
        "dev_labels_or_metrics_inspected": False,
        "thresholds_are_viability_floors_not_performance_claims": True,
        "implementation": {
            "path": "scripts/freeze_scifact_agent_ab_config.py",
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "thresholds": {
            "minimum_format_valid_rate_all_arms": MIN_FORMAT_VALID_RATE,
            "minimum_nonempty_prediction_rate_main_arms": (
                MIN_NONEMPTY_PREDICTION_RATE
            ),
            "minimum_distinct_final_labels_main_arms": MIN_DISTINCT_FINAL_LABELS,
            "minimum_official_train_abstract_rationalized_f1_main_arms": (
                MIN_ABSTRACT_RATIONALIZED_F1
            ),
            "minimum_official_train_sentence_label_f1_main_arms": (
                MIN_SENTENCE_LABEL_F1
            ),
        },
        "observed": observed,
        "official_train_scores": scores,
        "evidence_sha256": {
            "train_manifest": _sha256(manifest_path),
            "cases": outputs["cases.jsonl"],
            "predictions": prediction_hashes,
        },
    }
    _write_exclusive(gate_report_path, _json_bytes(gate_report))
    gate_binding = {
        "schema_version": GATE_SCHEMA,
        "status": "passed",
        "report": {
            "path": str(gate_report_path),
            "sha256": _sha256(gate_report_path),
        },
    }
    value = {
        "schema_version": FREEZE_SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "train_run_manifest": {
            "path": str(manifest_path),
            "sha256": _sha256(manifest_path),
            "selected_claim_count": claim_count,
        },
        "locked_configuration": configuration,
        "dev_inputs": dev_inputs,
        "selection": {
            "train_used_for_viability_gate_only": True,
            "dev_tuning_permitted": False,
            "main_comparison": "judge_challenger_vs_single_self_review",
        },
        "train_admission_gate": gate_binding,
        "train_admission_gate_sha256": _sha256_json(gate_binding),
    }
    _write_exclusive(destination, _json_bytes(value))
    return destination


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-run-manifest", type=Path, required=True)
    parser.add_argument("--score", action="append", default=[])
    parser.add_argument("--dev-corpus", type=Path, required=True)
    parser.add_argument("--dev-claims", type=Path, required=True)
    parser.add_argument("--dev-retrieval", type=Path, required=True)
    parser.add_argument("--dev-retrieval-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        print(
            freeze(
                train_manifest_path=args.train_run_manifest,
                score_directories=_parse_score_arguments(args.score),
                dev_corpus_path=args.dev_corpus,
                dev_claims_path=args.dev_claims,
                dev_retrieval_path=args.dev_retrieval,
                dev_retrieval_manifest_path=args.dev_retrieval_manifest,
                output=args.out,
            )
        )
    except AgentABFreezeError as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    main()
