"""Run a bounded, train-only SciFact NLI feasibility diagnostic.

This script is intentionally not a benchmark runner.  It accepts only the pinned
official SciFact train files, selects the first 100 claims in file order, reuses
the frozen official TF-IDF Top-3 retrieval artifact, and marks every output as
``exploratory_not_generalization``.  It must never be used to read or score dev.
"""

from __future__ import annotations

# The available CUDA environment uses Python 3.9.  Keep Optional, timezone.utc,
# and zip() without strict= compatible with that interpreter.
# ruff: noqa: B905, UP017, UP045
import argparse
import hashlib
import importlib.metadata
import json
import math
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

SCHEMA_VERSION = "scifact-nli-train-diagnostic-v1"
EVALUATION_MODE = "exploratory_not_generalization"
SAMPLE_POLICY = "first_100_official_train_in_file_order"
SAMPLE_COUNT = 100
SOURCE_CLAIM_COUNT = 809
SOURCE_CORPUS_COUNT = 5183
TOP_K_DOCS = 3
LEXICAL_TOP_N = 10
PRIMARY_NLI_THRESHOLD = 0.95
BATCH_SIZE = 8
MAX_LENGTH = 384
MODEL_REVISION = "6f5cf0a2b59cabb106aca4c287eed12e357e90eb"
OFFICIAL_CORPUS_SHA256 = (
    "b8d6c89624cb2ed74dee8938effc4f5d8bd2086887880af8110d64be4ceade62"
)
OFFICIAL_TRAIN_CLAIMS_SHA256 = (
    "f4c8fa82d8bd0653a9cc8d61a6ea48c25eacea64e90af5dbf390ebb1b74372f0"
)
OFFICIAL_EVALUATOR_SHA256 = (
    "2554fed44c3f5592bdeed59ab0d3918a412b452ba1c77da73ba0f62e74c9987f"
)
EXPECTED_LABEL_MAP = {
    0: "entailment",
    1: "neutral",
    2: "contradiction",
}
DIRECT_THRESHOLDS = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99)
LEXICAL_TOP_NS = (1, 3, 5, 10)
LEXICAL_THRESHOLDS = (0.0, 0.8, 0.9, 0.95)
DEV_TOKEN = re.compile(r"(^|[_.-])dev([_.-]|$)", re.IGNORECASE)


class NliTrainDiagnosticError(RuntimeError):
    """Raised when the bounded diagnostic contract is violated."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _jsonl_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
        for row in rows
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_exclusive(path: Path, value: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(value)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise NliTrainDiagnosticError(
                    f"expected JSON object at {path}:{line_number}"
                )
            rows.append(value)
    return rows


def _assert_not_dev(path: Path, role: str) -> None:
    if any(DEV_TOKEN.search(part) for part in path.resolve().parts):
        raise NliTrainDiagnosticError(f"{role} path contains a forbidden dev token")


def _model_identity(model_dir: Path) -> dict[str, Any]:
    root = model_dir.resolve()
    if not root.is_dir() or root.name != MODEL_REVISION:
        raise NliTrainDiagnosticError("model must be the pinned local DeBERTa revision")
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or ".cache" in path.parts:
            continue
        relative = path.relative_to(root).as_posix()
        files[relative] = {"sha256": _sha256(path), "bytes": path.stat().st_size}
    required = {
        "config.json",
        "model.safetensors",
        "spm.model",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    if not required.issubset(files):
        raise NliTrainDiagnosticError("local model snapshot is incomplete")
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    label_map = {
        int(key): str(value).lower()
        for key, value in config["id2label"].items()
    }
    if label_map != EXPECTED_LABEL_MAP:
        raise NliTrainDiagnosticError("unexpected DeBERTa NLI label mapping")
    return {
        "path": str(root),
        "revision": MODEL_REVISION,
        "identity_sha256": _sha256_bytes(_canonical_bytes(files)),
        "files": files,
        "id2label": {str(key): value for key, value in label_map.items()},
    }


def _validate_inputs(
    *,
    corpus_path: Path,
    claims_path: Path,
    retrieval_path: Path,
    retrieval_manifest_path: Path,
    evaluator_repo: Path,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    Path,
]:
    for path, role in (
        (corpus_path, "corpus"),
        (claims_path, "claims"),
        (retrieval_path, "retrieval"),
        (retrieval_manifest_path, "retrieval manifest"),
    ):
        _assert_not_dev(path, role)
        if not path.is_file():
            raise NliTrainDiagnosticError(f"missing {role}")
    if _sha256(corpus_path) != OFFICIAL_CORPUS_SHA256:
        raise NliTrainDiagnosticError(
            "corpus is not the pinned official SciFact corpus"
        )
    if _sha256(claims_path) != OFFICIAL_TRAIN_CLAIMS_SHA256:
        raise NliTrainDiagnosticError("claims are not the pinned official train split")

    evaluator = evaluator_repo.resolve() / "evaluator" / "eval.py"
    _assert_not_dev(evaluator, "official evaluator")
    if not evaluator.is_file() or _sha256(evaluator) != OFFICIAL_EVALUATOR_SHA256:
        raise NliTrainDiagnosticError("official evaluator source is not pinned")

    corpus_rows = _read_jsonl(corpus_path)
    claim_rows = _read_jsonl(claims_path)
    retrieval_rows = _read_jsonl(retrieval_path)
    if len(corpus_rows) != SOURCE_CORPUS_COUNT:
        raise NliTrainDiagnosticError("official corpus count mismatch")
    if len(claim_rows) != SOURCE_CLAIM_COUNT:
        raise NliTrainDiagnosticError("official train claim count mismatch")
    if len(retrieval_rows) != SOURCE_CLAIM_COUNT:
        raise NliTrainDiagnosticError("train retrieval coverage is incomplete")

    corpus_ids = {row.get("doc_id") for row in corpus_rows}
    claim_ids = [row.get("id") for row in claim_rows]
    retrieval_ids = [row.get("claim_id") for row in retrieval_rows]
    if len(corpus_ids) != SOURCE_CORPUS_COUNT or any(
        not isinstance(doc_id, int) for doc_id in corpus_ids
    ):
        raise NliTrainDiagnosticError("official corpus IDs are invalid")
    if (
        len(set(claim_ids)) != SOURCE_CLAIM_COUNT
        or any(not isinstance(claim_id, int) for claim_id in claim_ids)
        or retrieval_ids != claim_ids
    ):
        raise NliTrainDiagnosticError("train claim/retrieval ordering is invalid")
    for row in retrieval_rows:
        doc_ids = row.get("doc_ids")
        if (
            not isinstance(doc_ids, list)
            or len(doc_ids) != TOP_K_DOCS
            or len(set(doc_ids)) != TOP_K_DOCS
            or any(doc_id not in corpus_ids for doc_id in doc_ids)
        ):
            raise NliTrainDiagnosticError("retrieval is not known-document Top-3")

    retrieval_manifest = json.loads(retrieval_manifest_path.read_text(encoding="utf-8"))
    if (
        retrieval_manifest.get("schema_version")
        != "scifact-official-tfidf-run-v1"
        or retrieval_manifest.get("data", {}).get("split") != "train"
        or retrieval_manifest.get("data", {}).get("claim_count")
        != SOURCE_CLAIM_COUNT
        or retrieval_manifest.get("configuration", {}).get("top_k") != TOP_K_DOCS
        or retrieval_manifest.get("outputs_sha256", {}).get(retrieval_path.name)
        != _sha256(retrieval_path)
        or retrieval_manifest.get("data", {}).get("files_sha256", {}).get(
            corpus_path.name
        )
        != OFFICIAL_CORPUS_SHA256
        or retrieval_manifest.get("data", {}).get("files_sha256", {}).get(
            claims_path.name
        )
        != OFFICIAL_TRAIN_CLAIMS_SHA256
    ):
        raise NliTrainDiagnosticError("retrieval manifest is not bound to train inputs")
    return corpus_rows, claim_rows, retrieval_rows, evaluator


def _build_candidates(
    corpus_rows: Sequence[dict[str, Any]],
    claims: Sequence[dict[str, Any]],
    retrieval_rows: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[int, list[int]]]:
    corpus = {int(row["doc_id"]): row for row in corpus_rows}
    retrieval = {int(row["claim_id"]): row for row in retrieval_rows}
    candidates: list[dict[str, Any]] = []
    by_claim: dict[int, list[int]] = defaultdict(list)
    for claim in claims:
        claim_id = int(claim["id"])
        for doc_rank, doc_id in enumerate(retrieval[claim_id]["doc_ids"], 1):
            abstract = corpus[int(doc_id)].get("abstract")
            if not isinstance(abstract, list) or any(
                not isinstance(sentence, str) for sentence in abstract
            ):
                raise NliTrainDiagnosticError("corpus abstract is invalid")
            for sentence_index, sentence in enumerate(abstract):
                pair_index = len(candidates)
                candidates.append(
                    {
                        "pair_index": pair_index,
                        "claim_id": claim_id,
                        "doc_rank": doc_rank,
                        "doc_id": int(doc_id),
                        "sentence_index": sentence_index,
                        "sentence": sentence,
                        "claim": claim["claim"],
                    }
                )
                by_claim[claim_id].append(pair_index)
    return candidates, dict(by_claim)


def _assign_lexical_scores(
    candidates: list[dict[str, Any]],
    claims: Sequence[dict[str, Any]],
    by_claim: dict[int, list[int]],
) -> dict[str, Any]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        sublinear_tf=True,
        stop_words="english",
    )
    matrix = vectorizer.fit_transform(
        [candidate["sentence"] for candidate in candidates]
        + [claim["claim"] for claim in claims]
    )
    sentence_matrix = matrix[: len(candidates)]
    claim_matrix = matrix[len(candidates) :]
    for claim_offset, claim in enumerate(claims):
        indices = by_claim[int(claim["id"])]
        scores = cosine_similarity(
            claim_matrix[claim_offset], sentence_matrix[indices]
        ).ravel()
        for pair_index, score in zip(indices, scores):
            candidates[pair_index]["lexical_score"] = float(score)
    return {
        "implementation": "sklearn.feature_extraction.text.TfidfVectorizer",
        "fit_scope": "all candidate sentences plus the 100 unlabeled claim texts",
        "lowercase": True,
        "ngram_range": [1, 2],
        "sublinear_tf": True,
        "stop_words": "english",
        "vocabulary_size": len(vectorizer.vocabulary_),
    }


def _gold_rationale_pairs(
    corpus_rows: Sequence[dict[str, Any]], claims: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    corpus = {int(row["doc_id"]): row for row in corpus_rows}
    rows: list[dict[str, Any]] = []
    for claim in claims:
        for raw_doc_id, rationale_sets in claim["evidence"].items():
            doc_id = int(raw_doc_id)
            abstract = corpus[doc_id]["abstract"]
            for rationale_index, rationale in enumerate(rationale_sets):
                sentence_indices = rationale["sentences"]
                rows.append(
                    {
                        "claim_id": int(claim["id"]),
                        "doc_id": doc_id,
                        "rationale_index": rationale_index,
                        "sentence_indices": sentence_indices,
                        "premise": " ".join(
                            abstract[index] for index in sentence_indices
                        ),
                        "claim": claim["claim"],
                        "gold_label": rationale["label"],
                    }
                )
    return rows


def _infer_probabilities(
    premises: Sequence[str],
    hypotheses: Sequence[str],
    *,
    model_dir: Path,
    device: str,
) -> tuple[list[list[float]], dict[str, Any]]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if device == "cuda" and not torch.cuda.is_available():
        raise NliTrainDiagnosticError("CUDA was requested but is unavailable")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), local_files_only=True
    )
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir), local_files_only=True, torch_dtype=dtype
    ).eval()
    model.to(device)
    label_map = {
        int(key): str(value).lower() for key, value in model.config.id2label.items()
    }
    if label_map != EXPECTED_LABEL_MAP:
        raise NliTrainDiagnosticError("loaded model has an unexpected label mapping")
    probabilities: list[list[float]] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(premises), BATCH_SIZE):
            encoded = tokenizer(
                list(premises[offset : offset + BATCH_SIZE]),
                list(hypotheses[offset : offset + BATCH_SIZE]),
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**encoded).logits.float()
            probabilities.extend(torch.softmax(logits, dim=-1).cpu().tolist())
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return probabilities, {
        "device": device,
        "batch_size": BATCH_SIZE,
        "max_length": MAX_LENGTH,
        "torch_dtype": str(dtype),
        "pair_count": len(premises),
        "elapsed_seconds": elapsed,
        "pairs_per_second": len(premises) / elapsed,
    }


def _rank_candidates(
    candidates: Sequence[dict[str, Any]], indices: Sequence[int]
) -> list[dict[str, Any]]:
    return sorted(
        (candidates[index] for index in indices),
        key=lambda candidate: (
            float(candidate["lexical_score"]),
            int(candidate["pair_index"]),
        ),
        reverse=True,
    )


def _select_prediction(
    *,
    claim_id: int,
    ranked_candidates: Sequence[dict[str, Any]],
    lexical_top_n: Optional[int],
    threshold: float,
) -> tuple[dict[str, Any], str, Optional[dict[str, Any]]]:
    considered = (
        list(ranked_candidates)
        if lexical_top_n is None
        else list(ranked_candidates[:lexical_top_n])
    )
    eligible: list[tuple[float, int, dict[str, Any], str]] = []
    for rank, candidate in enumerate(considered):
        entailment, neutral, contradiction = candidate["nli_probabilities"]
        relation_score = max(entailment, contradiction)
        label = "SUPPORT" if entailment >= contradiction else "CONTRADICT"
        if relation_score >= threshold and relation_score > neutral:
            eligible.append((relation_score, -rank, candidate, label))
    if not eligible:
        return {"id": claim_id, "evidence": {}}, "NEI", None
    _, _, selected, label = max(eligible, key=lambda item: (item[0], item[1]))
    prediction = {
        "id": claim_id,
        "evidence": {
            str(selected["doc_id"]): {
                "label": label,
                "sentences": [selected["sentence_index"]],
            }
        },
    }
    return prediction, label, selected


def _gold_label(claim: dict[str, Any]) -> str:
    if not claim["evidence"]:
        return "NEI"
    labels = {
        rationale["label"]
        for rationale_sets in claim["evidence"].values()
        for rationale in rationale_sets
    }
    if len(labels) != 1:
        raise NliTrainDiagnosticError("claim has inconsistent gold labels")
    return next(iter(labels))


def _f1(relevant: int, retrieved: int, correct: int) -> dict[str, float]:
    precision = correct / retrieved if retrieved else 0.0
    recall = correct / relevant if relevant else 0.0
    value = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": value}


def _official_style_metrics(
    claims: Sequence[dict[str, Any]], predictions: Sequence[dict[str, Any]]
) -> dict[str, dict[str, float]]:
    if [claim["id"] for claim in claims] != [row["id"] for row in predictions]:
        raise NliTrainDiagnosticError("prediction ordering/coverage mismatch")
    abstract = Counter()
    sentence = Counter()
    for claim, prediction_row in zip(claims, predictions):
        gold = claim["evidence"]
        abstract["relevant"] += len(gold)
        sentence["relevant"] += sum(
            len(rationale["sentences"])
            for rationale_sets in gold.values()
            for rationale in rationale_sets
        )
        for raw_doc_id, prediction in prediction_row["evidence"].items():
            abstract["retrieved"] += 1
            sentence["retrieved"] += len(prediction["sentences"])
            rationale_sets = gold.get(str(raw_doc_id), gold.get(int(raw_doc_id)))
            if not rationale_sets:
                continue
            labels = {rationale["label"] for rationale in rationale_sets}
            if len(labels) != 1:
                raise NliTrainDiagnosticError("gold document labels conflict")
            label_correct = prediction["label"] == next(iter(labels))
            abstract["correct_label_only"] += int(label_correct)
            gold_sets = [set(rationale["sentences"]) for rationale in rationale_sets]
            shortest = min(len(gold_set) for gold_set in gold_sets)
            abstract_sentences = set(prediction["sentences"][: max(3, shortest)])
            abstract["correct_rationalized"] += int(
                label_correct
                and any(gold_set.issubset(abstract_sentences) for gold_set in gold_sets)
            )
            predicted_sentences = set(prediction["sentences"])
            correct_sentences = 0
            for index in predicted_sentences:
                containing = [gold_set for gold_set in gold_sets if index in gold_set]
                if len(containing) > 1:
                    raise NliTrainDiagnosticError(
                        "one sentence occurs in multiple gold rationale alternatives"
                    )
                if containing and containing[0].issubset(predicted_sentences):
                    correct_sentences += 1
            sentence["correct_selection"] += correct_sentences
            sentence["correct_label"] += int(label_correct) * correct_sentences
    metrics = {
        "abstract_label_only": _f1(
            abstract["relevant"],
            abstract["retrieved"],
            abstract["correct_label_only"],
        ),
        "abstract_rationalized": _f1(
            abstract["relevant"],
            abstract["retrieved"],
            abstract["correct_rationalized"],
        ),
        "sentence_selection": _f1(
            sentence["relevant"],
            sentence["retrieved"],
            sentence["correct_selection"],
        ),
        "sentence_label": _f1(
            sentence["relevant"],
            sentence["retrieved"],
            sentence["correct_label"],
        ),
    }
    if any(
        not math.isfinite(value)
        for group in metrics.values()
        for value in group.values()
    ):
        raise NliTrainDiagnosticError("non-finite official-style metric")
    return metrics


def _strategy_result(
    *,
    claims: Sequence[dict[str, Any]],
    candidates: Sequence[dict[str, Any]],
    by_claim: dict[int, list[int]],
    retrieval_by_claim: dict[int, list[int]],
    lexical_top_n: Optional[int],
    threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    predictions: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    labels: list[str] = []
    for claim in claims:
        claim_id = int(claim["id"])
        ranked = _rank_candidates(candidates, by_claim[claim_id])
        prediction, label, selected = _select_prediction(
            claim_id=claim_id,
            ranked_candidates=ranked,
            lexical_top_n=lexical_top_n,
            threshold=threshold,
        )
        predictions.append(prediction)
        labels.append(label)
        cases.append(
            {
                "claim_id": claim_id,
                "gold_label": _gold_label(claim),
                "predicted_label": label,
                "prediction": prediction,
                "selected_pair_index": (
                    selected["pair_index"] if selected is not None else None
                ),
            }
        )
    positive_indices = [
        index for index, claim in enumerate(claims) if claim["evidence"]
    ]
    reachable_indices = [
        index
        for index in positive_indices
        if {int(doc_id) for doc_id in claims[index]["evidence"]}
        & set(retrieval_by_claim[int(claims[index]["id"])])
    ]

    def predicted_gold_document(index: int) -> bool:
        return bool(
            {int(doc_id) for doc_id in predictions[index]["evidence"]}
            & {int(doc_id) for doc_id in claims[index]["evidence"]}
        )

    summary = {
        "lexical_top_n": lexical_top_n,
        "nli_threshold": threshold,
        "predicted_label_distribution": dict(Counter(labels)),
        "nonempty_prediction_rate": sum(label != "NEI" for label in labels)
        / len(labels),
        "claim_label_accuracy": sum(
            label == _gold_label(claim) for label, claim in zip(labels, claims)
        )
        / len(labels),
        "gold_document_hit_all_positive": sum(
            predicted_gold_document(index) for index in positive_indices
        )
        / len(positive_indices),
        "gold_document_hit_retrieval_reachable": sum(
            predicted_gold_document(index) for index in reachable_indices
        )
        / len(reachable_indices),
        "official_style_metrics": _official_style_metrics(claims, predictions),
    }
    return summary, predictions, cases


def _locator_recall(
    claims: Sequence[dict[str, Any]],
    candidates: Sequence[dict[str, Any]],
    by_claim: dict[int, list[int]],
    retrieval_by_claim: dict[int, list[int]],
) -> dict[str, Any]:
    positive = [claim for claim in claims if claim["evidence"]]
    reachable = [
        claim
        for claim in positive
        if {int(doc_id) for doc_id in claim["evidence"]}
        & set(retrieval_by_claim[int(claim["id"])])
    ]

    def hits(claim: dict[str, Any], top_n: int) -> bool:
        gold = {
            (int(doc_id), sentence_index)
            for doc_id, rationale_sets in claim["evidence"].items()
            for rationale in rationale_sets
            for sentence_index in rationale["sentences"]
        }
        ranked = _rank_candidates(candidates, by_claim[int(claim["id"])])[:top_n]
        return bool(
            gold
            & {
                (int(candidate["doc_id"]), int(candidate["sentence_index"]))
                for candidate in ranked
            }
        )

    return {
        "positive_claim_count": len(positive),
        "retrieval_reachable_positive_count": len(reachable),
        "tfidf_top3_gold_document_recall_over_positive_claims": len(reachable)
        / len(positive),
        "sentence_locator_recall": {
            str(top_n): {
                "all_positive": sum(hits(claim, top_n) for claim in positive)
                / len(positive),
                "retrieval_reachable": sum(
                    hits(claim, top_n) for claim in reachable
                )
                / len(reachable),
            }
            for top_n in (1, 3, 5, 10, 20)
        },
    }


def _oracle_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    three_way_labels = ("SUPPORT", "NEI", "CONTRADICT")
    for row in rows:
        probabilities = row["nli_probabilities"]
        row["three_way_prediction"] = three_way_labels[
            max(range(3), key=lambda index: probabilities[index])
        ]
        row["forced_relation_prediction"] = (
            "SUPPORT" if probabilities[0] >= probabilities[2] else "CONTRADICT"
        )
    return {
        "scope": "oracle_gold_rationales_train_only_diagnostic",
        "rationale_count": len(rows),
        "multi_sentence_rationale_count": sum(
            len(row["sentence_indices"]) > 1 for row in rows
        ),
        "gold_label_distribution": dict(Counter(row["gold_label"] for row in rows)),
        "three_way_prediction_distribution": dict(
            Counter(row["three_way_prediction"] for row in rows)
        ),
        "three_way_accuracy": sum(
            row["three_way_prediction"] == row["gold_label"] for row in rows
        )
        / len(rows),
        "forced_relation_prediction_distribution": dict(
            Counter(row["forced_relation_prediction"] for row in rows)
        ),
        "forced_relation_accuracy": sum(
            row["forced_relation_prediction"] == row["gold_label"] for row in rows
        )
        / len(rows),
        "forced_relation_accuracy_by_gold": {
            label: sum(
                row["forced_relation_prediction"] == label
                for row in rows
                if row["gold_label"] == label
            )
            / sum(row["gold_label"] == label for row in rows)
            for label in ("SUPPORT", "CONTRADICT")
        },
    }


def _flatten_metrics(metrics: dict[str, dict[str, float]]) -> dict[str, float]:
    return {
        f"{group}_{metric}": value
        for group, values in metrics.items()
        for metric, value in values.items()
    }


def _same_metrics(left: dict[str, Any], right: dict[str, float]) -> bool:
    return set(left) == set(right) and all(
        isinstance(left[key], (int, float))
        and abs(float(left[key]) - right[key]) <= 1e-12
        for key in right
    )


def run(
    *,
    corpus_path: Path,
    claims_path: Path,
    retrieval_path: Path,
    retrieval_manifest_path: Path,
    model_dir: Path,
    evaluator_repo: Path,
    output: Path,
    device: str,
) -> Path:
    destination = output.resolve()
    _assert_not_dev(destination, "output")
    if destination.exists():
        raise NliTrainDiagnosticError("output directory already exists (exclusive run)")
    if device not in {"cuda", "cpu"}:
        raise NliTrainDiagnosticError("device must be cuda or cpu")

    corpus_rows, all_claims, all_retrieval, evaluator = _validate_inputs(
        corpus_path=corpus_path.resolve(),
        claims_path=claims_path.resolve(),
        retrieval_path=retrieval_path.resolve(),
        retrieval_manifest_path=retrieval_manifest_path.resolve(),
        evaluator_repo=evaluator_repo.resolve(),
    )
    model = _model_identity(model_dir)
    claims = all_claims[:SAMPLE_COUNT]
    retrieval_rows = all_retrieval[:SAMPLE_COUNT]
    retrieval_by_claim = {
        int(row["claim_id"]): [int(doc_id) for doc_id in row["doc_ids"]]
        for row in retrieval_rows
    }
    candidates, by_claim = _build_candidates(corpus_rows, claims, retrieval_rows)
    lexical_configuration = _assign_lexical_scores(candidates, claims, by_claim)
    oracle_rows = _gold_rationale_pairs(corpus_rows, claims)
    premises = [candidate["sentence"] for candidate in candidates] + [
        row["premise"] for row in oracle_rows
    ]
    hypotheses = [candidate["claim"] for candidate in candidates] + [
        row["claim"] for row in oracle_rows
    ]
    probabilities, inference = _infer_probabilities(
        premises, hypotheses, model_dir=model_dir.resolve(), device=device
    )
    candidate_probabilities = probabilities[: len(candidates)]
    oracle_probabilities = probabilities[len(candidates) :]
    for candidate, values in zip(candidates, candidate_probabilities):
        candidate["nli_probabilities"] = values
    for row, values in zip(oracle_rows, oracle_probabilities):
        row["nli_probabilities"] = values

    primary, primary_predictions, primary_cases = _strategy_result(
        claims=claims,
        candidates=candidates,
        by_claim=by_claim,
        retrieval_by_claim=retrieval_by_claim,
        lexical_top_n=LEXICAL_TOP_N,
        threshold=PRIMARY_NLI_THRESHOLD,
    )
    strategy_sweep: list[dict[str, Any]] = []
    for threshold in DIRECT_THRESHOLDS:
        result, _, _ = _strategy_result(
            claims=claims,
            candidates=candidates,
            by_claim=by_claim,
            retrieval_by_claim=retrieval_by_claim,
            lexical_top_n=None,
            threshold=threshold,
        )
        strategy_sweep.append(result)
    for top_n in LEXICAL_TOP_NS:
        for threshold in LEXICAL_THRESHOLDS:
            result, _, _ = _strategy_result(
                claims=claims,
                candidates=candidates,
                by_claim=by_claim,
                retrieval_by_claim=retrieval_by_claim,
                lexical_top_n=top_n,
                threshold=threshold,
            )
            strategy_sweep.append(result)

    ranked_by_claim = {
        claim_id: _rank_candidates(candidates, indices)
        for claim_id, indices in by_claim.items()
    }
    for case in primary_cases:
        case["retrieved_doc_ids"] = retrieval_by_claim[case["claim_id"]]
        case["ranked_candidates"] = ranked_by_claim[case["claim_id"]]

    configuration = {
        "schema_version": SCHEMA_VERSION,
        "evaluation_mode": EVALUATION_MODE,
        "reportable_as_generalization": False,
        "sample_policy": SAMPLE_POLICY,
        "source_split": "train",
        "sample_count": SAMPLE_COUNT,
        "top_k_documents": TOP_K_DOCS,
        "primary_strategy": {
            "sentence_locator": "global_word_tfidf",
            "lexical_top_n": LEXICAL_TOP_N,
            "nli_relation_score_threshold": PRIMARY_NLI_THRESHOLD,
            "selection": (
                "within lexical Top-N choose highest max(entailment, contradiction); "
                "require relation score >= threshold and > neutral"
            ),
            "empty_policy": "no eligible sentence means NEI and empty evidence",
        },
        "lexical_locator": lexical_configuration,
        "nli": {
            "premise": "one abstract sentence",
            "hypothesis": "claim",
            "probability_order": ["entailment", "neutral", "contradiction"],
            "relation_map": {
                "entailment": "SUPPORT",
                "contradiction": "CONTRADICT",
            },
            "batch_size": BATCH_SIZE,
            "max_length": MAX_LENGTH,
        },
        "strategy_sweep": {
            "direct_thresholds": list(DIRECT_THRESHOLDS),
            "lexical_top_ns": list(LEXICAL_TOP_NS),
            "lexical_thresholds": list(LEXICAL_THRESHOLDS),
            "warning": "exploratory train-only sweep; selecting the best row is tuning",
        },
    }
    diagnostic = {
        "schema_version": SCHEMA_VERSION,
        "evaluation_mode": EVALUATION_MODE,
        "warning": (
            "These train-only, same-sample exploratory measurements are not dev, "
            "holdout, test, or generalization results."
        ),
        "gold_label_distribution": dict(Counter(_gold_label(c) for c in claims)),
        "locator": _locator_recall(
            claims, candidates, by_claim, retrieval_by_claim
        ),
        "primary": primary,
        "strategy_sweep": strategy_sweep,
        "oracle_relation_diagnostic": _oracle_summary(oracle_rows),
    }

    destination.mkdir(parents=True, exist_ok=False)
    selected_claims_path = destination / "claims_train_first100.jsonl"
    selected_retrieval_path = destination / "retrieval_train_first100.jsonl"
    predictions_path = destination / "predictions_primary.jsonl"
    _write_exclusive(selected_claims_path, _jsonl_bytes(claims))
    _write_exclusive(selected_retrieval_path, _jsonl_bytes(retrieval_rows))
    _write_exclusive(destination / "configuration.json", _json_bytes(configuration))
    _write_exclusive(predictions_path, _jsonl_bytes(primary_predictions))
    _write_exclusive(destination / "cases.jsonl", _jsonl_bytes(primary_cases))
    _write_exclusive(destination / "oracle_rationales.jsonl", _jsonl_bytes(oracle_rows))
    _write_exclusive(destination / "diagnostic_metrics.json", _json_bytes(diagnostic))

    official_metrics_path = destination / "official_metrics.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(evaluator),
            "--labels_file",
            str(selected_claims_path),
            "--preds_file",
            str(predictions_path),
            "--metrics_output_file",
            str(official_metrics_path),
            "--verbose",
        ],
        cwd=evaluator_repo.resolve(),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    _write_exclusive(destination / "official_stdout.txt", completed.stdout.encode())
    _write_exclusive(destination / "official_stderr.txt", completed.stderr.encode())
    if completed.returncode:
        raise NliTrainDiagnosticError(
            f"official evaluator failed with exit code {completed.returncode}"
        )
    official_metrics = json.loads(official_metrics_path.read_text(encoding="utf-8"))
    independently_flattened = _flatten_metrics(primary["official_style_metrics"])
    if not _same_metrics(official_metrics, independently_flattened):
        raise NliTrainDiagnosticError(
            "official evaluator disagrees with independent official-style scoring"
        )

    output_hashes = {
        path.name: _sha256(path)
        for path in sorted(destination.iterdir())
        if path.is_file()
    }
    script_path = Path(__file__).resolve()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_status": {
            "mode": EVALUATION_MODE,
            "reportable_as_generalization": False,
            "source_split": "train",
            "dev_read_or_scored": False,
        },
        "implementation": {
            "path": str(script_path),
            "sha256": _sha256(script_path),
        },
        "inputs": {
            "corpus": {
                "path": str(corpus_path.resolve()),
                "sha256": _sha256(corpus_path.resolve()),
                "row_count": len(corpus_rows),
            },
            "claims_train": {
                "path": str(claims_path.resolve()),
                "sha256": _sha256(claims_path.resolve()),
                "source_row_count": len(all_claims),
                "selected_row_count": len(claims),
                "selection": SAMPLE_POLICY,
                "selected_ids_sha256": _sha256_bytes(
                    _canonical_bytes([claim["id"] for claim in claims])
                ),
            },
            "tfidf_top3_train": {
                "path": str(retrieval_path.resolve()),
                "sha256": _sha256(retrieval_path.resolve()),
                "manifest_path": str(retrieval_manifest_path.resolve()),
                "manifest_sha256": _sha256(retrieval_manifest_path.resolve()),
                "source_row_count": len(all_retrieval),
                "selected_row_count": len(retrieval_rows),
            },
            "official_evaluator": {
                "path": str(evaluator),
                "sha256": _sha256(evaluator),
            },
        },
        "model": model,
        "configuration_sha256": _sha256(destination / "configuration.json"),
        "counts": {
            "claims": len(claims),
            "candidate_sentence_pairs": len(candidates),
            "oracle_rationale_pairs": len(oracle_rows),
            "total_nli_pairs": len(probabilities),
        },
        "inference": inference,
        "environment": {
            "python": sys.version,
            "torch": importlib.metadata.version("torch"),
            "transformers": importlib.metadata.version("transformers"),
            "scikit_learn": importlib.metadata.version("scikit-learn"),
        },
        "validation": {
            "official_input_hashes": True,
            "retrieval_manifest_train_binding": True,
            "exact_first_100_file_order": True,
            "official_evaluator_cross_check": True,
            "absolute_metric_tolerance": 1e-12,
        },
        "outputs_sha256": output_hashes,
    }
    _write_exclusive(destination / "run_manifest.json", _json_bytes(manifest))
    return destination


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--claims-train", type=Path, required=True)
    parser.add_argument("--retrieval-train", type=Path, required=True)
    parser.add_argument("--retrieval-manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--official-evaluator-repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args(argv)


def main() -> None:
    args = _parse_args()
    print(
        run(
            corpus_path=args.corpus,
            claims_path=args.claims_train,
            retrieval_path=args.retrieval_train,
            retrieval_manifest_path=args.retrieval_manifest,
            model_dir=args.model,
            evaluator_repo=args.official_evaluator_repo,
            output=args.out,
            device=args.device,
        )
    )


if __name__ == "__main__":
    main()
