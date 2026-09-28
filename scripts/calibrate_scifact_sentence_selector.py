"""Calibrate an auditable SciFact sentence selector on official train only.

The executable contract is deliberately narrow: all 809 official train claims,
the frozen TF-IDF Top-3 retrieval artifact, and one pinned DeBERTa inference pass.
No dev/test input exists in the CLI.  Selection quality is estimated with nested
out-of-fold predictions whose groups are connected components induced by shared
Top-3 documents.  A deployable full-train model is emitted only after every
pre-registered acceptance check passes.
"""

from __future__ import annotations

# The pinned CUDA environment is Python 3.9.
# ruff: noqa: B905, E501, RUF001, RUF046, UP017, UP032, UP045
import argparse
import hashlib
import importlib.metadata
import json
import math
import random
import re
import statistics
import sys
import time
import warnings
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

SCHEMA_VERSION = "scifact-sentence-selector-calibration-v1"
EVALUATION_MODE = "nested_oof_official_train_only_not_generalization"
SOURCE_SPLIT = "train"
SOURCE_CLAIM_COUNT = 809
SOURCE_CORPUS_COUNT = 5183
RELATION_CLAIM_COUNT = 505
NEI_CLAIM_COUNT = 304
RETRIEVABLE_RELATION_CLAIM_COUNT = 396
CANDIDATE_PAIR_COUNT = 22_266
POSITIVE_PAIR_COUNT = 760
CONNECTED_COMPONENT_COUNT = 228
LARGEST_COMPONENT_CLAIM_COUNT = 191
TOP_K_DOCS = 3
OUTER_FOLDS = 4
INNER_FOLDS = 3
BOOTSTRAP_SAMPLES = 10_000
RANDOM_SEED = 20260904
BATCH_SIZE = 8
MAX_LENGTH = 384
MODEL_REVISION = "6f5cf0a2b59cabb106aca4c287eed12e357e90eb"
OFFICIAL_CORPUS_SHA256 = (
    "b8d6c89624cb2ed74dee8938effc4f5d8bd2086887880af8110d64be4ceade62"
)
OFFICIAL_TRAIN_CLAIMS_SHA256 = (
    "f4c8fa82d8bd0653a9cc8d61a6ea48c25eacea64e90af5dbf390ebb1b74372f0"
)
OFFICIAL_TFIDF_TRAIN_RETRIEVAL_SHA256 = (
    "ce1b6a9bb8e2b02214dc1fa1ae62ea62a2a13660192e67bbbf609e0509edf7a1"
)
OFFICIAL_TFIDF_TRAIN_MANIFEST_SHA256 = (
    "76d1707c8b10d1c705579cdad36b090929d26ef0c1dec6ba556fffc48137f2a2"
)
OFFICIAL_SCIFACT_COMMIT = "68b98a56d93e0f9da0d2aab4e6c3294699a0f72e"
OFFICIAL_SCIFACT_ARCHIVE_SHA256 = (
    "aba85f44b80e4014a93b905de8f91929104d1bc4886950e2ec33be243355f75b"
)
PINNED_MODEL_IDENTITY_SHA256 = (
    "9194298eb4045bb0a48daa2c01ecadfa981fe0300fa87c5975c18f5ec972a5e2"
)
EXPECTED_LABEL_MAP = {0: "entailment", 1: "neutral", 2: "contradiction"}
DEV_TOKEN = re.compile(r"(^|[_.-])dev([_.-]|$)", re.IGNORECASE)

STATIC_FEATURE_NAMES = (
    "word_jaccard",
    "claim_token_coverage",
    "sentence_token_coverage",
    "numeric_token_set_match",
    "negation_presence_match",
    "length_ratio",
    "reciprocal_doc_rank",
    "normalized_sentence_position",
    "nli_entailment",
    "nli_neutral",
    "nli_contradiction",
    "nli_relation_probability",
    "nli_relation_vs_neutral_margin",
    "nli_polarity_margin_abs",
)
FOLD_FIT_FEATURE_NAMES = (
    "word_tfidf_cosine",
    "char_tfidf_cosine",
)
FEATURE_NAMES = FOLD_FIT_FEATURE_NAMES + STATIC_FEATURE_NAMES
THRESHOLDS = (0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)
MAX_SENTENCES = (1, 2, 3, 4)
STRATEGIES = ("hybrid", "lexical_only", "nli_only")
NEGATION_TERMS = frozenset(
    {"no", "not", "never", "neither", "nor", "none", "without", "cannot"}
)
NUMBER_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:e[+-]?\d+)?%?",
    re.IGNORECASE,
)

MIN_PRIMARY_DELTA = 0.020
MIN_ALL_RELATION_ANY_GOLD_RECALL = 0.50
MIN_RETRIEVABLE_ANY_GOLD_RECALL = 0.65
MIN_NEI_CORRECT_EMPTY = 0.70
MIN_NONINFERIOR_OUTER_FOLDS = 3
LEXICAL_BASELINE_FORMULA = (
    "0.30*word_tfidf_cosine + 0.20*char_tfidf_cosine + "
    "0.15*word_jaccard + 0.10*claim_token_coverage + "
    "0.05*sentence_token_coverage + 0.05*numeric_token_set_match + "
    "0.05*negation_presence_match + 0.04*length_ratio + "
    "0.04*reciprocal_doc_rank + "
    "0.02*(1-normalized_sentence_position)"
)


class SelectorCalibrationError(RuntimeError):
    """Raised when the train-only calibration contract is violated."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
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
                raise SelectorCalibrationError(
                    "expected JSON object at {}:{}".format(path, line_number)
                )
            rows.append(value)
    return rows


def _assert_not_dev(path: Path, role: str) -> None:
    if any(DEV_TOKEN.search(part) for part in path.resolve().parts):
        raise SelectorCalibrationError(
            "{} path contains a forbidden dev token".format(role)
        )


def _model_identity(model_dir: Path) -> dict[str, Any]:
    root = model_dir.resolve()
    if not root.is_dir() or root.name != MODEL_REVISION:
        raise SelectorCalibrationError("model must be the pinned local DeBERTa revision")
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
        raise SelectorCalibrationError("local model snapshot is incomplete")
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    label_map = {
        int(key): str(value).lower() for key, value in config["id2label"].items()
    }
    if label_map != EXPECTED_LABEL_MAP:
        raise SelectorCalibrationError("unexpected DeBERTa NLI label mapping")
    identity = {
        "path": str(root),
        "revision": MODEL_REVISION,
        "identity_sha256": _sha256_bytes(_canonical_bytes(files)),
        "files": files,
        "id2label": {str(key): value for key, value in label_map.items()},
    }
    if identity["identity_sha256"] != PINNED_MODEL_IDENTITY_SHA256:
        raise SelectorCalibrationError("pinned DeBERTa model identity mismatch")
    return identity


def _validate_inputs(
    *,
    corpus_path: Path,
    claims_path: Path,
    retrieval_path: Path,
    retrieval_manifest_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    for path, role in (
        (corpus_path, "corpus"),
        (claims_path, "claims train"),
        (retrieval_path, "retrieval train"),
        (retrieval_manifest_path, "retrieval train manifest"),
    ):
        _assert_not_dev(path, role)
        if not path.is_file():
            raise SelectorCalibrationError("missing {}".format(role))
    if _sha256(corpus_path) != OFFICIAL_CORPUS_SHA256:
        raise SelectorCalibrationError("corpus is not the pinned official SciFact corpus")
    if _sha256(claims_path) != OFFICIAL_TRAIN_CLAIMS_SHA256:
        raise SelectorCalibrationError("claims are not the pinned official train split")
    if _sha256(retrieval_path) != OFFICIAL_TFIDF_TRAIN_RETRIEVAL_SHA256:
        raise SelectorCalibrationError("retrieval is not the pinned TF-IDF train artifact")
    if _sha256(retrieval_manifest_path) != OFFICIAL_TFIDF_TRAIN_MANIFEST_SHA256:
        raise SelectorCalibrationError("TF-IDF train manifest identity mismatch")

    corpus_rows = _read_jsonl(corpus_path)
    claim_rows = _read_jsonl(claims_path)
    retrieval_rows = _read_jsonl(retrieval_path)
    if len(corpus_rows) != SOURCE_CORPUS_COUNT:
        raise SelectorCalibrationError("official corpus count mismatch")
    if len(claim_rows) != SOURCE_CLAIM_COUNT:
        raise SelectorCalibrationError("official train claim count mismatch")
    if len(retrieval_rows) != SOURCE_CLAIM_COUNT:
        raise SelectorCalibrationError("train retrieval coverage is incomplete")

    corpus_ids = {row.get("doc_id") for row in corpus_rows}
    claim_ids = [row.get("id") for row in claim_rows]
    retrieval_ids = [row.get("claim_id") for row in retrieval_rows]
    if len(corpus_ids) != SOURCE_CORPUS_COUNT or any(
        not isinstance(doc_id, int) for doc_id in corpus_ids
    ):
        raise SelectorCalibrationError("official corpus IDs are invalid")
    if (
        len(set(claim_ids)) != SOURCE_CLAIM_COUNT
        or any(not isinstance(claim_id, int) for claim_id in claim_ids)
        or retrieval_ids != claim_ids
    ):
        raise SelectorCalibrationError("train claim/retrieval ordering is invalid")
    for row in retrieval_rows:
        doc_ids = row.get("doc_ids")
        if (
            not isinstance(doc_ids, list)
            or len(doc_ids) != TOP_K_DOCS
            or len(set(doc_ids)) != TOP_K_DOCS
            or any(doc_id not in corpus_ids for doc_id in doc_ids)
        ):
            raise SelectorCalibrationError("retrieval is not known-document Top-3")

    manifest = json.loads(retrieval_manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != "scifact-official-tfidf-run-v1"
        or manifest.get("official_source", {}).get("commit")
        != OFFICIAL_SCIFACT_COMMIT
        or manifest.get("official_source", {}).get("archive_sha256")
        != OFFICIAL_SCIFACT_ARCHIVE_SHA256
        or manifest.get("data", {}).get("split") != SOURCE_SPLIT
        or manifest.get("data", {}).get("claim_count") != SOURCE_CLAIM_COUNT
        or manifest.get("configuration", {}).get("top_k") != TOP_K_DOCS
        or manifest.get("configuration", {}).get("min_gram") != 1
        or manifest.get("configuration", {}).get("max_gram") != 2
        or manifest.get("configuration", {}).get("stop_words") != "english"
        or manifest.get("configuration", {}).get("document_text")
        != "title + ' '.join(abstract)"
        or manifest.get("outputs_sha256", {}).get(retrieval_path.name)
        != _sha256(retrieval_path)
        or manifest.get("data", {}).get("files_sha256", {}).get(corpus_path.name)
        != OFFICIAL_CORPUS_SHA256
        or manifest.get("data", {}).get("files_sha256", {}).get(claims_path.name)
        != OFFICIAL_TRAIN_CLAIMS_SHA256
    ):
        raise SelectorCalibrationError("retrieval manifest is not bound to train inputs")

    relation_count = sum(bool(row.get("evidence")) for row in claim_rows)
    if relation_count != RELATION_CLAIM_COUNT:
        raise SelectorCalibrationError("official train relation-claim count mismatch")
    if len(claim_rows) - relation_count != NEI_CLAIM_COUNT:
        raise SelectorCalibrationError("official train NEI count mismatch")
    return corpus_rows, claim_rows, retrieval_rows


def _gold_alternatives(claim: Mapping[str, Any]) -> list[frozenset[tuple[int, int]]]:
    alternatives: list[frozenset[tuple[int, int]]] = []
    evidence = claim.get("evidence")
    if not isinstance(evidence, dict):
        raise SelectorCalibrationError("claim evidence must be an object")
    for raw_doc_id, rationale_sets in evidence.items():
        doc_id = int(raw_doc_id)
        if not isinstance(rationale_sets, list):
            raise SelectorCalibrationError("gold rationale sets must be a list")
        for rationale in rationale_sets:
            indices = rationale.get("sentences")
            if not isinstance(indices, list) or not indices:
                raise SelectorCalibrationError("gold rationale must be nonempty")
            pairs = frozenset((doc_id, int(index)) for index in indices)
            if len(pairs) != len(indices):
                raise SelectorCalibrationError("gold rationale has duplicate sentences")
            alternatives.append(pairs)
    return alternatives


def _token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _numeric_tokens(text: str) -> set[str]:
    from decimal import Decimal, InvalidOperation

    normalized: set[str] = set()
    for match in NUMBER_PATTERN.findall(text.replace(",", "")):
        percent = match.endswith("%")
        raw = match[:-1] if percent else match
        try:
            value = Decimal(raw).normalize()
        except InvalidOperation as exc:
            raise SelectorCalibrationError("numeric token normalization failed") from exc
        token = format(value, "f")
        if "." in token:
            token = token.rstrip("0").rstrip(".")
        if token == "-0":
            token = "0"
        normalized.add(token + ("%" if percent else ""))
    return normalized


def _has_negation(text: str) -> bool:
    lowered = text.casefold()
    tokens = set(re.findall(r"[a-z]+", lowered))
    return bool(tokens & NEGATION_TERMS) or bool(re.search(r"\b[a-z]+n['’]t\b", lowered))


def _lexical_features(claim: str, sentence: str) -> dict[str, float]:
    claim_tokens = _token_set(claim)
    sentence_tokens = _token_set(sentence)
    overlap = claim_tokens & sentence_tokens
    union = claim_tokens | sentence_tokens
    shorter = min(len(claim_tokens), len(sentence_tokens))
    longer = max(len(claim_tokens), len(sentence_tokens))
    return {
        "word_jaccard": _safe_ratio(len(overlap), len(union)),
        "claim_token_coverage": _safe_ratio(len(overlap), len(claim_tokens)),
        "sentence_token_coverage": _safe_ratio(len(overlap), len(sentence_tokens)),
        "numeric_token_set_match": float(
            _numeric_tokens(claim) == _numeric_tokens(sentence)
        ),
        "negation_presence_match": float(_has_negation(claim) == _has_negation(sentence)),
        "length_ratio": _safe_ratio(shorter, longer),
    }


def _build_candidates(
    corpus_rows: Sequence[dict[str, Any]],
    claims: Sequence[dict[str, Any]],
    retrieval_rows: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[int, list[int]], dict[int, list[int]]]:
    corpus = {int(row["doc_id"]): row for row in corpus_rows}
    retrieval = {int(row["claim_id"]): row for row in retrieval_rows}
    candidates: list[dict[str, Any]] = []
    by_claim: dict[int, list[int]] = defaultdict(list)
    retrieval_by_claim: dict[int, list[int]] = {}
    for claim in claims:
        claim_id = int(claim["id"])
        claim_text = claim.get("claim")
        if not isinstance(claim_text, str) or not claim_text.strip():
            raise SelectorCalibrationError("claim text is invalid")
        doc_ids = [int(value) for value in retrieval[claim_id]["doc_ids"]]
        retrieval_by_claim[claim_id] = doc_ids
        for doc_rank, doc_id in enumerate(doc_ids, 1):
            abstract = corpus[doc_id].get("abstract")
            if not isinstance(abstract, list) or any(
                not isinstance(sentence, str) for sentence in abstract
            ):
                raise SelectorCalibrationError("corpus abstract is invalid")
            for sentence_index, sentence in enumerate(abstract):
                pair_index = len(candidates)
                row = {
                    "pair_index": pair_index,
                    "claim_id": claim_id,
                    "doc_id": doc_id,
                    "doc_rank": doc_rank,
                    "sentence_index": sentence_index,
                    "document_sentence_count": len(abstract),
                    "claim": claim_text,
                    "sentence": sentence,
                }
                row.update(_lexical_features(claim_text, sentence))
                candidates.append(row)
                by_claim[claim_id].append(pair_index)
    return candidates, dict(by_claim), retrieval_by_claim


def _validate_candidate_keys(
    candidates: Sequence[dict[str, Any]],
    by_claim: Mapping[int, Sequence[int]],
    expected_claim_ids: set[int],
) -> dict[str, Any]:
    pair_indices = [int(row["pair_index"]) for row in candidates]
    keys = [
        (int(row["claim_id"]), int(row["doc_id"]), int(row["sentence_index"]))
        for row in candidates
    ]
    indexed = [index for claim_id in sorted(by_claim) for index in by_claim[claim_id]]
    result = {
        "pair_index_contiguous": pair_indices == list(range(len(candidates))),
        "candidate_key_unique": len(set(keys)) == len(keys),
        "by_claim_coverage_complete": sorted(indexed) == list(range(len(candidates))),
        "by_claim_claim_coverage_complete": set(by_claim) == expected_claim_ids,
    }
    if not all(result.values()):
        raise SelectorCalibrationError("candidate key uniqueness/coverage check failed")
    return result


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
        raise SelectorCalibrationError("CUDA was requested but is unavailable")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir), local_files_only=True, torch_dtype=dtype
    ).eval()
    model.to(device)
    label_map = {
        int(key): str(value).lower() for key, value in model.config.id2label.items()
    }
    if label_map != EXPECTED_LABEL_MAP:
        raise SelectorCalibrationError("loaded model has an unexpected label mapping")
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
        "inference_pass_count": 1,
        "device": device,
        "batch_size": BATCH_SIZE,
        "max_length": MAX_LENGTH,
        "torch_dtype": str(dtype),
        "pair_count": len(premises),
        "elapsed_seconds": elapsed,
        "pairs_per_second": len(premises) / elapsed,
    }


def _freeze_features(
    candidates: list[dict[str, Any]], probabilities: Sequence[Sequence[float]]
) -> list[dict[str, Any]]:
    if len(candidates) != len(probabilities):
        raise SelectorCalibrationError("NLI output coverage mismatch")
    frozen: list[dict[str, Any]] = []
    for candidate, raw_values in zip(candidates, probabilities):
        values = [float(value) for value in raw_values]
        if (
            len(values) != 3
            or any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in values)
            or abs(sum(values) - 1.0) > 1e-4
        ):
            raise SelectorCalibrationError("invalid NLI probability vector")
        entailment, neutral, contradiction = values
        feature_values = {
            "word_jaccard": float(candidate["word_jaccard"]),
            "claim_token_coverage": float(candidate["claim_token_coverage"]),
            "sentence_token_coverage": float(candidate["sentence_token_coverage"]),
            "numeric_token_set_match": float(candidate["numeric_token_set_match"]),
            "negation_presence_match": float(candidate["negation_presence_match"]),
            "length_ratio": float(candidate["length_ratio"]),
            "reciprocal_doc_rank": 1.0 / int(candidate["doc_rank"]),
            "normalized_sentence_position": (int(candidate["sentence_index"]) + 1)
            / int(candidate["document_sentence_count"]),
            "nli_entailment": entailment,
            "nli_neutral": neutral,
            "nli_contradiction": contradiction,
            "nli_relation_probability": max(entailment, contradiction),
            "nli_relation_vs_neutral_margin": max(entailment, contradiction) - neutral,
            "nli_polarity_margin_abs": abs(entailment - contradiction),
        }
        if tuple(feature_values) != STATIC_FEATURE_NAMES:
            raise SelectorCalibrationError("feature whitelist violation")
        frozen.append(
            {
                "pair_index": int(candidate["pair_index"]),
                "claim_id": int(candidate["claim_id"]),
                "doc_id": int(candidate["doc_id"]),
                "doc_rank": int(candidate["doc_rank"]),
                "sentence_index": int(candidate["sentence_index"]),
                "document_sentence_count": int(candidate["document_sentence_count"]),
                "claim_sha256": _sha256_bytes(candidate["claim"].encode("utf-8")),
                "sentence_sha256": _sha256_bytes(candidate["sentence"].encode("utf-8")),
                "claim": candidate["claim"],
                "sentence": candidate["sentence"],
                "static_features": feature_values,
            }
        )
    return frozen


def _build_targets(
    claims: Sequence[dict[str, Any]], frozen: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    gold_union: dict[int, set[tuple[int, int]]] = {}
    for claim in claims:
        alternatives = _gold_alternatives(claim)
        gold_union[int(claim["id"])] = (
            set().union(*alternatives) if alternatives else set()
        )
    return [
        {
            "pair_index": int(row["pair_index"]),
            "claim_id": int(row["claim_id"]),
            "is_gold_sentence": (int(row["doc_id"]), int(row["sentence_index"]))
            in gold_union[int(row["claim_id"])],
        }
        for row in frozen
    ]


class _DisjointSet:
    def __init__(self, values: Sequence[int]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: int) -> int:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def _connected_components(
    claim_ids: Sequence[int],
    retrieval_by_claim: Mapping[int, Sequence[int]],
    claim_text_by_id: Mapping[int, str],
) -> list[dict[str, Any]]:
    disjoint = _DisjointSet(claim_ids)
    first_claim_by_doc: dict[int, int] = {}
    for claim_id in claim_ids:
        for doc_id in retrieval_by_claim[claim_id]:
            prior = first_claim_by_doc.setdefault(int(doc_id), claim_id)
            disjoint.union(claim_id, prior)
    first_claim_by_text: dict[str, int] = {}
    for claim_id in claim_ids:
        normalized = claim_text_by_id[claim_id].strip().casefold()
        if not normalized:
            raise SelectorCalibrationError("claim text is empty after normalization")
        prior = first_claim_by_text.setdefault(normalized, claim_id)
        disjoint.union(claim_id, prior)
    claims_by_root: dict[int, list[int]] = defaultdict(list)
    for claim_id in claim_ids:
        claims_by_root[disjoint.find(claim_id)].append(claim_id)
    components: list[dict[str, Any]] = []
    for component_claims in claims_by_root.values():
        ordered_claims = sorted(component_claims)
        docs = sorted(
            {
                int(doc_id)
                for claim_id in ordered_claims
                for doc_id in retrieval_by_claim[claim_id]
            }
        )
        component_id = _sha256_bytes(_canonical_bytes(ordered_claims))[:16]
        components.append(
            {
                "component_id": component_id,
                "claim_ids": ordered_claims,
                "doc_ids": docs,
                "claim_count": len(ordered_claims),
                "normalized_claim_text_count": len(
                    {claim_text_by_id[value].strip().casefold() for value in ordered_claims}
                ),
            }
        )
    return sorted(components, key=lambda row: row["component_id"])


def _assign_component_folds(
    components: Sequence[dict[str, Any]], fold_count: int
) -> list[dict[str, Any]]:
    if fold_count < 2 or len(components) < fold_count:
        raise SelectorCalibrationError(
            "at least {} connected components are required".format(fold_count)
        )
    folds = [
        {"fold": index, "component_ids": [], "claim_ids": [], "doc_ids": []}
        for index in range(fold_count)
    ]
    ordered = sorted(
        components,
        key=lambda row: (-int(row["claim_count"]), str(row["component_id"])),
    )
    for component in ordered:
        target = min(folds, key=lambda row: (len(row["claim_ids"]), row["fold"]))
        target["component_ids"].append(component["component_id"])
        target["claim_ids"].extend(component["claim_ids"])
        target["doc_ids"].extend(component["doc_ids"])
    for fold in folds:
        fold["component_ids"].sort()
        fold["claim_ids"].sort()
        fold["doc_ids"] = sorted(set(fold["doc_ids"]))
        if not fold["claim_ids"]:
            raise SelectorCalibrationError("component fold is empty")
    return folds


def _validate_fold_isolation(
    folds: Sequence[dict[str, Any]], expected_claim_ids: set[int]
) -> dict[str, Any]:
    seen_claims: set[int] = set()
    claim_overlap = 0
    doc_overlap = 0
    for fold in folds:
        claims = set(fold["claim_ids"])
        docs = set(fold["doc_ids"])
        claim_overlap += len(seen_claims & claims)
        seen_claims.update(claims)
        for other in folds:
            if int(other["fold"]) >= int(fold["fold"]):
                continue
            doc_overlap += len(docs & set(other["doc_ids"]))
    return {
        "claim_coverage_count": len(seen_claims),
        "claim_coverage_complete": seen_claims == expected_claim_ids,
        "cross_fold_claim_overlap_count": claim_overlap,
        "cross_fold_doc_overlap_count": doc_overlap,
    }


def _claim_label(claim: Mapping[str, Any]) -> str:
    alternatives = claim.get("evidence")
    if not alternatives:
        return "NEI"
    labels = {
        rationale["label"]
        for rationale_sets in alternatives.values()
        for rationale in rationale_sets
    }
    if len(labels) != 1 or not labels.issubset({"SUPPORT", "CONTRADICT"}):
        raise SelectorCalibrationError("claim gold labels are inconsistent")
    return next(iter(labels))


def _fold_statistics(
    folds: Sequence[dict[str, Any]],
    claims_by_id: Mapping[int, dict[str, Any]],
    retrievable_relation_ids: set[int],
) -> list[dict[str, Any]]:
    statistics_rows: list[dict[str, Any]] = []
    for fold in folds:
        labels = Counter(_claim_label(claims_by_id[value]) for value in fold["claim_ids"])
        statistics_rows.append(
            {
                "fold": int(fold["fold"]),
                "claim_count": len(fold["claim_ids"]),
                "component_count": len(fold["component_ids"]),
                "top3_document_count": len(fold["doc_ids"]),
                "label_counts": {
                    label: int(labels[label])
                    for label in ("SUPPORT", "CONTRADICT", "NEI")
                },
                "retrievable_relation_claim_count": sum(
                    value in retrievable_relation_ids for value in fold["claim_ids"]
                ),
            }
        )
    return statistics_rows


def _tfidf_parameters(analyzer: str, ngram_range: tuple[int, int]) -> dict[str, Any]:
    return {
        "input": "content",
        "encoding": "utf-8",
        "decode_error": "strict",
        "strip_accents": None,
        "lowercase": True,
        "preprocessor": None,
        "tokenizer": None,
        "analyzer": analyzer,
        "stop_words": None,
        "token_pattern": r"(?u)\b\w\w+\b",
        "ngram_range": list(ngram_range),
        "max_df": 1.0,
        "min_df": 1,
        "max_features": None,
        "vocabulary": None,
        "binary": False,
        "dtype": "float64",
        "norm": "l2",
        "use_idf": True,
        "smooth_idf": True,
        "sublinear_tf": True,
    }


def _protocol() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "evaluation_mode": EVALUATION_MODE,
        "source_split": SOURCE_SPLIT,
        "dev_read_or_scored": False,
        "reportable_as_generalization": False,
        "fixed_train_contract": {
            "claims": SOURCE_CLAIM_COUNT,
            "relation_claims": RELATION_CLAIM_COUNT,
            "nei_claims": NEI_CLAIM_COUNT,
            "retrievable_relation_claims": RETRIEVABLE_RELATION_CLAIM_COUNT,
            "candidate_sentence_pairs": CANDIDATE_PAIR_COUNT,
            "positive_candidate_pairs": POSITIVE_PAIR_COUNT,
            "connected_components": CONNECTED_COMPONENT_COUNT,
            "largest_component_claims": LARGEST_COMPONENT_CLAIM_COUNT,
            "retrieval": "pinned_official_tfidf_top3_train",
        },
        "grouping": (
            "connected components induced by shared Top-3 documents or duplicate "
            "claim.strip().casefold() text"
        ),
        "cross_validation": {
            "outer_folds": OUTER_FOLDS,
            "inner_folds": INNER_FOLDS,
            "fit_isolation": "claim IDs and all Top-3 document IDs are disjoint",
        },
        "features": {
            "whitelist": list(FEATURE_NAMES),
            "feature_count": len(FEATURE_NAMES),
            "identifiers_or_labels_allowed": False,
            "nli_inference_passes": 1,
            "fold_fit_tfidf": {
                "word": _tfidf_parameters("word", (1, 2)),
                "char_wb": _tfidf_parameters("char_wb", (3, 5)),
                "fit_scope": (
                    "only training-fold claim texts and training-fold candidate "
                    "sentences; validation is transform-only"
                ),
                "empty_vocabulary_policy": "fail_closed",
            },
            "numeric_token_normalization_regex": NUMBER_PATTERN.pattern,
            "numeric_token_set_match": (
                "normalized numeric sets are exactly equal; both empty is a match"
            ),
            "negation_terms": sorted(NEGATION_TERMS),
            "negation_contraction_pattern": r"\b[a-z]+n['’]t\b",
        },
        "hybrid": {
            "scaler": "sample-weighted StandardScaler fit inside each training fold",
            "classifier": "L2 LogisticRegression(C=1, class_weight=None, liblinear)",
            "sample_weighting": {
                "claim_total": 1.0,
                "relation_with_both_classes": {
                    "positive_candidates_total": 0.5,
                    "negative_candidates_total": 0.5,
                },
                "relation_without_positive_or_nei": "all negative candidates total 1.0",
                "applied_to": ["StandardScaler.fit", "LogisticRegression.fit"],
            },
        },
        "baselines": {
            "lexical_only": LEXICAL_BASELINE_FORMULA,
            "nli_only": "max(entailment, contradiction)",
        },
        "inner_selection": {
            "thresholds": list(THRESHOLDS),
            "max_sentences": list(MAX_SENTENCES),
            "rule": (
                "prefer configurations with NEI correct-empty >=0.70; maximize "
                "custom relation best-rationale macro-F1, then complete-rationale "
                "recall, then fewer sentences and higher threshold"
            ),
        },
        "custom_metric": {
            "name": "relation_claim_best_rationale_set_macro_f1",
            "semantics": "gold alternatives outer-OR; sentences within one rationale inner-set",
        },
        "official_compatible_metric": {
            "name": "label_free_pooled_sentence_selection_f1",
            "semantics": (
                "sum per-claim relevant/retrieved/correct-selection sufficient counts "
                "then recompute pooled precision, recall, and F1"
            ),
            "external_official_evaluator_invoked": False,
        },
        "acceptance": {
            "official_f1_delta_vs_each_baseline": MIN_PRIMARY_DELTA,
            "official_cluster_bootstrap_95ci_lower_vs_each_baseline_strictly_above": 0.0,
            "noninferior_to_fold_max_baseline_outer_folds": MIN_NONINFERIOR_OUTER_FOLDS,
            "all_relation_complete_rationale_recall": MIN_ALL_RELATION_ANY_GOLD_RECALL,
            "retrievable_relation_complete_rationale_recall": MIN_RETRIEVABLE_ANY_GOLD_RECALL,
            "nei_correct_empty": MIN_NEI_CORRECT_EMPTY,
            "oof_coverage": 1.0,
            "cross_fold_claim_and_top3_doc_overlap": 0,
            "feature_whitelist_violations": 0,
        },
        "fail_closed_structural_checks": [
            "pinned corpus/train/retrieval/manifest SHA256",
            "pinned official source commit/archive and TF-IDF configuration",
            "pinned DeBERTa revision plus complete model identity SHA256",
            "exact fixed train counts and candidate key coverage",
            "exact outer/inner transformer fit claim IDs",
            "exact OOF per-strategy claim/relation coverage",
            "zero cross-fold claim and Top-3 document overlap",
            "unchanged implementation, preregistration, and inputs during run",
            "exact 10000-sample dual-metric dual-baseline bootstrap keys",
        ],
        "bootstrap_protocols": {
            "custom_best_rationale_macro": {
                "unit": "all_228_connected_components",
                "paired": True,
                "samples": BOOTSTRAP_SAMPLES,
                "seed": RANDOM_SEED,
                "comparisons": ["hybrid_vs_lexical_only", "hybrid_vs_nli_only"],
            },
            "official_compatible_pooled_sentence_selection": {
                "unit": "all_228_connected_components",
                "paired": True,
                "samples": BOOTSTRAP_SAMPLES,
                "seed": RANDOM_SEED,
                "aggregation": "component multiplicity -> sufficient counts -> pooled F1",
                "comparisons": ["hybrid_vs_lexical_only", "hybrid_vs_nli_only"],
            },
        },
        "final_refit": {
            "only_after_gate_pass": True,
            "threshold": "median of four outer-fold inner selections",
            "max_sentences": "mode of four outer-fold selections; ties choose smaller",
            "full_train_retuning": False,
            "final_tfidf_state": "full vocabulary and idf serialized only here",
        },
    }


def _new_vectorizer(analyzer: str, ngram_range: tuple[int, int]) -> Any:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer

    return TfidfVectorizer(
        input="content",
        encoding="utf-8",
        decode_error="strict",
        strip_accents=None,
        lowercase=True,
        preprocessor=None,
        tokenizer=None,
        analyzer=analyzer,
        stop_words=None,
        token_pattern=r"(?u)\b\w\w+\b",
        ngram_range=ngram_range,
        max_df=1.0,
        min_df=1,
        max_features=None,
        vocabulary=None,
        binary=False,
        dtype=np.float64,
        norm="l2",
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=True,
    )


def _serialize_vectorizer(vectorizer: Any, analyzer: str) -> dict[str, Any]:
    ngram_range = (1, 2) if analyzer == "word" else (3, 5)
    vocabulary = {
        term: int(index)
        for term, index in sorted(vectorizer.vocabulary_.items(), key=lambda item: item[0])
    }
    idf = [float(value) for value in vectorizer.idf_]
    state = {
        "parameters": _tfidf_parameters(analyzer, ngram_range),
        "vocabulary": vocabulary,
        "idf_by_index": idf,
    }
    return {
        **state,
        "state_sha256": _sha256_bytes(_canonical_bytes(state)),
        "vocabulary_size": len(vocabulary),
    }


def _fit_fold_transformer(
    frozen: Sequence[dict[str, Any]], train_pair_indices: Sequence[int]
) -> dict[str, Any]:
    if not train_pair_indices:
        raise SelectorCalibrationError("TF-IDF training fold has no candidates")
    train_claims: dict[int, str] = {}
    sentences: list[str] = []
    for pair_index in train_pair_indices:
        row = frozen[pair_index]
        train_claims[int(row["claim_id"])] = str(row["claim"])
        sentences.append(str(row["sentence"]))
    fit_texts = [train_claims[key] for key in sorted(train_claims)] + sentences
    word = _new_vectorizer("word", (1, 2))
    char = _new_vectorizer("char_wb", (3, 5))
    try:
        word.fit(fit_texts)
        char.fit(fit_texts)
    except ValueError as exc:
        raise SelectorCalibrationError("TF-IDF fold has an empty vocabulary") from exc
    return {
        "word": word,
        "char_wb": char,
        "fit_claim_ids": sorted(train_claims),
        "fit_pair_count": len(train_pair_indices),
    }


def _serialize_fold_transformer(transformer: Mapping[str, Any]) -> dict[str, Any]:
    state = {
        "fit_claim_ids": list(transformer["fit_claim_ids"]),
        "fit_pair_count": int(transformer["fit_pair_count"]),
        "word": _serialize_vectorizer(transformer["word"], "word"),
        "char_wb": _serialize_vectorizer(transformer["char_wb"], "char_wb"),
    }
    return {**state, "state_sha256": _sha256_bytes(_canonical_bytes(state))}


def _fold_transformer_binding(transformer: Mapping[str, Any]) -> dict[str, Any]:
    cached = transformer.get("audit_binding")
    if isinstance(cached, dict):
        return cached
    state = _serialize_fold_transformer(transformer)
    claim_ids = state["fit_claim_ids"]
    binding = {
        "state_sha256": state["state_sha256"],
        "rebuild_contract": "bound inputs + implementation + exact fit claim IDs",
        "fit_claim_ids": claim_ids,
        "fit_claim_ids_sha256": _sha256_bytes(_canonical_bytes(claim_ids)),
        "fit_claim_count": len(claim_ids),
        "fit_pair_count": state["fit_pair_count"],
        "word": {
            "parameters": state["word"]["parameters"],
            "state_sha256": state["word"]["state_sha256"],
            "vocabulary_size": state["word"]["vocabulary_size"],
        },
        "char_wb": {
            "parameters": state["char_wb"]["parameters"],
            "state_sha256": state["char_wb"]["state_sha256"],
            "vocabulary_size": state["char_wb"]["vocabulary_size"],
        },
    }
    if isinstance(transformer, dict):
        transformer["audit_binding"] = binding
    return binding


def _fold_feature_rows(
    frozen: Sequence[dict[str, Any]],
    pair_indices: Sequence[int],
    transformer: Mapping[str, Any],
) -> list[dict[str, float]]:
    import numpy as np

    claims = [str(frozen[index]["claim"]) for index in pair_indices]
    sentences = [str(frozen[index]["sentence"]) for index in pair_indices]
    word_claim = transformer["word"].transform(claims)
    word_sentence = transformer["word"].transform(sentences)
    char_claim = transformer["char_wb"].transform(claims)
    char_sentence = transformer["char_wb"].transform(sentences)
    word_cosines = np.asarray(word_claim.multiply(word_sentence).sum(axis=1)).ravel()
    char_cosines = np.asarray(char_claim.multiply(char_sentence).sum(axis=1)).ravel()
    rows: list[dict[str, float]] = []
    for offset, pair_index in enumerate(pair_indices):
        static = frozen[pair_index]["static_features"]
        if tuple(static) != STATIC_FEATURE_NAMES:
            raise SelectorCalibrationError("static feature whitelist violation")
        features = {
            **{name: float(static[name]) for name in STATIC_FEATURE_NAMES},
            "word_tfidf_cosine": float(word_cosines[offset]),
            "char_tfidf_cosine": float(char_cosines[offset]),
        }
        ordered = {name: features[name] for name in FEATURE_NAMES}
        if tuple(ordered) != FEATURE_NAMES or any(
            not math.isfinite(value) for value in ordered.values()
        ):
            raise SelectorCalibrationError("feature whitelist violation")
        rows.append(ordered)
    return rows


def _feature_matrix(feature_rows: Sequence[Mapping[str, float]]) -> list[list[float]]:
    return [[float(row[name]) for name in FEATURE_NAMES] for row in feature_rows]


def _claim_sample_weights(
    targets: Sequence[dict[str, Any]], pair_indices: Sequence[int]
) -> list[float]:
    by_claim: dict[int, list[int]] = defaultdict(list)
    for pair_index in pair_indices:
        by_claim[int(targets[pair_index]["claim_id"])].append(pair_index)
    weights_by_pair: dict[int, float] = {}
    for indices in by_claim.values():
        positive = [index for index in indices if targets[index]["is_gold_sentence"]]
        negative = [index for index in indices if not targets[index]["is_gold_sentence"]]
        if positive and negative:
            for index in positive:
                weights_by_pair[index] = 0.5 / len(positive)
            for index in negative:
                weights_by_pair[index] = 0.5 / len(negative)
        else:
            only_class = positive or negative
            for index in only_class:
                weights_by_pair[index] = 1.0 / len(only_class)
    for indices in by_claim.values():
        total = sum(weights_by_pair[index] for index in indices)
        if abs(total - 1.0) > 1e-12:
            raise SelectorCalibrationError("per-claim sample weights do not sum to one")
    return [weights_by_pair[index] for index in pair_indices]


def _fit_hybrid(
    frozen: Sequence[dict[str, Any]],
    targets: Sequence[dict[str, Any]],
    train_pair_indices: Sequence[int],
    transformer: Optional[Mapping[str, Any]] = None,
    include_vectorizer_state: bool = False,
) -> tuple[Any, Any, dict[str, Any], Mapping[str, Any]]:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    labels = [int(bool(targets[index]["is_gold_sentence"])) for index in train_pair_indices]
    if len(set(labels)) != 2:
        raise SelectorCalibrationError("hybrid training fold lacks both target classes")
    fold_transformer = (
        transformer
        if transformer is not None
        else _fit_fold_transformer(frozen, train_pair_indices)
    )
    feature_rows = _fold_feature_rows(frozen, train_pair_indices, fold_transformer)
    sample_weights = _claim_sample_weights(targets, train_pair_indices)
    scaler = StandardScaler()
    raw_matrix = _feature_matrix(feature_rows)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        scaler.fit(raw_matrix, sample_weight=sample_weights)
    unexpected = [
        item
        for item in caught
        if "invalid value encountered in sqrt" not in str(item.message)
    ]
    arrays = (scaler.mean_, scaler.var_, scaler.scale_)
    if unexpected or any(not np.all(np.isfinite(value)) for value in arrays):
        raise SelectorCalibrationError("non-finite weighted StandardScaler state")
    if np.any(scaler.var_ < -1e-12):
        raise SelectorCalibrationError("materially negative weighted feature variance")
    scaler.var_ = np.maximum(scaler.var_, 0.0)
    matrix = scaler.transform(raw_matrix)
    classifier = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="liblinear",
        class_weight=None,
        random_state=RANDOM_SEED,
        max_iter=2000,
    )
    classifier.fit(matrix, labels, sample_weight=sample_weights)
    if list(classifier.classes_) != [0, 1]:
        raise SelectorCalibrationError("unexpected logistic-regression class order")
    model = _serialize_model(scaler, classifier)
    model["sample_weighting"] = {
        "claim_total_weight": 1.0,
        "relation_with_both_classes": "positive total=0.5; negative total=0.5",
        "single_class_claim": "the present class totals 1.0",
        "applied_to": ["StandardScaler.fit", "LogisticRegression.fit"],
    }
    if include_vectorizer_state:
        model["tfidf"] = _serialize_fold_transformer(fold_transformer)
    else:
        model["tfidf"] = _fold_transformer_binding(fold_transformer)
    return scaler, classifier, model, fold_transformer


def _serialize_model(scaler: Any, classifier: Any) -> dict[str, Any]:
    values = [
        *[float(value) for value in scaler.mean_],
        *[float(value) for value in scaler.scale_],
        *[float(value) for value in classifier.coef_[0]],
        float(classifier.intercept_[0]),
    ]
    if any(not math.isfinite(value) for value in values):
        raise SelectorCalibrationError("non-finite fitted coefficient")
    return {
        "feature_names": list(FEATURE_NAMES),
        "standard_scaler": {
            "mean": [float(value) for value in scaler.mean_],
            "variance": [float(value) for value in scaler.var_],
            "scale": [float(value) for value in scaler.scale_],
            "sample_weighted": True,
        },
        "logistic_regression": {
            "penalty": "l2",
            "C": 1.0,
            "solver": "liblinear",
            "class_weight": None,
            "random_state": RANDOM_SEED,
            "classes": [int(value) for value in classifier.classes_],
            "coefficients": [float(value) for value in classifier.coef_[0]],
            "intercept": float(classifier.intercept_[0]),
            "iterations": [int(value) for value in classifier.n_iter_],
        },
    }


def _score_pairs(
    strategy: str,
    frozen: Sequence[dict[str, Any]],
    pair_indices: Sequence[int],
    transformer: Optional[Mapping[str, Any]] = None,
    fitted: Optional[tuple[Any, Any]] = None,
) -> dict[int, float]:
    if strategy not in STRATEGIES:
        raise SelectorCalibrationError("unknown selector strategy")
    if strategy == "hybrid":
        if fitted is None:
            raise SelectorCalibrationError("hybrid strategy requires a fitted model")
        if transformer is None:
            raise SelectorCalibrationError("hybrid strategy requires fold TF-IDF")
        scaler, classifier = fitted
        feature_rows = _fold_feature_rows(frozen, pair_indices, transformer)
        probabilities = classifier.predict_proba(
            scaler.transform(_feature_matrix(feature_rows))
        )[:, 1]
        return {
            pair_index: float(value)
            for pair_index, value in zip(pair_indices, probabilities)
        }
    result: dict[int, float] = {}
    lexical_rows: Optional[list[dict[str, float]]] = None
    if strategy == "lexical_only":
        if transformer is None:
            raise SelectorCalibrationError("lexical baseline requires fold TF-IDF")
        lexical_rows = _fold_feature_rows(frozen, pair_indices, transformer)
    for offset, pair_index in enumerate(pair_indices):
        features = frozen[pair_index]["static_features"]
        if strategy == "lexical_only":
            assert lexical_rows is not None
            fold_features = lexical_rows[offset]
            score = (
                0.30 * fold_features["word_tfidf_cosine"]
                + 0.20 * fold_features["char_tfidf_cosine"]
                + 0.15 * float(features["word_jaccard"])
                + 0.10 * float(features["claim_token_coverage"])
                + 0.05 * float(features["sentence_token_coverage"])
                + 0.05 * float(features["numeric_token_set_match"])
                + 0.05 * float(features["negation_presence_match"])
                + 0.04 * float(features["length_ratio"])
                + 0.04 * float(features["reciprocal_doc_rank"])
                + 0.02 * (1.0 - float(features["normalized_sentence_position"]))
            )
        else:
            score = float(features["nli_relation_probability"])
        result[pair_index] = score
    return result


def _select_pairs(
    claim_ids: Sequence[int],
    by_claim: Mapping[int, Sequence[int]],
    scores: Mapping[int, float],
    frozen: Sequence[dict[str, Any]],
    *,
    threshold: float,
    max_sentences: int,
) -> dict[int, list[int]]:
    selected: dict[int, list[int]] = {}
    for claim_id in claim_ids:
        eligible = [
            pair_index
            for pair_index in by_claim[claim_id]
            if pair_index in scores and scores[pair_index] >= threshold
        ]
        eligible.sort(
            key=lambda pair_index: (
                -scores[pair_index],
                int(frozen[pair_index]["doc_rank"]),
                int(frozen[pair_index]["sentence_index"]),
                int(frozen[pair_index]["doc_id"]),
                pair_index,
            )
        )
        selected[claim_id] = eligible[:max_sentences]
    return selected


def _pair_f1(predicted: set[tuple[int, int]], gold: set[tuple[int, int]]) -> float:
    if not predicted or not gold:
        return 0.0
    correct = len(predicted & gold)
    precision = correct / len(predicted)
    recall = correct / len(gold)
    return 2.0 * precision * recall / (precision + recall) if correct else 0.0


def _selector_metrics(
    claim_ids: Sequence[int],
    claims_by_id: Mapping[int, dict[str, Any]],
    frozen: Sequence[dict[str, Any]],
    selected: Mapping[int, Sequence[int]],
    retrievable_relation_ids: set[int],
) -> dict[str, Any]:
    relation_scores: list[float] = []
    all_relation_complete = 0
    retrievable_complete = 0
    all_relation_any_gold = 0
    nei_empty = 0
    relation_ids: list[int] = []
    nei_ids: list[int] = []
    pooled_counts = Counter()
    gold_document_count = 0
    complete_gold_document_count = 0
    per_claim: dict[str, dict[str, Any]] = {}
    for claim_id in claim_ids:
        alternatives = _gold_alternatives(claims_by_id[claim_id])
        predicted_pairs = {
            (int(frozen[index]["doc_id"]), int(frozen[index]["sentence_index"]))
            for index in selected[claim_id]
        }
        if alternatives:
            relation_ids.append(claim_id)
            score = max(
                _pair_f1(predicted_pairs, set(alternative))
                for alternative in alternatives
            )
            gold_union = set().union(*alternatives)
            hit = bool(predicted_pairs & gold_union)
            complete = any(set(alternative).issubset(predicted_pairs) for alternative in alternatives)
            gold_doc_ids = {doc_id for alternative in alternatives for doc_id, _ in alternative}
            complete_gold_docs = 0
            for doc_id in gold_doc_ids:
                predicted_in_doc = {
                    sentence_index
                    for predicted_doc_id, sentence_index in predicted_pairs
                    if predicted_doc_id == doc_id
                }
                rationale_sets = [
                    {sentence_index for candidate_doc_id, sentence_index in alternative}
                    for alternative in alternatives
                    if {candidate_doc_id for candidate_doc_id, _ in alternative} == {doc_id}
                ]
                complete_gold_docs += int(
                    any(rationale.issubset(predicted_in_doc) for rationale in rationale_sets)
                )
            gold_document_count += len(gold_doc_ids)
            complete_gold_document_count += complete_gold_docs
            relation_scores.append(score)
            all_relation_any_gold += int(hit)
            all_relation_complete += int(complete)
            retrievable_complete += int(
                complete and claim_id in retrievable_relation_ids
            )
            relevant = sum(len(alternative) for alternative in alternatives)
            correct = 0
            for doc_id in gold_doc_ids:
                predicted_in_doc = {
                    sentence_index
                    for predicted_doc_id, sentence_index in predicted_pairs
                    if predicted_doc_id == doc_id
                }
                gold_in_doc = [
                    {sentence_index for _, sentence_index in alternative}
                    for alternative in alternatives
                    if {candidate_doc_id for candidate_doc_id, _ in alternative} == {doc_id}
                ]
                for sentence_index in predicted_in_doc:
                    containing = [value for value in gold_in_doc if sentence_index in value]
                    if len(containing) > 1:
                        raise SelectorCalibrationError(
                            "one sentence occurs in multiple rationale alternatives"
                        )
                    if containing and containing[0].issubset(predicted_in_doc):
                        correct += 1
            sufficient = {
                "relevant": relevant,
                "retrieved": len(predicted_pairs),
                "correct_selection": correct,
            }
            pooled_counts.update(sufficient)
            per_claim[str(claim_id)] = {
                "is_relation": True,
                "best_rationale_set_f1": score,
                "any_gold_hit": hit,
                "contains_complete_gold_rationale": complete,
                "retrievable": claim_id in retrievable_relation_ids,
                "gold_document_count": len(gold_doc_ids),
                "complete_gold_document_count": complete_gold_docs,
                "selected_count": len(predicted_pairs),
                "official_sentence_selection_sufficient_counts": sufficient,
            }
        else:
            nei_ids.append(claim_id)
            empty = not predicted_pairs
            nei_empty += int(empty)
            sufficient = {
                "relevant": 0,
                "retrieved": len(predicted_pairs),
                "correct_selection": 0,
            }
            pooled_counts.update(sufficient)
            per_claim[str(claim_id)] = {
                "is_relation": False,
                "correct_empty": empty,
                "selected_count": len(predicted_pairs),
                "gold_document_count": 0,
                "complete_gold_document_count": 0,
                "official_sentence_selection_sufficient_counts": sufficient,
            }
    precision = _safe_ratio(
        pooled_counts["correct_selection"], pooled_counts["retrieved"]
    )
    recall = _safe_ratio(
        pooled_counts["correct_selection"], pooled_counts["relevant"]
    )
    pooled_f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "custom_best_rationale_macro_f1": {
            "name": "relation_claim_best_rationale_set_macro_f1",
            "semantics": (
                "for each relation claim, compute pair-set F1 against every gold "
                "rationale alternative and retain the best; alternatives are outer-OR, "
                "sentences inside one rationale are an inner set"
            ),
            "claim_count": len(relation_ids),
            "value": sum(relation_scores) / len(relation_scores)
            if relation_scores
            else 0.0,
        },
        "official_compatible_pooled_sentence_selection": {
            "external_official_evaluator_invoked": False,
            "label_free": True,
            "sufficient_counts": {
                "relevant": int(pooled_counts["relevant"]),
                "retrieved": int(pooled_counts["retrieved"]),
                "correct_selection": int(pooled_counts["correct_selection"]),
            },
            "precision": precision,
            "recall": recall,
            "f1": pooled_f1,
        },
        "gold_document_complete_rationale": {
            "gold_document_count": gold_document_count,
            "complete_gold_document_count": complete_gold_document_count,
            "recall": _safe_ratio(
                complete_gold_document_count, gold_document_count
            ),
        },
        "all_relation_complete_rationale_recall": _safe_ratio(
            all_relation_complete, len(relation_ids)
        ),
        "retrievable_relation_complete_rationale_recall": _safe_ratio(
            retrievable_complete,
            sum(claim_id in retrievable_relation_ids for claim_id in relation_ids),
        ),
        "all_relation_any_gold_recall_diagnostic": _safe_ratio(
            all_relation_any_gold, len(relation_ids)
        ),
        "nei_correct_empty_rate": _safe_ratio(nei_empty, len(nei_ids)),
        "relation_claim_count": len(relation_ids),
        "retrievable_relation_claim_count": sum(
            claim_id in retrievable_relation_ids for claim_id in relation_ids
        ),
        "nei_claim_count": len(nei_ids),
        "selected_sentence_count": sum(len(selected[claim_id]) for claim_id in claim_ids),
        "per_claim": per_claim,
    }


def _configuration_key(metrics: dict[str, Any], threshold: float, maximum: int) -> tuple[Any, ...]:
    nei_rate = float(metrics["nei_correct_empty_rate"])
    primary = float(metrics["custom_best_rationale_macro_f1"]["value"])
    eligible = nei_rate >= MIN_NEI_CORRECT_EMPTY
    return (
        int(eligible),
        primary if eligible else nei_rate,
        primary,
        float(metrics["all_relation_complete_rationale_recall"]),
        -maximum,
        threshold,
    )


def _tune_selection(
    *,
    strategy: str,
    inner_folds: Sequence[dict[str, Any]],
    outer_train_claim_ids: Sequence[int],
    frozen: Sequence[dict[str, Any]],
    targets: Sequence[dict[str, Any]],
    by_claim: Mapping[int, Sequence[int]],
    claims_by_id: Mapping[int, dict[str, Any]],
    retrievable_relation_ids: set[int],
    transformer_cache: Optional[dict[tuple[int, ...], Mapping[str, Any]]] = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scores_by_claim: dict[int, dict[int, float]] = {}
    fold_models: list[dict[str, Any]] = []
    outer_set = set(outer_train_claim_ids)
    cache = transformer_cache if transformer_cache is not None else {}
    for inner in inner_folds:
        validation_ids = list(inner["claim_ids"])
        train_ids = sorted(outer_set - set(validation_ids))
        train_pairs = [index for claim_id in train_ids for index in by_claim[claim_id]]
        validation_pairs = [
            index for claim_id in validation_ids for index in by_claim[claim_id]
        ]
        fitted: Optional[tuple[Any, Any]] = None
        model: Optional[dict[str, Any]] = None
        transformer: Optional[Mapping[str, Any]] = None
        if strategy in {"hybrid", "lexical_only"}:
            cache_key = tuple(train_pairs)
            transformer = cache.get(cache_key)
            if transformer is None:
                transformer = _fit_fold_transformer(frozen, train_pairs)
                cache[cache_key] = transformer
            if list(transformer["fit_claim_ids"]) != train_ids:
                raise SelectorCalibrationError(
                    "inner transformer fit claim IDs differ from expected train IDs"
                )
        if strategy == "hybrid":
            scaler, classifier, model, transformer = _fit_hybrid(
                frozen,
                targets,
                train_pairs,
                transformer=transformer,
                include_vectorizer_state=False,
            )
            fitted = (scaler, classifier)
        elif strategy == "lexical_only":
            assert transformer is not None
            model = {"tfidf": _fold_transformer_binding(transformer)}
        scores = _score_pairs(
            strategy,
            frozen,
            validation_pairs,
            transformer=transformer,
            fitted=fitted,
        )
        for claim_id in validation_ids:
            scores_by_claim[claim_id] = {
                index: scores[index] for index in by_claim[claim_id]
            }
        fold_models.append(
            {
                "inner_fold": int(inner["fold"]),
                "train_claim_count": len(train_ids),
                "validation_claim_count": len(validation_ids),
                "model": model,
            }
        )
    if set(scores_by_claim) != outer_set:
        raise SelectorCalibrationError("inner OOF score coverage is incomplete")
    flat_scores = {
        pair_index: score
        for claim_scores in scores_by_claim.values()
        for pair_index, score in claim_scores.items()
    }
    sweep: list[dict[str, Any]] = []
    best: Optional[dict[str, Any]] = None
    for threshold in THRESHOLDS:
        for maximum in MAX_SENTENCES:
            selected = _select_pairs(
                outer_train_claim_ids,
                by_claim,
                flat_scores,
                frozen,
                threshold=threshold,
                max_sentences=maximum,
            )
            metrics = _selector_metrics(
                outer_train_claim_ids,
                claims_by_id,
                frozen,
                selected,
                retrievable_relation_ids,
            )
            row = {
                "threshold": threshold,
                "max_sentences": maximum,
                "metrics": {key: value for key, value in metrics.items() if key != "per_claim"},
            }
            sweep.append(row)
            if best is None or _configuration_key(metrics, threshold, maximum) > _configuration_key(
                best["_metrics"], best["threshold"], best["max_sentences"]
            ):
                best = {**row, "_metrics": metrics}
    if best is None:
        raise SelectorCalibrationError("selection sweep is empty")
    best.pop("_metrics")
    return {"selected": best, "sweep": sweep, "inner_models": fold_models}, sweep


def _run_nested_oof(
    *,
    claims: Sequence[dict[str, Any]],
    frozen: Sequence[dict[str, Any]],
    targets: Sequence[dict[str, Any]],
    by_claim: Mapping[int, Sequence[int]],
    components: Sequence[dict[str, Any]],
    outer_folds: Sequence[dict[str, Any]],
    retrievable_relation_ids: set[int],
) -> tuple[
    dict[str, Any],
    dict[str, dict[int, list[int]]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    claims_by_id = {int(row["id"]): row for row in claims}
    all_claim_ids = sorted(claims_by_id)
    component_by_id = {row["component_id"]: row for row in components}
    selected_by_strategy = {strategy: {} for strategy in STRATEGIES}
    audit_folds: list[dict[str, Any]] = []
    for outer in outer_folds:
        transformer_cache: dict[tuple[int, ...], Mapping[str, Any]] = {}
        validation_ids = list(outer["claim_ids"])
        outer_train_ids = sorted(set(all_claim_ids) - set(validation_ids))
        train_component_ids = sorted(
            set(component_by_id) - set(outer["component_ids"])
        )
        inner_components = [component_by_id[value] for value in train_component_ids]
        inner_folds = _assign_component_folds(inner_components, INNER_FOLDS)
        inner_isolation = _validate_fold_isolation(inner_folds, set(outer_train_ids))
        if (
            not inner_isolation["claim_coverage_complete"]
            or inner_isolation["cross_fold_claim_overlap_count"]
            or inner_isolation["cross_fold_doc_overlap_count"]
        ):
            raise SelectorCalibrationError("inner component isolation failed")
        train_pairs = [index for claim_id in outer_train_ids for index in by_claim[claim_id]]
        validation_pairs = [index for claim_id in validation_ids for index in by_claim[claim_id]]
        outer_audit = {
            "outer_fold": int(outer["fold"]),
            "train_claim_count": len(outer_train_ids),
            "validation_claim_count": len(validation_ids),
            "inner_folds": inner_folds,
            "inner_isolation": inner_isolation,
            "strategies": {},
        }
        for strategy in STRATEGIES:
            tuning, _ = _tune_selection(
                strategy=strategy,
                inner_folds=inner_folds,
                outer_train_claim_ids=outer_train_ids,
                frozen=frozen,
                targets=targets,
                by_claim=by_claim,
                claims_by_id=claims_by_id,
                retrievable_relation_ids=retrievable_relation_ids,
                transformer_cache=transformer_cache,
            )
            fitted: Optional[tuple[Any, Any]] = None
            outer_model: Optional[dict[str, Any]] = None
            transformer: Optional[Mapping[str, Any]] = None
            if strategy in {"hybrid", "lexical_only"}:
                cache_key = tuple(train_pairs)
                transformer = transformer_cache.get(cache_key)
                if transformer is None:
                    transformer = _fit_fold_transformer(frozen, train_pairs)
                    transformer_cache[cache_key] = transformer
                if list(transformer["fit_claim_ids"]) != outer_train_ids:
                    raise SelectorCalibrationError(
                        "outer transformer fit claim IDs differ from expected train IDs"
                    )
            if strategy == "hybrid":
                scaler, classifier, outer_model, transformer = _fit_hybrid(
                    frozen,
                    targets,
                    train_pairs,
                    transformer=transformer,
                    include_vectorizer_state=False,
                )
                fitted = (scaler, classifier)
            elif strategy == "lexical_only":
                assert transformer is not None
                outer_model = {"tfidf": _fold_transformer_binding(transformer)}
            scores = _score_pairs(
                strategy,
                frozen,
                validation_pairs,
                transformer=transformer,
                fitted=fitted,
            )
            chosen = tuning["selected"]
            selected = _select_pairs(
                validation_ids,
                by_claim,
                scores,
                frozen,
                threshold=float(chosen["threshold"]),
                max_sentences=int(chosen["max_sentences"]),
            )
            selected_by_strategy[strategy].update(selected)
            fold_metrics = _selector_metrics(
                validation_ids,
                claims_by_id,
                frozen,
                selected,
                retrievable_relation_ids,
            )
            outer_audit["strategies"][strategy] = {
                "inner_tuning": tuning,
                "outer_model": outer_model,
                "outer_validation_metrics": {
                    key: value for key, value in fold_metrics.items() if key != "per_claim"
                },
            }
        audit_folds.append(outer_audit)
        transformer_cache.clear()
    metrics: dict[str, Any] = {}
    cases: list[dict[str, Any]] = []
    for strategy in STRATEGIES:
        if set(selected_by_strategy[strategy]) != set(all_claim_ids):
            raise SelectorCalibrationError("outer OOF coverage is incomplete")
        result = _selector_metrics(
            all_claim_ids,
            claims_by_id,
            frozen,
            selected_by_strategy[strategy],
            retrievable_relation_ids,
        )
        metrics[strategy] = {key: value for key, value in result.items() if key != "per_claim"}
        strategy_rows = [row for row in cases if row["strategy"] == strategy]
        if strategy_rows:
            raise SelectorCalibrationError("strategy cases were populated out of order")
        for claim_id in all_claim_ids:
            cases.append(
                {
                    "strategy": strategy,
                    "claim_id": claim_id,
                    "selected_pair_indices": selected_by_strategy[strategy][claim_id],
                    **result["per_claim"][str(claim_id)],
                }
            )
        strategy_rows = [row for row in cases if row["strategy"] == strategy]
        if (
            len(strategy_rows) != len(all_claim_ids)
            or len({int(row["claim_id"]) for row in strategy_rows})
            != len(all_claim_ids)
            or sum(bool(row["is_relation"]) for row in strategy_rows)
            != RELATION_CLAIM_COUNT
        ):
            raise SelectorCalibrationError("per-strategy OOF case coverage is invalid")
    return metrics, selected_by_strategy, audit_folds, cases


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise SelectorCalibrationError("cannot take quantile of empty values")
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _metric_from_case_rows(
    rows: Sequence[dict[str, Any]], metric: str
) -> tuple[float, bool]:
    if metric == "custom_best_rationale_macro":
        values = [
            float(row["best_rationale_set_f1"])
            for row in rows
            if row["is_relation"]
        ]
        return (_safe_ratio(sum(values), len(values)), not values)
    if metric != "official_compatible_pooled_sentence_selection":
        raise SelectorCalibrationError("unknown bootstrap metric")
    counts = Counter()
    for row in rows:
        counts.update(row["official_sentence_selection_sufficient_counts"])
    precision = _safe_ratio(counts["correct_selection"], counts["retrieved"])
    recall = _safe_ratio(counts["correct_selection"], counts["relevant"])
    value = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return value, counts["relevant"] == 0


def _paired_component_bootstrap(
    *,
    components: Sequence[dict[str, Any]],
    hybrid_cases: Mapping[int, dict[str, Any]],
    baseline_cases: Mapping[int, dict[str, Any]],
    metric: str,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    if samples != BOOTSTRAP_SAMPLES or len(components) < 2:
        raise SelectorCalibrationError(
            "paired component bootstrap requires exactly 10000 samples"
        )
    all_claim_ids = {
        int(claim_id) for component in components for claim_id in component["claim_ids"]
    }
    if set(hybrid_cases) != all_claim_ids or set(baseline_cases) != all_claim_ids:
        raise SelectorCalibrationError("paired bootstrap claim coverage mismatch")
    component_claims = [list(row["claim_ids"]) for row in components]
    rng = random.Random(seed)
    deltas: list[float] = []
    zero_denominator_resamples = 0
    for _ in range(samples):
        sampled = [rng.choice(component_claims) for _ in component_claims]
        claim_ids = [claim_id for component in sampled for claim_id in component]
        hybrid, hybrid_zero = _metric_from_case_rows(
            [hybrid_cases[claim_id] for claim_id in claim_ids], metric
        )
        baseline, baseline_zero = _metric_from_case_rows(
            [baseline_cases[claim_id] for claim_id in claim_ids], metric
        )
        zero_denominator_resamples += int(hybrid_zero or baseline_zero)
        deltas.append(hybrid - baseline)
    hybrid_point, _ = _metric_from_case_rows(list(hybrid_cases.values()), metric)
    baseline_point, _ = _metric_from_case_rows(list(baseline_cases.values()), metric)
    return {
        "unit": "all_shared_top3_document_or_duplicate_claim_connected_components",
        "all_components_in_sampling_frame": True,
        "samples": samples,
        "seed": seed,
        "component_count": len(component_claims),
        "aggregation": (
            "resample components with replacement; apply component multiplicity to "
            "claim sufficient statistics; recompute metric"
        ),
        "point_hybrid": hybrid_point,
        "point_baseline": baseline_point,
        "point_delta": hybrid_point - baseline_point,
        "zero_denominator_resamples": zero_denominator_resamples,
        "confidence_interval": {
            "method": "paired_percentile_connected_component_cluster_bootstrap",
            "level": 0.95,
            "lower": _quantile(deltas, 0.025),
            "upper": _quantile(deltas, 0.975),
        },
    }


def _bootstrap_comparisons(
    *,
    components: Sequence[dict[str, Any]],
    cases: Sequence[dict[str, Any]],
    samples: int = BOOTSTRAP_SAMPLES,
) -> dict[str, Any]:
    expected_case_count = SOURCE_CLAIM_COUNT * len(STRATEGIES)
    if len(cases) != expected_case_count:
        raise SelectorCalibrationError("bootstrap case count mismatch")
    by_strategy = {
        strategy: {
            int(row["claim_id"]): row
            for row in cases
            if row["strategy"] == strategy
        }
        for strategy in STRATEGIES
    }
    metrics = (
        "custom_best_rationale_macro",
        "official_compatible_pooled_sentence_selection",
    )
    baselines = ("lexical_only", "nli_only")
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "samples": samples,
        "seed": RANDOM_SEED,
        "component_count": len(components),
        "metrics": {},
    }
    for metric in metrics:
        result["metrics"][metric] = {}
        for baseline in baselines:
            key = "hybrid_vs_{}".format(baseline)
            result["metrics"][metric][key] = _paired_component_bootstrap(
                components=components,
                hybrid_cases=by_strategy["hybrid"],
                baseline_cases=by_strategy[baseline],
                metric=metric,
                samples=samples,
                seed=RANDOM_SEED,
            )
    expected_keys = {"hybrid_vs_lexical_only", "hybrid_vs_nli_only"}
    if set(result["metrics"]) != set(metrics) or any(
        set(result["metrics"][metric]) != expected_keys for metric in metrics
    ):
        raise SelectorCalibrationError("bootstrap comparison keys are incomplete")
    return result


def _official_style_metrics(
    claims: Sequence[dict[str, Any]],
    frozen: Sequence[dict[str, Any]],
    selected: Mapping[int, Sequence[int]],
) -> dict[str, dict[str, float]]:
    abstract = Counter()
    sentence = Counter()
    for claim in claims:
        claim_id = int(claim["id"])
        gold = claim["evidence"]
        abstract["relevant"] += len(gold)
        sentence["relevant"] += sum(
            len(rationale["sentences"])
            for rationale_sets in gold.values()
            for rationale in rationale_sets
        )
        selected_by_doc: dict[int, list[int]] = defaultdict(list)
        for pair_index in selected[claim_id]:
            selected_by_doc[int(frozen[pair_index]["doc_id"])].append(pair_index)
        for doc_id, pair_indices in selected_by_doc.items():
            abstract["retrieved"] += 1
            sentence["retrieved"] += len(pair_indices)
            rationale_sets = gold.get(str(doc_id), gold.get(doc_id))
            if not rationale_sets:
                continue
            labels = {rationale["label"] for rationale in rationale_sets}
            if len(labels) != 1:
                raise SelectorCalibrationError("gold document labels conflict")
            entailment = max(
                float(frozen[index]["static_features"]["nli_entailment"])
                for index in pair_indices
            )
            contradiction = max(
                float(frozen[index]["static_features"]["nli_contradiction"])
                for index in pair_indices
            )
            predicted_label = "SUPPORT" if entailment >= contradiction else "CONTRADICT"
            label_correct = predicted_label == next(iter(labels))
            predicted_sentence_list = [
                int(frozen[index]["sentence_index"]) for index in pair_indices
            ]
            predicted_sentences = set(predicted_sentence_list)
            gold_sets = [set(rationale["sentences"]) for rationale in rationale_sets]
            shortest = min(len(gold_set) for gold_set in gold_sets)
            capped = set(predicted_sentence_list[: max(3, shortest)])
            abstract["correct_label_only"] += int(label_correct)
            abstract["correct_rationalized"] += int(
                label_correct and any(gold_set.issubset(capped) for gold_set in gold_sets)
            )
            correct = 0
            for sentence_index in predicted_sentences:
                containing = [value for value in gold_sets if sentence_index in value]
                if len(containing) > 1:
                    raise SelectorCalibrationError(
                        "one sentence occurs in multiple rationale alternatives"
                    )
                if containing and containing[0].issubset(predicted_sentences):
                    correct += 1
            sentence["correct_selection"] += correct
            sentence["correct_label"] += int(label_correct) * correct

    def score(counts: Counter[str], key: str) -> dict[str, float]:
        precision = _safe_ratio(counts[key], counts["retrieved"])
        recall = _safe_ratio(counts[key], counts["relevant"])
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {"precision": precision, "recall": recall, "f1": f1}

    return {
        "sentence_selection": score(sentence, "correct_selection"),
        "sentence_label": score(sentence, "correct_label"),
        "abstract_label_only": score(abstract, "correct_label_only"),
        "abstract_rationalized": score(abstract, "correct_rationalized"),
    }


def _gate(
    *,
    metrics: Mapping[str, dict[str, Any]],
    audit_folds: Sequence[dict[str, Any]],
    cases: Sequence[dict[str, Any]],
    bootstrap: dict[str, Any],
    isolation: dict[str, Any],
    feature_whitelist_violations: int,
) -> dict[str, Any]:
    hybrid = metrics["hybrid"]
    baselines = ("lexical_only", "nli_only")
    official_deltas = {
        baseline: float(
            hybrid["official_compatible_pooled_sentence_selection"]["f1"]
        )
        - float(metrics[baseline]["official_compatible_pooled_sentence_selection"]["f1"])
        for baseline in baselines
    }
    custom_deltas = {
        baseline: float(hybrid["custom_best_rationale_macro_f1"]["value"])
        - float(metrics[baseline]["custom_best_rationale_macro_f1"]["value"])
        for baseline in baselines
    }
    noninferior_folds = sum(
        float(
            fold["strategies"]["hybrid"]["outer_validation_metrics"][
                "official_compatible_pooled_sentence_selection"
            ]["f1"]
        )
        >= max(
            float(
                fold["strategies"][baseline]["outer_validation_metrics"][
                    "official_compatible_pooled_sentence_selection"
                ]["f1"]
            )
            for baseline in baselines
        )
        for fold in audit_folds
    )
    coverage_by_strategy = {}
    for strategy in STRATEGIES:
        rows = [row for row in cases if row["strategy"] == strategy]
        coverage_by_strategy[strategy] = {
            "rows": len(rows),
            "unique_claims": len({int(row["claim_id"]) for row in rows}),
            "relation_claims": sum(bool(row["is_relation"]) for row in rows),
        }
    official_bootstrap = bootstrap["metrics"][
        "official_compatible_pooled_sentence_selection"
    ]
    custom_bootstrap = bootstrap["metrics"]["custom_best_rationale_macro"]
    checks = {
        "official_f1_delta_vs_lexical_at_least_0_020": official_deltas[
            "lexical_only"
        ]
        >= MIN_PRIMARY_DELTA,
        "official_f1_delta_vs_nli_at_least_0_020": official_deltas["nli_only"]
        >= MIN_PRIMARY_DELTA,
        "official_cluster_ci_lower_vs_lexical_above_zero": float(
            official_bootstrap["hybrid_vs_lexical_only"]["confidence_interval"][
                "lower"
            ]
        )
        > 0.0,
        "official_cluster_ci_lower_vs_nli_above_zero": float(
            official_bootstrap["hybrid_vs_nli_only"]["confidence_interval"]["lower"]
        )
        > 0.0,
        "at_least_3_of_4_outer_folds_noninferior_to_fold_max_baseline": noninferior_folds
        >= MIN_NONINFERIOR_OUTER_FOLDS,
        "all_505_relation_complete_rationale_recall_at_least_0_50": (
            hybrid["relation_claim_count"] == RELATION_CLAIM_COUNT
            and float(hybrid["all_relation_complete_rationale_recall"])
            >= MIN_ALL_RELATION_ANY_GOLD_RECALL
        ),
        "all_396_retrievable_complete_rationale_recall_at_least_0_65": (
            hybrid["retrievable_relation_claim_count"]
            == RETRIEVABLE_RELATION_CLAIM_COUNT
            and float(hybrid["retrievable_relation_complete_rationale_recall"])
            >= MIN_RETRIEVABLE_ANY_GOLD_RECALL
        ),
        "all_304_nei_correct_empty_at_least_0_70": (
            hybrid["nei_claim_count"] == NEI_CLAIM_COUNT
            and float(hybrid["nei_correct_empty_rate"]) >= MIN_NEI_CORRECT_EMPTY
        ),
        "oof_claim_coverage_100_percent": (
            all(
                values
                == {
                    "rows": SOURCE_CLAIM_COUNT,
                    "unique_claims": SOURCE_CLAIM_COUNT,
                    "relation_claims": RELATION_CLAIM_COUNT,
                }
                for values in coverage_by_strategy.values()
            )
            and isolation["claim_coverage_complete"]
        ),
        "cross_fold_claim_overlap_zero": isolation[
            "cross_fold_claim_overlap_count"
        ]
        == 0,
        "cross_fold_top3_doc_overlap_zero": isolation["cross_fold_doc_overlap_count"]
        == 0,
        "feature_whitelist_violations_zero": feature_whitelist_violations == 0,
        "bootstrap_protocol_and_keys_exact": (
            bootstrap["samples"] == BOOTSTRAP_SAMPLES
            and bootstrap["component_count"] == CONNECTED_COMPONENT_COUNT
            and set(bootstrap["metrics"])
            == {
                "custom_best_rationale_macro",
                "official_compatible_pooled_sentence_selection",
            }
            and set(official_bootstrap)
            == {"hybrid_vs_lexical_only", "hybrid_vs_nli_only"}
            and set(custom_bootstrap)
            == {"hybrid_vs_lexical_only", "hybrid_vs_nli_only"}
        ),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "pass" if all(checks.values()) else "reject",
        "full_train_model_permitted": all(checks.values()),
        "official_f1_deltas": official_deltas,
        "custom_best_rationale_macro_f1_deltas": custom_deltas,
        "custom_bootstrap_reported_not_gated": custom_bootstrap,
        "coverage_by_strategy": coverage_by_strategy,
        "noninferior_outer_fold_count": noninferior_folds,
        "checks": checks,
        "failure_policy": "any failed check forbids full-train fitting and model emission",
    }


def _full_train_model(
    *,
    claims: Sequence[dict[str, Any]],
    frozen: Sequence[dict[str, Any]],
    targets: Sequence[dict[str, Any]],
    by_claim: Mapping[int, Sequence[int]],
    audit_folds: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    thresholds = [
        float(fold["strategies"]["hybrid"]["inner_tuning"]["selected"]["threshold"])
        for fold in audit_folds
    ]
    maxima = [
        int(
            fold["strategies"]["hybrid"]["inner_tuning"]["selected"][
                "max_sentences"
            ]
        )
        for fold in audit_folds
    ]
    if len(thresholds) != OUTER_FOLDS or len(maxima) != OUTER_FOLDS:
        raise SelectorCalibrationError("final parameters require all four outer folds")
    threshold = float(statistics.median(thresholds))
    counts = Counter(maxima)
    highest_count = max(counts.values())
    maximum = min(value for value, count in counts.items() if count == highest_count)
    pair_indices = list(range(len(frozen)))
    _, _, model, transformer = _fit_hybrid(
        frozen, targets, pair_indices, include_vectorizer_state=True
    )
    expected_claim_ids = sorted({int(row["claim_id"]) for row in frozen})
    if list(transformer["fit_claim_ids"]) != expected_claim_ids:
        raise SelectorCalibrationError(
            "full transformer fit claim IDs differ from all train IDs"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "training_scope": "all_809_official_train_claims_after_nested_oof_gate_pass",
        "source_split": SOURCE_SPLIT,
        "reportable_as_generalization": False,
        "selection": {
            "threshold": threshold,
            "threshold_rule": "median of four outer-fold inner-selected thresholds",
            "outer_fold_thresholds": thresholds,
            "max_sentences": maximum,
            "max_sentences_rule": "mode of four outer-fold choices; ties choose smaller",
            "outer_fold_max_sentences": maxima,
            "post_gate_full_train_retuning": False,
        },
        "model": model,
    }


def run(
    *,
    corpus_path: Path,
    claims_path: Path,
    retrieval_path: Path,
    retrieval_manifest_path: Path,
    model_dir: Path,
    output: Path,
    device: str,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
) -> Path:
    destination = output.resolve()
    _assert_not_dev(destination, "output")
    if destination.exists():
        raise SelectorCalibrationError("output directory already exists (exclusive run)")
    if device not in {"cuda", "cpu"}:
        raise SelectorCalibrationError("device must be cuda or cpu")
    if bootstrap_samples != BOOTSTRAP_SAMPLES:
        raise SelectorCalibrationError("formal bootstrap sample count must equal 10000")

    corpus_rows, claims, retrieval_rows = _validate_inputs(
        corpus_path=corpus_path.resolve(),
        claims_path=claims_path.resolve(),
        retrieval_path=retrieval_path.resolve(),
        retrieval_manifest_path=retrieval_manifest_path.resolve(),
    )
    model_identity = _model_identity(model_dir)
    input_paths = {
        "corpus": corpus_path.resolve(),
        "claims_train": claims_path.resolve(),
        "tfidf_top3_train": retrieval_path.resolve(),
        "tfidf_manifest": retrieval_manifest_path.resolve(),
    }
    initial_input_hashes = {
        name: _sha256(path) for name, path in input_paths.items()
    }
    script_path = Path(__file__).resolve()
    initial_script_sha256 = _sha256(script_path)
    protocol = _protocol()
    protocol_sha256 = _sha256_bytes(_canonical_bytes(protocol))
    destination.mkdir(parents=True, exist_ok=False)
    preregistration = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_nli_or_oof": True,
        "implementation": {
            "path": str(script_path),
            "sha256": initial_script_sha256,
        },
        "inputs": {
            name: {"path": str(path), "sha256": initial_input_hashes[name]}
            for name, path in input_paths.items()
        },
        "model": model_identity,
        "protocol": protocol,
        "protocol_sha256": protocol_sha256,
        "crash_policy": "retain incomplete exclusive output directory",
    }
    preregistration_path = destination / "preregistration.json"
    _write_exclusive(preregistration_path, _json_bytes(preregistration))
    preregistration_sha256 = _sha256(preregistration_path)
    candidates, by_claim, retrieval_by_claim = _build_candidates(
        corpus_rows, claims, retrieval_rows
    )
    claim_ids = [int(row["id"]) for row in claims]
    candidate_key_validation = _validate_candidate_keys(
        candidates, by_claim, set(claim_ids)
    )
    if len(candidates) != CANDIDATE_PAIR_COUNT:
        raise SelectorCalibrationError("fixed candidate-pair count mismatch")
    targets = _build_targets(claims, candidates)
    if sum(bool(row["is_gold_sentence"]) for row in targets) != POSITIVE_PAIR_COUNT:
        raise SelectorCalibrationError("fixed positive candidate-pair count mismatch")
    positive_pairs = {
        (
            int(row["claim_id"]),
            int(candidates[int(row["pair_index"])]["doc_id"]),
            int(candidates[int(row["pair_index"])]["sentence_index"]),
        )
        for row in targets
        if row["is_gold_sentence"]
    }
    retrievable_relation_ids = {value[0] for value in positive_pairs}
    if len(retrievable_relation_ids) != RETRIEVABLE_RELATION_CLAIM_COUNT:
        raise SelectorCalibrationError(
            "fixed Top-3 retrievable relation-claim count mismatch"
        )
    claim_text_by_id = {int(row["id"]): str(row["claim"]) for row in claims}
    components = _connected_components(
        claim_ids, retrieval_by_claim, claim_text_by_id
    )
    if len(components) != CONNECTED_COMPONENT_COUNT:
        raise SelectorCalibrationError("fixed connected-component count mismatch")
    if (
        max(int(row["claim_count"]) for row in components)
        != LARGEST_COMPONENT_CLAIM_COUNT
    ):
        raise SelectorCalibrationError("fixed largest-component claim count mismatch")
    outer_folds = _assign_component_folds(components, OUTER_FOLDS)
    isolation = _validate_fold_isolation(outer_folds, set(claim_ids))
    if (
        not isolation["claim_coverage_complete"]
        or isolation["cross_fold_claim_overlap_count"]
        or isolation["cross_fold_doc_overlap_count"]
    ):
        raise SelectorCalibrationError("outer component isolation failed")
    fold_statistics = _fold_statistics(
        outer_folds,
        {int(row["id"]): row for row in claims},
        retrievable_relation_ids,
    )
    probabilities, inference = _infer_probabilities(
        [row["sentence"] for row in candidates],
        [row["claim"] for row in candidates],
        model_dir=model_dir.resolve(),
        device=device,
    )
    frozen = _freeze_features(candidates, probabilities)
    metrics, selections, audit_folds, cases = _run_nested_oof(
        claims=claims,
        frozen=frozen,
        targets=targets,
        by_claim=by_claim,
        components=components,
        outer_folds=outer_folds,
        retrievable_relation_ids=retrievable_relation_ids,
    )
    bootstrap = _bootstrap_comparisons(
        components=components,
        cases=cases,
        samples=bootstrap_samples,
    )
    feature_whitelist_violations = sum(
        tuple(row["static_features"]) != STATIC_FEATURE_NAMES for row in frozen
    ) + int(len(FEATURE_NAMES) != 16)
    gate = _gate(
        metrics=metrics,
        audit_folds=audit_folds,
        cases=cases,
        bootstrap=bootstrap,
        isolation=isolation,
        feature_whitelist_violations=feature_whitelist_violations,
    )
    official_metrics = {
        strategy: _official_style_metrics(claims, frozen, selections[strategy])
        for strategy in STRATEGIES
    }

    configuration = protocol
    _write_exclusive(destination / "configuration.json", _json_bytes(configuration))
    _write_exclusive(destination / "frozen_features.jsonl", _jsonl_bytes(frozen))
    _write_exclusive(destination / "training_targets.jsonl", _jsonl_bytes(targets))
    _write_exclusive(destination / "connected_components.json", _json_bytes(components))
    _write_exclusive(destination / "outer_folds.json", _json_bytes(outer_folds))
    _write_exclusive(
        destination / "outer_fold_statistics.json", _json_bytes(fold_statistics)
    )
    _write_exclusive(destination / "nested_oof_audit.json", _json_bytes(audit_folds))
    _write_exclusive(destination / "nested_oof_cases.jsonl", _jsonl_bytes(cases))
    _write_exclusive(
        destination / "oof_metrics.json",
        _json_bytes(
            {
                "custom_metric_is_not_official_sentence_selection_f1": True,
                "gate_metric": "official_compatible_pooled_sentence_selection.f1",
                "selector_metrics": metrics,
                "independent_official_style_metrics_reported_separately": official_metrics,
                "official_evaluator_invoked": False,
            }
        ),
    )
    _write_exclusive(destination / "cluster_bootstrap.json", _json_bytes(bootstrap))
    _write_exclusive(destination / "acceptance_gate.json", _json_bytes(gate))
    if gate["full_train_model_permitted"]:
        final_model = _full_train_model(
            claims=claims,
            frozen=frozen,
            targets=targets,
            by_claim=by_claim,
            audit_folds=audit_folds,
        )
        _write_exclusive(destination / "selector_model.json", _json_bytes(final_model))

    ending_script_sha256 = _sha256(script_path)
    if ending_script_sha256 != initial_script_sha256:
        raise SelectorCalibrationError(
            "implementation changed during run; leave incomplete directory"
        )
    if _sha256(preregistration_path) != preregistration_sha256:
        raise SelectorCalibrationError(
            "preregistration changed during run; leave incomplete directory"
        )
    ending_input_hashes = {name: _sha256(path) for name, path in input_paths.items()}
    if ending_input_hashes != initial_input_hashes:
        raise SelectorCalibrationError(
            "input changed during run; leave incomplete directory"
        )
    output_hashes = {
        path.name: _sha256(path)
        for path in sorted(destination.iterdir())
        if path.is_file()
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_status": {
            "mode": EVALUATION_MODE,
            "source_split": SOURCE_SPLIT,
            "dev_read_or_scored": False,
            "reportable_as_generalization": False,
            "acceptance": gate["status"],
            "full_train_model_emitted": (destination / "selector_model.json").is_file(),
        },
        "implementation": {
            "path": str(script_path),
            "sha256": initial_script_sha256,
            "unchanged_during_run": True,
        },
        "preregistration": {
            "path": str(preregistration_path),
            "sha256": preregistration_sha256,
            "protocol_sha256": protocol_sha256,
        },
        "inputs": {
            name: {"path": str(path), "sha256": initial_input_hashes[name]}
            for name, path in input_paths.items()
        },
        "model": model_identity,
        "inference": inference,
        "counts": {
            "claims": len(claims),
            "relation_claims": RELATION_CLAIM_COUNT,
            "nei_claims": NEI_CLAIM_COUNT,
            "retrievable_relation_claims": len(retrievable_relation_ids),
            "candidate_sentence_pairs": len(frozen),
            "connected_components": len(components),
        },
        "validation": {
            **isolation,
            **candidate_key_validation,
            "preregistration_unchanged_during_run": True,
            "inputs_unchanged_during_run": True,
            "feature_whitelist_violations": feature_whitelist_violations,
            "nli_inference_pass_count": inference["inference_pass_count"],
            "fixed_counts": {
                "candidate_sentence_pairs": len(frozen),
                "positive_candidate_pairs": sum(
                    bool(row["is_gold_sentence"]) for row in targets
                ),
                "connected_components": len(components),
                "largest_component_claims": max(
                    int(row["claim_count"]) for row in components
                ),
            },
            "outer_fold_statistics": fold_statistics,
        },
        "environment": {
            "python": sys.version,
            "torch": importlib.metadata.version("torch"),
            "transformers": importlib.metadata.version("transformers"),
            "scikit_learn": importlib.metadata.version("scikit-learn"),
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
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args(argv)


def main() -> None:
    args = _parse_args()
    destination = run(
        corpus_path=args.corpus,
        claims_path=args.claims_train,
        retrieval_path=args.retrieval_train,
        retrieval_manifest_path=args.retrieval_manifest,
        model_dir=args.model,
        output=args.out,
        device=args.device,
        bootstrap_samples=BOOTSTRAP_SAMPLES,
    )
    gate = json.loads((destination / "acceptance_gate.json").read_text(encoding="utf-8"))
    print(destination)
    if gate["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
