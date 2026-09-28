"""Run a pre-registered, train-only Qwen SciFact relation oracle gate.

This is a diagnostic of binary SUPPORT/CONTRADICT capability when a gold
rationale is supplied.  It is not a generalization metric, retrieval score, or
Agent score.  The runner never accepts or reads a dev input.
"""

from __future__ import annotations

# The pinned CUDA environment uses Python 3.9.
# ruff: noqa: B905, UP017, UP045
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import sys
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

SCHEMA_VERSION = "scifact-qwen-relation-oracle-gate-v1"
CASE_SCHEMA_VERSION = "scifact-qwen-relation-oracle-case-v1"
PREREGISTRATION_SCHEMA_VERSION = "scifact-qwen-oracle-preregistration-v1"
EVALUATION_SCOPE = "oracle_gold_rationale_official_train_only_diagnostic"
EXPECTED_TRAIN_CLAIMS = 809
EXPECTED_RELATION_CLAIMS = 505
EXPECTED_LABEL_COUNTS = {"SUPPORT": 332, "CONTRADICT": 173}
OFFICIAL_CORPUS_SHA256 = (
    "b8d6c89624cb2ed74dee8938effc4f5d8bd2086887880af8110d64be4ceade62"
)
OFFICIAL_TRAIN_CLAIMS_SHA256 = (
    "f4c8fa82d8bd0653a9cc8d61a6ea48c25eacea64e90af5dbf390ebb1b74372f0"
)
MODEL_REPOSITORY = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
LABELS = ("SUPPORT", "CONTRADICT")
LETTER_IDS = {"A": 32, "B": 33}
MIN_CLASS_RECALL = 0.60
MIN_MACRO_F1_GAIN_OVER_MAJORITY = 0.05
TIE_BREAK_LABEL = "SUPPORT"
DEV_TOKEN = re.compile(r"(^|[_.-])dev([_.-]|$)", re.IGNORECASE)

SYSTEM_PROMPT = (
    "You are a scientific claim relation classifier. Use only the supplied "
    "premise and follow the label mapping exactly."
)
USER_PROMPT_TEMPLATE = """Decide whether the premise SUPPORTS or CONTRADICTS the claim.
Return only the mapped letter as the next token.

Label mapping:
A = {a_label}
B = {b_label}

Premise:
{premise}

Claim:
{claim}"""
ASSISTANT_PREFILL = "Answer: "
MAPPINGS = (
    {"mapping_id": "support_a", "A": "SUPPORT", "B": "CONTRADICT"},
    {"mapping_id": "support_b", "A": "CONTRADICT", "B": "SUPPORT"},
)


class OracleGateError(RuntimeError):
    """Raised when the frozen diagnostic contract is violated."""


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


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise OracleGateError(f"expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def _assert_not_dev(path: Path, role: str) -> None:
    if any(DEV_TOKEN.search(part) for part in path.resolve().parts):
        raise OracleGateError(f"{role} path contains forbidden dev token")


def _write_exclusive(path: Path, value: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(value)


def _validate_source_files(
    corpus_path: Path, claims_path: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    for path, role in ((corpus_path, "corpus"), (claims_path, "claims")):
        _assert_not_dev(path, role)
        if not path.is_file():
            raise OracleGateError(f"missing {role} file")
    if claims_path.name != "claims_train.jsonl":
        raise OracleGateError("only the official claims_train.jsonl is accepted")
    if _sha256(corpus_path) != OFFICIAL_CORPUS_SHA256:
        raise OracleGateError("official corpus hash mismatch")
    if _sha256(claims_path) != OFFICIAL_TRAIN_CLAIMS_SHA256:
        raise OracleGateError("official train claims hash mismatch")
    corpus = _read_jsonl(corpus_path)
    claims = _read_jsonl(claims_path)
    if len(corpus) != 5183 or len(claims) != EXPECTED_TRAIN_CLAIMS:
        raise OracleGateError("official SciFact input counts mismatch")
    return corpus, claims


def _valid_rationale_candidates(
    claim: dict[str, Any], corpus_by_id: dict[int, dict[str, Any]]
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    evidence = claim.get("evidence")
    if not isinstance(evidence, dict):
        raise OracleGateError(f"claim {claim.get('id')} has invalid evidence")
    for raw_doc_id, groups in evidence.items():
        try:
            doc_id = int(raw_doc_id)
        except (TypeError, ValueError) as exc:
            raise OracleGateError("invalid evidence document id") from exc
        document = corpus_by_id.get(doc_id)
        if document is None:
            raise OracleGateError(f"missing gold document {doc_id}")
        abstract = document.get("abstract")
        if not isinstance(abstract, list) or any(
            not isinstance(sentence, str) for sentence in abstract
        ):
            raise OracleGateError(f"invalid abstract for document {doc_id}")
        if not isinstance(groups, list):
            raise OracleGateError("invalid evidence group list")
        for group_index, group in enumerate(groups):
            if not isinstance(group, dict):
                raise OracleGateError("invalid evidence group")
            label = group.get("label")
            raw_sentences = group.get("sentences")
            if label not in LABELS or not isinstance(raw_sentences, list):
                raise OracleGateError("invalid relation evidence group")
            if not raw_sentences or any(
                type(index) is not int for index in raw_sentences
            ):
                raise OracleGateError("invalid rationale sentence indices")
            sentence_ids = tuple(sorted(raw_sentences))
            if len(set(sentence_ids)) != len(sentence_ids) or any(
                index < 0 or index >= len(abstract) for index in sentence_ids
            ):
                raise OracleGateError("out-of-range or duplicate rationale sentence")
            premise = " ".join(abstract[index] for index in sentence_ids)
            if not premise.strip():
                raise OracleGateError("empty gold rationale text")
            candidates.append(
                {
                    "doc_id": doc_id,
                    "group_index": group_index,
                    "sentence_ids": list(sentence_ids),
                    "label": label,
                    "premise": premise,
                }
            )
    return candidates


def _select_oracle_cases(
    corpus_rows: Sequence[dict[str, Any]], claims: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    corpus_by_id = {int(row["doc_id"]): row for row in corpus_rows}
    cases: list[dict[str, Any]] = []
    for claim in claims:
        evidence = claim.get("evidence")
        if not evidence:
            continue
        candidates = _valid_rationale_candidates(claim, corpus_by_id)
        if not candidates:
            raise OracleGateError(f"relation claim {claim.get('id')} has no rationale")
        labels = {candidate["label"] for candidate in candidates}
        if len(labels) != 1:
            raise OracleGateError(f"claim {claim.get('id')} has conflicting labels")
        # Frozen rule: fewest sentences, then fewest Unicode code points in the
        # joined rationale, then numeric doc id, sentence tuple, annotation order.
        selected = min(
            candidates,
            key=lambda item: (
                len(item["sentence_ids"]),
                len(item["premise"]),
                item["doc_id"],
                tuple(item["sentence_ids"]),
                item["group_index"],
            ),
        )
        cases.append(
            {
                "claim_id": int(claim["id"]),
                "claim": str(claim["claim"]),
                "gold_label": selected["label"],
                "gold_doc_id": selected["doc_id"],
                "gold_sentence_ids": selected["sentence_ids"],
                "gold_group_index": selected["group_index"],
                "premise": selected["premise"],
            }
        )
    return cases


def _validate_selected_cases(cases: Sequence[dict[str, Any]]) -> None:
    if len(cases) != EXPECTED_RELATION_CLAIMS:
        raise OracleGateError("expected all 505 official train relation claims")
    counts = Counter(str(case["gold_label"]) for case in cases)
    if dict(counts) != EXPECTED_LABEL_COUNTS:
        raise OracleGateError("official train relation label counts mismatch")
    claim_ids = [int(case["claim_id"]) for case in cases]
    if len(set(claim_ids)) != len(claim_ids):
        raise OracleGateError("duplicate train claim id")


def _prompt_messages(
    case: dict[str, Any], mapping: dict[str, str]
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_PROMPT_TEMPLATE.format(
                a_label=mapping["A"],
                b_label=mapping["B"],
                premise=case["premise"],
                claim=case["claim"],
            ),
        },
        {"role": "assistant", "content": ASSISTANT_PREFILL},
    ]


def _mapped_probabilities(
    mapping: dict[str, str], probability_a: float, probability_b: float
) -> dict[str, float]:
    return {
        mapping["A"]: probability_a,
        mapping["B"]: probability_b,
    }


def _average_swapped_probabilities(
    scored_mappings: Sequence[dict[str, Any]],
) -> dict[str, float]:
    if len(scored_mappings) != 2:
        raise OracleGateError("exactly two swapped mappings are required")
    by_label = {label: [] for label in LABELS}
    for result in scored_mappings:
        mapped = _mapped_probabilities(
            result["mapping"], result["probability_a"], result["probability_b"]
        )
        for label in LABELS:
            by_label[label].append(mapped[label])
    return {label: sum(by_label[label]) / 2.0 for label in LABELS}


def _class_metrics(
    gold_labels: Sequence[str], predicted_labels: Sequence[str]
) -> dict[str, Any]:
    per_class: dict[str, Any] = {}
    f1_values: list[float] = []
    for label in LABELS:
        true_positive = sum(
            gold == label and predicted == label
            for gold, predicted in zip(gold_labels, predicted_labels)
        )
        false_positive = sum(
            gold != label and predicted == label
            for gold, predicted in zip(gold_labels, predicted_labels)
        )
        false_negative = sum(
            gold == label and predicted != label
            for gold, predicted in zip(gold_labels, predicted_labels)
        )
        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative
        precision = (
            true_positive / precision_denominator if precision_denominator else 0.0
        )
        recall = true_positive / recall_denominator if recall_denominator else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_class[label] = {
            "support": recall_denominator,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        f1_values.append(f1)
    return {
        "accuracy": sum(
            gold == predicted
            for gold, predicted in zip(gold_labels, predicted_labels)
        )
        / len(gold_labels),
        "binary_macro_f1": sum(f1_values) / len(f1_values),
        "per_class": per_class,
    }


def _evaluate_gate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    gold = [str(row["gold_label"]) for row in rows]
    predicted = [str(row["predicted_label"]) for row in rows]
    measured = _class_metrics(gold, predicted)
    majority_label = Counter(gold).most_common(1)[0][0]
    baseline = _class_metrics(gold, [majority_label] * len(gold))
    gain = measured["binary_macro_f1"] - baseline["binary_macro_f1"]
    checks = {
        "support_recall_at_least_0_60": (
            measured["per_class"]["SUPPORT"]["recall"] >= MIN_CLASS_RECALL
        ),
        "contradict_recall_at_least_0_60": (
            measured["per_class"]["CONTRADICT"]["recall"] >= MIN_CLASS_RECALL
        ),
        "macro_f1_gain_over_majority_at_least_0_05": (
            gain >= MIN_MACRO_F1_GAIN_OVER_MAJORITY
        ),
    }
    return {
        "status": "go" if all(checks.values()) else "no_go",
        "checks": checks,
        "thresholds": {
            "minimum_recall_per_class": MIN_CLASS_RECALL,
            "minimum_binary_macro_f1_absolute_gain_over_majority": (
                MIN_MACRO_F1_GAIN_OVER_MAJORITY
            ),
        },
        "qwen_oracle": measured,
        "fixed_train_majority_baseline": {
            "prediction": majority_label,
            **baseline,
        },
        "binary_macro_f1_absolute_gain_over_majority": gain,
    }


def _model_identity(model_dir: Path) -> dict[str, Any]:
    root = model_dir.resolve()
    _assert_not_dev(root, "model")
    if not root.is_dir() or root.name != MODEL_REVISION:
        raise OracleGateError("model must be the pinned local Qwen revision")
    provenance_path = root / "local_provenance.json"
    if not provenance_path.is_file():
        raise OracleGateError("missing local model provenance")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if (
        provenance.get("repository") != MODEL_REPOSITORY
        or provenance.get("revision") != MODEL_REVISION
    ):
        raise OracleGateError("local model provenance mismatch")
    declared_files = provenance.get("files_sha256")
    if (
        not isinstance(declared_files, dict)
        or "model.safetensors" not in declared_files
    ):
        raise OracleGateError("incomplete local model provenance")
    verified: dict[str, dict[str, Any]] = {}
    for relative_path, expected_hash in sorted(declared_files.items()):
        path = root / relative_path
        if not path.is_file() or _sha256(path) != expected_hash:
            raise OracleGateError(f"model file hash mismatch: {relative_path}")
        verified[relative_path] = {
            "sha256": expected_hash,
            "bytes": path.stat().st_size,
        }
    return {
        "path": str(root),
        "repository": MODEL_REPOSITORY,
        "revision": MODEL_REVISION,
        "local_provenance_sha256": _sha256(provenance_path),
        "files": verified,
        "identity_sha256": _sha256_bytes(_canonical_bytes(verified)),
    }


class QwenNextTokenScorer:
    """Score only the A/B next-token logits; never generate free-form text."""

    def __init__(self, model_dir: Path) -> None:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise OracleGateError("CUDA is required for the pinned Qwen diagnostic")
        if not torch.cuda.is_bf16_supported():
            raise OracleGateError("CUDA device does not support bfloat16")
        self.torch = torch
        self.device = torch.device("cuda")
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir), local_files_only=True
        )
        for letter, expected_id in LETTER_IDS.items():
            actual = self.tokenizer.encode(letter, add_special_tokens=False)
            if actual != [expected_id]:
                raise OracleGateError(f"{letter} is not the pinned single token")
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        ).to(self.device)
        self.model.eval()
        self.device_name = torch.cuda.get_device_name(self.device)
        self.max_position_embeddings = int(
            getattr(self.model.config, "max_position_embeddings", 0)
        )

    def score(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        started = time.perf_counter()
        rendered = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            continue_final_message=True,
        )
        encoded = self.tokenizer(
            rendered,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_tokens = int(encoded["input_ids"].shape[1])
        if self.max_position_embeddings and input_tokens > self.max_position_embeddings:
            raise OracleGateError("oracle prompt exceeds model context window")
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        self.torch.cuda.synchronize()
        inference_started = time.perf_counter()
        with self.torch.inference_mode():
            logits = self.model(**encoded).logits[0, -1].float()
            selected = logits[
                self.torch.tensor(
                    [LETTER_IDS["A"], LETTER_IDS["B"]], device=self.device
                )
            ]
            probabilities = self.torch.softmax(selected, dim=0)
        self.torch.cuda.synchronize()
        inference_ms = (time.perf_counter() - inference_started) * 1000.0
        end_to_end_ms = (time.perf_counter() - started) * 1000.0
        return {
            "rendered_prompt_sha256": _sha256_text(rendered),
            "rendered_prompt_character_count": len(rendered),
            "input_token_ids_sha256": _sha256_bytes(
                _canonical_bytes(encoded["input_ids"][0].detach().cpu().tolist())
            ),
            "input_tokens": input_tokens,
            "next_token_id_a": LETTER_IDS["A"],
            "next_token_id_b": LETTER_IDS["B"],
            "logit_a": float(selected[0].item()),
            "logit_b": float(selected[1].item()),
            "probability_a": float(probabilities[0].item()),
            "probability_b": float(probabilities[1].item()),
            "synchronized_inference_latency_ms": inference_ms,
            "end_to_end_latency_ms": end_to_end_ms,
        }


def _percentile(values: Sequence[float], proportion: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise OracleGateError("cannot summarize empty latency values")
    index = max(0, min(len(ordered) - 1, int(len(ordered) * proportion + 0.999999) - 1))
    return ordered[index]


def _preregistration(
    *, script_path: Path, corpus_path: Path, claims_path: Path, model: dict[str, Any]
) -> dict[str, Any]:
    protocol = {
        "scope": EVALUATION_SCOPE,
        "interpretation": (
            "oracle diagnostic only; not a generalization metric, retrieval metric, "
            "or Agent score"
        ),
        "case_selection": (
            "all 505 relation claims in official train file order; choose the valid "
            "gold rationale minimizing (sentence_count, joined_unicode_length, "
            "numeric_doc_id, sorted_sentence_ids, annotation_index)"
        ),
        "premise_construction": "join selected abstract sentences with one ASCII space",
        "decision": (
            "score only single-token A and B next-token logits; within each mapping "
            "apply a two-token softmax, map probabilities to relation labels, then "
            "arithmetic-mean each relation probability across the two swaps"
        ),
        "tie_break_label": TIE_BREAK_LABEL,
        "mappings": MAPPINGS,
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt_template": USER_PROMPT_TEMPLATE,
        "assistant_prefill": ASSISTANT_PREFILL,
        "gate": {
            "support_recall_minimum": MIN_CLASS_RECALL,
            "contradict_recall_minimum": MIN_CLASS_RECALL,
            "binary_macro_f1_absolute_gain_over_fixed_train_majority_minimum": (
                MIN_MACRO_F1_GAIN_OVER_MAJORITY
            ),
            "failure_status": "no_go",
        },
        "prompt_tuning_after_observing_results": "forbidden",
        "free_form_generation": "forbidden",
    }
    return {
        "schema_version": PREREGISTRATION_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": protocol,
        "protocol_sha256": _sha256_bytes(_canonical_bytes(protocol)),
        "script": {"path": str(script_path.resolve()), "sha256": _sha256(script_path)},
        "inputs": {
            "corpus": {
                "path": str(corpus_path.resolve()),
                "sha256": _sha256(corpus_path),
            },
            "claims": {
                "path": str(claims_path.resolve()),
                "sha256": _sha256(claims_path),
            },
        },
        "model": model,
    }


def run(
    *, corpus_path: Path, claims_path: Path, model_dir: Path, destination: Path
) -> dict[str, Any]:
    _assert_not_dev(destination, "output")
    if destination.exists():
        raise OracleGateError("exclusive output destination already exists")
    if not destination.parent.is_dir():
        raise OracleGateError("output parent directory does not exist")
    corpus, claims = _validate_source_files(corpus_path, claims_path)
    cases = _select_oracle_cases(corpus, claims)
    _validate_selected_cases(cases)
    model_identity = _model_identity(model_dir)
    script_path = Path(__file__).resolve()
    preregistration = _preregistration(
        script_path=script_path,
        corpus_path=corpus_path,
        claims_path=claims_path,
        model=model_identity,
    )

    destination.mkdir(exist_ok=False)
    preregistration_path = destination / "preregistration.json"
    _write_exclusive(preregistration_path, _json_bytes(preregistration))

    scorer = QwenNextTokenScorer(model_dir)
    output_rows: list[dict[str, Any]] = []
    for case in cases:
        mapping_results: list[dict[str, Any]] = []
        for mapping in MAPPINGS:
            messages = _prompt_messages(case, mapping)
            result = scorer.score(messages)
            mapping_results.append(
                {
                    "mapping_id": mapping["mapping_id"],
                    "mapping": {"A": mapping["A"], "B": mapping["B"]},
                    "messages_sha256": _sha256_bytes(_canonical_bytes(messages)),
                    **result,
                }
            )
        averaged = _average_swapped_probabilities(mapping_results)
        predicted = (
            "SUPPORT"
            if averaged["SUPPORT"] >= averaged["CONTRADICT"]
            else "CONTRADICT"
        )
        output_rows.append(
            {
                "schema_version": CASE_SCHEMA_VERSION,
                "claim_id": case["claim_id"],
                "claim_sha256": _sha256_text(case["claim"]),
                "gold_label": case["gold_label"],
                "oracle_rationale": {
                    "doc_id": case["gold_doc_id"],
                    "sentence_ids": case["gold_sentence_ids"],
                    "group_index": case["gold_group_index"],
                    "premise_sha256": _sha256_text(case["premise"]),
                    "premise_character_count": len(case["premise"]),
                },
                "mapping_scores": mapping_results,
                "averaged_relation_probabilities": averaged,
                "predicted_label": predicted,
                "correct": predicted == case["gold_label"],
            }
        )

    gate = _evaluate_gate(output_rows)
    latency_values = [
        float(result["synchronized_inference_latency_ms"])
        for row in output_rows
        for result in row["mapping_scores"]
    ]
    input_token_values = [
        int(result["input_tokens"])
        for row in output_rows
        for result in row["mapping_scores"]
    ]
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "scope": EVALUATION_SCOPE,
        "interpretation": (
            "oracle diagnostic only; not a generalization metric, retrieval metric, "
            "or Agent score"
        ),
        "case_count": len(output_rows),
        "physical_forward_passes": len(output_rows) * len(MAPPINGS),
        "gate": gate,
        "telemetry": {
            "input_tokens": {
                "total": sum(input_token_values),
                "min": min(input_token_values),
                "max": max(input_token_values),
                "mean": sum(input_token_values) / len(input_token_values),
            },
            "synchronized_inference_latency_ms": {
                "mean": sum(latency_values) / len(latency_values),
                "p50": _percentile(latency_values, 0.50),
                "p95": _percentile(latency_values, 0.95),
                "max": max(latency_values),
            },
        },
    }
    cases_path = destination / "cases.jsonl"
    metrics_path = destination / "metrics.json"
    _write_exclusive(cases_path, _jsonl_bytes(output_rows))
    _write_exclusive(metrics_path, _json_bytes(metrics))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": gate["status"],
        "scope": EVALUATION_SCOPE,
        "preregistration": {
            "path": "preregistration.json",
            "sha256": _sha256(preregistration_path),
            "protocol_sha256": preregistration["protocol_sha256"],
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": importlib.metadata.version("torch"),
            "transformers": importlib.metadata.version("transformers"),
            "device": scorer.device_name,
            "dtype": "torch.bfloat16",
            "attention_implementation": "eager",
            "free_form_generation_used": False,
        },
        "model": model_identity,
        "inputs": preregistration["inputs"],
        "selected_cases_sha256": _sha256_bytes(
            _canonical_bytes(
                [
                    {
                        "claim_id": case["claim_id"],
                        "gold_label": case["gold_label"],
                        "gold_doc_id": case["gold_doc_id"],
                        "gold_sentence_ids": case["gold_sentence_ids"],
                        "claim_sha256": _sha256_text(case["claim"]),
                        "premise_sha256": _sha256_text(case["premise"]),
                    }
                    for case in cases
                ]
            )
        ),
        "counts": {
            "official_train_claims": len(claims),
            "relation_claims": len(output_rows),
            "label_counts": dict(Counter(row["gold_label"] for row in output_rows)),
            "physical_forward_passes": len(output_rows) * len(MAPPINGS),
        },
        "outputs": {
            "cases.jsonl": {"sha256": _sha256(cases_path), "rows": len(output_rows)},
            "metrics.json": {"sha256": _sha256(metrics_path)},
        },
        "gate": gate,
    }
    manifest_path = destination / "run_manifest.json"
    _write_exclusive(manifest_path, _json_bytes(manifest))
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--claims", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = run(
            corpus_path=args.corpus,
            claims_path=args.claims,
            model_dir=args.model,
            destination=args.out,
        )
    except (OracleGateError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"status": manifest["status"], "out": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
