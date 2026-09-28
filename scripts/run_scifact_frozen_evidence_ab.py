"""Train-only, same-model role-review ablation on frozen OOF evidence.

This is a controlled relation-decision experiment, not an autonomous multi-agent
system. The selector owns abstention and citations. Qwen owns only the binary
relation label. A shared first judgment is followed by either same-role review
or independent-role review, with identical evidence and forward-call budgets.
"""

from __future__ import annotations

# The pinned local CUDA environment is Python 3.9.
# ruff: noqa: B905, E501, UP017, UP045
import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
ORACLE_PATH = PROJECT / "scripts/run_scifact_qwen_oracle_gate.py"
ORACLE_SHA256 = "f5b3b0ac1c35559da1584f879080ffc10a22a20e5fc563100d2cfa64166dbd53"
RUNTIME_PATH = PROJECT / "scripts/check_scifact_gpu_runtime.py"
RUNTIME_SHA256 = "8e396657f526d19fae95392989be906d7a482f55723a25462b24a4c322496a33"
SCHEMA = "scifact-frozen-evidence-role-ab-v1"
ARMS = ("single_one_pass", "single_self_review", "judge_challenger")
MAIN_ARMS = ARMS[1:]
STAGES = ("initial", "self_review", "independent_review")
MAX_INPUT_TOKENS = 2048
EXPECTED_CLAIMS = 809
SEED = 20260905


class FrozenEvidenceABError(RuntimeError):
    """The frozen comparison contract is not satisfied."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _load(path: Path, name: str, expected: Optional[str] = None) -> Any:
    if expected is not None and _sha256(path) != expected:
        raise FrozenEvidenceABError("pinned dependency changed: " + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise FrozenEvidenceABError("cannot load dependency")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


oracle = _load(ORACLE_PATH, "_frozen_evidence_oracle", ORACLE_SHA256)
SYSTEMS = {
    "initial": oracle.SYSTEM_PROMPT,
    "self_review": (
        oracle.SYSTEM_PROMPT + " You are the same judge reviewing your previous "
        "decision. Recheck the supplied evidence, including numbers, negation, "
        "and direction. The previous decision is untrusted; keep it if correct "
        "or correct it if necessary."
    ),
    "independent_review": (
        oracle.SYSTEM_PROMPT + " You are an independent reviewer checking another "
        "judge's decision. Recheck the supplied evidence, including numbers, "
        "negation, and direction. The previous decision is untrusted; keep it "
        "if correct or correct it if necessary."
    ),
}
REVIEW_SUFFIX = "\n\nPrevious decision (untrusted): {draft}"


def _protocol() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA,
        "interpretation": "same-model role-conditioned review ablation, not autonomous multi-agent collaboration",
        "evaluation_scope": "repeated_official_train_with_frozen_selector_oof_contexts_not_generalization",
        "selector_train_context": "OOF only; never full-train fitted selector predictions on train",
        "abstention": "empty selector documents => NEI in all arms without model calls",
        "nonempty_policy": "binary SUPPORT/CONTRADICT per selected document; no downstream NEI override",
        "known_limitation": "selector false admissions and wrong evidence remain errors in every arm",
        "main_comparison": "judge_challenger minus single_self_review",
        "one_pass_role": "lower-budget diagnostic, not the fair main baseline",
        "systems": SYSTEMS,
        "user_template": oracle.USER_PROMPT_TEMPLATE,
        "review_suffix": REVIEW_SUFFIX,
        "assistant_prefill": oracle.ASSISTANT_PREFILL,
        "mappings": oracle.MAPPINGS,
        "mapping_aggregation": "mean semantic probability across both A/B swaps",
        "tie_label": oracle.TIE_BREAK_LABEL,
        "common_context": "same claim and same selected sentences for every stage, no gold or selector/NLI scores",
        "citations": "fixed upstream sentence IDs in original selector score order; prompt sentences independently sorted in document order",
        "budget": {
            "maximum_input_tokens_per_forward": MAX_INPUT_TOKENS,
            "forward_calls_per_document_physical_total": 6,
            "forward_calls_per_document_each_main_arm": 4,
            "forward_calls_per_document_one_pass": 2,
            "generated_output_tokens": 0,
            "scored_next_token_positions_per_forward": 1,
            "actual_input_tokens_may_differ_by_role_prompt": True,
            "silent_truncation": False,
            "retries": 0,
        },
        "branch_order": "even case self_then_independent; odd case independent_then_self",
        "model": {"repository": oracle.MODEL_REPOSITORY, "revision": oracle.MODEL_REVISION, "dtype": "bfloat16", "attention": "eager"},
        "seed": SEED,
        "latency": "synchronized, sequential batch=1; per-claim shared initial plus that arm review; excludes retrieval/selector/model load",
        "cost": "external API USD=0; local compute cost unknown without a measured billing basis; GPU inference seconds reported separately",
        "train_gate": {
            "minimum_abstract_rationalized_f1_each_main_arm": 0.05,
            "minimum_sentence_label_f1_each_main_arm": 0.05,
            "minimum_distinct_relation_labels_each_main_arm": 2,
            "positive_challenger_gain_required": False,
            "meaning": "non-degenerate and scoreable; not competitive quality",
        },
        "followup": "no dev input accepted; freeze a future dev protocol only after external official scoring agrees",
        "crash_policy": "fsync each complete claim journal record; no resume; retain incomplete exclusive directory after failure",
    }


def _messages(context: Mapping[str, Any], document: Mapping[str, Any], stage: str,
              mapping: Mapping[str, str], draft: Optional[str] = None) -> list[dict[str, str]]:
    if stage not in STAGES or mapping not in oracle.MAPPINGS:
        raise FrozenEvidenceABError("unknown stage or label mapping")
    if stage != "initial" and draft not in oracle.LABELS:
        raise FrozenEvidenceABError("review requires a valid shared initial label")
    if stage == "initial" and draft is not None:
        raise FrozenEvidenceABError("initial decision must not receive a draft")
    premise = " ".join(str(row["text"]) for row in document["sentences"])
    if not premise or not str(context["claim"]):
        raise FrozenEvidenceABError("empty claim or evidence premise")
    user = oracle.USER_PROMPT_TEMPLATE.format(
        a_label=mapping["A"], b_label=mapping["B"], premise=premise, claim=context["claim"],
    )
    if stage != "initial":
        user += REVIEW_SUFFIX.format(draft=draft)
    return [{"role": "system", "content": SYSTEMS[stage]},
            {"role": "user", "content": user},
            {"role": "assistant", "content": oracle.ASSISTANT_PREFILL}]


def _validate_score(score: Mapping[str, Any]) -> None:
    count = score.get("input_tokens")
    if type(count) is not int or not 1 <= count <= MAX_INPUT_TOKENS:
        raise FrozenEvidenceABError("input token budget violated")
    for key in ("probability_a", "probability_b", "logit_a", "logit_b",
                "synchronized_inference_latency_ms", "end_to_end_latency_ms"):
        value = score.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise FrozenEvidenceABError("invalid model telemetry: " + key)
    if not 0 <= score["probability_a"] <= 1 or not 0 <= score["probability_b"] <= 1 or abs(score["probability_a"] + score["probability_b"] - 1) > 1e-5:
        raise FrozenEvidenceABError("invalid binary probabilities")
    if score["synchronized_inference_latency_ms"] < 0 or score["end_to_end_latency_ms"] < score["synchronized_inference_latency_ms"]:
        raise FrozenEvidenceABError("invalid synchronized timing")
    for key in ("rendered_prompt_sha256", "input_token_ids_sha256"):
        value = score.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise FrozenEvidenceABError("invalid prompt provenance hash: " + key)


def _decision(row: Mapping[str, Any], document: Mapping[str, Any], stage: str,
              scorer: Any, draft: Optional[str] = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    calls = []
    for mapping in oracle.MAPPINGS:
        messages = _messages(row["context"], document, stage, mapping, draft)
        score = scorer.score(messages)
        _validate_score(score)
        owners = list(ARMS) if stage == "initial" else ["single_self_review" if stage == "self_review" else "judge_challenger"]
        calls.append({
            **score, "call_id": "{}:{}:{}:{}".format(row["claim_id"], document["doc_id"], stage, mapping["mapping_id"]),
            "claim_id": row["claim_id"], "doc_id": document["doc_id"], "stage": stage,
            "mapping": dict(mapping), "attributed_arms": owners,
            "canonical_context_sha256": row["canonical_context_sha256"],
            "messages_sha256": _digest(messages), "draft_label": draft,
            "generated_output_tokens": 0, "scored_next_token_positions": 1,
            "external_api_cost_usd": 0.0, "local_compute_cost_usd": None,
        })
    probabilities = oracle._average_swapped_probabilities(calls)
    label = "SUPPORT" if probabilities["SUPPORT"] >= probabilities["CONTRADICT"] else "CONTRADICT"
    return {"label": label, "probabilities": probabilities, "call_ids": [call["call_id"] for call in calls]}, calls


def _account(calls: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "forward_calls": len(calls), "input_tokens": sum(call["input_tokens"] for call in calls),
        "generated_output_tokens": 0, "scored_next_token_positions": len(calls),
        "gpu_inference_ms": sum(call["synchronized_inference_latency_ms"] for call in calls),
        "agent_path_ms": sum(call["end_to_end_latency_ms"] for call in calls),
        "external_api_cost_usd": 0.0, "local_compute_cost_usd": None, "total_cost_usd": None,
    }


def _run_case(row: Mapping[str, Any], scorer: Any, case_index: int) -> dict[str, Any]:
    if row.get("source") != "oof":
        raise FrozenEvidenceABError("train requires OOF evidence, not full-fit predictions")
    context = row["context"]
    if _digest(context) != row["canonical_context_sha256"]:
        raise FrozenEvidenceABError("context hash mismatch")
    citation_order = row.get("citation_sentence_indices")
    if not isinstance(citation_order, dict) or set(citation_order) != {str(doc["doc_id"]) for doc in context["documents"]}:
        raise FrozenEvidenceABError("missing original selector citation order")
    for document in context["documents"]:
        indices = citation_order[str(document["doc_id"])]
        if not isinstance(indices, list) or any(type(index) is not int for index in indices) or len(indices) != len(set(indices)) or set(indices) != {item["sentence_index"] for item in document["sentences"]}:
            raise FrozenEvidenceABError("citation order differs from supplied evidence sentences")
    calls: list[dict[str, Any]] = []
    documents = []
    evidence: dict[str, dict[str, Any]] = {arm: {} for arm in ARMS}
    order = ("self_review", "independent_review") if case_index % 2 == 0 else ("independent_review", "self_review")
    for document in context["documents"]:
        initial, initial_calls = _decision(row, document, "initial", scorer)
        calls.extend(initial_calls)
        decisions = {"initial": initial}
        for stage in order:
            decision, review_calls = _decision(row, document, stage, scorer, initial["label"])
            decisions[stage] = decision
            calls.extend(review_calls)
        indices = citation_order[str(document["doc_id"])]
        for arm, stage in zip(ARMS, STAGES):
            evidence[arm][str(document["doc_id"])] = {"label": decisions[stage]["label"], "sentences": indices}
        documents.append({"doc_id": document["doc_id"], "decisions": decisions})
    expected = 6 * len(context["documents"])
    if len(calls) != expected or len({call["call_id"] for call in calls}) != expected:
        raise FrozenEvidenceABError("physical call coverage mismatch")
    accounts = {arm: _account([call for call in calls if arm in call["attributed_arms"]]) for arm in ARMS}
    if any(accounts[arm]["forward_calls"] != 4 * len(context["documents"]) for arm in MAIN_ARMS):
        raise FrozenEvidenceABError("main arm forward budgets differ")
    return {
        "schema_version": SCHEMA, "claim_id": row["claim_id"],
        "canonical_context_sha256": row["canonical_context_sha256"],
        "citation_sentence_indices": citation_order,
        "selector_source": "oof", "document_count": len(context["documents"]),
        "branch_order": list(order), "document_decisions": documents, "calls": calls,
        "predictions": {arm: {"id": row["claim_id"], "evidence": evidence[arm]} for arm in ARMS},
        "arm_accounting": accounts, "physical_accounting": _account(calls),
    }


class BoundedQwenScorer(oracle.QwenNextTokenScorer):
    def score(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        started = time.perf_counter()
        rendered = self.tokenizer.apply_chat_template(messages, tokenize=False, continue_final_message=True)
        count = len(self.tokenizer.encode(rendered, add_special_tokens=False))
        if count > MAX_INPUT_TOKENS:
            raise FrozenEvidenceABError("prompt exceeds fixed input budget; no silent truncation")
        result = super().score(messages)
        result["end_to_end_latency_ms"] = (time.perf_counter() - started) * 1000
        if result["input_tokens"] != count:
            raise FrozenEvidenceABError("token budget check disagrees with model tokenization")
        return result


def _percentile(values: Sequence[float], proportion: float) -> Optional[float]:
    if not values:
        return None
    return sorted(values)[max(0, math.ceil(len(values) * proportion) - 1)]


def _quality(claims: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Independently reproduce official pooled scoring; external scoring is separate."""
    by_id = {row["id"]: row for row in predictions}
    if len(by_id) != len(predictions) or set(by_id) != {row["id"] for row in claims}:
        raise FrozenEvidenceABError("quality scoring requires exact claim coverage")
    abstract, sentence = Counter(), Counter()
    labels = Counter()
    for claim in claims:
        gold = claim["evidence"]
        abstract["relevant"] += len(gold)
        sentence["relevant"] += sum(len(r["sentences"]) for group in gold.values() for r in group)
        for doc_id, prediction in by_id[claim["id"]]["evidence"].items():
            labels[prediction["label"]] += 1
            abstract["retrieved"] += 1
            sentence["retrieved"] += len(prediction["sentences"])
            annotations = gold.get(str(doc_id), [])
            if not annotations:
                continue
            correct_label = prediction["label"] == annotations[0]["label"]
            sets = [set(annotation["sentences"]) for annotation in annotations]
            cap = max(3, min(map(len, sets)))
            abstract["correct_label_only"] += int(correct_label)
            abstract["correct_rationalized"] += int(correct_label and any(s <= set(prediction["sentences"][:cap]) for s in sets))
            picked = set(prediction["sentences"])
            correct = sum(any(index in rationale and rationale <= picked for rationale in sets) for index in picked)
            sentence["correct_selection"] += correct
            sentence["correct_label"] += int(correct_label) * correct
    def score(counts: Mapping[str, int], key: str) -> dict[str, float]:
        precision = counts[key] / counts["retrieved"] if counts["retrieved"] else 0.0
        recall = counts[key] / counts["relevant"] if counts["relevant"] else 0.0
        return {"precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}
    return {
        "metrics": {"abstract_label_only": score(abstract, "correct_label_only"),
                    "abstract_rationalized": score(abstract, "correct_rationalized"),
                    "sentence_selection": score(sentence, "correct_selection"),
                    "sentence_label": score(sentence, "correct_label")},
        "counts": {"abstract": dict(abstract), "sentence": dict(sentence)},
        "predicted_document_label_counts": dict(labels),
    }


def _summary(cases: Sequence[Mapping[str, Any]], claims: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    quality = {arm: _quality(claims, [row["predictions"][arm] for row in cases]) for arm in ARMS}
    totals = {}
    for arm in ARMS:
        calls = [call for row in cases for call in row["calls"] if arm in call["attributed_arms"]]
        totals[arm] = {
            **_account(calls),
            "agent_path_p50_ms": _percentile([row["arm_accounting"][arm]["agent_path_ms"] for row in cases], .5),
            "agent_path_p95_ms": _percentile([row["arm_accounting"][arm]["agent_path_ms"] for row in cases], .95),
            "nonempty_agent_path_p95_ms": _percentile([row["arm_accounting"][arm]["agent_path_ms"] for row in cases if row["document_count"]], .95),
        }
    checks = {
        "exact_claim_coverage": len(cases) == EXPECTED_CLAIMS and len({row["claim_id"] for row in cases}) == EXPECTED_CLAIMS,
        "same_main_forward_budgets_every_case": all(row["arm_accounting"][MAIN_ARMS[0]]["forward_calls"] == row["arm_accounting"][MAIN_ARMS[1]]["forward_calls"] == 4 * row["document_count"] for row in cases),
        "exact_physical_call_coverage": all(len(row["calls"]) == 6 * row["document_count"] for row in cases),
    }
    for arm in MAIN_ARMS:
        checks[arm + "_at_least_two_relation_labels"] = len(quality[arm]["predicted_document_label_counts"]) >= 2
        for metric in ("abstract_rationalized", "sentence_label"):
            checks[arm + "_" + metric + "_f1_at_least_0_05"] = quality[arm]["metrics"][metric]["f1"] >= .05
    differences = {metric: quality["judge_challenger"]["metrics"][metric]["f1"] - quality["single_self_review"]["metrics"][metric]["f1"] for metric in quality["single_self_review"]["metrics"]}
    return {
        "scope": _protocol()["evaluation_scope"], "reportable_as_generalization": False,
        "official_evaluator_invoked": False, "external_official_crosscheck_required": True,
        "quality": quality, "accounting": totals,
        "unique_physical_accounting": _account([call for row in cases for call in row["calls"]]),
        "challenger_minus_self_review_f1": differences,
        "empty_selector_claims": sum(row["document_count"] == 0 for row in cases),
        "latency_excludes": ["model_loading", "retrieval", "selector", "disk_io"],
        "train_viability_gate": {"checks": checks, "status": "pass" if all(checks.values()) else "reject", "positive_gain_required": False, "dev_frozen": False},
    }


def _write(path: Path, value: Any) -> None:
    with path.open("xb") as handle:
        handle.write((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))


def run(*, frozen_path: Path, model_dir: Path, output: Path) -> Path:
    runtime = _load(RUNTIME_PATH, "_role_ab_runtime", RUNTIME_SHA256)
    runtime._native_error_mode()
    environment = runtime._verify_environment()
    for path in (frozen_path, model_dir, output):
        if any(token in path.name.casefold().replace("-", "_").split("_") for token in ("dev", "test")):
            raise FrozenEvidenceABError("this runner accepts train only")
    output = output.resolve()
    if output.exists():
        raise FrozenEvidenceABError("exclusive output already exists")
    freeze_module_path = PROJECT / "scripts/freeze_scifact_selector_upstream.py"
    upstream = _load(freeze_module_path, "_role_ab_upstream")
    frozen = upstream.load_frozen(frozen_path)
    frozen_document_path = frozen_path / "freeze.json" if frozen_path.is_dir() else frozen_path
    frozen_document_sha256 = _sha256(frozen_document_path)
    rows = frozen["contexts"]
    if len(rows) != EXPECTED_CLAIMS or len({row["claim_id"] for row in rows}) != EXPECTED_CLAIMS:
        raise FrozenEvidenceABError("frozen train OOF coverage is incomplete")
    source_hashes = {str(path.resolve()): _sha256(path) for path in (Path(__file__), ORACLE_PATH, RUNTIME_PATH, freeze_module_path)}
    model_identity = oracle._model_identity(model_dir)
    protocol = _protocol()
    output.mkdir(parents=True, exist_ok=False)
    prereg = {"schema_version": SCHEMA, "created_at": datetime.now(timezone.utc).isoformat(),
              "protocol": protocol, "protocol_sha256": _digest(protocol), "sources_sha256": source_hashes,
              "upstream_freeze": {"path": str(frozen_path.resolve()), "sha256": frozen_document_sha256},
              "model": model_identity, "environment": environment,
              "written_before_model_load": True, "dev_read_or_scored": False}
    _write(output / "preregistration.json", prereg)
    prereg_sha = _sha256(output / "preregistration.json")
    import torch
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model_started = time.perf_counter()
    scorer = BoundedQwenScorer(model_dir)
    loading_seconds = time.perf_counter() - model_started
    synthetic_context = {"claim": "The treatment increases survival.", "documents": []}
    synthetic_document = {"sentences": [{"sentence_index": 0, "text": "The treatment increases survival."}]}
    warmup = []
    for mapping in oracle.MAPPINGS:
        result = scorer.score(_messages(synthetic_context, synthetic_document, "initial", mapping))
        _validate_score(result)
        warmup.append(result)
    _write(output / "runtime_preflight.json", {"environment": environment, "gpu": scorer.device_name,
           "model_loading_seconds": loading_seconds, "synthetic_warmup_calls": warmup,
           "excluded_from_evaluation_accounting": True, "loaded_native_modules": runtime._loaded_native_modules()})
    cases = []
    started = time.perf_counter()
    previous_hash = prereg_sha
    with (output / "case_journal.jsonl").open("xb") as journal:
        for index, row in enumerate(rows):
            case = _run_case(row, scorer, index)
            envelope = {"index": index, "previous_sha256": previous_hash, "case": case}
            envelope["sha256"] = _digest(envelope)
            previous_hash = envelope["sha256"]
            journal.write(_canonical(envelope) + b"\n")
            journal.flush()
            os.fsync(journal.fileno())
            cases.append(case)
            if (index + 1) % 25 == 0 or index + 1 == len(rows):
                print(f"Completed {index + 1}/{len(rows)} train claims", flush=True)
    measured_seconds = time.perf_counter() - started
    # Labels are loaded for scoring only, after all model decisions are complete.
    train = WORKSPACE / "env/datasets/scifact/raw/data/claims_train.jsonl"
    if _sha256(train) != oracle.OFFICIAL_TRAIN_CLAIMS_SHA256:
        raise FrozenEvidenceABError("train scoring labels hash mismatch")
    claims = [json.loads(line) for line in train.read_text(encoding="utf-8").splitlines() if line]
    summary = _summary(cases, claims)
    for arm in ARMS:
        with (output / ("predictions_" + arm + ".jsonl")).open("xb") as handle:
            for case in cases:
                handle.write(_canonical(case["predictions"][arm]) + b"\n")
    _write(output / "comparison.json", summary)
    if {path: _sha256(Path(path)) for path in source_hashes} != source_hashes or _sha256(output / "preregistration.json") != prereg_sha:
        raise FrozenEvidenceABError("bound source or preregistration changed during run")
    if oracle._model_identity(model_dir) != model_identity:
        raise FrozenEvidenceABError("Qwen model changed during run")
    if _sha256(frozen_document_path) != frozen_document_sha256 or upstream.load_frozen(frozen_path) != frozen:
        raise FrozenEvidenceABError("upstream freeze changed during run")
    manifest = {"schema_version": SCHEMA, "evaluation_scope": protocol["evaluation_scope"],
                "dev_read_or_scored": False, "reportable_as_generalization": False,
                "preregistration_sha256": prereg_sha, "sources_sha256": source_hashes,
                "upstream_freeze": prereg["upstream_freeze"], "model": model_identity,
                "claim_count": len(cases), "final_journal_sha256": previous_hash,
                "evaluation_wall_seconds_including_journaling": measured_seconds,
                "train_viability_status": summary["train_viability_gate"]["status"],
                "outputs_sha256": {path.name: _sha256(path) for path in sorted(output.iterdir()) if path.is_file()},
                "next_step_requires_official_score_crosscheck": True}
    _write(output / "run_manifest.json", manifest)
    print("Train role-ablation completed; viability={}; official crosscheck still required".format(summary["train_viability_gate"]["status"]), flush=True)
    return output


def main() -> None:
    import faulthandler
    faulthandler.enable()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=WORKSPACE / "env/models/Qwen--Qwen2.5-1.5B-Instruct" / oracle.MODEL_REVISION)
    args = parser.parse_args()
    run(frozen_path=args.frozen, model_dir=args.model, output=args.out)


if __name__ == "__main__":
    main()
