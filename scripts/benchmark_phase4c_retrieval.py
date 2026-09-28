"""Compare the frozen pre-fix CJK retriever with the current implementation.

The baseline is loaded directly from ``git HEAD`` and must identify itself as
``lexical-cjk-ngram-v1``.  The working-tree implementation is the candidate.
Both are evaluated on the same frozen Phase 4C diagnostic cases with literal
evidence-span ranking.  This avoids using the Latin-only deterministic Judge
as a proxy for Chinese retrieval quality.

The script also runs a policy-level empty-recall probe.  It verifies that a
multi-chunk, available source returns no fabricated lexical hit by default and
that an explicit fallback is one exact bounded source slice with score zero.
It does not claim that a live model route executed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from evidencetrace.audit_models import (
    EVIDENCE_EXACT_MAX_BYTES,
    EVIDENCE_EXACT_MAX_CHARS,
)
from evidencetrace.eval.baselines import build_evidence_chunks
from evidencetrace.eval.dataset import load_dataset
from evidencetrace.retrieval.rank import (
    FULL_CONTEXT_FALLBACK_SCORE,
    RETRIEVAL_POLICY_VERSION,
    LexicalRetriever,
)


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = (
    PROJECT
    / "eval_sets"
    / "phase4c_public_zh_diagnostic"
    / "holdout_public_zh.jsonl"
)
DEFAULT_OUTPUT = (
    PROJECT
    / "eval_runs"
    / "phase4c_public_zh_diagnostic"
    / "retrieval_comparison.json"
)
DEFAULT_RUNNER_MANIFEST = (
    PROJECT
    / "eval_runs"
    / "phase4c_public_zh_diagnostic"
    / "runner_test"
    / "run_manifest.json"
)
LEGACY_PATH = "src/evidencetrace/retrieval/rank.py"
LATIN_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[._-][A-Za-z0-9]+)*%?")
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _git(*args: str) -> bytes:
    completed = subprocess.run(
        ("git", *args),
        cwd=PROJECT,
        check=True,
        capture_output=True,
    )
    return completed.stdout


def _load_legacy_retriever() -> tuple[type[Any], dict[str, str]]:
    source = _git("show", f"HEAD:{LEGACY_PATH}")
    namespace: dict[str, Any] = {
        "__name__": "phase4c_git_head_legacy_rank",
        "__file__": f"git:HEAD:{LEGACY_PATH}",
    }
    exec(compile(source, namespace["__file__"], "exec"), namespace)
    policy = str(namespace.get("RETRIEVAL_POLICY_VERSION", ""))
    if policy != "lexical-cjk-ngram-v1":
        raise RuntimeError(
            "git HEAD no longer contains the preregistered v1 retrieval baseline"
        )
    return namespace["LexicalRetriever"], {
        "git_commit": _git("rev-parse", "HEAD").decode("ascii").strip(),
        "path": LEGACY_PATH,
        "policy_version": policy,
        "source_sha256": _sha256_bytes(source),
    }


def _literal_rank(results: Iterable[Any], gold_span: str) -> int | None:
    for index, item in enumerate(results, 1):
        text = str(item.text)
        if gold_span in text or text in gold_span:
            return index
    return None


def _mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def _percentile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        return 0.0
    index = min(len(sorted_values) - 1, max(0, int(fraction * len(sorted_values))))
    return sorted_values[index]


def _paired_bootstrap(
    deltas: list[float], *, samples: int, seed: int
) -> dict[str, float | int]:
    if not deltas:
        return {"samples": samples, "seed": seed, "mean_delta": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    rng = random.Random(seed)
    n = len(deltas)
    observed = _mean(deltas)
    draws = sorted(
        _mean([deltas[rng.randrange(n)] for _ in range(n)]) for _ in range(samples)
    )
    return {
        "samples": samples,
        "seed": seed,
        "mean_delta": observed,
        "ci95_low": _percentile(draws, 0.025),
        "ci95_high": _percentile(draws, 0.975),
    }


def _summary(
    rows: list[dict[str, Any]], *, bootstrap_samples: int, seed: int
) -> dict[str, Any]:
    old_ranks = [row["baseline_v1_rank"] for row in rows]
    new_ranks = [row["candidate_v3_rank"] for row in rows]
    old_at_1 = [1.0 if rank == 1 else 0.0 for rank in old_ranks]
    new_at_1 = [1.0 if rank == 1 else 0.0 for rank in new_ranks]
    old_at_5 = [1.0 if rank is not None and rank <= 5 else 0.0 for rank in old_ranks]
    new_at_5 = [1.0 if rank is not None and rank <= 5 else 0.0 for rank in new_ranks]
    old_rr = [0.0 if rank is None else 1.0 / rank for rank in old_ranks]
    new_rr = [0.0 if rank is None else 1.0 / rank for rank in new_ranks]
    rank_deltas = [right - left for left, right in zip(old_rr, new_rr, strict=True)]
    return {
        "case_count": len(rows),
        "baseline_v1": {
            "literal_recall_at_1": _mean(old_at_1),
            "literal_recall_at_5": _mean(old_at_5),
            "literal_mrr": _mean(old_rr),
            "hit_at_5_count": int(sum(old_at_5)),
        },
        "candidate_v3": {
            "literal_recall_at_1": _mean(new_at_1),
            "literal_recall_at_5": _mean(new_at_5),
            "literal_mrr": _mean(new_rr),
            "hit_at_5_count": int(sum(new_at_5)),
        },
        "paired_candidate_minus_baseline": {
            "rank_outcomes": {
                "candidate_better": sum(delta > 0 for delta in rank_deltas),
                "tie": sum(delta == 0 for delta in rank_deltas),
                "candidate_worse": sum(delta < 0 for delta in rank_deltas),
            },
            "literal_recall_at_1": _paired_bootstrap(
                [right - left for left, right in zip(old_at_1, new_at_1, strict=True)],
                samples=bootstrap_samples,
                seed=seed + 1,
            ),
            "literal_recall_at_5": _paired_bootstrap(
                [right - left for left, right in zip(old_at_5, new_at_5, strict=True)],
                samples=bootstrap_samples,
                seed=seed + 2,
            ),
            "literal_mrr": _paired_bootstrap(
                rank_deltas,
                samples=bootstrap_samples,
                seed=seed + 3,
            ),
        },
    }


def _runner_live_status(path: Path, *, dataset_hash: str) -> dict[str, Any]:
    if not path.is_file():
        return {
            "requested": False,
            "status": "runner_artifact_missing",
            "reason": "No standard runner manifest was supplied to this analysis.",
            "manifest_path": str(path),
        }
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if value.get("dataset_hash") != dataset_hash:
        raise RuntimeError("runner manifest does not match the frozen diagnostic dataset")
    statuses = value.get("live_baseline_statuses", {})
    requested = any(status != "not_requested" for status in statuses.values())
    return {
        "requested": requested,
        "status": value.get("live_baseline", "unknown"),
        "baseline_statuses": statuses,
        "reason": value.get("skip_reason") or "See the standard runner manifest.",
        "manifest_path": str(path.resolve()),
        "manifest_sha256": _sha256_bytes(raw),
        "model_id": value.get("model_id"),
        "maximum_expected_live_model_calls": value.get(
            "maximum_expected_live_model_calls"
        ),
    }


def _policy_probe(dataset: Any) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    query = "量子纠缠熵"
    for source in dataset.sources.values():
        if not source.available:
            continue
        chunks = build_evidence_chunks(source)
        retriever = LexicalRetriever(chunks, neighbor_window=0)
        default_results = retriever.search(
            query, top_k=5, fallback_to_full_context=False
        )
        explicit_results = retriever.search(
            query, top_k=5, fallback_to_full_context=True
        )
        fallback = explicit_results[0] if len(explicit_results) == 1 else None
        rows.append(
            {
                "source_id": source.source_id,
                "chunk_count": len(chunks),
                "source_chars": len(source.content),
                "source_utf8_bytes": len(source.content.encode("utf-8")),
                "default_result_count": len(default_results),
                "explicit_result_count": len(explicit_results),
                "score_is_zero": bool(
                    fallback is not None
                    and fallback.score == FULL_CONTEXT_FALLBACK_SCORE
                ),
                "literal_full_source": bool(
                    fallback is not None
                    and fallback.text == source.content
                    and fallback.chunk.char_start == 0
                    and fallback.chunk.char_end == len(source.content)
                ),
                "bounded": bool(
                    fallback is not None
                    and len(fallback.text) <= EVIDENCE_EXACT_MAX_CHARS
                    and len(fallback.text.encode("utf-8")) <= EVIDENCE_EXACT_MAX_BYTES
                ),
            }
        )
    return {
        "probe_query": query,
        "probe_role": "offline_policy_integration_probe_not_live_agent_quality",
        "source_count": len(rows),
        "initial_empty_count": sum(row["default_result_count"] == 0 for row in rows),
        "fallback_returned_count": sum(row["explicit_result_count"] == 1 for row in rows),
        "zero_score_count": sum(row["score_is_zero"] for row in rows),
        "literal_full_source_count": sum(row["literal_full_source"] for row in rows),
        "bounded_count": sum(row["bounded"] for row in rows),
        "exact_limits": {
            "max_chars": EVIDENCE_EXACT_MAX_CHARS,
            "max_utf8_bytes": EVIDENCE_EXACT_MAX_BYTES,
        },
        "rows": rows,
    }


def _markdown(payload: dict[str, Any]) -> str:
    def row(label: str, values: dict[str, Any]) -> str:
        old = values["baseline_v1"]
        new = values["candidate_v3"]
        delta = values["paired_candidate_minus_baseline"]
        d5 = delta["literal_recall_at_5"]
        dmrr = delta["literal_mrr"]
        return (
            f"| {label} | {values['case_count']} | "
            f"{old['literal_recall_at_5']:.4f} | {new['literal_recall_at_5']:.4f} | "
            f"{d5['mean_delta']:+.4f} [{d5['ci95_low']:+.4f}, {d5['ci95_high']:+.4f}] | "
            f"{old['literal_mrr']:.4f} | {new['literal_mrr']:.4f} | "
            f"{dmrr['mean_delta']:+.4f} [{dmrr['ci95_low']:+.4f}, {dmrr['ci95_high']:+.4f}] |"
        )

    summaries = payload["retrieval_quality"]
    probe = payload["empty_retrieval_policy_probe"]
    live = payload["live_evaluation"]
    lines = [
        "# EvidenceTrace Phase 4C 检索诊断",
        "",
        "本报告比较冻结在 git HEAD 的 `lexical-cjk-ngram-v1` 与当前工作树的 CJK 检索实现。指标只使用 gold evidence 的**字面片段命中**，不使用仍为 Latin-token 口径的确定性 Judge 充当中文语义指标。",
        "",
        f"数据：{payload['dataset']['case_count']} 例（test {payload['dataset']['test_count']}），有效性 `{payload['dataset']['benchmark_validity']}`，public eligible = false。",
        "",
        "| 子集 | N | v1 Recall@5 | v3 Recall@5 | 配对差值 95% bootstrap CI | v1 MRR | v3 MRR | 配对差值 95% bootstrap CI |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        row("test substantive（主诊断）", summaries["test_substantive"]),
        row("all substantive", summaries["all_substantive"]),
        row("test mixed CJK+Latin", summaries["test_mixed_cjk_latin"]),
        row("all mixed CJK+Latin", summaries["all_mixed_cjk_latin"]),
        "",
        "`substantive` 仅含有冻结 gold evidence span 的 entailed / partially_entailed / contradicted。CI 是小样本配对 bootstrap 描述区间，不是总体显著性结论。",
        "",
        f"全部可用 case 的空结果率：v1 {payload['empty_result_rates_all_available_cases']['baseline_v1']:.4f}，v3 {payload['empty_result_rates_all_available_cases']['candidate_v3']:.4f}。",
        "",
        "## 空召回退 policy probe",
        "",
        f"对 {probe['source_count']} 个多 chunk 可用来源使用无词汇重叠查询：初始空召回 {probe['initial_empty_count']}/{probe['source_count']}；显式 fallback 返回 {probe['fallback_returned_count']}/{probe['source_count']}；score=0、完整字面切片、大小有界分别为 {probe['zero_score_count']}/{probe['source_count']}、{probe['literal_full_source_count']}/{probe['source_count']}、{probe['bounded_count']}/{probe['source_count']}。",
        "",
        "这是离线 policy/integration probe，不是在线 Agent 效果。默认检索没有伪造正分命中；只有调用方显式请求时才返回有界 full-source 候选。",
        "",
        "## 在线模型状态",
        "",
        f"Live requested: {str(live['requested']).lower()}；状态：`{live['status']}`。{live['reason']}",
        "",
        "## 边界",
        "",
        "- 该数据由官方公开文档摘录和确定性标签组成，没有独立人工复核，不能当公开 benchmark 或泛化结论。",
        "- EvidenceTrace runner 的 relation macro-F1 会另行原样落盘，但其确定性 Judge/token-F1 仍只识别 Latin/数字，不能解释为中文语义质量。",
        "- 这里的 baseline 是同一冻结数据、同一 chunker、同一 top-k 下的 git-HEAD v1 检索源码；版本和源码 SHA-256 已写入 JSON。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--runner-manifest", type=Path, default=DEFAULT_RUNNER_MANIFEST
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260902)
    args = parser.parse_args()
    if args.bootstrap_samples <= 0:
        raise SystemExit("--bootstrap-samples must be positive")

    dataset = load_dataset(args.dataset)
    legacy_retriever, baseline_meta = _load_legacy_retriever()
    current_rank_path = PROJECT / LEGACY_PATH
    current_meta = {
        "path": LEGACY_PATH,
        "policy_version": RETRIEVAL_POLICY_VERSION,
        "source_sha256": _sha256_bytes(current_rank_path.read_bytes()),
    }
    if RETRIEVAL_POLICY_VERSION != "lexical-cjk-ngram-v3":
        raise RuntimeError("candidate retrieval policy is not the expected v3")

    quality_rows: list[dict[str, Any]] = []
    availability_rows: list[dict[str, Any]] = []
    for case in dataset.cases:
        source = dataset.sources[case.source_id]
        if not source.available:
            continue
        chunks = build_evidence_chunks(source)
        query = case.claim_text or case.document_path or ""
        old_results = legacy_retriever(chunks, neighbor_window=0).search(query, top_k=5)
        new_results = LexicalRetriever(chunks, neighbor_window=0).search(
            query, top_k=5, fallback_to_full_context=False
        )
        cjk = bool(CJK_RE.search(query))
        latin_count = len(LATIN_TOKEN_RE.findall(query))
        availability_rows.append(
            {
                "case_id": case.case_id,
                "split": case.split,
                "contains_cjk": cjk,
                "latin_token_count": latin_count,
                "baseline_v1_result_count": len(old_results),
                "candidate_v3_result_count": len(new_results),
            }
        )
        if case.gold_evidence_span is None:
            continue
        old_rank = _literal_rank(old_results, case.gold_evidence_span)
        new_rank = _literal_rank(new_results, case.gold_evidence_span)
        quality_rows.append(
            {
                "case_id": case.case_id,
                "split": case.split,
                "gold_relation": case.gold_relation.value,
                "source_id": case.source_id,
                "contains_cjk": cjk,
                "latin_token_count": latin_count,
                "mixed_cjk_latin": cjk and latin_count >= 2,
                "baseline_v1_rank": old_rank,
                "candidate_v3_rank": new_rank,
                "baseline_v1_result_count": len(old_results),
                "candidate_v3_result_count": len(new_results),
                "baseline_v1_top_texts": [item.text for item in old_results],
                "candidate_v3_top_texts": [item.text for item in new_results],
            }
        )

    def subset(*, split: str | None = None, mixed: bool = False) -> list[dict[str, Any]]:
        return [
            row
            for row in quality_rows
            if (split is None or row["split"] == split)
            and (not mixed or row["mixed_cjk_latin"])
        ]

    retrieval_quality = {
        "test_substantive": _summary(
            subset(split="test"),
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed + 10,
        ),
        "all_substantive": _summary(
            subset(),
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed + 20,
        ),
        "test_mixed_cjk_latin": _summary(
            subset(split="test", mixed=True),
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed + 30,
        ),
        "all_mixed_cjk_latin": _summary(
            subset(mixed=True),
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed + 40,
        ),
    }
    payload: dict[str, Any] = {
        "schema_version": "phase4c-retrieval-comparison-v1",
        "dataset": {
            "path": str(dataset.path),
            "hash": dataset.raw_hash,
            "split_hash": dataset.split_hash,
            "case_count": len(dataset.cases),
            "dev_count": len(dataset.dev_cases),
            "test_count": len(dataset.test_cases),
            "benchmark_validity": "diagnostic_contaminated",
            "public_benchmark_eligible": False,
        },
        "baseline": baseline_meta,
        "candidate": current_meta,
        "top_k": 5,
        "metric_definition": (
            "A hit occurs when the frozen gold evidence span is a literal substring "
            "of a retrieved chunk, or the chunk is a literal substring of the span."
        ),
        "bootstrap": {"samples": args.bootstrap_samples, "seed": args.seed},
        "retrieval_quality": retrieval_quality,
        "empty_result_rates_all_available_cases": {
            "case_count": len(availability_rows),
            "baseline_v1": _mean(
                [1.0 if row["baseline_v1_result_count"] == 0 else 0.0 for row in availability_rows]
            ),
            "candidate_v3": _mean(
                [1.0 if row["candidate_v3_result_count"] == 0 else 0.0 for row in availability_rows]
            ),
        },
        "empty_retrieval_policy_probe": _policy_probe(dataset),
        "live_evaluation": _runner_live_status(
            args.runner_manifest, dataset_hash=dataset.raw_hash
        ),
        "case_rows": quality_rows,
        "availability_rows": availability_rows,
        "limitations": [
            "Official public-source diagnostic with deterministic labels; no independent human review.",
            "Small label-stratified test split; paired bootstrap intervals are descriptive.",
            "Relation macro-F1 from the Latin-token deterministic Judge is not a CJK semantic metric.",
            "The fallback probe validates policy invariants only and is not live Agent quality.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    json_payload = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    args.output.write_text(json_payload, encoding="utf-8")
    markdown_path = args.output.with_suffix(".md")
    markdown_path.write_text(_markdown(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "markdown": str(markdown_path),
                "test_substantive": retrieval_quality["test_substantive"],
                "test_mixed_cjk_latin": retrieval_quality["test_mixed_cjk_latin"],
                "policy_probe": payload["empty_retrieval_policy_probe"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
