"""Bounded train-only local-generation diagnostic through the production Judge.

The Python 3.11 application calls an isolated, pinned Python 3.9 CUDA worker.
No cloud endpoint, retry, Challenger, retrieval, or selector inference is used.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import queue
import re
import subprocess
import sys
import threading
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from evidencetrace.eval.frozen_judge import build_judge_input, judge_frozen_row
from evidencetrace.eval.scifact_official import (
    compute_official_pipeline_metrics,
    validate_official_data,
)
from evidencetrace.model_client import OpenAICompatibleClient

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
PYTHON = WORKSPACE / "env/evidencetrace-scifact/Scripts/python.exe"
GPU_PYTHON = WORKSPACE / "env/scifact-verisci-py39-gpu/Scripts/python.exe"
WORKER = PROJECT / "scripts/compat/scifact_local_judge_worker.py"
FROZEN = PROJECT / "eval_runs/scifact_selector_train_oof_frozen_v1"
FREEZER = PROJECT / "scripts/freeze_scifact_selector_upstream.py"
FREEZER_SHA = "7ff6615c3107ac4353e70bf68f5882f3ac63026ee8cc082947a617806d81d5d7"
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
SCHEMA = "scifact-local-production-judge-pilot-v1"
PILOT_SIZE = 30
MAX_OUTPUT_TOKENS = 512
MAX_INPUT_TOKENS = 4096


class LocalJudgeRunError(RuntimeError):
    """The bounded single-judge diagnostic could not complete."""


def canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("xb") as handle:
        for row in rows:
            handle.write(canonical(row) + b"\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def load_freezer():
    if sha(FREEZER) != FREEZER_SHA:
        raise LocalJudgeRunError("frozen upstream verifier source changed")
    spec = importlib.util.spec_from_file_location("_single_judge_freezer", FREEZER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def select_pilot(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deterministic selection uses IDs only, never gold or model outcomes."""
    return sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"{SCHEMA}:{row['claim_id']}".encode("ascii")
        ).hexdigest(),
    )[:PILOT_SIZE]


def synthetic_probes() -> list[dict[str, Any]]:
    definitions = [
        ("The trial enrolled adults.", "The trial enrolled adults.", "entailed"),
        (
            "The treatment increased survival.",
            "The treatment reduced survival.",
            "contradicted",
        ),
        (
            "The trial enrolled adults.",
            "The telescope observed distant galaxies.",
            "not_in_source",
        ),
    ]
    rows = []
    for index, (claim, evidence, expected) in enumerate(definitions):
        context = {
            "claim": claim,
            "documents": [
                {
                    "doc_id": 900000 + index,
                    "rank": 1,
                    "title": "Synthetic probe",
                    "sentences": [{"sentence_index": 0, "text": evidence}],
                }
            ],
        }
        rows.append(
            {
                "claim_id": 900000 + index,
                "source": "synthetic_diagnostic",
                "outer_fold": -1,
                "context": context,
                "canonical_context_sha256": digest(context),
                "citation_sentence_indices": {str(900000 + index): [0]},
                "top3_doc_ids": [900000 + index],
                "expected_relation_for_scoring_only": expected,
            }
        )
    return rows


class LocalWorkerTransport(httpx.BaseTransport):
    """Real model inference transport, not a fixture or canned-response client."""

    def __init__(self, output: Path, max_calls: int) -> None:
        self.output = output
        self.events: list[dict[str, Any]] = []
        self.pending: queue.Queue[str | None] = queue.Queue()
        self.log = (output / "worker_stderr.log").open("xb")
        self.audit = (output / "generation_journal.jsonl").open("xb")
        self.audit_chain = "0" * 64
        self.process: subprocess.Popen[str] | None = None
        self.closed = False
        env = os.environ.copy()
        env.update(
            {
                "PYTHONIOENCODING": "utf-8",
                "PYTHONDONTWRITEBYTECODE": "1",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            }
        )
        try:
            self.process = subprocess.Popen(
                [
                    str(GPU_PYTHON),
                    "-B",
                    "-u",
                    "-X",
                    "faulthandler",
                    str(WORKER),
                    "--max-new-tokens",
                    str(MAX_OUTPUT_TOKENS),
                    "--max-input-tokens",
                    str(MAX_INPUT_TOKENS),
                    "--max-calls",
                    str(max_calls),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self.log,
                text=True,
                encoding="utf-8",
                bufsize=1,
                cwd=PROJECT,
                env=env,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            threading.Thread(target=self._read_lines, daemon=True).start()
            self.ready = self._receive(300)
            if self.ready.get("type") != "ready":
                raise LocalJudgeRunError("local worker did not become ready")
            write_json(output / "worker_ready.json", self.ready)
        except BaseException:
            self.shutdown(abort=True)
            raise

    def _read_lines(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        try:
            for line in self.process.stdout:
                self.pending.put(line)
        finally:
            self.pending.put(None)

    def _receive(self, timeout: float) -> dict[str, Any]:
        try:
            line = self.pending.get(timeout=timeout)
        except queue.Empty:
            raise LocalJudgeRunError("local worker timed out") from None
        if line is None:
            raise LocalJudgeRunError("local worker exited before its response")
        try:
            item = json.loads(line)
        except ValueError:
            raise LocalJudgeRunError(
                "local worker emitted invalid protocol JSON"
            ) from None
        if isinstance(item, dict) and item.get("type") == "fatal":
            write_json(self.output / "worker_fatal.json", item)
            raise LocalJudgeRunError("local worker reported a fatal inference error")
        if not isinstance(item, dict):
            raise LocalJudgeRunError("local worker emitted an invalid protocol object")
        return item

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        if request.method != "POST" or str(request.url) != (
            "http://local-inference.invalid/v1/chat/completions"
        ):
            raise LocalJudgeRunError("unexpected local transport destination")
        body = json.loads(request.content)
        request_id = len(self.events)
        assert self.process is not None and self.process.stdin is not None
        self.process.stdin.write(
            json.dumps(
                {
                    "request_id": request_id,
                    "body": body,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        self.process.stdin.flush()
        result = self._receive(300)
        if result.get("type") != "response" or result.get("request_id") != request_id:
            raise LocalJudgeRunError("local worker response identity mismatch")
        audit = result["audit"]
        response_body = result["body"]
        usage = response_body["usage"]
        if (
            audit["request_body_sha256"] != digest(body)
            or audit["messages_sha256"] != digest(body["messages"])
            or response_body["model"] != MODEL_ID
            or audit["raw_output"]
            != response_body["choices"][0]["message"]["content"]
            or usage["prompt_tokens"] != audit["input_tokens"]
            or usage["completion_tokens"] != audit["generated_output_tokens"]
            or usage["total_tokens"]
            != usage["prompt_tokens"] + usage["completion_tokens"]
        ):
            raise LocalJudgeRunError("local request, response or usage binding differs")
        record = {
            "request_id": request_id,
            "request_body": body,
            "response": result,
            "previous_sha256": self.audit_chain,
        }
        self.audit_chain = digest(record)
        record["record_sha256"] = self.audit_chain
        self.audit.write(canonical(record) + b"\n")
        self.audit.flush()
        os.fsync(self.audit.fileno())
        self.events.append(result)
        return httpx.Response(200, json=result["body"], request=request)

    def close(self) -> None:
        # The production client closes an httpx.Client after every request.
        # The isolated worker deliberately survives until the diagnostic ends.
        pass

    def shutdown(self, *, abort: bool = False) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            process = self.process
            if process is not None and process.poll() is None:
                if not abort and process.stdin is not None:
                    try:
                        process.stdin.write('{"type":"shutdown"}\n')
                        process.stdin.flush()
                        process.wait(timeout=15)
                    except (OSError, subprocess.TimeoutExpired):
                        process.terminate()
                else:
                    process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
            if process is not None:
                for stream in (process.stdin, process.stdout):
                    if stream is not None:
                        stream.close()
                write_json(self.output / "worker_exit.json", {
                    "exit_code": process.returncode, "abort_requested": abort,
                })
        finally:
            self.audit.close()
            self.log.close()


def source_bindings() -> dict[str, str]:
    paths = [
        Path(__file__),
        WORKER,
        FREEZER,
        PROJECT / "scripts/run_scifact_qwen_oracle_gate.py",
        PROJECT / "scripts/check_scifact_gpu_runtime.py",
    ]
    paths.extend((PROJECT / "src/evidencetrace").rglob("*.py"))
    return {str(path.relative_to(PROJECT)): sha(path) for path in sorted(paths)}


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)]


def summarize(
    records: list[dict[str, Any]],
    claims: list[dict[str, Any]],
    corpus: list[dict[str, Any]],
) -> dict[str, Any]:
    gold = {row["id"]: row for row in claims}
    predictions = [row["prediction"] for row in records]
    outcomes = [item for row in records for item in row["outcomes"]]
    nei_rows = [row for row in records if not gold[row["claim_id"]]["evidence"]]
    nonempty_nei = [
        row
        for row in nei_rows
        if any(item["doc_id"] is not None for item in row["outcomes"])
    ]
    complete_docs = []
    for row in records:
        for item in row["outcomes"]:
            rationales = gold[row["claim_id"]]["evidence"].get(str(item["doc_id"]), [])
            selected = {
                offset["sentence_index"]
                for group in item["evidence_groups"]
                for offset in group["sentence_offsets"]
            }
            if not any(
                set(rationale["sentences"]) <= selected for rationale in rationales
            ):
                continue
            label = {"entailed": "SUPPORT", "contradicted": "CONTRADICT"}.get(
                item["relation"]
            )
            complete_docs.append(
                {
                    "claim_id": row["claim_id"],
                    "doc_id": item["doc_id"],
                    "status": item["status"],
                    "gold_label": rationales[0]["label"],
                    "validated_label_correct": (
                        item["status"] == "ok" and label == rationales[0]["label"]
                    ),
                }
            )
    return {
        "claim_count": len(records),
        "claim_status_counts": dict(Counter(row["status"] for row in records)),
        "outcome_status_counts": dict(Counter(item["status"] for item in outcomes)),
        "explicit_nei_correct": sum(row["pure_nei_eligible"] for row in nei_rows),
        "nei_claim_count": len(nei_rows),
        "nonempty_nei_claim_count": len(nonempty_nei),
        "nonempty_nei_explicit_correct": sum(
            row["pure_nei_eligible"] for row in nonempty_nei
        ),
        "error_categories": dict(
            Counter(item["error"]["category"] for item in outcomes if item["error"])
        ),
        "complete_gold_evidence_available_docs": complete_docs,
        "complete_gold_evidence_available_doc_count": len(complete_docs),
        "validated_label_correct_on_complete_evidence": sum(
            item["validated_label_correct"] for item in complete_docs
        ),
        "official_compatible_metrics": compute_official_pipeline_metrics(
            predictions,
            validate_official_data(corpus, claims),
        ),
        "failure_policy": "failed/unsupported docs omitted from official predictions; "
        "all claims remain in denominator; errors NEVER count as explicit correct NEI",
        "citation_policy": "actual exact model quote mapped to original sentence IDs; "
        "not all selected sentences; a partial quote is not a full-sentence quote",
        "scope": "30-claim ID-selected repeated train diagnostic, NOT generalization",
    }


def run(output: Path, *, execute: bool) -> dict[str, Any]:
    if Path(sys.executable).resolve() != PYTHON.resolve():
        raise LocalJudgeRunError("use the fixed application Python 3.11 interpreter")
    output = output.resolve()
    if (
        output.parent != (PROJECT / "eval_runs").resolve()
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", output.name)
        or "train" not in output.name.split("_")
        or re.search(r"(^|[_.-])(dev|test)([_.-]|$)", output.name)
        or output.exists()
    ):
        raise LocalJudgeRunError("output must be a new project-local train run")
    print("[single-judge] Verifying frozen upstream; no model loaded.", flush=True)
    bundle = load_freezer().load_frozen(FROZEN)
    all_rows = bundle["contexts"]
    rows = select_pilot(all_rows)
    probes = synthetic_probes()
    # Validate all 809 input adaptations without inference, not only the pilot.
    for row in all_rows + probes:
        for document in row["context"]["documents"] or [None]:
            build_judge_input(row, document)
    bindings = source_bindings()
    freeze_hash = sha(FROZEN / "freeze.json")
    inputs = bundle["freeze"]["bindings"]
    claims_path = Path(inputs["input/claims_train"]["path"])
    corpus_path = Path(inputs["input/corpus"]["path"])
    selected_ids = {row["claim_id"] for row in rows}
    claims = [row for row in read_jsonl(claims_path) if row["id"] in selected_ids]
    corpus = read_jsonl(corpus_path)
    max_calls = sum(len(row["context"]["documents"]) for row in rows + probes)
    if max_calls > 100 or len(rows) != PILOT_SIZE:
        raise LocalJudgeRunError("pilot budget or sample size invalid")
    protocol = {
        "schema_version": SCHEMA,
        "model": MODEL_ID,
        "upstream_freeze_sha256": freeze_hash,
        "source_bindings": bindings,
        "selected_claim_ids": [row["claim_id"] for row in rows],
        "sampling": "30 smallest SHA256(schema_version + ':' + claim_id); no gold used",
        "main_contexts_sha256": digest(rows),
        "synthetic_probes_sha256": digest(probes),
        "main_scope": "production ClaimJudgeAgent, one attempt per selected document",
        "exclusions": [
            "cloud",
            "Challenger",
            "retries",
            "retrieval",
            "reranking",
            "dev",
        ],
        "input_protocol": "current production serializer/prompts/schema/guards, "
        "version pinned by source_bindings (not a claim of unchanged historical prompts); "
        "consecutive selected sentences grouped with newline; no title or hidden gold; "
        "no unselected sentences; neutral score=0",
        "output_protocol": "greedy raw JSON; NO JSON grammar, repair or prefill; "
        "production schema and citation guards remain enabled",
        "projection": "entailed -> SUPPORT; contradicted -> CONTRADICT; "
        "others omitted, raw relations/errors retained; only not_in_source is true NEI",
        "max_provider_attempts": max_calls,
        "max_input_tokens": MAX_INPUT_TOKENS,
        "max_new_tokens": MAX_OUTPUT_TOKENS,
        "retries": 0,
        "diagnostic_probe_calls": 3,
        "performance": "sequential observed latency, excludes model load/upstream; "
        "not a stable benchmark; synthetic probes excluded from main totals",
        "cost": "API cost 0; local compute and total cost unpriced/null",
        "acceptance": "report raw outcomes, schema/scope/guard errors, explicit NEI "
        "and official scores; no posthoc quality threshold or prompt search",
        "next_action": "do not auto-call DeepSeek; interpret local failures first",
        "reportable_as_generalization": False,
    }
    output.mkdir(exist_ok=False)
    write_json(
        output / "preregistration.json",
        {
            "created_at": datetime.now(UTC).isoformat(),
            "protocol": protocol,
            "protocol_sha256": digest(protocol),
        },
    )
    write_jsonl(output / "contexts_train_pilot.jsonl", rows)
    write_jsonl(output / "claims_train_pilot.jsonl", claims)
    write_jsonl(output / "synthetic_probes.jsonl", probes)
    preflight = {
        "validated_frozen_claims": len(all_rows),
        "pilot_claims": len(rows),
        "pilot_selected_documents": max_calls - 3,
        "pilot_empty_claims": sum(not row["context"]["documents"] for row in rows),
        "max_model_calls_including_probes": max_calls,
        "status": "input_contracts_validated_no_inference_yet",
    }
    write_json(output / "preflight.json", preflight)
    print("[single-judge] " + json.dumps(preflight), flush=True)
    if not execute:
        return preflight
    transport = None
    records = []
    try:
        transport = LocalWorkerTransport(output, max_calls)
        client = OpenAICompatibleClient(
            model_id=MODEL_ID,
            api_key="local-worker-no-external-credential",
            base_url="http://local-inference.invalid/v1",
            temperature=0,
            max_calls=max_calls,
            max_tokens=MAX_OUTPUT_TOKENS,
            transport=transport,
        )
        probe_records = []
        for probe in probes:
            result = judge_frozen_row(probe, client)
            result["expected_relation_for_scoring_only"] = probe[
                "expected_relation_for_scoring_only"
            ]
            probe_records.append(result)
            print(
                f"[single-judge] synthetic {probe['claim_id']}: {result['status']}",
                flush=True,
            )
        write_jsonl(output / "synthetic_results.jsonl", probe_records)
        model_events_before_main = len(client.telemetry_events)
        with (output / "judgements_train.jsonl").open("xb") as journal:
            for index, row in enumerate(rows, 1):
                started = time.perf_counter()
                result = judge_frozen_row(row, client)
                result["observed_judge_path_ms"] = (
                    time.perf_counter() - started
                ) * 1000
                records.append(result)
                journal.write(canonical(result) + b"\n")
                journal.flush()
                os.fsync(journal.fileno())
                print(
                    f"[single-judge] {index}/{len(rows)} claim={row['claim_id']} "
                    f"status={result['status']}",
                    flush=True,
                )
        telemetry = client.telemetry_events[model_events_before_main:]
        write_jsonl(
            output / "predictions_train.jsonl", [r["prediction"] for r in records]
        )
        report = summarize(records, claims, corpus)
        complete_usage = all(
            item.usage_status == "reported"
            and item.input_tokens is not None
            and item.output_tokens is not None
            for item in telemetry
        )
        report["telemetry"] = {
            "main_provider_calls": len(telemetry),
            "probe_provider_calls": model_events_before_main,
            "physical_model_calls": len(transport.events),
            "input_tokens": sum(item.input_tokens for item in telemetry)
            if complete_usage else None,
            "output_tokens": sum(item.output_tokens for item in telemetry)
            if complete_usage else None,
            "usage_reported_for_every_call": complete_usage,
            "observed_judge_path_p95_ms": percentile(
                [row["observed_judge_path_ms"] for row in records],
                0.95,
            ),
            "api_cost_usd": 0,
            "local_compute_cost_usd": None,
            "total_cost_usd": None,
        }
        if len(transport.events) != len(client.telemetry_events):
            raise LocalJudgeRunError(
                "physical generation and provider accounting differ"
            )
        if source_bindings() != bindings or sha(FROZEN / "freeze.json") != freeze_hash:
            raise LocalJudgeRunError("source changed during the frozen diagnostic")
        write_json(output / "report.json", report)
        transport.shutdown()
        if transport.process is None or transport.process.returncode != 0:
            raise LocalJudgeRunError("local worker did not exit cleanly")
        write_json(
            output / "run_manifest.json",
            {
                "schema_version": SCHEMA,
                "completed_claims": len(records),
                "status": "completed_diagnostic_not_a_quality_acceptance",
                "protocol_sha256": digest(protocol),
                "dev_read_or_scored": False,
                "cloud_calls": 0,
                "reportable_as_generalization": False,
                "generation_journal_final_sha256": transport.audit_chain,
                "outputs_sha256": {
                    path.name: sha(path)
                    for path in sorted(output.iterdir())
                    if path.is_file()
                },
            },
        )
        return report
    except BaseException as error:
        if transport is not None:
            transport.shutdown(abort=True)
        write_json(
            output / "failure.json",
            {
                "status": "incomplete_diagnostic",
                "completed_claims": len(records),
                "exception_type": type(error).__name__,
                "worker_exit_code": transport.process.returncode
                if transport is not None and transport.process is not None
                else None,
                "resume_supported": False,
            },
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually run the bounded local GPU diagnostic",
    )
    args = parser.parse_args()
    result = run(args.out, execute=args.execute)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
