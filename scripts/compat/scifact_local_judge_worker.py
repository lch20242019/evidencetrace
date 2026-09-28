"""Bounded local Qwen generation worker for unchanged production Judge requests.

The JSONL transport carries no credentials. This backend does not implement the
provider's json_object grammar: its raw generation is checked by the caller's
unchanged production response parser. Importing this module loads no model.
"""

from __future__ import annotations

# The pinned inference environment is Python 3.9.
# ruff: noqa: E501, UP045
import argparse
import faulthandler
import hashlib
import importlib.util
import json
import os
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Optional

PROJECT = Path(__file__).resolve().parents[2]
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
ORACLE_PATH = PROJECT / "scripts/run_scifact_qwen_oracle_gate.py"
ORACLE_SHA256 = "f5b3b0ac1c35559da1584f879080ffc10a22a20e5fc563100d2cfa64166dbd53"
RUNTIME_PATH = PROJECT / "scripts/check_scifact_gpu_runtime.py"
RUNTIME_SHA256 = "8e396657f526d19fae95392989be906d7a482f55723a25462b24a4c322496a33"


def _digest(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _load(path: Path, name: str, expected: str) -> Any:
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError("pinned dependency hash mismatch")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError("dependency cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _validate_body(body: Any, maximum: int) -> dict[str, Any]:
    expected = {"model", "temperature", "max_tokens", "messages", "response_format"}
    if not isinstance(body, dict) or set(body) != expected:
        raise ValueError("unsupported request fields")
    if body["model"] != MODEL_ID or isinstance(body["temperature"], bool) or body["temperature"] != 0:
        raise ValueError("model or greedy temperature differs from contract")
    if type(body["max_tokens"]) is not int or not 1 <= body["max_tokens"] <= maximum:
        raise ValueError("requested output budget exceeds worker ceiling")
    if body["response_format"] != {"type": "json_object"}:
        raise ValueError("unexpected production response format")
    messages = body["messages"]
    if not isinstance(messages, list) or not messages:
        raise ValueError("request messages must be nonempty")
    for message in messages:
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            raise ValueError("unsupported message fields")
        if not isinstance(message["role"], str) or message["role"] not in {"system", "user", "assistant"} or not isinstance(message["content"], str):
            raise ValueError("unsupported message type")
    return body


def _validate_limits(max_new_tokens: int, max_input_tokens: int, max_calls: int) -> None:
    for value, ceiling in ((max_new_tokens, 512), (max_input_tokens, 4096), (max_calls, 100)):
        if type(value) is not int or not 1 <= value <= ceiling:
            raise ValueError("worker limits exceed fixed contract")


def _generate(scorer: Any, body: dict[str, Any], max_input_tokens: int) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.perf_counter()
    tokenizer, torch = scorer.tokenizer, scorer.torch
    rendered = tokenizer.apply_chat_template(body["messages"], tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=False, truncation=False)
    input_ids = encoded["input_ids"][0].tolist()
    input_count = len(input_ids)
    if not 1 <= input_count <= max_input_tokens:
        raise ValueError("input budget exceeded; no truncation is permitted")
    if input_count + body["max_tokens"] > scorer.max_position_embeddings:
        raise ValueError("requested generation exceeds model context")
    encoded = {name: value.to(scorer.device) for name, value in encoded.items()}
    config = scorer.model.generation_config
    eos = config.eos_token_id if config.eos_token_id is not None else tokenizer.eos_token_id
    pad = config.pad_token_id if config.pad_token_id is not None else tokenizer.pad_token_id
    if eos is None or pad is None:
        raise ValueError("pinned model must supply EOS and padding IDs")
    parameters = dict(max_new_tokens=body["max_tokens"], do_sample=False, num_beams=1,
                      num_return_sequences=1, repetition_penalty=1.0, eos_token_id=eos,
                      pad_token_id=pad, use_cache=True)
    torch.cuda.synchronize()
    inference_started = time.perf_counter()
    with torch.inference_mode():
        sequences = scorer.model.generate(**encoded, **parameters)
    torch.cuda.synchronize()
    inference_ms = (time.perf_counter() - inference_started) * 1000
    output_ids = sequences[0, input_count:].detach().cpu().tolist()
    if not output_ids or len(output_ids) > body["max_tokens"]:
        raise ValueError("invalid generated sequence length")
    content = tokenizer.decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    eos_ids = eos if isinstance(eos, (list, tuple)) else [eos]
    if output_ids[-1] not in eos_ids and len(output_ids) != body["max_tokens"]:
        raise ValueError("generation ended before its budget without EOS")
    finish = "stop" if output_ids[-1] in eos_ids else "length"
    response = {"object": "chat.completion", "created": int(time.time()), "model": MODEL_ID,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish}],
                "usage": {"prompt_tokens": input_count, "completion_tokens": len(output_ids),
                          "total_tokens": input_count + len(output_ids)}}
    audit = {"pid": os.getpid(), "request_body_sha256": _digest(body), "messages_sha256": _digest(body["messages"]),
             "rendered_prompt_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
             "input_token_ids_sha256": _digest(input_ids), "generated_token_ids_sha256": _digest(output_ids),
             "raw_output": content, "raw_output_with_special_tokens": tokenizer.decode(output_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False),
             "input_tokens": input_count, "generated_output_tokens": len(output_ids), "generation_parameters": parameters,
             "synchronized_generation_latency_ms": inference_ms, "end_to_end_latency_ms": (time.perf_counter() - started) * 1000,
             "requested_response_format": body["response_format"], "json_grammar_enforced": False}
    return response, audit


def _emit(value: dict[str, Any]) -> None:
    print(json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False), flush=True)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=PROJECT.parent / "env/models/Qwen--Qwen2.5-1.5B-Instruct" / MODEL_REVISION)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--max-calls", type=int, default=100)
    args = parser.parse_args(argv)
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    faulthandler.enable(file=sys.stderr)
    stage, request_id = "startup", None
    try:
        _validate_limits(args.max_new_tokens, args.max_input_tokens, args.max_calls)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        with redirect_stdout(sys.stderr):
            runtime = _load(RUNTIME_PATH, "_local_judge_runtime", RUNTIME_SHA256)
            runtime._native_error_mode()
            environment = runtime._verify_environment()
            oracle = _load(ORACLE_PATH, "_local_judge_oracle", ORACLE_SHA256)
            identity = oracle._model_identity(args.model)
            scorer = oracle.QwenNextTokenScorer(args.model)
        _emit({"type": "ready", "model": identity, "runtime": environment, "device": scorer.device_name,
               "pid": os.getpid(), "limits": {"max_new_tokens": args.max_new_tokens, "max_input_tokens": args.max_input_tokens, "max_calls": args.max_calls},
               "json_grammar_enforced": False, "sources_sha256": {str(ORACLE_PATH): ORACLE_SHA256, str(RUNTIME_PATH): RUNTIME_SHA256,
                    str(Path(__file__).resolve()): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}})
        seen: set[int] = set()
        for line in sys.stdin:
            stage, request_id = "request_validation", None
            request = json.loads(line)
            if request == {"type": "shutdown"}:
                return 0
            if not isinstance(request, dict) or set(request) != {"request_id", "body"}:
                raise ValueError("invalid worker request")
            request_id = request["request_id"]
            if type(request_id) is not int or request_id < 0 or request_id in seen or len(seen) >= args.max_calls:
                request_id = None
                raise ValueError("invalid request identity or exhausted call budget")
            body = _validate_body(request["body"], args.max_new_tokens)
            seen.add(request_id)
            stage = "generation"
            with redirect_stdout(sys.stderr):
                response, audit = _generate(scorer, body, args.max_input_tokens)
            response["id"] = f"local-judge-{os.getpid()}-{request_id}"
            _emit({"type": "response", "request_id": request_id, "body": response, "audit": audit})
        return 0
    except Exception as exc:
        _emit({"type": "fatal", "request_id": request_id, "code": stage + "_error", "exception_type": type(exc).__name__})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
