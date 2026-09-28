from __future__ import annotations

import copy
import importlib.util
import io
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/compat/scifact_local_judge_worker.py"


@pytest.fixture
def worker():
    spec = importlib.util.spec_from_file_location("local_judge_worker_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def body(max_tokens=4):
    return {
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": "Use the supplied JSON contract."},
            {"role": "user", "content": '原文 {"claim":"unaltered"}'},
        ],
    }


def test_body_validation_preserves_original_messages(worker):
    value = body()
    saved = copy.deepcopy(value)
    assert worker._validate_body(value, 512) is value
    assert value == saved


@pytest.mark.parametrize(
    "field,value",
    [
        ("model", "another-model"), ("temperature", True), ("temperature", 0.1),
        ("max_tokens", True), ("max_tokens", 0), ("max_tokens", 513),
        ("response_format", {"type": "json_schema"}),
        ("messages", []), ("messages", [{"role": [], "content": "x"}]),
        ("messages", [{"role": "user", "content": {"image": "unsupported"}}]),
    ],
)
def test_invalid_provider_request_is_rejected(worker, field, value):
    payload = body()
    payload[field] = value
    with pytest.raises(ValueError):
        worker._validate_body(payload, 512)


def test_unknown_fields_and_excess_requested_budget_are_rejected(worker):
    with pytest.raises(ValueError):
        worker._validate_body({**body(), "tools": []}, 512)
    with pytest.raises(ValueError):
        worker._validate_body(body(5), 4)


@pytest.mark.parametrize(
    "limits",
    [(True, 4096, 100), (512, True, 100), (512, 4096, True),
     (0, 4096, 100), (513, 4096, 100), (512, 4097, 100),
     (512, 4096, 0), (512, 4096, 101)],
)
def test_worker_limits_reject_bool_zero_and_excess(worker, limits):
    with pytest.raises(ValueError):
        worker._validate_limits(*limits)


class Tensor:
    def __init__(self, values):
        self.values = values

    def __getitem__(self, key):
        if isinstance(key, tuple):
            return Tensor(self.values[key[0]][key[1]])
        return Tensor(self.values[key])

    def tolist(self):
        return self.values

    def to(self, _device):
        return self

    def detach(self):
        return self

    def cpu(self):
        return self


class Tokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.template_options = kwargs
        return json.dumps(messages, ensure_ascii=False) + "<assistant>"

    def __call__(self, text, **kwargs):
        self.tokenization_options = kwargs
        return {"input_ids": Tensor([[11, 12, 13]])}

    def decode(self, ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        assert clean_up_tokenization_spaces is False
        return "".join("<eos>" if value == 2 else f"<{value}>"
                       for value in ids if not (skip_special_tokens and value == 2))


def scorer(output_ids):
    model = SimpleNamespace(
        generation_config=SimpleNamespace(eos_token_id=[2, 3], pad_token_id=0),
        calls=[],
    )

    def generate(**kwargs):
        model.calls.append(kwargs)
        return Tensor([[11, 12, 13, *output_ids]])

    model.generate = generate
    return SimpleNamespace(
        tokenizer=Tokenizer(), model=model, device="fake-device",
        device_name="fake-gpu",
        max_position_embeddings=8192,
        torch=SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None),
                              inference_mode=nullcontext),
    )


def test_generate_keeps_messages_and_counts_only_new_tokens(worker):
    fake = scorer([91, 2])
    payload = body()
    response, audit = worker._generate(fake, payload, 4096)
    assert fake.tokenizer.messages is payload["messages"]
    assert fake.tokenizer.template_options == {
        "tokenize": False, "add_generation_prompt": True,
    }
    assert fake.tokenizer.tokenization_options == {
        "return_tensors": "pt", "add_special_tokens": False, "truncation": False,
    }
    assert response["usage"] == {
        "prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5,
    }
    assert response["choices"][0]["finish_reason"] == "stop"
    assert response["choices"][0]["message"]["content"] == audit["raw_output"] == "<91>"
    assert audit["raw_output_with_special_tokens"] == "<91><eos>"
    assert audit["input_token_ids_sha256"] == worker._digest([11, 12, 13])
    assert audit["generated_token_ids_sha256"] == worker._digest([91, 2])
    assert audit["messages_sha256"] == worker._digest(payload["messages"])
    assert audit["json_grammar_enforced"] is False
    params = fake.model.calls[0]
    assert params["do_sample"] is False and params["num_beams"] == 1
    assert params["repetition_penalty"] == 1.0 and params["max_new_tokens"] == 4


def test_output_at_budget_without_eos_is_length(worker):
    response, audit = worker._generate(scorer([91, 92, 93, 94]), body(), 4096)
    assert response["choices"][0]["finish_reason"] == "length"
    assert audit["generated_output_tokens"] == 4


def test_short_output_without_eos_is_not_misreported_as_length(worker):
    with pytest.raises(ValueError, match="without EOS"):
        worker._generate(scorer([91]), body(), 4096)


def test_input_and_context_limits_fail_before_generate(worker):
    fake = scorer([91, 2])
    with pytest.raises(ValueError, match="no truncation"):
        worker._generate(fake, body(), 2)
    assert fake.model.calls == []
    fake.max_position_embeddings = 6
    with pytest.raises(ValueError, match="model context"):
        worker._generate(fake, body(), 4096)
    assert fake.model.calls == []


def test_main_protocol_enforces_max_calls_without_loading_model(worker, monkeypatch):
    runtime = SimpleNamespace(
        _native_error_mode=lambda: None, _verify_environment=lambda: {"stub": True},
    )
    oracle = SimpleNamespace(_model_identity=lambda _path: {"stub": True},
                             QwenNextTokenScorer=lambda _path: scorer([91, 2]))
    monkeypatch.setattr(
        worker, "_load", lambda path, *args:
        runtime if path == worker.RUNTIME_PATH else oracle,
    )
    monkeypatch.setattr(worker.faulthandler, "enable", lambda **kwargs: None)
    requests = [json.dumps({"request_id": i, "body": body()}) for i in (0, 1)]
    output = io.StringIO()
    monkeypatch.setattr(sys, "stdin", io.StringIO("\n".join(requests) + "\n"))
    monkeypatch.setattr(sys, "stdout", output)
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    assert worker.main(["--max-calls", "1"]) == 1
    lines = [json.loads(line) for line in output.getvalue().splitlines()]
    assert [line["type"] for line in lines] == ["ready", "response", "fatal"]
    assert lines[1]["request_id"] == 0
    assert lines[2]["code"] == "request_validation_error"
    assert "body" not in lines[2] and "message" not in lines[2]
