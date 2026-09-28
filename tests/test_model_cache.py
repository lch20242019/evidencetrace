from __future__ import annotations

import hashlib
import json
from pathlib import Path

import httpx
import pytest
from pydantic import BaseModel

from evidencetrace.audit_models import (
    DEFAULT_MINER_MAX_WINDOW_CHARS,
    JUDGE_VERDICT_OWNERSHIP_POLICY_VERSION,
    MINER_COVERAGE_POLICY_VERSION,
    MINER_DRAFT_CONTRACT_VERSION,
    MINER_WINDOW_POLICY_VERSION,
)
from evidencetrace.cache import CacheStore, cache_key
from evidencetrace.checks.deterministic import (
    DETERMINISTIC_SIGNAL_POLICY_VERSION,
)
from evidencetrace.model_client import (
    DEFAULT_MAX_TOKENS,
    MODEL_PROVIDER_CONTRACT_VERSION,
    SCHEMA_RECOVERY_POLICY_VERSION,
    DeterministicFakeModel,
    ModelConfigError,
    ModelResponseError,
    OpenAICompatibleClient,
)


class Output(BaseModel):
    value: int


def test_cache_key_invalidates_every_required_dimension() -> None:
    base = {
        "source_hash": "source-a",
        "claim_hash": "claim-a",
        "model_id": "model-a",
        "prompt_version": "prompt-a",
        "retrieval_config_version": "retrieval-a",
        "deterministic_signal_policy_version": (DETERMINISTIC_SIGNAL_POLICY_VERSION),
        "miner_contract_version": MINER_DRAFT_CONTRACT_VERSION,
        "miner_window_policy_version": MINER_WINDOW_POLICY_VERSION,
        "miner_coverage_policy_version": MINER_COVERAGE_POLICY_VERSION,
        "miner_window_limit": 30,
        "miner_max_window_chars": DEFAULT_MINER_MAX_WINDOW_CHARS,
        "judge_verdict_ownership_policy_version": (
            JUDGE_VERDICT_OWNERSHIP_POLICY_VERSION
        ),
        "model_max_tokens": DEFAULT_MAX_TOKENS,
        "model_provider_contract_version": MODEL_PROVIDER_CONTRACT_VERSION,
    }
    keys = {cache_key(**base)}
    for field in base:
        changed = dict(base)
        changed[field] = (
            int(changed[field]) + 1
            if field
            in {"model_max_tokens", "miner_window_limit", "miner_max_window_chars"}
            else str(changed[field]) + "-changed"
        )
        keys.add(cache_key(**changed))

    assert len(keys) == 15


def test_signal_policy_version_invalidates_legacy_cache_key(
    tmp_path: Path,
) -> None:
    legacy_value = {
        "source_hash": "source-a",
        "claim_hash": "claim-a",
        "model_id": "model-a",
        "prompt_version": "prompt-a",
        "retrieval_config_version": "retrieval-a",
    }
    legacy_key = hashlib.sha256(
        json.dumps(
            legacy_value,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    current_key = cache_key(**legacy_value)
    prior_policy_key = cache_key(
        **legacy_value,
        deterministic_signal_policy_version="deterministic-signals-v1",
    )
    prior_miner_key = cache_key(
        **legacy_value,
        miner_contract_version="live-miner-draft-v2",
    )
    prior_recovery_key = cache_key(
        **legacy_value,
        schema_recovery_policy_version="schema-recovery-disabled-v0",
    )
    prior_ownership_key = cache_key(
        **legacy_value,
        judge_verdict_ownership_policy_version="model-owned-verdict-v0",
    )
    prior_window_key = cache_key(
        **legacy_value,
        miner_window_policy_version="paragraph-miner-policy-v0",
    )
    prior_coverage_key = cache_key(
        **legacy_value,
        miner_coverage_policy_version="miner-coverage-policy-v1",
    )
    prior_window_limit_key = cache_key(
        **legacy_value,
        miner_window_limit=29,
    )
    prior_window_bound_key = cache_key(
        **legacy_value,
        miner_max_window_chars=DEFAULT_MINER_MAX_WINDOW_CHARS + 1,
    )

    assert current_key != legacy_key
    assert current_key != prior_policy_key
    assert current_key != prior_miner_key
    assert current_key != prior_recovery_key
    assert current_key != prior_ownership_key
    assert current_key != prior_window_key
    assert current_key != prior_coverage_key
    assert current_key != prior_window_limit_key
    assert current_key != prior_window_bound_key
    assert MINER_DRAFT_CONTRACT_VERSION == "live-miner-draft-v3"
    assert MINER_WINDOW_POLICY_VERSION == "miner-window-policy-v1"
    assert MINER_COVERAGE_POLICY_VERSION == "miner-coverage-policy-v2"
    assert SCHEMA_RECOVERY_POLICY_VERSION == "schema-recovery-v1"
    with CacheStore(tmp_path / "cache.sqlite3") as cache:
        cache.put_model(legacy_key, {"relation": "entailed"})
        assert cache.get_model(current_key) is None


def test_sqlite_cache_round_trip(tmp_path: Path) -> None:
    with CacheStore(tmp_path / "cache.sqlite3") as cache:
        cache.put_model("key", {"value": 3})
        cache.put_source("https://example.test", {"hash": "abc"})

        assert cache.get_model("key") == {"value": 3}
        assert cache.get_source("https://example.test") == {"hash": "abc"}
        assert cache.get_model("missing") is None


def test_openai_adapter_requires_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = OpenAICompatibleClient(model_id="test-model", api_key=None)

    with pytest.raises(ModelConfigError, match="OPENAI_API_KEY"):
        client.complete("task", {})


def test_openai_adapter_validates_schema_and_keeps_key_out_of_result() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer super-secret"
        body = {"choices": [{"message": {"content": json.dumps({"value": 7})}}]}
        return httpx.Response(200, request=request, json=body)

    client = OpenAICompatibleClient(
        model_id="test-model",
        api_key="super-secret",
        base_url="https://model.example/v1",
        transport=httpx.MockTransport(handler),
    )

    assert client.complete_model("task", {"text": "x"}, Output) == Output(value=7)
    assert "super-secret" not in repr(client.complete("task", {}))


def test_invalid_model_schema_is_rejected() -> None:
    model = DeterministicFakeModel(lambda _task, _payload: {"value": "not-int"})

    with pytest.raises(ModelResponseError, match="schema"):
        model.complete_model("task", {}, Output)
