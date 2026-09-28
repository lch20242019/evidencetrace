"""SQLite cache keys and storage for source/model results."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

from evidencetrace.audit_models import (
    DEFAULT_MINER_MAX_WINDOW_CHARS,
    JUDGE_VERDICT_OWNERSHIP_POLICY_VERSION,
    MINER_COVERAGE_POLICY_VERSION,
    MINER_DRAFT_CONTRACT_VERSION,
    MINER_WINDOW_POLICY_VERSION,
)
from evidencetrace.checks.deterministic import (
    CHECKABILITY_POLICY_VERSION,
    DETERMINISTIC_SIGNAL_POLICY_VERSION,
)
from evidencetrace.eval.router import ADAPTIVE_ROUTER_POLICY_VERSION
from evidencetrace.model_client import (
    DEFAULT_MAX_TOKENS,
    MODEL_PROVIDER_CONTRACT_VERSION,
    SCHEMA_RECOVERY_POLICY_VERSION,
)
from evidencetrace.relation_policy import RELATION_DEFINITION_POLICY_VERSION
from evidencetrace.retrieval.rank import RETRIEVAL_POLICY_VERSION


def cache_key(
    *,
    source_hash: str,
    claim_hash: str,
    model_id: str,
    prompt_version: str,
    retrieval_config_version: str,
    deterministic_signal_policy_version: str = (DETERMINISTIC_SIGNAL_POLICY_VERSION),
    checkability_policy_version: str = CHECKABILITY_POLICY_VERSION,
    retrieval_policy_version: str = RETRIEVAL_POLICY_VERSION,
    router_policy_version: str = ADAPTIVE_ROUTER_POLICY_VERSION,
    relation_definition_policy_version: str = (RELATION_DEFINITION_POLICY_VERSION),
    miner_contract_version: str = MINER_DRAFT_CONTRACT_VERSION,
    miner_window_policy_version: str = MINER_WINDOW_POLICY_VERSION,
    miner_coverage_policy_version: str = MINER_COVERAGE_POLICY_VERSION,
    miner_window_limit: int = 30,
    miner_max_window_chars: int = DEFAULT_MINER_MAX_WINDOW_CHARS,
    schema_recovery_policy_version: str = SCHEMA_RECOVERY_POLICY_VERSION,
    model_max_tokens: int = DEFAULT_MAX_TOKENS,
    model_provider_contract_version: str = MODEL_PROVIDER_CONTRACT_VERSION,
    judge_verdict_ownership_policy_version: str = (
        JUDGE_VERDICT_OWNERSHIP_POLICY_VERSION
    ),
) -> str:
    value = {
        "source_hash": source_hash,
        "claim_hash": claim_hash,
        "model_id": model_id,
        "prompt_version": prompt_version,
        "retrieval_config_version": retrieval_config_version,
        "deterministic_signal_policy_version": (deterministic_signal_policy_version),
        "checkability_policy_version": checkability_policy_version,
        "retrieval_policy_version": retrieval_policy_version,
        "router_policy_version": router_policy_version,
        "relation_definition_policy_version": (relation_definition_policy_version),
        "miner_contract_version": miner_contract_version,
        "miner_window_policy_version": miner_window_policy_version,
        "miner_coverage_policy_version": miner_coverage_policy_version,
        "miner_window_limit": miner_window_limit,
        "miner_max_window_chars": miner_max_window_chars,
        "schema_recovery_policy_version": schema_recovery_policy_version,
        "model_max_tokens": model_max_tokens,
        "model_provider_contract_version": model_provider_contract_version,
        "judge_verdict_ownership_policy_version": (
            judge_verdict_ownership_policy_version
        ),
    }
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class CacheStore:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS model_cache "
            "(cache_key TEXT PRIMARY KEY, payload TEXT NOT NULL)"
        )
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS source_cache "
            "(url TEXT PRIMARY KEY, payload TEXT NOT NULL)"
        )
        self.db.commit()

    def get_model(self, key: str) -> dict[str, Any] | None:
        row = self.db.execute(
            "SELECT payload FROM model_cache WHERE cache_key=?", (key,)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def put_model(self, key: str, payload: dict[str, Any]) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO model_cache VALUES (?,?)",
            (key, json.dumps(payload, sort_keys=True)),
        )
        self.db.commit()

    def get_source(self, url: str) -> dict[str, Any] | None:
        row = self.db.execute(
            "SELECT payload FROM source_cache WHERE url=?", (url,)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def put_source(self, url: str, payload: dict[str, Any]) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO source_cache VALUES (?,?)",
            (url, json.dumps(payload, sort_keys=True)),
        )
        self.db.commit()

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> CacheStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


__all__ = ["CacheStore", "cache_key"]
