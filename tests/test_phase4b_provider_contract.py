from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

from evidencetrace.eval.full_document import (
    FULL_DOCUMENT_MAX_TOKENS,
    FullDocumentBenchmarkError,
    FullDocumentFailureArtifact,
    FullDocumentGoldClaim,
    FullDocumentMetadata,
    LoadedFullDocumentDataset,
    SingleAgentDocumentOutput,
    load_full_document_dataset,
    run_full_document_benchmark,
    safe_full_document_failure,
)
from evidencetrace.eval.models import SourceFixture
from evidencetrace.model_client import (
    ModelSchemaError,
    OpenAICompatibleClient,
)
from evidencetrace.models import Checkability, Relation

ROOT = Path(__file__).parents[1]
PACK = ROOT / "eval_sets/full_document_v1"
HISTORICAL_FAILURE = (
    ROOT / "eval_runs" / "phase4b_full_document_live_dev" / "failed_attempt.json"
)
API_SECRET = "phase4b-provider-contract-api-secret"
RAW_SECRET = "phase4b-provider-contract-raw-secret"
SOURCE_URL = "https://synthetic.example.test/widget"
SOURCE_TEXT = "Widget ships v2."
DOCUMENT_TEXT = f"# Synthetic\n\n{SOURCE_TEXT} [spec]({SOURCE_URL})\n"


def _semantic_claim(
    *, reason: str = "The source states the version."
) -> dict[str, Any]:
    return {
        "text": SOURCE_TEXT,
        "claim_type": "versioned_capability",
        "checkability": "checkable",
        "citation_urls": [SOURCE_URL],
        "relation": "entailed",
        "confidence": 0.91,
        "reason": reason,
        "evidence_span": SOURCE_TEXT,
    }


def _body(
    content: object,
    *,
    finish_reason: object = "stop",
    usage: object | None = None,
) -> dict[str, Any]:
    return {
        "choices": [
            {
                "message": {
                    "content": content,
                    "reasoning_content": RAW_SECRET,
                },
                "finish_reason": finish_reason,
            }
        ],
        "usage": (
            usage
            if usage is not None
            else {
                "prompt_tokens": 101,
                "completion_tokens": 37,
                "total_tokens": 138,
                "completion_tokens_details": {"reasoning_tokens": 11},
            }
        ),
    }


def _client(
    body: dict[str, Any],
    *,
    requests: list[dict[str, Any]] | None = None,
    max_tokens: int = FULL_DOCUMENT_MAX_TOKENS,
) -> OpenAICompatibleClient:
    def handler(request: httpx.Request) -> httpx.Response:
        if requests is not None:
            requests.append(json.loads(request.content))
        return httpx.Response(200, request=request, json=body)

    return OpenAICompatibleClient(
        model_id="deepseek-v4-flash",
        api_key=API_SECRET,
        base_url="https://provider.example.test/v1",
        temperature=0.0,
        max_tokens=max_tokens,
        transport=httpx.MockTransport(handler),
    )


def _dataset(tmp_path: Path) -> LoadedFullDocumentDataset:
    start = DOCUMENT_TEXT.index(SOURCE_TEXT)
    end = start + len(SOURCE_TEXT)
    source = SourceFixture(
        source_id="synthetic_widget_source",
        url=SOURCE_URL,
        content=SOURCE_TEXT,
        provenance="fully synthetic provider contract fixture",
        content_hash=hashlib.sha256(SOURCE_TEXT.encode()).hexdigest(),
    )
    metadata = FullDocumentMetadata(
        document_id="synthetic_document",
        path="docs/synthetic.md",
        split="dev",
        stratum="short",
        content_sha256=hashlib.sha256(DOCUMENT_TEXT.encode()).hexdigest(),
        gold_claim_count=1,
    )
    gold = FullDocumentGoldClaim(
        annotation_status="author_constructed_deterministic_provisional",
        document_id=metadata.document_id,
        document_path=metadata.path,
        claim_id="synthetic_gold_001",
        text=SOURCE_TEXT,
        offset_start=start,
        offset_end=end,
        line_start=3,
        line_end=3,
        claim_type="versioned_capability",
        checkability=Checkability.CHECKABLE,
        citation_urls=(SOURCE_URL,),
        source_ids=(source.source_id,),
        gold_relation=Relation.ENTAILED,
        gold_evidence_span=SOURCE_TEXT,
        gold_evidence_source_id=source.source_id,
        construction_note="Synthetic contract fixture.",
    )
    return LoadedFullDocumentDataset(
        root=tmp_path,
        documents=(metadata,),
        gold_claims=(gold,),
        sources={source.source_id: source},
        sources_by_url={source.url: source},
        document_text={metadata.document_id: DOCUMENT_TEXT},
        dataset_sha256="a" * 64,
    )


def test_document_contract_valid_response_uses_8192_and_safe_provenance(
    tmp_path: Path,
) -> None:
    requests: list[dict[str, Any]] = []
    client = _client(
        _body(json.dumps({"claims": [_semantic_claim()]})),
        requests=requests,
    )

    artifact = run_full_document_benchmark(
        _dataset(tmp_path),
        "single_agent_document_live",
        client,
        run_id="synthetic-provider-contract",
        started_at=datetime(2035, 1, 2, tzinfo=UTC),
        max_calls=2,
        max_total_tokens=10_000,
    )

    assert requests[0]["max_tokens"] == 8192
    assert requests[0]["response_format"] == {"type": "json_object"}
    assert "thinking" not in requests[0]
    assert "reasoning_effort" not in requests[0]
    assert artifact.request_provenance.max_tokens_requested == 8192
    assert artifact.request_provenance.thinking_mode == "provider_default"
    assert artifact.request_provenance.temperature_effective is None
    assert artifact.request_provenance.output_schema_names == (
        "SingleAgentDocumentOutput",
    )
    stage = artifact.stage_telemetry["single_agent"]
    assert stage.calls == 1
    assert stage.finish_reasons == ("stop",)
    assert stage.max_tokens_requested == (8192,)
    assert stage.reasoning_tokens == 11
    assert stage.suspected_token_truncations == 0
    assert artifact.schema_diagnostics == ()
    assert RAW_SECRET not in artifact.model_dump_json()
    assert API_SECRET not in artifact.model_dump_json()


def test_miner_and_judge_share_hardened_contract_and_separate_telemetry(
    tmp_path: Path,
) -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        request_body = json.loads(request.content)
        requests.append(request_body)
        user = json.loads(request_body["messages"][1]["content"])
        task = user["task"]
        if task == "claim_mining":
            payload = {
                "claims": [
                    {
                        "text": SOURCE_TEXT,
                        "claim_type": "versioned_capability",
                        "checkability": "checkable",
                    }
                ]
            }
        elif task in {
            "claim_judgement",
            "single_agent_full_source_verification",
        }:
            payload = {
                key: value
                for key, value in _semantic_claim().items()
                if key in {"relation", "confidence", "reason", "evidence_span"}
            }
        else:
            raise AssertionError("unexpected synthetic task")
        return httpx.Response(
            200,
            request=request,
            json=_body(json.dumps(payload)),
        )

    client = OpenAICompatibleClient(
        model_id="deepseek-v4-flash",
        api_key=API_SECRET,
        base_url="https://provider.example.test/v1",
        temperature=0.0,
        max_tokens=FULL_DOCUMENT_MAX_TOKENS,
        transport=httpx.MockTransport(handler),
    )
    artifact = run_full_document_benchmark(
        _dataset(tmp_path),
        "miner_adaptive_judge_live",
        client,
        run_id="synthetic-multi-provider-contract",
        started_at=datetime(2035, 1, 2, tzinfo=UTC),
        max_calls=10,
        max_total_tokens=10_000,
    )

    tasks = [json.loads(item["messages"][1]["content"])["task"] for item in requests]
    assert tasks[0] == "claim_mining"
    assert set(tasks[1:]) <= {
        "claim_judgement",
        "single_agent_full_source_verification",
    }
    assert all(item["max_tokens"] == 8192 for item in requests)
    assert artifact.stage_telemetry["miner"].calls == 1
    assert artifact.stage_telemetry["judge"].calls == 1
    assert artifact.stage_telemetry["single_agent"].calls == 0
    assert artifact.request_provenance.output_schema_names == (
        "LiveMinerDraftOutput",
        "LiveJudgeSemanticOutput",
    )
    judge_request = json.loads(requests[-1]["messages"][1]["content"])
    judge_schema = judge_request["output_contract"]["output_schema"]
    assert judge_schema["properties"]["reason"]["maxLength"] == 240
    assert set(judge_schema["required"]) == set(judge_schema["properties"])
    assert judge_schema["additionalProperties"] is False


def test_document_contract_rejects_default_output_budget_before_request(
    tmp_path: Path,
) -> None:
    requests: list[dict[str, Any]] = []
    client = _client(
        _body(json.dumps({"claims": [_semantic_claim()]})),
        requests=requests,
        max_tokens=2048,
    )

    with pytest.raises(FullDocumentBenchmarkError) as raised:
        run_full_document_benchmark(
            _dataset(tmp_path),
            "single_agent_document_live",
            client,
            run_id="wrong-output-budget",
            started_at=datetime(2035, 1, 2, tzinfo=UTC),
        )

    assert raised.value.stage == "provider_contract"
    assert raised.value.code == "provider_output_budget_mismatch"
    assert requests == []


def test_finish_reason_length_has_bounded_truncation_diagnostic() -> None:
    client = _client(
        _body(
            json.dumps({"claims": [_semantic_claim()]}),
            finish_reason="length",
            usage={
                "prompt_tokens": 400,
                "completion_tokens": 8192,
                "total_tokens": 8592,
                "completion_tokens_details": {"reasoning_tokens": 2048},
            },
        )
    )

    with pytest.raises(ModelSchemaError) as raised:
        client.complete_model(
            "single_agent_document_verification", {}, SingleAgentDocumentOutput
        )

    diagnostic = raised.value.diagnostic
    assert diagnostic is not None
    assert diagnostic.contract_stage == "content_contract"
    assert diagnostic.failure_category == "finish_reason"
    assert diagnostic.finish_reason == "length"
    assert diagnostic.max_tokens_requested == 8192
    assert diagnostic.input_tokens == 400
    assert diagnostic.output_tokens == 8192
    assert diagnostic.reasoning_tokens == 2048
    assert diagnostic.response_content_present is True
    assert diagnostic.suspected_token_truncation is True


def test_truncated_json_is_json_decode_without_payload_leak() -> None:
    content = '{"claims":[{"text":"' + RAW_SECRET
    client = _client(_body(content))

    with pytest.raises(ModelSchemaError) as raised:
        client.complete_model(
            "single_agent_document_verification", {}, SingleAgentDocumentOutput
        )

    diagnostic = raised.value.diagnostic
    assert diagnostic is not None
    assert diagnostic.contract_stage == "json_decode"
    assert diagnostic.failure_category == "invalid_json"
    assert diagnostic.response_content_present is True
    assert diagnostic.response_content_length == len(content)
    assert diagnostic.suspected_token_truncation is False
    encoded = diagnostic.model_dump_json() + repr(raised.value)
    assert RAW_SECRET not in encoded
    assert API_SECRET not in encoded


@pytest.mark.parametrize(
    ("claim", "category", "path", "error_type"),
    [
        (
            {
                key: value
                for key, value in _semantic_claim().items()
                if key != "relation"
            },
            "missing_field",
            ("claims", 0, "relation"),
            "missing",
        ),
        (
            {**_semantic_claim(), "provider_private_field": RAW_SECRET},
            "extra_field",
            ("claims", 0, "unknown_field"),
            "extra_forbidden",
        ),
        (
            {**_semantic_claim(), "relation": "supported"},
            "enum",
            ("claims", 0, "relation"),
            "enum",
        ),
        (
            _semantic_claim(reason="x" * 241 + RAW_SECRET),
            "schema_constraint",
            ("claims", 0, "reason"),
            "string_too_long",
        ),
    ],
    ids=["missing", "extra", "enum", "long-reason"],
)
def test_document_schema_validation_diagnostics_are_bounded(
    claim: dict[str, Any],
    category: str,
    path: tuple[str | int, ...],
    error_type: str,
) -> None:
    client = _client(_body(json.dumps({"claims": [claim]})))

    with pytest.raises(ModelSchemaError) as raised:
        client.complete_model(
            "single_agent_document_verification", {}, SingleAgentDocumentOutput
        )

    diagnostic = raised.value.diagnostic
    assert diagnostic is not None
    assert diagnostic.contract_stage == "schema_validation"
    assert diagnostic.failure_category == category
    assert diagnostic.issues[0].location == path
    assert diagnostic.issues[0].error_type == error_type
    assert diagnostic.max_tokens_requested == 8192
    encoded = diagnostic.model_dump_json() + repr(raised.value)
    assert RAW_SECRET not in encoded
    assert API_SECRET not in encoded


def test_full_document_schema_failure_artifact_retains_only_safe_diagnostic(
    tmp_path: Path,
) -> None:
    client = _client(
        _body(
            json.dumps({"claims": [{**_semantic_claim(), "reason": RAW_SECRET * 40}]})
        )
    )

    with pytest.raises(FullDocumentBenchmarkError) as raised:
        run_full_document_benchmark(
            _dataset(tmp_path),
            "single_agent_document_live",
            client,
            run_id="safe-document-failure",
            started_at=datetime(2035, 1, 2, tzinfo=UTC),
            max_calls=2,
        )

    failure = safe_full_document_failure(
        "single_agent_document_live",
        raised.value,
    )
    assert failure.artifact_version == "full-document-failure-v2"
    assert failure.failure_code == "model_schema_invalid"
    assert failure.schema_diagnostic is not None
    assert failure.schema_diagnostic.contract_stage == "schema_validation"
    encoded = failure.model_dump_json(exclude_none=True)
    assert RAW_SECRET not in encoded
    assert API_SECRET not in encoded
    assert "reasoning_content" not in encoded


def test_dev_loader_uses_prefix_path_and_never_calls_full_jsonl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import evidencetrace.eval.full_document as full_document

    def forbidden_full_loader(*_args: Any, **_kwargs: Any) -> tuple[Any, ...]:
        raise AssertionError("dev loader must not parse the full JSONL file")

    monkeypatch.setattr(full_document, "_jsonl", forbidden_full_loader)
    dataset = load_full_document_dataset(PACK, split="dev")

    assert len(dataset.documents) == 12
    assert len(dataset.gold_claims) == 60
    assert len(dataset.sources) == 36
    assert all(
        item.split == "dev" and "/test/" not in item.path for item in dataset.documents
    )


def test_historical_full_document_failure_remains_byte_identical_and_readable() -> None:
    raw = HISTORICAL_FAILURE.read_bytes()

    assert hashlib.sha256(raw).hexdigest() == (
        "e175b57583f621432cb6a96a5624bf294e71e841c1dd481c58feb514da3e2241"
    )
    artifact = FullDocumentFailureArtifact.model_validate_json(raw)
    assert artifact.artifact_version == "full-document-failure-v1"
    assert artifact.schema_diagnostic is None
