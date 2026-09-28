from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "compat"
    / "run_scifact_hf_agent_ab.py"
)


def _load_runner():
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("scifact_hf_agent_ab", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def runner():
    return _load_runner()


def test_parser_accepts_plain_and_exact_json_fence(runner) -> None:
    raw = '{"evidence":{"10":{"label":"SUPPORT","sentences":[1,2]}}}'
    plain = runner.parse_agent_output(raw, {10: 3, 20: 1, 30: 1})
    fenced = runner.parse_agent_output(
        "```json\n" + raw + "\n```", {10: 3, 20: 1, 30: 1}
    )

    assert plain["valid"] is True
    assert plain["transport_unwrapped"] is False
    assert fenced["valid"] is True
    assert fenced["transport_unwrapped"] is True
    assert fenced["value"] == plain["value"]


@pytest.mark.parametrize(
    ("raw", "failure"),
    [
        ('{"evidence":{},"extra":1}', "invalid_root_schema"),
        (
            '{"evidence":{"99":{"label":"SUPPORT","sentences":[0]}}}',
            "document_out_of_scope",
        ),
        (
            '{"evidence":{"10":{"label":"SUPPORT","sentences":[1,1]}}}',
            "duplicate_sentence_index",
        ),
        (
            '{"evidence":{"10":{"label":"SUPPORT","sentences":[3]}}}',
            "sentence_out_of_scope",
        ),
        (
            '{"evidence":{"10":{"label":"SUPPORT","sentences":[]}}}',
            "invalid_sentences",
        ),
        ('{"evidence":{},"evidence":{}}', "invalid_json"),
        (" ```json\n{\"evidence\":{}}\n```", "invalid_json"),
    ],
)
def test_parser_rejects_repairs_and_scope_violations(runner, raw, failure) -> None:
    parsed = runner.parse_agent_output(raw, {10: 3, 20: 1, 30: 1})

    assert parsed["valid"] is False
    assert parsed["failure_code"] == failure
    assert parsed["value"] is None


def test_context_is_canonical_and_drops_gold_fields(runner) -> None:
    claim = {
        "id": 7,
        "claim": "A claim.",
        "evidence": {"10": [{"label": "SUPPORT", "sentences": [0]}]},
        "cited_doc_ids": [10],
    }
    corpus = {
        10: {"title": "T1", "abstract": ["S1", "S2"]},
        20: {"title": "T2", "abstract": ["S3"]},
        30: {"title": "T3", "abstract": ["S4"]},
    }

    context, canonical, digest = runner._build_context(
        claim, [10, 20, 30], corpus
    )

    assert set(context) == {"claim", "documents"}
    assert "evidence" not in canonical
    assert "cited_doc_ids" not in canonical
    assert digest == runner._sha256_text(canonical)
    assert [doc["rank"] for doc in context["documents"]] == [1, 2, 3]
    assert context["documents"][0]["sentences"][1] == {
        "index": 1,
        "text": "S2",
    }


def test_every_role_repeats_the_strict_output_contract(runner) -> None:
    forbidden_keys = "claim, documents, text, found_documents, evidence_schema"
    for system_prompt in (
        runner.INITIAL_SYSTEM,
        runner.SELF_REVIEW_SYSTEM,
        runner.CHALLENGER_SYSTEM,
    ):
        assert "First inspect every supplied sentence" in system_prompt
        assert "directly restates or entails" in system_prompt
        assert "incompatible number, negation, or direction" in system_prompt
        assert "Only after checking all supplied sentences" in system_prompt
        assert "Do not default to NEI" in system_prompt
    assert "exactly the keys analysis and candidates" in runner.INITIAL_SYSTEM
    assert "non-empty JSON string" in runner.INITIAL_SYSTEM
    assert "three-item array" in runner.INITIAL_SYSTEM
    assert "retrieval rank 1 through 3" in runner.INITIAL_SYSTEM
    for system_prompt in (
        runner.SELF_REVIEW_SYSTEM,
        runner.CHALLENGER_SYSTEM,
    ):
        assert "only root key is evidence" in system_prompt
        assert "always an object, never a list" in system_prompt
        assert forbidden_keys in system_prompt
        assert "a second JSON object" in system_prompt
    assert runner.INITIAL_DRAFT_PREFILL == '{"analysis":"'
    assert runner.FINAL_EVIDENCE_PREFILL == '{"evidence":{'
    assert runner.INITIAL_USER_TEMPLATE.endswith("root closing brace.")
    assert runner.REVIEW_USER_TEMPLATE.endswith("brace.")


@pytest.mark.parametrize(
    ("value", "closed"),
    [
        ('{"evidence":', False),
        ('{"evidence":{}}', True),
        ('{"evidence":{"1":{"label":"SUP}PORT"', False),
        ('{"evidence":{"1":{"sentences":[0,1]}}}', True),
        ('{"evidence":{}}\nExplanation', True),
    ],
)
def test_json_root_stopping_detector(runner, value, closed) -> None:
    assert runner._json_root_closed(value) is closed


def test_initial_draft_parser_and_projection_are_strict(runner) -> None:
    raw = (
        '{"analysis":"direct support","candidates":['
        '[2,"SUPPORT",[0]]]}'
    )
    parsed = runner.parse_initial_draft(raw, {10: 3, 20: 1, 30: 1})

    assert parsed["valid"] is True
    assert runner._draft_to_evidence(parsed["value"], [10, 20, 30]) == {
        "evidence": {
            "20": {"label": "SUPPORT", "sentences": [0]},
        }
    }
    empty = runner.parse_initial_draft(
        '{"analysis":"no relation","candidates":[]}',
        {10: 3, 20: 1, 30: 1},
    )
    assert empty["valid"] is True
    assert runner._draft_to_evidence(empty["value"], [10, 20, 30]) == {
        "evidence": {}
    }
    nei = runner.parse_initial_draft(
        '{"analysis":"no relation","candidates":[[1,"NEI",[0,2]]]}',
        {10: 3, 20: 1, 30: 1},
    )
    assert nei["valid"] is True
    assert runner._draft_to_evidence(nei["value"], [10, 20, 30]) == {
        "evidence": {}
    }


@pytest.mark.parametrize(
    ("raw", "failure"),
    [
        ('{"analysis":"x","candidates":{}}', "invalid_draft_candidates"),
        (
            '{"analysis":"x","candidates":['
            '[4,"SUPPORT",[0]]]}',
            "draft_rank_out_of_scope",
        ),
        (
            '{"analysis":"x","candidates":['
            '[1,"SUPPORT",[3]]]}',
            "draft_sentence_out_of_scope",
        ),
        (
            '{"analysis":"x","candidates":['
            '[1,"NEI",[0]],[2,"SUPPORT",[0]]]}',
            "mixed_nei_and_relation_candidates",
        ),
        (
            '{"analysis":"x","candidates":[[1,"NEI",[3]]]}',
            "draft_sentence_out_of_scope",
        ),
        (
            '{"analysis":"x","candidates":[[1,["SUPPORT"],[0]]]}',
            "invalid_draft_label",
        ),
    ],
)
def test_initial_draft_parser_rejects_invalid_candidates(
    runner, raw, failure
) -> None:
    parsed = runner.parse_initial_draft(raw, {10: 3, 20: 1, 30: 1})

    assert parsed["valid"] is False
    assert parsed["failure_code"] == failure


def test_journal_hash_chain_detects_tampering(runner) -> None:
    record = {"case_index": 0, "claim_id": 7}
    envelope = runner._journal_envelope(record, None)

    assert envelope["record_sha256"] == runner._sha256_text(
        runner._canonical_json(record)
    )
    envelope["record"]["claim_id"] = 8
    assert envelope["record_sha256"] != runner._sha256_text(
        runner._canonical_json(envelope["record"])
    )


def test_case_uses_three_calls_and_alternates_by_case_index(runner) -> None:
    class FakeGenerator:
        def __init__(self) -> None:
            self.kinds = []

        def generate(
            self,
            call_id,
            call_kind,
            messages,
            logical_attribution,
            assistant_prefill,
        ):
            self.kinds.append(call_kind)
            if call_kind == "shared_initial_judge":
                raw = (
                    '{"analysis":"support","candidates":['
                    '[1,"SUPPORT",[0]]]}'
                )
            else:
                raw = (
                    '{"evidence":{"10":'
                    '{"label":"SUPPORT","sentences":[0]}}}'
                )
            rendered_prompt = "rendered" + assistant_prefill
            generated_fragment = raw[len(assistant_prefill) :]
            return {
                "schema_version": "scifact-agent-physical-call-v1",
                "call_id": call_id,
                "call_kind": call_kind,
                "logical_attribution": logical_attribution,
                "device": "fake",
                "messages_sha256": runner._sha256_text(
                    runner._canonical_json(messages)
                ),
                "rendered_prompt": rendered_prompt,
                "rendered_prompt_sha256": runner._sha256_text(rendered_prompt),
                "prompt_sha256": runner._sha256_text(rendered_prompt),
                "prompt_character_count": len(rendered_prompt),
                "assistant_prefill": assistant_prefill,
                "assistant_prefill_sha256": runner._sha256_text(
                    assistant_prefill
                ),
                "assistant_prefill_in_input_tokens": True,
                "generated_fragment": generated_fragment,
                "generated_fragment_sha256": runner._sha256_text(
                    generated_fragment
                ),
                "raw_output": raw,
                "raw_output_sha256": runner._sha256_text(raw),
                "input_tokens": 100,
                "output_tokens": 10,
                "total_tokens": 110,
                "synchronized_latency_ms": 5.0,
                "end_to_end_latency_ms": 6.0,
                "generation_succeeded": True,
                "generation_error": None,
                "json_root_stopping_triggered": True,
                "stopping_policy": runner.JSON_ROOT_STOPPING_POLICY,
                "stopping_policy_sha256": runner._sha256_text(
                    runner.JSON_ROOT_STOPPING_POLICY
                ),
                "external_api_cost_usd": 0.0,
                "compute_cost_usd": None,
            }

    corpus = {
        10: {"title": "T1", "abstract": ["S1"]},
        20: {"title": "T2", "abstract": ["S2"]},
        30: {"title": "T3", "abstract": ["S3"]},
    }
    generator = FakeGenerator()
    record = runner._run_case(
        generator,
        case_index=0,
        claim={"id": 1, "claim": "A claim."},
        doc_ids=[10, 20, 30],
        corpus=corpus,
    )

    assert generator.kinds == [
        "shared_initial_judge",
        "self_review",
        "independent_challenger",
    ]
    assert record["case"]["physical_call_count"] == 3
    assert record["case"]["arms"]["single_self_review"]["logical_call_count"] == 2
    assert record["case"]["arms"]["judge_challenger"]["total_tokens"] == 220
    assert set(record["predictions"]) == set(runner.ARMS)

    envelope = runner._journal_envelope(record, None)
    assert (
        runner._validate_journal_record(
            envelope,
            expected_previous=None,
            expected_index=0,
            claim={"id": 1, "claim": "A claim."},
            doc_ids=[10, 20, 30],
            corpus=corpus,
        )
        == envelope["record_sha256"]
    )
