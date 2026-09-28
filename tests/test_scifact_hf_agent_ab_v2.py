from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "compat"
    / "run_scifact_hf_agent_ab_v2.py"
)


def _load_runner():
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("scifact_hf_agent_ab_v2", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def runner():
    return _load_runner()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            '{"label":"SUPPORT","citations":[[2,[0]]]}',
            {"20": {"label": "SUPPORT", "sentences": [0]}},
        ),
        (
            '{"label":"CONTRADICT","citations":[[1,[0,2]]]}',
            {"10": {"label": "CONTRADICT", "sentences": [0, 2]}},
        ),
        ('{"label":"NEI","citations":[]}', {}),
    ],
)
def test_explicit_decision_parser_projects_official_evidence(
    runner, raw, expected
) -> None:
    parsed = runner.parse_agent_output(raw, {10: 3, 20: 1, 30: 1})

    assert parsed["valid"] is True
    assert parsed["projected_evidence"] == expected
    assert set(parsed["value"]) == {"label", "citations"}


@pytest.mark.parametrize(
    ("raw", "failure"),
    [
        ('{"label":"SUPPORT","citations":[]}', "relation_without_citations"),
        (
            '{"label":"NEI","citations":[[1,[0]]]}',
            "nei_with_citations",
        ),
        ('{"label":["SUPPORT"],"citations":[]}', "invalid_label"),
        (
            '{"label":"SUPPORT|CONTRADICT|NEI","citations":[]}',
            "invalid_label",
        ),
        (
            '{"label":"SUPPORT","citations":[[4,[0]]]}',
            "citation_rank_out_of_scope",
        ),
        (
            '{"label":"SUPPORT","citations":[[true,[0]]]}',
            "citation_rank_out_of_scope",
        ),
        (
            '{"label":"SUPPORT","citations":[[1,[0]],[1,[1]]]}',
            "duplicate_citation_rank",
        ),
        (
            '{"label":"SUPPORT","citations":[[1,[]]]}',
            "invalid_citation_sentences",
        ),
        (
            '{"label":"SUPPORT","citations":[[1,[3]]]}',
            "citation_sentence_out_of_scope",
        ),
        (
            '{"label":"SUPPORT","citations":[[1,[0,0]]]}',
            "duplicate_citation_sentence_index",
        ),
        (
            '{"citations":[[1,[0]]],"label":"SUPPORT"}',
            None,
        ),
        (
            '{"label":"SUPPORT","citations":[{"rank":1,"sentences":[0]}]}',
            "invalid_citation_schema",
        ),
    ],
)
def test_explicit_decision_parser_rejects_degenerate_or_unsafe_outputs(
    runner, raw, failure
) -> None:
    parsed = runner.parse_agent_output(raw, {10: 3, 20: 1, 30: 1})

    assert parsed["valid"] is (failure is None)
    assert parsed["failure_code"] == failure
    if failure is not None:
        assert parsed["projected_evidence"] is None


def test_initial_parser_uses_distinct_failure_codes(runner) -> None:
    parsed = runner.parse_initial_draft(
        '{"label":"CONTRADICT","citations":[]}',
        {10: 3, 20: 1, 30: 1},
    )

    assert parsed["valid"] is False
    assert parsed["failure_code"] == "draft_relation_without_citations"


@pytest.mark.parametrize(
    ("raw", "closed"),
    [
        ('{"a":[1]}', True),
        ('{"a":[1}}', False),
        ('{"a":{"b":1]}', False),
        ('{"a":"}"}', True),
        ('{"a":[1]', False),
        ('[1,2,3]', False),
    ],
)
def test_json_root_stopper_requires_matching_container_types(
    runner, raw, closed
) -> None:
    assert runner._json_root_closed(raw) is closed


def test_protocol_avoids_the_empty_object_prefill_and_pipe_placeholder(
    runner,
) -> None:
    assert runner.INITIAL_DRAFT_PREFILL == '{"label":"'
    assert runner.FINAL_EVIDENCE_PREFILL == '{"label":"'
    assert '{"evidence":{' not in runner.FINAL_EVIDENCE_PREFILL
    assert runner.JSON_ROOT_STOPPING_POLICY.startswith("during_generation")
    assert runner.DRAFT_PARSER_POLICY["root_exact_keys"] == [
        "label",
        "citations",
    ]
    assert runner.DRAFT_PARSER_POLICY["version"].endswith("v2")
    assert runner.FINAL_PARSER_POLICY["version"].endswith("v2")


def test_configuration_explicitly_locks_effective_generation_penalty(runner) -> None:
    lock = runner._configuration_lock(Path("ignored"), {}, 256, 7)

    assert lock["agent_protocol_version"] == "scifact-agent-decision-v2"
    assert lock["generation"]["repetition_penalty"] == 1.0
    assert lock["generation"]["repetition_penalty_source"].startswith(
        "explicit_protocol_override"
    )
    dependency = lock["implementation_dependencies"]["audited_runner_base"]
    assert Path(dependency["path"]).is_file()
    assert len(dependency["sha256"]) == 64


def test_run_case_projects_all_three_decisions(runner) -> None:
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
                raw = '{"label":"SUPPORT","citations":[[1,[0]]]}'
            elif call_kind == "self_review":
                raw = '{"label":"CONTRADICT","citations":[[2,[0]]]}'
            else:
                raw = '{"label":"NEI","citations":[]}'
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
    record = runner._base._run_case(
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
    assert record["predictions"]["single_one_pass"]["evidence"] == {
        "10": {"label": "SUPPORT", "sentences": [0]}
    }
    assert record["predictions"]["single_self_review"]["evidence"] == {
        "20": {"label": "CONTRADICT", "sentences": [0]}
    }
    assert record["predictions"]["judge_challenger"]["evidence"] == {}

    envelope = runner._base._journal_envelope(record, None)
    assert (
        runner._base._validate_journal_record(
            envelope,
            expected_previous=None,
            expected_index=0,
            claim={"id": 1, "claim": "A claim."},
            doc_ids=[10, 20, 30],
            corpus=corpus,
        )
        == envelope["record_sha256"]
    )

def test_smoke_only_freeze_path_is_disabled(runner, tmp_path: Path) -> None:
    with pytest.raises(
        runner.AgentABError, match=r"quality-gated|full train scores"
    ):
        runner.write_frozen_config(None, tmp_path / "frozen.json", {}, {})


def _write_quality_gated_freeze(runner, tmp_path: Path) -> tuple[Path, dict]:
    train_manifest = tmp_path / "train" / "run_manifest.json"
    train_manifest.parent.mkdir(parents=True)
    train_manifest.write_text('{"outputs_sha256":{}}\n', encoding="utf-8")
    train_sha = hashlib.sha256(train_manifest.read_bytes()).hexdigest()
    gate_script = Path(__file__).parents[1] / "scripts" / (
        "freeze_scifact_agent_ab_config.py"
    )
    report = {
        "schema_version": "scifact-agent-ab-train-admission-v1",
        "status": "passed",
        "scope": "complete_official_train_only",
        "agent_protocol_version": runner.AGENT_PROTOCOL_VERSION,
        "dev_labels_or_metrics_inspected": False,
        "implementation": {
            "path": "scripts/freeze_scifact_agent_ab_config.py",
            "sha256": hashlib.sha256(gate_script.read_bytes()).hexdigest(),
        },
        "thresholds": dict(runner.EXPECTED_TRAIN_GATE_THRESHOLDS),
        "observed": {
            arm: {
                "format_valid_count": 809,
                "nonempty_prediction_count": 405,
                "label_counts": {
                    "SUPPORT": 202,
                    "CONTRADICT": 203,
                    "NEI": 404,
                },
                "claim_count": 809,
                "format_valid_rate": 1.0,
                "nonempty_prediction_rate": 405 / 809,
                "distinct_label_count": 3,
            }
            for arm in runner.ARMS
        },
        "evidence_sha256": {"train_manifest": train_sha},
    }
    report_path = tmp_path / "gate.json"
    report_path.write_text(json.dumps(report) + "\n", encoding="utf-8")
    gate = {
        "schema_version": "scifact-agent-ab-train-admission-v1",
        "status": "passed",
        "report": {
            "path": str(report_path),
            "sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        },
    }
    configuration = {"agent_protocol_version": runner.AGENT_PROTOCOL_VERSION}
    inputs = {"split": "dev", "opaque": "hash-bound"}
    frozen = {
        "schema_version": "scifact-agent-ab-frozen-config-v2",
        "locked_configuration": configuration,
        "dev_inputs": inputs,
        "train_run_manifest": {
            "path": str(train_manifest),
            "sha256": train_sha,
            "selected_claim_count": 809,
        },
        "train_admission_gate": gate,
        "train_admission_gate_sha256": runner._sha256_text(
            runner._canonical_json(gate)
        ),
    }
    frozen_path = tmp_path / "frozen.json"
    frozen_path.write_text(json.dumps(frozen) + "\n", encoding="utf-8")
    return frozen_path, {"configuration": configuration, "inputs": inputs}


def test_frozen_v2_validator_requires_bound_train_quality_gate(
    runner, tmp_path: Path
) -> None:
    frozen_path, expected = _write_quality_gated_freeze(runner, tmp_path)

    value = runner._validate_frozen_config(
        frozen_path, expected["configuration"], expected["inputs"]
    )

    assert value["train_admission_gate"]["status"] == "passed"


def test_frozen_v2_validator_rejects_tampered_gate_binding(
    runner, tmp_path: Path
) -> None:
    frozen_path, expected = _write_quality_gated_freeze(runner, tmp_path)
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    frozen["train_admission_gate_sha256"] = "0" * 64
    frozen_path.write_text(json.dumps(frozen) + "\n", encoding="utf-8")

    with pytest.raises(runner.AgentABError, match="binding hash"):
        runner._validate_frozen_config(
            frozen_path, expected["configuration"], expected["inputs"]
        )


def _rewrite_bound_gate(frozen_path: Path, mutate) -> None:
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    report_path = Path(frozen["train_admission_gate"]["report"]["path"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    mutate(report)
    report_path.write_text(json.dumps(report) + "\n", encoding="utf-8")
    frozen["train_admission_gate"]["report"]["sha256"] = hashlib.sha256(
        report_path.read_bytes()
    ).hexdigest()
    frozen["train_admission_gate_sha256"] = hashlib.sha256(
        json.dumps(
            frozen["train_admission_gate"],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    frozen_path.write_text(json.dumps(frozen) + "\n", encoding="utf-8")


def test_frozen_v2_validator_rejects_changed_gate_thresholds(
    runner, tmp_path: Path
) -> None:
    frozen_path, expected = _write_quality_gated_freeze(runner, tmp_path)
    _rewrite_bound_gate(
        frozen_path,
        lambda report: report["thresholds"].update(
            {"minimum_distinct_final_labels_main_arms": 1}
        ),
    )

    with pytest.raises(runner.AgentABError, match="thresholds"):
        runner._validate_frozen_config(
            frozen_path, expected["configuration"], expected["inputs"]
        )


def test_frozen_v2_validator_rejects_constant_label_main_arm(
    runner, tmp_path: Path
) -> None:
    frozen_path, expected = _write_quality_gated_freeze(runner, tmp_path)

    def collapse(report) -> None:
        arm = report["observed"]["single_self_review"]
        arm["label_counts"] = {"SUPPORT": 809, "CONTRADICT": 0, "NEI": 0}
        arm["nonempty_prediction_count"] = 809
        arm["nonempty_prediction_rate"] = 1.0
        arm["distinct_label_count"] = 1

    _rewrite_bound_gate(frozen_path, collapse)

    with pytest.raises(runner.AgentABError, match="main arm collapsed"):
        runner._validate_frozen_config(
            frozen_path, expected["configuration"], expected["inputs"]
        )
