"""Generate the deterministic provisional full-document-v1 benchmark."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "eval_sets" / "full_document_v1"
DOCUMENTS = OUTPUT / "documents"

RELATIONS = (
    "entailed",
    "partially_entailed",
    "contradicted",
    "not_in_source",
    "source_unavailable",
    "not_checkable",
)
ENTITIES = (
    "Aster Queue",
    "Beryl Relay",
    "Cinder Ledger",
    "Dahlia Runtime",
    "Ember Catalog",
    "Fjord Monitor",
    "Garnet Proxy",
    "Harbor Index",
    "Iris Scheduler",
    "Juniper Vault",
    "Kestrel Builder",
    "Lumen Registry",
    "Mica Stream",
    "Nacre Console",
    "Opal Gateway",
    "Pine Compiler",
    "Quartz Broker",
    "Rill Engine",
    "Spruce Archive",
    "Tidal Worker",
    "Umber Store",
    "Violet Daemon",
    "Willow Cache",
    "Xenon Inspector",
)
PROFILES = (
    "Cedar",
    "Drift",
    "Elm",
    "Flint",
    "Grove",
    "Heath",
    "Ivory",
    "Jade",
)
CODE_PATHS = [
    "src/evidencetrace/markdown.py",
    "src/evidencetrace/agents/miner.py",
    "src/evidencetrace/agents/judge.py",
    "src/evidencetrace/checks/deterministic.py",
    "src/evidencetrace/eval/baselines.py",
    "src/evidencetrace/eval/router.py",
    "src/evidencetrace/eval/full_document.py",
    "src/evidencetrace/policy.py",
    "src/evidencetrace/sarif.py",
]
PROMPT_SCHEMA_PATHS = [
    "src/evidencetrace/audit_models.py",
    "src/evidencetrace/eval/full_document.py",
    "src/evidencetrace/model_client.py",
]


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def stable_json(payload: Any) -> str:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8", newline="\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    write_text(
        path,
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for row in rows
        ),
    )


def bundle_hash(paths: list[str]) -> str:
    payload = "".join(f"{sha256_file(ROOT / path)}  {path}\n" for path in paths)
    return sha256_bytes(payload.encode())


@dataclass(frozen=True)
class Facts:
    entity: str
    peer: str
    capacity: int
    version: str
    date: str
    profile: str
    percentage: int

    @property
    def capacity_sentence(self) -> str:
        return f"{self.entity} accepts {self.capacity} jobs per batch."

    @property
    def release_sentence(self) -> str:
        return (
            f"{self.entity} version {self.version} entered stable support "
            f"on {self.date}."
        )

    @property
    def profile_sentence(self) -> str:
        return f"{self.entity} supports the {self.profile} operating profile."

    @property
    def negative_sentence(self) -> str:
        return f"{self.entity} does not export unencrypted backups."

    @property
    def comparison_sentence(self) -> str:
        return (
            f"In the Atlas-5 benchmark, {self.entity} completed "
            f"{self.percentage}% more tasks than {self.peer}."
        )


@dataclass(frozen=True)
class ClaimSpec:
    relation: str
    text: str
    claim_type: str
    checkability: str
    citation_urls: tuple[str, ...]
    evidence: str | None
    evidence_source_id: str | None
    construction_note: str


def facts_for(index: int) -> Facts:
    month = (index % 9) + 1
    day = (index * 3 % 24) + 2
    return Facts(
        entity=ENTITIES[index],
        peer=f"{PROFILES[(index + 3) % len(PROFILES)]} Node",
        capacity=29 + index * 7,
        version=f"v{1 + index % 4}.{2 + index % 7}",
        date=f"2026-{month:02d}-{day:02d}",
        profile=PROFILES[index % len(PROFILES)],
        percentage=11 + index * 2,
    )


def source_content(facts: Facts, stratum: str) -> str:
    facts_sentences = (
        facts.capacity_sentence,
        facts.release_sentence,
        facts.profile_sentence,
        facts.negative_sentence,
        facts.comparison_sentence,
    )
    if stratum == "short":
        return "; ".join(facts_sentences)
    if stratum == "medium":
        return "\n\n".join(
            (
                f"{facts.capacity_sentence} {facts.release_sentence}",
                facts.profile_sentence,
                facts.negative_sentence,
                facts.comparison_sentence,
            )
        )
    distractors = (
        (f"{facts.peer} accepts {facts.capacity + 13} jobs in its laboratory profile."),
        (
            f"A compatibility memo dated 2025-11-17 discusses version "
            f"v{2 + int(facts.version[1])}.0 of a different product."
        ),
        "The telemetry appendix describes color themes but no product capability.",
        (
            f"An unrelated sample reports {facts.percentage + 7}% utilization "
            "for a staging cluster."
        ),
    )
    return "\n\n".join((*facts_sentences, *distractors))


def claim_for(
    relation: str,
    facts: Facts,
    *,
    variant: int,
    primary_url: str,
    unavailable_url: str,
    primary_source_id: str,
    unavailable_source_id: str,
) -> ClaimSpec:
    if relation == "entailed":
        options = (
            (facts.capacity_sentence, "numeric_measurement"),
            (facts.release_sentence, "dated_release"),
            (facts.negative_sentence, "negative_capability"),
            (facts.comparison_sentence, "benchmark_comparison"),
        )
        text, claim_type = options[variant % len(options)]
        return ClaimSpec(
            relation=relation,
            text=text,
            claim_type=claim_type,
            checkability="checkable",
            citation_urls=(primary_url,),
            evidence=text,
            evidence_source_id=primary_source_id,
            construction_note="direct source support",
        )
    if relation == "partially_entailed":
        profile_index = PROFILES.index(facts.profile)
        unsupported_profile = PROFILES[
            (profile_index + 1 + variant % (len(PROFILES) - 1)) % len(PROFILES)
        ]
        text = (
            f"{facts.entity} supports the {facts.profile} operating profile "
            f"and the {unsupported_profile} profile."
        )
        return ClaimSpec(
            relation=relation,
            text=text,
            claim_type="compound_capability",
            checkability="checkable",
            citation_urls=(primary_url,),
            evidence=facts.profile_sentence,
            evidence_source_id=primary_source_id,
            construction_note="one supported and one unsupported clause",
        )
    if relation == "contradicted":
        mode = variant % 5
        if mode == 0:
            text = f"{facts.entity} accepts {facts.capacity + 1} jobs per batch."
            evidence = facts.capacity_sentence
            claim_type = "numeric_measurement"
        elif mode == 1:
            changed_date = facts.date[:-2] + f"{int(facts.date[-2:]) + 1:02d}"
            text = (
                f"{facts.entity} version {facts.version} entered stable support "
                f"on {changed_date}."
            )
            evidence = facts.release_sentence
            claim_type = "dated_release"
        elif mode == 2:
            major, minor = facts.version[1:].split(".")
            changed_version = f"v{major}.{int(minor) + 1}"
            text = (
                f"{facts.entity} version {changed_version} entered stable support "
                f"on {facts.date}."
            )
            evidence = facts.release_sentence
            claim_type = "versioned_release"
        elif mode == 3:
            text = f"{facts.entity} exports unencrypted backups."
            evidence = facts.negative_sentence
            claim_type = "negative_capability"
        else:
            text = (
                f"In the Atlas-5 benchmark, {facts.entity} completed "
                f"{facts.percentage + 1}% more tasks than {facts.peer}."
            )
            evidence = facts.comparison_sentence
            claim_type = "benchmark_comparison"
        return ClaimSpec(
            relation=relation,
            text=text,
            claim_type=claim_type,
            checkability="checkable",
            citation_urls=(primary_url,),
            evidence=evidence,
            evidence_source_id=primary_source_id,
            construction_note="single deterministic slot conflict",
        )
    if relation == "not_in_source":
        return ClaimSpec(
            relation=relation,
            text=(
                f"{facts.entity} includes a violet incident map for satellite "
                "operators."
            ),
            claim_type="uncited_capability",
            checkability="checkable",
            citation_urls=(primary_url,),
            evidence=None,
            evidence_source_id=None,
            construction_note="same entity with absent predicate",
        )
    if relation == "source_unavailable":
        return ClaimSpec(
            relation=relation,
            text=(
                f"{facts.entity} field recorder keeps archives for {43 + variant} days."
            ),
            claim_type="retention_limit",
            checkability="checkable",
            citation_urls=(unavailable_url,),
            evidence=None,
            evidence_source_id=None,
            construction_note="frozen unavailable citation",
        )
    return ClaimSpec(
        relation=relation,
        text=(
            f"Thoughtful operators should prefer {facts.entity} "
            "for an elegant workflow."
        ),
        claim_type="subjective_recommendation",
        checkability="not_checkable",
        citation_urls=() if (variant // 5) % 2 == 0 else (primary_url,),
        evidence=None,
        evidence_source_id=None,
        construction_note="subjective claim without an objective criterion",
    )


def citation_markup(urls: tuple[str, ...], label: str = "Source") -> str:
    suffixes = ("primary", "secondary", "tertiary", "quaternary")
    return " ".join(
        f"[{label} {suffixes[index]}]({url})" for index, url in enumerate(urls)
    )


def render_document(
    document_id: str,
    entity: str,
    stratum: str,
    claims: list[ClaimSpec],
) -> str:
    lines = [f"# {entity} integration notes", ""]
    if stratum in {"medium", "long"}:
        lines.extend(
            [
                (
                    "This document records release observations for a local "
                    "integration review."
                ),
                "",
            ]
        )
    if stratum == "long":
        lines.extend(
            [
                "## Context",
                "",
                "Background context is intentionally repeated.",
                "",
                "Background context is intentionally repeated.",
                "",
                "The appendix contains workflow organization prose.",
                "",
                "## Assertions",
                "",
            ]
        )

    if document_id == "fdv1_doc_001":
        shared = replace(claims[0], citation_urls=claims[0].citation_urls[:1])
        second = replace(claims[1], citation_urls=shared.citation_urls)
        claims[0] = shared
        claims[1] = second
        lines.extend(
            [
                (
                    f"{shared.text} {second.text} "
                    f"{citation_markup(shared.citation_urls, 'Shared')}"
                ),
                "",
            ]
        )
        remaining = claims[2:]
    else:
        remaining = claims

    for claim in remaining:
        suffix = (
            f" {citation_markup(claim.citation_urls)}" if claim.citation_urls else ""
        )
        lines.extend([claim.text + suffix, ""])

    if stratum == "long":
        lines.extend(
            [
                "## Non-claim material",
                "",
                "Deployment prose here is deliberately non-assertive.",
                "",
                "```text",
                "sample configuration omitted",
                "```",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def generate() -> dict[str, Any]:
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    DOCUMENTS.mkdir(parents=True)

    source_rows: list[dict[str, Any]] = []
    document_rows: list[dict[str, Any]] = []
    gold_rows: list[dict[str, Any]] = []
    document_paths: list[str] = []

    for index, entity in enumerate(ENTITIES):
        number = index + 1
        document_id = f"fdv1_doc_{number:03d}"
        split = "dev" if number <= 12 else "test"
        stratum = ("short", "medium", "long")[index % 3]
        facts = facts_for(index)
        base_url = f"https://full-document.example.test/{document_id}"
        primary_url = f"{base_url}/reference"
        mirror_url = f"{base_url}/mirror"
        unavailable_url = f"{base_url}/retired"
        primary_id = f"fdv1_s{number:03d}_primary"
        mirror_id = f"fdv1_s{number:03d}_mirror"
        unavailable_id = f"fdv1_s{number:03d}_unavailable"
        content = source_content(facts, stratum)
        mirror_content = (
            f"Independent release mirror. {content}"
            if stratum != "long"
            else f"Independent release mirror.\n\n{content}"
        )
        for source_id, url, body, available in (
            (primary_id, primary_url, content, True),
            (mirror_id, mirror_url, mirror_content, True),
            (unavailable_id, unavailable_url, "", False),
        ):
            source_rows.append(
                {
                    "available": available,
                    "content": body,
                    "content_hash": sha256_bytes(body.encode()),
                    "provenance": (
                        "Author-constructed deterministic full-document-v1 "
                        "fixture; no network retrieval."
                    ),
                    "source_id": source_id,
                    "url": url,
                }
            )

        claims = [
            claim_for(
                RELATIONS[(index * 5 + offset) % len(RELATIONS)],
                facts,
                variant=index * 5 + offset,
                primary_url=primary_url,
                unavailable_url=unavailable_url,
                primary_source_id=primary_id,
                unavailable_source_id=unavailable_id,
            )
            for offset in range(5)
        ]

        if document_id == "fdv1_doc_002":
            target = next(
                position
                for position, claim in enumerate(claims)
                if claim.evidence is not None
            )
            claims[target] = replace(
                claims[target],
                citation_urls=(primary_url, mirror_url),
                construction_note="one claim bound to two corroborating citations",
            )
        if document_id == "fdv1_doc_003":
            target = next(
                position
                for position, claim in enumerate(claims)
                if " " in claim.text and claim.relation != "not_checkable"
            )
            before, after = claims[target].text.split(" ", 1)
            claims[target] = replace(
                claims[target],
                text=before + "\n" + after,
                construction_note="claim span crosses a Markdown source line",
            )
        if document_id == "fdv1_doc_004":
            duplicate = replace(
                claims[0],
                construction_note="duplicate claim text at a distinct source span",
            )
            claims[1] = duplicate

        relative_path = f"eval_sets/full_document_v1/documents/{split}/{document_id}.md"
        markdown = render_document(document_id, entity, stratum, claims)
        write_text(ROOT / relative_path, markdown)
        document_paths.append(relative_path)

        cursor = 0
        source_ids_by_url = {
            primary_url: primary_id,
            mirror_url: mirror_id,
            unavailable_url: unavailable_id,
        }
        for claim_number, claim in enumerate(claims, 1):
            offset_start = markdown.find(claim.text, cursor)
            if offset_start < 0:
                raise RuntimeError(f"claim span missing in {document_id}")
            offset_end = offset_start + len(claim.text)
            cursor = offset_end
            line_start = markdown.count("\n", 0, offset_start) + 1
            line_end = markdown.count("\n", 0, max(offset_start, offset_end - 1)) + 1
            gold_rows.append(
                {
                    "annotation_status": (
                        "author_constructed_deterministic_provisional"
                    ),
                    "checkability": claim.checkability,
                    "citation_urls": list(claim.citation_urls),
                    "claim_id": f"{document_id}_claim_{claim_number:02d}",
                    "claim_type": claim.claim_type,
                    "construction_note": claim.construction_note,
                    "document_id": document_id,
                    "document_path": relative_path,
                    "gold_evidence_source_id": claim.evidence_source_id,
                    "gold_evidence_span": claim.evidence,
                    "gold_relation": claim.relation,
                    "line_end": line_end,
                    "line_start": line_start,
                    "offset_end": offset_end,
                    "offset_start": offset_start,
                    "source_ids": [
                        source_ids_by_url[url] for url in claim.citation_urls
                    ],
                    "text": claim.text,
                }
            )

        document_rows.append(
            {
                "content_sha256": sha256_bytes(markdown.encode()),
                "document_id": document_id,
                "gold_claim_count": len(claims),
                "path": relative_path,
                "split": split,
                "stratum": stratum,
            }
        )

    write_jsonl(OUTPUT / "documents.jsonl", document_rows)
    write_jsonl(OUTPUT / "gold_claims.jsonl", gold_rows)
    write_jsonl(OUTPUT / "sources.jsonl", source_rows)

    dataset_paths = [
        *document_paths,
        "eval_sets/full_document_v1/documents.jsonl",
        "eval_sets/full_document_v1/gold_claims.jsonl",
        "eval_sets/full_document_v1/sources.jsonl",
    ]
    dataset_sha256 = bundle_hash(dataset_paths)
    relation_support = {
        relation: sum(row["gold_relation"] == relation for row in gold_rows)
        for relation in RELATIONS
    }
    paragraph_ceiling = 12
    claim_ceiling = 12
    split_documents = 12
    retry_multiplier = 2
    maximum_provider_calls_per_split = (
        split_documents + split_documents * (paragraph_ceiling + claim_ceiling)
    ) * retry_multiplier
    execution_policy = {
        "adaptive_router_policy": "adaptive-router-v1",
        "maximum_claims_per_document": claim_ceiling,
        "maximum_markdown_paragraphs_per_document": paragraph_ceiling,
        "maximum_provider_calls_per_split": maximum_provider_calls_per_split,
        "maximum_total_tokens_per_split": 1_000_000,
        "schema_recovery_policy": "schema-recovery-v1",
        "schema_retry_limit": 1,
        "semantic_repair": 0,
    }
    execution_policy_sha256 = sha256_bytes(
        json.dumps(execution_policy, sort_keys=True, separators=(",", ":")).encode()
    )
    implementation_freeze = {
        "code_bundle_paths": CODE_PATHS,
        "code_bundle_sha256": bundle_hash(CODE_PATHS),
        "execution_policy_sha256": execution_policy_sha256,
        "prompt_schema_bundle_paths": PROMPT_SCHEMA_PATHS,
        "prompt_schema_bundle_sha256": bundle_hash(PROMPT_SCHEMA_PATHS),
    }
    preregistration = {
        "artifact_version": "full-document-v1-preregistration-v1",
        "baseline_definitions": {
            "miner_adaptive_judge_live": (
                "Paragraph-scoped Claim Miner, deterministic local citation "
                "binding, existing Adaptive Router/Retrieval, then bounded "
                "single-claim Judge."
            ),
            "single_agent_document_live": (
                "One primary structured-output call per Markdown document for "
                "claim extraction, citation binding, and source verdicts."
            ),
        },
        "call_budget": {
            "maximum_claims_per_document": claim_ceiling,
            "maximum_markdown_paragraphs_per_document": paragraph_ceiling,
            "maximum_provider_calls_per_split": maximum_provider_calls_per_split,
            "maximum_total_tokens_per_split": 1_000_000,
            "schema_retry_limit": 1,
            "semantic_repair": 0,
        },
        "dataset_bundle_sha256": dataset_sha256,
        "implementation_freeze": implementation_freeze,
        "metrics": [
            "exact_claim_extraction_precision_recall_f1",
            "relaxed_span_token_f1",
            "atomicity_violation_rate",
            "citation_binding_accuracy",
            "line_locator_accuracy",
            "relation_observed_and_fixed_six_macro_f1",
            "relation_weighted_f1",
            "relation_balanced_accuracy",
            "contradiction_recall",
            "not_checkable_f1",
            "evidence_span_f1",
            "end_to_end_claim_relation_f1",
            "document_policy_decision_accuracy",
            "miner_judge_handoff_failure_rate",
            "calls_tokens_p50_p95_latency",
            "single_agent_multi_agent_quality_and_resource_delta",
        ],
        "reporting_rules": [
            "Report dev and test separately and never tune on test outcomes.",
            "Count every missed gold claim as an end-to-end and relation failure.",
            "Use one-to-one claim alignment; never reuse a prediction.",
            "Report Miner and Judge calls, tokens, and latency separately.",
            "Do not call Router or Retrieval independent agents.",
            "This provisional benchmark is rerunnable and is not a blind holdout.",
            (
                "A later live run requires separate explicit model and budget "
                "authorization."
            ),
        ],
        "scope": {
            "dev_documents": 12,
            "test_documents": 12,
            "test_execution": "not_run",
            "dev_execution": "not_run",
        },
        "status": "registered_not_executed",
    }
    write_text(OUTPUT / "preregistration.json", stable_json(preregistration))

    readme = """# full-document-v1

This is an author-constructed, deterministic provisional benchmark for the
complete Markdown -> Claim Miner -> citation binding -> Adaptive/Judge ->
policy/SARIF path. It is not blind, not a public benchmark, and may be rerun
after versioned engineering changes.

The dataset contains 24 Markdown documents split evenly into dev and test.
Gold claims retain exact source offsets, one-based line ranges, citation URLs,
locally assigned source IDs, relation labels, and continuous evidence spans.
All cited sources are frozen local fixtures under `sources.jsonl`; `.test`
URLs are identifiers and must never be fetched.
"""
    write_text(OUTPUT / "README.md", readme)

    artifact_paths = [
        *dataset_paths,
        "eval_sets/full_document_v1/preregistration.json",
        "eval_sets/full_document_v1/README.md",
    ]
    manifest = {
        "annotation": {
            "counts_as_human_review": False,
            "method": "author_constructed_deterministic_provisional",
        },
        "artifact_version": "full-document-v1-manifest-v1",
        "benchmark_kind": "full_document_end_to_end",
        "blind_holdout": False,
        "case_counts": {
            "documents": len(document_rows),
            "gold_atomic_claims": len(gold_rows),
            "sources": len(source_rows),
        },
        "dataset_bundle_paths": dataset_paths,
        "dataset_bundle_sha256": dataset_sha256,
        "features": [
            "shared_citation",
            "multiple_citations",
            "no_citation",
            "duplicate_text",
            "cross_line_claim",
            "multiple_facts_in_one_paragraph",
            "numeric_date_version_negation_comparison",
            "partial_support",
            "source_unavailable",
            "long_source_distractors",
        ],
        "files": {
            path: {
                "bytes": (ROOT / path).stat().st_size,
                "sha256": sha256_file(ROOT / path),
            }
            for path in artifact_paths
        },
        "implementation_freeze": implementation_freeze,
        "gold_claims_per_document": {
            "maximum": 5,
            "minimum": 5,
        },
        "public_benchmark_eligible": False,
        "relation_support": relation_support,
        "rerunnable": True,
        "split_counts": {"dev": 12, "test": 12},
        "status": "provisional_benchmark_ready",
        "stratum_counts": {
            stratum: sum(row["stratum"] == stratum for row in document_rows)
            for stratum in ("short", "medium", "long")
        },
    }
    write_text(OUTPUT / "manifest.json", stable_json(manifest))
    return manifest


if __name__ == "__main__":
    result = generate()
    print(
        stable_json(
            {
                "dataset_bundle_sha256": result["dataset_bundle_sha256"],
                **result["case_counts"],
            }
        ),
        end="",
    )
