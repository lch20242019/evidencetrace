# Phase 4B Full-Document Live Dev Outcome

Status: smoke failed closed; full dev and test were not run.

## Scope And Freeze

The execution started from
`033f32dc89aa04afa95dd793f6dafcb82fe5746b` and used
`deepseek-v4-flash`, provider-default thinking, requested temperature `0.0`,
one schema-only retry per logical call, no semantic repair, and no cache. The
global limits were 600 provider attempts and 1,000,000 total tokens.

All registered dataset, manifest, preregistration, gold, source, code,
prompt/schema, and execution-policy hashes matched before the first request.
The executor streamed only the frozen dev prefixes: 12 document records, 60
gold records, 36 source fixtures, and 12 `documents/dev/*.md` files. Test
JSONL rows were not parsed and test Markdown files were not opened. Whole-file
access was limited to required byte-level SHA-256 verification.

## Smoke Outcome

The deterministic smoke document was `fdv1_doc_001`, selected as the first dev
document in frozen manifest order. Its `single_agent_document_live` call
failed strict schema validation on the first attempt and again after the one
allowed schema-only recovery.

The safe classification is:

- baseline: `single_agent_document_live`
- stage: `single_agent`
- category: `schema`
- code: `model_schema_invalid`
- logical calls: 1
- provider attempts: 2
- first-attempt schema failures: 1
- schema retry calls: 1
- recovered schema failures: 0
- unrecovered schema failures: 1
- input tokens: 1,972
- output tokens: 4,096
- total tokens: 6,068
- cost: null

The artifacts do not retain the provider response, invalid fields, finish
reason, model rationale, prompt, Markdown/source body, headers, or
credentials. The failure subtype cannot be narrowed further without guessing.
Per-call latency was not retained in the safe failure outcome, so p50/p95 are
unavailable.

The Miner plus Adaptive Judge smoke did not start. The two complete
12-document dev baselines did not start. The test split was not loaded,
parsed, or run.

## Metrics Boundary

No extraction, citation, locator, relation, evidence, end-to-end, document
policy, handoff, resource-delta, or Multi-Agent quality metric exists for this
attempt. Missing gold and spurious claim counts also do not exist because no
document produced a completed canonical prediction artifact. These values are
not estimated.

This provisional dev attempt is not blind, human-reviewed, formal, or a
test-split result. It supports no claim that the Multi-Agent system is better
or worse than the document Single Agent.

## Artifacts

- `eval_runs/phase4b_full_document_live_dev/failed_attempt.json`:
  `e175b57583f621432cb6a96a5624bf294e71e841c1dd481c58feb514da3e2241`
- `eval_runs/phase4b_full_document_live_dev/execution_outcome.json`:
  `ffee09d489132de01b4047d5252f0f7c2ef2add960e1b95e993e4c394ee0745f`
- one-shot executor:
  `483c4d21c9e4ae9772fd3d419875c5007d3c19e3d8eb2de01f0aa04571a29fc9`

The repository remains an engineering continuation with known limitations.
Phase 3 stays `completed_with_known_limitations`, `phase4_eligible=false`, and
the test split is not ready for live execution.

## Verification

The same boundary-compliant selected suite passed 498 tests before and after
the attempt; it excludes seven consumed-corpus loader modules containing 41
tests. Compileall, `git diff --check`, exact-credential and artifact privacy
scans, and Ruff lint/format checks for the frozen one-shot executor passed.
All eight Phase 4B freeze hashes, the historical Phase 3H failure hashes, and
the byte-only consumed-corpus hashes remained unchanged. No repository Python
or production file was modified.
