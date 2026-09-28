# Phase 4B.1 Provider Contract Hardening

Status: offline implementation frozen; both bounded dev smokes completed.

## Historical Failure Boundary

The retained Phase 4B failure identifies only
`single_agent / schema / model_schema_invalid`. It contains no finish reason,
response length, JSON/Pydantic stage, field path, or provider content. The two
attempts used the former generic 2,048-token output ceiling and reported 4,096
output tokens in aggregate. That saturation makes truncation a diagnostic
hypothesis, not a proven root cause. The historical classification therefore
remains `unknown_schema_failure`; no missing provider output is reconstructed.

## Contract Audit

The OpenAI-compatible adapter uses `response_format={"type":"json_object"}`.
The complete Pydantic JSON schema, strict required-field instruction,
`additionalProperties=false`, and a valid example are carried in the user
contract. No provider-specific thinking parameter is sent, so thinking remains
provider-default. JSON envelope validation precedes content validation;
Pydantic strict JSON parsing then distinguishes JSON decode from schema
validation. Only `ModelSchemaError` can receive one schema-only retry.

The full-document provider contract now requires the project maximum
`max_tokens=8192`. Calls using the generic 2,048-token default fail locally
before a provider request. The request budget appears in canonical provenance,
per-call telemetry, safe diagnostics, and cache-key dimensions. The model
still owns only semantic extraction and verdict fields. Deterministic IDs,
offsets, line locators, source IDs, evidence locators, versions, and
corroboration remain locally owned.

Single Agent document output, full-context pair output, and Judge output retain
required relation/evidence semantics while limiting model-authored `reason` to
240 characters. Missing fields, additional fields, bad enums, type errors, and
overlong reasons remain strict schema failures. No JSON repair, field
completion, enum correction, or semantic normalization is performed.

## Safe Diagnostics

New diagnostics may retain only:

- an allowlisted contract stage and finish reason;
- requested max tokens;
- reported input, output, and reasoning-token integers;
- whether content exists and its character count;
- bounded Pydantic error codes and allowlisted field paths; and
- a boolean suspected-token-truncation signal.

They never retain response content, reasoning content, claim/source text,
prompt data, headers, credentials, validation input, or exception context.
Historical failure artifacts remain byte-identical and readable without
invented diagnostics.

## Live Boundary

The bounded smoke may read only the frozen dev prefix and one physical dev
Markdown document. Full-file dataset access is limited to byte-level hash
verification. Single Agent and Miner plus Adaptive Judge are independent
smokes; a Single Agent failure does not suppress the Multi-Agent smoke.
Together they may use at most 20 provider attempts and 200,000 reported
tokens, with no cache and no full dev or test execution.

## Bounded Smoke Result

Both independent smokes used `fdv1_doc_001` and completed operationally. The
document Single Agent used two provider attempts: its first response reached
`finish_reason=stop` but failed at `schema_validation`, with the safe
Pydantic path `claims/0` and code `value_error`. It was not suspected to be
truncated. The one allowed schema-only retry succeeded. This is reported as
one first-attempt schema failure, one retry, one recovered failure, and zero
unrecovered failures.

The Multi-Agent smoke used four first-pass Miner calls and five first-pass
Judge calls. All nine reached `finish_reason=stop`; none required recovery,
and no schema, transport, or suspected-truncation failure occurred. Router and
Retrieval remain deterministic components rather than Agents.

The combined execution used 11 of 20 provider attempts and 18,597 of 200,000
reported tokens. All usage records were complete. Cache status was
`disabled_not_available`, and cost remains null because there is no frozen
price snapshot. The test split was not loaded, parsed, or run.

The single smoke document is operational evidence only. Its extraction,
relation, and policy values are not a full-dev quality result and do not
support a Multi-Agent advantage claim. The implementation is engineering-ready
for a separately authorized complete dev run. Test execution remains
unauthorized and must depend on the complete dev outcome.

## Validation Note

One preliminary pytest invocation used an incorrect historical exclusion
list and executed `test_eval_v2_candidates.py` in an isolated process. It
printed no candidate content, made no provider call, and is not counted as a
boundary-compliant result. The corrected suite excluded all seven documented
consumed-corpus loader modules and passed 510 tests before live execution.
This process deviation is retained rather than silently described as
hash-only compliance.
