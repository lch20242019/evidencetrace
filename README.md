# EvidenceTrace

EvidenceTrace is a bounded fact-audit Agent for Markdown, ADRs, RFCs, READMEs,
and Markdown Git diffs. It extracts atomic claims, verifies citations first,
discovers evidence only when authorized, challenges high-risk verdicts, and
reports findings at the original source line through the terminal, canonical
JSON, and SARIF.

EvidenceTrace is not a truth detector. A completed verdict states how the
available evidence relates to a claim; `needs_human` and partial document
results remain explicit.

## Quickstart

From this source checkout:

```bash
python3 -m pip install -e .
evidencetrace demo
evidencetrace check <path>
evidencetrace check <path> --discover --sarif evidencetrace.sarif
evidencetrace check <path> --suggest-patch suggestions.diff
evidencetrace check <path> --changed-from <git-revision>
```

The current self-use path also accepts multiple explicit Markdown/TXT targets,
shared local references, and a separate interactive `fix` command:

```bash
uv sync --frozen --extra dev

uv run evidencetrace check TARGET1.md TARGET2.txt \
  --reference REF1.md --reference REF2.txt --discover

uv run evidencetrace fix TARGET1.md TARGET2.txt \
  --reference REF1.md --reference REF2.txt --discover
```

`demo` is fully offline and byte-stable. Its small repository fixture exercises
a cited claim, an uncited claim that triggers Scout, a numeric contradiction
that triggers Challenger, and a subjective claim that is skipped as
not-checkable.

This public source snapshot omits local evaluation datasets and generated run
artifacts (`eval_sets/` and `eval_runs/`). Evaluation commands and tests that
read those directories require the corresponding local data.

Model-backed Agents are optional:

```bash
export EVIDENCETRACE_MODEL=<openai-compatible-model>
export OPENAI_API_KEY=<configured-secret>
export OPENAI_BASE_URL=<provider-base-url>
export TAVILY_API_KEY=<optional-discovery-secret>
```

With no model credential, the deterministic Controller, Miner, retrieval,
Judge, and fallback plans still run. With `--discover` but no Tavily secret,
discovery returns `discovery_unavailable` without making a request or
terminating unrelated claims.

The bounded audit path is:

```text
explicit Markdown/TXT targets + shared references
  -> Claim Miner
  -> Audit Coordinator initial plan
  -> citation / local reference / authorized Scout discovery
  -> fix-only human evidence trust or conflict selection when required
  -> Claim Judge
  -> Audit Coordinator review plan
  -> conditional Challenger
  -> policy and canonical artifacts
  -> fix only: deterministic scalar repair
       -> per-candidate apply confirmation
       -> stale checks, atomic write, and reversible diffs
```

The Controller owns tools, budgets, retries, artifacts, policy, and final plan
validation. A claim-level Agent or source failure produces a safe status and
does not stop other claims. `--suggest-patch` writes a candidate unified diff;
on the read-only `check` path it never applies or commits it.

Legacy single-document product runs below `.evidencetrace/runs/<run-id>/` may
contain canonical `audit.json`, `product_run.json`, and derived `audit.md`.
Current batch/fix runs use `batch-manifest.json` plus
`targets/<safe-target-id>/` with per-target `audit.json`, `audit.md`,
`results.sarif`, and suggested/applied/reverse diffs. See
[architecture and safety boundaries](docs/architecture.md) for the exact
layout.

For the v2 self-use path, exit code `0` means no unresolved, rejected, stale,
Agent, or apply error; `2` is a safe partial; and `1` is a fatal preflight,
initialization, or read failure. Legacy single-document policy checks may also
use `1` for a completed policy failure. Inspect canonical outcomes rather than
inferring state from the code alone.

The reusable workflow at `.github/workflows/evidencetrace.yml` keeps
`contents: read` and `security-events: write`, uploads SARIF and a job summary
before propagating a nonzero result, and degrades to deterministic checks on
fork PRs without secrets.

Safe fetch rejects private/metadata targets, credential-bearing URLs,
non-text or oversized responses, and unsafe redirects. Search snippets select
candidate URLs only; they can never become evidence.

PyPI publishing and a public Action release are not complete. Therefore
`uvx evidencetrace ...` is the intended release interface, not a currently
verified installation path.

## Documentation

- [Documentation index](docs/README.md)
- [Interview preparation guide](docs/interview-guide.md)
- [Component deep dive](docs/component-deep-dive.md)
- [Engineering decisions and trade-offs](docs/engineering-decisions.md)
- [Problems and lessons learned](docs/problems-and-lessons.md)
- [Current project handoff](docs/PROJECT_HANDOFF.md)
- [Architecture and safety boundaries](docs/architecture.md)
- [Setup and migration recovery](docs/setup-and-recovery.md)
- [Known limitations](docs/known-limitations.md)

The Projects-root `evidencetrace_self_use_mvp_plan_2026-07-28.md` v2.0 remains
the normative self-use plan. Current code and canonical artifacts are the
implementation facts; dated handoffs and historical evaluation reports must
not override them.

## Historical Evaluation

Phase 3 remains `completed_with_known_limitations` and
`phase4_eligible=false`. The consumed holdouts are retained as immutable
history and no formal blind-holdout F1 was measured. The pair benchmark did
not run Claim Miner, and the provisional full-document smoke did not establish
a Multi-Agent quality advantage. Product engineering continues with those
limitations; no v5 or new evaluation phase is planned.

## Run the Phase 3B dev evaluation

The exposed v1 test split is now diagnostic-only. Development runs must select
the dev split explicitly and write a new output directory:

```bash
PYTHONPATH=src python3 -m evidencetrace eval eval_sets/core.jsonl \
  --out eval_runs/phase3b_dev --split dev
```

New pair-level runs use accurate deterministic baseline names:
`lexical_rules`, `lexical_full_source`, and
`retrieval_judge_deterministic`. Pair cases already provide an atomic claim, so
they go directly through retrieval and Judge; they do not call Claim Miner.
Claim-extraction metrics are therefore `null / not_applicable`. Extraction
precision and recall require a full Markdown benchmark with gold claim spans
and counts.
Reports distinguish observed-label macro-F1 from fixed six-label macro-F1,
show label support and coverage, and label repeated-fixture Recall@5 as a
sanity metric rather than a retrieval benchmark.

Live comparison is explicit:

```bash
PYTHONPATH=src python3 -m evidencetrace eval eval_sets/core.jsonl \
  --out eval_runs/phase3b_dev_live --split dev --live --model <model-id>
```

With credentials this adds `single_agent_live`, `retrieval_judge_live`, and
`adaptive_live` in a separate report table. Each makes at most one model
request per case and none calls Miner, so the combined pair-level hard ceiling
is `3 * case_count`. Adaptive deterministic routes often make no request.
Without `OPENAI_API_KEY`, all three are recorded as
`skipped_missing_credentials`; deterministic output is never presented as
model performance.

The historical `miner_judge_deterministic` and `miner_judge_live` names remain
readable only for frozen artifacts produced before Phase 3E.2. They are no
longer active pair-level baselines. A future `miner_judge_live` comparison must
start from full Markdown or paragraph input and measure extraction as well as
verification; see
[`docs/full-document-benchmark-design.md`](docs/full-document-benchmark-design.md).

The Phase 3A artifacts in `eval_runs/core/` remain immutable history and keep
their legacy names. The v1 test is `diagnostic_contaminated`, not a release
gate. Reviewer A's validated labels now produce a separate 60-case frozen
dataset at `eval_sets/v2/holdout_single_human.jsonl`. Its validity is exactly
`single_human_synthetic_holdout`; it is not double-human, adjudicated, or a
public benchmark, and every run must report
`public_benchmark_eligible: false`.

The first live dev preflight was attempted on 2026-07-15 with
`deepseek-v4-flash`. It failed closed on a local percentage-token validation
bug before artifacts were written; the v2 holdout was not run. The generic
Miner fix is frozen. Because the failed run consumed an estimated 26–33 calls,
the owner authorized exactly one complete 54-call rerun. That authorization is
now consumed.
The runner enforces 54 before each provider request; failed requests count,
automatic retries are zero, and no automatic rerun is allowed. A failed run
persists only secret-free operational telemetry in `failed_attempt.json`; it
does not serialize keys, headers, prompts, source payloads, or exception text.
Both live baselines use provider-default thinking and requested temperature 0.0.

The consumed dev command was the following; do not rerun it without new approval:

```bash
PYTHONPATH=src python3 -m evidencetrace eval eval_sets/v2/dev.jsonl \
  --out eval_runs/phase3d_v2_dev_live_rerun --split dev --live \
  --model "$EVIDENCETRACE_MODEL"
```

It executed once from frozen commit `bcda80b9ea9c21d46bda05fa2160b88e19f60fa0`
and stopped on `v2_dev_001`: the first `single_agent_live` provider response
failed schema validation after exactly one call. Reported usage was 259 input,
391 output, and 650 total tokens; latency was 3,571.082 ms; cost is null because
no price snapshot was supplied. The only output is the safe
`eval_runs/phase3d_v2_dev_live_rerun/failed_attempt.json`; no complete dev
metrics exist, no automatic rerun is allowed, and the holdout was not run.

Phase 3D.1 hardens only the OpenAI-compatible output contract, entirely
offline. Adapter prompt version `openai-compatible-v2` sends
`response_format: {"type": "json_object"}`, the exact Pydantic JSON
schema, one compact schema-valid example, and bounded `max_tokens` (default
2,048; validated range 1–8,192). Both system and user instructions require one
JSON object, all required fields, no extra fields, and no Markdown fence.
Pydantic strict JSON validation remains the final boundary; invalid JSON,
empty content, truncation, wrong types, extra fields, and invalid relations
fail closed without retry, aliasing, normalization, or a model repair call.

Future v2 failure artifacts may contain only allowlisted schema names,
redacted/allowlisted Pydantic locations and error types, a category-owned static
message, an allowlisted finish reason, and bounded response content length.
Schema, envelope, and transport exceptions are detached from their raw causes
and contexts before propagation. Artifacts never contain model content,
reasoning content, prompt/source/claim payloads, headers, or credentials. The
historical v1 failure artifact and SHA-256 remain unchanged; its schema path
cannot be recovered or inferred, and the v1 schema rejects invented diagnostics.
The offline hardening checkpoint itself made zero real model calls and is an
instrumentation repair, not evidence of improved model quality. Under
provider-default thinking, temperature 0.0 remains requested-only and actual
effective temperature remains unknown.

A later, separately authorized smoke used the frozen Phase 3D.1 contract for
exactly one `v2_dev_001` `single_agent_live` request. It passed strict schema
validation and source-substring grounding; its relation matched the dev gold.
The request reported 553 input, 177 output, and 730 total tokens, 2,151.273 ms
latency, and `finish_reason=stop`. Cost remains null because no explicit price
snapshot was supplied. The only artifact is
`eval_runs/phase3d1_v2_dev_001_smoke/smoke_attempt.json`, SHA-256
`b0d2ca17493efa3be7b2d37bac4de12fbc38d2d79653bfbd0463bf193745cf14`.
It contains only hashes, booleans/enums, configuration, and safe telemetry—not
model content, reasoning, reason, prompt, claim/source text, headers, or
credentials.

A second, separately authorized smoke exercised only the isolated
`miner_judge_live` chain for `v2_dev_001`. It made exactly two provider calls:
one Miner call followed by one Judge call, with zero retry or repair calls. The
strict `MinerOutput` and `JudgeOutput` schemas passed. Miner paragraph,
citation, protected-fact, and token-provenance guards passed; Judge claim/source/
locator scope, candidate-substring, substantive-evidence, and deterministic-
conflict guards passed. The case-local final relation matched dev gold and the
evidence was a literal candidate substring.

The Miner call reported 613 input, 475 output, and 1,088 total tokens with
4,316.504 ms latency. The Judge call reported 1,014 input, 535 output, and
1,549 total tokens with 4,378.621 ms latency. Combined usage was 1,627 input,
1,010 output, and 2,637 total tokens; sequential call latency totaled
8,695.125 ms. Cost is null because no explicit model price snapshot was
provided. The safe artifact is
`eval_runs/phase3d1_v2_dev_001_miner_judge_smoke/smoke_attempt.json`, SHA-256
`f01d4cfdb33a2923c5a2ca72b5235b0ee2e94368a60a41fa908b6fe870a273cd`.
It contains only frozen hashes, contract/guard results, safe telemetry, and
safety declarations—not model content, reasoning, prompts, claim/source text,
relations, evidence text, headers, or credentials.

Across these two one-case smokes, all three live response schemas
(`SingleAgentLiveOutput`, `MinerOutput`, and `JudgeOutput`) have passed the
hardened provider contract once. These are provider-contract smokes, not a
complete dev baseline, benchmark metric, model-quality conclusion, or Phase 3
gate result. Both smoke authorizations are consumed. At that checkpoint, a
complete dev comparison still required new explicit call/cost authorization;
the v2 holdout remained unexecuted and unauthorized.

The preceding paragraph records the historical state at those smoke
checkpoints. The v2 holdout was subsequently attempted once and is now
consumed. Its command is intentionally no longer shown: rerunning it cannot
create a valid gate result. Phase 3E.2 is entirely offline and does not
authorize a dev or holdout run. No v3 holdout exists, and Phase 4 remains
blocked.

The live Miner contract is now `live-miner-draft-v2`. It requires a `claims`
array, and each model-authored item may contain only `text`, `claim_type`, and
`checkability`. Exact paragraph offsets, stable claim IDs, file and line
metadata, citation URLs, model ID, and prompt version are filled locally;
`slots` is locally fixed to an empty object because it has no downstream
business consumer. A linear, non-recursive matcher accepts a duplicate span
only when its earliest forward and latest reverse monotonic mappings agree.
Non-source or ambiguous spans, missing protected tokens, and incomplete
fragments fail closed through a payload-free `MinerScopeError`. There is no
retry, output repair, aliasing, or normalization. The deterministic signal
policy remains
`deterministic-signals-v2`, and the transport/structured-output contract remains
`openai-compatible-v2`.

Phase 3E.3 makes Judge output validation distinguish integrity from model
quality. Claim/source/evidence scope violations still fail closed through a
payload-free, allowlisted `JudgeScopeError`; deterministic factual conflicts
still fail through `DeterministicConflictError`. A legal in-scope relation,
including an incorrect `not_checkable` decision, is now a scored prediction
instead of a fatal local `ValueError`. Strict JSON, zero retry, and zero repair
remain unchanged.

The frozen repair at commit `ad72cecfabf8d6969a51d739ef27b820bda4fbbc`
passed a five-call `v2_dev_017` smoke and three independent complete dev runs.
All three runs completed with zero schema, transport, guard, or local-validation
failures. `retrieval_judge_live` macro-F1 was `0.942857` in every run versus
`0.652778` for `single_agent_live`, a per-run delta of `+0.290079`. Relation
agreement was 18/18 for Single Agent and 17/18 for Retrieval-to-Judge;
`v2_dev_017` was the only unstable case. The aggregate is
`eval_runs/phase3e3_pair_dev_stability/aggregate.json` (SHA-256
`a30c43342a12ea0ed01e23afd54632d2268db9d281fe8ee5905c443c7038c78b`).
These are provisional dev results, not a Phase 3 gate. They establish the
prerequisite for separately creating and freezing a new v3 holdout; no v3 was
created, consumed v2 was not loaded or rerun, and Phase 4 remains blocked.

## Phase 3F-C v3_zh internal gate

The 72-case internal Chinese holdout was executed exactly once on 2026-07-23
from commit `f3c07304195807cba751843e498369cd390a929c`. The run completed all
five pair baselines with zero schema, transport, scope, guard, or local
validation failures. It used frozen Chinese source snapshots, made 111 provider
calls and zero Claim Miner calls, and consumed 133,833 reported tokens.

The headline `retrieval_judge_live` fixed-six macro-F1 was `0.613757` and
contradiction recall was `0.769231`. It therefore missed the preregistered
internal threshold of `0.70`, the separate v0.1 target of `0.75`, and the
contradiction-recall target of `0.80`. `single_agent_live` macro-F1 was
`0.727360`; the Retrieval-to-Judge delta was `-0.113603`. This is an
operationally complete metric failure: Phase 3 did not pass and Phase 4 is not
eligible.

Canonical artifacts and the append-only execution record are under
`eval_runs/phase3f_v3_zh_internal_gate/`. The record SHA-256 is
`74540003d8200410038028dc4a898aee422d00f1683a03ddfd465c8f01febf4a`.
The v3_zh holdout is consumed and must not be rerun as a formal gate. This
claim-source pair benchmark does not measure Claim Miner extraction and cannot
establish complete Claim Miner plus Judge Multi-Agent advantage.

## Phase 3G-B adaptive Chinese dev

Phase 3G-B is an offline implementation checkpoint. Chinese retrieval retains
the existing Latin path and adds Unicode-normalized character bigram/trigram
features only when ordinary lexical terms are absent or sparse. Multi-chunk
sources remain lexically ranked. A single available non-empty chunk cannot
disappear solely because of lexical mismatch.

The new `adaptive_live` baseline uses four explicit routes: unavailable source
to deterministic `source_unavailable`; purely subjective language without
objective fact slots to deterministic `not_checkable`; one chunk whose exact
serialized full-source prompt fits the configured 32,768-byte budget to the
existing Single Agent; and all other cases to Retrieval-to-Judge. Artifacts
record only allowlisted route telemetry: route, reason code, chunk count, and
estimated context size. Existing baseline meanings are unchanged.

The independent provisional dataset is
`eval_sets/phase3g_dev_zh/dev.jsonl`: 72 pair cases over 24 new source fixtures,
with 12 examples per relation. Available sources comprise seven short, seven
medium, and six long documents; four more are explicitly unavailable. Long
documents contain genuine operational distractor chunks. The pack and its
leakage audit are frozen in `eval_sets/phase3g_dev_zh/manifest.json`.

No live model was called, v3_zh was not rerun, and v4 was not created. Future
live dev work requires separate authorization and is preregistered to compare
`single_agent_live`, `retrieval_judge_live`, and `adaptive_live` overall and
separately for short, medium, and long sources. Phase 3 remains unpassed and
Phase 4 remains ineligible.

## Phase 3G-C adaptive Chinese live dev

The separately authorized three-run dev attempt used the frozen Phase 3G-B
code, prompts, policies, and 72-case dataset. Its four-route smoke passed with
two model calls and zero calls for the deterministic `not_checkable` and
unavailable paths.

Runs 01 and 02 then completed all six baselines and 72 cases. Each used 168
provider calls; Adaptive used exactly 48 calls and retained the frozen
17/31/12/12 route distribution. Adaptive observed-label and fixed-six
macro-F1, contradiction recall, and not-checkable recall were all `1.0` in
both completed runs. Single Agent macro-F1 was `0.958115` and `0.956703`.
Adaptive and Retrieval-to-Judge had perfect two-run relation agreement;
Single Agent agreement was `0.930556`.

Run 03 failed strict schema validation after 46 provider responses. It stopped
fail-closed with zero retry and zero repair, retained only a sanitized failure
artifact, and was not rerun. Therefore 3/3 operational success and the
all-failures-zero condition failed. Three-run metrics, relation agreement, and
the remaining quality/readiness conditions are not evaluated or inferred.
Total usage including smoke and the failed round was 384 calls and 542,644
reported tokens.

The incomplete aggregate is under
`eval_runs/phase3g_adaptive_zh_dev_stability/`. Phase 3 remains formally
unpassed, this result does not make a v4 blind holdout eligible, v3_zh was not
rerun, and no v4 or Phase 4 implementation was created.

## Phase 3G-D bounded schema recovery

Offline diagnosis classifies the Run 03 event as
`unknown_schema_failure`. The frozen safe records prove that it occurred on
`single_agent_live`, case `phase3g_zh_dev_046`, at
`single_agent_full_source_verification`, and that it crossed the provider
schema/JSON contract boundary rather than a local business validator. They do
not retain the finish reason, max-token state, assistant content envelope, or
Pydantic issue code, so no narrower subtype is claimed.

`schema-recovery-v1` permits exactly one retry after `ModelSchemaError`.
The retry uses the same claim/source scope and Pydantic schema plus a fixed,
content-free schema reminder; it never receives the first response. There is
no JSON repair, field completion, enum correction, or semantic normalization.
Transport, budget, scope, evidence, deterministic guard, and local-validation
failures do not trigger a retry. Every retry remains subject to the existing
pre-request global budget boundary and is recorded as attempt 2.

Canonical run telemetry now reports first-attempt schema failures, retry
calls, recovered and unrecovered failures, first-attempt contract success,
final operational success, and retry token/latency totals. Cache wiring remains
`not_available`; failed responses are not cached, and
`schema-recovery-v1` is part of future cache keys.

The next live dev attempt is preregistered in
`eval_runs/phase3g_schema_recovery_readiness/preregistration.json`. It requires
three entirely fresh complete runs; old Runs 01/02 do not count. The new code,
prompt/schema, and config/recovery hashes are respectively
`69c7554c6865ad789e7d8b4a74816ccf0efecd98b86729226562f5fad99149fe`,
`6dbe47902acc27f7c81f623c2c8c6c4546d685cfa9848ca5a931aa5b1a3fb658`,
and `d670ac287f9d2ae0779ddc5fb09ae59f98a0e0397303764797b876909aa4b15a`.
No model was called, Run 03 was not rerun, and no v4 or Phase 4 work was
created.

## Phase 3G-E schema-recovery live dev

The newly authorized attempt used commit
`060db3b066f19093a4b8cd97edfa4b47e73e519a` and the frozen Phase 3G-D
code, prompt/schema, config/recovery policy, and 72-case Chinese dev dataset.
The five-operation smoke succeeded with three live calls and zero calls on the
deterministic routes. All first attempts were contract-valid.

Fresh Runs 01 and 02 completed all six baselines and retained the
17/31/12/12 Adaptive route distribution. Adaptive and Retrieval-to-Judge
macro-F1 were `1.000000` in both runs. Single Agent macro-F1 was `0.972028`
and `0.986087`. Adaptive contradiction recall and not-checkable recall were
`1.000000`; its observed-label macro-F1 was also `1.000000` for short,
medium, long, and unavailable strata in both completed runs.

Run 03 stopped fail-closed in `retrieval_judge_live` on
`phase3g_zh_dev_059`. One first-attempt schema failure was recovered by the
single permitted retry, then the canonical Judge result failed the unchanged
scope contract with `JudgeScopeError` code `claim_id_mismatch`. Scope failures
are not retryable. The round retained only its sanitized
`failed_attempt.json`; no predictions or metrics were reconstructed, and no
three-run aggregate was generated.

The attempted dev calls achieved `455/456 = 0.997807` first-attempt contract
success, with one recovered and zero unrecovered schema failures. Final
operational success was only 2/3, however, and one scope failure remained.
Quality, stability, and product-value conditions requiring three complete
runs are therefore not evaluated. Total usage, including smoke and the failed
round, was 460 calls and 672,550 reported tokens; retry usage was 2,023 tokens.

The append-only outcome is
`eval_runs/phase3g_recovery_zh_dev_stability/execution_outcome.json`; the
complete retained-file hash index is `artifact_manifest.json` beside it.
Phase 3 remains formally unpassed, a v4 blind holdout is not yet eligible, and
Phase 4 remains ineligible. No consumed holdout was loaded or rerun, and no v4
was created. The boundary-compliant selected suite passed 439 tests both
before and after live execution; it excludes five consumed-corpus loader files
containing 26 tests and is not an unconditional full-suite claim. Compileall,
diff check, secret scan, frozen-input verification, and byte-only consumed
hash verification also passed.

## Phase 3G-F local verdict ownership

The live Judge now requests only `relation`, `confidence`, `reason`, and one
nullable-but-required `evidence_span` field from the model. The strict
`LiveJudgeSemanticOutput` schema requires all four keys and forbids additional
properties. `claim_id`, source identity, locator, corroboration, Judge/model/
prompt versions, and other correlation metadata are attached only from
trusted local request context.

The model prompt no longer contains the claim ID, source ID, locator, URL,
file/line provenance, or output-version fields. A returned `claim_id` or other
locally owned field is therefore an extra-field `ModelSchemaError`, eligible
only for the existing single `schema-recovery-v1` retry. It is never dropped
or repaired locally. Candidate-substring validation, local source/locator
scope, deterministic conflict checks, and unavailable-source checks remain
fail-closed. The canonical `JudgeOutput` and `Verdict` formats remain readable
for historical artifacts.

The offline freeze records code, prompt/schema, ownership-policy, and combined
config/recovery/ownership hashes in
`eval_runs/phase3g_local_verdict_ownership_readiness/preregistration.json`.
The boundary-compliant suite now passes 453 tests (the prior 439 plus 14 new
tests) while still excluding the five consumed-corpus loader files containing
26 tests. No model call, historical rerun, v4 creation, or Phase 4
implementation occurred during this offline checkpoint.

## Phase 3G-F ownership live dev

The fresh ownership freeze completed its smoke and all three Chinese dev
runs. Smoke used two live calls and zero calls for both deterministic routes.
Each full run completed all six baselines with Adaptive routes fixed at
17 full-context, 31 Retrieval-to-Judge, 12 deterministic not-checkable, and
12 unavailable cases.

Adaptive and Retrieval-to-Judge both achieved `1.000000` macro-F1 in every
run. Single Agent macro-F1 was `0.897198`, `0.957672`, and `0.986087`.
Adaptive contradiction recall, not-checkable recall, and observed-label
short/medium/long/unavailable macro-F1 were all `1.000000` in all three
runs. Adaptive relation agreement and route agreement were `1.000000`;
Single Agent agreement was `0.888889`.

There were four first-attempt schema failures: zero in Run 01 and two each in
Runs 02 and 03. All four recovered through the one permitted schema-only
retry, leaving zero unrecovered schema failures and zero final operational
failures. Combined first-attempt contract success was
`500/504 = 0.992063`. The run used 510 provider calls including smoke,
446,877 reported tokens, and 3,932 retry tokens; overall latency was
2,268.055 ms p50 and 3,775.860 ms p95.

Every preregistered dev-readiness condition passed. Adaptive exceeded Single
Agent mean macro-F1 by `0.053014`, improved long-stratum macro-F1 by
`0.073228`, and reduced input tokens by `36.22%`. This provisional pair-level
dev evidence makes creation of a new v4 blind holdout eligible. It does not
pass Phase 3, make Phase 4 eligible, or establish full Claim Miner plus Judge
Multi-Agent superiority. No consumed holdout was loaded or rerun.

The aggregate and complete byte-hash index are
`eval_runs/phase3g_ownership_zh_dev_stability/aggregate.json` and
`artifact_manifest.json` beside it. The post-run 453-test
boundary-compliant selected suite, compileall, changed-files Ruff, diff check,
secret scan, frozen-input verification, historical-artifact verification, and
byte-only consumed hashes all passed.

## Phase 3H-A v4_zh blind candidates

`eval_sets/v4_zh/` freezes a new Chinese blind candidate pack with 72 cases
and 24 sources, exactly three cases per source. Its 20 available sources are
short official Chinese snapshots from PingCAP, OceanBase, EMQ Technologies,
StarRocks, Apache Doris, vueComponent, and OpenHarmony; four additional
sources are frozen unavailable and will not be refetched during evaluation.
Production chunking yields 7 short, 7 medium, 6 long, and 4 unavailable
sources. The corresponding expected Adaptive routes are 17 full-context,
31 Retrieval-to-Judge, 12 deterministic not-checkable, and 12 unavailable.

The candidate JSONL and reviewer packet contain no relation, gold evidence,
model proposal, expected route, or case-level construction target. Only the
manifest records the aggregate 12-per-class construction balance. Human
annotation has not started, no gold dataset exists, and the one-time v4_zh
formal gate has not run.

Leakage checks found no cross-corpus ID, URL, entity, content-hash,
raw/normalized claim, structured numeric/date/version fingerprint, or
threshold-level n-gram collision against data that may legally be read.
Consumed v2/v3/v3_zh inputs were checked only by byte hash and retained
non-label manifests/fingerprints. One intentional within-pack, same-source
single-number contrast was manually accepted and documented. Phase 3H-A
changes no production code or frozen code/prompt/config/ownership hash and
makes zero model calls.

## Phase 3H-B v4_zh single-human gold

The completed blind reviewer packet was first copied byte-for-byte to the
private external review-record directory, then the repository packet was
restored to its original blank SHA-256. The external record passed exact
72-case ID/source/claim matching, strict relation enumeration, non-empty
notes, reviewer provenance, and evidence validation. Its final labels contain
11 entailed, 11 partially entailed, 11 contradicted, 15 not-in-source,
12 source-unavailable, and 12 not-checkable cases.

`holdout_single_human_zh.jsonl` contains 33 continuous canonical-source
evidence spans and 39 required null spans. Reviewer identity, reviewer ID,
and full notes remain outside the repository; public rows retain only the
external record hash and generic single-human provenance. The rich source
snapshot remains byte-identical, while a mechanically derived
`SourceFixture` JSONL lets the existing dataset loader consume the gold
without a production-code change.

The review manifest marks the benchmark single-human complete, internal-gate
eligible, non-public, blind to the evaluation model, and `not_run`. The
original one-time gate preregistration remains unchanged. No model call or
v4_zh execution occurred during annotation freeze.

## Phase 3H-C final v4_zh gate

The one authorized v4_zh execution failed closed during the third
`single_agent_live` call, after two completed predictions. The response had
passed JSON/Pydantic validation, but the frozen post-schema path raised an
untyped local `ValueError`. The safe artifact retained no model content and
could not distinguish the precise local condition. Execution stopped after
3 calls and 5,444 tokens; no complete metrics were generated.

v4_zh is consumed and cannot be rerun or used for tuning. Phase 3 is
`completed_with_known_limitations`, not passed, and `phase4_eligible=false`.
No formal blind-holdout F1 was measured. No v5 will be created.

## Phase 4A reliability and SARIF engineering continuation

Phase 4A proceeds only as engineering continuation with the limitation above;
it does not change the Phase 3 or eligibility status. Post-schema local
invariants now use payload-free typed errors and allowlisted codes. A
schema-valid but wrong relation on an available source remains a scoreable
prediction. Forged evidence, unavailable-source metadata conflict,
deterministic guard/scope failures, telemetry divergence, and invalid
canonical assembly remain fail-closed and do not trigger schema retry.

`evidencetrace check <path> --sarif <output>` derives deterministic SARIF 2.1.0
from canonical `audit.json` using the existing policy mapping. The checked-in
workflow uses minimum `contents: read` and `security-events: write`
permissions, uploads a generated SARIF after policy failure, and then restores
the EvidenceTrace exit status.

The failed holdout was pair-level: atomic claims were supplied directly and
Claim Miner did not participate. Full Claim Miner plus Judge Multi-Agent value
still requires a separately designed full-document benchmark; Phase 4A does
not provide that evidence.

## Phase 4B full-document benchmark readiness

`eval_sets/full_document_v1/` is a deterministic, author-constructed
provisional benchmark for the complete path:

```text
Markdown
  -> Claim Miner
  -> local citation/source binding
  -> Adaptive Router or Retrieval
  -> Claim Judge
  -> policy decision and SARIF
```

It contains 24 Markdown documents, split into 12 dev and 12 test documents,
with 120 exact-span gold claims and 72 frozen local source fixtures. Short,
medium, and long document strata each contain eight documents. The six
relations all have support, and the corpus includes shared citations,
multiple citations for one claim, no citation, duplicate text, cross-line
claims, multiple facts in one paragraph, unavailable sources, and long-source
distractors. This dataset is not blind, not human-reviewed, and not public
benchmark eligible. Its `.test` URLs are fixture identifiers and must not be
fetched.

The two registered live baselines are:

- `single_agent_document_live`: one primary structured-output call per
  document performs extraction, citation binding, and verdict prediction.
- `miner_adaptive_judge_live`: the existing paragraph-scoped Claim Miner
  extracts continuous spans; local code attaches IDs, source locations, and
  citation bindings; the existing Adaptive pair verifier selects
  deterministic, full-context, or retrieval-backed Judge handling.

Only Miner and Judge are Agents. Router and Retrieval are deterministic
components. Both baselines use strict Pydantic output, existing schema-only
recovery, bounded calls/tokens, canonical `AuditArtifact`, existing policy,
and the existing SARIF renderer.

Registered metrics include exact and relaxed claim extraction, atomicity,
citation and line binding, fixed-six relation quality, evidence quality,
end-to-end claim-plus-relation F1, document policy accuracy, handoff failures,
stage-specific calls/tokens/latency, and baseline deltas. Missed gold claims
remain failures in relation and end-to-end metrics.

Deterministic fake-model tests complete both systems across all 24 documents.
Those perfect fixture scores validate contracts and metric arithmetic only;
they are not live quality measurements or formal F1.

The separately authorized Phase 4B live dev attempt stopped during the first
`single_agent_document_live` smoke document. The initial strict-schema response
and its one permitted schema-only recovery both failed validation. The attempt
therefore ended after two provider calls and before the Multi-Agent smoke or
the 12-document dev run. No quality metric was generated, and the test split
was not loaded, parsed, or run. See
[`docs/phase4b-live-dev.md`](docs/phase4b-live-dev.md). Phase 3 remains
`completed_with_known_limitations` and `phase4_eligible=false`.

## Phase 4B.1 provider contract smoke

Full-document calls now use the project provider ceiling of 8,192 output
tokens, a smaller model-owned schema, 240-character reasons, strict required
fields and extra-field rejection, request provenance, cache-key invalidation,
and payload-free JSON/Pydantic diagnostics. The historical Phase 4B failure
still cannot be attributed beyond `unknown_schema_failure`.

The separately authorized one-document dev smoke completed both registered
paths. The document Single Agent had one first-attempt schema validation
failure and recovered with its single schema-only retry. Four Miner calls and
five Judge calls all succeeded on their first attempt. Combined usage was 11
provider attempts and 18,597 reported tokens; all finish reasons were `stop`,
with no suspected truncation or unrecovered failure. The test split was not
loaded, parsed, or run.

This is operational smoke evidence, not a full-dev quality result and not
evidence of Multi-Agent superiority. Complete dev execution requires separate
authorization. Phase 3 remains `completed_with_known_limitations`, and
`phase4_eligible=false`.
