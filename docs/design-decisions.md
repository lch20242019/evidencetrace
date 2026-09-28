# Design decisions and plan deviations

This log records decisions that differ from, or make explicit an ambiguity in,
the project plan. Decisions are append-only; later reversals receive a new
entry.

## DD-0001 — keep the colliding name only as a local working name

- Date: 2026-07-10
- Phase: 0
- Status: accepted for local implementation; blocked for publication
- Plan context: the plan calls EvidenceTrace a working name and requires a name
  availability check before coding/publication.
- Evidence: PyPI returned 404 for the normalized name, while GitHub returned two
  exact public repository-name matches, including an overlapping verification
  project.
- Decision: keep the user-requested `/data/yz_data/evidencetrace` directory and
  provisional package import name during Phase 0–1. Do not present the public
  repository or Action name as available.
- Impact: no change to the product Idea or technical scope; public branding and
  release remain blocked pending maintainer direction and legal review.

## DD-0002 — exactly three Markdown input fixtures

- Date: 2026-07-10
- Phase: 0
- Status: accepted
- Plan context: Phase 0 requires a bad document; the implementation request also
  asks for bug, simple-work, and complex-work Markdown documents.
- Decision: maintain exactly three input fixtures in `examples/`: the bad demo,
  a simple work document, and a complex work document. Expected audit and
  project documentation are artifacts, not input fixtures.
- Impact: the simple and complex documents become reusable Phase 1 parser
  fixtures without adding more work-document Markdown files.

## DD-0003 — reserved `.invalid` URLs plus an offline manifest

- Date: 2026-07-10
- Phase: 0
- Status: accepted
- Plan context: the demo must be deterministic and require no API key.
- Decision: fixture citations use `fixtures.evidencetrace.invalid`, with fixed
  source text and availability state in `examples/evidence/source_manifest.json`.
- Impact: Phase 1 parses ordinary HTTP(S) citation strings. Resolving those URLs
  to offline content remains Phase 2 work and is not implemented early.


## DD-0004 — deterministic footnote scanner instead of a plugin dependency

- Date: 2026-07-10
- Phase: 1
- Status: accepted
- Plan context: footnotes are required in the deterministic parser, while the
  base markdown-it package does not implement the extension.
- Decision: scan footnote definitions and occurrences over markdown-it block
  ranges with a small source-preserving parser. Do not add
  `mdit-py-plugins` in the base package.
- Impact: exact source spans remain available without expanding the runtime
  dependency set. Later CommonMark extension behavior must be covered by new
  fixtures before changing this choice.

## DD-0005 — no new-side context for deletion-only hunks

- Date: 2026-07-10
- Phase: 1
- Status: accepted
- Decision: preserve a deletion hunk's `deletion_anchor` as metadata, but leave
  `ChangeSelection.changed_ranges` empty when no new-side line exists. Selecting
  a neighboring paragraph is context strategy and belongs to a later phase.


## DD-0006 — stdlib HTML extractor in Phase 2

- Date: 2026-07-10
- Phase: 2
- Status: accepted
- Plan context: the technology section names trafilatura for HTML main-content
  extraction.
- Decision: use a small stdlib `HTMLParser` extractor for static HTML instead
  of adding trafilatura in this environment.
- Reason: it keeps the MVP dependency set small, supports deterministic unit
  tests, and covers the required script/style/navigation removal and
  heading/paragraph source locators. It does not claim full browser-quality
  extraction.
- Impact: dynamic/login pages remain unsupported; replace or augment the
  extractor only with retrieval eval evidence in a later phase.

## DD-0007 — reject credential-bearing citation URLs before HTTP

- Date: 2026-07-10
- Phase: 2
- Status: accepted
- Plan context: API keys, Authorization values, and sensitive headers must not
  enter logs or artifacts.
- Decision: reject URL query parameters such as `api_key`, `token`,
  `Authorization`, and cloud-provider signatures before a request enters httpx.
- Reason: client logging can expose complete request URLs before a result can be
  redacted. Rejecting is safer than retaining a secret-bearing citation URL.
- Impact: signed/authenticated URLs are intentionally unsupported in Phase 2;
  the finding becomes `source_unavailable` rather than exposing a secret.

## DD-0008 — FTS5 primary with tested in-memory BM25 fallback

- Date: 2026-07-10
- Phase: 2
- Status: accepted
- Decision: use SQLite FTS5 when available and an in-memory lexical/BM25-style
  fallback when SQLite lacks FTS5.
- Reason: this preserves citation-first retrieval without vector dependencies
  or embedding credentials.
- Impact: the fallback is deliberately small and suitable only for the
  per-source chunk volumes of this MVP; no dense retrieval was added.

## DD-0009 — render only after rereading canonical audit.json

- Date: 2026-07-10
- Phase: 2
- Status: accepted
- Decision: write and schema-read `audit.json` before rendering `audit.md` and
  terminal text.
- Reason: this makes the JSON artifact the actual state source rather than
  merely a parallel serialization of in-memory state.

## DD-0010 — code-budget warning

- Date: 2026-07-10
- Phase: 2
- Status: accepted with warning
- Decision: record the physical-line count above the plan's core-package target
  rather than compressing security and contract tests prematurely.
- Reason: safe fetch, source contracts, and narrow agent validation are
  high-value MVP safeguards. Future phases must delete/defer scope rather than
  extend the package casually.

## DD-0011 — local Git initialization and duplicate-module cleanup

- Date: 2026-07-10
- Phase: 2.5
- Status: accepted
- Decision: initialize an empty local Git repository only when the checkout
  has no `.git/`; do not add a remote or create a commit. Remove unreferenced
  root-level copies of packaged Phase 2 modules and a compared `.orig` backup.
- Reason: repository hygiene and reproducibility require local Git metadata,
  while duplicate modules are not included by the `src/` package discovery and
  can create ambiguous imports. The backup had no unique behavior.
- Impact: public package imports and Phase 2 behavior are unchanged. Publication
  and remote collaboration remain outside this checkpoint.

## DD-0012 — frozen fictional seed data with provisional difficult cases

- Date: 2026-07-10
- Phase: 3A
- Status: accepted
- Decision: use 80 fictional offline claim-source pairs with checked source
  hashes and a checked-in case-hash map. The 30 natural and 30 single-slot
  mutation cases are `deterministic_gold`; all 20 difficult cases are
  `provisional` pending human review.
- Reason: no second annotator was available in this implementation environment.
  Calling model- or author-generated difficult labels human gold would
  overstate benchmark quality.
- Impact: provisional metrics are reported separately and never enter headline
  or stop-gate metrics. There are initially zero `human_reviewed` cases.

## DD-0013 — macro-F1 averages labels present in gold or predictions

- Date: 2026-07-10
- Phase: 3A
- Status: accepted
- Decision: keep a complete six-label confusion matrix and per-label metrics,
  while relation macro-F1 averages labels with gold or predicted support in the
  selected evaluation subset.
- Reason: averaging absent labels as zero would cap a subset below the 0.70
  stop gate even with perfect predictions. Zero-denominator per-label values
  remain explicit zeroes.
- Impact: the metric is reproducible and its label support is visible in
  `metrics.json`; comparisons must use the same split and annotation filter.

## DD-0014 — live baseline replaces only the eval miner-plus-judge client

- Date: 2026-07-10
- Phase: 3A
- Status: accepted
- Decision: offline evaluation always runs all three deterministic baselines.
  Explicit `--live --model <id>` swaps an OpenAI-compatible `ModelClient` into
  the eval-only miner-plus-judge baseline. Missing credentials are a recorded
  clean skip, not a passing live result.
- Reason: this preserves identical case/source inputs and keeps the product's
  default Phase 2 architecture unchanged. It adds no second provider.
- Impact: live cost remains `null` until real usage/token accounting is
  available; no price or cost is fabricated.

## DD-0015 — keep the plan budget unchanged after Phase 3A audit

- Date: 2026-07-10
- Phase: 3A
- Status: accepted with warning
- Decision: do not edit the source-of-truth budget. Record the local
  recommendation of 7,500 production and 7,500 test/eval physical lines in
  `docs/code-budget-review.md`. Retain the eval runtime at 1,286 lines, 86 over
  its approximate target, rather than remove frozen/composite hashing, artifact
  schemas, error attribution, annotation separation, or live-skip contracts.
- Reason: the excess is evaluation integrity and reproducibility scope, not a
  new product feature. Clear dead duplicates and unused wrappers were removed
  before accepting the warning.
- Impact: Phase 4 and 5 have consumed no budget and remain out of scope. Any
  later increase must be justified against the unchanged plan and stop gate.

## DD-0016 — Phase 3A local checkpoint commit

- Date: 2026-07-10
- Phase: 3B checkpoint 0
- Status: accepted
- Decision: after the clean 274-test/compileall baseline, create local commit
  `49c6aef phase3a-provisional-baseline` using the already configured Git
  identity. Do not change local/global identity and do not add a remote.
- Reason: Phase 3B changes benchmark semantics; a content-addressed Phase 3A
  state is needed before replacing artifact contracts.
- Impact: `eval_sets/core*` and `eval_runs/core/` are additionally guarded
  by exact SHA-256 tests.

## DD-0017 — semantic baseline names and dual metric denominators

- Date: 2026-07-10
- Phase: 3B
- Status: accepted; supersedes DD-0013/DD-0014 for new runs
- Decision: rename deterministic comparisons to `lexical_rules`,
  `lexical_full_source`, and `miner_judge_deterministic`. Reserve
  `single_agent_live` and `miner_judge_live` for real client calls and show
  them separately. Pair-level extraction is not applicable. Report
  observed-gold-label macro-F1 and fixed six-label macro-F1 together, plus
  weighted-F1, balanced accuracy, support, and coverage.
- Reason: lexical/fake execution cannot establish Single-Agent versus
  Multi-Agent model quality, and a supplied claim cannot measure extraction.
- Impact: legacy artifacts keep their names; only new schemas/runs use the
  corrected taxonomy.

## DD-0018 — shared hard-conflict and caution contracts

- Date: 2026-07-10
- Phase: 3B
- Status: accepted
- Decision: aligned numeric/date/version/entity/negation conflicts are typed
  errors consumed identically by deterministic and live Judge paths. Explicit
  opposing qualifiers are also blocking; incomplete qualifiers/conjuncts and
  comparisons are cautions that prefer partial support or cap confidence.
- Reason: the prior negation warning/error mismatch silently disabled a
  required guard. High lexical overlap is insufficient for high-confidence
  entailment.
- Impact: one Phase 2 test expectation changed from warning to error, while the
  public signal and verdict schemas remain source-grounded and validated.

## DD-0019 — v1 diagnostic contamination and synthetic v2 review packet

- Date: 2026-07-10
- Phase: 3B
- Status: accepted
- Decision: classify any new v1 test result as
  `diagnostic_contaminated`; do not use it as a release gate. Prepare 60
  offline synthetic v2 candidates with label proposals hidden from the review
  packet and blank human fields. Do not execute the holdout.
- Reason: v1 failures were already published to developers. Network collection
  was unavailable, and author/model proposals cannot be called human gold.
- Process note: legacy runner unit tests initially exercised v1 all-split
  behavior in temporary directories during integration. Their predictions were
  not inspected or retained; the tests were then changed to dev-only. This
  prevents an inaccurate claim of renewed v1 blindness.
- Impact: formal Phase 3 requires independent annotation and an uncontaminated
  holdout. v2 remains provisional and synthetic.

## DD-0020 — dev code freeze and Phase 3B budget warning

- Date: 2026-07-10
- Phase: 3B
- Status: accepted with warning
- Decision: after the dev gate passed, freeze the judgement/eval/CLI files at
  `eb5a01fa0065a5b6643c4c1d60cca746d1f626015e54ce0d4382343859120f62`.
  Accept a net Phase 3B production increment of 896 lines, 96 over the
  approximate target, without changing the source-of-truth budget.
- Reason: the excess implements the required live comparison separation,
  validity-aware metrics/gates, and shared deterministic/live conflict guards.
- Impact: the v2 manifest records the hash and remains `not_run`. Phase 4 is
  still blocked; passing dev does not make this a release benchmark.

## DD-0021 — single-human freeze with separate model provenance

- Date: 2026-07-14
- Phase: 3C
- Status: accepted; formal Phase 3 gate remains blocked
- Decision: allow `single_human_review_complete` only after the 60-case human
  record passes ID/source/hash, relation, note, and exact-substring evidence
  validation and has truthful, complete reviewer name, ID, date, guide version,
  attestation, and explicit `viewed_model_review` metadata. Do not default that
  field to false. Treat underscore-only evidence for non-evidence relations as
  empty and require cases `v2_holdout_041` through `v2_holdout_050` to remain
  `source_unavailable`.
- Provenance: preserve the former model-filled Reviewer B record under an
  explicit `model_m2` name, restore a blank Human Reviewer B packet, and leave
  adjudication blank. The human A record is the sole source of final relation
  and evidence. Model notes remain separate and may be referenced only when the
  human and model relations agree; they are never merged into human notes.
- Reason: a truthful `viewed_model_review: true` records exposure before the
  provenance freeze without converting a genuine human annotation into a model
  review. It also does not create a second independent human annotation.
- Impact: the result is at most `single_human_review` with human-model
  comparison. It must not be called double-human review, inter-human agreement,
  human Cohen's kappa, or adjudicated human gold. The v2 holdout is not run;
  the formal Phase 3 gate and Phase 4 remain blocked pending a second independent
  human review and required live-model baseline evaluation.
- Supersession: DD-0022 later removes only the second-human requirement from
  the internal development gate. All provenance and public-claim limits here
  remain in force.

## DD-0022 — single-human synthetic holdout for the internal Phase 3 gate

- Date: 2026-07-14
- Phase: 3D
- Status: accepted for internal evaluation only; Phase 4 remains blocked until
  the live holdout gate is actually evaluated
- Decision: derive a 60-case frozen test dataset exclusively from the validated
  Reviewer A relation and evidence fields. Give it the distinct benchmark
  validity `single_human_synthetic_holdout`, annotation status
  `single_human_review`, and `public_benchmark_eligible: false`. Do not import
  candidate model proposals, model-review relations or notes, or private human
  identity fields into gold data.
- Gate: after a successful live dev preflight and a clean code/prompt/data
  freeze, run the holdout at most once. Only `miner_judge_live` relation
  macro-F1 >= 0.70 can pass the internal Phase 3 gate. All six labels must have
  support and substantive verdict evidence must remain a literal source
  substring. Keep the unchanged v0.1 targets of 0.75 macro-F1 and 0.80
  contradiction recall separate.
- Human-review tradeoff: a second independent human is no longer an internal
  development blocker. It remains a recommended future quality enhancement and
  is still required before claiming double-human review, inter-human agreement,
  or adjudicated human gold. The blank Reviewer B and adjudication records stay
  preserved.
- Live boundary: compare `single_agent_live` and `miner_judge_live` with the
  same provider, model, and temperature. Do not run without credentials, an
  explicit model ID, a successful dev preflight, authorization for up to about
  180 calls and cost, and a clean frozen worktree. Missing conditions produce a
  ready checkpoint, not a passing or skipped-as-passed evaluation.
- Impact: synthetic wording, repeated offline sources, and one human reviewer
  prevent public benchmark claims. A score below 0.70 keeps Phase 4 blocked and
  requires a new v3 holdout for any post-holdout tuning; a score at or above
  0.70 makes Phase 4 internally eligible but does not remove these limitations.

## DD-0023 — failed live dev attempts do not inherit fresh call authorization

- Date: 2026-07-15
- Phase: 3D live dev preflight
- Status: accepted; the one explicitly authorized rerun was consumed
- Observation: the first `deepseek-v4-flash` dev comparison failed closed on
  case `v2_dev_008` before artifacts were written. Failed-run telemetry was not
  persisted, so the actual provider-call count is unknown; fixed execution
  order bounds it at 26–33 calls.
- Root cause: percentage text such as `1.8%` was tokenized as `1.8` by the
  protected-token regex but as `1.8%` by the normal tokenizer. This rejected a
  verbatim model claim. The general fix prioritizes percentage tokens and makes
  protected-fact validation fail closed even when a model returns empty claims.
- Scope discipline: no dev/test/holdout label, evidence, retrieval rule, Judge
  rule, task prompt, or model output was changed. The fix has newly written
  regression cases for verbatim decimal percentages and empty-claim bypass.
- Cost boundary: at that point a complete rerun could consume 54 more calls and
  bring the cumulative dev total to 80–87. The owner granted one new 54-call
  authorization; it was consumed by the schema-failed attempt and cannot be
  reused. Holdout authorization remains separate and capped at 180 calls.
- Provider configuration: the adapter requested temperature 0.0 while DeepSeek
  V4 used its provider-default thinking mode. Reports describe 0.0 as the
  requested value and do not assert effective deterministic sampling. Any
  future dev attempt requires a new authorization and a frozen configuration.
- Impact: no successful live baseline metric or gate artifact exists. The v2
  holdout remains unexecuted and Phase 4 remains blocked.


## DD-0024 — hard live-call budget and failure-only telemetry artifact

- Date: 2026-07-15
- Phase: 3D authorized dev rerun
- Status: accepted, frozen, and exercised by the one authorized execution
- Decision: enforce the calculated 54-call dev ceiling inside the shared model
  client before every provider POST. Failed provider requests consume one slot;
  the blocked next request consumes none. No automatic transport retry or eval
  rerun is permitted.
- Failure artifact: on any live transport, schema, grounding, or local contract
  failure, atomically write only `failed_attempt.json` and re-raise. Do not write
  the four canonical success artifacts for a partial run. Persist per-call safe
  usage/latency/failure events and the failed baseline/case, but never exception
  text, credentials, headers, base URLs, requests, responses, prompts, or source
  payloads.
- Configuration semantics: both live baselines share one provider/model client
  and provider-default thinking. Temperature 0.0 is recorded as requested;
  actual effective temperature remains null with status unknown. Automatic retry
  count is explicitly zero.
- Impact: the new code and prompt-bearing hashes must be frozen before the one
  authorized dev execution. A failed attempt cannot be automatically repeated.


## DD-0025 — authorized dev rerun stopped on provider schema failure

- Date: 2026-07-15
- Phase: 3D live dev preflight
- Status: recorded failure; no automatic rerun and no holdout execution
- Execution: run `deepseek-v4-flash` once with provider-default thinking and
  requested temperature 0.0 from frozen commit
  `bcda80b9ea9c21d46bda05fa2160b88e19f60fa0` and code bundle
  `66284bd98b75f7ea1a7160bcf9d8662d2b37401f988378e92264a33df18e9df2`.
- Result: stop on `v2_dev_001` in the first `single_agent_live` call because
  provider output failed the required Pydantic schema. Reported usage is 259
  input, 391 output, and 650 total tokens; p50/p95 latency is 3,571.082 ms;
  there is one schema failure, zero transport failures, and zero retries. Cost
  is null because no explicit price snapshot was supplied.
- Failure provenance: retain only
  `eval_runs/phase3d_v2_dev_live_rerun/failed_attempt.json`, SHA-256
  `e6b7fda4daefe7d3f98911a238c7fdc492c71ab57a4a5d39406274d5f43a640c`.
  No canonical success artifact, request/response content, credential, header,
  exception message, or complete dev metric is present.
- Interpretation: this is an adapter/output-schema preflight failure, not a
  model-quality result and not a Phase 3 gate result. Do not alter prompt,
  schema, rule, or label after this consumed attempt. Another dev attempt needs
  new authorization. The v2 holdout remains unauthorized and unexecuted;
  no Phase 3 gate result exists, and Phase 4 remains blocked.


## DD-0026 — schema-aware JSON mode with bounded safe diagnostics

- Date: 2026-07-15
- Phase: 3D.1 offline provider-contract remediation
- Status: accepted and offline-frozen; later one-call smoke passed
- Problem: the historical DeepSeek attempt proves only that
  `SingleAgentLiveOutput` validation failed. Its raw content and Pydantic
  details were intentionally not retained, so attributing the failure to any
  specific field would be fabrication. The prior adapter also learned the
  target schema only after the provider call.
- Decision: make `complete_model` schema-aware before POST. Send JSON-object
  mode, target JSON schema, one valid example, and bounded output tokens.
  Require a single unfenced object in both system and user instructions, then
  validate the original content with strict Pydantic JSON parsing.
- Failure policy: empty, non-string, invalid/fenced, truncated, missing,
  additional, wrong-type, enum, and other constraint failures stop after the
  original call. Do not strip fences, translate relations, accept aliases,
  coerce wrong types, retry transport, or ask a model to repair its output.
- Diagnostics policy: a future v2 failure artifact may contain only code-owned
  schema names, static categories/messages, at most eight Pydantic loc/type
  pairs extracted without input or URL, a sanitized finish reason, and content
  length. Unknown provider-controlled path components become
  `unknown_field`; unknown schema/error names and finish reasons become
  fixed safe placeholders. Static diagnostic models reject untrusted field
  text, and propagated schema/transport exceptions are detached from original
  causes and contexts. Never persist provider content, reasoning, Pydantic
  `msg`/`ctx`/`input`, prompts, claims, sources, headers, or credentials.
- Compatibility: continue accepting the historical v1 artifact without
  diagnostics, explicitly reject any invented v1 diagnostic, and do not rewrite
  the historical file. Preserve call/token/latency telemetry,
  hard call limits, zero retry, Agent isolation, relation/evidence schema
  strictness, and provider-default thinking. Requested temperature 0.0 is not
  represented as effective provider temperature.
- Scope and interpretation: this is instrumentation and wire-contract
  hardening, not semantic prompt tuning or evidence that model quality
  improved. The freeze itself made zero real API calls. Its later separately
  authorized one-case smoke passed, as recorded in DD-0027; full dev and
  holdout execution remain separately blocked.

## DD-0027 — one-case structured-output smoke passed without widening scope

- Date: 2026-07-15
- Phase: 3D.1 live provider-contract smoke
- Status: accepted and frozen; authorization consumed
- Authorization boundary: execute exactly one `deepseek-v4-flash`
  `single_agent_live` call for `v2_dev_001`, with `max_calls=1`, zero retry,
  zero Miner/Judge calls, no full dev split, and no holdout. Use the direct
  `run_baseline` library path because the full eval runner would exceed scope.
- Freeze: execute from commit
  `291314a59f949f676da1d9a10501409382567b9c`, code hash
  `79fc7fa4ec8d886556baa8406c0d61253abbcc5010b1ae0a8cd27da26a9ca308`,
  and prompt hash
  `43e4f32204726f608e0eed44d5c0f3e4779e01984f15d0ecdb14ad27add6326e`.
- Result: one provider call, contract passed, dev relation matched gold, and the
  evidence passed exact substring grounding. Usage was 553 input, 177 output,
  and 730 total tokens; latency was 2,151.273 ms and finish reason was `stop`.
  Cost is null because no explicit price snapshot was supplied.
- Artifact policy: persist only the fixed single-file smoke artifact and its
  hash. Store configuration, frozen identifiers/hashes, safe telemetry,
  contract/quality/grounding booleans, and the evidence hash. Do not persist
  model content, reasoning, reason, relation labels, prompt, claim/source text,
  base URL, request/response headers, request ID, or credentials.
- Interpretation: this validates the provider contract once; it is not a full
  live dev comparison, a benchmark score, or a Phase 3 gate. The one-call
  authorization is consumed. Full dev and holdout still require separate new
  authorization, and Phase 4 remains blocked.

## DD-0028 — isolated Miner/Judge contract smoke completes schema coverage

- Date: 2026-07-15
- Phase: 3D.1 live provider-contract smoke
- Status: accepted and recorded; the two-call authorization is consumed
- Authorization boundary: execute only `v2_dev_001` through
  `miner_judge_live`, with at most one Miner request followed by at most one
  Judge request, a shared `max_calls=2` ceiling, and zero retry, repair,
  single-agent, full-dev, or holdout calls.
- Result: Miner and Judge each made one `deepseek-v4-flash` request and each
  passed its strict JSON/Pydantic and local scope contracts. The final relation
  matched dev gold and the evidence passed candidate-substring validation.
  Aggregate usage was 1,627 input, 1,010 output, and 2,637 total tokens; cost is
  null because no explicit price snapshot was supplied.
- Artifact boundary: the single safe smoke artifact records fixed identifiers,
  hashes, contract/grounding booleans, bounded usage, latency, finish reason,
  and failure counts. It does not contain raw model content, reasoning, prompts,
  claims, sources, evidence text, headers, credentials, or Pydantic input
  values. Its SHA-256 is
  `f01d4cfdb33a2923c5a2ca72b5235b0ee2e94368a60a41fa908b6fe870a273cd`.
- Freeze and interpretation: production code and prompts remain unchanged at
  code hash
  `79fc7fa4ec8d886556baa8406c0d61253abbcc5010b1ae0a8cd27da26a9ca308`
  and prompt hash
  `43e4f32204726f608e0eed44d5c0f3e4779e01984f15d0ecdb14ad27add6326e`.
  Together with DD-0027, all three live output schemas now have isolated smoke
  coverage. This is not a complete live dev comparison, holdout execution,
  model-quality result, or Phase 3 gate. Full dev needs fresh authorization;
  the holdout remains unexecuted and Phase 4 remains blocked.

## DD-0029 — complete live dev is provisional evidence, not the holdout gate

- Date: 2026-07-19
- Phase: 3D live dev comparison
- Status: accepted and frozen; the 54-call authorization is consumed
- Execution boundary: run the 18-case `dev` split exactly once at commit
  `57aac924abb81ebf8f522c3658ff18d3f6112077`, with
  `deepseek-v4-flash`, provider-default thinking, a pre-request hard ceiling of
  54 calls, and zero retry, repair, automatic rerun, or holdout access. Code,
  prompt, dev labels/gold, and frozen holdout data remain byte-stable.
- Result: the run completed in 48 calls: 18 Single Agent, 18 Miner, and 12
  Judge. Both live baselines produced 0.720635 observed/fixed-six macro-F1,
  1.0 contradiction recall, 1.0 entailed precision, 0.8 partial-support F1,
  and 0.066667 false-block rate. Miner/Judge improved source-span token F1 by
  0.028509, but used 12 more calls and 29,675 more tokens, had 2,926.753 ms
  higher p50 latency and 4,575.226 ms higher p95 latency, and had 0.008547
  higher high-confidence error. It did not improve relation metrics on dev.
- Contract result: all 90 baseline-case records completed; deterministic calls,
  schema failures, transport failures, retry, and repair were zero. All
  substantive evidence spans passed source-substring validation. Guard failures
  are recorded as a derived zero completion invariant because the success
  schema has no separate guard-failure field.
- Cost and provenance: canonical cost stays null without an explicit price
  snapshot and cache hit/miss accounting. Four canonical artifacts are frozen
  under bundle SHA-256
  `5ee8691195ebd97927f48de2e1cebd493ecbfb2314e99eb241e59e339a208411`;
  no credential appeared in either exact-key or generic-pattern scans.
- Gate interpretation: 0.720635 meets the provisional 0.70 dev threshold but
  misses the unchanged 0.75 v0.1 macro-F1 target. Dev validity is provisional,
  so this is neither a formal Phase 3 pass nor public benchmark evidence. Phase
  3 still requires the one-time frozen holdout; Phase 4 remains blocked.

## DD-0030 — a fail-closed v2 holdout attempt is consumed and cannot be rerun

- Date: 2026-07-20
- Phase: 3D one-time frozen holdout
- Status: accepted failure provenance; Phase 3 gate failed; Phase 4 blocked
- Frozen boundary: execute the 60-case single-human synthetic holdout once from
  commit `34f4077c6a38a3dde98560530a2a3ab35fa3c483`, code hash
  `79fc7fa4ec8d886556baa8406c0d61253abbcc5010b1ae0a8cd27da26a9ca308`,
  prompt hash
  `43e4f32204726f608e0eed44d5c0f3e4779e01984f15d0ecdb14ad27add6326e`,
  and dataset file SHA-256
  `0d19004b627f4ecabf9dd40fd28303b0539b89f36f3f7f5cec135c47a6582c75`.
  Use at most 180 calls with zero retry/repair and no adaptive changes.
- Outcome: after all 60 Single Agent calls and four Miner/Judge calls, the
  second Judge result triggered the deterministic-conflict guard on
  `v2_holdout_002`. The attempt stopped after 64 calls, with 61 completed live
  predictions, zero schema/transport failures, and one local guard failure.
- Safe failure record: preserve only `failed_attempt.json`, SHA-256
  `e7ea20d1b7d18e4a1472c14f5053440e342c8b44ec89d49998fc1fce7337fa60`.
  It contains safe telemetry and bounded identifiers but no raw model content,
  claim/source/evidence, prompt, header, credential, or exception input.
- Metrics policy: the fail-closed runner removed all partial success artifacts.
  Therefore quality metrics, confusion matrices, span coverage, and live
  baseline deltas are unavailable and must not be recovered by rerunning or
  fabricated from memory. Canonical cost remains null.
- Gate policy: v2 is consumed despite incomplete Miner/Judge execution. It
  fails 60/60 completion and zero-guard-failure requirements and cannot be
  reused for a formal gate. Remediation is dev-only; any later gate requires a
  new v3 holdout. Phase 4 remains blocked.
- Scope limitation: this pair-level holdout directly provides atomic claims.
  It cannot evaluate claim extraction or prove Miner value. Full-document
  extraction and Scout/Challenger ablations remain separate future work and are
  not implemented by this checkpoint.

## DD-0031 — entity hard conflicts require absence from bounded context

- Date: 2026-07-21
- Phase: 3E offline deterministic-guard remediation
- Status: accepted and offline-frozen; complete dev validation remains pending
- Evidence boundary: the historical v2 failure artifact proves only a generic
  local deterministic conflict and remains unchanged. The more specific cause
  was established through the project owner's confirmed context and a new
  generic offline reproduction; it is not retroactively attributed to fields
  that the safe artifact did not preserve.
- Problem: entity detection compared a Claim with only the most-aligned
  evidence clause. A neighboring sentence could establish the Claim entity,
  while a sentence-initial account or tier descriptor in the aligned clause
  was treated as a different entity. This produced an error-level false
  conflict even though the entity remained present inside the Judge's bounded
  source context.
- Decision: `entity_mismatch` is a hard conflict only when the Claim has an
  entity, the aligned clause names a disjoint entity, the Claim entity is
  completely absent from the full bounded evidence context, and predicate
  context is sufficiently aligned. The bounded context is the deduplicated
  source chunk text of the top-k candidates already in `JudgeInput`; it does
  not grant the Judge document-wide access.
- Producer consistency: Pipeline, eval Miner/Judge, and the Judge's own derived
  signals pass the same bounded context. Clause selection and every
  numeric/date/version/negation/qualifier/conjunction rule remain unchanged.
  Retrieval windows and evidence/source/citation/Claim scope guards also
  remain unchanged.
- Tradeoff: if the Claim entity appears anywhere in the already-bounded top-k
  context, this policy conservatively abstains from an entity hard conflict.
  That can reduce some entity-conflict recall, but it follows the explicit
  requirement that hard rejection needs complete bounded-context absence.
  Clearly aligned cross-entity statements still emit an error and block an
  `entailed` live verdict.
- Safe error contract: replace only the deterministic-entailment guard's
  generic `ValueError` with `DeterministicConflictError`. The exception has
  a fixed message and carries only sorted, unique allowlisted codes. A future
  `phase3e-live-failure-v3` artifact may persist those codes; historical v1
  and v2 artifacts may not. Other unknown `ValueError` and `RuntimeError`
  failures retain the existing fail-closed path.
- Versioning: name this semantic policy `deterministic-signals-v2` and add it
  as an independent cache-key dimension. This invalidates both legacy
  five-dimension keys and v1 policy keys. There is currently no production
  cache call site, so this remains an enforced key/storage contract rather
  than runtime cache wiring. Future eval manifests record the policy version.
- Contract separation: keep the structured-output contract at
  `openai-compatible-v2`. Output schemas, examples, provider, retry policy,
  and model prompts are not tuned in this remediation. The prompt bundle hash
  changes only because Judge/baseline orchestration code changed.
- Benchmark consequence: mark v2
  `diagnostic_consumed_guard_failure`. Do not rerun it, reconstruct deleted
  outputs, or infer missing metrics. Do not create v3 before the repaired
  policy is validated on full dev. Phase 3 remains unpassed and Phase 4
  blocked.
- Scope: this checkpoint made zero model/API calls and added no dependency,
  Agent, provider, retrieval feature, Phase 4 feature, or benchmark data.

## DD-0032 — incomplete context-guard dev run remains failure provenance

- Date: 2026-07-21
- Phase: 3E authorized live dev validation
- Status: accepted failure provenance; authorization consumed; no rerun
- Execution boundary: run the unchanged 18-case v2 dev split once from commit
  `8eaa17b7e0d0ec3b907e022743f8d3dcecef1e45` with
  `deepseek-v4-flash`, provider-default thinking, a global pre-request
  ceiling of 54 calls, zero retry/repair, and no holdout access. Freeze code,
  config, prompt, labels, and gold throughout the run.
- Outcome: all 18 Single Agent predictions and 13 Miner/Judge predictions
  completed. The fourteenth Miner output failed its protected-slot scope guard
  before Judge, stopping the process after 44 calls: 18 Single Agent, 14 Miner,
  and 12 Judge. This failure is unrelated to the deterministic entity guard.
- Safe provenance: preserve only the v2-format `failed_attempt.json`, SHA-256
  `a17b64973c70da1e6b01fd9c1e673885c41f58818759232b52c27b270c204596`.
  The artifact retains call/token/latency/configuration telemetry but no raw
  Miner/model output, reasoning, Claim, source, evidence, prompt, header, or
  credential. The specific introduced slot is unrecoverable and must not be
  guessed.
- Guard interpretation: schema, transport, retry, and typed deterministic
  guard failures were zero before the Miner scope failure. This is positive
  partial evidence for the context-aware entity fix, not a successful full-dev
  validation. Only offline generic tests cover both adjacent-context
  acceptance and true cross-entity rejection across the complete policy.
- Metrics policy: the fail-closed runner removed all partial success
  artifacts. Relation metrics, per-label metrics, span F1, confusion matrices,
  and the 0.70 dev threshold are unavailable and may not be reconstructed.
  Resource summaries may be derived from safe ordered telemetry; partial
  Miner/Judge resources cannot be presented as a like-for-like improvement
  over the earlier complete run.
- Sampling caveat: requested temperature remains 0.0 but actual effective
  temperature is unknown under provider-default thinking. Any output, token,
  or latency difference from Phase 3D can reflect provider/model variability
  and cannot be attributed entirely to the entity policy.
- Versioning: retain code hash
  `41c5bf4a9f5bdd9159dad26b37eeb3688efa09db3604dfb5bb5b3057d96083cf`,
  config hash
  `071f58d6f7345a63b583686908bebd7682cd649c40dc7ad18a01a993dd7a9d4e`,
  prompt hash
  `6ccb4e34abc29819ef7fdd5d6b1648acbe8d86e96aab46168d4629dfdbd75e43`,
  signal policy `deterministic-signals-v2`, and structured-output contract
  `openai-compatible-v2`.
- Next gate: do not create v3 while full dev remains unstable. A generic Miner
  scope remediation, if authorized, must be dev-only and independently frozen
  before another live attempt. Consumed v2 remains diagnostic, Phase 3 remains
  unpassed, and Phase 4 remains blocked.

## DD-0033 — model-owned Miner drafts and pair-level Judge baselines

- Date: 2026-07-21
- Phase: 3E.2 offline contract correction
- Status: accepted; supersedes DD-0017 only for new pair-level runs
- Miner contract: a live model returns a required `LiveMinerDraftOutput.claims`
  array containing only claim `text`, `claim_type`, and `checkability`.
  Local deterministic code owns IDs, file and line mapping, citation URLs,
  model/prompt provenance, and the empty `slots` mapping. The model cannot
  return or override metadata.
- Span mapping: a linear, non-recursive matcher computes the earliest forward
  and latest reverse monotonic non-overlapping exact mappings. Only equality
  proves uniqueness; absence or disagreement fails closed.
- Slots audit: Judge, retrieval, pipeline, cache, and artifact rendering have no
  `AtomicClaim.slots` consumer. The sole production read is the Miner's local
  post-enrichment assertion that the locally written mapping remains `{}`.
  Keeping the public field preserves serialized compatibility without asking
  the model for unused structure.
- Scope failures: use payload-free `MinerScopeError` with exactly one
  allowlisted code: `non_source_span`, `ambiguous_span`,
  `missing_protected_token`, or `invalid_claim_fragment`. Do not retain or
  expose claim/paragraph/model content, and do not retry, repair, normalize, or
  guess a mapping.
- Pair baseline semantics: a claim-source pair already supplies an atomic
  claim. New runs therefore use `retrieval_judge_deterministic` or
  `retrieval_judge_live`, with local claim construction, existing retrieval,
  and Judge only. The live path makes at most one Judge call per eligible case;
  combined with `single_agent_live`, the pair-run ceiling is `2N`. Extraction
  metrics remain not applicable.
- Compatibility: legacy `miner_judge_deterministic` and `miner_judge_live`
  literals remain schema-readable for immutable historical artifacts but are
  excluded from active pair runner collections. Historical reports, hashes,
  metrics, and names are not rewritten.
- Evaluation interpretation: `miner_judge_live` is reserved for a future
  full-document flow from Markdown/paragraphs through extraction, retrieval,
  and Judge. It must be evaluated against gold claim spans/counts and
  end-to-end relation labels before any Multi-Agent benefit claim. Phase 3E.2
  records only the design and does not implement that benchmark.
- Versioning: `live-miner-draft-v2` is a separate prompt/schema and cache-key
  dimension. `deterministic-signals-v2` and `openai-compatible-v2` remain
  unchanged. Old Miner cache entries are invalid under the new contract. The
  prompt/schema freeze includes `audit_models.py` and `eval/models.py`, so a
  schema change invalidates its hash. The frozen code/prompt/config hashes are
  `309c570e…be64b`, `01dde7fb…77c0a`, and `d20df2b9…15a9`.
- Scope: zero real model/API calls; no dev or consumed-v2 rerun, no v3, no
  dependency/provider/Agent expansion, and no Phase 4 implementation. Phase 3
  remains unpassed and Phase 4 blocked.

## DD-0034 — incomplete stability series is an operational failure, not an aggregate

- Date: 2026-07-21
- Phase: 3E.2 pair-dev live stability evaluation
- Status: accepted failure provenance; no automatic retry or continuation
- Frozen boundary: execute one isolated Retrieval→Judge smoke and then up to
  three identical complete dev runs from commit
  `ac6f3ab21d8ce78dbfc2514bfabdeca9121196ae`, with the frozen Phase 3E.2
  code/prompt/config hashes, `deepseek-v4-flash`, provider-default thinking,
  zero retry/repair, 109 calls, and a 950,000-token pre-request stop line.
- Smoke decision: the successful one-call smoke establishes local
  `EvalCase`→`AtomicClaim`→retrieval→Judge wiring and zero Miner calls. It is
  a contract check, not one of the three full dev runs and not a quality gate.
- Outcome: run 01 completed and produced its canonical artifacts. Run 02
  stopped after 31 provider calls when an untyped local validation `ValueError`
  occurred on `v2_dev_017`. Schema and transport failures were
  zero. The safe failure artifact does not identify the specific validation
  condition, so the cause may not be inferred or reconstructed.
- Fail-closed decision: classify this as an operational stability failure.
  Preserve run 01 and the run-02 failure artifact, do not retry run 02, do not
  start run 03, and do not synthesize missing canonical artifacts or partial
  quality metrics.
- Aggregate validity: a stability mean/range/standard deviation and per-case
  relation consistency require all three independently completed runs. With
  only one complete run, no aggregate is generated and run 01 must not be
  selected as a representative or best run. Repeated model results, when
  available, are model self-consistency and never inter-human agreement.
- Accounting: the smoke and two attempted runs consumed 63 calls and 74,459
  tokens in total; Miner calls were zero throughout. Combined call latency
  p50/p95 was 2,811.118 / 7,038.013 ms. Operational success was 1/2 attempted
  and 1/3 planned dev runs. No retry or repair call occurred. Run 03 and all
  holdout execution remained outside the consumed authorization.
- Data boundary: only the dev dataset was selected. Its loader used the shared
  source fixtures needed for dev. The consumed v2 holdout was not loaded,
  parsed, evaluated, or run; a post-validation byte-level SHA check only
  confirmed immutability. No v3 was created.
- Gate consequence: run 01's provisional dev metrics do not pass Phase 3 or
  make Phase 4 eligible. The stopped stability series requires separate
  offline diagnosis and a newly frozen validation path. Phase 4 remains
  blocked.

## DD-0035 — separate Judge integrity failures from relation-quality errors

- Date: 2026-07-22
- Phase: 3E.3 Judge validation repair and dev stability
- Status: accepted and validated on three independent complete dev runs
- Diagnosis boundary: the Phase 3E.2 run-02 artifact retained no raw model
  output, so the exact unsaved relation is not guessed. A deterministic stub
  reproduced the structural defect: a schema-valid and in-scope
  `not_checkable` relation could be promoted to a generic fatal `ValueError`
  solely because local checkability heuristics disagreed.
- Semantic decision: relation choice is model quality. Any schema-valid,
  in-scope relation is canonicalized and scored, even when wrong. It must not
  abort the evaluation.
- Integrity decision: claim-ID mismatch, source scope, evidence-span scope,
  source availability, and missing substantive evidence remain fail closed.
  They use `JudgeScopeError` with only an allowlisted code and a fixed message.
  Factual deterministic contradictions continue to use
  `DeterministicConflictError`.
- Payload boundary: typed exceptions and failure artifacts may not contain a
  claim, source, evidence text, prompt, model output, headers, credentials, or
  arbitrary exception text. Strict JSON, zero retry, and zero repair remain
  unchanged.
- Generality: the repair has no case-ID branch. Parameterized tests cover
  subjective/best claims, subjective claims with an objective metric, legal
  partial entailment, forged evidence spans, and factual contradictions.
- Live result: five `v2_dev_017` smoke calls produced five canonical
  predictions with zero operational failures. Three complete dev runs then
  succeeded on commit `ad72cecfabf8d6969a51d739ef27b820bda4fbbc` with the
  same code, prompt/schema, config, dataset, model, temperature, and thinking
  mode. No Phase 3E.2 run was mixed into the aggregate.
- Stability interpretation: Retrieval-to-Judge macro-F1 was 0.942857 in all
  runs, but relation agreement was 17/18. `v2_dev_017` varied among
  `not_in_source` and `partially_entailed` in the three complete runs, so zero
  macro-F1 standard deviation does not imply perfect per-case stability.
- Budget and safety: smoke plus full runs used 98 calls and 117,883 tokens,
  below the authorized 150/1,000,000 ceilings. Schema, transport, guard, and
  local-validation failures were all zero. Consumed v2 was SHA-checked only,
  not loaded or rerun.
- Gate consequence: the operational dev prerequisite for creating and freezing
  an uncontaminated v3 holdout is met. No v3 is created by this decision, Phase
  3 remains unpassed, and Phase 4 remains blocked.

## DD-0036 — consume v3_zh once and preserve a metric failure

- Date: 2026-07-23
- Phase: 3F-C one-time v3_zh formal internal gate
- Status: accepted metric failure; holdout consumed; rerun prohibited
- Frozen boundary: use commit
  `f3c07304195807cba751843e498369cd390a929c`, gold SHA-256
  `34518f925e82bd3a1d716bc42ab07ab6b6b3b4333408fab45bfb573d38e67302`,
  freeze bundle `1fe3b8c5ec33b8c8fd25904d8e357b52cde674183011016044bd9d736e85c9b9`,
  and the frozen Phase 3E.3 code, prompt/schema, and config hashes. Do not
  rewrite the v3_zh manifest.
- Execution decision: run all five current pair baselines over all 72 cases
  exactly once, using frozen Chinese snapshots and no source refetch. The pair
  benchmark bypasses Claim Miner. Single Agent and Retrieval-to-Judge may each
  make at most one model request per case, with 144 calls, 1,000,000 tokens,
  zero retry, and zero repair as hard authorization boundaries.
- Operational outcome: all five baselines completed all 72 cases. The run used
  111 calls and 133,833 reported tokens, with zero schema, transport, scope,
  guard, local-validation, or budget failures. Provider-default thinking and
  requested temperature 0.0 were used; effective temperature remains unknown.
  Cache accounting is `not_available` because no production evaluation cache
  call site is wired.
- Quality outcome: Retrieval-to-Judge fixed-six macro-F1 was `0.613757`,
  contradiction recall was `0.769231`, and Single Agent macro-F1 was
  `0.727360`. The Retrieval-to-Judge macro-F1 delta was `-0.113603`.
- Gate decision: completion and operational-integrity requirements passed, but
  the headline result missed the preregistered `0.70` internal threshold, the
  separate `0.75` v0.1 target, and the `0.80` contradiction-recall target.
  Preserve this as a metric failure. Phase 3 did not pass and Phase 4 is not
  eligible.
- Consumption and claims: the v3_zh authorization and formal gate are consumed
  and must not be rerun. Post-result tuning cannot rehabilitate this holdout as
  a formal gate. Because claims were supplied directly, the result compares
  pair-level verification only and cannot establish complete Claim Miner plus
  Judge Multi-Agent advantage.
- Provenance: keep the four canonical artifacts plus the append-only execution
  record under `eval_runs/phase3f_v3_zh_internal_gate/`. The canonical bundle
  SHA-256 is
  `f7601e1076d3ec5fc175c165a0d09c80dbd9f147ae256f46d15164904dcab851`;
  the execution-record SHA-256 is
  `74540003d8200410038028dc4a898aee422d00f1683a03ddfd465c8f01febf4a`.

## DD-0037 — route Chinese pair verification by checkability and context fit

- Date: 2026-07-23
- Phase: 3G-B adaptive Chinese dev
- Status: accepted for offline dev; no live quality claim
- Retrieval: preserve the Latin path except for the required one-chunk
  invariant. When a query contains CJK and fewer than two ordinary lexical
  terms, add NFKC-derived character bigrams/trigrams. Score all chunks
  lexically and retain numbers, versions, dates, negation, and identifiers in
  original source text.
- Router: add `adaptive_live` without changing historical baseline semantics.
  Unavailable source and pure subjective/no-fact-slot claims are deterministic.
  A single chunk uses full-context Single Agent only when the exact serialized
  payload fits the configured context budget. Empty, multi-chunk, or over-
  budget sources use Retrieval-to-Judge. No v3-derived character cutoff exists.
- Telemetry: route, reason, chunk count, and estimated context size are
  required on adaptive predictions. Routes and reasons are allowlisted enums;
  no claim/source/model text is copied into route telemetry. Each adaptive case
  permits at most one model call.
- Checkability: Chinese recommendations and superlatives are deterministic
  `not_checkable` only when objective anchors are absent. Numbers, dates,
  versions, named benchmarks, source-attributed rankings, explicit platform
  compatibility, and capacity limits remain model-verifiable.
- Data: freeze 72 provisional Chinese dev pairs over 24 new sources, with
  balanced relations and short/medium/long coverage. The pack is not a
  holdout. Later live work must report all three baselines overall and by
  source stratum.
- Boundary: zero model calls; consumed v3_zh is used only for leakage
  comparison and is not rerun or modified. No v4 or Phase 4 feature is created.

## DD-0038 — fail Phase 3G-C readiness after the third-run schema error

- Date: 2026-07-23
- Phase: 3G-C adaptive Chinese live dev stability
- Status: accepted operational failure; no rerun
- Freeze: use commit `f5d5521803ffb9b058afafdd33e852541b86da5a`
  and the frozen Phase 3G-B code, prompt/schema, policy/config, and dev dataset
  hashes. Do not modify these after the first model call.
- Execution: the four-route smoke and Runs 01 and 02 succeeded. Each complete
  run made 168 calls; Adaptive made 48 and retained route counts 17/31/12/12.
  Adaptive macro-F1, contradiction recall, and not-checkable recall were 1.0
  in both complete runs.
- Failure: Run 03 hit strict schema validation after 46 provider responses.
  Stop immediately, keep only the sanitized failure artifact, do not retry,
  repair, continue, or rerun, and do not infer missing predictions or metrics.
- Stability interpretation: two-run Adaptive relation and route agreement were
  1.0, but they cannot be promoted to the preregistered three-run result.
  Single Agent two-run relation agreement was 0.930556.
- Readiness decision: fail `3/3 operational success` and
  `all operational failures=0`. Mark the quality and product-value conditions
  that require three complete runs as not evaluated, even though the first two
  Adaptive runs scored 1.0.
- Budget: smoke, two complete runs, and the failed third run used 384 calls and
  542,644 reported tokens, below the 650/1,000,000 authorization limits. Retry
  and repair counts were zero; cache accounting remains `not_available`.
- Boundary: this is provisional pair-level dev evidence, not a holdout and not
  an end-to-end Claim Miner plus Judge evaluation. Phase 3 remains unpassed,
  v4 creation is not eligible from this result, v3_zh is not rerun, and no
  Phase 4 implementation is added.

## DD-0039 - retry one provider schema-contract failure without repair

- Date: 2026-07-23
- Phase: 3G-D schema failure diagnosis and bounded recovery
- Status: accepted for a new, separately authorized dev stability attempt
- Diagnosis: classify the historical Run 03 event as
  `unknown_schema_failure`. Baseline, case, and task are derivable from frozen
  ordering and 46 recorded calls. The provider contract boundary is proven by
  the safe `schema` taxonomy; the missing diagnostic prevents any narrower
  malformed/missing/enum/truncation claim.
- Trigger: retry only `ModelSchemaError`, with `schema_retry_limit=1` and at
  most two provider calls for one live case. A valid first attempt never
  retries. Transport, budget, scope/span, deterministic conflict, semantic
  disagreement, and local business-validation errors never initiate retry.
- Request integrity: preserve the same claim/source scope and Pydantic output
  schema. Add only a fixed, case-independent schema reminder and attempt index.
  Never include the prior response, locally repair JSON, fill fields, correct
  enums, or normalize semantics.
- Failure behavior: if pre-request budget is unavailable, do not send attempt
  2. If attempt 2 fails, stop fail-closed. Keep only allowlisted schema
  diagnostics and payload-free telemetry. No failed response enters cache.
- Observability: report first-attempt failures, retries, recovered and
  unrecovered counts, first-attempt contract rate, final operational success,
  and retry usage/latency. A recovered failure is visible and cannot be
  represented as zero first-attempt failures.
- Next evaluation: discard old Runs 01/02 for readiness counting and require
  three fresh complete runs on one new freeze. Require 3/3 final success,
  unrecovered failures `0`, first-attempt contract success at least `0.99`,
  and at most two recovered failures per run. Preserve every existing quality
  and product-value threshold.
- Boundary: this decision authorizes no model call, holdout run, v4 creation,
  or Phase 4 implementation. Historical Phase 3G-C artifacts remain
  byte-identical.

## DD-0040 - stop Phase 3G-E after a recovered schema error exposes scope failure

- Date: 2026-07-23
- Phase: 3G-E schema-recovery adaptive Chinese live dev
- Status: accepted operational failure; no rerun
- Freeze: use commit `060db3b066f19093a4b8cd97edfa4b47e73e519a`
  and the frozen Phase 3G-D code, prompt/schema, config/recovery policy, and
  Phase 3G Chinese dev dataset. The execution uses provider-default thinking,
  requested temperature 0.0, no semantic repair, and no cache.
- Successful work: the five-operation smoke and fresh Runs 01 and 02 completed
  with valid telemetry and route counts 17/31/12/12. Adaptive and
  Retrieval-to-Judge macro-F1 were 1.0 in both complete runs.
- Failure sequence: in Run 03, one `ModelSchemaError` triggered the single
  permitted retry and recovered. The resulting canonical Judge output then
  raised typed `JudgeScopeError` with allowlisted code `claim_id_mismatch` on
  `retrieval_judge_live / phase3g_zh_dev_059`.
- Recovery boundary: report the event simultaneously as one first-attempt
  schema failure, one recovered schema failure, and a failed final operation.
  Do not retry the subsequent scope violation. Retain only the sanitized
  failure artifact and never reconstruct the model response.
- Readiness: observed dev contract success was 455/456, recovered/unrecovered
  counts were 1/0, and no run exceeded two recoveries. Nevertheless, final
  operational success was 2/3 and scope failures were not zero. Quality and
  product-value gates requiring three complete runs are not evaluated.
- Budget: smoke, two complete runs, and the failed third run used 460 calls
  and 672,550 reported tokens, including 2,023 retry tokens. This remained
  below the 520/1,000,000 authorization.
- Provenance: do not create the conditional three-run aggregate. Preserve the
  execution preregistration, budget ledger, two complete run directories, Run
  03 safe failure artifact, and append-only `execution_outcome.json`.
- Consequence: Phase 3 remains unpassed. The evidence is insufficient to
  create v4, and Phase 4 remains ineligible. No consumed holdout was loaded or
  rerun, and no v4 or Phase 4 implementation was created.

## DD-0041 - keep canonical verdict identity under local ownership

- Date: 2026-07-23
- Phase: 3G-F local verdict ownership
- Status: accepted for a fresh, separately frozen live dev stability attempt
- Diagnosis: the Phase 3G-E scope failure was not a defective scope validator.
  The contract unnecessarily asked the model to echo deterministic IDs and
  association metadata, allowing a schema-valid response to disagree with the
  trusted request.
- Model boundary: live Judge output owns only relation, confidence, reason,
  and evidence span. All four keys are required, extra properties are
  forbidden, and the prompt contains no claim/source IDs, locator/provenance,
  or output-version metadata.
- Local boundary: build canonical claim/source identity, locator,
  corroboration, and version fields from `JudgeInput` and configured client
  metadata. Match evidence to a real ordered candidate substring before
  attaching its local locator.
- Recovery: an extra `claim_id`, source ID, locator, or version is a
  `ModelSchemaError`. Permit the one existing schema-only retry, without
  passing the first response or performing repair. Scope, span, guard,
  transport, budget, and local validation remain non-retryable.
- Compatibility: retain canonical `JudgeOutput`/`Verdict` and legacy scope
  codes for old artifact readers. `claim_id_mismatch` remains readable but
  cannot arise from a schema-valid response in the new live contract.
- Evaluation: freeze a new code/prompt/config/ownership bundle, then discard
  all older completed runs for readiness counting. Require three fresh runs
  under the unchanged Phase 3G readiness and product-value thresholds.
- Boundary: no holdout execution, v4 creation, Phase 4 implementation, cache
  fabrication, semantic repair, or historical artifact mutation is permitted.

## DD-0042 - advance the ownership freeze to v4 construction eligibility

- Date: 2026-07-23
- Phase: 3G-F ownership live dev
- Status: accepted dev-readiness pass
- Execution: the smoke and three fresh runs completed under one frozen
  code/prompt/config/dataset bundle. Adaptive routes remained 17/31/12/12,
  and every final operational failure count was zero.
- Ownership result: no `claim_id_mismatch` occurred. Under the new contract a
  schema-valid model response cannot author claim/source identity, locator, or
  version fields; local trusted context supplies them before unchanged scope
  and span validation.
- Recovery result: four of 504 initial dev decisions failed schema validation
  and all recovered via the single schema-only retry. Combined first-attempt
  contract success was 0.992063, no round exceeded two recoveries, and no
  unrecovered failure occurred.
- Quality result: Adaptive and Retrieval-to-Judge macro-F1 were 1.0 in every
  run. Single Agent mean macro-F1 was 0.946986. Adaptive had 1.0 relation and
  route agreement, a +0.073228 long-stratum macro-F1 delta, and 36.22% lower
  input-token use than Single Agent.
- Decision: all frozen dev-readiness and product-value gates pass, so creation
  of a genuinely new v4 blind holdout is eligible under separate
  authorization.
- Boundary: this is provisional pair-level dev evidence. Phase 3 remains
  formally unpassed, Phase 4 remains ineligible, and no conclusion is made
  about complete Claim Miner plus Judge Multi-Agent superiority. No consumed
  holdout was loaded or rerun.

## DD-0043 - freeze v4_zh as a blind, human-first candidate pack

- Date: 2026-07-24
- Phase: 3H-A v4_zh blind candidates
- Status: accepted; awaiting independent human annotation
- Corpus shape: freeze 72 cases over 24 sources with three cases per source,
  source strata 7/7/6/4 for short/medium/long/unavailable, and aggregate
  intended-construction support of 12 per relation.
- Blindness: do not persist case-level intended relations. Candidate and
  reviewer surfaces contain only identifiers, frozen source text, claims,
  and blank annotation fields. Gold and model proposals do not exist.
- Routing: derive source length and the aggregate 17/31/12/12 expected route
  distribution by running the frozen production chunker and Adaptive Router,
  not by character-count thresholds or hidden case mappings.
- Provenance: use official native-Chinese snapshots obtained through the
  existing no-retry safe-fetch boundary. Preserve both upstream body and
  canonical excerpt hashes; freeze unavailable sources without guessed text
  or evaluation-time refetch.
- Leakage: compare full text only with readable non-consumed datasets.
  Consumed corpora remain byte-only and may contribute only pre-existing
  non-label manifests or fingerprints. Disclose this coverage boundary rather
  than reconstructing consumed records.
- Gate: preregister the one-time formal thresholds before annotation and
  model exposure. Candidate construction does not itself make the pack
  eligible to run; human annotation and validation are separate future work.
- Freeze: make no production, Judge, Router, Retrieval, prompt/schema,
  recovery, ownership, dependency, remote, or Phase 4 change.

## DD-0044 - separate private human provenance from v4_zh public gold

- Date: 2026-07-24
- Phase: 3H-B v4_zh single-human freeze
- Status: accepted; internal gate eligible but not run
- Migration: preserve the completed packet byte-for-byte outside the
  repository before restoring the committed blind packet. Never overwrite a
  differing external record or discard the only filled copy.
- Human boundary: require exact case/source/claim matching, exact relation
  enums or fixed Chinese mappings, complete notes, continuous evidence
  substrings, and explicit reviewer provenance. Do not infer or repair labels.
- Privacy: publish only the reviewer-record hash and provenance-presence
  booleans. Keep reviewer name/alias, reviewer ID, and full notes in the
  private external record.
- Source compatibility: retain the rich native-Chinese source snapshots
  byte-for-byte and derive a strict runtime `SourceFixture` file
  deterministically. This avoids changing the production dataset loader.
- Status: single-human review counts as one human review and is sufficient for
  the preregistered internal gate, but is not eligible as a public benchmark
  and does not claim second-human adjudication.
- Boundary: do not use intended construction strata, load consumed gold,
  call a model, execute v4_zh, alter gate thresholds, or modify production
  code.

## DD-0045 - consume v4_zh on its fail-closed operational outcome

- Date: 2026-07-24
- Phase: 3H-C final internal gate
- Status: accepted operational failure; Phase 3 closed with known limitations
- Preflight: the exact commit, clean/no-remote repository, frozen dataset and
  implementation hashes, provider configuration, and 468-test
  boundary-compliant suite all passed before the first call. No smoke,
  preview, cross-run cache, source refetch, or threshold change occurred.
- Outcome: the one authorized run stopped during `single_agent_live` on
  `v4_zh_candidate_003`, after two completed predictions and three provider
  calls. The third response was schema-valid, then failed a non-retryable
  local `ValueError` validation. The safe artifact has no finer allowlisted
  subtype or model content, so the precise local condition remains unknown.
- Resources: 3 calls used 2,731 input, 2,713 output, and 5,444 total tokens;
  schema/recovery/transport events were zero. Cost remains null because no
  frozen pricing snapshot exists.
- Gate decision: treat the formal attempt as consumed operational failure.
  Do not invent missing metrics, rerun v4_zh, tune against it, or create v5.
  Set Phase 3 to `completed_with_known_limitations` and keep
  `phase4_eligible=false`.
- Engineering decision: preserve the immutable failure provenance and stop
  the Phase 3 pair-level holdout sequence. Any future offline investigation
  may improve local-validation taxonomy, but it cannot retroactively change
  this gate or authorize Phase 4.
- Scope: the benchmark supplied atomic claims and never exercised Claim
  Miner. It evaluates only the verifier/adaptive boundary and cannot prove a
  complete Claim Miner plus Judge Multi-Agent advantage.

## DD-0046 - continue Phase 4 engineering without a Phase 3 quality pass

- Date: 2026-07-25
- Phase: 4A reliability and GitHub integration
- Status: accepted engineering continuation with known limitations
- Gate boundary: keep Phase 3 as `completed_with_known_limitations` and
  `phase4_eligible=false`. The consumed v4_zh run has no formal quality
  metrics, cannot be rerun or debug-loaded, and will not be replaced by v5.
- Local validation: use one payload-free `LocalValidationError` and a closed
  code allowlist for evidence grounding, unavailable-source conflict,
  call/telemetry accounting, and canonical assembly. Clear exception chains
  before artifact serialization.
- Quality boundary: a schema-valid, in-scope relation disagreement is a
  prediction to score, including `source_unavailable` on an available source.
  Forged evidence, bounded-scope violations, deterministic conflicts, and
  unavailable-source metadata contradictions remain fail-closed.
- Recovery: preserve schema-recovery-v1 exactly. Only `ModelSchemaError` may
  receive its one schema-only retry; local validation never retries.
- SARIF: derive SARIF 2.1.0 only from canonical audit data and the existing
  policy decision. Use stable rules/order/fingerprints and repository-relative
  locations. Exclude model-authored and source content rather than attempting
  to sanitize it after serialization.
- CLI: formalize exit statuses as 0 policy pass, 1 completed policy failure,
  and 2 configuration/operational failure. Atomic SARIF output precedes exit
  1 so CI can retain findings.
- GitHub: use a reusable/PR workflow with only `contents: read` and
  `security-events: write`. Continue after the check step solely to upload an
  existing SARIF, then propagate the original nonzero status.
- Validation gap: this work productizes verifier output but does not establish
  blind-holdout F1 or Claim Miner plus Judge performance. Full-document claim
  extraction and verification still require an independent benchmark.

## DD-0047 - evaluate Miner plus Judge through canonical full-document paths

- Date: 2026-07-25
- Phase: 4B full-document benchmark readiness
- Status: accepted engineering continuation; no live execution
- Dataset: use a versioned, author-constructed deterministic provisional
  corpus with physical Markdown files, frozen local sources, exact claim
  offsets/lines, citation/source bindings, relations, and continuous evidence.
  It is explicitly rerunnable, non-blind, non-human-reviewed, and non-public.
- Baselines: compare one document-level Single Agent call with the real
  paragraph-scoped Miner plus existing Adaptive verifier. Do not provide
  atomic claims to either baseline.
- Ownership: models own semantic extraction/verdict fields. Local trusted code
  attaches claim/source IDs, raw Markdown spans, line locators, citation scope,
  source bindings, and evidence locators. Additional fields remain forbidden.
- Reuse: continue through `parse_markdown`, `ClaimMinerAgent`,
  `adaptive_live`, `ClaimJudgeAgent`, deterministic checks, canonical audit,
  policy, and SARIF. Do not create benchmark-only relation or policy logic.
- Terminology: Miner and Judge are Agents. Adaptive Router and lexical
  Retrieval are deterministic components and must not be counted as Agents.
- Alignment: match exact spans first, then use deterministic one-to-one
  relaxed alignment. Never reuse a prediction. Missing gold claims enter
  relation confusion as `__missing__` and cannot disappear from denominators.
- Safety: preserve schema-only recovery. Scope, citation, evidence, guard,
  transport, budget, and local handoff failures remain payload-free and
  fail-closed. No source fixture may be fetched during evaluation.
- Readiness boundary: deterministic fake runs validate implementation and
  arithmetic only. They do not establish live F1, blind-holdout quality, or a
  Multi-Agent advantage.
- Phase status: retain `completed_with_known_limitations` and
  `phase4_eligible=false`; do not create v5 or reinterpret the consumed gate.

## DD-0048 - stop full-document live dev on unrecovered smoke schema failure

- Date: 2026-07-25
- Phase: 4B full-document live dev
- Status: accepted fail-closed outcome; no full dev or test execution
- Scope: execute only the provisional `full_document_v1` dev split after one
  document is exercised by both registered baselines. Never parse or run the
  test split under dev authorization.
- Outcome: the first document-level Single Agent logical call failed its
  initial strict schema and the one permitted schema-only recovery. Retain
  only stage/category/allowlisted code plus aggregate budget telemetry.
- Interpretation: two provider attempts and a final schema failure establish
  one first-attempt schema failure, one retry, zero recoveries, and one
  unrecovered failure. They do not establish which field or provider behavior
  caused the failure because response content was not retained.
- Stop rule: do not run the second smoke baseline, the full 12-document dev
  comparison, or the test split after an unrecovered smoke failure. Do not
  fabricate quality metrics or dev error categories.
- Readiness: test execution is not engineering-ready. Diagnose offline with
  synthetic/dev-only fixtures, freeze any general contract change, and require
  separate live-dev authorization before reconsidering test authorization.
- Phase status: retain `completed_with_known_limitations` and
  `phase4_eligible=false`; this result is not blind, formal, or evidence of a
  Multi-Agent advantage.

## DD-0049 - harden the full-document provider contract before full dev

- Date: 2026-07-25
- Phase: 4B.1 provider contract hardening
- Status: accepted; bounded Single and Multi dev smokes completed
- Historical diagnosis: retain `unknown_schema_failure`. The old safe
  artifact lacks finish reason, response length, JSON/Pydantic stage, and
  field path, so output-token saturation cannot prove truncation.
- Output budget: require the existing project maximum of 8,192 tokens for
  full-document calls. Include it in provenance, budget telemetry, and cache
  identity; do not send an invented provider thinking parameter.
- Schema ownership: models return only extraction, citation selection, and
  verdict semantics. Local code continues to attach deterministic IDs,
  offsets, lines, source bindings, locators, versions, and corroboration.
- Strictness: cap reasons at 240 characters while preserving required fields,
  forbidden extras, strict enums/types, and no JSON repair or semantic
  normalization.
- Diagnostics: persist only allowlisted contract stages, finish reasons,
  numeric usage, content presence/length, fixed Pydantic code/path, and a
  suspected-truncation boolean. Never persist provider content or exception
  input/context.
- Live observation: Single Agent failed its first schema validation at
  `claims/0` with `value_error`, then recovered once. Four Miner and five Judge
  calls were first-pass valid. All finish reasons were `stop`, and no call was
  suspected to be truncated.
- Readiness: these results establish operational readiness for a separately
  authorized complete dev run, not quality or Multi-Agent superiority. Test
  remains unauthorized. Keep Phase 3 `completed_with_known_limitations` and
  `phase4_eligible=false`.

## DD-0050 - pivot v0.1 from evaluation expansion to a bounded product loop

- Date: 2026-07-26
- Status: accepted product direction; offline vertical slice implemented
- Problem: the project over-invested in formal evaluation while users still
  lacked a complete, resilient document-audit loop. The consumed gates did not
  yield a formal quality pass, and another holdout would not resolve the
  product gap.
- Decision: freeze the old evaluation surface as history. Do not create v5,
  another annotation task, or another evaluation phase. Measure v0.1 progress
  through real user closure: check a document or Git diff, preserve actionable
  line findings, continue after one claim fails, emit SARIF, and optionally
  produce a reviewable patch.
- Architecture: the deterministic Controller retains permissions, budgets,
  retries, tools, artifacts, and policy. Claim Miner and Claim Judge retain
  their existing bounded contracts. Add a two-call Audit Coordinator, a
  plan-only Evidence Scout backed by Controller-owned safe search/fetch, and a
  one-shot Challenger for contradictions, low confidence, structured facts,
  and comparisons.
- Citation-first rule: an initial plan cannot send a cited claim to discovery.
  Citation evidence is checked first; only an unresolved review may authorize
  counter-search. Search snippets select URLs but never become evidence.
- Failure semantics: Agent/source failures are isolated per claim and use safe
  status codes. Parse, configuration, and artifact failures terminate the
  document. Suggested patches are output-only and never applied or committed.
- Release boundary: keep Phase 3 `completed_with_known_limitations` and
  `phase4_eligible=false`. The offline demo and source-checkout Action are
  implemented, while real dogfood, PyPI/`uvx`, and a public demo repository
  remain release work.
