# Code-budget review

Date: 2026-07-10 UTC  
Scope: Checkpoint A, before Phase 3A implementation

The project was not a Git repository at the start of this review. A local
empty repository was initialized with `git init`; no remote and no commit were
created. Generated Python caches were removed and `.gitignore` now explicitly
covers the requested build, test, virtual-environment, cache, and
`.evidencetrace/` paths.

## Current physical line counts

Counts use `wc -l` on Python source files and do not treat them as a quality
metric by themselves.

| Area | Lines | Plan context |
|---|---:|---|
| `src/evidencetrace/markdown.py` | 910 | Markdown/parser and source mapping |
| `src/evidencetrace/models.py` | 748 | schemas/config/artifact contracts |
| `src/evidencetrace/gitdiff.py` | 579 | diff parsing and changed-line mapping |
| `src/evidencetrace/retrieval/fetch.py` | 449 | safe fetch boundary |
| `src/` total | 4,627 | Phase 2 core package |
| `tests/` total | 3,085 | existing Phase 0–2 tests |

The plan's core-package ceiling is 4,000 lines. The current core is therefore
627 lines above that ceiling. The overall source plus test count is 7,712
lines, before adding Phase 3A evaluation data and runtime.

## Module review

### `markdown.py` — 910 lines

The size comes from preserving original offsets while handling CommonMark
paragraphs, inline/reference/footnote/autolink/bare-URL citations, escaped and
code spans, duplicate definitions, suppression directives, and diagnostics.
These are independent safety and auditability contracts rather than wrappers.
No repeated block or dead branch was identified that can be removed without
weakening a tested public behavior. A future refactor may split citation
scanning from paragraph mapping, but that is not a safe LOC-only edit.

### `models.py` — 748 lines

This file contains the Phase 1 public schema, policy/configuration, diff
contracts, artifact contracts, source metadata, and validators. The validators
enforce path safety, source-span requirements, timezone correctness, and
cross-field invariants. Removing them would hide invalid artifacts rather than
reduce product scope. A future schema split is possible, but is intentionally
deferred to avoid changing import contracts.

### `gitdiff.py` — 579 lines

The excess is mostly Git path quoting/escaping, malformed and combined-diff
diagnostics, binary/deletion/rename/copy handling, and direct paragraph
selection. These cases are covered by the existing tests and prevent unsafe
guesses about changed source lines. No clearly dead code was found.

### `retrieval/fetch.py` — 449 lines

The fetcher is below the combined plan budget for fetch/retrieval/cache, but is
large for a single module because it implements scheme and IP validation, DNS
revalidation, IP pinning, redirect/retry policy, phase/total timeouts, MIME and
stream-size limits, and secret-safe metadata. Those checks are security
boundaries and are not candidates for compression.

## Safe deletions made

The repository root contained unreferenced duplicate Phase 2 modules
(`cache.py`, `model_client.py`, `agents/`, `checks/`, and `retrieval/`). The
packaging configuration only includes `src/`, all tests import
`evidencetrace.*` from `src/`, and the copies differed only by formatting or
older implementations. They were deleted as explicit dead duplicates. The
`src/evidencetrace/retrieval/fetch.py.orig` file was compared first; it differed
from `fetch.py` only by validating a redacted URL before the request. The
current implementation rejects credential-bearing URLs before httpx, so the
`.orig` file was a redundant backup and was deleted.

No production module was compressed or broadly refactored for LOC reduction.

## Estimated future increments

These are planning estimates, not permission to implement later phases early:

| Phase | Production increment | Tests/data/docs increment | Main risk |
|---|---:|---:|---|
| Phase 3A eval runtime, baselines, metrics | 900–1,200 | 1,500–2,300 | dataset provenance and label leakage |
| Phase 4 PR workflow/Action | 500–800 | 700–1,100 | secret and changed-line boundaries |
| Phase 5 Scout/Challenger path | 900–1,400 | 900–1,400 | cost, retrieval quality, and scope creep |

Phase 6 local-corpus RAG and Phase 7 release work are excluded from this
estimate.

## Recommended budgets

For the current security-heavy core and the explicitly requested evaluation,
the following local engineering budget is more realistic than compressing
tested safety code:

- Final production Python: 7,500 lines hard cap, with every Phase 3–5
  increment justified in this document or a later decision record.
- Final tests and evaluation fixtures/runtime: 7,500 lines/data-equivalent
  physical lines hard cap, with frozen test data kept readable.
- Do not change the source-of-truth plan's budget in this checkpoint. If a
  future phase approaches these recommendations, defer functionality before
  removing source mapping, security, schema, or evaluation checks.

These are recommendations only; the project plan's budget remains unchanged.

## Phase 3A actual increment

After implementation and dead-code cleanup:

| Area | Physical Python lines |
|---|---:|
| `src/evidencetrace/eval/` | 1,286 |
| Other Phase 3A production additions (`cli.py`) | approximately 19 |
| Phase 3A production increment | approximately 1,305 |
| `src/` total | 5,932 |
| Phase 3A test increment | 330 |
| `tests/` total | 3,415 |
| Checked-in eval JSONL/hash/guides | 246 physical lines |

The eval-package target was approximately 1,200 lines; it is eighty-six
physical lines over that target. The overage is retained for composite
source/data hashing, artifact schemas, retrieval-versus-Judge attribution, the
required per-label/annotation report, and the optional live-mode boundary.
Removing these checks would weaken required eval contracts. Clear dead code
found during review (an unused dataset summary, claim wrapper, mutation type
alias, imports, and a no-op provenance branch) was removed first.

The original source-of-truth budget is still unchanged. Phase 4 remains
blocked by the evaluation stop gate, so none of the estimated Phase 4 or 5
increment has been consumed.

## Phase 3B actual increment

Phase 3B replaced misleading eval semantics and repaired dev-only judgement
logic. Deletions offset additions as explicitly allowed by the task:

| Area | Insertions | Deletions | Net |
|---|---:|---:|---:|
| `eval/baselines.py` | 245 | 101 | +144 |
| `eval/metrics.py` | 132 | 20 | +112 |
| `eval/models.py` | 70 | 3 | +67 |
| `eval/runner.py` | 367 | 121 | +246 |
| Judge/Miner/audit signal contracts | 83 | 20 | +63 |
| `checks/deterministic.py` | 273 | 9 | +264 |
| CLI wording | 2 | 2 | 0 |
| **Phase 3B production total** | **1,172** | **276** | **+896** |

The approximate Phase 3B net target was 800 lines, so the result is 96 lines
over. The retained overage is attributable to:

- physically separate one-call single-agent and isolated Miner/Judge live
  execution paths with call counting and source-span validation;
- observed/fixed/weighted/balanced metrics, coverage and false-block
  definitions, three-state validity, and deterministic/live report separation;
- conservative entity, negation, scope, denominator, qualifier, conjunction,
  clause-alignment, and live-output guards with shared typed signal contracts.

These are the validity and safety repairs requested for this checkpoint, not a
new product feature. Removing them would recreate the misleading comparison or
allow model output to bypass deterministic conflicts. No large framework,
provider, retrieval system, or later-phase feature was added.

Current physical Python totals:

- `src/`: 6,828 lines (Phase 3A checkpoint: 5,932; net +896).
- `tests/`: 4,257 lines (Phase 3A checkpoint: 3,415; approximately +842).
- New Phase 3B dedicated tests: 811 lines; migrated existing tests account for
  the remaining net change.
- v2 data/guides/manifest: 804 physical lines, including a 599-line
  human-review packet.

The frozen judgement/eval/CLI code hash is
`eb5a01fa0065a5b6643c4c1d60cca746d1f626015e54ce0d4382343859120f62`.
The plan budget remains unchanged. Phase 4 has still consumed zero code.

## Phase 3D actual increment

Phase 3D adds a frozen single-human dataset boundary and operational telemetry;
it does not add another model provider, retrieval architecture, Agent, or
Phase 4 integration. Exact physical-line counts are measured against commit
`457e7b3242e057d15bf11568966d4adb11c861da`:

| Area | Phase 3C base | Phase 3D ready | Net |
|---|---:|---:|---:|
| `src/evidencetrace/eval/` | 1,855 | 2,441 | +586 |
| Other production `src/` | 4,973 | 5,149 | +176 |
| **All production `src/`** | **6,828** | **7,590** | **+762** |
| Holdout generator script | 0 | 326 | +326 |
| Tests | 4,273 | 5,028 | +755 |
| Frozen JSONL/hash records | 0 | 139 | +139 data lines |

The `src/` increment consists of single-human provenance and validity schema,
six-label/live-only gate semantics, source-substring enforcement, per-call
model usage and failure telemetry, same-configuration/call-ceiling checks, and
artifact/report aggregation. The standalone generator is intentionally kept
outside runtime production code and performs the one-way, privacy-minimized
projection from the external Reviewer A record. Its validation and freeze
logic is retained rather than replaced by a hand-edited gold file.

The earlier recommended 7,500-line final `src/` cap is exceeded by 90 lines at
this ready checkpoint. This is recorded as a budget warning, not hidden by
compressing readable validation or removing safety tests. Any Phase 4 work
should first identify genuine duplication or revise the recommendation in the
source-of-truth plan; this checkpoint does not revise that budget itself.

Phase 3D code/prompt bundle hashes are recorded in
`eval_sets/v2/manifest.json`. Phase 4 and Phase 5 have still consumed zero
production or test code.

## Phase 3D live-dev remediation increment

The first live dev preflight exposed a protected-token canonicalization bug.
The remediation changes no architecture or prompt and adds one net production
line: explicit percentage matching precedes semantic-version matching, and the
existing protected-fact subset check now also rejects an empty model claim set.

| Area | Phase 3D ready | Post-dev-fix | Net |
|---|---:|---:|---:|
| All production `src/` | 7,590 | 7,591 | +1 |
| Holdout generator script | 326 | 326 | 0 |
| Tests | 5,028 | 5,081 | +53 |

The test increment covers verbatim decimal percentages, empty-claim bypass,
and the failed-attempt manifest provenance. The recommended 7,500-line `src/`
cap is now exceeded by 91 lines. Phase 4 and Phase 5 remain at zero code.


## Phase 3D authorized-rerun safety increment

The authorized rerun safety layer adds a provider-call hard budget, typed
failure-only artifact, exact safe telemetry persistence, explicit sampling and
thinking semantics, and regression coverage. It does not add an Agent,
provider, retrieval feature, or product behavior.

| Area | Post-dev-fix | Safety freeze | Net |
|---|---:|---:|---:|
| `model_client.py` | 325 | 344 | +19 |
| `eval/models.py` | 340 | 426 | +86 |
| `eval/runner.py` | 868 | 1,183 | +315 |
| All production `src/` | 7,591 | 8,011 | +420 |
| Tests | 5,081 | 5,447 | +366 |

This 420-line production increment raises the 7,500-line recommendation
overage from 91 to 511 lines. The overage is retained because the user requires
a pre-request hard ceiling, failed-call accounting, mid-run token/latency/case
persistence, strict secret exclusion, requested-versus-effective configuration,
and no retry or rerun ambiguity. No code-budget line was reduced by weakening
safety or evaluation checks.

## Phase 3D failed-run provenance increment

The one authorized dev rerun stopped on its first provider response because
the output failed the required schema. Persisting and freezing that outcome
adds no production code. The additional test coverage validates the checked-in
failure artifact schema and fixed hash, its sole-file invariant, manifest
cross-references, and absence of sensitive payload fields.

| Area | Safety freeze | Failure provenance | Net |
|---|---:|---:|---:|
| All production `src/` | 8,011 | 8,011 | 0 |
| Holdout generator script | 326 | 326 | 0 |
| Tests | 5,447 | 5,509 | +62 |

The runtime `failed_attempt.json` is provenance data rather than production or
test code. Phase 4 and Phase 5 remain at zero code.


## Phase 3D.1 structured-output contract increment

Phase 3D.1 starts from 8,011 production and 5,509 test lines at commit
`ab2eaaa5d2f083685842c7611b92d587af7442b0`. It changes only the
OpenAI-compatible wire contract, safe schema-failure instrumentation, and
offline tests.

| Area | Insertions | Deletions | Net |
|---|---:|---:|---:|
| `model_client.py` | 493 | 30 | +463 |
| `eval/models.py` | 26 | 7 | +19 |
| `eval/runner.py` | 12 | 2 | +10 |
| **Production total** | **531** | **39** | **+492** |
| Existing tests | 33 | 6 | +27 |
| New structured-output tests | 621 | 0 | +621 |
| **Test total** | **654** | **6** | **+648** |

The approximate production target was +200 lines; this checkpoint is 292 lines
over. The retained production increment is itemized as follows:

- approximately 150 lines for a schema-aware one-request JSON-mode path,
  bounded output tokens, strict content/envelope handling, truncation failure,
  detached transport/schema exceptions, and unchanged pre-request budgeting;
- approximately 275 lines for frozen/extra-forbid diagnostic contracts,
  Pydantic error classification, bounded issue extraction, strict allowlists,
  constructed-instance revalidation, static messages, and raw-context removal;
- approximately 32 lines for the three compact output examples and duplicated
  system/user format instructions;
- 19 lines for v1/v2 failure-artifact compatibility and invariants;
- approximately 16 lines for safe diagnostic and actual adapter prompt-version
  propagation.

These boundaries directly implement the requested no-content diagnostics and
fail-closed behavior. Removing them to meet the target would either duplicate
schema knowledge in the runner, permit unsafe untyped artifact fields, or lose
required failure telemetry. Repetition was removed through one shared request
path and one failure helper before accepting the overage. No security check was
compressed into opaque logic solely to reduce physical LOC.

Current physical Python totals are 8,503 production lines, 6,157 test lines,
and the unchanged 326-line holdout generator. No dependency, provider, Agent,
retrieval feature, live call, holdout execution, or Phase 4 code was added. The
source-of-truth plan budget is unchanged.

## Phase 3D.1 one-case smoke provenance increment

The authorized smoke changes no production module or prompt. It adds one fixed
safe runtime artifact, v2 manifest/status/decision documentation, and one
regression test that freezes the artifact hash, exactly-one-call boundary,
zero-retry/Miner/dev/holdout flags, telemetry, and sensitive-field exclusions.

| Area | Contract freeze | Smoke checkpoint | Net |
|---|---:|---:|---:|
| Production `src/` | 8,503 | 8,503 | 0 |
| Tests | 6,157 | 6,236 | +79 |

The runtime JSON is provenance data, not production or test code. No provider,
Agent, retry path, eval runner, prompt, frozen dataset, or Phase 4 functionality
was added or changed. The source-of-truth plan budget remains unchanged.

## Phase 3D.1 Miner/Judge smoke provenance increment

The isolated Miner/Judge smoke changes no production module or prompt. It made
one Miner request and, only after that contract passed with one legal claim,
one Judge request. Both output schemas and their local scope/grounding guards
passed. The safe runtime artifact is provenance data and contains no raw model
content, reasoning, prompt, claim, source, evidence text, header, credential,
or Pydantic input value. Cost remains null because no explicit model-price
snapshot was provided.

Final physical Python totals after the fixed-artifact regression was added are:

| Area | Single-agent smoke checkpoint | Final checkpoint | Net |
|---|---:|---:|---:|
| Production `src/` | 8,503 | 8,503 | 0 |
| Tests | 6,236 | 6,373 | +137 |

The test-only increment freezes artifact integrity, call/token arithmetic,
scope results, and sensitive-field exclusions. Production code and prompt
hashes remain unchanged. This provenance does not represent a full dev run,
holdout run, quality metric, Phase 3 gate, or Phase 4 implementation. The
source-of-truth plan budget remains unchanged.

## Phase 3E context-aware entity-guard increment

Phase 3E starts from 8,503 production and 6,373 test Python lines. It changes
only deterministic entity context, typed guard-failure safety, semantic cache
versioning, future eval provenance, and their offline regressions.

| Production area | Physical LOC | Insertions | Deletions | Net |
|---|---:|---:|---:|---:|
| `checks/deterministic.py` | 416 | 46 | 2 | +44 |
| `agents/judge.py` | 267 | 14 | 6 | +8 |
| `cache.py` | 77 | 28 | 4 | +24 |
| `eval/baselines.py` | 451 | 12 | 3 | +9 |
| `pipeline.py` | 190 | 9 | 2 | +7 |
| `eval/models.py` | 483 | 38 | 0 | +38 |
| `eval/runner.py` | 1,210 | 17 | 0 | +17 |
| **Production total** | **8,650** | **164** | **17** | **+147** |

The +147 net production change is below the requested approximate +150 target.
The largest increment is the backward-compatible v3 typed guard artifact
contract and its invariants; the entity semantic change itself is a single
additional bounded-context absence condition plus one shared context helper.
No lines were compressed or safety checks removed to meet the target.

| Test area | Physical LOC | Net |
|---|---:|---:|
| New `test_phase3e_entity_guard.py` | 330 | +330 |
| Existing manifest freeze tests | 545 | +31 |
| Existing cache tests | 117 | +37 |
| **All tests** | **6,771** | **+398** |

Tests intentionally exceed the production increment because they cover both
false-positive and true-conflict entity paths, preceding/following and
separate-candidate context, live Judge acceptance/rejection, unchanged
numeric/date/version/negation signals, safe exception contents, future
failure-artifact allowlisting, historical freeze hashes, and legacy cache
invalidation. No v2 fixture text or case-specific rule was copied.

Phase 4 and Phase 5 still have zero code in this checkpoint. The
source-of-truth plan budget remains unchanged.

## Phase 3E live-dev failure provenance increment

The authorized dev attempt changes no production module, prompt, config,
dataset label, or gold. It adds one checked-in safe failure artifact,
manifest/status/decision documentation, and a fixed regression for the
artifact's hash, Pydantic schema, call budget, token arithmetic, failure
classification, zero guard codes, and sensitive-field exclusions.

| Area | Offline Phase 3E checkpoint | Live-attempt checkpoint | Net |
|---|---:|---:|---:|
| Production `src/` | 8,650 | 8,650 | 0 |
| Tests | 6,771 | 6,852 | +81 |

The runtime JSON is provenance data rather than production/test code. The
additional test lines are retained because an incomplete live run has no
canonical metrics artifact; freezing its sole failure record is the only
repeatable way to prove the 44-call ceiling, telemetry, fail-closed cleanup,
and absence of sensitive payloads. No code-budget target is exceeded by this
provenance-only checkpoint. Phase 4 and Phase 5 remain at zero code.

## Phase 3E.2 Miner-contract and pair-baseline increment

Phase 3E.2 starts from 8,650 production and 6,852 test Python lines. The current
checkpoint contains 8,858 production and 7,229 test lines: net +208 production
and +377 tests. The production increment stays below the requested approximate
+300 target.

| Production area | Insertions | Deletions | Net |
|---|---:|---:|---:|
| `agents/miner.py` | 139 | 33 | +106 |
| `audit_models.py` | 34 | 1 | +33 |
| `cache.py` | 3 | 0 | +3 |
| `eval/baselines.py` | 63 | 47 | +16 |
| `eval/models.py` | 35 | 2 | +33 |
| `eval/runner.py` | 30 | 19 | +11 |
| `model_client.py` | 6 | 0 | +6 |
| **Production total** | **310** | **102** | **+208** |

The largest change replaces model-owned enriched claims with required exact-span
drafts, linear earliest/latest uniqueness matching, and deterministic enrichment.
The matcher is non-recursive and does not enumerate occurrence combinations.
The remaining production lines separate active pair baselines from legacy
artifact literals, add a safe Miner scope
error field, and version the Miner cache contract. This is contract and eval
validity work, not a new Agent, provider, retrieval system, or feature.

Tests add 466 lines and remove 89, net +377. The retained coverage exercises
required/strict draft fields, exact/multiline/ambiguous occurrence mapping,
large repeated-span and 1,100-claim non-recursive cases, protected-token and
qualifier retention, local metadata ownership, payload-free
errors, cache invalidation, one-call retrieval/Judge behavior, `2N` call
ceilings, pair extraction N/A, and historical artifact readability. The new
full-document benchmark file is design documentation only.

No dependency or Phase 4/5 production code was added. The source-of-truth plan
budget remains unchanged.

## Phase 3E.2 pair-dev stability provenance increment

The stability attempt changed no production or test Python. It adds only
generated runtime provenance plus status, decision, budget, and manifest
documentation. Production remains approximately 8,858 lines and tests 7,229
lines, so the Phase 3E.2 implementation delta remains +208 production and +377
tests from its Phase 3E starting point.

The runtime footprint consists of one safe smoke artifact, four canonical
run-01 artifacts, and one safe run-02 failure artifact. Run 03 and the
three-run aggregate were intentionally not generated after the operational
failure. Retaining the complete run separately from the safe failure record is
necessary for reproducible provenance and does not add a product feature,
dependency, Agent, provider, or Phase 4/5 code. The source-of-truth plan budget
remains unchanged.

Final offline checks did not change the code budget: 402 tests passed in 1.80
seconds and compileall passed. Ruff 0.15.21 was available and both requested
checks were run; lint reported 157 findings, including 34 fixable, while format
check reported 49 files would reformat and 7 already formatted. No automatic
fix or formatting was applied because it would invalidate the frozen code
hash. `mypy` was not installed and was not run.

## Phase 3E.3 Judge-validation and stability increment

Phase 3E.3 starts from 8,858 production and 7,229 test Python lines at commit
`1663bb0766780865bcbd775058bb8d3c0d1b7e8a`. The offline repair checkpoint has
9,673 production and 7,800 test lines: net +815 production and +571 tests.

| Area | Insertions | Deletions | Net |
|---|---:|---:|---:|
| `agents/judge.py` | 42 | 12 | +30 |
| `eval/models.py` | 33 | 1 | +32 |
| `eval/runner.py` | 18 | 5 | +13 |
| `eval/stability.py` | 740 | 0 | +740 |
| **Production total** | **833** | **18** | **+815** |
| Tests | 583 | 12 | +571 |

The 740-line stability module is the dominant increment. It owns two bounded,
secret-free artifacts: the repeated single-case smoke and the three-run
aggregate. The aggregate validates frozen provenance, preserves every baseline
headline metric, computes macro-F1 distribution statistics and live-baseline
deltas, records per-case relation agreement, and handles failed attempts
without fabricating partial metrics. This is evaluation/provenance machinery,
not a new product Agent, provider, retriever, Phase 4 feature, or dependency.

The smaller production changes type Judge scope failures, version that local
validation contract, and carry only the allowlisted failure code into future
safe artifacts. Tests scale with the integrity boundary and aggregate blast
radius: semantic relation acceptance, evidence scope, deterministic conflicts,
payload exclusion, smoke repetition, successful aggregation, and failed-run
accounting are covered.

Live artifacts and documentation add no production/test Python. The five-call
smoke and three complete dev runs consumed 98 calls and 117,883 tokens with no
retry or repair. The 15-artifact bundle is retained as runtime provenance. No
v3 holdout, Phase 4/5 code, dependency, remote, or formatting sweep was added;
the source-of-truth plan budget remains unchanged.

Final verification passed 413 tests and compileall. Ruff lint passed on only
the eight modified Python files. The corresponding format check reported that
all eight would be reformatted; it was intentionally not applied after the
live freeze, so the recorded code and prompt/schema hashes remain valid.

## Phase 3G-B adaptive Chinese dev increment

Phase 3G-B adds 337 net production lines. This is slightly below the
approximate 400–700 target because the implementation reuses the existing
Single Agent, Judge, source model, and metrics contracts, and changed-file Ruff
formatting compacted several historical modules. No requested behavior or
safety check was omitted to reduce the count.

| Production area | Net lines |
|---|---:|
| Chinese retrieval and checkability | +70 |
| Adaptive baseline and shared prompt/chunk helpers | +132 |
| Router and bilingual relation policy modules | +111 |
| Models, runner, cache, CLI, stability, dataset compatibility | +24 |
| **Production total** | **+337** |

Focused tests add approximately 442 net lines. The deterministic generator and
leakage auditor are non-production eval tooling and are intentionally kept
outside `src/`. They construct 72 cases, recompute routes and hashes, and audit
prior corpora without calling a model. No dependency, embedding store, vector
database, Phase 4 code, v4 data, or unrelated formatting sweep was added.

## Phase 3G-C live-dev provenance increment

Phase 3G-C changes no production or test Python. It adds only generated
evaluation provenance, a pre-run readiness registration, and documentation.
The runtime footprint is one successful smoke artifact, five artifacts for
each of two complete runs, one sanitized failed-attempt artifact for Run 03,
one persistent call/token ledger, and the incomplete aggregate.

Run 03 stopped on its first schema failure after 46 provider responses. No
success artifacts or partial metrics were retained for that round, and no
retry, repair, continuation, or rerun occurred. The aggregate keeps complete
metrics and pairwise diagnostics for Runs 01 and 02 while marking three-run
metrics and readiness quality gates unavailable. This is provenance and
evaluation reporting, not product code.

No dependency, Agent, provider, retriever, vector store, v4 data, Phase 4
feature, remote, or formatting sweep was added. The source-of-truth production
budget and all Phase 3G-B frozen code hashes remain unchanged.

## Phase 3G-D schema recovery increment

Phase 3G-D adds 240 net production lines, within the requested approximate
150–300 line range.

| Production area | Insertions | Deletions | Net |
|---|---:|---:|---:|
| `cache.py` | 3 | 0 | +3 |
| `eval/baselines.py` | 14 | 9 | +5 |
| `eval/metrics.py` | 34 | 41 | -7 |
| `eval/models.py` | 22 | 4 | +18 |
| `eval/runner.py` | 54 | 19 | +35 |
| `model_client.py` | 245 | 59 | +186 |
| **Production total** | **372** | **132** | **+240** |

The dominant increment is the bounded recovery state machine and its
payload-free summary contract. The remaining changes account for physical
provider attempts, raise per-case/live-run ceilings to include one schema-only
retry, version cache keys, preserve historical artifact defaults, and expose
recovery policy and telemetry in canonical run/failure artifacts. The metrics
module's negative net is changed-file Ruff cleanup; no metric behavior was
removed.

Focused tests use a deterministic sequence provider to cover the two-attempt
state machine, strict boundaries, budget checks, all live paths, canonical
artifacts, leakage resistance, and historical hashes. New diagnosis,
recovery-policy, and preregistration JSON files are non-production evaluation
provenance. No dependency, Agent, retriever, v4 data, Phase 4 feature, remote,
or unrelated refactor was added.

## Phase 3G-E live-dev provenance increment

Phase 3G-E changes no production or test Python. It adds only generated
evaluation provenance and result documentation. The runtime footprint is one
successful smoke artifact, five canonical artifacts plus one execution record
for each of two complete runs, one sanitized failed-attempt artifact for Run
03, and an append-only execution preregistration, call/token ledger, and
execution outcome, plus their hash manifest.

Run 03 first recovered one provider schema failure, then stopped on a typed
Judge scope violation. No complete-run artifacts or partial metrics were
retained for that round, no further call was made, and the conditional
three-run aggregate was not generated. Keeping the recovered schema telemetry
alongside the final scope failure is required provenance; it does not add a
product feature.

The attempt consumed 460 calls and 672,550 reported tokens, including one
retry and 2,023 retry tokens. No dependency, Agent, provider, retriever,
vector store, v4 data, Phase 4 feature, remote, or formatting sweep was added.
The Phase 3G-D production/test code budget and every frozen implementation hash
remain unchanged. The same 439-test boundary-compliant suite passed before and
after execution; compileall, diff, secret, and frozen-hash checks also passed.

## Phase 3G-F local verdict ownership increment

Phase 3G-F adds 119 net production lines, within the requested approximate
100–250 line range.

| Production area | Insertions | Deletions | Net |
|---|---:|---:|---:|
| Live Judge semantic contract and local assembly | 110 | 12 | +98 |
| Ownership-aware cache and run provenance | 17 | 2 | +15 |
| Provider schema/example version | 7 | 1 | +6 |
| **Production total** | **134** | **15** | **+119** |

The behavior increment remains narrow: one semantic schema, one local
assembler, and ownership policy versioning. It adds no Agent, provider,
retriever, dependency, cache implementation, holdout, v4 data, or Phase 4
feature.

Tests add 14 cases and 208 net lines. Existing fake responses were migrated
from the obsolete full-verdict shape to the semantic shape; changed-file Ruff
formatting accounts for most surrounding churn. Coverage proves strict
required/additional-property behavior, local ID and locator attachment,
forged-span rejection, extra-ID recovery and terminal second failure,
payload-safe exceptions, cache invalidation, all three live paths, and
historical artifact compatibility.

## Phase 3G-F live-dev provenance increment

The live portion changes no production or test Python after the offline
ownership freeze. It adds one smoke artifact, five canonical artifacts for
each of three complete runs, a preregistration, budget ledger, aggregate
JSON/Markdown, and a byte-hash manifest.

The three rounds used 508 calls plus two smoke calls and 446,877 total
reported tokens. Four first-attempt schema failures consumed four bounded
retries and 3,932 retry tokens; every retry recovered, and all final
operational failure counts were zero. Keeping those failures explicit is
evaluation provenance, not a new product feature.

No dependency, Agent, provider, retriever, cache implementation, holdout,
v4 dataset, Phase 4 feature, remote, or formatting sweep was added. The
offline production increment remains +119 net lines, and all frozen code,
prompt/schema, config/recovery/ownership, policy, and dev-dataset hashes
remain unchanged.

## Phase 3H-A blind-data increment

Phase 3H-A adds zero production lines and changes no existing test or runtime
behavior. The increment is confined to the six-file `eval_sets/v4_zh/`
candidate pack, one focused data-contract test module, and status
documentation.

The data footprint is deliberate: 24 compact source records, 72 three-per-
source candidate records, a blind reviewer packet, an annotation guide, a
leakage report, and a hash-bearing preregistration manifest. Long snapshots
retain only enough genuine multi-topic technical text to exercise retrieval;
available snapshots do not mirror full upstream pages.

No dependency, Agent, provider, retriever, cache, production policy, gold
dataset, evaluation result, Phase 4 feature, remote, or repository-wide
formatting change is included. Model calls are zero, and consumed corpora are
not parsed by the new tests.

## Phase 3H-B single-human freeze increment

Phase 3H-B adds zero production lines. Its executable increment is one
non-production freeze script plus focused data-contract tests. The generated
data consists of 72 gold rows, 72 case hashes, 24 strict runtime source
fixtures, and one review manifest.

The runtime fixture is a mechanical projection of the unchanged rich source
snapshot, not a second source corpus. It exists solely to satisfy the current
strict `SourceFixture` loader contract without widening production schemas.
No reviewer identity, reviewer ID, or full note is copied into repository
artifacts.

No dependency, provider call, Agent, retriever, cache, prompt/schema, policy,
production module, gate execution, Phase 4 feature, remote, or formatting
sweep was added. Consumed inputs remain byte-hash-only.

## Phase 4A reliability and SARIF increment

Phase 4A adds 479 net production Python lines, within the requested 400–700
line target.

| Production area | Net lines |
|---|---:|
| Deterministic SARIF renderer and bounded atomic writer | +322 |
| Typed local-validation and failure-code contract | +73 |
| Runner, baseline, artifact, CLI, and compatibility integration | +84 |
| **Production total** | **+479** |

The SARIF module intentionally owns format construction, stable ordering,
privacy exclusion, URI encoding, and the output filesystem boundary in one
place. The small error module is shared by baselines, artifact schema, and
runner taxonomy so the allowlist is not duplicated. Existing policy,
canonical audit models, source locators, and CLI pipeline are reused.

No new provider, Agent, retriever, cache, database, web search, UI, or
dependency is added. The GitHub workflow and tests are non-production code.
Phase 3 remains closed with known limitations and Phase 4 eligibility remains
false; this increment is engineering continuation, not a release-gate pass.

## Phase 4B full-document benchmark increment

Phase 4B adds 1,594 net production Python lines, exceeding the requested
500–900 line target by 694 lines. The excess is accepted and itemized because
the requested deliverable combines a new strict dataset contract, two
end-to-end orchestration modes, extraction alignment, fifteen metric families,
bounded stage telemetry, safe failure artifacts, and canonical output. No
line count is attributed to a copied Judge, Router, Retrieval, policy, SARIF,
provider, or schema-recovery implementation.

| Production area | Net lines |
|---|---:|
| Full-document schemas, dataset hash/span/binding validation | +320 |
| Provider-attempt/token budget and payload-free failure taxonomy | +145 |
| Single Agent and Miner/Adaptive/Judge orchestration | +455 |
| One-to-one alignment and end-to-end metric contract | +275 |
| Canonical audit, policy/SARIF, stable artifact output and comparison | +382 |
| Model schema registration and public eval exports | +17 |
| **Production total** | **+1,594** |

The larger increment remains one cohesive eval module because its internal
types share the same claim-span, citation-binding, telemetry, and artifact
invariants. Splitting those invariants across parallel benchmark services
would reduce file size visibility without reducing production behavior.

Existing production paths are reused directly:

- Markdown parsing and source locations;
- paragraph-scoped `ClaimMinerAgent`;
- `adaptive_live`, source chunking, and lexical retrieval;
- `ClaimJudgeAgent` scope/evidence/guard validation;
- schema-recovery-v1 and provider telemetry;
- canonical `AuditArtifact`, policy decisions, and SARIF.

Non-production additions are the deterministic dataset generator, 24
Markdown documents, 120 gold rows, 72 source fixtures, preregistration and
manifest files, plus focused tests. No dependency, provider, vector store,
web search, UI, Challenger, Scout, second provider, holdout, or v5 is added.

## Phase 4B.1 provider-contract increment

Phase 4B.1 adds 356 net production Python lines: 455 additions and 99
deletions across the existing model adapter, full-document evaluator, cache
identity, and semantic output models.

| Production area | Net lines |
|---|---:|
| Safe provider telemetry and schema diagnostics | +151 |
| Full-document dev-only loading, provenance, and failure propagation | +195 |
| Output reason constraints and policy/cache versioning | +10 |
| **Production total** | **+356** |

The increment reuses the existing OpenAI-compatible client, Pydantic models,
schema-only recovery, full-document orchestration, and cache key. It adds no
Agent, provider, retriever, database, web search, UI, holdout, or test-split
execution. New test and smoke artifacts are non-production.

## v0.1 bounded product vertical slice

The pivot starts at commit
`a0051dd62c5f4123a0e5a99929a00927b24be41d`. Its production target is
900-1,400 net lines, with a hard design-reduction checkpoint before 1,500.
The implementation finishes at 1,493 net production Python lines: 1,571 lines
in new modules plus a net 78-line reduction across existing production files.
This is 93 lines above the target range and seven lines below the mandatory
design-reduction boundary.

| Production area | Net lines |
|---|---:|
| Audit Coordinator typed planning contract | +118 |
| Evidence Scout, search protocol, Tavily adapter, and fixture search | +302 |
| One-shot high-risk Challenger | +187 |
| Product Controller, statuses, trace, persistence, and patch output | +964 |
| CLI, model schema registry, and replacement of the old pipeline | -78 |
| **Production total** | **+1,493** |

The result stays below the 1,500-line stop boundary. The main size driver is
the 964-line Controller because it owns citation binding, discovery dispatch,
per-claim isolation, review/challenge dispatch, canonical artifact assembly,
terminal telemetry, patch generation, and typed budget-exhaustion handling.
The 93-line target overage preserves existing one-shot schema recovery and
classifies exhausted Miner, Scout, Judge, and Challenger calls without
collapsing them into generic Agent errors. The previous citation pipeline was
replaced by a compatibility export rather than retained as a second
orchestration path.

No Agent SDK, vector store, second provider, Web UI, PDF path, benchmark,
holdout, or automatic Markdown rewrite was added. Tests, fixture content,
workflow YAML, and documentation are outside the production-Python count.
