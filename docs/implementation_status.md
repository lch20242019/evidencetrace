# Implementation status

Source of truth: `/Users/USERNAME/Projects/multi_agent_researchops_project_plan.md`
(version 3.0, 2026-07-26).

Last updated: 2026-07-26 UTC

## v0.1 product vertical slice

Status: **bounded product workflow implemented; real-document dogfood and
publishing remain**

The active deliverable is a user-facing technical-document audit workflow,
not another benchmark. `evidencetrace check` now runs the existing Markdown
parser and Claim Miner through a deterministic Controller, a two-plan Audit
Coordinator, citation-first source resolution, an optional bounded Evidence
Scout, the isolated Claim Judge, a one-shot high-risk Challenger, existing
policy, and SARIF.

The Controller owns plans, tools, budgets, artifacts, retries, and policy.
Unknown claim IDs, duplicate work, citation bypass, invalid stage actions,
and over-budget discovery plans are rejected. Coordinator failure selects a
deterministic fallback. Claim-level Miner, Scout, source, Judge, or Challenger
failure is recorded with a safe allowlisted code while unrelated claims
continue. Global parse and artifact failures remain document-terminating.

The CLI supports `check <path>`, `--discover`, `--changed-from`,
`--suggest-patch`, `--sarif`, and a byte-stable offline `demo`. Suggested
patches are written as candidate diffs and are never applied. The GitHub
workflow uploads SARIF and a summary before preserving policy/partial status;
fork PRs without secrets run deterministic checks and report skipped live
capabilities.

The demo repository is `examples/product-demo/`. Search and source access use
fixtures only. This implementation made zero real model and Tavily calls.
PyPI publication, public Action packaging, and dogfood against independently
chosen real ADR/RFC/README changes are still pending.

Historical status is unchanged: Phase 3 is
`completed_with_known_limitations`, `phase4_eligible=false`, and no formal
blind-holdout F1 was measured. Consumed datasets and historical artifacts are
retained without extension or reinterpretation.

## Phase 0 — product contract and demo

Status: **complete for implementation; public naming remains blocked**

Delivered:

- README first screen with the literal category, target `uvx` commands,
  before/after contradiction, exact source span, minimal Action configuration,
  and “not a truth detector” boundary.
- Exactly three Markdown input fixtures: bad/bug, simple work, and complex work.
- Eight controlled bad-document error classes backed by a deterministic offline
  source manifest.
- Hand-authored expected `audit.md` with 1-based source lines.
- Relation, corroboration, and default severity definitions.
- GitHub/PyPI/trademark preflight record.
- Design-decision and interview-note logs.

Verification run on 2026-07-10:

```text
python3 -m pytest -q tests/test_phase0_contract.py
6 passed in 0.01s

python3 -m compileall -q tests
passed
```

Static-tool status:

- `ruff` and `mypy` are not installed on this server.
- A project virtual environment could not be created because system
  `ensurepip`/`python3-venv` is unavailable.
- Per maintainer direction, no sudo/system install and no further tool download
  will be attempted. Full ruff/mypy checks are deferred to the maintainer's
  local environment; this is an environment limitation, not a passing result.

Known release blocker:

- GitHub contains two exact `EvidenceTrace` repository names, including one
  materially overlapping verification project. See `docs/name-availability.md`.
  Local implementation continues under the explicit working directory/name;
  publication does not.

Plan differences/clarifications recorded:

- `DD-0001`: keep the colliding name only for local implementation.
- `DD-0002`: treat the requested three documents as input fixtures; expected
  audit and project docs are not counted as work-document fixtures.
- `DD-0003`: use reserved `.invalid` URLs plus an offline source manifest. The
  resolver/fetcher is deliberately not implemented before Phase 2.

Deferred by phase boundary:

- No fetch, retrieval, model, Agent, Judge, renderer, GitHub Action, or live CLI
  behavior has been implemented.

## Phase 1 — deterministic core

Status: **complete for the deterministic core**

Delivered (scope gate):

- Package skeleton and Pydantic contracts.
- Markdown paragraph/link/reference/footnote/bare-URL parsing with source lines.
- Git unified-diff parsing and direct changed-paragraph selection.
- Canonical `audit.json` artifact manager, suppression, and policy rules.
- At least 20 parser fixture cases plus schema, diff, policy, and artifact tests.
- No Phase 2 behavior.


Delivered:

- Package skeleton and Pydantic contracts in `src/evidencetrace/`.
- Markdown paragraph/link/reference/footnote/autolink/bare-URL parsing with source lines, columns, and original-string offsets.
- Git unified-diff parsing and direct changed-paragraph selection for added, modified, deleted, renamed, copied, binary, and combined-diff diagnostics.
- Canonical `audit.json` artifact manager with schema validation, atomic write, collision refusal, and path traversal guards.
- Suppression directives, claim-type suppression, include/exclude policy, and plan-defined severity mapping.
- 29 named parser cases (including parametrized cases) plus schema, diff, policy, artifact, package, and Phase 0 contract tests.
- No Phase 2/4 behavior: no fetch, retrieval, model, Agent, Judge, renderer, GitHub Action, cache, or live audit command.

Verification run on 2026-07-10:

```text
python3 -m pytest -q
124 passed in 0.28s

python3 -m compileall -q src tests
passed
```

Static-tool status remains the Phase 0 environment limitation: ruff and mypy are not installed on this server, and no sudo/system installation was used. Run the project-local quality gate in the maintainer's local environment:

```bash
ruff check .
ruff format --check .
mypy src
```

Plan differences/clarifications recorded for Phase 1:

- `artifacts.py` is an explicit module because the plan requires a run artifact manager even though the example tree omitted the file.
- `mdit-py-plugins` was not added: footnote definitions and occurrences use a small deterministic scanner over markdown-it block ranges, avoiding a new dependency while preserving exact source spans.
- `parse_markdown_file` accepts `repo_root` for absolute input paths and emits only project-relative POSIX paths; `parse_markdown` itself rejects absolute paths through the schema.
- Pure deletion hunks retain `deletion_anchor` metadata but select no new-side paragraph. Neighbor context is explicitly deferred to the plan's later workflow phase.
- Phase 1 writes only canonical `audit.json`; claims and verdicts remain empty typed collections until the citation-audit phase.

## Phase 2 — citation audit MVP

Status: **complete for the citation-audit MVP**

Delivered:

- SSRF-aware HTTP(S) fetcher with DNS/IP validation, redirect revalidation,
  IP pinning for live connections, bounded total/phase timeouts, MIME and
  decompressed response-size limits, and bounded retry policy.
- Sanitized HTML/text extraction into heading-aware chunks with source URL,
  retrieval time, locators, and literal-source evidence text.
- SQLite FTS5 lexical retrieval with a minimal BM25 fallback, numeric/date/
  version/entity boosts, and bounded adjacent context.
- Small `ModelClient` protocol, OpenAI-compatible adapter, deterministic fake
  model, and schema validation for every model response.
- Paragraph-local Claim Miner and narrow-context Claim Judge. Judge validates
  that substantive verdict spans are real candidate source substrings.
- Deterministic numeric, date, semantic-version, negation, comparison, and
  source-span signals supplied to the Judge; they do not replace semantic
  judgement.
- Citation-audit pipeline, `evidencetrace demo`, live `evidencetrace check`,
  canonical `audit.json`, derived `audit.md`, and terminal rendering from the
  reread canonical artifact.
- SQLite cache key/storage contracts including source hash, claim hash, model
  id, prompt version, and retrieval configuration version.

Security decisions:

- Only HTTP(S) is accepted. Local/private/loopback/link-local/reserved/
  multicast/unspecified and cloud metadata destinations are rejected, including
  every resolved DNS address and each redirect target.
- URLs with credential-bearing query keys are rejected before entering httpx;
  request/response secrets, cookies, Authorization, and sensitive headers are
  neither persisted nor returned in source metadata.
- All network tests use mock transports or monkeypatched DNS/time; no test
  makes a real network request.

Deferred by phase boundary:

- No formal eval dataset/baseline/mutation generator, GitHub Action/SARIF/PR
  annotations, Scout, Challenger, Web search/query rewrite, local corpus,
  PDF, dense retrieval/embedding, second provider, UI, report generation, or
  automatic Markdown editing.

Known Phase 2 limitations:

- The deterministic demo Judge is conservative lexical-plus-signal logic; it
  is not an evaluated or calibrated NLI system. Live semantic quality requires
  a configured OpenAI-compatible model.
- Credential-bearing/signed URLs are intentionally unsupported to prevent
  secrets from entering HTTP logs. Authenticated/private sources therefore
  require a future safe credential design.
- HTML extraction uses the stdlib parser and targets static HTML; it does not
  execute JavaScript or support login flows.

Verification:

```text
python3 -m pytest -q
257 passed in 0.49s

python3 -m compileall -q src tests
compileall completed successfully
```

Static-tool status: `ruff` and `mypy` are not installed on this server and were
not downloaded. No sudo or system package installation was used.

Code size snapshot (physical Python lines):

- `src/`: approximately 4,628 lines.
- `tests/`: approximately 3,085 lines.
- The core package has exceeded the plan's 4,000-line target by about 628
  physical lines. This is recorded as a scope/budget warning rather than hidden.

## Checkpoint A — repository hygiene and budget audit

Status: **complete**

- The project was not a Git repository at the start of the checkpoint. A local
  empty `.git/` was initialized with no remote and no commit.
- `.gitignore` explicitly excludes Python caches, pytest/mypy/ruff caches,
  virtual environments, build outputs, egg-info, and `.evidencetrace/`.
- Generated `__pycache__/`, `.pyc`, and `.pyo` files were removed.
- `retrieval/fetch.py.orig` was compared with `fetch.py`; it contained only an
  older one-line redaction-before-validation variant and no unique behavior.
  It was removed as a redundant backup.
- Unreferenced root-level duplicate Phase 2 modules were removed. The packaged
  implementation remains solely under `src/evidencetrace/`.
- The detailed physical-line review and future increment estimates are in
  [`docs/code-budget-review.md`](code-budget-review.md). The source-of-truth
  plan's budget was not modified.

Checkpoint verification is recorded together with the Phase 3A gate below
after the complete test suite is run.

## Phase 3A — eval infrastructure, seed dataset, and baselines

Status: **infrastructure complete; 0.70 stop gate not reached; Phase 4 blocked**

Delivered:

- Pydantic eval/source/result/manifest schemas and JSONL loading with source
  content hashes, derived case hashes, checked-in frozen hashes, duplicate
  claim/split-leakage checks, and literal gold-span validation.
- 80 offline pairs: 30 hand-authored natural pairs, 30 deterministic
  single-slot mutations, and 20 difficult cases. Splits are 40 dev / 40 test.
- Annotation counts: 60 `deterministic_gold`, 0 `human_reviewed`, and 20
  `provisional`. Provisional cases are queued in `eval_sets/REVIEW_QUEUE.md`
  and excluded from headline metrics.
- Because no test case is human-reviewed, `metrics.json`, `run_manifest.json`,
  and `eval_report.md` mark the whole benchmark `provisional`; these metrics are
  internal stop-gate results and are not README/public headline claims.
- Eval-only `rules_only`, `single_agent_full_document`, and
  `miner_plus_judge` baselines over identical case/source inputs. The
  single-agent baseline is not a product default.
- Optional OpenAI-compatible live miner-plus-judge client behind explicit
  `--live --model`; absent credentials are recorded as
  `skipped_missing_credentials`, not reported as a pass.
- Claim extraction, relation/per-label/confusion, contradiction/entailed/
  partial/high-confidence/abstention, source-span token F1, Recall@5/MRR/
  coverage, latency/calls/cache/cost/false-block metrics. Unknown cost is
  `null` with an explanation.
- `evidencetrace eval` artifacts: `eval_report.md`, `eval_results.jsonl`,
  `metrics.json`, and `run_manifest.json`, including dataset/split hashes,
  model/prompt/retrieval configuration, Python version, execution time, and
  Git state (`uncommitted` because the repository intentionally has no commit).

Internal stop-gate results use the 30 non-provisional test cases
(`deterministic_gold` or `human_reviewed`) only:

| Baseline | Relation macro-F1 | Contradiction recall | Recall@5 | Span token F1 |
|---|---:|---:|---:|---:|
| `rules_only` | 0.596 | 0.733 | 1.000 | 0.662 |
| `single_agent_full_document` | 0.475 | 0.733 | 1.000 | 0.579 |
| `miner_plus_judge` | 0.548 | 0.600 | 1.000 | 0.662 |

The primary `miner_plus_judge` result is below the Phase 3 0.70 macro-F1 stop
gate. Labels were not changed and test cases were not used for prompt tuning.
Phase 4 work must not begin; future fixes must be developed on dev and rerun
against the frozen test split.

Verification on 2026-07-10:

```text
python3 -m pytest -q
274 passed in 1.75s

python3 -m compileall -q src tests
compileall completed successfully
```

`ruff` and `mypy` were not present and were not run or installed. No sudo or
system package installation was used.

Current physical Python lines:

- `src/evidencetrace/eval/`: approximately 1,286.
- Phase 3A production increment: approximately 1,305 including CLI changes.
- `src/` total: approximately 5,932.
- Phase 3A test increment: 330.
- `tests/` total: approximately 3,415.

Benchmark limitations:

- The seed sources and labels are fictional and author-generated; there is no
  second annotator and no `human_reviewed` case yet.
- Pair-level extraction metrics are trivially strong because each seed pair
  contains one expected claim; the set is not a full-document extraction
  benchmark.
- The difficult queue is provisional and cannot support headline claims.
- The small repeated-source fixture set makes retrieval Recall@5 optimistic.
- Deterministic baselines make zero model calls; live quality, token usage, and
  cost were not measured.

No Phase 4 GitHub Action, SARIF, PR annotation, Scout, Challenger, Web search,
query rewrite, local corpus, PDF, vectors/embeddings, second provider, Web UI,
report generation, or Markdown rewriting was implemented.

## Phase 3B — eval validity repair and dev-only remediation

Status: **dev gate reached; benchmark remains provisional; Phase 4 blocked**

### Checkpoint 0

- The pre-change gate remained `274 passed`; `compileall` passed.
- Secret-pattern and generated-file checks found no credential file or
  committable cache. Python/test/type-check caches and `.evidencetrace/` were
  ignored.
- A preconfigured Git identity was present; neither local nor global identity
  was changed. Local commit `49c6aef phase3a-provisional-baseline` was
  created. No remote exists.
- `eval_sets/core.jsonl`, its frozen hashes, and every
  `eval_runs/core/` file are protected by byte-for-byte SHA-256 regression
  tests and remain unchanged.

### Baseline and metric validity

New artifacts use these exact names:

- deterministic: `lexical_rules`, `lexical_full_source`,
  `miner_judge_deterministic`;
- live-only: `single_agent_live`, `miner_judge_live`.

`single_agent_live` makes exactly one schema-validated model call with one
claim and the complete source. It does not call Miner, Retriever, or Judge.
`miner_judge_live` uses the isolated Miner and Judge path. Deterministic paths
record zero model calls. The two live paths use the same configured client,
case/source input, and final `EvalPrediction` contract. They appear in a
separate table and are never synthesized from deterministic output.

Pair-level extraction metrics are now `null` with
`status=not_applicable`. Verification reports:

- observed-gold-label and fixed-six-label macro-F1;
- weighted-F1, balanced accuracy, and per-label support;
- explicit coverage status and thresholds;
- an auditable false-block numerator/denominator, where a block is predicted
  `contradicted` for a non-contradicted gold case;
- repeated-source Recall@5 only as `fixture_sanity`.

Phase 3B originally defined `benchmark_validity` as `provisional`,
`diagnostic_contaminated`, or `human_reviewed_holdout`; Phase 3D adds the
distinct `single_human_synthetic_holdout` state. Coverage failure or diagnostic
contamination prevents a release PASS regardless of a scalar F1.

### Dev-only repair record

All rule changes were derived from v1 dev and covered with new, generic
fixtures. No case ID or complete benchmark sentence was hard-coded.

| Saved stage | Baseline name | Observed macro-F1 | Contradiction recall | Entailed precision | Partial F1 | High-confidence error |
|---|---|---:|---:|---:|---:|---:|
| `phase3b_dev_before` | legacy `miner_plus_judge` | 0.422 | 0.471 | 0.600 | 0.333 | 0.348 |
| `phase3b_dev_iteration1` | `miner_judge_deterministic` | 0.694 | 0.765 | 0.909 | 0.556 | 0.333 |
| `phase3b_dev_after` | `miner_judge_deterministic` | 0.828 | 0.882 | 1.000 | 0.769 | 0.125 |

The final dev result also has fixed-taxonomy macro-F1 0.552, weighted-F1
0.910, and balanced accuracy 0.941. Core semantic supports are 17 entailed,
5 partially-entailed, and 17 contradicted, exceeding the configured minimum of
3. `not_in_source` and `source_unavailable` have zero dev support, and
`not_checkable` has one case; the fixed-taxonomy score and provisional
validity remain essential caveats.

Final deterministic dev comparison:

| Baseline | Observed macro-F1 | Fixed macro-F1 | Contradiction recall | Entailed precision |
|---|---:|---:|---:|---:|
| `lexical_rules` | 0.484 | 0.322 | 0.647 | 0.625 |
| `lexical_full_source` | 0.461 | 0.307 | 0.647 | 0.630 |
| `miner_judge_deterministic` | 0.828 | 0.552 | 0.882 | 1.000 |

The automated dev conditions are met, but
`phase4_eligible=false`: this is a tuning result, not a Phase 3 release
result. The frozen Phase 3B code hash is
`eb5a01fa0065a5b6643c4c1d60cca746d1f626015e54ce0d4382343859120f62`.

### Judge remediation

- Negation now has one producer/consumer contract: an aligned
  `negation_mismatch` is an `error`.
- Recommendation morphology covers recommend/recommends/recommended/
  recommending, prefer forms, should/ought-to, and Chinese advice markers.
- Conservative entity mismatches require aligned predicate/context; pronouns
  are not treated as replacement entities.
- Scope, denominator, qualifier, conjunction, and aligned-clause signals prefer
  partial support when evidence is incomplete and contradiction for explicit
  opposing qualifiers.
- Deterministic and live Judge paths derive the same signals. A live
  `entailed` output cannot bypass numeric, date, version, entity, negation, or
  explicit qualifier conflicts. Every substantive span remains a literal
  candidate substring.
- Lexical overlap alone caps deterministic entailment confidence; comparison
  and qualifier cautions cap both deterministic and live confidence.

### v1 and v2 benchmark state

No post-repair v1 test benchmark artifact was generated, and no v1 test result
was inspected or used for remediation. The previously exposed v1 test is
`diagnostic_contaminated` and cannot be a final gate. During integration,
legacy unit tests briefly exercised the old all-split runner in temporary
directories before those tests were converted to dev-only; their outputs were
not retained or consulted. This process detail is recorded rather than
claiming new blindness for v1.

`eval_sets/v2/` contains:

- 18 new synthetic dev cases;
- 60 unreviewed holdout candidates, with 10 model proposals for each relation;
- 12 hashed synthetic sources (10 available, 2 unavailable);
- an annotation guide, a 60-item label-hidden review packet, and a hashed
  manifest.

Every holdout item is `synthetic_candidate`,
`unreviewed_candidate`, `provisional`, and `not_run`. Human relation,
evidence, and reviewer-note fields are blank; no proposal is called gold or
`human_reviewed`. The holdout was not executed. Its next action is human
annotation by the maintainer or a second annotator.

### Model and scope boundary

`OPENAI_API_KEY` was absent. Both `single_agent_live` and
`miner_judge_live` are recorded as `skipped_missing_credentials`; no real
model evaluation, token accounting, or cost claim was made.

Phase 3B added 896 net production Python lines (96 over the approximate
800-line target), bringing `src/` to 6,828 lines. Tests grew by approximately
842 lines to 4,257. The overage rationale is in
`docs/code-budget-review.md`; the source-of-truth budget was not changed.

Final verification on 2026-07-10:

```text
python3 -m pytest -q
316 passed in 1.51s

python3 -m compileall -q src tests
completed successfully
```

`ruff` and `mypy` are not installed and were not run, downloaded, or
reported as passing. No sudo or system package installation was used.
`git diff --check`, the v1 frozen-hash audit, the v2 manifest/hash audit, and
the credential-pattern scan passed. Generated caches remain ignored and are
not repository changes.

No Phase 4 Action/SARIF/annotation work, Scout, Challenger, Web search, local
corpus, PDF, embeddings/vectors, second provider, Web UI, report-generation
product feature, or Markdown rewriting was implemented.

## Phase 3C — single-human annotation freeze and human-model comparison

Status: **single human review complete; formal Phase 3 gate and Phase 4
blocked**

Reviewer A's 60-case record was validated before freezing: it contains exactly
60 unique cases; every relation is in the six-label taxonomy; every relation
and note is populated; every required evidence span is a literal contiguous
source substring; cases `v2_holdout_041` through `v2_holdout_050` are
`source_unavailable`; and case IDs, source IDs, source text, availability,
and source hashes match the candidate packet and fixtures. No substantive
human relation, evidence span, or note was rewritten during the provenance
freeze.

The project owner confirmed Reviewer A's provenance on 2026-07-14. The record
is now `single_human_review_complete`, counts as one human review, and records
`viewed_model_review=true`. This truthfully records that the owner viewed the
model comparison after completing the substantive judgments; it does not
remove the record's human status or elevate it beyond a single human review.
Its pre-freeze SHA-256 was
`36ae72349da5a25f20564ede487f020bf20a7d14304731bcf934f7d0f4427ffb`;
its frozen SHA-256 is
`4b4d3d8e25960d71e9b7cb151b1caec25fe9089dcd7bf1dd0a5352e13e470192`.

The file previously named `human_reviewer_b.md` was a completed model review,
not a human annotation. It was preserved byte-for-byte outside the repository
as `model_m2_openai_codex.md`, SHA-256
`635c7ad243b14d454d1d01b4a47e1f66dd6079fd209bde3595f87f83438fe2b6`.
A new blank `human_reviewer_b.md` is reserved for a future second person, and
`adjudication.md` remains blank. The earlier `model_m1_codex.md` remains
preserved at SHA-256
`e982bb36e137d1035a76b06808230a6b8497158f6c4547a88d23843f467f9527`.
All private records remain outside Git.

The derived external `human_model_resolution.md` keeps human and model fields
separate. Final relation and evidence always equal Reviewer A; model notes are
referenced only where relation agrees and are never merged into human notes.
Human A and Model M2 agree on 56 of 60 cases and disagree on 4, for raw
relation agreement `0.933333`. This is **human-model agreement**, not
inter-human agreement; no inter-human Cohen's kappa was computed. Artifact
SHA-256:
`2b273f70219e52bb815beec69fc50112e6874b504cced18a3db75a327aa377ac`.

At the Phase 3C checkpoint, the benchmark's highest provenance was
`single_human_review`; there was no second independent human review or
adjudicated human gold, and the v2 holdout had not been executed. Phase 3D
decision DD-0022 later made a second human optional for the internal gate while
preserving those provenance limits. The current blockers are separate
authorization for and successful completion of the one-case live smoke,
followed by separate authorization for any full dev rerun and the one-time live
holdout. The frozen `review_manifest.json` is the historical Phase 3C annotation
snapshot; current execution status is recorded in `eval_sets/v2/manifest.json`.
No rules or prompts were adjusted and no Phase 4 functionality was implemented.

Phase 3C verification on 2026-07-14:

```text
python3 -m pytest -q
316 passed in 2.50s

python3 -m compileall -q src tests
completed successfully

git diff --check
passed
```

`ruff` and `mypy` were not installed, so they were not run or reported as
passing. No package was installed and no sudo was used.

## Phase 3D — single-human synthetic holdout ready checkpoint

Status: **one authorized dev rerun consumed and failed schema validation;
no automatic rerun; v2 holdout unauthorized; Phase 4 blocked**

The project decision now permits one internal gate on the frozen Reviewer A
labels without waiting for a second human. The benchmark validity is exactly
`single_human_synthetic_holdout`, and every schema, manifest, metrics artifact,
and report exposes `public_benchmark_eligible=false`. A second independent
human remains an optional quality enhancement and is still necessary for any
future double-human, inter-human, or adjudicated claim. Blank Reviewer B and
adjudication records remain preserved outside Git.

### Frozen dataset

`eval_sets/v2/holdout_single_human.jsonl` contains exactly 60 test cases and
all six relations: 10 entailed, 5 partially entailed, 14 contradicted, 11 not
in source, 10 source unavailable, and 10 not checkable. All 29 substantive
gold evidence spans are exact contiguous source substrings. Gold relation and
evidence come only from Reviewer A. Candidate/model proposals, M1/M2 labels or
notes, Reviewer A's private notes, name, and ID are excluded.

Frozen provenance:

- Reviewer A SHA-256:
  `4b4d3d8e25960d71e9b7cb151b1caec25fe9089dcd7bf1dd0a5352e13e470192`
- Dataset file SHA-256:
  `0d19004b627f4ecabf9dd40fd28303b0539b89f36f3f7f5cec135c47a6582c75`
- Dataset plus source-fixture hash:
  `b5c09264eae626a541a0a79454699e561b336cc1cb7c9606a7fe3dc6885233cd`
- Frozen case-map SHA-256:
  `ca00b89d0b006bc92ba3344a9d018f95ff74d626e9b769bad6fa371cdb610cf5`
- Generator SHA-256:
  `142816b732b21f59cf35f4e0ea24cd4fc31a3f7e2bcaf841263b18d16d534b2b`
- Ready-checkpoint code bundle SHA-256:
  `d5517f6fc6bbe99a0f356ff1409a831eda5fae776d0d77f5cf4fdaa448be6138`
- Post-dev-fix code bundle SHA-256:
  `98e834e554ce0df4c21f911174f60d43b27241a60d5358ba44eeb5656acdb728`
- Ready-checkpoint prompt-bearing bundle SHA-256:
  `12ff21e50c46fe129d0a5e83f5ac2a0c402e606d13c72b6726f8e72e914b0d99`
- Post-dev-fix prompt-bearing bundle SHA-256:
  `be33624084af0e32ce184171ce7c4a0cb789c0573f04444dbcd0eaaab00555f9`

- Authorized-rerun safety code bundle SHA-256:
  `66284bd98b75f7ea1a7160bcf9d8662d2b37401f988378e92264a33df18e9df2`
- Authorized-rerun prompt-bearing bundle SHA-256:
  `845b72bacf789fe3bb08e3a11b178e4dc6e96fb49b4867349c37c9ecb753619c`

The original `holdout_candidates.jsonl`, private Reviewer A record, v1 frozen
artifacts, Phase 3A run, and Phase 3B dev artifacts remain unchanged. No v2
holdout run directory exists.

### Validity, gate, and telemetry

Pair-file validation now supports physically separated dev and test files
while still rejecting an empty dataset, duplicate claims/IDs, source/hash
mismatches, and frozen-case drift. Single-human cases require source, reviewer,
and generator SHA-256 provenance. The internal stop gate requires all six
labels, a real `miner_judge_live` headline, relation macro-F1 >= 0.70, and a
literal source substring for every substantive verdict. The unchanged v0.1
targets of macro-F1 >= 0.75 and contradiction recall >= 0.80 are reported
separately.

The OpenAI-compatible adapter and eval artifacts now record secret-free
per-baseline model calls, provider/model/temperature, provider usage when
available, input/output/total tokens, model-call p50/p95 latency, and
schema/transport failure counts. `single_agent_live` is constrained to one
call per case; isolated Miner plus Judge uses at most two; 60 cases across both
live baselines therefore have a hard expected ceiling of 180. Deterministic
baselines remain zero-call. Cost stays null with an explicit explanation unless
records contain an explicit pricing snapshot; no token price is invented.
API keys, Authorization, request headers, and prompt/source payloads are not
part of telemetry or artifacts.

### Live execution status

On 2026-07-15 the owner configured runtime-only DeepSeek credentials, selected
`deepseek-v4-flash`, and authorized one dev comparison with a maximum of 54
provider calls. The 18-case live dev preflight was started. No v2 holdout
evaluation was run, and no holdout output directory was created.

The run failed closed on `v2_dev_008` with `miner dropped protected factual
tokens: 1.8` before any eval artifact was written. The failure was a local
token-canonicalization defect: the protected-token regex matched `1.8%` as
`1.8`, while the normal tokenizer correctly retained `1.8%`. The fix gives an
explicit percentage branch priority over semantic-version matching and also
prevents an empty `claims` result from bypassing protected-fact validation.
No dev label, holdout data, retrieval rule, Judge rule, or prompt was changed.

Because failed-run telemetry was not persisted, the exact provider call count
is unknown. The fixed execution order bounds it at 26–33 calls: 18 completed
`single_agent_live` calls, seven prior Miner calls, zero to seven prior Judge
calls, and the failing eighth Miner call. At that point, a full dev rerun could
use up to 54 more calls, bringing the cumulative bound to 80–87. The owner then
granted exactly one fresh complete rerun with `deepseek-v4-flash`,
provider-default thinking, requested temperature 0.0, and a hard ceiling of 54
calls. That authorization was consumed by the one-call schema failure described
below and cannot be reused. Failed requests counted toward the ceiling;
automatic retries and automatic reruns were disabled.

Before that rerun, the shared client gained a pre-request hard budget check.
Any live transport, schema, grounding, or local contract failure writes a typed
`failed_attempt.json` with the actual safe call events, reported token
subtotals, latency, failure baseline, and failure case, then stops. It writes
no partial canonical success artifacts and serializes no credentials, headers,
request/response payloads, or exception messages.

The adapter requested temperature 0.0 but did not override DeepSeek V4's
provider-default thinking mode. Therefore 0.0 is a requested parameter, not a
claim about effective provider sampling. Both live baselines use the same
shared provider, model, requested temperature, and thinking configuration.

The one authorized rerun executed once from frozen commit
`bcda80b9ea9c21d46bda05fa2160b88e19f60fa0` with code bundle
`66284bd98b75f7ea1a7160bcf9d8662d2b37401f988378e92264a33df18e9df2`.
It stopped on `v2_dev_001` during the first `single_agent_live` call because
provider output did not satisfy `SingleAgentLiveOutput`.

The typed failure artifact at
`eval_runs/phase3d_v2_dev_live_rerun/failed_attempt.json` records exactly one
model call: 259 input, 391 output, and 650 total tokens; p50/p95 latency
3,571.082 ms; one schema failure; zero transport failures; zero retries; and
zero completed predictions. It is the only file in the run directory, so no
canonical success artifact or partial metric exists. Cost remains null because
no explicit DeepSeek price snapshot was supplied. Artifact SHA-256 is
`e6b7fda4daefe7d3f98911a238c7fdc492c71ab57a4a5d39406274d5f43a640c`.
The authorization is consumed; no automatic rerun is allowed. A new dev
attempt requires new authorization, and no dev quality metric can be reported.

Only a successful dev preflight plus a clean code/prompt/data freeze and
separate authorization for up to 180 holdout calls permits the one-time
holdout run. The runner refuses a nonempty holdout output directory, preventing
a successful artifact from being silently overwritten.

A provider transport or schema failure remains a failed run rather than a
manufactured verdict; the adapter classifies and counts the failed call, but a
successful gate artifact is not produced from a partial evaluation. If the
eventual frozen holdout macro-F1 is below 0.70, Phase 4 remains blocked and any
post-result tuning requires a new v3 holdout. Passing 0.70 would make Phase 4
internally eligible while retaining the single-human, synthetic, non-public
limitations.

Phase 3D ready verification on 2026-07-14:

```text
python3 -m pytest -q
333 passed in 1.86s

python3 -m compileall -q src tests
completed successfully

git diff --check
passed
```

Post-dev-remediation verification on 2026-07-15:

```text
python3 -m pytest -q
335 passed in 1.77s

python3 -m compileall -q src tests
completed successfully

git diff --check
passed
```

Authorized-rerun safety-freeze verification on 2026-07-15:

```text
python3 -m pytest -q
339 passed in 2.77s

python3 -m compileall -q src tests
completed successfully

git diff --check
passed
```

Authorized-rerun execution and failure-provenance verification on 2026-07-15:

```text
python3 -m pytest -q
340 passed

python3 -m compileall -q src tests
completed successfully

git diff --check
passed
```

`ruff` and `mypy` are not installed and were not run or reported as passing.
No dependency was downloaded and no sudo or system installation was used.
Physical Python totals are 8,011 lines under `src/`, 326 lines in the holdout
generator, and 5,509 test lines. The authorized-rerun safety layer adds 420
production lines and 366 test lines over the post-dev-fix checkpoint. Freezing
the failed-run provenance adds no production code and 62 test lines. The
7,500-line recommendation is exceeded by 511 production lines and is
explicitly recorded in `docs/code-budget-review.md`.

No GitHub Action, SARIF, PR annotation, Scout, Challenger, Web search product
feature, query rewrite, local-corpus RAG, PDF, embedding/vector database,
second model provider, Web UI, report-generation product, or Markdown
rewriting was implemented. Phase 4 has not started.


## Phase 3D.1 — OpenAI-compatible structured-output contract hardening

Status: **offline implementation frozen; subsequent one-call contract smoke
passed; full dev rerun and v2 holdout unauthorized; Phase 4 blocked**

This checkpoint repairs the provider format contract and failure
instrumentation only. It does not tune semantic prompts, rules, retrieval,
labels, or model quality. It made zero real model API calls.

Delivered:

- `OpenAICompatibleClient.complete_model` now sends
  `response_format={"type": "json_object"}`, the target
  `model_json_schema()`, a compact schema-valid example, and bounded
  `max_tokens`. The default is 2,048 and the accepted configuration range is
  1–8,192.
- System and user instructions independently require exactly one JSON object,
  every required field, no additional fields, and no Markdown/code fence.
  Adapter prompt version is `openai-compatible-v2`.
- `SingleAgentLiveOutput`, `MinerOutput`, and `JudgeOutput` each have a
  request example validated with the real strict schema. Their relation enums,
  required fields, evidence requirements, and Agent input isolation remain
  unchanged.
- Response parsing accepts only `message.content`; `reasoning_content` is
  ignored. Empty/non-string content, invalid or fenced JSON, and
  `finish_reason=length` fail closed. Final validation uses strict Pydantic
  JSON parsing and performs no field aliasing, relation normalization,
  guessing, retry, or model repair call.
- Future `phase3d1-live-failure-v2` artifacts can persist a bounded
  `schema_diagnostic`: an allowlisted schema name, failure category, at
  most eight redacted/allowlisted Pydantic location/type entries, a category-
  owned static message, allowlisted finish reason, and bounded response content
  length. Validation uses `errors(include_input=False, include_url=False)`;
  messages, context, original input, model/reasoning content,
  prompt/claim/source payloads, credentials, and headers are not serialized.
  Schema, invalid-envelope, and transport exceptions are raised only after the
  original exception scope ends, leaving both cause and context empty.
- Model calls, reported token usage, latency, schema/transport failure counts,
  the pre-request hard budget, and zero automatic retries remain intact.
  Single Agent remains one full-source call; Miner still receives no source;
  Judge still receives only the isolated claim/evidence/source metadata
  contract.

Historical boundary:

- `eval_runs/phase3d_v2_dev_live_rerun/failed_attempt.json` remains byte-for-
  byte unchanged at SHA-256
  `e6b7fda4daefe7d3f98911a238c7fdc492c71ab57a4a5d39406274d5f43a640c`.
  It is a v1 artifact and can establish only that schema validation failed.
  Because raw content and ValidationError details were not retained, no exact
  field path can be recovered or invented. The artifact schema now explicitly
  rejects any diagnostic added to a historical v1 record.
- Frozen v2 gold, Reviewer A provenance, candidates, sources, and historical
  eval artifacts were not modified. The v2 holdout was not run.
- Provider-default thinking remains unchanged. Temperature 0.0 is recorded only
  as requested; actual effective temperature remains null/unknown.
- A subsequent, separately authorized `v2_dev_001` smoke call passed; its
  provenance is recorded below. This did not authorize or execute the full dev
  comparison. A full dev run and the holdout remain separately unauthorized.

Offline verification on 2026-07-15:

```text
python3 -m pytest -q
371 passed in 2.78s

python3 -m compileall -q src tests
completed successfully

git diff --check
passed
```

`ruff` and `mypy` are not installed and were not run or reported as
passing. No dependency was installed and no sudo was used.

Phase 3D.1 freeze:

- Base commit:
  `ab2eaaa5d2f083685842c7611b92d587af7442b0`
- Code bundle SHA-256:
  `79fc7fa4ec8d886556baa8406c0d61253abbcc5010b1ae0a8cd27da26a9ca308`
- Prompt-bearing bundle SHA-256:
  `43e4f32204726f608e0eed44d5c0f3e4779e01984f15d0ecdb14ad27add6326e`
- Physical Python totals: 8,503 production lines and 6,157 test lines.
- Net from the Phase 3D failed-run checkpoint: +492 production lines and +648
  test lines. The production overage is itemized in
  `docs/code-budget-review.md`.

No Action, SARIF, PR annotation, Scout, Challenger, Web search, second
provider, local-corpus RAG, PDF, embedding/vector database, Web UI, holdout
evaluation, or Phase 4 functionality was implemented. Phase 3 has no live gate
result and Phase 4 remains blocked.

## Phase 3D.1 — authorized one-case live contract smoke

Status: **contract passed; one request consumed; full dev and holdout not run;
Phase 4 blocked**

Execution on 2026-07-15 used the existing narrow library path, not the full
`evidencetrace eval --live` command:

- frozen execution commit
  `291314a59f949f676da1d9a10501409382567b9c` with a clean worktree;
- code bundle
  `79fc7fa4ec8d886556baa8406c0d61253abbcc5010b1ae0a8cd27da26a9ca308`
  and prompt bundle
  `43e4f32204726f608e0eed44d5c0f3e4779e01984f15d0ecdb14ad27add6326e`;
- exactly one selected case, `v2_dev_001`, and exactly one selected baseline,
  `single_agent_live`;
- `deepseek-v4-flash`, provider-default thinking, requested temperature 0.0,
  effective temperature null/unknown, `max_calls=1`, and zero retry/repair;
- no `miner_judge_live`, complete dev, deterministic baseline sweep, or v2
  holdout execution.

Result:

- actual provider/model calls: 1;
- strict `SingleAgentLiveOutput` contract: passed;
- relation matched dev gold: true;
- returned evidence existed and passed exact source-substring validation; only
  its SHA-256 is persisted;
- usage: 553 input, 177 output, 730 total tokens;
- latency: 2,151.273 ms; finish reason: `stop`;
- cost: null because no explicit model price snapshot was supplied.

The single safe artifact is
`eval_runs/phase3d1_v2_dev_001_smoke/smoke_attempt.json`, SHA-256
`b0d2ca17493efa3be7b2d37bac4de12fbc38d2d79653bfbd0463bf193745cf14`.
It stores case/source/evidence hashes, contract and quality booleans, frozen
configuration, usage, latency, and safety declarations. It does not store raw
model content, reasoning content, model reason, predicted/gold relation text,
prompt, claim/source text, headers, base URL, request ID, or credentials.

This smoke establishes only that the hardened provider contract worked once on
one dev case. It is not a complete live comparison, benchmark metric, or Phase
3 gate result. The consumed authorization cannot be reused. Discussing a new
full-dev authorization is now possible, but no such run may start without a
new explicit call/cost authorization. The holdout remains unauthorized and
unexecuted; Phase 4 remains blocked.

Post-smoke offline verification on 2026-07-15:

```text
python3 -m pytest -q
372 passed in 2.44s

python3 -m compileall -q src tests
completed successfully

git diff --check
passed
```

No production code or prompt changed. Physical totals are 8,503 production
lines and 6,236 test lines; the smoke provenance adds 79 net test lines only.
`ruff` and `mypy` were not installed, so they were not run or reported as
passing. No package was installed and no sudo was used.

## Phase 3D.1 — authorized Miner/Judge chain contract smoke

Status: **contract passed for both isolated stages; two requests consumed;
full dev and holdout not run; Phase 4 blocked**

Execution on 2026-07-15 used the existing narrow library path and selected
only case `v2_dev_001` and baseline `miner_judge_live`. It did not invoke the
full `evidencetrace eval --live` command. Preflight used clean execution commit
`3fe6b764e813c06c10c965cbfc0d35648cdee7d7`, unchanged code bundle
`79fc7fa4ec8d886556baa8406c0d61253abbcc5010b1ae0a8cd27da26a9ca308`,
and unchanged prompt-bearing bundle
`43e4f32204726f608e0eed44d5c0f3e4779e01984f15d0ecdb14ad27add6326e`.
The client hard limit was two calls. The run used `deepseek-v4-flash`,
provider-default thinking, requested temperature 0.0, effective temperature
null/unknown, and zero retry or repair calls.

Miner result:

- exactly one provider call; strict `MinerOutput` validation passed;
- one atomic claim was produced;
- paragraph and citation scope, protected-fact retention, and claim/slot token
  provenance guards all passed;
- 613 input, 475 output, and 1,088 total tokens;
- 4,316.504 ms latency and `finish_reason=stop`.

Because Miner succeeded with a legal nonempty claim, the chain made its second
and final authorized call. Judge result:

- exactly one provider call; strict `JudgeOutput` validation passed;
- claim ID, source ID, and evidence locator were all members of the restricted
  inputs;
- the evidence span passed literal candidate-substring and substantive-span
  validation, and the deterministic-conflict guard passed;
- the case-local final relation matched dev gold;
- 1,014 input, 535 output, and 1,549 total tokens;
- 4,378.621 ms latency and `finish_reason=stop`.

Total execution was exactly two provider calls, 1,627 input, 1,010 output, and
2,637 total tokens. The two sequential call latencies total 8,695.125 ms.
Schema failures and transport failures were both zero. Cost is null because no
explicit price snapshot was provided; no price was inferred.

The sole safe artifact is
`eval_runs/phase3d1_v2_dev_001_miner_judge_smoke/smoke_attempt.json`, SHA-256
`f01d4cfdb33a2923c5a2ca72b5235b0ee2e94368a60a41fa908b6fe870a273cd`.
It persists only case/source/evidence hashes, frozen configuration, stage
contract and guard results, safe usage/latency/finish-reason telemetry, and
safety declarations. It does not persist model or reasoning content, Miner or
Judge output objects, reason or relation text, prompts, claim/source/evidence
text, headers, base URL, request IDs, or credentials.

Together with the preceding one-call Single Agent smoke, all three live output
schemas (`SingleAgentLiveOutput`, `MinerOutput`, and `JudgeOutput`) have now
passed one narrowly authorized provider-contract smoke. This establishes only
format, schema, isolation, and local guard interoperability on one dev case. It
is not a complete dev comparison, a benchmark metric, a model-quality result,
or a Phase 3 gate result. Both smoke authorizations are consumed; no current
authorization permits another call. The complete dev split and v2 holdout were
not run. Any complete dev comparison requires a new explicit call/cost
authorization, the holdout remains separately unauthorized, and Phase 4
remains blocked. No production code or prompt changed: physical Python totals
are 8,503 production lines and 6,373 test lines, a test-only increase of 137
lines from the preceding checkpoint for the frozen artifact and manifest
regression.

Final offline validation passed: `python3 -m pytest -q` reported 373 passed in
2.92 seconds; `python3 -m compileall -q src tests` and `git diff --check` both
completed successfully. An exact scan for the runtime `OPENAI_API_KEY` across
tracked and unignored files found zero matches, and generic credential-pattern
scanning of changed files found zero matches. Generated `__pycache__`, `.pyc`,
and `.pytest_cache` files were removed afterward. `ruff` and `mypy` were not
installed, so neither was run or reported as passing; nothing was installed and
sudo was not used.

## Phase 3D — complete live dev comparison on the frozen contract

Status: **complete dev run succeeded; provisional numerical dev threshold met;
formal Phase 3 holdout gate not run; Phase 4 blocked**

The project owner explicitly authorized one complete `dev` execution on
2026-07-19 with a hard ceiling of 54 provider requests, zero retry/repair, no
automatic rerun, and no holdout access. Preflight confirmed clean commit
`57aac924abb81ebf8f522c3658ff18d3f6112077`, code hash
`79fc7fa4ec8d886556baa8406c0d61253abbcc5010b1ae0a8cd27da26a9ca308`,
prompt hash
`43e4f32204726f608e0eed44d5c0f3e4779e01984f15d0ecdb14ad27add6326e`,
model `deepseek-v4-flash`, provider-default thinking, requested temperature
0.0, effective temperature null/unknown, 18 unique dev cases, and a new output
directory. The API key was read only from the environment and was never
printed or persisted.

The exact authorized CLI completed successfully once between
`2026-07-19T16:23:31.754259Z` and `2026-07-19T16:27:53.901300Z`. It produced
90 validated results: 18 for each of three deterministic baselines and two live
baselines. All deterministic baselines recorded zero model calls. Live calls
were:

- `single_agent_live`: 18 calls, exactly one per case;
- Miner: 18 calls, exactly one per case;
- Judge: 12 calls after eligible Miner results; the other six cases were
  handled by the defined local short-circuit conditions;
- total: 48 of the authorized maximum 54, with zero retry or repair.

All 90 predictions completed. Schema and transport failure counts were zero.
The canonical schema has no separate guard-failure counter; zero guard failures
is therefore a derived completion invariant, not a stored counter. Every
substantive verdict had a nonempty evidence span that was a literal substring
of its source fixture.

### Dev quality metrics

| Baseline | Obs/fixed-six macro-F1 | Contradiction recall | Entailed precision | Partial F1 | High-conf error | False-block | Span token F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `lexical_rules` | 0.428571 / 0.428571 | 0.666667 | 0.500000 | 0.200000 | 0.400000 | 0.000000 | 0.959064 |
| `lexical_full_source` | 0.439815 / 0.439815 | 0.666667 | 0.600000 | 0.222222 | 0.400000 | 0.066667 | 0.585013 |
| `miner_judge_deterministic` | 0.650000 / 0.650000 | 0.666667 | 1.000000 | 0.500000 | 0.000000 | 0.000000 | 0.959064 |
| `single_agent_live` | 0.720635 / 0.720635 | 1.000000 | 1.000000 | 0.800000 | 0.222222 | 0.066667 | 0.930556 |
| `miner_judge_live` | 0.720635 / 0.720635 | 1.000000 | 1.000000 | 0.800000 | 0.230769 | 0.066667 | 0.959064 |

The false-block denominator is the 15 non-contradicted dev cases; both live
baselines produced one false block. Observed and fixed-six macro-F1 agree
because all six labels have support.

### Live telemetry

| Baseline | Calls | Input/output/total tokens | p50 latency | p95 latency | Schema/transport failures |
|---|---:|---:|---:|---:|---:|
| `single_agent_live` | 18 | 9,642 / 4,966 / 14,608 | 2,873.474 ms | 9,381.546 ms | 0 / 0 |
| `miner_judge_live` | 30 | 24,004 / 20,279 / 44,283 | 5,800.227 ms | 13,956.772 ms | 0 / 0 |
| Combined live | 48 | 33,646 / 25,245 / 58,891 | — | — | 0 / 0 |

`miner_judge_live - single_agent_live` was zero for observed/fixed macro-F1,
contradiction recall, entailed precision, partial F1, and false-block rate. It
was +0.008547 high-confidence error, +0.028509 source-span token F1, +12 calls,
+14,362 input tokens, +15,313 output tokens, +29,675 total tokens, +2,926.753
ms p50 latency, and +4,575.226 ms p95 latency. Thus Miner/Judge did not improve
relation metrics on this dev set; it improved evidence-span overlap while using
substantially more model resources.

Canonical `cost_usd` remains null because no explicit price snapshot or cache
hit/miss accounting was supplied. No estimate is represented as an actual
provider bill.

### Artifacts and interpretation

The new directory
`eval_runs/phase3d_v2_dev_live_contract_v2/` contains exactly the four
canonical files and no failure artifact:

- `eval_report.md`: SHA-256
  `3a197c2b830f041c44c45a821d16e8ef5af51b522df5849579132e19fde0e0fc`;
- `eval_results.jsonl`: SHA-256
  `0f04cac56d5b3a728604127b5ed90a219d3b1600082ef6339f894f20443d556d`;
- `metrics.json`: SHA-256
  `17fb9af2f243effef35b5fbd807983de01558594d4771ca236d337c295b29a03`;
- `run_manifest.json`: SHA-256
  `6cdf9054a55fda1c8d571ca4f39ad6db9c62936570c9c6d9514f172c25982628`.

The bundle SHA-256 is
`5ee8691195ebd97927f48de2e1cebd493ecbfb2314e99eb241e59e339a208411`,
computed over filename-sorted `filename`, two spaces, file SHA-256, newline
records. Pydantic validation passed for the metrics, run manifest, and all 90
JSONL results. Exact runtime-key and generic credential scans over all four
artifacts found zero matches.

Both live baselines meet the provisional dev numerical threshold of 0.70.
`miner_judge_live` does not meet the unchanged v0.1 macro-F1 target of 0.75,
although contradiction recall exceeds 0.80 and substantive evidence grounding
is complete. Dev is a tuning surface with `benchmark_validity=provisional` and
`public_benchmark_eligible=false`; this run cannot pass the formal Phase 3 gate.
The frozen v2 holdout was not loaded or run, its authorization is absent, and
Phase 4 remains blocked. No code, prompt, dev label, gold, frozen holdout, or
historical artifact changed during the evaluation.

Final offline validation after recording provenance passed:
`python3 -m pytest -q` reported 373 passed in 1.83 seconds;
`python3 -m compileall -q src tests` and `git diff --check` completed
successfully. An exact scan for the runtime `OPENAI_API_KEY` across tracked and
unignored repository files found zero matches, and generic credential-pattern
scanning across changed/untracked files found zero matches. `ruff` and `mypy`
were not installed, so neither was run or reported as passing. No dependency
was installed and sudo was not used. Production/test Python LOC remain 8,503 /
6,373 because this checkpoint changes only evaluation artifacts, manifest, and
documentation.

## Phase 3D — one-time frozen v2 holdout consumed by fail-closed guard

Status: **formal holdout run failed integrity; no rerun; Phase 3 internal gate
not passed; Phase 4 blocked; a future gate requires v3**

On 2026-07-20 the project owner authorized exactly one run of the frozen
60-case `holdout_single_human.jsonl`, with a pre-request ceiling of 180 provider
calls, zero retry/repair, no automatic rerun, and no permission to change code,
prompt, labels, gold, or execution behavior. Preflight passed before any model
request:

- HEAD `34f4077c6a38a3dde98560530a2a3ab35fa3c483`, clean worktree, no remote;
- code hash
  `79fc7fa4ec8d886556baa8406c0d61253abbcc5010b1ae0a8cd27da26a9ca308`;
- prompt hash
  `43e4f32204726f608e0eed44d5c0f3e4779e01984f15d0ecdb14ad27add6326e`;
- dataset file SHA-256
  `0d19004b627f4ecabf9dd40fd28303b0539b89f36f3f7f5cec135c47a6582c75`;
- 60 unique test cases, all `single_human_review`, all six relations with
  support, and all 29 gold substantive spans literal source substrings;
- `deepseek-v4-flash`, provider-default thinking, API key present only in the
  environment, and a nonexistent output directory;
- `pytest` 373 passed in 2.04 seconds, compileall and `git diff --check`
  passed;
- exact runtime-key scan found zero matches. The broad generic scan found only
  three unchanged, pre-existing synthetic credential fixtures under tests;
  after explicit allowlisting, non-test/production generic matches were zero.

The single authorized process started at `2026-07-20T03:38:45.129822Z` and
failed closed at `2026-07-20T03:42:01.783780Z`. `single_agent_live` completed
all 60 cases. `miner_judge_live` completed one case and, on
`v2_holdout_002`, its second Judge output triggered the deterministic-conflict
guard: an `entailed` verdict conflicted with an explicit deterministic mismatch.
No model content, claim, source, evidence, prompt, header, credential, or
exception input was persisted.

Actual provider calls were 64 of 180:

- Single Agent: 60;
- Miner: 2, derived from the two attempted Miner/Judge cases;
- Judge: 2, derived from the same four Miner/Judge calls;
- deterministic baselines: zero model calls;
- automatic retry and repair: zero.

The safe telemetry records 35,480 input, 18,335 output, and 53,815 total
tokens; p50/p95 model latency was 2,589.814 / 6,793.315 ms. All 64 calls
reported usage. Schema and transport failures were zero; the terminating local
guard failure count was one. Canonical `cost_usd` is null because no explicit
price snapshot or cache hit/miss accounting was supplied.

The output directory `eval_runs/phase3d_v2_holdout/` contains only
`failed_attempt.json`, SHA-256
`e7ea20d1b7d18e4a1472c14f5053440e342c8b44ec89d49998fc1fce7337fa60`.
It validates as `EvalFailureArtifact`, has
`canonical_artifacts_written=false` and `contains_sensitive_payloads=false`,
and passed call/usage arithmetic, runtime-key, generic-secret, and raw
claim/source/evidence absence checks. The four success artifacts were not
written and were not reconstructed.

Consequently no holdout macro-F1, contradiction recall, entailed precision,
partial F1, high-confidence error, false-block rate, span F1, confusion matrix,
or Single Agent versus Miner/Judge quality delta is available. Reporting any
of those values would require an unauthorized rerun or fabrication. The formal
gate fails the required 60/60 Miner/Judge completion and zero-guard-failure
conditions; the 0.70 internal threshold and the separate 0.75/0.80 v0.1 targets
are not evaluable.

This one-time v2 holdout is now consumed and must not be rerun. Phase 3 internal
gate remains unpassed and Phase 4 remains blocked. Any remediation must be
developed only on dev; a later formal gate requires a newly frozen v3 holdout.
No tuning is performed from the exposed failure. The dataset remains
`single_human_synthetic_holdout`, `public_benchmark_eligible=false`, and has no
second-human/adjudicated status.

The pair-level benchmark directly supplies atomic claims, so even a successful
run could only test relation generalization, not claim extraction or whether
Miner adds value. Demonstrating multi-Agent value still requires a separate
full-document claim-extraction benchmark; Scout/Challenger ablations remain
future work and were not implemented here.

Post-attempt offline validation passed: `python3 -m pytest -q` reported 373
passed in 1.69 seconds; `python3 -m compileall -q src tests` and
`git diff --check` completed successfully. The exact runtime API-key scan found
zero matches, and changed/untracked plus all non-test paths had zero generic
credential matches. Three unchanged synthetic credential assignments in
pre-existing test fixtures remain explicitly allowlisted. `ruff` and `mypy`
were not installed, so neither was run or reported as passing. No dependency
was installed and sudo was not used. Production/test Python LOC remain 8,503 /
6,373: no source, test, prompt, label, gold, frozen holdout, or historical
artifact was modified. Only the new safe failure artifact, v2 provenance
manifest, and documentation are included in this checkpoint.

## Phase 3E — context-aware deterministic entity guard

Status: **offline remediation complete; v2 remains diagnostic and consumed;
full dev validation is the next separately authorized step; Phase 4 blocked**

The v2 holdout failure artifact remains byte-for-byte unchanged at SHA-256
`e7ea20d1b7d18e4a1472c14f5053440e342c8b44ec89d49998fc1fce7337fa60`.
The frozen holdout also remains unchanged at SHA-256
`0d19004b627f4ecabf9dd40fd28303b0539b89f36f3f7f5cec135c47a6582c75`.
No partial predictions or metrics were recovered, inferred, or recreated.
The v2 status is now `diagnostic_consumed_guard_failure`; it was not rerun.

The user-confirmed root cause was reproduced with new offline fixtures. Before
this checkpoint, `entity_mismatch` compared the Claim only with the most
aligned clause. A sentence-initial account/scope word could therefore look
like a different entity while the actual Claim entity in an adjacent bounded
sentence was ignored. The new `deterministic-signals-v2` policy emits the
error-level signal only when all four conditions hold:

- the Claim contains an entity;
- the aligned clause contains a different entity;
- the Claim entity is absent from the complete bounded evidence context; and
- Claim and aligned-clause predicate/context are sufficiently aligned.

The bounded context is the deduplicated chunk text of only the top-k evidence
candidates already available to the Judge. Alignment and the numeric, date,
semantic-version, negation, qualifier, and conjunction checks continue using
the same best candidate/clause as before. Retrieval settings, source/span
grounding, citation scope, Claim scope, and model relations were not weakened
or rewritten. True cross-product conflicts still block an `entailed` verdict.

The generic guard rejection is now
`DeterministicConflictError`. It exposes only a sorted, unique tuple of
allowlisted signal codes and uses a fixed payload-free message. Future typed
guard failures can use the backward-compatible
`phase3e-live-failure-v3` artifact with `guard_signal_codes`; v1/v2
artifacts cannot contain that field. Unknown `ValueError` and
`RuntimeError` paths retain the existing fail-closed
`local_validation` behavior. The historical v2 artifact was neither
rewritten nor supplemented.

The cache-key contract now includes
`deterministic_signal_policy_version=deterministic-signals-v2` in addition to
source, Claim, model, prompt, and retrieval dimensions, so legacy five-field
keys and v1-policy keys do not resolve under the new key. This repository still
has no production cache call site; this is a versioned storage/key contract,
not a claim that Judge caching has been wired. Future eval run manifests record
the policy independently. The structured-output contract remains
`openai-compatible-v2`; its output schemas and examples did not change.

Phase 3E provenance:

- code bundle SHA-256:
  `41c5bf4a9f5bdd9159dad26b37eeb3688efa09db3604dfb5bb5b3057d96083cf`;
- config SHA-256:
  `071f58d6f7345a63b583686908bebd7682cd649c40dc7ad18a01a993dd7a9d4e`;
- prompt bundle SHA-256:
  `6ccb4e34abc29819ef7fdd5d6b1648acbe8d86e96aab46168d4629dfdbd75e43`.

The prompt bundle hash changes because Judge/baseline orchestration now derives
entity signals from bounded context. It does not indicate a structured-output
schema or semantic model-prompt tuning change.

Offline verification on 2026-07-21: the targeted guard/cache/Judge suite
reported 71 passed; the final full suite reported 387 passed. Compileall,
`git diff --check`, and the secret scan passed. `ruff` and `mypy` were
not installed and were not run; no package was installed and sudo was not
used. Physical Python LOC are 8,650 production and 6,771 tests, a Phase 3E net
change of +147 production and +398 test lines from the 8,503/6,373 baseline.
The production increment stays within the requested approximate +150 target.

Real model API calls in this checkpoint: **0**. No live dev run occurred, v2
was not rerun, v3 was not created, and no Phase 4 feature was implemented.
Phase 3 remains unpassed and Phase 4 remains blocked. The next step is a
separately authorized complete dev rerun; it is not part of this checkpoint.

## Phase 3E — authorized full-dev validation attempt

Status: **attempt consumed; failed closed on an unrelated Miner scope guard;
no rerun; entity fix only partially live-validated; Phase 4 blocked**

The owner authorized one complete 18-case dev run with
`deepseek-v4-flash`, provider-default thinking, at most 54 requests, and
zero retry/repair. Preflight at commit
`8eaa17b7e0d0ec3b907e022743f8d3dcecef1e45` confirmed a clean worktree,
no remote, an absent output directory, 18 dev cases, the frozen code/config/
prompt hashes, `deterministic-signals-v2`, and
`openai-compatible-v2`. Pytest reported 387 passed before the call;
compileall, diff check, and the count-only secret scan also passed.

The single process ran from `2026-07-21T04:32:44.773290Z` to
`2026-07-21T04:36:02.756522Z` and stopped after 44 of 54 allowed requests.
All 18 Single Agent cases completed. Thirteen Miner/Judge cases completed;
the fourteenth Miner response on `v2_dev_014` failed the existing
protected-slot scope guard before a Judge call. The stage split is therefore
18 Single Agent, 14 Miner, and 12 Judge calls. No request was retried or
repaired.

The safe artifact intentionally does not retain the Miner output or the
specific introduced slot. That detail is unrecoverable and is not guessed.
This was a local `ValueError`, not
`DeterministicConflictError`: schema failures, transport failures, and
deterministic entity-guard failures were all zero before termination.
Consequently the context-aware entity fix caused no observed hard conflict
through the 13 completed Miner/Judge cases, but the interrupted run does not
prove guard stability across all 18 cases. Generic neighboring-context and
true cross-entity behavior remain covered by the offline regressions.

Only
`eval_runs/phase3e_v2_dev_live_context_guard/failed_attempt.json` exists,
SHA-256
`a17b64973c70da1e6b01fd9c1e673885c41f58818759232b52c27b270c204596`.
It validates as `EvalFailureArtifact`, records no guard codes, and contains
no model content, reasoning, Claim/source/evidence, prompt payload, header, or
credential. The four canonical success artifacts were removed by the
fail-closed runner and were not reconstructed.

| Baseline | Completion | Quality/confusion | Calls | Input / output / total tokens | Model p50 / p95 ms |
|---|---|---|---:|---:|---:|
| `lexical_rules` | executed before failure | unavailable after fail-closed cleanup | 0 | 0 / 0 / 0 | n/a |
| `lexical_full_source` | executed before failure | unavailable after fail-closed cleanup | 0 | 0 / 0 / 0 | n/a |
| `miner_judge_deterministic` | executed before failure | unavailable after fail-closed cleanup | 0 | 0 / 0 / 0 | n/a |
| `single_agent_live` | 18/18 | unavailable after fail-closed cleanup | 18 | 9,642 / 4,031 / 13,673 | 2,250.190 / 3,201.200 |
| `miner_judge_live` | 13/18; Miner failed on case 14 | unavailable after fail-closed cleanup | 26 | 21,565 / 18,299 / 39,864 | 5,785.249 / 8,331.629 |

Aggregate usage was 31,207 input, 22,330 output, and 53,537 total tokens;
aggregate p50/p95 latency was 4,422.922 / 8,265.639 ms. All 44 calls reported
usage and provider-call failure kind `none`; the failure occurred in local
post-response Miner validation. Canonical `cost_usd` is null because no
explicit pricing snapshot or cache hit/miss accounting was supplied.

Observed/fixed-six macro-F1, contradiction recall, entailed precision,
partial-support F1, high-confidence error, false-block rate,
evidence-span F1, and confusion matrices are unavailable for every baseline.
The 0.70 Miner/Judge dev threshold is therefore not evaluable. Recovering
partial predictions or reporting partial metrics would violate the
fail-closed artifact contract.

The prior successful Phase 3D run remains the only complete comparison:
both live baselines had observed/fixed-six macro-F1 0.720635,
contradiction recall 1.0, entailed precision 1.0, partial F1 0.8, and
false-block rate 0.066667. Its high-confidence error was 0.222222 for Single
Agent and 0.230769 for Miner/Judge; span F1 was 0.930556 and 0.959064.
No Phase 3E quality delta can be computed.

For the fully completed Single Agent baseline, Phase 3E versus Phase 3D used
the same 18 calls and input tokens, 935 fewer output/total tokens, with model
p50/p95 lower by 623.283 / 6,180.346 ms. The partial Miner/Judge execution used
four fewer calls and 4,419 fewer total tokens than the former complete run;
that is not a like-for-like efficiency result. Provider-default thinking
leaves effective temperature unknown, so output/resource variation cannot be
attributed solely to the entity-guard change.

Post-attempt offline validation reported 388 tests passed. Compileall,
`git diff --check`, and secret scans passed; `ruff` and `mypy` remain
unavailable and were not run. Production code remains 8,650 lines and all
code/config/prompt hashes remain unchanged. Tests are 6,852 lines, +81 for the
fixed failure-provenance regression.

The authorization is consumed and the attempt was not rerun. Neither consumed
v2 holdout data nor v3 was run or created. Phase 3 remains unpassed and Phase 4
blocked. Because full dev was not stable, v3 candidate generation remains
deferred; any Miner scope remediation requires a separate offline checkpoint
and fresh authorization.

## Phase 3E.2 — minimal Miner contract and pair-eval baseline correction

Status: **offline implementation complete; no live rerun; Phase 3 unpassed;
Phase 4 blocked**

The live Miner no longer returns the public, fully enriched `MinerOutput`
schema. Its model-owned `live-miner-draft-v2` contract requires one top-level
`claims` array. Every entry contains only `text`, `claim_type`, and
`checkability`; missing `claims` and all additional fields are rejected.
Claim text must be an exact contiguous paragraph substring. The bounded matcher
computes the earliest forward and latest reverse monotonic non-overlapping
mappings; they must agree, otherwise it fails closed as absent or ambiguous.
It uses no recursive or combinatorial search. No Markdown fence, alias,
paraphrase, normalization, retry, or model repair is accepted.

After draft validation, deterministic local code constructs the public
`AtomicClaim` values. It assigns stable ordered claim IDs, copies the project
file and citations only from `MinerInput`, derives line ranges from exact source
offsets, and fills model/prompt provenance locally. An audit found no
downstream Judge, retrieval, pipeline, cache, or artifact consumer of
`AtomicClaim.slots`. Its sole current production read is the Miner's local
post-enrichment scope assertion, which requires the locally generated mapping
to remain empty. The public `AtomicClaim`, `MinerOutput`, and historical
serialized forms remain backward compatible.

Paragraph and factual-scope checks remain strict. All emitted text must come
from the paragraph, protected numeric/date/version/percentage/negation/
qualifier tokens must be covered by at least one claim, incomplete fragments
are rejected, and a model cannot supply file paths, line numbers, citations,
IDs, slots, or provenance. `MinerScopeError` carries only one of four
allowlisted codes: `non_source_span`, `ambiguous_span`,
`missing_protected_token`, or `invalid_claim_fragment`. It contains none of the
claim, paragraph, model response, source, or Pydantic input. There is no retry
or repair path.

Pair-level evaluation now reflects the supplied input. Because every
`EvalCase` already contains one atomic `claim_text`, active new runs use
`retrieval_judge_deterministic` and `retrieval_judge_live`; neither calls
Miner. The claim is built locally, existing lexical retrieval runs, and the
live path makes at most one Judge request for a case that requires semantic
judgement. Together with the one-call `single_agent_live` baseline, a pair run
has a maximum live-call ceiling of `2 * case_count`. Deterministic baselines
remain zero-call, and pair-level claim-extraction metrics remain
`null / not_applicable`.

The names `miner_judge_deterministic` and `miner_judge_live` remain accepted
only so frozen Phase 3A–3E artifacts can be read with their original semantics.
They are not active pair-level runner baselines. Future Miner/Judge evaluation
requires a separate full-document benchmark with gold claim spans and counts,
extraction precision/recall/F1, atomicity and line-mapping measures, and
end-to-end relation metrics. That benchmark is designed in
`docs/full-document-benchmark-design.md` but is not implemented here.

Version boundaries are explicit: Miner draft/cache contract
`live-miner-draft-v2`, deterministic policy `deterministic-signals-v2`, and
OpenAI-compatible transport/structured-output contract
`openai-compatible-v2`. The Miner contract is an independent cache-key
dimension, so legacy full-output Miner entries do not resolve under the new
contract. The prompt/schema bundle now includes both schema-definition modules,
not only prompt call sites. The frozen code, prompt/schema, and config hashes
are `309c570e4eb835384946bc1db210c2f8250e2b6a9b6630440839ef89425be64b`,
`01dde7fb745f2073d10f2ac1e0c3fe1497c51f51f190c4752a329ab7afb77c0a`,
and `d20df2b9f96bb7254d167df114554257d01339d149cb1c08c87cb31089bc15a9`.
The repository still has no production cache call site.

This checkpoint made **zero** real model/API calls. It did not rerun dev or use
consumed v2 content for development, create v3, or implement any Phase 4 feature.
The historical Phase 3E failure artifact remains immutable at SHA-256
`a17b64973c70da1e6b01fd9c1e673885c41f58818759232b52c27b270c204596`.
Phase 3 remains unpassed and Phase 4 remains blocked.

Final offline verification on 2026-07-21: `python3 -m pytest -q` passed all
402 tests; `python3 -m compileall -q src tests`, manifest JSON validation,
`git diff --check`, and credential-pattern scans passed. `ruff` and `mypy`
were not installed, so they were not run or reported as passing. The current
tree contains approximately 8,858 production Python lines and 7,229 test
Python lines. No evaluation command, network request, retry, or model repair
was executed in this checkpoint.

## Phase 3E.2 — pair-dev live stability attempt

Status: **stopped on an operational local validation `ValueError` after one
complete dev run; no three-run aggregate; Phase 3 unpassed; Phase 4 blocked**

The authorized attempt used the unchanged commit
`ac6f3ab21d8ce78dbfc2514bfabdeca9121196ae`, code hash
`309c570e4eb835384946bc1db210c2f8250e2b6a9b6630440839ef89425be64b`,
prompt/schema hash
`01dde7fb745f2073d10f2ac1e0c3fe1497c51f51f190c4752a329ab7afb77c0a`,
and config hash
`d20df2b9f96bb7254d167df114554257d01339d149cb1c08c87cb31089bc15a9`.
The model was `deepseek-v4-flash` with provider-default thinking, requested
temperature 0.0, and unknown effective temperature. The global limits were
109 provider calls and 1,000,000 total tokens, with no new request after
950,000 tokens. Retry and repair remained zero.

The one-case `v2_dev_003` wiring smoke passed in exactly one Judge request.
Its `AtomicClaim` was constructed locally from `EvalCase`, Miner and Single
Agent calls were zero, the predicted `entailed` relation matched gold, and
source, evidence-substring, locator, and `deterministic-signals-v2` scope
checks passed. It used 1,125 input, 442 output, and 1,567 total tokens, with
4,144.897 ms latency. The safe artifact is
`eval_runs/phase3e2_retrieval_judge_smoke/smoke_attempt.json`, SHA-256
`5ebfb2154dcb235b964d741d0e1bc66d93e55f9a0d8958cc417d13d3fa10e625`.

The first complete independent dev run succeeded with 31 model calls: 18
`single_agent_live`, 13 `retrieval_judge_live`, and zero Miner calls. It used
23,813 input, 12,862 output, and 36,675 total tokens. Schema and transport
failures were zero. Pair-level extraction metrics remained not applicable.

| Baseline | Observed/fixed-six macro-F1 | Contradiction recall | Entailed precision | Partial F1 | High-confidence error | False-block | Span F1 | Calls | Tokens | Model p50 / p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `lexical_rules` | 0.428571 | 0.666667 | 0.500000 | 0.200000 | 0.400000 | 0.000000 | 0.959064 | 0 | 0 | n/a |
| `lexical_full_source` | 0.439815 | 0.666667 | 0.600000 | 0.222222 | 0.400000 | 0.066667 | 0.585013 | 0 | 0 | n/a |
| `retrieval_judge_deterministic` | 0.650000 | 0.666667 | 1.000000 | 0.500000 | 0.000000 | 0.000000 | 0.959064 | 0 | 0 | n/a |
| `single_agent_live` | 0.652778 | 1.000000 | 1.000000 | 0.500000 | 0.277778 | 0.133333 | 0.930556 | 18 | 14,293 | 2,191.109 / 3,588.207 |
| `retrieval_judge_live` | 0.887302 | 1.000000 | 1.000000 | 0.666667 | 0.000000 | 0.066667 | 0.959064 | 13 | 22,382 | 5,398.727 / 8,889.010 |

These are provisional dev metrics from run 01, not holdout results or a Phase
3 gate. Its four canonical artifacts are preserved under
`eval_runs/phase3e2_pair_dev_run_01/` and are hashed individually in the v2
manifest.

Run 02 stopped fail closed on `v2_dev_017` after an untyped local validation
`ValueError`. It had made 31 calls (18 Single Agent and 13
Retrieval→Judge), used 23,813 input, 12,404 output, and 36,217 total tokens,
and recorded zero schema and transport failures before termination. Its
all-call model latency p50/p95 was 2,685.601 / 7,038.013 ms. Miner was never
called. The specific failed validation condition is not recoverable from the safe
artifact and is not guessed. No canonical run-02 report, results,
metrics, or run manifest exists; only
`eval_runs/phase3e2_pair_dev_run_02/failed_attempt.json` is retained, SHA-256
`c4fbd497ac266bef4014fdf7f6d92e304e2fe5801017d9fbc3900399d0d3a82c`.

Per the precommitted stop rule, run 02 was not retried, run 03 was not
started, and no three-run stability summary was generated. Reporting a mean,
minimum, maximum, standard deviation, per-case self-consistency, or a
best-of-three result would imply nonexistent complete runs, so those values
remain unavailable. This is an operational stability failure, not evidence
that run 01's provisional quality metrics repeated reliably.

The whole attempt consumed 63 provider calls, comprising one smoke Judge,
36 Single Agent, 26 Retrieval→Judge, and zero Miner calls. It used 48,751
input, 25,708 output, and 74,459 total tokens; combined call latency p50/p95
was 2,811.118 / 7,038.013 ms. No call was retried or repaired. Operational
success was 1/2 attempted dev runs and 1/3 planned dev runs.
The six-artifact bundle SHA-256 is
`f3abc490f9ca749af5d32b59cfd83a23b50225ad982d0685e48fdba0523c55e0`.
Its deterministic algorithm sorts relative paths lexicographically, emits one
compact UTF-8 JSON object with exactly `path`, `sha256`, and `bytes` keys for
each artifact using separators `,` and `:` and no trailing newline, then hashes
the resulting byte stream.

Only `eval_sets/v2/dev.jsonl` was selected. The dev loader read shared source
fixtures required by those dev cases. The consumed v2 holdout was not loaded,
parsed, evaluated, or run; post-validation checked its byte-level SHA only to
confirm immutability. No holdout was run, v3 was not created, and no Phase 4
feature was implemented. Frozen code, prompt/schema, and config hashes
remained unchanged. Phase 3 remains unpassed and Phase 4 remains blocked.

Final offline quality checks for this provenance update: `python3 -m pytest -q`
passed all 402 tests in 1.80 seconds, and
`python3 -m compileall -q src tests` passed. Ruff 0.15.21 was installed and
actually run: `python3 -m ruff check src tests` failed with 157 findings, 34
reported fixable; `python3 -m ruff format --check src tests` failed because 49
files would be reformatted and 7 were already formatted. These pre-existing
quality findings were not auto-fixed because doing so would change the frozen
code hash. Manifest and artifact schema validation, `git diff --check`, and
the exact-secret, Bearer-token, `sk`-like token, and forbidden JSON-key scans
passed. `mypy` was not installed and was not run.

## Phase 3E.3 — typed Judge validation and three-run dev stability

Status: **repair validated; 3/3 complete dev runs; Phase 3 unpassed; Phase 4
blocked**

Run 02's safe Phase 3E.2 failure artifact does not preserve raw model output,
so the exact unsaved relation is not reconstructed. The reproducible defect was
that Judge's local validator treated a schema-valid, in-scope
`not_checkable` relation as a fatal generic `ValueError` whenever local
checkability heuristics disagreed. That mixed an ordinary model-quality error
with integrity failures and could abort the complete evaluation on
`v2_dev_017`.

`judge-output-validation-v2` removes that semantic fatal branch. Legal
relations now become canonical predictions and are scored. Claim-ID mismatch,
source scope, evidence-span scope, source availability, and substantive
evidence violations remain fail closed through payload-free `JudgeScopeError`
codes; factual deterministic conflicts remain fail closed through
`DeterministicConflictError`. Neither exception carries claim, source, prompt,
model content, headers, credentials, or an arbitrary message. Strict JSON,
zero automatic retry, and zero repair are unchanged. Parameterized offline
coverage includes subjective/best claims, subjective claims with objective
metrics, legal partial entailment, forged evidence spans, and factual
contradictions.

The live validation used frozen commit
`ad72cecfabf8d6969a51d739ef27b820bda4fbbc`, code hash
`b05f9c35c72d06a3a45c6f08844db677fcceeaf868202e675b8cea0279b50c90`,
prompt/schema hash
`8dc3d1f611c0673aad90c83678629eb947a5c5281ac19e8c16d25a72f885b5dd`,
and config hash
`af7bce4126a4b74e63d3db4a93d158efe4c37f3cbdb6c99c7efba0f327e7f1e4`.
The model was `deepseek-v4-flash`, requested temperature 0.0, with
provider-default thinking and unknown effective temperature.

The five independent `v2_dev_017` smoke calls all produced valid canonical
predictions with zero schema, transport, guard, or local-validation failures.
The relations were `not_in_source`, `partially_entailed`, `not_checkable`,
`not_in_source`, and `not_in_source`; this is model variability, not an
operational failure. The smoke used 8,725 tokens. Its secret-free artifact is
`eval_runs/phase3e3_v2_dev_017_retrieval_judge_smoke/smoke_attempt.json`,
SHA-256 `2b1de7c62d67bb2c14c3ddb23062b18c4791a24f05c615b9723f899b7ea81c88`.

All three complete dev runs then succeeded on that same frozen boundary:

| Run | Calls | Tokens | Single Agent macro-F1 | Retrieval-to-Judge macro-F1 | Delta |
|---|---:|---:|---:|---:|---:|
| 01 | 31 | 37,684 | 0.652778 | 0.942857 | +0.290079 |
| 02 | 31 | 34,847 | 0.652778 | 0.942857 | +0.290079 |
| 03 | 31 | 36,627 | 0.652778 | 0.942857 | +0.290079 |

The deterministic macro-F1 values were also identical in all runs:
`lexical_rules=0.428571`, `lexical_full_source=0.439815`, and
`retrieval_judge_deterministic=0.650000`. Population standard deviation was
zero for every baseline macro-F1. The aggregate contains every headline
baseline metric and each per-run Retrieval-to-Judge minus Single Agent delta.
Across the three runs, Single Agent relation agreement was 18/18 and
Retrieval-to-Judge was 17/18. The only unstable case was `v2_dev_017`, whose
gold is `not_checkable` and whose three Retrieval-to-Judge predictions were
`not_in_source`, `not_in_source`, and `partially_entailed`.

The three full runs consumed 93 calls and 109,158 tokens, with combined model
latency p50/p95 of 2,826.125/8,946.451 ms. Including smoke, the authorized work
used 98 calls and 117,883 tokens, below the 150-call and 1,000,000-token hard
limits. Schema, transport, guard, and local-validation failure totals were all
zero. No request was retried or repaired.

The aggregate JSON is
`eval_runs/phase3e3_pair_dev_stability/aggregate.json`, SHA-256
`a30c43342a12ea0ed01e23afd54632d2268db9d281fe8ee5905c443c7038c78b`;
its Markdown rendering has SHA-256
`4557b8e62cff618f13ec4cfc5322d51457e2e0e2ece086ad54dd22895fc54689`.
The 15-artifact Phase 3E.3 bundle SHA-256 is
`6db95e1b5ee10d76fa3f60254cab1748ee003a901c123e2b1d4bc69568f0052e`.

Only `eval_sets/v2/dev.jsonl` was selected by the loader. The consumed v2
holdout was neither loaded nor parsed; only its byte-level SHA-256
`0d19004b627f4ecabf9dd40fd28303b0539b89f36f3f7f5cec135c47a6582c75`
was checked. No label or prompt was changed, no v3 holdout was created, and no
Phase 4 code was implemented. Phase 3E.3 now supplies the dev operational and
stability prerequisite for separately creating and freezing a new v3 holdout,
but Phase 3 itself remains unpassed until that uncontaminated gate is defined
and run.

Final validation passed 413 tests in 0.95 seconds, compileall, manifest and
artifact regressions, consumed-v2 SHA-only immutability, sensitive-key scans,
and `git diff --check`. Ruff lint passed on only the eight Phase 3E.3 Python
files. Ruff format check reported those eight files would be reformatted; no
formatting was applied after the live freeze because that would invalidate the
recorded code and prompt/schema hashes.

## Phase 3F-C — one-time v3_zh formal internal gate

Status: **operationally complete metric failure; v3_zh consumed; Phase 3
unpassed; Phase 4 not eligible**

The one authorized execution ran all 72 valid Chinese cases exactly once from
commit `f3c07304195807cba751843e498369cd390a929c`. Preflight verified a clean
worktree; all six relation labels had support; the holdout was eligible and
unexposed to the evaluation model; and the gold, freeze bundle, code,
prompt/schema, and config hashes matched their frozen values. The selected
419-test suite passed before execution. It excluded the six tests in
`tests/test_eval_v2_single_human.py` because those tests load consumed v2 gold,
so this was not an unconditional full-suite claim. Compileall, diff, and secret
checks also passed.

All five current pair baselines completed 72 cases each. Claim Miner calls were
zero. The live paths made 111 provider calls: 72 Single Agent and 39
Retrieval-to-Judge Judge calls. Reported usage was 83,171 input, 50,662 output,
and 133,833 total tokens. Combined model latency p50/p95 was
3,591.527/9,098.556 ms. Automatic retry and repair counts were zero. Schema,
transport, scope, guard, local-validation, and call-budget failures were all
zero. No source was fetched from the network. Runtime evaluation cache wiring
is absent, so cache hit/miss accounting is `not_available`; schema-default
`cache_hit` fields are not treated as observed cache activity.

| Baseline | Fixed-six macro-F1 | Weighted-F1 | Balanced acc. | Contradiction recall | Entailed precision | Partial F1 | High-conf. error | False-block | Span F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `lexical_rules` | 0.362555 | 0.380388 | 0.390082 | 0.230769 | 0.200000 | 0.173913 | 0.600000 | 0.033898 | 0.569253 |
| `lexical_full_source` | 0.362555 | 0.380388 | 0.390082 | 0.230769 | 0.200000 | 0.173913 | 0.600000 | 0.033898 | 0.569253 |
| `retrieval_judge_deterministic` | 0.308986 | 0.322850 | 0.362037 | 0.000000 | 0.200000 | 0.173913 | 1.000000 | 0.016949 | 0.488896 |
| `single_agent_live` | 0.727360 | 0.716505 | 0.783565 | 1.000000 | 0.833333 | 0.842105 | 0.214286 | 0.016949 | 0.890625 |
| `retrieval_judge_live` | 0.613757 | 0.610780 | 0.637001 | 0.769231 | 0.666667 | 0.777778 | 0.300000 | 0.016949 | 0.765411 |

Observed-label and fixed-six macro-F1 are identical because every relation has
support. Retrieval-to-Judge minus Single Agent was `-0.113603` macro-F1,
`-0.105725` weighted-F1, `-0.146563` balanced accuracy, `-0.230769`
contradiction recall, `-0.166667` entailed precision, `-0.064327` partial F1,
`+0.085714` high-confidence error, `0` false-block rate, and `-0.125214`
evidence-span F1. It used 33 fewer calls but 7,483 more total tokens; model
latency p50/p95 was 2,843.527/6,472.636 ms higher.

The headline confusion matrix, label support, all baseline metrics, and exact
resource accounting are preserved in `metrics.json` and
`execution_record.json`. Retrieval-to-Judge completed all 72 cases and retained
valid substantive source substrings, but fixed-six macro-F1 `0.613757` was
below both `0.70` and `0.75`; contradiction recall `0.769231` was below
`0.80`. The formal internal gate therefore failed on quality metrics. Phase 3
did not pass, Phase 4 is not eligible, and v3_zh is consumed with no rerun
permitted.

The canonical artifact bundle SHA-256 is
`f7601e1076d3ec5fc175c165a0d09c80dbd9f147ae256f46d15164904dcab851`.
The append-only execution record is
`eval_runs/phase3f_v3_zh_internal_gate/execution_record.json`, SHA-256
`74540003d8200410038028dc4a898aee422d00f1683a03ddfd465c8f01febf4a`.
The frozen v3_zh manifest was not rewritten; its pre/post SHA-256 remains
`eb540f6e8e61483fd3e725b2d77817c6bebeacd4babe0d81a8b851a3939f19d5`.
This remains a claim-source pair benchmark and cannot establish end-to-end
Claim Miner plus Judge Multi-Agent advantage.

Post-run validation passed the same 419-test selected suite in 1.21 seconds,
again excluding the six consumed-v2-gold-loading tests rather than claiming an
unconditional full suite. Compileall, `git diff --check`, exact frozen-hash
verification, exact runtime-key/base-URL scans, and generated-artifact
credential/header/raw-reasoning pattern scans all passed.

## Phase 3G-B — adaptive Chinese dev implementation

Status: **offline implementation complete; live dev not run**

Chinese retrieval now preserves the legacy Latin scoring path and adds
Unicode-NFKC CJK bigram/trigram fallback when Latin terms are absent or fewer
than two. Numeric, date, version, negation, and identifier text is never
rewritten in source or artifact output. Multi-chunk sources use deterministic
lexical scores; a sole available non-empty chunk is retained even under zero
lexical overlap.

`adaptive_live` is a new baseline and does not rename or alter historical
baselines. It routes unavailable and high-confidence subjective cases
deterministically, uses full-context Single Agent for a single chunk whose
exact serialized prompt fits a configurable byte budget, and otherwise uses
Retrieval-to-Judge. Every case is capped at one call. Route telemetry uses
allowlisted route/reason enums plus chunk count and estimated context size.

Chinese checkability recognizes recommendation, value-judgment, and
superlative forms only when objective slots are absent. Explicit numbers,
dates, versions, benchmark/ranking attribution, platform compatibility, and
capacity limits prevent the deterministic `not_checkable` short circuit.
Both live pair prompts now receive concise bilingual relation definitions.

The new `eval_sets/phase3g_dev_zh/` pack contains 72 provisional dev pairs and
24 new sources: 7 short, 7 medium, 6 long, and 4 unavailable. Every relation
has 12 cases. Case strata are 21 short, 21 medium, 18 long, and 12 unavailable.
Long fixtures have seven operational chunks with real distractors. The pack
preregisters later reporting for all three live baselines overall and by
short/medium/long stratum; pair extraction metrics remain not applicable.

Leakage audit covered IDs, hashes, URLs, entities, exact and normalized text,
CJK-aware lexical features, character 5-grams, and content numbers against
v1, v2 dev/candidates, v3, and consumed v3_zh diagnostic content. It found no
collision or threshold flag. The consumed v2 gold file was not parsed. Model
calls were zero, v3_zh was not rerun, v4 was not created, and Phase 4 remains
ineligible.

Final validation passed 448 tests in the documented selected suite, which
excludes the six tests that load consumed v2 gold and is therefore not an
unconditional full-suite claim. Compileall, changed-file Ruff lint and format,
`git diff --check`, secret scan, and consumed-artifact hash verification also
passed. Frozen hashes are code
`5d5e17ca0fb1fa1d9257482006aca602440d48643273a5fd1a6b6962e3addaf8`,
prompt/schema
`a6c81a19fe2307d55144c39317586e1efe398ca13f500ec615c5b3aa49101975`,
config/router/retrieval policy
`070e9485f37a6e85fb0b2821081b9fefbcc2d1e3267252b2f23774c3a531e4d0`,
dev dataset
`37a76001e027b82b34de07eacf2d8160a65583d1560bef393dc37d9155bb6761`,
and leakage audit
`60d96bb35b6af6b41d463024d64a75d14bbbc2754992c168872682c3e1d9736e`.

## Phase 3G-C — adaptive Chinese live dev stability attempt

Status: **2/3 complete runs; Run 03 schema failure; readiness failed**

The run used the Phase 3G-B freeze at commit
`f5d5521803ffb9b058afafdd33e852541b86da5a`, model
`deepseek-v4-flash`, provider-default thinking, requested temperature 0.0,
zero retry, zero repair, and no evaluation cache. The pre-registered global
limits were 650 provider calls and 1,000,000 reported tokens.

The four-route smoke succeeded. The full-context and Retrieval-to-Judge cases
made one call each; deterministic `not_checkable` and unavailable cases made
zero calls. Route, schema, scope, source-substring, and telemetry checks all
passed.

Runs 01 and 02 each produced 432 records over all six baselines and completed
with zero schema, transport, scope, guard, or local-validation failures. Each
made 168 calls, including exactly 48 Adaptive calls, with route counts
17 full-context, 31 Retrieval-to-Judge, 12 deterministic not-checkable, and 12
deterministic unavailable.

| Run | Baseline | Macro-F1 | Fixed-six | Contradiction recall | Not-checkable recall | Calls | Total tokens |
|---|---|---:|---:|---:|---:|---:|---:|
| 01 | `single_agent_live` | 0.958115 | 0.958115 | 1.000000 | 0.833333 | 72 | 65,174 |
| 01 | `retrieval_judge_live` | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 48 | 99,429 |
| 01 | `adaptive_live` | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 48 | 83,131 |
| 02 | `single_agent_live` | 0.956703 | 0.956703 | 1.000000 | 0.750000 | 72 | 65,876 |
| 02 | `retrieval_judge_live` | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 48 | 102,457 |
| 02 | `adaptive_live` | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 48 | 82,370 |

Adaptive scored 1.0 observed-label macro-F1 in short, medium, long, and
unavailable strata in both complete runs. Fixed-six macro-F1 was 0.833333 for
each available-source stratum and 0.166667 for unavailable, because those
subsets do not contain all six labels. Its completed-run long-stratum
macro-F1 advantage over Single Agent averaged +0.085079. Adaptive used 20.225%
more input tokens than Single Agent, so the token-reduction value path was not
met.

Run 03 encountered one strict schema failure after 46 provider responses and
40,726 reported tokens. It stopped immediately, was not retried or repaired,
did not continue, and retained only
`eval_runs/phase3g_adaptive_zh_dev_run_03/failed_attempt.json`. No Run 03
predictions or metrics are generated or inferred.

Across Runs 01 and 02, Adaptive and Retrieval-to-Judge relation agreement and
Adaptive route agreement were 1.0. Single Agent agreement was 0.930556, with
five unstable case IDs recorded in the aggregate. These are explicitly
two-completed-run diagnostics, not the requested three-run stability result.

The final call ledger records 384 calls, 383,200 input tokens, 159,444 output
tokens, and 542,644 total tokens, including smoke and the failed round. The
pre-registered readiness decision is fail: 3/3 operational success was not
achieved and operational failures were not all zero. All quality and product
value gates requiring three complete runs are marked not evaluated.

The incomplete aggregate is
`eval_runs/phase3g_adaptive_zh_dev_stability/aggregate.json`. Phase 3 remains
formally unpassed, Phase 4 remains ineligible, v3_zh was not rerun, and v4 was
not created.

## Phase 3G-D - bounded schema recovery ready

Status: **offline implementation complete; fresh live dev not authorized**

The sanitized Run 03 record and frozen execution path identify the failing
baseline as `single_agent_live`, the ordered case as
`phase3g_zh_dev_046`, and the model task as
`single_agent_full_source_verification`. The round recorded 46/46 reserved and
completed provider events with reported usage. The safe mapper's `schema`
code proves a provider wire/output-contract failure, not a downstream local
validator failure.

The exact subtype is `unknown_schema_failure`. Run 03 did not retain a safe
schema diagnostic, finish reason, max-token indication, chat message/content
envelope state, or Pydantic error code. The raw response was not saved and is
not reconstructed or guessed. Formal Run 03 artifacts, status, and missing
metrics remain unchanged.

Production now wraps live structured calls with `schema-recovery-v1`.
`ModelSchemaError` alone can trigger one second request. Attempt 2 uses the
same scoped input and output schema plus a fixed schema-only reminder, contains
none of attempt 1's response, and performs no repair or normalization. A
second schema failure fails closed. Transport, exhausted budget, scope/span,
guard/conflict, semantic disagreement, and local business validation are not
retry triggers.

Safe telemetry includes attempt index and all preregistered recovery counters,
tokens, and latency. The runner's maximum is two provider calls per live case,
and retries pass through the existing pre-request call/token budget wrapper.
Evaluation cache integration is still unavailable, so neither failed nor
successful responses are cached in this path; the recovery policy version is
nevertheless included in the cache key contract.

Seventeen focused tests cover valid-first, invalid-to-valid,
invalid-to-invalid, budget and transport behavior, scope/guard/local
boundaries, all three live baselines, safe success/failure artifacts, cache
invalidation, and historical compatibility. The documented selected suite
passed 439 tests in 1.92 seconds. It excludes all 26 tests in
`test_eval_v2_single_human.py`, `test_eval_v2_candidates.py`,
`test_eval_v3_candidates.py`, `test_eval_v3_zh_single_human.py`, and
`test_phase3g_v3_zh_analysis.py` so the final verification does not load a
consumed corpus. It is not an unconditional full-suite claim.

An earlier broad development invocation included v3/v3_zh parsing tests. It
made no provider call and wrote no holdout artifact, but it did not satisfy
this phase's hash-only consumed-data rule and is not counted as final
validation. The final consumed-data check was SHA-256 only.

Final compileall, changed-file Ruff lint/format checks, `git diff --check`,
and the changed/untracked secret scan passed. Sixteen Phase 3G-C artifact
hashes and five consumed-corpus hashes matched their frozen values.

Frozen hashes are code
`69c7554c6865ad789e7d8b4a74816ccf0efecd98b86729226562f5fad99149fe`,
prompt/schema
`6dbe47902acc27f7c81f623c2c8c6c4546d685cfa9848ca5a931aa5b1a3fb658`,
schema recovery policy
`849b98d7118e6624d7ee516a2b2a1b7c0057f7a5b4bd783105693fb99f14de20`,
and combined config/recovery policy
`d670ac287f9d2ae0779ddc5fb09ae59f98a0e0397303764797b876909aa4b15a`.
The next protocol requires three fresh runs, 3/3 final operational success,
zero unrecovered schema failures, at least 99% first-attempt contract success,
and no more than two recovered schema failures per run. Existing quality and
product-value thresholds are unchanged. Model calls were zero; Run 03 was not
rerun; v4 and Phase 4 remain absent.

## Phase 3G-E - schema-recovery live dev attempt

Status: **2/3 complete runs; Run 03 scope failure; readiness failed**

The execution used commit `060db3b066f19093a4b8cd97edfa4b47e73e519a`,
model `deepseek-v4-flash`, provider-default thinking, requested temperature
0.0, no semantic repair, and only the single retry allowed by
`schema-recovery-v1`. The global authorization was 520 provider calls and
1,000,000 reported tokens. Evaluation cache integration remained
`not_available`.

The smoke completed all five requested operations. Its three live cases each
used one valid first attempt; deterministic not-checkable and unavailable
routes used zero model calls. There were no smoke retries or operational
failures.

Runs 01 and 02 each produced all 432 records, used 168 calls, and completed
with zero schema, transport, scope, guard, local-validation, or budget
failure. Each retained 17 full-context, 31 Retrieval-to-Judge, 12
deterministic not-checkable, and 12 deterministic unavailable Adaptive routes.

| Run | Baseline | Macro-F1 | Fixed-six | Contradiction recall | Not-checkable recall | Calls | Total tokens |
|---|---|---:|---:|---:|---:|---:|---:|
| 01 | `single_agent_live` | 0.972028 | 0.972028 | 1.000000 | 0.833333 | 72 | 64,972 |
| 01 | `retrieval_judge_live` | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 48 | 101,990 |
| 01 | `adaptive_live` | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 48 | 85,588 |
| 02 | `single_agent_live` | 0.986087 | 0.986087 | 1.000000 | 0.916667 | 72 | 65,953 |
| 02 | `retrieval_judge_live` | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 48 | 101,234 |
| 02 | `adaptive_live` | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 48 | 83,200 |

Adaptive observed-label macro-F1 was `1.000000` in short, medium, long, and
unavailable strata in both complete runs. Single Agent was `0.893333` and
`0.949206` on the short stratum, and `1.000000` on the other three strata.
Available-source strata have five observed labels, so their fixed-six
macro-F1 is `0.833333`; unavailable has one observed label and fixed-six
macro-F1 `0.166667`.

Run 03 made 121 provider calls for 120 logical model decisions. A first
attempt schema failure was recovered by one retry, using 1,467 input, 556
output, and 2,023 total tokens with 5,646.498 ms retry latency. The validated
retry result then failed Judge scope integrity on
`retrieval_judge_live / phase3g_zh_dev_059` with allowlisted
`claim_id_mismatch`. The canonical failure taxonomy remains
`local_validation`, while the typed exception and code identify the
operational category as scope. Scope failures do not trigger schema retry, so
the run stopped immediately and retained only the sanitized failure artifact.

Across attempted dev model decisions, first-attempt contract success was
`455/456 = 0.997807`; recovered/unrecovered schema failures were `1/0`, and
the maximum recovered count in any run was one. Those individual recovery
conditions passed, but final operational success was 2/3 and operational
scope failures were not zero. All preregistered quality, stability, and
product-value conditions requiring three complete runs are not evaluated and
cannot pass readiness. No Run 03 metrics or three-run aggregate were generated.

Final usage was 460 calls, 471,393 input tokens, 201,157 output tokens, and
672,550 total tokens, including smoke and the failed round. Overall model
latency was 3,239.114 ms p50 and 9,482.966 ms p95. The append-only result is
`eval_runs/phase3g_recovery_zh_dev_stability/execution_outcome.json`; its
sibling `artifact_manifest.json` verifies all 15 retained input artifacts.

Phase 3 remains formally unpassed, v4 blind-holdout creation is not eligible,
and Phase 4 remains ineligible. Consumed v2/v3/v3_zh files were only
byte-hashed, v3_zh was not rerun, and no v4 was created.

The same boundary-compliant selected suite passed 439 tests before and after
execution. It excludes all 26 tests in the five documented consumed-corpus
loader files and is not an unconditional full-suite claim. Final compileall,
`git diff --check`, exact runtime-secret and changed-artifact sensitive-pattern
scans, frozen input hashes, runtime harness hashes, and all five byte-only
consumed-corpus hashes passed.

## Phase 3G-F - local verdict ownership ready

Status: **offline implementation complete; authorized live execution pending**

The Phase 3G-E failure was possible because the live Judge asked the provider
to generate the complete canonical `JudgeOutput`. That made `claim_id`,
`source_ids`, evidence `source_id`/`locator`, corroboration, and version fields
model-authored even though each was already known from trusted local input.
The recovered Run 03 response was schema-valid but its generated `claim_id`
did not equal the request claim ID, so the unchanged scope validator correctly
failed closed.

`LiveJudgeSemanticOutput` now contains exactly four required keys:
`relation`, `confidence`, `reason`, and nullable `evidence_span`.
`additionalProperties=false` follows from the strict Pydantic contract. The
Judge prompt payload removes claim/source IDs, URL, file/line provenance,
locator/offsets, and version metadata. Local assembly writes canonical
identity and association fields from `JudgeInput`, maps a returned evidence
substring to the first ordered containing candidate, and then runs the
existing scope, evidence, availability, and deterministic conflict guards.

As a result, a model cannot alter the canonical claim/source ID or locator.
Returning any of those fields is a provider schema-contract failure and may
trigger exactly one `schema-recovery-v1` retry. No response field is deleted,
filled, corrected, or semantically normalized. A forged evidence span remains
a typed `JudgeScopeError`; a second schema failure or any scope/span/guard/
transport/budget/local-validation failure remains terminal.

Single Agent was already semantic-only and is unchanged. Adaptive full-context
routes inherit that contract; Adaptive retrieval routes inherit the new local
Judge assembly. Historical `JudgeOutput`, `Verdict`, prediction, and failure
artifact models remain readable, including the legacy `claim_id_mismatch`
code. That code is retained for compatibility but is structurally unreachable
from a schema-valid response under the new live Judge contract.

Offline verification passed 196 focused tests and the 453-test
boundary-compliant selected suite. The latter excludes five consumed-corpus
loader files containing 26 tests and is not an unconditional full-suite claim.
Phase 3G-C's 16 and Phase 3G-E's 15 retained artifacts matched their frozen
hash indexes. Consumed v2/v3/v3_zh files were accessed only for byte-level
SHA-256 checks. Model calls were zero; no historical run was modified or
rerun; no v4 or Phase 4 work was created.

New freeze values:

- Code: `88ace6e1b9675febb6071101aa04d5ee274b9c6986bdc6652156225d52c3d641`
- Prompt/schema: `eb83e292e51dcdcf7b39679a6998cdfb4b727eb286b7768d36f3b0747438fff5`
- Ownership policy: `8432ead334bc0761b4bf78f1bccc19a81b1083741751275418245bff77b9cd8c`
- Config/recovery/ownership: `d8a3ce764614e31c6151ad0a2859e143206a489af9cc55aac581ab4240929ed2`

## Phase 3G-F - local verdict ownership live dev

Status: **complete; all preregistered dev-readiness conditions passed**

The frozen ownership contract completed its smoke and three entirely fresh
72-case runs. Smoke made two live calls and no calls for deterministic
not-checkable or unavailable cases. All runs retained Adaptive route counts
17/31/12/12 and completed all six baselines.

| Run | Single Agent macro-F1 | Retrieval-to-Judge macro-F1 | Adaptive macro-F1 | First/recovered/unrecovered schema failures |
|---|---:|---:|---:|---:|
| 01 | 0.897198 | 1.000000 | 1.000000 | 0 / 0 / 0 |
| 02 | 0.957672 | 1.000000 | 1.000000 | 2 / 2 / 0 |
| 03 | 0.986087 | 1.000000 | 1.000000 | 2 / 2 / 0 |

Adaptive fixed-six macro-F1, weighted-F1, balanced accuracy, contradiction
recall, entailed precision, partial F1, not-checkable precision/recall/F1,
and evidence-span F1 were all `1.000000` in each run. High-confidence error
and false-block rate were zero. Its observed-label macro-F1 was `1.000000`
for short, medium, long, and unavailable strata in all rounds. Fixed-six
stratum values remain lower where only a subset of labels is supported and
must not be read as observed-label quality.

Adaptive and Retrieval-to-Judge relation agreement were `1.000000`; Adaptive
route agreement was `1.000000`. Single Agent relation agreement was
`0.888889`, with eight unstable cases. Adaptive's mean macro-F1 delta versus
Single Agent was `+0.053014` and versus Retrieval-to-Judge was `0.000000`.
Its mean long-stratum delta versus Single Agent was `+0.073228`; input-token
reduction versus Single Agent was `36.22%`.

Combined first-attempt contract success was `500/504 = 0.992063`. All four
first-attempt schema failures were separately recorded and recovered; no run
exceeded two recoveries, no failure was concealed as first-attempt success,
and unrecovered/final operational failures were zero. Total use including
smoke was 510 calls, 330,546 input tokens, 116,331 output tokens, and 446,877
total tokens. Retry use was 2,775 input, 1,157 output, and 3,932 total tokens.
Overall latency was 2,268.055 ms p50 and 3,775.860 ms p95.

All frozen readiness checks passed. A new v4 blind holdout may now be
created under separate authorization, but Phase 3 remains formally unpassed
until a new one-time formal gate succeeds. Phase 4 remains ineligible. This
is a pair benchmark and does not establish full Claim Miner plus Judge
Multi-Agent value. No consumed corpus was loaded or rerun.

Post-run verification passed the same 453-test boundary-compliant selected
suite, compileall, changed-files Ruff lint/format checks, `git diff --check`,
secret scans, frozen implementation/dataset hashes, all Phase 3G-C/E
historical artifact hashes, and the five byte-only consumed-corpus hashes.
The 20-file result manifest has SHA-256
`62bcd641e902530543197d93bbc4e0d15a11b2190a0d3acdec655d4f5ff5ab00`.

## Phase 3H-A - v4_zh blind candidate pack

Status: **complete; awaiting independent human annotation**

The new `eval_sets/v4_zh/` pack contains 72 canonical Chinese pair
candidates across 24 sources, with exactly three cases per source. Source
strata are 7 short, 7 medium, 6 long, and 4 frozen unavailable; production
chunking and the frozen Adaptive Router produce aggregate routes of
17 full-context, 31 Retrieval-to-Judge, 12 deterministic not-checkable, and
12 unavailable. The six intended construction strata contain 12 cases each,
but no case-level target is recorded in the candidate file or reviewer
packet.

All 20 available snapshots are official, native `zh-CN` technical
documentation or release-note excerpts. Provenance includes publisher, URL,
retrieval UTC, HTTP status, upstream body hash, canonical content hash, and
license or terms reference. The four unavailable records preserve their
failed safe-fetch status and contain no inferred source content.

The blind packet has no gold relation/evidence, model proposal, answer-bearing
notes, or evaluation-model exposure. Leakage auditing found no cross-corpus
threshold hit against readable prior datasets and used only byte hashes plus
retained non-label metadata for consumed corpora. One high-similarity
within-pack pair is an intentional same-source, single-number contrast and is
explicitly documented for reviewer awareness without revealing its relation.

Production code, Judge, Router, Retrieval, prompt/schema, recovery, and
ownership policy remain byte-identical to checkpoint
`2ef101b6cd8bdd4d1db74e8b894ec5afb0958407`. No model was called, no consumed
holdout was loaded or rerun, no gold was generated, the v4_zh gate remains
`not_run`, and Phase 4 remains unimplemented.

## Phase 3H-B - v4_zh single-human gold freeze

Status: **complete; eligible for one separately authorized internal gate**

The filled reviewer packet was preserved outside the repository before the
repository copy was restored to the original blank packet hash. The final
private record contains 72 unique annotations, complete reviewer provenance,
an independent-human attestation, and `viewed_model_output=false`. Public
artifacts expose neither reviewer identity/ID nor full notes.

All 72 case/source/claim triples match the frozen candidates. Label support is
11 entailed, 11 partially entailed, 11 contradicted, 15 not-in-source,
12 source-unavailable, and 12 not-checkable. All 33 substantive evidence
spans are exact continuous substrings of the native Chinese canonical source;
the other 39 evidence fields are null as required.

The original rich source snapshots, candidates, blank reviewer packet, and
candidate bundle retain their Phase 3H-A hashes. A deterministic, derived
runtime source fixture was added because the production loader accepts the
strict `SourceFixture` contract rather than the richer provenance record. No
production behavior or frozen code/prompt/config/ownership input changed.

The review manifest records `single_human_review_complete`,
`counts_as_human_review=true`, internal-gate eligibility, no second human
review, no evaluation-model exposure, and `holdout_execution_status=not_run`.
Consumed corpora were touched only for byte-level SHA-256. Model calls and
holdout executions were zero.

## Phase 3H-C - one-time v4_zh final internal gate

Status: **consumed operational failure; Phase 3 completed with known
limitations; Phase 4 not eligible**

All pre-run boundaries passed at commit
`689b245be01df73a04757343eaab1362147610c9`: the worktree was clean, no
remote was configured, all gold/source/implementation freeze hashes matched,
the environment was configured for `deepseek-v4-flash`, and the
boundary-compliant selected suite passed 468 tests while excluding the five
consumed-corpus loader files containing 26 tests. No smoke, preview, cache
reuse, or source refetch preceded the formal execution.

The only authorized v4_zh execution failed closed on
`single_agent_live` case `v4_zh_candidate_003`. Two live predictions had
completed. The third provider response passed the JSON/Pydantic contract, but
the frozen local path then raised `ValueError`; the safe artifact classifies
this as `local_validation`. The artifact records no finer local-validation
code or model content, so the exact subtype is unknown and must not be
reconstructed or guessed. It was not a schema, transport, scope, guard,
recovery, or budget failure.

Execution stopped after 3 provider calls with 2,731 input tokens, 2,713 output
tokens, and 5,444 total tokens. All three attempts reported usage and passed
the first-attempt schema contract; schema retries and schema failures were
zero. Provider latency was 6,691.295 ms p50 and 9,783.875 ms p95. No frozen
pricing snapshot exists, so `cost_usd` remains null.

The run did not complete 72 cases, Retrieval-to-Judge and Adaptive were not
reached, and no canonical success artifacts or quality metrics were
generated. Missing metrics are not estimated. v4_zh is consumed by this one
formal attempt, rerun is forbidden, and no v5 will be created. Phase 3 is
therefore final as `completed_with_known_limitations`; `phase4_eligible=false`.

This was a pair-level benchmark with atomic claims supplied directly. Claim
Miner did not participate, so the attempt cannot establish the value of a
complete Claim Miner plus Judge Multi-Agent system.

The safe failure artifact SHA-256 is
`6cc55f01d52752c20efcd0a0f2f1196d826e7df285a671cba47ad014c250f034`;
the execution outcome SHA-256 is
`f312a738b3e150d17b200a11f7d6c6326399fb09f93f58257e519b20b000dde5`.
The one-shot lock, finalized budget ledger, and dataset execution-status
hashes are respectively
`d541fd26b4af6c11bd54f75954647ab9b484ae9f3eb71aa9cbfb5f8b1bebfbdd`,
`a086c00d4c0ac6fcac771ac658e8eb7ab2a9446236bfb6bd1f4db3f17c0da0c9`,
and
`9fc6b97e5c02a186bd988e412cf96dcac15817cb03d66461eaced9f535375392`.

Post-run verification passed the same 468-test selected suite, compileall,
`git diff --check`, exact-key and privacy scans, all v4_zh and implementation
freeze hashes, Phase 3G-C/E/F historical artifact hashes, and the five
byte-only consumed-corpus hashes. There were no changed Python files for a
repository changed-files Ruff run; the uncommitted one-shot executor itself
passed Ruff lint and format checks.

## Phase 4A - typed local validation, SARIF, and GitHub integration

Status: **engineering continuation with known limitations; Phase 3 not
passed; phase4_eligible=false**

Phase 4A does not reinterpret the consumed v4_zh outcome. Formal blind-holdout
quality was not measured, no official F1 exists, and no v5 or holdout rerun is
permitted. The Phase 3 status remains `completed_with_known_limitations`.

The `single_agent_live` post-schema audit now has explicit outcomes:

| Branch | Result |
|---|---|
| Wrong relation with otherwise valid in-scope output | Scoreable prediction |
| Available source classified `source_unavailable` | Scoreable prediction |
| Evidence is not a continuous source substring | Typed fail-closed |
| Unavailable source metadata is contradicted | Typed fail-closed |
| Model-call or telemetry accounting diverges | Typed fail-closed |
| Canonical prediction raises ValueError/AssertionError/KeyError | Typed fail-closed |
| Substantive prediction lacks grounded evidence | Typed fail-closed |

`LocalValidationError` contains only an allowlisted code. New failure artifacts
use `phase4a-local-validation-failure-v7`, record the failure stage/category
and fixed code, and clear exception cause/context before serialization.
Schema-only recovery remains unchanged; local, scope, evidence, guard,
transport, and budget failures are not retried. Historical artifact readers
remain compatible.

The new deterministic SARIF 2.1.0 renderer consumes canonical `AuditArtifact`
data and calls the existing `decide_policy` mapping. It emits six stable rule
IDs, policy-derived SARIF levels, percent-encoded repository-relative
Markdown URIs, one-based line/column locations, stable fingerprints, and
sorted results. Empty findings still produce a valid file. Atomic output is
restricted to the project root, including symlink/path-traversal checks.
Claim text, model reason, evidence/source text, URLs, credentials, headers,
absolute paths, and request content are not serialized.

`evidencetrace check <path> --sarif <output>` now writes SARIF before returning
the existing policy status contract: 0 pass, 1 policy failure, 2 operational
failure. `.github/workflows/evidencetrace.yml` is both PR-triggered and
reusable. It grants only `contents: read` plus job-level
`security-events: write`, preserves a policy-failing SARIF for Code Scanning
upload, then propagates the original exit code.

The allowed test suite passes 479 tests. It excludes seven consumed-corpus
loader modules containing 41 tests: the prior five files plus the two v4_zh
loader tests. New coverage uses only fully synthetic data and offline
bad/simple/complex fixtures. No model was called, consumed datasets were only
byte-hashed, and no historical artifact, gold, source, freeze hash, remote, or
provider was changed.

This product work cannot establish complete Multi-Agent value. The consumed
pair gate supplied atomic claims and did not run Claim Miner; a full-document
benchmark with gold claim extraction remains outstanding.

## Phase 4B - full-document Multi-Agent benchmark ready

Status: **engineering continuation with known limitations; registered but not
live-executed; Phase 3 not passed; phase4_eligible=false**

The new `full_document_v1` dataset is author-constructed and deterministic,
not blind or public. It contains 24 Markdown documents, 120 exact-span
provisional gold claims, and 72 local source fixtures. Dev and test each
contain 12 documents; short, medium, and long strata each contain eight.
Every document has five gold atomic claim units. Relation support is 20
entailed, 20 partially entailed, 20 contradicted, 21 not-in-source, 19
source-unavailable, and 20 not-checkable.

Gold rows preserve the exact Markdown substring and offsets, one-based start
and end lines, claim type/checkability, citation URLs, local source IDs,
relation, and a continuous evidence/source binding where required. Dataset
validation recomputes every document/source/bundle hash and checks all local
span and binding invariants. No consumed corpus participates in generation,
loading, or tests.

`single_agent_document_live` uses one primary strict Pydantic response per
document. The response owns only claim text/type/checkability, selected
citation URLs, relation, confidence, reason, and evidence text. Local code
maps unique source-order spans and adds claim IDs, line/offset locators,
source IDs, and evidence locators.

`miner_adaptive_judge_live` invokes the existing `ClaimMinerAgent` separately
for each allowed parser paragraph. Whitespace-normalized Miner spans are
mapped back to the unique continuous raw Markdown span, including soft
line-break claims. Citation binding is local and source-scoped. Existing
`adaptive_live` then supplies unavailable/not-checkable deterministic routes,
full-source Single Agent verification, or lexical Retrieval plus
`ClaimJudgeAgent`. Router and Retrieval are not reported as Agents.

Both systems share schema-recovery-v1, a provider-attempt/token budget,
payload-free error artifacts, canonical `AuditArtifact`, existing policy
decisions, and deterministic SARIF. Miner, Judge, and document Single Agent
telemetry remain separate. The preregistered per-split ceiling is 600
provider attempts and 1,000,000 reported tokens, including at most one schema
retry per logical call and no semantic repair.

Metric behavior is frozen before any live execution:

- exact claim extraction precision/recall/F1;
- one-to-one relaxed span token precision/recall/F1;
- atomicity violation rate;
- citation-binding and line-locator accuracy;
- observed-label and fixed-six relation macro-F1, weighted-F1, balanced
  accuracy, contradiction recall, and not-checkable F1;
- evidence-span F1 and strict end-to-end claim-plus-relation F1;
- document policy-decision accuracy and Miner/Judge handoff failure rate;
- Miner, Judge, and document Single Agent calls, tokens, p50/p95 latency;
- Single Agent versus Multi-Agent quality and resource deltas.

Relation metrics include a `__missing__` prediction for every unextracted gold
claim. End-to-end true positives require both an exact claim span and a
correct relation. No metric is restricted to successfully extracted claims.

The focused suite runs both systems over all 24 documents with the existing
deterministic fake model and produces byte-stable canonical result/SARIF
artifacts. Perfect fake-model values are test assertions, not benchmark
quality evidence. The boundary-compliant suite passes 498 tests while
excluding the same seven consumed-corpus loader modules containing 41 tests.
Real model calls remain zero and neither dev nor test has a live result.

The `full_document_v1` preregistration is `registered_not_executed`. A future
live run requires separate authorization naming the model, dev/test scope,
provider-call and token limits, cache policy, and failure handling. Phase 3
remains `completed_with_known_limitations`; `phase4_eligible` remains false.

## Phase 4B - full-document live dev attempt

Status: **smoke operational failure; full dev and test not run; Phase 3 not
passed; phase4_eligible=false**

The authorized run began from
`033f32dc89aa04afa95dd793f6dafcb82fe5746b`. The worktree was clean, no
remote was configured, all eight dataset/implementation freeze values
matched, the provider environment was configured for
`deepseek-v4-flash`, and the boundary-compliant selected suite passed 498
tests. The selected suite continued to exclude seven consumed-corpus loader
modules containing 41 tests.

A one-shot executor read only the first 12 dev document rows, their 60 gold
rows, the first 36 dev source fixtures, and the 12 physical dev Markdown
files. It did not parse a later JSONL row or open a test Markdown document.
Full-file access was limited to required byte-level frozen-hash verification.
The deterministic smoke selector chose `fdv1_doc_001`, the first dev document
in frozen manifest order.

The first `single_agent_document_live` smoke failed closed with stage
`single_agent`, category `schema`, and allowlisted code
`model_schema_invalid`. The frozen schema-recovery-v1 path made two provider
attempts for this one logical call. Because a second attempt is possible only
after the first `ModelSchemaError`, and the final safe code is also a schema
failure, the bounded counters are one first-attempt schema failure, one
schema-retry call, zero recovered failures, and one unrecovered failure. No
saved artifact reveals the malformed fields, finish reason, response content,
prompt, document/source body, headers, credentials, or model rationale; none
is reconstructed.

Both attempts reported usage: 1,972 input tokens, 4,096 output tokens, and
6,068 total tokens. Per-call latency was intentionally not retained in the
failure outcome, so p50/p95 are unavailable rather than estimated. Cost
remains null because no frozen pricing snapshot exists. Cache status was
disabled/not available.

The Miner plus Adaptive Judge smoke did not start. Neither full 12-document
dev baseline ran, and no test data was loaded, parsed, or run. Consequently
there are no extraction, relation, evidence, end-to-end, policy, handoff, or
relative Multi-Agent quality metrics and no dev error classification to
report. Missing metrics are not inferred.

The safe failure artifact is
`eval_runs/phase4b_full_document_live_dev/failed_attempt.json` with SHA-256
`e175b57583f621432cb6a96a5624bf294e71e841c1dd481c58feb514da3e2241`.
The execution outcome beside it has SHA-256
`ffee09d489132de01b4047d5252f0f7c2ef2add960e1b95e993e4c394ee0745f`.
The frozen one-shot executor SHA-256 is
`483c4d21c9e4ae9772fd3d419875c5007d3c19e3d8eb2de01f0aa04571a29fc9`.

This dev attempt is not blind, human-reviewed, formal, or test-split evidence.
It supplies no basis for a Multi-Agent advantage claim. Engineering is not
ready to execute the test split: a separately authorized future continuation
must first diagnose the payload-free schema failure using synthetic/dev-only
fixtures, freeze any general repair, and obtain fresh live-dev authorization.
Phase 3 remains `completed_with_known_limitations`; `phase4_eligible=false`.

Post-run verification passed the same 498-test boundary-compliant suite,
compileall, `git diff --check`, exact-credential and failure-artifact privacy
scans, the frozen executor Ruff lint/format checks, all eight Phase 4B freeze
hashes, the historical Phase 3H failure hashes, and byte-only consumed-corpus
hashes. No repository Python or production file changed.

## Phase 4B.1 - provider contract hardening and bounded smoke

Status: **both independent dev smokes operationally successful; full dev and
test not run; Phase 3 remains completed with known limitations**

The full-document adapter now enforces `max_tokens=8192`, records that value
in provenance and cache keys, limits model reasons to 240 characters, and
retains strict required fields plus `additionalProperties=false`. Safe
diagnostics distinguish response envelope, content contract, JSON decode, and
Pydantic schema validation. They record only allowlisted finish reason,
numeric usage, response presence/length, fixed error code/path, and suspected
truncation. No response content, reasoning, prompt, source body, validation
input, header, or credential is retained.

The prior failed artifact remains byte-identical and
`unknown_schema_failure`; its aggregate 2,048-token saturation is a hypothesis,
not proof of truncation. Synthetic tests cover length finish, truncated JSON,
missing/extra fields, bad enums and types, overlong reason, valid canonical
responses, cache invalidation, and diagnostics privacy.

The one-shot live execution used only `fdv1_doc_001`. The document Single
Agent's first response ended with `stop` and failed Pydantic validation at
`claims/0` with `value_error`; it was not suspected to be truncated. Its one
schema-only retry succeeded. The independent Multi-Agent smoke completed four
Miner and five Judge calls, all first-pass valid. Global usage was 11
attempts, 7,363 input tokens, 11,234 output tokens, and 18,597 total tokens.
No unrecovered schema, transport, scope, span, guard, local-validation, or
budget failure occurred. Test remained `not_loaded_not_parsed_not_run`.

One preliminary pytest command mistakenly executed the consumed v2 candidate
loader because it excluded `test_phase3b_v1_integrity.py` instead of
`test_eval_v2_candidates.py`. It printed no corpus content, made no provider
call, and is not counted as compliant verification. The corrected pre-live
suite excluded all seven documented consumed loaders and passed 510 tests.
The deviation remains disclosed.

This bounded smoke is not a complete dev result, blind benchmark, or evidence
that Multi-Agent quality exceeds Single Agent quality. The implementation is
engineering-ready for separately authorized complete dev execution. Test
execution is still unauthorized. Phase 3 remains
`completed_with_known_limitations`; `phase4_eligible=false`.
