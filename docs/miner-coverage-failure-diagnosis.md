# Miner Bounded-Window Coverage Failure Diagnosis

Date: 2026-07-27

Status: `offline_diagnosis_complete`

Miner P0: `improved_but_blocking`

Phase 3: `completed_with_known_limitations`

`phase4_eligible=false`

This is a read-only diagnosis of the six failed windows from the one-time
Miner Architecture Live Recheck. It is not a new live run, benchmark, model
evaluation, or implementation change.

## Evidence Inventory

Authoritative context:

- `multi_agent_researchops_project_plan.md`
- `docs/miner-architecture-decision.md`
- `docs/dogfood/round_1_2026-07-27.md`

Frozen execution root:

```text
/tmp/evidencetrace-dogfood-round1-miner-architecture-live-recheck
```

Repository and source boundaries:

| Item | Verified value |
|---|---|
| Git HEAD | `eca904ebdd77c78a838c18b24ba5be63e303e133` |
| `src/` aggregate | `19f8a253042756f6acf0ff96d4570c6bf308c8585ffd7f0f88456905ec22f558` |
| `tests/` aggregate | `e21b3a66e239eb116777de2d021cd3da33be0cde368b23b17825deabde9167de` |
| Round 1 report | `f2b02ea05454c36b5793bdd9ef18021a6b8a2d08a83e0a5fe265b2dc5a5da5c5` |
| Staged diff | empty |
| Git remote | none |

All 13 live artifact hashes matched the Round 1 report:

| Case | Artifact | SHA-256 |
|---|---|---|
| README | `audit.json` | `72c7957ba1eb768c6c41cf093589ec1c43992bd2b91f18582b11518175e207d6` |
| README | `product_run.json` | `cf33438c0247fbff7d633cccba071d573e795daf454c9af1132f4455c4e7f8d8` |
| README | `audit.md` | `344eda5da45318eadb51b6b654c54b76c896599c918dac4c663de256db689364` |
| README | SARIF | `c438ebfe95141951872e072b80faa55a6533934c6d3bdb44657b3df5d0c42d60` |
| ADR | `audit.json` | `97aa0b97b9058a038104639af1b9953cab3391a9966e4514a0d151848270f093` |
| ADR | `product_run.json` | `28f89b3cb85e7b1ad5d78c84cf78574946c49d06f28875ee74e37707f0e3e070` |
| ADR | `audit.md` | `979d182b9f552ec803ce4538bf6089a10435011c84e69eaeace1389f91fbd3d8` |
| ADR | SARIF | `589b3fafafde81ac05b0121e6aa80c404f40bf9c49cb7a02ec908bcd09cc22f5` |
| PR | `audit.json` | `39ba515d784b775da15d134041afa2358d68b1731c757ddced721ce5e2661a6e` |
| PR | `product_run.json` | `030d23b4dbe730532cd5b6b2a5911c056c8d7c29d2a82f0f4224196db5984b08` |
| PR | `audit.md` | `a7c179a04136167f3f998d1fed62e37d3b3c3cec4bee09d67a7eba80e0a6f054` |
| PR | SARIF | `f680210bda160f45f300d5fe6ac031011ec741ac82f557805ddb7a30db855309` |
| PR | suggested patch | `173271a1152c21b28a96dd9216dc12b87f6f105b9bdae89c68b1fc8ac2fe6975` |

No raw provider response, rejected draft, prompt, or reasoning is present.
Consequently, the direct local rejection layer is known, while the exact
model-authored failure content remains unknown.

## Six Failed Windows

Offsets below are half-open offsets in trusted model-visible paragraph text.
Protected occurrence values come from the deterministic local regex and
source map, not from model output. Window SHA-256 values are diagnostic
fingerprints; full window text is not persisted here.

| Case | Paragraph / window | Source lines | Length / window SHA-256 | Attempts | Typed code and direct layer | Protected occurrences | Claims | Core label and public coverage | Certainty |
|---|---|---:|---|---|---|---|---:|---|---|
| ADR | `p_0002` / `w_0001` (index 1) | 4 | 16 / `f1a91e22a1b9ed218c27d225337ef1edda2db1b0b5c4140996a72194dc073236` | 1, no recovery, `stop` | `miner_scope_missing_protected_token`; `ClaimMinerAgent._validate_drafts`, `agents/miner.py:545-578` | date `2020-03-31`, window and paragraph `[6,16)`, line 4 | 0 accepted, 0 withheld exact | ADR date metadata; paragraph ET2001 present | Direct layer proven; exact protected branch and draft unknown |
| ADR | `p_0003` / `w_0002` (index 2) | 9-10 | 122 / `ad7efffcb293a5d7295173402071564b818a8de8e0fafedfd2335a6aed396b0f` | 1, no recovery, `stop` | `miner_scope_invalid_claim_fragment`; `_validate_drafts`, `agents/miner.py:547-553` | none | 0 accepted, 0 withheld exact | repository-creation / first-release context; paragraph ET2001 present | Rejection rule proven; rejected fragment text and exactness unknown |
| ADR | `p_0006` / `w_0001` (index 1) | 19-20 | 99 / `654364695428bffb39b1aaa3e75428107c02c380fa1fa0381301ded294f37006` | 1, no recovery, `stop` | `miner_scope_missing_protected_token`; `_validate_drafts`, `agents/miner.py:545-578` | `0.0.0` at window/paragraph `[15,20)`; `1.0.0` at `[30,35)`; both line 19 | 0 accepted, 1 withheld exact | chosen option and initial maturity rationale; paragraph ET2001 present | Failure and one exact draft proven; omitted token and draft unknown |
| PR | `p_0034` / `w_0005` (index 5) | 261-262 | 154 / `10e2c70d1aa8803fdfd9190d1c021b27cb5c1429046bcffe339627759e0db783` | 1, no recovery, `stop` | `miner_scope_non_source_span`; `_unique_monotonic_offsets`, `agents/miner.py:419-452`, raised at line 433 or 443 | none | 0 accepted, 0 withheld exact | trigger context and write-access consequence; paragraph ET2001 present | Mapping failure proven; paraphrase, order, or forward/reverse branch unknown |
| PR | `p_0034` / `w_0007` (index 7) | 261-262 | 331 / `671bea73e4223b1898c274ced76c2420012533a78a4a400e2e5cd765a917556a` | 1, no recovery, `stop` | `miner_scope_missing_protected_token`; `_validate_drafts`, `agents/miner.py:545-578` | `only`, window `[147,151)`, paragraph `[965,969)`, lines 261-262 | 0 accepted, 1 withheld exact | attack/write/secret consequence and mitigation; paragraph ET2001 present | Failure and malformed leading boundary proven; causal model effect inferred |
| PR | `p_0034` / `w_0008` (index 8) | 261-262 | 108 / `f97281598a3a818666bd7580e7a53a949cd1faaf8d70b1247e3a6e03d06ab036` | 1, no recovery, `stop` | `miner_scope_missing_protected_token`; `_validate_drafts`, `agents/miner.py:545-578` | `more`, window `[56,60)`, paragraph `[1206,1210)`, lines 261-262 | 0 accepted, 0 withheld exact | navigational reference sentence, no required core fact; paragraph ET2001 present | `more details` false association proven; draft content unknown |

All six codes are emitted only after `LiveMinerDraftOutput` passes strict
provider JSON/Pydantic validation:

```text
model_client.py:948-965
-> agents/miner.py:472-490
-> agents/miner.py:541-583
```

Therefore there is no provider/schema contract failure among these six
windows. Scope failures are intentionally not retried.

The exact protected-coverage sub-branch cannot be reconstructed. Both lexical
coverage (`agents/miner.py:554-561`) and span coverage
(`agents/miner.py:571-578`) emit the same typed code. In addition,
`ProductAuditPipeline._mine` writes failed outcome coverage as zero at
`product.py:994-1004` instead of preserving the safe count carried by
`MinerScopeError`. The artifact's zero does not identify which occurrence was
missed.

## Core Fact To Window Mapping

### ADR

| Fact label | Window | Window result | Canonical result | Missing category |
|---|---|---|---|---|
| Initial-version presence/absence influences next version | `p_0003_w_0001` | complete | `c_0001` retained | none |
| Repository creation choice influences first release | `p_0003_w_0002` | invalid fragment | absent | explicit failed window |
| No initial version / first release `1.0.0` | `p_0004_w_0001` | complete | `c_0002`, `c_0003` retained | none, although `c_0002` is a short option fragment |
| Initial `0.0.0` / first release `0.1.0` | `p_0005_w_0001` | complete | `c_0004` retained | none |
| Chosen option `0.0.0` | `p_0006_w_0001` | missing protected token | absent | explicit failed window |
| `1.0.0` maturity / general-use rationale | `p_0006_w_0001` | missing protected token | absent | explicit failed window |
| Exceptional later upgrade to `1.0.0` | `p_0006_w_0002` | complete | `c_0005` retained | none |
| Strict versioning afterwards | `p_0006_w_0002` | complete | `c_0006` retained | none |

The chosen option's literal statement and its first rationale are entirely in
the failed 99-character window. Their absence is fully explained by that
window producing zero accepted claims. The artifact does not reveal the one
withheld exact draft, so it cannot prove which part the model attempted.

### PR #871

| Fact label | Window | Window result | Canonical result | Missing category |
|---|---|---|---|---|
| Labeler requires write permission | `p_0034_w_0001` | complete | `c_0001` retained | none |
| Fork `pull_request` receives read-only tokens | `p_0034_w_0002` | complete | `c_0002` retained with `only` and `at most` | none |
| Permission error consequence | `p_0034_w_0003` | complete | `c_0003` retained with negation | none |
| Switch workflow to `pull_request_target` | `p_0034_w_0004` | complete, empty | absent | **coverage-policy blind spot** |
| Trigger changes context and grants write access | `p_0034_w_0005` | non-source span | absent | explicit failed window |
| Security warning lead-in | `p_0034_w_0006` | complete | `c_0004` ends at the dotted abbreviation with an unmatched opening parenthesis | deterministic boundary plus fragment-validation blind spot |
| Attackers obtain write access or secrets | `p_0034_w_0007` | missing protected token | absent | explicit failed window; malformed leading boundary |
| Use trigger only in carefully constrained workflows | `p_0034_w_0007` | missing protected token | absent | explicit failed window |
| Documentation reference | `p_0034_w_0008` | missing protected token | absent | false hard-coverage association, not a core-fact loss |

`p_0034_w_0004` is a schema-valid, complete window containing core technical
identifiers but returning no claim and no window failure signal. This is the
confirmed answer to the complete-window blind-spot question.

There is a second, different blind spot in `w_0006`: the sentence splitter
treats the period in a dotted abbreviation as a boundary. The resulting
window ends with an unmatched opening parenthesis, the next window starts with
the unmatched closing parenthesis, and the fragment validator still accepts
the first part as complete. The merged span would be 463 characters, above
the current 384-character bound, so merely suppressing that boundary does not
prove a safe replacement split.

ET2001 correctly reports all paragraphs that contain explicit failed windows.
It does not fully represent all core coverage gaps:

- PR has one ET2001 because `w_0005`, `w_0007`, and `w_0008` failed.
- ET2001 does not identify the complete-empty `w_0004`.
- ET2001 does not identify the accepted but incomplete `w_0006` claim.
- If the same complete-empty behavior occurred without another failed window,
  no ET2001 would be produced.

## Validation Path

```text
Markdown token parsing
  markdown.py:545-606, 760-820, 973-1082
-> trusted model-visible source map
  models.py:189-252
-> deterministic window plan
  agents/miner.py:203-228, 331-403
-> model request with strict LiveMinerDraftOutput
  agents/miner.py:472-490
  model_client.py:948-965
-> fragment validity
  agents/miner.py:406-409, 541-553
-> lexical protected coverage
  agents/miner.py:146-186, 554-561
-> exact/order/unique mapping
  agents/miner.py:419-452, 562-570
-> span protected coverage
  agents/miner.py:571-583
-> local metadata scope
  agents/miner.py:586-612
-> window failure isolation and paragraph aggregation
  product.py:842-1118
-> canonical paragraph outcome
  product.py:639-669
-> terminal / Markdown / SARIF
  render.py:12-159
  sarif.py:237-280, 284-339
```

Failure-family conclusions:

| Family | Direct rejection | Contract-correct? | False-positive evidence | Planner evidence | Unknown |
|---|---|---|---|---|---|
| Provider/schema | none among six | n/a | none | none | Raw valid responses are not retained |
| Invalid fragment | `_is_complete_claim` in `_validate_drafts` | Yes for the rejected draft | Rejected text unavailable, so no validator false positive is proven | ADR window itself is a complete sentence; no planner defect proven there | Exactness and fragment content |
| Non-source span | `_unique_monotonic_offsets` | Yes, fail-closed | No false positive can be proven without draft text | PR `w_0005` is a complete bounded sentence | Paraphrase vs order vs mapping direction |
| Missing protected | lexical or span coverage in `_validate_drafts` | Safety intent is correct | PR `w_0008` proves generic `more` is falsely associated as a hard comparator in navigation; relation of other omitted occurrences remains unknown | PR `w_0007` begins at a broken abbreviation boundary | Which occurrence each response omitted |
| Window boundary | deterministic sentence regex | Exactness is preserved, but atomicity is not | `w_0006`/`w_0007` parenthesis imbalance is deterministic | Proven split inside dotted abbreviation | A safe replacement boundary below 384 chars |
| Paragraph coverage | failed windows aggregate correctly | Correct for explicit failures | PR `w_0004` proves schema-valid empty core content can be marked complete | Not a source-map error | General frequency outside these cases |

## Confirmed, Inferred, Unknown

### Confirmed

- All six failures are post-schema local scope failures.
- Every failed window used one attempt, no schema recovery, and safe
  `finish_reason=stop`.
- Window text, hashes, ordering, non-overlap, and source locations are
  deterministic and in bounds.
- ADR `p_0006_w_0001` and PR `w_0007` each withheld one individually exact
  complete draft, but accepted none.
- PR `w_0004` is a complete-empty core-fact window with no local coverage
  signal.
- PR splitting breaks a dotted abbreviation and separates matching
  parentheses across `w_0006` and `w_0007`.
- PR `w_0008` treats navigation phrase `more details` as hard comparison
  coverage.
- Failed-window safe coverage counts are flattened to zero by product
  assembly.

### Inferred

- The malformed `w_0007` input likely makes atomic extraction harder, but the
  artifact cannot establish causation.
- The hard `only` occurrence may belong to a different atomic clause than the
  withheld exact draft in `w_0007`; draft text is unavailable, so association
  is not proven.
- A source-span contract might reduce future non-source failures, but one
  event is insufficient to justify that migration.

### Unknown

- Rejected draft text, ordering, punctuation, and protected-token content.
- Which protected occurrence was omitted in each missing-token failure.
- Whether the missing-token errors arose at lexical or span coverage.
- Whether PR `w_0005` was paraphrased, reordered, duplicated, or otherwise
  unmappable.
- Provider reasoning, raw content, token split, and why it selected empty or
  incomplete claims.
- Whether any extraction-oriented change would recover these facts reliably
  on broader documents.

## Option Comparison

| Option | Confirmed issue it can solve | Cannot solve | Exact/protected safety | Parallel pipeline | Calls / limits | Artifact, cache, provenance | Minimum code and tests | Rollback condition |
|---|---|---|---|---|---|---|---|---|
| A. Keep current `needs_human` only | Existing failed windows stay visible | Complete-empty `w_0004`, incomplete `w_0006`, and extraction loss | unchanged | no | unchanged | none | none | n/a |
| B. Finer deterministic modifier/identifier-aware boundary | Dotted-abbreviation split and unmatched fragment | Protected failures, non-source draft, complete-empty window | can preserve exactness if offsets remain trusted | no | may reduce or reshape windows; must remain precomputed | bump window policy and cache provenance | planner plus deterministic abbreviation/clause fixtures | any non-exact split, unbounded call change, or systematic oversized regression |
| C. Stronger coverage signal without claim acceptance | Complete-empty core identifier window | Does not recover rejected drafts, fix boundary, or explain model behavior | exact validator and protected guard remain unchanged | no | no new call | bump coverage policy; add allowlisted reason and cache/provenance change | trusted source-map anchor kind, paragraph aggregation, historical-reader and empty-window fixtures | valid intentional-empty technical text becomes systematically noisy, or any claim acceptance changes |
| D. Source-span/offset/ID typed contract | Could structurally constrain future non-source output | Protected association, empty output, planner boundary, atomicity | strong only with strict local offset validation | substantial risk of becoming candidate-span pipeline | nominally same, but new schema/recovery behavior | major provider schema, contract, cache, artifact migration | Miner schema/client fixtures, span ownership, migration and broad recall tests | candidate recall loss, model-owned identity, or parallel extraction path |
| E. No implementation | Preserves safety while unknowns remain | Leaves confirmed complete-empty blind spot | unchanged | no | unchanged | none | diagnosis only | reconsider only with new non-live evidence |

## Selected Minimum Recommendation

Select **C, narrowed to a parser-owned inline-code identifier coverage-only
signal**.

Proposed contract:

1. The trusted source map distinguishes inline-code identifier spans from
   ordinary link-label spans; it must not infer this later by string search.
2. A schema-valid empty Miner result that covers one or more such identifier
   spans cannot be marked `complete`.
3. It produces one allowlisted window reason such as
   `miner_coverage_empty_identifier_window`, then a paragraph
   `partial/needs_human` outcome through the existing aggregation path.
4. It does not create a claim, accept a withheld draft, repair text, change a
   relation, or assert that the identifier-bearing text is true.
5. Exact substring, protected-token hard failure, one schema recovery,
   window/call budgets, and all local ownership checks remain unchanged.
6. `MINER_COVERAGE_POLICY_VERSION` and cache/provenance inputs must change;
   historical artifacts remain readable.

This is intentionally a visibility fix, not an extraction-recovery claim. It
is supported by PR `w_0004`, where the parser can prove that model-visible
inline-code identifiers exist while the window returned zero claims and was
marked complete.

The recommendation does not alter the `more` classifier, salvage either
withheld draft, or change dotted-abbreviation boundaries. Those issues require
separate evidence and must not be bundled into the next implementation.

Rejected directions:

- A is insufficient because a confirmed core window is currently silent.
- B is not selected now because the obvious merged PR span is 463 characters,
  above the frozen bound, and no safe replacement split has been proven.
- D is disproportionate to one non-source event and does not satisfy the
  previously documented prerequisites for reconsidering candidate spans.
- E is unnecessary because the complete-empty identifier blind spot is
  deterministic and admits a fail-safe visibility change.

## Safety Invariants

- Claims remain contiguous, character-for-character model-visible substrings.
- No rejected or withheld draft is accepted.
- No deterministic code asserts a fact or relation.
- Protected-token guard and scope validator are not weakened.
- Provider attempts, schema recovery, window limit, output tokens, and global
  budget do not change.
- No recursive windowing or new Agent/pipeline is introduced.
- New safe outcomes contain only local IDs, locators, counts, versions, and
  allowlisted codes.
- Raw source, response, draft, prompt, reasoning, header, and credential
  remain absent from artifacts.

## Next Independent Task

The only next task is a separately authorized, fully offline implementation of
the selected coverage-only identifier signal with deterministic fixtures.

Expected implementation surface:

- trusted source-map segment typing in `models.py` and `markdown.py`;
- coverage-only classification and aggregation in `agents/miner.py` /
  `product.py`;
- one allowlisted outcome code and coverage-policy version/provenance update;
- focused source-map, intentional-empty, identifier-empty, historical-reader,
  cache-invalidation, privacy, and zero-call-change tests.

It must not include boundary changes, source-span output contracts, new
provider calls, prompt tuning, retry/token changes, P1, or another live run.

## Prohibited Follow-Ups

- Do not rerun the three dogfood inputs.
- Do not infer or reconstruct missing provider output.
- Do not relax exact-span, fragment, protected-token, or ownership checks.
- Do not add retry, output tokens, recursive splitting, or open loops.
- Do not implement candidate-span selection or a parallel Miner pipeline.
- Do not run DeepSeek, Tavily, benchmark, holdout, or consumed corpora.
- Do not advance P1, ten-case dogfood, package release, or public claims.
- Do not mark Miner P0 resolved from this diagnosis.

## Plan Accountability

- **Plan basis:** real-user Miner blockers must be addressed before P1 or
  broader dogfood, while strict fact boundaries and bounded execution remain
  unchanged.
- **Observed basis:** six local failures are visible, but PR has one
  schema-valid complete-empty identifier window that the existing coverage
  policy does not signal.
- **Judgment:** retain the current bounded-window architecture; add only a
  coverage-only identifier signal in a separate offline task.
- **Modify plan:** yes, record this diagnosis and make that implementation the
  sole next task. Miner P0 remains `improved_but_blocking`; Phase 3 and
  eligibility remain unchanged.
