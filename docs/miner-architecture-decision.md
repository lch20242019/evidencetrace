# Miner Architecture Decision: Source-Mapped Bounded Windows

Date: 2026-07-27

Status: accepted and implemented; live verification found blocking coverage gaps

Implementation status: live verification complete / improved_but_blocking

Miner P0 status: `improved_but_blocking`

Phase 3 status: `completed_with_known_limitations`

`phase4_eligible=false`

## Decision

Adopt **B, deterministic source-mapped sentence/clause windowing**, as the
primary Miner architecture. Add only the minimum safety fallback from **D**:
keep local draft validity separate from paragraph coverage, and emit a
paragraph-level `needs_human` outcome whenever a window fails or protected
coverage is incomplete.

This is one product path:

```text
raw Markdown paragraph
-> deterministic model-visible text plus source map
-> bounded exact-substring windows
-> existing Miner contract, independently per window
-> strict Pydantic validation
-> exact/atomic/order/protected validation
-> accepted window claims
-> paragraph coverage aggregation
-> downstream claims plus explicit needs_human outcome
```

It is not a second Miner pipeline. It does not select option A, does not adopt
option C, and does not combine all four proposals.

| Required decision field | Record |
|---|---|
| Plan basis | The canonical v3.0 plan requires real-user blockers to be fixed before P1 or broader dogfood, while retaining exact substrings, protected qualifiers, typed contracts, bounded recovery, and failure isolation. |
| Observed basis | Contract v3 reduced Miner failures from 9 to 2, but ADR `p_0006` still failed paragraph-global protected-token validation and PR `p_0034` still produced 0 claims after a recovered schema attempt ended as `empty_content` with `finish_reason=length`. |
| Judgment | Adjust the Miner dispatch unit and status model. Keep the security boundary. |
| Modify plan | Yes. The canonical plan now makes this architecture the only next implementation task. |

Architecture completion does not resolve P0. Resolution still requires an
offline implementation and a separately authorized live recheck.

## Scope And Invariants

The selected design must preserve all of the following:

1. A model-authored claim is accepted only when it is a single contiguous,
   character-for-character substring of the exact `input.text` visible to the
   model.
2. Case, whitespace, punctuation, numbers, dates, versions, identifiers,
   negations, units, and comparison qualifiers are never locally repaired.
3. Model output cannot own file, paragraph, window, claim, line, citation,
   version, or provenance identifiers.
4. Pydantic remains strict and `additionalProperties=false`.
5. Only `ModelSchemaError` may receive the existing one schema-only recovery.
6. Scope, protected-token, order, overlap, local validation, transport, and
   budget failures are not retried.
7. No recursive splitting after provider failure and no
   while-until-success loop are permitted.
8. A failed window cannot delete accepted claims from another window.
9. Missing protected coverage can never yield a paragraph status of
   `complete`.
10. A claim-bearing paragraph that cannot be mined must produce a typed,
    source-located `needs_human` outcome instead of silently appearing as zero
    findings.

## Current Architecture

### Current data flow

```text
paragraph raw Markdown
-> markdown-it inline tokens
-> markdown._token_plain_text
   - keeps text and non-URL inline code
   - removes link destinations, images, HTML, footnote markers, and bare URLs
   - replaces line breaks with spaces
   - collapses all whitespace
-> MarkdownParagraph.plain_text
-> ProductAuditPipeline._mine
-> MinerInput(text=plain_text, paragraph line range, paragraph citations)
-> ClaimMinerAgent.mine
-> OpenAICompatibleClient JSON-object request
-> LiveMinerDraftOutput Pydantic validation
-> ClaimMinerAgent._validate_drafts
   - every draft must be a complete fragment
   - all paragraph protected tokens must occur in the aggregate draft output
   - all drafts must have one unique monotonic exact mapping
-> local AtomicClaim identity, lines, and citations
-> repeated _validate_scope
-> paragraph claims, or one exception for the entire paragraph
-> ProductAuditPipeline catches the exception, records a failed trace event,
   and continues with the next paragraph
```

### Direct answers to the architecture questions

1. **Does one invalid draft discard the other drafts in a paragraph?**

   Yes. `_validate_drafts` raises before `MinerOutput` is constructed.
   `_unique_monotonic_offsets` also validates the complete tuple. Any invalid
   fragment, missing protected token, non-source span, ambiguous span,
   out-of-order span, or overlap prevents every draft in that model response
   from becoming a canonical claim.

2. **Is the protected-token guard paragraph-global or claim-local?**

   It is paragraph-global. `_protected_tokens(input_data.text)` scans the
   entire model-visible paragraph. The result is compared with the union of
   tokens in every returned draft. It is neither occurrence-aware nor scoped
   to one claim span.

3. **Are claim validity and paragraph coverage conflated?**

   Yes. Exactness and fragment completeness are claim-validity questions.
   Whether every protected fact in a paragraph was extracted is a coverage
   question. Both currently raise the same paragraph-level exception and
   discard the complete response.

4. **What are the PR paragraph request boundaries?**

   Frozen PR `p_0034` has:

   - raw Markdown: 1,616 characters;
   - model-visible `plain_text`: 1,258 characters;
   - resolved citation URLs: 4;
   - `LiveMinerDraftOutput` compact JSON Schema: 1,555 characters;
   - complete compact user message: 3,790 characters;
   - configured output limit: 2,048 tokens;
   - provider adapter maximum configurable output: 8,192 tokens;
   - public product global provider-attempt ceiling: 304;
   - schema recovery: one retry, so at most two provider attempts for the
     logical dispatch.

   The current sentence-boundary function deterministically identifies eight
   exact fragments in this paragraph, with lengths
   `114, 145, 126, 142, 154, 131, 331, 108`. The longest is 331 characters.

   Actual input, output, and reasoning token counts were not persisted and
   remain unknown.

5. **Why did `finish_reason=length` become `empty_content`?**

   The adapter checks `content is None or blank` before it checks
   `finish_reason == "length"`. Therefore a response with empty content and a
   safe `length` finish reason is classified as `empty_content`, while the
   finish reason remains available in safe telemetry. A non-empty response
   with `length` would instead be classified as `finish_reason`.

   The original provider content was intentionally not stored. It is unknown
   whether reasoning consumed the output budget, whether JSON was ever
   started, or what either attempt tried to return.

6. **Can a paragraph be split safely today?**

   Text-level splitting is feasible: `_fragments` already returns trimmed
   exact substrings of the model-visible text, and the PR paragraph becomes
   eight bounded fragments. End-to-end source mapping is not yet sufficient:
   `_token_plain_text` flattens line breaks and exposes no raw-to-model-text
   offset map. `_fragment_lines` therefore cannot recover precise lines for a
   multi-line parser-produced `plain_text`.

   The implementation must first create a trusted model-visible source map.
   It must not infer line numbers by searching normalized text after the fact.
   Existing paragraph citation URLs can be inherited by every window without
   giving citation ownership to the model; finer citation binding is outside
   this P0 decision.

7. **Can the current architecture emit a paragraph-level `needs_human`
   finding?**

   Not as a canonical user-facing finding. It can set the document to
   `partial` and store a failed Miner trace event with a paragraph ID. However,
   `AuditArtifact` contains no paragraph mining outcome, SARIF is generated
   only from claim verdicts, and the terminal can still say that no checkable
   claims were found. PR `p_0034` therefore produced an empty SARIF despite a
   typed Miner failure.

### Basis for these answers

| Finding | Plan basis | Observed basis | Judgment | Modify plan |
|---|---|---|---|---|
| Paragraph-wide rejection is too coarse | Per-claim and per-paragraph failures must be isolated. | `_validate_drafts` raises before any draft is assembled; PR and ADR each lost a complete claim-bearing paragraph. | Adjust. | Yes. |
| The protected guard has two responsibilities | Key qualifiers cannot be dropped, but valid exact claims must not be discarded unnecessarily. | The guard compares the entire paragraph token set with the aggregate output token set. ADR `p_0006` failed this check after schema recovery. | Split hard window validity from paragraph coverage. | Yes. |
| Token/retry tuning is not a diagnosis | Recovery must remain finite and failures must not be hidden by repeated calls. | PR already used the allowed schema retry and still ended `length`/`empty_content`; raw response is unavailable. | Do not tune token or retry in this decision. | No. |
| A partial document needs a located mining outcome | Real users must see failed work, not only a process exit code. | PR had 0 claims and 0 SARIF results even though the product trace recorded a failed changed paragraph. | Add a P0 paragraph outcome; broader Policy/SARIF semantics remain P1. | Yes. |

## Two Confirmed Failure Paths

### PR #871: bounded-output failure becomes paragraph loss

Known path:

```text
p_0034 plain_text, 1,258 characters
-> one Miner logical dispatch
-> first schema failure, category not persisted
-> one allowed schema recovery
-> final response content absent or blank
-> safe finish_reason=length
-> ModelSchemaError(empty_content)
-> ProductAuditPipeline records miner_schema_empty_content
-> no MinerOutput exists
-> 0 claims from the only selected paragraph
-> empty policy/SARIF result, document partial
```

The model-visible paragraph contains fork permissions, `pull_request`,
`pull_request_target`, write access, and a security warning. None reached the
Coordinator or Judge.

Windowing addresses the structural blast radius. The same current sentence
boundary policy produces eight independent exact windows, so one
length/schema failure cannot erase the other seven outcomes. This is a design
hypothesis until a later authorized live run; it is not a claim that the
provider will necessarily succeed.

### ADR: paragraph-global coverage becomes claim rejection

Known path:

```text
p_0006 plain_text, 301 characters
-> one Miner logical dispatch
-> one schema recovery
-> final schema-valid LiveMinerDraftOutput
-> _validate_drafts scans protected tokens across the full paragraph
-> aggregate draft tokens miss at least one protected token
-> MinerScopeError(missing_protected_token)
-> no offsets or canonical claims are returned
-> chosen option and rationale both disappear
```

The current sentence boundary policy produces two exact windows of 99 and 201
characters. It is known that paragraph-global coverage failed. It is unknown
which token was absent, what drafts were present, or whether one draft was
otherwise valid, because model content was not persisted.

The selected architecture retains a hard protected-token check inside each
bounded modifier-bearing window. Protected occurrences outside an accepted
window become paragraph coverage signals. Missing coverage prevents
`complete`, but it does not delete claims already accepted from independent
windows.

## Known Facts And Unknowns

### Known

- Contract v3 reached the real provider request.
- README completed 10 of 10 Miner dispatches.
- ADR completed 5 of 6; `p_0006` ended
  `miner_scope_missing_protected_token`.
- PR completed 0 of 1; its final attempt ended
  `miner_schema_empty_content` with `finish_reason=length`.
- PR used two attempts and ADR `p_0006` used two attempts.
- Local scope errors do not retry.
- Schema errors may retry exactly once.
- One paragraph exception currently discards its complete response.
- Parser-produced `plain_text` has no raw-to-model-text character map.
- Product artifacts can mark the document partial and record a paragraph trace
  code, but cannot produce a paragraph-level audit/SARIF finding.

### Unknown

- The first-attempt schema category for the ADR and PR dispatches.
- Either attempt's provider content or reasoning.
- Actual prompt, completion, and reasoning token counts for these dispatches.
- Which ADR protected occurrence was omitted.
- Whether the ADR response contained any independently valid exact draft.
- Whether smaller windows will eliminate `finish_reason=length` for this
  provider.
- The live quality and stability of the selected architecture.

No implementation may turn these unknowns into inferred facts.

## Options

| Criterion | A. Paragraph text plus token/recovery tuning | B. Deterministic bounded windows | C. Deterministic candidate spans, model selects IDs | D. Separate draft validity, coverage, and needs-human |
|---|---|---|---|---|
| PR length/empty-content | Does not structurally solve it. A larger output budget may move the failure without bounding response complexity. | Directly reduces each request's extraction surface; one failed window does not erase the paragraph. Live success remains unverified. | Output can be small, but the candidate list may still require batching. | Makes the failure visible but does not prevent the original long dispatch. |
| ADR paragraph-global protected guard | No. | Narrows protected scope to a sentence/clause window, but pure B still needs coverage state. | Can bind protected occurrences to candidate spans, but still needs coverage semantics. | Directly separates claim validity from paragraph coverage. |
| Exact substring safety | Unchanged. | Preserved because every window and claim remains an exact `input.text` substring. | Structurally strongest if candidate mapping is correct. | Preserved if valid drafts continue through the existing exact validator. |
| Atomic claim quality | Same model burden and long-context failure mode. | Smaller semantic units reduce over-merged output; unsafe fragments still fail. | Depends heavily on deterministic candidate recall and boundary quality. | Does not improve model extraction granularity by itself. |
| Line and citation mapping | Existing coarse paragraph mapping remains. | Requires a trusted model-visible source map; paragraph citations can be inherited locally. | Requires the same source map and candidate provenance. | Reuses current mapping, including current multi-line limitation. |
| Failure isolation | Paragraph only. | Window-level. | Candidate or batch-level. | Draft-level where independently provable, plus paragraph-level fallback. |
| Calls and tokens | Fewest calls, potentially largest outputs; retry tuning raises cost. | More logical calls, smaller inputs/outputs, deterministically capped. | Moderate calls but larger local candidate construction and schema migration. | Same calls as current, with small artifact overhead. |
| Controller budget | Existing coarse global ceiling only. | Needs an explicit Miner window counter and at most two attempts per window. | Needs candidate/batch ceilings and potentially more complex accounting. | Reuses current call ceiling; adds no calls. |
| Schema/cache migration | Small max-token provenance change if tuning occurs. | New window/coverage policy versions; Miner contract changes only if output shape changes. | Major schema, prompt, cache, and provenance migration. | Product artifact/status version change; Miner schema can remain. |
| Implementation complexity | Low. | Medium. Parser source mapping is the largest required change. | High. Candidate generation and selection become a new extraction contract. | Medium-low. State separation is local but does not solve PR response size. |
| Regression risk | High recurrence risk despite small diff. | Medium; boundary and line-map fixtures can contain it. | High recall and atomicity regression risk. | Low-medium, but operational PR failure remains. |
| Parallel pipeline risk | None. | None if it calls the existing Miner from the existing product loop. | High: it can become a second span-extraction system beside the current Miner. | None. |
| Three-case overfitting | Token changes after one `length` event would be weakly justified. | General sentence/clause windows apply to arbitrary Markdown; no case IDs or phrases. | General in principle, but large machinery is not yet justified by three cases. | General state semantics, not case-specific. |
| Reversibility | Easy, but offers little structural value. | High with a versioned window policy and retained paragraph fallback. | Low-medium because the model contract and candidate representation change substantially. | High; optional fields and a versioned product artifact can preserve readers. |

### Option accountability

| Option | Plan basis | Observed basis | Judgment | Modify plan |
|---|---|---|---|---|
| A | Recovery must remain finite and engineering changes must follow confirmed causes. | The PR already used the one allowed recovery; raw content and actual token split are unknown. | Reject as the primary direction. | No token or retry change is added to the plan. |
| B | Reuse the parser, Miner, Controller, typed contract, and hard budget while improving failure isolation. | The PR model-visible text deterministically forms eight exact fragments, while one paragraph dispatch lost all content. | Select as the primary architecture. | Yes. |
| C | Exactness is mandatory, but the project must not create a parallel pipeline without evidence. | Current evidence shows dispatch granularity and coverage-state problems, not deterministic candidate-recall quality. | Defer. | No. |
| D | A bad unit must not silently delete otherwise usable work. | ADR and PR each became whole-paragraph loss, and current audit/SARIF cannot represent a paragraph mining failure. | Select only the minimum paragraph coverage and `needs_human` fallback around B. | Yes. |

## Selected Architecture In Detail

### 1. Deterministic model-visible source map

The parser continues to own Markdown interpretation. It must produce, for each
paragraph:

- the existing raw source span;
- the exact model-visible text;
- deterministic model-text ranges mapped to original line ranges;
- heading path;
- paragraph citation IDs and resolved URLs.

The map may conservatively cover a source line range, but it must never invent
a more precise line than the parser can prove. Existing `plain_text` readers
must remain compatible.

### 2. Bounded windows

Windows are created before any model call:

- use sentence terminators and explicit semicolon boundaries already supported
  by the current deterministic fragment policy;
- retain each window as an exact substring of model-visible paragraph text;
- never hard-cut inside a token, identifier, number, date, version, inline-code
  value, or citation label;
- do not split on arbitrary case-specific words;
- if one unsplittable fragment exceeds the configured serialized request
  bound, emit `needs_human` without calling the model for that fragment;
- retain source order and prohibit overlapping windows.

The first implementation should dispatch one deterministic fragment per
window. It must not add recursive re-windowing after a failed call.

### 3. Existing Miner, independent dispatch

Every window uses the existing `ClaimMinerAgent`, `ModelClient`, Pydantic
contract, schema-only recovery, and local identity assembly. The model still
returns semantic draft fields, not local IDs or locators.

Schema failure invalidates that window because raw failed output is not
available for safe salvage. Other windows continue.

### 4. Draft validity and protected coverage

The implementation separates three checks:

1. **Individual draft validity:** nonblank, atomic fragment, exact substring,
   and unique mapping inside its window.
2. **Collection validity:** source order and no overlap among accepted drafts.
3. **Coverage validity:** protected token occurrences in the window and then
   the paragraph are accounted for by accepted source spans.

Protected tokens retain a hard safety role. A protected occurrence in the
same deterministic modifier-bearing window cannot be omitted while that
window is declared complete. An uncovered occurrence prevents that window and
paragraph from being complete.

Independently exact drafts may be retained in the window outcome when another
draft is invalid, but they may enter the Judge path only when their window's
hard qualifier and collection checks pass. Otherwise they remain
source-located `needs_human` candidates. This preserves evidence without
silently certifying incomplete extraction.

### 5. Paragraph aggregation

The Controller assembles window outcomes in source order:

- claims from complete windows enter the existing Coordinator/Judge path;
- a failed window does not remove claims from another complete window;
- any schema, scope, coverage, budget, or local failure makes paragraph
  coverage `partial` or `needs_human`;
- zero claims plus claim-bearing or failed windows must be rendered as an
  explicit paragraph-level `needs_human` outcome;
- an intentionally empty non-claim window remains a successful empty result
  and must not become a false warning.

The deterministic distinction between an intentional empty result and an
operational failure is essential.

## New Contracts

Names below are architectural, not committed APIs.

### Input

```text
MinerWindowInput
  paragraph_id            locally owned
  window_id               locally owned
  text                    exact model-visible substring
  paragraph_char_start    locally owned
  paragraph_char_end      locally owned
  source_line_start       locally owned
  source_line_end         locally owned
  file                    locally owned
  heading_path            locally owned
  citation_urls           locally owned
  window_policy_version   locally owned
```

Only `text`, heading context, and citations are model-visible as data. IDs and
source locations are attached locally and cannot be overridden.

### Output

```text
MinerWindowOutcome
  window_id
  status:
    complete
    partial
    agent_error
    budget_exhausted
    needs_human
  accepted_claims
  withheld_exact_draft_count
  allowlisted_failure_codes
  protected_occurrences_total
  protected_occurrences_covered
  provider_attempts
  schema_retry
  finish_reason
```

No model content, prompt, reasoning, header, credential, source text, or
rejected draft text is written to the safe product artifact.

```text
ParagraphMiningOutcome
  paragraph_id
  source
  status:
    complete
    partial
    needs_human
  window_count
  completed_window_count
  accepted_claim_ids
  allowlisted_reason_codes
```

The product artifact and terminal must expose this outcome. Broader
`needs_human` Policy/SARIF semantics remain the separately deferred P1 task;
the P0 implementation must at least prevent a failed paragraph from appearing
as a normal zero-claim result.

## Budget And Failure Isolation

1. Add a Controller-owned Miner logical-window counter distinct from canonical
   claim count.
2. The first default Miner window ceiling should reuse the existing
   `max_changed_claims` numeric bound of 30, but record it under an explicit
   `miner_window_limit`; it must not continue conflating paragraph count,
   window count, and claim count in artifacts.
3. Each window has one initial provider attempt and at most one existing
   schema-only recovery. Therefore the default Miner provider-attempt ceiling
   is at most 60.
4. The existing global provider ceiling remains an outer bound.
5. Keep the current 2,048 output-token request for the first windowed
   implementation. This decision does not use a token increase as its fix.
6. Exhausted window budget creates source-located `needs_human` outcomes for
   unprocessed windows. It does not silently truncate the document.
7. Schema failure stops one window after the existing recovery. Scope,
   protected-token, order, overlap, local validation, transport, and budget
   errors do not retry.
8. No window may schedule another window. Only the deterministic Controller
   creates the finite window list before dispatch.

## Migration, Cache, And Provenance

- Add `MINER_WINDOW_POLICY_VERSION` and
  `MINER_COVERAGE_POLICY_VERSION`.
- Bump `MINER_DRAFT_CONTRACT_VERSION` only if the provider-visible output
  schema changes. A changed input/window policy must still invalidate cache
  provenance even if the draft schema remains identical.
- Include both new policy versions, window limits, serialized request bound,
  model max tokens, provider contract, and schema-recovery policy in the
  cache key.
- Never reuse paragraph-level v3 model results as window-level results.
- Version `ProductRunArtifact` for paragraph/window outcomes and retain a
  read path for `bounded-multi-agent-product-v1`.
- Record policy versions and limits in safe run provenance.
- The current public product path did not create or read a model cache during
  Round 1. Cache hit/miss behavior remains unverified and must not be claimed
  as already implemented.

## Expected Implementation Surface

The next task should be limited to:

| File or function | Expected change |
|---|---|
| `src/evidencetrace/models.py` | Add backward-compatible trusted model-text/source-map data needed for window line ranges. |
| `src/evidencetrace/markdown.py` `_token_plain_text`, `_paragraphs` | Produce deterministic model-visible text segments and preserve source-line mapping without changing citation ownership. |
| `src/evidencetrace/audit_models.py` | Add typed window/coverage contracts and policy versions; change provider schema only if required. |
| `src/evidencetrace/agents/miner.py` `_fragments`, `_validate_drafts`, `mine` | Reuse exact matching for one window, separate individual/collection validity from occurrence-based coverage, and retain strict fail-closed behavior. |
| `src/evidencetrace/product.py` `_mine`, budget and artifact models | Build the finite window list, enforce budgets, aggregate outcomes, preserve other windows, and expose paragraph `needs_human`. |
| `src/evidencetrace/cache.py` | Include window and coverage policy versions in cache keys. |
| `src/evidencetrace/render.py` | Show source-located Miner `needs_human` outcomes in terminal/audit Markdown. |

The P0 task must not change Judge, Coordinator, Scout, Challenger, discovery,
evidence retrieval, relation policy, or general provider behavior. A minimal
SARIF representation for a failed mining paragraph may be separately reviewed
with P1; it is not permission to redesign Policy in the P0 implementation.

## Expected Tests

Deterministic fixtures must cover:

1. model-visible windows are exact, ordered, non-overlapping substrings;
2. raw-to-window line mapping across soft breaks, links, and inline code;
3. the synthetic PR shape becomes multiple bounded windows without using its
   text, entity, case ID, or numeric constants;
4. a schema failure in one window preserves claims from other windows;
5. one invalid draft does not remove independently valid drafts when
   collection and protected checks remain provable;
6. missing protected tokens prevent window and paragraph completion;
7. protected tokens in another window do not invalidate a complete claim;
8. qualifier omission inside the same modifier-bearing window remains
   fail-closed;
9. out-of-order, overlapping, ambiguous, non-source, rewritten, and
   non-contiguous drafts remain rejected;
10. intentional empty non-claim windows remain valid;
11. unsplittable oversized windows and exhausted budgets become located
    `needs_human` outcomes with zero extra calls;
12. one initial call and one schema-only retry remain the hard per-window
    ceiling;
13. safe artifacts contain no model content, prompt, source text, header, or
    credential;
14. historical v1 product artifacts remain readable;
15. cache/provenance changes when either new policy version changes.

No live model is required for implementation acceptance.

## Rejected Options

### A: token or retry tuning

Rejected as the primary fix. It does not separate paragraph coverage from
claim validity, does not improve isolation, and cannot be justified from one
provider `length` event whose raw response and token split are unknown.
Recovery remains one.

### C: deterministic candidate-span selection

Deferred, not selected. It offers strong exactness, but candidate recall and
atomicity would become a new deterministic extraction problem. It requires a
larger schema and provenance migration and risks creating a parallel Miner
pipeline. Three dogfood cases do not justify that cost.

C may be reconsidered only if bounded windows still show systematic
non-source or qualifier failures on broader real dogfood after P0 is safely
implemented. It must not be implemented as part of the next task.

### D alone

Rejected as the primary fix. It correctly prevents silent loss and resolves
the paragraph-global status conflation, but it leaves the PR's one large
provider response unchanged. Its paragraph outcome is retained only as the
minimum fallback around B.

## Implementation Boundary

The next and only implementation task is:

```text
Implement source-mapped bounded Miner windows plus paragraph coverage outcomes
with deterministic fixtures, while preserving all current strict validators
and bounded recovery.
```

Explicitly deferred:

- real DeepSeek or Tavily calls;
- rerunning the three dogfood cases;
- P1 full telemetry;
- Policy and SARIF semantic redesign;
- Judge evidence tuning;
- Scout, Challenger, or discovery work;
- ten-case dogfood;
- PyPI, `uvx`, public demo, or release preparation;
- benchmarks, holdouts, or evaluation phases.

## Rollback Conditions

Do not authorize a live recheck, and revert to the v3 dispatch behavior while
retaining explicit paragraph failure visibility, if any of these occur:

1. a window cannot be proven to be an exact model-visible substring;
2. line mapping can point outside the original paragraph source span;
3. qualifier omission can reach a `complete` claim;
4. calls can exceed the precomputed window and recovery ceiling;
5. one window can recursively schedule more calls;
6. historical product artifacts become unreadable;
7. deterministic fixtures show systematic claim loss from the window policy
   without a `needs_human` outcome.

Rollback does not permit removal of exact-span or protected-token safety.

## Offline Implementation Record

The selected architecture is now implemented through its P0 public visibility
boundary:

1. The Markdown parser builds model-visible text and its trusted source map in
   one deterministic pass while preserving the established `plain_text`
   behavior.
2. `plan_miner_windows(...)` creates a finite, source-ordered list of exact,
   non-overlapping sentence/clause windows before any provider call.
3. The Controller owns a separate 30-window default budget and a hard maximum
   of two provider attempts per dispatched window, while the outer provider
   ceiling remains in force.
4. Individual draft validity, collection validity, window hard-qualifier
   coverage, and paragraph occurrence coverage are separate checks. A failed
   window cannot delete accepted claims from another complete window.
5. Safe `MinerWindowOutcome` and `ParagraphMiningOutcome` contracts retain
   only local identifiers, trusted locations, counts, allowlisted codes, and
   bounded attempt metadata.
6. Canonical `audit.json` now carries a backward-compatible
   `paragraph_mining_outcomes` collection. Terminal output, `audit.md`, and
   SARIF derive paragraph visibility only from that canonical collection.
7. A non-complete paragraph yields one source-located extraction warning in
   each public renderer. It is not assigned a claim relation and does not use
   the `not_in_source` SARIF rule. Valid claim verdicts from complete windows
   remain present beside a paragraph-level mining issue.

Offline verification passed 258 focused/regression tests and 581
boundary-compliant tests while excluding the seven documented
consumed-corpus loader files. Changed-production mypy, compileall, scoped Ruff
lint/format checks, diff validation, and privacy checks passed. Historical
canonical audits without the new optional collection remain readable, and the
offline demo remains byte-stable.

No DeepSeek, Tavily, dogfood, holdout, or consumed evaluation was run. The
provider-visible Miner contract remains `live-miner-draft-v3`; exact substring,
protected-token, retry, budget, and local ownership rules were not relaxed.
The only next task is a separately authorized live recheck on the original
three fixed dogfood inputs. Until that evidence exists, Miner P0 remains
`improved_but_blocking`; P1 telemetry/Policy work, ten-case dogfood, and release
preparation remain deferred.

## Confidence

Overall decision confidence: **medium-high**.

- High confidence: the current paragraph-wide discard, paragraph-global
  protected guard, missing paragraph finding, request limits, and
  `empty_content` precedence are directly proven by code and artifacts.
- Medium-high confidence: bounded windows materially reduce failure blast
  radius and request/output complexity; the frozen PR text already separates
  into eight general sentence windows.
- Medium confidence: the real provider will avoid `finish_reason=length` and
  produce adequate atomic claims after windowing. That requires later live
  evidence and is not asserted here.

The decision is intentionally reversible and makes unknown provider behavior
visible rather than filling it in with guesses.

## 2026-07-27 Live Recheck Record

The separately authorized live recheck ran each of the three frozen Round 1
inputs exactly once with `deepseek-v4-flash`. It did not change code, prompts,
schemas, validators, budgets, retry, model settings, or inputs, and it made no
Tavily request.

Observed architecture behavior:

- The README planned 11 windows; all 11 completed and produced 4 exact claims.
- ADR `p_0006` planned the expected exact windows of 99 and 201 characters.
  Overall, 5 of 8 ADR windows completed and produced 6 claims. Three
  non-complete paragraphs were visible as three ET2001 findings.
- PR `p_0034` planned the expected 8 exact windows of
  114/145/126/142/154/131/331/108 characters. Five windows completed and
  produced 4 claims, compared with 0 claims before this architecture. The
  remaining three failures were aggregated into one source-located ET2001
  finding while accepted claims remained in the audit.
- Every accepted claim was an ordered, non-overlapping, unique exact substring
  of its model-visible window. Every locator remained inside its paragraph,
  complete-window protected coverage was satisfied, and calls remained below
  the precomputed window and provider-attempt limits.
- Canonical audit, terminal, Markdown, and SARIF represented each non-complete
  paragraph exactly once. ET2001 remained a warning without a relation and did
  not masquerade as `not_in_source`.

No rollback condition was observed. Source mapping, bounded planning, failure
isolation, public visibility, and budget enforcement worked as designed.
However, the live quality gate for Miner P0 was not met: ADR still omitted the
literal chosen-option statement, and PR still omitted the
`pull_request_target` transition, its explicit write-access consequence, and
the complete security warning/mitigation.

Decision record:

- **Plan basis:** claim-bearing paragraphs must not disappear silently, strict
  fact boundaries cannot be relaxed, and all provider work must remain
  bounded.
- **Observed basis:** PR recovered 4 exact claims and a visible partial
  paragraph, but ADR and PR retained core coverage gaps despite structurally
  correct windows.
- **Judgment:** keep the selected architecture; classify Miner P0 as
  `improved_but_blocking`, not resolved and not rollback-required.
- **Plan change:** record this live evidence only. The next task is a separate
  offline bounded-window coverage failure diagnosis. Do not tune or rerun the
  three inputs, and continue to defer P1, ten-case dogfood, and release work.

Full commands, per-case telemetry, core-claim review, artifact hashes, and
side-effect checks are recorded in
`docs/dogfood/round_1_2026-07-27.md` under
“Miner Architecture Live Recheck”.

## 2026-07-27 Coverage Failure Diagnosis Record

The six remaining failed windows were diagnosed offline against their frozen
product artifacts, deterministic window plans, trusted source maps, and
current validation code. No provider, search service, CLI dogfood, benchmark,
or consumed corpus was run.

Confirmed failure distribution:

- ADR: two protected-token hard-coverage failures and one invalid-fragment
  failure.
- PR: two protected-token hard-coverage failures and one non-source-span
  failure.
- All six passed strict JSON/Pydantic validation first, used one provider
  attempt, did not recover, and ended with safe `finish_reason=stop`.
- No provider/schema contract failure occurred among the six.

The diagnosis also found three deterministic limitations outside the typed
failure count:

1. PR's explicit `pull_request_target` switch is in a schema-valid
   complete-empty window with no protected occurrence, so current coverage
   policy emits no window signal.
2. The sentence planner splits a dotted abbreviation, leaving one accepted
   fragment with an unmatched opening parenthesis and the next failed window
   with the unmatched close. The obvious merged span is 463 characters,
   above the current 384-character bound, so a safe replacement split is not
   yet proven.
3. Navigation phrase `more details` is classified as a hard comparative
   occurrence. Separately, failed outcome assembly flattens the safe covered
   count to zero, so artifacts cannot identify the omitted occurrence.

The selected minimum recommendation is option C, narrowed to a
**parser-owned inline-code identifier coverage-only signal**. A schema-valid
empty result over trusted inline-code identifier spans should become an
allowlisted `needs_human` coverage outcome rather than `complete`. This signal
must not generate or accept a claim, repair text, weaken the protected guard,
or change calls, tokens, retry, window bounds, or local ownership.

Option B is deferred because suppressing the proven abbreviation boundary
would create a 463-character span without a proven safe bounded replacement.
Option D remains deferred because one non-source event does not satisfy the
documented prerequisite for a candidate-span/offset contract migration.
Existing `needs_human` alone is insufficient because it misses the
complete-empty core window.

Decision accountability:

- **Plan basis:** prevent silent claim-bearing omissions before P1 or broader
  dogfood, without weakening exact/protected safety.
- **Observed basis:** explicit failures are visible, but PR has a deterministic
  complete-empty identifier coverage blind spot.
- **Judgment:** retain the architecture and implement only the coverage-only
  identifier signal in a separately authorized offline task.
- **Plan change:** record this task as the sole next implementation. Miner P0
  remains `improved_but_blocking`; Phase 3 remains
  `completed_with_known_limitations`; `phase4_eligible=false`.

The complete six-window evidence table, core-fact mapping, source locations,
window hashes, validation lines, confirmed/inferred/unknown separation, and
option comparison are in `docs/miner-coverage-failure-diagnosis.md`.

## 2026-07-27 Coverage-Only Identifier Signal Implementation Record

The separately authorized offline implementation adds only the narrow
coverage signal selected by the diagnosis. While building the trusted source
map, the Markdown parser now marks an inline-code span as an identifier only
when it is nonblank, has no whitespace or surrounding trim, is not URL-like,
starts with an ASCII letter, and contains one or more identifier separators
(`.`, `_`, `:`, or `-`) joining nonempty ASCII alphanumeric components.
Ordinary text, link labels, URLs, blank code spans, and natural-language code
phrases do not receive this trusted kind. Model-visible text, offsets, source
lines, paragraph ownership, and the provider request remain unchanged.

The deterministic window plan carries only the count of intersecting trusted
identifier spans. After a Miner response has passed JSON/Pydantic validation
and every existing scope, protected-token, fragment, budget, and recovery
path, a successful `claims=[]` result in such a window becomes
`needs_human` with the single allowlisted reason
`miner_coverage_empty_identifier_window`. Existing failures take precedence.
No claim, relation, draft text, identifier text, or downstream Agent task is
created. Paragraph aggregation therefore remains `partial` when another
window supplied accepted claims and becomes `needs_human` when none did.
Canonical audit continues to drive the existing terminal, Markdown, and
ET2001 output; no renderer, SARIF rule, or policy meaning changed.

Only `MINER_COVERAGE_POLICY_VERSION` changed, from
`miner-coverage-policy-v1` to `miner-coverage-policy-v2`; that version already
participates in cache-key and safe provenance construction. Historical v1
outcomes and provenance remain readable. `MINER_DRAFT_CONTRACT_VERSION`
remains `live-miner-draft-v3`, and `MINER_WINDOW_POLICY_VERSION` remains
`miner-window-policy-v1`. Calls, attempts, schema recovery, window budget,
provider schema, exact/order/atomic/protected validators, and ModelClient are
unchanged.

Offline verification passed 165 focused tests and 601 boundary-compliant
tests while excluding the same seven consumed-corpus loader files.
Changed-production mypy, scoped Ruff, compileall, diff checking, and privacy
checks passed. Two credential-free offline demo runs were byte-identical with
SHA-256
`d238b46540f2241b995e59a26d28495eb4d61dd1fd1501b76d23516f3b4cbbe8`.

This is a visibility fix, not an extraction fix. It does not prove that the
real provider will return a claim for `pull_request_target`, improve recall,
or repair abbreviation boundaries, protected-comparator classification, or
provider output. No live dogfood was rerun. Miner P0 remains
`improved_but_blocking`; Phase 3 remains
`completed_with_known_limitations`; `phase4_eligible=false`. A separate plan
review is required before any live run, P1 work, broader dogfood, or release
work.
