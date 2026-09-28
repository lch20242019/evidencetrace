# Phase 3G-A v3_zh consumed-gate postmortem

## Status and boundary

This is an offline, post-hoc diagnostic over the consumed v3_zh holdout. It is
not a rerun, a new holdout result, or a replacement formal gate. The formal
status remains `consumed_metric_failure`; its macro-F1, thresholds, execution
status, gold, and canonical artifacts are unchanged. Real model calls in this
phase are zero.

The analysis started from checkpoint
`91eab156c8467ce542b420066bfac845fa7e04a1`. Before reading the data, the
analyzer checks the frozen SHA-256 of the four canonical artifacts, the
append-only execution record, the v3_zh gold file, and the frozen manifest.
The 72-case structured comparison is in
`eval_runs/phase3f_v3_zh_internal_gate/error_analysis.json`.

## Comparison outcome

| Outcome | Cases |
|---|---:|
| Both correct | 46 |
| Single Agent only correct | 10 |
| Retrieval-to-Judge only correct | 1 |
| Both wrong | 15 |

Single Agent was correct on 56/72 cases and Retrieval-to-Judge on 47/72.
These are diagnostic case accuracies, not newly computed formal gate metrics.
All ten Single-Agent-only wins were substantive cases for which retrieval
returned no candidate. The sole Retrieval-to-Judge-only win was
`v3_holdout_candidate_070`, where the no-candidate route returned
`not_in_source` and Single Agent over-credited one supported conjunct.

## Relation analysis

| Gold relation | Cases | Single correct | Retrieval-to-Judge correct | Main Retrieval-to-Judge failure |
|---|---:|---:|---:|---|
| `entailed` | 10 | 10 | 4 | Six complete retrieval misses became `not_in_source` |
| `partially_entailed` | 9 | 8 | 7 | One retrieval miss; one relation-boundary ambiguity |
| `contradicted` | 13 | 13 | 10 | Three complete retrieval misses |
| `not_in_source` | 16 | 13 | 14 | Two annotation/translation ambiguity flags |
| `source_unavailable` | 12 | 12 | 12 | None |
| `not_checkable` | 12 | 0 | 0 | Ten `not_in_source`, two `partially_entailed` |

The zero accuracy for `not_checkable` is shared by both live baselines. For
Retrieval-to-Judge, eight of those claims reached Judge with the complete
available snapshot and were still classified as a source relation; four were
routed before a Judge call. This is a Chinese checkability/router and Judge
problem, not evidence truncation.

Three cases are flagged for annotation or translation ambiguity without
changing gold:

- `v3_holdout_candidate_053`: the Chinese source states support for a `.venv`
  path containing a centralized project environment, while gold is
  `not_in_source`.
- `v3_holdout_candidate_058`: the Chinese source directly identifies Nushell
  `0.114.1`, while gold is `not_in_source`.
- `v3_holdout_candidate_061`: one conjunct is supported and the other conflicts,
  leaving a defensible boundary question between `partially_entailed` and
  `contradicted`.

These flags are diagnostic review findings only. The formal human gold remains
unchanged and continues to determine the consumed result.

## Retrieval diagnostics

Exactly 32 cases have substantive gold evidence. Literal evidence retrieval
produced:

| Metric | Result |
|---|---:|
| Recall@1 | 22/32 = 0.687500 |
| Recall@3 | 22/32 = 0.687500 |
| Recall@5 | 22/32 = 0.687500 |
| MRR | 0.687500 |
| Exact substring coverage | 22/32 = 0.687500 |
| Complete misses | 10 |
| Partial-context cases | 0 |
| Truncated-evidence cases | 0 |

Every hit was rank 1 and every miss had zero candidates. Candidate count never
exceeded one. Increasing top-k therefore cannot fix these ten failures under
the current retrieval representation; the issue is candidate generation and
Chinese lexical matching, not ranking depth.

The source pack is not a general retrieval benchmark. It repeats 24 short
fixtures across 72 cases. The 20 available snapshots contain 35–127 Unicode
characters, with median 61; each is one blank-line paragraph and at most five
non-empty lines. Recall@5 here is only a consumed-set pipeline diagnostic.

## Error taxonomy

All 25 Retrieval-to-Judge relation errors were assigned exactly one category:

| Category | Count | Share | Representative cases |
|---|---:|---:|---|
| `retrieval_miss` | 10 | 40% | 003, 005, 009, 018, 026 |
| `judge_error_with_sufficient_evidence` | 8 | 32% | 004, 013, 027, 033, 044 |
| `deterministic_router_error` | 4 | 16% | 019, 021, 038, 052 |
| `annotation_or_translation_ambiguity` | 3 | 12% | 053, 058, 061 |
| `retrieval_partial_context` | 0 | 0% | none |
| `evidence_span_selection_error` | 0 | 0% | none |
| `unsupported_other` / `unknown` | 0 | 0% | none |

The classification order and thresholds are preregistered in the offline
analysis script. A literal miss takes precedence over downstream Judge
speculation. The three ambiguity flags are explicit manual review findings,
not automatic relabeling.

## Contradiction context

Ten of 13 contradicted cases received Judge context and were classified
correctly. The three that received no context were the three contradiction
errors. The feature audit is lexical and per-case; it does not resolve entity
aliases.

| Feature | Relevant cases | Context received | Claim feature cases / all entered | Source feature cases / all entered |
|---|---:|---:|---:|---:|
| Numbers | 3 | 3 | 1 / 1 | 3 / 3 |
| Dates | 0 | 0 | 0 / 0 | 0 / 0 |
| Versions | 12 | 9 | 12 / 2 | 6 / 5 |
| Entities | 13 | 10 | 13 / 3 | 10 / 8 |
| Negations | 9 | 7 | 8 / 0 | 3 / 1 |

Claim-side conflict values are often expected not to occur verbatim in a
contradicting source, so low claim-value entry is not itself a retrieval
failure. The important structural result is that context was all-or-nothing:
when a candidate existed it was the full short snapshot; the three failed
contradictions received none.

## Source-length pattern

| Source stratum | Cases | Single correct | Retrieval-to-Judge correct |
|---|---:|---:|---:|
| Available, at most 50 characters | 15 | 15 | 10 |
| Available, 51–75 characters | 30 | 20 | 16 |
| Available, 76–127 characters | 15 | 9 | 9 |
| Source unavailable | 12 | 12 | 12 |

All ten Single-Agent-only wins used available sources of 35–67 characters.
The advantage is therefore concentrated in very short sources and in
`entailed`/`contradicted`/`partially_entailed` cases lost before Judge. It is
not a `not_checkable` advantage: both systems failed all 12 such cases.

## Conclusion

The incremental Retrieval-to-Judge deficit versus Single Agent is primarily a
Chinese candidate-generation failure: exactly ten cases that Single Agent
answered correctly were denied any Judge context. The absolute
Retrieval-to-Judge error set is broader, with twelve checkability-related
Judge/router errors and three ambiguity flags. Both layers need independent
dev work.

This dataset contains no long source. It supports avoiding retrieval for these
short snapshots, but it cannot show whether Retrieval-to-Judge helps on long
documents. It also supplies claims directly, so it says nothing about Claim
Miner extraction recall, atomicity, line mapping, or full Claim Miner plus
Judge Multi-Agent value.
