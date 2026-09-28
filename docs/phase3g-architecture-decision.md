# Phase 3G-A/B architecture decision

## Decision

Adopt and freeze adaptive routing as a **dev benchmark baseline**, not as a
deployed product behavior:

- sources that fit safely in the verifier context and collapse to one short
  chunk should use a full-context Single Agent path;
- genuinely long sources should remain eligible for Retrieval-to-Judge;
- unavailable sources should keep the existing deterministic route.

Phase 3G-B implements this policy without a v3-derived character threshold.
The decision uses chunk count and the byte size of the exact serialized model
payload against a configured 32,768-byte budget. The consumed v3_zh holdout is
not used to tune a passing threshold and cannot be rerun as a formal gate.

The independent `phase3g_dev_zh` pack now provides 72 provisional pair cases
over short, medium, long, and unavailable sources. It preregisters a later
three-baseline live comparison, but no live run or quality conclusion is part
of this checkpoint.

## Evidence

The v3_zh sources are all short: available snapshots contain 35–127
characters, median 61, and at most five non-empty lines. Retrieval returned
either the entire snapshot as one candidate or nothing. Among 32 evidence
cases, all 22 hits were rank 1 and all ten misses had no candidate. There were
no partial or truncated contexts.

The ten Single-Agent-only wins exactly match those ten retrieval misses. Six
were `entailed`, three were `contradicted`, and one was
`partially_entailed`. Retrieval-to-Judge had only one exclusive win. This is
strong evidence that retrieval is harmful overhead for this short-source
regime.

The evidence does not justify replacing Retrieval-to-Judge everywhere. There
are no long documents in this consumed set, so its expected context-control
and evidence-localization benefits were never exercised.

## Questions resolved

1. **Chinese retrieval or Judge?** The performance gap against Single Agent is
   primarily Chinese retrieval/candidate generation: ten Single-Agent-correct
   cases never reached Judge. The absolute error set is mixed: ten retrieval
   misses, eight sufficient-context Judge errors, four deterministic router
   errors, and three annotation/translation ambiguity flags.
2. **Do short sources need retrieval?** Not for the observed 35–127-character
   snapshots. Retrieval added a failure point without reducing context.
3. **Where is Single Agent stronger?** All ten exclusive wins are in sources
   no longer than 67 characters and in substantive relations. Single Agent has
   no advantage on `not_checkable`; both baselines scored 0/12 there.
4. **Should routing be adaptive?** Yes, as a dev hypothesis. The route should
   depend on context fit and chunk count, not a holdout-derived literal
   127-character cutoff.
5. **Continue investing in Retrieval-to-Judge?** Only for long-source use
   cases, and only after a Chinese-aware retrieval dev benchmark shows value.
   The current evidence is sufficient to reject it as the default short-source
   path, but insufficient to reject it for long documents.
6. **What remains unknown about Multi-Agent value?** The pair benchmark
   supplies each claim. It cannot measure Claim Miner extraction recall,
   atomicity, deduplication, line mapping, downstream error propagation, or
   end-to-end Claim Miner plus Judge advantage.

## Implemented dev scope

Phase 3G-B completed the following work on new, non-holdout dev fixtures:

1. Added Chinese-aware candidate generation using character bigrams/trigrams,
   including mixed identifiers, numeric values, dates, and versions.
2. Added the one-chunk retrieval invariant: a safely bounded source
   should not disappear merely because lexical overlap is zero.
3. Defined adaptive routing from source availability, objective checkability,
   chunk count, serialized prompt size, and a configured context budget.
4. Added Chinese `not_checkable` positive and objective-slot counterexamples.
5. Built a 72-case stratified dev benchmark with genuine long-source
   distractors and preregistered all three live baselines.
6. Completed CJK-aware leakage checks against prior versions while preserving
   consumed data as historical inputs.

The remaining dev scope is three independently authorized live runs with zero
operational failures and overall plus stratum-level reporting. Those runs must
freeze code, prompt/schema, policy config, dataset, model, temperature, and
thinking mode before the first call.

This scope does not include Phase 4, a new production agent, prompt tuning on
v3_zh cases, or a v3_zh rerun.

## v4 requirement

A new blind holdout is required before another formal Phase 3 gate. Call it v4
only after the dev work above is complete and the code, prompt/schema, config,
retrieval policy, and routing policy are frozen.

Before creating v4:

- demonstrate the adaptive policy on independent multilingual dev fixtures;
- include long sources so Retrieval-to-Judge has a meaningful evaluation
  surface;
- complete annotation/translation QA without evaluation-model exposure;
- perform leakage checks against v1, consumed v2, dev, v3, and consumed v3_zh;
- preregister relation support, operational requirements, thresholds, call and
  token budgets, failure handling, and one-time execution;
- establish a clean checkpoint and repeatable offline test baseline.

The consumed v3_zh artifacts remain useful for diagnosis but cannot become v4
templates, tuning examples, or formal metrics. Phase 3 remains unpassed and
Phase 4 remains ineligible.
