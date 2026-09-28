# Full-document Claim Extraction Benchmark Design

Status: design only; not implemented in Phase 3E.2.

## Purpose

The existing pair-level evaluation supplies one atomic claim in every case. It
can measure retrieval and claim-source relation judgement, but it cannot
measure Claim Miner quality or demonstrate a Multi-Agent advantage. A future
full-document benchmark must begin with complete Markdown documents and score
the extraction and verification stages separately and end to end.

## Required inputs and gold annotations

Each benchmark document must preserve immutable content and provenance hashes
and contain reviewer-approved annotations for:

- every checkable and explicitly not-checkable gold claim;
- the exact contiguous claim span and character offsets in the Markdown;
- source file, heading path, paragraph identity, and start/end lines;
- the minimum atomic claim count for each source sentence or paragraph;
- claim type, checkability, citation scope, and protected factual tokens;
- gold relation and exact source evidence span for each atomic claim; and
- unavailable-source status where verification cannot run.

Dev and holdout documents must be physically or logically separated. Holdout
content, annotations, code, prompts, retrieval settings, and model identifiers
must be frozen before a one-time evaluation. Model proposals cannot be called
human gold.

## Claim matching and extraction metrics

Predicted and gold claims require a documented one-to-one matching rule based
on exact spans first and bounded token overlap only as an explicit secondary
analysis. One prediction cannot satisfy multiple gold claims. The report must
include:

- claim precision, recall, and F1;
- missed-claim and spurious-claim counts;
- atomicity error rate for over-merged and over-split claims;
- exact and token-overlap claim-span quality;
- line-mapping accuracy for both start and end lines; and
- protected-token retention by type.

Documents with no gold claims must remain in the denominator needed to measure
spurious extraction rather than being silently skipped.

## Verification and end-to-end metrics

Verification uses the same relation taxonomy, evidence-substring guard, and
retrieval metrics as pair evaluation. It must report relation macro-F1,
per-label precision/recall/F1 and support, contradiction recall, entailed
precision, partial-support F1, high-confidence error, false-block rate,
abstention, evidence-span token F1, Recall@5, and MRR.

End-to-end relation scoring must count a missed gold claim as a pipeline miss;
it must not score only claims successfully emitted by Miner. Reports must show
extraction failures, retrieval failures, Judge failures, and local guard
failures separately.

## Baselines and resource comparison

At minimum, run on the same frozen documents, sources, provider, model, and
sampling configuration:

1. a true single-agent full-document baseline that sees the complete document
   and relevant source material under one typed output contract; and
2. the isolated Miner -> retrieval -> Judge flow, where Miner sees only its
   bounded document context and Judge sees only one enriched atomic claim and
   bounded evidence.

Report model/tool calls, input/output/total tokens, p50/p95 latency, failures,
and cost only when an explicit price snapshot supports it. A Multi-Agent
benefit claim requires better extraction or end-to-end quality commensurate
with additional resources; pair-level relation parity is not evidence either
for or against Miner value.

## Scope boundary

Phase 3E.2 adds no full-document dataset, labels, runner, model calls, or
benchmark result. It does not implement Scout, Challenger, Web search, query
rewrite, Action, SARIF, PR annotations, or any Phase 4 feature. Creating and
annotating this dataset is a separately authorized future phase.
