# Expected audit: bad-agent-comparison.md

This is a hand-authored Phase 0 gold artifact, not generated output. Line
numbers are 1-based and inclusive. `corroboration` is `not_requested` for every
case because the fixed demo uses citation mode only.

## Summary

| Relation | Count | Severity |
|---|---:|---|
| `contradicted` | 5 | `error` |
| `partially_entailed` | 1 | `warning` |
| `not_in_source` | 1 | `warning` |
| `source_unavailable` | 1 | `notice` |

## ET-001 — numeric mismatch

- Claim line: `9`
- Claim: “Nimbus completed 82% of the locked benchmark tasks.”
- Citation: `https://fixtures.evidencetrace.invalid/nimbus/benchmark-2026`
- Relation / severity: `contradicted` / `error`
- Source locator: `Benchmark report > Results`
- Evidence span: “Nimbus completed 62% of the 200 tasks in the locked evaluation set.”
- Reason: the cited result is 62%, not 82%.

## ET-002 — version mismatch

- Claim line: `12`
- Claim: “Durable checkpoints shipped in Nimbus v1.2.”
- Citation: `https://fixtures.evidencetrace.invalid/nimbus/releases/v1.3`
- Relation / severity: `contradicted` / `error`
- Source locator: `Release notes > Nimbus v1.3`
- Evidence span: “Durable checkpoints are available starting with Nimbus v1.3.”
- Reason: the capability starts in v1.3, not v1.2.

## ET-003 — entity mismatch

- Claim line: `15`
- Claim: “The Falcon team maintains Nimbus.”
- Citation: `https://fixtures.evidencetrace.invalid/nimbus/ownership`
- Relation / severity: `contradicted` / `error`
- Source locator: `Service registry > Owner`
- Evidence span: “The Orion team owns and maintains the Nimbus runtime.”
- Reason: the source names Orion, not Falcon.

## ET-004 — scope expansion

- Claim line: `20`
- Claim: “Nimbus is production-ready for regulated workloads on every supported platform.”
- Citation: `https://fixtures.evidencetrace.invalid/nimbus/readiness`
- Relation / severity: `partially_entailed` / `warning`
- Source locator: `Preview notice > Limitations`
- Evidence span: “Nimbus is an experimental preview for Linux test environments and is not approved for regulated production workloads.”
- Reason: the claim removes the preview, platform, and regulatory limitations.

## ET-005 — date mismatch

- Claim line: `23`
- Claim: “Nimbus became generally available on 2025-03-18.”
- Citation: `https://fixtures.evidencetrace.invalid/nimbus/launch`
- Relation / severity: `contradicted` / `error`
- Source locator: `Launch record > General availability`
- Evidence span: “Nimbus became generally available on 2025-04-18.”
- Reason: the source gives April 18, not March 18.

## ET-006 — negation flip

- Claim line: `26`
- Claim: “Nimbus does not require a database for durable runs.”
- Citation: `https://fixtures.evidencetrace.invalid/nimbus/storage`
- Relation / severity: `contradicted` / `error`
- Source locator: `Operations guide > Storage requirements`
- Evidence span: “Durable runs require a supported SQL database.”
- Reason: the claim negates an explicit requirement.

## ET-007 — claim absent from a related source

- Claim line: `29`
- Claim: “Nimbus guarantees exactly-once execution across regions.”
- Citation: `https://fixtures.evidencetrace.invalid/nimbus/retries`
- Relation / severity: `not_in_source` / `warning`
- Source locator: `Retry guide > Failure handling`
- Evidence span: none; `not_in_source` must not invent one.
- Reason: the related retry guide does not state an exactly-once, cross-region guarantee.

## ET-008 — unavailable source

- Claim line: `32`
- Claim: “Nimbus outperformed every competing runtime in the 2026 latency study.”
- Citation: `https://fixtures.evidencetrace.invalid/nimbus/missing-latency-study`
- Relation / severity: `source_unavailable` / `notice`
- Source locator: unavailable fixture
- Evidence span: none; the source has no cached content.
- Reason: the source is intentionally unavailable, so the relation cannot be judged.

