# Agent Runtime Adoption Proposal

This intentionally incorrect document is the fixed bad-document demo for
EvidenceTrace CI. Each `ET-*` marker names one controlled evidence failure.

## Benchmark and release claims

<!-- ET-001: numeric_mismatch -->
Nimbus completed 82% of the locked benchmark tasks [in the benchmark report](https://fixtures.evidencetrace.invalid/nimbus/benchmark-2026).

<!-- ET-002: version_mismatch -->
Durable checkpoints shipped in Nimbus v1.2 [according to the release notes][nimbus-release].

<!-- ET-003: entity_mismatch -->
The Falcon team maintains Nimbus [according to the ownership registry](https://fixtures.evidencetrace.invalid/nimbus/ownership).

## Readiness and operations

<!-- ET-004: scope_expansion -->
Nimbus is production-ready for regulated workloads on every supported platform.[^readiness]

<!-- ET-005: date_mismatch -->
Nimbus became generally available on 2025-03-18. The launch record is https://fixtures.evidencetrace.invalid/nimbus/launch.

<!-- ET-006: negation_flip -->
Nimbus does not require a database for durable runs [as described in storage requirements](https://fixtures.evidencetrace.invalid/nimbus/storage).

<!-- ET-007: absent_claim -->
Nimbus guarantees exactly-once execution across regions [according to its retry guide][retry-guide].

<!-- ET-008: unavailable_source -->
Nimbus outperformed every competing runtime in the 2026 latency study [reported here](https://fixtures.evidencetrace.invalid/nimbus/missing-latency-study).

[nimbus-release]: https://fixtures.evidencetrace.invalid/nimbus/releases/v1.3 "Nimbus v1.3 release notes"
[retry-guide]: https://fixtures.evidencetrace.invalid/nimbus/retries "Nimbus retry guide"

[^readiness]: The preview notice is archived at https://fixtures.evidencetrace.invalid/nimbus/readiness.

