# Runtime Migration Decision Record

## Context

The current worker pool processes 4,000 jobs per hour, and the measurement was
recorded in the [capacity report][capacity]. The report also records a p95 queue
delay of 18 seconds.[^capacity-note]

> The operations handbook requires a rollback exercise before production
> migration: https://fixtures.evidencetrace.invalid/runtime/operations.

## Decision

We will deploy Nimbus v1.3 in two stages:

1. The staging rollout begins on 2026-08-11
   [under the approved plan](https://fixtures.evidencetrace.invalid/runtime/rollout).
2. Production traffic remains capped at 10% until the error budget review is
   complete [as required by the rollout policy][rollout-policy].

<!-- evidencetrace: ignore reason="explicit team preference" -->
We prefer Nimbus because its configuration is easier to review.

### Failure handling

The runtime retries failed work at least once; it does not promise exactly-once
delivery [in the retry contract](https://fixtures.evidencetrace.invalid/runtime/retries).

An inline example such as `curl https://fixtures.evidencetrace.invalid/not-evidence`
is code and must not become a citation.

```text
https://fixtures.evidencetrace.invalid/fenced-code-is-not-evidence
[also not evidence](https://fixtures.evidencetrace.invalid/in-code)
```

![Architecture sketch](https://fixtures.evidencetrace.invalid/runtime/diagram.png)

[capacity]: https://fixtures.evidencetrace.invalid/runtime/capacity "Capacity report"
[rollout-policy]: https://fixtures.evidencetrace.invalid/runtime/policy

[^capacity-note]: See the stable report at
    https://fixtures.evidencetrace.invalid/runtime/capacity-details and the
    measurement notes at https://fixtures.evidencetrace.invalid/runtime/method.

