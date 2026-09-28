# Labels and CI severity

EvidenceTrace separates claim-to-source relation from independent
corroboration. “Unsupported” is user-facing shorthand only; it is never an
internal relation label.

## Claim-to-source relation

| Label | Meaning |
|---|---|
| `entailed` | The source explicitly supports the complete claim. |
| `partially_entailed` | The source supports only part of the claim or uses different qualifications. |
| `contradicted` | The source explicitly conflicts with the claim. |
| `not_in_source` | The source is related, but it does not contain enough evidence for the claim. |
| `source_unavailable` | The source cannot be accessed or parsed. |
| `not_checkable` | The text is an opinion, prediction, recommendation, or otherwise unsuitable for factual checking. |

## Independent corroboration

| Status | Meaning |
|---|---|
| `cited_only` | Only the original citation supports the claim; no independent check was run. |
| `corroborated` | At least one policy-compliant independent source supports the claim. |
| `disputed` | Trusted sources materially conflict. |
| `no_evidence` | No adequate independent evidence was found. |
| `not_requested` | Independent corroboration was not requested. |

## Default severity mapping

| Severity | Labels/statuses |
|---|---|
| `error` | `contradicted` |
| `warning` | `partially_entailed`, `not_in_source`, `disputed`, `no_evidence` |
| `notice` | `source_unavailable`, `not_checkable` |
| `pass` | `entailed` |

An `entailed`, `partially_entailed`, or `contradicted` verdict is invalid without
a source span copied from cached source content. Missing evidence must resolve
to `not_in_source`, `source_unavailable`, or abstention rather than a fabricated
quotation.

