# Working-name availability check

Checked: 2026-07-10 UTC

This is an engineering preflight, not a trademark opinion or legal clearance.

| Surface | Query and result | Status |
|---|---|---|
| PyPI | `GET https://pypi.org/pypi/evidencetrace/json` returned HTTP 404. | The normalized distribution name appeared unregistered at check time. |
| GitHub | Repository search for `evidencetrace in:name` returned two exact public names: `z1000biker/EvidenceTrace` and `Boombaka3/EvidenceTrace`. | Exact-name collision. One result describes claim decomposition and assisted verification, so this is materially overlapping. |
| USPTO | Exact-term web and official-search discovery did not produce a reliable clearance result in the automated check. | Inconclusive; a qualified manual search is required before public release. |

Decision: retain **EvidenceTrace CI** only as the local working name because the
implementation directory and source-of-truth plan explicitly use it. Do not
claim that the GitHub name is available, publish under an assumed owner/name, or
reserve the PyPI package until a maintainer resolves naming and legal review.
Branding and publication are therefore **blocked**; Phase 0–1 deterministic implementation is not blocked.

