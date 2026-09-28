# EvidenceTrace

EvidenceTrace audits factual claims in technical Markdown, including READMEs,
ADRs, RFCs, and Git diffs. It checks cited and user-supplied evidence, can
discover more evidence when authorized, and maps findings back to source lines
in the terminal, JSON, and SARIF.

A verdict describes the available evidence. Unsupported and uncertain claims
remain explicit; EvidenceTrace is not a general truth detector.

## Quick start

Requires Python 3.11+. From this source checkout:

```bash
python -m pip install -e .
evidencetrace demo
evidencetrace check README.md
evidencetrace check README.md --reference docs/architecture.md --sarif audit.sarif
```

`demo` is offline and needs no credentials. `check` does not modify its targets.
Use `--discover` to authorize web discovery. The interactive `fix` command
asks for confirmation before applying supported repairs.
If evidence is insufficient, `check` exits with code `2` for a partial review.

## Configuration

Model-backed checks use `EVIDENCETRACE_MODEL` and `OPENAI_API_KEY`; set
`OPENAI_BASE_URL` for a compatible provider. Web discovery uses
`TAVILY_API_KEY`. Deterministic checks and fallbacks work without these keys.

## Repository scope

The public repository includes source, tests, examples, and documentation.
Local evaluation datasets and generated runs (`eval_sets/` and `eval_runs/`)
are omitted; evaluation commands and tests that read them need local data.
The reusable [GitHub workflow](.github/workflows/evidencetrace.yml) reports
findings as SARIF.

## Documentation

- [Documentation index](docs/README.md)
- [Architecture and safety boundaries](docs/architecture.md)
- [Setup and recovery](docs/setup-and-recovery.md)
- [Known limitations](docs/known-limitations.md)
- [Implementation and evaluation history](docs/implementation_status.md)
