# ruff: noqa: B008

"""Typer entry point for citation audit, demo, and offline evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from evidencetrace import __version__

app = typer.Typer(
    add_completion=False,
    help="EvidenceTrace CI — factual CI for technical Markdown.",
    no_args_is_help=True,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def root(
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show the package version and exit.",
        ),
    ] = None,
) -> None:
    """Expose global package options before dispatching a subcommand."""


def main() -> None:
    """Run the Typer application."""

    app()


def _has_interactive_tty() -> bool:
    import sys

    return sys.stdin.isatty() and sys.stdout.isatty()


__all__ = ["app", "main"]


@app.command()
def check(
    targets: list[Path] = typer.Argument(
        ...,
        help="One or more explicit Markdown or plain-text targets.",
    ),
    model: str | None = typer.Option(None, help="OpenAI-compatible model id."),
    discover: bool = typer.Option(
        False,
        "--discover",
        help=(
            "Explicitly authorize bounded Scout discovery for unresolved "
            "claims; a missing search key becomes discovery_unavailable."
        ),
    ),
    changed_from: str | None = typer.Option(
        None,
        "--changed-from",
        help="Audit only Markdown paragraphs changed from this Git revision.",
    ),
    suggest_patch: Path | None = typer.Option(
        None,
        "--suggest-patch",
        help=(
            "Write an evidence-gated candidate diff without applying, "
            "staging, or committing it."
        ),
    ),
    sarif: Path | None = typer.Option(
        None,
        "--sarif",
        help="Atomically write deterministic SARIF inside the project root.",
    ),
    reference: list[Path] | None = typer.Option(
        None,
        "--reference",
        help="Shared read-only Markdown/TXT evidence; repeat for multiple files.",
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        help="Use DIR as the canonical batch run root.",
    ),
) -> None:
    """Audit targets citation-first with bounded Agents; always read-only."""
    import os

    from evidencetrace.agents.scout import TavilySearchClient
    from evidencetrace.model_client import (
        ModelConfigError,
        OpenAICompatibleClient,
        SchemaRecoveryClient,
    )
    from evidencetrace.policy import exit_code
    from evidencetrace.product import (
        PRODUCT_MAX_PROVIDER_ATTEMPTS,
        DocumentRunStatus,
        ProductAuditPipeline,
        ProductPipelineError,
        write_suggestion_patch,
    )
    from evidencetrace.sarif import SarifError, audit_policy_findings, write_sarif
    from evidencetrace.self_use import (
        SelfUseError,
        SelfUseRunner,
        new_run_id,
        prepare_self_use_run,
    )
    from evidencetrace.self_use_inputs import (
        InputContractError,
        OutputPathRequest,
        preflight_self_use_inputs,
    )

    root = Path.cwd()
    references = tuple(reference or ())
    model_id = model or os.getenv("EVIDENCETRACE_MODEL")
    suffix = targets[0].suffix.casefold() if len(targets) == 1 else ""
    self_use_mode = (
        len(targets) != 1
        or bool(references)
        or output_dir is not None
        or suffix == ".txt"
        or discover
    )
    try:
        if self_use_mode:
            prepared = prepare_self_use_run(
                targets,
                references=references,
                mode="check",
                project_root=root,
                discover=discover,
                changed_from=changed_from,
                output_dir=output_dir,
                sarif=sarif,
                suggest_patch=suggest_patch,
            )
            client = (
                SchemaRecoveryClient(
                    OpenAICompatibleClient(
                        model_id=model_id,
                        max_calls=(
                            PRODUCT_MAX_PROVIDER_ATTEMPTS
                            * len(prepared.inputs.targets)
                        ),
                    )
                )
                if model_id and os.getenv("OPENAI_API_KEY")
                else None
            )
            batch_result = SelfUseRunner(
                project_root=root,
                model=client,
                search_client=TavilySearchClient() if discover else None,
            ).run_prepared(prepared)
            typer.echo(batch_result.terminal)
            if batch_result.exit_code:
                raise typer.Exit(code=batch_result.exit_code)
            return

        legacy_run_id = new_run_id()
        legacy_run_root = root / ".evidencetrace" / "runs" / legacy_run_id
        output_requests = tuple(
            request
            for request in (
                OutputPathRequest("sarif", sarif) if sarif is not None else None,
                (
                    OutputPathRequest("suggest_patch", suggest_patch)
                    if suggest_patch is not None
                    else None
                ),
                OutputPathRequest(
                    "legacy_run_root",
                    legacy_run_root,
                    origin="derived",
                ),
                OutputPathRequest(
                    "legacy_audit_json",
                    legacy_run_root / "audit.json",
                    origin="derived",
                ),
                OutputPathRequest(
                    "legacy_audit_markdown",
                    legacy_run_root / "audit.md",
                    origin="derived",
                ),
                OutputPathRequest(
                    "legacy_product_run",
                    legacy_run_root / "product_run.json",
                    origin="derived",
                ),
            )
            if request is not None
        )
        contract = preflight_self_use_inputs(
            targets,
            command="check",
            project_root=root,
            output_paths=output_requests,
        )
        if (
            changed_from is not None
            and contract.targets[0].display_path.startswith("external/")
        ):
            raise SelfUseError("changed_from_requires_internal_markdown")
        for output in contract.outputs:
            try:
                output.path.relative_to(root.resolve())
            except ValueError:
                raise SarifError("invalid_output_path") from None
            if output.origin == "derived" and (
                output.path.exists() or output.path.is_symlink()
            ):
                raise SarifError("invalid_output_path")
            if (
                output.origin == "explicit"
                and output.path.exists()
                and not output.path.is_file()
            ):
                raise SarifError("invalid_output_path")
        client = (
            SchemaRecoveryClient(
                OpenAICompatibleClient(
                    model_id=model_id,
                    max_calls=PRODUCT_MAX_PROVIDER_ATTEMPTS,
                )
            )
            if model_id and os.getenv("OPENAI_API_KEY")
            else None
        )
        legacy_result = ProductAuditPipeline(
            project_root=root,
            model=client,
            search_client=TavilySearchClient() if discover else None,
            enforce_human_evidence_gates=True,
        ).run(
            contract.targets[0].path,
            discover=discover,
            changed_from=changed_from,
            run_id=legacy_run_id,
        )
        if sarif is not None:
            write_sarif(legacy_result.audit, sarif, project_root=root)
        if suggest_patch is not None:
            write_suggestion_patch(
                contract.targets[0].path,
                legacy_result.audit,
                suggest_patch,
                project_root=root,
            )
        policy_code = exit_code(
            tuple(
                item.decision
                for item in audit_policy_findings(legacy_result.audit)
            ),
            legacy_result.audit.effective_config.policy,
        )
    except ModelConfigError as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error
    except InputContractError as error:
        typer.echo(f"Error: input rejected ({error.code}).", err=True)
        raise typer.Exit(code=1) from None
    except SelfUseError as error:
        typer.echo(f"Error: self-use check failed ({error.code}).", err=True)
        raise typer.Exit(code=1) from None
    except ProductPipelineError:
        typer.echo("Error: product audit failed safely.", err=True)
        raise typer.Exit(code=1) from None
    except SarifError as error:
        typer.echo(f"Error: SARIF output failed ({error.code}).", err=True)
        raise typer.Exit(code=1) from None
    except typer.Exit:
        raise
    except Exception:
        typer.echo("Error: audit failed.", err=True)
        raise typer.Exit(code=1) from None
    typer.echo(legacy_result.terminal)
    if legacy_result.product.document_status is DocumentRunStatus.PARTIAL:
        raise typer.Exit(code=2)
    if policy_code:
        raise typer.Exit(code=policy_code)


@app.command()
def fix(
    targets: list[Path] = typer.Argument(
        ...,
        help="One or more explicit .md, .markdown, or .txt targets.",
    ),
    model: str | None = typer.Option(None, help="OpenAI-compatible model id."),
    discover: bool = typer.Option(
        False,
        "--discover",
        help=(
            "Explicitly authorize bounded Scout discovery; Tavily-only exact "
            "evidence still requires a separate trust decision."
        ),
    ),
    reference: list[Path] | None = typer.Option(
        None,
        "--reference",
        help="Shared read-only Markdown/TXT evidence; repeat for multiple files.",
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        help="Use DIR as the canonical batch run root.",
    ),
) -> None:
    """Interactively apply evidence-grounded scalar repairs."""
    import os

    from evidencetrace.agents.scout import TavilySearchClient
    from evidencetrace.model_client import (
        ModelConfigError,
        OpenAICompatibleClient,
        SchemaRecoveryClient,
    )
    from evidencetrace.product import PRODUCT_MAX_PROVIDER_ATTEMPTS
    from evidencetrace.self_use import (
        InteractiveSession,
        PreparedSelfUseRun,
        SelfUseError,
        SelfUseRunner,
        prepare_self_use_run,
    )
    from evidencetrace.self_use_inputs import InputContractError

    if not _has_interactive_tty():
        typer.echo(
            "Error: fix requires an interactive TTY; no files were written.",
            err=True,
        )
        raise typer.Exit(code=2)

    root = Path.cwd()
    references = tuple(reference or ())
    model_id = model or os.getenv("EVIDENCETRACE_MODEL")
    prepared: PreparedSelfUseRun | None = None

    def recovery_hint() -> str:
        if prepared is None:
            return ""
        if output_dir is None:
            return (
                " Recovery artifacts, if created, are under "
                f".evidencetrace/runs/{prepared.run_id}/."
            )
        return (
            " Recovery artifacts, if created, are under the requested "
            f"--output-dir (run ID {prepared.run_id})."
        )

    try:
        prepared = prepare_self_use_run(
            targets,
            references=references,
            mode="fix",
            project_root=root,
            discover=discover,
            output_dir=output_dir,
        )
        client = (
            SchemaRecoveryClient(
                OpenAICompatibleClient(
                    model_id=model_id,
                    max_calls=(
                        PRODUCT_MAX_PROVIDER_ATTEMPTS
                        * len(prepared.inputs.targets)
                    ),
                )
            )
            if model_id and os.getenv("OPENAI_API_KEY")
            else None
        )
        result = SelfUseRunner(
            project_root=root,
            model=client,
            search_client=TavilySearchClient() if discover else None,
            interactive=InteractiveSession(),
        ).run_prepared(prepared)
    except ModelConfigError as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error
    except InputContractError as error:
        typer.echo(f"Error: input rejected ({error.code}).", err=True)
        raise typer.Exit(code=1) from None
    except SelfUseError as error:
        typer.echo(
            f"Error: interactive fix failed ({error.code})."
            + recovery_hint(),
            err=True,
        )
        raise typer.Exit(code=1) from None
    except Exception:
        typer.echo(
            "Error: interactive fix failed safely." + recovery_hint(),
            err=True,
        )
        raise typer.Exit(code=1) from None
    typer.echo(result.terminal)
    if result.exit_code:
        raise typer.Exit(code=result.exit_code)


@app.command()
def demo() -> None:
    """Run the byte-stable, offline bounded-Agent demo."""
    from datetime import UTC, datetime

    from evidencetrace.agents.scout import FixtureSearchClient
    from evidencetrace.pipeline import CitationAuditPipeline, fixture_transport
    from evidencetrace.retrieval.fetch import SafeFetcher

    root = Path.cwd()
    document = root / "examples" / "product-demo" / "README.md"
    manifest = root / "examples" / "evidence" / "source_manifest.json"
    if not document.exists() or not manifest.exists():
        typer.echo(
            "Error: demo fixtures are not present in the current project.", err=True
        )
        raise typer.Exit(code=2)
    pipeline = CitationAuditPipeline(
        project_root=root,
        fetcher=SafeFetcher(transport=fixture_transport(manifest), resolve_dns=False),
        search_client=FixtureSearchClient.from_manifest(manifest),
    )
    try:
        result = pipeline.run(
            document,
            discover=True,
            run_id="offline-product-demo",
            started_at=datetime(2030, 1, 1, tzinfo=UTC),
            persist=False,
        )
    except Exception:
        typer.echo("Error: demo failed safely.", err=True)
        raise typer.Exit(code=2) from None
    typer.echo(result.terminal)


@app.command("eval")
def eval_command(
    dataset: Path = typer.Argument(..., exists=True, readable=True),
    out: Path = typer.Option(
        Path("eval_runs/core"), "--out", help="Evaluation artifact directory."
    ),
    split: str = typer.Option("all", "--split", help="all, dev, or test."),
    live: bool = typer.Option(
        False,
        "--live",
        help=(
            "Request all live pair baselines; missing credentials are recorded "
            "as clean skips."
        ),
    ),
    model: str | None = typer.Option(
        None, "--model", help="OpenAI-compatible model id for --live."
    ),
) -> None:
    """Run explicit deterministic baselines and optional live comparisons."""
    from evidencetrace.eval.dataset import DatasetValidationError
    from evidencetrace.eval.runner import run_eval

    try:
        model_id = (
            model
            or __import__("os").getenv("EVIDENCETRACE_MODEL")
            or "deterministic-fake-v1"
        )
        result = run_eval(
            dataset, out, selected_split=split, model_id=model_id, live_requested=live
        )
    except (DatasetValidationError, OSError, RuntimeError, ValueError) as error:
        typer.echo(f"Error: eval failed: {error}", err=True)
        raise typer.Exit(code=2) from error
    typer.echo(f"Evaluation artifacts written to {result}")
    typer.echo(
        "Canonical files: eval_report.md, eval_results.jsonl, metrics.json, "
        "run_manifest.json"
    )
