"""Run SciFact public dev with one shared retriever and local offline NLI."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import re
import subprocess
import sys
from collections.abc import Iterable
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from evidencetrace.audit_models import EvidenceChunk, JudgeInput, RetrievedEvidence
from evidencetrace.checks.deterministic import (
    bounded_evidence_text,
    deterministic_signals,
)
from evidencetrace.eval.dataset import LoadedDataset, load_dataset
from evidencetrace.eval.models import EvalCase, SourceFixture
from evidencetrace.eval.scifact_judge import (
    SCIFACT_JUDGE_POLICY_VERSION,
    SCIFACT_RELATIONS,
    SciFactJudgeConfig,
    SemanticNliScorer,
    TransformersNliScorer,
    judge_scifact_3way,
)
from evidencetrace.models import (
    AtomicClaim,
    Checkability,
    Relation,
    SourceMetadata,
)
from evidencetrace.retrieval.rank import LexicalRetriever

RUNNER_VERSION = "scifact-public-dev-local-nli-runner-v2"
TASK_PROFILE_VERSION = "scifact-3way-prediction-profile-v1"
RETRIEVAL_TOP_K = 5
SENTENCE_INDEX_RELATIVE = "sources/sentence_index.jsonl"
BASELINES = ("lexical_rules", "retrieval_judge_scifact_nli")
SCIFACT_ALLOWED_RELATIONS = (
    Relation.ENTAILED,
    Relation.CONTRADICTED,
    Relation.NOT_IN_SOURCE,
)
SUBSTANTIVE_RELATIONS = {Relation.ENTAILED, Relation.CONTRADICTED}
REPO_ROOT = Path(__file__).resolve().parents[1]
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[._-][A-Za-z0-9]+)*")
STOPWORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "of",
    "to",
    "in",
    "on",
    "and",
    "for",
    "with",
}
WEIGHT_FILE_RE = re.compile(
    r"^(?:model(?:-\d{5}-of-\d{5})?\.safetensors|"
    r"pytorch_model(?:-\d{5}-of-\d{5})?\.bin)$"
)


class SciFactRunnerError(ValueError):
    """A pack, model identity, prediction, or output violates the run contract."""


@dataclass(frozen=True, slots=True)
class _ModelIdentity:
    local_path: Path
    upstream_model_id: str
    revision: str
    weights_sha256: str
    weights_hash_scope: str
    identity_binding_sha256: str
    file_sha256: dict[str, str]
    hf_metadata: dict[str, dict[str, str]]
    hf_metadata_sha256: dict[str, str]
    batch_size: int
    device: str


@dataclass(frozen=True, slots=True)
class _EvidenceRecord:
    source_id: str
    text: str
    locator: str
    char_start: int
    char_end: int
    rank: int
    score: float | None = None

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        if self.score is None:
            value.pop("score")
        return value


@dataclass(frozen=True, slots=True)
class _Prediction:
    case_id: str
    baseline: str
    relation: Relation
    confidence: float
    reason: str
    retrieved: tuple[_EvidenceRecord, ...]
    selected: tuple[_EvidenceRecord, ...]
    external_provider_calls: int = 0
    local_nli_score_many_calls: int = 0
    local_nli_pair_count: int = 0
    local_nli_subset_count: int = 0
    local_nli_batch_count: int = 0
    model_id: str | None = None
    model_revision: str | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def _jsonl_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for row in rows
    ).encode("utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SciFactRunnerError(f"required JSON file is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SciFactRunnerError(f"invalid JSON file: {path}") from error
    if not isinstance(value, dict):
        raise SciFactRunnerError(f"JSON file must contain an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise SciFactRunnerError(f"required JSONL file is missing: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise SciFactRunnerError(
                f"invalid JSONL row at {path}:{line_number}"
            ) from error
        if not isinstance(value, dict):
            raise SciFactRunnerError(
                f"JSONL row must be an object at {path}:{line_number}"
            )
        rows.append(value)
    if not rows:
        raise SciFactRunnerError(f"JSONL file is empty: {path}")
    return rows


def _assert_output_available(path: Path) -> None:
    if not path.exists():
        return
    if not path.is_dir():
        raise SciFactRunnerError(f"output path is not a directory: {path}")
    if any(path.iterdir()):
        raise SciFactRunnerError(
            f"output directory is not empty; refusing to overwrite: {path}"
        )


def _write_exclusive(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(content)
    except FileExistsError as error:
        raise SciFactRunnerError(f"refusing to overwrite output: {path}") from error


def _fixture_path(pack: Path, fixture: str) -> Path:
    relative = PurePosixPath(fixture)
    if relative.is_absolute() or ".." in relative.parts or "\\" in fixture:
        raise SciFactRunnerError(f"source fixture escapes pack: {fixture}")
    resolved = (pack / Path(*relative.parts)).resolve()
    try:
        resolved.relative_to(pack)
    except ValueError as error:
        raise SciFactRunnerError(f"source fixture escapes pack: {fixture}") from error
    if not resolved.is_file():
        raise SciFactRunnerError(f"source fixture is missing: {fixture}")
    return resolved


def _require_int(value: Any, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise SciFactRunnerError(f"sentence index requires integer {field}")
    return value


def _sentence_chunks(
    pack: Path, loaded: LoadedDataset
) -> tuple[dict[str, tuple[EvidenceChunk, ...]], Path]:
    """Load a label-free official sentence index and bind it to source bytes."""

    index_path = pack / Path(*PurePosixPath(SENTENCE_INDEX_RELATIVE).parts)
    rows = _read_jsonl(index_path)
    by_case: dict[str, tuple[EvidenceChunk, ...]] = {}
    expected_cases = {case.case_id: case for case in loaded.cases}
    for row in rows:
        if set(row) != {"case_id", "source_id", "documents", "sentence_spans"}:
            raise SciFactRunnerError("sentence index contains non-structural fields")
        case_id = row.get("case_id")
        source_id = row.get("source_id")
        if (
            not isinstance(case_id, str)
            or case_id not in expected_cases
            or case_id in by_case
        ):
            raise SciFactRunnerError("sentence index has an invalid case identity")
        case = expected_cases[case_id]
        if not isinstance(source_id, str) or source_id != case.source_id:
            raise SciFactRunnerError("sentence index source binding mismatch")
        source = loaded.sources[source_id]
        if not source.available:
            raise SciFactRunnerError("SciFact three-way profile requires a source")
        raw_spans = row.get("sentence_spans")
        if not isinstance(raw_spans, list) or not raw_spans:
            raise SciFactRunnerError("sentence index requires sentence spans")
        chunks: list[EvidenceChunk] = []
        ordered_identities: list[tuple[int, int]] = []
        prior_end = 0
        for span_index, raw_span in enumerate(raw_spans):
            if not isinstance(raw_span, dict) or set(raw_span) != {
                "doc_id",
                "sentence_id",
                "char_start",
                "char_end",
                "locator",
                "text_sha256",
            }:
                raise SciFactRunnerError("sentence index span schema mismatch")
            doc_id = _require_int(raw_span["doc_id"], field="doc_id")
            sentence_id = _require_int(raw_span["sentence_id"], field="sentence_id")
            start = _require_int(raw_span["char_start"], field="char_start")
            end = _require_int(raw_span["char_end"], field="char_end")
            locator = raw_span.get("locator")
            expected_hash = raw_span.get("text_sha256")
            identity = (doc_id, sentence_id)
            expected_locator = f"scifact://document/{doc_id}/sentence/{sentence_id}"
            expected_separator = "" if span_index == 0 else "\n"
            if (
                start < 0
                or end <= start
                or end > len(source.content)
                or source.content[prior_end:start] != expected_separator
                or locator != expected_locator
                or not isinstance(expected_hash, str)
                or not SHA256_RE.fullmatch(expected_hash)
                or identity in ordered_identities
            ):
                raise SciFactRunnerError("sentence index span identity is invalid")
            text = source.content[start:end]
            if not text.strip() or _sha256_bytes(text.encode("utf-8")) != expected_hash:
                raise SciFactRunnerError("sentence index text hash mismatch")
            chunks.append(
                EvidenceChunk(
                    source_id=source_id,
                    url=source.url,
                    text=text,
                    heading_path=(f"scifact document {doc_id}",),
                    locator=expected_locator,
                    char_start=start,
                    char_end=end,
                )
            )
            ordered_identities.append(identity)
            prior_end = end
        if prior_end != len(source.content):
            raise SciFactRunnerError("semantic source has trailing unindexed text")
        _validate_documents(row.get("documents"), ordered_identities, chunks)
        by_case[case_id] = tuple(chunks)
    if set(by_case) != set(expected_cases):
        raise SciFactRunnerError("sentence index and cases.jsonl IDs differ")
    return by_case, index_path


def _validate_documents(
    raw_documents: Any,
    ordered_identities: list[tuple[int, int]],
    chunks: list[EvidenceChunk],
) -> None:
    if not isinstance(raw_documents, list) or not raw_documents:
        raise SciFactRunnerError("sentence index requires document metadata")
    expected_doc_ids = list(dict.fromkeys(doc_id for doc_id, _ in ordered_identities))
    observed_doc_ids: list[int] = []
    for document in raw_documents:
        if not isinstance(document, dict) or set(document) != {
            "doc_id",
            "title",
            "char_start",
            "char_end",
            "sentence_keys",
        }:
            raise SciFactRunnerError("sentence index document schema mismatch")
        doc_id = _require_int(document["doc_id"], field="document.doc_id")
        start = _require_int(document["char_start"], field="document.char_start")
        end = _require_int(document["char_end"], field="document.char_end")
        title = document.get("title")
        sentence_keys = document.get("sentence_keys")
        positions = [
            index
            for index, identity in enumerate(ordered_identities)
            if identity[0] == doc_id
        ]
        expected_keys = [
            f"{ordered_identities[index][0]}:{ordered_identities[index][1]}"
            for index in positions
        ]
        if (
            not positions
            or doc_id in observed_doc_ids
            or not isinstance(title, str)
            or not title.strip()
            or sentence_keys != expected_keys
            or start != chunks[positions[0]].char_start
            or end != chunks[positions[-1]].char_end
        ):
            raise SciFactRunnerError("sentence index document identity is invalid")
        observed_doc_ids.append(doc_id)
    if observed_doc_ids != expected_doc_ids:
        raise SciFactRunnerError("sentence index document order mismatch")


def _input_hashes(
    pack: Path,
    cases_path: Path,
    sentence_index_path: Path,
    loaded: LoadedDataset,
) -> dict[str, str]:
    paths = {cases_path, sentence_index_path, pack / "manifest.json"}
    frozen_path = cases_path.with_name(f"{cases_path.stem}.frozen_hashes.json")
    if frozen_path.is_file():
        paths.add(frozen_path)
    for fixture in {case.source_fixture for case in loaded.cases}:
        paths.add(_fixture_path(pack, fixture))
    return {
        path.relative_to(pack).as_posix(): _sha256(path)
        for path in sorted(paths, key=lambda item: item.as_posix())
    }


def _validate_pack_manifest(pack: Path, sentence_index_path: Path) -> dict[str, Any]:
    manifest = _read_json(pack / "manifest.json")
    if manifest.get("schema_version") != "scifact-public-dev-v2":
        raise SciFactRunnerError("runner requires a v2 SciFact adapter pack")
    if (
        manifest.get("mapping", {}).get("prediction_sentence_index")
        != SENTENCE_INDEX_RELATIVE
    ):
        raise SciFactRunnerError("pack does not bind the prediction sentence index")
    expected_hash = manifest.get("output_sha256", {}).get(SENTENCE_INDEX_RELATIVE)
    if expected_hash != _sha256(sentence_index_path):
        raise SciFactRunnerError("sentence index manifest hash mismatch")
    return manifest


def _git_metadata() -> dict[str, Any]:
    git = ["git", "-c", f"safe.directory={REPO_ROOT}"]
    try:
        commit = subprocess.run(
            [*git, "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        status = subprocess.run(
            [*git, "diff", "--quiet", "HEAD", "--"],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            timeout=10,
        )
        tracked = (
            "clean"
            if status.returncode == 0
            else "dirty"
            if status.returncode == 1
            else "unavailable"
        )
        return {"commit": commit, "tracked_worktree": tracked}
    except (FileNotFoundError, subprocess.SubprocessError):
        return {"commit": None, "tracked_worktree": "unavailable"}


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for distribution in ("evidencetrace", "torch", "transformers", "tokenizers"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def _model_identity(
    model_path: Path | str,
    *,
    upstream_model_id: str,
    revision: str,
    weights_sha256: str,
    batch_size: int,
    device: str,
    require_hf_metadata: bool,
) -> _ModelIdentity:
    resolved = Path(model_path).resolve()
    if not resolved.is_dir():
        raise SciFactRunnerError("local NLI model path must be an existing directory")
    if not upstream_model_id.strip() or not REVISION_RE.fullmatch(revision):
        raise SciFactRunnerError(
            "NLI model requires an upstream ID and commit revision"
        )
    if not SHA256_RE.fullmatch(weights_sha256) or batch_size <= 0 or not device.strip():
        raise SciFactRunnerError("invalid local NLI model configuration")
    files = tuple(
        sorted(
            path
            for path in resolved.iterdir()
            if path.is_file() and not path.name.endswith((".lock", ".incomplete"))
        )
    )
    if not files:
        raise SciFactRunnerError("local NLI model directory is empty")
    file_hashes = {
        path.relative_to(resolved).as_posix(): _sha256(path) for path in files
    }
    weight_hashes = {
        name: digest
        for name, digest in file_hashes.items()
        if WEIGHT_FILE_RE.fullmatch(PurePosixPath(name).name)
    }
    if not weight_hashes:
        raise SciFactRunnerError("local NLI model has no supported weight files")
    if len(weight_hashes) == 1:
        observed_hash = next(iter(weight_hashes.values()))
        hash_scope = next(iter(weight_hashes))
    else:
        observed_hash = _sha256_bytes(_json_bytes(weight_hashes))
        hash_scope = "canonical JSON map of relative weight paths to file SHA-256"
    if observed_hash != weights_sha256:
        raise SciFactRunnerError("local NLI weights SHA-256 mismatch")
    metadata, metadata_hashes = _hf_download_metadata(
        resolved,
        file_hashes=file_hashes,
        weight_hashes=weight_hashes,
        revision=revision,
        require=require_hf_metadata,
    )
    identity_binding = {
        "upstream_model_id": upstream_model_id,
        "revision": revision,
        "weights_sha256": weights_sha256,
    }
    return _ModelIdentity(
        local_path=resolved,
        upstream_model_id=upstream_model_id,
        revision=revision,
        weights_sha256=weights_sha256,
        weights_hash_scope=hash_scope,
        identity_binding_sha256=_sha256_bytes(_json_bytes(identity_binding)),
        file_sha256=file_hashes,
        hf_metadata=metadata,
        hf_metadata_sha256=metadata_hashes,
        batch_size=batch_size,
        device=device,
    )


def _hf_download_metadata(
    model_path: Path,
    *,
    file_hashes: dict[str, str],
    weight_hashes: dict[str, str],
    revision: str,
    require: bool,
) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    """Validate local hf_hub metadata without claiming remote authentication."""

    if not require:
        return {}, {}
    metadata_dir = model_path / ".cache" / "huggingface" / "download"
    if not metadata_dir.is_dir():
        raise SciFactRunnerError("reportable run requires local hf_hub metadata")
    records: dict[str, dict[str, str]] = {}
    hashes: dict[str, str] = {}
    for relative_name in sorted(file_hashes):
        metadata_path = metadata_dir / f"{relative_name}.metadata"
        if not metadata_path.is_file():
            raise SciFactRunnerError("reportable run has incomplete hf_hub metadata")
        try:
            lines = metadata_path.read_text(encoding="utf-8").splitlines()
        except OSError as error:
            raise SciFactRunnerError("hf_hub metadata is unreadable") from error
        if len(lines) != 3 or not all(line.strip() for line in lines):
            raise SciFactRunnerError("hf_hub metadata format is invalid")
        metadata_revision, etag, downloaded_at = (line.strip() for line in lines)
        try:
            timestamp = float(downloaded_at)
        except ValueError as error:
            raise SciFactRunnerError("hf_hub metadata timestamp is invalid") from error
        if not math.isfinite(timestamp) or metadata_revision != revision:
            raise SciFactRunnerError("hf_hub metadata revision mismatch")
        if relative_name in weight_hashes and etag != weight_hashes[relative_name]:
            raise SciFactRunnerError("hf_hub weight etag does not match local SHA-256")
        metadata_relative = (
            PurePosixPath(".cache")
            / "huggingface"
            / "download"
            / f"{relative_name}.metadata"
        ).as_posix()
        records[relative_name] = {
            "revision": metadata_revision,
            "etag": etag,
            "downloaded_at": downloaded_at,
            "metadata_path": metadata_relative,
        }
        hashes[metadata_relative] = _sha256(metadata_path)
    return records, hashes


def _words(value: str) -> set[str]:
    return {
        token.casefold()
        for token in WORD_RE.findall(value)
        if token.casefold() not in STOPWORDS
    }


def _records(retrieved: tuple[RetrievedEvidence, ...]) -> tuple[_EvidenceRecord, ...]:
    return tuple(
        _EvidenceRecord(
            source_id=item.chunk.source_id,
            text=item.text,
            locator=item.chunk.locator,
            char_start=item.chunk.char_start,
            char_end=item.chunk.char_end,
            rank=rank,
            score=item.score,
        )
        for rank, item in enumerate(retrieved, 1)
    )


def _judge_input(
    case: EvalCase,
    source: SourceFixture,
    retrieved: tuple[RetrievedEvidence, ...],
) -> JudgeInput:
    if case.claim_text is None:
        raise SciFactRunnerError("SciFact pair profile requires claim_text")
    signals = deterministic_signals(
        case.claim_text,
        retrieved[0].text if retrieved else "",
        bounded_evidence_context=bounded_evidence_text(retrieved),
    )
    return JudgeInput(
        claim=AtomicClaim(
            claim_id=f"eval_{case.case_id}",
            text=case.claim_text,
            file=case.document_path or f"eval_sets/scifact/{case.case_id}.md",
            line_start=case.claim_line,
            line_end=case.claim_line,
            claim_type=case.claim_type,
            slots={},
            checkability=Checkability.CHECKABLE,
            citation_urls=(source.url,),
        ),
        evidence=retrieved,
        source=SourceMetadata(
            source_id=source.source_id,
            url=source.url,
            title=source.source_id,
            retrieved_at=datetime(2026, 9, 3, tzinfo=UTC),
            content_hash=source.content_hash,
            mime_type="text/plain",
            status="ok",
        ),
        signals=signals,
    )


def _lexical_prediction(input_data: JudgeInput, case: EvalCase) -> _Prediction:
    retrieved = input_data.evidence
    records = _records(retrieved)
    if not retrieved:
        return _Prediction(
            case_id=case.case_id,
            baseline="lexical_rules",
            relation=Relation.NOT_IN_SOURCE,
            confidence=0.0,
            reason="no_candidate_evidence",
            retrieved=records,
            selected=(),
        )
    best = retrieved[0]
    claim_words = _words(input_data.claim.text)
    overlap = len(claim_words & _words(best.text)) / max(len(claim_words), 1)
    signal_codes = {signal.code for signal in input_data.signals}
    slot_mismatch = bool(
        signal_codes & {"numeric_mismatch", "date_mismatch", "version_mismatch"}
    )
    if slot_mismatch and overlap >= 0.25:
        relation = Relation.CONTRADICTED
        reason = "deterministic_slot_mismatch"
    elif "negation_mismatch" in signal_codes and overlap >= 0.25:
        relation = Relation.CONTRADICTED
        reason = "deterministic_negation_mismatch"
    elif overlap >= 0.75:
        relation = Relation.ENTAILED
        reason = "strong_lexical_support"
    else:
        relation = Relation.NOT_IN_SOURCE
        reason = "insufficient_full_claim_coverage"
    selected = (records[0],) if relation in SUBSTANTIVE_RELATIONS else ()
    return _Prediction(
        case_id=case.case_id,
        baseline="lexical_rules",
        relation=relation,
        confidence=min(max(overlap, 0.0), 0.99) if selected else 0.0,
        reason=reason,
        retrieved=records,
        selected=selected,
    )


def _nli_prediction(
    input_data: JudgeInput,
    case: EvalCase,
    *,
    scorer: SemanticNliScorer,
    config: SciFactJudgeConfig,
    batch_size: int,
    canonical_model_id: str,
) -> _Prediction:
    decision = judge_scifact_3way(input_data, scorer=scorer, config=config)
    selected = tuple(
        _EvidenceRecord(
            source_id=item.source_id,
            text=item.text,
            locator=item.locator,
            char_start=item.char_start,
            char_end=item.char_end,
            rank=item.rank,
        )
        for item in decision.selected_evidence
    )
    return _Prediction(
        case_id=case.case_id,
        baseline="retrieval_judge_scifact_nli",
        relation=decision.relation,
        confidence=decision.confidence,
        reason=decision.reason_code,
        retrieved=_records(input_data.evidence),
        selected=selected,
        local_nli_score_many_calls=int(decision.semantic_pair_count > 0),
        local_nli_pair_count=decision.semantic_pair_count,
        local_nli_subset_count=decision.evaluated_subsets,
        local_nli_batch_count=math.ceil(decision.semantic_pair_count / batch_size),
        model_id=canonical_model_id,
        model_revision=decision.model_revision,
    )


def _validate_prediction(
    prediction: _Prediction,
    *,
    case: EvalCase,
    source: SourceFixture,
) -> None:
    if prediction.case_id != case.case_id or prediction.baseline not in BASELINES:
        raise SciFactRunnerError("prediction identity mismatch")
    if prediction.relation not in SCIFACT_RELATIONS:
        raise SciFactRunnerError("prediction escaped the SciFact three-way profile")
    if prediction.external_provider_calls != 0:
        raise SciFactRunnerError("SciFact local-NLI run attempted a provider call")
    if len(prediction.retrieved) > RETRIEVAL_TOP_K:
        raise SciFactRunnerError("retrieval budget exceeded")
    if prediction.relation in SUBSTANTIVE_RELATIONS and not prediction.selected:
        raise SciFactRunnerError("substantive prediction lacks evidence")
    if prediction.relation is Relation.NOT_IN_SOURCE and prediction.selected:
        raise SciFactRunnerError("NOINFO prediction must not select evidence")
    retrieved_identities = {
        (item.rank, item.locator, item.char_start, item.char_end, item.text)
        for item in prediction.retrieved
    }
    for expected_rank, item in enumerate(prediction.retrieved, 1):
        if item.rank != expected_rank:
            raise SciFactRunnerError("retrieval ranks must be contiguous")
    for item in (*prediction.retrieved, *prediction.selected):
        if (
            item.source_id != source.source_id
            or item.char_start < 0
            or item.char_end > len(source.content)
            or source.content[item.char_start : item.char_end] != item.text
        ):
            raise SciFactRunnerError("structured evidence is not an exact source span")
    if any(
        (item.rank, item.locator, item.char_start, item.char_end, item.text)
        not in retrieved_identities
        for item in prediction.selected
    ):
        raise SciFactRunnerError("selected evidence was not retrieved")


def _scorer_length_telemetry(
    scorer: SemanticNliScorer,
    *,
    reportable: bool,
) -> dict[str, int | bool | None]:
    values = {
        "max_input_tokens": getattr(scorer, "max_input_tokens", None),
        "max_observed_input_tokens": getattr(scorer, "max_observed_input_tokens", None),
        "rejected_overlength_pair_count": getattr(
            scorer, "rejected_overlength_pair_count", None
        ),
    }
    valid = all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
        for value in values.values()
    )
    if reportable:
        if not valid or not values["max_input_tokens"]:
            raise SciFactRunnerError(
                "reportable scorer did not expose valid input-length telemetry"
            )
        if values["rejected_overlength_pair_count"] != 0:
            raise SciFactRunnerError("reportable run rejected overlength NLI pairs")
        if values["max_observed_input_tokens"] > values["max_input_tokens"]:
            raise SciFactRunnerError("observed NLI input exceeds the declared limit")
    return {"available": valid, **values}


def _result_row(
    case: EvalCase,
    prediction: _Prediction,
    *,
    evaluation_status: dict[str, Any],
) -> dict[str, Any]:
    first = prediction.selected[0] if prediction.selected else None
    reportable = evaluation_status["reportable"] is True
    return {
        "case_id": case.case_id,
        "baseline": prediction.baseline,
        "predicted_relation": prediction.relation.value,
        "confidence": prediction.confidence,
        "reason": prediction.reason,
        "predicted_evidence_span": first.text if first is not None else None,
        "predicted_evidence": [item.as_dict() for item in prediction.selected],
        "retrieved_texts": [item.text for item in prediction.retrieved],
        "retrieved_evidence": [item.as_dict() for item in prediction.retrieved],
        "retrieval_rank": first.rank if first is not None else None,
        "extraction_count": 0,
        "predicted_line": case.claim_line,
        "external_provider_calls": prediction.external_provider_calls,
        "local_nli_score_many_calls": prediction.local_nli_score_many_calls,
        "local_nli_pair_count": prediction.local_nli_pair_count,
        "local_nli_subset_count": prediction.local_nli_subset_count,
        "local_nli_batch_count": prediction.local_nli_batch_count,
        "model_calls": prediction.local_nli_batch_count,
        "model_call_semantics": "local_nli_forward_batch",
        "model_usage_status": "not_applicable",
        "model_provider": (
            "local_transformers_nli"
            if reportable and prediction.model_id
            else "test_double_nonreportable"
            if prediction.model_id
            else None
        ),
        "model_id": prediction.model_id if reportable else None,
        "model_revision": prediction.model_revision if reportable else None,
        "evaluation_status": evaluation_status,
        "error_stage": (
            "judge" if prediction.relation is Relation.NOT_IN_SOURCE else "none"
        ),
        "source_id": case.source_id,
        "split": case.split,
    }


def run_pack(
    pack_dir: Path | str,
    out_dir: Path | str,
    *,
    nli_model_path: Path | str,
    nli_upstream_model_id: str,
    nli_revision: str,
    nli_weights_sha256: str,
    nli_batch_size: int = 16,
    nli_device: str = "cpu",
    scorer: SemanticNliScorer | None = None,
    judge_config: SciFactJudgeConfig | None = None,
    expected_case_count: int = 300,
) -> Path:
    """Run strict three-way rules/NLI paths and write scorer-compatible rows."""

    pack = Path(pack_dir).resolve()
    output = Path(out_dir).resolve()
    cases_path = pack / "cases.jsonl"
    if expected_case_count <= 0:
        raise SciFactRunnerError("expected_case_count must be positive")
    _assert_output_available(output)
    identity = _model_identity(
        nli_model_path,
        upstream_model_id=nli_upstream_model_id,
        revision=nli_revision,
        weights_sha256=nli_weights_sha256,
        batch_size=nli_batch_size,
        device=nli_device,
        require_hf_metadata=scorer is None,
    )
    loaded = load_dataset(cases_path)
    if len(loaded.cases) != expected_case_count:
        raise SciFactRunnerError(
            f"expected {expected_case_count} cases, found {len(loaded.cases)}"
        )
    if any(case.split != "dev" for case in loaded.cases):
        raise SciFactRunnerError("SciFact public-dev runner accepts only dev cases")
    chunks_by_case, sentence_index_path = _sentence_chunks(pack, loaded)
    pack_manifest = _validate_pack_manifest(pack, sentence_index_path)
    active_scorer = scorer or TransformersNliScorer(
        identity.local_path,
        revision=identity.revision,
        batch_size=identity.batch_size,
        device=identity.device,
        local_files_only=True,
    )
    reportable = scorer is None and type(active_scorer) is TransformersNliScorer
    evaluation_status = {
        "reportable": reportable,
        "mode": (
            "local_transformers_nli_reportable"
            if reportable
            else "test_double_nonreportable"
        ),
    }
    raw_scorer_id = str(getattr(active_scorer, "model_id", ""))
    scorer_identity_ok = raw_scorer_id == identity.upstream_model_id
    with suppress(OSError):
        scorer_identity_ok = scorer_identity_ok or (
            Path(raw_scorer_id).resolve() == identity.local_path
        )
    if (
        not scorer_identity_ok
        or str(getattr(active_scorer, "revision", "")) != identity.revision
    ):
        raise SciFactRunnerError(
            "semantic scorer identity disagrees with CLI provenance"
        )
    if getattr(active_scorer, "batch_size", identity.batch_size) != identity.batch_size:
        raise SciFactRunnerError("semantic scorer batch size disagrees with run config")
    config = judge_config or SciFactJudgeConfig()

    rows: list[dict[str, Any]] = []
    telemetry = {
        "external_provider_calls": 0,
        "local_nli_score_many_calls": 0,
        "local_nli_pair_count": 0,
        "local_nli_subset_count": 0,
        "local_nli_batch_count": 0,
    }
    counts_by_baseline = {baseline: 0 for baseline in BASELINES}
    for case in loaded.cases:
        source = loaded.sources[case.source_id]
        retrieved = LexicalRetriever(
            chunks_by_case[case.case_id], neighbor_window=0
        ).search(case.claim_text or "", top_k=RETRIEVAL_TOP_K)
        input_data = _judge_input(case, source, retrieved)
        predictions = (
            _lexical_prediction(input_data, case),
            _nli_prediction(
                input_data,
                case,
                scorer=active_scorer,
                config=config,
                batch_size=identity.batch_size,
                canonical_model_id=identity.upstream_model_id,
            ),
        )
        for prediction in predictions:
            _validate_prediction(prediction, case=case, source=source)
            rows.append(
                _result_row(
                    case,
                    prediction,
                    evaluation_status=evaluation_status,
                )
            )
            counts_by_baseline[prediction.baseline] += 1
            telemetry["external_provider_calls"] += prediction.external_provider_calls
            telemetry["local_nli_score_many_calls"] += (
                prediction.local_nli_score_many_calls
            )
            telemetry["local_nli_pair_count"] += prediction.local_nli_pair_count
            telemetry["local_nli_subset_count"] += prediction.local_nli_subset_count
            telemetry["local_nli_batch_count"] += prediction.local_nli_batch_count

    expected_rows = len(loaded.cases) * len(BASELINES)
    if len(rows) != expected_rows or telemetry["external_provider_calls"] != 0:
        raise SciFactRunnerError("runner row/provider-call accounting failed")
    length_telemetry = _scorer_length_telemetry(
        active_scorer,
        reportable=reportable,
    )
    allowed = {relation.value for relation in SCIFACT_ALLOWED_RELATIONS}
    if {row["predicted_relation"] for row in rows} - allowed:
        raise SciFactRunnerError("serialized results escaped the three-way profile")
    for case_id in {row["case_id"] for row in rows}:
        variants = {
            json.dumps(row["retrieved_evidence"], sort_keys=True)
            for row in rows
            if row["case_id"] == case_id
        }
        if len(variants) != 1:
            raise SciFactRunnerError("baselines did not reuse identical retrieval")

    results_bytes = _jsonl_bytes(rows)
    results_sha = _sha256_bytes(results_bytes)
    manifest = {
        "runner_version": RUNNER_VERSION,
        "task": "scifact_public_dev_oracle_cited_abstract_pair",
        "evaluation_status": evaluation_status,
        "task_profile": {
            "version": TASK_PROFILE_VERSION,
            "allowed_relations": [
                relation.value for relation in SCIFACT_ALLOWED_RELATIONS
            ],
            "claim_checkability": "forced_checkable_by_scifact_task_contract",
            "partial_support_policy": (
                "insufficient full-claim coverage becomes not_in_source during "
                "prediction; no scorer label mapping"
            ),
            "source_unavailable_policy": "fail_closed",
        },
        "baselines": list(BASELINES),
        "retrieval_budget": {
            "top_k_per_case": RETRIEVAL_TOP_K,
            "same_retrieval_object_reused_by_both_paths": True,
            "chunk_unit": "official_scifact_abstract_sentence",
            "sentence_index": SENTENCE_INDEX_RELATIVE,
        },
        "counts": {
            "cases": len(loaded.cases),
            "rows": len(rows),
            "rows_by_baseline": counts_by_baseline,
            **telemetry,
        },
        "local_nli": {
            "offline_only": True,
            "backend_class": (
                f"{type(active_scorer).__module__}.{type(active_scorer).__qualname__}"
            ),
            "upstream_model_id": identity.upstream_model_id,
            "local_model_directory": identity.local_path.name,
            "absolute_local_path_recorded": False,
            "revision": identity.revision,
            "weights_sha256": identity.weights_sha256,
            "weights_hash_scope": identity.weights_hash_scope,
            "identity_binding_sha256": identity.identity_binding_sha256,
            "file_hash_scope": (
                "regular files directly under the supplied model directory; "
                "nested caches, lock files, and incomplete files excluded"
            ),
            "batch_size": identity.batch_size,
            "device": identity.device,
            "model_file_sha256": identity.file_sha256,
            "hf_hub_download_metadata": identity.hf_metadata,
            "hf_hub_metadata_sha256": identity.hf_metadata_sha256,
            "provenance_binding_scope": (
                "Consistency check against local hf_hub download metadata; this "
                "does not cryptographically authenticate the remote repository."
            ),
            "package_versions": _package_versions(),
            "judge_policy_version": SCIFACT_JUDGE_POLICY_VERSION,
            "judge_config": asdict(config),
            "input_length_telemetry": length_telemetry,
        },
        "inputs": {
            "pack_schema_version": pack_manifest["schema_version"],
            "sha256": _input_hashes(pack, cases_path, sentence_index_path, loaded),
            "loaded_dataset_raw_sha256": loaded.raw_hash,
            "loaded_split_sha256": loaded.split_hash,
        },
        "outputs": {
            "sha256": {"eval_results.jsonl": results_sha},
            "hash_scope": (
                "run_manifest.json is excluded because a manifest cannot contain "
                "its own final file hash"
            ),
        },
        "runtime": {
            "python": {
                "implementation": platform.python_implementation(),
                "version": platform.python_version(),
                "executable": sys.executable,
            },
            "git": _git_metadata(),
        },
        "boundaries": [
            "Public SciFact dev is not a blind holdout or leaderboard result.",
            (
                "Each case receives its oracle cited abstract set; full-corpus "
                "document retrieval is not measured."
            ),
            "Claims are supplied by the adapter, so Claim Miner is not evaluated.",
            (
                "No external provider is called; the candidate performs the "
                "recorded number of local NLI forward batches."
            ),
            (
                "The NLI checkpoint is trained on general NLI rather than SciFact; "
                "these thresholds have not been calibrated on SciFact train."
            ),
            (
                "Selecting the maximum over as many as 31 subsets creates a "
                "multiple-comparison risk, and tokenizer truncation can omit text."
            ),
            (
                "Prediction-time sentence identity comes only from the label-free "
                "sentence index; gold_rationales.jsonl is not read by this runner."
            ),
        ],
    }

    _write_exclusive(output / "eval_results.jsonl", results_bytes)
    _write_exclusive(output / "run_manifest.json", _json_bytes(manifest))
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--nli-model-path", type=Path, required=True)
    parser.add_argument("--nli-upstream-model-id", required=True)
    parser.add_argument(
        "--nli-revision",
        required=True,
        help="Immutable 40-character upstream commit revision.",
    )
    parser.add_argument("--nli-weights-sha256", required=True)
    parser.add_argument("--nli-batch-size", type=int, default=16)
    parser.add_argument("--nli-device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    print(
        run_pack(
            args.pack,
            args.out,
            nli_model_path=args.nli_model_path,
            nli_upstream_model_id=args.nli_upstream_model_id,
            nli_revision=args.nli_revision,
            nli_weights_sha256=args.nli_weights_sha256,
            nli_batch_size=args.nli_batch_size,
            nli_device=args.nli_device,
        )
    )


if __name__ == "__main__":
    main()
