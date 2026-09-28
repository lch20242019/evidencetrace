"""Task-scoped semantic Judge for the SciFact three-way evaluation profile.

This module is intentionally separate from :mod:`evidencetrace.agents.judge`.
SciFact has a closed SUPPORT / CONTRADICT / NOINFO taxonomy, while the product
Judge serves a six-relation Chinese technical-document workflow.  Keeping the
profile here prevents benchmark-specific policy from changing production
behaviour.

The decision policy scores every retrieved singleton and every non-empty
subset of the bounded top five.  The semantic scorer is injected so the core
package does not acquire a mandatory Torch/Transformers dependency.  A lazy
Transformers adapter is provided for isolated evaluation environments.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Literal, Protocol

from evidencetrace.audit_models import JudgeInput, RetrievedEvidence
from evidencetrace.models import Checkability, Relation

SCIFACT_JUDGE_POLICY_VERSION = "scifact-three-way-semantic-aggregation-v1"
SCIFACT_MAX_EVIDENCE_ITEMS = 5
SCIFACT_MAX_CANDIDATE_SUBSETS = (1 << SCIFACT_MAX_EVIDENCE_ITEMS) - 1
SCIFACT_RELATIONS = frozenset(
    {Relation.ENTAILED, Relation.CONTRADICTED, Relation.NOT_IN_SOURCE}
)

SciFactJudgeErrorCode = Literal[
    "invalid_profile_input",
    "semantic_backend_unavailable",
    "semantic_input_limit_unavailable",
    "semantic_input_too_long",
    "semantic_score_count_mismatch",
    "semantic_score_type_invalid",
    "ambiguous_evidence_offset",
    "semantic_label_mapping_invalid",
    "semantic_inference_failed",
]
SciFactDecisionReason = Literal[
    "no_candidate_evidence",
    "semantic_entailment",
    "semantic_contradiction",
    "semantic_neutral",
    "mixed_semantic_conflict",
    "contradiction_gate_rejected",
]


class SciFactJudgeError(ValueError):
    """Payload-free failure raised by the task-scoped Judge."""

    def __init__(self, code: SciFactJudgeErrorCode) -> None:
        super().__init__("SciFact Judge contract failed")
        self.code = code


@dataclass(frozen=True, slots=True)
class NliScores:
    """Normalized probabilities for one ``(premise, hypothesis)`` pair."""

    entailment: float
    contradiction: float
    neutral: float

    def __post_init__(self) -> None:
        values = (self.entailment, self.contradiction, self.neutral)
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
            raise ValueError("NLI scores must be finite probabilities")
        if not math.isclose(sum(values), 1.0, rel_tol=1e-5, abs_tol=1e-5):
            raise ValueError("NLI probabilities must sum to one")


class SemanticNliScorer(Protocol):
    """Minimal semantic backend; premise is evidence and hypothesis is claim."""

    model_id: str
    revision: str

    def score_many(self, pairs: Sequence[tuple[str, str]]) -> Sequence[NliScores]: ...


@dataclass(frozen=True, slots=True)
class SciFactJudgeConfig:
    """Frozen, task-visible aggregation thresholds.

    These conservative defaults are a decision contract, not a claim of dev-set
    calibration.  Any later calibration must use training data and must be
    recorded in the run manifest rather than silently changing these values.
    """

    entailment_min_probability: float = 0.50
    entailment_min_margin_over_neutral: float = 0.00
    contradiction_min_probability: float = 0.60
    contradiction_min_margin: float = 0.10
    mixed_relation_margin: float = 0.05

    def __post_init__(self) -> None:
        values = (
            self.entailment_min_probability,
            self.entailment_min_margin_over_neutral,
            self.contradiction_min_probability,
            self.contradiction_min_margin,
            self.mixed_relation_margin,
        )
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
            raise ValueError("SciFact Judge thresholds must be finite probabilities")


@dataclass(frozen=True, slots=True)
class SciFactSelectedEvidence:
    """One exact, locally owned evidence sentence selected by the Judge."""

    rank: int
    source_id: str
    text: str
    locator: str
    char_start: int
    char_end: int

    def __post_init__(self) -> None:
        if not 1 <= self.rank <= SCIFACT_MAX_EVIDENCE_ITEMS:
            raise ValueError("selected evidence rank is outside the top-five budget")
        if (
            not self.source_id.strip()
            or not self.text.strip()
            or not self.locator.strip()
        ):
            raise ValueError("selected evidence identity fields must not be blank")
        if self.char_start < 0 or self.char_end <= self.char_start:
            raise ValueError("selected evidence offsets must be ordered")
        if self.char_end - self.char_start != len(self.text):
            raise ValueError("selected evidence offsets must exactly bound its text")


@dataclass(frozen=True, slots=True)
class SciFactJudgeDecision:
    """Closed-taxonomy decision plus exact multi-sentence evidence."""

    relation: Relation
    confidence: float
    reason_code: SciFactDecisionReason
    selected_evidence: tuple[SciFactSelectedEvidence, ...]
    evaluated_singletons: int
    evaluated_subsets: int
    semantic_pair_count: int
    model_id: str
    model_revision: str
    advisory_signal_codes: tuple[str, ...] = ()
    policy_version: str = SCIFACT_JUDGE_POLICY_VERSION

    def __post_init__(self) -> None:
        if self.relation not in SCIFACT_RELATIONS:
            raise ValueError("SciFact decision escaped the fixed three-way taxonomy")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("SciFact decision confidence must be a probability")
        if self.relation in {Relation.ENTAILED, Relation.CONTRADICTED}:
            if not self.selected_evidence:
                raise ValueError("substantive SciFact decisions require exact evidence")
        elif self.selected_evidence:
            raise ValueError("NOINFO decisions must not claim supporting evidence")
        if not 0 <= self.evaluated_singletons <= SCIFACT_MAX_EVIDENCE_ITEMS:
            raise ValueError("invalid evaluated singleton count")
        if not 0 <= self.evaluated_subsets <= SCIFACT_MAX_CANDIDATE_SUBSETS:
            raise ValueError("invalid evaluated subset count")
        if self.semantic_pair_count != self.evaluated_subsets:
            raise ValueError("semantic pair count must equal evaluated subset count")
        if not self.model_id.strip() or not self.model_revision.strip():
            raise ValueError("semantic model identity must not be blank")
        if tuple(sorted(set(self.advisory_signal_codes))) != self.advisory_signal_codes:
            raise ValueError("advisory signal codes must be sorted and unique")


@dataclass(frozen=True, slots=True)
class NliLabelIndices:
    """Explicit semantic label positions for a Transformers classifier."""

    entailment: int
    contradiction: int
    neutral: int

    def __post_init__(self) -> None:
        values = (self.entailment, self.contradiction, self.neutral)
        if any(value < 0 for value in values) or len(set(values)) != 3:
            raise ValueError("NLI label indices must be distinct and non-negative")


@dataclass(frozen=True, slots=True)
class _ScoredSubset:
    indices: tuple[int, ...]
    scores: NliScores


def _candidate_indices(count: int) -> tuple[tuple[int, ...], ...]:
    return tuple(
        candidate
        for size in range(1, count + 1)
        for candidate in combinations(range(count), size)
    )


def _premise(evidence: tuple[RetrievedEvidence, ...], indices: tuple[int, ...]) -> str:
    # Retrieval rank chooses candidates, while source offsets restore discourse
    # order inside a multi-sentence premise.
    ordered = sorted(
        (evidence[index] for index in indices),
        key=lambda item: (item.chunk.char_start, item.chunk.char_end),
    )
    return "\n".join(item.text for item in ordered)


def _best_for_relation(
    scored: tuple[_ScoredSubset, ...], relation: Relation, config: SciFactJudgeConfig
) -> _ScoredSubset | None:
    def probability(item: _ScoredSubset) -> float:
        return (
            item.scores.entailment
            if relation is Relation.ENTAILED
            else item.scores.contradiction
        )

    eligible: list[_ScoredSubset] = []
    for item in scored:
        value = probability(item)
        other = (
            item.scores.contradiction
            if relation is Relation.ENTAILED
            else item.scores.entailment
        )
        if relation is Relation.ENTAILED:
            threshold = config.entailment_min_probability
            margin = config.entailment_min_margin_over_neutral
        else:
            threshold = config.contradiction_min_probability
            margin = config.contradiction_min_margin
        if (
            value >= threshold
            and value >= other
            and value - item.scores.neutral >= margin
        ):
            eligible.append(item)
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda item: (
            probability(item),
            probability(item) - item.scores.neutral,
            -len(item.indices),
            tuple(-index for index in item.indices),
        ),
    )


def _selected(
    evidence: tuple[RetrievedEvidence, ...], indices: tuple[int, ...]
) -> tuple[SciFactSelectedEvidence, ...]:
    values: list[SciFactSelectedEvidence] = []
    for index in indices:
        item = evidence[index]
        relative_start = item.chunk.text.find(item.text)
        if (
            relative_start < 0
            or item.chunk.text.find(item.text, relative_start + 1) >= 0
        ):
            raise SciFactJudgeError("ambiguous_evidence_offset")
        char_start = item.chunk.char_start + relative_start
        values.append(
            SciFactSelectedEvidence(
                rank=index + 1,
                source_id=item.chunk.source_id,
                text=item.text,
                locator=item.chunk.locator,
                char_start=char_start,
                char_end=char_start + len(item.text),
            )
        )
    return tuple(values)


def _neutral_confidence(scored: tuple[_ScoredSubset, ...]) -> float:
    neutral_winners = [
        item.scores.neutral
        for item in scored
        if item.scores.neutral >= max(item.scores.entailment, item.scores.contradiction)
    ]
    return max(neutral_winners, default=0.0)


def judge_scifact_3way(
    input_data: JudgeInput,
    *,
    scorer: SemanticNliScorer,
    config: SciFactJudgeConfig | None = None,
) -> SciFactJudgeDecision:
    """Judge one SciFact pair without changing the product Judge's policy.

    All deterministic mismatch signals are retained only as diagnostics.  They
    never vote for CONTRADICTED.  A contradiction must independently clear the
    semantic probability and margin gate on the exact evidence subset.
    """

    active_config = config or SciFactJudgeConfig()
    if (
        input_data.claim.checkability is not Checkability.CHECKABLE
        or input_data.source.status != "ok"
        or len(input_data.evidence) > SCIFACT_MAX_EVIDENCE_ITEMS
    ):
        raise SciFactJudgeError("invalid_profile_input")
    model_id = str(getattr(scorer, "model_id", ""))
    model_revision = str(getattr(scorer, "revision", ""))
    if not model_id.strip() or not model_revision.strip():
        raise SciFactJudgeError("invalid_profile_input")
    advisory_codes = tuple(sorted({signal.code for signal in input_data.signals}))
    evidence = input_data.evidence
    if not evidence:
        return SciFactJudgeDecision(
            relation=Relation.NOT_IN_SOURCE,
            confidence=0.0,
            reason_code="no_candidate_evidence",
            selected_evidence=(),
            evaluated_singletons=0,
            evaluated_subsets=0,
            semantic_pair_count=0,
            model_id=model_id,
            model_revision=model_revision,
            advisory_signal_codes=advisory_codes,
        )

    candidates = _candidate_indices(len(evidence))
    pairs = tuple(
        (_premise(evidence, indices), input_data.claim.text) for indices in candidates
    )
    try:
        raw_scores = tuple(scorer.score_many(pairs))
    except SciFactJudgeError:
        raise
    except Exception:
        raise SciFactJudgeError("semantic_inference_failed") from None
    if len(raw_scores) != len(candidates):
        raise SciFactJudgeError("semantic_score_count_mismatch")
    if any(not isinstance(item, NliScores) for item in raw_scores):
        raise SciFactJudgeError("semantic_score_type_invalid")
    scored = tuple(
        _ScoredSubset(indices=indices, scores=scores)
        for indices, scores in zip(candidates, raw_scores, strict=True)
    )
    best_entailment = _best_for_relation(scored, Relation.ENTAILED, active_config)
    best_contradiction = _best_for_relation(
        scored, Relation.CONTRADICTED, active_config
    )
    common = {
        "evaluated_singletons": len(evidence),
        "evaluated_subsets": len(candidates),
        "semantic_pair_count": len(candidates),
        "model_id": model_id,
        "model_revision": model_revision,
        "advisory_signal_codes": advisory_codes,
    }

    if best_entailment is not None and best_contradiction is not None:
        entailment_probability = best_entailment.scores.entailment
        contradiction_probability = best_contradiction.scores.contradiction
        if (
            abs(entailment_probability - contradiction_probability)
            <= active_config.mixed_relation_margin
        ):
            return SciFactJudgeDecision(
                relation=Relation.NOT_IN_SOURCE,
                confidence=0.0,
                reason_code="mixed_semantic_conflict",
                selected_evidence=(),
                **common,
            )
        if contradiction_probability > entailment_probability:
            winner = best_contradiction
            relation = Relation.CONTRADICTED
        else:
            winner = best_entailment
            relation = Relation.ENTAILED
    elif best_entailment is not None:
        winner = best_entailment
        relation = Relation.ENTAILED
    elif best_contradiction is not None:
        winner = best_contradiction
        relation = Relation.CONTRADICTED
    else:
        near_contradiction = max(
            scored,
            key=lambda item: item.scores.contradiction,
        )
        contradiction_won = near_contradiction.scores.contradiction >= max(
            near_contradiction.scores.entailment,
            near_contradiction.scores.neutral,
        )
        return SciFactJudgeDecision(
            relation=Relation.NOT_IN_SOURCE,
            confidence=_neutral_confidence(scored),
            reason_code=(
                "contradiction_gate_rejected"
                if contradiction_won
                else "semantic_neutral"
            ),
            selected_evidence=(),
            **common,
        )

    probability = (
        winner.scores.entailment
        if relation is Relation.ENTAILED
        else winner.scores.contradiction
    )
    return SciFactJudgeDecision(
        relation=relation,
        confidence=probability,
        reason_code=(
            "semantic_entailment"
            if relation is Relation.ENTAILED
            else "semantic_contradiction"
        ),
        selected_evidence=_selected(evidence, winner.indices),
        **common,
    )


def _infer_label_indices(id2label: dict[Any, Any]) -> NliLabelIndices:
    found: dict[str, int] = {}
    aliases = {
        "entailment": ("entail", "support"),
        "contradiction": ("contrad", "refute"),
        "neutral": ("neutral", "noinfo", "not_enough_info"),
    }
    for raw_index, raw_label in id2label.items():
        label = str(raw_label).casefold().replace(" ", "_")
        for canonical, terms in aliases.items():
            if any(term in label for term in terms):
                found[canonical] = int(raw_index)
    if set(found) != set(aliases):
        raise SciFactJudgeError("semantic_label_mapping_invalid")
    return NliLabelIndices(
        entailment=found["entailment"],
        contradiction=found["contradiction"],
        neutral=found["neutral"],
    )


_TRANSFORMERS_UNBOUNDED_LENGTH_SENTINEL = 10**20


def _usable_input_limit(value: Any) -> int | None:
    """Return a concrete token limit, excluding Transformers' huge sentinel."""

    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if value <= 0 or value >= _TRANSFORMERS_UNBOUNDED_LENGTH_SENTINEL:
        return None
    return value


def _effective_max_input_tokens(tokenizer: Any, model: Any) -> int:
    """Resolve the strictest finite limit advertised by tokenizer and model."""

    candidates = (
        _usable_input_limit(getattr(tokenizer, "model_max_length", None)),
        _usable_input_limit(
            getattr(getattr(model, "config", None), "max_position_embeddings", None)
        ),
    )
    concrete = tuple(value for value in candidates if value is not None)
    if not concrete:
        raise SciFactJudgeError("semantic_input_limit_unavailable")
    return min(concrete)


def _pair_input_lengths(
    tokenizer: Any, pairs: Sequence[tuple[str, str]]
) -> tuple[int, ...]:
    """Tokenize without truncation and count complete pair inputs."""

    if not pairs:
        return ()
    encoded = tokenizer(
        [premise for premise, _ in pairs],
        [hypothesis for _, hypothesis in pairs],
        padding=False,
        truncation=False,
    )
    input_ids = encoded["input_ids"]
    if len(input_ids) != len(pairs):
        raise ValueError("tokenizer returned the wrong number of NLI inputs")
    lengths = tuple(len(row) for row in input_ids)
    if any(length <= 0 for length in lengths):
        raise ValueError("tokenizer returned an empty NLI input")
    return lengths


class TransformersNliScorer:
    """Lazy local adapter for a three-label Transformers NLI checkpoint.

    ``local_files_only`` defaults to true so importing or constructing normal
    EvidenceTrace components can never trigger an implicit network download.
    The model revision is required and exposed for run-manifest provenance.
    """

    def __init__(
        self,
        model: str | Path,
        *,
        revision: str,
        label_indices: NliLabelIndices | None = None,
        batch_size: int = 16,
        device: str = "cpu",
        local_files_only: bool = True,
    ) -> None:
        if not str(model).strip() or not revision.strip() or batch_size <= 0:
            raise ValueError("invalid Transformers NLI configuration")
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError:
            raise SciFactJudgeError("semantic_backend_unavailable") from None

        self.model_id = str(model)
        self.revision = revision
        self.batch_size = batch_size
        self.device = device
        self._torch = torch
        load_options: dict[str, Any] = {"local_files_only": local_files_only}
        if not Path(self.model_id).exists():
            load_options["revision"] = revision
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, **load_options
            )
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_id, **load_options
            )
            self._label_indices = label_indices or _infer_label_indices(
                dict(self._model.config.id2label)
            )
            label_count = int(self._model.config.num_labels)
            if label_count != 3 or (
                max(
                    self._label_indices.entailment,
                    self._label_indices.contradiction,
                    self._label_indices.neutral,
                )
                >= label_count
            ):
                raise SciFactJudgeError("semantic_label_mapping_invalid")
            self.max_input_tokens = _effective_max_input_tokens(
                self._tokenizer, self._model
            )
            self.max_observed_input_tokens = 0
            self.rejected_overlength_pair_count = 0
            self._model.to(device)
            self._model.eval()
        except SciFactJudgeError:
            raise
        except Exception:
            raise SciFactJudgeError("semantic_backend_unavailable") from None

    def score_many(self, pairs: Sequence[tuple[str, str]]) -> Sequence[NliScores]:
        values = tuple(pairs)
        if any(
            not premise.strip() or not hypothesis.strip()
            for premise, hypothesis in values
        ):
            raise ValueError("NLI pair text must not be blank")
        results: list[NliScores] = []
        try:
            input_lengths = _pair_input_lengths(self._tokenizer, values)
            if input_lengths:
                self.max_observed_input_tokens = max(
                    self.max_observed_input_tokens, max(input_lengths)
                )
            rejected_count = sum(
                length > self.max_input_tokens for length in input_lengths
            )
            if rejected_count:
                self.rejected_overlength_pair_count += rejected_count
                raise SciFactJudgeError("semantic_input_too_long")
            for start in range(0, len(values), self.batch_size):
                batch = values[start : start + self.batch_size]
                encoded = self._tokenizer(
                    [premise for premise, _ in batch],
                    [hypothesis for _, hypothesis in batch],
                    padding=True,
                    truncation=False,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                with self._torch.inference_mode():
                    logits = self._model(**encoded).logits
                    probabilities = self._torch.softmax(logits, dim=-1).cpu().tolist()
                for row in probabilities:
                    results.append(
                        NliScores(
                            entailment=float(row[self._label_indices.entailment]),
                            contradiction=float(row[self._label_indices.contradiction]),
                            neutral=float(row[self._label_indices.neutral]),
                        )
                    )
        except SciFactJudgeError:
            raise
        except Exception:
            raise SciFactJudgeError("semantic_inference_failed") from None
        return tuple(results)


__all__ = [
    "SCIFACT_JUDGE_POLICY_VERSION",
    "SCIFACT_MAX_CANDIDATE_SUBSETS",
    "SCIFACT_MAX_EVIDENCE_ITEMS",
    "SCIFACT_RELATIONS",
    "NliLabelIndices",
    "NliScores",
    "SciFactJudgeConfig",
    "SciFactJudgeDecision",
    "SciFactJudgeError",
    "SciFactSelectedEvidence",
    "SemanticNliScorer",
    "TransformersNliScorer",
    "judge_scifact_3way",
]
