from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from evidencetrace.agents.judge import ClaimJudgeAgent
from evidencetrace.audit_models import (
    CheckSignal,
    EvidenceChunk,
    JudgeInput,
    RetrievedEvidence,
)
from evidencetrace.eval.scifact_judge import (
    SCIFACT_MAX_CANDIDATE_SUBSETS,
    NliLabelIndices,
    NliScores,
    SciFactJudgeError,
    TransformersNliScorer,
    _effective_max_input_tokens,
    judge_scifact_3way,
)
from evidencetrace.models import AtomicClaim, Checkability, Relation, SourceMetadata

SOURCE_ID = "scifact_unit_source"
SOURCE_URL = "https://example.test/scifact"
NEUTRAL = NliScores(entailment=0.05, contradiction=0.05, neutral=0.90)


class RecordingScorer:
    model_id = "unit-nli"
    revision = "unit-revision"

    def __init__(
        self,
        responder: Callable[[str, str], NliScores] | None = None,
    ) -> None:
        self.responder = responder or (lambda _premise, _hypothesis: NEUTRAL)
        self.calls: list[tuple[tuple[str, str], ...]] = []

    def score_many(self, pairs: Sequence[tuple[str, str]]) -> Sequence[NliScores]:
        captured = tuple(pairs)
        self.calls.append(captured)
        return tuple(
            self.responder(premise, hypothesis) for premise, hypothesis in captured
        )


class _StubTensor:
    def __init__(self, batch_size: int) -> None:
        self.batch_size = batch_size

    def to(self, _device: str) -> _StubTensor:
        return self


class _StubTokenizer:
    def __init__(self, token_lengths: dict[str, int]) -> None:
        self.token_lengths = token_lengths
        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        premises: list[str],
        hypotheses: list[str],
        *,
        padding: bool,
        truncation: bool,
        return_tensors: str | None = None,
    ) -> dict[str, object]:
        assert len(premises) == len(hypotheses)
        self.calls.append(
            {
                "padding": padding,
                "truncation": truncation,
                "return_tensors": return_tensors,
            }
        )
        lengths = [self.token_lengths[premise] for premise in premises]
        if return_tensors is not None:
            return {"input_ids": _StubTensor(len(lengths))}
        return {"input_ids": [list(range(length)) for length in lengths]}


class _StubModel:
    def __init__(self) -> None:
        self.call_count = 0

    def __call__(self, *, input_ids: _StubTensor) -> SimpleNamespace:
        self.call_count += 1
        return SimpleNamespace(logits=input_ids.batch_size)


class _StubProbabilities:
    def __init__(self, batch_size: int) -> None:
        self.batch_size = batch_size

    def cpu(self) -> _StubProbabilities:
        return self

    def tolist(self) -> list[list[float]]:
        return [[0.70, 0.10, 0.20] for _ in range(self.batch_size)]


class _StubInferenceMode:
    def __enter__(self) -> None:
        return None

    def __exit__(
        self,
        _exc_type: object,
        _exc_value: object,
        _traceback: object,
    ) -> None:
        return None


class _StubTorch:
    @staticmethod
    def inference_mode() -> _StubInferenceMode:
        return _StubInferenceMode()

    @staticmethod
    def softmax(logits: int, *, dim: int) -> _StubProbabilities:
        assert dim == -1
        return _StubProbabilities(logits)


def _transformers_stub(
    token_lengths: dict[str, int], *, max_input_tokens: int
) -> tuple[TransformersNliScorer, _StubTokenizer, _StubModel]:
    scorer = TransformersNliScorer.__new__(TransformersNliScorer)
    tokenizer = _StubTokenizer(token_lengths)
    model = _StubModel()
    scorer.model_id = "stub-nli"
    scorer.revision = "stub-revision"
    scorer.batch_size = 16
    scorer.device = "cpu"
    scorer._torch = _StubTorch()
    scorer._tokenizer = tokenizer
    scorer._model = model
    scorer._label_indices = NliLabelIndices(entailment=0, contradiction=1, neutral=2)
    scorer.max_input_tokens = max_input_tokens
    scorer.max_observed_input_tokens = 0
    scorer.rejected_overlength_pair_count = 0
    return scorer, tokenizer, model


def _input(
    texts: tuple[str, ...],
    *,
    claim_text: str = "The treatment improves the outcome.",
    checkability: Checkability = Checkability.CHECKABLE,
    status: str = "ok",
    signals: tuple[CheckSignal, ...] = (),
) -> JudgeInput:
    evidence: list[RetrievedEvidence] = []
    offset = 0
    for index, text in enumerate(texts, 1):
        chunk = EvidenceChunk(
            source_id=SOURCE_ID,
            url=SOURCE_URL,
            text=text,
            locator=f"doc 7 sentence {index - 1}",
            char_start=offset,
            char_end=offset + len(text),
        )
        evidence.append(RetrievedEvidence(chunk=chunk, text=text, score=6.0 - index))
        offset += len(text) + 1
    return JudgeInput(
        claim=AtomicClaim(
            claim_id="scifact_unit_claim",
            text=claim_text,
            file="eval_sets/scifact/unit.md",
            line_start=1,
            line_end=1,
            claim_type="official_scifact_scientific_claim",
            checkability=checkability,
            citation_urls=(SOURCE_URL,),
        ),
        evidence=tuple(evidence),
        source=SourceMetadata(
            source_id=SOURCE_ID,
            url=SOURCE_URL,
            title="SciFact unit source",
            retrieved_at=datetime(2026, 9, 3, tzinfo=UTC),
            content_hash="a" * 64,
            mime_type="text/plain",
            status=status,
        ),
        signals=signals,
    )


def test_scores_every_top_five_singleton_and_all_31_nonempty_subsets() -> None:
    texts = tuple(f"Scientific sentence {index}." for index in range(1, 6))
    target = texts[4]
    scorer = RecordingScorer(
        lambda premise, _claim: (
            NliScores(entailment=0.92, contradiction=0.03, neutral=0.05)
            if premise == target
            else NEUTRAL
        )
    )

    decision = judge_scifact_3way(_input(texts), scorer=scorer)

    assert len(scorer.calls) == 1
    pairs = scorer.calls[0]
    assert len(pairs) == SCIFACT_MAX_CANDIDATE_SUBSETS == 31
    assert [premise for premise, _ in pairs[:5]] == list(texts)
    assert all(
        hypothesis == "The treatment improves the outcome." for _, hypothesis in pairs
    )
    assert decision.relation is Relation.ENTAILED
    assert decision.evaluated_singletons == 5
    assert decision.evaluated_subsets == decision.semantic_pair_count == 31
    assert tuple(item.rank for item in decision.selected_evidence) == (5,)
    assert decision.selected_evidence[0].text == target


def test_joint_candidate_returns_separate_exact_noncontiguous_evidence() -> None:
    texts = (
        "Background sentence.",
        "Treatment A lowers the measured marker.",
        "Unrelated method sentence.",
        "The lower marker predicts recovery.",
        "Closing sentence.",
    )
    joint = f"{texts[1]}\n{texts[3]}"
    scorer = RecordingScorer(
        lambda premise, _claim: (
            NliScores(entailment=0.94, contradiction=0.02, neutral=0.04)
            if premise == joint
            else NEUTRAL
        )
    )

    decision = judge_scifact_3way(_input(texts), scorer=scorer)

    assert decision.relation is Relation.ENTAILED
    assert tuple(item.rank for item in decision.selected_evidence) == (2, 4)
    assert tuple(item.text for item in decision.selected_evidence) == (
        texts[1],
        texts[3],
    )
    assert tuple(item.locator for item in decision.selected_evidence) == (
        "doc 7 sentence 1",
        "doc 7 sentence 3",
    )
    assert (
        decision.selected_evidence[0].char_end
        < decision.selected_evidence[1].char_start
    )
    assert "\n" not in decision.selected_evidence[0].text
    assert "\n" not in decision.selected_evidence[1].text


def test_hard_slot_signal_is_advisory_and_cannot_create_contradiction() -> None:
    signal = CheckSignal(
        code="numeric_mismatch",
        detail="an identifier-like number differs",
        severity="error",
    )
    scorer = RecordingScorer()

    decision = judge_scifact_3way(
        _input(
            ("GATA-3 expression is discussed in the study.",),
            claim_text="The study discusses GATA-4 expression.",
            signals=(signal,),
        ),
        scorer=scorer,
    )

    assert decision.relation is Relation.NOT_IN_SOURCE
    assert decision.reason_code == "semantic_neutral"
    assert decision.advisory_signal_codes == ("numeric_mismatch",)


def test_semantically_confirmed_numeric_conflict_passes_conservative_gate() -> None:
    scorer = RecordingScorer(
        lambda _premise, _claim: NliScores(
            entailment=0.02,
            contradiction=0.90,
            neutral=0.08,
        )
    )

    decision = judge_scifact_3way(
        _input(
            ("Drug A reduced mortality by 30 percent.",),
            claim_text="Drug A reduced mortality by 20 percent.",
        ),
        scorer=scorer,
    )

    assert decision.relation is Relation.CONTRADICTED
    assert decision.reason_code == "semantic_contradiction"
    assert decision.selected_evidence[0].text == (
        "Drug A reduced mortality by 30 percent."
    )


def test_weak_contradiction_argmax_falls_back_instead_of_forcing_conflict() -> None:
    scorer = RecordingScorer(
        lambda _premise, _claim: NliScores(
            entailment=0.20,
            contradiction=0.55,
            neutral=0.25,
        )
    )

    decision = judge_scifact_3way(
        _input(("A possibly conflicting sentence.",)), scorer=scorer
    )

    assert decision.relation is Relation.NOT_IN_SOURCE
    assert decision.reason_code == "contradiction_gate_rejected"
    assert decision.selected_evidence == ()


def test_close_support_and_conflict_candidates_abstain() -> None:
    first = "Evidence supporting the claim."
    second = "Evidence contradicting the claim."

    def respond(premise: str, _claim: str) -> NliScores:
        if premise == first:
            return NliScores(entailment=0.80, contradiction=0.05, neutral=0.15)
        if premise == second:
            return NliScores(entailment=0.05, contradiction=0.78, neutral=0.17)
        return NEUTRAL

    decision = judge_scifact_3way(
        _input((first, second)), scorer=RecordingScorer(respond)
    )

    assert decision.relation is Relation.NOT_IN_SOURCE
    assert decision.reason_code == "mixed_semantic_conflict"
    assert decision.confidence == 0.0


def test_empty_retrieval_is_closed_noinfo_without_semantic_call() -> None:
    scorer = RecordingScorer()

    decision = judge_scifact_3way(_input(()), scorer=scorer)

    assert scorer.calls == []
    assert decision.relation is Relation.NOT_IN_SOURCE
    assert decision.reason_code == "no_candidate_evidence"
    assert decision.semantic_pair_count == 0


@pytest.mark.parametrize(
    ("checkability", "status"),
    [
        (Checkability.NOT_CHECKABLE, "ok"),
        (Checkability.CHECKABLE, "unavailable"),
    ],
)
def test_profile_rejects_noncheckable_or_unavailable_inputs(
    checkability: Checkability, status: str
) -> None:
    with pytest.raises(SciFactJudgeError) as raised:
        judge_scifact_3way(
            _input(
                ("One sentence.",),
                checkability=checkability,
                status=status,
            ),
            scorer=RecordingScorer(),
        )

    assert raised.value.code == "invalid_profile_input"


def test_score_count_mismatch_fails_closed() -> None:
    class ShortScorer(RecordingScorer):
        def score_many(self, pairs: Sequence[tuple[str, str]]) -> Sequence[NliScores]:
            return (NEUTRAL,)

    with pytest.raises(SciFactJudgeError) as raised:
        judge_scifact_3way(
            _input(("First.", "Second.")),
            scorer=ShortScorer(),
        )

    assert raised.value.code == "semantic_score_count_mismatch"


def test_ambiguous_exact_span_offset_fails_closed() -> None:
    base = _input(("placeholder",))
    chunk = EvidenceChunk(
        source_id=SOURCE_ID,
        url=SOURCE_URL,
        text="echo and echo",
        locator="doc 7 sentence 0",
        char_start=0,
        char_end=len("echo and echo"),
    )
    ambiguous = base.model_copy(
        update={"evidence": (RetrievedEvidence(chunk=chunk, text="echo", score=1.0),)}
    )
    scorer = RecordingScorer(
        lambda _premise, _claim: NliScores(
            entailment=0.90, contradiction=0.02, neutral=0.08
        )
    )

    with pytest.raises(SciFactJudgeError) as raised:
        judge_scifact_3way(ambiguous, scorer=scorer)

    assert raised.value.code == "ambiguous_evidence_offset"


def test_default_product_judge_remains_top_one_lexical_policy() -> None:
    input_data = _input(
        (
            "Completely unrelated background.",
            "The treatment improves the outcome.",
        )
    )

    product_decision = ClaimJudgeAgent().judge(input_data).verdict
    semantic_decision = judge_scifact_3way(
        input_data,
        scorer=RecordingScorer(
            lambda premise, _claim: (
                NliScores(entailment=0.95, contradiction=0.01, neutral=0.04)
                if premise == "The treatment improves the outcome."
                else NEUTRAL
            )
        ),
    )

    assert product_decision.relation is Relation.NOT_IN_SOURCE
    assert semantic_decision.relation is Relation.ENTAILED
    assert semantic_decision.selected_evidence[0].rank == 2


@pytest.mark.parametrize(
    "values",
    [
        (0.4, 0.4, 0.4),
        (float("nan"), 0.0, 1.0),
        (-0.1, 0.1, 1.0),
    ],
)
def test_nli_scores_require_normalized_finite_probabilities(
    values: tuple[float, float, float],
) -> None:
    with pytest.raises(ValueError):
        NliScores(
            entailment=values[0],
            contradiction=values[1],
            neutral=values[2],
        )


def test_effective_input_limit_uses_strictest_finite_model_contract() -> None:
    tokenizer = SimpleNamespace(model_max_length=512)
    model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=514))

    assert _effective_max_input_tokens(tokenizer, model) == 512

    tokenizer.model_max_length = 10**30
    model.config.max_position_embeddings = 384
    assert _effective_max_input_tokens(tokenizer, model) == 384


def test_effective_input_limit_fails_closed_without_finite_contract() -> None:
    tokenizer = SimpleNamespace(model_max_length=10**30)
    model = SimpleNamespace(config=SimpleNamespace())

    with pytest.raises(SciFactJudgeError) as raised:
        _effective_max_input_tokens(tokenizer, model)

    assert raised.value.code == "semantic_input_limit_unavailable"


def test_transformers_scorer_accepts_exact_limit_without_truncation() -> None:
    scorer, tokenizer, model = _transformers_stub(
        {"exact-limit": 6}, max_input_tokens=6
    )

    scores = scorer.score_many((("exact-limit", "claim"),))

    assert scores == (NliScores(entailment=0.70, contradiction=0.10, neutral=0.20),)
    assert model.call_count == 1
    assert [call["truncation"] for call in tokenizer.calls] == [False, False]
    assert [call["padding"] for call in tokenizer.calls] == [False, True]
    assert scorer.max_input_tokens == 6
    assert scorer.max_observed_input_tokens == 6
    assert scorer.rejected_overlength_pair_count == 0


def test_transformers_scorer_rejects_every_overlength_pair_before_inference() -> None:
    scorer, tokenizer, model = _transformers_stub(
        {"within-limit": 6, "over-limit-a": 7, "over-limit-b": 9},
        max_input_tokens=6,
    )

    with pytest.raises(SciFactJudgeError) as raised:
        scorer.score_many(
            (
                ("within-limit", "claim"),
                ("over-limit-a", "claim"),
                ("over-limit-b", "claim"),
            )
        )

    assert raised.value.code == "semantic_input_too_long"
    assert model.call_count == 0
    assert len(tokenizer.calls) == 1
    assert tokenizer.calls[0] == {
        "padding": False,
        "truncation": False,
        "return_tensors": None,
    }
    assert scorer.max_input_tokens == 6
    assert scorer.max_observed_input_tokens == 9
    assert scorer.rejected_overlength_pair_count == 2
