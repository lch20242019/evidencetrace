"""Deterministic slot-alignment signals for the semantic Judge."""

from __future__ import annotations

import re
from collections.abc import Iterable

from evidencetrace.audit_models import (
    CheckSignal,
    EvidenceChunk,
    RetrievedEvidence,
)

DATE_RE = re.compile(r"\b20\d{2}[-/]\d{1,2}[-/]\d{1,2}\b")
VERSION_RE = re.compile(r"\bv?\d+(?:\.\d+){1,3}\b", re.I)
NUMBER_RE = re.compile(
    r"(?<![A-Za-z])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?(?![A-Za-z])"
)
NEGATION_RE = re.compile(
    r"\b(?:no|not|never|without|cannot|can't|won't|don't|doesn't|didn't|"
    r"isn't|aren't|wasn't|weren't|does\s+not|do\s+not|did\s+not|"
    r"is\s+not|are\s+not|was\s+not|were\s+not)\b",
    re.I,
)
COMPARISON_RE = re.compile(
    r"\b(?:more|less|better|worse|best|worst|faster|slower|higher|"
    r"lower|largest|smallest|greatest|least|every|all|only)\b",
    re.I,
)
RECOMMENDATION_RE = re.compile(
    r"\b(?:should|ought\s+to|prefer(?:s|red|ring)?|"
    r"recommend(?:s|ed|ing|ation|ations)?|advis(?:e|es|ed|ing|able))\b"
    r"|(?:建议|推荐|应当|应该|最好|最佳|最高|最低|最优秀|最值得|最理想|"
    r"值得推荐|更容易使用|宜)",
    re.I,
)
OBJECTIVE_FACT_RE = re.compile(
    r"(?:"
    r"\b(?:benchmark|specification|manual|documentation|release\s+note|"
    r"ranking|ranked)\b|"
    r"\b(?:supports?|runs?\s+on|for)\s+[A-Za-z0-9._ -]+"
    r"(?:platform|operating\s+system)\b|"
    r"(?:基准|测试|指标|手册|文档|说明|公告|引用|排名|名次|"
    r"列为|写明|上限|吞吐量|延迟)|"
    r"(?:支持|兼容|运行于|适用于).{0,16}(?:平台|系统)|"
    r"(?:第[一二三四五六七八九十\d]+名)|"
    r"(?:最多|最少|至少|至多)\s*\d"
    r")",
    re.I,
)
QUALIFIER_RE = re.compile(
    r"\b(?:only|all|every|each|any|exactly|at\s+least|at\s+most|"
    r"narrow|wide|premium|standard)\b",
    re.I,
)
SCOPE_RE = re.compile(
    r"\b(?:(?P<modifier>[A-Za-z][A-Za-z0-9_-]*)\s+)?"
    r"(?P<noun>regions?|accounts?|tiers?|samples?|benchmarks?)\b",
    re.I,
)
DENOMINATOR_RE = re.compile(
    r"\b(?:out\s+of|among|per)\s+\d[\d,]*(?:\.\d+)?"
    r"(?:\s+(?:regions?|accounts?|tiers?|samples?|benchmarks?))?\b",
    re.I,
)
ENTITY_RE = re.compile(
    r"\b(?:[A-Z]{2,}[A-Z0-9]*|[A-Z][a-z]+(?:[A-Z][A-Za-z0-9]*)?|"
    r"[A-Za-z][A-Za-z0-9]*(?:[-_.][A-Za-z0-9]+)+)\b"
)
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[._-][A-Za-z0-9]+)*")
CONJUNCTION_RE = re.compile(r"\b(?:and|or)\b", re.I)
CLAUSE_BOUNDARY_RE = re.compile(r"(?<=[.;!?])\s+|\s+(?:and|but)\s+", re.I)
OPPOSED_QUALIFIERS = (
    (frozenset({"only"}), frozenset({"all", "every"})),
    (frozenset({"at least"}), frozenset({"at most"})),
    (frozenset({"narrow"}), frozenset({"wide"})),
    (frozenset({"premium"}), frozenset({"standard"})),
)

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "was",
    "were",
    "with",
}
NON_ENTITY_WORDS = STOPWORDS | {
    "all",
    "any",
    "each",
    "every",
    "only",
    "service",
    "platform",
    "project",
    "product",
    "release",
    "server",
    "client",
    "version",
    "it",
    "this",
    "that",
    "these",
    "those",
}
SCOPE_SINGULAR = {
    "regions": "region",
    "accounts": "account",
    "tiers": "tier",
    "samples": "sample",
    "benchmarks": "benchmark",
}

# These are explicit aligned conflicts. Producers and consumers share the same
# code contract; every emitted hard conflict has error severity.
HARD_CONFLICT_CODES = frozenset(
    {
        "numeric_mismatch",
        "date_mismatch",
        "version_mismatch",
        "entity_mismatch",
        "negation_mismatch",
    }
)
DETERMINISTIC_SIGNAL_POLICY_VERSION = "deterministic-signals-v2"
CHECKABILITY_POLICY_VERSION = "bilingual-checkability-v1"
GUARD_SIGNAL_CODES = HARD_CONFLICT_CODES | {"qualifier_mismatch"}
CAUTION_CODES = frozenset(
    {"comparison_reminder", "qualifier_mismatch", "conjunction_mismatch"}
)


class DeterministicConflictError(ValueError):
    """A payload-free rejection containing only known guard signal codes."""

    def __init__(self, signal_codes: Iterable[str]) -> None:
        codes = tuple(sorted(set(signal_codes)))
        if not codes or not set(codes) <= GUARD_SIGNAL_CODES:
            raise ValueError("deterministic guard signal codes are invalid")
        self.signal_codes = codes
        super().__init__("entailed verdict conflicts with deterministic mismatch")


def bounded_evidence_text(
    evidence: Iterable[RetrievedEvidence | EvidenceChunk],
) -> str:
    """Join only the source chunks already present in the Judge input."""

    return "\n".join(
        dict.fromkeys(
            item.chunk.text if isinstance(item, RetrievedEvidence) else item.text
            for item in evidence
        )
    )


def _dates(text: str) -> set[str]:
    return {match.group(0).replace("/", "-") for match in DATE_RE.finditer(text)}


def _versions(text: str) -> set[str]:
    return {
        match.group(0).casefold().removeprefix("v")
        for match in VERSION_RE.finditer(text)
    }


def _numbers(text: str) -> set[str]:
    without_slots = DATE_RE.sub(" ", VERSION_RE.sub(" ", text))
    return {
        match.group(0).replace(",", "").casefold()
        for match in NUMBER_RE.finditer(without_slots)
    }


def _words(text: str) -> set[str]:
    return {
        match.group(0).casefold()
        for match in WORD_RE.finditer(text)
        if match.group(0).casefold() not in STOPWORDS
    }


def _entities(text: str) -> set[str]:
    return {
        match.group(0).casefold()
        for match in ENTITY_RE.finditer(text)
        if match.group(0).casefold() not in NON_ENTITY_WORDS
    }


def _context_words(text: str) -> set[str]:
    return _words(text) - _entities(text) - _versions(text) - _numbers(text)


def _contexts_align(claim_text: str, evidence_text: str) -> bool:
    claim_context = _context_words(claim_text)
    source_context = _context_words(evidence_text)
    shared = claim_context & source_context
    smaller = min(len(claim_context), len(source_context))
    if len(shared) >= 2 and len(shared) / max(smaller, 1) >= 0.5:
        return True
    if shared and _entities(claim_text) & _entities(evidence_text):
        return True
    return bool(shared and _versions(claim_text) & _versions(evidence_text))


def _aligned_evidence_text(claim_text: str, evidence_text: str) -> str:
    """Use the source clause most aligned to a non-compound claim."""

    if CONJUNCTION_RE.search(claim_text):
        return evidence_text
    claim_words = _words(claim_text)
    clauses = [
        clause.strip()
        for clause in CLAUSE_BOUNDARY_RE.split(evidence_text)
        if clause.strip()
    ]
    return max(
        clauses or [evidence_text],
        key=lambda clause: len(claim_words & _words(clause)),
    )


def _qualifiers(text: str) -> set[str]:
    values = {
        re.sub(r"\s+", " ", match.group(0).casefold())
        for match in QUALIFIER_RE.finditer(text)
    }
    for match in SCOPE_RE.finditer(text):
        noun = match.group("noun").casefold()
        noun = SCOPE_SINGULAR.get(noun, noun)
        modifier = (match.group("modifier") or "").casefold()
        if modifier in STOPWORDS:
            modifier = ""
        values.add(f"scope:{modifier}:{noun}")
    values.update(
        "denominator:" + re.sub(r"\s+", " ", match.group(0).casefold())
        for match in DENOMINATOR_RE.finditer(text)
    )
    return values


def _opposed_qualifiers(
    claim_qualifiers: set[str], source_qualifiers: set[str]
) -> bool:
    return any(
        (left & claim_qualifiers and right & source_qualifiers)
        or (right & claim_qualifiers and left & source_qualifiers)
        for left, right in OPPOSED_QUALIFIERS
    )


def _has_missing_conjunct(claim_text: str, evidence_text: str) -> bool:
    parts = [part.strip() for part in CONJUNCTION_RE.split(claim_text)]
    if len(parts) < 2:
        return False
    evidence_words = _words(evidence_text)
    coverage: list[float] = []
    for part in parts:
        part_words = _words(part)
        if part_words:
            coverage.append(len(part_words & evidence_words) / max(len(part_words), 1))
    return bool(
        len(coverage) >= 2
        and any(value >= 0.75 for value in coverage)
        and any(value < 0.75 for value in coverage)
    )


def is_high_confidence_subjective(text: str) -> bool:
    """Detect opinion/advice only when no objective fact slot is present."""

    if RECOMMENDATION_RE.search(text) is None:
        return False
    has_structured_slot = bool(
        DATE_RE.search(text) or VERSION_RE.search(text) or NUMBER_RE.search(text)
    )
    return not has_structured_slot and OBJECTIVE_FACT_RE.search(text) is None


def is_recommendation(text: str) -> bool:
    """Backward-compatible name for high-confidence subjective detection."""

    return is_high_confidence_subjective(text)


def deterministic_signals(
    claim_text: str,
    evidence_text: str,
    *,
    bounded_evidence_context: str | None = None,
) -> tuple[CheckSignal, ...]:
    signals: list[CheckSignal] = []
    aligned_evidence = _aligned_evidence_text(claim_text, evidence_text)
    comparisons = (
        ("date_mismatch", _dates(claim_text), _dates(aligned_evidence), "dates"),
        (
            "version_mismatch",
            _versions(claim_text),
            _versions(aligned_evidence),
            "versions",
        ),
        (
            "numeric_mismatch",
            _numbers(claim_text),
            _numbers(aligned_evidence),
            "values",
        ),
    )
    for code, claim_values, source_values, label in comparisons:
        if claim_values and source_values and not claim_values <= source_values:
            signals.append(
                CheckSignal(
                    code=code,
                    detail=(
                        f"claim {label} {sorted(claim_values)} differ from "
                        f"source {sorted(source_values)}"
                    ),
                    severity="error",
                )
            )
    if bool(NEGATION_RE.search(claim_text)) != bool(
        NEGATION_RE.search(aligned_evidence)
    ) and _contexts_align(claim_text, aligned_evidence):
        signals.append(
            CheckSignal(
                code="negation_mismatch",
                detail="claim and source differ in negation markers",
                severity="error",
            )
        )
    claim_entities = _entities(claim_text)
    source_entities = _entities(aligned_evidence)
    context_entities = _entities(
        evidence_text if bounded_evidence_context is None else bounded_evidence_context
    )
    if (
        claim_entities
        and source_entities
        and claim_entities.isdisjoint(source_entities)
        and claim_entities.isdisjoint(context_entities)
        and _contexts_align(claim_text, aligned_evidence)
    ):
        signals.append(
            CheckSignal(
                code="entity_mismatch",
                detail=(
                    f"aligned predicate/context names claim entities "
                    f"{sorted(claim_entities)} but source entities "
                    f"{sorted(source_entities)}"
                ),
                severity="error",
            )
        )
    claim_qualifiers = _qualifiers(claim_text)
    source_qualifiers = _qualifiers(aligned_evidence)
    if (
        claim_qualifiers != source_qualifiers
        and (claim_qualifiers or source_qualifiers)
        and _contexts_align(claim_text, aligned_evidence)
    ):
        severity = (
            "error"
            if _opposed_qualifiers(claim_qualifiers, source_qualifiers)
            else "warning"
        )
        signals.append(
            CheckSignal(
                code="qualifier_mismatch",
                detail=(
                    f"claim qualifiers {sorted(claim_qualifiers)} differ from "
                    f"source qualifiers {sorted(source_qualifiers)}"
                ),
                severity=severity,
            )
        )
    if _contexts_align(claim_text, evidence_text) and _has_missing_conjunct(
        claim_text, evidence_text
    ):
        signals.append(
            CheckSignal(
                code="conjunction_mismatch",
                detail="source covers only part of a compound claim",
                severity="warning",
            )
        )
    if COMPARISON_RE.search(claim_text):
        signals.append(
            CheckSignal(
                code="comparison_reminder",
                detail="comparative or absolute wording requires semantic review",
                severity="notice",
            )
        )
    return tuple(signals)


def validate_evidence_span(evidence_text: str, source_text: str) -> CheckSignal | None:
    if not evidence_text.strip() or evidence_text not in source_text:
        return CheckSignal(
            code="span_missing",
            detail="candidate evidence is not a non-empty literal source substring",
            severity="error",
        )
    return None


__all__ = [
    "CAUTION_CODES",
    "CHECKABILITY_POLICY_VERSION",
    "DETERMINISTIC_SIGNAL_POLICY_VERSION",
    "GUARD_SIGNAL_CODES",
    "HARD_CONFLICT_CODES",
    "DeterministicConflictError",
    "bounded_evidence_text",
    "deterministic_signals",
    "is_high_confidence_subjective",
    "is_recommendation",
    "validate_evidence_span",
]
