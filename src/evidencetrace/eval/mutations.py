"""Small deterministic mutation helpers used to audit seed-case construction."""

from __future__ import annotations

import re


DATE_RE = re.compile(r"\b20\d{2}[-/]\d{1,2}[-/]\d{1,2}\b")
VERSION_RE = re.compile(r"\bv?\d+(?:\.\d+){1,3}\b", re.I)
NUMBER_RE = re.compile(r"(?<![A-Za-z])\d+(?:[.,]\d+)?%?(?![A-Za-z])")
ENTITY_RE = re.compile(r"\b[A-Z][A-Za-z0-9_-]*\b")
NEGATION_RE = re.compile(r"\b(?:no|not|never|without|cannot|doesn't|does not|isn't|is not)\b", re.I)
COMPARISON_RE = re.compile(r"\b(?:more|less|better|worse|best|worst|faster|slower|higher|lower)\b", re.I)


def _values(pattern: re.Pattern[str], text: str) -> set[str]:
    return {match.group(0).casefold() for match in pattern.finditer(text)}


def changed_slot_categories(source_text: str, claim_text: str) -> frozenset[str]:
    """Return factual slot categories whose values differ between two strings."""

    categories: set[str] = set()
    if _values(DATE_RE, source_text) != _values(DATE_RE, claim_text):
        categories.add("date")
    if _values(VERSION_RE, source_text) != _values(VERSION_RE, claim_text):
        categories.add("version")
    source_numbers = _values(NUMBER_RE, DATE_RE.sub(" ", VERSION_RE.sub(" ", source_text)))
    claim_numbers = _values(NUMBER_RE, DATE_RE.sub(" ", VERSION_RE.sub(" ", claim_text)))
    if source_numbers != claim_numbers:
        categories.add("numeric")
    if bool(NEGATION_RE.search(source_text)) != bool(NEGATION_RE.search(claim_text)):
        categories.add("negation")
    if _values(COMPARISON_RE, source_text) != _values(COMPARISON_RE, claim_text):
        categories.add("comparison")
    source_entities = _values(ENTITY_RE, source_text)
    claim_entities = _values(ENTITY_RE, claim_text)
    if source_entities != claim_entities:
        categories.add("entity")
    source_scope = _values(re.compile(r"\b(?:all|only|every|each|enterprise|one|multi-region)\b", re.I), source_text)
    claim_scope = _values(re.compile(r"\b(?:all|only|every|each|enterprise|one|multi-region)\b", re.I), claim_text)
    if source_scope != claim_scope:
        categories.add("scope")
    return frozenset(categories)


EXPECTED_CATEGORY: dict[str, str] = {
    "percentage_swap": "numeric",
    "numeric_swap": "numeric",
    "range_swap": "numeric",
    "date_swap": "date",
    "version_swap": "version",
    "entity_swap": "entity",
    "negation_flip": "negation",
    "comparison_flip": "comparison",
    "scope_expansion": "scope",
    "scope_narrowing": "scope",
    "qualification_drop": "scope",
}


def validate_single_slot_mutation(
    source_text: str, claim_text: str, mutation_type: str
) -> frozenset[str]:
    """Raise when a claimed deterministic mutation changes multiple slot classes."""

    changed = changed_slot_categories(source_text, claim_text)
    expected = EXPECTED_CATEGORY.get(mutation_type)
    if expected is None:
        raise ValueError(f"unsupported deterministic mutation type: {mutation_type}")
    if expected not in changed:
        raise ValueError(
            f"mutation {mutation_type} did not change its expected {expected} slot"
        )
    if len(changed) != 1:
        raise ValueError(
            f"mutation {mutation_type} changed multiple slots: {sorted(changed)}"
        )
    return changed


__all__ = ["changed_slot_categories", "validate_single_slot_mutation"]
