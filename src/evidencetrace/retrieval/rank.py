"""Citation-first lexical retrieval using FTS5 with BM25 fallback."""

from __future__ import annotations

import math
import re
import sqlite3
import unicodedata
from collections import Counter
from collections.abc import Iterable
from itertools import pairwise

from evidencetrace.audit_models import (
    EVIDENCE_CONTEXT_MAX_BYTES,
    EVIDENCE_CONTEXT_MAX_CHARS,
    EVIDENCE_EXACT_MAX_BYTES,
    EVIDENCE_EXACT_MAX_CHARS,
    EvidenceChunk,
    RetrievedEvidence,
    evidence_collection_is_bounded,
)

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[._-][A-Za-z0-9]+)*%?")
CJK_RUN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]{2,}")
SPECIAL_RE = re.compile(
    r"(?:\b20\d{2}-\d{1,2}-\d{1,2}\b|\bv?\d+(?:\.\d+){1,3}\b|\b\d+(?:\.\d+)?%)",
    re.I,
)
RETRIEVAL_POLICY_VERSION = "lexical-cjk-ngram-v3"
# A zero score is deliberately reserved for the bounded full-context fallback.
# It means "no lexical hit was found", rather than a very small positive
# retrieval score that could be mistaken for a real ranking result.
FULL_CONTEXT_FALLBACK_SCORE = 0.0


def _tokens(text: str) -> list[str]:
    return [token.casefold() for token in TOKEN_RE.findall(text)]


def _cjk_ngrams(text: str) -> list[str]:
    """Build CJK-only features without mutating source or identifier text."""

    normalized = unicodedata.normalize("NFKC", text)
    features: list[str] = []
    for run in CJK_RUN_RE.findall(normalized):
        for width in (2, 3):
            features.extend(
                run[index : index + width] for index in range(len(run) - width + 1)
            )
    return features


class LexicalRetriever:
    def __init__(
        self, chunks: Iterable[EvidenceChunk], *, neighbor_window: int = 1
    ) -> None:
        if neighbor_window < 0:
            raise ValueError("neighbor_window cannot be negative")
        self.chunks = tuple(chunks)
        if any(
            len(chunk.text) > EVIDENCE_EXACT_MAX_CHARS
            or len(chunk.text.encode("utf-8")) > EVIDENCE_EXACT_MAX_BYTES
            for chunk in self.chunks
        ):
            raise ValueError("retrieval source chunk exceeds the hard size limit")
        self.neighbor_window = neighbor_window
        # Keep a small inverted index for CJK n-grams.  SQLite's default FTS5
        # tokenizer is useful for Latin identifiers but is not a reliable
        # tokenizer for unspaced Chinese, so CJK candidates are resolved
        # independently and then unioned with the FTS candidates below.
        self._cjk_postings: dict[str, set[int]] = {}
        for index, chunk in enumerate(self.chunks):
            for term in set(_cjk_ngrams(chunk.text)):
                self._cjk_postings.setdefault(term, set()).add(index)
        self.fts5 = False
        self.db: sqlite3.Connection | None = None
        try:
            self.db = sqlite3.connect(":memory:")
            self.db.execute("CREATE VIRTUAL TABLE evidence USING fts5(text, heading)")
            self.db.executemany(
                "INSERT INTO evidence(rowid,text,heading) VALUES (?,?,?)",
                (
                    (index + 1, chunk.text, " ".join(chunk.heading_path))
                    for index, chunk in enumerate(self.chunks)
                ),
            )
            self.db.commit()
            self.fts5 = True
        except sqlite3.OperationalError:
            if self.db is not None:
                self.db.close()
            self.db = None

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        fallback_to_full_context: bool = False,
    ) -> tuple[RetrievedEvidence, ...]:
        """Return lexical hits, optionally falling back to a bounded full source.

        The fallback is intentionally opt-in for multi-chunk sources.  A
        lexical retriever must not silently turn a large or discontinuous
        document into model context.  Callers that have a policy-level reason
        to continue after an empty recall can request it and then record that
        route explicitly.  A single bounded chunk keeps the historical
        invariant that a short source is not discarded solely because lexical
        overlap is zero.
        """
        if top_k <= 0:
            return ()
        terms = _tokens(query)
        # CJK and Latin/structured terms are complementary signals.  The old
        # implementation disabled CJK features as soon as a query contained
        # two Latin tokens (for example ``Product v1.2.3``), which made a
        # Chinese sentence with a product/version pair fall through to the
        # Latin-only FTS candidate list.  If the source used Chinese wording,
        # the relevant chunk was then never scored at all.  Keep both feature
        # families whenever the query contains CJK text.  The evaluation
        # sources are small bounded fixtures, so scanning all chunks in this
        # branch is intentional and avoids relying on an FTS tokenizer that
        # is not CJK-aware.
        cjk_terms = (
            _cjk_ngrams(query)
            if re.search(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]", query)
            else []
        )
        if not terms and not cjk_terms:
            return ()
        if cjk_terms:
            cjk_candidates: set[int] = set()
            for term in cjk_terms:
                cjk_candidates.update(self._cjk_postings.get(term, ()))
            latin_candidates = set(self._candidate_indexes(query)) if terms else set()
            candidate_indexes = sorted(cjk_candidates | latin_candidates)
            # A short source is cheap to scan and may contain a structured
            # slot (date/version/number) that the CJK index cannot represent.
            # Preserve that useful fallback without making every large
            # document a full scan.
            if not candidate_indexes and len(self.chunks) <= 32:
                candidate_indexes = list(range(len(self.chunks)))
        else:
            candidate_indexes = self._candidate_indexes(query)
        scored = [
            (index, self._bm25_score(index, terms, cjk_terms, query))
            for index in candidate_indexes
        ]
        scored = [item for item in scored if item[1] > 0]
        if not scored:
            fallback = self.full_context_fallback()
            if fallback is not None and (
                fallback_to_full_context or len(self.chunks) == 1
            ):
                return (fallback,)
        scored.sort(key=lambda item: (-item[1], item[0]))
        results: list[RetrievedEvidence] = []
        for index, score in scored[:top_k]:
            hit = self.chunks[index]
            context = self._with_neighbors(index)
            candidate = RetrievedEvidence(
                chunk=context,
                text=hit.text,
                score=score,
            )
            if not evidence_collection_is_bounded((*results, candidate)):
                break
            results.append(candidate)
        return tuple(results)

    def full_context_fallback(self) -> RetrievedEvidence | None:
        """Build one exact, bounded candidate for an empty lexical recall.

        Only chunks that can be joined as a literal source slice are accepted.
        The exact evidence text is capped at ``EVIDENCE_EXACT_MAX_CHARS`` and
        bytes, so this method cannot bypass the normal Judge payload limits.
        ``score=0`` is intentional and lets downstream routing distinguish this
        path from a lexical hit without inventing a ranking value.
        """

        if not self.chunks:
            return None
        ordered_pairs = tuple(
            sorted(
                enumerate(self.chunks),
                key=lambda pair: (pair[1].char_start, pair[0]),
            )
        )
        indexes = [index for index, _ in ordered_pairs]
        ordered = tuple(item for _, item in ordered_pairs)
        first = ordered[0]
        if any(
            item.source_id != first.source_id
            or item.url != first.url
            or item.char_end <= item.char_start
            or item.char_end - item.char_start != len(item.text)
            for item in ordered
        ):
            return None
        context = self._literal_context(indexes)
        if context is None:
            return None
        if (
            not context.strip()
            or len(context) > EVIDENCE_EXACT_MAX_CHARS
            or len(context.encode("utf-8")) > EVIDENCE_EXACT_MAX_BYTES
        ):
            return None
        last = ordered[-1]
        if len(ordered) == 1:
            # Preserve the original chunk identity/locator for the historical
            # one-chunk invariant; only the score communicates fallback origin.
            return RetrievedEvidence(
                chunk=first,
                text=first.text,
                score=FULL_CONTEXT_FALLBACK_SCORE,
            )
        chunk = EvidenceChunk(
            source_id=first.source_id,
            url=first.url,
            text=context,
            heading_path=first.heading_path,
            locator=(
                f"full-source fallback [chars {first.char_start}-{last.char_end}]"
            ),
            char_start=first.char_start,
            char_end=last.char_end,
        )
        return RetrievedEvidence(
            chunk=chunk,
            text=context,
            score=FULL_CONTEXT_FALLBACK_SCORE,
        )

    def _candidate_indexes(self, query: str) -> list[int]:
        if not self.fts5 or self.db is None:
            return list(range(len(self.chunks)))
        fts_terms = list(dict.fromkeys(re.findall(r"[A-Za-z0-9]+", query)))
        if not fts_terms:
            return []
        expression = " OR ".join(f'"{term}"' for term in fts_terms)
        try:
            rows = self.db.execute(
                "SELECT rowid FROM evidence WHERE evidence MATCH ?", (expression,)
            ).fetchall()
        except sqlite3.OperationalError:
            return list(range(len(self.chunks)))
        return [int(row[0]) - 1 for row in rows]

    def _bm25_score(
        self,
        index: int,
        query_terms: list[str],
        cjk_terms: list[str],
        query: str,
    ) -> float:
        document_terms = _tokens(self.chunks[index].text)
        counts = Counter(document_terms)
        score = 0.0
        total_documents = max(len(self.chunks), 1)
        for term in query_terms:
            frequency = counts.get(term, 0)
            if not frequency:
                continue
            document_frequency = sum(
                term in set(_tokens(chunk.text)) for chunk in self.chunks
            )
            inverse = math.log(
                1
                + (total_documents - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
            score += inverse * (frequency * 2.2) / (frequency + 1.2)
        cjk_counts = Counter(_cjk_ngrams(self.chunks[index].text))
        for term in cjk_terms:
            frequency = cjk_counts.get(term, 0)
            if not frequency:
                continue
            document_frequency = sum(
                term in set(_cjk_ngrams(chunk.text)) for chunk in self.chunks
            )
            inverse = math.log(
                1
                + (total_documents - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
            score += inverse * (frequency * 2.2) / (frequency + 1.2)
        for special in SPECIAL_RE.findall(query):
            if special.casefold() in self.chunks[index].text.casefold():
                score += 8.0
        return score

    def _with_neighbors(self, index: int) -> EvidenceChunk:
        hit = self.chunks[index]
        if self.neighbor_window == 0:
            return hit
        selected = [index]
        for distance in range(1, self.neighbor_window + 1):
            for candidate_index in (index - distance, index + distance):
                if not 0 <= candidate_index < len(self.chunks):
                    continue
                candidate = self.chunks[candidate_index]
                if (
                    candidate.source_id != hit.source_id
                    or candidate.url != hit.url
                ):
                    continue
                proposed = sorted((*selected, candidate_index))
                context = self._literal_context(proposed)
                if context is None:
                    continue
                if (
                    len(context) > EVIDENCE_CONTEXT_MAX_CHARS
                    or len(context.encode("utf-8")) > EVIDENCE_CONTEXT_MAX_BYTES
                ):
                    break
                selected = proposed
            else:
                continue
            break
        neighbors = [self.chunks[value] for value in selected]
        if len(neighbors) == 1:
            return hit
        context = self._literal_context(selected)
        assert context is not None
        return EvidenceChunk(
            source_id=hit.source_id,
            url=hit.url,
            text=context,
            heading_path=hit.heading_path,
            locator=hit.locator + " (with adjacent context)",
            char_start=min(chunk.char_start for chunk in neighbors),
            char_end=max(chunk.char_end for chunk in neighbors),
        )

    def _literal_context(self, indexes: list[int]) -> str | None:
        chunks = [
            self.chunks[value]
            for value in sorted(
                indexes,
                key=lambda value: (self.chunks[value].char_start, value),
            )
        ]
        if any(
            chunk.char_end - chunk.char_start != len(chunk.text)
            for chunk in chunks
        ):
            return None
        context = chunks[0].text
        for previous, current in pairwise(chunks):
            gap = current.char_start - previous.char_end
            if gap == 0:
                separator = ""
            elif (
                gap == 2
                and "paragraph " in previous.locator
                and "paragraph " in current.locator
                and " part " not in previous.locator
                and " part " not in current.locator
            ):
                separator = "\n\n"
            else:
                return None
            context += separator + current.text
        if chunks[-1].char_end - chunks[0].char_start != len(context):
            return None
        return context


__all__ = [
    "FULL_CONTEXT_FALLBACK_SCORE",
    "RETRIEVAL_POLICY_VERSION",
    "LexicalRetriever",
]
