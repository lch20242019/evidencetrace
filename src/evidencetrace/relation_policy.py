"""Shared bilingual relation guidance for live pair verification."""

from __future__ import annotations

RELATION_DEFINITION_POLICY_VERSION = "bilingual-relations-v1"
RELATION_DEFINITIONS_BILINGUAL = {
    "entailed": "Fully supported by the source / 来源完整支持该陈述。",
    "partially_entailed": "Only part is supported / 来源仅支持陈述的一部分。",
    "contradicted": "Source facts conflict / 来源事实与陈述冲突。",
    "not_in_source": "The source does not address it / 来源未提及。",
    "source_unavailable": "The source is unavailable / 来源不可用。",
    "not_checkable": (
        "Pure opinion without objective criteria / 无明确客观标准的纯主观判断。"
    ),
}

__all__ = [
    "RELATION_DEFINITIONS_BILINGUAL",
    "RELATION_DEFINITION_POLICY_VERSION",
]
