from __future__ import annotations

import pytest

from evidencetrace.eval.mutations import (
    changed_slot_categories,
    validate_single_slot_mutation,
)


def test_numeric_mutation_changes_one_slot() -> None:
    source = "Nimbus completed 62% of 200 tasks."
    claim = "Nimbus completed 82% of 200 tasks."

    assert changed_slot_categories(source, claim) == {"numeric"}
    assert validate_single_slot_mutation(source, claim, "percentage_swap") == {"numeric"}


def test_version_mutation_changes_one_slot() -> None:
    assert validate_single_slot_mutation(
        "Release 4.2.0 introduced snapshots.",
        "Release 4.3.0 introduced snapshots.",
        "version_swap",
    ) == {"version"}


def test_multi_slot_mutation_is_rejected() -> None:
    with pytest.raises(ValueError, match="multiple slots"):
        validate_single_slot_mutation(
            "Nimbus completed 62% of 200 tasks.",
            "Orion completed 82% of 300 tasks.",
            "percentage_swap",
        )
