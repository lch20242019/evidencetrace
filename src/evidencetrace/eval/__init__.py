"""Offline evaluation infrastructure for the citation-audit MVP."""

from evidencetrace.eval.dataset import (
    DatasetValidationError,
    load_dataset,
    validate_dataset,
)
from evidencetrace.eval.full_document import (
    compare_full_document_runs,
    load_full_document_dataset,
    run_full_document_benchmark,
)
from evidencetrace.eval.runner import run_eval

__all__ = [
    "DatasetValidationError",
    "compare_full_document_runs",
    "load_dataset",
    "load_full_document_dataset",
    "run_eval",
    "run_full_document_benchmark",
    "validate_dataset",
]
