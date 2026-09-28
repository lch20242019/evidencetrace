"""Typed, payload-free failures for local evaluation invariants."""

from __future__ import annotations

from typing import Literal

LocalValidationCode = Literal[
    "canonical_prediction_invalid",
    "deterministic_baseline_model_call",
    "evidence_span_not_in_source",
    "model_call_count_out_of_bounds",
    "model_telemetry_inconsistent",
    "source_availability_conflict",
    "substantive_evidence_missing",
]
LOCAL_VALIDATION_CODES: frozenset[str] = frozenset(
    {
        "canonical_prediction_invalid",
        "deterministic_baseline_model_call",
        "evidence_span_not_in_source",
        "model_call_count_out_of_bounds",
        "model_telemetry_inconsistent",
        "source_availability_conflict",
        "substantive_evidence_missing",
    }
)

FailureCode = Literal[
    "call_budget_exhausted",
    "canonical_prediction_invalid",
    "deterministic_baseline_model_call",
    "deterministic_conflict",
    "evidence_span_not_in_source",
    "judge_scope_violation",
    "miner_scope_violation",
    "model_call_count_out_of_bounds",
    "model_schema_failure",
    "model_telemetry_inconsistent",
    "model_transport_failure",
    "source_availability_conflict",
    "substantive_evidence_missing",
    "unexpected_exception",
]


class LocalValidationError(RuntimeError):
    """Reject a local invariant using only one allowlisted diagnostic code."""

    def __init__(self, code: LocalValidationCode) -> None:
        if code not in LOCAL_VALIDATION_CODES:
            raise ValueError("unsupported local validation code")
        super().__init__("local evaluation validation failed")
        self.code = code
        self.__cause__ = None
        self.__context__ = None
        self.__suppress_context__ = True


def clear_exception_chain(error: BaseException) -> None:
    """Remove chained payloads before a failure crosses the artifact boundary."""

    error.__cause__ = None
    error.__context__ = None
    error.__suppress_context__ = True


__all__ = [
    "LOCAL_VALIDATION_CODES",
    "FailureCode",
    "LocalValidationCode",
    "LocalValidationError",
    "clear_exception_chain",
]
