"""Phase 3B must preserve the exposed v1 diagnostic set and its history."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).parents[1]

FROZEN_PHASE3A_HASHES = {
    "eval_sets/core.jsonl": "e7ccc4b4e61a87f971865df94b7b2f296383de4b5c62afdea8e49528a81d0eda",
    "eval_sets/core.frozen_hashes.json": "4c1077a9d4dba10e9e6a7d626a139233af5201004d589a0138f5671703ab93b5",
    "eval_runs/core/eval_report.md": "f2da5bcc0dfc1a7a640ae0f361e6a7e6ff12792b5d37c3670020f0659796f10f",
    "eval_runs/core/eval_results.jsonl": "44026449f7cba1fc0ece490ba16ba20ba6170e3b525550eecc6e88d569f20a48",
    "eval_runs/core/metrics.json": "140b3c00a4546b5b9719d002317817ff94ea15d320d395212913984fde8e207b",
    "eval_runs/core/run_manifest.json": "1ad183b42c44c47c137cda9f9c0eb1ff949a6ca532277758ffd284335bf349cd",
}


def test_v1_diagnostic_set_and_historical_run_are_byte_for_byte_frozen() -> None:
    actual = {
        relative: hashlib.sha256(ROOT.joinpath(relative).read_bytes()).hexdigest()
        for relative in FROZEN_PHASE3A_HASHES
    }

    assert actual == FROZEN_PHASE3A_HASHES
