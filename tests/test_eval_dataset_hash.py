from __future__ import annotations

import shutil
from pathlib import Path

from evidencetrace.eval.dataset import load_dataset

ROOT = Path(__file__).parents[1]


def test_dataset_hash_covers_source_fixture_metadata(tmp_path: Path) -> None:
    eval_dir = tmp_path / "eval_sets"
    shutil.copytree(ROOT / "eval_sets", eval_dir)
    dataset_path = eval_dir / "core.jsonl"
    before = load_dataset(dataset_path).raw_hash
    sources = eval_dir / "sources/seed_sources.jsonl"
    content = sources.read_text(encoding="utf-8")
    sources.write_text(
        content.replace(
            "hand-authored fictional offline release fixture",
            "reviewed fictional offline release fixture",
            1,
        ),
        encoding="utf-8",
    )

    assert load_dataset(dataset_path).raw_hash != before
