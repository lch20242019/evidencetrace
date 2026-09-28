"""Run one frozen SciFact script with the legacy slow-tokenizer default.

Transformers 4.x defaults to a fast tokenizer, which rejects the ``zip`` object
passed by the frozen SciFact rationale script.  Transformers 2.7 used by the
upstream release resolved this checkpoint through the slow tokenizer.  The
upstream whole-abstract GPU batch also exceeds a 6 GiB GPU for the largest
SciFact abstract.  This launcher restores the slow tokenizer and splits only
the model forward pass; tokenization, padding, logits, and output order stay the
same before the unmodified script resumes.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class _MicroBatchModel(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, micro_batch_size: int) -> None:
        super().__init__()
        self.model = model
        self.micro_batch_size = micro_batch_size

    def forward(self, **kwargs: torch.Tensor) -> Any:
        batch_size = next(iter(kwargs.values())).size(0)
        if batch_size <= self.micro_batch_size:
            return self.model(**kwargs)
        logits = []
        for start in range(0, batch_size, self.micro_batch_size):
            batch = {
                key: tensor[start : start + self.micro_batch_size]
                for key, tensor in kwargs.items()
            }
            logits.append(self.model(**batch)[0])
        return (torch.cat(logits, dim=0),)


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(
            "usage: launcher.py MICRO_BATCH_SIZE OFFICIAL_SCRIPT [SCRIPT_ARGS ...]"
        )
    try:
        micro_batch_size = int(sys.argv[1])
    except ValueError as error:
        raise SystemExit("MICRO_BATCH_SIZE must be an integer") from error
    if micro_batch_size <= 0:
        raise SystemExit("MICRO_BATCH_SIZE must be positive")
    official_script = Path(sys.argv[2]).resolve()
    if not official_script.is_file():
        raise SystemExit(f"official script does not exist: {official_script}")

    original = AutoTokenizer.from_pretrained
    original_model = AutoModelForSequenceClassification.from_pretrained

    def from_pretrained(*args: Any, **kwargs: Any) -> Any:
        kwargs["use_fast"] = False
        kwargs["local_files_only"] = True
        return original(*args, **kwargs)

    def model_from_pretrained(*args: Any, **kwargs: Any) -> Any:
        kwargs["local_files_only"] = True
        return _MicroBatchModel(
            original_model(*args, **kwargs), micro_batch_size=micro_batch_size
        )

    AutoTokenizer.from_pretrained = from_pretrained
    AutoModelForSequenceClassification.from_pretrained = model_from_pretrained
    sys.argv = [str(official_script), *sys.argv[3:]]
    runpy.run_path(str(official_script), run_name="__main__")


if __name__ == "__main__":
    main()
