"""Small shared helpers: seeding, logging, run metadata, JSON I/O."""

from __future__ import annotations

import json
import logging
import os
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


def set_seed(seed: int) -> None:
    """Seed every RNG we use (python, numpy, and torch if installed) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_logger(name: str = "bhashalm", level: int = logging.INFO) -> logging.Logger:
    """Return a logger that prints ``time | level | message`` once (no duplicate handlers)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def git_commit_hash() -> str | None:
    """Return the current git commit hash, or None outside a git repo.

    Saved with every artifact so we always know which code produced it.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def write_json(obj: Any, path: str | Path) -> None:
    """Write ``obj`` as pretty UTF-8 JSON, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_json(path: str | Path) -> Any:
    """Read a UTF-8 JSON file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)
