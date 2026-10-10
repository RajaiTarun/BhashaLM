"""Typed configuration objects loaded from YAML.

Every run (data audit, tokenizer training, pretraining, finetuning) is driven by a
dataclass config. Configs are loaded from YAML, can be overridden from the command line
with dotted keys (``train.lr=3e-4``), and are saved next to every artifact so a run can be
reproduced exactly.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")

# Repo root = two levels above src/bhashalm/
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class DataConfig:
    """Where a language's corpus lives and how it is organised on disk."""

    language: str  # "hindi" | "nepali"
    lang_code: str  # ISO code stored in each document, e.g. "hi" | "ne"
    splits_dir: str  # directory containing train/ val/ test/ sub-folders of *.jsonl
    splits: list[str] = field(default_factory=lambda: ["train", "val", "test"])
    text_field: str = "text"
    seed: int = 1337

    def split_dir(self, split: str) -> Path:
        """Return the absolute directory for one split."""
        return resolve_path(self.splits_dir) / split


def resolve_path(path: str | Path) -> Path:
    """Resolve a path relative to the repo root (absolute paths are returned unchanged)."""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _coerce(value: str) -> Any:
    """Parse a CLI override value with YAML rules, so '3e-4' -> float and 'true' -> bool."""
    return yaml.safe_load(value)


def apply_overrides(raw: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    """Apply ``a.b.c=value`` style overrides to a nested dict in place and return it."""
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key=value, got {item!r}")
        key, value = item.split("=", 1)
        node = raw
        *parents, leaf = key.split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = _coerce(value)
    return raw


def from_dict(cls: type[T], raw: dict[str, Any]) -> T:
    """Build a (possibly nested) dataclass from a dict, rejecting unknown keys."""
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"{cls} is not a dataclass")
    known = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(raw) - set(known)
    if unknown:
        raise KeyError(f"Unknown config keys for {cls.__name__}: {sorted(unknown)}")
    kwargs = {}
    for name, value in raw.items():
        ftype = known[name].type
        # Nested dataclass fields are stored as dicts in YAML.
        sub = globals().get(ftype) if isinstance(ftype, str) else ftype
        if isinstance(value, dict) and dataclasses.is_dataclass(sub):
            value = from_dict(sub, value)
        kwargs[name] = value
    return cls(**kwargs)


def load_config(cls: type[T], path: str | Path, overrides: list[str] | None = None) -> T:
    """Load a YAML file into dataclass ``cls``, applying optional dotted overrides."""
    with open(resolve_path(path), encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return from_dict(cls, apply_overrides(raw, overrides))


def save_config(cfg: Any, path: str | Path) -> None:
    """Write a dataclass config to YAML (used to snapshot configs next to artifacts)."""
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(dataclasses.asdict(cfg), f, allow_unicode=True, sort_keys=False)
