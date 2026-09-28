"""Load the small, versioned EvidenceTrace TOML configuration."""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from evidencetrace.models import EffectiveConfig

DEFAULT_CONFIG_PATH = Path(".evidencetrace.toml")


class ConfigError(ValueError):
    """Raised when a configuration file cannot be read or validated."""


def config_from_mapping(data: Mapping[str, Any]) -> EffectiveConfig:
    """Validate an already decoded configuration mapping."""

    try:
        return EffectiveConfig.model_validate(dict(data))
    except ValidationError as error:
        raise ConfigError(f"invalid EvidenceTrace configuration: {error}") from error


def loads_config(text: str) -> EffectiveConfig:
    """Decode and validate TOML text."""

    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"invalid TOML: {error}") from error
    return config_from_mapping(data)


def load_config(path: Path | str | None = None) -> EffectiveConfig:
    """Load a config file, using defaults when the implicit file is absent.

    Passing ``None`` means "look for .evidencetrace.toml in the current
    directory" and falls back to defaults when it is absent. An explicitly
    supplied missing path is an error, so a typo cannot silently change policy.
    """

    implicit = path is None
    config_path = DEFAULT_CONFIG_PATH if implicit else Path(path)
    if not config_path.exists():
        if implicit:
            return EffectiveConfig()
        raise ConfigError(f"configuration file does not exist: {config_path}")
    if not config_path.is_file():
        raise ConfigError(f"configuration path is not a file: {config_path}")
    try:
        data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ConfigError(f"cannot read configuration: {config_path}") from error
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"invalid TOML in {config_path}: {error}") from error
    return config_from_mapping(data)


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "ConfigError",
    "config_from_mapping",
    "load_config",
    "loads_config",
]
