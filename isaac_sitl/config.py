"""Configuration overlay support independent of Isaac Sim."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge objects; arrays and scalar values replace the base."""
    merged = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _read_json(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    with resolved.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration root must be a JSON object: {resolved}")
    return value


def load_config_with_overlays(
    base_path: str | Path,
    overlay_paths: list[str | Path] | tuple[str | Path, ...] = (),
) -> dict[str, Any]:
    config = _read_json(base_path)
    for overlay_path in overlay_paths:
        config = deep_merge(config, _read_json(overlay_path))
    return config
