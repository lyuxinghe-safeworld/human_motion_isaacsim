from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any


def _load_registry() -> dict[str, dict[str, Any]]:
    """Read and parse the bundled ``registry.json`` model catalogue."""
    registry_text = resources.files("human_motion_isaacsim").joinpath("registry.json").read_text(
        encoding="utf-8"
    )
    data = json.loads(registry_text)
    if not isinstance(data, dict):
        raise ValueError("registry.json must contain a JSON object")
    return data


def list_models() -> list[dict[str, str]]:
    """Return a list of available model entries from the registry."""
    registry = _load_registry()
    return [
        {
            "name": model_name,
            "description": str(metadata.get("description", "")),
        }
        for model_name, metadata in registry.items()
    ]


def resolve_tracker_assets(model_name: str, *, repo_root: str | Path | None = None) -> Any:
    """Backward-compatible public wrapper around the checkpoint-backed asset resolver."""
    from human_motion_isaacsim.checkpoint import _resolve_tracker_assets

    return _resolve_tracker_assets(model_name, repo_root=repo_root)
