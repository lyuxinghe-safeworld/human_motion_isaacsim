from __future__ import annotations

import json
import os
from importlib import resources
from pathlib import Path
from typing import Any

from human_motion_isaacsim.checkpoint import (
    TrackerAssets,
    load_tracker_assets,
    resolved_config_path_for_checkpoint,
)


def _load_registry() -> dict[str, dict[str, Any]]:
    registry_text = resources.files("human_motion_isaacsim").joinpath("registry.json").read_text(
        encoding="utf-8"
    )
    data = json.loads(registry_text)
    if not isinstance(data, dict):
        raise ValueError("registry.json must contain a JSON object")
    return data


def list_models() -> list[dict[str, str]]:
    registry = _load_registry()
    return [
        {
            "name": model_name,
            "description": str(metadata.get("description", "")),
        }
        for model_name, metadata in registry.items()
    ]


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _repo_checkpoint_path(model_name: str, repo_root: Path) -> Path:
    return (
        repo_root
        / "third_party"
        / "ProtoMotions"
        / "data"
        / "pretrained_models"
        / "motion_tracker"
        / model_name
        / "last.ckpt"
    )


def _cache_checkpoint_path(model_name: str) -> Path:
    cache_home = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache_home / "human_motion_isaacsim" / model_name / "last.ckpt"


def _has_local_assets(checkpoint_path: Path) -> bool:
    return checkpoint_path.exists() and resolved_config_path_for_checkpoint(checkpoint_path).exists()


def resolve_tracker_assets(model_name: str, *, repo_root: str | Path | None = None) -> TrackerAssets:
    registry = _load_registry()
    if model_name not in registry:
        raise KeyError(f"Unknown model: {model_name}")

    repo_checkpoint = _repo_checkpoint_path(model_name, Path(repo_root) if repo_root else _default_repo_root())
    if _has_local_assets(repo_checkpoint):
        return load_tracker_assets(repo_checkpoint)

    cache_checkpoint = _cache_checkpoint_path(model_name)
    if _has_local_assets(cache_checkpoint):
        return load_tracker_assets(cache_checkpoint)

    checkpoint_gcs = registry[model_name].get("checkpoint_gcs")
    if checkpoint_gcs:
        raise FileNotFoundError(
            f"No local tracker assets found for {model_name}; remote fetch is not implemented for {checkpoint_gcs}"
        )

    raise FileNotFoundError(f"No tracker assets found for {model_name}")
