from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from human_motion_isaacsim.protomotions_path import ensure_protomotions_importable, resolve_protomotions_root


@dataclass(slots=True)
class TrackerAssets:
    checkpoint_path: Path
    resolved_config_path: Path
    robot_config: Any
    simulator_config: Any
    terrain_config: Any
    scene_lib_config: Any
    motion_lib_config: Any
    env_config: Any
    agent_config: Any


def _protomotions_root() -> Path:
    return resolve_protomotions_root()


def _ensure_tracker_protomotions_importable(protomotions_root: str | Path | None = None) -> Path:
    if protomotions_root is None:
        return ensure_protomotions_importable()

    root = Path(protomotions_root).resolve()
    if not (root / "protomotions" / "__init__.py").exists():
        raise FileNotFoundError(f"ProtoMotions checkout not found at {root}")

    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    return root


def _normalize_robot_asset_root(robot_config: Any, *, protomotions_root: str | Path | None = None) -> None:
    asset = getattr(robot_config, "asset", None)
    if asset is None:
        return

    asset_root = getattr(asset, "asset_root", "")
    if asset_root and not os.path.isabs(asset_root):
        root = Path(protomotions_root).resolve() if protomotions_root is not None else _protomotions_root()
        asset.asset_root = str((root / asset_root).resolve())


def resolved_config_path_for_checkpoint(checkpoint_path: str | Path) -> Path:
    checkpoint = Path(checkpoint_path).resolve()
    return checkpoint.parent / "resolved_configs_inference.pt"


def load_tracker_assets(
    checkpoint_path: str | Path, *, protomotions_root: str | Path | None = None
) -> TrackerAssets:
    checkpoint = Path(checkpoint_path).resolve()
    resolved = resolved_config_path_for_checkpoint(checkpoint)
    if not resolved.exists():
        raise FileNotFoundError(f"Missing resolved_configs_inference.pt next to {checkpoint}")

    # Torch unpickling imports ProtoMotions config classes, so the repo path must
    # be on sys.path before we touch resolved_configs_inference.pt.
    root = _ensure_tracker_protomotions_importable(protomotions_root)
    configs = torch.load(resolved, map_location="cpu", weights_only=False)
    robot_config = configs["robot"]
    _normalize_robot_asset_root(robot_config, protomotions_root=root)

    return TrackerAssets(
        checkpoint_path=checkpoint,
        resolved_config_path=resolved,
        robot_config=robot_config,
        simulator_config=configs["simulator"],
        terrain_config=configs.get("terrain"),
        scene_lib_config=configs["scene_lib"],
        motion_lib_config=configs["motion_lib"],
        env_config=configs["env"],
        agent_config=configs["agent"],
    )
