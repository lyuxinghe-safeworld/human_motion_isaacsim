from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from human_motion_isaacsim._registry import _load_registry
from human_motion_isaacsim.protomotions_path import ensure_protomotions_importable, resolve_protomotions_root


@dataclass(slots=True)
class TrackerAssets:
    """Immutable bundle of paths and deserialized configs loaded from a tracker checkpoint."""

    checkpoint_path: Path
    resolved_config_path: Path
    robot_config: Any
    simulator_config: Any
    terrain_config: Any
    scene_lib_config: Any
    motion_lib_config: Any
    env_config: Any
    agent_config: Any


def tracker_kinematic_layout(tracker_assets: Any) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Extract body and joint name tuples from a TrackerAssets' robot config."""
    robot_config = getattr(tracker_assets, "robot_config", None)
    kinematic_info = getattr(robot_config, "kinematic_info", None)
    body_names = getattr(kinematic_info, "body_names", None)
    joint_names = getattr(kinematic_info, "joint_names", None)

    if body_names is None or joint_names is None:
        raise ValueError("Tracker assets are missing robot_config.kinematic_info body/joint metadata.")

    return tuple(body_names), tuple(joint_names)


def _protomotions_root() -> Path:
    """Convenience wrapper returning the resolved ProtoMotions package root."""
    return resolve_protomotions_root()


def _loaded_protomotions_root() -> Path | None:
    """Return the root of the currently-imported protomotions package, or None if not loaded."""
    module = sys.modules.get("protomotions")
    module_file = getattr(module, "__file__", None) if module is not None else None
    if not module_file:
        return None
    return Path(module_file).resolve().parent.parent


def _ensure_tracker_protomotions_importable(protomotions_root: str | Path | None = None) -> Path:
    """Ensure the protomotions package from the given root is importable, hot-swapping if needed."""
    if protomotions_root is None:
        return ensure_protomotions_importable()

    root = Path(protomotions_root).resolve()
    if not (root / "protomotions" / "__init__.py").exists():
        raise FileNotFoundError(f"ProtoMotions checkout not found at {root}")

    if _loaded_protomotions_root() == root:
        return root

    root_str = str(root)
    sys.path[:] = [path for path in sys.path if path != root_str]
    sys.path.insert(0, root_str)

    stale_modules = [
        module_name
        for module_name in list(sys.modules)
        if module_name == "protomotions" or module_name.startswith("protomotions.")
    ]
    for module_name in stale_modules:
        sys.modules.pop(module_name, None)

    importlib.invalidate_caches()
    importlib.import_module("protomotions")

    return root


def _normalize_robot_asset_root(robot_config: Any, *, protomotions_root: str | Path | None = None) -> None:
    """Convert a relative robot asset root to an absolute path anchored at the ProtoMotions root."""
    asset = getattr(robot_config, "asset", None)
    if asset is None:
        return

    asset_root = getattr(asset, "asset_root", "")
    if asset_root and not os.path.isabs(asset_root):
        root = Path(protomotions_root).resolve() if protomotions_root is not None else _protomotions_root()
        asset.asset_root = str((root / asset_root).resolve())


def resolved_config_path_for_checkpoint(checkpoint_path: str | Path) -> Path:
    """Return the expected path of the resolved-config sidecar for a given checkpoint file."""
    checkpoint = Path(checkpoint_path).resolve()
    return checkpoint.parent / "resolved_configs_inference.pt"


def load_tracker_assets(
    checkpoint_path: str | Path, *, protomotions_root: str | Path | None = None
) -> TrackerAssets:
    """Load a TrackerAssets bundle from a checkpoint file and its resolved-config sidecar."""
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


def _default_repo_root() -> Path:
    """Return the root of this repository based on the location of this source file."""
    return Path(__file__).resolve().parents[2]


def _repo_checkpoint_path(model_name: str, repo_root: Path) -> Path:
    """Build the path to a model checkpoint inside the local ProtoMotions vendor tree."""
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


def _repo_protomotions_root(repo_root: Path) -> Path:
    """Return the ProtoMotions package root within the given repository checkout."""
    return repo_root / "third_party" / "ProtoMotions"


def _cache_checkpoint_path(model_name: str) -> Path:
    """Build the XDG-cache path where a downloaded model checkpoint is stored."""
    cache_home = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache_home / "human_motion_isaacsim" / model_name / "last.ckpt"


def _has_local_assets(checkpoint_path: Path) -> bool:
    """Return True when both the checkpoint and its resolved-config sidecar exist on disk."""
    return checkpoint_path.exists() and resolved_config_path_for_checkpoint(checkpoint_path).exists()


def _resolve_tracker_assets(model_name: str, *, repo_root: str | Path | None = None) -> TrackerAssets:
    """Locate and load tracker assets for a registered model name.

    Searches the local repository vendor tree first, then the XDG cache
    directory.  Raises ``FileNotFoundError`` if no local assets are found.
    """
    registry = _load_registry()
    if model_name not in registry:
        raise KeyError(f"Unknown model: {model_name}")

    resolved_repo_root = Path(repo_root) if repo_root else _default_repo_root()
    repo_checkpoint = _repo_checkpoint_path(model_name, resolved_repo_root)
    if _has_local_assets(repo_checkpoint):
        return load_tracker_assets(
            repo_checkpoint,
            protomotions_root=_repo_protomotions_root(resolved_repo_root),
        )

    cache_checkpoint = _cache_checkpoint_path(model_name)
    if _has_local_assets(cache_checkpoint):
        return load_tracker_assets(cache_checkpoint)

    checkpoint_gcs = registry[model_name].get("checkpoint_gcs")
    if checkpoint_gcs:
        raise FileNotFoundError(
            f"No local tracker assets found for {model_name}; remote fetch is not implemented for {checkpoint_gcs}"
        )

    raise FileNotFoundError(f"No tracker assets found for {model_name}")
