from __future__ import annotations

from pathlib import Path
from typing import Any

from human_motion_isaacsim._registry import (
    list_models as registry_list_models,
    resolve_tracker_assets,
)
from human_motion_isaacsim._state import PACKAGE_STATE


def init(
    model: str,
    world: Any,
    articulation: Any,
    headless: bool = True,
    reference_markers: bool = False,
) -> None:
    del headless
    del reference_markers

    PACKAGE_STATE.teardown()
    PACKAGE_STATE.model_name = model
    PACKAGE_STATE.tracker_assets = resolve_tracker_assets(model)
    PACKAGE_STATE.world = world
    PACKAGE_STATE.articulation = articulation


def run(motion_file: str | Path, video_output: str | Path | None = None) -> None:
    del motion_file
    del video_output

    if PACKAGE_STATE.model_name is None:
        raise RuntimeError("human_motion_isaacsim.init() must be called before run().")

    raise RuntimeError("The tracking loop is not implemented yet; Task 6 will wire the execution path.")


def list_models() -> list[dict[str, str]]:
    return registry_list_models()
