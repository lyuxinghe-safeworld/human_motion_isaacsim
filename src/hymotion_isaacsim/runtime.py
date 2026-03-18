from __future__ import annotations

from pathlib import Path
from typing import Callable, Any

from hymotion_isaacsim.binding import bind_fixed_humanoid
from hymotion_isaacsim.checkpoint import load_tracker_assets
from hymotion_isaacsim.motion_file import load_motion_metadata


class ProtoMotionIsaacSimController:
    def __init__(
        self,
        *,
        humanoid_prim_path: str,
        checkpoint_path: str | Path,
        lookup_articulation: Callable[[str], Any],
        bind_humanoid: Callable[..., Any] = bind_fixed_humanoid,
        load_assets: Callable[[str | Path], Any] = load_tracker_assets,
        motion_runner: Callable[[Any, Any, str | Path | None], Any] | None = None,
        restore_rest_pose: Callable[[Any], None] | None = None,
    ) -> None:
        self.humanoid_prim_path = humanoid_prim_path
        self.checkpoint_path = Path(checkpoint_path)
        self.lookup_articulation = lookup_articulation
        self.bound_humanoid = bind_humanoid(
            humanoid_prim_path,
            lookup_articulation=lookup_articulation,
        )
        self.tracker_assets = load_assets(self.checkpoint_path)
        self._motion_runner = motion_runner
        self._restore_rest_pose = restore_rest_pose
        self._busy = False

    def run_motion(self, motion_file: str, video_output: str | None = None):
        if self._busy:
            raise RuntimeError("Motion execution already in progress")

        # NOTE(v1): run_motion is intentionally blocking and exclusive. While a
        # motion is active, this controller owns env stepping and assumes no
        # external process is mutating the humanoid or stage state.
        metadata = load_motion_metadata(motion_file)
        self._busy = True
        try:
            if self._motion_runner is None:
                raise NotImplementedError(
                    "The controller shell is initialized, but the execution loop is not wired yet."
                )
            result = self._motion_runner(self, metadata, video_output)
        except Exception:
            if self._restore_rest_pose is not None:
                self._restore_rest_pose(self)
            raise
        else:
            if self._restore_rest_pose is not None:
                self._restore_rest_pose(self)
            return result
        finally:
            self._busy = False
