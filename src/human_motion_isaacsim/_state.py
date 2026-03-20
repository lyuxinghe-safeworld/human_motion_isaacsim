from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from human_motion_isaacsim.checkpoint import TrackerAssets


def _teardown_helper(helper: Any) -> None:
    """Call the first recognized teardown method on the given helper object."""
    for method_name in ("teardown", "destroy", "shutdown", "close"):
        method = getattr(helper, method_name, None)
        if callable(method):
            method()
            return


@dataclass(slots=True)
class PackageState:
    """Global mutable state for the initialized model, world, articulation, and related objects."""

    model_name: str | None = None
    tracker_assets: TrackerAssets | None = None
    world: Any | None = None
    articulation: Any | None = None
    body_rigid_view: Any | None = None
    headless: bool = True
    reference_markers: bool = False
    simulation_app: Any | None = None
    owned_helpers: list[Any] = field(default_factory=list)

    def teardown(self) -> None:
        """Reset all fields to defaults, tearing down owned helpers but preserving the simulation app."""
        first_error: Exception | None = None
        try:
            for helper in self.owned_helpers:
                if helper is self.simulation_app:
                    continue
                try:
                    _teardown_helper(helper)
                except Exception as exc:
                    if first_error is None:
                        first_error = exc
        finally:
            self.owned_helpers.clear()
            self.model_name = None
            self.tracker_assets = None
            self.world = None
            self.articulation = None
            self.body_rigid_view = None
            self.headless = True
            self.reference_markers = False

        if first_error is not None:
            raise first_error


PACKAGE_STATE = PackageState()


def _resolve_simulation_app(world: Any, articulation: Any) -> Any | None:
    """Search the world and articulation objects for an attached simulation app reference."""
    for owner in (world, articulation):
        for attr_name in ("simulation_app", "_simulation_app", "app", "_app"):
            candidate = getattr(owner, attr_name, None)
            if candidate is not None:
                return candidate
    return None


def _resolve_articulation_prim_path(articulation: Any) -> str:
    """Extract the USD prim path from an articulation, trying common attribute names."""
    for attr_name in ("prim_path", "_prim_path"):
        prim_path = getattr(articulation, attr_name, None)
        if prim_path:
            return str(prim_path)
    raise RuntimeError("Unable to determine articulation prim path for body view setup.")


def _resolve_body_rigid_view(world: Any, articulation: Any) -> Any | None:
    """Look up an existing body rigid-prim view from the articulation or world."""
    for owner in (articulation, world):
        for attr_name in ("body_rigid_view", "_body_rigid_view"):
            candidate = getattr(owner, attr_name, None)
            if candidate is not None:
                return candidate
    return None


def _cache_body_rigid_view(world: Any, articulation: Any, body_rigid_view: Any) -> None:
    """Store a body rigid-prim view back onto the articulation or world for later reuse."""
    if body_rigid_view is None:
        return

    for owner in (articulation, world):
        for attr_name in ("body_rigid_view", "_body_rigid_view"):
            try:
                setattr(owner, attr_name, body_rigid_view)
                return
            except Exception:
                continue


def _build_body_rigid_view(world: Any, articulation: Any, tracker_assets: Any) -> Any | None:
    """Construct a RigidPrimView covering every body in the tracker's kinematic layout.

    Returns ``None`` when the tracker assets lack body-name metadata.
    """
    robot_config = getattr(tracker_assets, "robot_config", None)
    kinematic_info = getattr(robot_config, "kinematic_info", None)
    body_names = getattr(kinematic_info, "body_names", None)
    if not body_names:
        return None

    from omni.isaac.core.prims import RigidPrimView

    prim_path = _resolve_articulation_prim_path(articulation)
    body_prim_paths = [f"{prim_path}/bodies/{name}" for name in body_names]
    view = RigidPrimView(
        prim_paths_expr=body_prim_paths,
        name=f"{getattr(articulation, 'name', 'humanoid')}_bodies",
    )

    scene = getattr(world, "scene", None)
    if scene is not None and hasattr(scene, "add"):
        added_view = scene.add(view)
        if added_view is not None:
            view = added_view

    initialize = getattr(view, "initialize", None)
    if callable(initialize):
        initialize()

    return view


def _resolve_scene_alignment_callback(world: Any, articulation: Any) -> Any | None:
    """Find a scene-alignment callback attached to the world or articulation, if any."""
    for owner in (world, articulation):
        for attr_name in ("scene_alignment_callback", "_scene_alignment_callback"):
            candidate = getattr(owner, attr_name, None)
            if callable(candidate):
                return candidate
    return None
