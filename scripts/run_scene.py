#!/usr/bin/env python3
"""Run a custom Isaac Sim scene with optional ProtoMotions humanoid control.

Standalone mode (no --motion-file):
  Humanoid stands in rest pose in a scene with static objects.

ProtoMotions mode (--motion-file provided):
  ProtoMotions agent controls the humanoid, tracking the given motion file.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a custom Isaac Sim scene with optional ProtoMotions control.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to ProtoMotions tracker checkpoint (needed for humanoid asset path)",
    )
    parser.add_argument(
        "--motion-file", type=str, default="",
        help="Path to .motion file. If omitted, runs standalone (rest pose).",
    )
    parser.add_argument("--headless", action="store_true", help="Run Isaac Sim headless")
    parser.add_argument(
        "--video-output", type=str, default="",
        help="Output MP4 path. If omitted, no video saved.",
    )
    parser.add_argument(
        "--reference-markers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render reference-motion marker spheres during headless video capture.",
    )
    return parser.parse_args()


def _resolve_model_for_checkpoint(checkpoint_path: str) -> str:
    """Map the wrapper checkpoint argument onto a registered public model name."""
    import human_motion_isaacsim as hmi
    from human_motion_isaacsim._registry import resolve_tracker_assets

    checkpoint = Path(checkpoint_path).resolve()
    for model_entry in hmi.list_models():
        model_name = model_entry["name"]
        try:
            tracker_assets = resolve_tracker_assets(model_name)
        except FileNotFoundError:
            continue
        if tracker_assets.checkpoint_path.resolve() == checkpoint:
            return model_name

    raise ValueError(
        f"Checkpoint {checkpoint} does not match any registered model. "
        "Use a checkpoint that belongs to a supported model."
    )


def build_scene(checkpoint_path: str, headless: bool):
    """Create the local Isaac Sim scene used by the wrapper script."""
    from human_motion_isaacsim.checkpoint import load_tracker_assets
    from human_motion_isaacsim.protomotions_path import ensure_protomotions_importable

    ensure_protomotions_importable()
    tracker_assets = load_tracker_assets(checkpoint_path)

    from isaacsim import SimulationApp
    from omni.isaac.core import World
    from omni.isaac.core.articulations import Articulation
    from omni.isaac.core.objects import GroundPlane
    from omni.isaac.core.prims import RigidPrimView
    from omni.isaac.core.utils.stage import add_reference_to_stage
    from scene_utils import GROUND_PLANE_PRIM_PATH, populate_scene

    simulation_app = SimulationApp({"headless": headless})

    fps = getattr(tracker_assets.simulator_config.sim, "fps", 60)
    world = World(physics_dt=1.0 / fps, rendering_dt=1.0 / fps)
    world.scene.add(GroundPlane(prim_path=GROUND_PLANE_PRIM_PATH, size=100.0))

    asset_root = tracker_assets.robot_config.asset.asset_root
    usd_file = tracker_assets.robot_config.asset.usd_asset_file_name
    humanoid_usd_path = str(Path(asset_root) / usd_file)
    add_reference_to_stage(humanoid_usd_path, "/World/Humanoid")
    articulation = world.scene.add(
        Articulation(prim_path="/World/Humanoid", name="humanoid")
    )

    populate_scene(world)

    body_names = tracker_assets.robot_config.kinematic_info.body_names
    body_prim_paths = [f"/World/Humanoid/bodies/{name}" for name in body_names]
    body_rigid_view = RigidPrimView(
        prim_paths_expr=body_prim_paths,
        name="humanoid_bodies",
    )
    world.scene.add(body_rigid_view)
    world.reset()

    return simulation_app, world, articulation, body_rigid_view, tracker_assets


def run_standalone(world, simulation_app, headless: bool):
    """Run the scene with the humanoid in rest pose."""
    print("Running standalone mode (humanoid in rest pose). Press Ctrl+C to exit.")
    try:
        while simulation_app.is_running():
            world.step(render=not headless)
    except KeyboardInterrupt:
        pass
    finally:
        simulation_app.close()


def run_protomotions(
    world,
    articulation,
    body_rigid_view,
    simulation_app,
    motion_file: str,
    model: str,
    headless: bool,
    video_output: str | None,
    reference_markers: bool,
):
    """Run ProtoMotions control through the package-owned API."""
    import human_motion_isaacsim as hmi
    from scene_utils import align_scene_to_humanoid_root

    articulation.body_rigid_view = body_rigid_view
    world.simulation_app = simulation_app
    world.scene_alignment_callback = lambda simulator: align_scene_to_humanoid_root(
        world,
        simulator,
    )
    try:
        hmi.init(
            model=model,
            world=world,
            articulation=articulation,
            headless=headless,
            reference_markers=reference_markers,
        )
        hmi.run(
            motion_file,
            video_output=video_output if video_output else None,
        )
    finally:
        simulation_app.close()


def main():
    args = parse_args()
    model = _resolve_model_for_checkpoint(args.checkpoint)

    simulation_app, world, articulation, body_rigid_view, _ = build_scene(
        args.checkpoint, args.headless,
    )

    if not args.motion_file:
        run_standalone(world, simulation_app, args.headless)
    else:
        run_protomotions(
            world=world,
            articulation=articulation,
            body_rigid_view=body_rigid_view,
            simulation_app=simulation_app,
            motion_file=args.motion_file,
            model=model,
            headless=args.headless,
            video_output=args.video_output if args.video_output else None,
            reference_markers=args.reference_markers,
        )


if __name__ == "__main__":
    main()
