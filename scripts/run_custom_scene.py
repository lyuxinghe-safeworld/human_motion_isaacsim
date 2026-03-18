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
    return parser.parse_args()


def build_scene(checkpoint_path: str, headless: bool):
    """Create the Isaac Sim world with ground plane, humanoid, and static objects.

    Returns (simulation_app, world, articulation, tracker_assets).
    """
    from hymotion_isaacsim.protomotions_path import ensure_protomotions_importable
    from hymotion_isaacsim.checkpoint import load_tracker_assets

    ensure_protomotions_importable()
    tracker_assets = load_tracker_assets(checkpoint_path)

    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": headless})

    from omni.isaac.core import World
    from omni.isaac.core.utils.stage import add_reference_to_stage
    from omni.isaac.core.articulations import Articulation
    from omni.isaac.core.objects import GroundPlane
    from hymotion_isaacsim.custom_scene import populate_scene

    # Physics dt from checkpoint config
    fps = getattr(tracker_assets.simulator_config.sim, "fps", 60)
    world = World(physics_dt=1.0 / fps, rendering_dt=1.0 / fps)

    # Ground plane
    world.scene.add(GroundPlane(prim_path="/World/GroundPlane", size=100.0))

    # Humanoid USD
    asset_root = tracker_assets.robot_config.asset.asset_root
    usd_file = tracker_assets.robot_config.asset.usd_asset_file_name
    humanoid_usd_path = str(Path(asset_root) / usd_file)
    add_reference_to_stage(humanoid_usd_path, "/World/Humanoid")
    articulation = world.scene.add(
        Articulation(prim_path="/World/Humanoid", name="humanoid")
    )

    # Static objects
    populate_scene(world)

    world.reset()

    return simulation_app, world, articulation, tracker_assets


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


def main():
    args = parse_args()
    simulation_app, world, articulation, tracker_assets = build_scene(
        args.checkpoint, args.headless,
    )

    if not args.motion_file:
        run_standalone(world, simulation_app, args.headless)
    else:
        # ProtoMotions mode — Task 7
        raise NotImplementedError("ProtoMotions mode not yet implemented. Pass --motion-file to use it.")


if __name__ == "__main__":
    main()
