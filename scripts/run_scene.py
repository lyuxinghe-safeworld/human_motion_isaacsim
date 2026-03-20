#!/usr/bin/env python3
"""Run a custom Isaac Sim scene with optional ProtoMotions humanoid control.

Standalone mode (no --motion-file):
  Humanoid stands in rest pose in a scene with static objects.

ProtoMotions mode (--motion-file provided):
  ProtoMotions agent controls the humanoid, tracking the given motion file.
"""
from __future__ import annotations

import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a custom Isaac Sim scene with optional ProtoMotions control.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="smpl",
        help="Registered human model name.",
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


def run_standalone(world, simulation_app, headless: bool):
    """Run the scene with the humanoid in rest pose."""
    print("Running standalone mode (humanoid in rest pose). Press Ctrl+C to exit.")
    try:
        world.reset()
        while simulation_app.is_running():
            world.step(render=not headless)
    except KeyboardInterrupt:
        pass
    finally:
        simulation_app.close()


def run_protomotions(
    world,
    articulation,
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
    from scene_utils import build_scene

    simulation_app, world, articulation = build_scene(
        args.model,
        args.headless,
    )

    if not args.motion_file:
        run_standalone(world, simulation_app, args.headless)
    else:
        run_protomotions(
            world=world,
            articulation=articulation,
            simulation_app=simulation_app,
            motion_file=args.motion_file,
            model=args.model,
            headless=args.headless,
            video_output=args.video_output if args.video_output else None,
            reference_markers=args.reference_markers,
        )


if __name__ == "__main__":
    main()
