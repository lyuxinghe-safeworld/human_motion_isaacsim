#!/usr/bin/env python3
"""Run a custom Isaac Sim scene with optional ProtoMotions humanoid control.

Standalone mode (no --motion-file):
  Humanoid stands in rest pose in a scene with static objects.

ProtoMotions mode (--motion-file provided):
  ProtoMotions agent controls the humanoid, tracking the given motion file.
"""
from __future__ import annotations

import argparse
import logging


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
        "--motion-file",
        action="append",
        default=[],
        help="Path or gs:// URI to a .motion file. Repeat to run multiple motions in sequence.",
    )
    parser.add_argument(
        "--manifest-path",
        action="append",
        default=[],
        help="Path or gs:// URI to a MotionBundle manifest. Repeat to run multiple manifests in sequence.",
    )
    parser.add_argument(
        "--representation",
        type=str,
        default="proto_motion",
        help="MotionBundle representation/derivative to run when using --manifest-path.",
    )
    parser.add_argument(
        "--staging-dir",
        type=str,
        default="",
        help="Optional local directory for staging GCS-backed manifests and motion files.",
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


def configure_logging() -> None:
    """Enable INFO-level API position logs for wrapper-driven motion runs."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


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
    motion_files: list[str],
    manifest_paths: list[str],
    model: str,
    headless: bool,
    video_output: str | None,
    reference_markers: bool,
    representation: str,
    staging_dir: str | None,
):
    """Run ProtoMotions control through the package-owned API."""
    import human_motion_isaacsim as hmi

    world.simulation_app = simulation_app
    try:
        hmi.init(
            model=model,
            world=world,
            articulation=articulation,
            headless=headless,
            reference_markers=reference_markers,
        )
        for motion_file in motion_files:
            hmi.run(motion_file, video_output=video_output if video_output else None)
        for manifest_path in manifest_paths:
            hmi.run(
                video_output=video_output if video_output else None,
                manifest_path=manifest_path,
                representation=representation,
                staging_dir=staging_dir,
            )
    finally:
        simulation_app.close()


def main():
    configure_logging()
    args = parse_args()
    from scene_utils import build_scene

    simulation_app, world, articulation = build_scene(
        args.model,
        args.headless,
    )

    if args.motion_file and args.manifest_path:
        raise SystemExit("Use either --motion-file or --manifest-path for a given run, not both.")

    if not args.motion_file and not args.manifest_path:
        run_standalone(world, simulation_app, args.headless)
    else:
        run_protomotions(
            world=world,
            articulation=articulation,
            simulation_app=simulation_app,
            motion_files=args.motion_file,
            manifest_paths=args.manifest_path,
            model=args.model,
            headless=args.headless,
            video_output=args.video_output if args.video_output else None,
            reference_markers=args.reference_markers,
            representation=args.representation,
            staging_dir=args.staging_dir or None,
        )


if __name__ == "__main__":
    main()
