#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from human_motion_isaacsim.protomotions_path import ensure_protomotions_importable


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run one ProtoMotions motion in Isaac Sim from an empty-scene setup.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to ProtoMotions tracker checkpoint")
    parser.add_argument("--motion-file", type=str, required=True, help="Path to ProtoMotions .motion file")
    parser.add_argument("--video-output", type=str, default="", help="Output MP4 path")
    parser.add_argument("--headless", action="store_true", help="Run Isaac Sim headless")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of environments")
    return parser.parse_args()


def main():
    args = parse_args()

    ensure_protomotions_importable()
    from protomotions.utils.simulator_imports import import_simulator_before_torch

    # Isaac Lab / Isaac Sim must be imported before torch.
    import_simulator_before_torch("isaaclab")

    from human_motion_isaacsim.motion_runner import MotionRunner

    runtime = MotionRunner.from_checkpoint_path(args.checkpoint)
    result = runtime.run_standalone_motion(
        checkpoint_path=args.checkpoint,
        motion_file=args.motion_file,
        video_output=args.video_output or None,
        headless=args.headless,
        num_envs=args.num_envs,
    )

    print(f"success={result.success}")
    print(f"motion_file={Path(result.motion_file).resolve()}")
    if result.video_output is not None:
        print(f"video_output={Path(result.video_output).resolve()}")
    print(f"num_steps={result.num_steps}")
    print(f"duration_seconds={result.duration_seconds}")


if __name__ == "__main__":
    main()
