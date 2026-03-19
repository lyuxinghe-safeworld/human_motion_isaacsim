#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Launch Isaac Sim with a visible viewport and capture one monitor-backed frame.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output", type=str, default="output/monitor_probe.png", help="PNG output path")
    parser.add_argument("--device", type=str, default="cuda:0", help="Isaac Sim device")
    parser.add_argument("--warmup-frames", type=int, default=3, help="Frames to render before capture")
    return parser.parse_args()


def main():
    args = parse_args()

    from isaaclab.app import AppLauncher

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    app_launcher = AppLauncher({"headless": False, "device": args.device})
    try:
        for _ in range(max(args.warmup_frames, 0)):
            app_launcher.app.update()

        from human_motion_isaacsim.recording import capture_active_viewport_to_file

        capture_active_viewport_to_file(output_path, simulation_app=app_launcher.app)
        print(f"monitor_probe={output_path}", flush=True)
    finally:
        # This is a one-shot probe script, so we prefer an eager Kit shutdown.
        app_launcher.app.close(wait_for_replicator=False)


if __name__ == "__main__":
    main()
