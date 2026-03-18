from __future__ import annotations

import importlib
from pathlib import Path

from moviepy.video.io.ImageSequenceClip import ImageSequenceClip


def frame_path_for_step(frames_dir: str | Path, step: int) -> Path:
    return Path(frames_dir) / f"{step:06d}.png"


def capture_active_viewport_to_file(
    file_path: str | Path,
    *,
    simulation_app,
    flush_updates: int = 3,
) -> None:
    if simulation_app is None:
        raise ValueError("simulation_app is required for viewport capture")

    viewport_utility = importlib.import_module("omni.kit.viewport.utility")
    kit_app = importlib.import_module("omni.kit.app")
    renderer_capture = importlib.import_module("omni.renderer_capture")

    async def _capture():
        helper = viewport_utility.capture_viewport_to_file(
            viewport_utility.get_active_viewport(),
            file_path=str(file_path),
        )
        await helper.wait_for_result()

        app = kit_app.get_app()
        capture_iface = renderer_capture.acquire_renderer_capture_interface()
        for _ in range(flush_updates):
            capture_iface.wait_async_capture()
            await app.next_update_async()

    simulation_app.run_coroutine(_capture())


def compile_video(frame_paths: list[Path], video_path: str | Path, fps: int = 30) -> None:
    clip = ImageSequenceClip([str(path) for path in frame_paths], fps=fps)
    clip.write_videofile(
        str(video_path),
        codec="libx264",
        audio=False,
        threads=4,
        preset="veryfast",
        ffmpeg_params=[
            "-profile:v",
            "main",
            "-level",
            "4.0",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-crf",
            "23",
        ],
    )
