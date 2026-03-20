from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(slots=True)
class MotionMetadata:
    """Lightweight descriptor for a .motion file: path, frame rate, and frame count."""

    path: Path
    fps: int
    num_frames: int

    @property
    def duration_seconds(self) -> float:
        """Total clip duration computed from frame count and FPS."""
        return self.num_frames / self.fps


def load_motion_metadata(path: str | Path) -> MotionMetadata:
    """Read a .motion file and return its metadata without loading full tensor data."""
    motion_path = Path(path)
    if motion_path.suffix != ".motion":
        raise ValueError(f"Motion file must use the .motion suffix: {motion_path}")

    payload = torch.load(motion_path, map_location="cpu", weights_only=False)
    rigid_body_pos = payload["rigid_body_pos"]
    fps = int(payload.get("fps", 30))
    num_frames = int(rigid_body_pos.shape[0])
    return MotionMetadata(path=motion_path, fps=fps, num_frames=num_frames)
