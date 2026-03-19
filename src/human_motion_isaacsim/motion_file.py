from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(slots=True)
class MotionMetadata:
    path: Path
    fps: int
    num_frames: int

    @property
    def duration_seconds(self) -> float:
        return self.num_frames / self.fps


def load_motion_metadata(path: str | Path) -> MotionMetadata:
    motion_path = Path(path)
    if motion_path.suffix != ".motion":
        raise ValueError(f"Motion file must use the .motion suffix: {motion_path}")

    payload = torch.load(motion_path, map_location="cpu", weights_only=False)
    rigid_body_pos = payload["rigid_body_pos"]
    fps = int(payload.get("fps", 30))
    num_frames = int(rigid_body_pos.shape[0])
    return MotionMetadata(path=motion_path, fps=fps, num_frames=num_frames)
