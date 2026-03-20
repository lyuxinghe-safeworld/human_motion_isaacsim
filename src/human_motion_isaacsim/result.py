from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class MotionRunResult:
    success: bool
    motion_file: Path
    video_output: Path | None = None
    output_video_path: Path | None = None
    num_steps: int = 0
    duration_seconds: float = 0.0
    tracking_score: float | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        if self.output_video_path is None:
            self.output_video_path = self.video_output
        elif self.video_output is None:
            self.video_output = self.output_video_path
