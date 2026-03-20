from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class MotionRunResult:
    """Outcome of a single motion-tracking run, including optional video path and metrics."""

    success: bool
    motion_file: Path
    video_output: Path | None = None
    num_steps: int = 0
    duration_seconds: float = 0.0
    error_message: str | None = None
    tracking_score: float | None = None

    @property
    def output_video_path(self) -> Path | None:
        """Alias for video_output, kept for backward compatibility."""
        return self.video_output

    @output_video_path.setter
    def output_video_path(self, value: Path | None) -> None:
        self.video_output = value
