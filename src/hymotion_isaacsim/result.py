from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class MotionRunResult:
    success: bool
    motion_file: Path
    video_output: Path | None = None
    num_steps: int = 0
    duration_seconds: float = 0.0
    error_message: str | None = None
