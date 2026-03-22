from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def is_gcs_uri(value: str | Path | None) -> bool:
    """Return True when the provided value is a gs:// URI."""
    if value is None:
        return False
    return str(value).strip().startswith("gs://")


def stage_gcs_uri(uri: str, destination_root: str | Path) -> Path:
    """Download a GCS object into destination_root and return the staged local path."""
    destination = Path(destination_root)
    destination.mkdir(parents=True, exist_ok=True)

    file_name = Path(uri.rstrip("/")).name or "downloaded"
    output_path = destination / file_name

    command = _copy_command(uri, output_path)
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else ""
        raise RuntimeError(f"Failed to download {uri} to {output_path}: {stderr}") from exc

    return output_path


def _copy_command(uri: str, output_path: Path) -> list[str]:
    """Choose an installed GCS CLI for object download."""
    if shutil.which("gcloud"):
        return ["gcloud", "storage", "cp", uri, str(output_path)]
    if shutil.which("gsutil"):
        return ["gsutil", "cp", uri, str(output_path)]
    raise RuntimeError("GCS download requires either 'gcloud' or 'gsutil' to be installed.")
