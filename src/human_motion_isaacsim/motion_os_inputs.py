from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from human_motion_isaacsim.gcs import is_gcs_uri, stage_gcs_uri

Downloader = Callable[[str, Path], Path]


@dataclass(frozen=True, slots=True)
class ResolvedMotionInput:
    """Resolved runtime input for a replayable ProtoMotions motion."""

    motion_file: Path
    representation: str
    source_uri: str
    manifest_path: Path | None = None
    staging_dir: Path | None = None


def resolve_motion_input(
    *,
    motion_file: str | Path | None = None,
    manifest_path: str | Path | None = None,
    representation: str = "proto_motion",
    staging_dir: str | Path | None = None,
    downloader: Downloader | None = None,
) -> ResolvedMotionInput:
    """Resolve either a direct .motion file or a MotionBundle manifest into a local .motion path."""
    if bool(motion_file) == bool(manifest_path):
        raise ValueError("Provide exactly one of motion_file or manifest_path.")

    download = downloader or stage_gcs_uri
    stage_root = _resolve_staging_dir(staging_dir, needs_staging=bool(manifest_path) or is_gcs_uri(motion_file))

    if motion_file is not None:
        resolved_source = _normalize_source_reference(motion_file)
        resolved_motion = _stage_input_reference(
            resolved_source,
            staging_root=stage_root,
            stage_subdir=representation,
            downloader=download,
            base_dir=None,
            must_exist=False,
        )
        return ResolvedMotionInput(
            motion_file=resolved_motion,
            representation=representation,
            source_uri=resolved_source,
            manifest_path=None,
            staging_dir=stage_root,
        )

    manifest_source = _normalize_source_reference(manifest_path)
    local_manifest_path = _stage_input_reference(
        manifest_source,
        staging_root=stage_root,
        stage_subdir="manifest",
        downloader=download,
        base_dir=None,
        must_exist=True,
    )
    manifest_payload = _read_json_object(local_manifest_path)
    artifact_source = _select_representation_source(manifest_payload, representation)
    normalized_artifact_source = _normalize_source_reference(
        artifact_source,
        base_dir=local_manifest_path.parent,
    )
    local_motion_path = _stage_input_reference(
        normalized_artifact_source,
        staging_root=stage_root,
        stage_subdir=representation,
        downloader=download,
        base_dir=local_manifest_path.parent,
        must_exist=False,
    )
    return ResolvedMotionInput(
        motion_file=local_motion_path,
        representation=representation,
        source_uri=normalized_artifact_source,
        manifest_path=local_manifest_path.resolve(),
        staging_dir=stage_root,
    )


def _resolve_staging_dir(staging_dir: str | Path | None, *, needs_staging: bool) -> Path | None:
    if staging_dir is not None:
        path = Path(staging_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path
    if not needs_staging:
        return None
    return Path(tempfile.mkdtemp(prefix="human-motion-input-"))


def _stage_input_reference(
    source: str,
    *,
    staging_root: Path | None,
    stage_subdir: str,
    downloader: Downloader,
    base_dir: Path | None,
    must_exist: bool,
) -> Path:
    if is_gcs_uri(source):
        if staging_root is None:
            raise RuntimeError(f"Staging directory is required for GCS source: {source}")
        return downloader(source, staging_root / stage_subdir)

    path = Path(source)
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    path = path.resolve()
    if must_exist and not path.exists():
        raise FileNotFoundError(path)
    return path


def _normalize_source_reference(value: str | Path, base_dir: Path | None = None) -> str:
    if isinstance(value, Path):
        path_value = value
        if path_value.is_absolute():
            return str(path_value.resolve())
        if base_dir is not None:
            return str((base_dir / path_value).resolve())
        return str(path_value.resolve())

    string_value = str(value).strip()
    if not string_value:
        raise ValueError("Motion source reference must be a non-empty path or URI.")
    if is_gcs_uri(string_value):
        return string_value

    path = Path(string_value)
    if not path.is_absolute() and base_dir is not None:
        return str((base_dir / path).resolve())
    return str(path.resolve())


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Manifest is not valid JSON: {path}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"Manifest must be a JSON object: {path}")
    return payload


def _select_representation_source(manifest: dict[str, Any], representation: str) -> str:
    for container_name in ("derivatives", "representations", "runtime_derivatives"):
        entry = _lookup_container_entry(manifest.get(container_name), representation)
        if entry is not None:
            resolved = _extract_path_from_entry(entry, representation)
            if resolved is not None:
                return resolved

    if representation == "proto_motion":
        for key in ("proto_motion", "motion_file", "proto_motion_path"):
            value = manifest.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    gcs_paths = manifest.get("gcs_paths")
    if isinstance(gcs_paths, dict):
        for key in (representation, "proto_motion", "motion_file"):
            value = gcs_paths.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    raise KeyError(f"Manifest does not contain representation '{representation}'.")


def _lookup_container_entry(container: Any, representation: str) -> Any | None:
    if not isinstance(container, dict):
        return None
    if representation in container:
        return container[representation]

    for entry in container.values():
        if not isinstance(entry, dict):
            continue
        names = {
            str(entry.get("representation") or "").strip(),
            str(entry.get("kind") or "").strip(),
            str(entry.get("derivative_kind") or "").strip(),
            str(entry.get("name") or "").strip(),
        }
        if representation in names:
            return entry
    return None


def _extract_path_from_entry(entry: Any, representation: str) -> str | None:
    if isinstance(entry, str) and entry.strip():
        return entry.strip()

    if not isinstance(entry, dict):
        return None

    for key in (
        "local_path",
        "path",
        "gcs_path",
        "uri",
        "source_uri",
        "artifact_path",
        "motion_file",
        "file",
    ):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    gcs_paths = entry.get("gcs_paths")
    if isinstance(gcs_paths, dict):
        for key in (representation, "proto_motion", "motion", "artifact", "file"):
            value = gcs_paths.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    return None
