from __future__ import annotations

import json
from pathlib import Path


def test_resolve_motion_input_returns_direct_local_motion_file(tmp_path: Path) -> None:
    from human_motion_isaacsim.motion_os_inputs import resolve_motion_input

    motion_path = tmp_path / "sample.motion"
    motion_path.write_text("payload", encoding="utf-8")

    result = resolve_motion_input(motion_file=motion_path)

    assert result.motion_file == motion_path.resolve()
    assert result.manifest_path is None
    assert result.representation == "proto_motion"
    assert result.source_uri == str(motion_path.resolve())


def test_resolve_motion_input_stages_direct_gcs_motion_file(tmp_path: Path) -> None:
    from human_motion_isaacsim.motion_os_inputs import resolve_motion_input

    staged_motion = tmp_path / "staged" / "remote.motion"
    calls: list[tuple[str, Path]] = []

    def fake_downloader(uri: str, destination_root: Path) -> Path:
        calls.append((uri, destination_root))
        staged_motion.parent.mkdir(parents=True, exist_ok=True)
        staged_motion.write_text("payload", encoding="utf-8")
        return staged_motion

    result = resolve_motion_input(
        motion_file="gs://bucket/motions/remote.motion",
        staging_dir=tmp_path / "staging-root",
        downloader=fake_downloader,
    )

    assert result.motion_file == staged_motion
    assert result.manifest_path is None
    assert result.representation == "proto_motion"
    assert result.source_uri == "gs://bucket/motions/remote.motion"
    assert calls == [("gs://bucket/motions/remote.motion", tmp_path / "staging-root" / "proto_motion")]


def test_resolve_motion_input_uses_proto_motion_from_local_manifest(tmp_path: Path) -> None:
    from human_motion_isaacsim.motion_os_inputs import resolve_motion_input

    motion_path = tmp_path / "local.motion"
    motion_path.write_text("payload", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "asset_id": "asset-123",
                "derivatives": {
                    "proto_motion": {
                        "format": "protomotions_motion_v1",
                        "gcs_path": str(motion_path),
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    result = resolve_motion_input(
        manifest_path=manifest_path,
        representation="proto_motion",
    )

    assert result.motion_file == motion_path.resolve()
    assert result.manifest_path == manifest_path.resolve()
    assert result.representation == "proto_motion"
    assert result.source_uri == str(motion_path.resolve())


def test_resolve_motion_input_stages_gcs_manifest_and_proto_motion(tmp_path: Path) -> None:
    from human_motion_isaacsim.motion_os_inputs import resolve_motion_input

    manifest_uri = "gs://bucket/assets/asset-123/manifest.json"
    motion_uri = "gs://bucket/assets/asset-123/derivatives/proto.motion"
    calls: list[tuple[str, Path]] = []

    def fake_downloader(uri: str, destination_root: Path) -> Path:
        calls.append((uri, destination_root))
        destination_root.mkdir(parents=True, exist_ok=True)
        if uri == manifest_uri:
            manifest_path = destination_root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "derivatives": {
                            "proto_motion": {
                                "format": "protomotions_motion_v1",
                                "gcs_path": motion_uri,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            return manifest_path

        if uri == motion_uri:
            motion_path = destination_root / "proto.motion"
            motion_path.write_text("payload", encoding="utf-8")
            return motion_path

        raise AssertionError(f"unexpected uri: {uri}")

    result = resolve_motion_input(
        manifest_path=manifest_uri,
        representation="proto_motion",
        staging_dir=tmp_path / "staging-root",
        downloader=fake_downloader,
    )

    assert result.manifest_path == tmp_path / "staging-root" / "manifest" / "manifest.json"
    assert result.motion_file == tmp_path / "staging-root" / "proto_motion" / "proto.motion"
    assert result.source_uri == motion_uri
    assert calls == [
        (manifest_uri, tmp_path / "staging-root" / "manifest"),
        (motion_uri, tmp_path / "staging-root" / "proto_motion"),
    ]
