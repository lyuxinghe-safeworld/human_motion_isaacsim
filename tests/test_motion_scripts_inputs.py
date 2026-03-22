from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script_module(module_name: str, script_name: str):
    script_path = REPO_ROOT / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"unable to load script module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_smoke_motion_parse_args_accepts_manifest_inputs(monkeypatch) -> None:
    module = _load_script_module("test_smoke_motion_script", "smoke_motion.py")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "smoke_motion.py",
            "--checkpoint",
            "/tmp/last.ckpt",
            "--manifest-path",
            "gs://bucket/assets/manifest.json",
            "--representation",
            "proto_motion",
            "--staging-dir",
            "/tmp/staging",
        ],
    )

    args = module.parse_args()

    assert args.checkpoint == "/tmp/last.ckpt"
    assert args.motion_file is None
    assert args.manifest_path == "gs://bucket/assets/manifest.json"
    assert args.representation == "proto_motion"
    assert args.staging_dir == "/tmp/staging"


def test_run_scene_parse_args_accepts_manifest_sequences(monkeypatch) -> None:
    module = _load_script_module("test_run_scene_script", "run_scene.py")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_scene.py",
            "--manifest-path",
            "/tmp/manifest-a.json",
            "--manifest-path",
            "gs://bucket/assets/manifest-b.json",
            "--representation",
            "proto_motion",
            "--staging-dir",
            "/tmp/staging",
        ],
    )

    args = module.parse_args()

    assert args.motion_file == []
    assert args.manifest_path == ["/tmp/manifest-a.json", "gs://bucket/assets/manifest-b.json"]
    assert args.representation == "proto_motion"
    assert args.staging_dir == "/tmp/staging"
