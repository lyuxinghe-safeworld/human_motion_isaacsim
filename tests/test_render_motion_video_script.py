from __future__ import annotations

import json
import os
import stat
import subprocess
from pathlib import Path


def _write_executable(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _prepare_fake_repo(tmp_path: Path) -> tuple[Path, Path]:
    repo_root = tmp_path / "repo"
    wrapper_source = Path("scripts/run_custom_scene.sh").read_text()
    _write_executable(repo_root / "scripts" / "run_custom_scene.sh", wrapper_source)
    (repo_root / "scripts" / "run_custom_scene.py").write_text("# stub\n")

    log_path = repo_root / "wrapper-log.json"
    _write_executable(
        repo_root / "env" / ".venv" / "bin" / "python",
        """#!/usr/bin/env python3
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

keys = [
    "DISPLAY",
    "LD_LIBRARY_PATH",
    "MASTER_ADDR",
    "MASTER_PORT",
    "MPLCONFIGDIR",
    "NCCL_IB_DISABLE",
    "NCCL_NET",
    "OMNI_KIT_ACCEPT_EULA",
]
Path(os.environ["FAKE_WRAPPER_LOG"]).write_text(
    json.dumps(
        {
            "argv": sys.argv[1:],
            "env": {key: os.environ.get(key) for key in keys},
        }
    )
)
""",
    )

    default_checkpoint = (
        repo_root
        / "third_party"
        / "ProtoMotions"
        / "data"
        / "pretrained_models"
        / "motion_tracker"
        / "smpl"
        / "last.ckpt"
    )
    default_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    default_checkpoint.write_text("checkpoint")

    return repo_root, log_path


def _run_wrapper(repo_root: Path, log_path: Path, *args: str) -> dict:
    env = os.environ.copy()
    env.pop("DISPLAY", None)
    env.pop("MPLCONFIGDIR", None)
    env["FAKE_WRAPPER_LOG"] = str(log_path)

    subprocess.run(
        [str(repo_root / "scripts" / "run_custom_scene.sh"), *args],
        cwd=repo_root,
        env=env,
        check=True,
    )

    return json.loads(log_path.read_text())


def test_run_custom_scene_wrapper_defaults_to_headless(tmp_path):
    repo_root, log_path = _prepare_fake_repo(tmp_path)
    motion_file = repo_root / "motions" / "walk_cycle.motion"
    motion_file.parent.mkdir(parents=True, exist_ok=True)
    motion_file.write_text("motion")

    result = _run_wrapper(repo_root, log_path, "--motion-file", str(motion_file))

    assert result["argv"] == [
        str(repo_root / "scripts" / "run_custom_scene.py"),
        "--checkpoint",
        str(
            repo_root
            / "third_party"
            / "ProtoMotions"
            / "data"
            / "pretrained_models"
            / "motion_tracker"
            / "smpl"
            / "last.ckpt"
        ),
        "--motion-file",
        str(motion_file),
        "--video-output",
        str(repo_root / "output" / "walk_cycle.mp4"),
        "--headless",
        "--reference-markers",
    ]
    assert result["env"]["OMNI_KIT_ACCEPT_EULA"] == "YES"
    assert result["env"]["MPLCONFIGDIR"] == "/tmp/matplotlib"
    assert result["env"]["DISPLAY"] is None
    assert result["env"]["NCCL_IB_DISABLE"] == "1"
    assert result["env"]["NCCL_NET"] == "Socket"
    assert result["env"]["MASTER_ADDR"] == "127.0.0.1"
    assert result["env"]["MASTER_PORT"] == "29500"
    assert str(
        repo_root
        / "env"
        / ".venv"
        / "lib"
        / "python3.11"
        / "site-packages"
        / "isaacsim"
        / "kit"
        / "extscore"
        / "omni.client.lib"
        / "bin"
    ) in result["env"]["LD_LIBRARY_PATH"]


def test_run_custom_scene_wrapper_sets_display_when_not_headless(tmp_path):
    repo_root, log_path = _prepare_fake_repo(tmp_path)
    motion_file = repo_root / "motions" / "gesture.motion"
    motion_file.parent.mkdir(parents=True, exist_ok=True)
    motion_file.write_text("motion")
    checkpoint = repo_root / "checkpoints" / "custom.ckpt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_text("checkpoint")
    video_output = repo_root / "output" / "custom.mp4"

    result = _run_wrapper(
        repo_root,
        log_path,
        "--motion-file",
        str(motion_file),
        "--headless",
        "false",
        "--checkpoint",
        str(checkpoint),
        "--reference-markers",
        "false",
        "--video-output",
        str(video_output),
        "--display",
        ":1",
    )

    assert result["argv"] == [
        str(repo_root / "scripts" / "run_custom_scene.py"),
        "--checkpoint",
        str(checkpoint),
        "--motion-file",
        str(motion_file),
        "--video-output",
        str(video_output),
        "--no-reference-markers",
    ]
    assert result["env"]["DISPLAY"] == ":1"
