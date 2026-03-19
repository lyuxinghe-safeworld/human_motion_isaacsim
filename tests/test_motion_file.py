from pathlib import Path

import pytest
import torch


def test_load_motion_metadata_reads_fps_and_frame_count(tmp_path):
    from human_motion_isaacsim import load_motion_metadata

    motion_path = tmp_path / "example.motion"
    torch.save(
        {
            "fps": 30,
            "rigid_body_pos": torch.zeros((90, 24, 3)),
        },
        motion_path,
    )

    meta = load_motion_metadata(motion_path)

    assert meta.path == motion_path
    assert meta.fps == 30
    assert meta.num_frames == 90
    assert meta.duration_seconds == 3.0


def test_load_motion_metadata_rejects_non_motion_suffix(tmp_path):
    from human_motion_isaacsim import load_motion_metadata

    bad_path = tmp_path / "example.pt"
    torch.save({"fps": 30, "rigid_body_pos": torch.zeros((1, 1, 3))}, bad_path)

    with pytest.raises(ValueError, match="\\.motion"):
        load_motion_metadata(bad_path)
