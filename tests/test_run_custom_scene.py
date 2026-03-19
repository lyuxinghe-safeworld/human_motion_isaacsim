from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch


def test_parse_args_enables_reference_markers_by_default(monkeypatch):
    from scripts.run_custom_scene import parse_args

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_custom_scene.py",
            "--checkpoint", "/tmp/checkpoint.ckpt",
            "--motion-file", "/tmp/sample.motion",
        ],
    )

    args = parse_args()

    assert args.reference_markers is True


def test_parse_args_allows_disabling_reference_markers(monkeypatch):
    from scripts.run_custom_scene import parse_args

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_custom_scene.py",
            "--checkpoint", "/tmp/checkpoint.ckpt",
            "--motion-file", "/tmp/sample.motion",
            "--no-reference-markers",
        ],
    )

    args = parse_args()

    assert args.reference_markers is False


def test_align_scene_to_humanoid_root_uses_root_xy(monkeypatch):
    from scripts import run_custom_scene

    set_scene_origin = MagicMock()
    monkeypatch.setattr(
        "hymotion_isaacsim.custom_scene.set_scene_origin",
        set_scene_origin,
    )

    simulator = SimpleNamespace(
        _get_simulator_root_state=MagicMock(
            return_value=SimpleNamespace(root_pos=torch.tensor([[12.5, -3.0, 0.9]]))
        )
    )

    world = object()
    run_custom_scene._align_scene_to_humanoid_root(world, simulator)

    set_scene_origin.assert_called_once_with(world, (12.5, -3.0, 0.0))


def test_enable_reference_markers_for_capture_builds_markers_when_headless():
    from scripts.run_custom_scene import _enable_reference_markers_for_capture

    markers = {"body_markers_red": object()}
    env = SimpleNamespace(
        control_manager=SimpleNamespace(
            create_visualization_markers=MagicMock(return_value=markers)
        )
    )
    simulator = SimpleNamespace(
        headless=True,
        _build_visualization_markers=MagicMock(),
    )

    _enable_reference_markers_for_capture(env, simulator)

    env.control_manager.create_visualization_markers.assert_called_once_with(headless=False)
    simulator._build_visualization_markers.assert_called_once_with(markers)


def test_update_reference_markers_for_capture_restores_headless_flag():
    from scripts.run_custom_scene import _update_reference_markers_for_capture

    marker_state = {"body_markers_red": object()}
    env = SimpleNamespace(
        control_manager=SimpleNamespace(
            get_markers_state=MagicMock(return_value=marker_state)
        )
    )
    simulator = SimpleNamespace(
        headless=True,
        _update_simulator_markers=MagicMock(),
    )

    _update_reference_markers_for_capture(env, simulator)

    assert simulator.headless is True
    env.control_manager.get_markers_state.assert_called_once_with()
    simulator._update_simulator_markers.assert_called_once_with(marker_state)


def test_prepare_headless_capture_for_video_builds_markers_and_primes_camera(monkeypatch):
    from scripts.run_custom_scene import _prepare_headless_capture_for_video

    enable_markers = MagicMock()
    monkeypatch.setattr(
        "scripts.run_custom_scene._enable_reference_markers_for_capture",
        enable_markers,
    )
    env = SimpleNamespace()
    simulator = SimpleNamespace(headless=True, prepare_headless_capture=MagicMock())

    _prepare_headless_capture_for_video(env, simulator)

    enable_markers.assert_called_once_with(env, simulator)
    simulator.prepare_headless_capture.assert_called_once_with()


def test_prepare_headless_capture_for_video_skips_markers_when_disabled(monkeypatch):
    from scripts.run_custom_scene import _prepare_headless_capture_for_video

    enable_markers = MagicMock()
    monkeypatch.setattr(
        "scripts.run_custom_scene._enable_reference_markers_for_capture",
        enable_markers,
    )
    env = SimpleNamespace()
    simulator = SimpleNamespace(headless=True, prepare_headless_capture=MagicMock())

    _prepare_headless_capture_for_video(env, simulator, enable_reference_markers=False)

    enable_markers.assert_not_called()
    simulator.prepare_headless_capture.assert_called_once_with()


def test_plan_motion_max_steps_uses_env_step_rate():
    from scripts.run_custom_scene import _plan_motion_max_steps

    simulator_config = SimpleNamespace(sim=SimpleNamespace(fps=120, decimation=4))

    assert _plan_motion_max_steps(3.0, simulator_config) == 90
