from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace


def _install_fake_torch(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))


def _install_fake_protomotions_imports(monkeypatch) -> None:
    protomotions_module = types.ModuleType("protomotions")
    utils_module = types.ModuleType("protomotions.utils")
    simulator_imports_module = types.ModuleType("protomotions.utils.simulator_imports")
    simulator_imports_module.import_simulator_before_torch = lambda _name: None

    monkeypatch.setitem(sys.modules, "protomotions", protomotions_module)
    monkeypatch.setitem(sys.modules, "protomotions.utils", utils_module)
    monkeypatch.setitem(sys.modules, "protomotions.utils.simulator_imports", simulator_imports_module)


def _install_fake_runtime_modules(monkeypatch) -> None:
    simulator_adapter_module = types.ModuleType("human_motion_isaacsim.simulator_adapter")
    simulator_adapter_module.SimulatorAdapter = object

    viewport_capture_module = types.ModuleType("human_motion_isaacsim.viewport_capture")
    viewport_capture_module.compile_video = lambda frame_paths, video_path, fps: None
    viewport_capture_module.frame_path_for_step = lambda frames_dir, step: frames_dir / f"{step:06d}.png"

    monkeypatch.setitem(sys.modules, "human_motion_isaacsim.simulator_adapter", simulator_adapter_module)
    monkeypatch.setitem(sys.modules, "human_motion_isaacsim.viewport_capture", viewport_capture_module)


def test_api_run_resolves_manifest_backed_proto_motion(monkeypatch, tmp_path: Path) -> None:
    _install_fake_torch(monkeypatch)
    _install_fake_runtime_modules(monkeypatch)

    api = importlib.import_module("human_motion_isaacsim._api")
    package_state = importlib.import_module("human_motion_isaacsim._state").PACKAGE_STATE

    resolved_motion = tmp_path / "resolved.motion"
    seen: dict[str, object] = {}

    def fake_resolve_motion_input(**kwargs):
        seen["resolver_kwargs"] = kwargs
        return SimpleNamespace(motion_file=resolved_motion)

    def fake_build_runtime_bundle(motion_path: Path):
        seen["motion_path"] = motion_path
        return {
            "motion_metadata": SimpleNamespace(duration_seconds=1.0),
            "max_steps": 0,
            "env": object(),
            "agent": type("_Agent", (), {"eval": lambda self: None})(),
            "simulator": object(),
            "helpers": [],
        }

    monkeypatch.setattr(api, "resolve_motion_input", fake_resolve_motion_input)
    monkeypatch.setattr(api, "_build_runtime_bundle", fake_build_runtime_bundle)

    package_state.model_name = "smpl"
    package_state.world = object()
    package_state.articulation = object()
    package_state.reference_markers = False
    package_state.completed_run_count = 0
    package_state.next_run_root_position = None
    package_state.scene_reference_positions = {}

    result = api.run(
        manifest_path=tmp_path / "manifest.json",
        representation="proto_motion",
        staging_dir=tmp_path / "staging",
    )

    assert seen["resolver_kwargs"] == {
        "motion_file": None,
        "manifest_path": tmp_path / "manifest.json",
        "representation": "proto_motion",
        "staging_dir": tmp_path / "staging",
    }
    assert seen["motion_path"] == resolved_motion
    assert result.success is True
    assert result.motion_file == resolved_motion

    package_state.teardown()


def test_motion_runner_run_standalone_motion_resolves_manifest_backed_proto_motion(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _install_fake_torch(monkeypatch)
    _install_fake_protomotions_imports(monkeypatch)
    _install_fake_runtime_modules(monkeypatch)

    motion_runner_module = importlib.import_module("human_motion_isaacsim.motion_runner")
    motion_file_module = importlib.import_module("human_motion_isaacsim.motion_file")

    runtime = motion_runner_module.MotionRunner(
        tracker_assets=SimpleNamespace(
            env_config=SimpleNamespace(_target_="env.target"),
            agent_config=SimpleNamespace(_target_="agent.target"),
        )
    )

    resolved_motion = tmp_path / "resolved.motion"
    seen: dict[str, object] = {}

    class _FakeAgent:
        def eval(self):
            return None

        def add_agent_info_to_obs(self, obs):
            return obs

        def obs_dict_to_tensordict(self, obs):
            return obs

        def model(self, _obs_td):
            return {"action": object()}

    class _FakeEnv:
        def reset(self, _done_indices):
            return {}, {}

        def step(self, _actions):
            return {}, None, _FakeDones(), None, {}

    class _FakeDones:
        def nonzero(self, as_tuple=False):
            return self

        def squeeze(self, _index):
            return self

    class _FakeSimulator:
        def _write_viewport_to_file(self, _file_name):
            return None

        def close(self):
            return None

    def fake_resolve_motion_input(**kwargs):
        seen["resolver_kwargs"] = kwargs
        return SimpleNamespace(motion_file=resolved_motion)

    def fake_load_motion_metadata(_path):
        return motion_file_module.MotionMetadata(path=resolved_motion, fps=30, num_frames=0)

    def fake_build_standalone_runner(self, **kwargs):
        seen["build_kwargs"] = kwargs
        return {
            "app_launcher": None,
            "simulation_app": None,
            "fabric": object(),
            "env": _FakeEnv(),
            "agent": _FakeAgent(),
            "simulator": _FakeSimulator(),
        }

    compile_calls: dict[str, object] = {}

    monkeypatch.setattr(motion_runner_module, "ensure_protomotions_importable", lambda: None)
    monkeypatch.setattr(motion_runner_module, "resolve_motion_input", fake_resolve_motion_input)
    monkeypatch.setattr(motion_file_module, "load_motion_metadata", fake_load_motion_metadata)
    monkeypatch.setattr(
        motion_runner_module.MotionRunner,
        "build_standalone_runner",
        fake_build_standalone_runner,
    )
    monkeypatch.setattr(
        sys.modules["human_motion_isaacsim.viewport_capture"],
        "compile_video",
        lambda frame_paths, video_path, fps: compile_calls.update(
            {"video_path": video_path, "fps": fps, "frame_count": len(frame_paths)}
        ),
    )

    result = runtime.run_standalone_motion(
        checkpoint_path=tmp_path / "last.ckpt",
        manifest_path=tmp_path / "manifest.json",
        representation="proto_motion",
        staging_dir=tmp_path / "staging",
        video_output=tmp_path / "walk.mp4",
        headless=False,
        num_envs=1,
    )

    assert seen["resolver_kwargs"] == {
        "motion_file": None,
        "manifest_path": tmp_path / "manifest.json",
        "representation": "proto_motion",
        "staging_dir": tmp_path / "staging",
    }
    assert seen["build_kwargs"]["motion_file"] == resolved_motion
    assert result.success is True
    assert result.motion_file == resolved_motion
    assert compile_calls == {"video_path": tmp_path / "walk.mp4", "fps": 30, "frame_count": 0}
