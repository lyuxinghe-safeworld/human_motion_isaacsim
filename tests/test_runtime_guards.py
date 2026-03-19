from dataclasses import dataclass
from pathlib import Path
import pytest
import torch
from types import SimpleNamespace


class _FakeArticulation:
    def __init__(self, body_names, joint_names):
        self.body_names = body_names
        self.joint_names = joint_names


def test_validate_humanoid_path_rejects_missing_prim():
    from human_motion_isaacsim.binding import bind_fixed_humanoid, StageBindingError

    def lookup(_prim_path):
        return None

    with pytest.raises(StageBindingError, match="not found"):
        bind_fixed_humanoid("/World/Humanoid", lookup_articulation=lookup)


def test_validate_humanoid_binding_rejects_incompatible_body_layout():
    from human_motion_isaacsim.binding import bind_fixed_humanoid, StageBindingError

    articulation = _FakeArticulation(
        body_names=["Pelvis", "WrongBody"],
        joint_names=["L_Hip_x"],
    )

    with pytest.raises(StageBindingError, match="body names"):
        bind_fixed_humanoid(
            "/World/Humanoid",
            lookup_articulation=lambda _prim_path: articulation,
        )


def test_load_tracker_checkpoint_requires_resolved_configs(tmp_path):
    from human_motion_isaacsim.checkpoint import load_tracker_assets

    checkpoint = tmp_path / "last.ckpt"
    checkpoint.write_bytes(b"checkpoint")

    with pytest.raises(FileNotFoundError, match="resolved_configs_inference.pt"):
        load_tracker_assets(checkpoint)


def test_load_tracker_checkpoint_returns_expected_targets(tmp_path):
    from human_motion_isaacsim.checkpoint import load_tracker_assets

    checkpoint = tmp_path / "last.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    resolved = checkpoint.parent / "resolved_configs_inference.pt"
    torch.save(
        {
            "robot": SimpleNamespace(asset=SimpleNamespace(asset_root="protomotions/data/assets")),
            "simulator": SimpleNamespace(_target_="sim.target"),
            "terrain": SimpleNamespace(_target_="terrain.target"),
            "scene_lib": SimpleNamespace(_target_="scene.target"),
            "motion_lib": SimpleNamespace(_target_="motion.target"),
            "env": SimpleNamespace(_target_="env.target"),
            "agent": SimpleNamespace(_target_="agent.target"),
        },
        resolved,
    )

    assets = load_tracker_assets(checkpoint)

    assert assets.resolved_config_path == resolved
    assert assets.env_config._target_ == "env.target"
    assert assets.agent_config._target_ == "agent.target"
    assert assets.robot_config.asset.asset_root.startswith("/")


def test_load_tracker_checkpoint_bootstraps_protomotions_before_torch_load(tmp_path, monkeypatch):
    import human_motion_isaacsim.checkpoint as checkpoint_module

    checkpoint = tmp_path / "last.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    resolved = checkpoint.parent / "resolved_configs_inference.pt"
    resolved.write_bytes(b"placeholder")

    call_order = []

    def fake_bootstrap():
        call_order.append("bootstrap")
        return tmp_path

    def fake_load(*args, **kwargs):
        call_order.append("torch.load")
        return {
            "robot": SimpleNamespace(asset=SimpleNamespace(asset_root="/tmp/assets")),
            "simulator": SimpleNamespace(_target_="sim.target"),
            "terrain": SimpleNamespace(_target_="terrain.target"),
            "scene_lib": SimpleNamespace(_target_="scene.target"),
            "motion_lib": SimpleNamespace(_target_="motion.target"),
            "env": SimpleNamespace(_target_="env.target"),
            "agent": SimpleNamespace(_target_="agent.target"),
        }

    monkeypatch.setattr(checkpoint_module, "ensure_protomotions_importable", fake_bootstrap)
    monkeypatch.setattr(checkpoint_module.torch, "load", fake_load)

    checkpoint_module.load_tracker_assets(checkpoint)

    assert call_order == ["bootstrap", "torch.load"]


def test_controller_initialization_binds_humanoid_and_checkpoint(tmp_path):
    from human_motion_isaacsim.runtime import ProtoMotionIsaacSimController

    checkpoint = tmp_path / "last.ckpt"
    checkpoint.write_bytes(b"checkpoint")

    expected_bound_humanoid = object()
    expected_assets = object()

    controller = ProtoMotionIsaacSimController(
        humanoid_prim_path="/World/Humanoid",
        checkpoint_path=checkpoint,
        lookup_articulation=lambda _prim_path: object(),
        bind_humanoid=lambda prim_path, *, lookup_articulation: expected_bound_humanoid,
        load_assets=lambda path: expected_assets,
    )

    assert controller.bound_humanoid is expected_bound_humanoid
    assert controller.tracker_assets is expected_assets


def test_controller_rejects_overlapping_run_requests(tmp_path):
    from human_motion_isaacsim.runtime import ProtoMotionIsaacSimController

    checkpoint = tmp_path / "last.ckpt"
    checkpoint.write_bytes(b"checkpoint")

    controller = ProtoMotionIsaacSimController(
        humanoid_prim_path="/World/Humanoid",
        checkpoint_path=checkpoint,
        lookup_articulation=lambda _prim_path: object(),
        bind_humanoid=lambda prim_path, *, lookup_articulation: object(),
        load_assets=lambda path: object(),
    )
    controller._busy = True

    with pytest.raises(RuntimeError, match="already in progress"):
        controller.run_motion("walk.motion")


def test_build_runtime_reuses_motion_tracker_agent_config():
    from human_motion_isaacsim.protomotions_runtime import ProtoMotionsRuntime

    assets = SimpleNamespace(
        env_config=SimpleNamespace(_target_="env.target"),
        agent_config=SimpleNamespace(_target_="agent.target"),
    )

    runtime = ProtoMotionsRuntime(tracker_assets=assets)

    assert runtime.env_target == "env.target"
    assert runtime.agent_target == "agent.target"


def test_runtime_plans_steps_from_motion_metadata():
    from human_motion_isaacsim.motion_file import MotionMetadata
    from human_motion_isaacsim.protomotions_runtime import ProtoMotionsRuntime

    assets = SimpleNamespace(
        env_config=SimpleNamespace(_target_="env.target"),
        agent_config=SimpleNamespace(_target_="agent.target"),
    )
    runtime = ProtoMotionsRuntime(tracker_assets=assets)
    motion = MotionMetadata(path=Path("walk.motion"), fps=30, num_frames=90)

    assert runtime.plan_num_steps(motion) == 90


def test_runtime_plans_steps_from_env_step_rate():
    from human_motion_isaacsim.motion_file import MotionMetadata
    from human_motion_isaacsim.protomotions_runtime import ProtoMotionsRuntime

    assets = SimpleNamespace(
        env_config=SimpleNamespace(_target_="env.target"),
        agent_config=SimpleNamespace(_target_="agent.target"),
        simulator_config=SimpleNamespace(sim=SimpleNamespace(fps=200, decimation=4)),
    )
    runtime = ProtoMotionsRuntime(tracker_assets=assets)
    motion = MotionMetadata(path=Path("walk.motion"), fps=30, num_frames=60)

    assert runtime.plan_num_steps(motion) == 100


def test_runtime_extends_episode_length_for_full_motion(tmp_path, monkeypatch):
    from human_motion_isaacsim.protomotions_runtime import ProtoMotionsRuntime

    assets = SimpleNamespace(
        robot_config=SimpleNamespace(asset=SimpleNamespace(asset_root="/tmp/assets")),
        simulator_config=SimpleNamespace(_target_="sim.target"),
        terrain_config=SimpleNamespace(_target_="terrain.target"),
        scene_lib_config=SimpleNamespace(_target_="scene.target"),
        motion_lib_config=SimpleNamespace(_target_="motion.target", motion_file=""),
        env_config=SimpleNamespace(_target_="env.target", max_episode_length=120),
        agent_config=SimpleNamespace(_target_="agent.target"),
    )
    runtime = ProtoMotionsRuntime(tracker_assets=assets)

    env_config_seen = {}
    fabric_config_seen = {}
    app_launcher_seen = {}

    class _FakeFabric:
        def __init__(self, **kwargs):
            self.device = "cpu"

        def launch(self):
            return None

    class _FakeAppLauncher:
        def __init__(self, flags):
            app_launcher_seen["flags"] = flags
            self.app = object()

    @dataclass
    class _FakeFabricConfig:
        accelerator: str = "gpu"
        devices: int = 1
        num_nodes: int = 1
        strategy: object = None
        precision: str = "32-true"
        loggers: list | None = None
        callbacks: list | None = None

        def __post_init__(self):
            fabric_config_seen["strategy"] = self.strategy

    def fake_build_all_components(**kwargs):
        return {
            "terrain": object(),
            "scene_lib": object(),
            "motion_lib": object(),
            "simulator": object(),
        }

    def fake_get_class(_target):
        class _FakeEnv:
            def __init__(self, **kwargs):
                env_config_seen["config"] = kwargs["config"]

        return _FakeEnv

    monkeypatch.setattr("human_motion_isaacsim.protomotions_runtime.ensure_protomotions_importable", lambda: tmp_path)
    monkeypatch.setattr("lightning.fabric.Fabric", _FakeFabric)
    monkeypatch.setattr("protomotions.utils.component_builder.build_all_components", fake_build_all_components)
    monkeypatch.setattr("protomotions.utils.fabric_config.FabricConfig", _FakeFabricConfig)
    monkeypatch.setattr("protomotions.utils.hydra_replacement.get_class", fake_get_class)
    monkeypatch.setattr("protomotions.utils.inference_utils.apply_backward_compatibility_fixes", lambda *args: None)
    monkeypatch.setattr("protomotions.utils.simulator_imports.import_simulator_before_torch", lambda _name: _FakeAppLauncher)
    monkeypatch.setattr("protomotions.envs.component_manager.TORCH_COMPILE_AVAILABLE", True)
    monkeypatch.setattr(
        "protomotions.simulator.base_simulator.utils.convert_friction_for_simulator",
        lambda terrain_config, simulator_config: (terrain_config, simulator_config),
    )

    class _FakeAgent:
        def __init__(self, **kwargs):
            pass

        def setup(self):
            return None

        def load(self, *args, **kwargs):
            return None

    monkeypatch.setattr("protomotions.utils.hydra_replacement.get_class", lambda _target: _FakeAgent if _target == "agent.target" else fake_get_class(_target))

    runtime.build_standalone_runner(
        checkpoint_path=tmp_path / "last.ckpt",
        motion_file=tmp_path / "walk.motion",
        max_steps=300,
        headless=True,
        enable_cameras=True,
        num_envs=1,
    )

    assert env_config_seen["config"].max_episode_length == 400
    assert env_config_seen["config"] is not assets.env_config
    assert fabric_config_seen["strategy"] == "auto"
    assert app_launcher_seen["flags"]["enable_cameras"] is True
    from protomotions.envs import component_manager as base_manager_module

    assert base_manager_module.TORCH_COMPILE_AVAILABLE is False


def test_compile_video_uses_hymotion_codec_settings(tmp_path, monkeypatch):
    from human_motion_isaacsim import recording

    calls = {}

    class _FakeClip:
        def __init__(self, frames, fps):
            calls["frames"] = frames
            calls["fps"] = fps

        def write_videofile(self, path, **kwargs):
            calls["path"] = path
            calls["kwargs"] = kwargs

    monkeypatch.setattr(recording, "ImageSequenceClip", _FakeClip)

    frame_paths = [tmp_path / "000000.png", tmp_path / "000001.png"]
    for frame_path in frame_paths:
        frame_path.write_bytes(b"png")

    recording.compile_video(frame_paths, tmp_path / "clip.mp4", fps=30)

    assert calls["fps"] == 30
    assert calls["path"].endswith("clip.mp4")
    assert calls["kwargs"]["codec"] == "libx264"
    assert calls["kwargs"]["audio"] is False
    assert calls["kwargs"]["preset"] == "veryfast"
    assert "-pix_fmt" in calls["kwargs"]["ffmpeg_params"]


def test_frame_path_sequence_is_zero_padded(tmp_path):
    from human_motion_isaacsim.recording import frame_path_for_step

    assert frame_path_for_step(tmp_path, 7).name == "000007.png"


def test_run_standalone_motion_uses_protomotions_viewport_capture(tmp_path, monkeypatch):
    from human_motion_isaacsim.motion_file import MotionMetadata
    from human_motion_isaacsim.protomotions_runtime import ProtoMotionsRuntime

    assets = SimpleNamespace(
        env_config=SimpleNamespace(_target_="env.target"),
        agent_config=SimpleNamespace(_target_="agent.target"),
    )
    runtime = ProtoMotionsRuntime(tracker_assets=assets)

    capture_calls = []
    compile_calls = {}

    class _FakeAgent:
        def eval(self):
            return None

        def add_agent_info_to_obs(self, obs):
            return obs

        def obs_dict_to_tensordict(self, obs):
            return obs

        def model(self, obs_td):
            return {"action": torch.zeros((1, 1))}

    class _FakeEnv:
        def reset(self, done_indices):
            return {}, {}

        def step(self, actions):
            return {}, torch.zeros(1), torch.zeros(1, dtype=torch.bool), None, {}

    class _FakeSimulator:
        def _write_viewport_to_file(self, file_name):
            capture_calls.append(file_name)
            Path(file_name).write_bytes(b"png")

        def close(self):
            return None

    monkeypatch.setattr(
        "human_motion_isaacsim.motion_file.load_motion_metadata",
        lambda _path: MotionMetadata(path=Path("walk.motion"), fps=30, num_frames=2),
    )
    monkeypatch.setattr(
        "protomotions.utils.simulator_imports.import_simulator_before_torch",
        lambda _name: None,
    )
    monkeypatch.setattr(
        ProtoMotionsRuntime,
        "build_standalone_runner",
        lambda self, **kwargs: {
            "app_launcher": None,
            "simulation_app": None,
            "fabric": object(),
            "env": _FakeEnv(),
            "agent": _FakeAgent(),
            "simulator": _FakeSimulator(),
        },
    )
    def fake_compile(frame_paths, video_path, fps):
        compile_calls["frame_paths"] = [Path(path) for path in frame_paths]
        compile_calls["video_path"] = Path(video_path)
        compile_calls["fps"] = fps

    monkeypatch.setattr("human_motion_isaacsim.recording.compile_video", fake_compile)

    result = runtime.run_standalone_motion(
        checkpoint_path=tmp_path / "last.ckpt",
        motion_file=tmp_path / "walk.motion",
        video_output=tmp_path / "walk.mp4",
        headless=False,
        num_envs=1,
    )

    assert len(capture_calls) == 2
    assert capture_calls[0].endswith("000000.png")
    assert capture_calls[1].endswith("000001.png")
    assert compile_calls["fps"] == 30
    assert compile_calls["video_path"] == tmp_path / "walk.mp4"
    assert result.success is True
    assert result.num_steps == 2


def test_run_motion_returns_success_result_for_valid_motion(tmp_path):
    from human_motion_isaacsim.runtime import ProtoMotionIsaacSimController
    from human_motion_isaacsim.result import MotionRunResult

    checkpoint = tmp_path / "last.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    motion_file = tmp_path / "walk.motion"
    torch.save({"fps": 30, "rigid_body_pos": torch.zeros((60, 24, 3))}, motion_file)

    controller = ProtoMotionIsaacSimController(
        humanoid_prim_path="/World/Humanoid",
        checkpoint_path=checkpoint,
        lookup_articulation=lambda _prim_path: object(),
        bind_humanoid=lambda prim_path, *, lookup_articulation: object(),
        load_assets=lambda path: object(),
        motion_runner=lambda controller, metadata, video_output: MotionRunResult(
            success=True,
            motion_file=metadata.path,
            video_output=Path(video_output),
            num_steps=60,
            duration_seconds=metadata.duration_seconds,
        ),
    )

    result = controller.run_motion(motion_file, video_output=tmp_path / "walk.mp4")

    assert result.success is True
    assert result.motion_file == motion_file
    assert result.video_output == tmp_path / "walk.mp4"
    assert result.num_steps == 60


def test_run_motion_restores_rest_pose_on_runtime_error(tmp_path):
    from human_motion_isaacsim.runtime import ProtoMotionIsaacSimController

    checkpoint = tmp_path / "last.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    motion_file = tmp_path / "walk.motion"
    torch.save({"fps": 30, "rigid_body_pos": torch.zeros((60, 24, 3))}, motion_file)

    restored = {"called": False}

    def restore(_controller):
        restored["called"] = True

    def fail(_controller, _metadata, _video_output):
        raise RuntimeError("boom")

    controller = ProtoMotionIsaacSimController(
        humanoid_prim_path="/World/Humanoid",
        checkpoint_path=checkpoint,
        lookup_articulation=lambda _prim_path: object(),
        bind_humanoid=lambda prim_path, *, lookup_articulation: object(),
        load_assets=lambda path: object(),
        motion_runner=fail,
        restore_rest_pose=restore,
    )

    with pytest.raises(RuntimeError, match="boom"):
        controller.run_motion(motion_file)

    assert restored["called"] is True


def test_resolve_protomotions_root_prefers_repo_local_submodule(tmp_path, monkeypatch):
    import human_motion_isaacsim.protomotions_path as protomotions_path

    repo_root = tmp_path / "human_motion_isaacsim"
    local_root = repo_root / "third_party" / "ProtoMotions"
    module_file = repo_root / "src" / "human_motion_isaacsim" / "protomotions_path.py"
    (local_root / "protomotions").mkdir(parents=True)
    (local_root / "protomotions" / "__init__.py").write_text("", encoding="utf-8")

    monkeypatch.delenv("PROTOMOTIONS_ROOT", raising=False)
    monkeypatch.delenv("PROTO_MOTIONS_ROOT", raising=False)
    monkeypatch.setattr(protomotions_path, "__file__", str(module_file))
    monkeypatch.setattr(protomotions_path, "find_spec", lambda _name: None)
    monkeypatch.setattr(protomotions_path.Path, "home", lambda: tmp_path / "home")

    assert protomotions_path.resolve_protomotions_root() == local_root.resolve()
