import pytest
import torch


def test_init_is_exported():
    import human_motion_isaacsim as hmi

    assert callable(hmi.init)


def test_run_requires_init():
    import human_motion_isaacsim as hmi

    with pytest.raises(RuntimeError, match="init"):
        hmi.run("walk.motion")


def test_run_after_init_executes_tracking_loop(monkeypatch, tmp_path):
    import human_motion_isaacsim as hmi
    from human_motion_isaacsim import _api
    from human_motion_isaacsim._state import PACKAGE_STATE

    monkeypatch.setattr(_api, "resolve_tracker_assets", lambda model: object())
    compiled = {}
    teardown_calls = []

    class _FakeAgent:
        def eval(self):
            return None

        def add_agent_info_to_obs(self, obs):
            return obs

        def obs_dict_to_tensordict(self, obs):
            return obs

        def model(self, _obs_td):
            return {"action": torch.zeros((1, 1))}

        def close(self):
            teardown_calls.append("agent")

    class _FakeEnv:
        def reset(self, _done_indices):
            return {}, {}

        def step(self, _actions):
            dones = torch.tensor([False], dtype=torch.bool)
            return {}, torch.zeros(1), dones, None, {}

        def teardown(self):
            teardown_calls.append("env")

    class _FakeSimulator:
        headless = False

        def __init__(self):
            self.frames = []

        def _write_viewport_to_file(self, file_name):
            self.frames.append(file_name)

        def _get_simulator_root_state(self):
            return type(
                "_RootState",
                (),
                {"root_pos": torch.zeros((1, 3), dtype=torch.float32)},
            )()

        def close(self):
            teardown_calls.append("simulator")

    fake_simulator = _FakeSimulator()

    def fake_build_runtime_bundle(_motion_path):
        return {
            "motion_metadata": type("_MotionMetadata", (), {"duration_seconds": 2.0})(),
            "max_steps": 2,
            "env": _FakeEnv(),
            "agent": _FakeAgent(),
            "simulator": fake_simulator,
            "helpers": [fake_simulator, _FakeEnv(), _FakeAgent()],
        }

    monkeypatch.setattr(_api, "_build_runtime_bundle", fake_build_runtime_bundle)
    monkeypatch.setattr(_api, "_align_scene_to_humanoid_root", lambda world, simulator: None)
    monkeypatch.setattr(
        _api,
        "compile_video",
        lambda frame_paths, video_path, fps: compiled.update(
            {
                "frame_paths": [str(path) for path in frame_paths],
                "video_path": str(video_path),
                "fps": fps,
            }
        ),
    )
    PACKAGE_STATE.teardown()

    hmi.init("smpl", world=object(), articulation=object())
    result = hmi.run(tmp_path / "walk.motion", video_output=tmp_path / "walk.mp4")

    assert result.success is True
    assert result.motion_file == tmp_path / "walk.motion"
    assert result.video_output == tmp_path / "walk.mp4"
    assert result.num_steps == 2
    assert PACKAGE_STATE.model_name == "smpl"
    assert teardown_calls == ["simulator", "env", "agent"]
    assert compiled == {}
    PACKAGE_STATE.teardown()


def test_failed_init_preserves_pre_init_run_behavior(monkeypatch):
    import human_motion_isaacsim as hmi
    from human_motion_isaacsim import _api
    from human_motion_isaacsim._state import PACKAGE_STATE

    def fail_resolution(_model):
        raise FileNotFoundError("missing assets")

    PACKAGE_STATE.teardown()
    monkeypatch.setattr(_api, "resolve_tracker_assets", fail_resolution)

    with pytest.raises(FileNotFoundError, match="missing assets"):
        hmi.init("smpl", world=object(), articulation=object())

    with pytest.raises(RuntimeError, match="init"):
        hmi.run("walk.motion")

    PACKAGE_STATE.teardown()


def test_failed_reinit_preserves_last_successful_state(monkeypatch):
    import human_motion_isaacsim as hmi
    from human_motion_isaacsim import _api
    from human_motion_isaacsim._state import PACKAGE_STATE

    original_world = object()
    original_articulation = object()
    original_assets = object()

    def resolve_assets(model):
        if model == "smpl":
            return original_assets
        raise FileNotFoundError("missing replacement assets")

    PACKAGE_STATE.teardown()
    monkeypatch.setattr(_api, "resolve_tracker_assets", resolve_assets)

    hmi.init("smpl", world=original_world, articulation=original_articulation)

    with pytest.raises(FileNotFoundError, match="missing replacement assets"):
        hmi.init("replacement", world=object(), articulation=object())

    assert PACKAGE_STATE.model_name == "smpl"
    assert PACKAGE_STATE.world is original_world
    assert PACKAGE_STATE.articulation is original_articulation
    assert PACKAGE_STATE.tracker_assets is original_assets

    monkeypatch.setattr(
        _api,
        "_build_runtime_bundle",
        lambda _motion_path: {
            "motion_metadata": type("_MotionMetadata", (), {"duration_seconds": 1.0})(),
            "max_steps": 0,
            "env": object(),
            "agent": type("_Agent", (), {"eval": lambda self: None})(),
            "simulator": object(),
            "helpers": [],
        },
    )

    result = hmi.run("walk.motion")

    assert result.success is True
    assert result.num_steps == 0

    PACKAGE_STATE.teardown()


def test_motion_run_result_preserves_positional_constructor_and_video_alias():
    from pathlib import Path

    from human_motion_isaacsim.result import MotionRunResult

    result = MotionRunResult(
        True,
        Path("walk.motion"),
        Path("walk.mp4"),
        60,
        2.0,
        None,
    )

    assert result.video_output == Path("walk.mp4")
    assert result.output_video_path == Path("walk.mp4")
    result.output_video_path = Path("alias.mp4")
    assert result.video_output == Path("alias.mp4")


def test_list_models_returns_smpl_first_model():
    import human_motion_isaacsim as hmi

    models = hmi.list_models()

    assert models[0]["name"] == "smpl"
