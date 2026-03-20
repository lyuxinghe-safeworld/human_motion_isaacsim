import pytest


def test_init_is_exported():
    import human_motion_isaacsim as hmi

    assert callable(hmi.init)


def test_run_requires_init():
    import human_motion_isaacsim as hmi

    with pytest.raises(RuntimeError, match="init"):
        hmi.run("walk.motion")


def test_run_after_init_reports_tracking_loop_not_implemented(monkeypatch):
    import human_motion_isaacsim as hmi
    from human_motion_isaacsim import _api
    from human_motion_isaacsim._state import PACKAGE_STATE

    monkeypatch.setattr(_api, "resolve_tracker_assets", lambda model: object())
    PACKAGE_STATE.teardown()

    hmi.init("smpl", world=object(), articulation=object())

    with pytest.raises(RuntimeError, match="tracking loop.*not implemented"):
        hmi.run("walk.motion")

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

    with pytest.raises(RuntimeError, match="tracking loop.*not implemented"):
        hmi.run("walk.motion")

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
