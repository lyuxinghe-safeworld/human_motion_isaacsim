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


def test_list_models_returns_smpl_first_model():
    import human_motion_isaacsim as hmi

    models = hmi.list_models()

    assert models[0]["name"] == "smpl"
