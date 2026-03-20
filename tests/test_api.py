import pytest


def test_run_requires_init():
    import human_motion_isaacsim as hmi

    with pytest.raises(RuntimeError, match="init"):
        hmi.run("walk.motion")


def test_list_models_reads_packaged_registry():
    import human_motion_isaacsim as hmi

    models = hmi.list_models()

    assert models[0]["name"] == "smpl"
