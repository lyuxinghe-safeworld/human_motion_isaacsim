from types import SimpleNamespace

import torch


def _write_resolved_configs(directory, *, asset_root):
    torch.save(
        {
            "robot": SimpleNamespace(asset=SimpleNamespace(asset_root=asset_root)),
            "simulator": SimpleNamespace(_target_="sim.target"),
            "terrain": SimpleNamespace(_target_="terrain.target"),
            "scene_lib": SimpleNamespace(_target_="scene.target"),
            "motion_lib": SimpleNamespace(_target_="motion.target"),
            "env": SimpleNamespace(_target_="env.target"),
            "agent": SimpleNamespace(_target_="agent.target"),
        },
        directory / "resolved_configs_inference.pt",
    )


def test_list_models_reads_packaged_registry():
    import human_motion_isaacsim._registry as _registry

    models = _registry.list_models()

    assert models == [
        {
            "name": "smpl",
            "description": "SMPL body model",
        }
    ]


def test_resolve_tracker_assets_prefers_repo_local_assets(monkeypatch, tmp_path):
    repo_root = tmp_path / "checkout"
    local_checkpoint = (
        repo_root
        / "third_party"
        / "ProtoMotions"
        / "data"
        / "pretrained_models"
        / "motion_tracker"
        / "smpl"
        / "last.ckpt"
    )
    local_checkpoint.parent.mkdir(parents=True)
    local_checkpoint.write_bytes(b"local checkpoint")
    _write_resolved_configs(local_checkpoint.parent, asset_root="protomotions/data/assets")

    cache_checkpoint = (
        tmp_path
        / "cache"
        / "human_motion_isaacsim"
        / "smpl"
        / "last.ckpt"
    )
    cache_checkpoint.parent.mkdir(parents=True)
    cache_checkpoint.write_bytes(b"cached checkpoint")
    _write_resolved_configs(cache_checkpoint.parent, asset_root="/tmp/cached-assets")

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    import human_motion_isaacsim._registry as _registry

    assets = _registry.resolve_tracker_assets("smpl", repo_root=repo_root)

    assert assets.checkpoint_path == local_checkpoint
    assert assets.resolved_config_path == local_checkpoint.parent / "resolved_configs_inference.pt"


def test_package_state_teardown_clears_owned_references_without_closing_simulation_app():
    from human_motion_isaacsim._state import PackageState

    helper = SimpleNamespace(teardown_calls=0)
    world = object()
    articulation = object()
    simulation_app = SimpleNamespace(close=lambda: (_ for _ in ()).throw(AssertionError("should not close")))

    def helper_teardown():
        helper.teardown_calls += 1

    state = PackageState(
        model_name="smpl",
        tracker_assets=object(),
        world=world,
        articulation=articulation,
        simulation_app=simulation_app,
        owned_helpers=[SimpleNamespace(teardown=helper_teardown)],
    )

    state.teardown()

    assert helper.teardown_calls == 1
    assert state.model_name is None
    assert state.tracker_assets is None
    assert state.world is None
    assert state.articulation is None
    assert state.simulation_app is simulation_app
    assert state.owned_helpers == []
