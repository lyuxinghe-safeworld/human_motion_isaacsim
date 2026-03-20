from types import SimpleNamespace

import importlib
import sys
from pathlib import Path

import pytest
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
    protomotions_root = repo_root / "third_party" / "ProtoMotions"
    local_checkpoint = (
        protomotions_root
        / "data"
        / "pretrained_models"
        / "motion_tracker"
        / "smpl"
        / "last.ckpt"
    )
    local_checkpoint.parent.mkdir(parents=True)
    local_checkpoint.write_bytes(b"local checkpoint")
    _write_resolved_configs(local_checkpoint.parent, asset_root="protomotions/data/assets")
    (protomotions_root / "protomotions").mkdir(parents=True)
    (protomotions_root / "protomotions" / "__init__.py").write_text("", encoding="utf-8")

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
    assert assets.robot_config.asset.asset_root == str((protomotions_root / "protomotions" / "data" / "assets").resolve())


def test_load_tracker_assets_explicit_protomotions_override_replaces_stale_imports(monkeypatch, tmp_path):
    import human_motion_isaacsim.checkpoint as checkpoint_module

    stale_root = tmp_path / "stale_proto"
    override_root = tmp_path / "override_proto"
    for root, label in ((stale_root, "stale"), (override_root, "override")):
        package_dir = root / "protomotions"
        package_dir.mkdir(parents=True)
        (package_dir / "__init__.py").write_text("", encoding="utf-8")
        (package_dir / "marker.py").write_text(f'ROOT = "{label}"\n', encoding="utf-8")

    checkpoint = tmp_path / "last.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    resolved = checkpoint.parent / "resolved_configs_inference.pt"
    resolved.write_bytes(b"placeholder")

    monkeypatch.syspath_prepend(str(override_root))
    monkeypatch.syspath_prepend(str(stale_root))
    importlib.invalidate_caches()
    sys.modules.pop("protomotions", None)
    sys.modules.pop("protomotions.marker", None)
    importlib.import_module("protomotions.marker")

    assert sys.path.index(str(stale_root)) < sys.path.index(str(override_root))

    def fake_load(*args, **kwargs):
        marker = importlib.import_module("protomotions.marker")
        return {
            "robot": SimpleNamespace(asset=SimpleNamespace(asset_root="protomotions/data/assets")),
            "simulator": SimpleNamespace(_target_="sim.target"),
            "terrain": SimpleNamespace(_target_="terrain.target"),
            "scene_lib": SimpleNamespace(_target_="scene.target"),
            "motion_lib": SimpleNamespace(_target_="motion.target"),
            "env": SimpleNamespace(_target_=marker.ROOT),
            "agent": SimpleNamespace(_target_="agent.target"),
        }

    monkeypatch.setattr(checkpoint_module.torch, "load", fake_load)

    assets = checkpoint_module.load_tracker_assets(checkpoint, protomotions_root=override_root)

    assert assets.env_config._target_ == "override"
    assert assets.robot_config.asset.asset_root == str((override_root / "protomotions" / "data" / "assets").resolve())
    assert "override_proto" in Path(sys.modules["protomotions"].__file__).as_posix()


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


def test_package_state_teardown_clears_state_when_helper_teardown_raises():
    from human_motion_isaacsim._state import PackageState

    simulation_app = SimpleNamespace(close=lambda: (_ for _ in ()).throw(AssertionError("should not close")))
    call_order = []

    def teardown_first():
        call_order.append("first")

    def teardown_boom():
        call_order.append("boom")
        raise RuntimeError("boom")

    def teardown_last():
        call_order.append("last")

    state = PackageState(
        model_name="smpl",
        tracker_assets=object(),
        world=object(),
        articulation=object(),
        simulation_app=simulation_app,
        owned_helpers=[
            SimpleNamespace(teardown=teardown_first),
            SimpleNamespace(teardown=teardown_boom),
            SimpleNamespace(teardown=teardown_last),
        ],
    )

    with pytest.raises(RuntimeError, match="boom"):
        state.teardown()

    assert call_order == ["first", "boom", "last"]
    assert state.model_name is None
    assert state.tracker_assets is None
    assert state.world is None
    assert state.articulation is None
    assert state.simulation_app is simulation_app
    assert state.owned_helpers == []
