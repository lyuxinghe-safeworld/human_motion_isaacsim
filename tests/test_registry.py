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
    (local_checkpoint.parent / "resolved_configs_inference.pt").write_bytes(b"local configs")

    cache_checkpoint = (
        tmp_path
        / "cache"
        / "human_motion_isaacsim"
        / "smpl"
        / "last.ckpt"
    )
    cache_checkpoint.parent.mkdir(parents=True)
    cache_checkpoint.write_bytes(b"cached checkpoint")
    (cache_checkpoint.parent / "resolved_configs_inference.pt").write_bytes(b"cached configs")

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    import human_motion_isaacsim._registry as _registry

    assets = _registry.resolve_tracker_assets("smpl", repo_root=repo_root)

    assert assets.checkpoint_path == local_checkpoint
    assert assets.resolved_config_path == local_checkpoint.parent / "resolved_configs_inference.pt"
