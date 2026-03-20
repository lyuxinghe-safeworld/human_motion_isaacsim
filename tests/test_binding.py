from types import SimpleNamespace


def test_validate_articulation_uses_checkpoint_layout(monkeypatch):
    import human_motion_isaacsim.binding as binding

    checkpoint_layout = SimpleNamespace(
        body_names=("Pelvis", "Spine", "Chest"),
        joint_names=("Pelvis_x", "Pelvis_y", "Pelvis_z"),
    )

    monkeypatch.setattr(binding, "EXPECTED_SMPL_BODY_NAMES", ("not", "used"))
    monkeypatch.setattr(binding, "EXPECTED_SMPL_JOINT_NAMES", ("not", "used"))

    body_names, joint_names = binding.validate_humanoid_layout(
        checkpoint_layout.body_names,
        checkpoint_layout.joint_names,
    )

    assert body_names == checkpoint_layout.body_names
    assert joint_names == checkpoint_layout.joint_names
