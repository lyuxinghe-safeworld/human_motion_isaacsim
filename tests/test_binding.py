from types import SimpleNamespace


def test_validate_articulation_uses_checkpoint_layout():
    import human_motion_isaacsim.binding as binding

    articulation = SimpleNamespace(
        body_names=("Pelvis", "Spine", "Tail"),
        joint_names=("Pelvis_tx", "Pelvis_ty", "Pelvis_tz"),
    )
    checkpoint_layout = SimpleNamespace(
        robot_config=SimpleNamespace(
            kinematic_info=SimpleNamespace(
                body_names=("Pelvis", "Spine", "Tail"),
                joint_names=("Pelvis_tx", "Pelvis_ty", "Pelvis_tz"),
            )
        )
    )

    bound = binding.validate_articulation(
        articulation,
        tracker_assets=checkpoint_layout,
    )

    assert bound.body_names == checkpoint_layout.robot_config.kinematic_info.body_names
    assert bound.joint_names == checkpoint_layout.robot_config.kinematic_info.joint_names
