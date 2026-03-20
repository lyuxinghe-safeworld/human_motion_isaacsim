from types import SimpleNamespace

import pytest


SMPL_BODY_NAMES = (
    "Pelvis",
    "L_Hip",
    "L_Knee",
    "L_Ankle",
    "L_Toe",
    "R_Hip",
    "R_Knee",
    "R_Ankle",
    "R_Toe",
    "Torso",
    "Spine",
    "Chest",
    "Neck",
    "Head",
    "L_Thorax",
    "L_Shoulder",
    "L_Elbow",
    "L_Wrist",
    "L_Hand",
    "R_Thorax",
    "R_Shoulder",
    "R_Elbow",
    "R_Wrist",
    "R_Hand",
)

SMPL_JOINT_NAMES = (
    "L_Hip_x",
    "L_Hip_y",
    "L_Hip_z",
    "L_Knee_x",
    "L_Knee_y",
    "L_Knee_z",
    "L_Ankle_x",
    "L_Ankle_y",
    "L_Ankle_z",
    "L_Toe_x",
    "L_Toe_y",
    "L_Toe_z",
    "R_Hip_x",
    "R_Hip_y",
    "R_Hip_z",
    "R_Knee_x",
    "R_Knee_y",
    "R_Knee_z",
    "R_Ankle_x",
    "R_Ankle_y",
    "R_Ankle_z",
    "R_Toe_x",
    "R_Toe_y",
    "R_Toe_z",
    "Torso_x",
    "Torso_y",
    "Torso_z",
    "Spine_x",
    "Spine_y",
    "Spine_z",
    "Chest_x",
    "Chest_y",
    "Chest_z",
    "Neck_x",
    "Neck_y",
    "Neck_z",
    "Head_x",
    "Head_y",
    "Head_z",
    "L_Thorax_x",
    "L_Thorax_y",
    "L_Thorax_z",
    "L_Shoulder_x",
    "L_Shoulder_y",
    "L_Shoulder_z",
    "L_Elbow_x",
    "L_Elbow_y",
    "L_Elbow_z",
    "L_Wrist_x",
    "L_Wrist_y",
    "L_Wrist_z",
    "L_Hand_x",
    "L_Hand_y",
    "L_Hand_z",
    "R_Thorax_x",
    "R_Thorax_y",
    "R_Thorax_z",
    "R_Shoulder_x",
    "R_Shoulder_y",
    "R_Shoulder_z",
    "R_Elbow_x",
    "R_Elbow_y",
    "R_Elbow_z",
    "R_Wrist_x",
    "R_Wrist_y",
    "R_Wrist_z",
    "R_Hand_x",
    "R_Hand_y",
    "R_Hand_z",
)


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


def test_validate_articulation_rejects_smpl_articulation_for_non_smpl_metadata():
    import human_motion_isaacsim.binding as binding

    articulation = SimpleNamespace(
        body_names=SMPL_BODY_NAMES,
        joint_names=SMPL_JOINT_NAMES,
    )
    checkpoint_layout = SimpleNamespace(
        robot_config=SimpleNamespace(
            kinematic_info=SimpleNamespace(
                body_names=("Pelvis", "Spine", "Tail"),
                joint_names=("Pelvis_tx", "Pelvis_ty", "Pelvis_tz"),
            )
        )
    )

    with pytest.raises(
        binding.StageBindingError,
        match="selected model|articulation layout|body names",
    ):
        binding.validate_articulation(
            articulation,
            tracker_assets=checkpoint_layout,
        )


def test_bind_fixed_humanoid_uses_tracker_layout_metadata():
    import human_motion_isaacsim.binding as binding

    articulation = SimpleNamespace(
        body_names=("Pelvis", "Spine", "Tail"),
        joint_names=("Pelvis_tx", "Pelvis_ty", "Pelvis_tz"),
    )
    tracker_assets = SimpleNamespace(
        robot_config=SimpleNamespace(
            kinematic_info=SimpleNamespace(
                body_names=("Pelvis", "Spine", "Tail"),
                joint_names=("Pelvis_tx", "Pelvis_ty", "Pelvis_tz"),
            )
        )
    )

    bound = binding.bind_fixed_humanoid(
        "/World/Humanoid",
        lookup_articulation=lambda _prim_path: articulation,
        tracker_assets=tracker_assets,
    )

    assert bound.prim_path == "/World/Humanoid"
    assert bound.body_names == ("Pelvis", "Spine", "Tail")
    assert bound.joint_names == ("Pelvis_tx", "Pelvis_ty", "Pelvis_tz")


def test_validate_articulation_rewraps_tracker_metadata_shape_errors():
    import human_motion_isaacsim.binding as binding

    articulation = SimpleNamespace(
        body_names=("Pelvis", "Spine", "Tail"),
        joint_names=("Pelvis_tx", "Pelvis_ty", "Pelvis_tz"),
    )
    malformed_tracker_assets = SimpleNamespace(robot_config=SimpleNamespace())

    with pytest.raises(
        binding.StageBindingError,
        match="tracker metadata|binding|robot_config\\.kinematic_info",
    ):
        binding.validate_articulation(
            articulation,
            tracker_assets=malformed_tracker_assets,
        )


def test_validate_articulation_rewraps_non_iterable_tracker_metadata():
    import human_motion_isaacsim.binding as binding

    articulation = SimpleNamespace(
        body_names=("Pelvis", "Spine", "Tail"),
        joint_names=("Pelvis_tx", "Pelvis_ty", "Pelvis_tz"),
    )
    malformed_tracker_assets = SimpleNamespace(
        robot_config=SimpleNamespace(
            kinematic_info=SimpleNamespace(
                body_names=123,
                joint_names=("Pelvis_tx", "Pelvis_ty", "Pelvis_tz"),
            )
        )
    )

    with pytest.raises(
        binding.StageBindingError,
        match="tracker metadata|binding|robot_config\\.kinematic_info",
    ):
        binding.validate_articulation(
            articulation,
            tracker_assets=malformed_tracker_assets,
        )
