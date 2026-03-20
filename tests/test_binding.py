from types import SimpleNamespace

import pytest


def test_validate_articulation_uses_checkpoint_layout():
    import human_motion_isaacsim.binding as binding

    articulation = SimpleNamespace(
        body_names=("Unexpected_Body",),
        joint_names=("Unexpected_Joint",),
    )
    checkpoint_layout = SimpleNamespace(
        robot_config=SimpleNamespace(
            kinematic_info=SimpleNamespace(
                body_names=("Pelvis", "Spine", "Chest"),
                joint_names=("Pelvis_x", "Pelvis_y", "Pelvis_z"),
            )
        )
    )

    with pytest.raises(binding.StageBindingError, match="layout|body names"):
        binding.validate_articulation(
            articulation,
            tracker_assets=checkpoint_layout,
        )
