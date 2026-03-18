import torch


def test_restore_rest_pose_preserves_root_translation():
    from hymotion_isaacsim.rest_pose import capture_rest_pose_defaults, compose_rest_pose_state

    defaults = capture_rest_pose_defaults(
        root_orientation=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        joint_positions=torch.tensor([0.1, 0.2, 0.3]),
    )

    state = compose_rest_pose_state(
        defaults,
        target_root_position=torch.tensor([3.0, 4.0, 5.0]),
    )

    assert torch.equal(state.root_position, torch.tensor([3.0, 4.0, 5.0]))
    assert torch.equal(state.joint_positions, torch.tensor([0.1, 0.2, 0.3]))


def test_restore_rest_pose_zeros_joint_and_root_velocities():
    from hymotion_isaacsim.rest_pose import capture_rest_pose_defaults, compose_rest_pose_state

    defaults = capture_rest_pose_defaults(
        root_orientation=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        joint_positions=torch.tensor([0.1, 0.2, 0.3]),
    )

    state = compose_rest_pose_state(
        defaults,
        target_root_position=torch.tensor([3.0, 4.0, 5.0]),
    )

    assert torch.equal(state.root_linear_velocity, torch.zeros(3))
    assert torch.equal(state.root_angular_velocity, torch.zeros(3))
    assert torch.equal(state.joint_velocities, torch.zeros(3))
