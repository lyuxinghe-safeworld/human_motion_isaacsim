from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class RestPoseDefaults:
    root_orientation: torch.Tensor
    joint_positions: torch.Tensor


@dataclass(slots=True)
class RestPoseState:
    root_position: torch.Tensor
    root_orientation: torch.Tensor
    joint_positions: torch.Tensor
    root_linear_velocity: torch.Tensor
    root_angular_velocity: torch.Tensor
    joint_velocities: torch.Tensor


def capture_rest_pose_defaults(
    *,
    root_orientation: torch.Tensor,
    joint_positions: torch.Tensor,
) -> RestPoseDefaults:
    return RestPoseDefaults(
        root_orientation=root_orientation.clone(),
        joint_positions=joint_positions.clone(),
    )


def compose_rest_pose_state(
    defaults: RestPoseDefaults,
    *,
    target_root_position: torch.Tensor,
) -> RestPoseState:
    zeros3 = torch.zeros(3, dtype=target_root_position.dtype, device=target_root_position.device)
    joint_velocities = torch.zeros_like(defaults.joint_positions)
    return RestPoseState(
        root_position=target_root_position.clone(),
        root_orientation=defaults.root_orientation.clone(),
        joint_positions=defaults.joint_positions.clone(),
        root_linear_velocity=zeros3,
        root_angular_velocity=zeros3.clone(),
        joint_velocities=joint_velocities,
    )
