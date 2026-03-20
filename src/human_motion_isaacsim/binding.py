from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Any


# TODO(task 5): replace these legacy SMPL constants with layout metadata
# from the selected tracker assets.
EXPECTED_SMPL_BODY_NAMES = (
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

EXPECTED_SMPL_JOINT_NAMES = (
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


class StageBindingError(ValueError):
    """Raised when the controller cannot bind to the expected humanoid."""


@dataclass(slots=True)
class BoundHumanoid:
    prim_path: str
    articulation: Any
    body_names: tuple[str, ...]
    joint_names: tuple[str, ...]


def _as_tuple(names: Iterable[str]) -> tuple[str, ...]:
    return tuple(names)


def validate_humanoid_layout(
    body_names: Iterable[str],
    joint_names: Iterable[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    body_names_tuple = _as_tuple(body_names)
    joint_names_tuple = _as_tuple(joint_names)

    if body_names_tuple != EXPECTED_SMPL_BODY_NAMES:
        raise StageBindingError(
            "Bound humanoid body names do not match the expected ProtoMotions SMPL layout."
        )

    if joint_names_tuple != EXPECTED_SMPL_JOINT_NAMES:
        raise StageBindingError(
            "Bound humanoid joint names do not match the expected ProtoMotions SMPL layout."
        )

    return body_names_tuple, joint_names_tuple


def bind_fixed_humanoid(
    prim_path: str,
    *,
    lookup_articulation: Callable[[str], Any],
) -> BoundHumanoid:
    # NOTE(v2): This path is intentionally fixed for the first controller version.
    articulation = lookup_articulation(prim_path)
    if articulation is None:
        raise StageBindingError(f"Humanoid prim not found at {prim_path}")

    body_names, joint_names = validate_humanoid_layout(
        articulation.body_names,
        articulation.joint_names,
    )
    return BoundHumanoid(
        prim_path=prim_path,
        articulation=articulation,
        body_names=body_names,
        joint_names=joint_names,
    )
