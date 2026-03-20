from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Any

from human_motion_isaacsim.checkpoint import tracker_kinematic_layout


class StageBindingError(ValueError):
    """Raised when the controller cannot bind to the expected humanoid articulation."""


@dataclass(slots=True)
class BoundHumanoid:
    """A validated humanoid articulation with confirmed body and joint name layout."""

    prim_path: str
    articulation: Any
    body_names: tuple[str, ...]
    joint_names: tuple[str, ...]


def _as_tuple(names: Iterable[str]) -> tuple[str, ...]:
    """Coerce an iterable of strings into a tuple."""
    return tuple(names)


def _tracker_layout_for_binding(tracker_assets: Any) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Extract the expected kinematic layout from tracker assets, raising StageBindingError on failure."""
    try:
        return tracker_kinematic_layout(tracker_assets)
    except (TypeError, ValueError) as exc:
        raise StageBindingError(
            "Cannot validate the supplied articulation for binding because the selected model "
            "has missing or malformed tracker metadata at robot_config.kinematic_info body/joint names."
        ) from exc


def validate_humanoid_layout(
    body_names: Iterable[str],
    joint_names: Iterable[str],
    *,
    expected_body_names: Iterable[str],
    expected_joint_names: Iterable[str],
    model_label: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Compare articulation body/joint names against expected names, raising on mismatch."""
    body_names_tuple = _as_tuple(body_names)
    joint_names_tuple = _as_tuple(joint_names)
    expected_body_names_tuple = _as_tuple(expected_body_names)
    expected_joint_names_tuple = _as_tuple(expected_joint_names)

    if body_names_tuple != expected_body_names_tuple:
        raise StageBindingError(
            f"The selected model does not match the supplied articulation layout: "
            f"articulation body names do not match {model_label}."
        )

    if joint_names_tuple != expected_joint_names_tuple:
        raise StageBindingError(
            f"The selected model does not match the supplied articulation layout: "
            f"articulation joint names do not match {model_label}."
        )

    return body_names_tuple, joint_names_tuple


def validate_articulation(
    articulation: Any,
    *,
    tracker_assets: Any,
) -> BoundHumanoid:
    """Validate that an articulation matches the tracker's expected kinematic layout."""
    expected_body_names, expected_joint_names = _tracker_layout_for_binding(tracker_assets)
    checkpoint_path = getattr(tracker_assets, "checkpoint_path", None)
    model_label = (
        f"tracker asset metadata from {checkpoint_path}"
        if checkpoint_path is not None
        else "the selected model's tracker asset metadata"
    )
    body_names, joint_names = validate_humanoid_layout(
        articulation.body_names,
        articulation.joint_names,
        expected_body_names=expected_body_names,
        expected_joint_names=expected_joint_names,
        model_label=model_label,
    )
    return BoundHumanoid(
        prim_path="",
        articulation=articulation,
        body_names=body_names,
        joint_names=joint_names,
    )


def bind_fixed_humanoid(
    prim_path: str,
    *,
    lookup_articulation: Callable[[str], Any],
    tracker_assets: Any,
) -> BoundHumanoid:
    """Look up an articulation by prim path and validate it against tracker asset metadata."""
    # NOTE(v2): This path is intentionally fixed for the first controller version.
    articulation = lookup_articulation(prim_path)
    if articulation is None:
        raise StageBindingError(f"Humanoid prim not found at {prim_path}")

    bound = validate_articulation(
        articulation,
        tracker_assets=tracker_assets,
    )
    return BoundHumanoid(
        prim_path=prim_path,
        articulation=articulation,
        body_names=bound.body_names,
        joint_names=bound.joint_names,
    )
