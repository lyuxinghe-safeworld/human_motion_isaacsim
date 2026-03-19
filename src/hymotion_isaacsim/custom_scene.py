from __future__ import annotations

from typing import Any

SCENE_ROOT_PRIM_PATH = "/World/custom_scene"
GROUND_PLANE_PRIM_PATH = "/World/GroundPlane"

SCENE_OBJECTS = [
    {
        "type": "cube",
        "prim_path": f"{SCENE_ROOT_PRIM_PATH}/box",
        "size": 1.0,
        "position": (2.0, 1.0, 0.5),
        "color": (0.8, 0.2, 0.2),
        "fixed_base": True,
    },
    {
        "type": "cylinder",
        "prim_path": f"{SCENE_ROOT_PRIM_PATH}/cylinder",
        "radius": 0.3,
        "height": 1.5,
        "position": (-1.0, 2.0, 0.75),
        "color": (0.2, 0.8, 0.2),
        "fixed_base": True,
    },
    {
        "type": "sphere",
        "prim_path": f"{SCENE_ROOT_PRIM_PATH}/sphere",
        "radius": 0.5,
        "position": (1.0, -1.5, 0.5),
        "color": (0.2, 0.2, 0.8),
        "fixed_base": True,
    },
]


def populate_scene(world: Any) -> None:
    """Add hardcoded static objects to the Isaac Sim world.

    Uses omni.isaac.core object prims. Each object is a rigid body with a
    collider and fixed_base=True (static — won't fall or move).
    """
    import numpy as np
    from omni.isaac.core.objects import FixedCuboid, FixedCylinder, FixedSphere

    _BUILDERS = {
        "cube": lambda obj: FixedCuboid(
            prim_path=obj["prim_path"],
            size=obj["size"],
            position=np.array(obj["position"]),
            color=np.array(obj["color"]),
        ),
        "cylinder": lambda obj: FixedCylinder(
            prim_path=obj["prim_path"],
            radius=obj["radius"],
            height=obj["height"],
            position=np.array(obj["position"]),
            color=np.array(obj["color"]),
        ),
        "sphere": lambda obj: FixedSphere(
            prim_path=obj["prim_path"],
            radius=obj["radius"],
            position=np.array(obj["position"]),
            color=np.array(obj["color"]),
        ),
    }

    for obj in SCENE_OBJECTS:
        prim = _BUILDERS[obj["type"]](obj)
        world.scene.add(prim)


def set_scene_origin(world: Any, origin: tuple[float, float, float]) -> None:
    """Reposition the authored scene so its local origin follows the humanoid spawn."""
    import numpy as np
    from pxr import Gf

    stage = world.stage
    scene_origin = np.asarray(origin, dtype=np.float64)
    if scene_origin.shape != (3,):
        raise ValueError(f"Expected a 3D scene origin, got shape {scene_origin.shape}")

    ground_plane = stage.GetPrimAtPath(GROUND_PLANE_PRIM_PATH)
    if not ground_plane.IsValid():
        raise RuntimeError(f"Ground plane prim does not exist: {GROUND_PLANE_PRIM_PATH}")
    ground_plane_translate_attr = ground_plane.GetAttribute("xformOp:translate")
    if not ground_plane_translate_attr.IsValid():
        raise RuntimeError(
            f"Ground plane is missing xformOp:translate: {GROUND_PLANE_PRIM_PATH}"
        )
    ground_plane_translate_attr.Set(Gf.Vec3d(*scene_origin.tolist()))

    for obj in SCENE_OBJECTS:
        prim = stage.GetPrimAtPath(obj["prim_path"])
        if not prim.IsValid():
            raise RuntimeError(f"Scene prim does not exist: {obj['prim_path']}")
        position = np.asarray(obj["position"], dtype=np.float64) + scene_origin
        translate_attr = prim.GetAttribute("xformOp:translate")
        if not translate_attr.IsValid():
            raise RuntimeError(f"Scene prim is missing xformOp:translate: {obj['prim_path']}")
        translate_attr.Set(Gf.Vec3d(*position.tolist()))
