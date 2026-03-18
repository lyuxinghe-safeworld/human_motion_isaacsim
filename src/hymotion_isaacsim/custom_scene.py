from __future__ import annotations

from typing import Any

SCENE_OBJECTS = [
    {
        "type": "cube",
        "prim_path": "/World/custom_scene/box",
        "size": 1.0,
        "position": (2.0, 1.0, 0.5),
        "color": (0.8, 0.2, 0.2),
        "fixed_base": True,
    },
    {
        "type": "cylinder",
        "prim_path": "/World/custom_scene/cylinder",
        "radius": 0.3,
        "height": 1.5,
        "position": (-1.0, 2.0, 0.75),
        "color": (0.2, 0.8, 0.2),
        "fixed_base": True,
    },
    {
        "type": "sphere",
        "prim_path": "/World/custom_scene/sphere",
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
