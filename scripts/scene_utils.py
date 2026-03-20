from __future__ import annotations

from pathlib import Path
from typing import Any

SCENE_ROOT_PRIM_PATH = "/World/custom_scene"
GROUND_PLANE_PRIM_PATH = "/World/GroundPlane"
REPO_ROOT = Path(__file__).resolve().parents[1]

SCENE_MODELS = {
    "smpl": {
        "asset_root": REPO_ROOT
        / "third_party"
        / "ProtoMotions"
        / "protomotions"
        / "data"
        / "assets",
        "usd_file": "usd/smpl_humanoid.usda",
        "physics_fps": 120,
    }
}

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
    """Add hardcoded static objects to the Isaac Sim world."""
    import numpy as np
    from omni.isaac.core.objects import FixedCuboid, FixedCylinder, FixedSphere

    builders = {
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
        world.scene.add(builders[obj["type"]](obj))


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


def align_scene_to_humanoid_root(world: Any, simulator: Any) -> None:
    """Keep the authored local scene centered on the humanoid spawn/reset point."""
    root_pos = simulator._get_simulator_root_state().root_pos[0].detach().cpu().numpy()
    set_scene_origin(world, (float(root_pos[0]), float(root_pos[1]), 0.0))


def _resolve_scene_model_config(model: str) -> dict[str, Any]:
    try:
        return SCENE_MODELS[model]
    except KeyError as exc:
        raise KeyError(f"Scene assets are not configured for model: {model}") from exc


def build_scene(model: str, headless: bool):
    """Create the local Isaac Sim scene used by the wrapper script."""
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": headless})
    scene_model = _resolve_scene_model_config(model)

    from omni.isaac.core import World
    from omni.isaac.core.articulations import Articulation
    from omni.isaac.core.objects import GroundPlane
    from omni.isaac.core.utils.stage import add_reference_to_stage

    fps = int(scene_model.get("physics_fps", 60))
    world = World(physics_dt=1.0 / fps, rendering_dt=1.0 / fps)
    world.scene.add(GroundPlane(prim_path=GROUND_PLANE_PRIM_PATH, size=100.0))

    asset_root = Path(scene_model["asset_root"])
    usd_file = str(scene_model["usd_file"])
    humanoid_usd_path = str(Path(asset_root) / usd_file)
    add_reference_to_stage(humanoid_usd_path, "/World/Humanoid")
    articulation = world.scene.add(
        Articulation(prim_path="/World/Humanoid", name="humanoid")
    )

    populate_scene(world)
    return simulation_app, world, articulation
