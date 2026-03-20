from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


def _install_omni_stubs():
    """Install minimal omni.isaac.core stubs so populate_scene can be imported
    and called without a running Isaac Sim environment."""

    class _FakePrim:
        def __init__(self, **kwargs):
            self._kwargs = kwargs

    # Build the omni.isaac.core.objects stub hierarchy.
    omni_mod = types.ModuleType("omni")
    isaac_mod = types.ModuleType("omni.isaac")
    core_mod = types.ModuleType("omni.isaac.core")
    objects_mod = types.ModuleType("omni.isaac.core.objects")

    objects_mod.FixedCuboid = _FakePrim
    objects_mod.FixedCylinder = _FakePrim
    objects_mod.FixedSphere = _FakePrim

    omni_mod.isaac = isaac_mod
    isaac_mod.core = core_mod
    core_mod.objects = objects_mod

    sys.modules.setdefault("omni", omni_mod)
    sys.modules.setdefault("omni.isaac", isaac_mod)
    sys.modules.setdefault("omni.isaac.core", core_mod)
    sys.modules.setdefault("omni.isaac.core.objects", objects_mod)


def _install_pxr_stubs():
    """Install minimal pxr stubs for scene repositioning helpers."""

    pxr_mod = types.ModuleType("pxr")
    gf_mod = types.ModuleType("pxr.Gf")

    class _FakeVec3d(tuple):
        def __new__(cls, x, y, z):
            return super().__new__(cls, (x, y, z))

    gf_mod.Vec3d = _FakeVec3d

    pxr_mod.Gf = gf_mod

    sys.modules.setdefault("pxr", pxr_mod)
    sys.modules.setdefault("pxr.Gf", gf_mod)


_install_omni_stubs()
_install_pxr_stubs()


def _load_run_scene_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "run_scene.py"
    spec = importlib.util.spec_from_file_location("test_run_scene_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_scene_utils_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "scene_utils.py"
    spec = importlib.util.spec_from_file_location("test_scene_utils_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _parse_run_scene_args(monkeypatch, *args):
    run_scene = _load_run_scene_module()
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_scene.py", *args],
    )
    return run_scene.parse_args()


def test_populate_scene_adds_three_objects():
    """populate_scene should add box, cylinder, sphere to the world."""
    scene_utils = _load_scene_utils_module()

    assert len(scene_utils.SCENE_OBJECTS) == 3

    added = []

    class _FakeWorld:
        class scene:
            @staticmethod
            def add(obj):
                added.append(obj)

    scene_utils.populate_scene(_FakeWorld())
    assert len(added) == 3


def test_scene_objects_have_expected_prim_paths():
    scene_utils = _load_scene_utils_module()

    paths = [obj["prim_path"] for obj in scene_utils.SCENE_OBJECTS]
    assert all(p.startswith("/World/custom_scene/") for p in paths)


def test_scene_objects_are_static():
    """All scene objects should have fixed_base=True (static)."""
    scene_utils = _load_scene_utils_module()

    for obj in scene_utils.SCENE_OBJECTS:
        assert obj.get("fixed_base", False) is True, f"{obj['prim_path']} is not static"


def test_set_scene_origin_offsets_all_object_translations():
    scene_utils = _load_scene_utils_module()

    class _FakePrim:
        def __init__(self):
            self.translate = None

        def IsValid(self):
            return True

        def GetAttribute(self, attr_name):
            assert attr_name == "xformOp:translate"

            class _FakeAttr:
                def __init__(self, prim):
                    self._prim = prim

                def IsValid(self):
                    return True

                def Set(self, value):
                    self._prim.translate = tuple(value)

            return _FakeAttr(self)

    class _FakeStage:
        def __init__(self):
            self._prims = {obj["prim_path"]: _FakePrim() for obj in scene_utils.SCENE_OBJECTS}
            self._prims[scene_utils.GROUND_PLANE_PRIM_PATH] = _FakePrim()

        def GetPrimAtPath(self, prim_path):
            return self._prims[prim_path]

    stage = _FakeStage()
    world = types.SimpleNamespace(stage=stage)

    scene_utils.set_scene_origin(world, (10.0, 20.0, 0.0))

    ground_plane = stage.GetPrimAtPath(scene_utils.GROUND_PLANE_PRIM_PATH)
    assert ground_plane.translate == pytest.approx((10.0, 20.0, 0.0))

    for obj in scene_utils.SCENE_OBJECTS:
        prim = stage.GetPrimAtPath(obj["prim_path"])
        expected = tuple(a + b for a, b in zip(obj["position"], (10.0, 20.0, 0.0)))
        assert prim.translate == pytest.approx(expected)


def test_run_scene_defaults_model_to_smpl(monkeypatch):
    args = _parse_run_scene_args(
        monkeypatch,
        "--motion-file",
        "walk.motion",
    )

    assert args.model == "smpl"


def test_run_scene_rejects_checkpoint_argument(monkeypatch):
    with pytest.raises(SystemExit):
        _parse_run_scene_args(
            monkeypatch,
            "--motion-file",
            "walk.motion",
            "--checkpoint",
            "last.ckpt",
        )
