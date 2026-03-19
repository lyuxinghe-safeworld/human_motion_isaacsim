from __future__ import annotations

import sys
import types

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


def test_populate_scene_adds_three_objects():
    """populate_scene should add box, cylinder, sphere to the world."""
    from hymotion_isaacsim.custom_scene import SCENE_OBJECTS, populate_scene

    assert len(SCENE_OBJECTS) == 3

    added = []

    class _FakeWorld:
        class scene:
            @staticmethod
            def add(obj):
                added.append(obj)

    populate_scene(_FakeWorld())
    assert len(added) == 3


def test_scene_objects_have_expected_prim_paths():
    from hymotion_isaacsim.custom_scene import SCENE_OBJECTS

    paths = [obj["prim_path"] for obj in SCENE_OBJECTS]
    assert all(p.startswith("/World/custom_scene/") for p in paths)


def test_scene_objects_are_static():
    """All scene objects should have fixed_base=True (static)."""
    from hymotion_isaacsim.custom_scene import SCENE_OBJECTS

    for obj in SCENE_OBJECTS:
        assert obj.get("fixed_base", False) is True, f"{obj['prim_path']} is not static"


def test_set_scene_origin_offsets_all_object_translations():
    from hymotion_isaacsim.custom_scene import (
        GROUND_PLANE_PRIM_PATH,
        SCENE_OBJECTS,
        set_scene_origin,
    )

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
            self._prims = {obj["prim_path"]: _FakePrim() for obj in SCENE_OBJECTS}
            self._prims[GROUND_PLANE_PRIM_PATH] = _FakePrim()

        def GetPrimAtPath(self, prim_path):
            return self._prims[prim_path]

    stage = _FakeStage()
    world = types.SimpleNamespace(stage=stage)

    set_scene_origin(world, (10.0, 20.0, 0.0))

    ground_plane = stage.GetPrimAtPath(GROUND_PLANE_PRIM_PATH)
    assert ground_plane.translate == pytest.approx((10.0, 20.0, 0.0))

    for obj in SCENE_OBJECTS:
        prim = stage.GetPrimAtPath(obj["prim_path"])
        expected = tuple(a + b for a, b in zip(obj["position"], (10.0, 20.0, 0.0)))
        assert prim.translate == pytest.approx(expected)
