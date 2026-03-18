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


_install_omni_stubs()


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
