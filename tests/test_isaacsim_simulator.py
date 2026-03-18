from __future__ import annotations

import pytest
import torch
from types import SimpleNamespace
from unittest.mock import MagicMock


def _make_fake_articulation(body_names=None, dof_names=None):
    from hymotion_isaacsim.binding import EXPECTED_SMPL_BODY_NAMES, EXPECTED_SMPL_JOINT_NAMES
    art = MagicMock()
    art.body_names = list(body_names or EXPECTED_SMPL_BODY_NAMES)
    art.dof_names = list(dof_names or EXPECTED_SMPL_JOINT_NAMES)
    art.num_bodies = len(art.body_names)
    art.num_dof = len(art.dof_names)
    return art


class TestIsaacSimSimulatorConstruction:
    def test_adapter_stores_world_and_articulation(self):
        from hymotion_isaacsim.isaacsim_simulator import IsaacSimSimulator
        art = _make_fake_articulation()
        world = MagicMock()
        sim_app = MagicMock()
        adapter = IsaacSimSimulator.__new__(IsaacSimSimulator)
        adapter._world = world
        adapter._articulation = art
        adapter._simulation_app = sim_app
        assert adapter._world is world
        assert adapter._articulation is art
        assert adapter._simulation_app is sim_app

    def test_get_sim_body_ordering_returns_articulation_names(self):
        from hymotion_isaacsim.isaacsim_simulator import IsaacSimSimulator
        art = _make_fake_articulation(
            body_names=["Pelvis", "L_Hip", "R_Hip"],
            dof_names=["L_Hip_x", "L_Hip_y", "R_Hip_x"],
        )
        adapter = IsaacSimSimulator.__new__(IsaacSimSimulator)
        adapter._articulation = art
        ordering = adapter._get_sim_body_ordering()
        assert ordering.body_names == ["Pelvis", "L_Hip", "R_Hip"]
        assert ordering.dof_names == ["L_Hip_x", "L_Hip_y", "R_Hip_x"]
