from __future__ import annotations

import pytest
import torch
from types import SimpleNamespace
from unittest.mock import MagicMock

from protomotions.simulator.base_simulator.simulator_state import (
    ResetState, StateConversion,
)


def _make_fake_articulation(body_names=None, dof_names=None):
    from hymotion_isaacsim.binding import EXPECTED_SMPL_BODY_NAMES, EXPECTED_SMPL_JOINT_NAMES
    art = MagicMock()
    body = list(body_names or EXPECTED_SMPL_BODY_NAMES)
    dof = list(dof_names or EXPECTED_SMPL_JOINT_NAMES)
    art.body_names = body
    art.dof_names = dof
    art.num_bodies = len(body)
    art.num_dof = len(dof)
    # The adapter accesses art._articulation_view for body_names and batched APIs
    art._articulation_view.body_names = body
    art._articulation_view.dof_names = dof
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


class TestStateGetters:
    def test_get_simulator_root_state_returns_root_only_state(self):
        from hymotion_isaacsim.isaacsim_simulator import IsaacSimSimulator
        art = _make_fake_articulation()
        # SingleArticulation.get_world_pose() returns unbatched (3,) and (4,)
        art.get_world_pose.return_value = (
            torch.tensor([1.0, 2.0, 3.0]),
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
        )
        art.get_world_velocity.return_value = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        adapter = IsaacSimSimulator.__new__(IsaacSimSimulator)
        adapter._articulation = art
        adapter.device = "cpu"
        adapter.num_envs = 1
        state = adapter._get_simulator_root_state()
        assert state.root_pos.shape == (1, 3)
        assert state.root_rot.shape == (1, 4)
        assert state.root_vel.shape == (1, 3)
        assert state.root_ang_vel.shape == (1, 3)
        assert state.state_conversion == StateConversion.SIMULATOR

    def test_get_simulator_root_state_with_env_ids(self):
        from hymotion_isaacsim.isaacsim_simulator import IsaacSimSimulator
        art = _make_fake_articulation()
        # Return batched (already has batch dim)
        art.get_world_pose.return_value = (
            torch.tensor([1.0, 2.0, 3.0]),
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
        )
        art.get_world_velocity.return_value = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        adapter = IsaacSimSimulator.__new__(IsaacSimSimulator)
        adapter._articulation = art
        adapter.device = "cpu"
        adapter.num_envs = 1
        state = adapter._get_simulator_root_state(env_ids=torch.tensor([0]))
        assert state.root_pos.shape == (1, 3)

    def test_get_simulator_dof_state_returns_robot_state(self):
        from hymotion_isaacsim.isaacsim_simulator import IsaacSimSimulator
        num_dof = 69
        art = _make_fake_articulation()
        art.get_joint_positions.return_value = torch.zeros((1, num_dof))
        art.get_joint_velocities.return_value = torch.zeros((1, num_dof))
        adapter = IsaacSimSimulator.__new__(IsaacSimSimulator)
        adapter._articulation = art
        adapter.device = "cpu"
        adapter.num_envs = 1
        state = adapter._get_simulator_dof_state()
        assert state.dof_pos.shape == (1, num_dof)
        assert state.dof_vel.shape == (1, num_dof)
        assert state.state_conversion == StateConversion.SIMULATOR

    def test_get_simulator_dof_forces_returns_robot_state(self):
        from hymotion_isaacsim.isaacsim_simulator import IsaacSimSimulator
        num_dof = 69
        art = _make_fake_articulation()
        art.get_measured_joint_forces.return_value = torch.zeros((1, num_dof))
        adapter = IsaacSimSimulator.__new__(IsaacSimSimulator)
        adapter._articulation = art
        adapter.device = "cpu"
        adapter.num_envs = 1
        state = adapter._get_simulator_dof_forces()
        assert state.dof_forces.shape == (1, num_dof)
        assert state.state_conversion == StateConversion.SIMULATOR

    def test_get_simulator_dof_limits_for_verification(self):
        from hymotion_isaacsim.isaacsim_simulator import IsaacSimSimulator
        num_dof = 69
        art = _make_fake_articulation()
        limits = torch.zeros((1, num_dof, 2))
        limits[..., 0] = -3.14
        limits[..., 1] = 3.14
        # DOF limits are read from the view, not the single articulation
        art._articulation_view.get_dof_limits.return_value = limits
        adapter = IsaacSimSimulator.__new__(IsaacSimSimulator)
        adapter._articulation = art
        adapter.device = "cpu"
        lower, upper = adapter._get_simulator_dof_limits_for_verification()
        assert lower.shape == (num_dof,)
        assert upper.shape == (num_dof,)
        assert torch.allclose(lower, torch.tensor([-3.14]).expand(num_dof))
        assert torch.allclose(upper, torch.tensor([3.14]).expand(num_dof))

    def test_get_simulator_object_root_state_returns_empty(self):
        from hymotion_isaacsim.isaacsim_simulator import IsaacSimSimulator
        adapter = IsaacSimSimulator.__new__(IsaacSimSimulator)
        adapter.device = "cpu"
        adapter.num_envs = 1
        state = adapter._get_simulator_object_root_state()
        assert state.root_pos is not None
        assert state.root_pos.shape == (1, 0, 3)
        assert state.root_rot.shape == (1, 0, 4)
        assert state.root_vel.shape == (1, 0, 3)
        assert state.root_ang_vel.shape == (1, 0, 3)
        assert state.state_conversion == StateConversion.SIMULATOR

    def test_get_simulator_object_contact_buf_returns_empty(self):
        from hymotion_isaacsim.isaacsim_simulator import IsaacSimSimulator
        adapter = IsaacSimSimulator.__new__(IsaacSimSimulator)
        adapter.device = "cpu"
        adapter.num_envs = 1
        state = adapter._get_simulator_object_contact_buf()
        assert state.contact_forces is not None
        assert state.contact_forces.shape == (1, 0, 3)
        assert state.state_conversion == StateConversion.SIMULATOR


class TestStateSetterAndControl:
    def test_set_simulator_env_state_writes_to_articulation(self):
        from hymotion_isaacsim.isaacsim_simulator import IsaacSimSimulator
        art = _make_fake_articulation()
        adapter = IsaacSimSimulator.__new__(IsaacSimSimulator)
        adapter._articulation = art
        adapter.device = "cpu"
        adapter.num_envs = 1
        new_states = ResetState(
            root_pos=torch.zeros((1, 3)),
            root_rot=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            root_vel=torch.zeros((1, 3)),
            root_ang_vel=torch.zeros((1, 3)),
            dof_pos=torch.zeros((1, 69)),
            dof_vel=torch.zeros((1, 69)),
            state_conversion=StateConversion.SIMULATOR,
        )
        adapter._set_simulator_env_state(new_states, env_ids=torch.tensor([0]))
        art.set_world_pose.assert_called_once()
        art.set_linear_velocity.assert_called_once()
        art.set_angular_velocity.assert_called_once()
        art.set_joint_positions.assert_called_once()
        art.set_joint_velocities.assert_called_once()
        art.set_joint_position_targets.assert_called_once()

    def test_apply_simulator_pd_targets_calls_articulation(self):
        from hymotion_isaacsim.isaacsim_simulator import IsaacSimSimulator
        art = _make_fake_articulation()
        adapter = IsaacSimSimulator.__new__(IsaacSimSimulator)
        adapter._articulation = art
        targets = torch.zeros((1, 69))
        adapter._apply_simulator_pd_targets(targets)
        art.set_joint_position_targets.assert_called_once()

    def test_apply_simulator_torques_calls_articulation(self):
        from hymotion_isaacsim.isaacsim_simulator import IsaacSimSimulator
        art = _make_fake_articulation()
        adapter = IsaacSimSimulator.__new__(IsaacSimSimulator)
        adapter._articulation = art
        torques = torch.zeros((1, 69))
        adapter._apply_simulator_torques(torques)
        art.set_joint_efforts.assert_called_once()

    def test_apply_root_velocity_impulse_sets_velocity(self):
        from hymotion_isaacsim.isaacsim_simulator import IsaacSimSimulator
        art = _make_fake_articulation()
        adapter = IsaacSimSimulator.__new__(IsaacSimSimulator)
        adapter._articulation = art
        lin_vel = torch.tensor([[1.0, 0.0, 0.0]])
        ang_vel = torch.tensor([[0.0, 0.0, 1.0]])
        env_ids = torch.tensor([0])
        adapter._apply_root_velocity_impulse(lin_vel, ang_vel, env_ids)
        art.set_linear_velocity.assert_called_once()
        art.set_angular_velocity.assert_called_once()

    def test_physics_step_calls_world_step_decimation_times(self):
        from hymotion_isaacsim.isaacsim_simulator import IsaacSimSimulator
        world = MagicMock()
        adapter = IsaacSimSimulator.__new__(IsaacSimSimulator)
        adapter._world = world
        adapter._articulation = _make_fake_articulation()
        adapter.device = "cpu"
        adapter.num_envs = 1
        adapter.decimation = 4
        adapter.headless = True
        adapter._common_actions = torch.zeros((1, 69))
        adapter._apply_control = MagicMock()
        adapter._physics_step()
        assert world.step.call_count == 4
        assert adapter._apply_control.call_count == 4

    def test_physics_step_renders_when_not_headless(self):
        from hymotion_isaacsim.isaacsim_simulator import IsaacSimSimulator
        world = MagicMock()
        adapter = IsaacSimSimulator.__new__(IsaacSimSimulator)
        adapter._world = world
        adapter._articulation = _make_fake_articulation()
        adapter.device = "cpu"
        adapter.num_envs = 1
        adapter.decimation = 2
        adapter.headless = False
        adapter._common_actions = torch.zeros((1, 69))
        adapter._apply_control = MagicMock()
        adapter._physics_step()
        assert world.step.call_count == 2
        world.render.assert_called_once()

    def test_close_calls_simulation_app(self):
        from hymotion_isaacsim.isaacsim_simulator import IsaacSimSimulator
        sim_app = MagicMock()
        adapter = IsaacSimSimulator.__new__(IsaacSimSimulator)
        adapter._simulation_app = sim_app
        adapter._simulation_running = True
        adapter.close()
        sim_app.close.assert_called_once()
        assert adapter._simulation_running is False

    def test_init_camera_is_noop(self):
        from hymotion_isaacsim.isaacsim_simulator import IsaacSimSimulator
        adapter = IsaacSimSimulator.__new__(IsaacSimSimulator)
        # Should not raise
        adapter._init_camera()


class TestViewportAndMarkers:
    def test_write_viewport_to_file_uses_viewport_api(self):
        from hymotion_isaacsim.isaacsim_simulator import IsaacSimSimulator
        adapter = IsaacSimSimulator.__new__(IsaacSimSimulator)
        # Pre-set a mock viewport API so the method doesn't try to import omni
        mock_vp = MagicMock()
        adapter._viewport_api = mock_vp
        adapter._capture_viewport = MagicMock()
        adapter._write_viewport_to_file("/tmp/test_frame.png")
        # Verify the capture was called with the viewport API and file path
        adapter._capture_viewport.assert_called_once_with(mock_vp, "/tmp/test_frame.png")

    def test_update_simulator_markers_does_not_raise(self):
        from hymotion_isaacsim.isaacsim_simulator import IsaacSimSimulator
        adapter = IsaacSimSimulator.__new__(IsaacSimSimulator)
        adapter._update_simulator_markers(None)
        adapter._update_simulator_markers({})


class TestContactSensors:
    def test_get_simulator_bodies_contact_buf_returns_zeros_without_sensors(self):
        from hymotion_isaacsim.isaacsim_simulator import IsaacSimSimulator
        num_bodies = 24
        art = _make_fake_articulation()
        adapter = IsaacSimSimulator.__new__(IsaacSimSimulator)
        adapter._articulation = art
        adapter.device = "cpu"
        adapter.num_envs = 1
        adapter._num_bodies = num_bodies
        adapter._contact_sensors = {}
        result = adapter._get_simulator_bodies_contact_buf()
        assert result.rigid_body_contact_forces.shape == (1, num_bodies, 3)
        assert (result.rigid_body_contact_forces == 0).all()
        assert result.state_conversion == StateConversion.SIMULATOR
