from __future__ import annotations

import pytest
import torch
import numpy as np
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

from protomotions.simulator.base_simulator.simulator_state import (
    ResetState, StateConversion,
)
from protomotions.robot_configs.base import ControlType
from protomotions.simulator.base_simulator.config import MarkerState


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
    art._articulation_view._physics_view = None
    art._articulation_view.is_physics_handle_valid.return_value = False
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
        art.get_measured_joint_efforts.return_value = torch.zeros((1, num_dof))
        adapter = IsaacSimSimulator.__new__(IsaacSimSimulator)
        adapter._articulation = art
        adapter.device = "cpu"
        adapter.num_envs = 1
        state = adapter._get_simulator_dof_forces()
        assert state.dof_forces.shape == (1, num_dof)
        assert state.state_conversion == StateConversion.SIMULATOR

    def test_get_simulator_bodies_state_reorders_common_body_view_into_sim_order(self):
        from hymotion_isaacsim.isaacsim_simulator import IsaacSimSimulator

        common_body_names = ["Pelvis", "Head", "L_Hand"]
        sim_body_names = ["Head", "Pelvis", "L_Hand"]
        art = _make_fake_articulation(body_names=sim_body_names)
        body_rigid_view = MagicMock()
        body_rigid_view.get_world_poses.return_value = (
            torch.tensor(
                [
                    [10.0, 0.0, 0.0],  # Pelvis in common/body-view order
                    [20.0, 0.0, 0.0],  # Head
                    [30.0, 0.0, 0.0],  # L_Hand
                ]
            ),
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                ]
            ),
        )
        body_rigid_view.get_velocities.return_value = torch.tensor(
            [
                [1.0, 1.0, 1.0, 11.0, 11.0, 11.0],
                [2.0, 2.0, 2.0, 22.0, 22.0, 22.0],
                [3.0, 3.0, 3.0, 33.0, 33.0, 33.0],
            ]
        )

        adapter = IsaacSimSimulator.__new__(IsaacSimSimulator)
        adapter._articulation = art
        adapter._body_rigid_view = body_rigid_view
        adapter.robot_config = SimpleNamespace(
            kinematic_info=SimpleNamespace(body_names=common_body_names)
        )
        adapter.device = "cpu"
        adapter.num_envs = 1

        state = adapter._get_simulator_bodies_state()

        assert np.allclose(
            state.rigid_body_pos[0].numpy(),
            np.array(
                [
                    [20.0, 0.0, 0.0],  # Head first in simulator order
                    [10.0, 0.0, 0.0],  # Pelvis second
                    [30.0, 0.0, 0.0],
                ]
            ),
        )
        assert np.allclose(
            state.rigid_body_vel[0].numpy(),
            np.array(
                [
                    [2.0, 2.0, 2.0],
                    [1.0, 1.0, 1.0],
                    [3.0, 3.0, 3.0],
                ]
            ),
        )
        assert state.state_conversion == StateConversion.SIMULATOR

    def test_get_simulator_bodies_state_prefers_physics_view_link_data(self):
        from hymotion_isaacsim.isaacsim_simulator import IsaacSimSimulator

        art = _make_fake_articulation(body_names=["Pelvis", "Head"])
        art._articulation_view.is_physics_handle_valid.return_value = True
        art._articulation_view._physics_view = MagicMock()
        art._articulation_view._physics_view.get_link_transforms.return_value = torch.tensor(
            [
                [
                    [1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.4],
                    [4.0, 5.0, 6.0, 0.5, 0.6, 0.7, 0.8],
                ]
            ]
        )
        art._articulation_view._physics_view.get_link_velocities.return_value = torch.tensor(
            [
                [
                    [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
                    [20.0, 21.0, 22.0, 23.0, 24.0, 25.0],
                ]
            ]
        )

        adapter = IsaacSimSimulator.__new__(IsaacSimSimulator)
        adapter._articulation = art
        adapter._world = SimpleNamespace(physics_sim_view=MagicMock())
        adapter.device = "cpu"
        adapter.num_envs = 1

        state = adapter._get_simulator_bodies_state()

        assert np.allclose(state.rigid_body_pos[0].numpy(), np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        assert np.allclose(
            state.rigid_body_rot[0].numpy(),
            np.array([[0.4, 0.1, 0.2, 0.3], [0.8, 0.5, 0.6, 0.7]]),
        )
        assert np.allclose(
            state.rigid_body_vel[0].numpy(),
            np.array([[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]]),
        )
        assert np.allclose(
            state.rigid_body_ang_vel[0].numpy(),
            np.array([[13.0, 14.0, 15.0], [23.0, 24.0, 25.0]]),
        )

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

    def test_apply_simulator_pd_targets_calls_articulation(self):
        from hymotion_isaacsim.isaacsim_simulator import IsaacSimSimulator
        art = _make_fake_articulation()
        adapter = IsaacSimSimulator.__new__(IsaacSimSimulator)
        adapter._articulation = art
        targets = torch.zeros((1, 69))
        adapter._apply_simulator_pd_targets(targets)
        art._articulation_view.set_joint_position_targets.assert_called_once()

    def test_apply_simulator_torques_calls_articulation(self):
        from hymotion_isaacsim.isaacsim_simulator import IsaacSimSimulator
        art = _make_fake_articulation()
        adapter = IsaacSimSimulator.__new__(IsaacSimSimulator)
        adapter._articulation = art
        torques = torch.zeros((1, 69))
        adapter._apply_simulator_torques(torques)
        art._articulation_view.set_joint_efforts.assert_called_once()

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

    def test_create_simulation_configures_builtin_pd_drives(self):
        from hymotion_isaacsim.isaacsim_simulator import IsaacSimSimulator
        art = _make_fake_articulation()
        adapter = IsaacSimSimulator.__new__(IsaacSimSimulator)
        adapter._articulation = art
        adapter._visualization_markers = None
        control_info = {
            name: SimpleNamespace(
                stiffness=800.0 + idx,
                damping=80.0 + idx,
                effort_limit=500.0 + idx,
                velocity_limit=100.0 + idx,
            )
            for idx, name in enumerate(art._articulation_view.dof_names)
        }
        adapter.robot_config = SimpleNamespace(
            control=SimpleNamespace(
                control_type=ControlType.BUILT_IN_PD,
                control_info=control_info,
            ),
            kinematic_info=SimpleNamespace(
                dof_names=list(art._articulation_view.dof_names),
            ),
        )

        adapter._create_simulation()

        art._articulation_view.switch_control_mode.assert_called_once_with(mode="position")
        art._articulation_view.set_gains.assert_called_once()
        _, kwargs = art._articulation_view.set_gains.call_args
        assert np.allclose(kwargs["kps"], np.array([[800.0 + i for i in range(69)]], dtype=np.float32))
        assert np.allclose(kwargs["kds"], np.array([[80.0 + i for i in range(69)]], dtype=np.float32))
        art._articulation_view.set_max_efforts.assert_called_once()

    def test_init_camera_positions_follow_view_behind_root(self):
        from hymotion_isaacsim.isaacsim_simulator import IsaacSimSimulator
        adapter = IsaacSimSimulator.__new__(IsaacSimSimulator)
        adapter._perspective_view = MagicMock()
        adapter._camera_target = {"env": 0, "element": 0}
        adapter._get_simulator_root_state = MagicMock(
            return_value=SimpleNamespace(root_pos=torch.tensor([[1.0, 2.0, 3.0]]))
        )

        adapter._init_camera()

        adapter._perspective_view.set_camera_view.assert_called_once()
        eye, target = adapter._perspective_view.set_camera_view.call_args.args
        assert np.allclose(eye, np.array([1.0, -3.5, 4.2]))
        assert np.allclose(target, np.array([1.0, 2.0, 3.7]))
        assert np.allclose(adapter._cam_prev_char_pos, np.array([1.0, 2.0, 3.0]))

    def test_configure_follow_camera_lens_sets_wide_angle_defaults(self):
        from hymotion_isaacsim.isaacsim_simulator import IsaacSimSimulator

        clipping_attr = MagicMock()
        focal_attr = MagicMock()
        horiz_attr = MagicMock()
        vert_attr = MagicMock()
        camera = MagicMock()
        camera.GetClippingRangeAttr.return_value = clipping_attr
        camera.GetFocalLengthAttr.return_value = focal_attr
        camera.GetHorizontalApertureAttr.return_value = horiz_attr
        camera.GetVerticalApertureAttr.return_value = vert_attr
        fake_gf = SimpleNamespace(Vec2f=lambda x, y: (x, y))
        sys.modules["pxr"] = SimpleNamespace(Gf=fake_gf)

        adapter = IsaacSimSimulator.__new__(IsaacSimSimulator)
        try:
            adapter._configure_follow_camera_lens(camera)
        finally:
            sys.modules.pop("pxr", None)

        clipping_attr.Set.assert_called_once_with((0.01, 10000.0))
        focal_attr.Set.assert_called_once_with(24.0)
        horiz_attr.Set.assert_called_once_with(36.0)
        vert_attr.Set.assert_called_once_with(20.25)


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

    def test_prepare_headless_capture_primes_camera_with_render_async(self):
        from hymotion_isaacsim.isaacsim_simulator import IsaacSimSimulator

        adapter = IsaacSimSimulator.__new__(IsaacSimSimulator)
        adapter.headless = True
        adapter._world = MagicMock()
        adapter._simulation_app = MagicMock()
        adapter._headless_capture_ready = False

        fake_camera = MagicMock()
        fake_camera.get_rgba.side_effect = [
            np.zeros((0,), dtype=np.uint8),
            np.ones((2, 2, 4), dtype=np.uint8),
        ]
        adapter._ensure_headless_capture_camera = MagicMock(return_value=fake_camera)

        adapter.prepare_headless_capture()

        adapter._ensure_headless_capture_camera.assert_called_once_with()
        assert adapter._world.render_async.call_count == 2
        assert adapter._simulation_app.run_coroutine.call_count == 2
        assert adapter._headless_capture_ready is True

    def test_capture_headless_follow_camera_rgba_uses_render_only_path(self):
        from hymotion_isaacsim.isaacsim_simulator import IsaacSimSimulator

        adapter = IsaacSimSimulator.__new__(IsaacSimSimulator)
        adapter._world = MagicMock()
        adapter.prepare_headless_capture = MagicMock()
        adapter._update_headless_capture_camera_pose = MagicMock()
        adapter._headless_capture_camera = MagicMock()
        rgba = np.ones((4, 5, 4), dtype=np.uint8)
        adapter._headless_capture_camera.get_rgba.return_value = rgba

        result = adapter._capture_headless_follow_camera_rgba()

        adapter.prepare_headless_capture.assert_called_once_with()
        adapter._update_headless_capture_camera_pose.assert_called_once_with()
        adapter._world.render.assert_called_once_with()
        adapter._headless_capture_camera.get_rgba.assert_called_once_with()
        assert result is rgba

    def test_update_simulator_markers_updates_marker_prims(self):
        from hymotion_isaacsim.isaacsim_simulator import IsaacSimSimulator
        adapter = IsaacSimSimulator.__new__(IsaacSimSimulator)
        marker_a = MagicMock()
        marker_b = MagicMock()
        adapter._marker_prim_groups = {"body_markers_red": [marker_a, marker_b]}

        adapter._update_simulator_markers(
            {
                "body_markers_red": MarkerState(
                    translation=torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]),
                    orientation=torch.tensor([[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]]),
                )
            }
        )

        marker_a.set_world_pose.assert_called_once()
        marker_b.set_world_pose.assert_called_once()
        marker_a_kwargs = marker_a.set_world_pose.call_args.kwargs
        marker_b_kwargs = marker_b.set_world_pose.call_args.kwargs
        assert np.allclose(marker_a_kwargs["position"], np.array([1.0, 2.0, 3.0]))
        assert np.allclose(marker_a_kwargs["orientation"], np.array([1.0, 0.0, 0.0, 0.0]))
        assert np.allclose(marker_b_kwargs["position"], np.array([4.0, 5.0, 6.0]))
        assert np.allclose(marker_b_kwargs["orientation"], np.array([1.0, 0.0, 0.0, 0.0]))


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
