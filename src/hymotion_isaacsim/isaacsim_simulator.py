"""IsaacSim adapter implementing the ProtoMotions Simulator base class interface."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from protomotions.simulator.base_simulator.simulator import Simulator
from protomotions.simulator.base_simulator.config import SimBodyOrdering, SimulatorConfig
from protomotions.simulator.base_simulator.simulator_state import (
    RobotState,
    RootOnlyState,
    ObjectState,
    ResetState,
    StateConversion,
)
from protomotions.robot_configs.base import RobotConfig
from protomotions.components.scene_lib import SceneLib
from protomotions.components.terrains.terrain import Terrain


class IsaacSimSimulator(Simulator):
    """Adapter that wraps an Isaac Sim world and articulation behind ProtoMotions' Simulator API.

    The constructor stores the Isaac Sim objects and then delegates to the base class
    ``__init__``. Callers must set ``self._world``, ``self._articulation``, and
    ``self._simulation_app`` **before** ``super().__init__()`` is invoked, which is
    why those assignments appear at the top of ``__init__``.
    """

    def __init__(
        self,
        world,
        articulation,
        simulation_app,
        config: SimulatorConfig,
        robot_config: RobotConfig,
        scene_lib: SceneLib,
        device: torch.device,
        terrain: Optional[Terrain] = None,
    ) -> None:
        # Store Isaac Sim objects BEFORE calling super().__init__() so that any
        # overridden method invoked during base construction can access them.
        self._world = world
        self._articulation = articulation
        self._simulation_app = simulation_app
        self._contact_sensors: Dict = {}
        self._contact_body_prim_paths: list = []

        super().__init__(
            config=config,
            robot_config=robot_config,
            terrain=terrain,
            device=device,
            scene_lib=scene_lib,
        )

    # ------------------------------------------------------------------
    # Group 1: Initialization
    # ------------------------------------------------------------------

    def _create_simulation(self) -> None:
        """No-op: the Isaac Sim world and articulation are created by the caller."""
        pass

    @property
    def _view(self):
        """The underlying ArticulationView, which exposes batched + per-body APIs."""
        return self._articulation._articulation_view

    def _get_sim_body_ordering(self) -> SimBodyOrdering:
        """Return the body/DOF ordering as reported by the Isaac Sim articulation.

        ``body_names`` lives on the articulation *view*, not the single articulation.
        ``dof_names`` is available on both but we use the view for consistency.
        """
        return SimBodyOrdering(
            body_names=list(self._view.body_names),
            dof_names=list(self._view.dof_names),
        )

    def close(self) -> None:
        """Close the simulator and the Isaac Sim application."""
        self._simulation_running = False
        self._simulation_app.close()

    # ------------------------------------------------------------------
    # Group 2: Environment Setup
    # ------------------------------------------------------------------

    def _get_simulator_dof_limits_for_verification(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (lower_limits, upper_limits) as 1-D tensors from the articulation.

        The view's ``get_dof_limits()`` returns ``[N, num_dof, 2]``.
        """
        dof_limits = self._view.get_dof_limits()
        # Shape: [N, num_dof, 2] — take first env, split lower/upper
        lower = dof_limits[0, :, 0].to(self.device)
        upper = dof_limits[0, :, 1].to(self.device)
        return lower, upper

    # ------------------------------------------------------------------
    # Group 3: Simulation Steps
    # ------------------------------------------------------------------

    def _physics_step(self) -> None:
        """Advance the simulation by ``self.decimation`` sub-steps."""
        for _ in range(self.decimation):
            self._apply_control()
            self._world.step(render=False)
        if not self.headless:
            self._world.render()

    def _set_simulator_env_state(
        self,
        new_states: ResetState,
        new_object_states: Optional[ObjectState] = None,
        env_ids: Optional[torch.Tensor] = None,
    ) -> None:
        """Write root pose, root velocity, joint positions/velocities to the articulation.

        ``SingleArticulation`` uses *singular* setters (``set_world_pose``,
        ``set_linear_velocity``, etc.) that expect unbatched tensors, so we
        squeeze the batch dimension before calling them.
        """
        # Root pose — squeeze batch dim for single articulation
        root_pos = new_states.root_pos.squeeze(0)
        root_rot = new_states.root_rot.squeeze(0)
        self._articulation.set_world_pose(position=root_pos, orientation=root_rot)

        # Root velocity
        root_vel = new_states.root_vel.squeeze(0)
        root_ang_vel = new_states.root_ang_vel.squeeze(0)
        self._articulation.set_linear_velocity(root_vel)
        self._articulation.set_angular_velocity(root_ang_vel)

        # Joint state — these accept batched or unbatched
        self._articulation.set_joint_positions(new_states.dof_pos.squeeze(0))
        self._articulation.set_joint_velocities(new_states.dof_vel.squeeze(0))
        self._articulation.set_joint_position_targets(new_states.dof_pos.squeeze(0))

    def _apply_root_velocity_impulse(
        self,
        linear_velocity: torch.Tensor,
        angular_velocity: torch.Tensor,
        env_ids: torch.Tensor,
    ) -> None:
        """Set root velocities on the articulation for the given environments."""
        self._articulation.set_linear_velocity(linear_velocity.squeeze(0))
        self._articulation.set_angular_velocity(angular_velocity.squeeze(0))

    # ------------------------------------------------------------------
    # Group 4: State Getters
    # ------------------------------------------------------------------

    def _get_simulator_root_state(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> RootOnlyState:
        """Read root pose/velocity from the articulation and return a ``RootOnlyState``.

        The single articulation returns unbatched tensors, so we unsqueeze to
        ``(1, ...)`` to match the ``(num_envs, ...)`` convention.
        """
        pos, quat = self._articulation.get_world_pose()
        vel = self._articulation.get_world_velocity()

        # SingleArticulation returns (3,) and (4,) — add batch dim
        root_pos = pos.unsqueeze(0) if pos.dim() == 1 else pos
        root_rot = quat.unsqueeze(0) if quat.dim() == 1 else quat
        # get_world_velocity may not exist; fall back to get_velocities on view
        if vel is None:
            vels = self._view.get_velocities()
            root_vel = vels[:, :3]
            root_ang_vel = vels[:, 3:]
        elif vel.dim() == 1:
            root_vel = vel[:3].unsqueeze(0)
            root_ang_vel = vel[3:].unsqueeze(0)
        else:
            root_vel = vel[:, :3]
            root_ang_vel = vel[:, 3:]

        if env_ids is not None:
            root_pos = root_pos[env_ids]
            root_rot = root_rot[env_ids]
            root_vel = root_vel[env_ids]
            root_ang_vel = root_ang_vel[env_ids]

        return RootOnlyState(
            root_pos=root_pos,
            root_rot=root_rot,
            root_vel=root_vel,
            root_ang_vel=root_ang_vel,
            state_conversion=StateConversion.SIMULATOR,
        )

    def _get_simulator_bodies_state(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> RobotState:
        """Read per-body transforms and velocities from the articulation view.

        The view's ``get_world_poses()`` returns ``(positions [N,num_bodies,3],
        quats [N,num_bodies,4])``. ``get_velocities()`` returns
        ``[N, num_bodies, 6]`` (lin xyz + ang xyz). If the view only returns
        root-level data we fall back to ``get_linear_velocities`` /
        ``get_angular_velocities`` which are explicitly per-body.
        """
        positions, quats = self._view.get_world_poses()
        # Ensure batch dim
        if positions.dim() == 2:
            # (num_bodies, 3) -> (1, num_bodies, 3)
            positions = positions.unsqueeze(0)
            quats = quats.unsqueeze(0)

        lin_vels = self._view.get_linear_velocities()
        ang_vels = self._view.get_angular_velocities()
        if lin_vels.dim() == 2:
            lin_vels = lin_vels.unsqueeze(0)
            ang_vels = ang_vels.unsqueeze(0)

        if env_ids is not None:
            positions = positions[env_ids]
            quats = quats[env_ids]
            lin_vels = lin_vels[env_ids]
            ang_vels = ang_vels[env_ids]

        return RobotState(
            rigid_body_pos=positions,
            rigid_body_rot=quats,
            rigid_body_vel=lin_vels,
            rigid_body_ang_vel=ang_vels,
            state_conversion=StateConversion.SIMULATOR,
        )

    def _get_simulator_dof_state(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> RobotState:
        """Read joint positions and velocities from the articulation."""
        dof_pos = self._articulation.get_joint_positions()
        dof_vel = self._articulation.get_joint_velocities()

        # SingleArticulation may return 1-D tensors; ensure batch dim
        if dof_pos.dim() == 1:
            dof_pos = dof_pos.unsqueeze(0)
            dof_vel = dof_vel.unsqueeze(0)

        if env_ids is not None:
            dof_pos = dof_pos[env_ids]
            dof_vel = dof_vel[env_ids]

        return RobotState(
            dof_pos=dof_pos,
            dof_vel=dof_vel,
            state_conversion=StateConversion.SIMULATOR,
        )

    def _get_simulator_dof_forces(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> RobotState:
        """Read measured joint forces from the articulation."""
        dof_forces = self._articulation.get_measured_joint_forces()

        if dof_forces.dim() == 1:
            dof_forces = dof_forces.unsqueeze(0)

        if env_ids is not None:
            dof_forces = dof_forces[env_ids]

        return RobotState(
            dof_forces=dof_forces,
            state_conversion=StateConversion.SIMULATOR,
        )

    def setup_contact_sensors(self, body_prim_paths: list) -> None:
        """Create ContactSensor objects for each body prim path and store them.

        The sensors are keyed by prim path so they can be queried later in
        ``_get_simulator_bodies_contact_buf``.  This method requires a running
        Isaac Sim instance because it imports ``omni.isaac.sensor``.

        Args:
            body_prim_paths: List of USD prim paths for which to create sensors.
        """
        from omni.isaac.sensor import ContactSensor  # type: ignore[import]

        self._contact_body_prim_paths = list(body_prim_paths)
        self._contact_sensors = {}
        for path in self._contact_body_prim_paths:
            sensor = ContactSensor(prim_path=path)
            self._contact_sensors[path] = sensor

    def _get_simulator_bodies_contact_buf(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Return contact forces as a ``(num_envs, num_bodies, 3)`` tensor.

        If no contact sensors have been configured (``_contact_sensors`` is
        empty), a zero tensor is returned.  This is the correct behaviour for
        policies that do not use contact information.

        When sensors are available the force reading from each sensor is stacked
        in body order.  The result is always moved to ``self.device``.
        """
        n = self.num_envs if env_ids is None else len(env_ids)

        if not self._contact_sensors:
            return torch.zeros(n, self._num_bodies, 3, device=self.device)

        forces = []
        for path in self._contact_body_prim_paths:
            sensor = self._contact_sensors[path]
            # get_contact_force returns a (3,) or (N, 3) tensor depending on
            # the Isaac Sim version; we normalise to (N, 1, 3).
            force = sensor.get_contact_force()
            if force.dim() == 1:
                # Shape (3,) — broadcast across all envs
                force = force.unsqueeze(0).expand(n, -1)
            force = force.unsqueeze(1)  # (N, 1, 3)
            forces.append(force)

        contact_buf = torch.cat(forces, dim=1)  # (N, num_bodies, 3)

        if env_ids is not None:
            contact_buf = contact_buf[env_ids]

        return contact_buf.to(self.device)

    def _get_simulator_object_root_state(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> ObjectState:
        """Return empty object state (no scene objects in this adapter)."""
        n = self.num_envs if env_ids is None else len(env_ids)
        return ObjectState(
            root_pos=torch.zeros(n, 0, 3, device=self.device),
            root_rot=torch.zeros(n, 0, 4, device=self.device),
            root_vel=torch.zeros(n, 0, 3, device=self.device),
            root_ang_vel=torch.zeros(n, 0, 3, device=self.device),
            state_conversion=StateConversion.SIMULATOR,
        )

    def _get_simulator_object_contact_buf(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> ObjectState:
        """Return empty object contact state (no scene objects in this adapter)."""
        n = self.num_envs if env_ids is None else len(env_ids)
        return ObjectState(
            contact_forces=torch.zeros(n, 0, 3, device=self.device),
            state_conversion=StateConversion.SIMULATOR,
        )

    # ------------------------------------------------------------------
    # Group 5: Control
    # ------------------------------------------------------------------

    def _apply_simulator_pd_targets(self, pd_targets: torch.Tensor) -> None:
        """Apply PD position targets to the articulation.

        Squeeze batch dim for SingleArticulation which expects 1-D input.
        """
        self._articulation.set_joint_position_targets(pd_targets.squeeze(0))

    def _apply_simulator_torques(self, torques: torch.Tensor) -> None:
        """Apply raw torques to the articulation.

        Squeeze batch dim for SingleArticulation which expects 1-D input.
        """
        self._articulation.set_joint_efforts(torques.squeeze(0))

    # ------------------------------------------------------------------
    # Group 6: Rendering & Visualization
    # ------------------------------------------------------------------

    def _write_viewport_to_file(self, file_name: str) -> None:
        """Capture the current viewport render to a file.

        Lazily acquires the active viewport API on first call, then delegates
        to ``self._capture_viewport`` (which may be overridden in tests) or to
        ``omni.kit.viewport.utility.capture_viewport_to_file`` in real usage.
        """
        if not hasattr(self, '_viewport_api') or self._viewport_api is None:
            from omni.kit.viewport.utility import get_active_viewport
            self._viewport_api = get_active_viewport()

        if hasattr(self, '_capture_viewport') and callable(self._capture_viewport):
            self._capture_viewport(self._viewport_api, file_name)
        else:
            import omni.kit.viewport.utility as vp_utils
            vp_utils.capture_viewport_to_file(self._viewport_api, file_name)

    def _init_camera(self) -> None:
        """Initialize camera view. No-op for now; can be extended later."""
        pass

    def _update_simulator_markers(
        self, markers_state: Optional[Dict] = None
    ) -> None:
        """No-op. Visualization markers are optional and not yet supported."""
        pass  # Markers not supported in pure Isaac Sim adapter
