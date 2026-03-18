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

    def _get_sim_body_ordering(self) -> SimBodyOrdering:
        """Return the body/DOF ordering as reported by the Isaac Sim articulation."""
        return SimBodyOrdering(
            body_names=list(self._articulation.body_names),
            dof_names=list(self._articulation.dof_names),
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
        """Return (lower_limits, upper_limits) as 1-D tensors from the articulation."""
        dof_limits = self._articulation.get_dof_limits()
        # dof_limits shape: [N, num_dof, 2] — take first env, split lower/upper
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
        """Write root pose, root velocity, joint positions/velocities to the articulation."""
        self._articulation.set_world_poses(
            positions=new_states.root_pos,
            orientations=new_states.root_rot,
        )
        self._articulation.set_velocities(
            torch.cat([new_states.root_vel, new_states.root_ang_vel], dim=-1)
        )
        self._articulation.set_joint_positions(new_states.dof_pos)
        self._articulation.set_joint_velocities(new_states.dof_vel)
        self._articulation.set_joint_position_targets(new_states.dof_pos)

    def _apply_root_velocity_impulse(
        self,
        linear_velocity: torch.Tensor,
        angular_velocity: torch.Tensor,
        env_ids: torch.Tensor,
    ) -> None:
        """Set root velocities on the articulation for the given environments."""
        vel = torch.cat([linear_velocity, angular_velocity], dim=-1)
        self._articulation.set_velocities(vel)

    # ------------------------------------------------------------------
    # Group 4: State Getters
    # ------------------------------------------------------------------

    def _get_simulator_root_state(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> RootOnlyState:
        """Read root pose/velocity from the articulation and return a ``RootOnlyState``."""
        positions, quaternions = self._articulation.get_world_poses()
        velocities = self._articulation.get_velocities()

        root_pos = positions
        root_rot = quaternions
        root_vel = velocities[:, :3]
        root_ang_vel = velocities[:, 3:]

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
        """Read per-link transforms and velocities from the articulation view."""
        # Access the underlying articulation view for per-body data
        art_view = self._articulation._articulation_view

        # get_link_transforms returns [N, num_bodies, 7] (pos xyz + quat wxyz)
        link_transforms = art_view.get_link_transforms()
        body_pos = link_transforms[:, :, :3]
        body_rot = link_transforms[:, :, 3:]

        # get_link_velocities returns [N, num_bodies, 6] (lin_vel xyz + ang_vel xyz)
        link_velocities = art_view.get_link_velocities()
        body_vel = link_velocities[:, :, :3]
        body_ang_vel = link_velocities[:, :, 3:]

        if env_ids is not None:
            body_pos = body_pos[env_ids]
            body_rot = body_rot[env_ids]
            body_vel = body_vel[env_ids]
            body_ang_vel = body_ang_vel[env_ids]

        return RobotState(
            rigid_body_pos=body_pos,
            rigid_body_rot=body_rot,
            rigid_body_vel=body_vel,
            rigid_body_ang_vel=body_ang_vel,
            state_conversion=StateConversion.SIMULATOR,
        )

    def _get_simulator_dof_state(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> RobotState:
        """Read joint positions and velocities from the articulation."""
        dof_pos = self._articulation.get_joint_positions()
        dof_vel = self._articulation.get_joint_velocities()

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
        """Apply PD position targets to the articulation."""
        self._articulation.set_joint_position_targets(pd_targets)

    def _apply_simulator_torques(self, torques: torch.Tensor) -> None:
        """Apply raw torques to the articulation."""
        self._articulation.set_joint_efforts(torques)

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
