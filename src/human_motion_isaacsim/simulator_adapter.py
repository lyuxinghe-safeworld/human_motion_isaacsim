"""IsaacSim adapter implementing the ProtoMotions Simulator base class interface."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch

from protomotions.simulator.base_simulator.simulator import Simulator
from protomotions.simulator.base_simulator.config import (
    ProjectileConfig,
    SimBodyOrdering,
    SimulatorConfig,
)
from protomotions.simulator.base_simulator.simulator_state import (
    RobotState,
    RootOnlyState,
    ObjectState,
    ResetState,
    StateConversion,
)
from protomotions.robot_configs.base import ControlType, RobotConfig
from protomotions.components.scene_lib import SceneLib
from protomotions.components.terrains.terrain import Terrain


class SimulatorAdapter(Simulator):
    """Adapter that wraps an Isaac Sim world and articulation behind ProtoMotions' Simulator API.

    The constructor stores the Isaac Sim objects and then delegates to the base class
    ``__init__``. Callers must set ``self._world``, ``self._articulation``, and
    ``self._simulation_app`` **before** ``super().__init__()`` is invoked, which is
    why those assignments appear at the top of ``__init__``.
    """

    _FOLLOW_CAMERA_OFFSET = np.array([0.0, -5.5, 1.2], dtype=np.float32)
    _FOLLOW_CAMERA_TARGET_OFFSET = np.array([0.0, 0.0, 0.7], dtype=np.float32)
    _FOLLOW_CAMERA_FOCAL_LENGTH = 24.0
    _FOLLOW_CAMERA_HORIZONTAL_APERTURE = 36.0
    _FOLLOW_CAMERA_VERTICAL_APERTURE = 20.25

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
        self._body_rigid_view = None  # Set externally after construction
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
        """Finalize the pre-built world for ProtoMotions control and visualization."""
        self._configure_articulation_drives()
        self._build_visualization_markers(self._visualization_markers)

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
        # May return numpy array — convert to tensor first
        if not isinstance(dof_limits, torch.Tensor):
            dof_limits = torch.tensor(dof_limits, dtype=torch.float32)
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
        ``set_linear_velocity``, etc.) that expect unbatched **numpy** arrays,
        so we squeeze the batch dimension and convert to numpy before calling.
        """
        def _to_np(t):
            return t.detach().cpu().numpy()

        # Root pose — squeeze batch dim for single articulation
        self._articulation.set_world_pose(
            position=_to_np(new_states.root_pos.squeeze(0)),
            orientation=_to_np(new_states.root_rot.squeeze(0)),
        )

        # Root velocity
        self._articulation.set_linear_velocity(_to_np(new_states.root_vel.squeeze(0)))
        self._articulation.set_angular_velocity(_to_np(new_states.root_ang_vel.squeeze(0)))

        # Joint state
        self._articulation.set_joint_positions(_to_np(new_states.dof_pos.squeeze(0)))
        self._articulation.set_joint_velocities(_to_np(new_states.dof_vel.squeeze(0)))
        self._update_articulation_kinematics()

    def _apply_root_velocity_impulse(
        self,
        linear_velocity: torch.Tensor,
        angular_velocity: torch.Tensor,
        env_ids: torch.Tensor,
    ) -> None:
        """Set root velocities on the articulation for the given environments."""
        self._articulation.set_linear_velocity(
            linear_velocity.squeeze(0).detach().cpu().numpy()
        )
        self._articulation.set_angular_velocity(
            angular_velocity.squeeze(0).detach().cpu().numpy()
        )

    def _create_projectiles(self, config: ProjectileConfig) -> None:
        """Keep an in-memory projectile state cache for base-class bookkeeping.

        This adapter does not spawn separate projectile rigid bodies in Isaac Sim,
        but the ProtoMotions base class now expects a projectile pool to exist.
        Maintaining a cached state keeps recording and reset flows consistent
        without introducing unsupported simulator objects here.
        """
        self._projectile_positions = torch.zeros(
            self.num_envs,
            config.num_projectiles,
            3,
            device=self.device,
        )
        self._projectile_rotations_xyzw = torch.zeros(
            self.num_envs,
            config.num_projectiles,
            4,
            device=self.device,
        )
        self._projectile_rotations_xyzw[..., 3] = 1.0

    def _set_projectile_root_states(
        self,
        proj_indices: torch.Tensor,
        positions: torch.Tensor,
        rotations_xyzw: torch.Tensor,
        velocities: torch.Tensor,
        ang_velocities: torch.Tensor,
        env_ids: torch.Tensor,
    ) -> None:
        """Update the cached projectile state for the requested env/projectile pairs."""
        del velocities, ang_velocities
        if not hasattr(self, "_projectile_positions") or not hasattr(
            self, "_projectile_rotations_xyzw"
        ):
            num_projectiles = int(
                getattr(getattr(self, "_proj_config", None), "num_projectiles", 0)
            )
            self._create_projectiles(ProjectileConfig(num_projectiles=num_projectiles))

        self._projectile_positions[env_ids, proj_indices] = positions.to(self.device)
        self._projectile_rotations_xyzw[env_ids, proj_indices] = rotations_xyzw.to(
            self.device
        )

    def _get_projectile_positions_rotations(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cached projectile poses in ProtoMotions' common xyzw format."""
        if not hasattr(self, "_projectile_positions") or not hasattr(
            self, "_projectile_rotations_xyzw"
        ):
            num_projectiles = int(
                getattr(getattr(self, "_proj_config", None), "num_projectiles", 0)
            )
            self._create_projectiles(ProjectileConfig(num_projectiles=num_projectiles))

        return self._projectile_positions, self._projectile_rotations_xyzw

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
        pos = self._ensure_tensor(pos)
        quat = self._ensure_tensor(quat)

        # SingleArticulation returns (3,) and (4,) — add batch dim
        root_pos = pos.unsqueeze(0) if pos.dim() == 1 else pos
        root_rot = quat.unsqueeze(0) if quat.dim() == 1 else quat

        # Try get_world_velocity first; fall back to view's get_velocities
        try:
            vel = self._articulation.get_world_velocity()
            vel = self._ensure_tensor(vel)
        except (AttributeError, TypeError):
            vel = None

        if vel is None:
            vels = self._ensure_tensor(self._view.get_velocities())
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

    def _ensure_tensor(self, data, dtype=torch.float32) -> torch.Tensor:
        """Convert numpy arrays to torch tensors if needed."""
        if isinstance(data, torch.Tensor):
            return data.to(device=self.device, dtype=dtype)
        # numpy → CPU tensor → move to device
        return torch.as_tensor(data, dtype=dtype).to(self.device)

    def _get_simulator_bodies_state(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> RobotState:
        """Read per-body transforms and velocities via the ``RigidPrimView``.

        The ``_body_rigid_view`` must be set externally before this method is
        called (typically in the entry script after creating the world).  It
        wraps all body prim paths (e.g. ``/World/Humanoid/bodies/Pelvis``)
        and returns ``(num_bodies, 3)`` positions, ``(num_bodies, 4)``
        quaternions, and ``(num_bodies, 6)`` velocities.
        """
        physics_view = getattr(self._view, "_physics_view", None)
        if physics_view is not None and self._view.is_physics_handle_valid():
            self._update_articulation_kinematics()
            num_bodies = len(self._view.body_names)
            link_transforms = self._ensure_tensor(physics_view.get_link_transforms())
            link_velocities = self._ensure_tensor(physics_view.get_link_velocities())

            positions = link_transforms[:, :num_bodies, 0:3]
            # Isaac Sim physics tensors expose quaternions as xyzw.
            quats_xyzw = link_transforms[:, :num_bodies, 3:7]
            quats = torch.cat([quats_xyzw[..., 3:4], quats_xyzw[..., :3]], dim=-1)
            lin_vels = link_velocities[:, :num_bodies, 0:3]
            ang_vels = link_velocities[:, :num_bodies, 3:6]
        else:
            if self._body_rigid_view is None:
                raise RuntimeError(
                    "_body_rigid_view not set — call set_body_rigid_view() before using the simulator"
                )

            positions, quats = self._body_rigid_view.get_world_poses()
            positions = self._ensure_tensor(positions)
            quats = self._ensure_tensor(quats)

            vels = self._ensure_tensor(self._body_rigid_view.get_velocities())
            lin_vels = vels[:, :3]
            ang_vels = vels[:, 3:]

            # The external RigidPrimView is constructed in robot-config/common order.
            # Reorder into simulator articulation order before returning a SIMULATOR state.
            body_view_to_sim = self._get_body_view_to_sim_order().to(positions.device)
            positions = positions[body_view_to_sim]
            quats = quats[body_view_to_sim]
            lin_vels = lin_vels[body_view_to_sim]
            ang_vels = ang_vels[body_view_to_sim]

            # RigidPrimView returns (num_bodies, dim) — add batch dim (1, num_bodies, dim)
            positions = positions.unsqueeze(0)
            quats = quats.unsqueeze(0)
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

    def _get_body_view_to_sim_order(self) -> torch.Tensor:
        """Map the common-order RigidPrimView body layout into articulation simulator order."""
        if hasattr(self, "_body_view_to_sim_order"):
            return self._body_view_to_sim_order

        body_view_names = list(self.robot_config.kinematic_info.body_names)
        name_to_body_view_idx = {name: idx for idx, name in enumerate(body_view_names)}
        sim_body_names = list(self._view.body_names)
        try:
            indices = [name_to_body_view_idx[name] for name in sim_body_names]
        except KeyError as exc:
            missing_name = exc.args[0]
            raise RuntimeError(
                f"Body {missing_name!r} exists in articulation order but not in body_rigid_view order"
            ) from exc

        self._body_view_to_sim_order = torch.tensor(
            indices, dtype=torch.long, device=self.device
        )
        return self._body_view_to_sim_order

    def _update_articulation_kinematics(self) -> None:
        """Refresh cached articulation link kinematics when the physics view supports it."""
        world = getattr(self, "_world", None)
        if world is None:
            return
        physics_sim_view = getattr(world, "physics_sim_view", None)
        if physics_sim_view is not None:
            physics_sim_view.update_articulations_kinematic()

    def _get_simulator_dof_state(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> RobotState:
        """Read joint positions and velocities from the articulation."""
        dof_pos = self._ensure_tensor(self._articulation.get_joint_positions())
        dof_vel = self._ensure_tensor(self._articulation.get_joint_velocities())

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
        """Read measured joint efforts (per-DOF forces) from the articulation.

        Note: ``get_measured_joint_forces()`` returns (num_joints, 6) reaction
        forces, not per-DOF torques. ``get_measured_joint_efforts()`` returns
        the per-DOF effort values matching the ``(num_dof,)`` shape.
        """
        dof_forces = self._ensure_tensor(self._articulation.get_measured_joint_efforts())

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
    ) -> RobotState:
        """Return contact forces as a ``RobotState`` with ``rigid_body_contact_forces``.

        If no contact sensors have been configured (``_contact_sensors`` is
        empty), a zero tensor is returned.  This is the correct behaviour for
        policies that do not use contact information.

        When sensors are available the force reading from each sensor is stacked
        in body order.  The result is always moved to ``self.device``.
        """
        n = self.num_envs if env_ids is None else len(env_ids)

        if not self._contact_sensors:
            return RobotState(
                rigid_body_contact_forces=torch.zeros(n, self._num_bodies, 3, device=self.device),
                state_conversion=StateConversion.SIMULATOR,
            )

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

        return RobotState(
            rigid_body_contact_forces=contact_buf.to(self.device),
            state_conversion=StateConversion.SIMULATOR,
        )

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

        ``SingleArticulation`` does not have ``set_joint_position_targets``;
        the batched ``ArticulationView`` does.  We use the view and pass a
        2-D numpy array ``(1, num_dof)``.
        """
        self._view.set_joint_position_targets(
            pd_targets.detach().cpu().numpy()
        )

    def _apply_simulator_torques(self, torques: torch.Tensor) -> None:
        """Apply raw torques to the articulation.

        Use the batched view for consistency with PD targets.
        """
        self._view.set_joint_efforts(
            torques.detach().cpu().numpy()
        )

    # ------------------------------------------------------------------
    # Group 6: Rendering & Visualization
    # ------------------------------------------------------------------

    def _write_viewport_to_file(self, file_name: str) -> None:
        """Capture the current viewport render to a file.

        Uses Replicator's ``rgb`` annotator attached to a render product for
        reliable frame capture that works in both windowed and headless mode.

        On first call, lazily creates a render product and annotator.  If a
        ``_capture_viewport`` callable is set (for test mocking), it is used
        instead.
        """
        if hasattr(self, '_capture_viewport') and callable(self._capture_viewport):
            vp = getattr(self, '_viewport_api', None)
            self._capture_viewport(vp, file_name)
            return

        if self.headless:
            rgba = self._capture_headless_follow_camera_rgba()
            if hasattr(rgba, "numpy"):
                rgba = rgba.numpy()
            rgba = np.asarray(rgba)

            from PIL import Image

            Image.fromarray(rgba[:, :, :3]).save(file_name)
            return

        # Lazy init: create camera, light, render product, and rgb annotator
        if not hasattr(self, '_rep_annotator') or self._rep_annotator is None:
            import omni.replicator.core as rep
            from pxr import UsdLux, UsdGeom, Gf

            stage = self._world.stage

            # Add a dome light for uniform illumination regardless of position
            light_path = "/World/DomeLight"
            if not stage.GetPrimAtPath(light_path).IsValid():
                light_prim = stage.DefinePrim(light_path, "DomeLight")
                UsdLux.DomeLight(light_prim).GetIntensityAttr().Set(1000.0)

            # Create camera via USD prim so we can reposition with set_camera_view
            # Create camera via USD so we can move it with set_camera_view
            from pxr import UsdGeom, Gf
            self._cam_prim_path = "/World/FollowCamera"
            cam_prim = stage.DefinePrim(self._cam_prim_path, "Camera")
            self._configure_follow_camera_lens(UsdGeom.Camera(cam_prim))

            from omni.isaac.core.utils.viewports import set_camera_view
            set_camera_view(
                eye=self._FOLLOW_CAMERA_OFFSET,
                target=self._FOLLOW_CAMERA_TARGET_OFFSET,
                camera_prim_path=self._cam_prim_path,
            )

            rp = rep.create.render_product(self._cam_prim_path, (1280, 720))
            self._rep_annotator = rep.AnnotatorRegistry.get_annotator('rgb')
            self._rep_annotator.attach([rp])
            self._rep_module = rep

        # Move camera to follow the humanoid using USD xform
        root_pos = self._ensure_tensor(
            self._articulation.get_world_pose()[0]
        ).cpu().numpy()
        eye = root_pos + self._FOLLOW_CAMERA_OFFSET
        target = root_pos + self._FOLLOW_CAMERA_TARGET_OFFSET

        from omni.isaac.core.utils.viewports import set_camera_view
        set_camera_view(eye=eye, target=target, camera_prim_path=self._cam_prim_path)

        self._rep_module.orchestrator.step()
        data = self._rep_annotator.get_data()
        if data is not None and data.size > 0:
            from PIL import Image
            img = Image.fromarray(data[:, :, :3])
            img.save(file_name)

    def prepare_headless_capture(self) -> None:
        """Warm the headless follow-camera pipeline before the motion loop starts."""
        if not self.headless or getattr(self, "_headless_capture_ready", False):
            return

        camera = self._ensure_headless_capture_camera()
        for _ in range(3):
            self._simulation_app.run_coroutine(self._world.render_async())
            rgba = camera.get_rgba()
            if rgba is not None and getattr(rgba, "size", 0) > 0:
                self._headless_capture_ready = True
                return

        raise RuntimeError("Headless capture camera failed to warm up")

    def _ensure_headless_capture_camera(self):
        """Create a dedicated follow-camera sensor for drift-free headless capture."""
        camera = getattr(self, "_headless_capture_camera", None)
        if camera is not None:
            return camera

        from isaacsim.sensors.camera import Camera
        from omni.isaac.core.utils.viewports import set_camera_view
        from pxr import UsdLux, UsdGeom

        stage = self._world.stage
        light_path = "/World/DomeLight"
        if not stage.GetPrimAtPath(light_path).IsValid():
            light_prim = stage.DefinePrim(light_path, "DomeLight")
            UsdLux.DomeLight(light_prim).GetIntensityAttr().Set(1000.0)

        self._headless_capture_camera_path = "/World/FollowCamera"
        camera = Camera(
            prim_path=self._headless_capture_camera_path,
            name="follow_camera_capture",
            resolution=(1280, 720),
        )
        self._configure_follow_camera_lens(
            UsdGeom.Camera(stage.GetPrimAtPath(self._headless_capture_camera_path))
        )

        root_pos = self._ensure_tensor(
            self._articulation.get_world_pose()[0]
        ).cpu().numpy()
        set_camera_view(
            eye=root_pos + self._FOLLOW_CAMERA_OFFSET,
            target=root_pos + self._FOLLOW_CAMERA_TARGET_OFFSET,
            camera_prim_path=self._headless_capture_camera_path,
        )
        camera.initialize()

        self._headless_capture_camera = camera
        self._headless_capture_ready = False
        return camera

    def _update_headless_capture_camera_pose(self) -> None:
        """Keep the dedicated headless capture camera aligned to the humanoid root."""
        from omni.isaac.core.utils.viewports import set_camera_view

        root_pos = self._ensure_tensor(
            self._articulation.get_world_pose()[0]
        ).cpu().numpy()
        set_camera_view(
            eye=root_pos + self._FOLLOW_CAMERA_OFFSET,
            target=root_pos + self._FOLLOW_CAMERA_TARGET_OFFSET,
            camera_prim_path=self._headless_capture_camera_path,
        )

    def _capture_headless_follow_camera_rgba(self):
        """Render the current headless frame without advancing physics."""
        self.prepare_headless_capture()
        self._update_headless_capture_camera_pose()
        self._world.render()

        rgba = self._headless_capture_camera.get_rgba()
        if rgba is None or getattr(rgba, "size", 0) == 0:
            raise RuntimeError("Headless capture camera returned an empty frame")
        return rgba

    def render(self) -> None:
        """Update the interactive camera before delegating to ProtoMotions rendering."""
        if not self.headless:
            if not hasattr(self, "_perspective_view"):
                from human_motion_isaacsim.protomotions_path import ensure_protomotions_importable

                ensure_protomotions_importable()
                from protomotions.simulator.isaaclab.utils.perspective_viewer import (
                    PerspectiveViewer,
                )
                from pxr import UsdGeom

                self._perspective_view = PerspectiveViewer()
                self._configure_follow_camera_lens(
                    UsdGeom.Camera(self._world.stage.GetPrimAtPath("/OmniverseKit_Persp"))
                )
                self._init_camera()
            else:
                self._update_camera()
        super().render()

    def _init_camera(self) -> None:
        """Initialize the active viewport to a wider follow-camera view."""
        root_state = self._get_simulator_root_state()
        char_root_pos = root_state.root_pos[self._camera_target["env"]].detach().cpu().numpy()
        self._cam_prev_char_pos = char_root_pos
        self._perspective_view.set_camera_view(
            char_root_pos + self._FOLLOW_CAMERA_OFFSET,
            char_root_pos + self._FOLLOW_CAMERA_TARGET_OFFSET,
        )

    def _update_camera(self) -> None:
        """Keep the interactive camera tracking the humanoid root."""
        if not hasattr(self, "_cam_prev_char_pos"):
            self._init_camera()
            return

        root_state = self._get_simulator_root_state()
        char_root_pos = root_state.root_pos[self._camera_target["env"]].detach().cpu().numpy()
        cam_pos = np.array(self._perspective_view.get_camera_state())
        cam_delta = cam_pos - self._cam_prev_char_pos
        self._perspective_view.set_camera_view(
            char_root_pos + cam_delta,
            char_root_pos + self._FOLLOW_CAMERA_TARGET_OFFSET,
        )
        self._cam_prev_char_pos[:] = char_root_pos

    def _update_simulator_markers(
        self, markers_state: Optional[Dict] = None
    ) -> None:
        """Update visual marker prims from ProtoMotions marker state."""
        if not markers_state:
            return

        marker_groups = getattr(self, "_marker_prim_groups", {})
        for marker_name, state in markers_state.items():
            marker_prims = marker_groups.get(marker_name)
            if not marker_prims or state.translation.numel() == 0:
                continue

            positions = state.translation.detach().cpu().numpy().reshape(-1, 3)
            orientations = state.orientation.detach().cpu().numpy().reshape(-1, 4)

            for prim, position, orientation in zip(marker_prims, positions, orientations):
                if np.linalg.norm(orientation) < 1e-6:
                    orientation = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
                prim.set_world_pose(position=position, orientation=orientation)

    def _configure_articulation_drives(self) -> None:
        """Apply ProtoMotions drive mode, gains, and effort limits to the articulation."""
        control_info = getattr(self.robot_config.control, "control_info", None)
        if control_info is None:
            return

        dof_names = list(self._view.dof_names)
        num_dof = len(dof_names)
        kps = np.zeros((1, num_dof), dtype=np.float32)
        kds = np.zeros((1, num_dof), dtype=np.float32)
        max_efforts = np.zeros((1, num_dof), dtype=np.float32)

        control_type = getattr(self, "control_type", self.robot_config.control.control_type)
        use_builtin_pd = control_type == ControlType.BUILT_IN_PD
        for idx, dof_name in enumerate(dof_names):
            dof_info = control_info[dof_name]
            if use_builtin_pd:
                kps[0, idx] = float(dof_info.stiffness)
                kds[0, idx] = float(dof_info.damping)
            if dof_info.effort_limit is not None:
                max_efforts[0, idx] = float(dof_info.effort_limit)

        self._view.switch_control_mode(mode="position" if use_builtin_pd else "effort")
        self._view.set_gains(kps=kps, kds=kds)
        self._view.set_max_efforts(values=max_efforts)

    def _configure_follow_camera_lens(self, camera) -> None:
        """Configure a wide-angle lens so follow cameras show the full scene context."""
        from pxr import Gf

        camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 10000.0))
        camera.GetFocalLengthAttr().Set(self._FOLLOW_CAMERA_FOCAL_LENGTH)
        camera.GetHorizontalApertureAttr().Set(self._FOLLOW_CAMERA_HORIZONTAL_APERTURE)
        camera.GetVerticalApertureAttr().Set(self._FOLLOW_CAMERA_VERTICAL_APERTURE)

    def _build_visualization_markers(self, visualization_markers: Optional[Dict]) -> None:
        """Create simple sphere markers for the non-headless ProtoMotions visualizations."""
        self._marker_prim_groups = {}
        if not visualization_markers:
            return

        from isaacsim.core.api.objects import VisualSphere

        size_to_radius = {
            "tiny": 0.007,
            "small": 0.01,
            "regular": 0.05,
        }

        for marker_name, marker_cfg in visualization_markers.items():
            if marker_cfg.type != "sphere":
                continue

            color = np.array(marker_cfg.color, dtype=np.float32)
            marker_prims = []
            for idx, marker in enumerate(marker_cfg.markers):
                radius = size_to_radius.get(marker.size, 0.05)
                prim = VisualSphere(
                    prim_path=f"/World/Visuals/{marker_name}_{idx:03d}",
                    name=f"{marker_name}_{idx:03d}",
                    color=color,
                    radius=radius,
                )
                prim.initialize()
                marker_prims.append(prim)
            self._marker_prim_groups[marker_name] = marker_prims
