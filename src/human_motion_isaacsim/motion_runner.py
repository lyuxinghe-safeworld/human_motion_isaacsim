from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from human_motion_isaacsim.binding import bind_fixed_humanoid
from human_motion_isaacsim.checkpoint import load_tracker_assets
from human_motion_isaacsim.motion_file import load_motion_metadata
from human_motion_isaacsim.protomotions_path import ensure_protomotions_importable


class MotionController:
    def __init__(
        self,
        *,
        humanoid_prim_path: str,
        checkpoint_path: str | Path,
        lookup_articulation: Callable[[str], Any],
        bind_humanoid: Callable[..., Any] = bind_fixed_humanoid,
        load_assets: Callable[[str | Path], Any] = load_tracker_assets,
        motion_runner: Callable[[Any, Any, str | Path | None], Any] | None = None,
        restore_rest_pose: Callable[[Any], None] | None = None,
    ) -> None:
        self.humanoid_prim_path = humanoid_prim_path
        self.checkpoint_path = Path(checkpoint_path)
        self.lookup_articulation = lookup_articulation
        self.tracker_assets = load_assets(self.checkpoint_path)
        self.bound_humanoid = bind_humanoid(
            humanoid_prim_path,
            lookup_articulation=lookup_articulation,
            tracker_assets=self.tracker_assets,
        )
        self._motion_runner = motion_runner
        self._restore_rest_pose = restore_rest_pose
        self._busy = False

    def run_motion(self, motion_file: str, video_output: str | None = None):
        if self._busy:
            raise RuntimeError("Motion execution already in progress")

        # NOTE(v1): run_motion is intentionally blocking and exclusive. While a
        # motion is active, this controller owns env stepping and assumes no
        # external process is mutating the humanoid or stage state.
        metadata = load_motion_metadata(motion_file)
        self._busy = True
        try:
            if self._motion_runner is None:
                raise NotImplementedError(
                    "The controller shell is initialized, but the execution loop is not wired yet."
                )
            result = self._motion_runner(self, metadata, video_output)
        except Exception:
            if self._restore_rest_pose is not None:
                self._restore_rest_pose(self)
            raise
        else:
            if self._restore_rest_pose is not None:
                self._restore_rest_pose(self)
            return result
        finally:
            self._busy = False


@dataclass(slots=True)
class MotionRunner:
    tracker_assets: Any
    simulator_name: str = "isaaclab"

    @classmethod
    def from_checkpoint_path(
        cls,
        checkpoint_path: str | Path,
        *,
        simulator_name: str = "isaaclab",
    ) -> "MotionRunner":
        from human_motion_isaacsim.checkpoint import load_tracker_assets as _load

        return cls(
            tracker_assets=_load(checkpoint_path),
            simulator_name=simulator_name,
        )

    @property
    def env_target(self) -> str | None:
        return getattr(self.tracker_assets.env_config, "_target_", None)

    @property
    def agent_target(self) -> str | None:
        return getattr(self.tracker_assets.agent_config, "_target_", None)

    def plan_num_steps(self, motion_metadata, *, sim_fps: int = 30) -> int:
        clip_seconds = motion_metadata.duration_seconds
        sim_cfg = getattr(self.tracker_assets, "simulator_config", None)
        sim = getattr(sim_cfg, "sim", None)
        fps = getattr(sim, "fps", sim_fps)
        decimation = max(1, int(getattr(sim, "decimation", 1)))
        return int(clip_seconds * fps / decimation)

    def build_standalone_runner(
        self,
        *,
        checkpoint_path: str | Path,
        motion_file: str | Path,
        max_steps: int | None = None,
        headless: bool = False,
        enable_cameras: bool = False,
        num_envs: int = 1,
    ) -> dict[str, Any]:
        from copy import deepcopy
        from dataclasses import asdict

        ensure_protomotions_importable()
        from lightning.fabric import Fabric
        from protomotions.envs import component_manager as component_manager_module
        from protomotions.utils.component_builder import build_all_components
        from protomotions.utils.fabric_config import FabricConfig
        from protomotions.utils.hydra_replacement import get_class
        from protomotions.utils.inference_utils import apply_backward_compatibility_fixes
        from protomotions.utils.simulator_imports import import_simulator_before_torch
        from protomotions.simulator.base_simulator.utils import convert_friction_for_simulator

        AppLauncher = import_simulator_before_torch(self.simulator_name)

        import torch

        robot_config = deepcopy(self.tracker_assets.robot_config)
        simulator_config = deepcopy(self.tracker_assets.simulator_config)
        terrain_config = deepcopy(self.tracker_assets.terrain_config)
        scene_lib_config = deepcopy(self.tracker_assets.scene_lib_config)
        motion_lib_config = deepcopy(self.tracker_assets.motion_lib_config)
        env_config = deepcopy(self.tracker_assets.env_config)
        agent_config = deepcopy(self.tracker_assets.agent_config)

        apply_backward_compatibility_fixes(robot_config, simulator_config, env_config)
        simulator_config.num_envs = num_envs
        simulator_config.headless = headless
        motion_lib_config.motion_file = str(Path(motion_file).resolve())
        if max_steps is not None and hasattr(env_config, "max_episode_length"):
            env_config.max_episode_length = max(env_config.max_episode_length, max_steps + 100)
        # NOTE(v1): manager-level torch.compile adds a long first-run warmup and
        # can stall this function-triggered controller path. Disable it here so
        # motion execution starts predictably from the first request.
        component_manager_module.TORCH_COMPILE_AVAILABLE = False

        # NOTE(v1): this controller runs one blocking inference stream, so we
        # keep Fabric in single-device mode instead of DDP. In this pip-based
        # Isaac Sim stack, DDP Fabric initialization can destabilize Kit startup.
        fabric = Fabric(
            **asdict(
                FabricConfig(
                    devices=1,
                    num_nodes=1,
                    strategy="auto",
                    loggers=[],
                    callbacks=[],
                )
            )
        )
        fabric.launch()

        simulator_extra_params = {}
        app_launcher = None
        if self.simulator_name == "isaaclab":
            app_launcher = AppLauncher(
                {
                    "headless": headless,
                    "device": str(fabric.device),
                    "enable_cameras": enable_cameras,
                }
            )
            simulator_extra_params["simulation_app"] = app_launcher.app

        terrain_config, simulator_config = convert_friction_for_simulator(
            terrain_config,
            simulator_config,
        )

        components = build_all_components(
            terrain_config=terrain_config,
            scene_lib_config=scene_lib_config,
            motion_lib_config=motion_lib_config,
            simulator_config=simulator_config,
            robot_config=robot_config,
            device=fabric.device,
            **simulator_extra_params,
        )

        EnvClass = get_class(env_config._target_)
        env = EnvClass(
            config=env_config,
            robot_config=robot_config,
            device=fabric.device,
            terrain=components["terrain"],
            scene_lib=components["scene_lib"],
            motion_lib=components["motion_lib"],
            simulator=components["simulator"],
        )

        AgentClass = get_class(agent_config._target_)
        agent = AgentClass(
            config=agent_config,
            env=env,
            fabric=fabric,
            root_dir=Path(checkpoint_path).resolve().parent,
        )
        agent.setup()
        agent.load(str(Path(checkpoint_path).resolve()), load_env=False)

        return {
            "app_launcher": app_launcher,
            "simulation_app": app_launcher.app if app_launcher is not None else None,
            "fabric": fabric,
            "env": env,
            "agent": agent,
            "simulator": components["simulator"],
        }

    def run_standalone_motion(
        self,
        *,
        checkpoint_path: str | Path,
        motion_file: str | Path,
        video_output: str | Path | None = None,
        headless: bool = False,
        num_envs: int = 1,
    ):
        ensure_protomotions_importable()
        from protomotions.utils.simulator_imports import import_simulator_before_torch

        import_simulator_before_torch(self.simulator_name)

        from human_motion_isaacsim.motion_file import load_motion_metadata as _load_meta
        from human_motion_isaacsim.viewport_capture import compile_video, frame_path_for_step
        from human_motion_isaacsim.result import MotionRunResult

        motion_metadata = _load_meta(motion_file)
        max_steps = self.plan_num_steps(motion_metadata)
        bundle = self.build_standalone_runner(
            checkpoint_path=checkpoint_path,
            motion_file=motion_file,
            max_steps=max_steps,
            headless=headless,
            # Isaac Lab only needs camera-enabled rendering when running
            # headless. Under a VNC-backed monitor session, the regular GUI kit
            # provides the active viewport that ProtoMotions tracks and records.
            enable_cameras=headless,
            num_envs=num_envs,
        )

        app_launcher = bundle["app_launcher"]
        env = bundle["env"]
        agent = bundle["agent"]
        simulator = bundle["simulator"]
        simulation_app = bundle["simulation_app"]

        video_path = Path(video_output) if video_output is not None else Path(motion_file).with_suffix(".mp4")
        video_path.parent.mkdir(parents=True, exist_ok=True)
        frames_dir = video_path.with_suffix("") / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        # NOTE(v1): this standalone path owns the full stepping loop for the
        # duration of the motion and assumes the Isaac Sim world is otherwise
        # quiescent. The future stage-bound adapter can relax that constraint.
        agent.eval()
        done_indices = None
        try:
            for step in range(max_steps):
                obs, _ = env.reset(done_indices)
                obs = agent.add_agent_info_to_obs(obs)
                obs_td = agent.obs_dict_to_tensordict(obs)
                model_outs = agent.model(obs_td)
                actions = model_outs.get("mean_action", model_outs.get("action"))
                obs, rewards, dones, terminated, extras = env.step(actions)
                frame_path = frame_path_for_step(frames_dir, step)
                # Match the exact ProtoMotions / hymotion_isaaclab capture path.
                # Running extra Kit update ticks here can advance physics beyond
                # the controller step and desync the humanoid from the markers.
                simulator._write_viewport_to_file(str(frame_path))
                done_indices = dones.nonzero(as_tuple=False).squeeze(-1)

            frame_paths = sorted(frames_dir.glob("*.png"))
            compile_video(frame_paths, video_path, fps=30)
            return MotionRunResult(
                success=True,
                motion_file=Path(motion_file),
                video_output=video_path,
                num_steps=max_steps,
                duration_seconds=motion_metadata.duration_seconds,
            )
        finally:
            try:
                # NOTE(v1): the standalone smoke path owns the Kit app lifecycle.
                # Closing through SimulationApp avoids the slower default shutdown
                # path that the simulator wrapper uses for interactive sessions.
                # We use eager cleanup here because this wrapper is a one-shot
                # batch runner, not the future long-lived stage-bound controller.
                if simulation_app is not None and hasattr(simulation_app, "close"):
                    simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
                else:
                    simulator.close()
            except Exception:
                pass
