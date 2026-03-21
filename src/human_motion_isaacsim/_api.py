from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

from human_motion_isaacsim._registry import list_models as registry_list_models
from human_motion_isaacsim.checkpoint import _resolve_tracker_assets
from human_motion_isaacsim._state import (
    PACKAGE_STATE,
    _build_body_rigid_view,
    _cache_body_rigid_view,
    _resolve_body_rigid_view,
    _resolve_scene_alignment_callback,
    _resolve_simulation_app,
)
from human_motion_isaacsim.motion_file import load_motion_metadata
from human_motion_isaacsim.result import MotionRunResult
from human_motion_isaacsim.simulator_adapter import SimulatorAdapter
from human_motion_isaacsim.viewport_capture import compile_video, frame_path_for_step


def _existing_frame_count(frames_dir: Path) -> int:
    """Return the number of existing PNG frames in a capture directory."""
    return len(list(frames_dir.glob("*.png")))


def _clear_existing_frames(frames_dir: Path) -> None:
    """Remove stale PNG frames when starting a fresh capture session."""
    for frame_path in frames_dir.glob("*.png"):
        frame_path.unlink()


def _init_reference_markers(env: Any, simulator: Any) -> None:
    """Create visualization markers for headless video capture when running without a display."""
    if not simulator.headless:
        return

    visualization_markers = env.control_manager.create_visualization_markers(headless=False)
    if visualization_markers:
        simulator._build_visualization_markers(visualization_markers)


def _update_reference_markers(
    env: Any,
    simulator: Any,
    *,
    enable_reference_markers: bool,
) -> None:
    """Refresh visualization marker positions for the current frame during headless capture."""
    if not simulator.headless or not enable_reference_markers:
        return

    original_headless = simulator.headless
    simulator.headless = False
    try:
        markers_state = env.control_manager.get_markers_state()
    finally:
        simulator.headless = original_headless

    simulator._update_simulator_markers(markers_state)


def _prepare_headless_capture_for_video(
    env: Any,
    simulator: Any,
    *,
    enable_reference_markers: bool,
) -> None:
    """Initialize the headless capture pipeline and optional reference markers before the motion loop."""
    if not simulator.headless:
        return

    if enable_reference_markers:
        _init_reference_markers(env, simulator)
    simulator.prepare_headless_capture()


def _plan_motion_max_steps(duration_seconds: float, simulator_config: Any) -> int:
    """Calculate the number of simulation steps needed for a motion of the given duration."""
    sim_cfg = getattr(simulator_config, "sim", None)
    sim_fps = getattr(sim_cfg, "fps", 30)
    decimation = max(1, int(getattr(sim_cfg, "decimation", 1)))
    return int(duration_seconds * sim_fps / decimation)


def _build_runtime_bundle(motion_path: Path) -> dict[str, Any]:
    """Assemble all ProtoMotions components (fabric, env, agent, simulator) for a single motion run."""
    if PACKAGE_STATE.tracker_assets is None:
        raise RuntimeError("human_motion_isaacsim.init() must be called before run().")

    from lightning.fabric import Fabric
    from protomotions.components.motion_lib import MotionLib
    from protomotions.components.scene_lib import SceneLib
    from protomotions.envs import component_manager as component_manager_module
    from protomotions.utils.component_builder import build_terrain_from_config
    from protomotions.utils.fabric_config import FabricConfig
    from protomotions.utils.hydra_replacement import get_class
    from protomotions.utils.inference_utils import apply_backward_compatibility_fixes

    tracker_assets = PACKAGE_STATE.tracker_assets
    motion_metadata = load_motion_metadata(motion_path)

    component_manager_module.TORCH_COMPILE_AVAILABLE = False

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

    robot_config = deepcopy(tracker_assets.robot_config)
    simulator_config = deepcopy(tracker_assets.simulator_config)
    terrain_config = deepcopy(tracker_assets.terrain_config)
    motion_lib_config = deepcopy(tracker_assets.motion_lib_config)
    env_config = deepcopy(tracker_assets.env_config)
    agent_config = deepcopy(tracker_assets.agent_config)

    apply_backward_compatibility_fixes(robot_config, simulator_config, env_config)

    simulator_config.num_envs = 1
    simulator_config.headless = PACKAGE_STATE.headless
    motion_lib_config.motion_file = str(motion_path.resolve())

    max_steps = _plan_motion_max_steps(motion_metadata.duration_seconds, simulator_config)
    if hasattr(env_config, "max_episode_length"):
        env_config.max_episode_length = max(env_config.max_episode_length, max_steps + 100)

    terrain = build_terrain_from_config(terrain_config, num_envs=1, device=fabric.device)
    scene_lib = SceneLib.empty(num_envs=1, device=str(fabric.device), terrain=terrain)
    motion_lib = MotionLib(motion_lib_config, device=str(fabric.device))

    simulator = SimulatorAdapter(
        world=PACKAGE_STATE.world,
        articulation=PACKAGE_STATE.articulation,
        simulation_app=PACKAGE_STATE.simulation_app,
        config=simulator_config,
        robot_config=robot_config,
        scene_lib=scene_lib,
        device=fabric.device,
        terrain=terrain,
    )
    simulator._body_rigid_view = PACKAGE_STATE.body_rigid_view

    EnvClass = get_class(env_config._target_)
    env = EnvClass(
        config=env_config,
        robot_config=robot_config,
        device=fabric.device,
        terrain=terrain,
        scene_lib=scene_lib,
        motion_lib=motion_lib,
        simulator=simulator,
    )

    AgentClass = get_class(agent_config._target_)
    agent = AgentClass(
        config=agent_config,
        env=env,
        fabric=fabric,
        root_dir=tracker_assets.checkpoint_path.resolve().parent,
    )
    agent.setup()
    agent.load(str(tracker_assets.checkpoint_path.resolve()), load_env=False)

    return {
        "motion_metadata": motion_metadata,
        "max_steps": max_steps,
        "env": env,
        "agent": agent,
        "simulator": simulator,
        "helpers": [simulator, env, agent, scene_lib, motion_lib, terrain],
    }


def _resolve_current_root_position(simulator: Any) -> tuple[float, float, float] | None:
    """Read the current humanoid root position from the simulator if available."""
    get_root_state = getattr(simulator, "_get_simulator_root_state", None)
    if not callable(get_root_state):
        get_root_state = getattr(simulator, "get_root_state", None)
    if not callable(get_root_state):
        return None

    root_state = get_root_state()
    root_pos = getattr(root_state, "root_pos", None)
    if root_pos is None:
        return None

    root_pos_tensor = root_pos[0]
    if hasattr(root_pos_tensor, "detach"):
        root_pos_tensor = root_pos_tensor.detach().cpu()
    root_pos_values = tuple(float(value) for value in root_pos_tensor.tolist())
    if len(root_pos_values) != 3:
        return None
    return root_pos_values


def _build_reset_env_ids(env: Any):
    """Create an env-id tensor for reset/alignment helpers."""
    import torch

    return torch.arange(env.num_envs, dtype=torch.long, device=env.device)


def _apply_carried_position_to_initial_reset(
    env: Any,
    *,
    carried_root_position: tuple[float, float, float] | None,
):
    """Reset the env for the first step, reusing the previous clip's world position when present."""
    if carried_root_position is None:
        obs, _ = env.reset(None)
        return obs

    align_motion = getattr(env, "align_motion_with_humanoid", None)
    if not callable(align_motion):
        obs, _ = env.reset(None)
        return obs

    import torch

    env_ids = _build_reset_env_ids(env)
    env.reset(env_ids)
    target_root_position = torch.tensor(
        [carried_root_position],
        dtype=torch.float32,
        device=env.device,
    ).repeat(env.num_envs, 1)
    align_motion(env_ids, target_root_position)
    obs, _ = env.reset(env_ids, disable_motion_resample=True)
    return obs


def _reset_humanoid_to_neutral_standing_pose(
    env: Any,
    simulator: Any,
    *,
    carried_root_position: tuple[float, float, float],
) -> None:
    """Reset the humanoid to the default standing pose at the carried XY position."""
    default_reset_state = getattr(env, "default_reset_state", None)
    if default_reset_state is None:
        get_default_reset_state = getattr(simulator, "get_default_robot_reset_state", None)
        if not callable(get_default_reset_state):
            return
        default_reset_state = get_default_reset_state()
    else:
        clone = getattr(default_reset_state, "clone", None)
        if callable(clone):
            default_reset_state = clone()

    root_pos = default_reset_state.root_pos.clone()
    root_pos[:, 0] = carried_root_position[0]
    root_pos[:, 1] = carried_root_position[1]
    default_reset_state.root_pos = root_pos
    simulator.reset_envs(default_reset_state, None, _build_reset_env_ids(env))


def _teardown_run_helpers(helpers: list[Any]) -> None:
    """Call the first available teardown method on each helper, re-raising the first error encountered."""
    first_error: Exception | None = None
    for helper in helpers:
        for method_name in ("teardown", "destroy", "shutdown", "close"):
            method = getattr(helper, method_name, None)
            if callable(method):
                try:
                    method()
                except Exception as exc:
                    if first_error is None:
                        first_error = exc
                break
    if first_error is not None:
        raise first_error


def init(
    model: str,
    world: Any,
    articulation: Any,
    headless: bool = True,
    reference_markers: bool = False,
) -> None:
    """Initialize the package state with a model, world, and articulation for subsequent motion runs."""
    tracker_assets = _resolve_tracker_assets(model)
    simulation_app = _resolve_simulation_app(world, articulation)
    body_rigid_view = _resolve_body_rigid_view(world, articulation)
    built_body_rigid_view = False
    if body_rigid_view is None and getattr(tracker_assets, "robot_config", None) is not None:
        body_rigid_view = _build_body_rigid_view(world, articulation, tracker_assets)
        _cache_body_rigid_view(world, articulation, body_rigid_view)
        built_body_rigid_view = body_rigid_view is not None

    if built_body_rigid_view:
        reset = getattr(world, "reset", None)
        if callable(reset):
            reset()

    PACKAGE_STATE.teardown()
    PACKAGE_STATE.model_name = model
    PACKAGE_STATE.tracker_assets = tracker_assets
    PACKAGE_STATE.world = world
    PACKAGE_STATE.articulation = articulation
    PACKAGE_STATE.body_rigid_view = body_rigid_view
    PACKAGE_STATE.headless = headless
    PACKAGE_STATE.reference_markers = reference_markers
    PACKAGE_STATE.simulation_app = simulation_app


def run(
    motion_file: str | Path,
    video_output: str | Path | None = None,
) -> MotionRunResult:
    """Execute a motion-tracking loop and optionally produce a video, returning a MotionRunResult."""
    if PACKAGE_STATE.model_name is None:
        raise RuntimeError("human_motion_isaacsim.init() must be called before run().")

    motion_path = Path(motion_file)
    bundle = _build_runtime_bundle(motion_path)
    motion_metadata = bundle["motion_metadata"]
    max_steps = bundle["max_steps"]
    env = bundle["env"]
    agent = bundle["agent"]
    simulator = bundle["simulator"]
    helpers = bundle.get("helpers", [])

    frames_dir: Path | None = None
    video_path: Path | None = None
    if video_output is not None:
        video_path = Path(video_output)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        frames_dir = video_path.with_suffix("") / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        if PACKAGE_STATE.carried_root_position is None:
            _clear_existing_frames(frames_dir)
        _prepare_headless_capture_for_video(
            env,
            simulator,
            enable_reference_markers=PACKAGE_STATE.reference_markers,
        )

    done_indices = None
    agent.eval()
    try:
        initial_obs = None
        if max_steps > 0:
            initial_obs = _apply_carried_position_to_initial_reset(
                env,
                carried_root_position=PACKAGE_STATE.carried_root_position,
            )

        for step in range(max_steps):
            if step == 0:
                obs = initial_obs
                did_reset = True
            else:
                obs, _ = env.reset(done_indices)
                did_reset = done_indices.numel() > 0

            if did_reset:
                scene_alignment_callback = _resolve_scene_alignment_callback(
                    PACKAGE_STATE.world,
                    PACKAGE_STATE.articulation,
                )
                if scene_alignment_callback is not None:
                    scene_alignment_callback(simulator)
            obs = agent.add_agent_info_to_obs(obs)
            obs_td = agent.obs_dict_to_tensordict(obs)
            model_outs = agent.model(obs_td)
            actions = model_outs.get("mean_action", model_outs.get("action"))
            _, _, dones, _, _ = env.step(actions)
            if frames_dir is not None:
                _update_reference_markers(
                    env,
                    simulator,
                    enable_reference_markers=PACKAGE_STATE.reference_markers,
                )
                frame_path = frame_path_for_step(
                    frames_dir,
                    _existing_frame_count(frames_dir),
                )
                simulator._write_viewport_to_file(str(frame_path))
            done_indices = dones.nonzero(as_tuple=False).squeeze(-1)

        if max_steps > 0:
            carried_root_position = _resolve_current_root_position(simulator)
            if carried_root_position is not None:
                _reset_humanoid_to_neutral_standing_pose(
                    env,
                    simulator,
                    carried_root_position=carried_root_position,
                )
                PACKAGE_STATE.carried_root_position = carried_root_position

        if frames_dir is not None and video_path is not None:
            frame_paths = sorted(frames_dir.glob("*.png"))
            if frame_paths:
                compile_video(frame_paths, video_path, fps=30)

        return MotionRunResult(
            success=True,
            motion_file=motion_path,
            video_output=video_path,
            num_steps=max_steps,
            duration_seconds=motion_metadata.duration_seconds,
        )
    finally:
        _teardown_run_helpers(helpers)


def list_models() -> list[dict[str, str]]:
    """Return the list of available models from the registry."""
    return registry_list_models()
