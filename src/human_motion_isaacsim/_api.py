from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import logging
from pathlib import Path
from typing import Any

from human_motion_isaacsim._registry import list_models as registry_list_models
from human_motion_isaacsim.checkpoint import _resolve_tracker_assets
from human_motion_isaacsim._state import (
    PACKAGE_STATE,
    _build_body_rigid_view,
    _cache_body_rigid_view,
    _resolve_articulation_prim_path,
    _resolve_body_rigid_view,
    _resolve_simulation_app,
)
from human_motion_isaacsim.motion_file import load_motion_metadata
from human_motion_isaacsim.motion_os_inputs import resolve_motion_input
from human_motion_isaacsim.result import MotionRunResult
from human_motion_isaacsim.simulator_adapter import SimulatorAdapter
from human_motion_isaacsim.viewport_capture import compile_video, frame_path_for_step


LOGGER = logging.getLogger(__name__)


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

    return _extract_position_tuple(root_pos[0])


def _extract_position_tuple(value: Any) -> tuple[float, float, float] | None:
    """Best-effort conversion of tensor/vector-like values into a 3D position tuple."""
    if value is None:
        return None

    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()

    try:
        values = tuple(float(component) for component in value)
    except TypeError:
        return None

    if len(values) != 3:
        return None
    return values


def _resolve_articulation_root_position(articulation: Any) -> tuple[float, float, float] | None:
    """Read the current humanoid root position directly from the articulation if available."""
    get_world_pose = getattr(articulation, "get_world_pose", None)
    if not callable(get_world_pose):
        return None

    world_pose = get_world_pose()
    if not isinstance(world_pose, tuple) or len(world_pose) < 1:
        return None

    return _extract_position_tuple(world_pose[0])


def _resolve_scene_reference_positions(
    world: Any,
    articulation: Any,
) -> dict[str, tuple[float, float, float]]:
    """Capture the absolute positions of non-humanoid scene prims with authored translations."""
    stage = getattr(world, "stage", None)
    traverse = getattr(stage, "Traverse", None)
    if stage is None or not callable(traverse):
        return {}

    articulation_prim_path = None
    try:
        articulation_prim_path = _resolve_articulation_prim_path(articulation)
    except Exception:
        articulation_prim_path = None

    scene_positions: dict[str, tuple[float, float, float]] = {}
    for prim in traverse():
        prim_path = prim.GetPath()
        prim_path_str = getattr(prim_path, "pathString", str(prim_path))
        if articulation_prim_path and (
            prim_path_str == articulation_prim_path
            or prim_path_str.startswith(f"{articulation_prim_path}/")
        ):
            continue

        translate_attr = prim.GetAttribute("xformOp:translate")
        is_valid = getattr(translate_attr, "IsValid", None)
        if not callable(is_valid) or not is_valid():
            continue

        position = _extract_position_tuple(translate_attr.Get())
        if position is not None:
            scene_positions[prim_path_str] = position

    return scene_positions


def _format_position(position: tuple[float, float, float] | None) -> str:
    """Format a 3D point consistently for boundary logs."""
    if position is None:
        return "None"
    return f"({position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f})"


def _format_position_delta(
    position: tuple[float, float, float],
    reference: tuple[float, float, float] | None,
) -> str:
    """Format a delta from a reference position for scene-invariant logs."""
    if reference is None:
        return "None"
    delta = tuple(position[index] - reference[index] for index in range(3))
    return _format_position(delta)


def _log_position_snapshot(
    boundary: str,
    *,
    humanoid_root_position: tuple[float, float, float] | None,
    world: Any,
    articulation: Any,
) -> None:
    """Log the humanoid root and absolute scene prim positions at a run boundary."""
    LOGGER.info(
        "%s humanoid_root=%s desired_next_run_root=%s",
        boundary,
        _format_position(humanoid_root_position),
        _format_position(PACKAGE_STATE.next_run_root_position),
    )

    scene_positions = _resolve_scene_reference_positions(world, articulation)
    for prim_path, position in sorted(scene_positions.items()):
        LOGGER.info(
            "%s scene_prim=%s position=%s delta=%s",
            boundary,
            prim_path,
            _format_position(position),
            _format_position_delta(
                position,
                PACKAGE_STATE.scene_reference_positions.get(prim_path),
            ),
        )


def _build_reset_env_ids(env: Any):
    """Create an env-id tensor for reset/alignment helpers."""
    import torch

    return torch.arange(env.num_envs, dtype=torch.long, device=env.device)


def _normalize_env_ids(env: Any, env_ids: Any):
    """Normalize env-id inputs to a torch tensor on the env device."""
    import torch

    if env_ids is None:
        return _build_reset_env_ids(env)
    if isinstance(env_ids, list):
        return torch.tensor(env_ids, dtype=torch.long, device=env.device)
    if hasattr(env_ids, "to"):
        return env_ids.to(device=env.device, dtype=torch.long)
    return torch.as_tensor(env_ids, dtype=torch.long, device=env.device)


def _apply_next_run_root_position_to_initial_reset(
    env: Any,
    *,
    next_run_root_position: tuple[float, float, float] | None,
):
    """Reset the env for the first step, anchoring the clip to the desired world root."""
    if next_run_root_position is None:
        obs, _ = env.reset(None)
        return obs

    update_respawn_root_offset = getattr(env, "update_respawn_root_offset_by_env_ids", None)
    if not callable(update_respawn_root_offset):
        obs, _ = env.reset(None)
        return obs

    import torch

    target_root_position = torch.tensor(
        [next_run_root_position],
        dtype=torch.float32,
        device=env.device,
    ).repeat(env.num_envs, 1)
    terrain = getattr(env, "terrain", None)
    original_sample_valid_locations = getattr(terrain, "sample_valid_locations", None)

    def wrapped_update_respawn_root_offset_by_env_ids(
        env_ids: Any,
        ref_state: Any = None,
        sample_flat: bool = False,
    ):
        env_ids_tensor = _normalize_env_ids(env, env_ids)
        target_root_xy = target_root_position[env_ids_tensor, :2]

        if ref_state is None:
            return update_respawn_root_offset(env_ids_tensor, ref_state=ref_state, sample_flat=sample_flat)

        if callable(original_sample_valid_locations):
            def sample_valid_locations(num_envs: int, sample_flat: bool = False):
                if num_envs == len(env_ids_tensor):
                    return target_root_xy.clone()
                return original_sample_valid_locations(
                    num_envs=num_envs,
                    sample_flat=sample_flat,
                )

            terrain.sample_valid_locations = sample_valid_locations

        try:
            result = update_respawn_root_offset(
                env_ids_tensor,
                ref_state=ref_state,
                sample_flat=sample_flat,
            )
        finally:
            if callable(original_sample_valid_locations):
                terrain.sample_valid_locations = original_sample_valid_locations

        if hasattr(env, "respawn_root_offset"):
            env.respawn_root_offset[env_ids_tensor, :2] = (
                target_root_xy - ref_state.rigid_body_pos[:, 0, :2]
            )
        return result

    setattr(env, "update_respawn_root_offset_by_env_ids", wrapped_update_respawn_root_offset_by_env_ids)
    try:
        obs, _ = env.reset(None)
    finally:
        setattr(env, "update_respawn_root_offset_by_env_ids", update_respawn_root_offset)
        if callable(original_sample_valid_locations):
            terrain.sample_valid_locations = original_sample_valid_locations
    return obs


def _reset_humanoid_to_neutral_standing_pose(
    env: Any,
    simulator: Any,
    *,
    root_position: tuple[float, float, float],
) -> tuple[float, float, float] | None:
    """Reset the humanoid to the default standing pose at the carried XY position."""
    default_reset_state = getattr(env, "default_reset_state", None)
    if default_reset_state is None:
        get_default_reset_state = getattr(simulator, "get_default_robot_reset_state", None)
        if not callable(get_default_reset_state):
            return None
        default_reset_state = get_default_reset_state()
    else:
        clone = getattr(default_reset_state, "clone", None)
        if callable(clone):
            default_reset_state = clone()

    root_pos = default_reset_state.root_pos.clone()
    root_pos[:, 0] = root_position[0]
    root_pos[:, 1] = root_position[1]
    default_reset_state.root_pos = root_pos
    simulator.reset_envs(default_reset_state, None, _build_reset_env_ids(env))
    return _extract_position_tuple(root_pos[0])


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
    initial_root_position = _resolve_articulation_root_position(articulation)
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
        if initial_root_position is None:
            initial_root_position = _resolve_articulation_root_position(articulation)

    PACKAGE_STATE.teardown()
    PACKAGE_STATE.model_name = model
    PACKAGE_STATE.tracker_assets = tracker_assets
    PACKAGE_STATE.world = world
    PACKAGE_STATE.articulation = articulation
    PACKAGE_STATE.body_rigid_view = body_rigid_view
    PACKAGE_STATE.headless = headless
    PACKAGE_STATE.reference_markers = reference_markers
    PACKAGE_STATE.simulation_app = simulation_app
    PACKAGE_STATE.initial_root_position = initial_root_position
    PACKAGE_STATE.next_run_root_position = initial_root_position
    PACKAGE_STATE.scene_reference_positions = _resolve_scene_reference_positions(
        world,
        articulation,
    )

    _log_position_snapshot(
        "init",
        humanoid_root_position=initial_root_position,
        world=world,
        articulation=articulation,
    )


def run(
    motion_file: str | Path | None = None,
    video_output: str | Path | None = None,
    *,
    manifest_path: str | Path | None = None,
    representation: str = "proto_motion",
    staging_dir: str | Path | None = None,
) -> MotionRunResult:
    """Execute a motion-tracking loop and optionally produce a video, returning a MotionRunResult."""
    if PACKAGE_STATE.model_name is None:
        raise RuntimeError("human_motion_isaacsim.init() must be called before run().")

    resolved_input = resolve_motion_input(
        motion_file=motion_file,
        manifest_path=manifest_path,
        representation=representation,
        staging_dir=staging_dir,
    )
    motion_path = resolved_input.motion_file
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
        if PACKAGE_STATE.completed_run_count == 0:
            _clear_existing_frames(frames_dir)
        _prepare_headless_capture_for_video(
            env,
            simulator,
            enable_reference_markers=PACKAGE_STATE.reference_markers,
        )

    done_indices = None
    agent.eval()
    try:
        _log_position_snapshot(
            "run.start",
            humanoid_root_position=_resolve_articulation_root_position(PACKAGE_STATE.articulation),
            world=PACKAGE_STATE.world,
            articulation=PACKAGE_STATE.articulation,
        )

        initial_obs = None
        executed_steps = 0
        if max_steps > 0:
            initial_obs = _apply_next_run_root_position_to_initial_reset(
                env,
                next_run_root_position=PACKAGE_STATE.next_run_root_position,
            )
            _log_position_snapshot(
                "run.post_reset",
                humanoid_root_position=_resolve_current_root_position(simulator),
                world=PACKAGE_STATE.world,
                articulation=PACKAGE_STATE.articulation,
            )

        for step in range(max_steps):
            if step == 0:
                obs = initial_obs
            else:
                obs, _ = env.reset(done_indices)
                if done_indices.numel() > 0:
                    _log_position_snapshot(
                        f"run.reset.step_{step}",
                        humanoid_root_position=_resolve_current_root_position(simulator),
                        world=PACKAGE_STATE.world,
                        articulation=PACKAGE_STATE.articulation,
                    )

            if step == 0:
                _log_position_snapshot(
                    "run.pre_step_0",
                    humanoid_root_position=_resolve_current_root_position(simulator),
                    world=PACKAGE_STATE.world,
                    articulation=PACKAGE_STATE.articulation,
                )
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
            executed_steps = step + 1
            if done_indices.numel() == getattr(env, "num_envs", len(dones)):
                break

        final_root_position = None
        if executed_steps > 0:
            final_root_position = _resolve_current_root_position(simulator)
            _log_position_snapshot(
                "run.final",
                humanoid_root_position=final_root_position,
                world=PACKAGE_STATE.world,
                articulation=PACKAGE_STATE.articulation,
            )
            if final_root_position is not None:
                PACKAGE_STATE.next_run_root_position = final_root_position
                neutral_root_position = _reset_humanoid_to_neutral_standing_pose(
                    env,
                    simulator,
                    root_position=final_root_position,
                )
                _log_position_snapshot(
                    "run.neutral",
                    humanoid_root_position=neutral_root_position,
                    world=PACKAGE_STATE.world,
                    articulation=PACKAGE_STATE.articulation,
                )

        if frames_dir is not None and video_path is not None:
            frame_paths = sorted(frames_dir.glob("*.png"))
            if frame_paths:
                compile_video(frame_paths, video_path, fps=30)

        PACKAGE_STATE.completed_run_count += 1

        return MotionRunResult(
            success=True,
            motion_file=motion_path,
            video_output=video_path,
            num_steps=executed_steps,
            duration_seconds=motion_metadata.duration_seconds,
        )
    finally:
        _teardown_run_helpers(helpers)


def list_models() -> list[dict[str, str]]:
    """Return the list of available models from the registry."""
    return registry_list_models()
