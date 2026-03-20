from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

from human_motion_isaacsim._registry import list_models as registry_list_models
from human_motion_isaacsim.checkpoint import _resolve_tracker_assets
from human_motion_isaacsim._state import PACKAGE_STATE
from human_motion_isaacsim.motion_file import load_motion_metadata
from human_motion_isaacsim.result import MotionRunResult
from human_motion_isaacsim.simulator_adapter import SimulatorAdapter
from human_motion_isaacsim.viewport_capture import compile_video, frame_path_for_step


def _resolve_simulation_app(world: Any, articulation: Any) -> Any | None:
    for owner in (world, articulation):
        for attr_name in ("simulation_app", "_simulation_app", "app", "_app"):
            candidate = getattr(owner, attr_name, None)
            if candidate is not None:
                return candidate
    return None


def _resolve_articulation_prim_path(articulation: Any) -> str:
    for attr_name in ("prim_path", "_prim_path"):
        prim_path = getattr(articulation, attr_name, None)
        if prim_path:
            return str(prim_path)
    raise RuntimeError("Unable to determine articulation prim path for body view setup.")


def _resolve_body_rigid_view(world: Any, articulation: Any) -> Any | None:
    for owner in (articulation, world):
        for attr_name in ("body_rigid_view", "_body_rigid_view"):
            candidate = getattr(owner, attr_name, None)
            if candidate is not None:
                return candidate
    return None


def _cache_body_rigid_view(world: Any, articulation: Any, body_rigid_view: Any) -> None:
    if body_rigid_view is None:
        return

    for owner in (articulation, world):
        for attr_name in ("body_rigid_view", "_body_rigid_view"):
            try:
                setattr(owner, attr_name, body_rigid_view)
                return
            except Exception:
                continue


def _build_body_rigid_view(world: Any, articulation: Any, tracker_assets: Any) -> Any | None:
    robot_config = getattr(tracker_assets, "robot_config", None)
    kinematic_info = getattr(robot_config, "kinematic_info", None)
    body_names = getattr(kinematic_info, "body_names", None)
    if not body_names:
        return None

    from omni.isaac.core.prims import RigidPrimView

    prim_path = _resolve_articulation_prim_path(articulation)
    body_prim_paths = [f"{prim_path}/bodies/{name}" for name in body_names]
    view = RigidPrimView(
        prim_paths_expr=body_prim_paths,
        name=f"{getattr(articulation, 'name', 'humanoid')}_bodies",
    )

    scene = getattr(world, "scene", None)
    if scene is not None and hasattr(scene, "add"):
        added_view = scene.add(view)
        if added_view is not None:
            view = added_view

    initialize = getattr(view, "initialize", None)
    if callable(initialize):
        initialize()

    return view


def _resolve_scene_alignment_callback(world: Any, articulation: Any) -> Any | None:
    for owner in (world, articulation):
        for attr_name in ("scene_alignment_callback", "_scene_alignment_callback"):
            candidate = getattr(owner, attr_name, None)
            if callable(candidate):
                return candidate
    return None


def _enable_reference_markers_for_capture(env: Any, simulator: Any) -> None:
    if not simulator.headless:
        return

    visualization_markers = env.control_manager.create_visualization_markers(headless=False)
    if visualization_markers:
        simulator._build_visualization_markers(visualization_markers)


def _update_reference_markers_for_capture(
    env: Any,
    simulator: Any,
    *,
    enable_reference_markers: bool,
) -> None:
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
    if not simulator.headless:
        return

    if enable_reference_markers:
        _enable_reference_markers_for_capture(env, simulator)
    simulator.prepare_headless_capture()


def _plan_motion_max_steps(duration_seconds: float, simulator_config: Any) -> int:
    sim_cfg = getattr(simulator_config, "sim", None)
    sim_fps = getattr(sim_cfg, "fps", 30)
    decimation = max(1, int(getattr(sim_cfg, "decimation", 1)))
    return int(duration_seconds * sim_fps / decimation)


def _build_runtime_bundle(motion_path: Path) -> dict[str, Any]:
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


def _teardown_run_helpers(helpers: list[Any]) -> None:
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
    tracker_assets = _resolve_tracker_assets(model)
    simulation_app = _resolve_simulation_app(world, articulation)
    body_rigid_view = _resolve_body_rigid_view(world, articulation)
    if body_rigid_view is None and getattr(tracker_assets, "robot_config", None) is not None:
        body_rigid_view = _build_body_rigid_view(world, articulation, tracker_assets)
        _cache_body_rigid_view(world, articulation, body_rigid_view)

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
        _prepare_headless_capture_for_video(
            env,
            simulator,
            enable_reference_markers=PACKAGE_STATE.reference_markers,
        )

    done_indices = None
    agent.eval()
    try:
        for step in range(max_steps):
            obs, _ = env.reset(done_indices)
            if done_indices is None or done_indices.numel() > 0:
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
                _update_reference_markers_for_capture(
                    env,
                    simulator,
                    enable_reference_markers=PACKAGE_STATE.reference_markers,
                )
                frame_path = frame_path_for_step(frames_dir, step)
                simulator._write_viewport_to_file(str(frame_path))
            done_indices = dones.nonzero(as_tuple=False).squeeze(-1)

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
    return registry_list_models()
