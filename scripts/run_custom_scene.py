#!/usr/bin/env python3
"""Run a custom Isaac Sim scene with optional ProtoMotions humanoid control.

Standalone mode (no --motion-file):
  Humanoid stands in rest pose in a scene with static objects.

ProtoMotions mode (--motion-file provided):
  ProtoMotions agent controls the humanoid, tracking the given motion file.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a custom Isaac Sim scene with optional ProtoMotions control.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to ProtoMotions tracker checkpoint (needed for humanoid asset path)",
    )
    parser.add_argument(
        "--motion-file", type=str, default="",
        help="Path to .motion file. If omitted, runs standalone (rest pose).",
    )
    parser.add_argument("--headless", action="store_true", help="Run Isaac Sim headless")
    parser.add_argument(
        "--video-output", type=str, default="",
        help="Output MP4 path. If omitted, no video saved.",
    )
    parser.add_argument(
        "--reference-markers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render reference-motion marker spheres during headless video capture.",
    )
    return parser.parse_args()


def _align_scene_to_humanoid_root(world, simulator) -> None:
    """Move the authored scene so its local origin stays near the humanoid spawn."""
    from human_motion_isaacsim.custom_scene import set_scene_origin

    root_pos = simulator._get_simulator_root_state().root_pos[0].detach().cpu().numpy()
    set_scene_origin(world, (float(root_pos[0]), float(root_pos[1]), 0.0))


def _enable_reference_markers_for_capture(env, simulator) -> None:
    """Create reference-motion markers even for headless video capture."""
    if not simulator.headless:
        return

    visualization_markers = env.control_manager.create_visualization_markers(headless=False)
    if visualization_markers:
        simulator._build_visualization_markers(visualization_markers)


def _update_reference_markers_for_capture(
    env,
    simulator,
    enable_reference_markers: bool = True,
) -> None:
    """Update reference-motion markers during headless capture without enabling the viewport."""
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
    env,
    simulator,
    enable_reference_markers: bool = True,
) -> None:
    """Prime marker + camera capture state before the motion loop starts."""
    if not simulator.headless:
        return

    if enable_reference_markers:
        _enable_reference_markers_for_capture(env, simulator)
    simulator.prepare_headless_capture()


def _plan_motion_max_steps(duration_seconds: float, simulator_config) -> int:
    """Convert clip duration into env steps instead of raw physics substeps."""
    sim_cfg = getattr(simulator_config, "sim", None)
    sim_fps = getattr(sim_cfg, "fps", 30)
    decimation = max(1, int(getattr(sim_cfg, "decimation", 1)))
    return int(duration_seconds * sim_fps / decimation)


def build_scene(checkpoint_path: str, headless: bool):
    """Create the Isaac Sim world with ground plane, humanoid, and static objects.

    Returns (simulation_app, world, articulation, body_rigid_view, tracker_assets).
    """
    from human_motion_isaacsim.protomotions_path import ensure_protomotions_importable
    from human_motion_isaacsim.checkpoint import load_tracker_assets

    ensure_protomotions_importable()
    tracker_assets = load_tracker_assets(checkpoint_path)

    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": headless})

    from omni.isaac.core import World
    from omni.isaac.core.utils.stage import add_reference_to_stage
    from omni.isaac.core.articulations import Articulation
    from omni.isaac.core.objects import GroundPlane
    from omni.isaac.core.prims import RigidPrimView
    from human_motion_isaacsim.custom_scene import populate_scene

    # Physics dt from checkpoint config
    fps = getattr(tracker_assets.simulator_config.sim, "fps", 60)
    world = World(physics_dt=1.0 / fps, rendering_dt=1.0 / fps)

    # Ground plane
    world.scene.add(GroundPlane(prim_path="/World/GroundPlane", size=100.0))

    # Humanoid USD
    asset_root = tracker_assets.robot_config.asset.asset_root
    usd_file = tracker_assets.robot_config.asset.usd_asset_file_name
    humanoid_usd_path = str(Path(asset_root) / usd_file)
    add_reference_to_stage(humanoid_usd_path, "/World/Humanoid")
    articulation = world.scene.add(
        Articulation(prim_path="/World/Humanoid", name="humanoid")
    )

    # Static objects
    populate_scene(world)

    # Per-body rigid view — needed for per-body transforms/velocities.
    # The USD hierarchy puts bodies under /World/Humanoid/bodies/<name>.
    # Must be added to scene BEFORE world.reset() so it gets initialized.
    body_names = tracker_assets.robot_config.kinematic_info.body_names
    body_prim_paths = [f"/World/Humanoid/bodies/{name}" for name in body_names]
    body_rigid_view = RigidPrimView(
        prim_paths_expr=body_prim_paths, name="humanoid_bodies"
    )
    world.scene.add(body_rigid_view)

    world.reset()

    return simulation_app, world, articulation, body_rigid_view, tracker_assets


def run_standalone(world, simulation_app, headless: bool):
    """Run the scene with the humanoid in rest pose."""
    print("Running standalone mode (humanoid in rest pose). Press Ctrl+C to exit.")
    try:
        while simulation_app.is_running():
            world.step(render=not headless)
    except KeyboardInterrupt:
        pass
    finally:
        simulation_app.close()


def run_protomotions(
    world,
    articulation,
    body_rigid_view,
    simulation_app,
    tracker_assets,
    motion_file: str,
    checkpoint_path: str,
    headless: bool,
    video_output: str | None,
    reference_markers: bool,
):
    """Run ProtoMotions agent to control the humanoid in our custom Isaac Sim scene."""
    from copy import deepcopy
    from dataclasses import asdict

    from human_motion_isaacsim.protomotions_path import ensure_protomotions_importable

    ensure_protomotions_importable()

    # Disable torch.compile warmup (same rationale as protomotions_runtime.py)
    from protomotions.envs import component_manager as component_manager_module

    component_manager_module.TORCH_COMPILE_AVAILABLE = False

    from lightning.fabric import Fabric
    from protomotions.utils.fabric_config import FabricConfig
    from protomotions.utils.hydra_replacement import get_class
    from protomotions.utils.inference_utils import apply_backward_compatibility_fixes
    from protomotions.components.scene_lib import SceneLib
    from protomotions.components.motion_lib import MotionLib
    from protomotions.utils.component_builder import build_terrain_from_config

    from human_motion_isaacsim.isaacsim_simulator import IsaacSimSimulator
    from human_motion_isaacsim.motion_file import load_motion_metadata
    from human_motion_isaacsim.viewport_capture import compile_video, frame_path_for_step

    # --- Fabric (single device, no DDP) ---
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

    # --- Deep-copy configs from checkpoint ---
    robot_config = deepcopy(tracker_assets.robot_config)
    simulator_config = deepcopy(tracker_assets.simulator_config)
    terrain_config = deepcopy(tracker_assets.terrain_config)
    motion_lib_config = deepcopy(tracker_assets.motion_lib_config)
    env_config = deepcopy(tracker_assets.env_config)
    agent_config = deepcopy(tracker_assets.agent_config)

    apply_backward_compatibility_fixes(robot_config, simulator_config, env_config)

    # --- Override configs for single-env custom scene ---
    simulator_config.num_envs = 1
    simulator_config.headless = headless
    motion_lib_config.motion_file = str(Path(motion_file).resolve())

    # Compute max_steps from motion metadata
    motion_metadata = load_motion_metadata(motion_file)
    max_steps = _plan_motion_max_steps(motion_metadata.duration_seconds, simulator_config)

    if hasattr(env_config, "max_episode_length"):
        env_config.max_episode_length = max(env_config.max_episode_length, max_steps + 100)

    # --- Create terrain, empty SceneLib, and MotionLib from motion file ---
    terrain = build_terrain_from_config(terrain_config, num_envs=1, device=fabric.device)
    scene_lib = SceneLib.empty(num_envs=1, device=str(fabric.device), terrain=terrain)
    motion_lib = MotionLib(motion_lib_config, device=str(fabric.device))

    # --- Create our IsaacSimSimulator adapter ---
    simulator = IsaacSimSimulator(
        world=world,
        articulation=articulation,
        simulation_app=simulation_app,
        config=simulator_config,
        robot_config=robot_config,
        scene_lib=scene_lib,
        device=fabric.device,
        terrain=terrain,
    )
    simulator._body_rigid_view = body_rigid_view
    # NOTE: Do NOT call simulator._initialize_with_markers() here.
    # The Env.__init__ calls it internally with visualization markers.

    # --- Create Env ---
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

    # --- Create Agent, load weights ---
    AgentClass = get_class(agent_config._target_)
    agent = AgentClass(
        config=agent_config,
        env=env,
        fabric=fabric,
        root_dir=Path(checkpoint_path).resolve().parent,
    )
    agent.setup()
    agent.load(str(Path(checkpoint_path).resolve()), load_env=False)

    # --- Optional video capture setup ---
    frames_dir = None
    video_path = None
    if video_output:
        video_path = Path(video_output)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        frames_dir = video_path.with_suffix("") / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        _prepare_headless_capture_for_video(
            env,
            simulator,
            enable_reference_markers=reference_markers,
        )

    # --- Inference loop ---
    agent.eval()
    done_indices = None
    try:
        for step in range(max_steps):
            obs, _ = env.reset(done_indices)
            if done_indices is None or done_indices.numel() > 0:
                _align_scene_to_humanoid_root(world, simulator)
            obs = agent.add_agent_info_to_obs(obs)
            obs_td = agent.obs_dict_to_tensordict(obs)
            model_outs = agent.model(obs_td)
            actions = model_outs.get("mean_action", model_outs.get("action"))
            obs, rewards, dones, terminated, extras = env.step(actions)
            if frames_dir is not None:
                _update_reference_markers_for_capture(
                    env,
                    simulator,
                    enable_reference_markers=reference_markers,
                )
                frame_path = frame_path_for_step(frames_dir, step)
                simulator._write_viewport_to_file(str(frame_path))
            done_indices = dones.nonzero(as_tuple=False).squeeze(-1)

        # Compile video if frames were captured
        if frames_dir is not None and video_path is not None:
            frame_paths = sorted(frames_dir.glob("*.png"))
            if frame_paths:
                compile_video(frame_paths, video_path, fps=30)
                print(f"Video saved to {video_path}")
    finally:
        simulation_app.close()


def main():
    args = parse_args()
    simulation_app, world, articulation, body_rigid_view, tracker_assets = build_scene(
        args.checkpoint, args.headless,
    )

    if not args.motion_file:
        run_standalone(world, simulation_app, args.headless)
    else:
        run_protomotions(
            world=world,
            articulation=articulation,
            body_rigid_view=body_rigid_view,
            simulation_app=simulation_app,
            tracker_assets=tracker_assets,
            motion_file=args.motion_file,
            checkpoint_path=args.checkpoint,
            headless=args.headless,
            video_output=args.video_output if args.video_output else None,
            reference_markers=args.reference_markers,
        )
        
        # INSTALL:
        # install with pip
        # FUNCTIONS:
        # .init
        # .control


if __name__ == "__main__":
    main()
