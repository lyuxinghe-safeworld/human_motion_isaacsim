# Custom Isaac Sim Scene with ProtoMotions Humanoid Control

**Date:** 2026-03-18
**Status:** Draft

## Problem

The current pipeline uses ProtoMotions' `build_all_components` which delegates all scene creation to Isaac Lab's `InteractiveScene`. This means the user always sees an Isaac Lab environment (with `/World/envs/env_*/` hierarchy and checkpoint-baked terrain/objects) instead of a clean Isaac Sim scene they control.

## Goal

Build a pure Isaac Sim environment with a flat ground plane, static objects, and the same SMPL humanoid model currently used by ProtoMotions. The environment must work in two modes:

1. **Standalone mode** — the scene runs with the humanoid in rest pose, no ProtoMotions dependency at runtime.
2. **ProtoMotions mode** — the ProtoMotions agent controls the humanoid via a `.motion` file, same inference loop as today.

Both modes share the same scene setup. `num_envs=1` only.

## Design

### Architecture

Three-phase pipeline:

1. **Launch Isaac Sim + build scene** — create `SimulationApp`, `World`, ground plane, humanoid articulation, and static objects using `omni.isaac.core` APIs. No Isaac Lab.
2. **Optionally attach ProtoMotions** — instantiate a custom simulator adapter, then create the ProtoMotions `Env` + `Agent` from checkpoint configs.
3. **Run** — either a bare stepping loop (standalone) or the ProtoMotions inference loop.

### New Files

#### 1. `src/hymotion_isaacsim/isaacsim_simulator.py` — Custom Simulator Adapter

Subclasses ProtoMotions' `Simulator` base class. Replaces Isaac Lab's `InteractiveScene` with `omni.isaac.core.World` + `Articulation`.

**Responsibilities:**
- Wraps an already-created `World` and humanoid `Articulation` (does not create them — the entry script does)
- Implements robot state reading via `articulation.get_world_poses()`, `articulation.get_joint_positions()`, etc.
- Implements robot state writing via `articulation.set_world_poses()`, `articulation.set_joint_positions()`, etc.
- Implements action application via `articulation.apply_action(ArticulationAction(...))`
- Implements physics stepping via `world.step(render=False)` with decimation
- Implements rendering via `world.render()`
- Implements frame capture via `omni.kit.viewport` APIs
- Builds state ordering conversion tensors (simulator ↔ common) the same way `IsaacLabSimulator._finalize_setup()` does — reads joint/body names from the articulation and maps to ProtoMotions' common ordering
- Implements PD control (proportional/built-in) using `ArticulationAction`

**Key interface methods (from Simulator base):**

| Method | Implementation |
|---|---|
| `_create_simulation()` | No-op — world and articulation already exist |
| `_physics_step()` | Loop `decimation` times: `_apply_control()` → `world.step(render=False)` → `world.render()` at boundary |
| `_apply_control()` | Compute PD targets/torques, apply via `ArticulationAction` |
| `_get_simulator_root_state()` | Read root pose/velocity from articulation |
| `_get_simulator_dof_state()` | Read joint positions/velocities from articulation |
| `_get_simulator_bodies_state()` | Read rigid body states (FK computed by PhysX) |
| `_set_simulator_env_state()` | Write root + joint state to articulation |
| `render()` | `world.render()` |
| `_write_viewport_to_file()` | Viewport capture via Kit APIs |
| `close()` | `simulation_app.close()` |

#### 2. `src/hymotion_isaacsim/custom_scene.py` — Scene Builder

A single function:

```python
def populate_scene(world: World) -> None
```

Adds hardcoded static objects to the stage under `/World/custom_scene/`:
- Box (1m cube) at position (2, 1, 0.5)
- Cylinder (r=0.3, h=1.5) at position (-1, 2, 0.75)
- Sphere (r=0.5) at position (1, -1.5, 0.5)

Each object is a rigid body with a collider and `fixed_base=True` (static). Sizes and positions are module-level constants.

#### 3. `scripts/run_custom_scene.py` — Entry Script

CLI arguments:
- `--checkpoint` (required) — path to ProtoMotions tracker checkpoint
- `--motion-file` (optional) — path to `.motion` file. If omitted, runs in standalone mode.
- `--headless` — run without GUI
- `--video-output` — MP4 output path. If omitted, no video saved.
- `--num-envs` — ignored for now (always 1), kept for CLI compatibility

**Standalone mode flow** (no `--motion-file`):
1. Launch Isaac Sim via `SimulationApp`
2. Create `World` with physics dt from checkpoint's simulator config
3. Load humanoid USD asset from checkpoint's `robot_config.asset` path
4. Call `populate_scene(world)` to add static objects
5. `world.reset()`
6. Loop: `world.step()` + `world.render()` — humanoid in rest pose, scene is live

**ProtoMotions mode flow** (`--motion-file` provided):
1. Launch Isaac Sim via `SimulationApp`
2. Create `World` with physics dt from checkpoint's simulator config
3. Load humanoid USD asset from checkpoint's `robot_config.asset` path
4. Call `populate_scene(world)` to add static objects
5. `world.reset()`
6. Instantiate `IsaacSimSimulator` adapter wrapping the world + articulation
7. Create empty `SceneLib`, load `MotionLib` from `--motion-file`
8. Create ProtoMotions `Env` from checkpoint's env config, passing our simulator
9. Create ProtoMotions `Agent` from checkpoint's agent config, load weights
10. Run inference loop: `agent.eval()` → for each step: `env.reset()` → `agent.model(obs)` → `env.step(actions)`
11. Optionally capture frames per step and compile video

Steps 1–5 are shared between both modes.

### Existing Files Modified

None.

### Dependencies

- `omni.isaac.core` (ships with Isaac Sim)
- ProtoMotions (already a dependency)
- No new packages

## Validation

Run the same motion file through both pipelines and compare:
- `scripts/smoke_run_motion.py` (existing — Isaac Lab path)
- `scripts/run_custom_scene.py` (new — pure Isaac Sim path)

The humanoid motion should be visually identical. Minor rendering differences (lighting, camera angle) are expected.

## Risks

1. **Simulator adapter correctness** — state ordering conversion and PD control must exactly match ProtoMotions' expectations. A mismatch means the learned policy won't transfer. Mitigation: compare robot state tensors between old and new pipelines step-by-step.
2. **Frame capture** — `omni.kit.viewport` capture may differ from Isaac Lab's path. Minor rendering differences are acceptable.
3. **ProtoMotions coupling** — the adapter subclasses `Simulator` and depends on its internal interface. ProtoMotions updates could break it. Mitigation: pin ProtoMotions version, add integration test.
