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

Subclasses ProtoMotions' `Simulator` base class (`protomotions.simulator.base_simulator.simulator.Simulator`). Replaces Isaac Lab's `InteractiveScene` with `omni.isaac.core.World` + `Articulation`.

**Responsibilities:**
- Wraps an already-created `World`, `SimulationApp`, and humanoid `Articulation` (does not create them — the entry script does)
- Implements all abstract methods from the `Simulator` base class (see table below)
- State ordering conversion is handled by the inherited `_finalize_setup()` method — the adapter only needs to implement `_get_sim_body_ordering()` to return body/DOF name lists from the articulation; the base class builds the `DataConversionMapping` tensors automatically

**Constructor:** Accepts `world`, `articulation`, and `simulation_app` in addition to the base class arguments (`config`, `robot_config`, `terrain`, `device`, `scene_lib`). The `terrain` parameter will be `None` (flat ground is part of our scene, not a ProtoMotions `Terrain` object). The `scene_lib` will be an empty `SceneLib` (constructed via `SceneLib` with no `scene_file`).

**Quaternion convention:** Isaac Sim's `omni.isaac.core` uses `wxyz` quaternion ordering. The adapter must set `w_last=False` in the simulator config so that ProtoMotions' state conversion layer correctly handles the `wxyz` ↔ `xyzw` (common) mapping.

**Abstract methods to implement (all from Simulator base):**

| Abstract Method | Implementation |
|---|---|
| `_create_simulation()` | No-op — world and articulation already exist |
| `_get_sim_body_ordering()` | Return `SimBodyOrdering` with body/DOF names read from the articulation |
| `_get_simulator_root_state(env_ids)` | Read root pose/velocity via `articulation.get_world_poses()`, `get_velocities()` |
| `_get_simulator_dof_state(env_ids)` | Read joint positions/velocities via `articulation.get_joint_positions()`, `get_joint_velocities()` |
| `_get_simulator_bodies_state(env_ids)` | Read all rigid body states (FK computed by PhysX) via body handle queries |
| `_get_simulator_dof_forces(env_ids)` | Read joint forces via `articulation.get_measured_joint_forces()` |
| `_get_simulator_dof_limits_for_verification()` | Return DOF lower/upper limits from `articulation.get_dof_limits()` |
| `_get_simulator_bodies_contact_buf(env_ids)` | Read contact forces from `ContactSensor` prims attached to each humanoid body |
| `_get_simulator_object_root_state(env_ids)` | Return empty tensor — no SceneLib-managed objects |
| `_get_simulator_object_contact_buf(env_ids)` | Return empty tensor — no SceneLib-managed objects |
| `_set_simulator_env_state(new_states, new_object_states, env_ids)` | Write root + joint state via `articulation.set_world_poses()`, `set_joint_positions()`, `set_joint_velocities()` |
| `_apply_simulator_pd_targets(targets)` | Apply PD position targets via `articulation.apply_action(ArticulationAction(joint_positions=...))` |
| `_apply_simulator_torques(torques)` | Apply raw torques via `articulation.apply_action(ArticulationAction(joint_efforts=...))` |
| `_apply_root_velocity_impulse(env_ids, impulse)` | Set root velocity on articulation for push randomization |
| `_physics_step()` | Loop `decimation` times: inherited `_apply_control()` → `world.step(render=False)` |
| `_write_viewport_to_file(path)` | Viewport capture via `omni.kit.viewport` APIs |
| `_init_camera()` | Set initial camera position/target via `set_camera_view()` |
| `_update_simulator_markers(markers_state)` | Update visualization marker prims (or no-op if markers are not needed) |

**Inherited methods (NOT overridden):**
- `_finalize_setup()` — builds `DataConversionMapping`, PD action scaling, verifies DOF limits. Calls our `_get_sim_body_ordering()`.
- `_apply_control()` — dispatches to `_apply_simulator_pd_targets()` or `_apply_simulator_torques()` based on control mode. Handles PD target computation, action noise injection, domain randomization.
- `render()` — handles video recording state management, calls `_write_viewport_to_file()` when recording.
- `step()` — calls `_physics_step()`, updates markers, calls `render()`.
- `reset_envs()` — converts state from common to sim ordering, calls `_set_simulator_env_state()`.
- `close()` — sets `_simulation_running = False`. Our adapter extends this via `super().close()` then `simulation_app.close()`.

**Contact sensors:** During scene creation, the entry script attaches an `omni.isaac.sensor.ContactSensor` to each humanoid body that ProtoMotions expects contact data from (matching the body names in `robot_config`). The adapter's `_get_simulator_bodies_contact_buf()` reads from these sensors. This ensures the policy receives real contact forces, matching the Isaac Lab pipeline's behavior.

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
- `--checkpoint` (required) — path to ProtoMotions tracker checkpoint. Always required because it contains the humanoid USD asset path and physics config, even in standalone mode.
- `--motion-file` (optional) — path to `.motion` file. If omitted, runs in standalone mode.
- `--headless` — run without GUI
- `--video-output` — MP4 output path. If omitted, no video saved.

Note: `--num-envs` is not included since this pipeline is `num_envs=1` only.

**Standalone mode flow** (no `--motion-file`):
1. Load checkpoint → extract `robot_config` (for humanoid USD path) and `simulator_config` (for physics dt)
2. Launch Isaac Sim via `SimulationApp`
3. Create `World` with physics dt
4. Add ground plane via `omni.isaac.core.objects.GroundPlane`
5. Load humanoid USD asset via `omni.isaac.core.articulations.Articulation` at `/World/Humanoid`
6. Call `populate_scene(world)` to add static objects
7. `world.reset()` — initializes physics for all prims
8. Loop: `world.step()` + `world.render()` — humanoid in rest pose, scene is live

**ProtoMotions mode flow** (`--motion-file` provided):
1. Load checkpoint → extract all configs (robot, simulator, env, agent, motion_lib)
2. Launch Isaac Sim via `SimulationApp`
3. Create `World` with physics dt from simulator config
4. Add ground plane via `omni.isaac.core.objects.GroundPlane`
5. Load humanoid USD asset via `omni.isaac.core.articulations.Articulation` at `/World/Humanoid`
6. Call `populate_scene(world)` to add static objects
7. `world.reset()` — initializes physics for all prims
8. Create empty `SceneLib` (no scene_file, no objects)
9. Create `MotionLib` from `--motion-file`
10. Instantiate `IsaacSimSimulator` adapter wrapping the world + articulation + simulation_app, passing `terrain=None`, the empty `SceneLib`, and checkpoint configs
11. Simulator's two-phase init: `_initialize_with_markers({})` → `_create_simulation()` (no-op) → `_finalize_setup()` (builds conversion tensors)
12. Create ProtoMotions `Env` from checkpoint's env config, passing our simulator + empty scene_lib + motion_lib
13. Create ProtoMotions `Agent` from checkpoint's agent config, load weights
14. Run inference loop: `agent.eval()` → for each step: `env.reset()` → `agent.model(obs)` → `env.step(actions)`
15. Optionally capture frames per step and compile video

Steps 1–7 are shared between both modes.

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

The humanoid motion should be visually identical. Minor rendering differences (lighting, camera angle) are expected. Additionally, compare robot state tensors (root pos/rot, joint positions) between old and new pipelines step-by-step to verify numerical equivalence.

## Risks

1. **Simulator adapter correctness** — state ordering conversion and PD control must exactly match ProtoMotions' expectations. A mismatch means the learned policy won't transfer. Mitigation: compare robot state tensors between old and new pipelines step-by-step.
2. **Contact sensor fidelity** — `omni.isaac.sensor.ContactSensor` may report forces at different precision or timing than Isaac Lab's contact reporting. Mitigation: compare contact force tensors between old and new pipelines step-by-step.
3. **Quaternion convention** — Isaac Sim uses `wxyz`, ProtoMotions common uses `xyzw`. The adapter must set `w_last=False`. Incorrect setting silently produces wrong rotations. Mitigation: verify quaternion output matches Isaac Lab pipeline in first integration test.
4. **Frame capture** — `omni.kit.viewport` capture may differ from Isaac Lab's path. Minor rendering differences are acceptable.
5. **ProtoMotions coupling** — the adapter subclasses `Simulator` and depends on its internal interface (17 abstract methods). ProtoMotions updates could break it. Mitigation: pin ProtoMotions version, add integration test.
