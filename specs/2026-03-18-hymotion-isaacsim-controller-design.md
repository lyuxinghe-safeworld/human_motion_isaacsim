# HY-Motion Isaac Sim ProtoMotion Controller Design

**Date:** 2026-03-18

**Status:** Approved for planning

## Goal

Implement an Isaac Sim resident motion-execution runtime under `/home/lyuxinghe/code/hymotion_isaacsim` that:

1. Attaches to an already-running Isaac Sim session and stage.
2. Binds to the default SMPL humanoid used by `/home/lyuxinghe/code/hymotion_isaaclab`.
3. Accepts a ProtoMotions `.motion` file on demand.
4. Runs ProtoMotions tracking when triggered by a blocking function call.
5. Saves video with the same viewport-capture behavior used by `hymotion_isaaclab`.
6. Leaves the humanoid at the same world position in its default rest pose after each motion.
7. Stays alive across multiple sequential motion executions in one Isaac Sim session.

## Non-Goals

This `v1` does not attempt to:

1. Discover humanoids dynamically in the stage.
2. Support raw HY-Motion `.npz` input at runtime.
3. Support concurrent motion runs.
4. Support arbitrary humanoid assets that differ from the default SMPL asset expected by the ProtoMotions checkpoint.
5. Preserve the final tracked root orientation after the motion completes.
6. Own long-running scenario generation or world creation outside the execution window.

## Working Constraints

The design is constrained by the current behavior of the existing repos:

1. `/home/lyuxinghe/code/hymotion_isaaclab/scripts/run_tracking.py` is a thin inline wrapper around ProtoMotions inference setup, not a separate controller stack.
2. `/home/lyuxinghe/code/ProtoMotions/protomotions/inference_agent.py` and the IsaacLab simulator path assume ProtoMotions owns simulator bootstrap, env construction, and the simulation loop.
3. ProtoMotions video generation in the IsaacLab path is based on capturing the active viewport via `simulator._write_viewport_to_file(...)`, while the ProtoMotions camera logic maintains a follow camera.
4. The user's scenario generation already happens inside Isaac Sim and cannot be replaced.
5. During a motion execution window, the controller may assume no external interference and may own environment stepping exclusively.

## Versioning Decision

Current NVIDIA docs now describe newer Isaac Sim Python packaging, but the working local stack used by the existing HY-Motion IsaacLab flow is already pinned in `/home/lyuxinghe/code/env_isaaclab`:

1. Python `3.11.15`
2. `isaacsim 5.1.0.0`
3. `isaaclab 2.3.0`

`v1` will mirror this known-good stack instead of targeting the latest docs-only stack. The `env/` setup should use `uv` and pin the Isaac Sim / Isaac Lab versions to match the working environment above. This reduces the risk of API drift between current ProtoMotions code and newer Isaac Sim releases.

## Recommended Approach

Use a persistent, env-bound ProtoMotions runtime that attaches to an existing Isaac Sim session and humanoid, then temporarily owns stepping while a blocking `run_motion(...)` call executes.

This is preferred over:

1. A standalone runner that owns the entire Isaac Sim session, because the scenario system already owns stage creation.
2. A native Isaac Sim controller that only reuses ProtoMotions weights, because that would require rebuilding large parts of the observation/action bridge and would diverge from the known-good inference path.

## High-Level Architecture

The implementation lives entirely under `/home/lyuxinghe/code/hymotion_isaacsim`.

Planned top-level layout:

1. `env/`
   Stores the `uv` environment bootstrap, dependency pins, and setup helpers for the Isaac Sim + Isaac Lab + ProtoMotions runtime.
2. `specs/`
   Stores this design doc and future design artifacts.
3. `plans/`
   Stores the implementation plan derived from this design.
4. `src/hymotion_isaacsim/`
   Stores the controller runtime and supporting modules.
5. `tests/`
   Stores unit tests and environment-aware integration/smoke tests.
6. `scripts/`
   Stores local entrypoints for manual validation.

Core runtime modules:

1. `runtime.py`
   Owns persistent lifecycle. Exposes `ProtoMotionIsaacSimController`.
2. `binding.py`
   Binds the controller to a fixed humanoid prim path for `v1`.
3. `checkpoint.py`
   Loads ProtoMotions checkpoint/config artifacts once and keeps the model warm.
4. `motion_executor.py`
   Runs a blocking motion execution against the bound humanoid.
5. `recording.py`
   Captures viewport frames and compiles MP4 output with the same active-viewport behavior as the IsaacLab path.
6. `rest_pose.py`
   Restores the humanoid to its default rest pose at the desired world position after execution.
7. `result.py`
   Defines a small structured result object returned by `run_motion(...)`.

## Ownership Boundary

The runtime must respect this ownership model:

1. The Isaac Sim application is already running before controller construction.
2. The surrounding stage and scenario objects are owned by external scenario-generation code.
3. The controller only assumes ownership of:
   - the fixed-path humanoid binding
   - its own persistent policy/checkpoint state
   - stepping during a `run_motion(...)` call
   - video capture during a `run_motion(...)` call
4. After the run returns, the stage remains alive and controller state remains warm for the next motion.

This keeps `v1` compatible with a changing Isaac Sim stage while still allowing deterministic motion execution windows.

## Humanoid Binding Contract

`v1` uses a fixed binding contract:

1. The humanoid exists at a fixed prim path, for example `/World/Humanoid`.
2. The humanoid corresponds to the same default SMPL asset identity expected by the ProtoMotions motion tracker checkpoint and used by `hymotion_isaaclab`.
3. The articulation layout is stable across runs in the same session.

Binding-time validation must check:

1. The prim path exists.
2. The prim represents a valid articulation.
3. The articulation DOF/body naming is compatible with the expected SMPL robot config.

Code notes must be added at each fixed-path and fixed-asset assumption to make future `v2` dynamic discovery straightforward.

## Runtime API

The primary API is library-first and persistent:

```python
controller = ProtoMotionIsaacSimController(
    humanoid_prim_path="/World/Humanoid",
    checkpoint_path="...",
    simulator_context=...,
)

result = controller.run_motion(
    motion_file="/path/to/file.motion",
    video_output="/path/to/output.mp4",
)
```

`run_motion(...)` semantics:

1. Blocking call.
2. Exclusive control during execution.
3. One motion at a time.
4. Raises or returns structured failure on invalid setup/runtime errors.

Suggested result fields:

1. `success`
2. `motion_file`
3. `video_output`
4. `num_steps`
5. `duration_seconds`
6. `error_message`

## Motion Execution Flow

Controller initialization:

1. Validate Isaac Sim environment availability.
2. Bind to the fixed humanoid prim path.
3. Load ProtoMotions tracker checkpoint and frozen inference configs.
4. Resolve the same SMPL robot configuration used by the existing tracking path.
5. Initialize persistent policy/model state and reusable buffers.
6. Do not start stepping yet.

Per-motion execution:

1. Validate `.motion` file path.
2. Reject the request if another run is already active.
3. Load the motion into a ProtoMotions-compatible motion library/runtime.
4. Snapshot the humanoid root transform before execution.
5. Start recording state if video output is requested.
6. Step the environment in a blocking loop:
   - collect live humanoid state
   - build ProtoMotions observations
   - run policy inference
   - apply actions to the humanoid
   - step the simulation
   - capture the active viewport frame
7. Stop once the motion duration is reached or on failure/interruption.
8. Compile frames into MP4.
9. Restore the humanoid to the default rest pose while keeping the desired root world position.
10. Return a structured result.

## Rest Pose Semantics

The agreed `v1` behavior is:

1. After motion completion, the humanoid stays at the same world position.
2. The humanoid returns to the default asset pose.
3. The final tracked orientation is not preserved unless it matches the default pose orientation.

Operationally, `rest_pose.py` should:

1. Read the default joint configuration from the bound humanoid / expected asset defaults.
2. Preserve the root translation target.
3. Zero or reset root velocities and joint velocities.
4. Write the resulting state back to the articulation.
5. Step a small stabilization window if needed so the pose settles consistently before returning.

## Recording Strategy

The recording must match the behavior of `hymotion_isaaclab`, which currently:

1. Uses the ProtoMotions/IsaacLab follow-camera behavior.
2. Captures the active viewport each step through the simulator viewport capture path.
3. Compiles the frame sequence into an MP4.

`v1` should therefore:

1. Reuse the active viewport capture approach, not replace it with a separate Replicator or standalone rendering pipeline.
2. Keep controller-owned camera behavior close to ProtoMotions' existing follow-camera logic so the output clip resembles the current IsaacLab output.
3. Compile MP4 files with the same codec/pixel-format assumptions currently used in `hymotion_isaaclab`.

## Integration Strategy With ProtoMotions

The main design challenge is that ProtoMotions expects to build its own simulator/env stack. `v1` should avoid a full rewrite while also avoiding a black-box subprocess runner.

Recommended integration strategy:

1. Reuse checkpoint loading, motion library loading, policy inference, robot config metadata, and existing recording behavior.
2. Extract or wrap the minimal inference-time pieces needed to run against an already-bound Isaac Sim humanoid and simulation context.
3. Introduce an adapter layer that presents the necessary state/action interface to the reused ProtoMotions logic.

This implies a thin controller-specific adaptation layer rather than trying to call `protomotions/inference_agent.py` directly.

## Error Handling

Initialization failures must fail fast when:

1. The fixed humanoid prim path does not exist.
2. The prim is not a valid articulation.
3. The articulation is incompatible with the expected SMPL robot layout.
4. The ProtoMotions checkpoint or frozen config artifacts are missing.
5. Required Isaac Sim / Isaac Lab imports are unavailable in the configured `uv` environment.

Runtime failures must:

1. Prevent overlapping runs.
2. Stop recording cleanly when possible.
3. Attempt to restore the humanoid to default rest pose at the last stable root position.
4. Return or raise a clear error with motion path and execution phase context.

## Testing Strategy

Unit-level tests:

1. Humanoid binding validation.
2. Busy-state guard for overlapping runs.
3. Motion file validation.
4. Rest-pose transform composition and velocity reset behavior.
5. Result object population for success/failure cases.

Environment-aware integration tests:

1. Bind to the default SMPL humanoid at the fixed prim path.
2. Execute a known `.motion` file successfully.
3. Verify MP4 output is created.
4. Verify post-run humanoid root position is preserved while joint pose returns to default.
5. Verify a second motion can execute in the same session without rebuilding the controller.

Manual smoke path:

1. Create a small standalone validation script under `scripts/`.
2. Use it to start an empty Isaac Sim scene, load the default SMPL humanoid, execute one known `.motion`, and save video.
3. Use this script for initial end-to-end verification even though the production target remains an env-bound controller.

## Environment Setup Plan

The `env/` directory should provide a `uv`-based setup that mirrors the known-good local stack:

1. Create a `uv` virtual environment with Python `3.11.x`.
2. Install `torch` and the Isaac Sim pip packages pinned to the working local version.
3. Install `isaaclab==2.3.0`.
4. Install ProtoMotions and any supporting runtime dependencies needed by the controller.
5. Provide a reproducible setup script and a compact requirements/constraints file.
6. Document required environment variables such as `OMNI_KIT_ACCEPT_EULA=YES`.

The implementation should prefer this pinned runtime over the latest public versions until the controller is proven stable.

## Open Technical Risks

1. The largest risk is the adaptation boundary between ProtoMotions' existing IsaacLab inference path and an already-running Isaac Sim session.
2. The follow-camera behavior may need a small amount of controller-local glue to preserve output parity when the surrounding stage is externally owned.
3. Restoring the humanoid to default pose may require a short stabilization step window to avoid transient drift.
4. Exact compatibility depends on matching the working Isaac Sim / Isaac Lab package versions already present in `/home/lyuxinghe/code/env_isaaclab`.

## Implementation Direction

The implementation should proceed in this order:

1. Create the pinned `uv` environment definition under `env/`.
2. Build a minimal standalone smoke path in Isaac Sim that can load the default SMPL humanoid and step a scene.
3. Add the fixed-path humanoid binder and validation.
4. Factor a persistent controller runtime around ProtoMotions checkpoint/model reuse.
5. Add the blocking `.motion` execution path.
6. Add recording parity with the IsaacLab path.
7. Add rest-pose restoration.
8. Add automated tests and smoke-test documentation.

## Sources Reviewed

Local repos:

1. `/home/lyuxinghe/code/hymotion_isaaclab/scripts/run_tracking.py`
2. `/home/lyuxinghe/code/hymotion_isaaclab/README.md`
3. `/home/lyuxinghe/code/ProtoMotions/protomotions/inference_agent.py`
4. `/home/lyuxinghe/code/ProtoMotions/protomotions/utils/component_builder.py`
5. `/home/lyuxinghe/code/ProtoMotions/protomotions/simulator/isaaclab/simulator.py`
6. `/home/lyuxinghe/code/ProtoMotions/protomotions/robot_configs/smpl.py`
7. `/home/lyuxinghe/code/ProtoMotions/examples/tutorial/5_motion_manager.py`
8. `/home/lyuxinghe/code/ProtoMotions/examples/tutorial/6_mimic_environment.py`

Environment references:

1. `/home/lyuxinghe/code/env_isaaclab/pyvenv.cfg`
2. `/home/lyuxinghe/code/env_isaaclab/lib/python3.11/site-packages/isaaclab-2.3.0.dist-info`
3. `/home/lyuxinghe/code/env_isaaclab/lib/python3.11/site-packages/isaacsim-5.1.0.0.dist-info`
