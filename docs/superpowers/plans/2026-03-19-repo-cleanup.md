# Repo Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename files/classes for clarity, merge two runtime modules, delete unused code, and update all references — with no logic changes.

**Architecture:** Flat package layout preserved. Each task handles one rename/merge/delete operation plus all its downstream import and string reference updates. Tasks are ordered so earlier renames don't break later tasks.

**Tech Stack:** Python 3.11, pytest, bash, git

**Spec:** `docs/superpowers/specs/2026-03-19-repo-cleanup-design.md`

---

### Task 1: Delete `rest_pose.py` and its tests

**Files:**
- Delete: `src/human_motion_isaacsim/rest_pose.py`
- Delete: `tests/test_rest_pose.py`

- [ ] **Step 1: Verify rest_pose has no runtime imports**

Run: `grep -r "from human_motion_isaacsim.rest_pose" src/ scripts/ --include="*.py"`
Expected: No output (only tests import it)

- [ ] **Step 2: Delete the files**

```bash
git rm src/human_motion_isaacsim/rest_pose.py tests/test_rest_pose.py
```

- [ ] **Step 3: Run tests to confirm nothing breaks**

Run: `pytest tests/ -v --ignore=third_party`
Expected: All remaining tests pass

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore: remove unused rest_pose module and tests"
```

---

### Task 2: Rename `recording.py` → `viewport_capture.py`

**Files:**
- Rename: `src/human_motion_isaacsim/recording.py` → `src/human_motion_isaacsim/viewport_capture.py`
- Rename: `tests/test_recording.py` → `tests/test_viewport_capture.py`
- Modify: `src/human_motion_isaacsim/__init__.py`
- Modify: `scripts/test_isaacsim_monitor.py` (soon to be renamed itself, but update imports now)
- Modify: `scripts/run_custom_scene.py` (soon to be renamed itself, but update imports now)
- Modify: `src/human_motion_isaacsim/protomotions_runtime.py` (soon to be merged, but update imports now)
- Modify: `tests/test_runtime_guards.py`

- [ ] **Step 1: Rename source file**

```bash
git mv src/human_motion_isaacsim/recording.py src/human_motion_isaacsim/viewport_capture.py
```

- [ ] **Step 2: Rename test file**

```bash
git mv tests/test_recording.py tests/test_viewport_capture.py
```

- [ ] **Step 3: Update `__init__.py`**

Change:
```python
    "recording",
```
to:
```python
    "viewport_capture",
```

Change:
```python
    if name == "recording":
        import importlib

        return importlib.import_module("human_motion_isaacsim.recording")
```
to:
```python
    if name == "viewport_capture":
        import importlib

        return importlib.import_module("human_motion_isaacsim.viewport_capture")
```

- [ ] **Step 4: Update `scripts/test_isaacsim_monitor.py` import**

Change:
```python
from human_motion_isaacsim.recording import capture_active_viewport_to_file
```
to:
```python
from human_motion_isaacsim.viewport_capture import capture_active_viewport_to_file
```

- [ ] **Step 5: Update `scripts/run_custom_scene.py` import**

Change:
```python
from human_motion_isaacsim.recording import compile_video, frame_path_for_step
```
to:
```python
from human_motion_isaacsim.viewport_capture import compile_video, frame_path_for_step
```

- [ ] **Step 6: Update `src/human_motion_isaacsim/protomotions_runtime.py` import**

Change (inside `run_standalone_motion`):
```python
from human_motion_isaacsim.recording import compile_video, frame_path_for_step
```
to:
```python
from human_motion_isaacsim.viewport_capture import compile_video, frame_path_for_step
```

- [ ] **Step 7: Update `tests/test_runtime_guards.py` references**

Change line 295:
```python
from human_motion_isaacsim import recording
```
to:
```python
from human_motion_isaacsim import viewport_capture
```

Then update all `recording.` references in that test function to `viewport_capture.` (lines 308, 314).

Change line 325:
```python
from human_motion_isaacsim.recording import frame_path_for_step
```
to:
```python
from human_motion_isaacsim.viewport_capture import frame_path_for_step
```

Change line 396:
```python
monkeypatch.setattr("human_motion_isaacsim.recording.compile_video", fake_compile)
```
to:
```python
monkeypatch.setattr("human_motion_isaacsim.viewport_capture.compile_video", fake_compile)
```

- [ ] **Step 8: Run tests**

Run: `pytest tests/ -v --ignore=third_party`
Expected: All tests pass

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "refactor: rename recording.py to viewport_capture.py"
```

---

### Task 3: Rename `isaacsim_simulator.py` → `simulator_adapter.py`, class `IsaacSimSimulator` → `SimulatorAdapter`

**Files:**
- Rename: `src/human_motion_isaacsim/isaacsim_simulator.py` → `src/human_motion_isaacsim/simulator_adapter.py`
- Rename: `tests/test_isaacsim_simulator.py` → `tests/test_simulator_adapter.py`
- Modify: `scripts/run_custom_scene.py`
- Modify: `tests/test_simulator_adapter.py` (after rename)

- [ ] **Step 1: Rename source file**

```bash
git mv src/human_motion_isaacsim/isaacsim_simulator.py src/human_motion_isaacsim/simulator_adapter.py
```

- [ ] **Step 2: Rename class in `simulator_adapter.py`**

Change:
```python
class IsaacSimSimulator(Simulator):
```
to:
```python
class SimulatorAdapter(Simulator):
```

Also do a global find-and-replace of `IsaacSimSimulator` → `SimulatorAdapter` within this file (docstrings, any internal references).

- [ ] **Step 3: Rename test file**

```bash
git mv tests/test_isaacsim_simulator.py tests/test_simulator_adapter.py
```

- [ ] **Step 4: Update all `IsaacSimSimulator` references in `tests/test_simulator_adapter.py`**

Replace all occurrences of:
- `from human_motion_isaacsim.isaacsim_simulator import IsaacSimSimulator` → `from human_motion_isaacsim.simulator_adapter import SimulatorAdapter`
- `IsaacSimSimulator` → `SimulatorAdapter` (class references, `__new__` calls, etc.)
- `TestIsaacSimSimulatorConstruction` → `TestSimulatorAdapterConstruction`

- [ ] **Step 5: Update `scripts/run_custom_scene.py`**

Change:
```python
from human_motion_isaacsim.isaacsim_simulator import IsaacSimSimulator
```
to:
```python
from human_motion_isaacsim.simulator_adapter import SimulatorAdapter
```

Change the comment (line 248):
```python
# --- Create our IsaacSimSimulator adapter ---
```
to:
```python
# --- Create our SimulatorAdapter ---
```

Change the instantiation (line 249):
```python
simulator = IsaacSimSimulator(
```
to:
```python
simulator = SimulatorAdapter(
```

- [ ] **Step 6: Run tests**

Run: `pytest tests/ -v --ignore=third_party`
Expected: All tests pass

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "refactor: rename IsaacSimSimulator to SimulatorAdapter"
```

---

### Task 4: Merge `runtime.py` + `protomotions_runtime.py` → `motion_runner.py`, rename classes

**Files:**
- Delete: `src/human_motion_isaacsim/runtime.py`
- Delete: `src/human_motion_isaacsim/protomotions_runtime.py`
- Create: `src/human_motion_isaacsim/motion_runner.py`
- Modify: `src/human_motion_isaacsim/__init__.py`
- Rename: `tests/test_runtime_guards.py` → `tests/test_motion_runner.py`
- Modify: `tests/test_motion_runner.py` (after rename)
- Modify: `scripts/smoke_run_motion.py`
- Modify: `scripts/run_custom_scene.py`
- Modify: `tests/test_result.py`

- [ ] **Step 1: Create `motion_runner.py` by merging both files**

The merged file contains `MotionController` (from `runtime.py`) listed first, then `MotionRunner` (from `protomotions_runtime.py`). Shared imports are deduplicated.

```python
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
        self.bound_humanoid = bind_humanoid(
            humanoid_prim_path,
            lookup_articulation=lookup_articulation,
        )
        self.tracker_assets = load_assets(self.checkpoint_path)
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
        # Deferred import preserved from original protomotions_runtime.py
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
```

- [ ] **Step 2: Delete old files**

```bash
git rm src/human_motion_isaacsim/runtime.py src/human_motion_isaacsim/protomotions_runtime.py
```

- [ ] **Step 3: Update `__init__.py`**

Change:
```python
    "ProtoMotionIsaacSimController",
```
to:
```python
    "MotionController",
```

Change:
```python
    if name == "ProtoMotionIsaacSimController":
        from human_motion_isaacsim.runtime import ProtoMotionIsaacSimController

        return ProtoMotionIsaacSimController
```
to:
```python
    if name == "MotionController":
        from human_motion_isaacsim.motion_runner import MotionController

        return MotionController
```

- [ ] **Step 4: Rename test file**

```bash
git mv tests/test_runtime_guards.py tests/test_motion_runner.py
```

- [ ] **Step 5: Update all imports in `tests/test_motion_runner.py`**

Replace all occurrences:
- `from human_motion_isaacsim.runtime import ProtoMotionIsaacSimController` → `from human_motion_isaacsim.motion_runner import MotionController`
- `ProtoMotionIsaacSimController(` → `MotionController(`
- `from human_motion_isaacsim.protomotions_runtime import ProtoMotionsRuntime` → `from human_motion_isaacsim.motion_runner import MotionRunner`
- `ProtoMotionsRuntime(` → `MotionRunner(`
- `ProtoMotionsRuntime.from_checkpoint_path` → `MotionRunner.from_checkpoint_path`
- `"human_motion_isaacsim.protomotions_runtime.ensure_protomotions_importable"` → `"human_motion_isaacsim.motion_runner.ensure_protomotions_importable"`
- `monkeypatch.setattr(ProtoMotionsRuntime, "build_standalone_runner", ...)` → `monkeypatch.setattr(MotionRunner, "build_standalone_runner", ...)`

- [ ] **Step 6: Update `scripts/smoke_run_motion.py`**

Change:
```python
from human_motion_isaacsim.protomotions_runtime import ProtoMotionsRuntime
```
to:
```python
from human_motion_isaacsim.motion_runner import MotionRunner
```

Change:
```python
runtime = ProtoMotionsRuntime.from_checkpoint_path(args.checkpoint)
```
to:
```python
runtime = MotionRunner.from_checkpoint_path(args.checkpoint)
```

- [ ] **Step 7: Update `scripts/run_custom_scene.py`**

Change the comment referencing `protomotions_runtime.py`:
```python
# Disable torch.compile warmup (same rationale as protomotions_runtime.py)
```
to:
```python
# Disable torch.compile warmup (same rationale as motion_runner.py)
```

- [ ] **Step 8: Update `tests/test_result.py`**

Change line 4:
```python
assert hasattr(human_motion_isaacsim, "ProtoMotionIsaacSimController")
```
to:
```python
assert hasattr(human_motion_isaacsim, "MotionController")
```

- [ ] **Step 9: Run tests**

Run: `pytest tests/ -v --ignore=third_party`
Expected: All tests pass

- [ ] **Step 10: Commit**

```bash
git add -A
git commit -m "refactor: merge runtime + protomotions_runtime into motion_runner.py"
```

---

### Task 5: Rename scripts

**Files:**
- Rename: `scripts/smoke_run_motion.py` → `scripts/smoke_motion.py`
- Rename: `scripts/test_isaacsim_monitor.py` → `scripts/smoke_monitor.py`
- Rename: `scripts/run_custom_scene.py` → `scripts/run_scene.py`
- Rename: `scripts/run_custom_scene.sh` → `scripts/run_scene.sh`
- Modify: `scripts/run_scene.sh` (internal reference to .py)
- Rename: `tests/test_run_custom_scene.py` → `tests/test_run_scene.py`
- Rename: `tests/test_render_motion_video_script.py` → `tests/test_run_scene_script.py`
- Modify: `tests/test_run_scene.py` (all `run_custom_scene` → `run_scene` references)
- Modify: `tests/test_run_scene_script.py` (hardcoded path strings)
- Modify: `tests/test_result.py` (doc assertion strings)

- [ ] **Step 1: Rename script files**

```bash
git mv scripts/smoke_run_motion.py scripts/smoke_motion.py
git mv scripts/test_isaacsim_monitor.py scripts/smoke_monitor.py
git mv scripts/run_custom_scene.py scripts/run_scene.py
git mv scripts/run_custom_scene.sh scripts/run_scene.sh
```

- [ ] **Step 2: Update `scripts/run_scene.sh` internal reference**

Change line 6:
```
Usage: scripts/run_custom_scene.sh --motion-file PATH [options]
```
to:
```
Usage: scripts/run_scene.sh --motion-file PATH [options]
```

Change line 125:
```bash
"$repo_root/scripts/run_custom_scene.py"
```
to:
```bash
"$repo_root/scripts/run_scene.py"
```

- [ ] **Step 3: Rename test files**

```bash
git mv tests/test_run_custom_scene.py tests/test_run_scene.py
git mv tests/test_render_motion_video_script.py tests/test_run_scene_script.py
```

- [ ] **Step 4: Update all imports in `tests/test_run_scene.py`**

Replace all occurrences of `run_custom_scene` with `run_scene`:
- `from scripts.run_custom_scene import parse_args` → `from scripts.run_scene import parse_args`
- `from scripts import run_custom_scene` → `from scripts import run_scene`
- `run_custom_scene._align_scene_to_humanoid_root(` → `run_scene._align_scene_to_humanoid_root(`
- `from scripts.run_custom_scene import _enable_reference_markers_for_capture` → `from scripts.run_scene import _enable_reference_markers_for_capture`
- `from scripts.run_custom_scene import _update_reference_markers_for_capture` → `from scripts.run_scene import _update_reference_markers_for_capture`
- `from scripts.run_custom_scene import _prepare_headless_capture_for_video` → `from scripts.run_scene import _prepare_headless_capture_for_video`
- `"scripts.run_custom_scene._enable_reference_markers_for_capture"` → `"scripts.run_scene._enable_reference_markers_for_capture"`
- `from scripts.run_custom_scene import _plan_motion_max_steps` → `from scripts.run_scene import _plan_motion_max_steps`
- `"run_custom_scene.py"` in `sys.argv` monkeypatches → `"run_scene.py"`

- [ ] **Step 5: Update hardcoded path strings in `tests/test_run_scene_script.py`**

Change line 18:
```python
wrapper_source = Path("scripts/run_custom_scene.sh").read_text()
```
to:
```python
wrapper_source = Path("scripts/run_scene.sh").read_text()
```

Change line 19:
```python
_write_executable(repo_root / "scripts" / "run_custom_scene.sh", wrapper_source)
```
to:
```python
_write_executable(repo_root / "scripts" / "run_scene.sh", wrapper_source)
```

Change line 20:
```python
(repo_root / "scripts" / "run_custom_scene.py").write_text("# stub\n")
```
to:
```python
(repo_root / "scripts" / "run_scene.py").write_text("# stub\n")
```

Change line 76:
```python
[str(repo_root / "scripts" / "run_custom_scene.sh"), *args],
```
to:
```python
[str(repo_root / "scripts" / "run_scene.sh"), *args],
```

Change line 94 (inside assertion):
```python
str(repo_root / "scripts" / "run_custom_scene.py"),
```
to:
```python
str(repo_root / "scripts" / "run_scene.py"),
```

Change line 163 (inside assertion):
```python
str(repo_root / "scripts" / "run_custom_scene.py"),
```
to:
```python
str(repo_root / "scripts" / "run_scene.py"),
```

Change test function names:
- `test_run_custom_scene_wrapper_defaults_to_headless` → `test_run_scene_wrapper_defaults_to_headless`
- `test_run_custom_scene_wrapper_sets_display_when_not_headless` → `test_run_scene_wrapper_sets_display_when_not_headless`

- [ ] **Step 6: Update `tests/test_result.py` doc assertion strings**

Change line 37:
```python
assert "scripts/smoke_run_motion.py" in text
```
to:
```python
assert "scripts/smoke_motion.py" in text
```

Change line 40:
```python
def test_docs_reference_run_custom_scene_wrapper_script():
```
to:
```python
def test_docs_reference_run_scene_wrapper_script():
```

Change lines 46-47:
```python
assert "scripts/run_custom_scene.sh" in readme_text
assert "scripts/run_custom_scene.sh" in env_readme_text
```
to:
```python
assert "scripts/run_scene.sh" in readme_text
assert "scripts/run_scene.sh" in env_readme_text
```

Change line 57:
```python
assert "scripts/test_isaacsim_monitor.py" in text
```
to:
```python
assert "scripts/smoke_monitor.py" in text
```

- [ ] **Step 7: Run tests (expect failures from README/env doc references — that's Task 6)**

Run: `pytest tests/ -v --ignore=third_party -k "not test_docs_reference and not test_env_readme_documents_vnc and not test_env_readme_contains_smoke"`
Expected: All non-doc tests pass

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "refactor: rename scripts to smoke_*/run_scene conventions"
```

---

### Task 6: Update README.md and env/README.md references

**Files:**
- Modify: `README.md`
- Modify: `env/README.md`

- [ ] **Step 1: Update `README.md`**

All references to old names:

Line 22: `scripts/test_isaacsim_monitor.py` → `scripts/smoke_monitor.py`
Line 23: `scripts/smoke_run_motion.py` → `scripts/smoke_motion.py`
Line 27: `scripts/run_custom_scene.sh` → `scripts/run_scene.sh`; `scripts/run_custom_scene.py` → `scripts/run_scene.py`
Lines 40, 47, 56: `scripts/run_custom_scene.sh` → `scripts/run_scene.sh`
Lines 73-75 (key runtime code):
```
- `src/human_motion_isaacsim/protomotions_runtime.py`
- `src/human_motion_isaacsim/runtime.py`
- `src/human_motion_isaacsim/recording.py`
```
→
```
- `src/human_motion_isaacsim/motion_runner.py`
- `src/human_motion_isaacsim/viewport_capture.py`
```

- [ ] **Step 2: Update `env/README.md`**

Line 53: `scripts/run_custom_scene.sh` → `scripts/run_scene.sh`; `scripts/run_custom_scene.py` → `scripts/run_scene.py`
Lines 58, 65: `scripts/run_custom_scene.sh` → `scripts/run_scene.sh`
Line 121: `scripts/test_isaacsim_monitor.py` → `scripts/smoke_monitor.py`
Lines 151, 160: `scripts/smoke_run_motion.py` → `scripts/smoke_motion.py`
Line 174: `scripts/run_custom_scene.sh` → `scripts/run_scene.sh`; `scripts/run_custom_scene.py` → `scripts/run_scene.py` (this line contains BOTH references and must be fully updated)

- [ ] **Step 3: Run full test suite**

Run: `pytest tests/ -v --ignore=third_party`
Expected: All tests pass (including doc assertion tests)

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "docs: update README references to match renamed files"
```

---

### Task 7: Final verification

- [ ] **Step 1: Run full test suite**

Run: `pytest tests/ -v --ignore=third_party`
Expected: All tests pass

- [ ] **Step 2: Verify no stale references remain**

```bash
grep -rn "IsaacSimSimulator\|isaacsim_simulator\|ProtoMotionIsaacSimController\|ProtoMotionsRuntime\|protomotions_runtime\|rest_pose\|run_custom_scene\|smoke_run_motion\|test_isaacsim_monitor\|recording\.py" src/ scripts/ tests/ README.md env/README.md --include="*.py" --include="*.sh" --include="*.md" | grep -v "third_party/" | grep -v "__pycache__" | grep -v "docs/superpowers/"
```

Expected: No output (all old names are gone)

- [ ] **Step 3: Verify all source files exist at new paths**

```bash
ls src/human_motion_isaacsim/simulator_adapter.py \
   src/human_motion_isaacsim/motion_runner.py \
   src/human_motion_isaacsim/viewport_capture.py \
   scripts/smoke_motion.py \
   scripts/smoke_monitor.py \
   scripts/run_scene.py \
   scripts/run_scene.sh \
   tests/test_simulator_adapter.py \
   tests/test_motion_runner.py \
   tests/test_viewport_capture.py \
   tests/test_run_scene.py \
   tests/test_run_scene_script.py
```

Expected: All files listed

- [ ] **Step 4: Verify old files are gone**

```bash
ls src/human_motion_isaacsim/isaacsim_simulator.py \
   src/human_motion_isaacsim/runtime.py \
   src/human_motion_isaacsim/protomotions_runtime.py \
   src/human_motion_isaacsim/recording.py \
   src/human_motion_isaacsim/rest_pose.py \
   tests/test_rest_pose.py \
   tests/test_isaacsim_simulator.py \
   tests/test_runtime_guards.py \
   tests/test_recording.py \
   tests/test_run_custom_scene.py \
   tests/test_render_motion_video_script.py \
   scripts/smoke_run_motion.py \
   scripts/test_isaacsim_monitor.py \
   scripts/run_custom_scene.py \
   scripts/run_custom_scene.sh 2>&1
```

Expected: All "No such file or directory"
