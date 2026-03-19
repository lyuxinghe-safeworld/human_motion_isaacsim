# Repo Cleanup: Rename, Merge, and Remove Unused Files

**Date:** 2026-03-19
**Approach:** A — Rename + Merge Only (no logic changes, no file splitting)

## Goals

1. **Readability:** Every file name immediately tells you what it does
2. **Consistency:** Scripts follow prefix conventions (`smoke_*`, `run_*`, `test_*`)
3. **Cleanup:** Remove dead code that isn't used in any runtime path
4. **Constraints:** File names ≤ 2 words (snake_case), class names ≤ 3 words (PascalCase)

## Source Module Renames

| Current File | New File | Current Class/Symbols | New Class/Symbols |
|---|---|---|---|
| `isaacsim_simulator.py` | `simulator_adapter.py` | `IsaacSimSimulator` | `SimulatorAdapter` |
| `runtime.py` + `protomotions_runtime.py` | `motion_runner.py` | `ProtoMotionIsaacSimController` + `ProtoMotionsRuntime` | `MotionController` + `MotionRunner` |
| `recording.py` | `viewport_capture.py` | (functions only — unchanged) | (functions only — unchanged) |

### Unchanged Source Files

These already have clear, ≤ 2-word names:

- `binding.py` — SMPL layout validation
- `checkpoint.py` — ProtoMotions checkpoint loading
- `motion_file.py` — Motion file metadata extraction
- `result.py` — Motion run result dataclass
- `custom_scene.py` — Static scene population
- `protomotions_path.py` — ProtoMotions root resolution
- `__init__.py` — Package public API (lazy exports)

## Script Renames

| Current | New | Reason |
|---|---|---|
| `scripts/smoke_run_motion.py` | `scripts/smoke_motion.py` | Consistent `smoke_*` prefix |
| `scripts/test_isaacsim_monitor.py` | `scripts/smoke_monitor.py` | Not a pytest — it's a display smoke check |
| `scripts/run_custom_scene.py` | `scripts/run_scene.py` | Shorter, still clear |
| `scripts/run_custom_scene.sh` | `scripts/run_scene.sh` | Matches the `.py` it wraps |

## Test File Renames

Tests follow the module they cover:

| Current | New |
|---|---|
| `tests/test_isaacsim_simulator.py` | `tests/test_simulator_adapter.py` |
| `tests/test_runtime_guards.py` | `tests/test_motion_runner.py` |
| `tests/test_recording.py` | `tests/test_viewport_capture.py` |
| `tests/test_rest_pose.py` | **DELETE** |
| `tests/test_run_custom_scene.py` | `tests/test_run_scene.py` |
| `tests/test_render_motion_video_script.py` | `tests/test_run_scene_script.py` |

### Unchanged Test Files

- `tests/test_motion_file.py`
- `tests/test_result.py`
- `tests/test_custom_scene.py`

## Deletions

| File | Reason |
|---|---|
| `src/human_motion_isaacsim/rest_pose.py` | Unused in any runtime path; only tested in isolation |
| `tests/test_rest_pose.py` | Tests the deleted module |

Any `rest_pose` references in `__init__.py` will be removed. The `restore_rest_pose` callback parameter on `MotionController.__init__()` is retained — it accepts any `Callable[[Any], None]` and does not depend on the deleted module.

## Merge: `runtime.py` + `protomotions_runtime.py` → `motion_runner.py`

Both files share imports (`checkpoint`, `motion_file`) and represent two halves of the same concern (controlling and executing motions). The merged file will contain:

1. `MotionController` (from `runtime.py`'s `ProtoMotionIsaacSimController`) — thin controller shell with `_busy` guard and `run_motion()` method (listed first as the simpler, more commonly used class)
2. `MotionRunner` (from `protomotions_runtime.py`'s `ProtoMotionsRuntime`) — standalone orchestrator with `build_standalone_runner()` and `run_standalone_motion()`

No logic changes — just two classes in one file with shared imports deduplicated.

## Import Updates

All internal imports across source, scripts, and tests will be updated to reflect new module and class names:

- `from human_motion_isaacsim.isaacsim_simulator import IsaacSimSimulator` → `from human_motion_isaacsim.simulator_adapter import SimulatorAdapter`
- `from human_motion_isaacsim.runtime import ProtoMotionIsaacSimController` → `from human_motion_isaacsim.motion_runner import MotionController`
- `from human_motion_isaacsim.protomotions_runtime import ProtoMotionsRuntime` → `from human_motion_isaacsim.motion_runner import MotionRunner`
- `from human_motion_isaacsim.recording import ...` → `from human_motion_isaacsim.viewport_capture import ...`
- `__init__.py` lazy exports updated to point to new module names and new symbol names
- `run_scene.sh` updated to call `run_scene.py`

### `__init__.py` Public API Changes

The exported symbol names change to match the new class names:

| Old Export | New Export |
|---|---|
| `ProtoMotionIsaacSimController` | `MotionController` |
| `recording` (module alias) | `viewport_capture` (module alias) |

`MotionMetadata`, `MotionRunResult`, and `load_motion_metadata` are unchanged.

### Hardcoded Path String Updates

Beyond Python imports, these files contain hardcoded path strings that must also be updated:

- `tests/test_run_scene_script.py` (currently `test_render_motion_video_script.py`): references `"scripts/run_custom_scene.sh"` and `"scripts/run_custom_scene.py"` in `Path(...)` constructors — update to `"scripts/run_scene.sh"` and `"scripts/run_scene.py"`
- `scripts/run_scene.sh` (currently `run_custom_scene.sh`): references `run_custom_scene.py` — update to `run_scene.py`
- `README.md`: references to old file/class names in code pointers and examples

These are coupled changes — the shell script rename and its test assertions must be updated in lockstep.

## Out of Scope

- No logic changes to any function or class
- No splitting of `simulator_adapter.py` (stays ~830 lines)
- No new files beyond what's listed
- No package restructuring (flat layout preserved)
