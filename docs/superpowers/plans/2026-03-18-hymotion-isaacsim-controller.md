# HY-Motion Isaac Sim Controller Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a persistent Isaac Sim resident controller under `hymotion_isaacsim` that can bind to a fixed-path SMPL humanoid, execute ProtoMotions `.motion` files on demand, save MP4 output with viewport parity to `hymotion_isaaclab`, and restore the humanoid to default rest pose at the same world position after each run.

**Architecture:** The implementation starts by pinning a known-good `uv` environment that matches the existing local `env_isaaclab` stack, then builds a small `hymotion_isaacsim` Python package around three boundaries: stage binding, ProtoMotions checkpoint/runtime reuse, and blocking motion execution with recording/rest-pose cleanup. A standalone smoke script is built early to de-risk Isaac Sim startup and humanoid loading before the env-bound controller is wired for repeated runs in one session.

**Tech Stack:** Python 3.11, `uv`, Isaac Sim 5.1.0 pip packages, Isaac Lab 2.3.0, ProtoMotions, PyTorch, pytest

---

## File Structure

Planned files and responsibilities:

- Create: `hymotion_isaacsim/pyproject.toml`
  Purpose: package metadata, editable install, test extras, pinned package entrypoints.
- Create: `hymotion_isaacsim/env/README.md`
  Purpose: document the pinned `uv` environment and required environment variables.
- Create: `hymotion_isaacsim/env/install.sh`
  Purpose: one-shot reproducible `uv` setup matching the known-good local Isaac Sim / Isaac Lab versions.
- Create: `hymotion_isaacsim/env/requirements.lock`
  Purpose: pinned runtime dependencies for the controller package.
- Create: `hymotion_isaacsim/src/hymotion_isaacsim/__init__.py`
  Purpose: package exports.
- Create: `hymotion_isaacsim/src/hymotion_isaacsim/result.py`
  Purpose: result dataclass returned by motion execution.
- Create: `hymotion_isaacsim/src/hymotion_isaacsim/motion_file.py`
  Purpose: validate `.motion` paths and load motion metadata needed for duration/step planning.
- Create: `hymotion_isaacsim/src/hymotion_isaacsim/rest_pose.py`
  Purpose: rest-pose state capture and restoration utilities.
- Create: `hymotion_isaacsim/src/hymotion_isaacsim/binding.py`
  Purpose: fixed-path humanoid binding and validation for `v1`.
- Create: `hymotion_isaacsim/src/hymotion_isaacsim/checkpoint.py`
  Purpose: load and validate ProtoMotions checkpoint plus frozen inference config once.
- Create: `hymotion_isaacsim/src/hymotion_isaacsim/recording.py`
  Purpose: viewport frame capture and MP4 compilation with the same codec/pixel settings as `hymotion_isaaclab`.
- Create: `hymotion_isaacsim/src/hymotion_isaacsim/protomotions_runtime.py`
  Purpose: thin adapter around reused ProtoMotions agent/model/config pieces needed for inference-time action generation.
- Create: `hymotion_isaacsim/src/hymotion_isaacsim/runtime.py`
  Purpose: persistent `ProtoMotionIsaacSimController` public API and run-state guard.
- Create: `hymotion_isaacsim/scripts/smoke_run_motion.py`
  Purpose: standalone smoke harness for empty-scene startup, humanoid loading, single-motion execution, and recording.
- Create: `hymotion_isaacsim/tests/test_motion_file.py`
  Purpose: validate motion file loading and metadata handling.
- Create: `hymotion_isaacsim/tests/test_result.py`
  Purpose: validate result object semantics.
- Create: `hymotion_isaacsim/tests/test_rest_pose.py`
  Purpose: validate root-position-preserving rest-pose logic.
- Create: `hymotion_isaacsim/tests/test_runtime_guards.py`
  Purpose: validate busy-state and error-surface behavior.

## Chunk 1: Environment And Package Skeleton

### Task 1: Create the Python package scaffold

**Files:**
- Create: `hymotion_isaacsim/pyproject.toml`
- Create: `hymotion_isaacsim/src/hymotion_isaacsim/__init__.py`
- Test: `hymotion_isaacsim/tests/test_result.py`

- [ ] **Step 1: Write the failing package import test**

```python
def test_package_exports_controller_symbol():
    import hymotion_isaacsim

    assert hasattr(hymotion_isaacsim, "ProtoMotionIsaacSimController")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/lyuxinghe/code/hymotion_isaacsim && pytest tests/test_result.py::test_package_exports_controller_symbol -v`
Expected: FAIL with `ModuleNotFoundError` or missing symbol.

- [ ] **Step 3: Write minimal package metadata and export stub**

```toml
[project]
name = "hymotion-isaacsim"
version = "0.1.0"
requires-python = ">=3.11,<3.12"
```

```python
class ProtoMotionIsaacSimController:
    pass
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/lyuxinghe/code/hymotion_isaacsim && pytest tests/test_result.py::test_package_exports_controller_symbol -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git -C /home/lyuxinghe/code add hymotion_isaacsim/pyproject.toml hymotion_isaacsim/src/hymotion_isaacsim/__init__.py hymotion_isaacsim/tests/test_result.py
git -C /home/lyuxinghe/code commit -m "feat: scaffold hymotion isaac sim package"
```

### Task 2: Pin the known-good `uv` environment

**Files:**
- Create: `hymotion_isaacsim/env/install.sh`
- Create: `hymotion_isaacsim/env/requirements.lock`
- Create: `hymotion_isaacsim/env/README.md`
- Test: `hymotion_isaacsim/scripts/smoke_run_motion.py`

- [ ] **Step 1: Write the failing environment documentation check**

```python
from pathlib import Path


def test_env_docs_reference_known_good_versions():
    text = Path("env/README.md").read_text()
    assert "isaacsim==5.1.0.0" in text
    assert "isaaclab==2.3.0" in text
    assert "Python 3.11" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/lyuxinghe/code/hymotion_isaacsim && pytest tests/test_result.py::test_env_docs_reference_known_good_versions -v`
Expected: FAIL because `env/README.md` does not exist yet.

- [ ] **Step 3: Write the pinned env files**

```bash
uv venv env/.venv --python 3.11
uv pip install --python env/.venv/bin/python --index-url https://download.pytorch.org/whl/cu128 torch==2.7.0
uv pip install --python env/.venv/bin/python --extra-index-url https://pypi.nvidia.com "isaacsim[all,extscache]==5.1.0.0" "isaaclab==2.3.0"
```

Add `OMNI_KIT_ACCEPT_EULA=YES` and the editable install command for the local package to `env/README.md`.

- [ ] **Step 4: Run the check and a non-GUI smoke import**

Run: `cd /home/lyuxinghe/code/hymotion_isaacsim && pytest tests/test_result.py::test_env_docs_reference_known_good_versions -v`
Expected: PASS

Run: `cd /home/lyuxinghe/code/hymotion_isaacsim && env/.venv/bin/python -c "import isaacsim, isaaclab; print('ok')"`
Expected: prints `ok`

- [ ] **Step 5: Commit**

```bash
git -C /home/lyuxinghe/code add hymotion_isaacsim/env/install.sh hymotion_isaacsim/env/requirements.lock hymotion_isaacsim/env/README.md hymotion_isaacsim/tests/test_result.py
git -C /home/lyuxinghe/code commit -m "build: pin uv Isaac Sim runtime"
```

### Task 3: Add result and motion metadata helpers

**Files:**
- Create: `hymotion_isaacsim/src/hymotion_isaacsim/result.py`
- Create: `hymotion_isaacsim/src/hymotion_isaacsim/motion_file.py`
- Create: `hymotion_isaacsim/tests/test_motion_file.py`
- Modify: `hymotion_isaacsim/src/hymotion_isaacsim/__init__.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_load_motion_metadata_reads_fps_and_frame_count(tmp_path):
    ...
    assert meta.fps == 30
    assert meta.num_frames == 90
    assert meta.duration_seconds == 3.0


def test_load_motion_metadata_rejects_non_motion_suffix(tmp_path):
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/lyuxinghe/code/hymotion_isaacsim && pytest tests/test_motion_file.py -v`
Expected: FAIL with missing module/function errors.

- [ ] **Step 3: Write minimal implementation**

```python
@dataclass(slots=True)
class MotionMetadata:
    path: Path
    fps: int
    num_frames: int

    @property
    def duration_seconds(self) -> float:
        return self.num_frames / self.fps
```

Use `torch.load(..., weights_only=False)` and validate the file suffix is `.motion`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/lyuxinghe/code/hymotion_isaacsim && pytest tests/test_motion_file.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git -C /home/lyuxinghe/code add hymotion_isaacsim/src/hymotion_isaacsim/result.py hymotion_isaacsim/src/hymotion_isaacsim/motion_file.py hymotion_isaacsim/tests/test_motion_file.py hymotion_isaacsim/src/hymotion_isaacsim/__init__.py
git -C /home/lyuxinghe/code commit -m "feat: add motion metadata helpers"
```

## Chunk 2: Binding And Controller Core

### Task 4: Implement fixed-path humanoid binding and validation

**Files:**
- Create: `hymotion_isaacsim/src/hymotion_isaacsim/binding.py`
- Create: `hymotion_isaacsim/tests/test_runtime_guards.py`

- [ ] **Step 1: Write the failing tests for fixed-path validation**

```python
def test_validate_humanoid_path_rejects_missing_prim():
    ...


def test_validate_humanoid_binding_rejects_incompatible_body_layout():
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/lyuxinghe/code/hymotion_isaacsim && pytest tests/test_runtime_guards.py -k humanoid -v`
Expected: FAIL with missing module/function errors.

- [ ] **Step 3: Implement the binding helper**

```python
@dataclass(slots=True)
class BoundHumanoid:
    prim_path: str
    body_names: tuple[str, ...]
    joint_names: tuple[str, ...]
```

Validate the prim exists, is an articulation, and matches the expected SMPL body/joint naming. Add `NOTE(v2)` comments for the fixed-path assumption.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/lyuxinghe/code/hymotion_isaacsim && pytest tests/test_runtime_guards.py -k humanoid -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git -C /home/lyuxinghe/code add hymotion_isaacsim/src/hymotion_isaacsim/binding.py hymotion_isaacsim/tests/test_runtime_guards.py
git -C /home/lyuxinghe/code commit -m "feat: add fixed path humanoid binding"
```

### Task 5: Implement checkpoint/config loading

**Files:**
- Create: `hymotion_isaacsim/src/hymotion_isaacsim/checkpoint.py`
- Modify: `hymotion_isaacsim/tests/test_runtime_guards.py`

- [ ] **Step 1: Write the failing checkpoint tests**

```python
def test_load_tracker_checkpoint_requires_resolved_configs():
    ...


def test_load_tracker_checkpoint_returns_expected_targets():
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/lyuxinghe/code/hymotion_isaacsim && pytest tests/test_runtime_guards.py -k checkpoint -v`
Expected: FAIL with missing helper errors.

- [ ] **Step 3: Implement minimal checkpoint loader**

```python
@dataclass(slots=True)
class TrackerAssets:
    checkpoint_path: Path
    resolved_config_path: Path
    env_config: object
    agent_config: object
```

Load `resolved_configs_inference.pt`, validate required keys, and normalize the robot asset root path for ProtoMotions assets.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/lyuxinghe/code/hymotion_isaacsim && pytest tests/test_runtime_guards.py -k checkpoint -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git -C /home/lyuxinghe/code add hymotion_isaacsim/src/hymotion_isaacsim/checkpoint.py hymotion_isaacsim/tests/test_runtime_guards.py
git -C /home/lyuxinghe/code commit -m "feat: add tracker checkpoint loader"
```

### Task 6: Implement rest-pose utilities

**Files:**
- Create: `hymotion_isaacsim/src/hymotion_isaacsim/rest_pose.py`
- Create: `hymotion_isaacsim/tests/test_rest_pose.py`

- [ ] **Step 1: Write the failing rest-pose tests**

```python
def test_restore_rest_pose_preserves_root_translation():
    ...


def test_restore_rest_pose_zeros_joint_and_root_velocities():
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/lyuxinghe/code/hymotion_isaacsim && pytest tests/test_rest_pose.py -v`
Expected: FAIL with missing module/function errors.

- [ ] **Step 3: Implement minimal rest-pose helpers**

```python
@dataclass(slots=True)
class RestPoseState:
    root_position: torch.Tensor
    root_orientation: torch.Tensor
    joint_positions: torch.Tensor
```

Provide helpers to capture defaults and compose a rest-pose target at a supplied world position.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/lyuxinghe/code/hymotion_isaacsim && pytest tests/test_rest_pose.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git -C /home/lyuxinghe/code add hymotion_isaacsim/src/hymotion_isaacsim/rest_pose.py hymotion_isaacsim/tests/test_rest_pose.py
git -C /home/lyuxinghe/code commit -m "feat: add rest pose utilities"
```

### Task 7: Build the persistent controller shell

**Files:**
- Create: `hymotion_isaacsim/src/hymotion_isaacsim/runtime.py`
- Modify: `hymotion_isaacsim/src/hymotion_isaacsim/__init__.py`
- Modify: `hymotion_isaacsim/tests/test_runtime_guards.py`

- [ ] **Step 1: Write the failing controller lifecycle tests**

```python
def test_controller_initialization_binds_humanoid_and_checkpoint():
    ...


def test_controller_rejects_overlapping_run_requests():
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/lyuxinghe/code/hymotion_isaacsim && pytest tests/test_runtime_guards.py -k controller -v`
Expected: FAIL with missing class or lifecycle methods.

- [ ] **Step 3: Implement the controller shell**

```python
class ProtoMotionIsaacSimController:
    def __init__(...):
        self._busy = False

    def run_motion(...):
        raise NotImplementedError
```

Wire in the binding and checkpoint loader, and surface a clear busy-state guard before any runtime stepping work is added.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/lyuxinghe/code/hymotion_isaacsim && pytest tests/test_runtime_guards.py -k controller -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git -C /home/lyuxinghe/code add hymotion_isaacsim/src/hymotion_isaacsim/runtime.py hymotion_isaacsim/src/hymotion_isaacsim/__init__.py hymotion_isaacsim/tests/test_runtime_guards.py
git -C /home/lyuxinghe/code commit -m "feat: add controller lifecycle shell"
```

## Chunk 3: ProtoMotions Runtime, Recording, And Smoke Validation

### Task 8: Reuse ProtoMotions inference-time runtime pieces

**Files:**
- Create: `hymotion_isaacsim/src/hymotion_isaacsim/protomotions_runtime.py`
- Modify: `hymotion_isaacsim/tests/test_runtime_guards.py`

- [ ] **Step 1: Write the failing runtime-adapter tests**

```python
def test_build_runtime_reuses_motion_tracker_agent_config():
    ...


def test_runtime_plans_steps_from_motion_metadata():
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/lyuxinghe/code/hymotion_isaacsim && pytest tests/test_runtime_guards.py -k runtime -v`
Expected: FAIL with missing adapter methods.

- [ ] **Step 3: Implement the thin ProtoMotions runtime adapter**

```python
class ProtoMotionsRuntime:
    def build_agent(self):
        ...

    def plan_num_steps(self, motion_metadata):
        return int(motion_metadata.duration_seconds * 30)
```

Reuse the checkpoint's frozen configs, model loading, and inference-time action path. Keep adapter boundaries narrow and document the parts that still depend on fixed-path SMPL assumptions.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/lyuxinghe/code/hymotion_isaacsim && pytest tests/test_runtime_guards.py -k runtime -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git -C /home/lyuxinghe/code add hymotion_isaacsim/src/hymotion_isaacsim/protomotions_runtime.py hymotion_isaacsim/tests/test_runtime_guards.py
git -C /home/lyuxinghe/code commit -m "feat: add protomotions runtime adapter"
```

### Task 9: Implement viewport recording parity

**Files:**
- Create: `hymotion_isaacsim/src/hymotion_isaacsim/recording.py`
- Modify: `hymotion_isaacsim/tests/test_runtime_guards.py`

- [ ] **Step 1: Write the failing recording tests**

```python
def test_compile_video_uses_hymotion_codec_settings(tmp_path):
    ...


def test_frame_path_sequence_is_zero_padded(tmp_path):
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/lyuxinghe/code/hymotion_isaacsim && pytest tests/test_runtime_guards.py -k recording -v`
Expected: FAIL with missing module/function errors.

- [ ] **Step 3: Implement recording helpers**

```python
def compile_video(frame_paths: list[Path], video_path: Path, fps: int = 30) -> None:
    ...
```

Mirror the `moviepy` settings from `hymotion_isaaclab/scripts/run_tracking.py` and add a wrapper for active-viewport capture calls.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/lyuxinghe/code/hymotion_isaacsim && pytest tests/test_runtime_guards.py -k recording -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git -C /home/lyuxinghe/code add hymotion_isaacsim/src/hymotion_isaacsim/recording.py hymotion_isaacsim/tests/test_runtime_guards.py
git -C /home/lyuxinghe/code commit -m "feat: add viewport recording helpers"
```

### Task 10: Wire `run_motion(...)` end to end

**Files:**
- Modify: `hymotion_isaacsim/src/hymotion_isaacsim/runtime.py`
- Modify: `hymotion_isaacsim/src/hymotion_isaacsim/recording.py`
- Modify: `hymotion_isaacsim/src/hymotion_isaacsim/rest_pose.py`
- Modify: `hymotion_isaacsim/src/hymotion_isaacsim/protomotions_runtime.py`
- Modify: `hymotion_isaacsim/tests/test_runtime_guards.py`

- [ ] **Step 1: Write the failing end-to-end controller tests**

```python
def test_run_motion_returns_success_result_for_valid_motion(tmp_path):
    ...


def test_run_motion_restores_rest_pose_on_runtime_error(tmp_path):
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/lyuxinghe/code/hymotion_isaacsim && pytest tests/test_runtime_guards.py -k run_motion -v`
Expected: FAIL with `NotImplementedError` or missing behavior.

- [ ] **Step 3: Implement the blocking motion loop**

```python
def run_motion(self, motion_file: str, video_output: str | None = None):
    self._busy = True
    try:
        ...
    finally:
        self._busy = False
```

Validate motion metadata, step the runtime, capture frames, compile the video, restore rest pose, and return a populated result object.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/lyuxinghe/code/hymotion_isaacsim && pytest tests/test_runtime_guards.py -k run_motion -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git -C /home/lyuxinghe/code add hymotion_isaacsim/src/hymotion_isaacsim/runtime.py hymotion_isaacsim/src/hymotion_isaacsim/recording.py hymotion_isaacsim/src/hymotion_isaacsim/rest_pose.py hymotion_isaacsim/src/hymotion_isaacsim/protomotions_runtime.py hymotion_isaacsim/tests/test_runtime_guards.py
git -C /home/lyuxinghe/code commit -m "feat: implement blocking motion execution"
```

### Task 11: Add the standalone smoke script

**Files:**
- Create: `hymotion_isaacsim/scripts/smoke_run_motion.py`
- Modify: `hymotion_isaacsim/env/README.md`

- [ ] **Step 1: Write the failing smoke-script documentation check**

```python
def test_env_readme_contains_smoke_command():
    text = Path("env/README.md").read_text()
    assert "scripts/smoke_run_motion.py" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/lyuxinghe/code/hymotion_isaacsim && pytest tests/test_result.py::test_env_readme_contains_smoke_command -v`
Expected: FAIL because the command is not documented yet.

- [ ] **Step 3: Implement the smoke script and docs**

```bash
env/.venv/bin/python scripts/smoke_run_motion.py \
  --checkpoint /home/lyuxinghe/code/ProtoMotions/data/pretrained_models/motion_tracker/smpl/last.ckpt \
  --motion-file /home/lyuxinghe/code/hymotion_isaaclab/output/a_person_is_reaching_out_his_left_hand_and_walking_000.motion \
  --video-output output/smoke.mp4
```

The script should create an empty scene, load the default SMPL humanoid, instantiate the controller, run one motion, and print the saved MP4 path.

- [ ] **Step 4: Run the documentation check and smoke command**

Run: `cd /home/lyuxinghe/code/hymotion_isaacsim && pytest tests/test_result.py::test_env_readme_contains_smoke_command -v`
Expected: PASS

Run: `cd /home/lyuxinghe/code/hymotion_isaacsim && env/.venv/bin/python scripts/smoke_run_motion.py --help`
Expected: PASS and print CLI usage.

- [ ] **Step 5: Commit**

```bash
git -C /home/lyuxinghe/code add hymotion_isaacsim/scripts/smoke_run_motion.py hymotion_isaacsim/env/README.md hymotion_isaacsim/tests/test_result.py
git -C /home/lyuxinghe/code commit -m "feat: add Isaac Sim smoke script"
```

### Task 12: Final verification

**Files:**
- Modify: `hymotion_isaacsim/env/README.md`

- [ ] **Step 1: Run the fast unit test suite**

Run: `cd /home/lyuxinghe/code/hymotion_isaacsim && pytest tests/test_motion_file.py tests/test_result.py tests/test_rest_pose.py tests/test_runtime_guards.py -v`
Expected: PASS

- [ ] **Step 2: Run the smoke script against a real `.motion` file**

Run: `cd /home/lyuxinghe/code/hymotion_isaacsim && env/.venv/bin/python scripts/smoke_run_motion.py --checkpoint /home/lyuxinghe/code/ProtoMotions/data/pretrained_models/motion_tracker/smpl/last.ckpt --motion-file /home/lyuxinghe/code/hymotion_isaaclab/output/a_person_is_reaching_out_his_left_hand_and_walking_000.motion --video-output output/a_person_is_reaching_out_his_left_hand_and_walking_000.mp4`
Expected: PASS, saved MP4 path printed, controller returns success.

- [ ] **Step 3: Update the env docs with any final runtime caveats**

Document:
1. Fixed prim path requirement.
2. Known-good version pins.
3. `OMNI_KIT_ACCEPT_EULA=YES`
4. Production expectation that external processes do not interfere while `run_motion(...)` owns stepping.

- [ ] **Step 4: Commit the final verified state**

```bash
git -C /home/lyuxinghe/code add hymotion_isaacsim
git -C /home/lyuxinghe/code commit -m "feat: add Isaac Sim ProtoMotions motion controller"
```
