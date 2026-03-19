# Human Motion Isaac Sim Restructure Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename the repo and package to `human_motion_isaacsim`, vendor ProtoMotions as `third_party/ProtoMotions`, switch install/docs to Python `venv`, and verify the custom-scene video path still works.

**Architecture:** The change is mostly a repo-structure migration. First lock in tests around the rename-sensitive behavior, then rename the Python package/import paths, then wire the repo-local ProtoMotions submodule into runtime and install scripts, and finally verify with unit tests plus a real render command from the renamed checkout.

**Tech Stack:** Python 3.11, setuptools editable installs, Python `venv`, Isaac Sim 5.1.0.0, Isaac Lab 2.3.0, PyTorch 2.7.0, git submodules

---

## Chunk 1: Rename-Sensitive Tests

### Task 1: Add tests for the new repo-local ProtoMotions resolution and `venv` docs

**Files:**
- Modify: `tests/test_runtime_guards.py`
- Modify: `tests/test_result.py`

- [ ] **Step 1: Write failing tests for new path resolution behavior**

Add tests that assert ProtoMotions discovery prefers a repo-local `third_party/ProtoMotions` checkout when explicit env vars are unset.

- [ ] **Step 2: Run targeted tests to verify they fail**

Run: `python -m pytest tests/test_runtime_guards.py tests/test_result.py -k 'protomotions or env_readme or readme' -v`

Expected: FAIL because code and docs still point to the old naming and old install model.

- [ ] **Step 3: Write minimal code and doc updates later to satisfy those tests**

Reserved for later tasks after the tests are in place.

## Chunk 2: Rename Repo and Python Package

### Task 2: Rename the repo checkout and package imports

**Files:**
- Modify: `pyproject.toml`
- Modify: `scripts/run_custom_scene.py`
- Modify: `scripts/smoke_run_motion.py`
- Modify: `scripts/test_isaacsim_monitor.py`
- Modify: `tests/*.py`
- Move: `src/hymotion_isaacsim` -> `src/human_motion_isaacsim`

- [ ] **Step 1: Rename checkout directory on disk**

Run: `mv /home/lyuxinghe/code/protomotions_isaacsim /home/lyuxinghe/code/human_motion_isaacsim`

- [ ] **Step 2: Verify git status from the renamed checkout**

Run: `git status --short`

Expected: moved paths and pending modifications, no missing `.git`.

- [ ] **Step 3: Rename the package directory and imports**

Update the import graph from `hymotion_isaacsim` to `human_motion_isaacsim` and rename the distribution to `human-motion-isaacsim`.

- [ ] **Step 4: Run targeted package/import tests**

Run: `python -m pytest tests/test_result.py tests/test_motion_file.py tests/test_rest_pose.py tests/test_custom_scene.py tests/test_run_custom_scene.py tests/test_runtime_guards.py -v`

Expected: PASS for rename-related coverage.

## Chunk 3: Add ProtoMotions Submodule and Wiring

### Task 3: Add the `third_party/ProtoMotions` submodule and runtime/install integration

**Files:**
- Create: `.gitmodules`
- Modify: `env/install.sh`
- Modify: `env/README.md`
- Modify: `README.md`
- Modify: `src/human_motion_isaacsim/protomotions_path.py`
- Modify: `src/human_motion_isaacsim/checkpoint.py`

- [ ] **Step 1: Add the git submodule**

Run: `git submodule add git@github.com:NVlabs/ProtoMotions.git third_party/ProtoMotions`

- [ ] **Step 2: Update runtime discovery and install script**

Make `third_party/ProtoMotions` the default local path while preserving env-var overrides.

- [ ] **Step 3: Update docs to use `venv` consistently**

Remove `uv` instructions and replace them with Python `venv` and `pip` commands.

- [ ] **Step 4: Run tests that cover the path and doc changes**

Run: `python -m pytest tests/test_runtime_guards.py tests/test_result.py -v`

Expected: PASS.

## Chunk 4: Install and End-to-End Verification

### Task 4: Build the environment and generate a real video

**Files:**
- Modify: `README.md` if command details need final correction
- Modify: `env/README.md` if install steps differ from reality

- [ ] **Step 1: Install the environment from the renamed checkout**

Run: `./env/install.sh`

- [ ] **Step 2: Run the full test suite in the installed environment**

Run: `env/.venv/bin/python -m pytest tests -v`

- [ ] **Step 3: Generate the video with `run_custom_scene.py`**

Run: `OMNI_KIT_ACCEPT_EULA=YES ... env/.venv/bin/python scripts/run_custom_scene.py --checkpoint /home/lyuxinghe/code/human_motion_isaacsim/third_party/ProtoMotions/data/pretrained_models/motion_tracker/smpl/last.ckpt --motion-file /home/lyuxinghe/code/hymotion_isaaclab/output/a_person_is_reaching_out_his_left_hand_and_walking_000.motion --video-output output/custom_scene.mp4 --headless`

- [ ] **Step 4: Verify the video file exists**

Run: `ls -lh output/custom_scene.mp4`

Expected: file exists with non-zero size.
