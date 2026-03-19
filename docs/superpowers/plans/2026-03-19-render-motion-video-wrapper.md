# Render Motion Video Wrapper Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a bash wrapper that sets the required render environment and runs `scripts/run_custom_scene.py` with sensible defaults for rendering a motion file to video.

**Architecture:** Keep all simulation behavior in the existing Python entrypoint and implement only shell-side argument parsing, default resolution, validation, and env export in a new wrapper script. Cover the wrapper with subprocess tests that run against a temporary fake repo layout so the shell behavior is verified without launching Isaac Sim.

**Tech Stack:** Bash, pytest, subprocess, existing `scripts/run_custom_scene.py`

---

## Chunk 1: Test-First Wrapper Coverage

### Task 1: Add failing shell-wrapper tests

**Files:**
- Create: `tests/test_render_motion_video_script.py`

- [ ] **Step 1: Write the failing test**

Add subprocess coverage for:
- default headless mode
- repo-local default checkpoint
- default output path from motion stem
- non-headless mode setting `DISPLAY=:1`
- reference-marker flag forwarding

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_render_motion_video_script.py -v`
Expected: FAIL because `scripts/render_motion_video.sh` does not exist yet.

### Task 2: Implement the wrapper

**Files:**
- Create: `scripts/render_motion_video.sh`

- [ ] **Step 3: Write minimal implementation**

Implement:
- strict shell mode
- repo-root discovery from script location
- argument parsing for `--motion-file`, `--headless`, `--checkpoint`, `--reference-markers`, `--video-output`, `--display`
- default path resolution
- env export
- validation of required files
- forwarding into `env/.venv/bin/python scripts/run_custom_scene.py`

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_render_motion_video_script.py -v`
Expected: PASS

## Chunk 2: Docs and Guard Updates

### Task 3: Update docs and existing assertions

**Files:**
- Modify: `README.md`
- Modify: `env/README.md`
- Modify: `tests/test_result.py`

- [ ] **Step 5: Update docs**

Document the new wrapper as the preferred command for rendering a motion file and keep `run_custom_scene.py` as the underlying entrypoint.

- [ ] **Step 6: Update guard assertions**

Update tests that assert the ProtoMotions submodule URL and docs content so they match the current SSH submodule remote and the new wrapper command.

- [ ] **Step 7: Run targeted verification**

Run: `pytest tests/test_render_motion_video_script.py tests/test_result.py -v`
Expected: PASS

## Chunk 3: Final Verification

### Task 4: Verify broader coverage

**Files:**
- Verify only

- [ ] **Step 8: Run related full checks**

Run: `pytest tests/test_run_custom_scene.py tests/test_render_motion_video_script.py tests/test_result.py -v`
Expected: PASS

- [ ] **Step 9: Make the wrapper executable**

Run: `chmod +x scripts/render_motion_video.sh`
Expected: script is executable for direct shell use
