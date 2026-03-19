# `human_motion_isaacsim`

This repo contains the Isaac Sim inference path for running ProtoMotions tracking against the same `.motion` files produced by [ProtoMotions](https://github.com/NVlabs/ProtoMotions).

Clone with the bundled upstream dependency:

```bash
git lfs install
git clone --recurse-submodules <repo-url> /home/lyuxinghe/code/human_motion_isaacsim
```

If you already cloned the repo, initialize the submodule with:

```bash
git -C /home/lyuxinghe/code/human_motion_isaacsim submodule update --init --recursive
```

Start with `env/README.md` for:

- environment creation with `uv`
- TurboVNC / `DISPLAY=:1` monitor setup
- Isaac Sim monitor probing with `scripts/smoke_monitor.py`
- standalone ProtoMotions smoke runs with `scripts/smoke_motion.py`

## Run the custom scene

`scripts/run_scene.sh` is the preferred entrypoint for rendering a motion file to video. It sets the required env vars, defaults to headless mode, and then calls `scripts/run_scene.py` underneath.

Prerequisites:

- the `env/.venv` environment from `env/README.md`
- the `third_party/ProtoMotions` submodule initialized
- `OMNI_KIT_ACCEPT_EULA=YES`
- for GUI runs, a working `DISPLAY=:1` TurboVNC session as described in `env/README.md`
- `PROTOMOTIONS_ROOT` is optional if you want to override the bundled submodule checkout

Preferred headless render:

```bash
scripts/run_scene.sh \
  --motion-file assets/a_person_is_reaching_out_his_left_hand_and_walking.motion
```

Monitor-backed run on `DISPLAY=:1`:

```bash
scripts/run_scene.sh \
  --motion-file assets/a_person_is_reaching_out_his_left_hand_and_walking.motion \
  --headless false \
  --display :1
```

Override the checkpoint, output path, or marker rendering:

```bash
scripts/run_scene.sh \
  --motion-file assets/a_person_is_reaching_out_his_left_hand_and_walking.motion \
  --checkpoint /home/lyuxinghe/code/human_motion_isaacsim/third_party/ProtoMotions/data/pretrained_models/motion_tracker/smpl/last.ckpt \
  --video-output /home/lyuxinghe/code/human_motion_isaacsim/output/custom_scene.mp4 \
  --reference-markers false
```

Outputs:

- the MP4 is written to the path passed to `--video-output`
- if `--video-output` is omitted, the wrapper writes `output/<motion-file-stem>.mp4`
- extracted frames are written to `<video-output without .mp4>/frames`
- in headless video runs, the red reference markers are rendered into the saved frames so you can compare the tracked body against the input motion
- pass `--reference-markers false` when you want a clean render without the red target spheres

## Known issues

### Humanoid teleports on episode reset

ProtoMotions calls `env.reset()` whenever an episode boundary is reached (the `dones` tensor has nonzero entries). The reset writes a new root pose via `SimulatorAdapter._set_simulator_env_state()`, which teleports the humanoid to a different world-space position and orientation. In `scripts/run_scene.py` the custom-scene path compensates by calling `_align_scene_to_humanoid_root()` after every reset to reposition the ground plane and static objects, but the standalone smoke path in `MotionRunner.run_standalone_motion()` does not — if a reset fires mid-clip, the humanoid will appear to jump to a new location while the camera follows but the scene stays behind.

Practically this means:

- Short motions that finish within one episode are unaffected.
- Longer motions or motions with early termination will show a visible snap when the humanoid is respawned.
- The custom-scene script handles this; the standalone smoke runner does not.

### Non-headless mode (`--headless false`) produces degraded tracking

Running with `--headless false` (monitor-backed / GUI mode) results in the humanoid failing to accurately track the input motion. The root cause is a render-timing difference in the two capture pipelines:

- **Headless:** `_capture_headless_follow_camera_rgba()` calls `self._world.render()` once after physics is done, then reads the frame from a dedicated Isaac Sim Camera sensor. No extra state is advanced.
- **Non-headless:** `_physics_step()` calls `self._world.render()` at the end of each step (inside `env.step()`), and then `_write_viewport_to_file()` calls `self._rep_module.orchestrator.step()` to capture from Replicator's annotator pipeline. The Replicator orchestrator step can advance internal state that desyncs the articulation readback from the policy's expectations.

Additionally, `_physics_step()` injects an extra `self._world.render()` call only when `headless=False` (line 131 of `simulator_adapter.py`), which does not happen in headless mode. This extra render call changes the timing of physics readback relative to the policy step.

Use `--headless true` (the default) for reliable motion tracking and video output.

Key runtime code:

- `src/human_motion_isaacsim/motion_runner.py`
- `src/human_motion_isaacsim/viewport_capture.py`

The intended root-level layout is:

- `src/human_motion_isaacsim/` for the controller runtime
- `third_party/ProtoMotions/` for the upstream submodule checkout
- `scripts/` for smoke and monitor-entry scripts
- `tests/` for unit and guard coverage
- `env/` for the `uv` bootstrap and pinned dependency setup
- `plans/` and `specs/` for the design and implementation artifacts


## TODOs:
create a git workflow that will deploy this repo to gcp python package and test it end-to-end (python registry)