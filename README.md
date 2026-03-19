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
- Isaac Sim monitor probing with `scripts/test_isaacsim_monitor.py`
- standalone ProtoMotions smoke runs with `scripts/smoke_run_motion.py`

## Run the custom scene

`scripts/run_custom_scene.sh` is the preferred entrypoint for rendering a motion file to video. It sets the required env vars, defaults to headless mode, and then calls `scripts/run_custom_scene.py` underneath.

Prerequisites:

- the `env/.venv` environment from `env/README.md`
- the `third_party/ProtoMotions` submodule initialized
- `OMNI_KIT_ACCEPT_EULA=YES`
- for GUI runs, a working `DISPLAY=:1` TurboVNC session as described in `env/README.md`
- `PROTOMOTIONS_ROOT` is optional if you want to override the bundled submodule checkout

Preferred headless render:

```bash
scripts/run_custom_scene.sh \
  --motion-file assets/a_person_is_reaching_out_his_left_hand_and_walking.motion
```

Monitor-backed run on `DISPLAY=:1`:

```bash
scripts/run_custom_scene.sh \
  --motion-file assets/a_person_is_reaching_out_his_left_hand_and_walking.motion \
  --headless false \
  --display :1
```

Override the checkpoint, output path, or marker rendering:

```bash
scripts/run_custom_scene.sh \
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

Key runtime code:

- `src/human_motion_isaacsim/protomotions_runtime.py`
- `src/human_motion_isaacsim/runtime.py`
- `src/human_motion_isaacsim/recording.py`

The intended root-level layout is:

- `src/human_motion_isaacsim/` for the controller runtime
- `third_party/ProtoMotions/` for the upstream submodule checkout
- `scripts/` for smoke and monitor-entry scripts
- `tests/` for unit and guard coverage
- `env/` for the `uv` bootstrap and pinned dependency setup
- `plans/` and `specs/` for the design and implementation artifacts
