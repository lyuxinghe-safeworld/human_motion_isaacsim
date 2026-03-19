# `protomotions_isaacsim`

This repo contains the Isaac Sim inference path for running ProtoMotions tracking against the same `.motion` files produced by [ProtoMotions](https://github.com/NVlabs/ProtoMotions).

Start with `env/README.md` for:

- environment creation with `uv`
- TurboVNC / `DISPLAY=:1` monitor setup
- Isaac Sim monitor probing with `scripts/test_isaacsim_monitor.py`
- standalone ProtoMotions smoke runs with `scripts/smoke_run_motion.py`

## Run the custom scene

`scripts/run_custom_scene.py` is the main entrypoint for rendering the Isaac Sim custom scene with the ProtoMotions tracker.

Prerequisites:

- the `env/.venv` environment from `env/README.md`
- `PROTOMOTIONS_ROOT=/home/lyuxinghe/code/ProtoMotions`
- `OMNI_KIT_ACCEPT_EULA=YES`
- for GUI runs, a working `DISPLAY=:1` TurboVNC session as described in `env/README.md`

Interactive run on the VNC display:

```bash
DISPLAY=:1 \
OMNI_KIT_ACCEPT_EULA=YES \
PROTOMOTIONS_ROOT=/home/lyuxinghe/code/ProtoMotions \
LD_LIBRARY_PATH="$HOME/code/env_isaaclab/lib/python3.11/site-packages/isaacsim/kit/extscore/omni.client.lib/bin:${LD_LIBRARY_PATH:-}" \
NCCL_IB_DISABLE=1 \
NCCL_NET=Socket \
MASTER_ADDR=127.0.0.1 \
MASTER_PORT=29500 \
env/.venv/bin/python scripts/run_custom_scene.py \
  --checkpoint /home/lyuxinghe/code/ProtoMotions/data/pretrained_models/motion_tracker/smpl/last.ckpt \
  --motion-file /home/lyuxinghe/code/hymotion_isaaclab/output/a_person_is_reaching_out_his_left_hand_and_walking_000.motion
```

Headless render to MP4:

```bash
OMNI_KIT_ACCEPT_EULA=YES \
PROTOMOTIONS_ROOT=/home/lyuxinghe/code/ProtoMotions \
LD_LIBRARY_PATH="$HOME/code/env_isaaclab/lib/python3.11/site-packages/isaacsim/kit/extscore/omni.client.lib/bin:${LD_LIBRARY_PATH:-}" \
NCCL_IB_DISABLE=1 \
NCCL_NET=Socket \
MASTER_ADDR=127.0.0.1 \
MASTER_PORT=29500 \
env/.venv/bin/python scripts/run_custom_scene.py \
  --checkpoint /home/lyuxinghe/code/ProtoMotions/data/pretrained_models/motion_tracker/smpl/last.ckpt \
  --motion-file /home/lyuxinghe/code/hymotion_isaaclab/output/a_person_is_reaching_out_his_left_hand_and_walking_000.motion \
  --video-output output/custom_scene.mp4 \
  --headless
```

Outputs:

- the MP4 is written to the path passed to `--video-output`
- extracted frames are written to `<video-output without .mp4>/frames`
- in headless video runs, the red reference markers are rendered into the saved frames so you can compare the tracked body against the input motion
- add `--no-reference-markers` when you want a clean render without the red target spheres

Key runtime code:

- `src/hymotion_isaacsim/protomotions_runtime.py`
- `src/hymotion_isaacsim/runtime.py`
- `src/hymotion_isaacsim/recording.py`

The intended root-level layout is:

- `src/hymotion_isaacsim/` for the controller runtime
- `scripts/` for smoke and monitor-entry scripts
- `tests/` for unit and guard coverage
- `env/` for the `uv` bootstrap and pinned dependency setup
- `plans/` and `specs/` for the design and implementation artifacts
