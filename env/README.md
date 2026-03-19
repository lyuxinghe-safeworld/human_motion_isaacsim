# Environment Setup

This repo provides a standalone Isaac Sim + ProtoMotions smoke path that mirrors the monitor-backed workflow in `/home/lyuxinghe/code/hymotion_isaaclab`.

Pinned versions:

- Python 3.11
- `torch==2.7.0`
- `torchvision==0.22.0`
- `isaacsim==5.1.0.0`
- `isaaclab==2.3.0`

## VNC / monitor assumptions

This setup is meant to run on the same kind of GCP VM flow described in `/home/lyuxinghe/code/hymotion_isaaclab/README.md`:

- TurboVNC provides the X server
- Isaac Sim uses that monitor through `DISPLAY=:1`
- ProtoMotions runs from the bundled `third_party/ProtoMotions` submodule by default

Start or restart TurboVNC if needed:

```bash
vncserver :1 -geometry 1920x1080 -depth 24
```

Stop it:

```bash
vncserver -kill :1
```

Required environment variables for this repo:

```bash
export OMNI_KIT_ACCEPT_EULA=YES
export DISPLAY=:1
export LD_LIBRARY_PATH="$HOME/code/env_isaaclab/lib/python3.11/site-packages/isaacsim/kit/extscore/omni.client.lib/bin:${LD_LIBRARY_PATH:-}"
export NCCL_IB_DISABLE=1
export NCCL_NET=Socket
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
```

Optional override if you want to use a different ProtoMotions checkout:

```bash
export PROTOMOTIONS_ROOT=/path/to/ProtoMotions
```

## Preferred render wrapper

Use `scripts/run_custom_scene.sh` for normal motion renders. It exports the required Isaac Sim runtime variables for you, defaults to headless mode, and forwards into `scripts/run_custom_scene.py`.

Headless by default:

```bash
scripts/run_custom_scene.sh \
  --motion-file /home/lyuxinghe/code/hymotion_isaaclab/output/a_person_is_reaching_out_his_left_hand_and_walking_000.motion
```

Monitor-backed on `DISPLAY=:1`:

```bash
scripts/run_custom_scene.sh \
  --motion-file /home/lyuxinghe/code/hymotion_isaaclab/output/a_person_is_reaching_out_his_left_hand_and_walking_000.motion \
  --headless false \
  --display :1
```

Useful overrides:

- `--checkpoint /path/to/last.ckpt`
- `--video-output /path/to/output.mp4`
- `--reference-markers false`

## Create the uv environment

Install Git LFS first if it is not already available:

```bash
git lfs install
```

```bash
cd /home/lyuxinghe/code/human_motion_isaacsim
git submodule update --init --recursive
./env/install.sh
```

What `env/install.sh` does:

- creates `env/.venv` with `uv` and Python 3.11
- runs `git lfs pull` for the ProtoMotions checkout when it is a git repo
- installs `torch==2.7.0` and `torchvision==0.22.0`
- installs `isaacsim==5.1.0.0` and `isaaclab==2.3.0`
- installs ProtoMotions Isaac Lab Python requirements from the bundled submodule or `$PROTOMOTIONS_ROOT`
- installs ProtoMotions editable from the same resolved root
- installs this local package editable from the repo root

Install notes:

- `isaaclab==2.3.0` currently needs `flatdict==4.0.1` preinstalled without build isolation, so `env/install.sh` handles that explicitly.
- ProtoMotions stores checkpoints and some assets in Git LFS, so `git lfs install` must be available before running `./env/install.sh`.
- This repo uses `uv` to provision the Python 3.11 environment instead of depending on `/isaac-sim/kit/python/bin/python3`.
- Runtime imports auto-discover ProtoMotions from `PROTOMOTIONS_ROOT`, `PROTO_MOTIONS_ROOT`, `third_party/ProtoMotions`, or `/home/$USER/code/ProtoMotions`.

Quick import check:

```bash
OMNI_KIT_ACCEPT_EULA=YES env/.venv/bin/python -c "import isaacsim, isaaclab; print('ok')"
```

## Monitor sanity check

Before running ProtoMotions, verify that Isaac Sim can see the TurboVNC monitor and that viewport capture works:

```bash
DISPLAY=:1 \
OMNI_KIT_ACCEPT_EULA=YES \
env/.venv/bin/python scripts/test_isaacsim_monitor.py \
  --output output/monitor_probe.png
```

Expected result:

- Isaac Sim launches with a visible viewport on the VNC display
- the script writes `output/monitor_probe.png`

## ProtoMotions smoke run

This smoke path:

- starts from an empty Isaac Sim scene
- builds the default SMPL human used by the ProtoMotions motion-tracker checkpoint
- loads a `.motion` file from `hymotion_isaaclab`
- runs the tracker policy
- captures viewport frames
- compiles them into an MP4

Monitor-backed run:

```bash
DISPLAY=:1 \
OMNI_KIT_ACCEPT_EULA=YES \
LD_LIBRARY_PATH="$HOME/code/env_isaaclab/lib/python3.11/site-packages/isaacsim/kit/extscore/omni.client.lib/bin:${LD_LIBRARY_PATH:-}" \
NCCL_IB_DISABLE=1 \
NCCL_NET=Socket \
MASTER_ADDR=127.0.0.1 \
MASTER_PORT=29500 \
env/.venv/bin/python scripts/smoke_run_motion.py \
  --checkpoint /home/lyuxinghe/code/human_motion_isaacsim/third_party/ProtoMotions/data/pretrained_models/motion_tracker/smpl/last.ckpt \
  --motion-file /home/lyuxinghe/code/hymotion_isaaclab/output/a_person_is_reaching_out_his_left_hand_and_walking_000.motion \
  --video-output output/smoke_vnc.mp4
```

Headless alternative:

```bash
OMNI_KIT_ACCEPT_EULA=YES env/.venv/bin/python scripts/smoke_run_motion.py \
  --checkpoint /home/lyuxinghe/code/human_motion_isaacsim/third_party/ProtoMotions/data/pretrained_models/motion_tracker/smpl/last.ckpt \
  --motion-file /home/lyuxinghe/code/hymotion_isaaclab/output/a_person_is_reaching_out_his_left_hand_and_walking_000.motion \
  --video-output output/smoke_headless.mp4 \
  --headless
```

Expected outputs:

- MP4 at the path passed to `--video-output`
- extracted PNG frames under `<video-name>/frames`

## Notes

- `scripts/run_custom_scene.sh` is the preferred user-facing entrypoint; `scripts/run_custom_scene.py` remains the underlying Python runtime entrypoint.
- During motion execution, the runner uses the same active-viewport capture call used by the ProtoMotions Isaac Lab path: `simulator._write_viewport_to_file(...)`. That keeps camera behavior aligned with `/home/lyuxinghe/code/hymotion_isaaclab`.
- In monitor mode, the smoke runner uses the normal Isaac Sim GUI kit on `DISPLAY=:1`, which matches the `hymotion_isaaclab` monitor workflow.
- The standalone smoke runner owns the stepping loop for the duration of the motion. The future stage-bound controller should reuse the same policy/capture pieces without taking over an externally managed world lifecycle.
