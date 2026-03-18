# Environment Setup

This repo provides a standalone Isaac Sim + ProtoMotions smoke path that mirrors the monitor-backed workflow in `/home/lyuxinghe/code/hymotion_isaaclab`.

Pinned versions:

- Python 3.11
- `torch==2.7.0`
- `isaacsim==5.1.0.0`
- `isaaclab==2.3.0`

## VNC / monitor assumptions

This setup is meant to run on the same kind of GCP VM flow described in `/home/lyuxinghe/code/hymotion_isaaclab/README.md`:

- TurboVNC provides the X server
- Isaac Sim uses that monitor through `DISPLAY=:1`
- ProtoMotions runs inside the local `ProtoMotions` checkout

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
export PROTOMOTIONS_ROOT=/home/lyuxinghe/code/ProtoMotions
export DISPLAY=:1
export LD_LIBRARY_PATH="$HOME/code/env_isaaclab/lib/python3.11/site-packages/isaacsim/kit/extscore/omni.client.lib/bin:${LD_LIBRARY_PATH:-}"
export NCCL_IB_DISABLE=1
export NCCL_NET=Socket
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
```

## Create the uv environment

```bash
cd /home/lyuxinghe/code/hymotion_isaacsim
./env/install.sh
```

What `env/install.sh` does:

- creates `env/.venv` with `uv`
- installs `torch==2.7.0`
- installs `isaacsim==5.1.0.0` and `isaaclab==2.3.0`
- installs ProtoMotions Isaac Lab Python requirements from `$PROTOMOTIONS_ROOT/requirements_isaaclab.txt`
- installs this local package editable from the repo root

Install notes:

- `isaaclab==2.3.0` currently needs `flatdict==4.0.1` preinstalled without build isolation, so `env/install.sh` handles that explicitly.
- Runtime imports auto-discover the local ProtoMotions checkout from `PROTOMOTIONS_ROOT`, `PROTO_MOTIONS_ROOT`, or `/home/$USER/code/ProtoMotions`.

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
PROTOMOTIONS_ROOT=/home/lyuxinghe/code/ProtoMotions \
LD_LIBRARY_PATH="$HOME/code/env_isaaclab/lib/python3.11/site-packages/isaacsim/kit/extscore/omni.client.lib/bin:${LD_LIBRARY_PATH:-}" \
NCCL_IB_DISABLE=1 \
NCCL_NET=Socket \
MASTER_ADDR=127.0.0.1 \
MASTER_PORT=29500 \
env/.venv/bin/python scripts/smoke_run_motion.py \
  --checkpoint /home/lyuxinghe/code/ProtoMotions/data/pretrained_models/motion_tracker/smpl/last.ckpt \
  --motion-file /home/lyuxinghe/code/hymotion_isaaclab/output/a_person_is_reaching_out_his_left_hand_and_walking_000.motion \
  --video-output output/smoke_vnc.mp4
```

Headless alternative:

```bash
OMNI_KIT_ACCEPT_EULA=YES env/.venv/bin/python scripts/smoke_run_motion.py \
  --checkpoint /home/lyuxinghe/code/ProtoMotions/data/pretrained_models/motion_tracker/smpl/last.ckpt \
  --motion-file /home/lyuxinghe/code/hymotion_isaaclab/output/a_person_is_reaching_out_his_left_hand_and_walking_000.motion \
  --video-output output/smoke_headless.mp4 \
  --headless
```

Expected outputs:

- MP4 at the path passed to `--video-output`
- extracted PNG frames under `<video-name>/frames`

## Notes

- During motion execution, the runner uses the same active-viewport capture call used by the ProtoMotions Isaac Lab path: `simulator._write_viewport_to_file(...)`. That keeps camera behavior aligned with `/home/lyuxinghe/code/hymotion_isaaclab`.
- In monitor mode, the smoke runner uses the normal Isaac Sim GUI kit on `DISPLAY=:1`, which matches the `hymotion_isaaclab` monitor workflow.
- The standalone smoke runner owns the stepping loop for the duration of the motion. The future stage-bound controller should reuse the same policy/capture pieces without taking over an externally managed world lifecycle.
