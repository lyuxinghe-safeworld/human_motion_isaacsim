# Render Motion Video Wrapper Design

## Goal

Add a bash entrypoint that sets the required Isaac Sim and ProtoMotions environment variables, then renders a motion file through the existing `scripts/run_custom_scene.py` pipeline with sensible defaults.

## Constraints

- Keep `scripts/run_custom_scene.py` as the single source of truth for runtime behavior.
- Default to headless rendering.
- Use `DISPLAY=:1` only when the caller explicitly disables headless mode.
- Prefer repo-local defaults for the checkpoint, Python executable, and output path.
- Avoid introducing a second install or runtime stack.

## Options Considered

### 1. Thin bash wrapper around `run_custom_scene.py`

Pros:
- Keeps env setup in shell, where it naturally belongs.
- Leaves Python runtime logic in one place.
- Easy to call from terminals and automation.

Cons:
- Needs dedicated subprocess coverage because it is shell code.

### 2. Move env bootstrapping into `run_custom_scene.py`

Pros:
- One fewer script.

Cons:
- Mixes process setup with runtime logic.
- Harder to reuse the Python entrypoint from tests and other scripts.

### 3. Shared `env.sh` plus wrapper

Pros:
- More reusable if more shell entrypoints appear later.

Cons:
- Extra indirection without a current need.

## Decision

Implement option 1 with a new `scripts/render_motion_video.sh` wrapper.

## Interface

Required:

- `--motion-file <path>`

Optional:

- `--headless <true|false>` with default `true`
- `--checkpoint <path>` with default `third_party/ProtoMotions/data/pretrained_models/motion_tracker/smpl/last.ckpt`
- `--reference-markers <true|false>` with default `true`
- `--video-output <path>` with default `output/<motion-file-stem>.mp4`
- `--display <display>` with default `:1` and only used when `--headless false`

## Environment Behavior

The wrapper exports:

- `OMNI_KIT_ACCEPT_EULA=YES`
- `MPLCONFIGDIR=/tmp/matplotlib` unless already set
- `LD_LIBRARY_PATH=<repo>/env/.venv/lib/python3.11/site-packages/isaacsim/kit/extscore/omni.client.lib/bin:${LD_LIBRARY_PATH:-}`
- `NCCL_IB_DISABLE=1`
- `NCCL_NET=Socket`
- `MASTER_ADDR=127.0.0.1`
- `MASTER_PORT=29500`
- `DISPLAY=<display>` only for non-headless runs

## Validation

The wrapper should fail fast when:

- `env/.venv/bin/python` does not exist
- the motion file does not exist
- the checkpoint file does not exist

## Testing

- Add subprocess-based tests that execute the real wrapper script from a temporary fake repo layout.
- Verify default headless behavior, default checkpoint resolution, default output path, and exported env vars.
- Verify non-headless behavior sets `DISPLAY=:1` and forwards `--no-reference-markers`.
