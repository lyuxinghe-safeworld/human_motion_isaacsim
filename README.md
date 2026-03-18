# `hymotion_isaacsim`

This repo contains the Isaac Sim inference path for running ProtoMotions tracking against the same `.motion` files produced by `/home/lyuxinghe/code/hymotion_isaaclab`.

Start with `env/README.md` for:

- environment creation with `uv`
- TurboVNC / `DISPLAY=:1` monitor setup
- Isaac Sim monitor probing with `scripts/test_isaacsim_monitor.py`
- standalone ProtoMotions smoke runs with `scripts/smoke_run_motion.py`

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
