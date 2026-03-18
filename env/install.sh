#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
PROTO_ROOT="${PROTOMOTIONS_ROOT:-${PROTO_MOTIONS_ROOT:-$HOME/code/ProtoMotions}}"

if [[ ! -f "$PROTO_ROOT/requirements_isaaclab.txt" ]]; then
  echo "ProtoMotions Isaac Lab requirements not found at: $PROTO_ROOT/requirements_isaaclab.txt" >&2
  exit 1
fi

uv venv env/.venv --python 3.11
uv pip install --python env/.venv/bin/python --index-url https://download.pytorch.org/whl/cu128 torch==2.7.0
uv pip install --python env/.venv/bin/python setuptools==69.5.1 wheel
uv pip install --python env/.venv/bin/python --no-build-isolation flatdict==4.0.1
uv pip install --python env/.venv/bin/python --extra-index-url https://pypi.nvidia.com -r env/requirements.lock
uv pip install --python env/.venv/bin/python -r "$PROTO_ROOT/requirements_isaaclab.txt"
uv pip install --python env/.venv/bin/python -e .
