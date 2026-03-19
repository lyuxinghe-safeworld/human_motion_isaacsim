#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
DEFAULT_PROTO_ROOT="$(pwd)/third_party/ProtoMotions"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required to create the unified Python 3.11 environment." >&2
  exit 1
fi

if [[ -n "${PROTOMOTIONS_ROOT:-}" ]]; then
  PROTO_ROOT="${PROTOMOTIONS_ROOT}"
elif [[ -n "${PROTO_MOTIONS_ROOT:-}" ]]; then
  PROTO_ROOT="${PROTO_MOTIONS_ROOT}"
elif [[ -f "$DEFAULT_PROTO_ROOT/requirements_isaaclab.txt" ]]; then
  PROTO_ROOT="${DEFAULT_PROTO_ROOT}"
else
  PROTO_ROOT="$HOME/code/ProtoMotions"
fi

if [[ ! -f "$PROTO_ROOT/requirements_isaaclab.txt" ]]; then
  echo "ProtoMotions Isaac Lab requirements not found at: $PROTO_ROOT/requirements_isaaclab.txt" >&2
  echo "Initialize the submodule with: git submodule update --init --recursive" >&2
  exit 1
fi

if git -C "$PROTO_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  if ! command -v git-lfs >/dev/null 2>&1; then
    echo "git-lfs is required to fetch ProtoMotions checkpoints and assets. Install git-lfs and rerun." >&2
    exit 1
  fi
  git -C "$PROTO_ROOT" lfs pull
fi

uv venv env/.venv --python 3.11
uv pip install --python env/.venv/bin/python --index-url https://download.pytorch.org/whl/cu128 torch==2.7.0 torchvision==0.22.0
uv pip install --python env/.venv/bin/python setuptools==69.5.1 wheel
uv pip install --python env/.venv/bin/python --no-build-isolation flatdict==4.0.1
uv pip install --python env/.venv/bin/python --extra-index-url https://pypi.nvidia.com -r env/requirements.lock
uv pip install --python env/.venv/bin/python -r "$PROTO_ROOT/requirements_isaaclab.txt"
uv pip install --python env/.venv/bin/python -e "$PROTO_ROOT"
uv pip install --python env/.venv/bin/python -e .
