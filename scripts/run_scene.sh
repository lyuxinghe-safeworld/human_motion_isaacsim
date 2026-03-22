#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/run_scene.sh --motion-file PATH [options]

Options:
  --motion-file PATH          Path to a .motion file to render. Repeat to build a sequence.
  --model NAME                Registered human model name. Default: smpl.
  --headless true|false       Render headless. Default: true.
  --reference-markers true|false
                              Render reference markers. Default: true.
  --video-output PATH         Output MP4 path. Default: output/<motion-stem-sequence>.mp4
  --display DISPLAY           X display for non-headless runs. Default: :1.
  -h, --help                  Show this help text.
EOF
}

normalize_bool() {
  local value
  value="$(printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]')"
  case "$value" in
    true|false)
      printf '%s\n' "$value"
      ;;
    *)
      echo "Expected boolean true|false, got: ${1:-<empty>}" >&2
      exit 1
      ;;
  esac
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd -- "$script_dir/.." && pwd -P)"
python_bin="$repo_root/env/.venv/bin/python"

motion_files=()
model="smpl"
headless="true"
reference_markers="true"
video_output=""
display=":1"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --motion-file)
      motion_files+=("${2:-}")
      shift 2
      ;;
    --model)
      model="${2:-}"
      shift 2
      ;;
    --headless)
      headless="$(normalize_bool "${2:-}")"
      shift 2
      ;;
    --reference-markers)
      reference_markers="$(normalize_bool "${2:-}")"
      shift 2
      ;;
    --video-output)
      video_output="${2:-}"
      shift 2
      ;;
    --display)
      display="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ "${#motion_files[@]}" -eq 0 ]]; then
  echo "At least one --motion-file is required." >&2
  usage >&2
  exit 1
fi

if [[ -z "$video_output" ]]; then
  motion_name=""
  for motion_file in "${motion_files[@]}"; do
    motion_stem="$(basename -- "${motion_file%.*}")"
    if [[ -z "$motion_name" ]]; then
      motion_name="$motion_stem"
    else
      motion_name="${motion_name}__${motion_stem}"
    fi
  done
  video_output="$repo_root/output/${motion_name}.mp4"
fi

if [[ ! -x "$python_bin" ]]; then
  echo "Missing Python environment: $python_bin" >&2
  echo "Run ./env/install.sh first." >&2
  exit 1
fi

for motion_file in "${motion_files[@]}"; do
  if [[ ! -f "$motion_file" ]]; then
    echo "Motion file not found: $motion_file" >&2
    exit 1
  fi
done

mkdir -p -- "$(dirname -- "$video_output")"

export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
mkdir -p -- "$MPLCONFIGDIR"

isaacsim_client_lib="$repo_root/env/.venv/lib/python3.11/site-packages/isaacsim/kit/extscore/omni.client.lib/bin"
export LD_LIBRARY_PATH="${isaacsim_client_lib}:${LD_LIBRARY_PATH:-}"
export NCCL_IB_DISABLE="1"
export NCCL_NET="Socket"
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT="29500"

cmd=(
  "$python_bin"
  "$repo_root/scripts/run_scene.py"
  --model "$model"
)

for motion_file in "${motion_files[@]}"; do
  cmd+=(--motion-file "$motion_file")
done

cmd+=(--video-output "$video_output")

if [[ "$headless" == "true" ]]; then
  cmd+=(--headless)
else
  export DISPLAY="${DISPLAY:-$display}"
fi

if [[ "$reference_markers" == "true" ]]; then
  cmd+=(--reference-markers)
else
  cmd+=(--no-reference-markers)
fi

exec "${cmd[@]}"
