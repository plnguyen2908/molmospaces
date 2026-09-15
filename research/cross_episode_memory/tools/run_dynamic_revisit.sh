#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"
python_bin="${MOLMOSPACES_PYTHON:-/nobackup2/le/molmospaces/.venv/bin/python}"
: "${MLSPACES_ASSETS_DIR:?Set MLSPACES_ASSETS_DIR to the MolmoSpaces asset directory}"
stamp="$(date +%Y%m%d_%H%M%S)"
output_dir="${1:-$repo_root/research/cross_episode_memory/artifacts/dynamic_revisit_$stamp}"
case "$output_dir" in "$repo_root"/*) ;; *) echo "Output must be inside $repo_root" >&2; exit 2;; esac
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}" MUJOCO_GL=egl PYTHONPATH="$repo_root"
exec "$python_bin" research/cross_episode_memory/tools/check_dynamic_revisit.py \
  --assets "$MLSPACES_ASSETS_DIR" --output "$output_dir" --video-fps 25 --video-speedup 5
