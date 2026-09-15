#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"
python_bin="${MOLMOSPACES_PYTHON:-/nobackup2/le/molmospaces/.venv/bin/python}"
: "${MLSPACES_ASSETS_DIR:?Set MLSPACES_ASSETS_DIR to the MolmoSpaces asset directory}"
output_dir="$(realpath -m "${1:-research/cross_episode_memory/artifacts/reorder_chain_$(date +%Y%m%d_%H%M%S)}")"
case "$output_dir" in "$repo_root"/*) ;; *) echo "Output must stay inside $repo_root" >&2; exit 2;; esac
if [[ -e "$output_dir" ]]; then echo "Output already exists: $output_dir" >&2; exit 2; fi
if (( $# > 0 )); then shift; fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}" MUJOCO_GL=egl PYTHONPATH="$repo_root"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
exec "$python_bin" research/cross_episode_memory/tools/check_reorder_chain.py \
  --assets "$MLSPACES_ASSETS_DIR" --output "$output_dir" "$@"
