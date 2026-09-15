#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"
python_bin="${MOLMOSPACES_PYTHON:-/nobackup2/le/molmospaces/.venv/bin/python}"
if [[ ! -x "$python_bin" ]]; then
  echo "Missing project Python: $python_bin. See research/cross_episode_memory/CHECK_ITHOR_TRANSFER.md" >&2
  exit 1
fi
: "${MLSPACES_ASSETS_DIR:?Export MLSPACES_ASSETS_DIR; see CHECK_ITHOR_TRANSFER.md}"
if [[ ! -d "$MLSPACES_ASSETS_DIR" ]]; then
  echo "Assets directory does not exist: $MLSPACES_ASSETS_DIR" >&2
  exit 1
fi
output_dir="${1:-research/cross_episode_memory/artifacts/native_annotated_egg_$(date +%Y%m%d_%H%M%S)}"
output_dir="$(realpath -m -- "$output_dir")"
case "$output_dir" in
  "$repo_root"/*) ;;
  *) echo "Output must stay inside $repo_root" >&2; exit 1 ;;
esac
if [[ -e "$output_dir" ]]; then
  echo "Refusing to overwrite existing output: $output_dir" >&2
  exit 1
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}" MUJOCO_GL=egl PYTHONPATH="$repo_root"
echo "Python: $python_bin"
echo "Output: $output_dir"
exec "$python_bin" research/cross_episode_memory/tools/check_annotated_grasp.py \
  --assets "$MLSPACES_ASSETS_DIR" --output "$output_dir" \
  --kitchen --native-object --soft-finger --carry-only \
  --object-name egg_45a3d68915c9f19164541ddf4da76856_1_0_0 \
  --start-base-x -.91 --start-base-y -1.05 --start-base-yaw 3.136592653589793 \
  --pickup-stance-x -.91 --pickup-stance-y -1.05 \
  --base-x -.61 --base-y -.05 --reverse-undock .3 \
  --use-torso 6 --clearance 0 --motion-slowdown 3 \
  --grip-open .05 --grip-force 10 --lift-retreat .20 --lift-height .12 \
  --base-servo-scale 2 --door-angle 0 --door2-angle 0
