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
output_dir="${1:-research/cross_episode_memory/artifacts/native_annotated_fridge_$(date +%Y%m%d_%H%M%S)}"
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
exec "$python_bin" \
  research/cross_episode_memory/tools/check_annotated_fridge_task.py \
  --assets "$MLSPACES_ASSETS_DIR" \
  --output "$output_dir" \
  --kitchen --native-object --operate-door --soft-finger \
  --object-name egg_45a3d68915c9f19164541ddf4da76856_1_0_0 \
  --start-base-x .01 --start-base-y .68 --start-base-yaw 0 \
  --door-stance-x .01 --door-stance-y 2.08 \
  --base-x .01 --base-y 1.78 \
  --pickup-stance-x -.91 --pickup-stance-y -1.05 \
  --reverse-undock .3 --use-torso 6 --clearance 0 --motion-slowdown 8 --video-fps 25 --defer-video \
  --grip-open .05 --grip-force 10 --door-grip-force 100 --lift-retreat .20 --lift-height .12 \
  --open-angle 75 --second-open-angle 0 --close-stage-angle 0 \
  --close-final-angle 0 --close-tolerance 3 --base-servo-scale 2
