#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_dir"
python_bin="${MOLMOSPACES_PYTHON:-/nobackup2/le/molmospaces/.venv/bin/python}"
assets_dir="${MLSPACES_ASSETS_DIR:-/nobackup2/le/.cache/molmospaces/assets/L25vYmFja3VwMi9sZS9tb2xtb3NwYWNlcw}"
output_dir="${1:-research/cross_episode_memory/artifacts/procthor_population_$(date +%Y%m%d_%H%M%S)}"
export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec "$python_bin" research/cross_episode_memory/tools/populate_procthor_receptacles.py \
  --assets "$assets_dir" \
  --scene "$assets_dir/scenes/procthor-10k-train/train_8.xml" \
  --objects-per-table 5 --objects-per-receptacle 1 --book-overhang .04 \
  --output "$output_dir"
