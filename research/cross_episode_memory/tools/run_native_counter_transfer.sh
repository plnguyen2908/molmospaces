#!/usr/bin/env bash
set -euo pipefail
cd /nobackup/le/molmospaces
export CUDA_VISIBLE_DEVICES=2 MUJOCO_GL=egl PYTHONPATH=.
exec /nobackup2/le/molmospaces/.venv/bin/python research/cross_episode_memory/tools/check_navigation_transfer.py --assets "$MLSPACES_ASSETS_DIR" --output "${1:-research/cross_episode_memory/artifacts/native_counter_transfer_01}" --kitchen --native-object --soft-finger --start-base-x -.91 --start-base-y .68 --start-base-yaw 3.136592653589793 --base-x .16 --base-y 1.66 --pickup-stance-x -.91 --pickup-stance-y .68 --reverse-undock .3 --use-torso 6 --clearance 0 --motion-slowdown 8 --grip-open .08 --grip-close .08 --lift-retreat .20 --lift-height .12 --base-servo-scale 2
