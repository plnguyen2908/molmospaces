#!/usr/bin/env bash
set -euo pipefail
cd /nobackup/le/molmospaces
export CUDA_VISIBLE_DEVICES=2 MUJOCO_GL=egl PYTHONPATH=.
exec /nobackup2/le/molmospaces/.venv/bin/python research/cross_episode_memory/tools/check_navigation_transfer.py --assets "$MLSPACES_ASSETS_DIR" --output "${1:-research/cross_episode_memory/artifacts/kitchen_floor_fixed_transfer_02}" --kitchen --kitchen-table-pos 0 -0.5 --soft-finger --start-base-x -0.04 --base-x 0.26 --base-y 1.66 --pickup-stance-x -0.5 --pickup-stance-y -0.5 --loaf-pos 0 -0.5 0.97 --grasp-depth -0.003 --use-torso 3 --world-radius 1.5 --clearance 0 --motion-slowdown 8 --grip-force 100 --grip-kp 2500 --grip-open 0.08 --grip-close 0.08 --lift-retreat 0.30 --lift-height 0.12 --reverse-undock 0.3 --turn-speed 0.08 --nav-speed 0.12 --base-servo-scale 2 --gaze hybrid
