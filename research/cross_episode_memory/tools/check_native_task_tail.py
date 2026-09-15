"""Diagnose native pickup/placement/closing from a recorded post-opening state."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from research.cross_episode_memory.tools.check_navigation_transfer import (
    NavigationTransfer,
    parse_args,
)


def restore(check, row):
    check.data.qpos[:] = row["qpos"]
    check.data.qvel[:] = 0
    for aid in range(check.model.nu):
        if check.model.actuator_trntype[aid] == mujoco.mjtTrn.mjTRN_JOINT:
            jid = check.model.actuator_trnid[aid, 0]
            check.data.ctrl[aid] = check.data.qpos[check.model.jnt_qposadr[jid]]
    # Planar base actuators use site transmission, not joint transmission.
    for name in ("base_x", "base_y", "base_theta"):
        check.data.actuator("robot_0/" + name + "_act").ctrl[0] = check.data.joint("robot_0/" + name).qpos[0]
    mujoco.mj_forward(check.model, check.data)


class TailCheck(NavigationTransfer):
    def prepare_pickup(self):
        self.args.operate_door = False
        try:
            super().prepare_pickup()
        finally:
            self.args.operate_door = True


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--grasp-source", type=Path, required=True)
    local, remaining = parser.parse_known_args()
    check = TailCheck(parse_args(remaining))
    rows = json.loads((local.grasp_source / "trace.json").read_text())
    restore(check, next(r for r in reversed(rows) if r["stage"] == "grasp door handle"))
    check.grasp_in_handle = np.linalg.inv(check.handle_pose()) @ check.tcp()
    rows = json.loads((local.resume / "trace.json").read_text())
    restore(check, rows[-1])
    check.data.actuator("robot_0/right_finger_act").ctrl[0] = -0.05
    check.gaze_command = [
        float(check.data.joint("robot_0/head_" + str(i)).qpos[0]) for i in range(2)
    ]
    check.report["scope"] = "restored post-opening diagnostic; not continuous task evidence"
    check.report["restored_from"] = str(local.resume)
    return check.run()


if __name__ == "__main__":
    raise SystemExit(main())
