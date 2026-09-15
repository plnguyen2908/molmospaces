"""Run only the native closed-fridge approach and physical opening."""

import argparse
import json
import traceback
from pathlib import Path

import mujoco
import numpy as np

from research.cross_episode_memory.tools.check_navigation_transfer import (
    NavigationTransfer,
    parse_args,
)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--resume", type=Path)
    local, remaining = parser.parse_known_args()
    check = NavigationTransfer(parse_args(remaining))
    try:
        if local.resume:
            rows = json.loads((local.resume / "trace.json").read_text())
            row = next(r for r in reversed(rows) if r["stage"] == "opening")
            check.data.qpos[:] = row["qpos"]
            check.data.qvel[:] = 0
            for aid in range(check.model.nu):
                if check.model.actuator_trntype[aid] == mujoco.mjtTrn.mjTRN_JOINT:
                    jid = check.model.actuator_trnid[aid, 0]
                    check.data.ctrl[aid] = check.data.qpos[check.model.jnt_qposadr[jid]]
            check.data.actuator("robot_0/right_finger_act").ctrl[0] = 0
            mujoco.mj_forward(check.model, check.data)
            check.operating_door = True
            check.grasp_in_handle = np.linalg.inv(check.handle_pose()) @ check.tcp()
            check.planner = check.make_planner()
            check.arm_aids = check.actuator_ids(check.planner.names)
            check.follow_hinge(-np.radians(check.args.open_angle))
            check.gripper(True)
            check.retreat("withdraw from door")
            check.tuck_arm()
        else:
            check.tick(0.6)
            check.open_fridge()
        opened = float(abs(np.degrees(check.angle())))
        check.record(door_open_after_withdrawal_deg=opened)
        if opened < 70:
            raise RuntimeError(f"Door did not remain open: {opened:.1f} degrees")
        check.report["success"] = True
    except Exception as exc:
        check.report["error"] = str(exc)
        check.report["traceback"] = traceback.format_exc()
        print(check.report["traceback"], flush=True)
    finally:
        check.report["scope"] = (
            "native door diagnostic only; restored state"
            if local.resume
            else "native closed-fridge navigation and opening diagnostic only"
        )
        check.writer.close()
        check.head_writer.close()
        check.renderer.close()
        check.finalize_report()
        (check.output / "report.json").write_text(json.dumps(check.report, indent=2))
        (check.output / "trace.json").write_text(json.dumps(check.trace))
    return 0 if check.report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
