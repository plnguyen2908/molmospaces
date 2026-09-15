"""Diagnose the final closing arc with the torso fixed at a recorded grasp."""

import argparse
import json
import traceback
from pathlib import Path

import numpy as np

from research.cross_episode_memory.tools.check_native_task_tail import restore
from research.cross_episode_memory.tools.check_navigation_transfer import (
    NavigationTransfer,
    parse_args,
)


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--resume", type=Path, required=True)
    p.add_argument("--from-angle", type=float, default=40)
    local, remaining = p.parse_known_args()
    check = NavigationTransfer(parse_args(remaining))
    rows = json.loads((local.resume / "trace.json").read_text())
    row = min(
        (r for r in rows if r["stage"] == "closing"),
        key=lambda r: abs(abs(np.degrees(r["qpos"][0])) - local.from_angle),
    )
    restore(check, row)
    check.data.actuator("robot_0/right_finger_act").ctrl[0] = 0
    check.operating_door = True
    check.closing_door = "final"
    check.grasp_in_handle = np.linalg.inv(check.handle_pose()) @ check.tcp()
    check.planner = check.make_planner()
    check.arm_aids = check.actuator_ids(check.planner.names)
    try:
        check.follow_hinge(0.0)
        check.gripper(True)
        check.retreat("withdraw from closed door")
        check.tick(1)
        angle = abs(np.degrees(check.angle()))
        check.record(closed_angle_deg=float(angle))
        if angle > 3:
            raise RuntimeError(f"Door still at {angle:.1f} degrees")
        check.report["success"] = True
    except Exception as exc:
        check.report["error"] = str(exc)
        print(traceback.format_exc(), flush=True)
    finally:
        check.report["scope"] = "restored final-closing diagnostic only"
        check.writer.close()
        check.head_writer.close()
        check.renderer.close()
        (check.output / "report.json").write_text(json.dumps(check.report, indent=2))
        (check.output / "trace.json").write_text(json.dumps(check.trace))
    return 0 if check.report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
