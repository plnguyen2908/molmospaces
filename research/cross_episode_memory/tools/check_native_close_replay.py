"""Test closing by reversing the measured cuRobo opening motion at the same stance."""

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
    p.add_argument("--opening", type=Path, required=True)
    p.add_argument("--finish-only", action="store_true")
    local, remaining = p.parse_known_args()
    check = NavigationTransfer(parse_args(remaining))
    source = json.loads((local.opening / "trace.json").read_text())
    opening = [r for r in source if r["stage"] == "opening"]
    rows = json.loads((local.resume / "trace.json").read_text())
    if local.finish_only:
        close_rows = [r for r in rows if r["stage"] == "closing recorded opening path"]
        restore(check, close_rows[-21])
    else:
        restore(check, next(r for r in reversed(rows) if r["stage"] == "grasp to close"))
    check.data.actuator("robot_0/right_finger_act").ctrl[0] = -0.05
    check.operating_door = True
    check.planner = check.make_planner()
    check.arm_aids = check.actuator_ids(check.planner.names)
    ad = [
        check.model.jnt_qposadr[check.model.joint("robot_0/" + n).id] for n in check.planner.names
    ]
    try:
        if local.finish_only:
            check.gripper(False)
        else:
            check.stage = "return to opening posture"
            check.obstacles(articulating=False)
            current = [float(check.data.qpos[i]) for i in ad]
            target = [opening[-1]["qpos"][i] for i in ad]
            trajectory = check.planner.plan_joints(current, target)
            for q in trajectory:
                check.data.ctrl[check.arm_aids] = q
                check.tick(check.planner.dt * check.args.motion_slowdown)
            check.tick(0.5)
            check.gripper(False)
            check.stage = "closing recorded opening path"
            for row in reversed(opening):
                check.data.ctrl[check.arm_aids] = [row["qpos"][i] for i in ad]
                check.tick(0.04)
                if check.penetration_so_far() > 0.003:
                    raise RuntimeError("Collision during reverse opening path")
                if abs(check.angle() - row["qpos"][0]) > 0.12:
                    raise RuntimeError(
                        f"Door departed from reversed path: {np.degrees(check.angle()):.1f} versus {np.degrees(row['qpos'][0]):.1f}"
                    )
        check.door_move("closing finish at stop", check.handle_grasp_pose())
        push = check.tcp()
        handle = check.data.xpos[check.handle_bid].copy()
        root = check.data.body("refrigerator_b8b0ac7a848fb12e43f9a760bdd25981_1_0_0").xpos
        shut = np.array([root[0] - .3212, root[1] - .0939, handle[2]])
        direction = shut - handle
        push[:3, 3] += direction / np.linalg.norm(direction) * .04
        check.door_move("closing press to stop", push)
        check.record(door_deg=float(abs(np.degrees(check.angle()))))
        check.gripper(True)
        check.stage = "withdraw recorded handle approach"
        approach = [r for r in source if r["stage"] == "reach door handle"]
        for row in reversed(approach):
            check.data.ctrl[check.arm_aids] = [row["qpos"][i] for i in ad]
            check.tick(.04)
            if check.penetration_so_far() > .003:
                raise RuntimeError("Collision during recorded withdrawal")
        check.tick(1)
        angle = float(abs(np.degrees(check.angle())))
        check.record(closed_angle_deg=angle)
        if angle > 3:
            raise RuntimeError("Door not closed")
        check.report["success"] = True
    except Exception as exc:
        check.report["error"] = str(exc)
        print(traceback.format_exc(), flush=True)
    finally:
        check.report["scope"] = "restored reverse-opening-path closing diagnostic only"
        check.writer.close()
        check.head_writer.close()
        check.renderer.close()
        (check.output / "report.json").write_text(json.dumps(check.report, indent=2))
        (check.output / "trace.json").write_text(json.dumps(check.trace))
    return 0 if check.report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
