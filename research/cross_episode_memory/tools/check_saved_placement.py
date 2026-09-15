"""Placement-only diagnostic initialized from a recorded physical carry pose."""

import json
import sys
import traceback
from pathlib import Path

import mujoco
import numpy as np

from research.cross_episode_memory.tools.check_fridge_transfer import DOOR2_JOINT, JOINT, NS
from research.cross_episode_memory.tools.check_navigation_transfer import (
    NavigationTransfer,
    parse_args,
)


def main():
    source = Path(sys.argv[1])
    args = parse_args(sys.argv[2:])
    run = NavigationTransfer(args)
    try:
        rows = json.loads((source / "trace.json").read_text())
        state = next(r for r in reversed(rows) if r["stage"] == "drive forward bread to fridge")
        run.data.qpos[:] = state["qpos"]
        run.data.qvel[:] = 0
        run.data.joint(JOINT).qpos[0] = -np.radians(args.door_angle)
        run.data.joint(DOOR2_JOINT).qpos[0] = np.radians(args.door2_angle)
        for aid in range(run.model.nu):
            if run.model.actuator_trntype[aid] == mujoco.mjtTrn.mjTRN_JOINT:
                jid = run.model.actuator_trnid[aid, 0]
                run.data.ctrl[aid] = run.data.qpos[run.model.jnt_qposadr[jid]]
        run.data.actuator(NS + "right_finger_act").ctrl[0] = args.grip_close
        mujoco.mj_forward(run.model, run.data)
        run.report["scope"] = (
            "placement-only diagnostic restored from recorded qpos; not continuous task success"
        )
        run.report["source_trace"] = str(source)
        run.grasp_relative = np.linalg.inv(run.tcp()) @ run.bread_pose()
        run.attached = True
        run.placing = True
        run.planner = run.make_planner()
        run.arm_aids = run.actuator_ids(run.planner.names)
        run.load_world()
        run.cameras[0].azimuth = 0
        run.cameras[0].distance = 1.8
        run.cameras[0].elevation = -25
        run.cameras[0].lookat[:] = [0.5, 1.65, 0.95]
        run.cameras[1].azimuth = 0
        run.cameras[1].distance = 1.2
        run.place_payload()
    except Exception as e:
        run.report["error"] = str(e)
        run.report["traceback"] = traceback.format_exc()
        print(run.report["traceback"])
    finally:
        run.writer.close()
        run.head_writer.close()
        run.renderer.close()
        run.finalize_report()
        (run.output / "report.json").write_text(json.dumps(run.report, indent=2))
        (run.output / "trace.json").write_text(json.dumps(run.trace))
    raise SystemExit(0 if run.report["success"] else 1)


if __name__ == "__main__":
    main()
