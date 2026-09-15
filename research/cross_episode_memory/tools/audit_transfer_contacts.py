"""Audit recorded frames against all collidable kitchen scenery."""

import json
import os
import sys
from pathlib import Path

import mujoco

from molmo_spaces.configs.robot_configs import RBY1MConfig
from research.cross_episode_memory.kitchen_scene import make_kitchen
from research.cross_episode_memory.tools.check_fridge_door import HANDLE
from research.cross_episode_memory.tools.check_fridge_transfer import BREAD_PREFIX, NS, F


def main():
    folder = Path(sys.argv[1])
    report = json.loads((folder / "report.json").read_text())
    args = report["arguments"]
    model, data, _ = make_kitchen(
        Path(os.environ["MLSPACES_ASSETS_DIR"]),
        RBY1MConfig(),
        (F, BREAD_PREFIX),
        task_table_xy=args["kitchen_table_pos"],
        task_table_height=args["table_height"],
    )
    rows = json.loads((folder / "trace.json").read_text())
    worst = {"depth_m": 0.0}
    worst_finger_loaf = {"depth_m": 0.0}
    for row in rows:
        data.qpos[:] = row["qpos"]
        mujoco.mj_forward(model, data)
        for contact in data.contact:
            names = [model.body(model.geom_bodyid[g]).name for g in (contact.geom1, contact.geom2)]
            robot = [n.startswith(NS) for n in names]
            if robot[0] == robot[1]:
                continue
            ri = 0 if robot[0] else 1
            other = names[1 - ri]
            gid = (contact.geom1, contact.geom2)[1 - ri]
            if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_PLANE:
                continue
            if "ee_finger_r" in names[ri] and other.startswith(BREAD_PREFIX):
                if -float(contact.dist) > worst_finger_loaf["depth_m"]:
                    worst_finger_loaf = {
                        "depth_m": -float(contact.dist),
                        "pair": names,
                        "stage": row["stage"],
                        "time": row["time"],
                    }
                continue
            if "ee_finger_r" in names[ri] and args.get("operate_door", False) and other == HANDLE:
                continue
            if -float(contact.dist) > worst["depth_m"]:
                worst = {
                    "depth_m": -float(contact.dist),
                    "pair": names,
                    "stage": row["stage"],
                    "time": row["time"],
                }
    result = {
        "allowed_contacts": [
            "floor plane",
            "right fingers on loaf only within 1 mm",
            "right fingers on handle during door task",
        ],
        "frames": len(rows),
        "sample_hz": 25,
        "worst_robot_scene_contact": worst,
        "within_3mm": worst["depth_m"] <= 0.003,
        "worst_finger_loaf_contact": worst_finger_loaf,
        "finger_loaf_within_1mm": worst_finger_loaf["depth_m"] <= 0.001,
        "success": worst["depth_m"] <= 0.003 and worst_finger_loaf["depth_m"] <= 0.001,
    }
    (folder / "contact_audit_with_grasp.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
