"""Independently check the recorded native closed-fridge task's sequence and final state."""

import json
import os
import sys
from pathlib import Path

import mujoco
import numpy as np

from molmo_spaces.configs.robot_configs import RBY1MConfig
from research.cross_episode_memory.kitchen_scene import make_kitchen
from research.cross_episode_memory.tools.check_fridge_door import JOINT
from research.cross_episode_memory.tools.check_fridge_transfer import (
    BREAD_PREFIX,
    DOOR2_JOINT,
    F,
    collision_mesh,
)


def main():
    folder = Path(sys.argv[1])
    report = json.loads((folder / "report.json").read_text())
    rows = json.loads((folder / "trace.json").read_text())
    model, data, _ = make_kitchen(
        Path(os.environ["MLSPACES_ASSETS_DIR"]), RBY1MConfig(), (F, BREAD_PREFIX)
    )
    addresses = [model.jnt_qposadr[model.joint(n).id] for n in (JOINT, DOOR2_JOINT)]
    initial = np.degrees(np.asarray(rows[0]["qpos"])[addresses])
    final = np.degrees(np.asarray(rows[-1]["qpos"])[addresses])
    data.qpos[:] = rows[-1]["qpos"]
    mujoco.mj_forward(model, data)
    bread_bids = {b for b in range(model.nbody) if model.body(b).name.startswith(BREAD_PREFIX)}
    support = []
    for c in data.contact:
        bodies = [model.geom_bodyid[g] for g in (c.geom1, c.geom2)]
        for i in (0, 1):
            if bodies[i] in bread_bids and model.body(bodies[1 - i]).name.startswith(F):
                support.append(model.geom((c.geom1, c.geom2)[1 - i]).name)
    vertices = collision_mesh(model, data, lambda b: b in bread_bids)[0]
    shelf = model.site(report["target_shelf"]).id
    center = data.site_xpos[shelf]
    half_x, half_y = model.site_size[shelf][[2, 0]]
    inside = bool(
        vertices[:, 0].min() > center[0] - half_x - 0.01
        and vertices[:, 0].max() < center[0] + half_x + 0.01
        and vertices[:, 1].min() > center[1] - half_y - 0.01
        and vertices[:, 1].max() < center[1] + half_y + 0.01
        and abs(vertices[:, 2].min() - report["target_shelf_surface_z"]) < 0.015
    )
    stages = [r["stage"] for r in rows]
    ordered = [
        "opening",
        "drive forward to native counter",
        "lift bread",
        "insert loaf",
        "closing recorded opening path",
    ]
    indices = [stages.index(s) for s in ordered]
    before = next(s for s in report["stages"] if "door_open_deg" in s)["door_open_deg"]
    nav = report["navigation"]
    delta = np.asarray(rows[-1]["bread_pose"])[:3, 3] - np.asarray(rows[-2]["bread_pose"])[:3, 3]
    speed = float(np.linalg.norm(delta) / (rows[-1]["time"] - rows[-2]["time"]))
    checks = {
        "task_report_success": report["success"],
        "native_object_without_pose_override": report["arguments"]["native_object"]
        and report["arguments"]["loaf_pos"] is None
        and report["arguments"]["kitchen_table_pos"] is None,
        "initial_doors_closed": bool(np.max(abs(initial)) < 0.1),
        "action_order": indices == sorted(indices),
        "first_approach_at_least_half_metre": nav[0]["measured_distance_m"] > 0.5,
        "navigation_to_counter_before_pickup": len(nav) >= 4
        and nav[1]["measured_distance_m"] > 0.5
        and not nav[1]["carrying"],
        "door_remained_open_before_departure": before >= 70,
        "final_doors_closed": bool(np.max(abs(final)) <= 3),
        "loaf_still_supported_in_fridge": bool(support),
        "loaf_still_inside_target_shelf": inside,
        "loaf_still_settled": speed < 0.03,
    }
    result = {
        "success": all(checks.values()),
        "checks": checks,
        "initial_hinge_degrees": initial.tolist(),
        "final_hinge_degrees": final.tolist(),
        "final_loaf_support_geoms": sorted(set(support)),
        "final_loaf_speed_m_s": speed,
    }
    (folder / "sequence_validation.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
