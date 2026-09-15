"""Probe loaded docking poses using the recorded native pickup state."""

import json
import os
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from molmo_spaces.configs.robot_configs import RBY1MConfig
from research.cross_episode_memory.kitchen_scene import make_kitchen
from research.cross_episode_memory.tools.check_fridge_transfer import (
    BREAD_JOINT,
    BREAD_PREFIX,
    NS,
    F,
)

folder = Path(sys.argv[1])
rows = json.loads((folder / "trace.json").read_text())
state = next(r for r in reversed(rows) if r["stage"] == "lift bread")
m, d, _ = make_kitchen(Path(os.environ["MLSPACES_ASSETS_DIR"]), RBY1MConfig(), (F, BREAD_PREFIX))
q = np.array(state["qpos"])
ad = [m.jnt_qposadr[m.joint(NS + n).id] for n in ["base_x", "base_y", "base_theta"]]
ba = m.jnt_qposadr[m.joint(BREAD_JOINT).id]
start = q[ad]
rot = R.from_euler("z", -start[2])
for x in [0.089989, 0.189989, 0.289989]:
    for y in [1.578499, 1.678499, 1.778499, 1.878499]:
        worst = 0.0
        pair = None
        for dx, dy in [(0, 0), (-0.025, 0), (0.025, 0), (0, -0.025), (0, 0.025)]:
            d.qpos[:] = q
            d.qpos[ad] = [x + dx, y + dy, 0.0]
            d.qpos[ba : ba + 3] = rot.apply(q[ba : ba + 3] - [*start[:2], 0]) + [x + dx, y + dy, 0]
            d.qpos[ba + 3 : ba + 7] = (
                rot * R.from_quat(q[ba + 3 : ba + 7], scalar_first=True)
            ).as_quat(scalar_first=True)
            mujoco.mj_forward(m, d)
            for c in d.contact:
                names = [m.body(m.geom_bodyid[g]).name for g in [c.geom1, c.geom2]]
                moving = [n.startswith((NS, BREAD_PREFIX)) for n in names]
                if moving[0] == moving[1]:
                    continue
                other = c.geom2 if moving[0] else c.geom1
                if m.geom_type[other] == mujoco.mjtGeom.mjGEOM_PLANE:
                    continue
                if -float(c.dist) + 0.001 > worst:
                    worst = -float(c.dist) + 0.001
                    pair = names
        print(x, y, "clear" if worst == 0 else round(worst, 4), pair)
