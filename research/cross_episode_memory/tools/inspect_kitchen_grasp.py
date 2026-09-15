#!/usr/bin/env python3
"""Inspect conservative kitchen boxes around a proposed loaf grasp point."""

import argparse
from pathlib import Path

import mujoco
import numpy as np

from molmo_spaces.configs.robot_configs import RBY1MConfig
from research.cross_episode_memory.kitchen_scene import make_kitchen
from research.cross_episode_memory.tools.check_fridge_transfer import (
    BREAD,
    BREAD_JOINT,
    BREAD_PREFIX,
    F,
    geom_box,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--base", type=float, nargs=2, default=(0.06, -1.08))
    parser.add_argument("--loaf-pos", type=float, nargs=3, default=(0.586, -1.091, 1.044))
    parser.add_argument("--settle", type=float, default=0.6)
    args = parser.parse_args()

    model, data, _ = make_kitchen(
        args.assets, RBY1MConfig(), (F, BREAD_PREFIX), robot_xy=args.base
    )
    adr = model.jnt_qposadr[model.joint(BREAD_JOINT).id]
    data.qpos[adr : adr + 3] = args.loaf_pos
    mujoco.mj_forward(model, data)
    for _ in range(round(args.settle / model.opt.timestep)):
        mujoco.mj_step(model, data)

    bread_bids = {
        bid for bid in range(model.nbody) if model.body(bid).name.startswith(BREAD_PREFIX)
    }
    points = []
    for gid in range(model.ngeom):
        if model.geom_bodyid[gid] not in bread_bids:
            continue
        if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = model.geom_dataid[gid]
        start = model.mesh_vertadr[mid]
        vertices = model.mesh_vert[start : start + model.mesh_vertnum[mid]]
        points.append(vertices @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid])
    bounds = np.concatenate(points)
    grasp = np.array(
        [
            (bounds[:, 0].min() + bounds[:, 0].max()) / 2,
            (bounds[:, 1].min() + bounds[:, 1].max()) / 2,
            bounds[:, 2].max() - 0.003,
        ]
    )
    print(f"loaf_body_pose={data.body(BREAD).xpos.tolist()}")
    print(f"loaf_bounds={bounds.min(0).tolist()}..{bounds.max(0).tolist()}")
    print(f"grasp={grasp.tolist()} pre={(grasp + [0, 0, 0.12]).tolist()}")

    rows = []
    robot_prefix = "robot_0/"
    for gid in range(model.ngeom):
        if not (model.geom_contype[gid] or model.geom_conaffinity[gid]):
            continue
        bid = model.geom_bodyid[gid]
        owner = model.body(bid).name
        if owner.startswith(robot_prefix) or bid in bread_bids or owner.startswith(F):
            continue
        centre_offset, half = geom_box(model, gid)
        rotation = data.geom_xmat[gid].reshape(3, 3)
        centre = data.geom_xpos[gid] + rotation @ centre_offset
        local = np.abs(rotation.T @ (grasp - centre)) - half
        separation = float(np.linalg.norm(np.maximum(local, 0)))
        if separation <= 0.35:
            rows.append((separation, gid, owner, model.geom(gid).name, half.tolist()))
    for row in sorted(rows)[:40]:
        print(row)


if __name__ == "__main__":
    main()
