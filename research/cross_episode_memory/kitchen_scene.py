"""Build the real iTHOR kitchen for the reorder task, at a cost that is affordable.

The component checks so far used an isolated rig: the fridge and the loaf extracted
into an empty room with a plane floor. This builds the actual FloorPlan3 kitchen
instead, with the robot in it.

Straight out of the box that runs at 107 ms per 2 ms physics step -- 53x slower than
real time, about eight hours for a run that takes four minutes on the rig. Measured
breakdown: 1343 contacts, 4245 constraint rows, and constraint projection taking
~77% of every step.

Almost none of that is the clutter. Freezing every loose object gained 1.1x,
switching off collision beyond the work area 1.06x, and contact bitmasks 1.2x. The
cost is the FLOOR: iTHOR's collision floor is a mesh, and RB-Y1's base resting on it
generates over 500 contact points, where a plane generates two. Giving the floor a
plane for collision and keeping the mesh for appearance is 112x: 0.88 ms per step,
35 contacts, about four minutes per run.

Two scene assumptions follow from this, and both are deliberate:

- Loose props are frozen, so the robot cannot knock them over. Fine for this task;
  wrong for anything studying clutter disturbance.
- The floor is flat for collision purposes. True of this kitchen; it would hide a
  step or a threshold in a scene that had one.
"""

from pathlib import Path

import mujoco
import numpy as np

FLOOR_COLLISION_NAME = "kitchen_collision_floor"
# Bodies whose names start with these keep their physics: the robot, the fridge it
# operates, and the objects the task moves.
FLOORISH = ("floor", "decal", "mesh_")


def floor_height(model, data):
    """Top of the collidable floor geometry, in world z."""
    tops = []
    for gid in range(model.ngeom):
        name = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or "").lower()
        if "floor" not in name:
            continue
        if not (model.geom_contype[gid] or model.geom_conaffinity[gid]):
            continue
        tops.append(float(data.geom_xpos[gid][2] + abs(model.geom_size[gid][2])))
    return float(np.median(tops)) if tops else 0.0


def make_kitchen(
    assets: Path,
    robot_cfg,
    dynamic_prefixes: tuple[str, ...],
    robot_xy=(0.0, 0.0),
    scene_name: str = "FloorPlan3_physics.xml",
    plane_floor: bool = True,
    freeze_props: bool = True,
):
    """The kitchen with the robot in it, trimmed to something that runs.

    `dynamic_prefixes` names the bodies that must stay simulated -- the fridge and
    whatever the task moves. Everything else is frozen where it stands.
    """
    path = assets / "scenes/ithor" / scene_name
    spec = mujoco.MjSpec.from_file(str(path))
    # RB-Y1 must be inserted at the origin -- add_robot_to_scene asserts it -- and
    # then driven to where it belongs through its base joints.
    robot_cfg.robot_cls.add_robot_to_scene(
        robot_cfg, spec, "robot_0/", [0.0, 0.0], [1.0, 0.0, 0.0, 0.0]
    )
    keep = ("robot_0/",) + tuple(dynamic_prefixes)

    frozen = 0
    if freeze_props:
        for body in spec.bodies:
            if (body.name or "").startswith(keep):
                continue
            for joint in list(body.joints):
                spec.delete(joint)
                frozen += 1

    if plane_floor:
        spec.worldbody.add_geom(
            name=FLOOR_COLLISION_NAME,
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            pos=[0, 0, 0],
            size=[12, 12, 0.1],
            rgba=[0, 0, 0, 0],
        )

    model = spec.compile()
    data = mujoco.MjData(model)
    for axis, value in zip(("base_x", "base_y"), robot_xy):
        data.joint("robot_0/" + axis).qpos[0] = float(value)
        data.actuator("robot_0/" + axis + "_act").ctrl[0] = float(value)
    for i in range(model.nu):
        if model.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT and not data.ctrl[i]:
            data.ctrl[i] = data.qpos[model.jnt_qposadr[model.actuator_trnid[i, 0]]]
    mujoco.mj_forward(model, data)

    stats = {"frozen_joints": frozen, "plane_floor": plane_floor}
    if plane_floor:
        names = [model.body(b).name for b in range(model.nbody)]
        disabled = 0
        for gid in range(model.ngeom):
            owner = names[model.geom_bodyid[gid]]
            if owner.startswith(keep):
                continue
            gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            if gname == FLOOR_COLLISION_NAME:
                continue
            if any(k in owner.lower() for k in FLOORISH):
                model.geom_contype[gid] = 0
                model.geom_conaffinity[gid] = 0
                disabled += 1
        stats["mesh_floor_geoms_disabled"] = disabled
        mujoco.mj_forward(model, data)

    stats.update(bodies=model.nbody, geoms=model.ngeom, dofs=model.nv, contacts=int(data.ncon))
    return model, data, stats
