#!/usr/bin/env python3
"""Physical task-bread transfer from a table to an open fridge shelf.

An isolated component check using the actual FloorPlan3 bread and fridge assets.
"""

import argparse
import copy
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np

from molmo_spaces.configs.robot_configs import RBY1MConfig
from molmo_spaces.robots.rby1_actuators import configure_rby1_gripper_servos
from research.cross_episode_memory.tools.check_fridge_door import JOINT, F

DOOR2_JOINT = F + "_1_4_0_Mesh9a740b141_1"
BREAD_PREFIX = "bread_02590a72dc962788d308c804909b928a"
BREAD = BREAD_PREFIX + "_1_0_0"
BREAD_JOINT = BREAD + "_jntfree_0"


def collision_mesh(model, data, include_body):
    """Combine the model's collision triangles in world coordinates."""
    vertices, faces = [], []
    offset = 0
    for gid in range(model.ngeom):
        if not include_body(model.geom_bodyid[gid]):
            continue
        if not (model.geom_contype[gid] or model.geom_conaffinity[gid]):
            continue
        if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = model.geom_dataid[gid]
        va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        fa, fn = model.mesh_faceadr[mid], model.mesh_facenum[mid]
        v = model.mesh_vert[va : va + vn].copy()
        v = v @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]
        vertices.append(v)
        faces.append(model.mesh_face[fa : fa + fn].copy() + offset)
        offset += vn
    return np.concatenate(vertices), np.concatenate(faces)


def make_scene(
    assets,
    fridge_x=0.95,
    table_height=0.90,
    base_x=0.20,
    base_y=0.0,
    table_y_offset=0.0,
    table_foot_half_y=0.07,
    door_angle_deg=90.0,
    door2_angle_deg=0.0,
):
    path = assets / "scenes/ithor/FloorPlan3_physics.xml"
    src = ET.parse(path).getroot()
    fridge = copy.deepcopy(src.find(f'.//body[@name="{F}_1_0_0"]'))
    fridge.set("pos", f"{fridge_x} 0 1.21971")
    bread = copy.deepcopy(src.find(f'.//body[@name="{BREAD}"]'))
    bread.set("pos", f"0.45 {-0.13 + table_y_offset} 1.3")
    bread.set("quat", "0.7071067811865476 0.7071067811865475 0 0")
    parts = ET.Element("worldbody")
    parts.extend([fridge, bread])
    root = ET.Element("mujoco")
    root.append(copy.deepcopy(src.find("compiler")))
    root.append(copy.deepcopy(src.find("option")))
    root.append(copy.deepcopy(src.find("default")))
    asset = ET.SubElement(root, "asset")
    meshes = {g.get("mesh") for g in parts.iter("geom") if g.get("mesh")}
    mats = {g.get("material") for g in parts.iter("geom") if g.get("material")}
    textures = {
        m.get("texture") for m in src.find("asset") if m.tag == "material" and m.get("name") in mats
    }
    for e in src.find("asset"):
        if e.get("name") in meshes | mats | textures:
            e = copy.deepcopy(e)
            if e.get("file"):
                e.set("file", str((path.parent / e.get("file")).resolve()))
            asset.append(e)
    root.append(parts)
    fs = mujoco.MjSpec.from_string(ET.tostring(root).decode())
    rs = mujoco.MjSpec.from_file(str(assets / "robots/rby1m/rby1_v1.2_site_control.xml"))
    configure_rby1_gripper_servos(rs)
    for b in rs.bodies:
        if b.name.startswith("robot_0/"):
            b.gravcomp = 1
    rs.attach(fs, prefix="", suffix="", frame=rs.worldbody.add_frame())
    rs.worldbody.add_geom(
        name="table_top",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0.43, 0.01 + table_y_offset, table_height - 0.035],
        size=[0.11, 0.42, 0.035],
        rgba=[0.35, 0.5, 0.7, 1],
    )
    rs.worldbody.add_geom(
        name="table_pedestal",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0.49, 0.39 + table_y_offset, (table_height - 0.07) / 2],
        size=[0.025, 0.025, (table_height - 0.07) / 2],
        rgba=[0.3, 0.32, 0.35, 1],
    )
    rs.worldbody.add_geom(
        name="table_foot",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0.49, 0.39 + table_y_offset, 0.02],
        size=[0.07, table_foot_half_y, 0.02],
        rgba=[0.3, 0.32, 0.35, 1],
    )
    rs.worldbody.add_geom(
        name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.1], rgba=[0.25, 0.3, 0.35, 1]
    )
    rs.worldbody.add_light(pos=[0, -1, 3], dir=[0, 0, -1])
    rs.option = fs.option
    rs.memory = 256 * 1024 * 1024
    rs.compiler.balanceinertia = True
    rs.option.timestep = 0.002
    rs.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    m = rs.compile()
    d = mujoco.MjData(m)
    for side in ["right", "left"]:
        for i, q in enumerate([0.5, 0, 0, -2.3, 0, -0.5, 0]):
            d.joint(f"robot_0/{side}_arm_{i}").qpos[0] = q
        for i, q in [(1, -0.05), (2, 0.05)]:
            d.joint(f"robot_0/gripper_finger_{side[0]}{i}").qpos[0] = q
    d.joint("robot_0/base_x").qpos[0] = base_x
    d.joint("robot_0/base_y").qpos[0] = base_y
    for i in range(m.nu):
        if m.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT:
            d.ctrl[i] = d.qpos[m.jnt_qposadr[m.actuator_trnid[i, 0]]]
    d.actuator("robot_0/base_x_act").ctrl[0] = base_x
    d.actuator("robot_0/base_y_act").ctrl[0] = base_y
    mujoco.mj_forward(m, d)
    # Initial door angle. 90 is the fully open pose the placement was first
    # validated against; the robot can only swing the real handle to about 60,
    # so integrating a physical door opening needs the smaller angle to work.
    d.joint(JOINT).qpos[0] = -np.radians(door_angle_deg)
    # This is a French-door fridge. The left door was always left shut, so it can be
    # the thing blocking the approach rather than the right door not being wide
    # enough. Its hinge runs the other way, hence the positive sign.
    if door2_angle_deg:
        d.joint(DOOR2_JOINT).qpos[0] = np.radians(door2_angle_deg)
    mujoco.mj_forward(m, d)
    vertices, _ = collision_mesh(m, d, lambda bid: m.body(bid).name.startswith(BREAD_PREFIX))
    d.joint(BREAD_JOINT).qpos[2] += table_height + 0.002 - vertices[:, 2].min()
    mujoco.mj_forward(m, d)
    return m, d


import traceback
from importlib.metadata import version
from math import ceil

import imageio.v2 as imageio
from curobo._src.geom.types import Cuboid, Mesh, SceneCfg
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation as R

from research.cross_episode_memory.curobo_current import RightArmPlanner

NS = "robot_0/"
# Which interior shelf site to place on. Overridable because the reachable shelf
# depends on how far the door is open: at 60 degrees, the angle the robot can
# actually pull the handle to, the door blocks the approach to some of them.
SHELF_SUFFIX = os.environ.get("FRIDGE_SHELF", "FridgeBodyMeshf017e276Receptacle2_2")


class FridgeTransfer:
    def __init__(self, args):
        self.args = args
        self.output = args.output
        self.output.mkdir(parents=True, exist_ok=True)
        if getattr(args, "kitchen", False):
            # The real FloorPlan3 kitchen instead of the isolated rig. See
            # research/cross_episode_memory/kitchen_scene.py for why the floor gets a
            # collision plane and the loose props are frozen.
            from research.cross_episode_memory.kitchen_scene import make_kitchen

            self.model, self.data, self.kitchen_stats = make_kitchen(
                args.assets,
                RBY1MConfig(),
                (F, BREAD_PREFIX),
                robot_xy=(args.base_x, getattr(args, "base_y", 0.0)),
            )
            print(json.dumps({"kitchen": self.kitchen_stats}), flush=True)
        else:
            self.model, self.data = make_scene(
                args.assets,
                args.fridge_x,
                args.table_height,
                args.base_x,
                getattr(args, "base_y", 0.0),
                getattr(args, "table_y_offset", 0.0),
                getattr(args, "table_foot_half_y", 0.07),
                getattr(args, "door_angle", 90.0),
                getattr(args, "door2_angle", 0.0),
            )
        if args.soft_finger:
            # Explicit simulation assumption: a finite contact patch resists
            # twisting/rolling, unlike a single three-dimensional point contact.
            for gid in range(self.model.ngeom):
                if "ee_finger_r" in self.model.body(self.model.geom_bodyid[gid]).name and (
                    self.model.geom_contype[gid] or self.model.geom_conaffinity[gid]
                ):
                    self.model.geom_condim[gid] = 6
                    self.model.geom_priority[gid] = 1
                    self.model.geom_friction[gid] = [1.0, 0.005, 0.002]
        self.model.vis.headlight.ambient[:] = 0.5
        self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        self.cameras = [mujoco.MjvCamera(), mujoco.MjvCamera()]
        for camera, yaw, distance, lookat in zip(
            self.cameras, [45, 315], [3.5, 2.1], [[0.5, 0, 1.15], [0.65, -0.15, 1.5]]
        ):
            camera.azimuth = yaw
            camera.elevation = -20
            camera.distance = distance
            camera.lookat[:] = lookat
        self.writer = imageio.get_writer(str(self.output / "fridge_transfer.mp4"), fps=25)
        self.stage = "initialization"
        self.next_frame = 0.0
        self.trace = []
        self.report = {
            "success": False,
            "scope": "actual task bread, table to open fridge; isolated component",
            "object": BREAD,
            "fridge": F + "_1_0_0",
            "stages": [],
            "versions": {name: version(name) for name in ("mujoco", "nvidia-curobo")},
            "execution": "actuator controls only after initialization; no weld or teleport",
            "max_unintended_robot_penetration_m": 0.0,
            "grasp_depth_m": args.grasp_depth,
            "grasp_y_offset_m": args.grasp_y_offset,
            "grip_force_limit_n": args.grip_force,
            "grip_position_gain": args.grip_kp,
            "soft_finger_contact": args.soft_finger,
            "finger_friction_override": [1.0, 0.005, 0.002] if args.soft_finger else None,
            "motion_slowdown": args.motion_slowdown,
            "fridge_x": args.fridge_x,
            "base_x": args.base_x,
            "table_height": args.table_height,
        }
        self.planner = None
        self.arm_aids = self.actuator_ids([f"right_arm_{i}" for i in range(7)])
        self.bread_bids = {
            b for b in range(self.model.nbody) if self.model.body(b).name.startswith(BREAD_PREFIX)
        }
        self.fridge_bids = {
            b for b in range(self.model.nbody) if self.model.body(b).name.startswith(F)
        }
        # The door handle, gripped on purpose when operating the door.
        self.handle_bid = self.model.body(F + "_1_3_0").id
        self.table_gids = {
            g for g in range(self.model.ngeom) if self.model.geom(g).name.startswith("table_")
        }
        self.shelf_id = next(
            i for i in range(self.model.nsite) if self.model.site(i).name.endswith(SHELF_SUFFIX)
        )
        self.attached = False
        self.model.actuator_forcerange[self.model.actuator(NS + "right_finger_act").id] = [
            -args.grip_force,
            args.grip_force,
        ]

        aid = self.model.actuator(NS + "right_finger_act").id
        self.model.actuator_gainprm[aid, 0] = args.grip_kp
        self.model.actuator_biasprm[aid, 1] = -args.grip_kp

    def tcp(self):
        t = np.eye(4)
        site = self.data.site(NS + "ee_site_r")
        t[:3, :3], t[:3, 3] = site.xmat.reshape(3, 3), site.xpos
        return t

    def bread_pose(self):
        t = np.eye(4)
        body = self.data.body(BREAD)
        t[:3, :3], t[:3, 3] = body.xmat.reshape(3, 3), body.xpos
        return t

    def bread_vertices(self):
        return collision_mesh(self.model, self.data, lambda b: b in self.bread_bids)[0]

    def contacts(self):
        fingers = set()
        for c in self.data.contact:
            b1, b2 = self.model.geom_bodyid[[c.geom1, c.geom2]]
            if b1 in self.bread_bids or b2 in self.bread_bids:
                other = b2 if b1 in self.bread_bids else b1
                name = self.model.body(other).name
                if "ee_finger_r" in name:
                    fingers.add(name)
        return sorted(fingers)

    def unintended_penetration(self):
        worst = 0.0
        for c in self.data.contact:
            b1, b2 = self.model.geom_bodyid[[c.geom1, c.geom2]]
            n1, n2 = self.model.body(b1).name, self.model.body(b2).name
            if n1.startswith(NS):
                robot, other, gid = n1, b2, c.geom2
            elif n2.startswith(NS):
                robot, other, gid = n2, b1, c.geom1
            else:
                continue
            if other in self.bread_bids and "ee_finger_r" in robot:
                continue
            # Fingers on the door handle are the task, not an accident. Matches the
            # door check's own rule: only finger-on-handle is excused, the body
            # hitting the panel still counts. This value is a running maximum over
            # the whole run, so counting the grasp here poisoned every later check --
            # the run failed at pregrasp on contact from minutes earlier.
            if other == self.handle_bid and "ee_finger_r" in robot:
                continue
            if other in self.fridge_bids or other in self.bread_bids or gid in self.table_gids:
                worst = max(worst, -float(c.dist))
        return worst

    def record_camera_frames(self, frames):
        """Optional raw observation recording for derived component checks."""

    def finalize_report(self):
        """Hook for derived checks to add summary metrics before the report is written."""

    def before_step(self):
        """Hook for derived checks that steer actuators every physics step.

        Runs before `mj_step`, so anything written here is a command for the step
        about to be taken. Overrides must write `ctrl`, never `qpos`, to preserve
        the actuator-only guarantee this check reports.
        """

    def tick(self, seconds):
        for _ in range(ceil(seconds / self.model.opt.timestep)):
            self.before_step()
            mujoco.mj_step(self.model, self.data)
            self.report["max_unintended_robot_penetration_m"] = max(
                self.report["max_unintended_robot_penetration_m"], self.unintended_penetration()
            )
            if self.data.time + 1e-9 >= self.next_frame:
                frames = []
                for camera in self.cameras:
                    self.renderer.update_scene(self.data, camera=camera)
                    frames.append(self.renderer.render().copy())
                self.record_camera_frames(frames)
                frame = Image.fromarray(np.concatenate(frames, axis=1))
                draw = ImageDraw.Draw(frame)
                draw.rectangle((0, 0, frame.width, 48), fill="black")
                draw.text((10, 8), f"RB-Y1 / cuRobo | {self.stage}", fill="white")
                draw.text(
                    (10, 27),
                    f"Bread to fridge | simulation, {self.args.grip_force:g} N, "
                    f"{'finite-pad' if self.args.soft_finger else 'point'} contact | t={self.data.time:.2f}s",
                    fill="white",
                )
                draw.text((650, 8), "Manipulation detail", fill="white")
                if len(frames) > 2:
                    draw.text((1300, 8), "HEAD CAMERA | raw view recorded separately", fill="white")
                self.writer.append_data(np.asarray(frame))
                self.trace.append(
                    {
                        "time": float(self.data.time),
                        "stage": self.stage,
                        "qpos": self.data.qpos.tolist(),
                        "bread_pose": self.bread_pose().tolist(),
                        "tcp": self.tcp().tolist(),
                        "finger_contacts": self.contacts(),
                    }
                )
                self.next_frame += 0.04

    def record(self, **metrics):
        item = {"stage": self.stage, "time": float(self.data.time), **metrics}
        self.report["stages"].append(item)
        print(json.dumps(item), flush=True)

    def load_world(self):
        v, f = collision_mesh(self.model, self.data, lambda b: b in self.fridge_bids)
        v[:, 2] -= 0.005
        meshes = [
            Mesh(name="fridge", vertices=v.tolist(), faces=f.tolist(), pose=[0, 0, 0, 1, 0, 0, 0])
        ]
        boxes = []
        for gid in self.table_gids:
            boxes.append(
                Cuboid(
                    name=f"table_{gid}",
                    pose=list(self.data.geom_xpos[gid] - [0, 0, 0.005]) + [1, 0, 0, 0],
                    dims=list(2 * self.model.geom_size[gid]),
                )
            )
        self.planner.planner.update_world(SceneCfg(mesh=meshes, cuboid=boxes))
        self.record(collision_mesh_vertices=len(v), collision_mesh_triangles=len(f))

    def move(self, stage, pose):
        self.stage = stage
        positions = [float(self.data.joint(NS + n).qpos[0]) for n in self.planner.names]
        goal = list(pose[:3, 3] - [0, 0, 0.005]) + list(
            R.from_matrix(pose[:3, :3]).as_quat(scalar_first=True)
        )
        trajectory = self.planner.plan(positions, goal)
        dt = (
            ceil(self.planner.dt * self.args.motion_slowdown / self.model.opt.timestep)
            * self.model.opt.timestep
        )
        for q in trajectory:
            self.data.ctrl[self.arm_aids] = q
            self.tick(dt)
        self.tick(0.4)
        error = float(np.linalg.norm(self.tcp()[:3, 3] - pose[:3, 3]))
        self.record(
            tcp_error_m=error,
            finger_contacts=self.contacts(),
            bread_position=self.data.body(BREAD).xpos.tolist(),
        )
        if error > 0.025:
            raise RuntimeError(f"TCP missed {stage} by {error:.3f} m")
        if self.report["max_unintended_robot_penetration_m"] > 0.003:
            raise RuntimeError("Unintended robot contact exceeded 3 mm penetration")
        if self.attached:
            slip = float(
                np.linalg.norm(
                    (np.linalg.inv(self.tcp()) @ self.bread_pose())[:3, 3]
                    - self.grasp_relative[:3, 3]
                )
            )
            self.record(payload_slip_m=slip)
            if slip > 0.035 or not self.contacts():
                raise RuntimeError("Bread slipped or lost gripper contact during carry")

    def support_contacts(self, table=False):
        matches = []
        for c in self.data.contact:
            b1, b2 = self.model.geom_bodyid[[c.geom1, c.geom2]]
            if b1 in self.bread_bids:
                other, gid, normal = b2, c.geom2, -c.frame[:3]
            elif b2 in self.bread_bids:
                other, gid, normal = b1, c.geom1, c.frame[:3]
            else:
                continue
            if (gid in self.table_gids if table else other in self.fridge_bids) and normal[2] > 0.7:
                matches.append({"geom": self.model.geom(gid).name, "point": c.pos.tolist()})
        return matches

    def after_placement(self):
        """Hook after the loaf is on the shelf and released, before declaring success."""

    def prepare_pickup(self):
        """Optional navigation before the manipulation-only check."""

    def transport_payload(self):
        """Optional navigation after the physical lift, before shelf placement."""

    def actuator_ids(self, names):
        """MuJoCo actuator ids for planner joint names.

        The torso actuators are called link1_act..link6_act, not torso_N_act, which
        is why a search for "torso" actuators turns up nothing. They drive torso_0
        through torso_5.
        """
        ids = []
        for name in names:
            if name.startswith("torso_"):
                ids.append(self.model.actuator(NS + f"link{int(name.split('_')[1]) + 1}_act").id)
            elif name.startswith("right_arm_"):
                ids.append(self.model.actuator(NS + f"right_arm_{int(name.split('_')[2]) + 1}_act").id)
            else:
                ids.append(self.model.actuator(NS + name + "_act").id)
        return ids

    def unlocked_joints(self):
        """Extra joints the planner may move, beyond the right arm."""
        count = int(getattr(self.args, "use_torso", 0))
        return tuple(f"torso_{i}" for i in range(count))

    def make_planner(self):
        return RightArmPlanner(
            self.args.assets / "robots/rby1m",
            {
                "base_x": self.args.base_x,
                "base_y": getattr(self.args, "base_y", 0.0),
                "base_theta": 0.0,
            },
            unlock=self.unlocked_joints(),
            activation_distance=getattr(self.args, "clearance", 0.005),
        )

    def run(self):
        try:
            self.tick(0.6)
            if self.report["max_unintended_robot_penetration_m"] > 0.003:
                raise RuntimeError("Initial layout has unintended robot penetration")
            self.prepare_pickup()
            start = self.bread_pose().copy()
            bounds = self.bread_vertices()
            self.record(
                bread_bounds=[bounds.min(0).tolist(), bounds.max(0).tolist()],
                table_contacts=self.support_contacts(table=True),
            )
            if not self.support_contacts(table=True):
                raise RuntimeError("Bread was not initially supported by the table")
            self.planner = self.make_planner()
            self.arm_aids = self.actuator_ids(self.planner.names)
            self.load_world()
            grasp = np.eye(4)
            grasp[:3, :3] = R.from_quat(
                [0, np.sqrt(0.5), np.sqrt(0.5), 0], scalar_first=True
            ).as_matrix()
            grasp[:3, 3] = [
                (bounds[:, 0].min() + bounds[:, 0].max()) / 2,
                (bounds[:, 1].min() + bounds[:, 1].max()) / 2 + self.args.grasp_y_offset,
                bounds[:, 2].max() + self.args.grasp_depth,
            ]
            pre = grasp.copy()
            pre[2, 3] += 0.12
            self.move("pregrasp", pre)
            self.move("grasp approach", grasp)
            self.stage = "close around loaf"
            self.data.actuator(NS + "right_finger_act").ctrl[0] = 0
            self.tick(1)
            self.record(finger_contacts=self.contacts())
            if len(self.contacts()) != 2:
                raise RuntimeError("No bilateral loaf grasp")
            lift = self.tcp()
            lift[2, 3] += 0.15
            self.move("lift bread", lift)
            lifted = float(self.bread_pose()[2, 3] - start[2, 3])
            self.record(lifted_m=lifted)
            if lifted < 0.10 or len(self.contacts()) != 2 or self.support_contacts(table=True):
                raise RuntimeError("Bread did not remain physically grasped after lift")
            self.grasp_relative = np.linalg.inv(self.tcp()) @ self.bread_pose()
            if self.args.pickup_only:
                self.report["success"] = True
                self.report["scope"] = "pickup-only diagnostic; shelf placement not attempted"
                self.stage = "PASS: physical loaf pickup only"
                self.tick(1)
                return 0
            # Attach a conservative local bounding box to cuRobo only.
            object_pose = self.bread_pose()
            local = (self.bread_vertices() - object_pose[:3, 3]) @ object_pose[:3, :3]
            lo, hi = local.min(0), local.max(0)
            center = (lo + hi) / 2
            world_center = object_pose[:3, 3] + object_pose[:3, :3] @ center
            positions = [float(self.data.joint(NS + n).qpos[0]) for n in self.planner.names]
            self.planner.attach_box(
                positions,
                list(world_center - [0, 0, 0.005])
                + list(R.from_matrix(object_pose[:3, :3]).as_quat(scalar_first=True)),
                (hi - lo) / 2,
            )
            self.attached = True
            self.transport_payload()
            staging = self.tcp()
            staging[0, 3] += 0.08
            staging[1, 3] -= 0.04
            # Rotating the loaf flat happens here, in front of the fridge. With the
            # right door only 60 degrees open -- the most the arm can pull it -- the
            # loaf sweeps into the panel partway through the turn (measured: fails at
            # 30 to 40 degrees of the rotation). Backing the staging pose away from
            # the fridge first gives the loaf room to turn outside the door's space.
            staging[0, 3] -= getattr(self.args, "orient_standoff", 0.0)
            self.move("clear table with loaf upright", staging)
            object_pose = self.bread_pose()
            self.grasp_relative = np.linalg.inv(self.tcp()) @ object_pose
            # Roll the loaf onto its side so the hand approaches the shelf horizontally.
            rotated = object_pose.copy()
            rotated[:3, :3] = R.from_euler("y", -90, degrees=True).as_matrix() @ object_pose[:3, :3]
            rotated[:3, 3] = [0.67, -0.25, 1.70]
            # Short Cartesian waypoints prevent a redundant IK branch from swinging
            # the held loaf through an uncontrolled wrist rotation.
            for index in range(1, 10):
                fraction = index / 9
                waypoint = object_pose.copy()
                waypoint[:3, :3] = (
                    R.from_euler("y", -90 * fraction, degrees=True).as_matrix()
                    @ object_pose[:3, :3]
                )
                waypoint[:3, 3] = (1 - fraction) * object_pose[:3, 3] + fraction * rotated[:3, 3]
                self.move(
                    f"orient loaf {index * 10} degrees",
                    waypoint @ np.linalg.inv(self.grasp_relative),
                )
            shelf = self.data.site_xpos[self.shelf_id].copy()
            target_obj = rotated.copy()
            target_obj[:3, 3] = [shelf[0] - 0.06, shelf[1], shelf[2] + 0.04]
            # Probe the actual shelf surface, rather than using the fridge AABB roof.
            hit = np.array([-1], dtype=np.int32)
            distance = mujoco.mj_ray(
                self.model,
                self.data,
                np.array([shelf[0], shelf[1], shelf[2] + 0.04]),
                np.array([0.0, 0.0, -1.0]),
                np.array([0, 0, 0, 0, 1, 0], dtype=np.uint8),
                True,
                -1,
                hit,
            )
            if distance < 0 or self.model.geom_bodyid[hit[0]] not in self.fridge_bids:
                raise RuntimeError("Could not identify the physical interior shelf")
            shelf_z = float(shelf[2] + 0.04 - distance)
            rotated_local = local @ target_obj[:3, :3].T
            target_obj[2, 3] = shelf_z - rotated_local[:, 2].min() + 0.04
            self.report["target_shelf_surface_z"] = shelf_z
            self.report["target_shelf"] = self.model.site(self.shelf_id).name
            outside = target_obj.copy()
            outside[0, 3] -= 0.25
            outside[2, 3] += 0.02
            self.move("approach open shelf", outside @ np.linalg.inv(self.grasp_relative))
            self.move("insert loaf", target_obj @ np.linalg.inv(self.grasp_relative))
            self.stage = "release on shelf"
            self.data.actuator(NS + "right_finger_act").ctrl[0] = -0.05
            self.tick(1)
            self.planner.detach_block()
            self.attached = False
            back = self.tcp()
            back[0, 3] -= 0.16
            self.move("withdraw from loaf", back)
            self.tick(1.5)
            supports = self.support_contacts()
            final_bounds = self.bread_vertices()
            shelf_half_x = self.model.site_size[self.shelf_id][2]
            shelf_half_y = self.model.site_size[self.shelf_id][0]
            inside = bool(
                final_bounds[:, 0].min() > shelf[0] - shelf_half_x - 0.01
                and final_bounds[:, 0].max() < shelf[0] + shelf_half_x + 0.01
                and final_bounds[:, 1].min() > shelf[1] - shelf_half_y - 0.01
                and final_bounds[:, 1].max() < shelf[1] + shelf_half_y + 0.01
                and abs(final_bounds[:, 2].min() - shelf_z) < 0.015
            )
            speed = float(np.linalg.norm(self.data.joint(BREAD_JOINT).qvel[:3]))
            self.record(
                shelf_contacts=supports,
                inside_target_shelf=inside,
                released=not self.contacts(),
                bread_speed_m_s=speed,
            )
            if not supports or not inside or self.contacts() or speed > 0.03:
                raise RuntimeError(
                    "Final loaf placement was not supported, inside the shelf, released and settled"
                )
            self.after_placement()
            self.report["success"] = True
            self.stage = "PASS: bread placed inside fridge"
            self.tick(1)
        except Exception as exc:
            self.report["error"] = str(exc)
            self.report["traceback"] = traceback.format_exc()
            print(self.report["traceback"], flush=True)
            self.stage = "FAIL: " + self.stage
            self.tick(0.5)
        finally:
            self.writer.close()
            self.renderer.close()
            self.finalize_report()
            (self.output / "report.json").write_text(json.dumps(self.report, indent=2))
            (self.output / "trace.json").write_text(json.dumps(self.trace))
        return 0 if self.report["success"] else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--base-x", type=float, default=0.20)
    p.add_argument("--base-y", type=float, default=0.0)
    p.add_argument("--fridge-x", type=float, default=0.95)
    p.add_argument("--table-height", type=float, default=0.9)
    p.add_argument("--grasp-depth", type=float, default=-0.003)
    p.add_argument("--grasp-y-offset", type=float, default=0.0)
    p.add_argument("--motion-slowdown", type=float, default=3.0)
    p.add_argument("--grip-force", type=float, default=80.0)
    p.add_argument("--grip-kp", type=float, default=2000.0)
    p.add_argument(
        "--soft-finger", action="store_true", help="Explicit uncalibrated finite-pad contact model"
    )
    p.add_argument("--pickup-only", action="store_true")
    p.add_argument(
        "--clearance",
        type=float,
        default=0.005,
        help="How far planned paths keep off obstacles, in metres.",
    )
    p.add_argument(
        "--use-torso",
        type=int,
        default=0,
        help="Let the planner move this many torso joints (0-6). The shipped cuRobo "
        "config pins all six, so the arm reaches from a rigid trunk.",
    )
    p.add_argument(
        "--orient-standoff",
        type=float,
        default=0.0,
        help="Metres to back away from the fridge before turning the loaf flat, so "
        "it does not sweep into a partly open door.",
    )
    p.add_argument(
        "--door2-angle",
        type=float,
        default=0.0,
        help="Opening angle of the fridge's OTHER door, in degrees.",
    )
    p.add_argument(
        "--door-angle",
        type=float,
        default=90.0,
        help="Initial fridge door angle in degrees. The robot can only swing the "
        "real handle to about 60, so use that when the door is opened for real.",
    )
    raise SystemExit(FridgeTransfer(p.parse_args()).run())
