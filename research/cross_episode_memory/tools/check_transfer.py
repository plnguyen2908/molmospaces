#!/usr/bin/env python3
"""Physical RB-Y1 / cuRobo A-to-B check, before kitchen integration.

Uses the pinned robot in a deliberately small scene with two supports and a
5 cm block. Only initial conditions write qpos; execution writes actuator ctrl.
Always writes an MP4 and JSON report, including when planning or checks fail.
Run with the environment containing NVIDIA cuRobo 1.0, MuJoCo and imageio:

    CUDA_VISIBLE_DEVICES=2 MUJOCO_GL=egl PYTHONPATH=. python \
      research/cross_episode_memory/tools/check_transfer.py \
      --robot-dir "$MLSPACES_ASSETS_DIR/robots/rby1m" --output /tmp/transfer

This is an oracle manipulation component check, NOT a kitchen reorder result.
"""

import argparse
import json
import os
import traceback
from importlib.metadata import version
from math import ceil
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from molmo_spaces.robots.rby1_actuators import configure_rby1_gripper_servos
from research.cross_episode_memory.curobo_current import RightArmPlanner

NS = "robot_0/"
REST = [0.5, 0, 0, -2.3, 0, -0.5, 0]
A = np.array([0.40, -0.35, 0.82])
B = np.array([0.40, 0.00, 0.82])
HALF = 0.025


def make_scene(robot_dir):
    spec = mujoco.MjSpec.from_file(str(robot_dir / "rby1_v1.2_site_control.xml"))
    spec.option.timestep = 0.002
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    for body in spec.bodies:
        if body.name.startswith(NS):
            body.gravcomp = 1
    configure_rby1_gripper_servos(spec)
    spec.worldbody.add_geom(
        name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.1], rgba=[0.22, 0.25, 0.28, 1]
    )
    spec.worldbody.add_light(pos=[0, -1, 3], dir=[0, 0, -1])
    for name, pos, color in (("A", A, [0.25, 0.5, 0.8, 1]), ("B", B, [0.25, 0.7, 0.4, 1])):
        spec.worldbody.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=pos - [0, 0, 0.04],
            size=[0.14, 0.14, 0.04],
            rgba=color,
        )
    body = spec.worldbody.add_body(name="block", pos=A + [0, 0, HALF + 0.001])
    body.add_freejoint(name="block_free")
    body.add_geom(
        name="block_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[HALF] * 3,
        mass=0.08,
        friction=[1, 0.01, 0.001],
        rgba=[1, 0.5, 0.05, 1],
    )
    model = spec.compile()
    model.vis.headlight.ambient[:] = [0.5, 0.5, 0.5]
    data = mujoco.MjData(model)
    initial = {}
    for side in ("left", "right"):
        initial.update({f"{side}_arm_{i}": q for i, q in enumerate(REST)})
        letter = side[0]
        initial.update({f"gripper_finger_{letter}1": -0.05, f"gripper_finger_{letter}2": 0.05})
    for name, pos in initial.items():
        data.joint(NS + name).qpos[0] = pos
    for i in range(model.nu):
        if model.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT:
            jid = model.actuator_trnid[i, 0]
            data.ctrl[i] = data.qpos[model.jnt_qposadr[jid]]
    mujoco.mj_forward(model, data)
    return model, data


class Check:
    def __init__(self, robot_dir, output):
        self.model, self.data = make_scene(robot_dir)
        self.output = output
        output.mkdir(parents=True, exist_ok=True)
        self.report = {
            "scope": "isolated RB-Y1 block A-to-B; no navigation or fridge",
            "success": False,
            "stages": [],
            "teleports_during_execution": 0,
            "versions": {name: version(name) for name in ("mujoco", "nvidia-curobo")},
            "robot_dir": str(robot_dir),
        }
        self.stage = "initialization"
        self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        self.camera = mujoco.MjvCamera()
        self.camera.lookat[:] = [0.35, -0.05, 0.8]
        self.camera.distance = 2.5
        self.camera.azimuth = 135
        self.camera.elevation = -25
        self.writer = imageio.get_writer(str(output / "transfer.mp4"), fps=25)
        self.next_frame = 0
        self.samples = []
        self.planner = None
        self.robot_dir = robot_dir

    def tick(self, seconds):
        for _ in range(round(seconds / self.model.opt.timestep)):
            mujoco.mj_step(self.model, self.data)
            if self.data.time + 1e-9 >= self.next_frame:
                self.renderer.update_scene(self.data, camera=self.camera)
                frame = Image.fromarray(self.renderer.render())
                draw = ImageDraw.Draw(frame)
                draw.rectangle((0, 0, 640, 45), fill="black")
                draw.text(
                    (12, 8),
                    f"RB-Y1 / cuRobo | {self.stage} | t={self.data.time:.2f}s",
                    fill="white",
                )
                draw.text((12, 25), "Physical component check: blue A -> green B", fill="white")
                self.writer.append_data(np.asarray(frame))
                self.next_frame += 0.04
                self.samples.append(
                    {
                        "time": float(self.data.time),
                        "stage": self.stage,
                        "block": self.data.body("block").xpos.tolist(),
                        "arm": [
                            float(self.data.joint(NS + f"right_arm_{i}").qpos[0]) for i in range(7)
                        ],
                    }
                )

    def record(self, name, **metrics):
        item = {"stage": name, "time": float(self.data.time), **metrics}
        self.report["stages"].append(item)
        print(json.dumps(item), flush=True)

    def move(self, stage, xyz):
        self.stage = stage
        current = [float(self.data.joint(NS + name).qpos[0]) for name in self.planner.names]
        goal = list(np.asarray(xyz) - [0, 0, 0.005]) + [0, 1, 0, 0]  # TCP +Z points down.
        trajectory = self.planner.plan(current, goal)
        # Round UP to a physics step so playback cannot speed up the planned motion.
        waypoint_dt = ceil(self.planner.dt / self.model.opt.timestep) * self.model.opt.timestep
        for waypoint in trajectory:
            for name, target in zip(self.planner.names, waypoint):
                jid = self.model.joint(NS + name).id
                aid = np.flatnonzero(
                    (self.model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT)
                    & (self.model.actuator_trnid[:, 0] == jid)
                )[0]
                self.data.ctrl[aid] = target
            self.tick(waypoint_dt)
        self.tick(0.4)
        actual = self.data.site(NS + "ee_site_r").xpos
        error = float(np.linalg.norm(actual - xyz))
        self.record(stage, tcp_error_m=error, waypoints=len(trajectory))
        if error > 0.025:
            raise RuntimeError(f"{stage}: actual TCP missed target by {error:.3f} m")

    def finger_contacts(self):
        touched = set()
        block_id = self.model.geom("block_geom").id
        for contact in self.data.contact:
            if block_id in (contact.geom1, contact.geom2):
                other = contact.geom2 if contact.geom1 == block_id else contact.geom1
                body_name = self.model.body(self.model.geom_bodyid[other]).name
                if "ee_finger_r" in body_name:
                    touched.add(body_name)
        return touched

    def run(self):
        try:
            self.tick(0.5)
            start = self.data.body("block").xpos.copy()
            self.record("settled", block=start.tolist())
            self.planner = RightArmPlanner(
                self.robot_dir, {"base_x": 0.0, "base_y": 0.0, "base_theta": 0.0}
            )
            from curobo._src.geom.types import Cuboid, SceneCfg

            self.planner.planner.update_world(
                SceneCfg(
                    cuboid=[
                        Cuboid(
                            name=name,
                            pose=list(pos - [0, 0, 0.04]) + [1, 0, 0, 0],
                            dims=[0.28, 0.28, 0.08],
                        )
                        for name, pos in (("A", A), ("B", B))
                    ]
                )
            )
            self.move("pregrasp", start + [0, 0, 0.15])
            self.move("grasp approach", start + [0, 0, 0.005])
            self.stage = "close gripper"
            self.data.actuator(NS + "right_finger_act").ctrl[0] = 0
            self.tick(1)
            contacts = self.finger_contacts()
            self.record(self.stage, finger_contacts=sorted(contacts))
            if len(contacts) != 2:
                raise RuntimeError("No bilateral finger contact; stopping before lift")
            self.move("lift", start + [0, 0, 0.18])
            lift = float(self.data.body("block").xpos[2] - start[2])
            self.record("lift check", lifted_m=lift)
            if lift < 0.10 or len(self.finger_contacts()) != 2:
                raise RuntimeError("Block did not remain grasped during lift")
            positions = [float(self.data.joint(NS + name).qpos[0]) for name in self.planner.names]
            block = self.data.body("block")
            block_pose = list(block.xpos - [0, 0, 0.005]) + list(block.xquat)
            self.planner.attach_block(positions, block_pose, HALF)
            self.record("payload collision geometry enabled")
            self.move("carry to B", B + [0, 0, HALF + 0.18])
            if len(self.finger_contacts()) != 2:
                raise RuntimeError("Grasp lost during carry")
            self.move("lower onto B", B + [0, 0, HALF + 0.020])
            self.stage = "release"
            self.data.actuator(NS + "right_finger_act").ctrl[0] = -0.05
            self.tick(1)
            self.planner.detach_block()
            self.move("withdraw", B + [0, 0, HALF + 0.18])
            self.tick(1)
            final = self.data.body("block").xpos.copy()
            support = False
            ids = {self.model.geom("block_geom").id, self.model.geom("B").id}
            for contact in self.data.contact:
                support |= {contact.geom1, contact.geom2} == ids
            passed = bool(
                support and np.linalg.norm(final[:2] - B[:2]) < 0.10 and not self.finger_contacts()
            )
            self.record(
                "placement check",
                block=final.tolist(),
                supported_on_B=bool(support),
                released=not self.finger_contacts(),
            )
            if not passed:
                raise RuntimeError("Placement did not satisfy support and release checks")
            self.report["success"] = True
            self.stage = "PASS: lifted, carried, released on B"
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
            (self.output / "report.json").write_text(json.dumps(self.report, indent=2))
            (self.output / "trace.json").write_text(json.dumps(self.samples))
        return 0 if self.report["success"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return Check(args.robot_dir, args.output).run()


if __name__ == "__main__":
    raise SystemExit(main())
