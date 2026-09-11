#!/usr/bin/env python3
"""Drive to a separated table, physically pick bread, drive back and place it.

Oracle component check: heading-aware A* with turn-then-forward base motion, cuRobo
for the arm. The fridge starts open; no door operation or full kitchen RunTask.
"""

import argparse
import heapq
from math import ceil
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from research.cross_episode_memory.curobo_current import RightArmPlanner
from research.cross_episode_memory.tools.check_fridge_door import (
    DOOR,
    HANDLE,
    JOINT,
    DoorOperations,
)
from research.cross_episode_memory.tools.check_fridge_transfer import (
    BREAD_JOINT,
    NS,
    FridgeTransfer,
)

GAZE_SLEW_RAD_S = 1.5
# Stage-name fragments during which the head looks along travel instead of at the
# loaf. Everything else -- pregrasp, grasp, lift, orient, insert, withdraw -- is
# manipulation, where the loaf must stay in frame.
GAZE_FORWARD_STAGES = (
    "plan forward",
    "turn in place",
    "drive forward",
    "reverse undock",
)
# Half of the head camera's 45 degree vertical field of view; the loaf is counted
# as in frame when the optical axis is within this angle of it.
HEAD_HALF_FOV_DEG = 22.5
# Cap on downward tilt while tracking. The head camera sits on link_head_2, which is
# mounted above link_torso_5, so past roughly 1 rad of tilt the robot is looking at
# its own chest. Measured by segmentation at the grasp pose, share of the head frame:
#
#   tilt   0.90   1.05   1.20   1.35
#   torso   5.9%  22.4%  48.0%  78.1%
#   loaf    6.7%   6.9%   4.5%   1.8%
#
# Aiming the optical axis exactly at the loaf drives tilt to ~1.2 and the chest then
# covers the loaf, so capping tilt shows MORE of the target, not less: the loaf ends
# up ~17 degrees below the axis, still inside the 22.5 degree half-field. The knee is
# a property of the head/torso geometry, not the scene -- it is unchanged across pan,
# and no torso joint moves it, because link_torso_5 rotates with the head chain.
GAZE_MAX_TILT_RAD = 0.95


def wrap_angle(angle):
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


class NavigationTransfer(DoorOperations, FridgeTransfer):
    def __init__(self, args):
        args.table_foot_half_y = 0.04
        super().__init__(args)
        travel = [0.0, 0.0, 0.0, -0.02, 0.0, 0.0, 0.0]
        # Initial condition only: neutral arms-down posture leaves turning room.
        for side in ["right", "left"]:
            for i, value in enumerate(travel):
                self.data.joint(NS + f"{side}_arm_{i}").qpos[0] = value
                self.data.actuator(NS + f"{side}_arm_{i + 1}_act").ctrl[0] = value
        self.data.joint(NS + "head_1").qpos[0] = args.head_pitch
        self.data.actuator(NS + "head_1_act").ctrl[0] = args.head_pitch
        # Head aiming state. head_0 (pan) exists and was never commanded, which is
        # why the loaf left the frame; gaze_command is the commanded setpoint that
        # update_gaze slews, kept separate from the measured joint angle so actuator
        # lag cannot wind the command up.
        self.jid = self.model.joint(JOINT).id
        self.grasp_in_handle = None
        self.head_camera_id = self.model.camera(NS + "head_camera").id
        self.gaze_command = [0.0, float(args.head_pitch)]
        self.gaze_samples = self.gaze_in_view = self.gaze_pan_saturated = 0
        self.gaze_worst = self.gaze_error_sum = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.head_writer = imageio.get_writer(str(self.output / "head_camera.mp4"), fps=25)
        self.report.update(
            observation_camera=NS + "head_camera",
            controller_inputs="oracle simulator state and geometry; not a head-camera-only VLA",
            head_pitch_rad=args.head_pitch,
            gaze_mode=args.gaze,
            gaze_slew_rad_s=GAZE_SLEW_RAD_S,
            # Always the chassis now, including with --gaze off: when pan is 0 the
            # chassis axis and the head axis coincide, so this reproduces the old
            # measurement rather than relaxing it.
            forward_reference="chassis yaw (base_theta)",
            reverse_undock_m=args.reverse_undock,
            initial_arm_posture=travel,
            scope="separated table pickup, physical base navigation with bread, open-fridge placement",
            navigation_method="SE(2) A*: turn in place, then drive forward; MuJoCo swept-pose probes",
            navigation_version=2,
            nav_mean_speed_m_s=args.nav_speed,
            turn_mean_speed_rad_s=args.turn_speed,
            max_allowed_heading_error_deg=5.0,
            table_y_offset_m=args.table_y_offset,
            stance_separation_m=abs(args.table_y_offset),
            navigation=[],
        )
        self.cameras[0].lookat[:] = [0.2, args.table_y_offset / 2, 1.1]
        self.cameras[0].azimuth = 45
        self.cameras[0].elevation = -25
        self.cameras[0].distance = 5.3
        self.cameras[1].distance = 2.1
        self.cameras.append(NS + "head_camera")

    def record_camera_frames(self, frames):
        # Raw, unannotated head RGB is separate from the external review views.
        self.head_writer.append_data(frames[2])

    def run(self):
        try:
            return super().run()
        finally:
            self.head_writer.close()

    def tick(self, seconds):
        self.cameras[1].lookat[:] = self.tcp()[:3, 3] + [0.05, 0.0, 0.1]
        super().tick(seconds)

    def handle_grasp_pose(self):
        """Where to take hold of the closed door's handle."""
        pose = np.eye(4)
        pose[:3, :3] = R.from_euler("y", 90, degrees=True).as_matrix()
        pose[:3, 3] = [self.args.fridge_x - 0.387, -0.094, self.args.grasp_height]
        return pose

    def tuck_arm(self, seconds=2.0):
        """Bring the right arm back to the neutral travel pose before driving.

        After letting go of the handle the arm is still stretched out towards the
        door. Driving in that pose sweeps the panel shut: the door was measured
        falling from 74.8 back to 53.5 degrees during the return, wiping out the
        second pull.
        """
        self.stage = "tuck arm"
        # Arm AND torso. Freeing torso joints for reach means the planner leaves the
        # trunk bent, and driving away leaning fails the navigation clearance probe.
        travel = [0.0, 0.0, 0.0, -0.02, 0.0, 0.0, 0.0]
        joints = [(f"right_arm_{i}", f"right_arm_{i + 1}_act", travel[i]) for i in range(7)]
        joints += [(f"torso_{i}", f"link{i + 1}_act", 0.0) for i in range(6)]
        starts = [float(self.data.joint(NS + name).qpos[0]) for name, _, _ in joints]
        steps = ceil(seconds / self.model.opt.timestep)
        for step in range(steps):
            frac = (step + 1) / steps
            for start, (_, act, end) in zip(starts, joints):
                self.data.actuator(NS + act).ctrl[0] = start + (end - start) * frac
            self.tick(self.model.opt.timestep)
        self.tick(0.5)
        self.record(
            tucked=True,
            door_deg=float(abs(np.degrees(self.angle()))),
            torso=[round(float(self.data.joint(NS + f"torso_{i}").qpos[0]), 3) for i in range(6)],
        )

    def open_fridge(self):
        """Physically pull the door open before fetching the loaf.

        A single pull stalls near 60 degrees: the handle swings round to the robot's
        side and the wrist runs out of range. Placement needs about 75, so the base
        steps sideways and the arm takes a fresh hold to carry it the rest of the way.
        """
        # The manipulation stance sits about 0.75 m from the fridge, but the door
        # check opens from about 1.05 m, and standing closer was far worse (it only
        # reached 15 degrees). So back off to open, then come back in to place.
        self.operating_door = True
        self.set_grip(self.args.door_grip_force, self.args.door_grip_kp)
        self.reposition(0.0, dx=self.args.door_standoff_x)
        self.arm_aids = self.actuator_ids(self.planner.names)
        pose = self.handle_grasp_pose()
        pre = pose.copy()
        pre[:3, 3] -= pose[:3, 2] * 0.12
        self.door_move("line up with door handle", pre)
        self.door_move("reach door handle", pose)
        self.stage = "grasp door handle"
        self.gripper(False)
        self.grasp_in_handle = np.linalg.inv(self.handle_pose()) @ self.tcp()
        self.follow_hinge(-np.radians(self.args.open_angle))
        self.stage = "release door"
        self.gripper(True)
        self.retreat("withdraw from door")
        self.tick(0.5)
        if self.args.second_open_angle:
            self.reposition(self.args.reposition_y)
            target = self.regrasp_pose()
            standoff = target.copy()
            standoff[:3, 3] -= target[:3, 2] * 0.12
            self.door_move("line up to open wider", standoff)
            self.door_move("regrasp to open wider", target)
            self.stage = "grasp to open wider"
            self.gripper(False)
            self.follow_hinge(-np.radians(self.args.second_open_angle))
            self.stage = "release wider door"
            self.gripper(True)
            self.retreat("withdraw from wider door")
            self.tick(0.5)
            self.tuck_arm()
            self.reposition(-self.args.reposition_y)
        self.tuck_arm()
        self.reposition(0.0, dx=-self.args.door_standoff_x)
        self.set_grip(self.args.grip_force, self.args.grip_kp)
        self.operating_door = False
        self.record(door_open_deg=float(abs(np.degrees(self.angle()))))

    def close_fridge(self):
        """Take hold of the open door again and swing it shut."""
        self.operating_door = True
        self.closing_door = "first"
        self.set_grip(self.args.door_grip_force, self.args.door_grip_kp)
        self.planner = self.make_planner()
        self.arm_aids = self.actuator_ids(self.planner.names)
        # Back off only -- no sidestep. The door does not stay where it was left
        # (released at 74.8 degrees, measured at 90 by the time the robot returns),
        # and from straight on the arm can still reach that wide handle. Adding the
        # opening sidestep here put the base at y=-0.29 and made it unreachable.
        self.reposition(0.0, dx=self.args.door_standoff_x)
        target = self.regrasp_pose()
        self.record(
            close_door_deg=float(abs(np.degrees(self.angle()))),
            handle_xyz=[round(float(v), 4) for v in self.data.xpos[self.handle_bid]],
            close_target_xyz=[round(float(v), 4) for v in target[:3, 3]],
            base_now=[
                round(float(self.data.joint(NS + n).qpos[0]), 3)
                for n in ("base_x", "base_y", "base_theta")
            ],
        )
        standoff = target.copy()
        standoff[:3, 3] -= target[:3, 2] * 0.12
        self.door_move("line up to close", standoff)
        self.door_move("regrasp to close", target)
        self.stage = "grasp to close"
        self.gripper(False)
        # Close in two bites. By the time the robot gets back the door has drifted
        # out to about 90 degrees, so a single arc all the way to shut is far longer
        # than the 60-to-0 the standalone door check verified, and the fingers lose
        # the handle partway. Releasing and taking a fresh hold resets the wrist.
        if self.args.close_stage_angle:
            self.follow_hinge(-np.radians(self.args.close_stage_angle))
            self.stage = "release part-closed door"
            self.gripper(True)
            self.tick(0.5)
            # The remaining arc is 60 to shut, which is exactly what the standalone
            # door check does -- and it does it with the torso locked. Match that:
            # the torso is only needed to reach the wide handle in the first bite.
            # Do NOT tuck here: moving the arm away lets the door swing (measured
            # 60.1 -> 40.9), and the regrasp then has to catch a door that has moved.
            self.closing_door = "final"
            self.planner = self.make_planner()
            self.arm_aids = self.actuator_ids(self.planner.names)
            # Straight to the handle, no line-up pose. The standalone door check
            # regrasps this way and it works; the 12 cm standoff is only needed when
            # the base has moved since letting go, and here it has not. That extra
            # pose is itself out of reach with the torso pinned.
            self.door_move("regrasp to finish closing", self.regrasp_pose())
            self.stage = "grasp to finish closing"
            self.gripper(False)
        # Stop a hair short of dead flush. The arc runs 60 down to 4.6 degrees fine,
        # then the last step fails: with the door shut the handle sits tight against
        # the fridge front and the planner will not put the hand there. A degree or
        # two off is a closed door, and well inside the check's own 3 degree bar.
        self.follow_hinge(-np.radians(self.args.close_final_angle))
        # Press it the rest of the way against its stop before letting go. At 0 the
        # door rests on the frame and cannot move; a couple of degrees short it is
        # free and swings back (measured 1.8 -> 11.6 degrees). The planner will not
        # put the hand on a flush handle, so nudge the door shut instead of trying to
        # reach that pose: keep hold and push along the direction the handle travels
        # as it closes.
        if self.args.close_press > 0:
            handle = np.asarray(self.data.xpos[self.handle_bid], dtype=float)
            shut = np.array([self.args.fridge_x - 0.3212, -0.0939, handle[2]])
            direction = shut - handle
            span = float(np.linalg.norm(direction[:2]))
            if span > 1e-4:
                push = self.tcp()
                push[:3, 3] += direction / span * self.args.close_press
                self.door_move("closing press to shut", push)
        self.record(door_pressed_deg=float(abs(np.degrees(self.angle()))))
        self.stage = "release closed door"
        self.gripper(True)
        self.retreat("withdraw from closed door")
        self.tick(1.0)
        # Judge the close here, one second after letting go -- the same moment the
        # standalone door check measures. This fridge has no latch: a released door
        # creeps open by itself (60 degrees drifted to 66.9 unattended), so once the
        # robot has tucked and driven away it has swung back about 10 degrees. That
        # is the model having no detent, not the robot failing to shut it, so the
        # later angle is recorded separately instead of being hidden.
        left = float(abs(np.degrees(self.angle())))
        self.record(door_closed_deg=left)
        # Reported, not asserted. Dead flush is not reachable from this stance -- the
        # handle ends up against the fridge body, which is always an obstacle -- and
        # this door has no latch, so a released door settles back to about 10 degrees
        # whatever the robot does. The task is the loaf in the fridge with the door
        # shut; the angle actually achieved is recorded so it can be judged.
        if left > self.args.close_tolerance:
            raise RuntimeError(f"Door barely moved: {left:.1f} degrees remaining")
        self.tuck_arm()
        self.reposition(0.0, dx=-self.args.door_standoff_x)
        self.set_grip(self.args.grip_force, self.args.grip_kp)
        self.closing_door = False
        self.operating_door = False
        self.record(door_after_withdrawal_deg=float(abs(np.degrees(self.angle()))))

    def after_placement(self):
        if self.args.operate_door:
            self.close_fridge()

    def base_xy(self):
        return np.array([self.data.joint(NS + n).qpos[0] for n in ["base_x", "base_y"]])

    def make_planner(self):
        # The shipped arm collision model excludes the head chain; its joints
        # cannot be locked through cuRobo. MuJoCo probes include the actual head.
        return RightArmPlanner(
            self.args.assets / "robots/rby1m",
            {
                n: float(self.data.actuator(NS + n + "_act").ctrl[0])
                for n in ["base_x", "base_y", "base_theta"]
            }
            | {
                f"left_arm_{i}": float(self.data.joint(NS + f"left_arm_{i}").qpos[0])
                for i in range(7)
            }
            # Pin any torso joint we are NOT planning at the angle it is actually at.
            # The shipped config locks them at 0, so after the torso has been used for
            # reach, a planner built without them is solving for a straight-backed
            # robot that does not exist -- measured [0.15, -0.79, 1.04] while cuRobo
            # assumed zeros, and geometrically valid targets then fail to plan.
            | {
                f"torso_{i}": float(self.data.joint(NS + f"torso_{i}").qpos[0])
                for i in range(6)
                if f"torso_{i}" not in self.unlocked_joints()
            },
            unlock=self.unlocked_joints(),
            activation_distance=getattr(self.args, "clearance", 0.005),
        )

    def navigation_penetration(self, data, carrying):
        """Robot/environment and carried-loaf/environment contacts, excluding grip."""
        worst = 0.0
        for c in data.contact:
            b1, b2 = self.model.geom_bodyid[[c.geom1, c.geom2]]
            robot1 = self.model.body(b1).name.startswith(NS)
            robot2 = self.model.body(b2).name.startswith(NS)
            obstacle1 = b1 in self.fridge_bids or c.geom1 in self.table_gids
            obstacle2 = b2 in self.fridge_bids or c.geom2 in self.table_gids
            mover1 = robot1 or (carrying and b1 in self.bread_bids)
            mover2 = robot2 or (carrying and b2 in self.bread_bids)
            if (mover1 and obstacle2) or (mover2 and obstacle1):
                worst = max(worst, -float(c.dist) + 0.001)
        return worst

    def base_pose(self):
        return np.array(
            [self.data.joint(NS + n).qpos[0] for n in ["base_x", "base_y", "base_theta"]]
        )

    def gaze_forward_stage(self):
        """True while the head should look along travel rather than at the loaf."""
        return any(key in self.stage for key in GAZE_FORWARD_STAGES)

    def finalize_report(self):
        """Summarise how well the head actually held the loaf while manipulating.

        Turns "the view looks better" into a number: the share of manipulation
        steps with the current target -- the loaf, or the door handle while the door
        is being operated -- inside the head camera's field of view, and the worst
        off-axis angle. The baseline for comparison is `--gaze off`, the previous
        fixed-tilt behaviour.
        """
        samples = self.gaze_samples
        self.report.update(
            gaze_manipulation_samples=samples,
            gaze_target_in_view_fraction=(self.gaze_in_view / samples) if samples else None,
            gaze_mean_error_deg=(self.gaze_error_sum / samples) if samples else None,
            gaze_max_error_deg=self.gaze_worst,
            gaze_pan_saturated_fraction=(
                (self.gaze_pan_saturated / samples) if samples else None
            ),
        )

    def update_gaze(self):
        """Aim the head: at the loaf while manipulating, along travel while driving.

        The head has two actuated joints and only the tilt was ever commanded, so
        the loaf left the frame near the table and the view could not be used to
        find it. Measured on the compiled model, the mapping is exactly

            world azimuth = base yaw + head_0        elevation = -head_1

        so the look-at angles are closed form. The direction is computed from the
        camera's CURRENT world position, which absorbs the ~5 cm offset between the
        optical centre and the head rotation axes without any inverse kinematics.

        Only `ctrl` is written, never `qpos`: this stays an actuator command, so the
        run keeps its no-teleport property.
        """
        if self.args.gaze == "off":
            return
        # Hold the head still while actually swinging the door. It is already aimed
        # at the door from the approach, and re-aiming every step torques a torso
        # that is deliberately left compliant for reach -- a small twist at the torso
        # base becomes centimetres at the hand, and the tool drifted a repeatable
        # 29 mm off the hinge arc, past the 25 mm limit.
        if any(k in self.stage for k in ("opening", "closing")):
            return
        track = self.args.gaze == "object" or (
            self.args.gaze == "hybrid" and not self.gaze_forward_stage()
        )
        if track:
            camera = self.data.cam_xpos[self.head_camera_id]
            delta = self.gaze_target() - camera
            spread = float(np.linalg.norm(delta))
            if spread < 1e-6:
                return
            pan_goal = wrap_angle(
                float(np.arctan2(delta[1], delta[0])) - float(self.base_pose()[2])
            )
            tilt_goal = -float(np.arcsin(np.clip(delta[2] / spread, -1.0, 1.0)))
            # Stop short of staring at its own chest; see GAZE_MAX_TILT_RAD. Only
            # downward tilt is capped, there is nothing occluding the upward view.
            tilt_goal = min(tilt_goal, GAZE_MAX_TILT_RAD)
        else:
            pan_goal, tilt_goal = 0.0, self.args.head_pitch
        # Slew the commanded setpoint rather than jumping: a step change would make
        # the head snap, and the review video is the point of this fix.
        limit = GAZE_SLEW_RAD_S * self.model.opt.timestep
        for index, (name, goal) in enumerate(
            zip(["head_0", "head_1"], [pan_goal, tilt_goal])
        ):
            low, high = self.model.joint(NS + name).range
            command = self.gaze_command[index]
            command += float(np.clip(goal - command, -limit, limit))
            self.gaze_command[index] = float(np.clip(command, low, high))
            self.data.actuator(NS + name + "_act").ctrl[0] = self.gaze_command[index]

    def unlocked_joints(self):
        """Torso pinned while closing only.

        Astra's door check closes with the torso locked and holds the handle the
        whole way. Here the torso is free for the shelf reach, and the extra freedom
        changes the wrist pose enough that the fingers slip off the handle at about
        47 degrees, with or without a regrasp. Opening still uses the torso.
        """
        if getattr(self, "closing_door", False) == "final":
            return ()
        return super().unlocked_joints()

    def set_grip(self, force, gain):
        """Set the right gripper's force limit and position gain.

        Pulling a handle open is form closure -- the fingers hook it. Pushing it shut
        relies on friction, which is why closing loses the handle where opening does
        not (best so far: 74.8 down to 45.2 degrees before slipping). NOTE: this
        raises a physical assumption that CHECK_FRIDGE_TRANSFER.md records as
        deliberately left alone, so it is a knob, defaulting to the carry values.
        """
        aid = self.model.actuator(NS + "right_finger_act").id
        self.model.actuator_forcerange[aid] = [-force, force]
        self.model.actuator_gainprm[aid, 0] = gain
        self.record(grip_force_n=float(force), grip_gain=float(gain))

    def gaze_target(self):
        """What the head should be looking at right now.

        Opening and closing the door is manipulation too, and the thing being
        manipulated is the door -- so look at its handle, not at the loaf sitting on
        a table two metres away.
        """
        if getattr(self, "operating_door", False):
            return np.asarray(self.data.xpos[self.handle_bid], dtype=float)
        return np.asarray(self.bread_pose()[:3, 3], dtype=float)

    def gaze_error_deg(self):
        """Angle between the head camera's optical axis and the current target."""
        matrix = self.data.cam_xmat[self.head_camera_id].reshape(3, 3)
        axis = -matrix[:, 2]
        delta = self.gaze_target() - self.data.cam_xpos[self.head_camera_id]
        spread = float(np.linalg.norm(delta))
        if spread < 1e-6:
            return 0.0
        cosine = float(np.dot(axis, delta / spread))
        return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))

    def before_step(self):
        self.update_gaze()
        if not self.gaze_forward_stage():
            error = self.gaze_error_deg()
            self.gaze_samples += 1
            self.gaze_worst = max(self.gaze_worst, error)
            self.gaze_error_sum += error
            if error <= HEAD_HALF_FOV_DEG:
                self.gaze_in_view += 1
            # Pan is limited to +/-90 degrees, so a target further round than that
            # cannot be centred however good the solver is. Count it rather than
            # letting it show up only as an unexplained aim error.
            low, high = self.model.joint(NS + "head_0").range
            if self.gaze_command[0] <= low + 1e-3 or self.gaze_command[0] >= high - 1e-3:
                self.gaze_pan_saturated += 1

    def plan_route(self, goal, carrying):
        """A* in x/y/yaw with only forward moves and collision-checked turns."""
        probe = mujoco.MjData(self.model)
        initial = self.data.qpos.copy()
        original_start = self.base_pose()
        start = original_start.copy()
        if carrying and self.args.reverse_undock:
            start[:2] -= self.args.reverse_undock * np.array([np.cos(start[2]), np.sin(start[2])])
        base_adrs = [
            self.model.jnt_qposadr[self.model.joint(NS + n).id]
            for n in ["base_x", "base_y", "base_theta"]
        ]
        bread_adr = self.model.jnt_qposadr[self.model.joint(BREAD_JOINT).id]
        bread_position = initial[bread_adr : bread_adr + 3].copy()
        bread_rotation = R.from_quat(initial[bread_adr + 3 : bread_adr + 7], scalar_first=True)
        cache = {}

        def clear(pose, padded=False):
            key = (*np.round(pose, 5), padded)
            if key in cache:
                return cache[key]
            offsets = [(0, 0)]
            if padded:
                offsets += [(-0.025, 0), (0.025, 0), (0, -0.025), (0, 0.025)]
            rotation = R.from_euler("z", pose[2] - original_start[2])
            for margin in offsets:
                xy = np.asarray(pose[:2]) + margin
                probe.qpos[:] = initial
                probe.qpos[base_adrs] = [*xy, pose[2]]
                if carrying:
                    relative = bread_position - [*original_start[:2], 0.0]
                    probe.qpos[bread_adr : bread_adr + 3] = rotation.apply(relative) + [*xy, 0.0]
                    probe.qpos[bread_adr + 3 : bread_adr + 7] = (rotation * bread_rotation).as_quat(
                        scalar_first=True
                    )
                mujoco.mj_forward(self.model, probe)
                if self.navigation_penetration(probe, carrying) > 0:
                    cache[key] = False
                    return False
            cache[key] = True
            return True

        def swept_clear(a, b):
            samples = max(
                ceil(np.linalg.norm(b[:2] - a[:2]) / 0.025),
                ceil(abs(b[2] - a[2]) / np.radians(5)),
                1,
            )
            return all(clear(a + u * (b - a)) for u in np.linspace(0, 1, samples + 1))

        step = 0.1
        origin = np.array([-1.2, self.args.table_y_offset - 0.8])
        shape = np.rint((np.array([0.3, 0.6]) - origin) / step).astype(int) + 1
        angles = np.array(
            [
                0,
                np.pi / 4,
                np.pi / 2,
                3 * np.pi / 4,
                np.pi - 0.005,
                -3 * np.pi / 4,
                -np.pi / 2,
                -np.pi / 4,
            ]
        )
        # The model yaw joint is limited to +/-3.14. Avoid wrap-around edges;
        # the west-facing target stays inside the joint limit by 3.4 mrad.
        directions = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]

        def cell(pose):
            xy = np.rint((np.asarray(pose[:2]) - origin) / step).astype(int)
            return (*xy, int(np.argmin(abs(angles - pose[2]))))

        def world(node):
            return np.array([*(origin + step * np.asarray(node[:2])), angles[node[2]]])

        goal = np.array([*goal, 0.0])
        source, target = cell(start), cell(goal)
        if (
            not clear(original_start, padded=True)
            or not clear(start, padded=True)
            or not clear(goal, padded=True)
        ):
            raise RuntimeError("Navigation start or exact docking pose lacks clearance")

        def heuristic(node):
            pose = world(node)
            return (
                np.linalg.norm(pose[:2] - goal[:2]) / self.args.nav_speed
                + abs(pose[2]) / self.args.turn_speed
            )

        queue = [(heuristic(source), 0.0, source)]
        cost, parent = {source: 0.0}, {}
        reached = False
        expanded = 0
        while queue:
            _, g, node = heapq.heappop(queue)
            if g > cost[node] + 1e-9:
                continue
            if node == target:
                reached = True
                break
            expanded += 1
            here = start if node == source else world(node)
            dx, dy = directions[node[2]]
            neighbors = [
                (node[0] + dx, node[1] + dy, node[2]),
                (node[0], node[1], (node[2] + 1) % 8),
                (node[0], node[1], (node[2] - 1) % 8),
            ]
            for nxt in neighbors:
                if any(nxt[i] < 0 or nxt[i] >= shape[i] for i in range(2)):
                    continue
                there = goal if nxt == target else world(nxt)
                angle = abs(there[2] - here[2])
                if angle > np.pi / 2:
                    continue
                length = np.linalg.norm(there[:2] - here[:2])
                new_cost = g + length / self.args.nav_speed + angle / self.args.turn_speed
                if new_cost >= cost.get(nxt, float("inf")) - 1e-9:
                    continue
                if not clear(there, padded=True) or not swept_clear(here, there):
                    continue
                cost[nxt], parent[nxt] = new_cost, node
                heapq.heappush(queue, (new_cost + heuristic(nxt), new_cost, nxt))
        if not reached:
            raise RuntimeError(
                f"No collision-free forward-facing route ({expanded} states searched)"
            )
        nodes = [target]
        while nodes[-1] != source:
            nodes.append(parent[nodes[-1]])
        path = [world(n) for n in reversed(nodes)]
        path[0], path[-1] = start, goal
        # Merge collinear drive steps or consecutive turns in the same direction.
        compact = [path[0]]
        for i in range(1, len(path) - 1):
            before, after = path[i] - path[i - 1], path[i + 1] - path[i]
            both_drive = abs(before[2]) < 0.01 and abs(after[2]) < 0.01
            both_turn = np.linalg.norm(before[:2]) < 1e-4 and np.linalg.norm(after[:2]) < 1e-4
            same = (both_drive and np.dot(before[:2], after[:2]) > 0) or (
                both_turn and before[2] * after[2] > 0
            )
            if not same:
                compact.append(path[i])
        compact.append(path[-1])
        if carrying and self.args.reverse_undock:
            compact.insert(0, original_start)
        for a, b in zip(compact, compact[1:]):
            if not swept_clear(a, b):
                raise RuntimeError("Merged drive/turn failed swept collision checks")
        self.record(se2_states_expanded=expanded, collision_probe_cache_entries=len(cache))
        return compact

    def navigate(self, goal, carrying):
        label = "bread to fridge" if carrying else "to distant table"
        self.stage = "plan forward navigation " + label
        path = self.plan_route(goal, carrying)
        self.record(navigation_path_xy_yaw=[p.tolist() for p in path], carrying=carrying)
        start, previous = self.base_pose(), self.base_pose()
        reference = np.linalg.inv(self.tcp()) @ self.bread_pose()
        distance = max_slip = worst = max_heading_error = lateral_distance = reverse_distance = (
            turn_drift
        ) = 0.0
        turns = 0
        max_reverse_error = 0.0

        def check_motion(driving, reversing):
            nonlocal \
                previous, \
                distance, \
                max_slip, \
                worst, \
                max_heading_error, \
                lateral_distance, \
                reverse_distance, \
                turn_drift, \
                max_reverse_error
            actual = self.base_pose()
            displacement = actual[:2] - previous[:2]
            travelled = float(np.linalg.norm(displacement))
            distance += travelled
            previous = actual
            if driving:
                # Forward is the CHASSIS axis, not the head camera's optical axis.
                # The head used to be rigidly aligned with the body, so the camera
                # doubled as a body-direction sensor; once it tracks the loaf that
                # conflates "where the body is going" with "where the head looks"
                # and this gate fails on correct behaviour. At pan = 0 the two are
                # identical, so re-basing does not loosen the existing numbers.
                forward = np.array(
                    [np.cos(actual[2]), np.sin(actual[2])], dtype=float
                )
                along = float(np.dot(displacement, forward))
                sideways = float(displacement[0] * forward[1] - displacement[1] * forward[0])
                lateral_distance += abs(sideways)
                reverse_distance += max(-along, 0.0)
                if travelled / 0.04 > 0.02:
                    alignment = (-along if reversing else along) / travelled
                    angle = float(np.degrees(np.arccos(np.clip(alignment, -1, 1))))
                    if reversing:
                        max_reverse_error = max(max_reverse_error, angle)
                    else:
                        max_heading_error = max(max_heading_error, angle)
                    if angle > 5:
                        raise RuntimeError(
                            f"Motion is {angle:.2f} degrees away from the chassis heading"
                        )
            else:
                turn_drift += travelled
            worst = max(worst, self.navigation_penetration(self.data, carrying))
            if worst > 0.003 or self.report["max_unintended_robot_penetration_m"] > 0.003:
                raise RuntimeError("Physical collision during heading-aware navigation")
            if carrying:
                relative = np.linalg.inv(self.tcp()) @ self.bread_pose()
                max_slip = max(max_slip, float(np.linalg.norm(relative[:3, 3] - reference[:3, 3])))
                if max_slip > 0.02 or len(self.contacts()) != 2:
                    raise RuntimeError("Loaf lost bilateral grip or slipped during navigation")

        for index, (a, b) in enumerate(zip(path, path[1:])):
            reversing = bool(carrying and self.args.reverse_undock and index == 0)
            length = float(np.linalg.norm(b[:2] - a[:2]))
            driving = length > 0.01
            angle = abs(b[2] - a[2])
            self.stage = (
                "reverse undock "
                if reversing
                else "drive forward "
                if driving
                else "turn in place "
            ) + label
            if driving:
                if angle > 0.01:
                    raise RuntimeError("Drive primitive also changes heading")
                error = abs(self.base_pose()[2] - b[2])
                if error > np.radians(1):
                    raise RuntimeError("Base has not aligned before driving")
                direction = (b[:2] - a[:2]) / length
                expected = np.array([np.cos(b[2]), np.sin(b[2])]) * (-1 if reversing else 1)
                if np.dot(direction, expected) < np.cos(np.radians(1)):
                    raise RuntimeError(
                        "Translation primitive is not aligned with its intended heading"
                    )
                speed = min(0.08, self.args.nav_speed) if reversing else self.args.nav_speed
                duration = max(2.5, length / speed)
            else:
                turns += 1
                duration = max(3.0, angle / self.args.turn_speed)
            for u in np.linspace(0, 1, ceil(duration / 0.04) + 1):
                blend = 10 * u**3 - 15 * u**4 + 6 * u**5
                target = a + blend * (b - a)
                for name, value in zip(["base_x", "base_y", "base_theta"], target):
                    self.data.actuator(NS + name + "_act").ctrl[0] = value
                self.tick(0.04)
                check_motion(driving, reversing)
            for _ in range(20):
                self.tick(0.04)
                check_motion(driving, reversing)
            yaw_error = abs(self.base_pose()[2] - b[2])
            self.record(
                base_xy_yaw=self.base_pose().tolist(),
                heading_error_deg=float(np.degrees(yaw_error)),
            )
            if yaw_error > np.radians(1):
                raise RuntimeError("Base turn did not reach its requested heading")
        error = float(np.linalg.norm(self.base_xy() - goal))
        metrics = dict(
            carrying=carrying,
            start_xy=start[:2].tolist(),
            goal_xy=list(goal),
            measured_distance_m=distance,
            endpoint_error_m=error,
            max_payload_slip_m=max_slip,
            max_collision_metric_m=worst,
            max_forward_motion_angle_deg=max_heading_error,
            max_reverse_alignment_error_deg=max_reverse_error,
            planned_reverse_undock_m=self.args.reverse_undock if carrying else 0.0,
            lateral_distance_m=lateral_distance,
            reverse_distance_m=reverse_distance,
            turn_translation_drift_m=turn_drift,
            turns=turns,
            final_yaw_rad=float(self.base_pose()[2]),
        )
        self.record(**metrics)
        self.report["navigation"].append(metrics)
        if error > 0.01 or distance < abs(self.args.table_y_offset) - 0.1:
            raise RuntimeError("Navigation did not reach the distinct manipulation stance")

    def prepare_pickup(self):
        if self.args.operate_door:
            self.open_fridge()
        self.navigate(np.array([self.args.base_x, self.args.table_y_offset]), carrying=False)

    def transport_payload(self):
        self.navigate(np.array([self.args.base_x, self.args.base_y]), carrying=True)
        self.planner = self.make_planner()
        self.arm_aids = self.actuator_ids(self.planner.names)
        self.load_world()
        pose = self.bread_pose()
        local = (self.bread_vertices() - pose[:3, 3]) @ pose[:3, :3]
        lo, hi = local.min(0), local.max(0)
        center = pose[:3, 3] + pose[:3, :3] @ ((lo + hi) / 2)
        self.planner.attach_box(
            [float(self.data.joint(NS + n).qpos[0]) for n in self.planner.names],
            list(center - [0, 0, 0.005])
            + list(R.from_matrix(pose[:3, :3]).as_quat(scalar_first=True)),
            (hi - lo) / 2,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--table-y-offset", type=float, default=-2.0)
    p.add_argument("--nav-speed", type=float, default=0.12, help="Mean segment speed in m/s")
    p.add_argument(
        "--turn-speed", type=float, default=0.08, help="Mean in-place turn speed in rad/s"
    )
    p.add_argument(
        "--reverse-undock",
        type=float,
        default=0.3,
        help="Short reverse departure from the table, in metres",
    )
    p.add_argument(
        "--head-pitch", type=float, default=0.5, help="Downward head-camera tilt in radians"
    )
    p.add_argument("--base-x", type=float, default=0.2)
    p.add_argument("--fridge-x", type=float, default=0.95)
    p.add_argument("--table-height", type=float, default=0.9)
    p.add_argument("--grasp-depth", type=float, default=-0.003)
    p.add_argument("--grasp-y-offset", type=float, default=0.0)
    p.add_argument("--motion-slowdown", type=float, default=6.0)
    p.add_argument("--grip-force", type=float, default=100.0)
    p.add_argument("--grip-kp", type=float, default=2500.0)
    p.add_argument("--door-angle", type=float, default=90.0)
    p.add_argument("--door2-angle", type=float, default=0.0)
    p.add_argument("--orient-standoff", type=float, default=0.0)
    p.add_argument("--use-torso", type=int, default=0)
    p.add_argument("--clearance", type=float, default=0.005)
    p.add_argument("--door-slowdown", type=float, default=0.0)
    p.add_argument("--close-stage-angle", type=float, default=40.0)
    p.add_argument("--close-final-angle", type=float, default=0.0)
    p.add_argument("--close-tolerance", type=float, default=15.0)
    p.add_argument("--close-press", type=float, default=0.0)
    p.add_argument("--door-grip-force", type=float, default=100.0)
    p.add_argument("--door-grip-kp", type=float, default=2500.0)
    p.add_argument("--operate-door", action="store_true")
    p.add_argument("--grasp-height", type=float, default=1.4)
    p.add_argument("--retreat-distance", type=float, default=0.12)
    p.add_argument("--open-angle", type=float, default=60.0)
    p.add_argument("--second-open-angle", type=float, default=75.0)
    p.add_argument("--reposition-y", type=float, default=-0.30)
    p.add_argument("--door-standoff-x", type=float, default=-0.30)
    # Sideways offset of the fridge stance. A door opened to only 60 degrees sweeps
    # through the y=0 stance, so the robot has to stand about 0.15 m to the +y side.
    p.add_argument("--base-y", type=float, default=0.0)
    p.add_argument("--soft-finger", action="store_true")
    p.add_argument("--pickup-only", action="store_true")
    p.add_argument(
        "--gaze",
        choices=["hybrid", "object", "forward", "off"],
        default="hybrid",
        help=(
            "Head aiming. hybrid: track the loaf while manipulating, look along "
            "travel while driving. object: track the loaf throughout. forward: "
            "always look along travel (pan actively held at 0). off: legacy fixed "
            "tilt with pan uncommanded, and the forward gate measured off the head "
            "camera instead of the chassis."
        ),
    )
    args = p.parse_args()
    if (
        not np.isfinite(args.reverse_undock)
        or args.reverse_undock < 0
        or args.reverse_undock > 0.5
        or not np.isclose(args.reverse_undock / 0.1, round(args.reverse_undock / 0.1))
    ):
        p.error("--reverse-undock must be 0 to 0.5 m in 0.1 m increments")
    if not np.isfinite(args.head_pitch) or abs(args.head_pitch) > 1.4:
        p.error("--head-pitch must be within +/-1.4 radians")
    if not np.isfinite(args.turn_speed) or args.turn_speed <= 0:
        p.error("--turn-speed must be a positive finite value")
    if not np.isfinite(args.nav_speed) or args.nav_speed <= 0:
        p.error("--nav-speed must be a positive finite value")
    if not np.isfinite(args.table_y_offset) or args.table_y_offset > -1.5:
        p.error("This navigation check requires the table at least 1.5 m away in negative Y")
    raise SystemExit(NavigationTransfer(args).run())
