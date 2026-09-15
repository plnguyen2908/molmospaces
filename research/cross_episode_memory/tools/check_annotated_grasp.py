"""Use an existing filtered annotation on an unmodified native iTHOR object."""

import json

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from research.cross_episode_memory.tools.check_fridge_transfer import NS
from research.cross_episode_memory.tools.check_native_grasp import NativeGraspCheck
from research.cross_episode_memory.tools.check_navigation_transfer import parse_args


class AnnotatedGraspMixin:
    """Shared annotation selection and grip settings for diagnostics and full tasks."""

    def __init__(self, args):
        if not args.object_name:
            raise ValueError("Select a native object with --object-name")
        if getattr(args, "annotation_asset", None):
            self.annotation_asset = args.annotation_asset
        else:
            metadata = json.loads(
                (args.assets / "scenes/ithor/FloorPlan3_physics_metadata.json").read_text()
            )
            self.annotation_asset = metadata["objects"][args.object_name]["asset_id"]
        self.annotation_path = (
            args.assets
            / "grasps/droid"
            / self.annotation_asset
            / (self.annotation_asset + "_grasps_filtered.npz")
        )
        with np.load(self.annotation_path) as annotations:
            self.local_annotations = annotations["transforms"].copy()
        if len(self.local_annotations) == 0:
            raise ValueError(f"{self.annotation_asset} has no filtered grasp annotations")
        super().__init__(args)
        # The first annotated candidate is a 60 g egg, not the 553 g loaf.
        self.grasp_force_limit_n = 10.0
        self.grasp_target_force_n = 2.5
        self.grasp_stable_force_n = 1.0
        self.model.actuator_forcerange[self.model.actuator(NS + "right_finger_act").id] = [-10, 10]

    def validate_grasp_probe(self, probe):
        self.validate_grasp_contacts(probe)
        if getattr(self.args, 'gaze', 'off') != 'off' and hasattr(self, 'gaze_angles'):
            # cuRobo omits the head. Check the gaze posture used during grasping,
            # not only the earlier head orientation inherited from navigation.
            head = [self.model.jnt_qposadr[self.model.joint(NS+n).id]
                    for n in ('head_0', 'head_1')]
            initial = probe.qpos[head].copy()
            for _ in range(4):
                angles = self.gaze_angles(probe, probe.body(self.object_name).xpos)
                for name, angle in zip(('head_0', 'head_1'), angles):
                    probe.joint(NS+name).qpos[0] = np.clip(angle, *self.model.joint(NS+name).range)
                mujoco.mj_forward(self.model, probe)
            self.validate_grasp_contacts(probe)
            settled = probe.qpos[head].copy()
            probe.qpos[head] = .5*(initial+settled)
            mujoco.mj_forward(self.model, probe)
            self.validate_grasp_contacts(probe)
            probe.qpos[head] = settled
            mujoco.mj_forward(self.model, probe)

    def validate_grasp_contacts(self, probe):
        for contact in probe.contact:
            names = [self.model.body(self.model.geom_bodyid[g]).name
                     for g in (contact.geom1, contact.geom2)]
            robot = [n.startswith(NS) for n in names]
            if not any(robot):
                continue
            if all(robot):
                # Jaw-to-jaw contact is the normal empty-gripper closed stop.
                if all('ee_finger_' in n for n in names):
                    continue
            else:
                other_geom = contact.geom2 if robot[0] else contact.geom1
                if self.model.geom_type[other_geom] == mujoco.mjtGeom.mjGEOM_PLANE:
                    continue
            if contact.dist < -.0005:
                kind = 'Robot self collision' if all(robot) else 'Open hand/arm collision'
                raise RuntimeError(f'{kind}: {names}, depth={-contact.dist:.4f}, contact_xyz={contact.pos.tolist()}')

    def select_annotated_grasp(self):
        asset = self.annotation_asset
        path = self.annotation_path
        local = self.local_annotations
        annotation_count = len(local)
        if getattr(self, 'allow_grasp_symmetry', False):
            flip = np.diag([-1., -1., 1., 1.])
            local = np.concatenate((local, local @ flip))
        orientation_count = len(local)
        world = self.bread_pose() @ local
        offsets = getattr(self, 'annotation_vertical_offsets', (0.,))
        variants = []
        for offset in offsets:
            shifted = world.copy()
            shifted[:, 2, 3] += offset
            variants.append(shifted)
        world = np.concatenate(variants)
        local = np.linalg.inv(self.bread_pose()) @ world
        vertices = self.bread_vertices()
        candidates = []
        for index, pose in enumerate(world):
            width = float(np.ptp(vertices @ pose[:3, 1]))
            approach_ok = (self.annotation_approach_allowed(pose)
                           if hasattr(self, "annotation_approach_allowed") else pose[2, 2] <= -.9)
            if not approach_ok or width > .095:
                continue
            # RB-Y1 closes along tool Y, matching the library convention.
            # Filter conservatively with the full object projection, then check
            # the actual RB-Y1 hand/arm against the scene at all approach samples.
            score = width + 0.2 * (pose[2, 2] + 1) + .0002 * (index // orientation_count)
            score += getattr(self, "annotation_center_weight", 0.) * float(
                np.linalg.norm(pose[:2, 3] - self.bread_pose()[:2, 3]))
            candidates.append((score, index, width, pose))
        candidates.sort(key=lambda item: item[0])
        self.report["annotation_selection"] = {
            "asset_id": asset,
            "file": str(path),
            "library": "droid",
            "total_annotations": annotation_count,
            "tested_orientation_variants": len(local),
            "width_and_approach_candidates": len(candidates),
            "robot_compatibility": "RB-Y1 geometry and IK checked independently",
            "object_pose": self.bread_pose().tolist(),
            "rejected": [],
        }
        self.record(annotation_candidates=len(candidates), annotation_file=str(path))
        positions = [float(self.data.joint(NS + n).qpos[0]) for n in self.planner.names]
        joint_ids = [self.model.joint(NS + n).id for n in self.planner.names]
        addresses = self.model.jnt_qposadr[joint_ids]
        probe = mujoco.MjData(self.model)
        rejected_physical = getattr(self, "physically_rejected_annotation_variants", set())
        candidates = [item for item in candidates if item[1] not in rejected_physical]
        for _, index, width, pose in candidates[:getattr(self, "annotation_candidate_budget", 16)]:
            aperture = (min(.05, (width + .006) / 2)
                        if getattr(self, 'annotation_adaptive_aperture', False) else self.args.grip_open)
            pre = pose.copy()
            pre[:3, 3] -= getattr(self, "annotation_standoff", .10) * pose[:3, 2]
            goal = list(pre[:3, 3] - [0, 0, 0.005]) + list(
                Rotation.from_matrix(pre[:3, :3]).as_quat(scalar_first=True)
            )
            try:
                trajectory = self.planner.plan(positions, goal)
                q = np.asarray(trajectory[-1])
                contact_path = []
                mesh_contact = getattr(self, 'annotation_mesh_contact_approach', False)
                samples = 19 if mesh_contact else 6
                for fraction in np.linspace(0, 1, samples):
                    target = pre.copy()
                    target[:3, 3] = pre[:3, 3] + fraction * (pose[:3, 3] - pre[:3, 3])
                    kwargs = ({'preserve_self_clearance': True}
                              if getattr(self, 'preserve_planner_self_clearance', False) else {})
                    if getattr(self, 'annotation_joint_margin', 0.) > .02:
                        kwargs['joint_margin'] = self.annotation_joint_margin + .002
                    q = self.nearby_ik(target, q, **kwargs)
                    limits = self.model.jnt_range[joint_ids]
                    joint_margins = np.minimum(np.asarray(q)-limits[:,0], limits[:,1]-np.asarray(q))
                    margin = float(np.min(joint_margins))
                    if margin < getattr(self, 'annotation_joint_margin', 0.):
                        raise RuntimeError(f'Grasp approach reaches a joint limit: {self.planner.names[int(np.argmin(joint_margins))]}, remaining margin {margin:.4f} rad')
                    contact_path.append(np.asarray(q).copy())
                    probe.qpos[:] = self.data.qpos
                    probe.qpos[addresses] = q
                    probe.joint(NS + "gripper_finger_r1").qpos[0] = -aperture
                    probe.joint(NS + "gripper_finger_r2").qpos[0] = aperture
                    mujoco.mj_forward(self.model, probe)
                    self.validate_grasp_probe(probe)
                # cuRobo excludes some self pairs and the head chain. Check its
                # whole free-space approach with the actual MuJoCo geometry too.
                for waypoint_joints in trajectory:
                    probe.qpos[:] = self.data.qpos
                    probe.qpos[addresses] = waypoint_joints
                    probe.joint(NS + 'gripper_finger_r1').qpos[0] = -aperture
                    probe.joint(NS + 'gripper_finger_r2').qpos[0] = aperture
                    mujoco.mj_forward(self.model, probe)
                    self.validate_grasp_probe(probe)
                self.args.grip_open = aperture
                self.preplanned_moves = {"pregrasp": (pre.copy(), trajectory)}
                if mesh_contact:
                    for step in range(1, 4):
                        waypoint = pre.copy()
                        waypoint[:3, 3] += (step / 3) * (pose[:3, 3] - pre[:3, 3])
                        segment = contact_path[(step-1)*6+1:step*6+1]
                        self.preplanned_moves[f'grasp approach {step}/3'] = (waypoint, segment)
                    self.report['contact_approach_method'] = 'cuRobo outside approach; actual-mesh-checked IK contact approach'
                self.report["annotation_selection"].update(
                    selected_index=index % annotation_count,
                    parallel_gripper_roll_180=(index % orientation_count) >= annotation_count,
                    robot_adaptation_vertical_m=offsets[index // orientation_count],
                    projected_object_width_m=width,
                    open_command_m=aperture,
                    local_transform=local[index].tolist(),
                    world_transform=pose.tolist(),
                    checked_approach_samples=samples,
                )
                self.selected_annotation_variant = index
                self.record(selected_annotation=index % annotation_count, orientation_variant=index, projected_width_m=width)
                return pose.copy(), pre
            except RuntimeError as exc:
                self.report["annotation_selection"]["rejected"].append(
                    {"index": index, "reason": str(exc)}
                )
                self.record(rejected_annotation=index, reason=str(exc))
        raise RuntimeError("No annotated grasp passed RB-Y1 reachability and collision checks")


class AnnotatedGraspCheck(AnnotatedGraspMixin, NativeGraspCheck):
    def tick(self, seconds):
        if getattr(self, "holding_loaf", False):
            self.detail_target = self.bread_pose()[:3, 3].copy() + [0, 0, 0.06]
        super().tick(seconds)


if __name__ == "__main__":
    raise SystemExit(AnnotatedGraspCheck(parse_args()).run())
