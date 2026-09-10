"""Small adapter for NVIDIA cuRobo 1.0 (the MotionPlanner API).

Oracle validation only; this is not the study's learned policy. The shipped
RB-Y1 YAML uses the older MotionGen format. Convert it in memory, leaving the
pinned assets and shared Python environment untouched.
"""

from dataclasses import fields
from itertools import product
from math import sqrt
from pathlib import Path

import torch
import yaml
from curobo._src.collision.attachment_manager import AttachmentManager
from curobo._src.robot.loader.kinematics_loader_cfg import KinematicsLoaderCfg
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import GoalToolPose, JointState, Pose


class RightArmPlanner:
    def __init__(
        self,
        robot_dir: Path,
        locked_positions: dict[str, float],
        collision_cache=None,
        unlock: tuple[str, ...] = (),
        activation_distance: float = 0.005,
    ):
        config_dir = robot_dir / "curobo_config"
        raw = yaml.safe_load((config_dir / "rby1m_right_arm_holobase.yml").read_text())
        kin = raw["robot_cfg"]["kinematics"]
        kin["tool_frames"] = [kin.pop("ee_link")]
        allowed = {f.name for f in fields(KinematicsLoaderCfg)}
        # Old USD-only metadata is not part of the new URDF loader.
        kin = {k: v for k, v in kin.items() if k in allowed}
        kin["urdf_path"] = str(config_dir / "urdf/model_holobase.urdf")
        kin["asset_root_path"] = str(config_dir / "urdf")
        kin["collision_spheres"] = yaml.safe_load(
            (config_dir / "rby1m_holobase_spheres.yml").read_text()
        )["collision_spheres"]
        kin["lock_joints"].update(locked_positions)
        cspace = kin["cspace"]
        keep = [i for i, name in enumerate(cspace["joint_names"]) if name not in locked_positions]
        for key in ("joint_names", "retract_config", "null_space_weight", "cspace_distance_weight"):
            cspace[key] = [cspace[key][i] for i in keep]
        # The shipped config pins all six torso joints and leaves them out of the
        # planned space, so the arm reaches from a rigid trunk. They ARE actuated in
        # the MuJoCo model (as link1_act..link6_act), and collision spheres exist for
        # link_torso_1..5, so they can be planned. Limits come from the URDF.
        for name in unlock:
            if name in kin["lock_joints"]:
                del kin["lock_joints"][name]
            if name not in cspace["joint_names"]:
                cspace["joint_names"].append(name)
                cspace["retract_config"].append(0.0)
                cspace["null_space_weight"].append(1.0)
                cspace["cspace_distance_weight"].append(1.0)
        cspace["default_joint_position"] = cspace.pop("retract_config")
        self.planner = MotionPlanner(
            MotionPlannerCfg.create(
                robot={"kinematics": kin},
                collision_cache=collision_cache or {"cuboid": 16, "mesh": 2},
                num_ik_seeds=32,
                num_trajopt_seeds=4,
                # How far the optimiser keeps trajectories off obstacles. The table
            # is deliberately modelled 5 mm low (so the loaf resting on it is not
            # a collision), so at the default 5 mm a plan can sit flush with the
            # real surface and the gripper grazes it.
            optimizer_collision_activation_distance=activation_distance,
            )
        )
        self.planner.clear_scene_cache()
        # cuRobo 1.0 exposes an accessor whose TrajOptSolver member is missing.
        # The implementation itself is available and updates shared robot tensors.
        self.attachments = AttachmentManager(self.planner.kinematics)
        self.names = self.planner.joint_names
        self.dt = float(self.planner.trajopt_solver.config.interpolation_dt)

    def plan(self, positions, goal):
        state = JointState.from_position(
            torch.tensor([positions], device="cuda", dtype=torch.float32), joint_names=self.names
        )
        frame = self.planner.tool_frames[0]
        pose = Pose(
            position=torch.tensor([goal[:3]], device="cuda", dtype=torch.float32),
            quaternion=torch.tensor([goal[3:]], device="cuda", dtype=torch.float32),
            name=frame,
        )
        target = GoalToolPose.from_poses({frame: pose}, ordered_tool_frames=[frame], num_goalset=1)
        result = self.planner.plan_pose(target, state, max_attempts=5)
        if result is None or not result.success.any():
            raise RuntimeError(f"cuRobo failed to plan to {goal}; result={result}")
        b, s = result.success.nonzero(as_tuple=True)
        end = int(result.interpolated_last_tstep[b[0], s[0]].item())
        trajectory = result.interpolated_trajectory
        indices = [trajectory.joint_names.index(name) for name in self.names]
        return trajectory.position[b[0], s[0], :end, indices].cpu().numpy()

    def attach_block(self, positions, world_pose, half_size):
        """Collision geometry only: physical attachment remains gripper contact."""
        state = JointState.from_position(
            torch.tensor([positions], device="cuda", dtype=torch.float32), joint_names=self.names
        )
        # Eight enclosing spheres cover the block's eight octants.
        radius = sqrt(3) * half_size / 2
        spheres = [
            [x, y, z, radius] for x, y, z in product((-half_size / 2, half_size / 2), repeat=3)
        ]
        pose = Pose(
            position=torch.tensor([world_pose[:3]], device="cuda", dtype=torch.float32),
            quaternion=torch.tensor([world_pose[3:]], device="cuda", dtype=torch.float32),
        )
        self.attachments.update(
            torch.tensor(spheres, device="cuda", dtype=torch.float32),
            state,
            "attached_object_right",
            world_objects_pose_offset=pose,
        )

    def detach_block(self):
        self.attachments.detach("attached_object_right")

    def attach_box(self, positions, world_pose, half_sizes):
        """Conservatively cover a measured payload box with at most 40 spheres."""
        import numpy as np

        half_sizes = np.asarray(half_sizes, dtype=float)
        counts = np.full(3, 2, dtype=int)
        counts[np.argmax(half_sizes)] = 6
        cell_half = half_sizes / counts
        radius = float(np.linalg.norm(cell_half))
        axes = [np.linspace(-h + c, h - c, n) for h, c, n in zip(half_sizes, cell_half, counts)]
        spheres = [[x, y, z, radius] for x, y, z in product(*axes)]
        state = JointState.from_position(
            torch.tensor([positions], device="cuda", dtype=torch.float32), joint_names=self.names
        )
        pose = Pose(
            position=torch.tensor([world_pose[:3]], device="cuda", dtype=torch.float32),
            quaternion=torch.tensor([world_pose[3:]], device="cuda", dtype=torch.float32),
        )
        self.attachments.update(
            torch.tensor(spheres, device="cuda", dtype=torch.float32),
            state,
            "attached_object_right",
            world_objects_pose_offset=pose,
        )
