"""Complete ProcTHOR scene with RB-Y1, retaining native object poses and geometry."""
from pathlib import Path
import mujoco
import numpy as np
from molmo_spaces.configs.robot_configs import MobileFrankaRobotConfig, RBY1MConfig


def _tidybot_franka():
    from research.cross_episode_memory.tidybot_franka import TidyBotFrankaConfig
    return TidyBotFrankaConfig


ROBOT_CONFIGS = {'rby1m': RBY1MConfig, 'franka_droid': MobileFrankaRobotConfig,
                 'franka_tidybot': _tidybot_franka}


def make_house(scene_xml, dynamic_objects, robot_xy, robot_yaw=0., robot='rby1m'):
    spec = mujoco.MjSpec.from_file(str(scene_xml))
    original = spec.compile()
    original_data = mujoco.MjData(original)
    mujoco.mj_forward(original, original_data)
    native_poses = {name: original_data.body(name).xpos.copy() for name in dynamic_objects}
    # Preserve all rooms and collision geometry. Freeze only non-task degrees of
    # freedom; do not substitute a plane for unverified whole-house flooring.
    keep = set()
    for name in dynamic_objects:
        bid = original.body(name).id
        keep.add(name)
        descendants = {bid}
        for child in range(bid + 1, original.nbody):
            if original.body_parentid[child] in descendants:
                descendants.add(child)
                keep.add(original.body(child).name)
    frozen = 0
    for body in spec.bodies:
        if body.name in keep:
            continue
        for joint in list(body.joints):
            # Freezing articulated joints at their authored reference preserves
            # their initial physical transform, including native doorway doors.
            spec.delete(joint)
            frozen += 1
    # Which embodiment stands in the house is a caller's choice, not a constant.
    # Both share the holonomic base joints the navigation layer drives, so the
    # scene builder needs nothing robot-specific beyond the config itself.
    if robot not in ROBOT_CONFIGS:
        raise KeyError(f'unknown robot {robot!r}; have {sorted(ROBOT_CONFIGS)}')
    factory = ROBOT_CONFIGS[robot]
    cfg = (factory() if robot != 'franka_tidybot' else factory()())
    cfg.robot_cls.add_robot_to_scene(cfg, spec, 'robot_0/', [0., 0.], [1., 0., 0., 0.])
    cfg.robot_cls.apply_control_overrides(spec, cfg)
    model = spec.compile()
    data = mujoco.MjData(model)
    for name, value in zip(('base_x', 'base_y', 'base_theta'), (*robot_xy, robot_yaw)):
        data.joint('robot_0/' + name).qpos[0] = value
        data.actuator('robot_0/' + name + '_act').ctrl[0] = value
    # Straighten whatever arms this robot has, by profile rather than by RB-Y1's
    # naming: a one-armed Franka has no left_arm_* to write to.
    from research.cross_episode_memory.robot_profile import profile_for
    arms = list(profile_for(robot).arm_joints)
    if robot == 'rby1m':
        arms += [name.replace('right_', 'left_') for name in arms]
    for name in arms:
        joint = data.joint('robot_0/' + name)
        joint.qpos[0] = -.02 if name.endswith('_3') else 0.
    for aid in range(model.nu):
        if model.actuator_trntype[aid] == mujoco.mjtTrn.mjTRN_JOINT:
            data.ctrl[aid] = data.qpos[model.jnt_qposadr[model.actuator_trnid[aid, 0]]]
    mujoco.mj_forward(model, data)
    for name, pos in native_poses.items():
        if not np.allclose(data.body(name).xpos, pos, atol=1e-8):
            raise RuntimeError(f'Loading changed native position: {name}')
    return model, data, {'scene_xml': str(Path(scene_xml).resolve()), 'whole_house': True,
                         'original_bodies': original.nbody, 'original_geoms': original.ngeom,
                         'bodies_with_robot': model.nbody, 'geoms_with_robot': model.ngeom,
                         'frozen_joints': frozen, 'native_poses_preserved': True,
                         'floor_geometry': 'native, unchanged',
                         'dynamic_objects': list(dynamic_objects)}
