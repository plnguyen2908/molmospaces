"""Physical regression: position commands must actually open/close each finger."""

import os
from pathlib import Path

import mujoco
import numpy as np
import pytest

from molmo_spaces.robots.rby1_actuators import configure_rby1_gripper_servos


@pytest.mark.parametrize("side", ["right", "left"])
def test_finger_tracks_position_targets(side):
    assets = os.environ.get("MLSPACES_ASSETS_DIR")
    if not assets:
        pytest.skip("Set MLSPACES_ASSETS_DIR to the pinned robot assets")
    path = Path(assets) / "robots/rby1m/rby1_v1.2_site_control.xml"
    spec = mujoco.MjSpec.from_file(str(path))
    configure_rby1_gripper_servos(spec)
    model = spec.compile()
    model.body_gravcomp[:] = 1
    data = mujoco.MjData(model)
    # Lift both arms clear of the base so finger motion has no obstruction.
    for arm in ("left", "right"):
        for i, q in enumerate([0.5, 0, 0, -2.3, 0, -0.5, 0]):
            data.joint(f"robot_0/{arm}_arm_{i}").qpos[0] = q
    for a in range(model.nu):
        if model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT:
            data.ctrl[a] = data.qpos[model.jnt_qposadr[model.actuator_trnid[a, 0]]]
    mujoco.mj_forward(model, data)
    finger = data.joint(f"robot_0/gripper_finger_{side[0]}1")
    actuator = data.actuator(f"robot_0/{side}_finger_act")
    for target in (-0.045, -0.005, -0.045):
        actuator.ctrl[0] = target
        mujoco.mj_step(model, data, nstep=round(3 / model.opt.timestep))
        assert abs(finger.qpos[0] - target) < 0.002, (side, target, finger.qpos.copy())
        assert np.isfinite(data.qpos).all()
