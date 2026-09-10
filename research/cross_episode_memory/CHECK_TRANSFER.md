# Physical transfer component check

The verified artifact is [transfer.mp4](artifacts/transfer_verified/transfer.mp4)
(13.84 seconds, H.264, 640×480, 25 fps). The accompanying
[report](artifacts/transfer_verified/report.json) and
[trace](artifacts/transfer_verified/trace.json) contain stage results and measured
object/arm positions. This check uses the pinned RB-Y1 robot, two nearby flat
supports, and an 80 g, 5 cm block. It does not run the kitchen reorder episode.

The final run passed: bilateral finger contact, 18.8 cm lift, carry to B, release,
withdrawal, and object-to-B support contact after settling. Maximum endpoint TCP
position error was 0.74 mm. Simulation state is initialized once; subsequent
execution uses actuator controls and MuJoCo physics. There are no object welds,
teleports, or direct robot joint-position assignments during execution.

## Changes

- `curobo_current.py` converts the shipped legacy RB-Y1 YAML in memory for installed
  NVIDIA cuRobo 1.0's MotionPlanner API. That package is genuine cuRobo; the older
  description calling it unrelated was incorrect. Assets and shared dependencies
  remain unchanged.
- `molmo_spaces/robots/rby1_actuators.py` configures the asset's finger force motors
  as bounded position servos, matching the position commands that the existing
  RB-Y1 controller already produces. `RBY1.apply_control_overrides` applies this
  before optional user gain/force overrides. Both hands passed physical open,
  close, and reopen regression tests.
- `tools/check_transfer.py` runs approach, grasp, lift, carry, lower, release, and
  withdrawal with measured gates. It records failed runs too. cuRobo checks robot
  self-collisions and the supports; during carry/lowering it also checks a sphere
  approximation covering the held block. Planner attachment changes collision
  geometry only; contact between the fingers and block supplies the physical grasp.
- The installed MotionPlanner attachment accessor references a missing solver
  member. The adapter constructs its available AttachmentManager directly against
  the planner's kinematics instead of changing the installed package.
- MuJoCo's robot wrapper places the base 5 mm above the URDF origin. Targets and
  payload poses account for that offset. Trajectory playback rounds cuRobo's
  interpolation interval up to a simulation step, never speeding up the plan.

## Reproduce

Use a Python environment containing NVIDIA cuRobo 1.0, MuJoCo 3.5.0, torch,
PyYAML, imageio/ffmpeg, and Pillow. Select an available CUDA GPU.

```bash
CUDA_VISIBLE_DEVICES=2 MUJOCO_GL=egl PYTHONPATH=. \
  /nobackup2/le/molmospaces/.venv/bin/python \
  research/cross_episode_memory/tools/check_transfer.py \
  --robot-dir "$MLSPACES_ASSETS_DIR/robots/rby1m" \
  --output /tmp/rby1-transfer

PYTHONPATH=. /nobackup2/le/molmospaces/.venv/bin/python -m pytest \
  mlspaces_tests/test_rby1_gripper_servos.py -q
```

The first command exits nonzero if any gate fails. It writes `transfer.mp4`,
`report.json`, and `trace.json` in the output directory. Set
`MLSPACES_ASSETS_DIR` for the regression tests; otherwise they explicitly skip.
The recorded run's asset directory and installed versions are in its report.

## Kitchen integration still required

The existing `RBY1DemoEvalConfig` still selects the teleporting demo policy;
this new check does not silently replace its behavior. Its old success flags
must not be used as evidence of physical manipulation.

The next continuous validation must open the fridge, travel to the table,
grasp and lift the actual task object, travel carrying it to the fridge, place
and release it inside, then close the door. It must preserve the same simulation
state and stop at a failed stage. Use A* for room-scale base routes and cuRobo
for arm trajectories and constrained local approaches; the current component
check fixes the base and torso to isolate manipulation.

Before a full reorder run, integrate `RunTask` with the policy/config/schema,
replace intervention placement shortcuts with validated initial-state authoring,
and ensure episode chaining uses actual outcomes rather than assuming each work
instruction succeeded. This oracle check uses privileged poses; it is not a
learned-policy or episodic-memory benchmark result.

The subsequent [physical fridge-door check](CHECK_FRIDGE_DOOR.md) now also passes.
It validates opening, release/withdrawal, regrasp, and closure of the actual task
fridge in isolation. Actual-object shelf placement and continuous kitchen
integration remain the next checks.
