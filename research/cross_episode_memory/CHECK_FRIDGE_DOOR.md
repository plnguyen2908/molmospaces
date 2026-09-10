# Physical fridge-door check

[Watch the verified video](artifacts/fridge_door_verified/fridge_door.mp4)
(34.2 seconds, two views, 1280×480, 25 fps).
[Measured report](artifacts/fridge_door_verified/report.json) ·
[Recorded state/contact trace](artifacts/fridge_door_verified/trace.json).

The check passed: approach the real handle, grasp, open, release, withdraw,
regrasp, close, release, and withdraw again. The open door stayed at 59.46°
after withdrawal and a one-second settling interval. It finished 0.15° from
closed. Maximum endpoint TCP position error was 3.71 mm. Unintended
robot–fridge penetration was monitored every 2 ms; the recorded maximum was 0 m.

This uses the exact `Fridge_3` subtree from the reorder task's FloorPlan3 scene,
including its meshes, hinge limits, masses, friction, contact defaults, and scene
solver options. The fridge is translated to an isolated test position; the rest
of the kitchen is omitted. The RB-Y1 base and torso remain fixed in planning.
Only arm/gripper actuator controls execute the task. No door actuator, weld,
object teleport, or simulation joint-position overwrite drives the door.

Both fingers contacted the handle at every articulation waypoint. This does
not mean uninterrupted bilateral contact: at 25 Hz, opening recorded 227 samples
with two fingers, 35 with one, and 1 with none; closing recorded 223 with two,
37 with one, and 3 with none. These brief contact gaps remain visible in the trace.
The measured hinge followed each commanded arc step and both fingers reacquired
contact at each waypoint. The result establishes the physical open/close component,
not grasp robustness under arbitrary disturbances or starting positions.

## Reproduce

Use the same NVIDIA cuRobo 1.0 / MuJoCo 3.5.0 environment as the transfer check,
and select an available GPU:

```bash
CUDA_VISIBLE_DEVICES=2 MUJOCO_GL=egl PYTHONPATH=. \
  /nobackup2/le/molmospaces/.venv/bin/python \
  research/cross_episode_memory/tools/check_fridge_door.py \
  --assets "$MLSPACES_ASSETS_DIR" \
  --output /tmp/rby1-fridge-door
```

The verified defaults are `--fridge-x 1.05 --grasp-height 1.4 --open-angle 60
--retreat-distance 0.12`. The script exits nonzero on failure and saves video,
report, and trace for failed runs too. Ruff 0.15.0 lint and formatting checks pass.

## Implementation findings

- A pose near the handle can be reachable while the approach or withdrawal is
  not. Test the full articulation arc and release/withdrawal poses when choosing
  a stance. Straight withdrawal at the open-door wrist orientation failed here;
  rotating the released wrist halfway back toward its closed-door orientation
  while withdrawing 12 cm gives a reachable motion.
- Hinge targets are constructed from the measured hinge anchor/axis and the
  grasped TCP pose. The simulation's measured joint angle must follow each target;
  a failed plan or tracking/contact gate stops the check.
- cuRobo uses conservative oriented bounds per rigid fridge part. During hinge
  tracking, the active panel is omitted from the **static planner obstacles**
  because it moves with the hand. Its physical collision geometry stays enabled,
  and unintended robot/panel penetration is checked every simulation step.
  The panel is restored to planner obstacles after articulation. The grasped
  handle is an intentional contact target. Cabinet and other door obstacles remain.
- Those coarse exterior bounds are unsuitable for planning inside the fridge.
  Shelf placement will need finer interior collision geometry.

Next is the actual task-object transfer from a table into the already-open fridge.
The full continuous navigation/pick/place/close sequence and long reorder run
remain unverified. This is a privileged-state oracle component check, not an
assessment of the learned policy or episodic-memory conditions.
