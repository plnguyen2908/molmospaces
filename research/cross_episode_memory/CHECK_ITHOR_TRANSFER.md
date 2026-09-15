> Grasp validity correction: the previous videos show fingers penetrating the loaf.
> Historical success/contact reports exempted finger–loaf contacts and do not establish
> a realistic grasp. The new controller uses gradual contact-based closure and stops
> on finger–loaf penetration above 1 mm at every physics step. This tolerance permits
> small numerical contact compliance; it is not a calibrated bread deformation model.
> These changes still require a new simulation and video review.

# RB-Y1 native-object first task in iTHOR FloorPlan3

The current `run_native_closed_fridge_task.sh` launches the **annotated native egg**
through the full sequence: initially closed fridge → approach/open → navigate to
its original counter position → annotated grasp/lift → carry to fridge → lower to
3 mm above the measured shelf surface → release/withdraw → close. The target is
checked again after door closure and after the final recording dwell.

This integration passed non-simulation dispatch/argument checks. The egg's earlier
pickup/carry test passed; the complete egg-to-fridge sequence has **not yet been
physically tested**. The user will launch it. The runner uses the existing Python
virtual environment and writes videos/reports to a new
`artifacts/native_annotated_fridge_*` directory. It never invokes the stationary
diagnostic's navigation-skipping override.


The egg retains its authored initial pose; there is no added task table or object-pose override.

## Run and review

Use the existing environment at `/nobackup2/le/molmospaces/.venv`; activation is
unnecessary because the launcher calls its Python explicitly. This checkout at
`/nobackup/le/molmospaces` has no local `.venv`. Set `MOLMOSPACES_PYTHON` to an
alternate interpreter only if needed. Verified installed versions on 2026-09-11:
Python 3.11.14, MuJoCo 3.5.0, nvidia-curobo 1.0, PyTorch 2.7.1,
NumPy 2.4.6, SciPy 1.17.1, imageio 2.37.3, imageio-ffmpeg 0.6.0,
and Pillow 11.3.0. Linux, an NVIDIA CUDA GPU, EGL, and downloaded iTHOR/robot
assets are required. Project dependencies and the `mujoco`/`curobo` extras are
specified in `pyproject.toml`; cuRobo uses the pinned AllenAI fork there.
No package installation is needed for this existing environment.

From the repository root:

```bash
export MLSPACES_ASSETS_DIR="/nobackup2/le/.cache/molmospaces/assets/L25vYmFja3VwMi9sZS9tb2xtb3NwYWNlcw"
bash research/cross_episode_memory/tools/run_native_closed_fridge_task.sh
```

The launcher defaults to GPU 2 (`CUDA_VISIBLE_DEVICES` can override it), checks
the interpreter/assets paths, and chooses a timestamped output directory under
`research/cross_episode_memory/artifacts/native_annotated_fridge_*`. An optional
first argument selects a different new output directory inside this repository;
existing output directories are rejected. All artifacts stay under molmospaces.
The task writes `fridge_transfer.mp4`, `head_camera.mp4`, `report.json`, and
`trace.json`. Review-speed video and independent audits are separate postprocessing.
The stricter grasp implementation has passed local contact tests but has not yet
been validated in a new complete task run.

The historical recording `artifacts/native_closed_fridge_task_04` contains the
invalid grasp described above and these files:

- `review_4x.mp4`: compact review at four times simulation speed.
- `fridge_transfer.mp4`: full external/detail/head-camera recording.
- `head_camera.mp4`: raw head RGB.
- `report.json`, `trace.json`: arguments, execution metrics and recorded states.
- `sequence_validation.json`, `contact_audit.json`: independent checks.
- `initial_closed.jpg`, `final_closed.jpg`: extracted actual video frames.

## Historical result under the previous contact checks

The task report and independent sequence validator both pass. Both hinges begin
at zero. The robot travels 1.400 m to the fridge, physically opens it and leaves it
at 74.231 degrees after withdrawing and tucking. It then travels 2.300 m to the
bread before grasping, lifts it 0.11872 m, and carries it 1.691 m back to the
fridge. Maximum carry slip is 0.564 mm. The loaf is released, supported and settled
inside the target shelf. After closing, hand withdrawal and final arm tucking,
the door remains at 0.00349 degrees. The independent check confirms the loaf is
still inside and supported on the target shelf after closing.

All 15,780 recorded frames were audited at 25 Hz. Worst sampled unintended
robot/scenery penetration is 1.57 mm; the live 2 ms check reports 2.55 mm. Both
are within the existing 3 mm simulation tolerance. Navigation collision metrics
are zero. Intended floor, finger/loaf and finger/handle contact are exempted;
wrist/panel contact is not. This is one continuous pass, not a success-rate study.

Re-run validation with:

```bash
MUJOCO_GL=egl PYTHONPATH=. /nobackup2/le/molmospaces/.venv/bin/python \
  research/cross_episode_memory/tools/verify_native_sequence.py \
  research/cross_episode_memory/artifacts/native_closed_fridge_task_04
MUJOCO_GL=egl PYTHONPATH=. /nobackup2/le/molmospaces/.venv/bin/python \
  research/cross_episode_memory/tools/audit_transfer_contacts.py \
  research/cross_episode_memory/artifacts/native_closed_fridge_task_04
```

## Implementation

- Measure the source collision floor (0.2222066 m here) and translate the scene
  into a floor-relative frame before inserting RB-Y1. This fixes the robot being
  inserted below the kitchen floor. Explicit task positions share that frame.
- Native mode rejects table and object-pose overrides. The initial loaf position
  is (-1.51114, 0.66222, 1.15998) m, on its original counter at about 1.0917 m.
- Explicitly initialize both hinges to zero for door-operation tasks. Transform
  handle targets from the isolated fixture into the native fridge frame.
- Use heading-aware SE(2) A* with full-scene swept-pose checks for navigation.
  Reverse undocking precedes turns at fixtures; subsequent travel faces forward.
- Use all six torso joints and nearby IK with cuRobo joint trajectories for
  native pickup and placement. Raise and rotate together to avoid the wrist limit.
- Open the gripper before handle approach. Hinge steps prefer nearby IK and use
  collision-checked alternative pose planning when needed.
- Plan arm tucking around the door. Tucking and the closing standoff activate
  collision costs at 5 cm: nominal collision spheres alone let the wrist sensor
  brush the panel. A copied base-initialization block was removed from tucking.
- After placement, reverse clear of the shelf, tuck, and approach the handle.
  Seed a short standoff from the recorded opening posture, then align to the
  current handle before gripping. Close by reversing this episode's measured
  cuRobo opening path through actuator commands, monitoring hinge tracking and
  contacts. A final 4 cm commanded press reaches the shut stop. Hand withdrawal
  retraces the original approach; the door is checked again after final tucking.

No live execution step welds or teleports the object or writes hinge qpos.

## Scope and remaining limitations

This is an oracle component sequence, not the complete reorder RunTask or a
head-camera-only learned policy. Other kitchen props remain frozen, the flat
floor uses a collision plane, and finite-pad finger friction/force assumptions
are explicit in the report. Reachability of other counters is not established.

Head aiming still needs torso compensation. The measured target-in-view fraction
in this run is 22% of manipulation samples; physical success does not make this
a usable head-camera-only VLA demonstration yet.

## Focused diagnostics and earlier results

The `check_native_door.py`, `check_native_task_tail.py`, `check_native_closing.py`
and closing-tail/replay tools restore recorded states when requested. Their
reports identify them as diagnostics, never continuous-task evidence. The local
closing check `native_closing_04` retained 0.0035 degrees after final tucking.
Failed and interrupted attempts remain in the artifacts directory.

`tools/run_native_counter_transfer.sh` reproduces the earlier open-fridge native
counter check (`native_counter_transfer_02`). `tools/run_floor_fixed_transfer.sh`
reproduces the older added-table check (`kitchen_floor_fixed_transfer_06`). Neither
is evidence of the closed-fridge sequence above. `tools/check_saved_placement.py`
is a placement-only diagnostic; `tools/inspect_loaded_docking.py` and
`tools/inspect_transfer_stances.py` probe stances without live task execution.

## Stationary grasp correction

The last feedback test still slipped: saved contacts were on the rounded crown,
about 15 mm below the top, with contact-normal vertical components 0.61–0.65.
The actual loaf mass is 0.553 kg. The next focused test uses `--grasp-depth -0.012`
(9 mm deeper than the previous command), a 30 N actuator cap, and an 8 N per-finger
feedback target. It requires at least 6 N on both fingers for 300 ms before lifting,
then first lifts vertically 25 mm and verifies counter clearance before retreating.
The 1 mm penetration stop remains active. These settings are under physical test;
local controller checks do not establish successful pickup.

`tools/check_native_grasp.py` uses the existing native pickup arguments with
`--pickup-only` and skips navigation. It writes a fixed-camera `grasp_closeup.mp4`
with finger collision gap, both contact forces, loaf rise, and penetration, plus
frame telemetry in `report.json`. The original native object pose is preserved.

## Verified annotated native-object pickup and carry

The native `Egg_3` test passed in
`artifacts/native_annotated_egg_20260911_231156`. Its original counter pose was
preserved. The bread asset has zero filtered annotations and is now rejected by
the annotated runner before simulator creation. Egg_3 has 1,000 stored transforms;
57 passed the conservative width/approach filter. Annotation 586 was selected
without hand-authored offsets: world grasp = current object pose × stored transform.
The available library is Droid, so RB-Y1 compatibility was checked separately:
68.6 mm projected object width, cuRobo pregrasp planning, nearby IK, and collision
checks of the actual open hand and arm at six descent samples.

Measured outcome: 123.99 mm object rise, 1.30030 m carry (0.30 m straight reverse
undocking plus 1 m forward travel), three-second final hold, final finger forces
2.65/2.62 N, and zero measured finger/object penetration. The controller used a
10 N actuator cap, 2.5 N per-finger target, and required at least 1 N on both fingers
for 300 ms before lifting. Force loss and the 1 mm penetration guard remained
active during lifting and navigation. No physical attachment weld was used.
This validates one annotated-object pickup/carry episode; fridge placement and
the complete reorder chain are not part of this result.

Run using the Python environment and assets setup above:

```bash
bash research/cross_episode_memory/tools/run_native_annotated_egg.sh
```

The launcher creates a fresh output directory. `grasp_closeup.mp4` and
`fridge_transfer.mp4` are the original recordings. `annotated_egg_review.mp4` is
a side-view replay of the same recorded states with original measured forces;
carrying is shown at 5× speed. It changes the camera only, not the physical run.
`annotated_grasp_validation.json` verifies the source annotation, frame transform,
native pose, measured lift/carry, three-second hold, force and collision checks.

```bash
/nobackup2/le/molmospaces/.venv/bin/python \
  research/cross_episode_memory/tools/verify_annotated_grasp.py \
  research/cross_episode_memory/artifacts/native_annotated_egg_20260911_231156
```
