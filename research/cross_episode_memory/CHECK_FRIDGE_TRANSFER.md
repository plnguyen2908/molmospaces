# Actual bread: table to open fridge

`tools/check_fridge_transfer.py` is an oracle manipulation component check using
FloorPlan3's actual `Bread_3` and `Fridge_3` assets. It uses a small test table,
not the complete kitchen. The base starts at a fixed stance; the right door
starts open at 90 degrees. Navigation, door operation, RunTask integration, and
the continuous reorder instruction are outside this check.

Run from the repository root with the environment containing MuJoCo 3.5 and
NVIDIA cuRobo 1.0:

```bash
CUDA_VISIBLE_DEVICES=2 MUJOCO_GL=egl PYTHONPATH=. \
  /nobackup2/le/molmospaces/.venv/bin/python \
  research/cross_episode_memory/tools/check_fridge_transfer.py \
  --assets "$MLSPACES_ASSETS_DIR" \
  --soft-finger --grip-force 100 --grip-kp 2500 \
  --output research/cross_episode_memory/artifacts/bread_transfer_verified
```

The script writes `fridge_transfer.mp4`, `report.json`, and `trace.json`, including
on failure. `--pickup-only` isolates grasp/lift. A run succeeds only after the
loaf lifts clear of the table, remains grasped during carry, and is released,
supported, inside the selected shelf bounds, and settled after withdrawal.

Execution writes actuator controls only after initialization. cuRobo payload
attachment adds collision spheres; it does not weld or teleport the loaf in
MuJoCo. Bread dimensions, mass (about 0.553 kg), and asset parameters are preserved.
The optional finite-pad model changes the effective finger contact law, as
described below; it is not a calibrated hardware model.
The exact refrigerator collision triangles are used instead of an exterior box,
which would incorrectly block its interior shelves.

## Findings during development

- The loaf's roughly 12.5 cm width exceeds the gripper's 10 cm opening. A shallow
  top grasp contacts the sloped upper crust. Finger contact alone does not prove
  pickup: the initial attempts closed on the loaf and then slipped off.
- A bounded force diagnostic obtained a physical lift at an 80 N actuator limit
  with a position gain of 2000 N/m and slowed arm motion. These are explicit
  settings of this isolated check; the shared gripper helper's defaults remain
  20 N and 1000 N/m. The 80 N setting still slipped during rotation; the later
  diagnostic uses an explicit 100 N cap and 2500 N/m gain. Hardware suitability
  of the stronger setting is unverified.
- The shipped point-contact model has no contact moment resistance. The explicit
  `--soft-finger` option sets right-finger collision geoms to `condim=6`, priority
  1, and friction `[1.0, 0.005, 0.002]` (sliding coefficient, torsional length,
  rolling length). Sliding friction stays at the original effective value of 1.
  The rotational terms model a finite pad patch; their values are assumptions,
  not measured calibration. This option is confined to this check, defaults off,
  and is recorded in both the report and video. Point-contact failures are kept
  as evidence, rather than described as successful transfers.
- A long endpoint-only carry plan deviated from the intended rotation axis and
  dropped the loaf. Increasing force alone did not fix that path. The check now
  uses an upright clearance move and small rotation waypoints.
- The fixed stance must be held through the base's site-transmission position
  actuator. Initializing joint qpos alone leaves its target at zero and drives
  the robot away from the planning stance.
- Initial robot/table/fridge overlap is rejected. The test table is positioned
  clear of the open door and cabinet, and the MuJoCo arena is sized to avoid
  losing constraints in diagnostic initial states.

## Validation status

The isolated transfer passed on 2026-09-09 with the explicit 100 N, 2500 N/m,
finite-pad configuration above. This is one passing simulation run, not a
robustness or hardware claim.

- [Video, 49.2 seconds](artifacts/bread_transfer_verified/fridge_transfer.mp4)
- [Measured report](artifacts/bread_transfer_verified/report.json)
- [State/contact trace](artifacts/bread_transfer_verified/trace.json)

The loaf lifted about 15 cm, rotated 90 degrees, entered the actual middle-right
shelf, was released, and remained supported after withdrawal and settling.
All 675 sampled carry frames had both fingers in contact. Maximum measured
payload translation slip was 0.86 mm and maximum endpoint TCP error was 4.46 mm.
Unintended robot/table/fridge/bread penetration was zero at the 2 ms physics
checks. Final bread speed was 0.00000245 m/s, with three upward shelf contacts,
no gripper contact, and its measured geometry inside the selected shelf bounds.

Failed-run videos and reports remain under `artifacts/bread_pick_*` and
`artifacts/bread_transfer_*`. In particular, the original point-contact model
has not passed this transfer. Calibration of the contact law and grip force,
navigation while carrying, and the continuous open/pick/place/close task remain
outstanding. The earlier isolated door test opened to 60 degrees; this placement
check starts at 90 degrees, so that opening difference also needs integration.


Subsequent navigation checkpoint: [CHECK_NAVIGATION_TRANSFER.md](CHECK_NAVIGATION_TRANSFER.md)
records a passing test with the table shifted 2 m away and about 3 m of physical
base travel on each leg. The continuous door-operation sequence remains pending.
