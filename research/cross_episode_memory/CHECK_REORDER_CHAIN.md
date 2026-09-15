# Historical two-receptacle reorder run

## Current scope: right fridge door only

As requested on 2026-09-13, the task uses the existing native cuRobo handle pull
and physical recorded-path closure. The experimental panel-push controller has
been removed from the task's inheritance and door-motion path. The left door
stays closed; placement, retrieval, inspection, and dynamic placements use only
the right compartment. A full right compartment rejects the proposed transfer
before pickup; it never triggers left-door access. A tracked object on the left
cannot be counted as observed through the closed door.

The two task receptacles are the sink plus adjoining kitchen countertop, and the
right fridge compartment. The report lists the physical support bodies explicitly:
`receptacle_support_bodies` maps the counter to the sink and surrounding countertop.
The native objects remain at their authored initial poses. The default pool is
**egg, potato, and salt shaker**, all with filtered grasp annotations. `--objects`
can select a smaller pool or add pepper; four objects need `--change-cycles 3`.
Objects qualify for dynamic changes only after successful physical work in that
same run. An annotation alone does not qualify them. Finger contacts and forces
provide grasp/carry, with no held-object weld or teleport.

## Sequence

Start with all doors/drawers closed. There is no initial or pre-change inspection.
The default three-object/two-change sequence is:

1. Open; physically transfer egg counter→fridge and potato counter→fridge; close.
2. Change potato back to its demonstrated counter pickup pose. Revisit both
   receptacles and save the first post-change target: egg in fridge, the others
   on the counter.
3. Open; physically transfer egg fridge→counter and the **new salt shaker**
   counter→fridge; close.
4. Exchange salt shaker fridge→counter and potato counter→fridge, using each
   object's own demonstrated poses. Revisit both receptacles.
5. Restore the first post-change target: open; physically return potato to the
   counter, then egg to the fridge; close.

There are six physical transfers, two changes/revisits, and five right-door cycles.
Each work pair shares its open door; revisits have their own open/inspect/close
cycle. Restoration empties occupied fridge slots before inserting objects. Never
inspect or apply a change inside a work batch. With `--objects egg potato`, the
previous five-transfer/two-object schedule remains available. The scheduler
introduces unused objects in later pairs while retaining transfers in both directions.

**No feasibility rehearsal:** changes do not execute an extra robot rollout.
After each fully verified physical transfer, record its actual source and released
destination poses as witnesses. Dynamic changes require those witnesses, check
all proposed poses together for current collisions in a scratch state, and then
apply the explicit harness change and verify settled support. This does not prove
that every future trajectory will succeed; normal planning/contact/collision
checks still apply during live work. Failed work never qualifies an object.

This is a simulator-state oracle execution test, not a camera-only VLA result.
Right-only revisits do not claim inspection of the disabled left compartment.

## Run and outputs

From the repository root:

```bash
export MLSPACES_ASSETS_DIR=/nobackup2/le/.cache/molmospaces/assets/L25vYmFja3VwMi9sZS9tb2xtb3NwYWNlcw
bash research/cross_episode_memory/tools/run_reorder_chain.sh
```

The launcher uses `/nobackup2/le/molmospaces/.venv/bin/python` by default;
`MOLMOSPACES_PYTHON` can override it. Outputs stay under MolmoSpaces in
`research/cross_episode_memory/artifacts`. Existing output directories are rejected.

Each fresh run writes `run_manifest.json` before simulation, with its source
fingerprint, arguments, package versions and interpreter, and copies the loaded
repository sources into `source/`. The fingerprint also appears in the log and
final report so two invocations can be compared directly.

`report.json` records controller mode, enabled compartment, snapshots, transfer
and change events, contact/collision metrics, `successful_transfer_witnesses`, and
`intervention_eligibility` (including which witness and pose each change reused).
`door_cycle_policy: per_transfer_group`, `transfer_groups`, and each transfer's
`door_actions` record which transfers share access; standalone transfers still
open and close independently.
Terminal records distinguish `task_object` (egg/potato/salt shaker, with the full scene ID in
`task_object_id`) from `interaction_target` (the right fridge door handle during
opening/closing). Video captions also show target and payload separately. The
controller's payload binding does not change during door operation: arm targets
and gaze use the handle while `operating_door` is active. Reports retain the
legacy `object` payload field, stable internal `stage` IDs for saved-path recovery,
and `stage_label` for display; legacy `bread_*`/`loaf_*` fields refer to the payload.
Already-running processes retain the labels loaded when they started.
`trace.json` holds the continuous physics states. **Both successful and failed
full runs render after physics ends**, at 25 FPS and 5x playback. Failure videos
show execution up to the recorded endpoint and finish with a failure caption;
the report still records failure. No video renders while that run is executing.
`physics_result.json` reports the outcome before rendering and then updates video
status when encoding finishes. Empty traces skip video. Saved traces can also be
rendered without rerunning physics using `tools/render_reorder_trace.py RUN_DIR`.
Diagnostic component scripts remain report/trace-only unless explicitly replayed.

The current speed settings are 0.25 m/s mean forward travel, 0.25 rad/s mean
turning, and 4× arm trajectory time scaling. They can be overridden with
`--nav-speed`, `--turn-speed`, and `--motion-slowdown`. Reverse undocking remains
capped at 0.08 m/s. All contact, slip, heading, and collision checks retain their
original frequency and thresholds; physics still uses 2 ms steps. Static body
category caching and vectorized contact selection were equivalent on 90 recorded
native states and reduced the measured contact-check/physics loop time by 34%.

## Validation status

The expanded object pool and demonstrated-placement policy are being validated.
The previous full shared-door attempt completed all four ordinary transfers, then
failed in the now-removed final feasibility rehearsal at shelf alignment. That
failure is not a successful full task, and the videos below remain historical.
The new default must pass a fresh full shell-script run before it is declared
ready. Failed full runs now render their recorded execution after stopping, as
requested; older failed traces can be replayed with `render_reorder_trace.py`.

The current component regression `salt_native_pickup_v5` passed physical navigation,
annotated pickup, 119.8 mm lift and force hold of the untouched native salt shaker.
Annotation 70 held with 1.578/1.599 N final finger forces, no contact loss, zero
robot/environment penetration, and at most 0.540 mm finger/object penetration.
Its native mass is 21.9 g; the shaker uses a 1.5 N target/preload with the existing
1 N stability and 1 mm penetration gates. Egg/potato retain their previous forces.
The diagonal aisle stance avoids the cabinet; shelf inset is bounded by measured
object dimensions. cuRobo's cuboid cache now grows to fit the exact selected scene
geometry (1,537 boxes at the shaker), without discarding obstacles. All 76 focused
tests pass. This is a pickup component, not proof of complete transfer/history;
a fresh default shell-script run is required next.

### Earlier validation history


The next full attempt addresses `reorder_chain_20260913_140233`, which completed
both initial transfers and the first change/revisit, then rejected the loaded
egg's fridge departure. The scratch collision probe found the right forearm
crossing the open right door during reverse undocking. Retraction selection now
checks the arm path and resulting loaded navigation route together, before
moving; it also considers small lateral arm offsets. A 5 cm lateral offset
passed the saved-state route check. All existing collision margins are retained.
There are 70 passing focused tests, including rejection before arm motion and
propagation of physical carry failures. A fresh default shell-script validation
is in progress; saved-state checks are not full-history proof.

**Current shared-door change:** 68 focused tests pass. The later user run
`reorder_chain_20260913_132548` completed egg placement and carried the potato to
the fridge, then failed at the first shelf-alignment plan. A saved-state probe
isolated cuRobo's padded self-collision between `link_torso_4` and the folded left
arm. Actual MuJoCo meshes were clear; removing the payload proxy still failed.
The locally generated contact/lift IK had not enforced cuRobo's self padding.
Contact descent and lift IK now preserve that same padding, without reducing any
collision buffer or physical contact threshold. Free-space moves still use
cuRobo. `shared_door_clearance_pick_v1` passed the corrected physical potato
pickup/lift/hold. `shared_door_clearance_transfer_v1` also passed a continuous
potato pickup → carry → placement → release → close from the user's saved second
pickup boundary, with the egg already inside and the right door left open. It
selected annotation 33, used no planner fallback, maintained a minimum loaded
finger force of 1.172 N, and recorded zero robot/environment and finger/object
penetration. The sole door event was closure to 0.00358°. This is a completed
transfer regression, not a fresh full history; no video was rendered.
`shared_door_clearance_regression_v1` separately checks the original failed
annotation-334 carry posture: padded self clearance improves from −3.289 mm to
+2.000 mm at the same tool pose, then cuRobo plans the rejected alignment move.
The scratch recovery/alignment path has zero actual mesh penetration. This
regression is a planning check, not a physical replay of the complete transfer.
Consolidated evidence and tested source are in `artifacts/shared_door_clearance_validation`.

The user's
`reorder_chain_20260913_122737` run used the same arguments as v6 except its output
path, with identical MuJoCo/cuRobo versions and the same first opening and egg
annotation. It completed egg sink→fridge and failed at potato `grasp approach 2/3`
planning. Counter selection had checked an IK descent but cached only pregrasp;
execution then asked cuRobo to solve the descent again from the new arrival history.

Both counter and fridge pickups now execute all three cached, actual-mesh-checked
contact segments, with a 3 mm tracking cap. Missing segments abort before motion.
Both lift-off stages also use the existing physical contact servo: cuRobo's coarse
collision model could reject the grasped state as a planning start. Free-space
cuRobo planning, actual geometry checks, force limits and collision thresholds remain.

`shared_door_egg_grasp_v2` and `shared_door_potato_grasp_v3` passed saved-counter
pickup, lift and three-second force hold. Both lifted 119.7–119.8 mm, with zero
measured robot/environment, self, or finger/object penetration and no hold-contact
loss. Final finger forces were 2.735/2.706 N for egg and 2.766/2.914 N for potato.
The potato test selected annotation 334, matching the failed run. These component
checks render no video and do not establish full-history success. The complete
updated sequence remains unvalidated; videos below retain the previous seven-cycle
result. Comparison and physical evidence are in `artifacts/shared_door_grasp_comparison`.

**Previous fresh full run passed:** `artifacts/reorder_right_bidirectional_v6` (seed 0).
All 18 history-audit checks pass: five physical transfers including fridge→sink,
two validated dynamic changes, two post-change revisits, seven right-door cycles,
and a nontrivial restoration of the first saved arrangement. Under that previous policy, every transfer and
revisit finished closed; the left door remained at 0°. The final right-door angle
was 0.00359°. Egg finished in the fridge and potato on the sink counter.

Both complete videos rendered only after successful physics: [full three-view
video](artifacts/reorder_right_bidirectional_v6/fridge_transfer.mp4) and
[raw head camera](artifacts/reorder_right_bidirectional_v6/head_camera.mp4).
Each is 7:45 long, 25 FPS, at 5× playback. Change 1/revisit begins at 2:37/2:39,
change 2/revisit at 5:51/5:53, and historical restoration at 6:38. No failed run
or diagnostic component produced a video.

The run measured maximum finger/object penetration of 0.939 mm (1 mm limit),
robot/environment penetration of 0.346 mm, self penetration of 0.0994 mm,
and loaded-navigation slip of 17.43 mm (20 mm limit). Minimum loaded finger
force was 0.373 N with no recorded hold-contact loss. These are simulation
contact tolerances, not hardware-calibrated contact guarantees. Gaze target
coverage was 61.6% of manipulation samples; continuous visual observability
is not yet validated. This single successful seed validates the scoped oracle
sequence, not reliability across seeds or a camera-only VLA policy.

The launcher above now uses shared door access and defaults to planning seed 0
(`--seed` overrides it). The previous successful run's
source snapshot, report, trace, and `history_audit.json` are stored alongside
the videos. Earlier attempts and component findings follow for diagnosis.

- The first remaining-history diagnostic rejected change 1 because retrieval
  approached a head/left-door-handle contact. Gaze angles now use the actual torso
  frame, and annotated-grasp prechecks include the expected head posture. In the
  saved failure pose, the old base-frame formula had a 26.15° camera error;
  the corrected geometric target is aligned to numerical precision.
- `retrieval_gaze_frame_v1` physically grasped the egg, lifted it 0.120 m with
  about 2.6 N on each finger, and compacted it with no measured robot penetration.
  It then rejected the fixed 0.30 m reverse departure. A 0.20 m departure passed
  the complete route preflight, retaining all clearance margins.
- `retrieval_carry_departure_v1` physically carried that held egg 3.80 m,
  placed it on its native counter slot, and closed the right door to 0.0036°.
  Maximum carry slip was 16.34 mm (20 mm limit), with zero robot/environment
  or self penetration. These are components, not fresh full-history success.
- Loaded fridge departure now tries shorter validated reverse distances only
  when planning failed before any motion. Sub-millimetre final waypoint errors
  no longer produce unnecessary turn/drive/turn corrections.
- `reorder_right_bidirectional_v4` completed both initial transfers and their door
  cycles. Its isolated next-work check grasped and lifted the egg, then rejected
  a carry posture that would put the forearm through the open right panel.
  The live scene was not changed.
- Fridge retrieval now requires the upright wrist roll, clearance from joint
  limits, and a validated compact posture/departure. `retrieval_upright_roll_v1`
  passed pickup, lift, 3.80 m carry, counter placement, and right-door closure
  from the saved v4 pregrasp boundary, with no robot/environment or self penetration.
- Contact-approach feedback now corrects toward the validated joint target instead
  of repeatedly adding the existing actuator bias. A regression verifies recovery
  from stale gravity compensation without overshooting the target.
- The reference-carry-posture experiment failed planning and is excluded from the
  active controller; its source is preserved under `artifacts/carry_reference_experiment`.
- `reorder_right_bidirectional_v5` stopped on an actual self-collision during
  empty-arm folding after potato placement. Every fold now prechecks the whole
  path against actual meshes, including omitted self pairs and the expected head
  gaze; a rejected path can use the measured placement retreat before any motion.
  `placement_fold_mesh_check_v1` passed departure and closure from the preceding
  released boundary, with zero robot/environment and self penetration.

- `reorder_right_bidirectional_v3` closed after egg placement, then the planner
  failed to fold the empty arm after potato placement. A planning-only failure
  now falls back to the measured placement approach in reverse, preflighted
  against actual scene meshes, before retrying the fold. Physical execution
  failures still abort immediately. `right_placement_retreat_forced_v1` passed
  this fallback with zero recorded robot/environment and robot/self penetration;
  `right_placement_retreat_v1` separately completed physical door closure from
  the same released state. These checks rendered no video.

- `reorder_right_bidirectional_v2` completed egg placement at the faster settings,
  then its fixed 4 cm stop press exceeded the 25 mm TCP tracking limit after the
  door had seated. The stop press now uses bounded, measured increments.
  `right_close_tail_v1` passed at these settings: one push, 2.6 mm TCP error,
  bilateral handle contact, and a 0.0036° closed angle after release/withdrawal.
  Neither failed run nor component rendered video.

- `reorder_right_bidirectional_v1` completed the egg transfer and potato placement,
  then stopped because passive door motion put the fixed closing pregrasp outside
  the arm workspace. Closing now tries shorter live-handle standoffs and validates
  the full cuRobo approach against actual collision meshes.
  `right_close_recovery_v1` physically closed that saved failure state to 0.0035°
  with bilateral handle contact and unchanged object support. No video was rendered.

- Rollback checks: **57 focused tests pass**. The real iTHOR scene initializes
  with both doors closed and selects separate right-compartment slots for egg
  and potato. `artifacts/right_only_rollback_smoke/configuration_check.json`
  records this initialization/selection check; no manipulation or video was run.
- The scene check exposed a capacity-query crash for bodies without active mesh
  geometry. Empty mesh selections now return empty arrays and are skipped.
- The earlier right-door `reorder_history_doors_v4` run completed both initial
  physical transfers with door cycling, then failed in the feasibility rollout
  before dynamic change 1. That attempt did not validate the complete history;
  the later fresh v6 run above did.
- `reorder_close_boundary_v1` passed physical departure and right-door closure from
  a saved placement state: closed angle 0.0035 degrees, egg still supported in the
  fridge. This was a component, not a full-sequence success.
- Egg retrieval/lift and its later carry/counter placement passed in separate
  `reorder_return_egg_mesh_v3` and `reorder_return_egg_carry_v2` components.
- `reorder_chain_complete_v10` passed an older four-transfer, single-change task
  assembled from saved checkpoints. It is not proof of the current fresh run.
- The unfinished two-door/panel-push source snapshot is preserved under
  `artifacts/door_push_experiment_20260913`; it is not the active task controller.

Right-only regressions cover inspection, closing, rejection of left-side access,
full-right capacity without left fallback, current-cycle closing references,
physical clone isolation, and rejection before live mutation. Protocol tests retain
the exact transfer/change/revisit/restoration ordering and success-only rendering.
Additional checks enforce bidirectional ordinary work, a door cycle for each
work batch, no revisit during locomanip, and the correct next-work feasibility target.
