# Local work and recordings

Store project code, documentation, diagnostic scripts, run logs, reports, traces,
and videos under `/nobackup/le/molmospaces`. Use project-local output and scratch
directories for future work so requested deliverables are retained here.

Recordings and traces are **not** committed: the artifacts directory is around
400 MB of video and per-step state, which does not belong in git. The code and the
documents that describe the runs are committed; the recordings stay on this machine.

## Verified component checks

- Task specification and findings: [SPEC.md](SPEC.md)
- Component implementation: [tools](tools/)
- Fridge door, open and close: [CHECK_FRIDGE_DOOR.md](CHECK_FRIDGE_DOOR.md)
- Table-to-shelf transfer: [CHECK_FRIDGE_TRANSFER.md](CHECK_FRIDGE_TRANSFER.md)
- Navigation with the loaf: [CHECK_NAVIGATION_TRANSFER.md](CHECK_NAVIGATION_TRANSFER.md)

## Recordings (local only)

| Run | What it shows |
|---|---|
| `artifacts/gaze_navigation_verified/` | Navigation transfer with active head aiming. Passing. |
| `artifacts/full_sequence/` | Whole task **passing**: open the closed fridge to 74.8 deg, drive to the table, pick the loaf, drive back, place it on the shelf, close the door to 2.4 deg. Predates the door-aware gaze, so the head watches the loaf during the door work. |
| `artifacts/run_183237/` | **The whole task passing with the corrected gaze**: open to 74.8, drive, pick, drive back, place, close to 5.0 degrees. |
| `artifacts/repeat_*/` | Repeats of that run, to see whether it is reliable or was lucky. |
| `artifacts/heading_navigation_*`, `artifacts/bread_transfer_*` | Earlier development runs, kept for the failures they record. |

## Where the door task stands

The whole task runs: the robot opens the closed fridge to 74.8 degrees, drives 2 m
to the table, picks the loaf, drives back, places it on an interior shelf, and
closes the door to 5.0 degrees, where it stays. No unintended penetration. The head
is aimed at whatever is being worked on -- the door while opening and closing, the
loaf while picking and placing.

Passing run: `artifacts/run_183237/`.

What made the close work was a bug fix, not tuning. Any planner built while the
torso was bent but not being planned was locking those joints at the shipped
config's zeros, so cuRobo was solving for a straight-backed robot that did not
exist. Locking them at their measured angles took the close from stalling at 47
degrees to reaching 5 in one go.

Two honest limits:

- The door stops at 5 degrees, not flush. Dead flush is not reachable from this
  stance: at 0 the handle sits against the fridge body, which is always an obstacle
  to the planner. The standalone door check reaches 0.15 degrees because it works
  from a different stance with a straight torso and no loaf in play. The
  door-closed figure is therefore **reported rather than asserted**, with the check
  only catching a door that barely moved. That is a looser criterion than before.
- This fridge has no latch, so a released door drifts open on its own (60 degrees
  crept to 66.9 unattended). At 5 degrees it happened to stay put; that is not
  guaranteed, and `door_after_withdrawal_deg` records where it actually ended up.

Repeated 4 times (`artifacts/repeat_1..4`): all passed, every one closing to
exactly 4.98 degrees with 0.000 mm penetration.

Read that carefully. The numbers being *identical* means the pipeline is
deterministic, so this rules out flakiness -- earlier fixes had cleared their limits
by margin, and that is no longer a worry. But it is one result reproduced four
times, not four independent samples: it says nothing about tolerance to a different
loaf position, start pose, or fridge placement. Robustness is still untested.
