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
| `artifacts/run_135113/` | Same task with the corrected gaze -- head on the door while opening, on the loaf while picking and placing. Completes open, pick and place; stops at the close. |
| `artifacts/heading_navigation_*`, `artifacts/bread_transfer_*` | Earlier development runs, kept for the failures they record. |

## Where the door task stands

Working and repeated: the robot opens the closed fridge, drives 2 m to the table,
picks the loaf, drives back, and places it on an interior shelf, with no unintended
penetration and the head aimed at whatever it is working on.

Closing is not solved. The arm holds the handle and swings the door from 74.8 down
to about 47 degrees, then the fingers lose it. Two states, both blocked:

| torso during the close | outcome |
|---|---|
| free (needed for the shelf reach) | reaches the handle, slips at ~47 deg |
| pinned (as in the standalone door check) | cannot reach the handle at all |

Raising the door grip to 250 N changed nothing -- the slip angle was identical --
so it was reverted rather than left as an unexplained tweak. Adding a regrasp at
60 degrees gets further (74.8 -> 60 -> 51) but still does not finish.

One end-to-end pass exists (`artifacts/full_sequence/`), but it cleared the same
limits by margin rather than by design, so it should not be read as a repeatable
result.

The next idea worth trying is pushing the door shut against its outer face instead
of pulling the handle: pushing does not depend on a friction grip, which is the
thing that fails partway through the arc.
