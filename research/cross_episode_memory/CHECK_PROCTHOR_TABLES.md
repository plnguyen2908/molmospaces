# Full-house ProcTHOR two-table reorder

## Current scene population

The current scene is the complete `train_8` house, retaining native architecture
and fixed furniture/receptacles. Original loose props are removed and replaced
with unscaled native assets with nonempty filtered grasp annotations:
`Salt_Shaker_2`, `Potato_11`, `Mug_2`, and `Egg_3`.

Build this initial scene with:

```bash
bash research/cross_episode_memory/tools/run_procthor_population.sh
```

The builder attempts five objects per table and one per other receptacle site. Each table receives a Book_6 with up to 4 cm intentional edge overhang.
Placements prefer edges with a clear outward approach, retain the entire object
footprint on the receptacle, and leave at least 15 cm between object origins on
the same level. This heuristic does not establish robot reachability.
It checks placement collisions and two seconds of physical settling. Every
fixed receptacle furniture object must receive an object for overall population
success; unavailable individual sites (including closed drawer interiors) are
reported separately. `all_sites_populated` records that stricter condition.
The JSON report lists object provenance and annotation paths. No robot grasp
is implied by population success. This constructs initial conditions, not a
physical manipulation trajectory. All generated scenes and reports stay here.

Two selected tables will define the task pool; objects elsewhere remain context.
The reorder launcher uses the validated populated-scene manifest. At startup it randomly selects four distinct annotated asset types, balanced as
two objects per table. Those four objects supply the four primary locomanip
transfers across the two requested change cycles; the other table objects remain
physical context. Population and startup checks
do not establish successful physical manipulation.

Latest population validation: `artifacts/procthor_population_books_v1/population_report.json`
reports 84 spawned annotated objects across all 44 native receptacle furniture
objects. All 84 retain physical support after settling. Each dining table has five objects, including one overhanging book. 56 of 85 individual sites are filled; 29 unavailable sites are
listed explicitly. The scene preview is `population_preview.mp4` in that folder.
Robot reachability, physical grasping and the reorder chain are not validated
for this new population yet.

## Reorder protocol and prior component validation

The sequence is two transfers, dynamic change, revisit/save, two transfers,
another change, revisit, and restoration of the first saved post-change state.
Transfers include both directions. Only objects already successfully manipulated
can enter dynamic changes, and those changes reuse their demonstrated poses.
Revisit is separate from manipulation and also handles an empty table.

Run using the existing MolmoSpaces Python environment containing MuJoCo, PyTorch,
cuRobo, NumPy/SciPy and the repository dependencies:

```bash
export MLSPACES_ASSETS_DIR=/nobackup2/le/.cache/molmospaces/assets/L25vYmFja3VwMi9sZS9tb2xtb3NwYWNlcw
bash research/cross_episode_memory/tools/run_procthor_reorder.sh
```

The launcher defaults to `/nobackup2/le/molmospaces/.venv/bin/python`; override it
with `MOLMOSPACES_PYTHON`. All output stays under this repository. The previous
`run_reorder_chain.sh` remains the iTHOR fridge runner.

Physics traces are recorded in memory and saved when the run ends. Video replay
and encoding start only after termination, on both success and failure. The
physics result is published before rendering. A failed run remains a failure.

## Validation status

- Complete house loading and native initial object positions checked.
- `train_31`: 80 and 123 sampled collision-free empty-robot stances at the two tables.
  These checks do not establish loaded navigation or manipulation reachability.
- Shared protocol and existing annotation/output tests: 79 passed; three table-adapter checks also passed.
- Generic full-house physical pickup validated on `Remote_1` in `train_31`:
  119.8 mm lift, no contact-loss interval, maximum finger penetration 0.341 mm,
  no unintended robot/environment penetration. This is a component result in a
  component test, not proof that the complete sequence works.
- The selected objects and complete `train_31` chain are **not yet validated**.

The alternative `train_55` pair was rejected: chairs and a wall leave its pepper
shaker 1.55 m from the nearest clear sampled stance. No furniture was removed.

The phone controller uses a mass-scaled squeeze target (0.35 N per finger for
this 14 g asset) and 5 ms, critically damped finger-contact response (`solref`).
This is an explicit contact-model assumption recorded in the report. The hard
1 mm penetration limit remains unchanged. Other objects retain their native
contact response. The phone configuration still requires physical validation.

Destination docking is checked with the actual held-object pose, rather than
assuming an empty-arm stance works while carrying. Accepted navigation routes
are reused for execution. Placement also reserves other objects' demonstrated
destinations so later dynamic changes cannot silently reuse an occupied slot.
