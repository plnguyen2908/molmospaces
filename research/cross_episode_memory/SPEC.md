# Cross-Episode Episodic Memory for Loco-Manipulation — Build Spec

## Goal
An outer-loop episodic memory over a fixed base policy. New tasks are explored (try
decompositions until one succeeds); successful subtask executions are written to a
persistent store; later tasks retrieve them to skip exploration.

Claim: performance improves over repeated executions in the same environment.
No existing benchmark measures repeated-execution improvement: LA-EQA and
GOAT-Bench are multi-episode but evaluate each query once, and EchoVLA/MoMani —
the closest prior work, with genuine episodic memory — evaluates independent
trials with randomised starts rather than accumulation. **Verify EchoVLA's
protocol before building further** (appendix).

## Current plan (consolidated 2026-09-08)

*The appendix below is an append-only research log. This section is the current
state of decisions; where they conflict, this section wins.*

### Stack

| | Embodiment | Base | Arms | Navigation | Manipulation | Planner |
|---|---|---|---|---|---|---|
| **B1** | RB-Y1 | mobile | 2 | *(inside VLA)* | MolmoBot-SPOC-RBY1Rigid | none |
| **B2** | RB-Y1 | mobile | 2 | RING | MolmoBot-SPOC-RBY1Rigid | LLM |
| **B3** | Franka FR3 on TidyBot++ base | mobile | 1 | RING | pi0.5-DROID (`pi05_droid`) | LLM |
| **B4** | XLeRobot | mobile | 2 (independent) | RING | 2x G0.5 `g05-so101` | LLM |

B1 is the floor and costs nothing to run. B1->B2 isolates architecture with
embodiment fixed. B2->B3 and B3->B4 vary embodiment and policy, showing the
memory loop is not tied to one VLA.

**Ablations** (variants of our system, separate table from baselines): A1 no
memory, A2 within-episode only, A3 commonsense prior only, A4 naive concatenated
retrieval. A3 is the one most likely to kill a task -- design against it first.

### Non-negotiables
- **No privileged state in the execution loop.** Embodiment-mounted sensing is
  legitimate; anything only the simulator could produce is not. Bans A* (privileged
  occupancy map), cuRobo, grasp libraries, and the sim-only sensors listed in the
  audit. The LLM planner is agent-side and bound by this too.
- **Memory acts on the planner only.** No released VLA accepts memory as
  conditioning. Hold the base policy fixed and released so improvement is
  attributable to the memory loop rather than to retraining -- this is also what
  EchoVLA's design cannot show.
- **Planner action space:** `{navigate_to(X), pick(X, arm), place(X, Y, arm)}`.
  Bounded by the VLAs' training prompts; sample phrasing from their templates.
- **Cold start**, interleaved restoration queries, attempts-to-first-success
  plotted against episode index, 20-attempt cap with censored runs counted as
  failures.

### Tasks
Task 1 (relocation search) is viable **because RING supplies `navigate_to(X)`** --
MolmoBot's own vocabulary has no location slot, so a monolithic VLA would hide
search where memory cannot reach it. Task 2 (order restoration) is the strongest
task and the likely headline. Task 3 (setup reuse) is unchanged. **Task 4 moves to
Phase 2**: the only articulated checkpoint overwrites its language goal with a
privileged point, and MolmoSpaces articulation is a bare joint-range check.

### Current population policy (2026-09-13, user revision)

Keep the native house architecture and fixed receptacles. Remove the originally
spawned loose objects, then populate every native fixed receptacle with
objects having nonempty filtered grasp annotations. This supersedes the earlier
requirement to manipulate only originally spawned objects. Preserve asset size
and shape; do not manufacture easier geometry. Record each object's asset,
annotation file, initial support and placement. Validate collision-free placement
and settled physical support; annotation availability alone is not proof of a
successful robot grasp. Audit individual sites separately: closed drawer interiors
and obstructed shelf sites can remain unavailable, but report them explicitly;
a populated furniture object does not imply every one of its sites is filled.

The reorder sequence still operates between exactly two selected receptacles.
For the two-cycle run, randomly select four distinct annotated object instances,
balanced two per receptacle, so each of the four primary locomanip transfers
introduces a different object. Populate their task
objects near usable edges, with the footprint supported and space between
objects for gripper access. Books may intentionally overhang a table edge by up
to 4 cm so the gripper can approach the exposed edge; physical settling must
still establish stable support. Prefer outward approaches clear of nearby obstacles;
validate actual robot docking and reach separately. Other populated receptacles
remain scene context.
Dynamic changes still use only objects physically manipulated in that run.

### Current execution scene: full ProcTHOR house, two tables (2026-09-13)

The user has superseded the fridge scene for the next engineering test. Load a
complete native ProcTHOR house, preserving all rooms, furniture and obstacles.
Choose exactly two existing tables with naturally spawned objects. The eligible
pool is drawn exclusively from objects initially on those two tables; do not
import objects from other supports or insert synthetic props. Select suitable
annotated objects flexibly, with no two-object cap. Annotation availability is
screening only; retain physical contact, lift, carry, release and support checks.

Keep the two-transfer → dynamic-change → revisit history protocol and restoration
of an earlier post-change snapshot. Include transfers in both directions. Dynamic
changes may involve only objects already physically manipulated in that run and
reuse demonstrated placements. Render the execution after termination, including
failed runs through their failure endpoint. The older fridge requirements below
remain a record of the previous scenario, not constraints on the two-table scene.

### Two-receptacle execution test requirements (2026-09-13)

This bounded iTHOR/RB-Y1 engineering test uses an explicitly labelled simulator-state
oracle with cuRobo and annotated grasps. It validates physical execution; it does
not count as a no-privileged-state VLA baseline result above.

- Exactly two task receptacles: sink plus adjoining countertop, and fridge.
  Record their physical support-body mapping explicitly. Left/right fridge compartments
  are parts of the same receptacle, not a third task receptacle.
- Start with all doors closed. Execute **two locomanip transfers → dynamic change →
  revisit/save → two locomanip transfers → dynamic change → revisit → restore the
  first post-change snapshot**. No initial or pre-change inspection tour.
- Ordinary work must include both counter→fridge and fridge→counter physical
  transfers, introducing new native annotated objects in later batches. Default
  pool: egg, potato, salt shaker. First transfer egg/potato to fridge, reset potato
  to its demonstrated counter pose, revisit/save. Then retrieve egg and transfer
  salt shaker to fridge; exchange shaker back to counter and potato to fridge;
  revisit. Restore by retrieving potato and returning egg to fridge. This is six
  physical transfers; a fourth object requires at least three change cycles.
- Each consecutive pair of locomanip transfers shares one right-door access
  cycle: open before the first, keep it open between transfers, and close after
  the second. Both directions follow this rule. Each fridge revisit still opens
  and closes independently; the final restoration batch closes when finished.
  The two-change run uses five door cycles. Never revisit/inspect or apply a
  dynamic change inside a locomanip batch.
- **Current user-approved scope: right fridge door and right compartment only.**
  The left door stays closed. The sink counter and the accessible part of the
  fridge still count as exactly two task receptacles.
- Restore the existing cuRobo handle pull and physical recorded-path closure.
  The experimental partial-pull/panel-push controller is excluded from this task.
- Placement and dynamic-change feasibility use reachable, supported, collision-free
  right-compartment slots. If the right has no valid space, reject the proposed
  work/change before pickup or live mutation. Do not fall back to the left side.
- Revisit the accessible right compartment with its door open, then close it.
  Reject tracked objects on the disabled left side rather than claiming to observe
  or manipulate them through its closed door. Complete two-door inspection and
  left-side overflow are deferred until their physical controller is validated.
- **No duplicate feasibility rollout.** Only objects successfully picked,
  carried and released in this live run may be changed. Reuse that object's
  demonstrated pickup/placement poses on the requested support. Check current
  occupancy of the simultaneous proposed changes once, then verify settled
  support after the explicit harness relocation. A grasp annotation by itself
  does not qualify an object. Preserve the requested historical target. These
  past successes do not guarantee a future motion plan; retain live collision,
  force and support checks. Never silently count failed work as eligibility.
- Physical grasp/carry uses finger contacts and forces, with no held-object weld or
  pose teleport. Harness relocations occur only as explicit logged dynamic changes.
- Render only after the run ends. **Successful and failed full runs both produce
  videos**; failed runs show the recorded execution through the failure endpoint
  and remain labelled failures. Never render while that run is executing. Save
  the physics outcome before rendering. This supersedes the earlier no-failure-video rule.

The fresh `reorder_right_bidirectional_v6` run passed the previous policy
(close/reopen between transfers) with seed 0: five physical transfers, two changes/revisits, seven right-door
cycles, and restoration of the first post-change snapshot. Both full videos
rendered after successful physics. See [CHECK_REORDER_CHAIN.md](CHECK_REORDER_CHAIN.md)
for evidence, reproduction, contact metrics, and the remaining camera-coverage
limitation. The shared-door policy and counter-grasp fix have 66 passing focused checks
and physical pickup/lift/hold checks for both objects. The user's shared-door run
exposed a counter-approach planning failure after the egg transfer; the complete
updated sequence remains unvalidated. The previous result is one successful oracle run, not
a VLA or cross-seed reliability result.

### Ranked risks
1. **EchoVLA may already publish a repeated-execution curve.** Highest value
   check available; read its evaluation section before building further.
2. **RING's AI2-THOR -> MuJoCo gap is unmeasured.** Treat its success rate here as
   an open number.
3. **Base success rates must land in 20-60%** for every configuration, measured
   per embodiment, never inherited.
4. Integration work: RB-Y1 benchmark JSONs (B1/B2), TidyBot robot class (B3),
   XLeRobot robot class + dual client (B4).

### Build order
1. Verify EchoVLA's protocol. Re-scope if it pre-empts the claim.
2. Run B1 as shipped -- this is simultaneously the floor and the step-2 base-rate
   measurement. Zero implementation.
3. RING adapter (discrete actions -> `holo_joint_rel_planar_position`), then
   measure RING in MuJoCo.
4. Episode store + retrieval; schema records subtask, objects, spatial relations,
   order, parameters, **arm**, outcome. Retrieval key: subtask language + scene
   embedding + base pose.
5. Run A1-A4 against B1/B2 on tasks 1-3.

### Notes carried forward
- Base pose is a **retrieval key**, not a planner output -- the VLA or RING owns
  the base.
- Mount the exterior camera on the base for B3/B4, or the policy sees a valid
  image of the wrong place after every navigation leg.
- Log base-collision events separately; crude footprints cause navigation
  failures unrelated to policy quality.
- Pin all asset versions for the study. LinearBot, XLeRobot, Ridgeback meshes and
  TidyBot are **not** in the pinned set and will not fetch on a fresh machine.

## Episode structure (the core design)
A **run** is a sequence of episodes in one kitchen. Each episode is a real task
attempt (put things away, get ingredients out, clean up). Objects move as a
*byproduct of the work*, not by arbitrary teleport — this is what makes the state
changes defensible.

- Episode N: robot performs a task. Arrangement changes as a result.
- Between episodes: optional scripted changes (config-driven, logged).
- **Restoration queries are interleaved, not only at the end** — after episodes
  3, 6, 10, ask "restore to how it was at <earlier episode / time>". One query per
  run gives one data point and no curve.
- Query a *different* past state each time, and vary arrangements across episodes,
  so retrieving the wrong episode gives a wrong answer. Otherwise temporal
  indexing is untested.

**Observability rule:** the robot can only answer for states it was present for.
Each episode's task must route the robot past the objects that later queries will
reference, or the query is unanswerable for reasons unrelated to memory quality.
Verify this when authoring tasks.

Use **sim time** (seconds/minutes) not step counts in task phrasing — same
substance, more legible.

## Two axes — keep separate
- **Episode count** — how many task attempts have accumulated. This is the main
  result; attempts-to-success is plotted against it.
- **History length** — how far back memory must reach. Supporting experiment only.
  A long idle timeline gives a hard recall problem but no repetition, and
  repetition is the claim.

## Stack (Phase 1 — MolmoSpaces)
- **Sim:** MolmoSpaces (MuJoCo). Chosen for setup cost — pip-installable, no Isaac
  Sim dependency chain. Ships RB-Y1 + Franka FR3 assets (MJCF/USD, cuRobo URDF
  config for RB-Y1) via its asset manager.
- **Base policies:** four configurations, B1–B4 — see "Current plan" above. All
  are released, zero-shot, vision-only checkpoints; none is finetuned. Navigation
  is RING everywhere except B1, where it is internal to MolmoBot.
- **Planner:** LLM/VLM emits subtask sequences. Memory conditions subtask
  selection, object choice, parameters, and **arm**.
  *Not base pose* — the VLA (B1) or RING (B2–B4) owns the base, so the planner
  cannot set it. Base pose remains a **retrieval key**, observed not commanded.
- **Success signal:** MolmoSpaces-Bench predicates. Pick is `succ_pos_threshold =
  0.01` m *and* contact with the robot only (`pick_task.py:169`). Opening is
  `door_openness_threshold = 0.67` for doors but `task_success_threshold = 0.20`
  for generic articulation — set these explicitly per experiment and record the
  value. Log both the predicate outcome and what the system believed; report
  divergence rate.

Phase 2 (deferred): port state-dependent tasks to OmniGibson/BEHAVIOR-1K, the only
sim with the object-state model those tasks need.

## Constraints
- **Retrieval acts on the planner only.** No *released* VLA accepts a
  demonstration or trajectory as conditioning input — EchoVLA trains memory
  conditioning into the policy, but is unreleased, and G0.5's "visual memory" is a
  5-second within-episode buffer discarded at the final layer. Holding the base
  policy fixed is a deliberate choice: it makes gains attributable to memory rather
  than retraining, and lets one result span B1–B4. Memory changes which subtask runs with which
  parameters; the VLA executes whatever it is handed.
- **The memoryless baseline must be able to succeed, expensively.** If information
  is unrecoverable by acting, the baseline scores zero and there is no floor for
  the curve to rise from. Memory replaces *search* with *recall* — it does not
  supply unobtainable facts.
- Policy input is images + language only. No privileged sim state as policy input.
  (Privileged state as *evaluation* signal is fine and expected.)
- Retry loop is driven by ground-truth predicates, not self-monitoring. Released
  VLAs cannot reliably detect their own failures (FailBench: near-chance on
  contact-rich outcomes, biased toward predicting success).
- Robot assets are downloadable but policies are embodiment-bound. MolmoBot runs on
  RB-Y1 only — importing another URDF gives a robot, not a working policy.

## No-cheat rule (no privileged state in the execution loop)

**The test is embodiment-mounted sensing, not "is it sim state".** A signal is
legitimate if it is physically realizable on the robot and would read the same way
on hardware — cameras on the robot, joint encoders, force/torque, onboard depth or
LiDAR declared in the URDF. Adding sensors to the robot model is not cheating; it
is specifying the robot. A signal is a cheat when it could only be produced by
querying the simulator: object poses, floor-plan occupancy, scene inventories.

The reason to care is narrow and specific: memory is valuable only because sensing
is expensive and transient. If any agent-side component can query where things
are, the exploration cost that memory is supposed to eliminate never existed, and
the headline curve measures nothing.

**Forbidden in the execution loop** (agent-side, every episode):
- Ground-truth object poses or scene inventories reaching the policy *or the
  LLM/VLM planner*. The planner is agent-side. Handing it a scene object list is
  the easiest and most damaging leak: memory would then "recall" facts the robot
  never had to observe.
- Privileged-map navigation (`planner/astar_planner.py` — reads `ProcTHORMap` /
  `iTHORMap` occupancy).
- cuRobo motion planning, which plans from privileged state.
- Precomputed grasp libraries, which encode object geometry the robot cannot see.
- Sim-only sensors as policy input — those reporting object or scene state rather
  than robot state. See the audit for the specific list.

**Permitted** (not agent-side, or explicitly designed exceptions):
- Success predicates and all evaluation metrics. Privileged by design.
- The retry-loop trigger. The spec deliberately drives retries from ground-truth
  predicates rather than policy self-monitoring (FailBench). This *is* a
  privileged signal; it is a stated exception, not an oversight. Report it as
  such, and report the divergence between predicate outcome and what the system
  believed.
- Offline authoring: benchmark JSON generation, episode construction, scripted
  between-episode object moves. These run outside the agent's loop.
- cuRobo for seeding demonstrations, under the existing restriction — plan-level
  records only, never trajectories.

**Consequence for Phase 1.** The robot must move under its own learned policy or
not move at all; there is no scripted-navigation shortcut that preserves the claim.
This forces the MolmoBot/RB-Y1 path — which is cheaper than first estimated, since
MolmoBot ships MolmoSpaces eval configs and its RB-Y1 action space includes the
base. See the appendix.

**The planner is agent-side.** Memory acts on the planner and nowhere else (see
Constraints), so the planner is the component whose behavior the experiment
measures — the VLA is held fixed precisely so that improvements are attributable
to memory. A planner that can query ground truth therefore does not merely leak;
it removes the thing being measured. In task 1 the no-memory baseline would locate
the object immediately, score at ceiling from episode 1, and the curve would be
flat. "We are testing the VLA, not the planner" inverts this: the VLA is the fixed
substrate, the memory-planner loop is the system under test.

## Tasks — Phase 1 (runnable in MolmoSpaces)
1. **Object relocation search** — two phases.
   *Viable only because RING supplies `navigate_to(X)`.* MolmoBot's prompt
   vocabulary is `{pick, place, place_next_to}` with no location slot, so under a
   monolithic VLA the memory and no-memory conditions emit identical strings and
   the curve is flat. Do not run this task on B1.
   *Exploration:* "find A", "find B", "find C" repeated over several episodes as
   ordinary kitchen work. Each object swaps between two fixed locations on a
   deterministic schedule.
   *Query:* fetch a named object; metric is re-exploration count. Worst case is
   searching everywhere; memory should predict the location and go straight there.
   Memory stores observation records (object, location, time); the location pattern
   is **derived at query time by the planner**, not maintained as running
   statistics — keeps the mechanism identical to tasks 2–4.

   **Swap schedule is config: per object, a period and a phase offset.**
   - *Synchronized* (period 1, same phase for all) — easy version, use to verify
     the pipeline. Warning: one observation reveals the global phase, so the task
     reduces to tracking a single bit and the curve saturates by episode 3.
   - *Desynchronized* (different periods and offsets per object, e.g. A every
     episode, B every 2, C every 3) — the real experiment. Forces per-object
     tracking and rewards longer history.

   Vary the number of exploration rounds to show query performance improving as
   evidence accumulates. Two-panel figure: exploration cost, then query efficiency.

2. **Order restoration** — restore arrangement AND order to a named earlier state.
   Start with structured records (object, location, timestamp) as the memory
   content; rendered-frame and retrieved-frame variants are harder follow-ups.
   Give the memoryless baseline a partial cue (some items still in place, or a
   commonsense-plausible target order) so it can guess at a nonzero rate.
3. **Setup reuse / cross-room transfer** — same task performed in a second room
   with different object instances and layout. Tests whether retrieved *plan
   structure* transfers when geometry changes.
**Task 4 (per-instance quirk) has moved to Phase 2 — see below.**

## Tasks — Phase 2 (need OmniGibson object state)
4. **Per-instance quirk** — *moved here from Phase 1, twice blocked.* The early
   check resolved against it: MolmoSpaces articulation is a bare joint-range ratio
   (`opening_tasks.py:88-120`) with no latch or required approach, so a quirk would
   have to be manufactured. And the only articulated checkpoint,
   `MolmoBot-SPOC-RBY1Articulated`, **overwrites its language goal** with a generic
   string and routes object identity through a privileged ground-truth point
   (`spoc_policy.py:192-198`), leaving memory no channel to act through.
   Deferred variant: quirk with changed object size — tests whether the system
   knows when NOT to reuse. Different claim from "reuse what worked"; keep separate.
5. Consumable state — item too old to use, not visible without memory
6. Tool substitution — pot didn't fit last time, pick the smaller one
7. Duration — this oven takes 8 min to preheat

## Protocol
- **Cold start.** Memory begins empty. The first-encounter vs. repeat-encounter gap
  is the headline number and disappears if memory is pre-seeded.
- Script all between-episode changes as config: per episode, a list of
  (object, from, to). Same harness serves all tasks; log what changed alongside
  what the memory recorded.
- **Decide per task: observed vs. unobserved changes.** If the robot witnesses a
  relocation, that tests attribution. If it discovers things missing, that tests
  search-and-recall. Don't mix within one experiment.
- **Fixed swap schedule, no drift** for task 1. A schedule that changes mid-run
  tests adaptation — a good second experiment, a bad first one, because a failure
  to learn is indistinguishable from a failure to adapt.
- Cap attempts per episode (start with 20); censored runs count as failures.
- Exploration is **plan-level only** — subtask sequence, object choice, parameters,
  arm. Not grasp-level, not trajectory-level.

## Metrics
- **Success rate** and **attempts-to-first-success** (over successful runs only —
  report the cap, never silently drop failures from the mean).
- Plot both against episode index. Aggregate numbers hide the claim.
- Headline: first-encounter attempts vs. repeat-encounter attempts.
- Supporting curve: subtask executions per episode (comparable to IOM's metric).
- If every configuration eventually succeeds, success rate stops discriminating
  and attempts carries the result — design at least one task where the 20-attempt
  budget bites.

## Baselines vs. ablations

These are two different tables and were previously conflated. **Baselines are
systems that exist** — what a practitioner has available today. **Ablations are
variants of our own system**, used to attribute the gain to a specific mechanism.
A result needs both: baselines show it is worth building, ablations show why it
works.

### Baselines (systems people have)

**B1 — Released VLA, instruction only.** `RBY1RigidManipEvalConfig` as shipped:
the task string goes straight into `obs["task"]`, no planner, no memory. Zero
implementation — this is the existing eval path, and the same run that measures
base success rate for build-order step 2. This is the floor.

**B2 — Modular agentic pipeline.** LLM planner decomposing a task into subtasks,
dispatching manipulation to π and locomotion to a navigation VLA. This is the
standard "orchestrate released components" architecture and is the baseline most
likely to be raised as an objection, so it must exist. It has a planner but no
persistent memory: each episode starts cold. Feasibility is worked out in the
appendix — the embodiment and the nav policy both need decisions.

**B3+ — Published memory methods**, reimplemented on this harness. To be selected
later; not specified here.

**Floor constraint applies to baselines.** A baseline that scores ~0 is not a
floor. B1 given a single instruction is a fair attempt on atomic tasks (pick, one
pick-and-place) but will likely score near zero on multi-step episodes, since the
policy is trained on atomic prompts and nothing decomposes the instruction. Either
keep multi-step episodes out of the B1 comparison, or report B1 only on the atomic
subset and use B2 as the floor for multi-step tasks. Decide per task; state which
floor each number is against.

### Ablations (variants of our system)

Same planner, same VLA, same tasks. Only what memory hands the planner changes.

| # | Ablation | Memory input | Attributes |
|---|---|---|---|
| A1 | No memory | nothing | Cost of the planner alone |
| A2 | Within-episode only | this episode, wiped at the boundary | That *across*-episode is what matters |
| A3 | Commonsense prior only | none; LLM world knowledge free | That the task deviates from priors |
| A4 | Naive retrieval | past episodes as raw concatenated text | That *structure* matters — raw history can hurt via causal confusion |

A3 is the one most likely to kill a task: if an LLM guesses the right receptacle
from commonsense, memory adds nothing. Design against A3 first.

The headline claim requires beating B1 and B2, and being attributed by A2, A3,
and A4.

## Build order
1. Stand up MolmoSpaces + RB-Y1; run MolmoBot on MolmoSpaces-Bench tasks.
   Gate: can reset a scene deterministically and change exactly N objects.
2. Measure base success rate per task. **Target the 20–60% band.** If ~0%, memory
   has nothing to improve; if ~90%, no headroom. Do this before building anything.
3. Build the episode store + retrieval. Schema: subtask, objects, spatial relations,
   order, parameters, outcome. Retrieval key: subtask language + scene embedding +
   base pose — not images alone.
4. Run baselines 1–4 on tasks 1, 2, 3 first.

## Deferred / open
- **Seeded vs. cold-start** ablation once the loop works — separates "having good
  episodes" from "having explored them." If seeding with cuRobo demonstrations,
  store plan-level records only, never trajectories: cuRobo plans from privileged
  state and storing its output leaks information the robot could not observe.
- **Capacity scaling** as a separate figure: hold the task fixed, vary how much
  history memory holds. Keeps it out of the main result as a confound.
- Human-demonstration-video track: dropped. Scripted dynamics give ground-truth
  relocation records for free and avoid step extraction entirely.
- Custom URDF import (bimanual YAM on Flow Base) only if the project goes to
  hardware. Importing a robot needs controller config, cuRobo collision spheres,
  camera mounts, and grasp params — not just the URDF.

---

# Appendix — repo-grounded notes (verified 2026-09-08)

Findings from reading this checkout (`plnguyen2908/molmospaces`, fork of
`allenai/molmospaces` v0.2.0). These do not change the spec above; they pin it to
what the code actually provides and flag three places where the spec's
assumptions need adjusting.

## Success predicates — one correction
The spec says "pick = object raised ≥1cm, open = ≥67% joint range."

- **Pick ≥1cm confirmed.** `PickTaskConfig.succ_pos_threshold = 0.01` (meters) at
  `molmo_spaces/configs/task_configs.py:64`, applied in
  `molmo_spaces/tasks/pick_task.py:169`. Note the predicate is a *conjunction*:
  success also requires `only_robot_collision` — the lifted object must be
  contacting the robot and nothing else. A lift achieved by dragging against
  another object does not count.
- **The ≥67% figure is door-specific.** `door_openness_threshold = 0.67`
  (`task_configs.py:151`) governs the door-opening task. The *generic* articulated
  opening task uses `task_success_threshold = 0.20` (`task_configs.py:120`,
  applied at `molmo_spaces/tasks/opening_tasks.py:68`). Task 4 (per-instance
  quirk) targets generic articulation, so its predicate is 20% unless overridden.
  Both are config fields — set them explicitly per experiment rather than
  inheriting the default, and record the value in the run log.

## Base policy — MolmoBot ships its own MolmoSpaces eval configs
*(Revised after reading allenai/MolmoBot. The earlier estimate here — "integration
work, not a config flag" — was too pessimistic.)*

There is no MolmoBot policy class in *this* repo, but the MolmoBot repo already
depends on MolmoSpaces and provides the missing half.
`MolmoBot-SPOC/eval/config/rby1_eval_config.py` subclasses **our**
`JsonBenchmarkEvalConfig` and imports `RBY1MConfig` and
`RBY1GoProD455CameraSystem` from `molmo_spaces.configs`. Two eval configs are
shipped, each pinned to a released checkpoint:

| Config | `task_type` | Checkpoint |
|---|---|---|
| `RBY1RigidManipEvalConfig` | `pick` | `allenai/MolmoBot-SPOC-RBY1Rigid` |
| `RBY1ArticulatedManipEvalConfig` | `open` | `allenai/MolmoBot-SPOC-RBY1Articulated` |

`model_post_init` calls `snapshot_download(hf_model_name)`, so checkpoint fetch is
automatic. The policy class is `molmobot_spoc.eval.spoc_policy.SPOCModelPolicy`.

**The base is in the action space** — this is genuine loco-manipulation, not a
fixed-base policy with a nav wrapper:

    action_spec = {"base": 3,        # x, y, yaw
                   "torso": 1,       # articulated config only
                   "left_arm": 7, "right_arm": 7,
                   "left_gripper": 1, "right_gripper": 1}
    action_keys = {"base": "joint_pos_rel", ...}

matching `RBY1MConfig(command_mode={..., "base": "holo_joint_rel_planar_position"})`.
Cameras are `head_camera`, `wrist_camera_r`, `wrist_camera_l`; `use_proprioception`
is True; `chunk_size=16`, `inference_dt_ms=800`.

What remains is **authoring RB-Y1 benchmark JSONs** — both configs are
`JsonBenchmarkEvalConfig` subclasses and need `--benchmark_dir`. That is the real
build-order step 1 cost, not the policy wiring.

## RB-Y1 — assets yes, benchmarks no
`eval_main.py:69-70` installs `rby1`/`rby1m` robot assets, and
`molmo_spaces/robots/rby1.py` + `kinematics/rby1_kinematics.py` exist. But every
benchmark documented in `mb-bench.md` is **Franka** (Pick-MSProc, Pick-Classic,
Pick-Filament, Pick-RandCam, PnP-v2, PnP-NextTo, PnP-Color). There is no shipped
RB-Y1 benchmark JSON. Build-order step 1's "run MolmoBot on MolmoSpaces-Bench
tasks" therefore requires authoring RB-Y1 benchmark JSONs
(`molmo_spaces/evaluation/benchmark_schema.py` carries a `robot` field for
factory lookup), or running the Franka benchmarks with a different policy purely
to calibrate the harness.

*Superseded by the no-cheat rule.* A fixed-base Franka FR3 Phase 1 was considered
and is not viable: task 1 (object relocation search) has no search cost without
mobility, so it would gut the headline task. And the mobile-Franka variant only
works if something drives the base — which, with scripted navigation banned, there
is nothing to do honestly. Authoring RB-Y1 benchmark JSONs is therefore on the
critical path, not optional. Budget it into build-order step 1.

## Episode harness — what exists to build on
- **Determinism gate (build-order step 1) is satisfiable.** `BaseMujocoTaskSampler`
  seeds `random`, `numpy`, and `torch` from `config.seed`
  (`molmo_spaces/tasks/task_sampler.py:418-422`) and re-seeds on reset
  (`:526-528`). Randomizers draw from a separate stream derived from
  `current_seed + 1` (`:806-826`), so domain randomization can be varied
  independently of layout — useful for holding arrangement fixed while varying
  appearance.
- **The gym API is the natural episode loop.** `molmo_spaces/tasks/gym_env.py`
  wraps a task sampler so `reset()` samples a fresh episode; `env.task` exposes
  the task API. Read `docs/gym_compatibility.md` first: these envs declare
  neither `action_space` nor `observation_space`, so `check_env` and most
  gymnasium wrappers do not work, and only `n_batch == 1` is supported. A run
  (sequence of episodes in one kitchen) maps onto repeated `reset()` with
  `options={"house_index": ...}` — `RESET_OPTIONS` is limited to `house_index`
  and `force_advance_scene`.
- **Scripted between-episode changes** go through
  `molmo_spaces/env/object_manager.py`. It is largely query-oriented
  (`get_object`, `is_receptacle`, `has_free_joint`, `is_pickup_candidate`);
  the "move object from A to B" primitive the protocol needs is written against
  MuJoCo state directly for bodies where `has_free_joint()` is true. Use
  `is_receptacle`/`has_receptacle_site` to enumerate legal locations so the swap
  schedule references real placements.

## Task 4 viability — resolve early, as the spec says
The spec flags: "if MolmoSpaces articulation is just a joint range with no special
motion required, this task moves to Phase 2." Current reading points that way —
`opening_tasks.py:88-120` computes `percent_open` as
`|current_joint_state| / |joint_range|` and nothing more. There is no per-instance
latch, resistance, or required approach direction in the success predicate. A
"quirk" would have to be *manufactured* (e.g. a joint whose range is only
traversable from one base pose), which is legitimate but is authoring work, not a
property of the shipped assets. Decide before committing to task 4 in Phase 1.

## Practical
- This fork's `main` is safe for project files: `.github/workflows/sync-upstream.yaml`
  force-pushes upstream into a separate `main_public` branch, not `main`.
- Assets are version-pinned in `molmo_spaces/molmo_spaces_constants.py` and
  overridable via `MLSPACES_PINNED_ASSETS_FILE`. **Pin them for the whole study** —
  an asset version bump mid-run silently changes the environment and invalidates
  cross-episode comparisons.
- `MUJOCO_EGL_DEVICE_ID` selects the render device and its indices do not
  necessarily match `CUDA_VISIBLE_DEVICES`.

## Privileged-state audit (2026-09-08)

Where sim state can leak into the execution loop in this codebase, and the verdict
under the no-cheat rule.

### Banned — navigation and motion planning
- `molmo_spaces/planner/astar_planner.py:12,138` imports `ProcTHORMap` / `iTHORMap`
  and plans over `self.map.occupancy`. This is a ground-truth floor plan, not
  anything the robot perceives.
- `molmo_spaces/policy/solvers/navigation/astar_planner_policy.py` wraps that
  planner and emits `{"base": waypoint}` (`:497`), reading true base pose from
  `robot_view.base.pose` (`:337,378,416`). Unusable as an agent component.
- `molmo_spaces/planner/curobo_planner*.py` — plans from privileged state.

Consequence: **there is no usable scripted navigator.** Base motion in Phase 1
must come from the learned policy.

### Banned — all shipped manipulation solvers
Every policy in `molmo_spaces/policy/solvers/object_manipulation/` (pick,
pick-and-place, next-to, color, open/close, and both cuRobo variants) consumes
the precomputed grasp library. These are oracle policies: fine for generating
data and for authoring benchmarks, never as the agent under test.

### Banned — privileged sensors as policy input
From `molmo_spaces/env/sensors.py`, these expose sim state and must not enter the
policy or planner observation:
`ObjectPoseSensor` (:139), `ObjectStartPoseSensor` (:564), `EnvStateSensor` (:376),
`DoorStateSensor` (:598), `GraspStateSensor` (:465), `TaskInfoSensor` (:538),
`ObjectImagePointsSensor` (:692 — projects true object position into image
coordinates; especially tempting for visual grounding, especially disallowed).

Use them freely for logging, predicates, and metrics. Wire an assertion that the
observation dict handed to the agent contains none of them — this is the single
check most likely to catch a silent regression.

### Allowed — onboard sensing
Cameras (`sensors_cameras.py`) and proprioception: `RobotStateSensor` (:44),
`RobotJointPositionSensor` (:328), `RobotJointVelocitySensor` (:352),
`TCPPoseSensor` (:98), `LastCommanded*` (:177-246), `LastActionSensor` (:287).

`RobotBasePoseSensor` (:124) is a judgment call: odometry is realistic onboard
sensing, but in sim it is exact and drift-free. Permitted as a *retrieval key*
(the spec already keys on base pose) and for closing the base controller's own
loop. It must not be used to localize objects.

### The planner's inputs are the real risk
The LLM/VLM planner is agent-side. It may condition only on what the robot has
observed plus retrieved memory. Do not populate its context from
`env/object_manager.py` queries (`get_context_objects`, `get_natural_object_names`,
`is_receptacle`, ...) — those read the scene, not the camera. Object *names* for
task phrasing are fine, since language is given; object *locations, presence, or
inventory* are not.

### Note on π0 / openpi
`policy/learned_policy/pi_policy.py` is honest by construction — its observation
is two camera images plus `qpos["arm"][:7]` and gripper (`:136-146`), and its
action is `{"arm": <7>, "gripper": <1>}` (`:171-186`). No privileged input, and
no base output. It remains useful as a **fixed-base harness-calibration run** for
build-order step 2 (it is already wired via `PiPolicyEvalConfig`), but it cannot
carry a loco-manipulation task.

## MolmoBot RB-Y1 — clean on `pick`, privileged on `open` (2026-09-08)

Verdict per checkpoint, from `MolmoBot-SPOC/eval/config/spoc_policy_configs.py`.

### `MolmoBot-SPOC-RBY1Rigid` (pick) — clean
`SPOCRBY1RigidManipPolicyConfig` sets **`use_image_points = False`**. Its inputs
are three robot-mounted cameras (`head_camera`, `wrist_camera_r`,
`wrist_camera_l`) plus proprioception from `obs["qpos"]`. Every one of those is
embodiment-mounted sensing. Output includes a 3-DoF base command. This is an
honest loco-manipulation policy and is the right substrate for Phase 1.

### `MolmoBot-SPOC-RBY1Articulated` (open) — carries a privileged goal channel
`SPOCRBY1ArticulatedManipPolicyConfig` sets **`use_image_points = True`** with
`point_camera_key = "head_camera"`. At inference `spoc_policy.py:203-210` requires
`obs["object_image_points"]` and errors without it — that is our
`ObjectImagePointsSensor` (`molmo_spaces/env/sensors.py:692`), which projects a
**ground-truth object pose** into image coordinates. It looks up the key
`pickup_obj` or `door_handle` (`:268-277`).

Two mitigating details, neither of which makes it disappear:
- It is captured **once, on the first step** (`if self.pickup_obj_image_points is
  None`), not tracked per step. So it is closer to goal specification — "the handle
  is here" — than to continuous privileged perception.
- It is a fixed property of the trained model. Removing it means retraining, which
  is out of scope.

**Implications, to decide before committing to task 4.**
- Task 4 (per-instance articulated quirk) can only run on this checkpoint, so it
  inherits the pointer. Arguably tolerable: the memory question there is *how* to
  open, not *where* the handle is, and every baseline gets the same pointer. But it
  must be disclosed, not discovered by a reviewer. Combined with the finding that
  MolmoSpaces articulation is a bare joint-range check, task 4 is now the weakest
  Phase 1 task — consider moving it to Phase 2.
- Task 1 (relocation search) must **never** run on this checkpoint. A first-frame
  pointer to the target object is exactly the fact the search task exists to make
  expensive. It is also ill-defined when the object is out of view, which is the
  normal case during search.
- Net: **build Phase 1 on `pick` / RBY1Rigid.** Tasks 1, 2, and 3 are all
  expressible as pick-and-relocate work, which keeps the whole study on the clean
  checkpoint.

### Assertion to wire in build-order step 1
Fail loudly if the observation dict handed to the policy contains
`object_image_points` while a search-family task is running. This single check
covers the most damaging leak and catches it as a crash rather than as a
suspiciously good result.

## The gap, stated precisely: what the VLA does not do (2026-09-08)

Neither MolmoBot nor π0 has a planner. Both are flat, language-conditioned
policies: task string + images + proprioception → action chunk. No task
decomposition, no subtask sequencing, no memory, no termination reasoning. The
outer loop this project builds is not replacing a component — there is nothing
there. This section records the interface contract that loop must satisfy.

### 1. The handoff is a string, and the injection point exists
`spoc_policy.py:185-191` resolves the goal as `obs["task"]`, else `obs["goal"]`,
else `self.task.get_task_description()`. So a planner-issued subtask reaches the
VLA by writing `obs["task"]` — no policy modification needed. This is where a
memory-conditioned subtask goes.

### 2. The subtask vocabulary is bounded by the training distribution
`utils/constants/prompt_templates.py` defines templates in slot form —
`pick_and_place` covers ~30 phrasings of "Pick up the {pickup_name} and place it
in or on the {place_name}", plus pick-only and open groups. The interface accepts
arbitrary strings, but the policy only *reliably* executes what it was trained on.

**Design consequence:** the planner's action space is effectively
`{pick(X), place(X, Y), open(X)}` over object names the scene can express. Plans
must decompose to these primitives, and memory's "parameters" are the slot
fillers. Sample the phrasing from the templates rather than inventing wording —
phrasing drift is a confound that would show up as a success-rate change with no
relation to memory.

### 3. There is no termination signal
`use_done_action = False` for both RB-Y1 configs, and `spoc_policy.py` emits no
`done`. The VLA runs until something outside it stops the episode. Subtask
segmentation — deciding when a subtask has ended so the next one can start — is
therefore harness work, and it is part of the gap. This is consistent with the
existing decision to drive the retry loop from ground-truth predicates rather
than policy self-monitoring.

### 4. Decisive: the Articulated checkpoint has no language channel
`spoc_policy.py:192-198` — when image points are in use, the goal string is
**overwritten** with a generic sentence:

    if "pickup_obj_image_points" in self.required_obs_keys:
        if "place" in task_desc:
            observations["goal"] = "Pick up the object with point and place it in or on the receptacle with point."
        elif "Pick up" in task_desc:
            observations["goal"] = "Pick up the object with point."

Object identity reaches that model through the **ground-truth point**, not through
words. A planner cannot steer it by language, because the language is discarded.

This upgrades the earlier finding from "privileged, disclose it" to
**unusable for this project**: memory's entire mechanism is choosing the subtask,
the object, and the parameters, and on this checkpoint there is no channel through
which that choice can be expressed except the privileged one. Task 4 cannot run
here. Move it to Phase 2 or drop it.

`SPOCRBY1RigidManipPolicyConfig` sets `use_image_points = False`, keeps the
natural-language goal, and is unaffected. Phase 1 runs on it.

## B2 feasibility — the modular pipeline (2026-09-08)

B2 is "LLM planner + π for manipulation + nav VLA for locomotion, on a mobile
bimanual robot." Three pieces; two of them need a decision, and the requested
embodiment is not available off the shelf.

### There is no mobile bimanual robot in MolmoSpaces except RB-Y1
From `molmo_spaces/configs/robot_configs.py`:

| Config | Mobile base? | Arms |
|---|---|---|
| `RBY1Config` / `RBY1MConfig` (:252, :302) | yes — `holo_joint_planar_position` | bimanual, 7-DoF, + torso |
| `MobileFrankaRobotConfig` (:176) | yes — `holo_joint_planar_position` | single 7-DoF |
| `BimanualYamRobotConfig` (:391) | **no** — `command_mode` is `{arm, gripper}` only | bimanual, 6-DoF |
| `FrankaRobotConfig` (:139) | no | single 7-DoF |

`BimanualYamRobotConfig.base_size` is a static pedestal that raises the robot off
the floor, not a drivable base. So mobile + bimanual means RB-Y1 — the same
platform as B1, which weakens B2 as an independent comparison.

### π cannot drive RB-Y1's arms
Two π integrations ship, and neither fits:
- `pi_policy.py` (π0-FAST-DROID) — single 7-DoF Franka arm + gripper. Matches
  `FrankaRobotConfig` and `MobileFrankaRobotConfig` exactly.
- `bimanual_yam_pi_policy.py` (**π0.5** via LeRobot gRPC) — 3 cameras + 14-dim
  state in, 14-dim action out (two 6-DoF arms + two grippers). Matches
  `BimanualYamRobotConfig`. **No base in the action space.**

RB-Y1's arms are 7-DoF; the YAM checkpoint's 14-dim action does not fit, so
running π on RB-Y1 would require training. Out of scope.

### There is no learned navigation policy
The only navigator in the repo is the A* planner policy, which is banned under the
no-cheat rule (privileged occupancy map). `rum_client.py` is a gripper client, not
navigation. **B2's nav VLA must be sourced externally**, and it must satisfy the
same rule: vision-driven, no privileged map. This is the largest open item in B2
and should be resolved before committing to the baseline.

### Recommended B2 configuration
**`MobileFrankaRobotConfig` + π0-FAST-DROID for manipulation + an external nav
VLA for the base.** This is the only combination where a released checkpoint's
action space matches a mobile MolmoSpaces embodiment without training: π0-DROID's
7-DoF arm maps onto the mobile Franka's arm, and the base is commanded separately
by the nav policy.

The trade is that B2 becomes single-arm. That is acceptable — B2's purpose is to
represent the modular-orchestration architecture, not to match B1's embodiment.
State the embodiment difference when reporting, since B1 (RB-Y1, bimanual) and B2
(mobile Franka, single-arm) are not embodiment-matched and the comparison is
architecture-level, not like-for-like.

If a bimanual B2 is required, the fallback is fixed-base bimanual YAM + π0.5,
which drops locomotion entirely and belongs with the tabletop track rather than
the loco-manipulation result.

## RING as the navigation policy — and what it unblocks (2026-09-08)

[The One RING](https://one-ring-policy.allen.ai/) (arXiv 2412.14401, Ai2) is an
embodiment-agnostic indoor navigation generalist. It resolves B2's open item and,
more importantly, changes which architecture should carry the main experiment.

### Released and clean under the no-cheat rule
- Code: `github.com/Ainaz99/OneRING`. Weights: HF `AinazEftekhar/OneRING`,
  `ring_model_step_40356421.ckpt`. Data: `allenai/ring-data`.
- Observation (`online_evaluation/simple_inference.py`): `raw_navigation_camera`,
  `raw_manipulation_camera` (2 RGB), `goal` (natural language), `last_actions`.
- **No privileged input.** The paper is explicit that the policy "does not have
  access to any privileged information about its current body" — it infers
  embodiment from observations and transition dynamics, and needs no embodiment
  parameters at inference. No map, no occupancy grid, no ground-truth pose.
  Passes the no-cheat rule where A* does not.
- Trained purely in simulation on 1M randomized embodiments; evaluated zero-shot
  on real Stretch RE-1, LoCoBot, Unitree Go1, **and RB-Y1**.

### Integration is cheaper than it looks — same architecture family as MolmoBot
RING is built on the **SPOC** stack (`architecture/models/spoc_models/`,
`REGISTERED_MODELS`), the same family as `MolmoBot-SPOC`, and uses the identical
sensor names already present in MolmoBot's
`utils/constants/sensor_constants.py` (`raw_navigation_camera`,
`raw_manipulation_camera`). A RING policy class should closely mirror
`SPOCModelPolicy`.

**Action space is discrete** (`environment/action_spaces.py:170`,
`SPOCV1ActionSpace`): `move_ahead` (+0.2 m), `move_back` (−0.2 m),
`rotate_left`/`rotate_right` (±30°), `rotate_left_small`/`rotate_right_small`
(±6°), plus `done` / `sub_done`. These map onto
`holo_joint_rel_planar_position` as relative (x, y, θ) base targets — a small
adapter, not a redesign.

**Unverified:** RING was trained in AI2-THOR (Unity rendering); MolmoSpaces
renders the same THOR-derived scenes in MuJoCo. The sim-to-sim appearance gap is
real and must be *measured* before RING is relied upon — treat its success rate
in MolmoSpaces as an open number, not an inherited one.

### The consequence: RING restores task 1
The earlier blocker was that MolmoBot's prompt vocabulary is
`{pick(X), place(X,Y), place_next_to(X,Y)}` with no location slot, so a planner
holding "the apple is in the fridge" had no way to say it — search happens inside
the VLA, unreachable by language, and the memory and no-memory conditions emit
identical strings.

RING is an **object-goal navigation policy driven by a language instruction**. It
supplies the missing primitive: `navigate_to(X)`. With it the planner's action
space becomes `{navigate_to(X), pick(X), place(X, Y)}`, and memory can express
location knowledge directly — which is the whole mechanism task 1 needs.

**This promotes the modular stack from baseline to candidate substrate for the
main system.** A monolithic loco-manipulation VLA hides navigation where memory
cannot condition it; the modular stack exposes it. Reconsider the Stack section
on that basis — the argument for MolmoBot-as-substrate was that it avoids
scripted navigation, and RING satisfies that requirement while keeping navigation
addressable.

### Revised B2, embodiment-matched
Prefer **RING (navigation) + `MolmoBot-SPOC-RBY1Rigid` (manipulation) on RB-Y1**,
orchestrated by the LLM planner. Both policies are released, both are clean, and
RING has been run on RB-Y1. This makes B1 and B2 differ *only* in architecture —
monolithic VLA vs. planner-orchestrated modular stack — which is the comparison
worth reporting.

The mobile-Franka + π0-DROID variant remains available if a π-based B2 is wanted,
but it changes embodiment and manipulator simultaneously and so confounds the
comparison. π's serving path is already wired either way:
`pi_policy.py:72-95` connects via `openpi_client.websocket_client_policy`
(host/port from `remote_config`, default `localhost:8000`), so an openpi
`serve_policy.py` server needs no new code; π0.5 for bimanual YAM goes through
LeRobot gRPC instead (`bimanual_yam_pi_policy.py`).

## B3 — RING + π₀.₅ on mobile Franka (2026-09-08)

A second modular configuration, requested to test the memory loop on a different
embodiment. Feasible with released checkpoints and no model training.

### π₀.₅-DROID is the right π checkpoint, and it needs no new code
openpi releases **`pi05_droid`** (`gs://openpi-assets/checkpoints/pi05_droid`) —
π₀.₅ fine-tuned on DROID, single 7-DoF Franka arm + gripper. Its README
description is directly relevant: *"fast inference and good language-following."*

Language-following is the property that matters most here. The planner→VLA
interface is a string, so a policy that follows language well is worth more than
one that manipulates marginally better. On that basis **π₀.₅-DROID is preferable
to π₀-FAST-DROID** for every modular configuration, and it should replace
π₀-FAST-DROID wherever this spec previously named it.

Integration cost is zero. `pi_policy.py:_prepare_local_model` resolves its config
by checkpoint directory name — `_config.get_config(os.path.basename(checkpoint_path))`
— so a directory named `pi05_droid` picks up the right openpi config
automatically. The DROID observation schema is unchanged
(`observation/exterior_image_1_left`, `observation/wrist_image_left`,
`observation/joint_position` (7), `observation/gripper_position`, `prompt`), and
so is the 8-dim action. Set:

    PiPolicyConfig(checkpoint_path=".../pi05_droid",
                   remote_config=dict(host="localhost", port=8080))

against a server started with
`serve_policy.py policy:checkpoint --policy.config=pi05_droid --policy.dir=...`.
Note `PiPolicyConfig` defaults to port **8080**
(`molmo_spaces/configs/policy_configs_baselines.py:10`) while `pi_policy.py:82`
falls back to 8000 — set the port explicitly rather than relying on either.

*Do not* use `bimanual_yam_pi_policy.py` here: that is π₀.₅ bound to the bimanual
YAM interface (3 cameras, 14-dim state and action, LeRobot gRPC), and its action
vector does not fit a single-arm Franka.

### Embodiment: `MobileFrankaRobotConfig`
Mobile base (`holo_joint_planar_position`) + single 7-DoF Franka arm
(`robot_configs.py:176`). The arm matches π's DROID action space exactly; RING
drives the base through the same discrete-action adapter as B2. RING is
embodiment-agnostic by construction and needs no parameters for a new body, so
the mobile Franka is in-distribution for it in the way an arbitrary new robot
would be.

### What this configuration buys, and what it costs
**Buys:** a genuine embodiment ablation. B2 and B3 share the architecture (LLM
planner + RING + a manipulation VLA) and differ only in robot and manipulator, so
a memory effect appearing in both is evidence the mechanism is not
embodiment-specific. That is a stronger claim than a single-platform result.

**Costs:** B3 is single-arm, so tasks requiring two hands are out. It also needs
its own base-success-rate calibration — the 20–60% band must be re-measured for
π₀.₅-DROID on mobile Franka, not inherited from RB-Y1.

### Configuration summary

| | Embodiment | Navigation | Manipulation | Planner |
|---|---|---|---|---|
| B1 | RB-Y1 | *(inside VLA)* | MolmoBot-SPOC-RBY1Rigid | none |
| B2 | RB-Y1 | RING | MolmoBot-SPOC-RBY1Rigid | LLM |
| B3 | Mobile Franka | RING | π₀.₅-DROID (`pi05_droid`) | LLM |

B1→B2 isolates architecture on a fixed embodiment. B2→B3 isolates embodiment on a
fixed architecture. The memory system runs on top of B2 and B3 alike.

## B4 — GR00T on bimanual YAM (2026-09-08)

*Corrects the initial GR00T assessment in this appendix, which judged GR00T the
most expensive integration. That is true of the NVIDIA reference implementation on
mobile Franka. It is false for bimanual YAM, where GR00T is the **cheapest**
integration of any policy considered.*

### The existing YAM path is policy-agnostic
`molmo_spaces/policy/learned_policy/lerobot_grpc_client.py` is a generic LeRobot
policy-server client — its `connect()` takes `policy_type` as a parameter
("pi05", "act", "smolvla", ...). `bimanual_yam_pi_policy.py:115` reads it from
config: `self.remote_config.get("policy_type", "pi05")`, and
`BimanualYamPiPolicyConfig.remote_config`
(`configs/policy_configs_baselines.py:109-114`) exposes it directly.

GR00T N1.7 is served by LeRobot as the **`groot`** policy type. So switching the
bimanual YAM manipulator from π₀.₅ to GR00T is a config change:

    BimanualYamPiPolicyConfig(
        checkpoint_path="<yam-finetuned-groot-checkpoint>",
        remote_config=dict(host=..., port=..., policy_type="groot", device="cuda"),
    )

**Caveat:** `bimanual_yam_pi_policy.py:118-142` hardcodes `lerobot_features` to
match the π₀.₅ training dataset's schema. A GR00T checkpoint finetuned on a
different YAM dataset may use a different state layout, in which case
`lerobot_features` needs editing — a small change, but not zero.

### Checkpoint provenance must be checked before relying on it
A YAM-finetuned GR00T checkpoint exists third-party
(`robocurve/gr00t-n1.7-yam-molmoact2`, with `inspect-robots-yam` adapters for
I2RT YAM bimanual arms). It is **not** an NVIDIA or Ai2 release. Verify license,
training data, and quality before it carries any result. The alternative is
finetuning `nvidia/GR00T-N1.7-3B` with `NEW_EMBODIMENT` on YAM data, which is
training work and outside the current scope.

Note also that all GR00T checkpoints load the **gated** `nvidia/Cosmos-Reason2-2B`
backbone — an approved HF access request is a prerequisite, so start that early.

### B4 must be loco-manipulation — so YAM needs a base
Every configuration in this study carries a locomotion claim; a fixed-base B4 does
not qualify. As shipped, `BimanualYamRobotConfig` has `command_mode = {arm,
gripper}` and `bimanual_yam.py:56-60` builds move groups for `left_arm`,
`right_arm`, `left_gripper`, `right_gripper` only. `base_size` is a static
pedestal. There is nothing to navigate with.

GR00T cannot supply the locomotion itself for this embodiment.
`examples/GR00TWholeBodyControl` *is* genuine loco-manipulation — walking, table
approach, whole-body pickup — but it is for the **Unitree G1 humanoid**
(`UNITREE_G1_SONIC`), and it is a finetuning recipe from `nvidia/GR00T-N1.7-3B`,
not a zero-shot capability. Any YAM checkpoint outputs arms and grippers.

**Chosen route: add a holonomic base to bimanual YAM in simulation.**
*Revised — this is cheaper than first written. It is Python, not MJCF asset work.*

MolmoSpaces **synthesizes the mobile base in code at scene-build time**; it is not
in the robot asset. `franka_droid/model.xml` compiles to 8 actuators (7 arm +
gripper) and no base. `MobileFrankaRobot.add_robot_to_scene`
(`robots/mobile_franka.py:128-200`) then builds the base itself: a body, a
`base_size` box geom, a `base_site`, a world site, and three site-transmission
actuators (`base_x_act`, `base_y_act`, `base_theta_act`) via a local
`add_slider_act` helper, before attaching the unmodified arm spec on top.
Verified by construction: the assembled mobile Franka has `nu=11`, `nv=16`, with
base actuators present.

`BimanualYamRobot.add_robot_to_scene` (`robots/bimanual_yam.py:105-150`) is the
same structure with two differences: its base body is `mocap=True` (kinematically
posed, not actuated), and it adds no actuators or sites. Assembled, it has
`nu=14`, `nv=16`, and **no base actuators**.

So the work is:

1. `bimanual_yam.py`: drop `mocap=True`, add `base_site` and the world site, and
   add the three slider actuators — port `add_slider_act` from `mobile_franka.py`.
2. `BimanualYamRobotView`: expose a `base` move group.
3. `BimanualYamRobotConfig`: add `"base": "holo_joint_planar_position"` to
   `command_mode` and `base_control_params`; instantiate the base controller in
   `bimanual_yam.py` mirroring `mobile_franka.py:45-50`.

No MJCF editing, no new asset. The base is a **holonomic slab** — a box that
translates in x/y and rotates about z, with no wheels or suspension — so B4's
locomotion is as abstract as B3's, and the two remain comparable.

> **Bug found and fixed while verifying this.** `mobile_franka.py:170` referenced
> `R` (scipy `Rotation`) while the only import sat inside the file's
> `if __name__ == "__main__":` block, so `add_robot_to_scene` raised `NameError`
> on every library call — mobile Franka could not be built at all. No experiment
> config references `MobileFranka`, which is why it went unnoticed. Fixed by
> hoisting the import to module scope. **Implication for B3: mobile Franka is
> unexercised code, so budget for more latent breakage than the one line.**

Then RING drives the base and GR00T drives the arms, exactly as in B2/B3. No model
training anywhere.

**Why the fixed-base GR00T checkpoint stays in-distribution.** It was trained on a
stationary YAM, and the modular decomposition keeps it stationary during
manipulation: RING navigates, stops, then GR00T manipulates. Move-then-manipulate
is already how B2 and B3 work, so this is not a special accommodation — but it
does mean B4 cannot support manipulation-while-driving, and no task should require
it.

**Rejected alternatives.** *Unitree G1 + GR00T-WBC*: MolmoSpaces ships G1 assets
(per the README attributions) but has **no G1 robot class** — `molmo_spaces/robots/`
covers Franka, mobile Franka, RB-Y1, YAM variants, and floating grippers only. That
means a full robot integration plus `UNITREE_G1_SONIC` finetuning. *GR00T on
RB-Y1*: needs `NEW_EMBODIMENT` finetuning; feasible since MolmoSpaces can generate
RB-Y1 data, but it is a training project and duplicates B2's embodiment.

### What B4 covers that B1–B3 do not
B3 varies the *policy* on a mobile single-arm robot; B4 varies the *morphology*,
testing whether the memory loop holds with two arms. Only RB-Y1 (B1/B2) is
otherwise bimanual, and there the manipulator is welded to MolmoBot — so B4 is the
only way to separate "bimanual" from "MolmoBot".

### Updated configuration table

| | Embodiment | Base | Arms | Navigation | Manipulation | Planner |
|---|---|---|---|---|---|---|
| B1 | RB-Y1 | mobile | 2 | *(inside VLA)* | MolmoBot-SPOC-RBY1Rigid | none |
| B2 | RB-Y1 | mobile | 2 | RING | MolmoBot-SPOC-RBY1Rigid | LLM |
| B3 | Mobile Franka | mobile | 1 | RING | π₀.₅-DROID | LLM |
| B4 | XLeRobot | mobile | 2 (independent) | RING | 2x G0.5 `g05-so101` | LLM |

B1→B2 isolates architecture. B2→B3 isolates embodiment and policy. B3→B4 isolates
morphology. All four carry a locomotion claim; B4 requires the base-addition work
above before it does.

## Naming and camera mounting for B3/B4 bases (2026-09-08)

### Call it what it is
MolmoSpaces' "mobile Franka" is a Franka FR3 attached to an actuated box:
`add_robot_to_scene` builds a single 0.5 × 0.5 × 0.58 m `mjGEOM_BOX` and drives it
with three site actuators. No wheels, no chassis, no suspension. Verified: the
assembled model is 65 geoms, of which the base is one box.

This is defensible — real DROID hardware is a Franka on a rolling cart, and the
asset is named `franka_droid` with ZED cameras to match. The abstraction is that
MolmoSpaces *actuates* the cart, where DROID has a human move it. But do not write
"mobile Franka" as though it names a robot. Write **"Franka FR3 on an actuated
holonomic base."** The same applies to B4's YAM once its base is added.

The box does collide (`contype=1, conaffinity=1`), so navigation is physically
constrained. Its footprint is crude, though: a 0.5 m square slab will wedge in
doorways and clip furniture where a real base would not. **Log base-collision
events separately** so navigation failures caused by the footprint can be
distinguished from policy failures.

By contrast RB-Y1 is a fully modelled robot — chassis meshes on the base body,
1021 geoms. **B1/B2 are therefore the platform-credible loco-manipulation
results; B3/B4 are policy and morphology ablations and should not carry claims
about locomotion realism.**

### The exterior camera must be mounted on the base
`franka_droid/model.xml` defines only `wrist_cam`. Exterior cameras come from
config, and `FrankaRandomizedD405D455CameraSystem` uses
`RandomizedExocentricCameraConfig` — cameras placed around the **workspace
centre**, world-fixed.

π₀.₅-DROID consumes `exterior_image_1_left` + `wrist_image_left` and assumes the
exterior camera views its own workspace. With a world-fixed exo camera, the moment
the base drives to a new receptacle that assumption breaks and π receives an
exterior image of somewhere else.

**Required for B3 (and B4):** mount the exterior camera on the base via
`RobotMountedCameraConfig` (`configs/camera_configs.py:59`) so it travels with the
robot — matching real DROID, where the camera mount sits on the cart. This is a
config change, not code, but it is not optional: without it B3 produces quietly
wrong observations rather than an error.

## EchoVLA — closest prior work, and a challenge to a core assumption (2026-09-08)

[EchoVLA](https://arxiv.org/abs/2511.18112), *"Robotic Vision-Language-Action Model
with Synergistic Declarative Memory for Mobile Manipulation"* (Nov 2025). This is
the nearest neighbour to this project's thesis and was not in the related work.
It must be addressed.

**What it is.** A VLA with two memories: *scene memory* (spatial-semantic maps of
the evolving layout) and *episodic memory* (a buffer of task-level experiences —
prior instructions, observations, decisions). Retrieved representations are fused
by coarse- and fine-grained attention and condition base-arm diffusion policies.
It ships a benchmark, **MoMani**, built on **RoboCasa** (MuJoCo), with procedurally
generated sim tasks plus real-robot evaluation. Real platform: Kinova Gen3 7-DoF
on a holonomic mobile base (TidyBot++), front RGB-D and top-view stereo, 7×7 m
arena.

### It contradicts our stated constraint — and the constraint needs restating
This spec asserts: *"No released VLA accepts a demonstration or trajectory as
conditioning input. Memory changes which subtask runs with which parameters; the
VLA executes whatever it is handed."* EchoVLA fuses retrieved memory **directly
into the diffusion policy** as conditioning for base and arm actions.

The literal claim survives — EchoVLA is not a *released* VLA one can condition;
it is a VLA *trained* to be memory-conditioned, and no released checkpoint
(MolmoBot, π₀.₅, GR00T) accepts memory. But the architectural claim does not:
planner-level retrieval is a **choice**, not a necessity, and a reviewer will ask
why we did not train memory conditioning into the policy.

**Defensible answer, which should be stated in the paper rather than left
implicit:** we hold the base policy fixed and released so that any improvement is
attributable to the memory loop rather than to policy retraining, and so the
result transfers across policies (B1–B4 vary the VLA). EchoVLA's design confounds
memory with a bespoke trained policy and cannot show that.

### Does our differentiator survive? Probably — but VERIFY THIS FIRST
Our claim is that no benchmark measures **improvement over repeated executions in
the same environment**. EchoVLA's protocol reads as *independent* trials: "50
evaluation episodes per task" across "three random seeds" in sim, and "20
independent trials with randomized initial robot base positions" real-world.
Independent trials with randomized starts measure robustness, not accumulation.

**Unverified and load-bearing:** whether EchoVLA's episodic memory *persists
across* those evaluation episodes, and whether any result is reported as a
function of episode index. If memory persists and improvement is plotted, our
headline contribution is substantially pre-empted and must be re-scoped. Read the
evaluation section directly before building further — this is the highest-value
hour available right now.

### Practical notes
- **No code, weights, or MoMani release found.** So EchoVLA cannot be a
  drop-in baseline; comparing means reimplementing, which is a large task. Treat
  it as related work first, and only as a baseline if the reimplementation is
  scoped deliberately.
- **MoMani is RoboCasa/MuJoCo**, adjacent to MolmoSpaces. Even without a release,
  its task taxonomy and procedural generation are a useful design reference for
  our tasks 1–3, and a point of comparison when we justify using MolmoSpaces.
- Add EchoVLA to the related-work list alongside LA-EQA and GOAT-Bench, and
  update the "no existing benchmark measures repeated-execution improvement"
  sentence to name it explicitly and say why it does not.

## B4 revised — XLeRobot + RING + 2x G0.5-SO101 (2026-09-08)

Supersedes the LinearBot and based-YAM routes as the primary B4. Every component
is released, zero-shot, and requires no training.

| Component | Role | Source |
|---|---|---|
| XLeRobot | mobile dual-arm platform | `Vector-Wangel/XLeRobot`, Apache-2.0, native MJCF |
| RING | base / navigation | `AinazEftekhar/OneRING` |
| G0.5 `g05-so101` x2 | one arm each | `OpenGalaxea/G05` |
| LLM planner | subtask, object, parameters, **arm** | ours |

### Why this beats the earlier B4 routes
- **Dimensions match exactly.** `configs/data/so100.yaml` declares a single
  `right_arm` action of `raw_shape: 6`; XLeRobot has exactly 6 actuators per arm
  (Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw). No padding, no IK.
- **Zero-shot.** `g05-so101` is a released deployment checkpoint, unlike every
  other SO-100 route (SmolVLA, ACT, GR00T `examples/SO100`, `Maelic/openpi-SO100`),
  all of which require collecting data and finetuning.
- **The base is real.** `slide_joint_x`, `slide_joint_y`, `hinge_joint_z` map
  directly onto `holo_joint_planar_position`. A modelled chassis with omni-wheels,
  not a synthesized box (contrast B3, and the based-YAM proposal).
- **No sysID dependency.** Unlike LinearBot, nothing here waits on parameters
  Shuo has not yet released.

### Arm assignment is a planner decision
Two independent policy servers, one per arm, no coordination between them. The
LLM planner selects the arm, so the planner's action space becomes:

    {navigate_to(X), pick(X, arm), place(X, Y, arm)}

**This is a gain for the study, not a concession.** Arm choice is another
plan-level parameter, so it is another thing episodic memory can condition:
"the left arm could not reach the top shelf in this kitchen" is exactly the kind
of per-instance fact the memory schema already stores. Add `arm` to the record
schema alongside subtask, objects, spatial relations, order, parameters, outcome.

Describe it in writing as **two independently-controlled arms**, never as bimanual
manipulation — the arms cannot cooperate on a single object, and no task should
require them to.

### Integration checklist
1. Place XLeRobot MJCF + `assets/` (28 MB of STLs) in the **asset cache dir**, not
   the repo. It is unpinned like LinearBot, so record how it was obtained and
   note that a fresh machine will not fetch it automatically.
2. Robot class, robot view, and config following `docs/tutorials/add_robot.md`,
   with move groups `base`, `left_arm`, `right_arm`, `left_gripper`,
   `right_gripper`. Model on `bimanual_yam.py` plus `mobile_franka.py`.
3. **Cameras: the MJCF ships `ncam = 0`.** G0.5 wants `exterior`, `wrist_left`,
   `wrist_right`. Add wrist cameras via `RobotMountedCameraConfig` and a
   base-mounted exterior (the same fix B3 needs). The asset provides camera-mount
   geometry and pan/tilt head joints as mounting points.
4. Dual-client policy wrapper: two `LeRobotGRPCClient`-style clients on separate
   ports, merging into `{"left_arm", "right_arm", "left_gripper", "right_gripper"}`.
   Serve with `experiments/so100/start_server.sh <ckpt>` twice on different ports.
5. **Read `so100_policy_client.py` (48 KB) before wiring the cameras** — the
   `wrist_left` / `wrist_right` slot semantics for a single-arm checkpoint are
   undocumented, and getting the mapping backwards would silently degrade both
   arms rather than error.
6. Enable `proprio_guard: mode: "clip"` from `experiments/so100/client_config.yaml`
   and **log every activation**. It clamps joint state to the training
   distribution to stop out-of-distribution proprioception from "collapsing the
   model's action predictions" — so activations are a direct signal that sim has
   drifted outside the policy's training regime.

### Costs and unknowns
- **VRAM:** ~11 GB per checkpoint in bf16, so ~22 GB for two servers plus
  activations. Comfortable at 80 GB, tight at 40 GB, not viable at 24 GB. Prefer
  two GPUs.
- **Latency:** two forward passes; they serialise on one GPU. Mitigated by
  `--action_steps 32`, but that also means each arm runs 32 steps open-loop with
  no knowledge of the other's actions.
- **Two unmeasured sim-to-sim gaps:** `g05-so101` trained on real SO-101 data,
  RING trained in AI2-THOR's renderer, both running against MuJoCo here. Measure
  both in build-order step 2; inherit neither success rate.
- **Licence:** GalaxeaVLA is `NOASSERTION` with a custom `LICENSE-G0.5`. Read it
  before any published use.

### Keep LinearBot as the fallback
If XLeRobot integration stalls, LinearBot remains viable and its arms match the
YAM policy interface — but it is blocked on the sysID parameter transfer and its
arm servos currently miss commanded poses by ~2 cm at the gripper.

## Camera configuration and randomisation (2026-09-08)

### Where cameras come from
Not the URDF. MolmoSpaces builds them from a Python `CameraSystemConfig`
(`molmo_spaces/configs/camera_configs.py`) with three backing types:
`MjcfCameraConfig` (references a camera already present in the MJCF),
`RobotMountedCameraConfig` (attaches to a body by `reference_body_names` with
offsets — needs nothing in the asset), and `Fixed`/`RandomizedExocentricCameraConfig`
(world-placed).

### Decision: randomisation off for this study
**Cameras are frozen — no FOV, position, or orientation noise — and the same
camera pose is used for every episode within a run.**

*Why.* Viewpoint variance is an independent source of success-rate variance
layered on top of the effect being measured. If the camera moves between episode 3
and episode 7, the attempts-to-success curve mixes memory effects with viewpoint
effects and neither is recoverable.

*This is nearly free.* The benchmark schema has **no noise fields**:
`RobotMountedCameraSpec` carries `camera_offset`, `lookat_offset`,
`camera_quaternion`, `fov`, `record_depth`; `ExocentricCameraSpec` carries `pos`,
`forward`, `up`, `fov`. Randomisation is resolved into concrete numbers at
authoring time and baked into the `EpisodeSpec`, so a JSON-driven run is already
deterministic. The work is authoring one camera set per run and copying it down
the episode chain — not disabling machinery.

*What it costs, and it must be disclosed.* Randomisation at eval is **not** only a
datagen device. `RBY1GoProD455CameraSystem` — the system MolmoBot's own eval config
selects — carries per-camera noise, and `mb-bench.md` ships **Pick-RandCam** as a
deliberate randomised-camera benchmark, because viewpoint robustness is one of
MolmoBot's headline claims. Running fixed cameras therefore means **our success
rates are not directly comparable to published MolmoBot numbers**. Say so when
reporting. If a reviewer wants the robustness axis, it is a separate experiment:
hold memory fixed, vary camera noise.

### Optics must match each checkpoint's training rig
Camera *count* is not enough — a right-count/wrong-FOV rig is a silent
distribution shift. Status per configuration:

| | Cameras | Optics | Status |
|---|---|---|---|
| B1/B2 | `head_camera`, `wrist_camera_l/r` | GoPro 139° head, D455 58° wrists | **Shipped and correct** (`RBY1GoProD455CameraSystem`); all `MjcfCameraConfig`, present in the RB-Y1 asset |
| B3 | exterior + wrist | ZED, DROID mounting geometry | System exists (`FrankaRandomizedD405D455CameraSystem`) but the **exterior camera is world-placed** and must be re-declared as `RobotMountedCameraConfig` on the base — otherwise π sees the old workspace after every navigation leg |
| B4 | `exterior`, `wrist_left`, `wrist_right` | **Unknown** — read `experiments/so100/so100_policy_client.py` | XLeRobot MJCF has `ncam = 0`; all three must be created as `RobotMountedCameraConfig` |

B4's optics and the `wrist_left`/`wrist_right` slot semantics are both undocumented
and both live in the same 48 KB client file. Read it once and settle both before
authoring any B4 episode.

## Task 2 concrete design — two-receptacle restoration (2026-09-08)

Settles the semantics of "restore arrangement AND order". Supersedes the
left-to-right reading.

### Why not spatial ordering
An earlier design judged left-to-right order along a table axis. **It is not
commandable.** Every VLA's placement vocabulary is `pick_and_place` and
`pick_and_place_next_to`, and the next-to templates say only "next to" / "near" —
`PickAndPlaceNextToTaskSpec` exposes `min_/max_surface_to_surface_gap`, distances
with no direction. There is no left/right/above/below anywhere. A planner could
never aim for a specific spatial order, so the metric would measure manipulation
scatter, not memory.

### The design
Order is **categorical, by receptacle**.

- Pick **two receptacles** `R1`, `R2` anywhere in the house — tables, counters,
  shelves, sinks. Select with `object_manager.is_receptacle` /
  `has_receptacle_site`; require enough separation that moving between them needs
  navigation, and enough surface for all tracked objects.
- Pick **N tracked objects**, N = 4–6, each passing `is_pickup_candidate`
  (free joint + valid identifier). Anything failing that filter may be a
  destination but never a tracked object.
- **Work episode:** move one or more tracked objects between `R1` and `R2`. Plain
  `pick_and_place` — the existing `PickAndPlaceTask`, no new class needed.
- **Restoration episode:** "restore to how it was at \<episode T\>". Each tracked
  object returns to the receptacle it occupied at T.

### Predicate — reuse, don't invent
For each tracked object, `is_object_supported_by_body(obj, receptacle,
frac_weight_threshold=receptacle_supported_weight_frac)` — the same check
`pick_and_place_task.py:120-133` uses, so the semantics match the shipped
benchmarks. Success = all N correct. `get_info()` reports per-object correctness,
giving a graded "k of N restored" signal for the attempts curve.

**Author runs so each tracked object has a distinct receptacle at any time.** Two
objects on one receptacle have unconstrained relative arrangement, unmeasurable
under this predicate and ambiguous to phrase for the VLA.

### Why this design is strong
- **Defeats A3 (commonsense prior).** Which receptacle an object sat on N episodes
  ago is arbitrary — no LLM prior predicts it. A3 is the ablation most likely to
  kill a memory task; this kills A3 for free.
- **Fully commandable.** `place(X, R1)` is the best-supported template in every
  configuration B1–B4.
- **Discrete.** No pose tolerance, no axis convention, no sensitivity to the ~2 cm
  placement scatter a VLA produces.
- **Navigation has a real role** without requiring search, since the receptacles
  are apart.

### Knobs
| Knob | Effect |
|---|---|
| N objects | Chance floor is 1/2^N (1/16 at N=4, 1/64 at N=6) |
| Query depth (how far back) | Difficulty, and **this is the history-length axis** — free, no separate experiment |
| Objects moved per work episode | Rate of divergence; too slow and arrangements do not differ enough to test temporal indexing |
| **Partial-cue fraction** | Fraction of objects left undisturbed, so the memoryless condition can infer rather than guess blind |

### One tension, recorded deliberately
The spec requires that memory replace *search* with *recall* and not supply
unobtainable facts. A past receptacle assignment **is** unobtainable by acting —
no exploration recovers it. So task 2 tests pure recall, unlike task 1 where the
object's current location is genuinely discoverable and memory only saves search
cost. The prescribed mitigation is the **partial cue**: leave some objects in
place so the memoryless condition can infer at better than chance. Build the cue
fraction in as a config knob from the start; retrofitting it later means
re-authoring every run.

## Task 2 run structure — episode taxonomy and cadence (2026-09-09)

### Why exogenous change is required, not optional
If only the robot moves objects, an earlier arrangement can be restored by
replaying the system's own action log in reverse. That is bookkeeping, not
episodic memory, and it invalidates the claim. Scripted changes the robot does not
cause break the shortcut: the world moves independently, so the action log is no
longer sufficient and the system must have **observed and stored** state.

### Four episode types

**1. Work episode.** One manipulation instruction — move a tracked object between
`R1` and `R2`. Plain `pick_and_place` on the shipped `PickAndPlaceTask`; no new
class. This is also exactly the prompt distribution the VLAs were trained on.

**2. Intervention.** Not an episode. Between episodes, config-driven, the harness
moves objects between `R1` and `R2` directly (`create_mlspaces_body(...).position`
— runtime, no rebuild). **Unobserved:** the robot is not present and does not see
it happen.

*Observed vs. unobserved is a per-task decision and must not be mixed.* Unobserved
is correct here: restoration needs only memory of the state at T (which the robot
observed) and the current state (which it can perceive), so an unobserved change
breaks neither — while still killing the action-log shortcut. Observed changes
would instead test *attribution* ("X moved and it was not me"), a different and
larger claim.

**3. Explore episode.** After each intervention, the robot is asked to visit both
receptacles and look. Two `navigate_to` subtasks, reusing the shipped nav task
(`NavToObjTaskSpec`, success radius `succ_pos_threshold = 1.5` m). No
manipulation. Its purpose is to write a **fresh observed snapshot** of the
arrangement into memory.

This is what makes the memory content well-defined: memory holds a sequence of
*observed* snapshots, not an inferred world model. It also gives navigation a role
independent of manipulation, and it makes a stale-model failure distinguishable
from a recall failure.

**4. Restoration episode.** "Restore to how it was at \<episode T\>." Target must
be a **previously observed snapshot** — i.e. an explore episode — which satisfies
the observability rule by construction. Uses `ReorderTask` and the
receptacle-support predicate.

### Cadence
Repeating unit of three episodes:

    work, work, [intervention], explore, work, work, [intervention], explore, ...

with restoration episodes inserted periodically, each targeting an earlier explore
snapshot. **Fixed cadence for the whole run** — a schedule that changes mid-run
tests adaptation, which is a good second experiment and a bad first one, because
failure-to-learn and failure-to-adapt become indistinguishable.

Query depth (how many explore snapshots back the target is) remains the
history-length axis.

### Logging
Log every intervention as `(episode, object, from, to)` beside what memory
recorded at the following explore episode. That pairing is what separates a
perception failure (the explore episode missed the change) from a recall failure
(memory had it and the planner did not use it). Without it the two are
indistinguishable in the results.

### Config knobs
| Knob | Default to start | Effect |
|---|---|---|
| Objects per intervention | 1 | Divergence rate; raise if arrangements do not differ enough across snapshots to test temporal indexing |
| Intervention cadence | every 2 work episodes | Fixed for the run |
| Query depth | varied | History-length axis |
| N tracked objects | 4–6 | Chance floor 1/2^N |
| Partial-cue fraction | config | Keeps the memoryless floor above chance |

## Build log — 2026-09-09

### Environment: version mismatch, worked around without touching the shared venv
The venv at `/nobackup2/le/molmospaces/.venv` has **`molmospaces-resources 0.0.1b4`**;
this checkout's `pyproject.toml` requires **`0.0.2`**. Anything touching the resource
manager — loading a scene, importing through `molmo_spaces.evaluation` — raised
`ImportError: Please ensure molmospaces_resources is >= min(0.0.2, ...)`.
pip confirms the installed `molmo-spaces 0.2.0` itself pins `0.0.1b4`, i.e. that
venv predates this checkout.

**Not fixed by upgrading the venv** — it belongs to `/nobackup2/le/molmospaces` and
upgrading could break that checkout. Instead the correct version is installed to an
isolated directory and prepended on `PYTHONPATH`, which is contained and reversible:

    pip install --target <shim> "molmospaces-resources==0.0.2"
    PYTHONPATH=<shim>:. MLSPACES_ASSETS_DIR=... MUJOCO_GL=egl <venv>/bin/python ...

Verified: `get_scenes("ithor", "val")` returns 431 scenes per split.

**Permanent fix for the user to make:** create a venv for this checkout, or
reinstall `molmo-spaces` here so its pin matches (`pip install -e ".[mujoco,dev]"`).
`ruff` is also absent from that venv (dev extra), so new code here is syntax- and
import-verified but not lint-checked.

### Done
- `molmo_spaces/tasks/reorder_task.py` — `ReorderTask` + `SupportTracker`.
  **Note:** the support predicate needed all three of `PickAndPlaceTask`'s tiers —
  contact force, `objects_on_receptacle` geometric fallback, *and* relative-pose
  carry-forward. The third matters: a lightweight object resting at equilibrium
  loses contact forces and would otherwise flicker out of "supported", reporting a
  false negative. `get_reward()` returns fraction-restored so the attempts curve
  has resolution.
- `molmo_spaces/tasks/explore_task.py` — `ExploreTask`. Visits **latch**: the robot
  must leave one receptacle to reach the next, so an instantaneous proximity check
  would never see them all simultaneously.
- `benchmark_schema.py` — `ReorderTaskSpec`, `ExploreTaskSpec`, added to `TaskSpec`
  union and `ALL_TASK_SPEC_CLASSES`. Validated standalone (construct + round-trip).
  `EpisodeSpec.task` is a flexible dict dispatched on `task_cls`, so no sampler
  change was needed.

### Next
`tools/build_run.py` chaining harness → DummyPolicy smoke test → video demo.

### Build log continued — chaining harness

`research/cross_episode_memory/tools/build_run.py` (371 lines, two modes).

**`inspect`** lists receptacle and tracked-object candidates from a scene's
`*_physics_metadata.json` — no simulator needed. `ObjectManager.has_receptacle_site`
is defined as `name_map.sites` being non-empty, and that map is exactly what the
metadata stores, so this is authoritative rather than a category guess. On
`FloorPlan3`: 27 receptacles, 31 movable objects (Apple, Mug, Bread, Tomato, Cup,
Potato, Plate, Bowl...). Usable receptacle pair: `Cube_001_5cc5a1f7_25` (static,
6 sites) and `Side_Table_3_1_32` (7 sites).

**`build`** emits the episode chain plus a manifest (intervention log, snapshots,
final assignment). Verified cadence `W W E W W E R ...` as designed.

#### Bug found by verification: trivially-solved restorations
The first generated run had **1 restoration in 3 requiring zero moves** — random
drift over two receptacles had returned the arrangement to the target state, and
cue objects shrank the gap further. Such an episode is already solved on arrival,
so *a memoryless baseline scores 100% on it* and the reported success rate is
silently inflated.

Fixed: `min_objects_to_move` (default 2). Restoration target selection now scores
every candidate snapshot by how many moves it needs **after cues are applied**,
keeps only those meeting the minimum, and among those picks the one nearest
`query_depth` (preserving query depth as the history-length axis). If no snapshot
qualifies, the restoration is **skipped and recorded** in
`manifest.skipped_restorations` with the reason — never silently dropped.

#### Divergence rate, measured
16 work episodes, `min_objects_to_move=2`, seed 7:

| tracked objects | objects/intervention | restorations skipped |
|---|---|---|
| 4 | 1 | 1 |
| 6 | 1 | 1 |
| **6** | **2** | **0** |

**Intervention rate dominates object count.** Going 4→6 objects changed nothing;
moving 2 objects per intervention instead of 1 eliminated all skips. Default
`objects_per_intervention` to 2. The spec predicted this qualitatively ("too slow
and arrangements do not differ enough"); these are the numbers.

#### Next
DummyPolicy smoke test (assert episode N+1 starts where N ended), then the video.

### Build log — smoke test against the simulator

All three episode types now run end-to-end under `DummyPolicy` on `FloorPlan3`.

| episode kind | task class | result | correct? |
|---|---|---|---|
| work | `PickAndPlaceTask` | 0/1 | yes — a do-nothing policy must fail |
| explore | `ExploreTask` | 0/1 | yes, **after a fix** (was 1/1) |
| restoration | `ReorderTask` | 0/1 | yes |

#### Five integration facts, each found by a failed run

1. **Registering a task takes three layers, not one.** A `TaskSpec` in
   `benchmark_schema.py` is not enough. Each task also needs a `*TaskConfig` in
   `configs/task_configs.py` (added to the `AllTaskConfigs` alias) and entries in
   **both** `TASK_CLASS_TO_CONFIG_CLASS` and `TASK_CLASS_TO_SPEC_CLASS` in
   `json_eval_task_sampler.py`. Missing the config mapping raises
   `Unknown task class '...' - no mapping to task config class`.

2. **EpisodeSpec fields take MuJoCo *body* names, not `object_id`.** `Bread_3_14`
   raises `KeyError: Invalid name`; the body is
   `bread_02590a72dc962788d308c804909b928a_1_0_0`. The metadata's `hash_name` is
   the body name. `build_run.py inspect` now prints `body=` and warns about this.

3. **`eval_main` requires one `task_horizon_sec` across the whole benchmark dir.**
   Mixed values raise `inconsistent task_horizon_sec`. Since explore (short nav)
   and restoration (several placements) want very different budgets, **a run must
   be driven one episode per invocation** — which the design needs anyway, because
   interventions are applied *between* episodes.

4. **`ithor` splits by index**: train 1–12, val 13–24, test 25–30. `FloorPlan3` is
   **train**; asking for it in `val` raises `No scene file for split 'val' index 3`.

5. **Videos are automatic.** `prepare_episode_for_saving` writes one mp4 per camera
   (`episode_*_<camera>_*.mp4`) whenever `--output_dir` is set. No flag needed.

#### Second silent-success bug: trivially-satisfiable explore
`ExploreTask` **passed with `DummyPolicy`** — a policy that never moves. Cause: the
first receptacle pair was 1.25 m apart while the explore radius is 1.5 m, so the
robot observed both from spawn. The chosen `robot_base_pose` was also (0,0), which
is exactly where that receptacle sits — the robot spawned inside the furniture.

This is the same failure class as the trivial restoration: **no exception, just a
meaningless 100%**. Added `build_run.py validate --scene-xml`, which compiles the
scene and checks that every referenced body exists, that receptacle separation
exceeds `2 x explore_radius`, and that the robot base is not inside a receptacle.
`build` accepts `--scene-xml` and refuses to build a config that fails.

Re-run with a validated pair (oven <-> fridge, **5.17 m apart**): explore correctly
reports 0/1.

**`FloorPlan3` has only 4 static above-floor receptacles**, and just three pairs
exceed 3 m: oven-fridge (5.17), fridge-drawer (4.09, 3.52). Scene choice is more
constrained than expected — worth checking separation before adopting any kitchen.

### Build log — video demo delivered

`research/cross_episode_memory/demo/run_demo.mp4` — a four-episode run
(work, work, explore, restoration) on `FloorPlan3` under `DummyPolicy`, stitched by
`tools/compose_run_video.py`.

Per-episode mp4s are the wrong unit for this study: the interesting thing is how
the arrangement drifts *across* episodes. Each clip is therefore preceded by a
title card naming the episode kind, the instruction, and the object->receptacle
assignment it starts from. Read in sequence the cards show bread moving
oven -> refrigerator as a byproduct of episode 0's work, further drift by the
explore snapshot, and the restoration starting exactly 2 objects away from its
target — matching `min_objects_to_move=2`.

#### Third performance bug: the geometric fallback stalls episodes
The restoration episode **hung for over four minutes inside a 12-second-horizon
rollout**. Two causes, fixed in order:

1. `judge_success`, `get_reward` and `get_info` each independently recomputed the
   full per-object status every step — 3N checks where N would do. Fixed by
   memoising on `(batch_index, episode_step_count)`.
2. That was not enough. The real cost is **tier 3**: `objects_on_receptacle` runs
   shapely polygon tests over *every geom of the receptacle*, and a fridge has
   hundreds. The original tier order ran it before the cheap carry-forward check.

Fixed by reordering the tiers to contact -> carry-forward -> geometric, and
throttling the geometric tier to every `fallback_every` steps (default 10), with
`use_geometric_fallback` to disable it entirely. Ordering matters more than the
throttle: once support has been observed once, carry-forward answers almost every
subsequent step for free.

**Watch for this when scaling up.** Cost grows with tracked objects x receptacle
geom count, so a run with more objects or a geometry-heavy receptacle can stall
without any error. If episodes start taking minutes, this is the first thing to
check.

#### Camera framing: still to improve
The demo uses the robot-mounted camera, which frames a close-up of the gripper —
fine for the policy, poor for a human watching a run. An exocentric
`demo_overview` camera (`/tmp/cfg_demo.json`, positioned to see both oven and
fridge) is configured and validated but its render had not finished at time of
writing. Re-run `build_run.py build --config cfg_demo.json` and re-stitch for a
legible overview shot; the policy cameras should stay as they are.

#### Also fixed
`build_run.py write_run` now emits `house_<idx>/episode_*.json`, the layout
`load_benchmark` globs for. An `episodes/` subdirectory silently yields zero
episodes — the run completes and reports 0/0 rather than erroring.

#### Demo camera placement: two failed attempts, and why
Getting a *legible* demo view is harder than it looks, and both naive choices fail:

1. **Robot-mounted at `camera_offset [0.1, 0, 0.9]`, `lookat [0.6, 0, 0.4]`** — frames
   a close-up of the gripper. The offsets are relative to `robot_0/base`, and the
   arm is mounted on that base, so looking forward-and-slightly-up from just above
   the base points straight at the arm.
2. **Exocentric at the receptacle midpoint + `[4.2, 0, 2.6]`** — lands *outside the
   kitchen*, showing a blank exterior wall and a doorway. Room bounds were never
   checked.

There is also a geometric limit worth knowing: with the receptacles 5.17 m apart,
**no single interior camera at normal ceiling height frames both well**. A demo
either follows the robot, or uses one camera per receptacle, or accepts seeing one
region at a time.

Current approach: a robot-mounted **chase** camera (`[-1.6, 0, 1.9]` looking to
`[1.4, 0, 0.2]`, 85 deg) — behind and above, looking past the arm rather than at
it. Note the policy cameras must stay exactly as the checkpoint expects; a demo
camera is an *additional* entry in `cameras`, never a replacement.

#### Root cause of the bad camera views: an invalid robot spawn
Three camera placements failed in a row. The cause was not camera geometry — it was
the **robot spawn position**. `robot_base_pose` was set to the midpoint of the two
receptacles, which in `FloorPlan3` sits **0.35 m from a stool and 0.47 m from the
counter**. The robot spawns wedged among furniture, so *any* base-mounted camera is
pressed into geometry, and the work episodes are likely unrealistic too.

The validator missed it because it only checked distance from the two *named*
receptacles. Fixed: `validate_run` now checks clearance against **every** geom
between 0.05 m and 2.0 m height, defaulting to a 0.6 m minimum, and names the three
nearest offenders. Re-validating the config the demo was built from now fails
correctly.

**This invalidates the demo's visuals, not its logic.** The episode structure,
chaining, drift and restoration targeting are all correct and visible in the title
cards. The camera views are unusable, and the fix is to choose a spawn with real
clearance rather than to move the camera.

**Next step for a clean demo:** pick a `robot_base_pose` in free space — `ithor`
ships an occupancy map per scene (`FloorPlan3_physics_map.png`, part of the
`_with_occupancy` asset version), which is the right source — then re-run
`build_run.py build --scene-xml ...` and re-stitch. Do not hand-pick a midpoint.

### Build log — RB-Y1 (spec embodiment) and the camera-orientation bug

#### Wrong robot: `DummyBenchmarkEvalConfig` has no mobile base
The first working demo ran on `DummyBenchmarkEvalConfig`, chosen because it is the
shipped no-op eval config. It hardcodes **`FrankaRobotConfig`**, whose
`command_mode` is `{arm, gripper}` — **no base**. Every episode therefore ran on a
robot that physically could not drive between receptacles 5.17 m apart, nor satisfy
an explore episode. The structure was right; the embodiment made it unperformable.
The 0/1 results were correct for the wrong reason.

**Never use `DummyBenchmarkEvalConfig` for this study.** Use
`research/cross_episode_memory/eval_configs.py:RBY1ReorderEvalConfig`, which
carries `RBY1MConfig` (`"base": "holo_joint_planar_position"`, seven move groups).

**RB-Y1 was chosen over TidyBot++ and XLeRobot** because it is the only spec
embodiment already integrated in MolmoSpaces — robot class, view, kinematics all
exist — so it needs no new integration. It is also the spec's platform-credible
loco-manipulation robot, and it ships six MJCF cameras.

Verified: all four episode kinds (work, work, explore, restoration) run on RB-Y1
and report 0/1 under `DummyPolicy`, which is correct.

#### The camera bug: identity quaternion, not placement
**Four camera placements failed in a row** — robot-mounted close-up, exocentric
outside the room, chase through a wall, head-mount inside the robot's own body. All
four shared one cause: `camera_quaternion: [1, 0, 0, 0]`.

**MuJoCo cameras look down −Z.** An identity quaternion therefore points the camera
into the robot no matter where it is placed. RB-Y1's real `head_camera` sits at
`[0.05, 0, 0.05]` — essentially where it had been guessed — but carries
`[0.5, 0.5, -0.5, -0.5]` to face forward. Hours went into moving a camera to fix an
orientation bug.

**Read mounts out of the MJCF instead of guessing.** RB-Y1 ships purpose-built
views; `camera_follower` is parent `robot_0/base`, pos `[-1.3, 0, 2.7]`, quat
`[0.6533, 0.2706, -0.2706, -0.6533]`, fovy 45. Used verbatim it produces a correct
third-person demo view first try.

    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "robot_0/camera_follower")
    model.cam_pos[cid], model.cam_quat[cid], model.cam_fovy[cid]

#### Limitation worth knowing for B1/B2
`EpisodeSpec.cameras` accepts only `RobotMountedCameraSpec` and
`ExocentricCameraSpec` — there is no MJCF-camera spec — and at
`json_eval_task_sampler.py:250` the episode spec's cameras **replace**
`exp_config.camera_config` wholesale. So RB-Y1's six MJCF cameras cannot be
referenced from a benchmark JSON; they must be re-declared as robot-mounted specs
with the pose copied from the model. This matters when wiring B1/B2's real policy
cameras, which must match `RBY1GoProD455CameraSystem` exactly.

### Finding: spawn selection — use the shipped LinearBot recipe

Four home-grown spawn selectors failed (distance-to-geom, world-space AABB, A*
plannability, physics-probe navgrid). The working recipe already exists in
`molmo_spaces/policy/linearbot_policy/planner/mobile_manip_task_sampler.py::_place_robot`:

1. Candidates come from `env.get_thormap(agent_radius=R).get_free_points()` —
   **not** from `AStarPlanner.graph` nodes. The map's free points already exclude
   outdoors, so the "robot drove outside the house" bug is impossible by construction.
2. Each candidate is verified with `env.check_if_robot_collision_at_base_pose`,
   which ignores floor contacts, self-contacts (compared on *root* body names), and
   any contact with `dist > 0` (touching but not penetrating).
3. **Yaw is re-rolled per candidate.** A map-free point clears obstacles only by
   `agent_radius` (~0.35 m), but RB-Y1's footprint is wider, so whether the pose is
   clear depends on heading. My earlier search fixed `theta=0` and concluded no
   in-graph cell was clean — that conclusion was an artifact of not sampling yaw.
4. If nothing verifies clear, it falls back to a map-free point with a warning
   rather than crashing.

Corollary already recorded above: `get_discrete_location` *snaps* to the nearest
navigable cell, so a non-`None` return means "there is floor near here", not "this
pose is free". Never treat it as a clearance test.

### Finding: spawn must satisfy FOUR conditions, and two planner API gotchas

A spawn is only usable if it is simultaneously:
1. **map-free** — from `iTHORMap.get_free_points()` (excludes outdoors by construction);
2. **in the planner graph** — else `motion_plan` raises `Non-plannable starting
   position` and every waypoint fails. A probe-clean pose is NOT enough: the spawn
   `(-3.090, 0.519)` was collision-free but outside the graph and beyond the 40-cell
   (~1 m) `max_start_goal_distance` snap radius, so task 0 logged `no route` for
   every leg;
3. **probe-clean** — see the contact-height finding below;
4. **connected** to every target via `nx.has_path` — navigable but stranded is a real
   failure mode (LinearBot guards it with `_reachable_free_points`).

Reachability must be tested against receptacle **standoffs**, not centres: a centre
sits inside furniture and `get_discrete_location` returns `None` for it.

Two API gotchas, both of which cost a run:
- **`AStarPlanner._compute_plan` already returns WORLD waypoints** (it applies
  `pos_px_to_m(waypoints * downscale)[:, :2]` internally). Converting its output from
  "grid cells" to world a second time yields garbage coordinates.
- **`get_discrete_location` has an inconsistent return type**: a numpy *array* when
  the cell is already in the graph, a *tuple* when it snapped via `find_close()`.
  Graph nodes are tuples, so the array form raises `NodeNotFound`. Normalise it.

### Finding: the collision probe must filter contacts by HEIGHT, not by body name

RB-Y1's base joints are planar (x/y/theta), so nothing lifts the base off the floor
plane -- it rests at z=0.01 and is permanently in contact with the floor surface.
That surface is split across `floor_*`, `decals_*` and `mesh_*` bodies, so a
name-based `"floor" in name` filter (as in
`BaseMujocoEnv.check_robot_collision_in_current_pose`) misses two of the three and
reports ~510 phantom collisions at EVERY pose -- which made all 200 map-free
candidates look occupied. LinearBot avoids this by lifting the robot:
`robot_z = _detect_floor_z(...) + 0.15`.

Filter by contact height instead. Measured on FloorPlan3: on open floor robot-scene
contacts span z=0.032..0.220; at a pose buried in the oven they span z=0.032..1.359
and involve `oven_*`/`cube_*`. `FLOOR_CLEARANCE = 0.25` sits in that gap. Verified:
oven centre 2076 contacts, fridge centre 1866, and 60/60 map-free points clean in
7.3 s.

Do NOT derive that threshold from `model.geom_pos` -- it is BODY-LOCAL, not world.
Doing so filtered out every contact and made even the oven interior read as clear.

Limitation: the probe cannot detect "outside the house" -- there is no geometry out
there to touch, so it returns 0. That is acceptable only because candidates come
from the occupancy map, which excludes outdoors.

### Finding: articulation ramps must cache the rest pose, not re-read it

`_door_joint` derives the door's `closed` value from the LIVE joint qpos and picks the
open target as whichever joint limit is further away:
`opened = hi if abs(hi - closed) > abs(closed - lo) else lo`.

Calling that every step of a ramp is self-corrupting. Observed on the fridge
(`..._1_2_0_joint_0`, range [-1.571, 0], rest -0.569) within a single 15-step open:

    closed=-0.569 open=-1.571   (step 1)
    closed=-0.589 open=-1.571
    closed=-0.757 open=-1.571
    closed=-0.924 open= 0.000   <-- open target INVERTED

Once the drifting `closed` crossed the range midpoint (-0.786), the "further limit"
test flipped and the door reversed direction mid-swing. Even without the inversion,
the closing ramp ends at the drifted baseline rather than the true rest pose.

Fix: resolve `(adr, closed, opened)` ONCE per receptacle and cache it for the whole
open/place/close sequence.

Note the rest pose is not necessarily zero -- this fridge door sits ~33 deg ajar at
rest, so "closed" means "back to its initial pose", not qpos 0.

Aside: `_door_joint` breaks ties on range width by first-seen. The fridge has two
joints of identical 1.571 width (`..._1_2_0_joint_0` and `..._1_4_0_Mesh9a740b141_1`),
so which panel swings is arbitrary. Tighten this if the wrong panel is animated.

### Finding: place on receptacle SITES, not the subtree-AABB top

iTHOR receptacles carry explicit `*Receptacle*` sites marking where objects belong.
Placing at the subtree-AABB top instead puts objects on the appliance ROOF:

| receptacle   | sites | site z range   | AABB-top placement |
|--------------|-------|----------------|--------------------|
| refrigerator | 16    | 0.637 .. 1.209 | z = 2.528 (roof)   |
| oven         | 3     | 0.434 .. 1.139 | (same failure)     |

The first end-to-end task-0 run placed bread at z=2.528 on a fridge whose interior
shelves sit at 0.637..1.209 -- so the open/close sequence was theatre, and the task's
success predicate (object on an interior shelf) correctly scored 0.

`placement.resting_position` now prefers a site, spreading across sites by distance to
already-placed objects, and only falls back to the AABB top when a receptacle has no
sites. Verified: bread -> fridge z=0.719, bread -> oven z=0.516.

Note the fridge's 16 sites include door-bin shelves (`fridgefreezerdoor...Receptacle*`
at z=0.637/0.755) as well as interior drawers (`fridgedrawer...` at z=1.209). Placing
in a DOOR bin while the door is swinging may need ordering care.

### Finding: `place_receptacle_start_pose` must be the RECEPTACLE's pose (generator bug)

`PickAndPlaceTask` success requires FOUR conditions (pick_and_place_task.py:213):
`supported_by_receptacle AND not robot_contact AND pos_displacement <= max AND
tilt_displacement <= max`. The displacement terms compare the receptacle's LIVE pose
against `task_config.place_receptacle_start_pose` -- they exist to reject runs where
the robot shoved the furniture.

`build_run.py` wrote `cfg.placements[f"{obj}|{dst}"]` into that field -- the pose where
the OBJECT rests on the receptacle, not the receptacle's own pose:

| receptacle | written                          | actual body pose                          |
|------------|----------------------------------|-------------------------------------------|
| fridge     | [0.953, 1.917, 1.5697] identity  | [1.013, 1.917, 1.2197] [-.5,-.5,.5,.5]    |
| oven       | [-0.627, -3.046, 0.773] identity | [-0.447, -3.046, 0.2227] [0,0,.707,.707]  |

0.355 m and ~120 deg off, so both displacement terms blow their thresholds AT STEP 0
and the episode can never succeed regardless of behaviour. This affected 16 of 28
episodes (every work episode; explore/restoration do not use the field).

Diagnosis required in-episode instrumentation -- offline the placement satisfied all
four conditions, so the disagreement was only visible from inside the run. The probe
showed `supported=True robot_contact=False obj_contacts=18-19` steadily, which left
displacement as the only candidate.

Fixed: `RunConfig.receptacle_poses` populated by `measure_body_poses()` from the
compiled scene, read via `receptacle_start_pose()`, which RAISES when a pose is
missing rather than falling back -- a silently wrong value here costs a full run to
detect.

Note `fail` in the output H5 is just `~success` (save_utils.py:684), NOT an
independent failure signal. "fail is True at step 0" means only that success was
never True.

### Verified physical component check — 2026-09-09

The previous appendix explanation that installed `nvidia-curobo 1.0` is an
unrelated package is incorrect. It is NVIDIA cuRobo with the newer MotionPlanner
API, incompatible with this checkout's older MotionGen imports. The new
`curobo_current.py` converts the pinned RB-Y1 YAML in memory and successfully
plans physical arm motion with the installed package.

`tools/check_transfer.py` now supplies a small oracle A-to-B check before kitchen
integration. The final recorded run physically grasped a block with both fingers,
lifted it 18.8 cm, carried it to B, released it, and withdrew. The object retained
support contact with B after settling; maximum endpoint position error was
0.74 mm. Robot and object qpos are only written during initialization. Payload
attachment is cuRobo collision geometry only, not a simulator weld. See
[CHECK_TRANSFER.md](CHECK_TRANSFER.md) for the video, measured report, reproduction
commands, and remaining integration work.

The shipped RB-Y1 XML finger actuators are force motors, but the existing robot
controller sends position commands. `configure_rby1_gripper_servos`, called by
`RBY1.apply_control_overrides`, now converts those motors to bounded position
servos in memory before optional RobotConfig overrides. Physical open/close/reopen
regressions pass for both hands. The pinned asset files are unchanged.

This validates isolated arm/gripper manipulation only. It does not validate the
continuous fridge-open → table-pick → fridge-place → fridge-close sequence, the
RunTask integration, or the episodic-memory experiment. The old demo policy still
teleports objects and remains a visualization rather than physical validation.

### Verified physical fridge-door component — 2026-09-09

`tools/check_fridge_door.py` now physically opens, releases, withdraws, regrasps,
closes, and withdraws from the actual FloorPlan3 `Fridge_3` handle. It extracts
that fridge into an isolated scene, preserving its physical asset parameters and
scene solver settings. The verified run held the released door at 59.46° and
finished 0.15° from closed; maximum endpoint TCP error was 3.71 mm. Execution uses
arm/gripper actuator controls only. The video, exact scope, sampled contact gaps,
collision-model limitations, and reproduction command are documented in
[CHECK_FRIDGE_DOOR.md](CHECK_FRIDGE_DOOR.md).

Two integration findings matter for the full task: a reachable grasp does not
imply reachable withdrawal at the door's open wrist orientation, and a moving
door panel cannot be treated as a frozen obstacle during hinge tracking. The check
rotates the released wrist during withdrawal; it retains the cabinet/other door
in cuRobo's collision world and monitors actual robot/fridge penetration every
physics step. Its conservative exterior part bounds must be replaced with finer
geometry before planning placement on interior shelves.

This completes the isolated door component. Actual-object shelf placement,
navigation while carrying, and the continuous first reorder instruction still
require physical validation.


### Actual-bread shelf component, explicit contact model — 2026-09-09

`tools/check_fridge_transfer.py` now has a passing isolated table-to-open-fridge
simulation using the original FloorPlan3 bread and refrigerator meshes. The run
physically lifted the loaf about 15 cm, rotated it through cuRobo waypoints,
placed it on the middle-right interior shelf, released, and withdrew. Both
fingers were in contact in all 675 sampled carry frames; maximum payload
translation slip was 0.86 mm, maximum endpoint TCP error 4.46 mm, and unintended
robot/environment penetration zero. Final shelf support, containment, release,
and settling checks passed. See [CHECK_FRIDGE_TRANSFER.md](CHECK_FRIDGE_TRANSFER.md)
for the video, report, reproduction command, and failed-attempt evidence.

This result depends on explicit simulation assumptions: a 100 N gripper actuator
cap, 2500 N/m position gain, and an opt-in finite-pad contact model (`condim=6`,
priority 1, friction `[1, .005, .002]`). These values are not calibrated to hardware;
the original point-contact configuration failed the transfer. The shared robot
helper defaults and asset files were not changed to force this result. The
configuration is recorded in the video and report and must remain visible when
using this component in later experiments.

The scene uses a fixed base stance held through its site-transmission actuator,
a small test table, and a door initialized at 90 degrees. Exact fridge collision
triangles expose its interior to cuRobo; conservative payload spheres represent
the held loaf only in planning. MuJoCo execution uses actuator controls, without
object teleports or welds. Navigation while carrying, reconciliation with the
60-degree door check, and the continuous first reorder instruction still require
validation. This is an oracle component result, not a memory-experiment result.


### Navigation with the actual loaf, separated table — 2026-09-09

`tools/check_navigation_transfer.py` passed a continuous drive-to-table, physical
pickup, loaded return, shelf-place/release/withdraw component test. The table is
shifted 2 m from the earlier layout and the base stances are 2 m apart. The robot
travelled 3.006 m on each leg around the open fridge door. Both fingers contacted
the loaf in all 957 recorded loaded-navigation frames; maximum sampled grip
translation slip during driving was 0.059 mm. Shelf support/containment/release
passed after arrival. See [CHECK_NAVIGATION_TRANSFER.md](CHECK_NAVIGATION_TRANSFER.md)
for video, measurements, reproduction and scope.

A* uses MuJoCo probes in a separate data copy, with the loaf included during
loaded navigation. Execution uses smooth base position-actuator targets and
cuRobo arm plans; it does not teleport execution state. This validates
fixed-heading holonomic translations in the small separated-receptacle scene,
not wheel-level control or cluttered-kitchen navigation. The test retains the
explicit, uncalibrated 100 N finite-pad grasp configuration. The door starts open
at 90 degrees; continuous opening/closing and RunTask integration remain pending.


### Heading-aware navigation and head-camera recording — 2026-09-09

The separated-table check now passes with turn-in-place/forward-drive primitives
and an explicitly approved 30 cm reverse table departure. The previous fixed-yaw
sideways route is retained as a historical result, not forward-view validation.
The current SE(2) A* checks swept robot/payload poses; execution still uses physical
base actuators and cuRobo arm trajectories without teleports or welds.

In the verified run, empty/loaded travel was 2.4545 /
2.7148 m, with four turns per leg. Actual forward travel
stayed within 0.0180 degrees of the head camera's
horizontal optical direction above 2 cm/s. All 2286 recorded loaded-navigation
frames had bilateral finger contact, maximum payload translation slip was
0.703 mm, and final supported shelf placement/release passed. Navigation
collision metrics were zero; maximum unintended robot/environment penetration
over the entire run was 0.155 mm, below the 3 mm failure threshold. A faster
0.25 rad/s mean-turn attempt lost the loaf; reducing mean turn speed to
0.08 rad/s and doubling arm trajectory duration passed without increasing the existing uncalibrated 100 N finite-pad grip assumptions.

The video includes the actual head camera plus a separate raw RGB recording.
Head yaw follows the base and pitch is fixed 0.5 rad downward. The camera faces
travel, but loses the loaf near the table; active manipulation gaze remains
pending. This remains an oracle scene/state controller, not a head-camera-only
VLA test. The supplied cuRobo arm collision model excludes head joints; actual
head geometry is present in MuJoCo navigation probes. The fridge starts open
at 90 degrees, with continuous door operation and RunTask integration pending.
See [CHECK_NAVIGATION_TRANSFER.md](CHECK_NAVIGATION_TRANSFER.md) for evidence,
reproduction and scope.


### Fresh native iTHOR historical reorder execution — 2026-09-13

`artifacts/reorder_right_bidirectional_v6` passed a continuous, fresh FloorPlan3
RB-Y1 run with native annotated egg and potato, sink counter/right fridge only,
and all authored articulations initially closed. Five physical transfers included
both directions during ordinary work. Two harness changes were each validated in
an independent physical clone before live mutation, followed by a live revisit.
The final transfer restored the first post-change snapshot (egg in fridge, potato
on counter), with no inspection inserted into a transfer. Seven right-door cycles
finished closed; the left remained closed. All 18 `history_audit.json` checks pass.

The complete three-view and head-camera videos were rendered after physics,
at 25 FPS and 5× playback, each 465.08 seconds long. Component and failed attempts
produced no videos. Runtime fixes include actual-mesh empty-arm path checks,
torso-frame gaze targets, upright retrieval rolls and joint-limit clearance,
validated compact carry/shorter undocking, adaptive closing approaches, measured
stop pushes, and contact-servo bias correction. The contact-selection optimization
preserved checks on 90 recorded native states while reducing measured loop time
by 34%; the focused regression suite has 57 passing tests.

This is a single seed-0 oracle-controller execution result. Its finite-pad grasp
parameters are not calibrated to hardware, and manipulation gaze target coverage
is 61.6%, so it does not establish continuous visual observability or camera-only
VLA execution. Full evidence and previous failures are in
[CHECK_REORDER_CHAIN.md](CHECK_REORDER_CHAIN.md).


### Shared access between consecutive locomanip transfers — 2026-09-13

The user removed the per-transfer close/reopen requirement. `run_history` now
batches each work pair, and restoration batches its required transfers. Both
shelf and counter placement retain their empty-arm fold but defer closure until
the final transfer in the batch. Later transfers reuse the open right door and
its recorded opening trajectory. A batch starts and ends closed; a failed
transfer aborts without trying extra door motion. Dynamic changes and revisits
are forbidden even between transfers inside a batch.

Feasibility clones use the same batching, and reports/audits check individual
door actions plus the shared batch boundaries. The default history now requires
five complete door cycles (two work pairs, two revisits, one restoration), saving
two redundant close/reopen operations. All 64 focused tests pass. A full physical
run of this changed policy has not yet been performed; the v6 videos document
the previous seven-cycle execution and have not been replaced.


### Selected-object labels in execution logs — 2026-09-13

The bread/loaf wording in inherited transfer stages was a display artifact.
`select_object` selects the current body, collision geometry, joint, and grasp
annotations; the planning pose reads that body's state. Terminal labels, errors,
and video captions now use the selected object name, and terminal records include
the full object ID. Report `stage_label` provides the display text while internal
stage IDs remain stable for contact-path recovery and saved traces. The binding
and formatting check in `artifacts/object_logging_check/label_check.json` covers
both egg and potato, and all 64 focused regressions still pass. This change does
not reload code into processes already running or rerender existing videos.


### User-run comparison and counter-grasp execution fix — 2026-09-13

`reorder_chain_20260913_122737` matches v6's runtime arguments except output,
uses MuJoCo 3.5.0/cuRobo 1.0, and reproduces its initial state, first door opening
and egg annotation 586. The updated shared-door sequence completed the egg
transfer, then failed planning potato approach 2/3 with annotation 334. The new
arrival history changes planning after the first placement; it was not a different
Python environment or an initial egg-grasp failure.

Counter annotation selection checked a local IK descent but cached only pregrasp,
then execution replanned each descent segment. The task now caches and executes
the dense, actual-mesh-checked contact approach on both supports, with a 3 mm
tracking limit and rejection of missing cached segments before motion. Lift-off
from either support uses the existing actual-mesh contact servo because the
coarse cuRobo model can reject the intentional grasp state. Free-space motion
continues to use cuRobo. Actual geometry, force and penetration checks remain.

The new `tools/check_counter_grasp_boundary.py` tests saved counter arrivals
without replaying navigation or rendering video. `shared_door_egg_grasp_v2` and
`shared_door_potato_grasp_v3` passed physical grasp, 119.7–119.8 mm lift and a
three-second bilateral force hold. Both recorded zero unintended, self and
finger/object penetration and no hold-contact loss. Potato used the originally
failing annotation 334. The preceding restricted-annotation diagnostic rejected
a self-colliding alternate pregrasp branch, and the first complete-library test
exposed the lift-planning failure; neither produced video. All 66 focused tests
pass, including contact tracking and rejection of uncached replanning during
descent. The complete corrected history has not yet passed a fresh run.


### Door interaction versus transfer payload in logs — 2026-09-13

During door access, the persistent payload remains egg or potato; the arm and
manipulation gaze use the door handle. Console records now distinguish
`task_object`/`task_object_id` from `interaction_target`, and video captions show
both target and payload. The legacy report `object` field still names the payload.
This is a reporting change only. AST comparison against the tested grasp-fix
snapshot found differences only in reporting methods; the active launcher and
Python environment were also verified. Evidence is stored in
`artifacts/object_logging_check/interaction_context_check.json`. No process was
interrupted, and already-running processes retain their loaded labels.
