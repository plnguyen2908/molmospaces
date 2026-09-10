#!/usr/bin/env python3
"""Author a cross-episode *run* as a chain of EpisodeSpec JSONs.

A run is a sequence of episodes in ONE house sharing ONE frozen camera set, in
which a set of tracked objects moves between exactly two receptacles. See
``research/cross_episode_memory/SPEC.md`` -- "Task 2 concrete design" and
"Task 2 run structure".

Episode cadence (repeating)::

    work, work, [intervention], explore, ...

with restoration episodes inserted periodically, each targeting an earlier
*explore* snapshot.

- **work**       one pick_and_place: move a tracked object to the other receptacle.
- **intervention** not an episode. The harness moves objects itself, unobserved,
                 and records it in the manifest.
- **explore**    visit both receptacles and look. Writes an *observed* snapshot.
- **restoration** put every tracked object back on the receptacle it occupied at
                 the target explore episode.

Chaining is **categorical**: what carries from one episode to the next is the
object -> receptacle assignment, not exact poses. Poses are re-canonicalised each
episode from a per-(object, receptacle) placement table. That avoids pose drift
accumulating over a long run and stops an object inheriting a half-off-the-edge
pose that breaks the next episode. It also matches what memory stores.

Two modes::

    python build_run.py inspect --metadata <scene>_physics_metadata.json
    python build_run.py build   --config run_config.json --out <dir>
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REORDER_CLS = "molmo_spaces.tasks.reorder_task.ReorderTask"
EXPLORE_CLS = "molmo_spaces.tasks.explore_task.ExploreTask"
PICK_PLACE_CLS = "molmo_spaces.tasks.pick_and_place_task.PickAndPlaceTask"


# ----------------------------------------------------------------------------
# Scene inspection
# ----------------------------------------------------------------------------


def inspect_scene(metadata_path: str | Path) -> dict[str, list[dict[str, Any]]]:
    """List receptacle and tracked-object candidates from a scene's metadata.

    Authoritative without loading the sim: ``ObjectManager.has_receptacle_site``
    is defined as ``name_map.sites`` being non-empty, and that map is exactly what
    the metadata JSON stores. Movability comes from ``is_static``.
    """
    objects = json.loads(Path(metadata_path).read_text())["objects"]

    receptacles, movables = [], []
    for meta in objects.values():
        sites = (meta.get("name_map") or {}).get("sites") or {}
        entry = {
            "object_id": meta.get("object_id"),
            "body": meta.get("hash_name"),
            "category": meta.get("category"),
            "is_static": bool(meta.get("is_static")),
            "n_sites": len(sites),
        }
        if sites:
            receptacles.append(entry)
        if not meta.get("is_static"):
            movables.append(entry)

    receptacles.sort(key=lambda e: -e["n_sites"])
    movables.sort(key=lambda e: str(e["object_id"]))
    return {"receptacles": receptacles, "movable": movables}


# ----------------------------------------------------------------------------
# Run configuration
# ----------------------------------------------------------------------------


@dataclass
class RunConfig:
    """Everything needed to author one run. Serialise this beside the episodes."""

    house_index: int
    scene_dataset: str
    data_split: str
    receptacles: list[str]              # exactly two
    tracked_objects: list[str]          # 4-6 recommended; chance floor is 1/2^N
    # (object, receptacle) -> 7-dim canonical placement pose in world frame.
    # Keyed "object|receptacle" for JSON friendliness.
    placements: dict[str, list[float]]
    robot_name: str
    robot_init_qpos: dict[str, list[float]]
    robot_base_pose: list[float]
    img_resolution: tuple[int, int]
    cameras: list[dict[str, Any]]       # frozen: identical in every episode
    initial_assignment: dict[str, str]  # object -> receptacle at episode 0
    # Receptacle body name -> its ACTUAL 7-dim world pose (pos + wxyz quat), measured
    # from the compiled scene. `PickAndPlaceTask` compares the receptacle's live pose
    # against this to reject episodes where the robot shoved the furniture, so it must
    # be the body's real pose. Writing an object resting-pose here (which is what an
    # earlier version did) makes pos/tilt displacement non-zero at step 0 and the
    # episode can never succeed -- measured 0.355 m and ~120 deg off for the fridge.
    receptacle_poses: dict[str, list[float]] = field(default_factory=dict)

    n_work_episodes: int = 12
    work_per_cycle: int = 2             # work episodes between interventions
    objects_per_intervention: int = 1
    restoration_every: int = 2          # insert a restoration after this many cycles
    query_depth: int = 2                # preferred snapshot depth to target
    cue_fraction: float = 0.0           # objects left in place, so the memoryless
                                        # baseline can infer rather than guess blind
    # Per-kind episode budgets, in simulated seconds. Different kinds need very
    # different time: an explore is a short two-waypoint navigation, a restoration
    # is several pick-and-places. eval_main requires this or an explicit override.
    horizon_sec_work: float = 60.0
    horizon_sec_explore: float = 45.0
    horizon_sec_restoration: float = 240.0
    min_objects_to_move: int = 2        # a restoration must require at least this
                                        # many moves *after* cues are placed --
                                        # otherwise the episode is already solved
                                        # and a memoryless baseline scores 100%
    seed: int = 0

    def receptacle_start_pose(self, receptacle: str) -> list[float]:
        """The receptacle's own world pose, for the task's displacement check."""
        pose = self.receptacle_poses.get(receptacle)
        if pose is None:
            raise KeyError(
                f"no measured pose for receptacle {receptacle!r}; run "
                f"`build_run.py measure-receptacles` and add `receptacle_poses` to the "
                f"config. Falling back to a placement pose silently breaks scoring."
            )
        return list(pose)

    def __post_init__(self) -> None:
        if len(self.receptacles) != 2:
            raise ValueError(f"need exactly two receptacles, got {self.receptacles}")
        if len(self.tracked_objects) < 2:
            raise ValueError("need at least two tracked objects")
        missing = [
            f"{o}|{r}"
            for o in self.tracked_objects
            for r in self.receptacles
            if f"{o}|{r}" not in self.placements
        ]
        if missing:
            raise ValueError(
                "placements table is missing canonical poses for: "
                + ", ".join(missing)
                + ". Every (object, receptacle) pair needs one, because chaining "
                "re-canonicalises poses each episode."
            )
        unknown = set(self.initial_assignment) - set(self.tracked_objects)
        if unknown:
            raise ValueError(f"initial_assignment references untracked objects: {sorted(unknown)}")
        for obj in self.tracked_objects:
            if self.initial_assignment.get(obj) not in self.receptacles:
                raise ValueError(f"object {obj!r} has no valid initial receptacle")

    def other(self, receptacle: str) -> str:
        a, b = self.receptacles
        return b if receptacle == a else a


@dataclass
class _RunState:
    assignment: dict[str, str]
    episodes: list[dict[str, Any]] = field(default_factory=list)
    interventions: list[dict[str, Any]] = field(default_factory=list)
    skipped_restorations: list[dict[str, Any]] = field(default_factory=list)
    snapshots: dict[int, dict[str, str]] = field(default_factory=dict)  # episode idx -> assignment


# ----------------------------------------------------------------------------
# Episode construction
# ----------------------------------------------------------------------------


def _object_poses(cfg: RunConfig, assignment: dict[str, str]) -> dict[str, list[float]]:
    return {obj: cfg.placements[f"{obj}|{rec}"] for obj, rec in assignment.items()}


def _base_episode(cfg: RunConfig, assignment: dict[str, str]) -> dict[str, Any]:
    """The fields every episode shares. Cameras are frozen across the whole run."""
    return {
        "house_index": cfg.house_index,
        "scene_dataset": cfg.scene_dataset,
        "data_split": cfg.data_split,
        "seed": cfg.seed,
        "robot": {"robot_name": cfg.robot_name, "init_qpos": cfg.robot_init_qpos},
        "img_resolution": list(cfg.img_resolution),
        "cameras": cfg.cameras,
        "scene_modifications": {
            "added_objects": {},
            "object_poses": _object_poses(cfg, assignment),
            "removed_objects": [],
        },
        "task_relevant_objects": list(cfg.tracked_objects) + list(cfg.receptacles),
    }



def measure_body_poses(scene_xml: str | Path, names: list[str]) -> dict[str, list[float]]:
    """Actual world poses (pos + wxyz quat) of the named bodies in the compiled scene.

    Use for `RunConfig.receptacle_poses`. The fridge in FloorPlan3 sits at
    [1.013, 1.917, 1.2197] with quat [-0.5, -0.5, 0.5, 0.5] -- neither the AABB
    centre nor an identity quaternion, so both must be measured, not assumed.
    """
    import mujoco

    model = mujoco.MjSpec.from_file(str(scene_xml)).compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    out: dict[str, list[float]] = {}
    for name in names:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise KeyError(f"body {name!r} not in {scene_xml}")
        out[name] = [float(v) for v in data.xpos[bid]] + [float(v) for v in data.xquat[bid]]
    return out


def _work_episode(cfg: RunConfig, assignment: dict[str, str], obj: str) -> dict[str, Any]:
    src = assignment[obj]
    dst = cfg.other(src)
    ep = _base_episode(cfg, assignment)
    ep["task"] = {
        "task_cls": PICK_PLACE_CLS,
        "task_type": "pick_and_place",
        "robot_base_pose": cfg.robot_base_pose,
        "pickup_obj_name": obj,
        "pickup_obj_start_pose": cfg.placements[f"{obj}|{src}"],
        "place_receptacle_name": dst,
        "place_receptacle_start_pose": cfg.receptacle_start_pose(dst),
        "task_horizon_sec": cfg.horizon_sec_work,
    }
    ep["language"] = {
        "task_description": f"Pick up the {obj} and place it in or on the {dst}.",
        "referral_expressions": {"pickup_name": obj, "place_name": dst},
    }
    return ep


def _explore_episode(cfg: RunConfig, assignment: dict[str, str]) -> dict[str, Any]:
    ep = _base_episode(cfg, assignment)
    r0, r1 = cfg.receptacles
    ep["task"] = {
        "task_cls": EXPLORE_CLS,
        "task_type": "explore",
        "robot_base_pose": cfg.robot_base_pose,
        "receptacle_names": list(cfg.receptacles),
        "succ_pos_threshold": 1.5,
        "task_horizon_sec": cfg.horizon_sec_explore,
    }
    ep["language"] = {
        "task_description": f"Go and look at the {r0} and the {r1}.",
        "referral_expressions": {r0: r0, r1: r1},
    }
    return ep


def _restoration_episode(
    cfg: RunConfig,
    assignment: dict[str, str],
    target: dict[str, str],
    source_episode: int,
    cue_objects: list[str],
) -> dict[str, Any]:
    ep = _base_episode(cfg, assignment)
    ep["task"] = {
        "task_cls": REORDER_CLS,
        "task_type": "reorder",
        "robot_base_pose": cfg.robot_base_pose,
        "target_assignment": dict(target),
        "source_episode": source_episode,
        "cue_objects": list(cue_objects),
        "receptacle_supported_weight_frac": 0.5,
        "task_horizon_sec": cfg.horizon_sec_restoration,
    }
    ep["language"] = {
        "task_description": (
            f"Put things back where they were at episode {source_episode}."
        ),
        "referral_expressions": {o: o for o in target},
    }
    return ep


def build_run(cfg: RunConfig) -> dict[str, Any]:
    """Generate the episode chain plus a manifest recording every change."""
    import random

    rng = random.Random(cfg.seed)
    state = _RunState(assignment=dict(cfg.initial_assignment))

    def emit(ep: dict[str, Any], kind: str) -> int:
        idx = len(state.episodes)
        ep["episode_index"] = idx
        ep["episode_kind"] = kind
        state.episodes.append(ep)
        return idx

    cycles = max(1, cfg.n_work_episodes // max(cfg.work_per_cycle, 1))
    for cycle in range(cycles):
        # --- work episodes: the robot itself moves objects ---
        for _ in range(cfg.work_per_cycle):
            obj = rng.choice(cfg.tracked_objects)
            emit(_work_episode(cfg, state.assignment, obj), "work")
            state.assignment[obj] = cfg.other(state.assignment[obj])

        # --- intervention: unobserved, breaks the action-log shortcut ---
        moved = rng.sample(cfg.tracked_objects, k=min(cfg.objects_per_intervention,
                                                      len(cfg.tracked_objects)))
        for obj in moved:
            src = state.assignment[obj]
            dst = cfg.other(src)
            state.assignment[obj] = dst
            state.interventions.append(
                {
                    "before_episode_index": len(state.episodes),
                    "object": obj,
                    "from": src,
                    "to": dst,
                    "observed": False,
                }
            )

        # --- explore: re-observe, writing a fresh snapshot into memory ---
        idx = emit(_explore_episode(cfg, state.assignment), "explore")
        state.snapshots[idx] = dict(state.assignment)

        # --- restoration, targeting an earlier snapshot ---
        if cfg.restoration_every and (cycle + 1) % cfg.restoration_every == 0:
            n_cue = int(round(cfg.cue_fraction * len(cfg.tracked_objects)))
            cue = rng.sample(cfg.tracked_objects, k=n_cue) if n_cue else []

            # Random drift over two receptacles can coincidentally return the
            # arrangement to an earlier snapshot, and cue objects shrink the gap
            # further. A restoration that needs zero moves is already solved, and a
            # memoryless baseline would score 100% on it -- so choose a snapshot
            # that still requires real work after cues are applied. Prefer the one
            # nearest `query_depth`, since query depth is the history-length axis.
            candidates = sorted(state.snapshots)[:-1]  # never target the newest

            def _moves_needed(idx: int) -> int:
                target = state.snapshots[idx]
                after_cue = dict(state.assignment)
                for obj in cue:
                    after_cue[obj] = target[obj]
                return sum(1 for o in target if after_cue.get(o) != target[o])

            viable = [i for i in candidates if _moves_needed(i) >= cfg.min_objects_to_move]
            if viable:
                preferred = candidates[-1 - cfg.query_depth] if len(
                    candidates
                ) > cfg.query_depth else candidates[0]
                target_idx = min(viable, key=lambda i: (abs(i - preferred), -i))
                target = state.snapshots[target_idx]
                for obj in cue:
                    state.assignment[obj] = target[obj]
                emit(
                    _restoration_episode(cfg, state.assignment, target, target_idx, cue),
                    "restoration",
                )
            else:
                state.skipped_restorations.append(
                    {
                        "after_cycle": cycle,
                        "reason": (
                            f"no snapshot required >= {cfg.min_objects_to_move} moves; "
                            "arrangement had drifted back. Raise n tracked objects or "
                            "objects_per_intervention to diverge faster."
                        ),
                    }
                )

    manifest = {
        "config": {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in cfg.__dict__.items()
        },
        "n_episodes": len(state.episodes),
        "episode_kinds": [e["episode_kind"] for e in state.episodes],
        "interventions": state.interventions,
        "snapshots": {str(k): v for k, v in state.snapshots.items()},
        "skipped_restorations": state.skipped_restorations,
        "final_assignment": state.assignment,
    }
    return {"episodes": state.episodes, "manifest": manifest}


def write_run(result: dict[str, Any], out_dir: str | Path) -> Path:
    """Write the chain in the layout JsonEvalRunner expects: house_<i>/episode_*.json."""
    out = Path(out_dir)
    house = f"house_{result['manifest']['config']['house_index']}"
    (out / house).mkdir(parents=True, exist_ok=True)
    for ep in result["episodes"]:
        path = out / house / f"episode_{ep['episode_index']:08d}.json"
        path.write_text(json.dumps(ep, indent=2) + "\n")
    (out / "manifest.json").write_text(json.dumps(result["manifest"], indent=2) + "\n")
    return out



# ----------------------------------------------------------------------------
# Geometric validation
# ----------------------------------------------------------------------------


def validate_run(
    cfg: RunConfig,
    scene_xml: str | Path,
    explore_radius: float = 1.5,
    clearance: float = 0.6,
) -> list[str]:
    """Check a run config's geometric preconditions against the compiled scene.

    These are the failure modes that do not raise -- they quietly produce
    episodes that are already solved, so a do-nothing policy scores 100% and the
    result is silently meaningless. Found the hard way: an explore episode passed
    with DummyPolicy because both receptacles sat inside one 1.5 m radius.

    Returns a list of problems; empty means the config is sound.
    """
    import mujoco
    import numpy as np

    problems: list[str] = []
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    def body_pos(name: str):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        return None if bid < 0 else data.xpos[bid].copy()

    # Every referenced body must exist. EpisodeSpec fields take MuJoCo *body*
    # names (hash_name), not the human-readable object_id.
    for name in list(cfg.receptacles) + list(cfg.tracked_objects):
        if body_pos(name) is None:
            problems.append(f"body not found in scene: {name!r} (use the hash_name, not object_id)")
    if problems:
        return problems

    r0, r1 = (body_pos(r) for r in cfg.receptacles)
    sep = float(np.linalg.norm(r0[:2] - r1[:2]))
    # The robot must not be able to sit within the explore radius of both at once,
    # or an explore episode is satisfied without moving.
    if sep <= 2 * explore_radius:
        problems.append(
            f"receptacles are {sep:.2f} m apart but the explore radius is "
            f"{explore_radius:.2f} m -- the robot can observe both without moving, so "
            f"explore episodes are trivially solved. Need > {2 * explore_radius:.2f} m."
        )

    base = np.asarray(cfg.robot_base_pose[:2], dtype=float)
    for name, pos in zip(cfg.receptacles, (r0, r1)):
        dist = float(np.linalg.norm(base - pos[:2]))
        if dist < 0.5:
            problems.append(
                f"robot base is {dist:.2f} m from receptacle {name!r} -- it will spawn "
                "inside the furniture."
            )

    # Spawn validity is the PHYSICS ENGINE's verdict. Three other tests were tried
    # and all three disagreed with the simulator:
    #   * distance-to-nearest-geom PASSES points outside the building;
    #   * world-space AABBs call every cell occupied (a rotated wall mesh's AABB
    #     spans the room);
    #   * A* plannability is wrong in BOTH directions on FloorPlan3 -- it rejected a
    #     pose with 0 contacts and accepted one with 510.
    # Placing the actual robot and counting robot-vs-non-floor contacts cannot
    # disagree with the simulator, because it IS the simulator.
    try:
        probe = _spawn_collision_probe(scene_xml)
        hits = probe(float(base[0]), float(base[1]))
        if hits:
            problems.append(
                f"robot base ({base[0]:.2f}, {base[1]:.2f}) puts the robot in "
                f"{hits} contact(s) with the scene. Use `build_run.py find-spawn`."
            )
    except Exception as exc:  # noqa: BLE001
        problems.append(f"could not probe the spawn for collisions: {exc}")

    return problems




def _spawn_collision_probe(scene_xml: str | Path):
    """Build the scene WITH the robot in it and return a probe(x, y) -> n_contacts.

    Analytic clearance tests were tried three times and all three disagreed with the
    simulator: point-to-AABB called every cell occupied, point-to-box reported
    0.378 m clearance at a pose the physics had 203 mm inside a wall. A point is not
    a robot -- RB-Y1 is ~0.5 m wide, over a metre tall, ~1000 geoms -- so the only
    test that cannot disagree with physics is to place the actual robot and let
    MuJoCo run broad/narrow phase.

    Mirrors `BaseMujocoEnv.check_robot_collision_in_current_pose` (env.py:641),
    counting robot<->scene contacts with negative distance.
    """
    import mujoco
    import numpy as np

    # Metres. Contacts below this height are ground contact, not collision.
    # Measured, not guessed: on open floor the robot's floor-surface contacts span
    # z=0.032..0.220 (bodies `floor_*`, `decals_*`, `mesh_*`); at a pose buried in the
    # oven they span z=0.032..1.359 and involve `oven_*`/`cube_*`. 0.25 sits in that
    # gap. An earlier attempt derived this from `model.geom_pos`, which is BODY-LOCAL,
    # not world -- the resulting threshold filtered out every contact and made even
    # the oven interior look clear.
    FLOOR_CLEARANCE = 0.25

    import molmo_spaces.configs  # noqa: F401  (avoids the astar<->configs cycle)
    from molmo_spaces.configs.robot_configs import RBY1MConfig

    cfg = RBY1MConfig()
    spec = mujoco.MjSpec.from_file(str(scene_xml))
    cfg.robot_cls.add_robot_to_scene(cfg, spec, "robot_0/", [0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
    model = spec.compile()
    data = mujoco.MjData(model)

    jid = {
        axis: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"robot_0/base_{axis}")
        for axis in ("x", "y", "theta")
    }
    if any(v < 0 for v in jid.values()):
        raise RuntimeError("RB-Y1 base joints not found; cannot probe spawns")
    adr = {k: model.jnt_qposadr[v] for k, v in jid.items()}

    def probe(x: float, y: float, theta: float = 0.0) -> int:
        mujoco.mj_resetData(model, data)
        data.qpos[adr["x"]] = x
        data.qpos[adr["y"]] = y
        data.qpos[adr["theta"]] = theta
        mujoco.mj_forward(model, data)
        n = 0
        for ci in range(int(data.ncon)):
            con = data.contact[ci]
            if con.dist > 0:
                continue
            r1 = int(model.body_rootid[int(model.geom_bodyid[int(con.geom1)])])
            r2 = int(model.body_rootid[int(model.geom_bodyid[int(con.geom2)])])
            n1 = model.body(r1).name or ""
            n2 = model.body(r2).name or ""
            robot1, robot2 = n1.startswith("robot_0/"), n2.startswith("robot_0/")
            if robot1 == robot2:
                continue  # robot self-collision, or scene-vs-scene
            other = n2 if robot1 else n1
            # Floor contact is unavoidable, not a collision: RB-Y1's base joints are
            # planar (x/y/theta), so nothing lifts the base clear of the floor plane
            # and it rests at z=0.01. A NAME filter is not enough -- the floor surface
            # in iTHOR is split across `floor_*`, `decals_*` and `mesh_*` bodies, and
            # the latter two produced 510 "collisions" at EVERY pose, which is what
            # made all 200 map-free points look occupied. Filter by contact HEIGHT
            # instead: a genuine obstacle (wall, counter, appliance) is struck above
            # the floor, so anything at floor level is ground contact.
            if float(con.pos[2]) < FLOOR_CLEARANCE:
                continue
            n += 1
        return n

    return probe


def find_spawn(
    scene_xml: str | Path,
    receptacles: list[str],
    clearance: float = 0.6,
    grid_step: float = 0.25,
) -> tuple[list[float], float]:
    """Pick a robot spawn the NAVIGATION PLANNER considers valid.

    An earlier version scored candidate points by distance to the nearest geom.
    That is not a validity test: a point outside the building has enormous
    clearance and scores best, which is exactly what happened -- the robot spawned
    off-map and A* reported "START is not plannable".

    The spawn must come from the same world model the planner routes in. A cell is
    acceptable only if ``AStarPlanner.get_discrete_location`` resolves it AND a
    route exists from it to both receptacles; otherwise the run cannot even begin.
    Among valid cells, prefer balanced travel to the two receptacles.
    """
    import mujoco
    import numpy as np

    # Import the package first: astar_planner and configs import each other, so
    # importing astar_planner directly hits a partially-initialised module.
    import molmo_spaces.configs  # noqa: F401
    from molmo_spaces.planner.astar_planner import AStarPlanner, AStarPlannerConfig

    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    targets = []
    for name in receptacles:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise ValueError(f"receptacle body not found: {name!r}")
        targets.append(data.xpos[bid][:3].copy())

    planner = AStarPlanner(AStarPlannerConfig(agent_radius=clearance), str(scene_xml))

    # Search around the receptacles: the robot must work near both, and the
    # navigable floor is bounded by them far more tightly than by the scene AABB.
    centre = (targets[0][:2] + targets[1][:2]) / 2.0
    span = float(np.linalg.norm(targets[0][:2] - targets[1][:2])) / 2.0 + 1.5

    # Score every plannable cell by geometric clearance and report the best one,
    # rather than demanding a fixed threshold: RB-Y1 is wide and FloorPlan3 is a
    # small kitchen, so a hard 0.55 m cut leaves no candidates at all. The caller
    # sees the achieved clearance and can judge.
    candidates = []
    steps = int((2 * span) / grid_step) + 1
    n_plannable = 0
    for ix in range(steps):
        for iy in range(steps):
            pt = centre + np.array([-span + ix * grid_step, -span + iy * grid_step])
            probe = np.array([pt[0], pt[1], 0.0])
            # TEST 1 -- the planner must be able to start here. Rejects points
            # outside the building, which have huge geometric clearance and would
            # otherwise score best.
            if planner.get_discrete_location(probe) is None:
                continue
            n_plannable += 1
            # TEST 2 -- the real robot must not collide here. The occupancy map does
            # NOT rasterise every collidable body (FloorPlan3's walls are `decals_*`
            # boxes on the __STRUCTURAL_MJT__ class), so plannable does not imply
            # collision-free.
            d = [float(np.linalg.norm(pt - t[:2])) for t in targets]
            candidates.append((0.0, -abs(d[0] - d[1]), pt))

    if not candidates:
        raise RuntimeError(
            "no plannable cell near the receptacles; the occupancy map may not cover "
            "this region, or agent_radius is too large"
        )
    # Prefer balanced access, then probe in that order until one is collision-free.
    candidates.sort(key=lambda c: -c[1])
    probe = _spawn_collision_probe(scene_xml)
    best, gap, n_probed = None, 0.0, 0
    for _, _, pt in candidates:
        n_probed += 1
        hits = probe(float(pt[0]), float(pt[1]))
        if hits == 0:
            best = pt
            break
    if best is None:
        raise RuntimeError(
            f"probed {n_probed} plannable cells; every one puts the robot in contact "
            "with the scene. Try a different scene or a narrower robot."
        )
    print(
        f"# {n_plannable} plannable cells; collision-free after probing {n_probed}",
        file=__import__("sys").stderr,
    )
    print(
        f"# {n_plannable} plannable cells; best clearance {gap:.3f} m",
        file=__import__("sys").stderr,
    )
    return [float(best[0]), float(best[1]), 0.0, 1.0, 0.0, 0.0, 0.0], float(gap)




# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_i = sub.add_parser("inspect", help="list receptacle / tracked-object candidates")
    p_i.add_argument("--metadata", required=True, help="<scene>_physics_metadata.json")
    p_i.add_argument("--limit", type=int, default=25)

    p_b = sub.add_parser("build", help="emit an EpisodeSpec chain")
    p_b.add_argument("--config", required=True, help="JSON matching RunConfig fields")
    p_b.add_argument("--out", required=True)
    p_b.add_argument("--scene-xml", help="if given, validate geometry before building")

    p_v = sub.add_parser("validate", help="check geometric preconditions only")
    p_v.add_argument("--config", required=True)
    p_v.add_argument("--scene-xml", required=True)

    p_s = sub.add_parser("find-spawn", help="search for a robot base pose with clearance")
    p_s.add_argument("--scene-xml", required=True)
    p_s.add_argument("--receptacles", nargs=2, required=True)
    p_s.add_argument("--clearance", type=float, default=0.6)

    args = ap.parse_args()

    if args.cmd == "inspect":
        found = inspect_scene(args.metadata)
        # NOTE: `body` is the MuJoCo body name and is what EpisodeSpec fields
        # (object_poses, pickup_obj_name, place_receptacle_name, target_assignment)
        # must use. `object_id` is human-readable only -- passing it raises
        # KeyError: Invalid name.
        print(f"=== RECEPTACLES (have receptacle sites): {len(found['receptacles'])}")
        for e in found["receptacles"][: args.limit]:
            print(f"   {str(e['object_id']):26} sites={e['n_sites']:<3} "
                  f"static={str(e['is_static']):5} body={e['body']}")
        print(f"\n=== MOVABLE (tracked-object candidates): {len(found['movable'])}")
        for e in found["movable"][: args.limit]:
            print(f"   {str(e['object_id']):26} {str(e['category']):16} body={e['body']}")
        print("\n>>> Use the `body=` names in RunConfig; object_id will raise KeyError.")
        return

    if args.cmd == "find-spawn":
        pose, gap = find_spawn(args.scene_xml, args.receptacles, clearance=args.clearance)
        print(json.dumps({"robot_base_pose": pose, "clearance_m": round(gap, 3)}, indent=2))
        return

    cfg_dict = json.loads(Path(args.config).read_text())
    cfg = RunConfig(**cfg_dict)

    if args.cmd == "validate":
        problems = validate_run(cfg, args.scene_xml)
        if problems:
            print(f"FAIL: {len(problems)} problem(s)")
            for pr in problems:
                print("  -", pr)
            raise SystemExit(1)
        print("OK: geometric preconditions satisfied")
        return

    if getattr(args, "scene_xml", None):
        problems = validate_run(cfg, args.scene_xml)
        if problems:
            print(f"REFUSING TO BUILD: {len(problems)} problem(s)")
            for pr in problems:
                print("  -", pr)
            raise SystemExit(1)

    result = build_run(cfg)
    out = write_run(result, args.out)
    kinds = result["manifest"]["episode_kinds"]
    print(f"wrote {len(result['episodes'])} episodes to {out}")
    print("  cadence:", " ".join(k[0].upper() for k in kinds))
    print("  interventions:", len(result["manifest"]["interventions"]))
    print("  snapshots:", len(result["manifest"]["snapshots"]))
    skipped = result["manifest"]["skipped_restorations"]
    if skipped:
        print(f"  SKIPPED {len(skipped)} restoration(s) as trivially-solved:")
        for s_ in skipped:
            print(f"     after cycle {s_['after_cycle']}: {s_['reason']}")


if __name__ == "__main__":
    main()
