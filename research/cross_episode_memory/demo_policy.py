"""Scripted policy for DEMO VIDEOS ONLY.

**This is a cheat and must never appear in a baseline or ablation.** It reads the
task's privileged target directly and teleports objects into place. Its only job is
to make a watchable video of the episode structure: the robot drives between
receptacles and the arrangement visibly changes.

Why it exists at all: the real oracle path (`CuroboPickAndPlacePlannerPolicy`) is
unavailable in this environment -- the installed `nvidia-curobo 1.0` is an
unrelated package that lacks `curobo.geom`, and building Ai2's fork needs a CUDA
toolchain. A no-op policy produces a motionless robot, which demonstrates nothing.

The no-cheat rule (SPEC.md) is unaffected: it governs the execution loop of systems
under test. A demo video is neither an experiment nor a baseline.
"""

import logging
from typing import Any

import numpy as np

from molmo_spaces.env.data_views import create_mlspaces_body
from molmo_spaces.policy.base_policy import BasePolicy

log = logging.getLogger(__name__)

# Metres per policy step for the kinematic base drive. RB-Y1's holonomic base is a
# plain position servo (`holo_joint_planar_position` -> JointPosController) that
# achieves only ~0.015 m/s against its gains -- roughly five minutes of simulation
# per 4 m leg, which makes a multi-episode demo impractical. This policy therefore
# writes the base joints directly along an interpolated path: smooth visible motion,
# deterministic duration. Another demo-only cheat; see the module docstring.
BASE_STEP = 0.18
# Steps spent ramping an articulated door open (and again closed). Writing the hinge
# qpos in one step would snap the door through any object beside it; ramping keeps
# the motion visible and lets contacts resolve between steps.
DOOR_STEPS = 12
# Steps spent extending the arm toward the receptacle, and again retracting it. The
# object still moves kinematically, but teleporting it while the robot stood
# motionless read as fake -- the arm now reaches out, the object transfers at full
# extension, and the arm comes back.
ARM_STEPS = 12
# RB-Y1 right-arm home pose (`RBY1MConfig.init_qpos`) and an extended reach pose:
# shoulder pitched forward and elbow opened out from its tucked -2.3 rad.
ARM_HOME = np.array([0.5, 0.0, 0.0, -2.3, 0.0, -0.5, 0.0])
ARM_REACH = np.array([0.95, -0.15, 0.0, -1.05, 0.0, -0.55, 0.0])
# Metres. A receptacle centre sits INSIDE furniture, so it is never navigable: the
# planner snaps the goal to the nearest free cell, which is roughly a robot radius
# away. Arrival is therefore judged against that standoff point, not the receptacle
# centre -- judging against the centre parked the robot 1.32 m "short" of a goal it
# had in fact reached.
STANDOFF_TOLERANCE = 0.55
# Metres. The standoff must clear the receptacle by more than the base radius: with
# the nearest free point (fridge: 0.75 m from centre) the parked base sits ~0.4 m from
# the appliance face and RB-Y1's right end-effector still struck it -- logged as
# `EE_BODY_R<->refrigerator_... F=47194N d=34mm` on approach. Prefer the nearest free
# point that is at least this far from the receptacle centre.
MIN_STANDOFF_FROM_CENTRE = 1.15


class ScriptedDemoPolicy(BasePolicy):
    """Drive the base between receptacles; teleport objects on arrival.

    Handles all three episode kinds:
      * **work** -- drive to the place receptacle, then move the picked object onto it.
      * **explore** -- visit each receptacle in turn; no object changes.
      * **restoration** -- visit each receptacle that has objects to restore, moving
        them as it arrives.
    """

    @property
    def type(self) -> str:
        return "scripted_demo"

    def reset(self) -> None:
        self._waypoints: list[tuple[np.ndarray, list[str], str]] | None = None
        self._wp_index = 0
        self._nav_planner = None
        self._path: list[np.ndarray] = []
        self._path_index = 0
        self._stage: str | None = None
        self._stage_step = 0
        self._standoff_cache: dict[str, np.ndarray] = {}
        self._door_cache: dict[str, tuple[int, float, float]] = {}

    def _receptacle_xy(self, name: str) -> np.ndarray | None:
        data = self.task.env.mj_datas[0]
        try:
            return create_mlspaces_body(data, name).position[:2].copy()
        except KeyError:
            return None

    def _plan(self) -> list[tuple[np.ndarray, list[str], str]]:
        """Build (waypoint_xy, objects_to_move, destination) triples."""
        cfg = self.task.config.task_config
        kind_plan: list[tuple[np.ndarray, list[str], str]] = []

        target_assignment = getattr(cfg, "target_assignment", None)
        receptacle_names = getattr(cfg, "receptacle_names", None)
        place_receptacle = getattr(cfg, "place_receptacle_name", None)

        if target_assignment:  # restoration
            by_receptacle: dict[str, list[str]] = {}
            for obj, rec in target_assignment.items():
                by_receptacle.setdefault(rec, []).append(obj)
            for rec, objs in by_receptacle.items():
                xy = self._receptacle_xy(rec)
                if xy is not None:
                    kind_plan.append((xy, objs, rec))
        elif receptacle_names:  # explore -- look, change nothing
            for rec in receptacle_names:
                xy = self._receptacle_xy(rec)
                if xy is not None:
                    kind_plan.append((xy, [], rec))
        elif place_receptacle:  # work
            xy = self._receptacle_xy(place_receptacle)
            obj = getattr(cfg, "pickup_obj_name", None)
            if xy is not None:
                kind_plan.append((xy, [obj] if obj else [], place_receptacle))
        return kind_plan

    def _planner(self):
        """MolmoSpaces' shipped A* over `iTHORMap`.

        Use this, not a hand-built grid. The map encodes *navigable floor* -- 8026
        cells for FloorPlan3, with outdoor space absent entirely -- which is
        information no contact test can reconstruct. A grid built from "does the
        robot touch anything here" scores empty space OUTSIDE the building as the
        freest cells in the scene, and A* duly routed the robot out of the kitchen,
        around the house, and back in: collision-free because it had left the world.

        Note `get_discrete_location` SNAPS to the nearest navigable cell within
        `max_start_goal_distance` (~1 m). A non-None return means "there is floor
        near here", NOT "this pose is collision-free" -- misreading that is what
        sent me building the grid in the first place.
        """
        if self._nav_planner is None:
            import molmo_spaces.configs  # noqa: F401  (astar<->configs import cycle)
            from molmo_spaces.planner.astar_planner import AStarPlanner, AStarPlannerConfig

            self._nav_planner = AStarPlanner(
                AStarPlannerConfig(agent_radius=0.35),
                self.task.env.current_model_path,
            )
            log.info("[nav] planner ready: %d navigable cells", len(self._nav_planner.graph))
        return self._nav_planner

    def _plan_path(self, target_xy: np.ndarray) -> list[np.ndarray]:
        """Waypoints to ``target_xy`` in world coords, or [] if unreachable."""
        robot_view = self.task.env.current_robot.robot_view
        planner = self._planner()
        target = np.array([target_xy[0], target_xy[1], 0.0])
        try:
            plan = planner.motion_plan(target, robot_view)
        except Exception as exc:  # noqa: BLE001
            log.error("[nav] planning failed: %s", exc)
            return []
        if plan is None or len(plan) == 0:
            return []
        # `AStarPlanner._compute_plan` already returns WORLD waypoints -- it does
        # `pos_px_to_m(waypoints * downscale)[:, :2]` internally. An earlier version
        # here re-applied that conversion, treating metres as grid cells and steering
        # the robot to nonsense coordinates. Take the plan as-is.
        pts = [np.asarray(w, dtype=float)[:2] for w in plan]
        log.info("[nav] planned %d waypoints to (%.2f, %.2f)", len(pts), target_xy[0], target_xy[1])
        return pts

    def _standoff_xy(self, receptacle: str, target_xy: np.ndarray) -> np.ndarray:
        """Nearest navigable point to the receptacle -- the pose to actually drive to.

        Uses the same occupancy map the planner does, so the returned point is
        guaranteed navigable and indoors.
        """
        if receptacle in self._standoff_cache:
            return self._standoff_cache[receptacle]
        try:
            thormap = self.task.env.get_thormap()
            free = np.asarray(thormap.get_free_points())[:, :2]
            d = np.linalg.norm(free - target_xy, axis=1)
            far = np.flatnonzero(d >= MIN_STANDOFF_FROM_CENTRE)
            if far.size:
                # Closest point that still clears the receptacle by the arm's reach.
                best = free[far[int(np.argmin(d[far]))]]
            else:
                log.warning("[nav] no free point >= %.2f m from %s; using nearest",
                            MIN_STANDOFF_FROM_CENTRE, receptacle)
                best = free[int(np.argmin(d))]
        except Exception as exc:  # noqa: BLE001
            log.warning("[nav] standoff lookup failed for %s (%s); using centre",
                        receptacle, exc)
            best = np.asarray(target_xy, dtype=float)
        self._standoff_cache[receptacle] = best
        log.info("[nav] standoff for %s: (%.2f, %.2f), %.2f m from centre",
                 receptacle, best[0], best[1],
                 float(np.linalg.norm(best - target_xy)))
        return best

    def _door_joint(self, receptacle: str):
        """(joint_id, qpos_adr, closed, open) for the receptacle's door, or None.

        Finds the widest-range hinge/slide joint anywhere in the receptacle's body
        subtree, so it works for the fridge (`Fridge_3_joint`) and the oven
        (`oven_FP3_door`) without hard-coding either name.
        """
        import mujoco

        model = self.task.env.current_model
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, receptacle)
        if bid < 0:
            return None
        # Bodies in this receptacle's subtree.
        subtree = {bid}
        for b in range(model.nbody):
            parent = b
            while parent > 0:
                if parent in subtree:
                    subtree.add(b)
                    break
                parent = int(model.body_parentid[parent])
        best = None
        for j in range(model.njnt):
            if int(model.jnt_bodyid[j]) not in subtree:
                continue
            if int(model.jnt_type[j]) not in (
                int(mujoco.mjtJoint.mjJNT_HINGE),
                int(mujoco.mjtJoint.mjJNT_SLIDE),
            ):
                continue
            lo, hi = (float(x) for x in model.jnt_range[j])
            if hi - lo <= 1e-6:
                continue
            if best is None or (hi - lo) > (best[3] - best[2]):
                best = (j, int(model.jnt_qposadr[j]), lo, hi)
        if best is None:
            return None
        j, adr, lo, hi = best
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        # Open toward whichever limit is further from the rest position.
        data = self.task.env.mj_datas[0]
        closed = float(data.qpos[adr])
        opened = hi if abs(hi - closed) > abs(closed - lo) else lo
        log.info("[door] %s -> joint `%s` closed=%.3f open=%.3f", receptacle, name,
                 closed, opened)
        return j, adr, closed, opened

    def _log_success_metrics(self, obj_name: str, receptacle: str) -> None:
        """Log the four conditions `PickAndPlaceTask` uses, in-situ.

        Offline the placement satisfies all four, yet the episode scored 0 at every
        step -- so the disagreement has to be measured inside the running episode
        rather than reasoned about.
        """
        import mujoco
        from molmo_spaces.utils.mujoco_scene_utils import is_object_supported_by_body

        model = self.task.env.current_model
        data = self.task.env.mj_datas[0]
        try:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
            fid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, receptacle)
            if bid < 0 or fid < 0:
                return
            root = int(model.body_rootid[bid])
            sup = bool(is_object_supported_by_body(data, root, fid, frac_weight_threshold=0.5))
            rc = False
            ncontact = 0
            for ci in range(int(data.ncon)):
                c = data.contact[ci]
                r1 = int(model.body_rootid[int(model.geom_bodyid[c.geom1])])
                r2 = int(model.body_rootid[int(model.geom_bodyid[c.geom2])])
                if (r1 == root) ^ (r2 == root):
                    other = r1 if r1 != root else r2
                    ncontact += 1
                    if (model.body(other).name or "").startswith("robot_0/"):
                        rc = True
            log.info(
                "[success] obj_z=%.3f supported=%s robot_contact=%s obj_contacts=%d",
                float(data.xpos[bid][2]), sup, rc, ncontact,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("[success] metric probe failed: %s", exc)

    def _drive_arm(self, frac: float) -> None:
        """Interpolate the right arm between its home and reach pose.

        Joint-space interpolation, not IK: there is no RB-Y1 IK module in the repo
        (`molmo_spaces/kinematics/` has no rby1 solver), and for a demo a repeatable
        reach beats a solver that may fail mid-episode. The arm is driven through the
        move group, so it moves as an articulated chain rather than being teleported.
        """
        robot_view = self.task.env.current_robot.robot_view
        try:
            group = robot_view.get_move_group("right_arm")
        except Exception as exc:  # noqa: BLE001
            log.warning("[arm] no right_arm move group (%s); skipping reach", exc)
            return
        f = float(np.clip(frac, 0.0, 1.0))
        group.joint_pos = ARM_HOME + (ARM_REACH - ARM_HOME) * f

    def _drive_door(self, receptacle: str, frac: float) -> bool:
        """Ramp the receptacle's door to `frac` of its travel. False if it has none.

        The rest pose is captured ONCE per receptacle and cached. `_door_joint` reads
        `closed` from the live qpos, so calling it every step measured each increment
        against a baseline that had already moved -- observed drifting -0.569 ->
        -0.589 within one open ramp, which both short-changes the opening and leaves
        the door away from its true rest position on closing.
        """
        if receptacle not in self._door_cache:
            info = self._door_joint(receptacle)
            if info is None:
                self._door_cache[receptacle] = None
            else:
                _, adr, closed, opened = info
                self._door_cache[receptacle] = (adr, closed, opened)
        cached = self._door_cache[receptacle]
        if cached is None:
            return False
        adr, closed, opened = cached
        data = self.task.env.mj_datas[0]
        data.qpos[adr] = closed + (opened - closed) * float(np.clip(frac, 0.0, 1.0))
        return True

    def _teleport(self, objects: list[str], receptacle: str) -> None:
        """The cheat: place objects onto the receptacle without manipulating.

        Kinematic, but it must not leave an object *inside* geometry. Placement is
        computed from the receptacle's measured subtree AABB plus the object's own
        half-height (see `placement.resting_position`); an earlier version added a
        guessed 0.55 m to the body origin and buried every object 0.56 m inside the
        oven, because that origin sits near the appliance's base, not its top.

        Objects already placed this call are passed as `occupied`, so they spread
        across the surface instead of stacking into one another. Knocking a
        pre-existing scene object is tolerable; interpenetrating one is not.
        """
        from research.cross_episode_memory.placement import resting_position

        data = self.task.env.mj_datas[0]
        model = self.task.env.current_model
        occupied: list[np.ndarray] = []
        for obj_name in objects:
            try:
                obj = create_mlspaces_body(data, obj_name)
            except KeyError:
                log.warning("teleport: object %r not in model", obj_name)
                continue
            pos = resting_position(model, data, obj_name, receptacle, occupied_xy=occupied)
            if pos is None:
                log.warning("teleport: no resting pose for %r on %r", obj_name, receptacle)
                continue
            obj.position = pos
            occupied.append(pos[:2])
            log.info("teleport: %s -> %s at z=%.3f", obj_name, receptacle, pos[2])

    def _log_progress(self, phase: str, **fields: Any) -> None:
        """One line per step: what the policy is doing and where it is.

        Event-only logging ("planned 3 waypoints", "reached fridge") cannot answer
        the question a stalled run actually raises -- *where is it now and what is
        it trying to do*. This prints the phase, the leg, the remaining distance and
        the base pose every step, so a stall is legible as a distance that stops
        shrinking rather than as silence.
        """
        step = getattr(self.task, "episode_step_count", 0)
        base = self.task.env.current_robot.robot_view.base.pose[:2, 3]
        extra = " ".join(f"{k}={v}" for k, v in fields.items())
        log.info(
            "[step %3d] phase=%-10s base=(%.2f,%.2f) %s",
            step, phase, base[0], base[1], extra,
        )

    def _log_collisions(self) -> None:
        """Report robot<->scene contacts. Anything here that is not a held object
        is the robot clipping the house, which is the thing to fix."""
        from research.cross_episode_memory.collision import robot_scene_contacts, summarize

        step = getattr(self.task, "episode_step_count", 0)
        if step % 5:
            return
        model = self.task.env.current_model
        data = self.task.env.mj_datas[0]
        hits = robot_scene_contacts(model, data)
        if hits:
            log.warning("[collision] step %d: %s", step, summarize(hits))
        else:
            log.info("[collision] step %d: clear", step)

    def get_action(self, observation) -> dict[str, Any]:
        self._log_collisions()
        robot_view = self.task.env.current_robot.robot_view
        action = robot_view.get_noop_ctrl_dict()

        if self._waypoints is None:
            self._waypoints = self._plan()
            self._wp_index = 0
        if self._wp_index >= len(self._waypoints):
            self._log_progress("DONE", waypoints=f"{self._wp_index}/{len(self._waypoints)}")
            return action

        target_xy, objects, receptacle = self._waypoints[self._wp_index]
        base_xy = robot_view.base.pose[:2, 3]
        # Drive to a navigable point beside the receptacle, not into it.
        goal_xy = self._standoff_xy(receptacle, target_xy)

        # Arrival sub-sequence: open the door, place, close it again. Ramped over
        # DOOR_STEPS so the door sweeps visibly instead of snapping through objects.
        if self._stage is not None:
            if self._stage == "open":
                frac = (self._stage_step + 1) / DOOR_STEPS
                has_door = self._drive_door(receptacle, frac)
                self._log_progress("OPEN", target=receptacle.split("_")[0],
                                   frac=f"{frac:.2f}")
                self._stage_step += 1
                if not has_door or self._stage_step >= DOOR_STEPS:
                    self._stage, self._stage_step = "reach", 0
                return action
            if self._stage == "reach":
                frac = (self._stage_step + 1) / ARM_STEPS
                self._drive_arm(frac)
                self._log_progress("REACH", target=receptacle.split("_")[0],
                                   frac=f"{frac:.2f}")
                self._stage_step += 1
                if self._stage_step >= ARM_STEPS:
                    self._stage, self._stage_step = "place", 0
                return action
            if self._stage == "place":
                self._log_progress("PLACE", target=receptacle.split("_")[0],
                                   objects=len(objects))
                if objects:
                    self._teleport(objects, receptacle)
                self._stage, self._stage_step = "retract", 0
                return action
            if self._stage == "retract":
                if objects and self._stage_step % 4 == 0:
                    self._log_success_metrics(objects[0], receptacle)
                frac = 1.0 - (self._stage_step + 1) / ARM_STEPS
                self._drive_arm(frac)
                self._log_progress("RETRACT", target=receptacle.split("_")[0],
                                   frac=f"{frac:.2f}")
                self._stage_step += 1
                if self._stage_step >= ARM_STEPS:
                    self._stage, self._stage_step = "close", 0
                return action
            if self._stage == "close":
                if objects and self._stage_step % 4 == 0:
                    self._log_success_metrics(objects[0], receptacle)
                frac = 1.0 - (self._stage_step + 1) / DOOR_STEPS
                has_door = self._drive_door(receptacle, frac)
                self._log_progress("CLOSE", target=receptacle.split("_")[0],
                                   frac=f"{frac:.2f}")
                self._stage_step += 1
                if not has_door or self._stage_step >= DOOR_STEPS:
                    log.info("[nav] finished %s", receptacle)
                    self._stage, self._stage_step = None, 0
                    self._wp_index += 1
                    self._path = []
                return action

        # Plan once per waypoint, then follow the A* path cell by cell.
        if not self._path:
            self._path = self._plan_path(goal_xy)
            self._path_index = 0
            if not self._path:
                # No route: do not lunge at the target in a straight line, which is
                # what drove the robot 20 cm into a wall. Skip the waypoint instead.
                self._log_progress("NO_ROUTE", target=receptacle.split("_")[0])
                log.error("[nav] no route to %s -- skipping waypoint", receptacle)
                self._wp_index += 1
                return action

        # Advance along the path; arrival at the final cell resolves the waypoint.
        while self._path_index < len(self._path) and float(
            np.linalg.norm(self._path[self._path_index] - base_xy)
        ) < BASE_STEP:
            self._path_index += 1

        if self._path_index >= len(self._path):
            gap = float(np.linalg.norm(goal_xy - base_xy))
            if gap <= STANDOFF_TOLERANCE:
                log.info("[nav] reached %s (standoff gap %.2f m)", receptacle, gap)
                self._stage, self._stage_step = "open", 0
            else:
                log.warning("[nav] path exhausted %.2f m short of the %s standoff",
                            gap, receptacle)
                self._wp_index += 1
                self._path = []
            return action

        goal = self._path[self._path_index]
        delta = goal - base_xy
        dist = float(np.linalg.norm(delta))
        self._log_progress(
            "NAVIGATE",
            leg=f"{self._wp_index + 1}/{len(self._waypoints)}",
            pt=f"{self._path_index + 1}/{len(self._path)}",
            to_goal=f"{float(np.linalg.norm(goal_xy - base_xy)):.2f}m",
            target=receptacle.split("_")[0],
        )
        step = delta / max(dist, 1e-9) * min(BASE_STEP, dist)
        nxt = base_xy + step
        heading = float(np.arctan2(delta[1], delta[0]))
        try:
            robot_view.get_move_group("base").joint_pos = np.array(
                [nxt[0], nxt[1], heading], dtype=float
            )
        except Exception:  # noqa: BLE001 - demo only
            pass
        if "base" in action:
            action["base"] = np.array([nxt[0], nxt[1], heading], dtype=float)
        return action

        # Kinematic drive: advance the base joints directly rather than asking the
        # servo to chase a distant target.
        heading = float(np.arctan2(delta[1], delta[0]))
        step = delta / max(dist, 1e-9) * min(BASE_STEP, dist)
        goal = base_xy + step
        try:
            group = robot_view.get_move_group("base")
            group.joint_pos = np.array([goal[0], goal[1], heading], dtype=float)
        except Exception:  # noqa: BLE001 - demo only; fall back to control if the
            pass          # move group shape differs on another embodiment
        if "base" in action:
            action["base"] = np.array([goal[0], goal[1], heading], dtype=float)
        return action
