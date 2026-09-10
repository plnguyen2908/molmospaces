"""One episode = one whole *run*: a sequence of instructions executed in order.

This replaces the earlier design of one `EpisodeSpec` per episode chained by
re-authoring. That design had two flaws:

1. **State was fictional.** The chain was generated up front assuming every work
   instruction succeeded. A failed pick left the object in place in simulation but
   the next episode still started as though it had moved.
2. **Seven scene loads.** Scene construction dominates wall-clock (~2 min), so an
   N-step run cost N loads.

Here the environment is built once and instructions are issued one at a time,
advancing when the step's predicate passes or its attempt budget is spent. State
carries because it is never reset: a failure leaves the world as the failure left
it, which is what the study's "objects move as a byproduct of the work" requires.
Interventions are applied in place between steps.

**Everything is logged.** Each tick records the active instruction, the base pose,
distance to the current navigation target, and the per-object support verdict, so a
failure can be attributed to navigation, manipulation, or the predicate rather than
guessed at. `get_info()` exposes the log and `write_log()` dumps it to JSON.
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.env.abstract_sensors import SensorSuite
from molmo_spaces.env.data_views import create_mlspaces_body
from molmo_spaces.env.sensors import get_core_sensors
from molmo_spaces.tasks.reorder_task import SupportTracker
from molmo_spaces.tasks.task import BaseMujocoTask

log = logging.getLogger(__name__)

WORK, EXPLORE, RESTORE = "work", "explore", "restore"


class RunTask(BaseMujocoTask):
    """Execute an ordered script of instructions inside a single episode.

    ``task_config`` provides:
        steps: list[dict]      each {"kind": work|explore|restore, ...}
        interventions: list[dict]  {"before_step": int, "moves": [{object, from, to}]}
        tracked_objects: list[str]
        receptacles: list[str]
        max_steps_per_instruction: int
        arrival_radius: float
        log_path: str | None
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._support = SupportTracker()
        self._step_index = 0
        self._steps_on_current = 0
        self._records: list[dict[str, Any]] = []
        self._ticks: list[dict[str, Any]] = []
        self._last_tick_at = -1
        self._applied_interventions: set[int] = set()
        # Fail-fast: a failed instruction invalidates every later step's
        # precondition, because the arrangement is no longer what the script
        # assumed. Continuing would produce meaningless results, so the run
        # terminates instead.
        self._run_failed = False
        self._failed_at: dict[str, Any] | None = None

    # ---------------------------------------------------------------- config

    @property
    def steps(self) -> list[dict[str, Any]]:
        return list(getattr(self.config.task_config, "steps", []) or [])

    @property
    def _max_steps(self) -> int:
        return int(getattr(self.config.task_config, "max_steps_per_instruction", 120))

    @property
    def _arrival_radius(self) -> float:
        return float(getattr(self.config.task_config, "arrival_radius", 0.55))

    @property
    def current_step(self) -> dict[str, Any] | None:
        """The instruction the policy should be executing right now."""
        if self._step_index >= len(self.steps):
            return None
        return self.steps[self._step_index]

    def get_task_description(self) -> str:
        step = self.current_step
        if step is None:
            return "Run complete."
        return step.get("instruction", step.get("kind", "step"))

    def get_task_objects(self, batch_index: int = 0) -> dict[str, str]:
        objs = super().get_task_objects(batch_index)
        cfg = self.config.task_config
        for name in list(getattr(cfg, "tracked_objects", []) or []) + list(
            getattr(cfg, "receptacles", []) or []
        ):
            objs[name] = name
        return objs

    def _create_sensor_suite_from_config(self, config: MlSpacesExpConfig) -> SensorSuite:
        # No privileged object-state sensors: the run's ground truth is logged by
        # the task itself, never handed to the policy. See the no-cheat rule.
        return SensorSuite(get_core_sensors(config))

    def reset(self):
        result = super().reset()
        self._support.reset()
        self._step_index = 0
        self._steps_on_current = 0
        self._records.clear()
        self._ticks.clear()
        self._applied_interventions.clear()
        self._run_failed = False
        self._failed_at = None
        self._apply_interventions_for(0)
        return result

    # ------------------------------------------------------------ mechanics

    def _xy(self, body_name: str) -> np.ndarray | None:
        try:
            return create_mlspaces_body(self._env.mj_datas[0], body_name).position[:2].copy()
        except KeyError:
            log.warning("RunTask: body %r not found", body_name)
            return None

    def _apply_interventions_for(self, step_index: int) -> None:
        """Scripted, unobserved world changes applied in place between steps."""
        cfg = self.config.task_config
        for i, iv in enumerate(getattr(cfg, "interventions", []) or []):
            if int(iv.get("before_step", -1)) != step_index or i in self._applied_interventions:
                continue
            self._applied_interventions.add(i)
            data = self._env.mj_datas[0]
            for mv in iv.get("moves", []):
                dest = self._xy(mv["to"])
                if dest is None:
                    continue
                try:
                    body = create_mlspaces_body(data, mv["object"])
                    rec = create_mlspaces_body(data, mv["to"])
                    body.position = rec.position + np.array([0.0, 0.0, 0.55])
                except KeyError:
                    log.warning("RunTask: intervention body missing: %r", mv)
                    continue
                self._records.append(
                    {
                        "type": "intervention",
                        "before_step": step_index,
                        "object": mv["object"],
                        "from": mv.get("from"),
                        "to": mv["to"],
                        "observed": False,
                        "episode_step": self.episode_step_count,
                    }
                )
                log.info("RunTask INTERVENTION %s -> %s", mv["object"], mv["to"])

    def _support_of(self, obj: str, receptacle: str) -> bool:
        frac = float(getattr(self.config.task_config, "receptacle_supported_weight_frac", 0.5))
        return self._support.is_supported(
            self._env, 0, obj, receptacle, frac_weight_threshold=frac, task=self
        )

    def _step_satisfied(self, step: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        """Evaluate the current instruction's predicate and return diagnostics."""
        kind = step.get("kind")
        if kind == WORK:
            ok = self._support_of(step["object"], step["to"])
            return ok, {"object": step["object"], "to": step["to"], "supported": ok}
        if kind == RESTORE:
            per = {o: self._support_of(o, r) for o, r in step["target_assignment"].items()}
            return all(per.values()), {"per_object": per,
                                       "n_restored": int(sum(per.values())),
                                       "n_tracked": len(per)}
        if kind == EXPLORE:
            base = self._env.robots[0].robot_view.base.pose[:2, 3]
            visited = set(step.setdefault("_visited", []))
            dists = {}
            for rec in step["receptacles"]:
                xy = self._xy(rec)
                if xy is None:
                    continue
                d = float(np.linalg.norm(xy - base))
                dists[rec] = round(d, 3)
                if d <= self._arrival_radius:
                    visited.add(rec)
            step["_visited"] = sorted(visited)
            ok = len(visited) == len(step["receptacles"])
            return ok, {"visited": sorted(visited), "distances": dists}
        return False, {"error": f"unknown step kind {kind!r}"}

    def _tick(self) -> None:
        """Advance the state machine at most once per environment step."""
        if self.episode_step_count == self._last_tick_at:
            return
        self._last_tick_at = self.episode_step_count

        step = self.current_step
        if step is None or self._run_failed:
            return

        satisfied, diag = self._step_satisfied(step)
        base = self._env.robots[0].robot_view.base.pose[:2, 3]
        self._ticks.append(
            {
                "episode_step": self.episode_step_count,
                "step_index": self._step_index,
                "kind": step.get("kind"),
                "instruction": step.get("instruction"),
                "base_xy": [round(float(base[0]), 3), round(float(base[1]), 3)],
                "satisfied": satisfied,
                **diag,
            }
        )
        self._steps_on_current += 1

        timed_out = self._steps_on_current >= self._max_steps
        if satisfied or timed_out:
            self._records.append(
                {
                    "type": "step_result",
                    "step_index": self._step_index,
                    "kind": step.get("kind"),
                    "instruction": step.get("instruction"),
                    "success": bool(satisfied),
                    "timed_out": bool(timed_out and not satisfied),
                    "steps_taken": self._steps_on_current,
                    "episode_step": self.episode_step_count,
                    "diagnostics": diag,
                }
            )
            log.info(
                "RunTask step %d (%s) %s after %d steps",
                self._step_index,
                step.get("kind"),
                "SUCCESS" if satisfied else "FAILED",
                self._steps_on_current,
            )
            if not satisfied:
                # Terminate the whole run. Do NOT advance -- later steps assume an
                # arrangement that no longer holds.
                self._run_failed = True
                self._failed_at = {
                    "step_index": self._step_index,
                    "kind": step.get("kind"),
                    "instruction": step.get("instruction"),
                    "steps_taken": self._steps_on_current,
                    "diagnostics": diag,
                }
                log.error(
                    "RunTask FAILED at step %d (%s): %s",
                    self._step_index,
                    step.get("kind"),
                    diag,
                )
                return
            self._step_index += 1
            self._steps_on_current = 0
            self._apply_interventions_for(self._step_index)

    # -------------------------------------------------------------- outputs

    def judge_success(self) -> bool:
        """The run succeeds only if EVERY instruction succeeded, in order."""
        self._tick()
        if self._run_failed:
            return False
        results = [r for r in self._records if r["type"] == "step_result"]
        return len(results) == len(self.steps) and all(r["success"] for r in results)

    def is_terminal(self) -> np.ndarray:
        """Terminate on first failure, or when the whole script is complete."""
        self._tick()
        done = self._run_failed or self._step_index >= len(self.steps)
        return np.array([bool(done)] * self._env.n_batch)

    def get_reward(self) -> np.ndarray:
        """Fraction of instructions completed successfully so far."""
        self._tick()
        results = [r for r in self._records if r["type"] == "step_result"]
        done = sum(1 for r in results if r["success"])
        return np.array([done / max(len(self.steps), 1)], dtype=float)

    def get_info(self) -> list[dict[str, Any]]:
        self._tick()
        results = [r for r in self._records if r["type"] == "step_result"]
        step = self.current_step
        return [
            {
                "success": self.judge_success(),
                "step_index": self._step_index,
                "n_steps": len(self.steps),
                "current_kind": step.get("kind") if step else None,
                "current_instruction": step.get("instruction") if step else None,
                "steps_on_current": self._steps_on_current,
                "completed": len(results),
                "succeeded": sum(1 for r in results if r["success"]),
                "run_failed": self._run_failed,
                "failed_at": self._failed_at,
                "episode_step": self.episode_step_count,
            }
        ]

    def write_log(self, path: str | Path | None = None) -> Path | None:
        """Dump the full run log: per-step results, interventions, per-tick trace."""
        path = path or getattr(self.config.task_config, "log_path", None)
        if not path:
            return None
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "steps": self.steps,
                    "run_failed": self._run_failed,
                    "failed_at": self._failed_at,
                    "records": self._records,
                    "ticks": self._ticks,
                },
                indent=2,
                default=str,
            )
            + "\n"
        )
        log.info("RunTask log written to %s", out)
        return out
