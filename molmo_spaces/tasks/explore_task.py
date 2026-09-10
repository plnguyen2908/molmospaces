"""Observation-only episode: visit a set of receptacles and look at them.

Explore episodes exist because interventions are *unobserved* -- the harness moves
objects between episodes while the robot is elsewhere. Without a subsequent look,
the agent's belief about the arrangement is stale and its memory would contain an
inferred world model rather than observations.

An explore episode therefore writes a fresh **observed** snapshot into memory. That
is what makes any later restoration targeting this episode satisfy the
observability rule by construction, and what lets a perception failure (the robot
never saw the change) be told apart from a recall failure (memory had it, the
planner ignored it).

There is no manipulation here and no success predicate on object state -- only on
having been close enough to each receptacle to observe it.
"""

from typing import Any

import numpy as np

from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.env.abstract_sensors import SensorSuite
from molmo_spaces.env.data_views import create_mlspaces_body
from molmo_spaces.env.sensors import get_core_sensors
from molmo_spaces.tasks.task import BaseMujocoTask


class ExploreTask(BaseMujocoTask):
    """Succeeds once the robot base has approached every listed receptacle.

    ``task_config`` must provide:
        receptacle_names: list[str]
        succ_pos_threshold: float   (metres, base-to-receptacle)
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Visits latch: the robot must leave one receptacle to reach the next, so
        # an instantaneous "is near" check would never see them all at once.
        self._visited: dict[int, set[str]] = {}

    @property
    def receptacle_names(self) -> list[str]:
        return list(self.config.task_config.receptacle_names)

    @property
    def _radius(self) -> float:
        return float(getattr(self.config.task_config, "succ_pos_threshold", 1.5))

    def get_task_description(self) -> str:
        refs = getattr(self.config.task_config, "referral_expressions", {}) or {}
        names = [refs.get(r, r) for r in self.receptacle_names]
        if len(names) == 2:
            return f"Go and look at the {names[0]} and the {names[1]}."
        return "Go and look at the " + ", ".join(names) + "."

    def get_task_objects(self, batch_index: int = 0) -> dict[str, str]:
        task_objects = super().get_task_objects(batch_index)
        for name in self.receptacle_names:
            task_objects[name] = name
        return task_objects

    def _create_sensor_suite_from_config(self, config: MlSpacesExpConfig) -> SensorSuite:
        return SensorSuite(get_core_sensors(config))

    def reset(self):
        result = super().reset()
        self._visited.clear()
        return result

    def _update_visits(self, batch_index: int) -> set[str]:
        """Latch any receptacle the base is currently within ``_radius`` of."""
        data = self._env.mj_datas[batch_index]
        base_pos = self._env.robots[batch_index].robot_view.base.pose[:3, 3]
        seen = self._visited.setdefault(batch_index, set())
        for name in self.receptacle_names:
            if name in seen:
                continue
            try:
                receptacle = create_mlspaces_body(data, name)
            except KeyError:
                continue
            # Planar distance: base height is irrelevant to whether it drove there.
            if float(np.linalg.norm(receptacle.position[:2] - base_pos[:2])) <= self._radius:
                seen.add(name)
        return seen

    def judge_success(self) -> bool:
        return all(
            len(self._update_visits(i)) == len(self.receptacle_names)
            for i in range(self._env.n_batch)
        )

    def get_reward(self) -> np.ndarray:
        """Fraction of receptacles visited so far -- graded, for partial progress."""
        n = max(len(self.receptacle_names), 1)
        return np.array(
            [len(self._update_visits(i)) / n for i in range(self._env.n_batch)], dtype=float
        )

    def get_info(self) -> list[dict[str, Any]]:
        infos = []
        for i in range(self._env.n_batch):
            seen = self._update_visits(i)
            infos.append(
                {
                    "success": len(seen) == len(self.receptacle_names),
                    "visited": sorted(seen),
                    "receptacles": self.receptacle_names,
                    "n_visited": len(seen),
                    "episode_step": self.episode_step_count,
                }
            )
        return infos
