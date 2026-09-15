"""Check full-task dispatch and failure gates without starting a simulation."""

import shlex
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from research.cross_episode_memory.tools.check_annotated_fridge_task import AnnotatedFridgeTask
from research.cross_episode_memory.tools.check_navigation_transfer import (
    NavigationTransfer,
    parse_args,
)


class FridgeWiringTest(unittest.TestCase):
    def make_task(self):
        script = Path(__file__).with_name("run_native_closed_fridge_task.sh").read_text()
        tokens = shlex.split(script[script.index('exec "$python_bin"') :].replace("\\\n", " "))
        values = {
            "$MLSPACES_ASSETS_DIR": "/nobackup2/le/.cache/molmospaces/assets/L25vYmFja3VwMi9sZS9tb2xtb3NwYWNlcw",
            "$output_dir": "/tmp/fridge-wiring-test-unused",
        }
        args = parse_args([values.get(t, t) for t in tokens[3:]])

        def init_without_physics(check, parsed):
            check.args = parsed
            check.report = {}
            check.model = SimpleNamespace(
                actuator=lambda _: SimpleNamespace(id=0), actuator_forcerange=np.zeros((1, 2))
            )

        with patch.object(NavigationTransfer, "__init__", init_without_physics):
            return AnnotatedFridgeTask(args)

    def test_launcher_and_open_before_navigation(self):
        task = self.make_task()
        self.assertTrue(task.args.operate_door)
        self.assertFalse(task.args.pickup_only or task.args.carry_only)
        self.assertIn("egg_", task.args.object_name)
        self.assertEqual(task.args.pickup_stance_y, -1.05)
        events = []
        task.open_fridge = lambda: events.append("open")
        task.pickup_stance = lambda: np.array([-0.91, -1.05])
        task.bread_pose = lambda: np.eye(4)
        task.navigate = lambda *a, **kw: events.append("navigate_to_object")
        task.prepare_pickup()
        self.assertEqual(events, ["open", "navigate_to_object"])
        self.assertEqual(task.grasp_force_limit_n, 10.0)

    def test_close_then_check_object(self):
        task = self.make_task()
        events = []
        task.close_native_fridge = lambda: events.append("close")
        task.check_final_payload = lambda: events.append("validate_object")
        task.after_placement()
        self.assertEqual(events, ["close", "validate_object"])

    def test_final_dwell_failure_revokes_success(self):
        task = self.make_task()
        task.report["success"] = True

        def fail():
            raise RuntimeError("object rolled off shelf")

        task.check_final_payload = fail
        with patch.object(NavigationTransfer, "finalize_report", lambda _: None):
            task.finalize_report()
        self.assertFalse(task.report["success"])
        self.assertIn("rolled off", task.report["error"])


if __name__ == "__main__":
    unittest.main()
