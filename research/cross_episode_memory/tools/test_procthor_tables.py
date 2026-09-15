"""Safety checks for the table adapter's support and simultaneous-change logic."""
import unittest
import mujoco
import numpy as np
from research.cross_episode_memory.tools.check_procthor_reorder import TableReorder


class TableAdapterTests(unittest.TestCase):
    def make(self):
        model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
          <body name="table_a"><geom type="box" size=".5 .5 .25" pos="0 0 .25"/>
            <site name="site_a" pos="0 0 .5"/></body>
          <body name="table_b" pos="2 0 0"><geom type="box" size=".5 .5 .25" pos="0 0 .25"/>
            <site name="site_b" pos="0 0 .5"/></body>
          <body name="one" pos="0 0 .55"><freejoint/><geom type="box" size=".05 .05 .05"/></body>
          <body name="two" pos="2 0 .55"><freejoint/><geom type="box" size=".05 .05 .05"/></body>
        </worldbody></mujoco>''')
        check = TableReorder.__new__(TableReorder)
        check.model = model; check.data = mujoco.MjData(model)
        check.objects = ('one', 'two'); check.receptacles = ('table_a', 'table_b')
        check.table_bids = {name: check.descendants(name) for name in check.receptacles}
        check.group_active = False; check.holding_loaf = False
        for _ in range(50):
            mujoco.mj_step(model, check.data)
        return check

    def test_assignment_requires_actual_upward_table_contact(self):
        check = self.make()
        self.assertEqual(check.assignment(), {'one': 'table_a', 'two': 'table_b'})
        check.data.qpos[2] += .2
        mujoco.mj_forward(check.model, check.data)
        with self.assertRaisesRegex(RuntimeError, 'support'):
            check.assignment()

    def test_occupied_dynamic_pose_rejected_without_mutating_live_scene(self):
        check = self.make()
        check.demonstrated = {'one': {'table_b': check.data.qpos[7:14].copy()}}
        before = check.data.qpos.copy()
        with self.assertRaisesRegex(RuntimeError, 'occupied'):
            check.intervene({'one': 'table_b'})
        np.testing.assert_array_equal(check.data.qpos, before)

    def test_revisit_handles_empty_table(self):
        check = self.make()
        check.data.qpos[7:10] = [.2, 0., .55]
        for _ in range(50):
            mujoco.mj_step(check.model, check.data)
        check.selection = {'tables': [{'sites': ['site_a']}, {'sites': ['site_b']}]}
        check.select_object = lambda obj: setattr(check, 'object_name', obj)
        check.bread_pose = lambda: np.eye(4)
        check.stance = lambda table, point: np.array([0., 0., 0.])
        visits = []
        check.navigate = lambda *args, **kwargs: visits.append(args)
        check.tick = lambda seconds: None
        check.record = lambda **kwargs: None
        check.gaze_error_deg = lambda: 0.
        observed = check.observe(check.receptacles)
        self.assertEqual(observed, {'one': 'table_a', 'two': 'table_a'})
        self.assertEqual(len(visits), 2)
        self.assertIsNone(check.inspection_target)


if __name__ == '__main__':
    unittest.main()
