"""Read-only collision probe of a saved chain's loaded posture."""
import json, os
from pathlib import Path
from types import SimpleNamespace
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R
from molmo_spaces.configs.robot_configs import RBY1MConfig
from research.cross_episode_memory.kitchen_scene import make_kitchen
from research.cross_episode_memory.tools.check_dynamic_revisit import EGG, POTATO
from research.cross_episode_memory.tools.check_fridge_transfer import F, NS
from research.cross_episode_memory.tools.check_reorder_chain import PhysicalReorderCheck

folder = Path(__import__('sys').argv[1])
rows = json.loads((folder / 'trace.json').read_text())
q = np.asarray(rows[-1]['qpos'])
m,d,_ = make_kitchen(Path(os.environ['MLSPACES_ASSETS_DIR']), RBY1MConfig(),
                    (F, EGG.rsplit('_1_0_0',1)[0], POTATO.rsplit('_1_0_0',1)[0]),
                    robot_xy=(.01,1.78))
check = PhysicalReorderCheck.__new__(PhysicalReorderCheck)
check.model = m; check.data = d; check.args = SimpleNamespace(native_object=True, reverse_undock=.3)
check.undock = True
check.object_name = POTATO
check.object_joint = POTATO + "_jntfree_0"
check.record = lambda **fields: print(fields, flush=True)
check.bread_bids = {b for b in range(m.nbody) if m.body(b).name.startswith(POTATO.rsplit('_1_0_0',1)[0])}
check.fridge_bids = {b for b in range(m.nbody) if m.body(b).name.startswith(F)}
check.table_gids = set()
ad = [m.jnt_qposadr[m.joint(NS+n).id] for n in ['base_x','base_y','base_theta']]
ba = m.jnt_qposadr[m.joint(POTATO+'_jntfree_0').id]
start=q[ad]; rot=R.from_euler('z',-start[2])
for x in [-.19,-.09,.01,.11]:
 for y in [1.48,1.58,1.68,1.78,1.88,1.98]:
  worst=0
  for dx,dy in [(0,0),(-.025,0),(.025,0),(0,-.025),(0,.025)]:
   d.qpos[:]=q; d.qpos[ad]=[x+dx,y+dy,0]
   d.qpos[ba:ba+3]=rot.apply(q[ba:ba+3]-[*start[:2],0])+[x+dx,y+dy,0]
   d.qpos[ba+3:ba+7]=(rot*R.from_quat(q[ba+3:ba+7],scalar_first=True)).as_quat(scalar_first=True)
   mujoco.mj_forward(m,d)
   worst=max(worst,check.navigation_penetration(d,True))
  if worst==0: print('clear',x,y,flush=True)

d.qpos[:] = q
mujoco.mj_forward(m,d)
goal = check.loaded_docking_pose(np.array([.01, 1.964142]))
assert not np.allclose(goal, [.01, 1.778498], atol=.01)
print("PASS: corrected slot-aware loaded docking", goal.tolist())
