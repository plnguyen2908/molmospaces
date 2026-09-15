"""Inspect grasp widths and approach directions at the actual settled shelf pose."""
import json
from pathlib import Path
import numpy as np,mujoco
from molmo_spaces.configs.robot_configs import RBY1MConfig
from research.cross_episode_memory.kitchen_scene import make_kitchen
from research.cross_episode_memory.tools.check_reorder_chain import POTATO,EGG,F,PhysicalReorderCheck
p=Path('research/cross_episode_memory/artifacts/reorder_chain_complete_v10');r=json.loads((p/'report.json').read_text());rows=json.loads((p/'trace.json').read_text());assets=Path(r['arguments']['assets'])
m,d,_=make_kitchen(assets,RBY1MConfig(),(F,EGG.rsplit('_1_0_0',1)[0],POTATO.rsplit('_1_0_0',1)[0]),robot_xy=(.01,1.78));d.qpos[:]=rows[-1]['qpos'];mujoco.mj_forward(m,d)
c=PhysicalReorderCheck.__new__(PhysicalReorderCheck);c.model=m;c.data=d;c.object_name=POTATO;c.bread_bids={b for b in range(m.nbody) if m.body(b).name.startswith(POTATO.rsplit('_1_0_0',1)[0])};c.object_joint=POTATO+'_jntfree_0'
local=np.load(assets/'grasps/droid/Potato_3/Potato_3_grasps_filtered.npz')['transforms'];world=c.bread_pose()@local;v=c.bread_vertices();out=[]
for i,pose in enumerate(world):
 width=np.ptp(v@pose[:3,1]);direction=pose[:3,2]
 if direction[0]>.5 and abs(pose[2,1])<.3:out.append((float(width),i,direction.tolist()))
for item in sorted(out)[:25]:print(item)
