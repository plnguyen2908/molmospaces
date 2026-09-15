"""Probe reverse-undock and first-turn clearance from a recorded held posture."""
import json, os, sys
from pathlib import Path
from types import SimpleNamespace
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R
from molmo_spaces.configs.robot_configs import RBY1MConfig
from research.cross_episode_memory.kitchen_scene import make_kitchen
from research.cross_episode_memory.tools.check_dynamic_revisit import EGG, POTATO
from research.cross_episode_memory.tools.check_fridge_transfer import F, NS
from research.cross_episode_memory.tools.check_navigation_transfer import NavigationTransfer

NATIVE_POTATO = POTATO
if len(sys.argv) > 2:
    POTATO = sys.argv[2]

q=np.asarray(json.loads((Path(sys.argv[1])/'trace.json').read_text())[-1]['qpos'])
m,d,_=make_kitchen(Path(os.environ['MLSPACES_ASSETS_DIR']),RBY1MConfig(),
    (F, EGG.rsplit('_1_0_0',1)[0],NATIVE_POTATO.rsplit('_1_0_0',1)[0]),robot_xy=(.01,1.78))
c=NavigationTransfer.__new__(NavigationTransfer); c.model=m
c.args=SimpleNamespace(native_object=True); c.table_gids=set()
c.fridge_bids={b for b in range(m.nbody) if m.body(b).name.startswith(F)}
c.bread_bids={b for b in range(m.nbody) if m.body(b).name.startswith(POTATO.rsplit('_1_0_0',1)[0])}
for b in sorted(c.bread_bids): print('payload_body',m.body(b).name,'parent',m.body(m.body_parentid[b]).name,'pos',m.body_pos[b],flush=True)
ad=[m.jnt_qposadr[m.joint(NS+n).id] for n in ('base_x','base_y','base_theta')]
ba=m.jnt_qposadr[m.joint(POTATO+'_jntfree_0').id]; start=q[ad]
for back in (.3,.4,.5,.6):
 xy=start[:2]-back*np.array([np.cos(start[2]),np.sin(start[2])])
 worst=0
 for yaw in np.linspace(start[2],np.pi*3/4,20):
  rot=R.from_euler('z',yaw-start[2]);d.qpos[:]=q;d.qpos[ad]=[*xy,yaw]
  d.qpos[ba:ba+3]=rot.apply(q[ba:ba+3]-[*start[:2],0])+[*xy,0]
  d.qpos[ba+3:ba+7]=(rot*R.from_quat(q[ba+3:ba+7],scalar_first=True)).as_quat(scalar_first=True)
  mujoco.mj_forward(m,d)
  depth=c.navigation_penetration(d,True)
  if depth>worst+1e-5:
   worst=depth
   for ct in d.contact:
    if abs(-float(ct.dist)+.001-depth)<1e-6:
     print('worst',back,float(yaw),[m.body(m.geom_bodyid[g]).name for g in (ct.geom1,ct.geom2)],float(ct.dist),flush=True)
 print('undock',back,'first_turn_collision',worst,flush=True)
 for ct in d.contact:
  names=[m.body(m.geom_bodyid[g]).name for g in (ct.geom1,ct.geom2)]
  if ct.dist<-.001 and any(n.startswith(NS) for n in names) and not all(n.startswith(NS) for n in names): print('contact',names,float(ct.dist),flush=True)

from research.cross_episode_memory.tools.check_reorder_chain import PhysicalReorderCheck
check=PhysicalReorderCheck.__new__(PhysicalReorderCheck)
check.model=m;check.data=d;check.args=SimpleNamespace(native_object=True,kitchen=True,assets=Path(os.environ['MLSPACES_ASSETS_DIR']),nav_speed=.12,turn_speed=.08)
check.undock=True;check.object_joint=POTATO+'_jntfree_0'
check.bread_bids=c.bread_bids;check.fridge_bids=c.fridge_bids;check.table_gids=set()
check.record=lambda **fields: print(fields,flush=True)
d.qpos[:]=q;mujoco.mj_forward(m,d)
for back in (.3,.4,.5):
 check.args.reverse_undock=back
 try:
  route=check.plan_route(np.array([-.89,-1.02]),True,face=np.pi-.005)
  print('ROUTE',back,[p.tolist() for p in route],flush=True)
 except RuntimeError as e: print('FAILED',back,str(e),flush=True)
