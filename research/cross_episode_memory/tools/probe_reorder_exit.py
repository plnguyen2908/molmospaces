"""Inspect actual robot/environment clearance at a saved released boundary."""
import json,os,sys
from pathlib import Path
from types import SimpleNamespace
import mujoco,numpy as np
from molmo_spaces.configs.robot_configs import RBY1MConfig
from research.cross_episode_memory.kitchen_scene import make_kitchen
from research.cross_episode_memory.tools.check_dynamic_revisit import EGG,POTATO
from research.cross_episode_memory.tools.check_fridge_transfer import F,NS
from research.cross_episode_memory.tools.check_navigation_transfer import NavigationTransfer
p=Path(sys.argv[1]);r=json.loads((p/'report.json').read_text());q=np.array(json.loads((p/'trace.json').read_text())[-1]['qpos'])
m,d,_=make_kitchen(Path(r['arguments']['assets']),RBY1MConfig(),(F,EGG.rsplit('_1_0_0',1)[0],POTATO.rsplit('_1_0_0',1)[0]),robot_xy=(.01,1.78))
c=NavigationTransfer.__new__(NavigationTransfer);c.model=m;c.args=SimpleNamespace(native_object=True);c.table_gids=set();c.bread_bids={b for b in range(m.nbody) if m.body(b).name.startswith(EGG.rsplit('_1_0_0',1)[0])};c.fridge_bids={b for b in range(m.nbody) if m.body(b).name.startswith(F)}
ad=[m.jnt_qposadr[m.joint(NS+n).id] for n in ('base_x','base_y','base_theta')];start=q[ad]
for back in (0,.1,.2,.3):
 worst=0;hits=[]
 for dx,dy in ((0,0),(-.025,0),(.025,0),(0,-.025),(0,.025)):
  d.qpos[:]=q;d.qpos[ad[:2]]=start[:2]-back*np.array([np.cos(start[2]),np.sin(start[2])])+[dx,dy];mujoco.mj_forward(m,d)
  depth=c.navigation_penetration(d,False)
  if depth>worst:
   worst=depth;hits=[(m.body(m.geom_bodyid[ct.geom1]).name,m.body(m.geom_bodyid[ct.geom2]).name,float(ct.dist)) for ct in d.contact if abs(-ct.dist+.001-depth)<1e-6]
 print(back,worst,hits,flush=True)
