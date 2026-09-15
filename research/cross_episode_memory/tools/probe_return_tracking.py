"""Inspect tracking constraints in a saved failed return approach."""
import json
from pathlib import Path
import mujoco,numpy as np
from molmo_spaces.configs.robot_configs import RBY1MConfig
from research.cross_episode_memory.kitchen_scene import make_kitchen
from research.cross_episode_memory.tools.check_reorder_chain import EGG,POTATO,F,NS
p=Path('research/cross_episode_memory/artifacts/reorder_return_egg_mesh_v2');r=json.loads((p/'report.json').read_text());rows=json.loads((p/'trace.json').read_text());m,d,_=make_kitchen(Path(r['arguments']['assets']),RBY1MConfig(),(F,EGG.rsplit('_1_0_0',1)[0],POTATO.rsplit('_1_0_0',1)[0]),robot_xy=(.01,1.78));d.qpos[:]=rows[-1]['qpos'];mujoco.mj_forward(m,d)
for j in range(m.njnt):
 if m.joint(j).name.startswith(NS+'torso'):
  adr=m.jnt_dofadr[j];print(m.joint(j).name,'q',d.qpos[m.jnt_qposadr[j]],'stiffness',m.jnt_stiffness[j],'bias',d.qfrc_bias[adr],'passive',d.qfrc_passive[adr],'gravcomp',d.qfrc_gravcomp[adr],'constraint',d.qfrc_constraint[adr])
for idx,c in enumerate(d.contact):
 names=[m.body(m.geom_bodyid[g]).name for g in (c.geom1,c.geom2)]
 if all(n.startswith(NS) for n in names):
  force=np.zeros(6);mujoco.mj_contactForce(m,d,idx,force)
  if force[0]>1:print('self contact',names,'depth',c.dist,'force',force[0],c.pos.tolist())
print('robot gravcomp',set(m.body_gravcomp[b] for b in range(m.nbody) if m.body(b).name.startswith(NS)))
