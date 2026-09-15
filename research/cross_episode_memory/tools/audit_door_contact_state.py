"""Inspect saved contact geometry without advancing simulation."""
import json
from pathlib import Path
import mujoco,numpy as np
from molmo_spaces.configs.robot_configs import RBY1MConfig
from research.cross_episode_memory.kitchen_scene import make_kitchen
from research.cross_episode_memory.tools.check_reorder_chain import F,EGG,POTATO,NS,JOINT
from research.cross_episode_memory.door_contact import panel_pose,HAND_BODIES
root=Path('research/cross_episode_memory/artifacts');r=json.loads((root/'reorder_right_panel_tail_v6/report.json').read_text())
m,d,_=make_kitchen(Path(r['arguments']['assets']),RBY1MConfig(),(F,EGG.rsplit('_1_0_0',1)[0],POTATO.rsplit('_1_0_0',1)[0]),robot_xy=(.01,2.08));jid=m.joint(JOINT).id;adr=m.jnt_qposadr[jid]
for name in ['reorder_right_pull_push_tail_v5','reorder_right_panel_tail_v6','reorder_panel_push_right_v1']:
 rows=json.loads((root/name/'trace.json').read_text())
 for i,row in enumerate(rows):
  if i and i<len(rows)-1 and row['stage']==rows[i-1]['stage']:continue
  d.qpos[:]=row['qpos'];mujoco.mj_forward(m,d);pose,direction,radial=panel_pose(m,d,jid,-1)
  print(name,row['stage'],round(row['time'],2),'angle',round(np.degrees(d.qpos[adr]),2),'actual',np.round(d.site(NS+'ee_site_r').xpos,3),'expected',np.round(pose[:3,3],3),'error',np.round(d.site(NS+'ee_site_r').xpos-pose[:3,3],4),'dotnormal',np.dot(d.site(NS+'ee_site_r').xpos-pose[:3,3],direction),flush=True)
  if i==len(rows)-1:
   print('finger joints',[(n,m.joint(NS+n).range.tolist(),d.joint(NS+n).qpos.tolist()) for n in ('gripper_finger_r1','gripper_finger_r2')],flush=True)
   for ct in d.contact:
    bn=[m.body(b).name for b in m.geom_bodyid[[ct.geom1,ct.geom2]]]
    if any(n in HAND_BODIES for n in bn):print('handcontact',bn,ct.dist,flush=True)
