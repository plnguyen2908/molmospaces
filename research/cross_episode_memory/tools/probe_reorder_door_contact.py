"""Probe a saved door contact posture without advancing physics or rendering."""
import argparse,json
from pathlib import Path
from types import SimpleNamespace
import mujoco,numpy as np
from scipy.spatial.transform import Rotation
from research.cross_episode_memory.door_contact import solve_contact_ik
from molmo_spaces.configs.robot_configs import RBY1MConfig
from research.cross_episode_memory.kitchen_scene import make_kitchen
from research.cross_episode_memory.tools.check_reorder_chain import PhysicalReorderCheck,F,EGG,POTATO,NS,DOOR2_JOINT
p=argparse.ArgumentParser();p.add_argument('source',type=Path);a=p.parse_args()
r=json.loads((a.source/'report.json').read_text());rows=json.loads((a.source/'trace.json').read_text())
m,d,_=make_kitchen(Path(r['arguments']['assets']),RBY1MConfig(),(F,EGG.rsplit('_1_0_0',1)[0],POTATO.rsplit('_1_0_0',1)[0]),robot_xy=(.01,1.78))
d.qpos[:]=rows[-1]['qpos'];mujoco.mj_forward(m,d)
c=PhysicalReorderCheck.__new__(PhysicalReorderCheck);c.model=m;c.data=d
names=[f'right_arm_{i}' for i in range(7)]+[f'torso_{i}' for i in range(6)];addresses=[m.jnt_qposadr[m.joint(NS+n).id] for n in names]
c.planner=SimpleNamespace(names=names);q=d.qpos[addresses].copy();pose=c.tcp();jid=m.joint(DOOR2_JOINT).id;angle=d.joint(DOOR2_JOINT).qpos[0];anchor=d.xanchor[jid].copy();axis=d.xaxis[jid].copy()
print('start',np.degrees(angle),pose.tolist(),flush=True)
for deg in np.arange(np.degrees(angle)+2,76,2):
 goal=pose.copy();rot=Rotation.from_rotvec(axis*(np.radians(deg)-angle)).as_matrix();goal[:3,3]=anchor+rot@(pose[:3,3]-anchor);goal[:3,:3]=rot@pose[:3,:3]
 try:
  d.joint(DOOR2_JOINT).qpos[0]=np.radians(deg)
  q=solve_contact_ik(m,d,goal,names,q,m.body(F+'_1_5_0').id)
 except RuntimeError as e:print('IK FAIL',deg,str(e),flush=True);break
 d.qpos[addresses]=q;d.joint(DOOR2_JOINT).qpos[0]=np.radians(deg);mujoco.mj_forward(m,d)
 bad=[]
 for contact in d.contact:
  b1,b2=m.geom_bodyid[[contact.geom1,contact.geom2]];bn=[m.body(b).name for b in (b1,b2)]
  if not any(n.startswith(NS) for n in bn) or contact.dist>=-.0005:continue
  if any('ee_finger_r' in n for n in bn) and (F+'_1_5_0' in bn):continue
  if all('ee_finger_' in n for n in bn):continue
  if any(m.geom_type[g]==mujoco.mjtGeom.mjGEOM_PLANE for g in (contact.geom1,contact.geom2)):continue
  bad.append([bn,-float(contact.dist)])
 print('angle',float(deg),'collisions',bad,flush=True)
 if bad:break
