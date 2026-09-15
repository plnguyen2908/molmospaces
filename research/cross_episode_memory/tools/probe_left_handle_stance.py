"""Check a fixed-torso left-handle pull from a stance facing its opening arc."""
import json
from pathlib import Path
from types import SimpleNamespace
import mujoco,numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from molmo_spaces.configs.robot_configs import RBY1MConfig
from research.cross_episode_memory.kitchen_scene import make_kitchen
from research.cross_episode_memory.tools.check_reorder_chain import PhysicalReorderCheck,F,EGG,POTATO,NS,DOOR2_JOINT
from research.cross_episode_memory.door_contact import contact_path_collision
p=Path('research/cross_episode_memory/artifacts/reorder_left_door_v3');cfg=json.loads((p/'report.json').read_text())['arguments']
m,d,_=make_kitchen(Path(cfg['assets']),RBY1MConfig(),(F,EGG.rsplit('_1_0_0',1)[0],POTATO.rsplit('_1_0_0',1)[0]),robot_xy=(0.,1.98))
for n,v in zip(('base_x','base_y','base_theta'),(0.,1.98,1.3)):d.joint(NS+n).qpos[0]=v
for side in ('right','left'):
 for i in range(7):d.joint(NS+f'{side}_arm_{i}').qpos[0]=-.02 if i==3 else 0.
for i in (1,2):d.joint(NS+f'gripper_finger_r{i}').qpos[0]=(-1 if i==1 else 1)*.013
mujoco.mj_forward(m,d)
c=PhysicalReorderCheck.__new__(PhysicalReorderCheck);c.model=m;c.data=d
names=[f'right_arm_{i}' for i in range(7)];addresses=[m.jnt_qposadr[m.joint(NS+n).id] for n in names];bounds=m.jnt_range[[m.joint(NS+n).id for n in names]]
anchor=d.xanchor[m.joint(DOOR2_JOINT).id].copy();axis=d.xaxis[m.joint(DOOR2_JOINT).id].copy();pose=np.eye(4);pose[:3,:3]=(Rotation.from_euler('y',90,degrees=True)*Rotation.from_euler('z',180,degrees=True)).as_matrix();root=d.body(F+'_1_0_0').xpos
pose[:3,3]=[root[0]-.387,root[1]+.104711,cfg['grasp_height']+root[2]-1.21971]
q=np.array([.0,.0,.0,-1.2,0.,0.,0.]);out=[]
for deg in range(0,76,5):
 rot=Rotation.from_rotvec(axis*np.radians(deg)).as_matrix();goal=pose.copy();goal[:3,:3]=rot@pose[:3,:3];goal[:3,3]=anchor+rot@(pose[:3,3]-anchor)
 def residual(q):
  d.qpos[addresses]=q;mujoco.mj_kinematics(m,d);t=c.tcp();return np.r_[t[:3,3]-goal[:3,3],.3*Rotation.from_matrix(goal[:3,:3]@t[:3,:3].T).as_rotvec(),.001*(q-seed)]
 seed=q.copy();sol=least_squares(residual,np.clip(q,bounds[:,0]+.02,bounds[:,1]-.02),bounds=(bounds[:,0]+.02,bounds[:,1]-.02),max_nfev=150);q=sol.x
 d.qpos[addresses]=q;d.joint(DOOR2_JOINT).qpos[0]=np.radians(deg);mujoco.mj_forward(m,d)
 error=float(np.linalg.norm(residual(q)[:6]));bad=contact_path_collision(m,d,m.body(F+'_1_5_0').id)
 row=dict(angle=deg,error=error,collision=bad,joints=q.tolist(),tcp=goal.tolist());out.append(row);print(json.dumps(row),flush=True)
 if error>.004 or bad:break
(p/'facing_handle_pull.json').write_text(json.dumps(out,indent=2))
