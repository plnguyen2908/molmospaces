"""Read-only base-clearance and handle-IK search from a recorded open-door state."""
import argparse,json
from pathlib import Path
from types import SimpleNamespace
import mujoco,numpy as np
from scipy.spatial.transform import Rotation
from molmo_spaces.configs.robot_configs import RBY1MConfig
from research.cross_episode_memory.kitchen_scene import make_kitchen
from research.cross_episode_memory.tools.check_reorder_chain import PhysicalReorderCheck,F,EGG,POTATO,NS,DOOR2_JOINT,JOINT,HANDLE
from research.cross_episode_memory.tools.check_navigation_transfer import NavigationTransfer
from research.cross_episode_memory.door_contact import solve_contact_ik,contact_path_collision
p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('--side',default='left');a=p.parse_args()
r=json.loads((a.source/'report.json').read_text());rows=json.loads((a.source/'trace.json').read_text())
m,d,_=make_kitchen(Path(r['arguments']['assets']),RBY1MConfig(),(F,EGG.rsplit('_1_0_0',1)[0],POTATO.rsplit('_1_0_0',1)[0]),robot_xy=(.01,2.08));d.qpos[:]=rows[-1]['qpos'];d.qvel[:]=0
c=PhysicalReorderCheck.__new__(PhysicalReorderCheck);c.model=m;c.data=d;c.args=SimpleNamespace(**r['arguments']);c.args.assets=Path(c.args.assets);c.active_door=a.side;c.door_open_sign=1 if a.side=='left' else -1
c.jid=m.joint(DOOR2_JOINT if a.side=='left' else JOINT).id;c.handle_bid=m.body(F+'_1_5_0' if a.side=='left' else HANDLE).id
names=[f'right_arm_{i}' for i in range(7)]+[f'torso_{i}' for i in range(6)];addresses=[m.jnt_qposadr[m.joint(NS+n).id] for n in names];c.planner=SimpleNamespace(names=names)
for n in names:d.joint(NS+n).qpos[0]=-.02 if n=='right_arm_3' else 0.
for n in ('gripper_finger_r1','gripper_finger_r2'):
 limits=m.joint(NS+n).range;d.joint(NS+n).qpos[0]=float(limits[np.argmax(abs(limits))])
mujoco.mj_forward(m,d);original=d.qpos.copy();angle=c.angle()
probe=mujoco.MjData(m);probe.qpos[:]=d.qpos;probe.joint(JOINT).qpos[0]=0;probe.joint(DOOR2_JOINT).qpos[0]=0;mujoco.mj_forward(m,probe)
c.closed_handles={side:probe.body(F+suffix).xpos.copy() for side,suffix in [('left','_1_5_0'),('right','_1_3_0')]}
c.right_closed_grasp=NavigationTransfer.handle_grasp_pose(c)
print('angle',np.degrees(angle),'hinge damping',m.dof_damping[m.jnt_dofadr[c.jid]],'friction',m.dof_frictionloss[m.jnt_dofadr[c.jid]],flush=True)
goal=c.live_handle_grasp(side_on=True);goal[2,3]=1.4;pre=goal.copy();pre[:3,3]-=.10*pre[:3,2]
c.record=lambda **kw:None;on_floor=c.nav_map_filter();out=[]
for x,y,yaw in [(x,y,yaw) for x in (.05,-.05) for y in (1.95,2.05) for yaw in (.5,1.1)]:
 d.qpos[:]=original
 for n,v in zip(('base_x','base_y','base_theta'),(x,y,yaw)):d.joint(NS+n).qpos[0]=v
 if on_floor and not on_floor((x,y,yaw)):continue
 mujoco.mj_forward(m,d)
 if contact_path_collision(m,d,-1):continue
 seed=np.array([-1.2 if n=='right_arm_3' else 0. for n in names]);probe.qpos[:]=d.qpos
 try:
  q=solve_contact_ik(m,probe,pre,names,seed,-1,trust_radius=10.,max_nfev=120)
  pre_q=q.copy()
  for f in np.linspace(0,1,16)[1:]:
   target=pre.copy();target[:3,3]=(1-f)*pre[:3,3]+f*goal[:3,3]
   q=solve_contact_ik(m,probe,target,names,q,c.handle_bid)
  item=dict(stance=[x,y,yaw],angle=np.degrees(angle),goal=goal.tolist(),pre_joints=pre_q.tolist(),grasp_joints=q.tolist());out.append(item)
  print(json.dumps(item),flush=True)
  if len(out)>=3:break
 except RuntimeError as exc:print('reject',[x,y,yaw],str(exc),contact_path_collision(m,probe,c.handle_bid),flush=True)
(a.source/('close_handle_stances_'+a.side+'.json')).write_text(json.dumps(out,indent=2))
print('feasible',len(out),flush=True)
