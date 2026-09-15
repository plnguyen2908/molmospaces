"""Read-only search for a hand posture beside the inside face of a door."""
import argparse,json
from pathlib import Path
from types import SimpleNamespace
import mujoco,numpy as np
from scipy.spatial.transform import Rotation
from molmo_spaces.configs.robot_configs import RBY1MConfig
from research.cross_episode_memory.kitchen_scene import make_kitchen
from research.cross_episode_memory.tools.check_reorder_chain import PhysicalReorderCheck,F,EGG,POTATO,NS,DOOR2_JOINT,JOINT,collision_mesh
p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('--side',default='left');p.add_argument('--other-open',type=float,default=0.);p.add_argument('--closing',action='store_true');a=p.parse_args()
r=json.loads((a.source/'report.json').read_text());m,d,_=make_kitchen(Path(r['arguments']['assets']),RBY1MConfig(),(F,EGG.rsplit('_1_0_0',1)[0],POTATO.rsplit('_1_0_0',1)[0]),robot_xy=(.01,2.08))
c=PhysicalReorderCheck.__new__(PhysicalReorderCheck);c.model=m;c.data=d
names=[f'right_arm_{i}' for i in range(7)]+[f'torso_{i}' for i in range(6)];addresses=[m.jnt_qposadr[m.joint(NS+n).id] for n in names];c.planner=SimpleNamespace(names=names)
for side_name in ('left','right'):
 for i in range(7):d.joint(NS+f'{side_name}_arm_{i}').qpos[0]=-.02 if i==3 else 0.
for i in (1,2):d.joint(NS+f'gripper_finger_r{i}').qpos[0]=0
mujoco.mj_forward(m,d);tcp=c.tcp();fv=collision_mesh(m,d,lambda b:'ee_finger_r' in m.body(b).name)[0];print('fingers tcp bounds',((fv-tcp[:3,3])@tcp[:3,:3]).min(0),((fv-tcp[:3,3])@tcp[:3,:3]).max(0),flush=True)
side=a.side;sign=1 if side=='left' else -1;joint=DOOR2_JOINT if sign==1 else JOINT;panel=m.body(F+('_1_4_0' if sign==1 else '_1_2_0')).id;handle=m.body(F+('_1_5_0' if sign==1 else '_1_3_0')).id;jid=m.joint(joint).id
d.joint(JOINT if side=='left' else DOOR2_JOINT).qpos[0]=(-1 if side=='left' else 1)*np.radians(a.other_open)
original=d.qpos.copy();out=[]
for angle in ((75,45,0) if a.closing else (45,75)):
 d.qpos[:]=original;d.joint(joint).qpos[0]=sign*np.radians(angle);mujoco.mj_forward(m,d)
 anchor=d.xanchor[jid].copy();radial=Rotation.from_rotvec(d.xaxis[jid]*sign*np.radians(angle)).apply([0,-sign,0]);tangent=np.cross(d.xaxis[jid]*sign,radial)*(-1. if a.closing else 1.)
 vertices=collision_mesh(m,d,lambda b:b==panel)[0];bottom=float(vertices[:,2].min());print('panel bottom',bottom,flush=True)
 for height in (1.4,):
  point=anchor+radial*.42;point[2]=height;start=point-tangent*.35
  hits=[mujoco.mj_rayMesh(m,d,g,start,tangent) for g in range(m.ngeom) if m.geom_bodyid[g]==panel and (m.geom_contype[g] or m.geom_conaffinity[g]) and m.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH];hits=[h for h in hits if h>=0]
  if not hits:
   print('no panel ray',angle,height,flush=True);continue
  surface=start+min(hits)*tangent
  goal=np.eye(4);goal[:3,2]=(-radial+tangent)/np.sqrt(2);goal[:3,0]=[0,0,1];goal[:3,1]=np.cross(goal[:3,2],goal[:3,0]);goal[:3,3]=surface-tangent*.030
  for x,y,yaw in (((.05,1.95,.5),(.05,2.15,1.2)) if a.closing else ((.05,1.95,.5),)) if side=='left' else ((.05,1.90,-.35),):
   d.qpos[:]=original;d.joint(joint).qpos[0]=sign*np.radians(angle)
   for n,v in zip(('base_x','base_y','base_theta'),(x,y,yaw)):d.joint(NS+n).qpos[0]=v
   for n in names:d.joint(NS+n).qpos[0]=0 if n!='right_arm_3' else -1.2
   mujoco.mj_forward(m,d);q=d.qpos[addresses].copy()
   try:
    for _ in range(12):q=c.nearby_ik(goal,q,names)
   except RuntimeError:
    # Global positioning can need more than the local IK's half-radian trust region.
    from scipy.optimize import least_squares
    bounds=m.jnt_range[[m.joint(NS+n).id for n in names]]
    def residual(q):
     d.qpos[addresses]=q;mujoco.mj_kinematics(m,d);tcp=c.tcp();return np.r_[tcp[:3,3]-goal[:3,3],.3*Rotation.from_matrix(goal[:3,:3]@tcp[:3,:3].T).as_rotvec(),.002*(q-np.array([-1.2 if n=='right_arm_3' else 0. for n in names]))]
    sol=least_squares(residual,np.clip(q,bounds[:,0]+.02,bounds[:,1]-.02),bounds=(bounds[:,0]+.02,bounds[:,1]-.02),max_nfev=150)
    if np.linalg.norm(residual(sol.x)[:6])>.003:
     print('ik fail',angle,height,(x,y,yaw),float(np.linalg.norm(residual(sol.x)[:6])),flush=True);continue
    q=sol.x
   # Refine against actual collision geometry, keeping a modest torso posture.
   from scipy.optimize import least_squares
   bounds=m.jnt_range[[m.joint(NS+n).id for n in names]]
   robot_bids=[b for b in range(m.nbody) if m.body(b).name.startswith(NS)]
   robot_index={b:i for i,b in enumerate(robot_bids)}
   preferred=np.array([-1.2 if n=='right_arm_3' else 0. for n in names])
   weights=np.array([.002 if n.startswith('right_arm') else .012 for n in names])
   def clear_residual(q):
    d.qpos[addresses]=q;mujoco.mj_forward(m,d);tcp=c.tcp()
    collision=np.zeros(len(robot_bids))
    for ct in d.contact:
     if ct.dist>=-.0001:continue
     b1,b2=m.geom_bodyid[[ct.geom1,ct.geom2]]
     if b1 not in robot_index and b2 not in robot_index:continue
     if all('ee_finger_' in m.body(b).name for b in (b1,b2)):continue
     if any(m.geom_type[g]==mujoco.mjtGeom.mjGEOM_PLANE for g in (ct.geom1,ct.geom2)):continue
     for b in (b1,b2):
      if b in robot_index:collision[robot_index[b]]=max(collision[robot_index[b]],-ct.dist+.0005)
    return np.r_[tcp[:3,3]-goal[:3,3],.3*Rotation.from_matrix(goal[:3,:3]@tcp[:3,:3].T).as_rotvec(),weights*(q-preferred),5*collision]
   sol=least_squares(clear_residual,np.clip(q,bounds[:,0]+.02,bounds[:,1]-.02),bounds=(bounds[:,0]+.02,bounds[:,1]-.02),max_nfev=120,ftol=1e-7)
   if np.linalg.norm(clear_residual(sol.x)[:6])>.004:
    print('collision-aware IK error',angle,height,np.linalg.norm(clear_residual(sol.x)[:6]),flush=True);continue
   q=sol.x
   d.qpos[addresses]=q;mujoco.mj_forward(m,d)
   bad=[];touch=[]
   for contact in d.contact:
    b1,b2=m.geom_bodyid[[contact.geom1,contact.geom2]];bn=[m.body(b).name for b in (b1,b2)]
    if not any(n.startswith(NS) for n in bn) or contact.dist>=0:continue
    if any(('ee_finger_r' in n or n == NS+'EE_BODY_R') for n in bn) and panel in (b1,b2):touch.append(-float(contact.dist));continue
    if all('ee_finger_' in n for n in bn):continue
    if any(m.geom_type[g]==mujoco.mjtGeom.mjGEOM_PLANE for g in (contact.geom1,contact.geom2)):continue
    if contact.dist<-.0005:bad.append([bn,-float(contact.dist)])
   if bad:print('collision',angle,height,(x,y,yaw),max(bad,key=lambda v:v[1]),flush=True)
   if not bad and max(touch,default=0.)<=.001:
    item={'closing':a.closing,'other_open':a.other_open,'radius':.42,'clearance':.03,'angle':angle,'height':height,'stance':[x,y,yaw],'goal':goal.tolist(),'joints':list(q),'finger_depths':touch};out.append(item);print(json.dumps(item),flush=True)
print('feasible',len(out),flush=True)
path=a.source/(('close_postures_' if a.closing else 'push_postures_')+side+'.json');path.write_text(json.dumps(out,indent=2))
