"""Audit a panel approach plan against simulator joint limits and forward kinematics."""
import json
from pathlib import Path
from types import SimpleNamespace
import mujoco,numpy as np
from scipy.spatial.transform import Rotation
from research.cross_episode_memory.tools.check_reorder_chain import PhysicalReorderCheck,NS
from research.cross_episode_memory.door_contact import panel_pose,contact_path_collision
src=Path('research/cross_episode_memory/artifacts/reorder_right_pull_push_tail_v2')
r=json.loads((src/'report.json').read_text());rows=json.loads((src/'trace.json').read_text());row=next(row for row in reversed(rows) if row['stage']!='approach interior panel')
args=SimpleNamespace(**r['arguments']);args.assets=Path(args.assets);args.output=Path('research/cross_episode_memory/artifacts/reorder_panel_plan_audit');args.panel_push_doors=True;args.resume_dir=None
c=PhysicalReorderCheck(args);c.bind_door('right');c.data.qpos[:]=row['qpos'];c.data.qvel[:]=0
for axis in ('base_x','base_y','base_theta'):c.data.actuator(NS+axis+'_act').ctrl[0]=c.data.joint(NS+axis).qpos[0]
mujoco.mj_forward(c.model,c.data)
c.stage='approach interior panel';c.planner=c.make_planner();c.arm_aids=c.actuator_ids(c.planner.names);c.load_world()
pose,direction,_=panel_pose(c.model,c.data,c.jid,c.door_open_sign);pose[:3,3]-=.025*direction
positions=[float(c.data.joint(NS+n).qpos[0]) for n in c.planner.names]
goal=list(pose[:3,3]-[0,0,.005])+Rotation.from_matrix(pose[:3,:3]).as_quat(scalar_first=True).tolist()
try:
 trajectory=c.planner.plan(positions,goal);q=trajectory[-1];probe=mujoco.MjData(c.model);probe.qpos[:]=c.data.qpos
 violations=[]
 for i,n in enumerate(c.planner.names):
  j=c.model.joint(NS+n);probe.joint(NS+n).qpos[0]=q[i]
  if min(trajectory[:,i])<j.range[0]-1e-5 or max(trajectory[:,i])>j.range[1]+1e-5:violations.append(dict(joint=n,trajectory_min=float(min(trajectory[:,i])),trajectory_max=float(max(trajectory[:,i])),limits=j.range.tolist()))
 mujoco.mj_forward(c.model,probe)
 result=dict(joint_names=c.planner.names,target=pose.tolist(),endpoint_joints=q.tolist(),fk_xyz=probe.site(NS+'ee_site_r').xpos.tolist(),fk_error_m=float(np.linalg.norm(probe.site(NS+'ee_site_r').xpos-pose[:3,3])),violations=violations,collision=contact_path_collision(c.model,probe,-1),start_qpos=c.data.qpos.tolist())
 (c.output/'audit.json').write_text(json.dumps(result,indent=2));np.save(c.output/'trajectory.npy',trajectory);print(json.dumps(result),flush=True)
finally:c.writer.close();c.head_writer.close();c.renderer.close()
