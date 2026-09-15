"""Inspect a rejected shelf-alignment endpoint in both collision models."""
import json,re
from pathlib import Path
import mujoco,numpy as np
from scipy.spatial.transform import Rotation
from research.cross_episode_memory.tools.check_reorder_placement_boundary import restore
from research.cross_episode_memory.tools.check_reorder_chain import NS
src=Path(__import__('sys').argv[1]);out=Path(__import__('sys').argv[2])
c=restore(src,out);r=json.loads((src/'report.json').read_text())
c.placing=True;c.planner=c.make_planner();c.arm_aids=c.actuator_ids(c.planner.names);c.load_world()
pose=c.bread_pose();local=(c.bread_vertices()-pose[:3,3])@pose[:3,:3];lo,hi=local.min(0),local.max(0)
center=pose[:3,3]+pose[:3,:3]@((lo+hi)/2)
q=[float(c.data.joint(NS+n).qpos[0]) for n in c.planner.names]
c.planner.attach_box(q,list(center-[0,0,.005])+list(Rotation.from_matrix(pose[:3,:3]).as_quat(scalar_first=True)),(hi-lo)/2)
values=[float(x) for x in re.findall(r'np.float64\(([^)]+)\)',r['error'])]
goal=np.eye(4);goal[:3,3]=np.asarray(values[:3])+[0,0,.005];goal[:3,:3]=Rotation.from_quat(values[3:7],scalar_first=True).as_matrix()
relative=np.linalg.inv(c.tcp())@c.bread_pose();addresses=[c.model.jnt_qposadr[c.model.joint(NS+n).id] for n in c.planner.names]
objadr=c.model.jnt_qposadr[c.model.joint(c.object_joint).id];results=[]
try:
 for dx in (0.,.05,.10,-.05,-.10):
  target=goal.copy();target[0,3]+=dx
  result=dict(x_delta=dx)
  try:
   joints=c.nearby_ik(target,q)
   probe=mujoco.MjData(c.model);probe.qpos[:]=c.data.qpos;probe.qpos[addresses]=joints
   obj=target@relative;probe.qpos[objadr:objadr+3]=obj[:3,3];probe.qpos[objadr+3:objadr+7]=Rotation.from_matrix(obj[:3,:3]).as_quat(scalar_first=True)
   mujoco.mj_forward(c.model,probe)
   contacts=[]
   for ct in probe.contact:
    names=[c.model.body(c.model.geom_bodyid[g]).name for g in (ct.geom1,ct.geom2)]
    if ct.dist<0 and any(n.startswith(NS) for n in names):contacts.append(dict(bodies=names,depth=-float(ct.dist)))
   result.update(actual_environment=c.navigation_penetration(probe,True),actual_self=c.robot_self_penetration(probe),contacts=contacts,self_clearance=c.planner.self_clearance(joints))
   from curobo.types import JointState
   import torch
   roll=c.planner.planner.graph_planner.auxiliary_rollout
   state=roll.compute_state_from_action_metrics(torch.tensor([[q],[joints]],device='cuda',dtype=torch.float32))
   metrics=roll.compute_metrics_from_state(state)
   constraints=metrics.costs_and_constraints.constraints
   result['curobo_constraints']={n: v.reshape(2,-1).max(dim=1).values.tolist() for n,v in zip(constraints.names,constraints.values)}
  except Exception as e:result['error']=str(e)
  print(json.dumps(result),flush=True);results.append(result)
 (out/'alignment.json').write_text(json.dumps(results,indent=2))
finally:c.writer.close();c.head_writer.close();c.renderer.close()
