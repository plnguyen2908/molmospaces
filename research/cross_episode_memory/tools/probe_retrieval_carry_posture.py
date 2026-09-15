"""Check loaded arm retraction and departure together in scratch physics states."""
import json
from pathlib import Path
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from research.cross_episode_memory.tools.check_reorder_placement_boundary import restore
from research.cross_episode_memory.tools.check_reorder_chain import NS

src=Path(__import__('sys').argv[1]);out=Path(__import__('sys').argv[2])
c=restore(src,out)
rows=json.loads((src/'trace.json').read_text())
index=next(i for i,r in enumerate(rows) if r['active_object']==c.object_name and r['stage']=='compact held object before navigation')
# Select the last such transfer, not any earlier diagnostic pose.
while index+1<len(rows) and rows[index+1]['time']<900:index+=1
# In the failing full run the first compact is already in the final retrieval.
state=rows[max(0,index-1)]
c.data.qpos[:]=state['qpos'];c.data.time=state['time'];mujoco.mj_forward(c.model,c.data)
c.source,c.destination=c.fridge,c.counter
c.planner=c.make_planner();c.arm_aids=c.actuator_ids(c.planner.names)
pose=c.bread_pose();vertices=c.bread_vertices();local=(vertices-pose[:3,3])@pose[:3,:3]
lo,hi=local.min(0),local.max(0);center=pose[:3,3]+pose[:3,:3]@((lo+hi)/2)
c.planner.attach_box([float(c.data.joint(NS+n).qpos[0]) for n in c.planner.names],list(center-[0,0,.005])+list(Rotation.from_matrix(pose[:3,:3]).as_quat(scalar_first=True)),(hi-lo)/2)
start=c.tcp();relative=np.linalg.inv(start)@c.bread_pose();live=c.data
addresses=[c.model.jnt_qposadr[c.model.joint(NS+n).id] for n in c.planner.names];objadr=c.model.jnt_qposadr[c.model.joint(c.object_joint).id]
results=[]
try:
 for lateral in (0.,.05,.10,.15,-.05):
  for retreat,height in ((.2,0.),(.15,0.),(.1,0.),(.15,.05),(.15,-.05)):
   pose=start.copy();pose[:3,3]+=[-retreat,lateral,height]
   result=dict(lateral=lateral,retreat=retreat,height=height)
   try:
    path=c.plan_contact_path('carry posture probe',pose)
    probe=mujoco.MjData(c.model);probe.qpos[:]=live.qpos;probe.qpos[addresses]=path[-1]
    obj=pose@relative;probe.qpos[objadr:objadr+3]=obj[:3,3];probe.qpos[objadr+3:objadr+7]=Rotation.from_matrix(obj[:3,:3]).as_quat(scalar_first=True)
    probe.time=live.time;mujoco.mj_forward(c.model,probe)
    c.data=probe
    errors=[]
    for reverse in (.3,.2,.1):
     c.args.reverse_undock=reverse
     try:
      route=c.plan_route(np.array([c.args.pickup_stance_x,c.args.pickup_stance_y]),True,np.pi-.005)
      result.update(success=True,reverse=reverse,route=[p.tolist() for p in route]);break
     except RuntimeError as e:errors.append(str(e))
    else:result.update(success=False,errors=errors)
   except RuntimeError as e:result.update(success=False,error=str(e))
   finally:c.data=live
   results.append(result);print(json.dumps(result),flush=True)
   if result['success']:break
  if result['success']:break
 (out/'postures.json').write_text(json.dumps(results,indent=2))
finally:c.writer.close();c.head_writer.close();c.renderer.close()
