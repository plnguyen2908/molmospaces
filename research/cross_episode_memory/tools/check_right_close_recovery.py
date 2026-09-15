"""Saved-state right-door closure check; never renders a video."""
import argparse,json,traceback
from pathlib import Path
from types import SimpleNamespace
import mujoco,numpy as np
from research.cross_episode_memory.tools.check_reorder_chain import PhysicalReorderCheck,NS
from research.cross_episode_memory.door_contact import contact_path_collision
p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('output',type=Path);p.add_argument('--probe-only',action='store_true');p.add_argument('--closure-tail',action='store_true');a=p.parse_args()
r=json.loads((a.source/'report.json').read_text());rows=json.loads((a.source/'trace.json').read_text());args=SimpleNamespace(**r['arguments']);args.assets=Path(args.assets);args.output=a.output;args.resume_dir=None;args.defer_video=True
c=PhysicalReorderCheck(args);c.select_object(rows[-1].get('active_object',args.object_name));c.counter,c.fridge=r['receptacles'];c.receptacles=tuple(r['receptacles'])
state=next(row for row in reversed(rows) if row['stage']=='closing finish at stop') if a.closure_tail else rows[-1]
c.data.qpos[:]=state['qpos'];c.data.qvel[:]=0.;c.data.time=0.
for aid in range(c.model.nu):
 if c.model.actuator_trntype[aid]==mujoco.mjtTrn.mjTRN_JOINT:c.data.ctrl[aid]=c.data.qpos[c.model.jnt_qposadr[c.model.actuator_trnid[aid,0]]]
for name in ('base_x','base_y','base_theta'):c.data.actuator(NS+name+'_act').ctrl[0]=c.data.joint(NS+name).qpos[0]
c.data.actuator(NS+'right_finger_act').ctrl[0]=-.05;mujoco.mj_forward(c.model,c.data)
c.gaze_command=[float(c.data.joint(NS+n).qpos[0]) for n in ('head_0','head_1')]
last_close=max((e['time'] for e in r.get('door_cycles',[]) if e['action']=='close'),default=-1.)
c.door_histories['right']=[row for row in rows if row['time']>last_close];c.bind_door('right');c.skip_native_close_navigation=True;c.review_phase='RIGHT DOOR CLOSURE RECOVERY';c.report.update(scope='saved-state closure only; no full task or video',continued_from=str(a.source),initial_door_angles_deg=c.door_angles())
try:
 if a.probe_only:
  names=[f'right_arm_{i}' for i in range(7)]+[f'torso_{i}' for i in range(6)];c.planner=SimpleNamespace(names=names)
  grasp=[row for row in c.opening_reference if row['stage']=='opening'][-1]
  probe=mujoco.MjData(c.model);probe.qpos[:]=grasp['qpos'];mujoco.mj_forward(c.model,probe)
  pose=np.eye(4);pose[:3,:3]=probe.xmat[c.handle_bid].reshape(3,3);pose[:3,3]=probe.xpos[c.handle_bid]
  c.grasp_in_handle=np.linalg.inv(pose)@np.asarray(grasp['tcp']);target=c.regrasp_pose()
  seed=[grasp['qpos'][c.model.jnt_qposadr[c.model.joint(NS+n).id]] for n in names]
  result=[];print('live angle',np.degrees(c.angle()),'recorded angle',np.degrees(probe.qpos[c.model.jnt_qposadr[c.jid]]),'target',target.tolist(),flush=True)
  for distance in (.03,.025,.02,.015):
   pre=target.copy();pre[:3,3]-=distance*pre[:3,2]
   for radius in (.5,):
    row=dict(standoff=distance,trust_radius=radius)
    try:
     q=c.nearby_ik(pre,seed,trust_radius=radius);probe.qpos[:]=c.data.qpos
     for n,v in zip(names,q):probe.joint(NS+n).qpos[0]=v
     mujoco.mj_forward(c.model,probe);row.update(solved=True,collision=contact_path_collision(c.model,probe,-1),joints=q)
    except RuntimeError as exc:row.update(solved=False,error=str(exc))
    result.append(row);print(json.dumps(row),flush=True)
  c.report['ik_probe']=result;c.report['success']=any(v.get('solved') and not v['collision'] for v in result)
 else:
  before=c.assignment()
  if a.closure_tail:
   c.operating_door=True;c.set_grip(c.args.door_grip_force,c.args.door_grip_kp)
   c.data.actuator(NS+'right_finger_act').ctrl[0]=0.
   c.planner=c.make_planner();c.arm_aids=c.actuator_ids(c.planner.names)
   c.finish_native_fridge_close([row for row in c.opening_reference if row['stage']=='reach door handle'])
  else:c.close_after_access()
  if c.assignment()!=before:raise RuntimeError('Stored objects lost support during closure')
  c.report.update(success=True,final_door_angles_deg=c.door_angles())
except Exception as exc:c.report.update(success=False,error=str(exc),traceback=traceback.format_exc());print(c.report['traceback'],flush=True)
finally:
 c.report['video_status']='skipped_component_test'
 (c.output/'report.json').write_text(json.dumps(c.report,indent=2));(c.output/'trace.json').write_text(json.dumps(c.trace))
 c.writer.close();c.head_writer.close();c.renderer.close()
 print(json.dumps({'finished':True,'success':c.report['success'],'output':str(c.output)}),flush=True)
raise SystemExit(0 if c.report['success'] else 1)
