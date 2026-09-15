"""Resume a released placement boundary to test departure and physical closure."""
import argparse,json,traceback
from pathlib import Path
from types import SimpleNamespace
import mujoco,numpy as np
from research.cross_episode_memory.tools.check_reorder_chain import PhysicalReorderCheck,EGG,NS
p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('output',type=Path);p.add_argument('--retract-only',action='store_true');p.add_argument('--before-tuck',action='store_true');a=p.parse_args()
r=json.loads((a.source/'report.json').read_text());rows=json.loads((a.source/'trace.json').read_text())
if a.before_tuck:
 index=max(i for i,row in enumerate(rows) if row['stage']=='withdraw along executed cuRobo approach');rows=rows[:index+1]
args=SimpleNamespace(**r['arguments']);args.assets=Path(args.assets);args.output=a.output;args.resume_dir=None
c=PhysicalReorderCheck(args);c.select_object(rows[-1].get('active_object',EGG))
c.data.qpos[:]=rows[-1]['qpos'];c.data.qvel[:]=0;c.data.time=0
for i in range(c.model.nu):
 if c.model.actuator_trntype[i]==mujoco.mjtTrn.mjTRN_JOINT:
  c.data.ctrl[i]=c.data.qpos[c.model.jnt_qposadr[c.model.actuator_trnid[i,0]]]
for axis in ('base_x','base_y','base_theta'):
 c.data.actuator(NS+axis+'_act').ctrl[0]=c.data.joint(NS+axis).qpos[0]
c.data.actuator(NS+'right_finger_act').ctrl[0]=-.05
c.gaze_command=[float(c.data.joint(NS+n).qpos[0]) for n in ('head_0','head_1')]
mujoco.mj_forward(c.model,c.data)
c.counter,c.fridge=r['receptacles'];c.receptacles=tuple(r['receptacles']);c.source,c.destination=c.counter,c.fridge
last_close=max((e['time'] for e in r['door_cycles'] if e['action']=='close'),default=-1)
c.opening_reference=[row for row in rows if row['time']>last_close]
c.door_histories['right']=c.opening_reference
c.trace=list(rows);c.next_trace=0.
c.undock=True;c.review_phase='RELEASED BOUNDARY: DEPART AND CLOSE'
c.report['scope']='saved released placement boundary; departure and closure only'
try:
 before=c.assignment()
 if a.retract_only:
  c.planner=c.make_planner();c.arm_aids=c.actuator_ids(c.planner.names)
  c.retract_empty_placement_arm();c.tuck_arm()
  c.report['scope']='saved placement; measured-path empty-arm retraction and tuck only'
 else:c.after_placement()
 if c.assignment()!=before:raise RuntimeError('Stored objects lost support during departure/closure')
 c.report['success']=True
except Exception as exc:
 c.report.update(success=False,error=str(exc),traceback=traceback.format_exc());print(c.report['traceback'],flush=True)
finally:
 c.report['video_status']='skipped_component_test'
 (c.output/'report.json').write_text(json.dumps(c.report,indent=2));(c.output/'trace.json').write_text(json.dumps(c.trace))
 c.writer.close();c.head_writer.close();c.renderer.close()
raise SystemExit(0 if c.report['success'] else 1)
