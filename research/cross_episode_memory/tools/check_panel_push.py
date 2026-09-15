"""Short panel force test from an explicitly initialized contact-ready posture.

This is a component test, not a full opening or full task. Never renders video.
"""
import argparse,json,traceback
from pathlib import Path
from types import SimpleNamespace
import mujoco,numpy as np
from research.cross_episode_memory.tools.check_reorder_chain import PhysicalReorderCheck,NS,JOINT,DOOR2_JOINT
p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('postures',type=Path);p.add_argument('output',type=Path);p.add_argument('--side',choices=('left','right'),default='left');p.add_argument('--target-angle',type=float,default=75);p.add_argument('--start-angle',type=float,default=55);p.add_argument('--height',type=float,default=1.4);a=p.parse_args()
r=json.loads((a.source/'report.json').read_text());args=SimpleNamespace(**r['arguments']);args.assets=Path(args.assets);args.output=a.output;args.resume_dir=None;args.operate_door=True
c=PhysicalReorderCheck(args);c.bind_door(a.side);c.operating_door=True
items=json.loads(a.postures.read_text());stance=[.05,1.95,.5] if a.side=='left' else [.05,1.90,-.35]
entry=next(v for v in items if v['angle']==a.start_angle and v['height']==a.height and v['stance']==stance)
c.panel_contact_profile={k:entry[k] for k in ('height','radius','clearance','style') if k in entry}
names=[f'right_arm_{i}' for i in range(7)]+[f'torso_{i}' for i in range(6)]
for name,value in zip(names,entry['joints']):c.data.joint(NS+name).qpos[0]=value
for name,value in zip(('base_x','base_y','base_theta'),stance):c.data.joint(NS+name).qpos[0]=value
c.data.qpos[c.model.jnt_qposadr[c.jid]]=c.door_open_sign*np.radians(a.start_angle)
c.data.joint(JOINT if a.side=='left' else DOOR2_JOINT).qpos[0]=(-1 if a.side=='left' else 1)*np.radians(entry.get('other_open',0.))
for i in (1,2):c.data.joint(NS+f'gripper_finger_r{i}').qpos[0]=0
for i in range(c.model.nu):
 if c.model.actuator_trntype[i]==mujoco.mjtTrn.mjTRN_JOINT:c.data.ctrl[i]=c.data.qpos[c.model.jnt_qposadr[c.model.actuator_trnid[i,0]]]
for name,value in zip(('base_x','base_y','base_theta'),stance):c.data.actuator(NS+name+'_act').ctrl[0]=value
c.data.actuator(NS+'right_finger_act').ctrl[0]=0
mujoco.mj_forward(c.model,c.data)
c.stage='panel push component';c.review_phase='PANEL FORCE COMPONENT';c.report['scope']='initialized partial-open hand posture; panel force component only'
c.report['other_panel_initial_angle_deg']=entry.get('other_open',0.);c.report['initial_panel_angle_deg']=a.start_angle;c.report['initial_push_stance']=stance
# No free-space planning in this diagnostic; production approaches use cuRobo.
c.planner=SimpleNamespace(names=names);c.arm_aids=c.actuator_ids(names);c.data.ctrl[c.arm_aids]=entry['joints']
try:
 c.push_panel_arc(c.door_open_sign*np.radians(a.target_angle),approach=False)
 c.report['final_panel_angle_deg']=abs(np.degrees(c.angle()));c.report['success']=bool(abs(abs(np.degrees(c.angle()))-a.target_angle)<3)
except Exception as e:
 c.report.update(success=False,error=str(e),traceback=traceback.format_exc());print(c.report['traceback'],flush=True)
finally:
 c.report['video_status']='skipped_component_test'
 try:
  (c.output/'report.json').write_text(json.dumps(c.report,indent=2));(c.output/'trace.json').write_text(json.dumps(c.trace))
 finally:
  c.writer.close();c.head_writer.close();c.renderer.close()
 print(json.dumps({'component_finished':True,'success':c.report['success'],'output':str(c.output)}),flush=True)
raise SystemExit(0 if c.report['success'] else 1)
