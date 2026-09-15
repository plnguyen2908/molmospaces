"""Continue a saved released-handle state; report/trace only, never video."""
import argparse,json,traceback
from pathlib import Path
from types import SimpleNamespace
import mujoco,numpy as np
from research.cross_episode_memory.tools.check_reorder_chain import PhysicalReorderCheck,NS

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('source',type=Path);p.add_argument('output',type=Path)
p.add_argument('--side',choices=('left','right'),default='left');p.add_argument('--phase',choices=('released-handle','push-approach','panel-contact','after-push'),default='released-handle');a=p.parse_args()
r=json.loads((a.source/'report.json').read_text());rows=json.loads((a.source/'trace.json').read_text())
args=SimpleNamespace(**r['arguments']);args.assets=Path(args.assets);args.output=a.output;args.resume_dir=None;args.operate_door=True;args.panel_push_doors=True
c=PhysicalReorderCheck(args);c.bind_door(a.side)
c.counter,c.fridge=r['receptacles'];c.receptacles=tuple(r['receptacles'])
c.report['receptacles']=list(c.receptacles)
c.data.qpos[:]=rows[-1]['qpos'];c.data.qvel[:]=0.;c.data.time=rows[-1]['time'];c.next_trace=c.data.time
for aid in range(c.model.nu):
    if c.model.actuator_trntype[aid]==mujoco.mjtTrn.mjTRN_JOINT:
        c.data.ctrl[aid]=c.data.qpos[c.model.jnt_qposadr[c.model.actuator_trnid[aid,0]]]
for axis in ('base_x','base_y','base_theta'):
    c.data.actuator(NS+axis+'_act').ctrl[0]=c.data.joint(NS+axis).qpos[0]
c.data.actuator(NS+'right_finger_act').ctrl[0]=-.05
mujoco.mj_forward(c.model,c.data)
c.gaze_command=[float(c.data.joint(NS+n).qpos[0]) for n in ('head_0','head_1')]
c.operating_door=True;c.review_phase='RELEASED HANDLE: TRANSITION, PUSH AND CLOSE'
c.report.update(scope='saved released-handle continuation; not a fresh closed-door run',continued_from=str(a.source),continuation_phase=a.phase)
c.planner=c.make_planner();c.arm_aids=c.actuator_ids(c.planner.names)
c.data.ctrl[c.arm_aids]=[c.data.joint(NS+n).qpos[0] for n in c.planner.names]
try:
    if a.phase=='released-handle':
        c.retreat('withdraw from door');c.tick(.5)
    if a.phase=='panel-contact':
        c.push_panel_arc(c.door_open_sign*np.radians(c.args.open_angle),approach=False)
        c.door_opening_modes[a.side]='pull_then_push'
    elif a.phase=='after-push':
        c.door_opening_modes[a.side]='pull_then_push'
    else:
        c.finish_panel_opening()
    c.tuck_arm();c.operating_door=False;c.undock=True
    c.report['opened_angles_deg']=c.door_angles()
    if c.door_angles()[a.side]<70:raise RuntimeError('Door did not remain open after the hand push')
    c.close_after_access();c.report['final_door_angles_deg']=c.door_angles();c.report['success']=True
except Exception as exc:
    c.report.update(success=False,error=str(exc),traceback=traceback.format_exc());print(c.report['traceback'],flush=True)
finally:
    c.report['video_status']='skipped_component_test'
    try:
        (c.output/'report.json').write_text(json.dumps(c.report,indent=2));(c.output/'trace.json').write_text(json.dumps(c.trace))
    finally:
        c.writer.close();c.head_writer.close();c.renderer.close()
    print(json.dumps({'component_finished':True,'success':c.report['success'],'output':str(c.output)}),flush=True)
raise SystemExit(0 if c.report['success'] else 1)
