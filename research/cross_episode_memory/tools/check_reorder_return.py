"""Isolated return-transfer diagnostic from a recorded open-fridge shelf state.

This is not a closed-door/history success. Door operations are excluded here;
the full history runner separately executes and validates them.
"""
import argparse,json,traceback
from pathlib import Path
from types import SimpleNamespace
import mujoco
import numpy as np
from research.cross_episode_memory.tools.check_reorder_chain import PhysicalReorderCheck,Transfer,POTATO,NS
p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('output',type=Path);p.add_argument('--lower-slot',action='store_true');p.add_argument('--shelf-index',type=int,default=5);p.add_argument('--resume-carry',action='store_true');p.add_argument('--native-source',type=Path);p.add_argument('--annotation',type=int);p.add_argument('--object-name',default=POTATO);a=p.parse_args();POTATO=a.object_name
r=json.loads((a.source/'report.json').read_text());rows=json.loads((a.source/'trace.json').read_text())
if a.resume_carry and not a.native_source: p.error('--resume-carry requires --native-source with the original counter poses')
native_rows=json.loads((a.native_source/'trace.json').read_text()) if a.native_source else rows
args=SimpleNamespace(**r['arguments']);args.assets=Path(args.assets);args.output=a.output;args.operate_door=False;args.resume_dir=None;args.change_cycles=2;args.move_retries=4
c=PhysicalReorderCheck(args)
if a.annotation is not None:
 asset,path,transforms=c.annotations[POTATO]
 c.annotations[POTATO]=(asset,path,transforms[[a.annotation]])
 c.report['diagnostic_library_index']=a.annotation
c.select_object(POTATO)
c.data.qpos[:]=rows[-1]['qpos'];c.data.qvel[:]=0;c.data.time=0
for i in range(c.model.nu):
 if c.model.actuator_trntype[i]==mujoco.mjtTrn.mjTRN_JOINT:
  c.data.ctrl[i]=c.data.qpos[c.model.jnt_qposadr[c.model.actuator_trnid[i,0]]]
for axis in ['base_x','base_y','base_theta']:
 c.data.actuator(NS+axis+'_act').ctrl[0]=c.data.joint(NS+axis).qpos[0]
c.data.actuator(NS+'right_finger_act').ctrl[0]=-.05
mujoco.mj_forward(c.model,c.data)
receptacles=json.loads((a.native_source/'report.json').read_text())['receptacles'] if a.native_source else r['receptacles']
c.counter,c.fridge=receptacles;c.receptacles=tuple(receptacles);c.report['receptacles']=receptacles
for obj in c.objects:
 adr=c.model.jnt_qposadr[c.model.joint(obj+'_jntfree_0').id]
 c.initial_poses[obj]=native_rows[0]['qpos'][adr:adr+7]
if a.lower_slot:
 sid=next(i for i in range(c.model.nsite) if c.model.site(i).name.endswith(f'FridgeBodyMeshf017e276Receptacle{a.shelf_index}_{a.shelf_index}'))
 shelf=c.data.site_xpos[sid].copy();hit=np.array([-1],dtype=np.int32)
 distance=mujoco.mj_ray(c.model,c.data,shelf+[0,0,.04],np.array([0.,0.,-1.]),np.array([0,0,0,0,1,0],dtype=np.uint8),True,-1,hit)
 if distance<0:raise RuntimeError('No lower shelf surface')
 surface=shelf[2]+.04-distance
 joint=c.data.joint(POTATO+'_jntfree_0');joint.qpos[:]=c.initial_poses[POTATO]
 joint.qpos[:2]=shelf[:2]+[-.04,.09];mujoco.mj_forward(c.model,c.data)
 joint.qpos[2]+=surface+.003-c.bread_vertices()[:,2].min();joint.qvel[:]=0
 mujoco.mj_forward(c.model,c.data);c.tick(2)
 c.report['authored_candidate']={'kind':'lower shelf with native stable orientation','surface_z':surface,'pose':joint.qpos.tolist()}
c.report['initial_poses']={o:np.asarray(q).tolist() for o,q in c.initial_poses.items()}
c.open_for_access=lambda: None;c.close_after_access=lambda: None
c.report['scope']='isolated fridge retrieval and counter placement; doors excluded'
c.undock=True;c.review_phase='FRIDGE TO COUNTER DIAGNOSTIC'
try:
 if a.resume_carry:
  c.report['scope']='saved lifted-state carry and placement diagnostic; pickup and doors excluded'
  c.source,c.destination=c.fridge,c.counter
  c.move_tracking_tolerance=.003
  c.holding_loaf=c.attached=True;c.unloaded_grasp_seconds=0.
  c.grasp_target_force_n=2.5;c.grasp_stable_force_n=1.5
  c.data.actuator(NS+'right_finger_act').ctrl[0]=r['loaf_hold_command_after_settle']
  c.grasp_relative=np.linalg.inv(c.tcp())@c.bread_pose()
  c.planner=c.make_planner();c.arm_aids=c.actuator_ids(c.planner.names);c.load_world()
  c.transport_payload();c.place_payload()
 else:
  c.transfer(Transfer(POTATO,c.fridge,c.counter))
 c.report['success']=c.assignment()[POTATO]==c.counter
except Exception as e:
 c.report.update(success=False,error=str(e),traceback=traceback.format_exc());print(c.report['traceback'],flush=True)
finally:
 c.report['video_status']='skipped_component_test'
 (c.output/'report.json').write_text(json.dumps(c.report,indent=2));(c.output/'trace.json').write_text(json.dumps(c.trace))
 c.writer.close();c.head_writer.close();c.renderer.close()
raise SystemExit(0 if c.report['success'] else 1)
