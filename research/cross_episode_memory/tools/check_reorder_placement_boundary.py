"""Diagnose or execute placement from a saved physical carry; never render video."""
import argparse
import json
import random
import traceback
from pathlib import Path
from types import SimpleNamespace
import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from research.cross_episode_memory.tools.check_reorder_chain import PhysicalReorderCheck, EGG, POTATO, NS


def restore(source, output):
    report = json.loads((source/'report.json').read_text())
    rows = json.loads((source/'trace.json').read_text())
    state = rows[-1]
    args = SimpleNamespace(**report['arguments'])
    args.assets, args.output = Path(args.assets), output
    args.resume_dir, args.defer_video = None, True
    for seed in (random.seed, np.random.seed, torch.manual_seed, torch.cuda.manual_seed_all):
        seed(getattr(args, 'seed', 0))
    c = PhysicalReorderCheck(args)
    c.counter, c.fridge = report['receptacles']
    c.receptacles = (c.counter, c.fridge)
    c.select_object(state['active_object'])
    c.source, c.destination = c.counter, c.fridge
    c.placement_shelf_offset = (0., -.09 if c.object_name == EGG else .09, 0.)
    c.data.qpos[:] = state['qpos']
    c.data.qvel[:] = 0.
    c.data.time = state['time']
    for aid in range(c.model.nu):
        if c.model.actuator_trntype[aid] == mujoco.mjtTrn.mjTRN_JOINT:
            c.data.ctrl[aid] = c.data.qpos[c.model.jnt_qposadr[c.model.actuator_trnid[aid, 0]]]
    for axis in ('base_x', 'base_y', 'base_theta'):
        c.data.actuator(NS+axis+'_act').ctrl[0] = c.data.joint(NS+axis).qpos[0]
    c.data.actuator(NS+'right_finger_act').ctrl[0] = report['loaf_hold_command_after_settle']
    c.gaze_command = [float(c.data.joint(NS+n).qpos[0]) for n in ('head_0','head_1')]
    for obj in c.objects:
        adr = c.model.jnt_qposadr[c.model.joint(obj+'_jntfree_0').id]
        c.initial_poses[obj] = np.asarray(rows[0]['qpos'])[adr:adr+7].copy()
    mujoco.mj_forward(c.model,c.data)
    last_close = max((e['time'] for e in report['door_cycles']
                      if e['action']=='close' and e['time']<=state['time']),default=-1.)
    c.door_histories['right'] = [r for r in rows if last_close<r['time']<=state['time']]
    c.bind_door('right')
    c.holding_loaf = c.attached = c.undock = True
    c.grasp_relative = np.linalg.inv(c.tcp()) @ c.bread_pose()
    c.grasp_target_force_n, c.grasp_stable_force_n = 2.5, 1.
    c.trace, c.next_trace = [], float(c.data.time)
    c.review_phase = 'SAVED CARRY: PLACEMENT AND CLOSURE'
    c.report.update(scope='saved carry placement component; not full history', source_run=str(source),
                    source_time=state['time'], video_status='skipped_component_test')
    return c


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path); p.add_argument('output',type=Path)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--verify-clearance',action='store_true')
    cli=p.parse_args()
    if cli.output.exists(): p.error('Use a new output directory')
    c=restore(cli.source,cli.output)
    try:
        if not cli.execute:
            class Captured(Exception): pass
            def capture(stage,pose):
                c.probe_goal=pose
                c.stage=stage
                raise Captured()
            c.move=capture
            try: c.place_payload()
            except Captured: pass
            q=[float(c.data.joint(NS+n).qpos[0]) for n in c.planner.names]
            target=c.nearby_ik(c.probe_goal,q)
            c.report['probe']={'q':q,'target':target,'goal':c.probe_goal.tolist(),
                               'padded_self_clearance_m':c.planner.self_clearance(q),
                               'actual_environment_penetration_m':c.navigation_penetration(c.data,True),
                               'actual_self_penetration_m':c.robot_self_penetration(c.data)}
            if cli.verify_clearance:
                # The original invalid carry state is evidence, not a live
                # execution start. Verify contact IK can keep the same tool pose
                # while restoring the unmodified cuRobo self-clearance margin.
                safe = c.nearby_ik(c.tcp(), q, preserve_self_clearance=True)
                target = c.nearby_ik(c.probe_goal, safe, preserve_self_clearance=True)
                path = c.planner.plan_joints(safe, target)
                probe = mujoco.MjData(c.model)
                addresses = [c.model.jnt_qposadr[c.model.joint(NS+n).id] for n in c.planner.names]
                obj_address = c.model.jnt_qposadr[c.model.joint(c.object_joint).id]
                worst_environment = worst_self = 0.
                for sample in np.vstack((np.linspace(q, safe, 30), path)):
                    probe.qpos[:] = c.data.qpos
                    probe.qpos[addresses] = sample
                    mujoco.mj_forward(c.model, probe)
                    site = probe.site(NS+'ee_site_r')
                    tcp = np.eye(4); tcp[:3,3] = site.xpos; tcp[:3,:3] = site.xmat.reshape(3,3)
                    obj = tcp @ c.grasp_relative
                    probe.qpos[obj_address:obj_address+3] = obj[:3,3]
                    probe.qpos[obj_address+3:obj_address+7] = Rotation.from_matrix(obj[:3,:3]).as_quat(scalar_first=True)
                    mujoco.mj_forward(c.model, probe)
                    worst_environment = max(worst_environment, c.navigation_penetration(probe, True))
                    worst_self = max(worst_self, c.robot_self_penetration(probe))
                result = dict(success=worst_environment<=.001 and worst_self<=.0005,
                              initial_self_clearance_m=c.planner.self_clearance(q),
                              corrected_self_clearance_m=c.planner.self_clearance(safe),
                              planned_samples=len(path), actual_environment_penetration_m=worst_environment,
                              actual_self_penetration_m=worst_self)
                c.report['probe']['plan_payload_True'] = result
                print(json.dumps(result), flush=True)
            else:
                for attached in (True,False):
                    if not attached: c.planner.detach_block()
                    try:
                        path=c.planner.plan_joints(q,target)
                        result={'success':True,'samples':len(path)}
                    except Exception as exc: result={'success':False,'error':str(exc)}
                    c.report['probe'][f'plan_payload_{attached}']=result
                    print('PLAN',attached,result,flush=True)
            c.report['diagnostic_complete']=True
            c.report['success']=c.report['probe']['plan_payload_True']['success']
        else:
            c.place_payload()
    except Exception as exc:
        c.report.update(success=False,error=str(exc),traceback=traceback.format_exc())
        print(c.report['traceback'],flush=True)
    finally:
        (c.output/'report.json').write_text(json.dumps(c.report,indent=2))
        (c.output/'trace.json').write_text(json.dumps(c.trace))
        c.writer.close(); c.head_writer.close(); c.renderer.close()
        print(json.dumps({'finished':True,'success':c.report['success'],'output':str(c.output)}),flush=True)
    return 0 if c.report['success'] else 1

if __name__=='__main__': raise SystemExit(main())
