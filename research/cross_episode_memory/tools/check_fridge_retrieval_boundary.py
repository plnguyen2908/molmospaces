"""Physical retrieval from a saved open-fridge pregrasp boundary; no video."""
import argparse,json,traceback
from pathlib import Path
from types import SimpleNamespace
import mujoco,numpy as np
from research.cross_episode_memory.tools.check_reorder_chain import PhysicalReorderCheck,EGG,NS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path);p.add_argument('output',type=Path);p.add_argument('--carry-only',action='store_true');p.add_argument('--compact-first',action='store_true');a=p.parse_args()
    r=json.loads((a.source/'report.json').read_text());rows=json.loads((a.source/'trace.json').read_text())
    if a.carry_only:state=rows[-1]
    else:
        index=max(i for i,row in enumerate(rows) if row['stage']=='pregrasp')
        while index>0 and rows[index-1]['stage']=='pregrasp':index-=1
        state=rows[max(0,index-1)]
    args=SimpleNamespace(**r['arguments']);args.assets=Path(args.assets);args.output=a.output;args.resume_dir=None;args.defer_video=True
    c=PhysicalReorderCheck(args);c.select_object(state.get('active_object',EGG))
    c.counter,c.fridge=r['receptacles'];c.receptacles=tuple(r['receptacles']);c.source,c.destination=c.fridge,c.counter
    c.initial_poses={o:np.asarray(q) for o,q in r['initial_object_poses'].items()}
    c.data.qpos[:]=state['qpos'];c.data.qvel[:]=0.;c.data.time=state['time']
    for aid in range(c.model.nu):
        if c.model.actuator_trntype[aid]==mujoco.mjtTrn.mjTRN_JOINT:
            c.data.ctrl[aid]=c.data.qpos[c.model.jnt_qposadr[c.model.actuator_trnid[aid,0]]]
    for n in ('base_x','base_y','base_theta'):c.data.actuator(NS+n+'_act').ctrl[0]=c.data.joint(NS+n).qpos[0]
    c.data.actuator(NS+'right_finger_act').ctrl[0]=-.05
    c.gaze_command=[float(c.data.joint(NS+n).qpos[0]) for n in ('head_0','head_1')]
    mujoco.mj_forward(c.model,c.data)
    reference=r;reference_rows=rows
    while not reference.get('door_cycles'):
        parent=Path(reference['continued_from']);reference=json.loads((parent/'report.json').read_text());reference_rows=json.loads((parent/'trace.json').read_text())
    last_close=max(e['time'] for e in reference['door_cycles'] if e['action']=='close')
    c.door_histories['right']=[row for row in reference_rows if row['time']>last_close]
    c.bind_door('right');c.undock=True;c.next_trace=float(c.data.time)
    def prepare():
        c.args.grip_open=.05;c.annotation_adaptive_aperture=True;c.move_tracking_tolerance=.003
        c.annotation_joint_margin=.035
        c.annotation_standoff=.25;c.annotation_mesh_contact_approach=True
        c.annotation_vertical_offsets=(0.,.005,.01,.015);c.annotation_candidate_budget=64;c.allow_grasp_symmetry=True
    c.prepare_pickup=prepare
    c.review_phase='SAVED OPEN FRIDGE: RETRIEVE AND RETURN'
    c.report.update(scope='saved open-fridge retrieval, carry, counter placement, closure; not full history',
                    continued_from=str(a.source),receptacles=list(c.receptacles),initial_object_poses=r['initial_object_poses'])
    try:
        if a.carry_only:
            prepare();c.holding_loaf=c.attached=True;c.unloaded_grasp_seconds=0.;c.grasp_force_stable_seconds=0.
            c.data.actuator(NS+'right_finger_act').ctrl[0]=r['loaf_hold_command_after_settle']
            c.grasp_relative=np.linalg.inv(c.tcp())@c.bread_pose()
            c.report['loaf_hold_command_after_settle']=r['loaf_hold_command_after_settle']
            c.report['scope']='saved lifted-state carry, counter placement, and closure only'
            c.tick(.2)
            if len(c.contacts())!=2:raise RuntimeError('Saved held state lost bilateral finger contact')
            if a.compact_first:
                c.planner=c.make_planner();c.arm_aids=c.actuator_ids(c.planner.names)
                c.compact_and_carry_from_fridge()
            else:c.carry_from_fridge()
            c.place_payload()
        else:c.execute_transfer()
        if c.assignment()[c.object_name]!=c.counter or max(c.door_angles().values())>args.close_tolerance:
            raise RuntimeError('Retrieval did not end with counter support and closed doors')
        c.report['success']=True
    except Exception as exc:
        c.report.update(success=False,error=str(exc),traceback=traceback.format_exc());print(c.report['traceback'],flush=True)
    finally:
        c.report['video_status']='skipped_component_test'
        (c.output/'report.json').write_text(json.dumps(c.report,indent=2));(c.output/'trace.json').write_text(json.dumps(c.trace))
        c.writer.close();c.head_writer.close();c.renderer.close()
        print(json.dumps({'finished':True,'success':c.report['success'],'output':str(c.output)}),flush=True)
    return 0 if c.report['success'] else 1


if __name__=='__main__':raise SystemExit(main())
