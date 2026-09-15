"""Execute the complete restoration batch from its saved closed-door boundary; no video."""
import argparse
import json
import random
import traceback
from pathlib import Path
from types import SimpleNamespace
import mujoco
import numpy as np
import torch
from research.cross_episode_memory.reorder_chain import TwoReceptacleChain
from research.cross_episode_memory.tools.check_reorder_chain import PhysicalReorderCheck, NS


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path);parser.add_argument('output',type=Path)
    cli=parser.parse_args()
    if cli.output.exists():parser.error('Use a new output directory')
    report=json.loads((cli.source/'report.json').read_text())
    rows=json.loads((cli.source/'trace.json').read_text())
    index=next(i for i,row in enumerate(rows) if row.get('review_phase','').startswith('RESTORE'))
    if index==0:raise ValueError('No saved pre-restoration boundary')
    state=rows[index-1]
    args=SimpleNamespace(**report['arguments']);args.assets,args.output=Path(args.assets),cli.output
    for seed in (random.seed,np.random.seed,torch.manual_seed,torch.cuda.manual_seed_all):seed(getattr(args,'seed',0))
    c=PhysicalReorderCheck(args);chain=None
    try:
        c.data.qpos[:]=state['qpos'];c.data.qvel[:]=0.;c.data.time=state['time']
        for aid in range(c.model.nu):
            if c.model.actuator_trntype[aid]==mujoco.mjtTrn.mjTRN_JOINT:
                c.data.ctrl[aid]=c.data.qpos[c.model.jnt_qposadr[c.model.actuator_trnid[aid,0]]]
        for name in ('base_x','base_y','base_theta'):
            c.data.actuator(NS+name+'_act').ctrl[0]=c.data.joint(NS+name).qpos[0]
        c.data.actuator(NS+'right_finger_act').ctrl[0]=-.05
        c.gaze_command=[float(c.data.joint(NS+n).qpos[0]) for n in ('head_0','head_1')]
        mujoco.mj_forward(c.model,c.data)
        c.counter,c.fridge=report['receptacles'];c.receptacles=(c.counter,c.fridge)
        c.receptacle_support_bodies=report['receptacle_support_bodies']
        for obj in c.objects:
            adr=c.model.jnt_qposadr[c.model.joint(obj+'_jntfree_0').id]
            c.initial_poses[obj]=np.asarray(rows[0]['qpos'][adr:adr+7])
        c.trace=[];c.next_trace=float(c.data.time);c.undock=True;c.restoring_history=True
        c.history_cycle=len(report['snapshots'])
        if max(c.door_angles().values())>args.close_tolerance:raise RuntimeError('Restoration boundary is not closed')
        if c.assignment()!=report['snapshots'][-1]['assignment']:raise RuntimeError('Restoration boundary differs from the final revisit')
        chain=TwoReceptacleChain(c.receptacles,c.objects,c);chain.snapshots=report['snapshots']
        target=chain.restore(0)
        c.report.update(success=c.assignment()==target,final_assignment=c.assignment(),
                        target_assignment=target,final_door_angles_deg=c.door_angles())
    except Exception as exc:
        c.report.update(success=False,error=str(exc),traceback=traceback.format_exc())
        print(c.report['traceback'],flush=True)
    finally:
        if chain:c.report.update(chain_events=chain.events,snapshots=chain.snapshots)
        c.report.update(scope='saved closed-door restoration batch; not a fresh full history',
                        source_run=str(cli.source),source_time=state['time'],
                        receptacles=list(c.receptacles),video_status='skipped_component_test')
        (cli.output/'report.json').write_text(json.dumps(c.report,indent=2))
        (cli.output/'trace.json').write_text(json.dumps(c.trace))
        c.writer.close();c.head_writer.close();c.renderer.close()
        print(json.dumps({'finished':True,'success':c.report['success']}),flush=True)
    return 0 if c.report['success'] else 1

if __name__=='__main__':raise SystemExit(main())
