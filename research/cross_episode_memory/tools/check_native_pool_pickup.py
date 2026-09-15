"""Physical navigation and annotated pickup from a saved scene; no video or full-run claim."""
import argparse
import json
import traceback
from pathlib import Path
from types import SimpleNamespace
import mujoco
import numpy as np
from research.cross_episode_memory.tools.check_reorder_chain import PhysicalReorderCheck, OBJECT_POOL, NS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--object', choices=tuple(OBJECT_POOL), default='salt')
    cli = parser.parse_args()
    if cli.output.exists(): parser.error('Use a new output directory')
    report = json.loads((cli.source/'report.json').read_text())
    rows = json.loads((cli.source/'trace.json').read_text())
    state = rows[-1]
    args = SimpleNamespace(**report['arguments'])
    args.assets, args.output = Path(args.assets), cli.output
    args.pickup_only, args.carry_only = True, False
    c = PhysicalReorderCheck(args)
    try:
        c.data.qpos[:] = state['qpos']; c.data.qvel[:] = 0.; c.data.time = state['time']
        for aid in range(c.model.nu):
            if c.model.actuator_trntype[aid] == mujoco.mjtTrn.mjTRN_JOINT:
                c.data.ctrl[aid] = c.data.qpos[c.model.jnt_qposadr[c.model.actuator_trnid[aid,0]]]
        for n in ('base_x','base_y','base_theta'):
            c.data.actuator(NS+n+'_act').ctrl[0] = c.data.joint(NS+n).qpos[0]
        c.data.actuator(NS+'right_finger_act').ctrl[0] = -.05
        c.gaze_command = [float(c.data.joint(NS+n).qpos[0]) for n in ('head_0','head_1')]
        mujoco.mj_forward(c.model,c.data)
        c.counter, c.fridge = report['receptacles']; c.receptacles = (c.counter,c.fridge)
        c.receptacle_support_bodies = report['receptacle_support_bodies']
        c.select_object(OBJECT_POOL[cli.object]); c.source,c.destination=c.counter,c.fridge
        c.trace, c.next_trace = [], float(c.data.time); c.undock=True
        c.review_phase = 'COMPONENT: NATIVE POOL NAVIGATION AND PICKUP'
        # Door access is outside this component. Navigation, annotation selection,
        # physical approach, closure, lift and force hold use the task methods.
        c.open_for_access = lambda side: None
        start_height = float(c.bread_pose()[2,3])
        c.execute_transfer()
        rise = float(c.bread_pose()[2,3]-start_height)
        contact = c.finger_object_contact()
        if rise < .10 or len(contact['fingers']) != 2 or c.support_contacts(table=True):
            raise RuntimeError('Native object did not remain physically lifted with both fingers')
        c.report.update(success=True,final_lift_m=rise,final_finger_object_contact=contact)
    except Exception as exc:
        c.report.update(success=False,error=str(exc),traceback=traceback.format_exc())
        print(c.report['traceback'],flush=True)
    finally:
        c.report.update(scope='saved-state native navigation/pickup component; door access omitted',
                        source_run=str(cli.source),video_status='skipped_component_test')
        (cli.output/'report.json').write_text(json.dumps(c.report,indent=2))
        (cli.output/'trace.json').write_text(json.dumps(c.trace))
        c.writer.close();c.head_writer.close();c.renderer.close()
        print(json.dumps({'finished':True,'success':c.report['success']}),flush=True)
    return 0 if c.report['success'] else 1

if __name__=='__main__': raise SystemExit(main())
