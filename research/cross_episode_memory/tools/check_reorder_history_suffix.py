"""Debug remaining history from a released boundary; never render a video."""
import argparse
import json
import traceback
from pathlib import Path
from types import SimpleNamespace
import mujoco
import numpy as np
from research.cross_episode_memory.reorder_chain import TwoReceptacleChain
from research.cross_episode_memory.tools.check_reorder_chain import PhysicalReorderCheck, EGG, POTATO, NS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('context', type=Path)
    p.add_argument('state', type=Path)
    p.add_argument('output', type=Path)
    a = p.parse_args()
    report = json.loads((a.context/'report.json').read_text())
    context_rows = json.loads((a.context/'trace.json').read_text())
    state = json.loads((a.state/'trace.json').read_text())[-1]
    args = SimpleNamespace(**report['arguments'])
    args.assets, args.output, args.resume_dir, args.defer_video = Path(args.assets), a.output, None, True
    c = PhysicalReorderCheck(args)
    chain = None
    try:
        c.select_object(state.get('active_object', POTATO))
        c.counter, c.fridge = report['receptacles']
        c.receptacles = (c.counter, c.fridge)
        c.data.qpos[:] = state['qpos']
        c.data.qvel[:] = 0.
        c.data.time = 0.
        for aid in range(c.model.nu):
            if c.model.actuator_trntype[aid] == mujoco.mjtTrn.mjTRN_JOINT:
                c.data.ctrl[aid] = c.data.qpos[c.model.jnt_qposadr[c.model.actuator_trnid[aid,0]]]
        for name in ('base_x','base_y','base_theta'):
            c.data.actuator(NS+name+'_act').ctrl[0] = c.data.joint(NS+name).qpos[0]
        c.data.actuator(NS+'right_finger_act').ctrl[0] = -.05
        c.gaze_command = [float(c.data.joint(NS+n).qpos[0]) for n in ('head_0','head_1')]
        for obj in c.objects:
            adr = c.model.jnt_qposadr[c.model.joint(obj+'_jntfree_0').id]
            c.initial_poses[obj] = np.asarray(context_rows[0]['qpos'])[adr:adr+7].copy()
        mujoco.mj_forward(c.model, c.data)
        c.undock = True
        c.report.update(scope='saved-boundary remaining history debug; not a fresh full run',
                        context=str(a.context), continued_from=str(a.state),
                        receptacles=list(c.receptacles), video_status='skipped_component_test')
        if c.assignment() != {obj:c.fridge for obj in c.objects} or max(c.door_angles().values()) > args.close_tolerance:
            raise RuntimeError('Suffix requires both placed objects and a closed fridge')
        chain = TwoReceptacleChain(c.receptacles, c.objects, c)
        c.history_cycle = 1
        chain.intervene({POTATO:c.counter})
        target_index = chain.explore(label='after change 1')
        c.history_cycle = 2
        chain.work_batch(((EGG,c.counter), (POTATO,c.fridge)))
        c.requested_restore_target = dict(chain.snapshots[target_index]['assignment'])
        chain.intervene({POTATO:c.counter})
        chain.explore(label='after change 2')
        c.restoring_history = True
        target = chain.restore(target_index)
        c.report.update(success=c.assignment()==target, final_assignment=c.assignment(),
                        target_assignment=target, final_door_angles_deg=c.door_angles())
    except Exception as exc:
        c.report.update(success=False,error=str(exc),traceback=traceback.format_exc())
        print(c.report['traceback'],flush=True)
    finally:
        if chain:
            c.report.update(chain_events=chain.events,snapshots=chain.snapshots)
        (c.output/'report.json').write_text(json.dumps(c.report,indent=2))
        (c.output/'trace.json').write_text(json.dumps(c.trace))
        c.writer.close(); c.head_writer.close(); c.renderer.close()
        print(json.dumps({'finished':True,'success':c.report['success'],'output':str(c.output)}),flush=True)
    return 0 if c.report['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
