"""Test annotated pickup at a saved counter arrival; no navigation replay or video."""
import argparse
import json
import random
import traceback
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import torch

from research.cross_episode_memory.tools.check_reorder_chain import PhysicalReorderCheck, EGG, POTATO, NS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--object', choices=('egg', 'potato'))
    parser.add_argument('--annotation-index', type=int)
    parser.add_argument('--complete-transfer', action='store_true')
    cli = parser.parse_args()
    if cli.output.exists():
        parser.error('Use a new output directory')
    report = json.loads((cli.source / 'report.json').read_text())
    rows = json.loads((cli.source / 'trace.json').read_text())
    obj = {'egg': EGG, 'potato': POTATO}.get(cli.object, rows[-1]['active_object'])
    index = max(i for i, row in enumerate(rows)
                if row['stage'] == 'pregrasp' and row['active_object'] == obj)
    while index > 0 and rows[index-1]['stage'] == 'pregrasp':
        index -= 1
    state = rows[max(0, index-1)]
    args = SimpleNamespace(**report['arguments'])
    args.assets, args.output = Path(args.assets), cli.output
    args.resume_dir, args.defer_video, args.pickup_only, args.carry_only = None, True, not cli.complete_transfer, False
    for seed in (random.seed, np.random.seed, torch.manual_seed, torch.cuda.manual_seed_all):
        seed(getattr(args, 'seed', 0))
    c = PhysicalReorderCheck(args)
    c.counter, c.fridge = report['receptacles']
    c.receptacles = (c.counter, c.fridge)
    c.select_object(obj)
    if cli.annotation_index is not None:
        c.local_annotations = c.local_annotations[[cli.annotation_index]]
    c.source, c.destination = c.counter, c.fridge
    c.placement_shelf_offset = (0., -.09 if obj == EGG else .09, 0.)
    c.data.qpos[:] = state['qpos']
    c.data.qvel[:] = 0.
    c.data.time = state['time']
    for aid in range(c.model.nu):
        if c.model.actuator_trntype[aid] == mujoco.mjtTrn.mjTRN_JOINT:
            c.data.ctrl[aid] = c.data.qpos[c.model.jnt_qposadr[c.model.actuator_trnid[aid, 0]]]
    for axis in ('base_x', 'base_y', 'base_theta'):
        c.data.actuator(NS + axis + '_act').ctrl[0] = c.data.joint(NS + axis).qpos[0]
    c.data.actuator(NS + 'right_finger_act').ctrl[0] = -.05
    c.gaze_command = [float(c.data.joint(NS+n).qpos[0]) for n in ('head_0', 'head_1')]
    for tracked in c.objects:
        adr = c.model.jnt_qposadr[c.model.joint(tracked + '_jntfree_0').id]
        c.initial_poses[tracked] = np.asarray(rows[0]['qpos'])[adr:adr+7].copy()
    mujoco.mj_forward(c.model, c.data)
    last_close = max((e['time'] for e in report['door_cycles']
                      if e['action'] == 'close' and e['time'] <= state['time']), default=-1.)
    c.door_histories['right'] = [row for row in rows if last_close < row['time'] <= state['time']]
    c.bind_door('right')
    c.undock = True
    c.trace, c.next_trace = [], float(c.data.time)
    c.review_phase = 'SAVED COUNTER ARRIVAL: ANNOTATED PICKUP AND HOLD'
    scope = ('saved counter arrival: complete physical transfer and door closure; not full history'
             if cli.complete_transfer else
             'saved counter arrival: annotated pickup, lift and three-second force hold; not full history')
    c.report.update(scope=scope, source_run=str(cli.source), source_time=state['time'],
                    object=obj, diagnostic_annotation_index=cli.annotation_index,
                    receptacles=list(c.receptacles), video_status='skipped_component_test')
    start_height = float(c.bread_pose()[2, 3])
    try:
        if c.assignment()[obj] != c.counter:
            raise RuntimeError('Saved object is not supported on the counter')
        c.execute_transfer()
        contact = c.finger_object_contact()
        rise = float(c.bread_pose()[2, 3] - start_height)
        approach = [row for row in c.report['stages'] if row['stage'].startswith('grasp approach')
                    and 'planner_method' in row]
        if len(approach) != 3 or any(row['planner_method'] != 'actual_mesh_checked_contact_ik'
                                     or row['tcp_error_m'] > .003 for row in approach):
            raise RuntimeError('Pickup did not execute all three validated approach segments')
        if cli.complete_transfer:
            if not c.report.get('success') or not c.support_contacts() or c.contacts():
                raise RuntimeError('Physical transfer did not complete with released shelf support')
            if max(c.door_angles().values()) > c.args.close_tolerance:
                raise RuntimeError('Transfer did not close the door')
        elif rise < .10 or len(contact['fingers']) != 2 or c.support_contacts(table=True):
            raise RuntimeError('Pickup did not retain a physical lift above 10 cm')
        c.report.update(success=True, final_lift_m=rise, final_finger_object_contact=contact,
                        validated_approach_segments=len(approach))
    except Exception as exc:
        c.report.update(success=False, error=str(exc), traceback=traceback.format_exc())
        print(c.report['traceback'], flush=True)
    finally:
        c.report.update(scope=scope, video_status='skipped_component_test')
        (c.output/'report.json').write_text(json.dumps(c.report, indent=2))
        (c.output/'trace.json').write_text(json.dumps(c.trace))
        c.writer.close(); c.head_writer.close(); c.renderer.close()
        print(json.dumps({'finished': True, 'success': c.report['success'], 'output': str(c.output)}), flush=True)
    return 0 if c.report['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
