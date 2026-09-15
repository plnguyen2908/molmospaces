"""Audit full history, physical-transfer witnesses and door cycles without rendering."""
import argparse
import json
from pathlib import Path


def audit(report):
    checks = {}
    checks['full_physics_success'] = report.get('success') is True
    checks['two_receptacles'] = len(set(report.get('receptacles', []))) == 2
    checks['right_compartment_only'] = report.get('enabled_fridge_compartments') == ['right']
    initial = report.get('initial_closed_articulations', {})
    checks['all_authored_articulations_start_closed'] = bool(initial) and all(abs(v) < 1e-8 for v in initial.values())
    events = report.get('chain_events', [])
    cycles = report.get('arguments', {}).get('change_cycles', 2)
    restored = events[-1].get('n_restored', 0) if events else 0
    expected_order = ['work', 'work', 'intervention', 'explore'] * cycles
    expected_order += ['restore_request'] + ['work'] * restored + ['reorder']
    checks['exact_event_order'] = [e['kind'] for e in events] == expected_order
    work = [e for e in events if e['kind'] == 'work']
    expected_work = 2 * cycles + restored
    checks['expected_physical_transfers'] = restored > 0 and len(work) == expected_work
    ordinary = work[:2 * cycles]
    checks['every_tracked_object_physically_used'] = {e['object_name'] for e in ordinary} == set(report.get('tracked_objects', []))
    directions = {(e['source'], e['destination']) for e in ordinary}
    checks['bidirectional_ordinary_work'] = len(directions) == 2 and all((b, a) in directions for a, b in directions)
    completed = [s for s in report.get('stages', []) if 'completed_transfer' in s]
    doors = report.get('door_cycles', [])
    expected_cycles = 2 * cycles + 1
    checks['each_transfer_verified_shared_access'] = len(completed) == expected_work and all(s.get('door_access_verified') for s in completed)
    expected_actions = []
    for count in [2] * cycles + [restored]:
        for i in range(count):
            expected_actions.append(([['open', 'right']] if i == 0 else []) +
                                    ([['close', 'right']] if i == count-1 else []))
    checks['no_close_reopen_inside_batches'] = [s.get('door_actions') for s in completed] == expected_actions
    groups = report.get('transfer_groups', [])
    checks['work_pairs_and_restore_batch'] = [g.get('transfer_count') for g in groups] == [2]*cycles+[restored] and all(g.get('door_cycle_verified') for g in groups)
    checks['groups_leave_separate_revisit_cycles'] = [(g.get('door_event_start'),g.get('door_event_end')) for g in groups] == [(4*i,4*i+2) for i in range(cycles+1)]
    checks['expected_right_door_cycles'] = [(e['action'], e['side']) for e in doors] == [('open', 'right'), ('close', 'right')] * expected_cycles
    tolerance = report.get('arguments', {}).get('close_tolerance', 3.)
    checks['all_closes_reach_stop'] = len(doors) == 2*expected_cycles and all(e['angle_deg'] <= tolerance for e in doors if e['action'] == 'close')
    snapshots = report.get('snapshots', [])
    checks['post_change_snapshots'] = [s.get('label') for s in snapshots] == [f'after change {i+1}' for i in range(cycles)]
    target = snapshots[0]['assignment'] if snapshots else None
    checks['restored_first_post_change_target'] = target is not None and report.get('target_assignment') == target == report.get('final_assignment')
    checks['restoration_was_nontrivial'] = bool(events) and events[-1].get('kind') == 'reorder' and events[-1].get('n_restored', 0) > 0 and events[-1].get('source_snapshot') == 0
    eligibility = report.get('intervention_eligibility', [])
    witnesses = report.get('successful_transfer_witnesses', [])
    checks['all_transfers_have_live_witnesses'] = len(witnesses) == len(work) and all(
        w.get('transfer') == {k: e[k] for k in ('object_name', 'source', 'destination')}
        for w, e in zip(witnesses, work))
    changes = [e for e in events if e['kind'] == 'intervention']
    witness_valid = len(eligibility) == len(changes) == cycles
    for item, event in zip(eligibility, changes):
        expected = {e['object_name']: e['destination'] for e in event['moves']}
        witness_valid &= item.get('accepted') is True and item.get('moves') == expected
        witness_valid &= set(item.get('placements', {})) == set(expected)
        for obj, placement in item.get('placements', {}).items():
            index = placement.get('witness_index', -1)
            if not 0 <= index < len(witnesses):
                witness_valid = False
                continue
            witness = witnesses[index]; transfer = witness['transfer']
            endpoint = 'source' if placement.get('evidence') == 'successful_pickup' else 'destination'
            witness_valid &= (transfer['object_name'] == obj
                              and transfer[endpoint] == expected.get(obj)
                              and witness[endpoint+'_pose'] == placement.get('pose')
                              and witness['completed_time'] < item['time'])
    checks['changes_use_prior_successful_transfer_poses'] = bool(witness_valid)
    checks['no_duplicate_physics_rehearsal'] = not report.get('intervention_feasibility') and len(eligibility) == cycles and all(e.get('rehearsal_executed') is False for e in eligibility)
    final_doors = report.get('final_door_angles_deg', {})
    checks['finish_closed'] = set(final_doors) == {'right', 'left'} and all(v <= tolerance for v in final_doors.values())
    checks['robot_collision_limit'] = report.get('max_unintended_robot_penetration_m', float('inf')) <= .003
    checks['positive_loaded_finger_force'] = report.get('minimum_loaded_finger_force_n', 0.) > 0.
    return {'passed': all(checks.values()), 'checks': checks,
            'physical_transfer_count': len(work), 'door_cycle_count': len(doors)//2,
            'snapshot_count': len(snapshots), 'target_assignment': target,
            'video_status': report.get('video_status')}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    args = p.parse_args()
    result = audit(json.loads((args.run/'report.json').read_text()))
    (args.run/'history_audit.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
