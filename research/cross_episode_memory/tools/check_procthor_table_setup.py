"""Load a whole house and screen empty robot docking; physical support checked at runtime."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from scipy.ndimage import label
from scipy.spatial import cKDTree
from molmo_spaces.utils.scene_maps import ProcTHORMap
from research.cross_episode_memory.procthor_scene import make_house


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', default='train_31')
    parser.add_argument('--objects', nargs='+', default=['Remote_1', 'Remote_3', 'Cellphone_7'])
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'artifacts/procthor_two_tables')
    parser.add_argument('--pair-json', type=Path, help='Explicit pre-screened receptacle pair')
    cli = parser.parse_args()
    output = cli.output
    output.mkdir(parents=True, exist_ok=True)
    if cli.pair_json:
        pair = json.loads(cli.pair_json.read_text())
    else:
        candidates = []
        for name in ('procthor_table_grasp_screen.json', 'procthor_table_alternative_screen.json'):
            candidates.extend(json.loads((output.parent / name).read_text()))
        pair = next(p for p in candidates if p['scene'] == cli.scene
                    and set(cli.objects) <= {o['asset'] for o in p['eligible_objects']})
    objects = [next(o for o in pair['eligible_objects'] if o['asset'] == asset) for asset in cli.objects]
    if any(not o.get('top_down_width_candidates', 0) for o in objects):
        raise ValueError('Selected object has no screened RB-Y1 top-down grasp')
    if len(objects) < 3:
        raise RuntimeError('Need at least three native annotated candidates')
    dynamic = [o['body'] for t in pair['tables'] for o in t['objects']]
    table = pair['tables'][0]
    model, data, stats = make_house(pair['scene_xml'], dynamic, table['position'][:2])
    robot = np.array([model.body(i).name.startswith('robot_0/') for i in range(model.nbody)])
    stances = {}
    for table in pair['tables']:
        centre = np.array(table['position'][:2])
        valid = []
        for radius in (.85, 1., 1.15):
            for angle in np.linspace(-np.pi, np.pi, 24, endpoint=False):
                xy = centre + radius * np.array([np.cos(angle), np.sin(angle)])
                for heading_offset in (0., -.35, .35, -.7, .7):
                    yaw = (angle + heading_offset + 2*np.pi) % (2*np.pi) - np.pi
                    for name, value in zip(('base_x', 'base_y', 'base_theta'), (*xy, yaw)):
                        data.joint('robot_0/' + name).qpos[0] = value
                    mujoco.mj_forward(model, data)
                    collision = False
                    for c in data.contact:
                        a, b = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
                        if robot[a] == robot[b]:
                            continue
                        # Wheel-floor support is legitimate; other robot contacts
                        # must be clear. Full mesh contacts, no furniture removal.
                        rb = a if robot[a] else b
                        other = b if robot[a] else a
                        floor = model.body(other).name.lower().startswith(('room_', 'floor'))
                        if floor and c.pos[2] < .08 and any(s in model.body(rb).name.lower() for s in ('base', 'wheel')):
                            continue
                        if c.dist < -.0001:
                            collision = True
                            break
                    if not collision:
                        valid.append([*xy.tolist(), float(yaw)])
        stances[table['body']] = valid
    # A collision-free pose can still be trapped between chairs. Keep only
    # stances belonging to the same traversable map component on both tables.
    scene_path = Path(pair['scene_xml'])
    house_map = ProcTHORMap.load(str(scene_path.with_name(scene_path.stem + '_map.png')), agent_radius=.35)
    components, _ = label(house_map.occupancy)
    pixels = np.argwhere(house_map.occupancy)
    tree = cKDTree(house_map.get_free_points()[:, :2])
    labels = {}
    for table, poses in stances.items():
        labels[table] = []
        for pose in poses:
            distance, index = tree.query(pose[:2])
            component = int(components[tuple(pixels[index])]) if distance <= .12 else 0
            labels[table].append(component)
    shared = set.intersection(*(set(values) - {0} for values in labels.values()))
    if not shared:
        raise RuntimeError('No shared native-map component between the selected table stances')
    component = max(shared, key=lambda c: int(np.sum(components == c)))
    stances = {table: [pose for pose, c in zip(poses, labels[table]) if c == component]
               for table, poses in stances.items()}
    pair['navigation_component'] = component
    reachable = {}
    for obj in objects:
        table = next(t for t in pair['tables'] if any(o['body'] == obj['body'] for o in t['objects']))
        reachable[obj['body']] = min((float(np.linalg.norm(np.asarray(pose[:2]) - obj['position'][:2]))
                                     for pose in stances[table['body']]), default=float('inf'))
    pair['minimum_object_stance_distance_m'] = reachable
    pair.update(selected_objects=objects, setup=stats, empty_robot_stances=stances,
                physical_manipulation_validated=False)
    (output / 'selection.json').write_text(json.dumps(pair, indent=2))
    print(json.dumps({'scene': pair['scene'], 'objects': [o['asset'] for o in objects],
                      'table_distance_m': pair['table_distance_m'],
                      'clear_stances': {k: len(v) for k, v in stances.items()},
                      'output': str(output / 'selection.json')}), flush=True)
    if any(distance > .9 for distance in reachable.values()):
        raise RuntimeError('Selected object exceeds conservative 0.9 m stance distance; choose another pair or object')
    if not all(stances.values()):
        raise RuntimeError('One table has no collision-free empty-robot stance')


if __name__ == '__main__':
    main()
