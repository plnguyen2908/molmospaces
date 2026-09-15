"""Inventory native table objects in complete cached ProcTHOR houses.

Selection is geometric screening, not a claim of physical grasp feasibility.
No objects, furniture, walls, or rooms are moved or removed.
"""
import argparse
import itertools
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


def inventory(path, assets):
    xml = ET.parse(path).getroot()
    meshes = {m.get("name"): m.get("file", "") for m in xml.findall("./asset/mesh")}
    bodies = {b.get("name"): b for b in xml.iter("body")}
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    roots = {}
    for bid in range(1, model.nbody):
        root = bid
        while model.body_parentid[root] != 0:
            root = int(model.body_parentid[root])
        roots[bid] = root
    tables = {}
    for sid in range(model.nsite):
        root = roots.get(int(model.site_bodyid[sid]))
        if root is None:
            continue
        name = model.body(root).name
        if not name.lower().startswith(('diningtable_', 'coffeetable_', 'sidetable_')):
            continue
        tables.setdefault(name, {'body': name, 'position': data.xpos[root].tolist(),
                                  'sites': [], 'objects': []})['sites'].append(sid)
    for jid in range(model.njnt):
        if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
            continue
        bid = int(model.jnt_bodyid[jid])
        name = model.body(bid).name
        asset = '_'.join(name.split('_')[:2])
        for geom in bodies[name].iter('geom'):
            parts = Path(meshes.get(geom.get('mesh'), '')).parts
            if 'Prefabs' in parts:
                asset = parts[parts.index('Prefabs') + 1]
                break
        grasp_path = assets / 'grasps/droid' / asset / (asset + '_grasps_filtered.npz')
        count = 0
        if grasp_path.exists():
            with np.load(grasp_path) as grasps:
                count = len(grasps['transforms'])
        matches = []
        for table in tables.values():
            for sid in table['sites']:
                local = data.site_xmat[sid].reshape(3, 3).T @ (data.xpos[bid] - data.site_xpos[sid])
                size = model.site_size[sid]
                # Sites use a local vertical axis. Allow object centres above
                # the surface; physical support is verified by the runner.
                vertical = int(np.argmax(np.abs(data.site_xmat[sid].reshape(3, 3)[2])))
                horizontal = [a for a in range(3) if a != vertical]
                world_dz = data.xpos[bid, 2] - data.site_xpos[sid, 2]
                if np.all(np.abs(local[horizontal]) <= size[horizontal]) and -.06 <= world_dz <= .25:
                    matches.append(table['body'])
                    break
        if len(matches) == 1:
            tables[matches[0]]['objects'].append({'body': name, 'asset': asset,
                'position': data.xpos[bid].tolist(), 'filtered_grasps': count,
                'grasp_path': str(grasp_path) if count else None})
    for table in tables.values():
        table['sites'] = [model.site(s).name for s in table['sites']]
    return {'scene': ET.parse(path).getroot().get('model'), 'scene_xml': str(path),
            'whole_house': True, 'bodies': model.nbody, 'geoms': model.ngeom,
            'tables': list(tables.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    scenes = {}
    folder = args.assets / 'scenes/procthor-10k-train'
    for path in sorted(folder.glob('*.xml')):
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError:
            continue
        name = root.get('model', '')
        if not name.startswith('train_') or name in scenes:
            continue
        bodies = root.find('worldbody')
        if bodies is None or sum(b.get('name', '').startswith(('diningtable_', 'coffeetable_', 'sidetable_')) for b in bodies.findall('body')) < 2:
            continue
        scenes[name] = path
    houses, pairs = [], []
    for name, path in scenes.items():
        try:
            house = inventory(path, args.assets)
        except Exception as exc:
            print(json.dumps({'scene': name, 'error': str(exc)}), flush=True)
            continue
        houses.append(house)
        for a, b in itertools.combinations(house['tables'], 2):
            pool = [o for table in (a, b) for o in table['objects'] if o['filtered_grasps']]
            if not all(any(o['filtered_grasps'] for o in t['objects']) for t in (a, b)) or len(pool) < 3:
                continue
            distance = float(np.linalg.norm(np.array(a['position'])[:2] - np.array(b['position'])[:2]))
            pairs.append({'scene': name, 'scene_xml': str(path), 'tables': [a, b],
                          'eligible_objects': pool, 'table_distance_m': distance,
                          'screening_only': True})
        print(json.dumps({'scene': name, 'tables': len(house['tables']),
                          'annotated_on_tables': sum(o['filtered_grasps'] > 0 for t in house['tables'] for o in t['objects'])}), flush=True)
    pairs.sort(key=lambda p: (-min(len(p['eligible_objects']), 6), p['table_distance_m']))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'houses': houses, 'candidate_pairs': pairs,
                                       'physical_validation': 'pending'}, indent=2))
    print(json.dumps({'candidate_pairs': len(pairs), 'output': str(args.output)}), flush=True)


if __name__ == '__main__':
    main()
