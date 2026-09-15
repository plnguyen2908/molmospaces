"""Screen object dimensions and annotated approach widths, without moving objects."""
import json
from pathlib import Path
import mujoco
import numpy as np


def screen(pair):
    model = mujoco.MjModel.from_xml_path(pair['scene_xml'])
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for obj in pair['eligible_objects']:
        bid = model.body(obj['body']).id
        descendants = {bid}
        for child in range(bid + 1, model.nbody):
            if model.body_parentid[child] in descendants:
                descendants.add(child)
        vertices = []
        for gid in range(model.ngeom):
            if model.geom_bodyid[gid] not in descendants or not (model.geom_contype[gid] or model.geom_conaffinity[gid]):
                continue
            if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH:
                mid = model.geom_dataid[gid]
                start, count = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
                local = model.mesh_vert[start:start+count]
            elif model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_BOX:
                local = np.array(list(__import__('itertools').product((-1, 1), repeat=3))) * model.geom_size[gid]
            else:
                continue
            vertices.append(local @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid])
        if not vertices:
            obj['screen_error'] = 'No mesh or box geometry'
            continue
        vertices = np.concatenate(vertices)
        with np.load(obj['grasp_path']) as archive:
            local = archive['transforms']
        world_rotation = data.xmat[bid].reshape(3, 3) @ local[:, :3, :3]
        widths = np.ptp(vertices @ world_rotation[:, :, 1].T, axis=0)
        allowed = (widths < .095) & (world_rotation[:, 2, 2] < -.8)
        obj.update(extent_m=np.ptp(vertices, axis=0).tolist(),
                   mass_kg=float(model.body_subtreemass[bid]),
                   top_down_width_candidates=int(allowed.sum()),
                   minimum_top_down_width_m=float(widths[allowed].min()) if allowed.any() else None)
    pair['screening_only'] = True
    return pair


def main():
    root = Path(__file__).resolve().parents[1] / 'artifacts'
    inventory = json.loads((root / 'procthor_table_inventory.json').read_text())
    candidates = []
    seen = set()
    for pair in inventory['candidate_pairs']:
        if pair['scene'] not in {'train_4', 'train_31', 'train_55', 'train_10'} or pair['scene'] in seen:
            continue
        seen.add(pair['scene'])
        candidates.append(screen(pair))
        print(json.dumps({'scene': pair['scene'], 'objects': [(o['asset'], o.get('top_down_width_candidates'), o.get('mass_kg')) for o in pair['eligible_objects']]}), flush=True)
    (root / 'procthor_table_grasp_screen.json').write_text(json.dumps(candidates, indent=2))


if __name__ == '__main__':
    main()
