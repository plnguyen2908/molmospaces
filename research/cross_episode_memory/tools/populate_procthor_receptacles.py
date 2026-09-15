"""Clear native loose props and populate fixed receptacles with annotated assets.

This constructs an initial condition, not a physical transfer demonstration.
Every placement and any unfilled receptacle are recorded explicitly.
"""
import argparse
import copy
import itertools
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


def descendants(model, bid):
    result = {bid}
    for child in range(bid + 1, model.nbody):
        if model.body_parentid[child] in result:
            result.add(child)
    return result


def bounds(model, data, bid):
    from research.cross_episode_memory.tools.check_fridge_transfer import geom_box
    points = []
    bids = descendants(model, bid)
    corners = np.array(list(itertools.product((-1, 1), repeat=3)))
    for gid in range(model.ngeom):
        if model.geom_bodyid[gid] not in bids or not (model.geom_contype[gid] or model.geom_conaffinity[gid]):
            continue
        if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = model.geom_dataid[gid]
            start, count = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
            local = model.mesh_vert[start:start+count]
        else:
            centre, half = geom_box(model, gid)
            local = centre + corners * half
        points.append(local @ data.geom_xmat[gid].reshape(3,3).T + data.geom_xpos[gid])
    vertices = np.concatenate(points)
    return vertices.min(0)-data.xpos[bid], vertices.max(0)-data.xpos[bid]


def load_source(path):
    root = ET.parse(path).getroot()
    for element in root.iter():
        if element.get('file'):
            element.set('file', str((path.parent / element.get('file')).resolve()))
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model); mujoco.mj_forward(model, data)
    metadata = json.loads(path.with_name(path.stem+'_metadata.json').read_text())['objects']
    return root, model, data, metadata


def import_template(root, source, model, data, metadata, asset, assets, prefix=''):
    name, info = next((n,i) for n,i in metadata.items() if i['asset_id'] == asset)
    body = copy.deepcopy(source.find(f".//body[@name='{name}']"))
    annotation = assets / 'grasps/droid' / asset / (asset+'_grasps_filtered.npz')
    with np.load(annotation) as archive:
        count = len(archive['transforms'])
    if not count:
        raise ValueError(f'No filtered grasp annotations: {asset}')
    if prefix:
        # Import only the asset dependencies of this native object, with a
        # namespace to avoid colliding with the house's materials or meshes.
        used = {g.get(key) for g in body.iter('geom') for key in ('mesh','material') if g.get(key)}
        for element in source.find('asset'):
            if element.get('name') in used and element.get('texture'):
                used.add(element.get('texture'))
        rename = {name: prefix+name for name in used}
        for element in source.find('asset'):
            if element.get('name') not in used:
                continue
            element = copy.deepcopy(element)
            for key,value in list(element.attrib.items()):
                if value in rename: element.set(key,rename[value])
            root.find('asset').append(element)
        for element in body.iter():
            for key in ('mesh','material'):
                if element.get(key) in rename: element.set(key,rename[element.get(key)])
    bid = model.body(name).id
    low, high = bounds(model,data,bid)
    body.set('quat',' '.join(map(str,data.xquat[bid])))
    return dict(asset=asset, body=body, low=low, high=high, annotation=str(annotation),
                annotation_count=count, original_name=name, metadata=info)


def clone(template, number):
    body = copy.deepcopy(template['body'])
    names = {element.get('name'): f'population_{number:03d}/'+element.get('name')
             for element in body.iter() if element.get('name')}
    for element in body.iter():
        for key,value in list(element.attrib.items()):
            if value in names: element.set(key,names[value])
    body.set('pos',f'{1000+number*2} 1000 10')
    return body


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets',type=Path,required=True)
    parser.add_argument('--scene',type=Path,required=True)
    parser.add_argument('--template-scene',type=Path,help='Optional house used only as a source of annotated object assets')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--objects-per-table',type=int,default=5)
    parser.add_argument('--objects-per-receptacle',type=int,default=1)
    parser.add_argument('--book-overhang',type=float,default=.04)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    args.output = args.output.resolve()
    if not args.output.is_relative_to(repo): raise ValueError('Output must be inside MolmoSpaces')
    args.output.mkdir(parents=True,exist_ok=False)
    root, original, original_data, metadata = load_source(args.scene)
    if args.template_scene:
        template_root, template_model, template_data, template_metadata = load_source(args.template_scene)
    else:
        template_root, template_model, template_data, template_metadata = root, original, original_data, metadata
    templates = [import_template(root,template_root,template_model,template_data,template_metadata,a,args.assets,
                                 prefix=('population_template/' if args.template_scene else ''))
                 for a in ('Salt_Shaker_2','Potato_11','Mug_2','Book_6')]
    egg_path = args.assets / 'scenes/ithor/FloorPlan3_physics.xml'
    egg_root, egg_model, egg_data, egg_metadata = load_source(egg_path)
    egg_asset = next(i['asset_id'] for i in egg_metadata.values() if i['category'].lower() == 'egg')
    templates.append(import_template(root,egg_root,egg_model,egg_data,egg_metadata,
                                     egg_asset,args.assets,prefix='population_egg/'))
    del egg_model, egg_data
    parents = {child: parent for parent in root.iter() for child in parent}
    removed, removed_names = [], set()
    for name,info in metadata.items():
        if info['is_static']: continue
        body = root.find(f".//body[@name='{name}']")
        if body is None: continue
        removed_names.update(e.get('name') for e in body.iter() if e.get('name'))
        parents[body].remove(body); removed.append(name)
    # Chairs obstruct the mobile manipulator without contributing to this
    # two-receptacle task. Remove every chair fixture as a complete subtree;
    # native loose props (including anything authored on a chair) were removed
    # by the loop above.
    removed_chairs = set()
    for name, info in metadata.items():
        if not info['is_static'] or 'chair' not in info['category'].lower():
            continue
        body = root.find(f".//body[@name='{name}']")
        if body is None:
            continue
        removed_names.update(e.get('name') for e in body.iter() if e.get('name'))
        parents[body].remove(body)
        removed_chairs.add(name)
    for section in ('contact','equality','actuator','sensor','tendon'):
        element = root.find(section)
        if element is not None:
            for child in list(element):
                if any(value in removed_names for entry in child.iter() for value in entry.attrib.values()):
                    element.remove(child)
    sites = []
    for name,info in metadata.items():
        if not info['is_static'] or name in removed_chairs: continue
        for site in info['name_map'].get('sites',{}):
            if 'receptacle' in site.lower():
                sites.append(dict(body=name,site=site,category=info['category']))
    if not sites: raise ValueError('No native fixed receptacles')
    original_site_count=len(sites)
    if args.objects_per_table < 1: raise ValueError('objects-per-table must be positive')
    if args.objects_per_receptacle < 1: raise ValueError('objects-per-receptacle must be positive')
    if not 0 <= args.book_overhang <= .06: raise ValueError('book-overhang must be in [0, .06] m')
    sites=[dict(site,slot=slot) for site in sites
           for slot in range(args.objects_per_table if site['category'] in ('DiningTable','CoffeeTable','SideTable')
                             else args.objects_per_receptacle)]
    clear_path = args.output / 'empty_house.xml'
    ET.ElementTree(root).write(clear_path,encoding='unicode')
    # Precompile candidates in a remote parking area; no simulation steps are
    # taken until unused candidates are deleted from the final scene.
    world = root.find('worldbody'); candidates=[]
    for index,site in enumerate(sites):
        preferred = (index + args.seed) % len(templates)
        order = [preferred] + [i for i in range(len(templates)) if i != preferred]
        for template_index in order:
            template=templates[template_index]; body=clone(template,len(candidates))
            world.append(body)
            candidates.append(dict(site_index=index,template=template,xml=body,name=body.get('name')))
    model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
    data=mujoco.MjData(model); mujoco.mj_forward(model,data)
    selected=[]; missing=[]
    groups=model.geom_group.copy()
    for index,site in enumerate(sites):
        sid=model.site(site['site']).id; support=descendants(model,model.body(site['body']).id)
        matrix=data.site_xmat[sid].reshape(3,3); centre=data.site_xpos[sid].copy()
        vertical=int(np.argmax(np.abs(matrix[2]))); horizontal=[a for a in range(3) if a!=vertical]
        half_z=float(np.abs(matrix[2]) @ model.site_size[sid])
        hits=[]
        model.geom_group[:]=5
        for gid in range(model.ngeom):
            if model.geom_bodyid[gid] in support and (model.geom_contype[gid] or model.geom_conaffinity[gid]): model.geom_group[gid]=4
        for u,v in itertools.product((0.,-.25,.25,-.5,.5,-.7,.7,-.85,.85,-.95,.95),repeat=2):
            local=np.zeros(3); local[horizontal]=np.array([u,v])*model.site_size[sid,horizontal]
            point=centre+matrix@local
            origin=np.array([*point[:2],centre[2]+half_z+.02])
            distance=mujoco.mj_ray(model,data,origin,np.array([0.,0.,-1.]),np.array([0,0,0,0,1,0],dtype=np.uint8),True,-1,None)
            if distance>=0:
                point[2]=origin[2]-distance
                if point[2]>=centre[2]-half_z-.08: hits.append(point)
        model.geom_group[:]=groups
        # Prefer a boundary with an unobstructed outward approach. This is a
        # placement heuristic; full robot docking/IK still needs validation.
        def edge_rank(point):
            local=(point-centre)@matrix
            margins=model.site_size[sid,horizontal]-np.abs(local[horizontal])
            axis=horizontal[int(np.argmin(margins))]
            outward=matrix[:,axis]*(-1 if local[axis]<0 else 1)
            origin=point+np.array([0.,0.,.15])
            distance=mujoco.mj_ray(model,data,origin,outward,None,True,-1,None)
            obstructed=distance>=0 and distance<.6
            return (obstructed,float(min(margins)),float(np.linalg.norm(local[horizontal])))
        hits.sort(key=edge_rank)
        accepted=None
        for candidate in (c for c in candidates if c['site_index']==index):
            template=candidate['template']; bid=model.body(candidate['name']).id
            bids=descendants(model,bid); jid=model.body_jntadr[bid]; adr=model.jnt_qposadr[jid]
            parked=data.qpos[adr:adr+7].copy()
            for point in hits:
                pos=point.copy();pos[:2]-=(template['low'][:2]+template['high'][:2])/2
                pos[2]-=template['low'][2]-.002
                corners=np.array(list(itertools.product(*zip(template['low'],template['high']))))+pos
                local=(corners-centre)@matrix
                excess=np.abs(local[:,horizontal])-model.site_size[sid,horizontal]
                is_book=template['asset'].startswith('Book_')
                # A flat book may extend beyond a horizontal table boundary so
                # a finger can approach its exposed edge. Settling below still
                # has to establish stable physical support.
                if is_book and site['category'] in ('DiningTable','CoffeeTable','SideTable'):
                    if np.any(excess > args.book_overhang): continue
                    overhanging=bool(np.max(excess) > .005)
                    if not overhanging: continue
                else:
                    if np.any(excess > -.002): continue
                    overhanging=False
                if any(previous['receptacle']==site['body'] and abs(previous['position'][2]-pos[2])<.12 and
                       np.linalg.norm(np.array(previous['position'])[:2]-pos[:2])<.10
                       for previous in selected): continue
                data.qpos[adr:adr+3]=pos; mujoco.mj_forward(model,data)
                if any(c.dist<-.0003 and any(model.geom_bodyid[g] in bids for g in (c.geom1,c.geom2)) for c in data.contact): continue
                candidate['xml'].set('pos',' '.join(map(str,pos)))
                accepted=dict(body=candidate['name'],asset=template['asset'],receptacle=site['body'],site=site['site'],
                              position=pos.tolist(),annotation_file=template['annotation'],annotation_count=template['annotation_count'],
                              scale='native, unchanged',placement_policy=('book edge overhang' if overhanging else 'edge-first, outward-clearance preferred'),
                              intentional_overhang=overhanging,
                              footprint_edge_margin_m=float(np.min(model.site_size[sid,horizontal]-np.max(np.abs(local[:,horizontal]),axis=0))))
                selected.append(accepted);break
            if accepted: break
            data.qpos[adr:adr+7]=parked; mujoco.mj_forward(model,data)
        if accepted is None: missing.append(dict(**site,reason='No supported, fitting, collision-free annotated candidate'))
        print(json.dumps({'receptacle':index+1,'total':len(sites),'asset':accepted['asset'] if accepted else None}),flush=True)
    selected_names={s['body'] for s in selected}
    for candidate in candidates:
        if candidate['name'] not in selected_names: world.remove(candidate['xml'])
    root.set('model',args.scene.stem+'_populated')
    scene_path=args.output/'populated_house.xml'
    ET.ElementTree(root).write(scene_path,encoding='unicode')
    generated_metadata={name:info for name,info in metadata.items()
                        if info['is_static'] and name not in removed_chairs}
    for item in selected:
        body_xml=next(c['xml'] for c in candidates if c['name']==item['body'])
        name_map={plural:{e.get('name'):e.get('name') for e in body_xml.iter(tag) if e.get('name')}
                  for tag,plural in (('body','bodies'),('joint','joints'),('freejoint','freejoints'),('site','sites'),('geom','geoms'))}
        name_map['joints'].update(name_map.pop('freejoints'))
        generated_metadata[item['body']]=dict(asset_id=item['asset'],is_static=False,name_map=name_map,category=item['asset'].split('_')[0],
                                               initial_receptacle=item['receptacle'],annotation_file=item['annotation_file'])
    scene_path.with_name(scene_path.stem+'_metadata.json').write_text(json.dumps({'objects':generated_metadata},indent=2))
    # Validate actual settling in the final model, with fixture articulation held
    # at its authored pose just as in the two-table execution loader.
    spec=mujoco.MjSpec.from_file(str(scene_path))
    for body in spec.bodies:
        for joint in list(body.joints):
            if joint.type != mujoco.mjtJoint.mjJNT_FREE: spec.delete(joint)
    final=spec.compile(); settled=mujoco.MjData(final)
    for _ in range(round(2/final.opt.timestep)): mujoco.mj_step(final,settled)
    failures=[]
    for item in selected:
        bids=descendants(final,final.body(item['body']).id)
        supports=descendants(final,final.body(item['receptacle']).id)
        contacts=[]
        for c in settled.contact:
            a,b=final.geom_bodyid[c.geom1],final.geom_bodyid[c.geom2]
            if a in bids and b in supports and -c.frame[2]>.7: contacts.append(float(c.dist))
            elif b in bids and a in supports and c.frame[2]>.7: contacts.append(float(c.dist))
        item['settled_position']=settled.body(item['body']).xpos.tolist()
        item['supported_after_settling']=bool(contacts)
        if not contacts: failures.append(item['body'])
    populated_bodies={item['receptacle'] for item in selected}
    unpopulated_bodies=sorted({site['body'] for site in sites}-populated_bodies)
    report=dict(success=not unpopulated_bodies and not failures,all_sites_populated=not missing,
                populated_receptacles=len(populated_bodies),unpopulated_receptacles=unpopulated_bodies,
                source_scene=str(args.scene),populated_scene=str(scene_path),
                empty_scene=str(clear_path),removed_loose_objects=removed,
                removed_chairs=sorted(removed_chairs),
                original_receptacle_sites=original_site_count,requested_object_slots=len(sites),
                populated_receptacle_sites=len({item['site'] for item in selected}),spawned_objects=len(selected),placements=selected,unpopulated=missing,support_failures=failures,
                robot_manipulation_validated=False,seed=args.seed)
    (args.output/'population_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'finished':True,'success':report['success'],'populated':len(selected),'total':len(sites),
                      'support_failures':len(failures),'output':str(args.output)}),flush=True)
    return 0 if report['success'] else 1


if __name__=='__main__': raise SystemExit(main())
