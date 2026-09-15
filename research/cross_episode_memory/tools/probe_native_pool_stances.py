"""Read-only native-pool docking probe from a failed run; no robot rollout/video."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import mujoco
import numpy as np
from research.cross_episode_memory.tools.check_reorder_chain import PhysicalReorderCheck, OBJECT_POOL, NS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--object', choices=tuple(OBJECT_POOL), default='salt')
    cli = p.parse_args()
    report = json.loads((cli.source/'report.json').read_text())
    state = json.loads((cli.source/'trace.json').read_text())[-1]
    args = SimpleNamespace(**report['arguments'])
    args.assets, args.output = Path(args.assets), cli.output
    c = PhysicalReorderCheck(args)
    try:
        c.data.qpos[:] = state['qpos']; c.data.qvel[:] = 0.; c.data.time = state['time']
        c.select_object(OBJECT_POOL[cli.object]); mujoco.mj_forward(c.model,c.data)
        start = c.data.qpos.copy()
        adr = [c.model.jnt_qposadr[c.model.joint(NS+n).id] for n in ('base_x','base_y','base_theta')]
        floor = c.nav_map_filter(); candidates = []
        position = c.data.body(c.object_name).xpos.copy()
        for yaw in (-np.pi/4, -np.pi/2, -3*np.pi/4, 0.):
            for x in np.arange(-.29, .72, .1):
                for y in np.arange(-2.42, -1.51, .1):
                    pose = np.array([x, y, yaw])
                    if not floor(pose): continue
                    reach = float(np.linalg.norm(position[:2]-pose[:2]))
                    if reach > .95: continue
                    worst = 0.
                    for dx,dy in ((0,0),(-.025,0),(.025,0),(0,-.025),(0,.025)):
                        c.data.qpos[:] = start; c.data.qpos[adr] = pose + [dx,dy,0]
                        mujoco.mj_forward(c.model,c.data)
                        worst = max(worst,c.navigation_penetration(c.data,False))
                    if worst <= 0:
                        lateral = float(np.dot(position[:2]-pose[:2],[-np.sin(yaw),np.cos(yaw)]))
                        forward = float(np.dot(position[:2]-pose[:2],[np.cos(yaw),np.sin(yaw)]))
                        if forward < .3: continue
                        candidates.append(dict(pose=pose.tolist(),distance=reach,
                                               right_arm_alignment=abs(lateral+.21)))
        candidates.sort(key=lambda item:item['distance']+.5*item['right_arm_alignment'])
        c.data.qpos[:] = start; c.data.qpos[adr] = [.51,-2.12,-np.pi/2]
        mujoco.mj_forward(c.model,c.data)
        contacts=[]
        for contact in c.data.contact:
            names=[c.model.body(c.model.geom_bodyid[g]).name for g in (contact.geom1,contact.geom2)]
            if any(n.startswith(NS) for n in names) and not all(n.startswith(NS) for n in names):
                if min(contact.dist,0)<0: contacts.append(dict(names=names,depth=-float(contact.dist)))
        result=dict(object=c.object_name,object_position=position.tolist(),candidates=candidates,
                    original_on_map=floor([.51,-2.12,-np.pi/2]),original_contacts=contacts)
        (cli.output/'stances.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(dict(result,candidates=candidates[:8]),indent=2))
    finally:
        c.writer.close();c.head_writer.close();c.renderer.close()

if __name__=='__main__': main()
