"""Read-only hinge sweep audit: never changes a live episode or renders video."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from molmo_spaces.configs.robot_configs import RBY1MConfig
from research.cross_episode_memory.kitchen_scene import make_kitchen
from research.cross_episode_memory.tools.check_fridge_transfer import F, DOOR2_JOINT
from research.cross_episode_memory.tools.check_fridge_door import JOINT
from research.cross_episode_memory.tools.check_dynamic_revisit import EGG, POTATO

def audit(model, data):
    result = {}
    for side,joint,panel,handle,sign in [('right',JOINT,F+'_1_2_0',F+'_1_3_0',-1),('left',DOOR2_JOINT,F+'_1_4_0',F+'_1_5_0',1)]:
        probe=mujoco.MjData(model);probe.qpos[:]=data.qpos
        for j in (JOINT,DOOR2_JOINT):probe.joint(j).qpos[0]=0.
        panel_id=model.body(panel).id
        moving={panel_id}
        for bid in range(panel_id+1,model.nbody):
            if model.body_parentid[bid] in moving:moving.add(bid)
        samples=[]
        for angle in range(0,91,2):
            probe.joint(joint).qpos[0]=sign*np.radians(angle);mujoco.mj_forward(model,probe)
            contacts=[]
            for c in probe.contact:
                b1,b2=model.geom_bodyid[[c.geom1,c.geom2]]
                if (b1 in moving)==(b2 in moving):continue
                other=b2 if b1 in moving else b1
                name=model.body(other).name
                if name.startswith(F) or name.startswith('robot_0/'):continue
                if c.dist < 0:contacts.append({'other':name,'depth_m':-float(c.dist),'xyz':c.pos.tolist()})
            if angle%10==0 or contacts:samples.append({'angle_deg':angle,'handle_xyz':probe.body(handle).xpos.tolist(),'contacts':contacts})
        result[side]={'hinge_anchor':probe.xanchor[model.joint(joint).id].tolist(),'hinge_axis':probe.xaxis[model.joint(joint).id].tolist(),'samples':samples,'maximum_external_penetration_m':max((c['depth_m'] for s in samples for c in s['contacts']),default=0.)}
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('template',type=Path);p.add_argument('output',type=Path);args=p.parse_args()
    cfg=json.loads((args.template/'report.json').read_text())['arguments']
    m,d,_=make_kitchen(Path(cfg['assets']),RBY1MConfig(),(F,EGG.rsplit('_1_0_0',1)[0],POTATO.rsplit('_1_0_0',1)[0]),robot_xy=(.01,1.78))
    result=audit(m,d);args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
