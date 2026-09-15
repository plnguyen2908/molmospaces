"""Render candidate review cameras from a recorded simulator state."""
import json,os
from pathlib import Path
import mujoco,numpy as np
from PIL import Image,ImageDraw
from molmo_spaces.configs.robot_configs import RBY1MConfig
from research.cross_episode_memory.kitchen_scene import make_kitchen
from research.cross_episode_memory.tools.check_dynamic_revisit import EGG,POTATO
from research.cross_episode_memory.tools.check_fridge_transfer import F
p=Path('research/cross_episode_memory/artifacts/reorder_chain_complete_v10')
rows=json.loads((p/'trace.json').read_text());i=next(i for i,r in enumerate(rows) if r.get('review_phase','').startswith('DYNAMIC'));row=rows[i-1]
m,d,_=make_kitchen(Path(os.environ['MLSPACES_ASSETS_DIR']),RBY1MConfig(),(F,EGG.rsplit('_1_0_0',1)[0],POTATO.rsplit('_1_0_0',1)[0]),robot_xy=(.01,1.78));d.qpos[:]=row['qpos'];mujoco.mj_forward(m,d)
r=mujoco.Renderer(m,height=240,width=320);frames=[]
for dist in [.35,.6]:
 strip=[]
 for az in [0,90,180,270]:
  c=mujoco.MjvCamera();c.lookat[:]=[1.0,1.66,1.46];c.distance=dist;c.azimuth=az;c.elevation=-10
  r.update_scene(d,camera=c);im=Image.fromarray(r.render().copy());ImageDraw.Draw(im).text((4,4),f'az={az} dist={dist}',fill='red');strip.append(np.asarray(im))
 frames.append(np.concatenate(strip,axis=1))
Image.fromarray(np.concatenate(frames,axis=0)).save(p/'camera_probe.jpg');r.close()
