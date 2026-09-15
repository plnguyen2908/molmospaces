"""Read-only inspection of the open left door and arm withdrawal."""
import json
from pathlib import Path
from types import SimpleNamespace
import mujoco,numpy as np
from molmo_spaces.configs.robot_configs import RBY1MConfig
from research.cross_episode_memory.kitchen_scene import make_kitchen
from research.cross_episode_memory.tools.check_reorder_chain import PhysicalReorderCheck,F,EGG,POTATO,NS,DOOR2_JOINT,JOINT
from research.cross_episode_memory.tools.check_navigation_transfer import NavigationTransfer
src=Path('research/cross_episode_memory/artifacts/reorder_left_pull_push_tail_v4');r=json.loads((src/'report.json').read_text());rows=json.loads((src/'trace.json').read_text())
m,d,_=make_kitchen(Path(r['arguments']['assets']),RBY1MConfig(),(F,EGG.rsplit('_1_0_0',1)[0],POTATO.rsplit('_1_0_0',1)[0]),robot_xy=(.01,2.08))
jid=m.joint(DOOR2_JOINT).id;adr=m.jnt_qposadr[jid]
for i,row in enumerate(rows):
 if i==0 or i==len(rows)-1 or row['stage']!=rows[i-1]['stage']:
  print(row['stage'],round(row['time'],2),round(np.degrees(row['qpos'][adr]),3),flush=True)
c=PhysicalReorderCheck.__new__(PhysicalReorderCheck);c.model=m;c.data=d;c.args=SimpleNamespace(**r['arguments']);c.active_door='left';c.door_open_sign=1.;c.jid=jid;c.handle_bid=m.body(F+'_1_5_0').id
d.qpos[:]=rows[-1]['qpos'];mujoco.mj_forward(m,d)
probe=mujoco.MjData(m);probe.qpos[:]=d.qpos;probe.joint(JOINT).qpos[0]=0;probe.joint(DOOR2_JOINT).qpos[0]=0;mujoco.mj_forward(m,probe)
c.closed_handles={side:probe.body(F+suffix).xpos.copy() for side,suffix in [('left','_1_5_0'),('right','_1_3_0')]};c.right_closed_grasp=NavigationTransfer.handle_grasp_pose(c)
print('handle',d.xpos[c.handle_bid],'closed',c.handle_grasp_pose(),'side',c.live_handle_grasp(True),flush=True)
cam=mujoco.MjvCamera();cam.lookat[:]=[.4,2.1,1.1];cam.distance=1.5;cam.azimuth=180;cam.elevation=-15
renderer=mujoco.Renderer(m,600,800);renderer.update_scene(d,camera=cam)
from PIL import Image
Image.fromarray(renderer.render()).save(src/'left_open_audit.png');renderer.close()
