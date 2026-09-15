"""Read-only departure-route search from a saved held-object state."""
import argparse,json
from pathlib import Path
from types import SimpleNamespace
import mujoco,numpy as np
from scipy.spatial.transform import Rotation
from research.cross_episode_memory.tools.check_reorder_chain import PhysicalReorderCheck,EGG,NS
p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
r=json.loads((a.source/'report.json').read_text());state=json.loads((a.source/'trace.json').read_text())[-1]
args=SimpleNamespace(**r['arguments']);args.assets=Path(args.assets);args.output=a.output;args.defer_video=True
c=PhysicalReorderCheck(args);c.select_object(state.get('active_object',EGG));c.data.qpos[:]=state['qpos'];mujoco.mj_forward(c.model,c.data);c.undock=True;c.stage='loaded departure planning probe'
initial=c.data.qpos.copy();base=c.base_pose();qadr=[c.model.jnt_qposadr[c.model.joint(NS+n).id] for n in ('base_x','base_y','base_theta')];objadr=c.model.jnt_qposadr[c.model.joint(c.object_joint).id];probe=mujoco.MjData(c.model);results=[]
try:
 for distance in (.3,.25,.2,.15,.1,.05,0.):
  c.args.reverse_undock=distance;probe.qpos[:]=initial;offset=-distance*np.array([np.cos(base[2]),np.sin(base[2])]);probe.qpos[qadr[:2]]+=offset;probe.qpos[objadr:objadr+2]+=offset;mujoco.mj_forward(c.model,probe)
  contacts=[]
  for ct in probe.contact:
   names=[c.model.body(c.model.geom_bodyid[g]).name for g in (ct.geom1,ct.geom2)]
   if any(n.startswith(NS) for n in names) and not all(n.startswith(NS) for n in names):contacts.append(dict(bodies=names,dist=float(ct.dist)))
  result={'reverse_m':distance,'contacts':contacts}
  try:
   route=c.plan_route(np.array([c.args.pickup_stance_x,c.args.pickup_stance_y]),True,np.pi-.005);result.update(success=True,path=[q.tolist() for q in route])
  except RuntimeError as exc:result.update(success=False,error=str(exc))
  results.append(result);print(json.dumps(result),flush=True)
  if result['success']:break
 (c.output/'route_probe.json').write_text(json.dumps(results,indent=2))
finally:c.writer.close();c.head_writer.close();c.renderer.close()
