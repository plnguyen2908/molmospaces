"""Read-only planning audit for the previously failed empty-arm tuck."""
import json,time
from pathlib import Path
from types import SimpleNamespace
import mujoco,numpy as np
from research.cross_episode_memory.tools.check_reorder_chain import PhysicalReorderCheck,NS
from research.cross_episode_memory.door_contact import contact_path_collision
root=Path('research/cross_episode_memory/artifacts');source=root/'reorder_history_doors_v5';output=root/'right_only_tuck_plan_probe'
r=json.loads((source/'report.json').read_text());rows=json.loads((source/'trace.json').read_text())
args=SimpleNamespace(**r['arguments']);args.assets=Path(args.assets);args.output=output;args.resume_dir=None;args.defer_video=True
c=PhysicalReorderCheck(args);c.select_object(rows[-1].get('active_object',args.object_name))
c.data.qpos[:]=rows[-1]['qpos'];c.data.qvel[:]=0
for aid in range(c.model.nu):
 if c.model.actuator_trntype[aid]==mujoco.mjtTrn.mjTRN_JOINT:c.data.ctrl[aid]=c.data.qpos[c.model.jnt_qposadr[c.model.actuator_trnid[aid,0]]]
for name in ('base_x','base_y','base_theta'):c.data.actuator(NS+name+'_act').ctrl[0]=c.data.joint(NS+name).qpos[0]
mujoco.mj_forward(c.model,c.data)
initial=c.data.qpos.copy();results=[]
try:
 for margin in (.05,.005,0.):
  c.args.clearance=margin;c.stage='empty arm tuck planning probe';start=time.perf_counter()
  c.planner=c.make_planner();c.arm_aids=c.actuator_ids(c.planner.names);c.load_world()
  addresses=[c.model.jnt_qposadr[c.model.joint(NS+n).id] for n in c.planner.names]
  current=c.data.qpos[addresses].copy();target=[-.02 if n=='right_arm_3' else 0. for n in c.planner.names]
  row={'activation_distance_m':margin,'physical_start_collision':contact_path_collision(c.model,c.data,-1)}
  try:
   trajectory=c.planner.plan_joints(current.tolist(),target);probe=mujoco.MjData(c.model);bad=None
   for q in trajectory:
    probe.qpos[:]=initial;probe.qpos[addresses]=q;mujoco.mj_forward(c.model,probe);bad=contact_path_collision(c.model,probe,-1)
    if bad:break
   row.update(planned=True,trajectory_samples=len(trajectory),actual_mesh_collision=bad)
  except RuntimeError as exc:row.update(planned=False,error=str(exc))
  row['wall_seconds']=time.perf_counter()-start;results.append(row);print(json.dumps(row),flush=True)
 assert np.array_equal(c.data.qpos,initial)
 (output/'planning_check.json').write_text(json.dumps({'scope':'saved-state planning only; no physics or video','source':str(source),'results':results},indent=2))
finally:c.writer.close();c.head_writer.close();c.renderer.close()
