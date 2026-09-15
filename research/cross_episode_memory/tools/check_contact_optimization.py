"""Compare optimized contact checks with saved originals on recorded native states."""
import ast,json,time
from pathlib import Path
from types import SimpleNamespace
import mujoco,numpy as np
from research.cross_episode_memory.tools.check_reorder_chain import PhysicalReorderCheck,EGG,NS
root=Path('research/cross_episode_memory/artifacts');out=root/'right_only_runtime_profile'
args=SimpleNamespace(**json.loads((root/'reorder_history_doors_v4/report.json').read_text())['arguments'])
args.assets=Path(args.assets);args.output=out/'equivalence';args.resume_dir=None;args.defer_video=True
c=PhysicalReorderCheck(args)
def original(path,name):
 tree=ast.parse(path.read_text())
 node=next(n for klass in tree.body if isinstance(klass,ast.ClassDef) for n in klass.body if isinstance(n,ast.FunctionDef) and n.name==name)
 namespace=dict(np=np,mujoco=mujoco,NS=NS)
 exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
 return namespace[name]
old={name:original(out/('before_check_reorder_chain.py' if name=='robot_self_penetration' else 'before_check_fridge_transfer.py'),name)
     for name in ('robot_self_penetration','finger_object_contact','unintended_penetration')}
try:
 c.initialize_pair();spec=mujoco.mjtState.mjSTATE_INTEGRATION;initial=np.empty(mujoco.mj_stateSize(c.model,spec));mujoco.mj_getState(c.model,c.data,initial,spec)
 checked=0
 for source in ('reorder_history_doors_v4','reorder_return_egg_carry_v2','reorder_panel_servo_right_v2'):
  rows=json.loads((root/source/'trace.json').read_text())
  for index in np.linspace(0,len(rows)-1,min(30,len(rows)),dtype=int):
   row=rows[index];c.select_object(row.get('active_object',EGG));c.data.qpos[:]=row['qpos'];c.data.qvel[:]=0;mujoco.mj_forward(c.model,c.data)
   assert old['robot_self_penetration'](c,c.data)==c.robot_self_penetration(c.data)
   assert old['unintended_penetration'](c)==c.unintended_penetration()
   assert old['finger_object_contact'](c)==c.finger_object_contact()
   checked+=1
 # Exercise selection changes and in-place set edits too.
 c.bread_bids.add(c.model.body(NS+'link_torso_2').id)
 assert old['finger_object_contact'](c)==c.finger_object_contact()
 wall={}
 for variant in ('original','optimized'):
  c.select_object(EGG);mujoco.mj_setState(c.model,c.data,initial,spec);mujoco.mj_forward(c.model,c.data)
  c.trace=[];c.next_trace=float(c.data.time);c.stage='contact-check benchmark'
  for name in old:
   if variant=='original':setattr(c,name,old[name].__get__(c))
   else:delattr(c,name)
  start=time.perf_counter();c.tick(2.);wall[variant]=time.perf_counter()-start
 result=dict(success=True,scope='contact-check equivalence and runtime only; no task or video',native_states_checked=checked,
             simulation_seconds_per_trial=2.,wall_seconds=wall,speedup=wall['original']/wall['optimized'],physics_timestep=c.model.opt.timestep)
 (out/'contact_checks_benchmark.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
finally:c.writer.close();c.head_writer.close();c.renderer.close()
