"""Physical two-door access component test. Saves report/trace; never renders video."""
import argparse,json,traceback
from pathlib import Path
from types import SimpleNamespace
from research.cross_episode_memory.tools.check_reorder_chain import PhysicalReorderCheck
p=argparse.ArgumentParser(description=__doc__);p.add_argument('template',type=Path);p.add_argument('output',type=Path);p.add_argument('--side',choices=('left','right','both'),default='left');p.add_argument('--stance-x',type=float,default=-.09);p.add_argument('--stance-y',type=float,default=2.08);p.add_argument('--panel-push',action=argparse.BooleanOptionalAction,default=True);a=p.parse_args()
r=json.loads((a.template/'report.json').read_text());args=SimpleNamespace(**r['arguments']);args.assets=Path(args.assets);args.output=a.output;args.resume_dir=None;args.operate_door=True;args.panel_push_doors=a.panel_push;args.start_base_x=a.stance_x;args.start_base_y=a.stance_y;args.start_base_yaw=0.
c=PhysicalReorderCheck(args);c.door_profiles['left']['stance']=(a.stance_x,a.stance_y)
c.review_phase='TWO-DOOR COMPONENT CHECK';c.report['scope']='physical fridge-door component test; no full-chain claim'
try:
 c.initialize_pair()
 for side in ('right','left') if a.side=='both' else (a.side,):c.open_for_access(side)
 c.report['opened_angles_deg']=c.door_angles()
 c.close_after_access();c.report['final_door_angles_deg']=c.door_angles();c.report['success']=True
except Exception as e:
 c.report.update(success=False,error=str(e),traceback=traceback.format_exc());print(c.report['traceback'],flush=True)
finally:
 c.report['video_status']='skipped_component_test'
 (c.output/'report.json').write_text(json.dumps(c.report,indent=2));(c.output/'trace.json').write_text(json.dumps(c.trace))
 c.writer.close();c.head_writer.close();c.renderer.close()
raise SystemExit(0 if c.report['success'] else 1)
