"""Render a finished successful or failed history from its saved trace, without physics."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from research.cross_episode_memory.tools.check_reorder_chain import PhysicalReorderCheck


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path);p.add_argument('--output',type=Path)
    cli=p.parse_args();source=cli.source.resolve()
    output=(cli.output or source/'failure_video').resolve()
    repo=Path(__file__).resolve().parents[3]
    if not output.is_relative_to(repo):p.error('Output must remain inside molmospaces')
    if output.exists():p.error('Use a new video output directory')
    report=json.loads((source/'report.json').read_text())
    rows=json.loads((source/'trace.json').read_text())
    if not rows:p.error('No recorded states to render')
    args=SimpleNamespace(**report['arguments']);args.assets=Path(args.assets);args.output=output;args.defer_video=True
    c=PhysicalReorderCheck(args)
    try:
        if len(rows[-1]['qpos'])!=c.model.nq:raise RuntimeError('Saved state does not match scene model')
        c.report=report;c.trace=rows
        c.render_deferred_video()
        c.writer.close();c.head_writer.close()
        result=dict(source_run=str(source),physics_success=bool(report.get('success')),
                    error=report.get('error'),video_status='complete',
                    video_outcome='success' if report.get('success') else 'failure',
                    recorded_end_sim_time=rows[-1]['time'],physics_steps_executed=0,
                    video=str(output/'fridge_transfer.mp4'),head_video=str(output/'head_camera.mp4'))
        (output/'render_result.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result),flush=True)
    finally:
        c.writer.close();c.head_writer.close();c.renderer.close()

if __name__=='__main__':main()
