"""Package intervention evidence followed by the recorded revisit and restoration."""
import argparse,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path);a=p.parse_args()
r=json.loads((a.folder/'report.json').read_text())
if not r['success']:raise RuntimeError('Cannot package an incomplete chain as success')
chapters=json.loads((a.folder/'video_chapters.json').read_text())
detail=json.loads((a.folder/'dynamic_change_detail.json').read_text())
start=next(t for phase,t in chapters.items() if phase.startswith('DYNAMIC CHANGE'))-2
subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(a.folder/'dynamic_change_detail.mp4'),'-ss',str(start),'-i',str(a.folder/'fridge_transfer.mp4'),'-filter_complex','[0:v]pad=1920:480:320:0,setsar=1[d];[1:v]setsar=1[m];[d][m]concat=n=2:v=1:a=0[v]','-map','[v]','-an','-c:v','libx264','-preset','fast','-crf','20','-threads','2','-movflags','+faststart',str(a.folder/'dynamic_revisit_restore.mp4')],check=True)
(a.folder/'review_clip.json').write_text(json.dumps({'source_start_seconds':start,'before_after_preface_seconds':detail['preface_seconds'],'sequence':'recorded before/after stills, then continuous dynamic intervention, revisit, and physical restoration','fps':25,'playback_speed':5},indent=2))
