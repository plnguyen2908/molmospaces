"""Render actual recorded before/after states of a successful chain intervention."""
import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from molmo_spaces.configs.robot_configs import RBY1MConfig
from research.cross_episode_memory.kitchen_scene import make_kitchen
from research.cross_episode_memory.tools.check_dynamic_revisit import EGG, POTATO
from research.cross_episode_memory.tools.check_fridge_transfer import F


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    args = parser.parse_args()
    report = json.loads((args.folder / 'report.json').read_text())
    if not report['success']:
        raise RuntimeError('Do not render a failed chain as completed')
    rows = json.loads((args.folder / 'trace.json').read_text())
    indices = [i for i, row in enumerate(rows)
               if row.get('review_phase', '').startswith('DYNAMIC CHANGE')]
    if not indices or indices[0] == 0:
        raise RuntimeError('No recorded dynamic-change interval')
    intervals = []
    for index in indices:
        if not intervals or index != intervals[-1][-1] + 1 or rows[index].get('review_phase') != rows[intervals[-1][-1]].get('review_phase'):
            intervals.append([])
        intervals[-1].append(index)
    changes = [e for e in report['chain_events'] if e['kind'] == 'intervention']
    if len(intervals) != len(changes):
        raise RuntimeError('Recorded change intervals do not match the event log')
    evidence = []
    for number, (interval, event) in enumerate(zip(intervals, changes), 1):
        names = ', '.join(m['object_name'].split('_')[0] for m in event['moves'])
        evidence.extend(((rows[interval[0]-1], f'CHANGE {number} BEFORE | moving {names}'),
                         (rows[interval[-1]], f'CHANGE {number} AFTER | moved {names}')))

    model, data, _ = make_kitchen(
        Path(report['arguments']['assets']), RBY1MConfig(),
        (F, EGG.rsplit('_1_0_0', 1)[0], POTATO.rsplit('_1_0_0', 1)[0]),
        robot_xy=(.01, 1.78),
    )
    renderer = mujoco.Renderer(model, height=480, width=640)
    panels = []
    try:
        for row, label in evidence:
            data.qpos[:] = row['qpos']
            mujoco.mj_forward(model, data)
            views = []
            for title, lookat, azimuth in (
                ('SINK COUNTER', [-1.62, -1.13, 1.15], 0),
                ('FRIDGE', [1.0, 1.66, 1.46], 0),
            ):
                camera = mujoco.MjvCamera()
                camera.lookat[:] = lookat
                camera.distance = .6
                camera.azimuth = azimuth
                camera.elevation = -65 if title == 'SINK COUNTER' else -10
                renderer.update_scene(data, camera=camera)
                frame = Image.fromarray(renderer.render().copy())
                draw = ImageDraw.Draw(frame)
                draw.rectangle((0, 0, 640, 48), fill='black')
                draw.text((12, 8), title + ' | ' + label, fill='white')
                draw.text((12, 27), 'Recorded state; closed doors stay closed | inspect on revisit', fill='white')
                views.append(np.asarray(frame))
            panels.append(np.concatenate(views, axis=1))
        Image.fromarray(np.concatenate(panels, axis=0)).save(args.folder / 'dynamic_change_before_after.jpg')
        (args.folder / 'dynamic_change_detail.json').write_text(json.dumps({
            'changes': len(changes), 'preface_seconds': 3 * len(panels),
            'labels': [label for _, label in evidence]}, indent=2))
        with imageio.get_writer(str(args.folder / 'dynamic_change_detail.mp4'), fps=25) as writer:
            for frame in panels:
                for _ in range(75):
                    writer.append_data(frame)
    finally:
        renderer.close()
    fps, speedup = report['video_fps'], report['video_speedup']
    phase_times = {}
    for row in rows:
        phase_times.setdefault(row.get('review_phase', ''), row['time'] / speedup)
    (args.folder / 'video_chapters.json').write_text(json.dumps(phase_times, indent=2))
    print(json.dumps(phase_times, indent=2))


if __name__ == '__main__':
    main()
