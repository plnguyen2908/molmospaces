"""Render a completed population report as a labelled scene preview, not a task run."""
import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    args = parser.parse_args()
    report = json.loads((args.folder / 'population_report.json').read_text())
    spec = mujoco.MjSpec.from_file(report['populated_scene'])
    for body in spec.bodies:
        for joint in list(body.joints):
            if joint.type != mujoco.mjtJoint.mjJNT_FREE:
                spec.delete(joint)
    model = spec.compile()
    data = mujoco.MjData(model)
    for _ in range(round(2 / model.opt.timestep)):
        mujoco.mj_step(model, data)
    metadata = json.loads(Path(report['populated_scene']).with_name('populated_house_metadata.json').read_text())['objects']
    tables = [name for name, info in metadata.items() if info['category'] == 'DiningTable']
    renderer = mujoco.Renderer(model, height=480, width=640)
    try:
        with imageio.get_writer(str(args.folder / 'population_preview.mp4'), fps=25) as writer:
            for index, name in enumerate(tables):
                items = [p for p in report['placements'] if p['receptacle'] == name]
                camera = mujoco.MjvCamera()
                camera.lookat[:] = np.mean([data.body(p['body']).xpos for p in items], axis=0)
                camera.distance = 2.4
                camera.azimuth = 135
                camera.elevation = -65
                renderer.update_scene(data, camera=camera)
                frame = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(frame)
                draw.rectangle((0, 0, 640, 54), fill='black')
                draw.text((10, 7), f'SCENE PREVIEW ONLY | Native table {index+1} | {len(items)} annotated objects', fill='white')
                draw.text((10, 29), 'Population settled; robot grasp and reorder not yet validated', fill='white')
                frame.save(args.folder / f'table_{index+1}.png')
                for _ in range(75):
                    writer.append_data(np.asarray(frame))
    finally:
        renderer.close()


if __name__ == '__main__':
    main()
