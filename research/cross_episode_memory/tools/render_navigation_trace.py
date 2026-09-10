#!/usr/bin/env python3
"""Render saved navigation states with a clear wide view; no physics re-execution."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from research.cross_episode_memory.tools.check_fridge_transfer import NS, make_scene


def render(assets, directory):
    report = json.loads((directory / "report.json").read_text())
    trace = json.loads((directory / "trace.json").read_text())
    model, data = make_scene(
        assets,
        fridge_x=report["fridge_x"],
        table_height=report["table_height"],
        base_x=report["base_x"],
        table_y_offset=report["table_y_offset_m"],
        table_foot_half_y=0.04,
    )
    model.vis.headlight.ambient[:] = 0.5
    # Hide only the unused non-colliding end-effector target markers.
    for gid in range(model.ngeom):
        if model.body(model.geom_bodyid[gid]).name.startswith("target_ee_pose"):
            if not (model.geom_contype[gid] or model.geom_conaffinity[gid]):
                model.geom_rgba[gid, 3] = 0
    renderer = mujoco.Renderer(model, height=480, width=640)
    wide, detail = mujoco.MjvCamera(), mujoco.MjvCamera()
    wide.lookat[:] = [0.2, report["table_y_offset_m"] / 2, 1.1]
    wide.azimuth, wide.elevation, wide.distance = 45, -25, 5.3
    detail.azimuth, detail.elevation, detail.distance = 315, -20, 2.1
    output = directory / "navigation_transfer.mp4"
    with imageio.get_writer(str(output), fps=25) as writer:
        for i, sample in enumerate(trace):
            data.qpos[:] = sample["qpos"]
            mujoco.mj_forward(model, data)
            detail.lookat[:] = data.site(NS + "ee_site_r").xpos + [0.05, 0, 0.1]
            frames = []
            for camera in [wide, detail]:
                renderer.update_scene(data, camera=camera)
                frames.append(renderer.render().copy())
            frame = Image.fromarray(np.concatenate(frames, axis=1))
            draw = ImageDraw.Draw(frame)
            draw.rectangle((0, 0, 1280, 48), fill="black")
            draw.text((10, 8), f"Recorded simulation replay | {sample['stage']}", fill="white")
            model_label = "finite-pad" if report["soft_finger_contact"] else "point"
            draw.text(
                (10, 27),
                f"{report['stance_separation_m']:g} m between stances | grip {report['grip_force_limit_n']:g} N, {model_label} | t={sample['time']:.2f}s",
                fill="white",
            )
            draw.text((900, 8), "Grasp / placement detail", fill="white")
            writer.append_data(np.asarray(frame))
            if i % 500 == 0:
                print(f"Rendered {i}/{len(trace)} recorded frames", flush=True)
    renderer.close()
    print(output, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    render(args.assets, args.directory)
