"""Render an unobstructed side-view replay of the recorded annotated-object run."""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from molmo_spaces.configs.robot_configs import RBY1MConfig
from research.cross_episode_memory.kitchen_scene import make_kitchen
from research.cross_episode_memory.tools.check_fridge_transfer import F


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--still-time", type=float)
    args = parser.parse_args()
    report = json.loads((args.folder / "report.json").read_text())
    rows = json.loads((args.folder / "trace.json").read_text())
    telemetry = report["grasp_telemetry"]
    assert len(rows) == len(telemetry)
    name = report["object"]
    model, data, _ = make_kitchen(
        Path(report["arguments"]["assets"]),
        RBY1MConfig(),
        (F, name.rsplit("_1_0_0", 1)[0]),
        robot_xy=(report["arguments"]["base_x"], report["arguments"]["base_y"]),
    )
    model.vis.headlight.ambient[:] = 0.5
    camera = mujoco.MjvCamera()
    camera.elevation = -12
    camera.distance = 0.52
    times = np.array([r["time"] for r in rows])
    if args.still_time is not None:
        indices = [int(np.argmin(abs(times - args.still_time)))]
    else:
        indices = []
        next_time = 0.0
        for i, row in enumerate(rows):
            if row["time"] + 1e-6 < next_time:
                continue
            indices.append(i)
            carrying = "bread to fridge" in row["stage"]
            next_time = row["time"] + (0.4 if carrying else 0.08)
    writer = (
        None
        if args.still_time is not None
        else imageio.get_writer(str(args.folder / "annotated_egg_review.mp4"), fps=12.5)
    )
    try:
        with mujoco.Renderer(model, height=480, width=640) as renderer:
            for i in indices:
                row, measured = rows[i], telemetry[i]
                data.qpos[:] = row["qpos"]
                mujoco.mj_forward(model, data)
                closing_axis = np.array(row["tcp"])[:2, 1]
                side = np.array([closing_axis[1], -closing_axis[0]])
                if side[0] < 0:
                    side = -side
                camera.azimuth = float(np.degrees(np.arctan2(side[1], side[0])))
                camera.lookat[:] = data.body(name).xpos + [0, 0, 0.05]
                renderer.update_scene(data, camera=camera)
                frame = Image.fromarray(renderer.render().copy())
                draw = ImageDraw.Draw(frame)
                draw.rectangle((0, 0, 640, 84), fill="black")
                forces = measured["finger_forces_n"]
                stage = (
                    row["stage"]
                    .replace("bread", "egg")
                    .replace("loaf", "egg")
                    .replace("to fridge", "along aisle")
                )
                draw.text(
                    (10, 5),
                    f"Native egg / stored grasp #{report['annotation_selection']['selected_index']} | {stage}",
                    fill="white",
                )
                draw.text(
                    (10, 24),
                    f"Gap {measured['collision_gap_m'] * 1000:.1f} mm | Fingers {forces[0]:.2f} / {forces[1]:.2f} N",
                    fill="white",
                )
                draw.text(
                    (10, 43),
                    f"Object rise {measured['object_rise_m'] * 1000:.1f} mm | Penetration {measured['penetration_m'] * 1000:.2f} mm",
                    fill="white",
                )
                speed = "5x carry" if "bread to fridge" in row["stage"] else "real time"
                draw.text(
                    (10, 62),
                    f"Recorded-state replay / original forces | t={row['time']:.2f}s | {speed}",
                    fill="white",
                )
                if writer is None:
                    frame.save(args.folder / "annotated_egg_side_proof.jpg")
                else:
                    writer.append_data(np.asarray(frame))
    finally:
        if writer is not None:
            writer.close()
    print(json.dumps({"rendered_frames": len(indices), "still": args.still_time is not None}))


if __name__ == "__main__":
    main()
