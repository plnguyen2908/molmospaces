#!/usr/bin/env python3
"""Stitch a run's per-episode videos into one annotated demo.

MolmoSpaces already writes one mp4 per camera per episode. That is the wrong unit
for this study: the thing worth watching is the *run* -- how the arrangement drifts
across episodes, what the unobserved intervention changed, and whether the
restoration puts it back. Per-episode files cannot show that.

This concatenates episodes in order with a title card before each, naming the
episode kind, the instruction, and the object->receptacle assignment it starts
from, so the drift is readable frame by frame.

    python compose_run_video.py --run-dir <run> --eval-root <dir> --out demo.mp4
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

KIND_COLOR = {
    "work": (38, 70, 110),
    "explore": (28, 92, 78),
    "restoration": (120, 52, 40),
    "intervention": (92, 78, 24),
}


def _short(name: str) -> str:
    """Body names are long hashes; keep the readable prefix."""
    return re.sub(r"_[0-9a-f]{16,}.*$", "", name) or name


def _title_card(size: tuple[int, int], ep: dict, assignment: dict[str, str],
                seconds: float, fps: float) -> list[np.ndarray]:
    w, h = size
    kind = ep["episode_kind"]
    img = Image.new("RGB", (w, h), KIND_COLOR.get(kind, (50, 50, 50)))
    dr = ImageDraw.Draw(img)

    lines = [
        f"EPISODE {ep['episode_index']}  -  {kind.upper()}",
        "",
        ep.get("language", {}).get("task_description", ""),
    ]
    if kind == "restoration":
        lines.append(f"target: snapshot from episode {ep['task'].get('source_episode')}")
    lines.append("")
    lines.append("arrangement at start:")
    for obj, rec in sorted(assignment.items()):
        lines.append(f"   {_short(obj):<14} -> {_short(rec)}")

    y = int(h * 0.12)
    for i, line in enumerate(lines):
        dr.text((int(w * 0.07), y), line, fill=(240, 240, 240))
        y += 18 if line else 10
    frame = np.array(img)
    return [frame] * max(1, int(seconds * fps))


def _intervention_card(size: tuple[int, int], moves: list[dict], before_idx: int,
                       seconds: float, fps: float) -> list[np.ndarray]:
    """Interventions happen BETWEEN episodes and are otherwise invisible.

    They are the mechanism that makes the task non-trivial: the world changes
    without the robot causing or seeing it, so an earlier arrangement cannot be
    recovered by replaying the action log. A demo that omits them hides the point.
    """
    w, h = size
    img = Image.new("RGB", (w, h), KIND_COLOR["intervention"])
    dr = ImageDraw.Draw(img)
    lines = [
        f"INTERVENTION  (before episode {before_idx})",
        "",
        "scripted, UNOBSERVED - the robot is not present",
        "",
    ]
    for mv in moves:
        lines.append(f"   {_short(mv['object']):<14} {_short(mv['from'])} -> {_short(mv['to'])}")
    lines += ["", "the robot must re-observe to learn this"]
    y = int(h * 0.15)
    for line in lines:
        dr.text((int(w * 0.07), y), line, fill=(245, 245, 235))
        y += 18 if line else 10
    return [np.array(img)] * max(1, int(seconds * fps))


def _assignment_from(ep: dict, placements: dict, receptacles: list[str]) -> dict[str, str]:
    out = {}
    for obj, pose in ep.get("scene_modifications", {}).get("object_poses", {}).items():
        for rec in receptacles:
            if placements.get(f"{obj}|{rec}") == pose:
                out[obj] = rec
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, help="directory written by build_run.py")
    ap.add_argument("--eval-root", required=True,
                    help="parent of the per-episode eval output dirs (demo_out_<idx>)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--camera", default="exterior")
    ap.add_argument("--prefix", default="demo_out",
                    help="per-episode eval output dir prefix, e.g. full_out")
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--card-seconds", type=float, default=2.5)
    args = ap.parse_args()

    import imageio.v2 as imageio

    run_dir = Path(args.run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    placements = manifest["config"]["placements"]
    receptacles = manifest["config"]["receptacles"]

    episodes = {}
    for p in sorted((run_dir).rglob("episode_*.json")):
        ep = json.loads(p.read_text())
        episodes[ep["episode_index"]] = ep

    # Interventions are keyed by the episode they precede.
    by_before: dict[int, list[dict]] = {}
    for iv in manifest.get("interventions", []):
        by_before.setdefault(int(iv["before_episode_index"]), []).append(iv)

    frames: list[np.ndarray] = []
    size = None
    used = []
    for idx in sorted(episodes):
        cand = list(Path(args.eval_root).glob(f"{args.prefix}_{idx}/**/*_{args.camera}_*.mp4"))
        if not cand:
            continue
        rdr = imageio.get_reader(str(cand[0]))
        clip = [np.asarray(f) for f in rdr]
        rdr.close()
        if not clip:
            continue
        if size is None:
            size = (clip[0].shape[1], clip[0].shape[0])
        ep = episodes[idx]
        if idx in by_before:
            frames += _intervention_card(size, by_before[idx], idx,
                                         args.card_seconds, args.fps)
            used.append((idx, "intervention", 0))
        frames += _title_card(size, ep, _assignment_from(ep, placements, receptacles),
                              args.card_seconds, args.fps)
        frames += clip
        used.append((idx, ep["episode_kind"], len(clip)))

    if not frames:
        raise SystemExit(
            f"no videos found under {args.eval_root} matching camera {args.camera!r}"
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(args.out, frames, format="mp4", fps=args.fps, quality=6)
    print(f"wrote {args.out}: {len(frames)} frames from {len(used)} episodes")
    for idx, kind, n in used:
        print(f"   episode {idx:2d}  {kind:12} {n:4d} frames")


if __name__ == "__main__":
    main()
