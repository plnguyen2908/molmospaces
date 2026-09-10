#!/usr/bin/env python3
"""Step-response tracking check for MolmoSpaces robot assets.

Every actuator in the YAM-family assets is a position servo, so ``ctrl`` is a
position target and each joint should settle at the value it was commanded.
This script commands one joint at a time, lets it settle, and reports the
steady-state error -- plus the diagnostics needed to tell *why* a joint missed:
contact, force saturation, or neither (which points at dynamics parameters).

Motivating use: as of 2026-09-08 the shipped YAM arm dynamics predate the
real2sim system identification (chirp trajectory + CMA-ES over armature,
friction, motor delay). Distal joints hold 15-38%% standing errors with no
contact and no saturation. Re-run this after the identified parameters land:

    python research/cross_episode_memory/tools/check_joint_tracking.py \
        $MLSPACES_ASSETS_DIR/robots/linearbot/linearbot_holobase.xml

Note the same arm dynamics are duplicated in i2rt_yam/yam.xml,
i2rt_yam/yam_ai2.xml and linearbot/linearbot_holobase.xml. Check all three --
a fix to one does not propagate.
"""

import argparse
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

# Fraction of target treated as acceptable steady-state error.
DEFAULT_TOL = 0.02


def actuated_joints(m):
    """Yield (actuator_id, actuator_name, joint_name, qpos_addr) for joint servos."""
    for a in range(m.nu):
        if m.actuator_trntype[a] != mujoco.mjtTrn.mjTRN_JOINT:
            continue  # site-driven base actuators have no single joint to read
        j = m.actuator_trnid[a, 0]
        yield (
            a,
            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a),
            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j),
            m.jnt_qposadr[j],
        )


def target_for(m, joint_id, frac=0.5):
    """A reachable target: `frac` of the way from 0 toward the joint's limit."""
    lo, hi = m.jnt_range[joint_id]
    if not m.jnt_limited[joint_id]:
        return 0.5
    return (hi * frac) if abs(hi) >= abs(lo) else (lo * frac)


def run(path, settle_s, tol, gravcomp):
    m = mujoco.MjModel.from_xml_path(path)
    if gravcomp is not None:
        m.body_gravcomp[:] = gravcomp
    steps = int(settle_s / m.opt.timestep)

    print(f"model: {path}")
    print(f"  timestep={m.opt.timestep}  settle={settle_s}s  tol={tol:.0%}"
          f"  gravcomp={'model default' if gravcomp is None else gravcomp}\n")
    hdr = f"{'joint':22} {'target':>9} {'final':>9} {'err':>9} {'err%':>7} {'force':>8} {'sat':>4} {'ncon':>5}"
    print(hdr); print("-" * len(hdr))

    failures = []
    for a, aname, jname, qadr in actuated_joints(m):
        jid = m.actuator_trnid[a, 0]
        tgt = target_for(m, jid)
        if tgt == 0.0:
            continue
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        d.ctrl[a] = tgt
        for _ in range(steps):
            mujoco.mj_step(m, d)

        final = d.qpos[qadr]
        err = final - tgt
        pct = abs(err) / abs(tgt)
        frange = m.actuator_forcerange[a]
        force = d.actuator_force[a]
        saturated = bool(frange[1] > frange[0] and abs(abs(force) - max(abs(frange))) < 1e-6)
        flag = "" if pct <= tol else "  <-- FAIL"
        print(f"{jname:22} {tgt:9.4f} {final:9.4f} {err:+9.4f} {pct*100:6.1f}% "
              f"{force:8.3f} {'yes' if saturated else 'no':>4} {d.ncon:5d}{flag}")
        if pct > tol:
            failures.append((jname, pct, saturated, d.ncon))

    print()
    if not failures:
        print(f"PASS: all actuated joints settled within {tol:.0%} of target.")
        return 0

    print(f"FAIL: {len(failures)} joint(s) outside {tol:.0%}.")
    for jname, pct, sat, ncon in failures:
        if sat:
            cause = "actuator force saturated -- forcerange too low for the load"
        elif ncon:
            cause = f"{ncon} contact(s) present -- may be blocked, inspect the pose"
        else:
            cause = "no contact, no saturation -- points at dynamics params "\
                    "(armature / frictionloss / damping) or servo gains"
        print(f"  {jname:22} {pct*100:5.1f}%  {cause}")
    return 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="path to a robot MJCF")
    ap.add_argument("--settle", type=float, default=5.0, help="seconds to settle (default 5)")
    ap.add_argument("--tol", type=float, default=DEFAULT_TOL,
                    help=f"allowed steady-state error as a fraction (default {DEFAULT_TOL})")
    ap.add_argument("--gravcomp", type=float, default=None,
                    help="override body_gravcomp for every body (0 or 1); "
                         "use to separate gravity sag from other causes")
    args = ap.parse_args()
    sys.exit(run(args.model, args.settle, args.tol, args.gravcomp))


if __name__ == "__main__":
    main()
