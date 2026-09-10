# Forward-facing navigation with the actual loaf

The heading-aware check with active head aiming passed on 2026-09-10: drive to the table, physically
pick up the loaf, reverse 30 cm to undock, turn and drive forward to the fridge,
then place, release and withdraw. The table and fridge stances are 2 m apart.
This is one continuous passing run, not a robustness benchmark.

- [Review video: overview, arm and head camera](artifacts/gaze_navigation_verified/navigation_transfer.mp4)
- [Raw head-camera video](artifacts/gaze_navigation_verified/head_camera.mp4)
- [Measured report](artifacts/gaze_navigation_verified/report.json)
- [State/contact trace](artifacts/gaze_navigation_verified/trace.json)
- [Trace audit](artifacts/gaze_navigation_verified/audit.json)

## What passed

| Measurement | Result |
|---|---:|
| Empty outbound base travel | 2.4545 m |
| Loaded return base travel, including reverse | 2.7148 m |
| Maximum forward travel/chassis angle, outbound | 0.0033 degrees |
| Maximum forward travel/chassis angle, loaded | 0.0070 degrees |
| Maximum reverse alignment error | 0.0164 degrees |
| Planned reverse departure | 0.300 m |
| Total measured backward travel during loaded translation, including settling | 0.3036 m |
| Turns per navigation leg | 4 |
| Maximum payload translation slip during loaded navigation | 0.703 mm |
| Loaded-navigation frames with both finger contacts | 2286 / 2286 |
| Maximum arm endpoint position error | 4.92 mm |
| Maximum unintended robot/environment penetration, entire run | 0.155 mm |
| Final bread speed after settling | 0.00000120 m/s |
| Loaf inside the head camera view while manipulating | 98.6 percent |
| Head pan saturated | 0 percent |

Navigation collision metrics were zero. Across the entire run, the maximum
unintended penetration was 0.155 mm, below the 3 mm failure threshold.
Final shelf support, containment, release and settling checks passed. The videos
are original recordings from this run, 265.48 seconds long.

## Implementation and scope

`tools/check_navigation_transfer.py` uses A* over x/y/yaw, with a 10 cm grid
and eight headings. It turns in place, settles within one degree, then drives
forward. The only planned reverse motion is the approved 30 cm straight departure
from the table before the loaded return. Navigation measures actual displacement
against the chassis heading `base_theta`; motion above 2 cm/s must be within five
degrees of forward, or backward during that explicitly marked departure. This used
to be measured against the head camera's optical axis, which only worked while the
head was locked to the body: once the head tracks the loaf, that test confuses where
the body is going with where the head is looking. When pan is zero the two axes are
the same, so the numbers are directly comparable to the earlier run. Tiny servo settling movements below that speed are included in
the distance totals but excluded from the angle gate.

Collision probes use a separate MuJoCo data copy, including the carried loaf.
Nodes are checked with +/-2.5 cm XY offsets, and swept translations and turns are
sampled at 2.5 cm / five-degree intervals. These are sampled checks, not a formal
continuous collision guarantee. Execution also monitors robot/environment contacts
every 2 ms and loaded grip, translation slip, and payload/environment contacts
every 40 ms. The robot begins with both arms down; this is an initial condition,
not a posture reset during execution.

Smooth position targets drive the base's existing site-transmission actuators.
Execution does not write qpos, weld the loaf, or simulate a wheel-level controller.
cuRobo plans arm motions, with the base and left arm locked at each manipulation
stance. Its supplied arm collision model excludes the head chain; MuJoCo
navigation probes include the actual head geometry. The test table's foot remains
narrower than in the original isolated manipulation layout to provide docking
clearance.

## Camera and policy limits

The review video contains overview, manipulation-detail, and actual head RGB
views. `head_camera.mp4` is the same head stream without annotations, at 640x480
and 25 fps.

The head now aims itself. RB-Y1 has two head joints, pan and tilt; only tilt was
ever commanded before, which is why the loaf left the frame near the table. The
head looks at the loaf while manipulating and along the direction of travel while
driving, switching smoothly at 1.5 rad/s. Use `--gaze off` for the old fixed-tilt
behaviour. The loaf is inside the view for 98.6 percent of manipulation steps.

Two limits are built into the robot, not the controller. Pan only reaches plus or
minus 90 degrees, so a target further round than that cannot be centred;
`gaze_pan_saturated_fraction` reports when this happens, and it was 0 here. Tilting
further down than about 1 rad points the camera at the robot's own chest, which is
mounted directly under the head and turns with it, so no body posture can move it
out of shot. Measured share of the head frame at the grasp:

| head tilt | 0.90 | 1.05 | 1.20 | 1.35 |
|---|---:|---:|---:|---:|
| own chest | 5.9% | 22.4% | 48.0% | 78.1% |
| loaf | 6.7% | 6.9% | 4.5% | 1.8% |

Pointing straight at the loaf drives tilt to about 1.2, where the chest covers the
loaf, so downward tilt is capped at 0.95 rad. The loaf then sits about 17 degrees
below the centre of the view, still well inside the 22.5 degree half-field, and more
of it is visible. Turning still sweeps the view, and the brief reverse departure
intentionally moves backward outside the forward view.

The controller still uses oracle simulator state and geometry. This is not a
head-camera-only VLA evaluation or proof of visual obstacle observability. Active
head aiming, visual navigation, and cluttered-kitchen testing remain outstanding.

The pass retains the explicit 100 N grip cap, 2500 N/m position gain and finite-pad
contact model from [CHECK_FRIDGE_TRANSFER.md](CHECK_FRIDGE_TRANSFER.md). These are
uncalibrated assumptions; the original point-contact setup is not validated.
The fridge starts open at 90 degrees. Continuous opening/closing and RunTask
integration remain pending.

## Reproduce

```bash
CUDA_VISIBLE_DEVICES=2 MUJOCO_GL=egl PYTHONPATH=. \
  /nobackup2/le/molmospaces/.venv/bin/python \
  research/cross_episode_memory/tools/check_navigation_transfer.py \
  --assets "$MLSPACES_ASSETS_DIR" \
  --soft-finger --turn-speed 0.08 --motion-slowdown 6 --reverse-undock 0.3 --head-pitch 0.5 \
  --gaze hybrid \
  --output research/cross_episode_memory/artifacts/heading_navigation_repeat
```

The table offset defaults to -2 m and must be at most -1.5 m. Mean forward segment
speed is 0.12 m/s; reverse speed is capped at a mean 0.08 m/s. The verified run
uses a mean turn speed of 0.08 rad/s and arm slowdown 6, now also the navigation
diagnostic defaults. Quintic ramps reach 1.875 times the mean speed and stop
at each route turn. The grasp settings above are defaults of this diagnostic; `--soft-finger`
is opt-in.

## Earlier results

The earlier [fixed-heading video](artifacts/navigation_transfer_verified/navigation_transfer.mp4)
physically carried and placed the loaf, but translated sideways while the robot
faced the fridge. Its 3.006 m paths and grip measurements are historical component
results and do not satisfy the forward-view navigation requirement.

The first loaded heading-aware attempt at a mean turn speed of 0.25 rad/s
[failed with a real grip loss](artifacts/heading_navigation_06/fridge_transfer.mp4)
on its second loaded turn. The failure is retained; the grip-force and contact
parameters were not increased to address it.

At 0.12 rad/s, navigation passed, but the loaf shifted about 1.5 mm and fell
during the subsequent 30-degree reorientation (heading_navigation_07). The final
run reduced mean turn speed to 0.08 rad/s and doubled arm trajectory duration,
without changing grip force, contact parameters, or grasp depth.
