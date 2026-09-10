"""Real mobile-base geometry for MolmoSpaces' synthesized holonomic bases.

MolmoSpaces does not model mobile bases: ``add_robot_to_scene`` extrudes one box
from ``base_size`` and textures it with wood, standing in for DROID's rolling
cart. That reads as a crate in renders and in robot-mounted camera views.

``dress_base`` replaces the *appearance* with Clearpath **Ridgeback** geometry --
a real omnidirectional (mecanum) base built to carry manipulators, which is the
closest commercial analogue to "Franka on a holonomic base". Meshes come from
``github.com/ridgeback/ridgeback`` (``ridgeback_description``), BSD-3-Clause,
Copyright 2021 Clearpath Robotics Inc.

Every geom added here is visual-only -- ``contype=0``, ``conaffinity=0``,
``density=0`` -- so collision geometry, mass, inertia and dynamics are unchanged
and prior tracking results remain valid.

**Scale matters.** Ridgeback's real chassis is 0.960 x 0.793 x 0.216 m with
0.0759 m wheels and a 0.280 m deck. MolmoSpaces' default ``base_size`` is
[0.5, 0.5, 0.58]. Two modes:

``fit`` (default)
    Uniformly scale Ridgeback to the configured ``base_size`` footprint, and add
    a riser column up to the configured deck height. Visual and collision agree.
    The robot is no longer true-to-scale Ridgeback, but nothing about navigation
    changes.

``true``
    Ridgeback at 1:1, with a riser to the configured deck height. Physically
    honest, but the visual footprint is then *larger* than the collision box, so
    the robot will appear to clip walls. Only use this if you also widen
    ``base_size`` to [0.960, 0.793, ...] -- which changes the collision footprint
    and invalidates prior navigation measurements.

Usage, immediately after the robot is added to the scene::

    cfg.robot_cls.add_robot_to_scene(cfg, spec, "robot_0/", [0, 0], [1, 0, 0, 0])
    dress_base(spec, "robot_0/", cfg.base_size)
"""

from pathlib import Path

import mujoco
import numpy as np

# Ridgeback geometry, from ridgeback_description/urdf/ridgeback.urdf.xacro
RB_CHASSIS = (0.960, 0.793, 0.216)
RB_WHEEL_RADIUS = 0.0759
RB_ROCKER_OFFSET = 0.319          # wheel pair offset along x
RB_ROCKER_WIDTH = 0.472
RB_WHEEL_WIDTH = 0.0790
RB_WHEEL_Y = RB_ROCKER_WIDTH / 2 + RB_WHEEL_WIDTH / 2

_HALF_PI = np.pi / 2
# rpy (pi/2, 0, 0) as a quaternion -- the wheel mesh's own frame in the URDF.
_WHEEL_QUAT = [np.cos(_HALF_PI / 2), np.sin(_HALF_PI / 2), 0.0, 0.0]

# Clearpath livery, from the xacro's material names.
YELLOW_RGBA = [0.95, 0.74, 0.05, 1.0]   # body + side covers
BLACK_RGBA = [0.10, 0.10, 0.11, 1.0]    # end covers, top plate
GREY_RGBA = [0.72, 0.73, 0.75, 1.0]     # riser
WHITE_RGBA = [0.93, 0.93, 0.93, 1.0]    # front lights

# rpy (0, 0, pi) -- the 180-degree z flip used by the mirrored covers/lights.
_FLIP_Z = [0.0, 0.0, 0.0, 1.0]

# Ridgeback deck/riser constants (xacro: deck_height, deck_thickness, riser_height).
RB_DECK_HEIGHT = 0.280
RB_DECK_THICKNESS = 0.005
RB_RISER_HEIGHT = 0.055

_MESHES = ("body", "top", "end-cover", "side-cover", "wheel", "rocker", "lights")


def _mesh_dir() -> Path:
    """Locate the Ridgeback meshes inside the MolmoSpaces asset tree."""
    import os

    root = os.environ.get("MLSPACES_ASSETS_DIR")
    if not root:
        raise RuntimeError("MLSPACES_ASSETS_DIR is not set")
    d = Path(root) / "robots" / "ridgeback_base" / "meshes"
    if not d.is_dir():
        raise FileNotFoundError(
            f"{d} not found. The Ridgeback meshes are NOT part of the pinned asset "
            "set -- fetch them from github.com/ridgeback/ridgeback "
            "(ridgeback_description/meshes, BSD-3-Clause)."
        )
    return d


def _visual(body, **kw):
    """A geom that renders but neither collides nor carries mass."""
    return body.add_geom(contype=0, conaffinity=0, density=0, group=0, **kw)


def dress_base(spec, prefix: str, base_size, mode: str = "fit", riser: bool = True):
    """Attach Ridgeback visual geometry to ``{prefix}base``.

    Args:
        spec: the MjSpec being assembled.
        prefix: robot namespace, e.g. ``"robot_0/"``.
        base_size: the config's ``[x, y, z]`` extents.
        mode: ``"fit"`` scales Ridgeback to ``base_size``; ``"true"`` keeps 1:1.
        riser: add a riser column from the Ridgeback deck up to ``base_size[2]``,
            so the arm keeps its configured mounting height. Real Ridgeback-plus-
            manipulator setups use a riser for exactly this reason.
    """
    if mode not in ("fit", "true"):
        raise ValueError(f"mode must be 'fit' or 'true', got {mode!r}")

    body = spec.body(f"{prefix}base")
    if body is None:
        raise ValueError(f"no body named {prefix}base -- call after add_robot_to_scene")

    sx, sy, sz = (float(v) for v in base_size)
    # Uniform scale so the mesh is never distorted; fit to the tighter axis.
    s = 1.0 if mode == "true" else min(sx / RB_CHASSIS[0], sy / RB_CHASSIS[1])
    scale = [s, s, s]

    mdir = _mesh_dir()
    for name in _MESHES:
        spec.add_mesh(name=f"{prefix}rb_{name}", file=str(mdir / f"{name}.stl"), scale=scale)

    # Ridgeback's chassis origin sits at axle height; lift so wheels touch z=0.
    z0 = RB_WHEEL_RADIUS * s

    # Chassis. Placements follow ridgeback.urdf.xacro link-by-link: the side
    # covers, end covers and lights each appear twice, the second flipped 180
    # degrees about z.
    _visual(body, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"{prefix}rb_body",
            pos=[0, 0, z0], rgba=YELLOW_RGBA)
    for quat, rgba in ((None, YELLOW_RGBA), (_FLIP_Z, YELLOW_RGBA)):
        kw = {} if quat is None else {"quat": quat}
        _visual(body, type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=f"{prefix}rb_side-cover", pos=[0, 0, z0], rgba=rgba, **kw)
    for quat in (None, _FLIP_Z):                      # front_cover / rear_cover
        kw = {} if quat is None else {"quat": quat}
        _visual(body, type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=f"{prefix}rb_end-cover", pos=[0, 0, z0],
                rgba=BLACK_RGBA, **kw)
    for quat, rgba in ((None, WHITE_RGBA), (_FLIP_Z, BLACK_RGBA)):  # front/rear lights
        kw = {} if quat is None else {"quat": quat}
        _visual(body, type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=f"{prefix}rb_lights", pos=[0, 0, z0], rgba=rgba, **kw)

    # top_link sits at riser_height - deck_height + deck_thickness above chassis.
    top_z = z0 + (RB_RISER_HEIGHT - RB_DECK_HEIGHT + RB_DECK_THICKNESS) * s
    _visual(body, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"{prefix}rb_top",
            pos=[0, 0, top_z], rgba=BLACK_RGBA)

    # Ridgeback's own riser: a 0.493 m box rotated 45 degrees about z, at
    # (0.225 - deck_thickness) above the chassis origin.
    rz = z0 + (0.225 - RB_DECK_THICKNESS) * s
    c45 = float(np.cos(np.pi / 8))
    _visual(body, type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.493 / 2 * s, 0.493 / 2 * s, RB_RISER_HEIGHT / 2 * s],
            pos=[0, 0, rz + RB_RISER_HEIGHT / 2 * s],
            quat=[c45, 0.0, 0.0, float(np.sin(np.pi / 8))], rgba=GREY_RGBA)

    for dx in (-RB_ROCKER_OFFSET * s, RB_ROCKER_OFFSET * s):
        _visual(body, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"{prefix}rb_rocker",
                pos=[dx, 0, z0], quat=_WHEEL_QUAT, rgba=BLACK_RGBA)
        for dy in (-RB_WHEEL_Y * s, RB_WHEEL_Y * s):
            _visual(body, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"{prefix}rb_wheel",
                    pos=[dx, dy, z0], quat=_WHEEL_QUAT, rgba=[0.09, 0.09, 0.10, 1.0])

    # Riser from the Ridgeback deck to the configured arm mounting height, so the
    # arm's workspace geometry stays where the manipulation policy expects it.
    # Extension column above Ridgeback's own 55 mm riser, up to the configured
    # arm mounting height. NOT part of the real robot -- Ridgeback's deck is
    # 0.28 m and base_size[2] defaults to 0.58 m, so the arm needs lifting to
    # keep its workspace where the manipulation policy expects it. Real
    # Ridgeback-plus-manipulator builds add a riser for the same reason.
    deck = rz + RB_RISER_HEIGHT * s
    if riser and sz > deck + 1e-3:
        h = sz - deck
        _visual(body, type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[0.493 / 2 * s * 0.62, 0.493 / 2 * s * 0.62, h / 2],
                pos=[0, 0, deck + h / 2], quat=[c45, 0.0, 0.0, float(np.sin(np.pi / 8))],
                rgba=GREY_RGBA)
        _visual(body, type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[sx / 2 * 0.52, sy / 2 * 0.52, 0.012],
                pos=[0, 0, sz - 0.012], rgba=BLACK_RGBA)
