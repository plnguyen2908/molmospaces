"""Compute resting placements: on a receptacle's surface, clear of what is there.

The demo moves objects kinematically, which is fine, but a kinematic move must not
leave an object *inside* geometry -- a mug embedded in an oven reads as a bug, not a
placement. This module computes where an object has to sit so it rests on the
surface and does not overlap its neighbours.

Two traps this exists to avoid, both hit in practice:

* **Body origin is not the top.** `FloorPlan3`'s oven has its origin at z=0.223
  while its top surface is at z=1.336, spread over 77 geoms on child bodies. A
  naive `origin + 0.55` put objects 0.56 m *below* the surface -- inside the
  appliance. Always measure the subtree AABB.
* **Geoms hang off descendants.** Looking only at geoms whose `geom_bodyid` equals
  the receptacle's body id finds nothing at all for these assets.
"""

from __future__ import annotations

import numpy as np

# Gap left between an object's underside and the surface, so it settles rather
# than starting interpenetrated.
CLEARANCE = 0.005
# Minimum planar separation from any object already on the receptacle.
NEIGHBOUR_RADIUS = 0.13


def _descendants(model, body_id: int) -> set[int]:
    """Body ids in the subtree rooted at ``body_id`` (inclusive)."""
    out = {body_id}
    changed = True
    while changed:
        changed = False
        for b in range(model.nbody):
            if b not in out and int(model.body_parentid[b]) in out:
                out.add(b)
                changed = True
    return out


def subtree_aabb(model, data, body_name: str) -> tuple[np.ndarray, np.ndarray] | None:
    """World-frame AABB over every geom in the body's subtree."""
    import mujoco

    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        return None
    bodies = _descendants(model, bid)
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    found = False
    for g in range(model.ngeom):
        if int(model.geom_bodyid[g]) not in bodies:
            continue
        found = True
        c = model.geom_aabb[g][:3]
        e = model.geom_aabb[g][3:]
        rot = data.geom_xmat[g].reshape(3, 3)
        pos = data.geom_xpos[g]
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    corner = pos + rot @ (c + np.array([sx * e[0], sy * e[1], sz * e[2]]))
                    lo = np.minimum(lo, corner)
                    hi = np.maximum(hi, corner)
    return (lo, hi) if found else None



def receptacle_sites(model, data, receptacle_name: str) -> list[np.ndarray]:
    """World positions of the receptacle's placement sites, lowest shelf first.

    iTHOR receptacles carry explicit `*Receptacle*` sites marking where objects are
    meant to rest -- inside the fridge, on its shelves and in its door bins. Using the
    subtree AABB top instead puts objects on the appliance ROOF: measured z=2.528 for
    a fridge whose interior shelves sit at z=0.637..1.209, which makes opening the
    door pointless. Prefer these sites.
    """
    import mujoco

    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, receptacle_name)
    if bid < 0:
        return []
    sub = {bid}
    for b in range(model.nbody):
        parent = b
        while parent > 0:
            if parent in sub:
                sub.add(b)
                break
            parent = int(model.body_parentid[parent])
    out = []
    for sid in range(model.nsite):
        if int(model.site_bodyid[sid]) not in sub:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, sid) or ""
        if "receptacle" not in name.lower():
            continue
        out.append(np.asarray(data.site_xpos[sid], dtype=float).copy())
    out.sort(key=lambda p: float(p[2]))
    return out


def resting_position(
    model,
    data,
    obj_name: str,
    receptacle_name: str,
    occupied_xy: list[np.ndarray] | None = None,
) -> np.ndarray | None:
    """Where ``obj_name`` should sit to rest on ``receptacle_name``.

    Height comes from the receptacle's top plus the object's own half-height, so the
    object sits *on* the surface. The planar spot is chosen from a ring of candidates
    around the surface centre, maximising distance to ``occupied_xy`` -- incidental
    contact is tolerable, stacking objects inside each other is not.
    """
    rec = subtree_aabb(model, data, receptacle_name)
    obj = subtree_aabb(model, data, obj_name)
    if rec is None or obj is None:
        return None
    rec_lo, rec_hi = rec
    obj_lo, obj_hi = obj

    half_height = float(obj_hi[2] - obj_lo[2]) / 2.0

    # Prefer an explicit placement site (an actual shelf) over the AABB top (the roof).
    sites = receptacle_sites(model, data, receptacle_name)
    if sites:
        occupied_pts = [np.asarray(p[:2], dtype=float) for p in (occupied_xy or [])]
        best_site, best_gap = None, -np.inf
        for site in sites:
            gap = (
                min(float(np.linalg.norm(site[:2] - o)) for o in occupied_pts)
                if occupied_pts
                else np.inf
            )
            if gap > best_gap:
                best_gap, best_site = gap, site
            if gap >= NEIGHBOUR_RADIUS:
                best_site = site
                break
        return np.array(
            [best_site[0], best_site[1], float(best_site[2]) + half_height + CLEARANCE],
            dtype=float,
        )

    z = float(rec_hi[2]) + half_height + CLEARANCE

    centre = (rec_lo[:2] + rec_hi[:2]) / 2.0
    half_extent = np.maximum((rec_hi[:2] - rec_lo[:2]) / 2.0 - 0.12, 0.02)

    occupied = [np.asarray(p[:2], dtype=float) for p in (occupied_xy or [])]
    best, best_score = centre, -np.inf
    # Centre first, then rings outward; keeps objects near the middle of the surface
    # unless a neighbour is already there.
    candidates = [centre]
    for radius_frac in (0.35, 0.7):
        for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            offset = np.array([np.cos(angle), np.sin(angle)]) * half_extent * radius_frac
            candidates.append(centre + offset)
    for cand in candidates:
        if not occupied:
            best = cand
            break
        score = min(float(np.linalg.norm(cand - o)) for o in occupied)
        if score > best_score:
            best_score, best = score, cand
        if score >= NEIGHBOUR_RADIUS:
            best = cand
            break
    return np.array([best[0], best[1], z], dtype=float)
