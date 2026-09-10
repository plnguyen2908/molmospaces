"""Report robot<->scene contacts, so "no collision" is verified rather than assumed.

The demo may move the arm kinematically -- that part is a deliberate shortcut. What
it must NOT do is drive the base through walls and furniture: a robot that phases
through the house looks fake and proves nothing about navigation.

The only contact that is expected is the gripper holding an object. Everything else
-- base against a counter, arm against a door frame, head against a header -- is a
defect, and naming the pair matters because the fixes differ: a base/door-frame
contact means the route is too tight, an arm/receptacle contact means the carry pose
is too wide.

Pattern follows `robot_scene_contacts` in the LinearBot planner
(`policy/linearbot_policy/planner/skills/base.py`, `evaluation` branch).
"""

from __future__ import annotations

import numpy as np


def robot_scene_contacts(
    model,
    data,
    namespace: str = "robot_0/",
    carried: tuple[str, ...] = (),
    top: int = 8,
    floor_clearance: float = 0.25,
) -> list[tuple[str, str, float, float]]:
    """Live robot<->scene contacts, worst first: ``(robot_body, other_body, force, depth)``.

    ``carried`` names objects held in a gripper; their contacts are excluded because
    they touch the fingers every tick by construction, and would otherwise drown the
    log in noise exactly when it needs to be read quickly.

    Contacts below ``floor_clearance`` are ground contact, not collision, and are
    excluded. RB-Y1's base joints are planar, so the base rests permanently on the
    floor plane; iTHOR splits that surface across ``floor_*``, ``decals_*`` and
    ``mesh_*`` bodies, so a name-based filter misses two of the three. Before this,
    the log reported "8 contacts, base<->decals F=2996094N d=203mm" on every step of
    a perfectly clean drive. See the matching fix in ``tools/build_run.py``.
    """
    import mujoco

    carried_roots = set()
    for name in carried:
        try:
            carried_roots.add(int(model.body_rootid[int(model.body(name).id)]))
        except (KeyError, ValueError):
            continue

    out: list[tuple[str, str, float, float]] = []
    buf = np.zeros(6, dtype=float)
    for ci in range(int(data.ncon)):
        con = data.contact[ci]
        b1 = int(model.geom_bodyid[int(con.geom1)])
        b2 = int(model.geom_bodyid[int(con.geom2)])
        n1 = model.body(b1).name or ""
        n2 = model.body(b2).name or ""
        r1, r2 = n1.startswith(namespace), n2.startswith(namespace)
        if r1 == r2:
            # robot-vs-robot self-collision, or scene-vs-scene -- not what this asks
            continue
        robot_body, other_body = (n1, n2) if r1 else (n2, n1)
        if int(model.body_rootid[b2 if r1 else b1]) in carried_roots:
            continue
        if float(con.pos[2]) < floor_clearance:
            continue
        mujoco.mj_contactForce(model, data, ci, buf)
        out.append((robot_body, other_body, abs(float(buf[0])), -float(con.dist)))
    out.sort(key=lambda t: -t[2])
    return out[:top]


def summarize(contacts: list[tuple[str, str, float, float]]) -> str:
    if not contacts:
        return "no robot<->scene contact"
    parts = [
        f"{rb.split('/')[-1]}<->{ob.split('/')[-1][:28]} F={f:.0f}N d={d * 1000:.0f}mm"
        for rb, ob, f, d in contacts[:4]
    ]
    return f"{len(contacts)} contact(s): " + "; ".join(parts)
