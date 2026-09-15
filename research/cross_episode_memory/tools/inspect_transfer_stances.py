"""Inspect candidate departure and docking poses without executing a rollout."""

import sys

import mujoco
import numpy as np

from research.cross_episode_memory.tools.check_fridge_transfer import NS
from research.cross_episode_memory.tools.check_navigation_transfer import (
    NavigationTransfer,
    parse_args,
)


def main():
    r = NavigationTransfer(parse_args(sys.argv[1:]))
    try:
        mujoco.mj_forward(r.model, r.data)
        gate = r.nav_map_filter()
        for x, y in [
            (r.args.base_x, r.args.base_y),
            (r.args.base_x - 0.3, r.args.base_y),
            (-0.54, -0.54),
            (0.16, 1.66),
            (0.06, 1.66),
            (0.26, 1.76),
            (0.26, 1.56),
        ]:
            r.data.joint(NS + "base_x").qpos[0] = x
            r.data.joint(NS + "base_y").qpos[0] = y
            mujoco.mj_forward(r.model, r.data)
            contacts = []
            for c in r.data.contact:
                ns = [r.model.body(r.model.geom_bodyid[g]).name for g in [c.geom1, c.geom2]]
                if (
                    any(n.startswith(NS) for n in ns)
                    and not all(n.startswith(NS) for n in ns)
                    and c.dist < 0.002
                    and c.pos[2] > 0.05
                ):
                    contacts.append((ns, float(c.dist)))
            print(
                "POSE",
                x,
                y,
                "map",
                gate(np.array([x, y, 0.0])),
                "metric",
                r.navigation_penetration(r.data, False),
                "contacts",
                sorted(contacts, key=lambda c: c[1])[:2],
            )
    finally:
        r.writer.close()
        r.head_writer.close()
        r.renderer.close()


if __name__ == "__main__":
    main()
