"""Closed fridge: open, fetch an annotated native egg, place, and close."""

import numpy as np

from research.cross_episode_memory.tools.check_annotated_grasp import AnnotatedGraspMixin
from research.cross_episode_memory.tools.check_navigation_transfer import (
    NavigationTransfer,
    parse_args,
)


class AnnotatedFridgeTask(AnnotatedGraspMixin, NavigationTransfer):
    def __init__(self, args):
        if not (args.kitchen and args.native_object and args.operate_door):
            raise ValueError("Full task requires --kitchen --native-object --operate-door")
        if args.pickup_only or args.carry_only:
            raise ValueError("Full fridge task cannot use pickup-only or carry-only")
        super().__init__(args)
        self.release_clearance_m = 0.003
        self.report["scope"] = (
            "closed fridge: approach/open, navigate to native annotated object, "
            "pick/carry, lower/release on shelf, withdraw and close"
        )

    def after_placement(self):
        # Keep the NavigationTransfer door sequence; the stationary diagnostic's
        # prepare_pickup override must never replace opening and navigation here.
        super().after_placement()
        self.check_final_payload()

    def check_final_payload(self):
        shelf = self.data.site_xpos[self.shelf_id]
        half_x = self.model.site_size[self.shelf_id][2]
        half_y = self.model.site_size[self.shelf_id][0]
        vertices = self.bread_vertices()
        inside = bool(
            vertices[:, 0].min() > shelf[0] - half_x - 0.01
            and vertices[:, 0].max() < shelf[0] + half_x + 0.01
            and vertices[:, 1].min() > shelf[1] - half_y - 0.01
            and vertices[:, 1].max() < shelf[1] + half_y + 0.01
            and abs(vertices[:, 2].min() - self.report["target_shelf_surface_z"]) < 0.015
        )
        supports = self.support_contacts()
        speed = float(np.linalg.norm(self.data.joint(self.object_joint).qvel[:3]))
        released = not self.contacts()
        result = {
            "inside_target_shelf": inside,
            "supported": bool(supports),
            "released": released,
            "speed_m_s": speed,
        }
        self.report["object_after_door_closed"] = result
        self.record(**result)
        if not inside or not supports or not released or speed > 0.03:
            raise RuntimeError(
                "Object did not remain released and supported inside the closed fridge"
            )

    def finalize_report(self):
        super().finalize_report()
        # Recheck after the final recording dwell as well as after closing.
        if self.report.get("success"):
            try:
                self.check_final_payload()
            except RuntimeError as exc:
                self.report["success"] = False
                self.report["error"] = str(exc)


if __name__ == "__main__":
    raise SystemExit(AnnotatedFridgeTask(parse_args()).run())
