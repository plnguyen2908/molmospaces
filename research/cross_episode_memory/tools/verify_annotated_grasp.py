"""Verify annotation provenance and recorded native pickup/carry evidence."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    report = json.loads((args.folder / "report.json").read_text())
    selection = report["annotation_selection"]
    with np.load(selection["file"]) as archive:
        original = archive["transforms"][selection["selected_index"]]
    telemetry = report["grasp_telemetry"]
    hold = [row for row in telemetry if row["stage"] == "hold loaf after carrying"]
    navigation = [row for row in report["navigation"] if row.get("carrying")]
    checks = {
        "controller_success": report["success"],
        "selected_pose_is_stored_annotation": bool(
            np.allclose(original, selection["local_transform"])
        ),
        "world_pose_uses_object_frame": bool(
            np.allclose(np.array(selection["object_pose"]) @ original, selection["world_transform"])
        ),
        "native_pose_not_overridden": report["kitchen_stats"]["native_object_unmodified"]
        and not report["arguments"].get("loaf_pos")
        and not report["arguments"].get("loaf_quat"),
        "object_lifted_10cm": max(row["object_rise_m"] for row in telemetry) >= 0.1,
        "carried_at_least_1m": sum(row["measured_distance_m"] for row in navigation) >= 1.0,
        "three_second_hold_recorded": len(hold) > 1 and hold[-1]["time"] - hold[0]["time"] >= 2.9,
        "both_fingers_loaded_at_finish": bool(hold) and min(hold[-1]["finger_forces_n"]) > 0.05,
        "penetration_within_1mm": report["max_finger_object_penetration_m"] <= 0.001,
        "no_sustained_contact_loss": report.get("max_hold_contact_loss_seconds", 0) < 0.25,
        "carry_collision_free": all(row["max_collision_metric_m"] <= 0.0 for row in navigation),
    }
    result = {
        "success": all(checks.values()),
        "checks": checks,
        "lift_m": max(row["object_rise_m"] for row in telemetry),
        "carry_distance_m": sum(row["measured_distance_m"] for row in navigation),
        "max_penetration_m": report["max_finger_object_penetration_m"],
        "final_finger_forces_n": hold[-1]["finger_forces_n"] if hold else [],
        "annotation_index": selection["selected_index"],
    }
    (args.folder / "annotated_grasp_validation.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
