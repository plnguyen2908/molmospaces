"""Run the production native closing sequence from a recorded placed-loaf state."""

import argparse
import json
import traceback
from pathlib import Path

from research.cross_episode_memory.tools.check_native_task_tail import restore
from research.cross_episode_memory.tools.check_navigation_transfer import (
    NavigationTransfer,
    parse_args,
)


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--resume", type=Path, required=True)
    p.add_argument("--opening", type=Path)
    p.add_argument("--at-handle", action="store_true")
    local, remaining = p.parse_known_args()
    check = NavigationTransfer(parse_args(remaining))
    rows = json.loads((local.resume / "trace.json").read_text())
    restore(
        check,
        next(r for r in reversed(rows) if r["stage"] == "turn in place to fridge handle")
        if local.at_handle
        else rows[-1],
    )
    check.skip_native_close_navigation = local.at_handle
    check.data.actuator("robot_0/right_finger_act").ctrl[0] = -0.05
    check.opening_reference = (
        json.loads((local.opening / "trace.json").read_text()) if local.opening else rows
    )
    check.report["navigation"] = [{"restored_post_placement_state": True}]
    try:
        check.close_native_fridge()
        check.report["success"] = True
    except Exception as exc:
        check.report["error"] = str(exc)
        print(traceback.format_exc(), flush=True)
    finally:
        check.report["scope"] = "restored post-placement closing diagnostic only"
        check.writer.close()
        check.head_writer.close()
        check.renderer.close()
        (check.output / "report.json").write_text(json.dumps(check.report, indent=2))
        (check.output / "trace.json").write_text(json.dumps(check.trace))
    return 0 if check.report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
