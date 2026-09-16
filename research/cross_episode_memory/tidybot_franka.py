"""FR3 on a TidyBot++ base -- benchmark B3 in SPEC.md.

Upstream TidyBot++ ships only a Kinova Gen3: there is no Franka anywhere in that
repository. The base is arm-agnostic by design, which is what makes B3's "Franka
FR3 on TidyBot++ base" a graft rather than a download.

MolmoSpaces' MobileFrankaRobotConfig already mounts an FR3 on a holonomic base,
but on a 0.5 x 0.5 x 0.58 m box. A real TidyBot++ base measures 0.481 x 0.481 m
and presents its arm plate at 0.335 m, so the stock config sits the arm 24.5 cm
too high -- which decides whether the arm can reach a 0.68 m table at all.
Dimensions here are measured from the vendored base mesh, not copied from a
datasheet: see assets/tidybot_base/ and its PROVENANCE.txt.
"""

from pathlib import Path

from molmo_spaces.configs.robot_configs import MobileFrankaRobotConfig

VENDORED_BASE = Path(__file__).parent / "assets" / "tidybot_base"
BASE_XML = VENDORED_BASE / "models" / "stanford_tidybot" / "base.xml"

# Measured from the vendored mesh: bounding box 0.481 x 0.481 x 0.308, spanning
# z 0.027..0.335, with the arm plate's top face -- where the arm bolts on -- at
# 0.335. base_size[2] is the mount height the robot class attaches the arm at.
TIDYBOT_FOOTPRINT_M = (0.481, 0.481)
TIDYBOT_ARM_MOUNT_M = 0.335


class TidyBotFrankaConfig(MobileFrankaRobotConfig):
    """The stock mobile FR3, stood on a TidyBot++-sized base.

    The base is still the parent class's box rather than the vendored mesh. That
    keeps the control wiring untouched while making the two things that decide
    reachability and collision honest: the footprint and the mount height. Swap
    in the mesh when visual fidelity matters more than the diff.
    """

    name: str = "franka_droid"
    base_size: list[float] = [*TIDYBOT_FOOTPRINT_M, TIDYBOT_ARM_MOUNT_M]
