"""Evaluation configs for the cross-episode memory study.

Embodiments must come from the spec's B1-B4 table. In particular the robot needs a
**mobile base**: the task moves objects between receptacles metres apart, and
explore episodes require driving to both. A fixed-base arm cannot satisfy any of it.

Do NOT use `DummyBenchmarkEvalConfig` here -- it hardcodes `FrankaRobotConfig`,
whose `command_mode` is `{arm, gripper}` with no base.
"""

from molmo_spaces.configs.camera_configs import RBY1GoProD455CameraSystem
from molmo_spaces.configs.policy_configs import BasePolicyConfig, DummyPolicyConfig
from molmo_spaces.policy.base_policy import PolicyFactory
from molmo_spaces.utils.function_utils import make_lenient
from molmo_spaces.configs.robot_configs import RBY1MConfig
from molmo_spaces.evaluation.configs.evaluation_configs import JsonBenchmarkEvalConfig


class RBY1ReorderEvalConfig(JsonBenchmarkEvalConfig):
    """B1/B2 embodiment: RB-Y1 mobile bimanual, no-op policy.

    RB-Y1 is used because it is already fully integrated in MolmoSpaces (robot
    class, view, kinematics) and is the spec's platform-credible loco-manipulation
    robot. It also ships real MJCF cameras -- head plus both wrists -- so the demo
    needs no hand-placed camera.

    With `DummyPolicy` every episode is expected to FAIL. The point is that the
    harness runs, the arrangement chains across episodes, and the predicates report
    correctly.
    """

    robot_config: RBY1MConfig = RBY1MConfig()
    policy_config: DummyPolicyConfig = DummyPolicyConfig()
    camera_config: RBY1GoProD455CameraSystem = RBY1GoProD455CameraSystem()
    policy_dt_ms: float = 200.0
    use_wandb: bool = False
    terminate_upon_success: bool = False

    @property
    def tag(self) -> str:
        return "rby1_reorder"

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        if self.robot_config.action_noise_config is not None:
            self.robot_config.action_noise_config.enabled = False



class ScriptedDemoPolicyConfig(BasePolicyConfig):
    """Config for the scripted demo policy. DEMO ONLY -- see demo_policy.py."""

    policy_type: str = "scripted_demo"
    policy_cls: type = None
    policy_factory: PolicyFactory | None = None

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        if self.policy_cls is None:
            from research.cross_episode_memory.demo_policy import ScriptedDemoPolicy

            self.policy_cls = ScriptedDemoPolicy
            self.policy_factory = make_lenient(ScriptedDemoPolicy)


class RBY1DemoEvalConfig(RBY1ReorderEvalConfig):
    """RB-Y1 driving the episode chain with the scripted demo policy.

    **For demo videos only.** The policy teleports objects and reads privileged
    targets; it is not a baseline and must never be reported as one.

    The legacy `CuroboPickAndPlacePlannerPolicy` expects the older Ai2 cuRobo
    API. Installed NVIDIA cuRobo 1.0 is genuine cuRobo with a newer API; see
    `curobo_current.py` and `tools/check_transfer.py` for the physical component
    check. This demo remains a teleporting visualization, not that check.
    """

    policy_config: ScriptedDemoPolicyConfig = ScriptedDemoPolicyConfig()
    # Keep the scene's designed simulation timestep. base_scene.xml sets
    # timestep=0.002, and raising sim_dt to 4 ms made the solver struggle with
    # contacts: measured 484 s for 20 policy steps versus 97 s at 2 ms -- 5x
    # slower despite HALF the sim steps. The datagen config uses 4 ms for its own
    # scenes; do not copy it here.
    policy_dt_ms: float = 200.0
    ctrl_dt_ms: float = 2.0
    sim_dt_ms: float = 2.0

    @property
    def tag(self) -> str:
        return "rby1_scripted_demo"

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        # Head tilted down so the head camera frames the workspace, matching the
        # shipped RBY1PickAndPlaceDataGenConfig.
        self.robot_config.init_qpos["head"][1] = 0.6
