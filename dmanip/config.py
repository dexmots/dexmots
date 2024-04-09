from dataclasses import dataclass
from dmanip.utils.common import *
from typing import Optional


@dataclass(frozen=True)
class RewardConfig:
    c_act: float = 0.1
    """Action penalty coefficient"""

    c_finger: float = 0.2
    """Finger position error coefficient"""

    c_q: float = 10.0
    """Object orientation error coefficient"""

    c_pos: float = 0.0
    """object position error coefficient"""

    c_ft: float = 0.0
    """object goal force-torque error coefficient"""

    reward_type: RewardType = RewardType.DELTA
    """reward type to use"""

    def numpy(self) -> np.ndarray:
        return np.array([self.c_act, self.c_finger, self.c_q, self.c_pos, self.c_ft])


@dataclass(frozen=True)
class ClawWarpConfig:
    num_envs: int = 1
    """number of envs to create in parallel"""

    episode_length: int = 600
    """number of steps to run in env"""

    no_grad: bool = True
    """whether or not to let env be differentiable"""

    seed: int = 42
    """seed for env"""

    render: bool = False
    """whether to render the env"""

    device: str = "cuda"
    """device to run env on"""

    stochastic_init: bool = False
    """whether to use a stochastic initial state"""

    goal_type: GoalType = GoalType.ORIENTATION
    """goal type to use"""

    action_type: ActionType = ActionType.TORQUE
    """type of action space"""

    object_type: ObjectType = ObjectType.OCTPRISM
    """type of object to use"""

    debug: bool = False
    """whether to run env in debug mode (extra logging slowdown)"""

    stage_path: Optional[str] = None
    """path to stage file"""

    goal_path: Optional[str] = None
    """path to load goal trajectory from"""

    action_strength: float = 10
    """joint torque limits applied"""

    rew_kw: RewardConfig = RewardConfig(c_finger=0.2, c_q=100, c_act=0.001)
    """reward coefficients array"""

    capture_graph: bool = False
    """whether to capture the computation graph"""


@dataclass(frozen=True)
class AllegroWarpConfig:
    num_envs: int = 1
    """number of envs to create in parallel"""

    episode_length: int = 600
    """number of steps to run in env"""

    no_grad: bool = True
    """whether or not to let env be differentiable"""

    seed: int = 42
    """seed for env"""

    render: bool = False
    """whether to render the env"""

    device: str = "cuda"
    """device to run env on"""

    stochastic_init: bool = False
    """whether to use a stochastic initial state"""

    goal_type: GoalType = GoalType.ORIENTATION
    """goal type to use"""

    action_type: ActionType = ActionType.TORQUE
    """type of action space"""

    object_type: ObjectType = ObjectType.OCTPRISM
    """type of object to use"""

    debug: bool = False
    """whether to run env in debug mode (extra logging slowdown)"""

    stage_path: Optional[str] = None
    """path to stage file"""

    goal_path: Optional[str] = None
    """path to load goal trajectory from"""

    action_strength: float = 0.1
    """joint torque limits applied"""

    rew_kw: RewardConfig = RewardConfig(c_finger=0.2, c_q=100, c_act=0.0)
    """reward coefficients array"""


@dataclass(frozen=True)
class ComputeGradConfig:
    env_name: str = "ClawWarpEnv"
    """"name of environment"""

    num_envs: int = 1
    """"number of environments to initialize and optimize actions over"""

    num_steps: int = 5
    """number of steps to take in env"""

    policy: str = "random"
    """type of policy to use"""

    num_opt_steps: int = 32
    """number of optimizer steps"""

    debug: bool = False
    """check opt by logging/saving loss/training/ac/acgrad vals"""

    debug_cuda: bool = False
    """whether to debug cuda"""

    debug_cpu: bool = False
    """whether to debug cpu"""

    debug_actions: Optional[str] = None
    """Optional path to saved actions, should equal length of episode"""

    save_fig: bool = False
    """whether to save figure animation of updating action"""

    save_act: bool = False
    """whether to save the action after computing gradient"""

    gradcheck: bool = False
    """whether to debug with a grad check"""

    lr: Optional[float] = None
    """learning rate for updating actions"""

    render: bool = False
    """whether to render an episode of env after"""
