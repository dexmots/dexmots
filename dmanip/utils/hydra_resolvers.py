from omegaconf import OmegaConf


def maybe_register(name, func):
    try:
        OmegaConf.register_new_resolver(name, func)
    except ValueError:
        print(f"{name} already registered")
    return


maybe_register("resolve_default", lambda default, arg: default if arg in ["", None] else arg)


def custom_eval(x):
    import numpy as np  # noqa
    import torch  # noqa

    return eval(x)


maybe_register("eval", custom_eval)


def resolve_child(default, node, arg):
    """
    Attempts to get a child node parameter `arg` from `node`. If not
        present, the return `default`
    """
    if arg in node:
        return node[arg]
    else:
        return default


maybe_register("resolve_child", resolve_child)


from dmanip.utils.common import ActionType, HandType, ObjectType, GoalType, RewardType
from dmanip.envs.environment import RenderMode

maybe_register("object", lambda x: ObjectType[x.upper()])
maybe_register("hand", lambda x: HandType[x.upper()])
maybe_register("action", lambda x: ActionType[x.upper()])
maybe_register("goal", lambda x: GoalType[x.upper()])
maybe_register("reward", lambda x: RewardType[x.upper()])
maybe_register("render", lambda x: RenderMode[x.upper()])
