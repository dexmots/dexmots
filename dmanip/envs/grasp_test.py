import warp as wp
import os
import warp.sim
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import json

from scipy.spatial.transform import Rotation as R
from localscope import localscope
from ..utils import builder as bu
from .hand_env import HandObjectTask
from ..utils import torch_utils as tu
from ..utils.common import HandType, ObjectType, ActionType, GraspParams, load_grasps_npy, run_env
from pyglet.math import Vec3 as PyVec3

wp.init()


def main(args):
    grasps = json.loads(open("./graspdata/good_grasps.txt").read())
    allegro_grasps = "./graspdata/allegro"
    # print("allegro grasps:", f"{allegro_grasps}/")
    # print("shadow grasps:", f"{shadow_grasps}/")

    def get_object_type_id(object_code):
        object_type = "_".join(filter(lambda x: not (x == "merged" or x.isdigit()), object_code.split("_")))
        object_id = object_code.rstrip("_merged").split("_")[-1]
        if not object_id.isdigit():
            object_id = None
        return object_type, object_id

    for grasp_dict in grasps["grasps"]:
        object_code = grasp_dict["object_code"]
        object_type, object_id = get_object_type_id(object_code)
        grasp_dict.update(dict(object_type=object_type, object_id=object_id))
        grasp_npy = os.path.join(allegro_grasps, f"{object_code}.npy")
        grasp_dict["grasp_file"] = grasp_npy
        grasp_params = map(
            lambda x: x[1], filter(lambda x: x[0] in grasp_dict["grasp_ids"], enumerate(load_grasps_npy(grasp_npy)))
        )
        grasp_dict["params"] = list(grasp_params)

    g = [g for g in grasps["grasps"] if g["object_code"] == args.object_code][args.grasp_id]

    object_type = g["object_type"]
    object_id = g.get("object_id", None)

    gparams = g["params"]
    from ..utils.rewards import l1_dist

    rew_params = {"hand_joint_pos_err": (l1_dist, ("target_qpos", "hand_qpos"), 1.0)}
    env = HandObjectTask(
        len(gparams),
        1,
        episode_length=1000,
        object_type=ObjectType[object_type.upper()],
        object_id=object_id,
        stochastic_init=True,
        reward_params=rew_params,
        load_grasps=True,
        grasp_id=2,
        hand_start_position=(0.1, 0.11485 * 2.3, 0),
        hand_start_orientation=(0, 0, 0),
        action_type=ActionType["POSITION"],
    )
    # env.grasp_joint_q = None
    # env.grasps = gparams
    pi = lambda x, t: tu.unscale(env.hand_init_q, env.action_bounds[0], env.action_bounds[1])
    run_env(env, pi, num_steps=1000)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--object_code", type=str, default="Stapler_102990_merged")
    parser.add_argument("--grasp_id", type=int, default=0)
    args = parser.parse_args()
    main(args)
