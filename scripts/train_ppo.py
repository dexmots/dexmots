# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner
import yaml
import os

import numpy as np
from gym import wrappers
from shac import envs
from dmanip.config import ClawWarpConfig, AllegroWarpConfig, RewardConfig
from dmanip.utils.run_utils import *


def create_warp_env(**env_kwargs):
    env_name = cfg_train["params"]["diff_env"].pop("name")
    env_fn = getattr(envs, env_name)
    parsed_kwargs = parse_diff_env_kwargs(cfg_train["params"]["diff_env"])
    # parsed kwargs from config.params.diff_env get final priority
    # env_kwargs originate from config.params.config.env_config
    env_kwargs.update(parsed_kwargs)
    frames = env_kwargs.pop("frames", 1)

    if env_name == "ClawWarpEnv":
        config = ClawWarpConfig(
            num_envs=cfg_train["params"]["config"]["num_actors"],
            render=args.render,
            no_grad=True,
            **env_kwargs,
        )
    elif env_name == "AllegroWarpEnv":
        config = AllegroWarpConfig(
            num_envs=cfg_train["params"]["config"]["num_actors"],
            render=args.render,
            no_grad=True,
            **env_kwargs,
        )
    env = env_fn(config)

    print("num_envs = ", env.num_envs)
    print("num_actions = ", env.num_actions)
    print("num_obs = ", env.num_obs)

    if frames > 1:
        env = wrappers.FrameStack(env, frames, False)

    return env


class RLGPUEnv(vecenv.IVecEnv):
    def __init__(self, config_name, num_actors, **kwargs):
        self.env = env_configurations.configurations[config_name]["env_creator"](
            **kwargs
        )

        self.full_state = {}

        self.rl_device = args.rl_device

        self.full_state["obs"] = self.env.reset(force_reset=True).to(self.rl_device)
        print(self.full_state["obs"].shape)

    def step(self, actions):
        self.full_state["obs"], reward, is_done, info = self.env.step(
            actions.to(self.env.device)
        )

        return (
            self.full_state["obs"].to(self.rl_device),
            reward.to(self.rl_device),
            is_done.to(self.rl_device),
            info,
        )

    def reset(self):
        self.full_state["obs"] = self.env.reset(force_reset=True)

        return self.full_state["obs"].to(self.rl_device)

    def get_number_of_agents(self):
        return 1

    def get_env_info(self):
        info = {}
        info["action_space"] = self.env.action_space
        info["observation_space"] = self.env.observation_space
        info["agents"] = self.get_number_of_agents()

        print(info["action_space"], info["observation_space"])

        return info

    def set_env_state(self, env_state):
        self.env.load_checkpoint(env_state)


vecenv.register(
    "WARP",
    lambda config_name, num_actors, **kwargs: RLGPUEnv(
        config_name, num_actors, **kwargs
    ),
)
env_configurations.register(
    "warp",
    {"env_creator": lambda **kwargs: create_warp_env(**kwargs), "vecenv_type": "WARP"},
)


def get_args():
    custom_parameters = [
        {
            "name": "--test",
            "action": "store_true",
            "default": False,
            "help": "Run trained policy, no training",
        },
        {
            "name": "--num_envs",
            "type": int,
            "default": 0,
            "help": "Number of envs",
        },
        {
            "name": "--cfg",
            "type": str,
            "default": "./cfg/rl/ant.yaml",
            "help": "Configuration file for training/playing",
        },
        {
            "name": "--play",
            "action": "store_true",
            "default": False,
            "help": "Run trained policy, the same as test",
        },
        {
            "name": "--checkpoint",
            "type": str,
            "default": None,
            "help": "Path to the saved weights, only for rl_games RL library",
        },
        {
            "name": "--rl_device",
            "type": str,
            "default": "cuda:0",
            "help": "Choose CPU or GPU device for inferencing policy network",
        },
        {
            "name": "--seed",
            "type": int,
            "default": None,
            "help": "Random seed, set to current time if None",
        },
        {
            "name": "--render",
            "action": "store_true",
            "default": False,
            "help": "whether generate rendering file.",
        },
        {
            "name": "--multi-gpu",
            "action": "store_true",
            "default": False,
            "help": "whether to use multi-gpu training.",
        },
        {"name": "--logdir", "type": str, "default": "logs/tmp/rl/"},
        {
            "name": "--no-time-stamp",
            "action": "store_true",
            "default": False,
            "help": "whether not add time stamp at the log path",
        },
        {
            "name": "--wandb",
            "action": "store_true",
            "default": False,
            "help": "whether not log with wandb",
        },
        {
            "name": "--notes",
            "type": str,
            "default": None,
            "help": "notes for wandb run",
        },
        # c_act, c_finger, c_q, c_pos, c_f/t
        {
            "name": "--c_act",
            "help": "act penalty coefficient",
            "type": float,
            "default": None,
        },
        {
            "name": "--c_finger",
            "help": "finger position error coefficient",
            "type": float,
            "default": None,
        },
        {
            "name": "--c_q",
            "help": "object orientation error coefficient",
            "type": float,
            "default": None,
        },
        {
            "name": "--c_pos",
            "help": "object position error coefficient",
            "type": float,
            "default": None,
        },
        {
            "name": "--c_ft",
            "help": "object goal force-torque error coefficient",
            "type": float,
            "default": None,
        },
        {"name": "--max_epochs", "type": int, "default": None, "help": "max epochs"},
    ]

    # parse arguments
    args = parse_arguments(description="RL Policy", custom_parameters=custom_parameters)

    return args


if __name__ == "__main__":

    args = get_args()

    with open(args.cfg, "r") as f:
        cfg_train = yaml.load(f, Loader=yaml.Loader)

    if args.play or args.test:
        cfg_train["params"]["config"]["num_actors"] = (
            cfg_train["params"]["config"].get("player", {}).get("num_actors", 1)
        )

    if not args.no_time_stamp:
        args.logdir = os.path.join(args.logdir, get_time_stamp())

    if args.num_envs > 0:
        cfg_train["params"]["config"]["num_actors"] = args.num_envs

    if args.multi_gpu:
        cfg_train["params"]["config"]["multi_gpu"] = True
        rank = int(os.getenv("LOCAL_RANK", "0"))
        world_rank = int(os.getenv("WORLD_SIZE", "1"))
        args.rl_device = "cuda:{}".format(rank)

    vargs = vars(args)
    multi_gpu = vargs.pop("multi_gpu")
    notes = vargs.pop("notes")

    # get reward coefficients from args
    cfg_train["params"]["config"]["env_config"] = {}
    rew_kw = cfg_train["params"]["config"]["env_config"].get("rew_kw", {})
    for key in ["c_act", "c_finger", "c_q", "c_pos", "c_ft"]:
        if key in vargs and vargs[key] is not None:
            rew_kw[key] = vargs.pop(key)
    cfg_train["params"]["config"]["env_config"]["rew_kw"] = RewardConfig(**rew_kw)

    cfg_train["params"]["general"] = {}
    for key in vargs.keys():
        if key == "seed" and vargs[key] is not None:
            if multi_gpu:
                vargs[key] += rank
            cfg_train["params"]["seed"] = vargs[key]
            cfg_train["params"]["config"]["env_config"]["seed"] = vargs[key]
        elif key == "max_epochs" and vargs[key] is not None:
            cfg_train["params"]["config"]["max_epochs"] = vargs[key]
        elif key == "rl_device":
            cfg_train["params"]["general"][key] = vargs[key]
            cfg_train["params"]["config"]["env_config"]["device"] = vargs[key]
        else:
            cfg_train["params"]["general"][key] = vargs[key]

    # save config
    if cfg_train["params"]["general"]["train"]:
        log_dir = cfg_train["params"]["general"]["logdir"]
        os.makedirs(log_dir, exist_ok=True)
        cfg_train["params"]["config"]["train_dir"] = log_dir
        cfg_train["params"]["config"]["logdir"] = log_dir
        # save config
        yaml.dump(cfg_train, open(os.path.join(log_dir, "cfg.yaml"), "w"))

    if args.wandb and (not multi_gpu or rank == 0):
        import wandb

        run = wandb.init(
            project="diff_manip",
            config=cfg_train["params"],
            entity="dmanip-rss",
            sync_tensorboard=True,
            resume="allow",
            notes=notes,
        )

    # add observer to log score keys
    if cfg_train["params"]["config"].get("score_keys"):
        algo_observer = RLGPUEnvAlgoObserver()
    else:
        algo_observer = None
    runner = Runner(algo_observer)
    runner.load(cfg_train)
    runner.reset()
    runner.run(vargs)

    if args.wandb and (not multi_gpu or rank == 0):
        wandb.save(os.path.join(log_dir, "nn/*.pth"))
        run.finish()
