# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# gradient-based policy optimization by actor critic method
from dmanip.algorithms.shac2 import SHAC
import os
import yaml
import torch
from copy import copy

import numpy as np

from dmanip.utils.run_utils import (
    parse_arguments,
    parse_diff_env_kwargs,
    get_time_stamp,
)


def get_args():  # TODO: delve into the arguments
    custom_parameters = [
        {
            "name": "--test",
            "action": "store_true",
            "default": False,
            "help": "Run trained policy, no training",
        },
        {
            "name": "--cfg",
            "type": str,
            "default": "./cfg/shac/claw-det.yaml",
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
            "help": "Path to the saved weights",
        },
        {
            "name": "--run-id",
            "type": str,
            "default": None,
            "help": "load and continue logging run_id",
        },
        {"name": "--logdir", "type": str, "default": "logs/tmp/shac/"},
        {"name": "--save-interval", "type": int, "default": 0},
        {
            "name": "--no-time-stamp",
            "action": "store_true",
            "default": False,
            "help": "whether not add time stamp at the log path",
        },
        {"name": "--device", "type": str, "default": "cuda:0"},
        {"name": "--multi-gpu", "action": "store_true", "default": False},
        {"name": "--seed", "type": int, "default": 0, "help": "Random seed"},
        {
            "name": "--render",
            "action": "store_true",
            "default": False,
            "help": "whether generate rendering file.",
        },
        {
            "name": "--wandb",
            "action": "store_true",
            "default": False,
            "help": "whether to log with wandb.",
        },
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
        {
            "name": "--critic_lr",
            "help": "critic learning rate",
            "type": float,
            "default": None,
        },
        {
            "name": "--actor_lr",
            "help": "actor learning rate",
            "type": float,
            "default": None,
        },
        {
            "name": "--target_alpha",
            "help": "target critic update polyak factor",
            "type": float,
            "default": None,
        },
        {
            "name": "--max_epochs",
            "help": "number of epochs to run",
            "type": int,
            "default": None,
        },
        {
            "name": "--notes",
            "help": "notes to add to wandb",
            "type": str,
            "default": None,
        },
    ]

    # parse arguments
    args = parse_arguments(description="SHAC", custom_parameters=custom_parameters)

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

    if args.multi_gpu:
        cfg_train["params"]["config"]["multi_gpu"] = True
    else:
        args.device = torch.device(args.device)
        cfg_train["params"]["config"]["multi_gpu"] = False

    vargs = vars(args)
    notes = vargs.pop("notes")
    multi_gpu = vargs.pop("multi_gpu")
    rank = int(os.getenv("LOCAL_RANK", "0"))
    world_rank = int(os.getenv("WORLD_SIZE", "1"))

    # get reward coefficients from args
    if "diff_env" not in cfg_train["params"]["config"]:
        cfg_train["params"]["config"]["diff_env"] = cfg_train["params"].get("diff_env", {})

    rew_kw = cfg_train["params"]["config"]["diff_env"].get("rew_kw", {})
    for key in ["c_act", "c_finger", "c_q", "c_pos", "c_ft"]:
        if key in vargs and vargs[key] is not None:
            rew_kw[key] = vargs.pop(key)
    cfg_train["params"]["config"]["diff_env"]["rew_kw"] = rew_kw

    cfg_train["params"]["general"] = {}
    for key in vargs.keys():
        if key == "seed" and vargs[key] is not None:
            if multi_gpu:
                vargs[key] += rank
            cfg_train["params"]["general"]["seed"] = vargs[key]
            cfg_train["params"]["config"]["diff_env"]["seed"] = vargs[key]
        elif key == "max_epochs" and vargs[key] is not None:
            cfg_train["params"]["config"]["max_epochs"] = vargs[key]
        elif key == "actor_lr" and vargs[key] is not None:
            cfg_train["params"]["config"]["actor_learning_rate"] = vargs[key]
        elif key == "critic_lr" and vargs[key] is not None:
            cfg_train["params"]["config"]["critic_learning_rate"] = vargs[key]
        elif key == "target_alpha" and vargs[key] is not None:
            cfg_train["params"]["config"]["target_critic_alpha"] = vargs[key]
        elif key == "device" and vargs[key] is not None and multi_gpu:
            if rank > world_rank:
                cfg_train["params"]["general"][key] = "cpu"
            else:
                cfg_train["params"]["general"][key] = "cuda:{}".format(rank)
        else:
            cfg_train["params"]["general"][key] = vargs[key]

    if args.wandb and (not multi_gpu or rank == 0):
        import wandb

        # if using default logdir, skip run_name
        if (
            vargs["logdir"] == "logs/tmp/shac/"
            or os.path.split(vargs["logdir"])[0] == "logs/tmp/shac/"
            or not vargs["no_time_stamp"]
            or vargs["run_id"]
        ):
            run_name = None
        else:
            run_name = os.path.split(vargs["logdir"].rstrip("/"))[1]
        run_id = vargs["run_id"]

        if run_id:
            wandb.init(
                project="diff_manip",
                entity="dmanip-rss",
                sync_tensorboard=True,
                resume="allow",
                name=run_name,
                id=run_id,
                notes=notes,
            )
            cfg_train["params"]["general"].update(wandb.config["general"])
        else:
            wandb.init(
                project="diff_manip",
                entity="dmanip-rss",
                sync_tensorboard=True,
                resume="allow",
                name=run_name,
                config=cfg_train["params"],
                notes=notes,
            )

    cfg_train["params"]["diff_env"] = parse_diff_env_kwargs(cfg_train["params"]["config"]["diff_env"])

    traj_optimizer = SHAC(cfg_train)

    if args.wandb and (not multi_gpu or rank == 0) and (wandb.run.resumed and not wandb.run.sweep_id):
        log_dir = wandb.config["general"]["logdir"]
        loadable = filter(lambda x: x.endswith(".pt"), os.listdir(log_dir))
        path = "crashed.pt" if "crashed.pt" in loadable else "best_policy.pt"
        model_path = os.path.join(log_dir, path)
        print("continuing from checkpoint:", model_path, "step:", wandb.run.summary._step)
        traj_optimizer.resume_from(
            model_path,
            cfg_train,
            wandb.run.summary._step,
            wandb.run.summary.global_step,
            wandb.run.summary["best_policy_loss/step"],
        )

    if args.train:
        if args.checkpoint:
            traj_optimizer.load(args.checkpoint, cfg_train)
        traj_optimizer.train()
    else:
        traj_optimizer.play(cfg_train)
