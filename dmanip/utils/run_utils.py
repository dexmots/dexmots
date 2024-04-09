import argparse
import torch


def parse_arguments(description="Testing Args", custom_parameters=[]):
    parser = argparse.ArgumentParser()

    for argument in custom_parameters:
        if ("name" in argument) and ("type" in argument or "action" in argument):
            help_str = ""
            if "help" in argument:
                help_str = argument["help"]

            if "type" in argument:
                if "default" in argument:
                    parser.add_argument(
                        argument["name"],
                        type=argument["type"],
                        default=argument["default"],
                        help=help_str,
                    )
                else:
                    print("ERROR: default must be specified if using type")
            elif "action" in argument:
                parser.add_argument(argument["name"], action=argument["action"], help=help_str)
        else:
            print()
            print("ERROR: command line argument name, type/action must be defined, argument not added to parser")
            print("supported keys: name, type, default, action, help")
            print()

    args = parser.parse_args()

    if args.test:
        args.play = args.test
        args.train = False
    elif args.play:
        args.train = False
    else:
        args.train = True

    return args


from datetime import datetime


def get_time_stamp():
    now = datetime.now()
    year = now.strftime("%Y")
    month = now.strftime("%m")
    day = now.strftime("%d")
    hour = now.strftime("%H")
    minute = now.strftime("%M")
    second = now.strftime("%S")
    return "{}-{}-{}-{}-{}-{}".format(month, day, year, hour, minute, second)


import os
from .common import GoalType, ActionType, ObjectType
from rl_games.common.algo_observer import AlgoObserver
from rl_games.algos_torch import torch_ext
from rl_games.common import env_configurations, vecenv
from tensorboardX import SummaryWriter
from dmanip import envs


def parse_diff_env_kwargs(diff_env):
    env_kwargs = {}
    for key, value in diff_env.items():
        if key == "goal_type":
            env_kwargs["goal_type"] = GoalType(value)
        elif key == "action_type":
            env_kwargs["action_type"] = ActionType(value)
        elif key == "object_type":
            env_kwargs["object_type"] = ObjectType(value)
        elif key == "stochastic_env":
            env_kwargs["stochastic_init"] = value
        elif key == "rew_kw" and isinstance(value, dict):
            env_kwargs["rew_kw"] = RewardConfig(**value)
        else:
            env_kwargs[key] = value
    print("parsed kwargs:", env_kwargs)
    return env_kwargs


def register_env_deprecated(cfg_train, render, rl_device):
    from dmanip.config import ClawWarpConfig, AllegroWarpConfig, RewardConfig
    from gym import wrappers

    def create_warp_env(**env_kwargs):
        env_name = cfg_train["params"]["diff_env"].pop("name")
        env_fn = getattr(envs, env_name)
        parsed_kwargs = parse_diff_env_kwargs(cfg_train["params"]["diff_env"])
        # parsed kwargs from config.params.diff_env get final priority
        # env_kwargs originate from config.params.config.env_config
        env_kwargs.update(parsed_kwargs)
        frames = env_kwargs.pop("frames", 1)
        if "no_grad" in env_kwargs:
            env_kwargs.pop("no_grad")
        if "render" in env_kwargs:
            env_kwargs.pop("render")
        if env_name == "ClawWarpEnv":
            config = ClawWarpConfig(
                num_envs=cfg_train["params"]["config"]["num_actors"],
                render=render,
                no_grad=True,
                **env_kwargs,
            )
        else:
            config = AllegroWarpConfig(
                num_envs=cfg_train["params"]["config"]["num_actors"],
                render=render,
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

    vecenv.register(
        "WARP",
        lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs),
    )
    env_configurations.register(
        "warp",
        {
            "env_creator": lambda **kwargs: create_warp_env(**kwargs),
            "vecenv_type": "WARP",
        },
    )


class RLGPUEnv(vecenv.IVecEnv):
    def __init__(self, config_name, num_actors, **kwargs):
        self.env = env_configurations.configurations[config_name]["env_creator"](**kwargs)

        self.full_state = {}

        self.rl_device = "cuda" if torch.cuda.is_available() else "cpu"

        self.full_state["obs"] = self.env.reset(force_reset=True).to(self.rl_device)
        print(self.full_state["obs"].shape)

    def step(self, actions):
        self.full_state["obs"], reward, is_done, info = self.env.step(actions.to(self.env.device))

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

    # def get_env_state(self):
    #     return self.env.get_checkpoint()
    #
    # def set_env_state(self, env_state=None):
    #     if env_state is not None and "joint_q" in env_state:
    #         self.env.load_checkpoint(env_state)


class RLGPUEnvAlgoObserver(AlgoObserver):
    def after_init(self, algo):
        self.algo = algo
        self.score_keys = self.algo.config.get("score_keys", [])
        # dummy mean_scores to keep parent methods from breaking
        # TODO: find way to integrate better
        games_to_track = 100
        if hasattr(self.algo, "games_to_track"):
            games_to_track = self.algo.games_to_track
        device = self.algo.config.get("device", "cuda:0")

        self.mean_scores_map = {
            k + "_final": torch_ext.AverageMeter(1, games_to_track).to(device) for k in self.score_keys
        }

        if hasattr(self.algo, "writer"):
            self.writer = self.algo.writer
        else:
            summaries_dir = self.algo.summaries_dir
            os.makedirs(summaries_dir, exist_ok=True)
            self.writer = SummaryWriter(summaries_dir)

    def process_infos(self, infos, done_indices):
        super().process_infos(infos, done_indices)
        if isinstance(infos, dict):
            for k, v in filter(lambda kv: kv[0] in self.score_keys, infos.items()):
                final_v = v[done_indices]
                if final_v.shape[0] > 0:
                    self.mean_scores_map[f"{k}_final"].update(final_v)

    def after_clear_stats(self):
        for score_values in self.mean_scores_map.values():
            score_values.clear()

    def after_print_stats(self, frame, epoch_num, total_time):
        super().after_print_stats(frame, epoch_num, total_time)
        for score_key in self.score_keys:
            score_values = self.mean_scores_map[score_key + "_final"]
            if score_values.current_size > 0:
                mean_scores = score_values.get_mean()
                self.writer.add_scalar(f"scores/{score_key}/step", mean_scores, frame)
                self.writer.add_scalar(f"scores/{score_key}/iter", mean_scores, epoch_num)
                self.writer.add_scalar(f"scores/{score_key}/time", mean_scores, total_time)
