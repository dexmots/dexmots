"""Renders and samples random actions for ClawEnv"""

import os
import pdb

import numpy as np
import torch
import warp as wp
import yaml
from rl_games.torch_runner import Runner

from dmanip.utils import run_utils
from dmanip.utils.plotting import render_fingertips
from dmanip.config import AllegroWarpConfig, ClawWarpConfig, RewardConfig
from dmanip.envs.claw_env import (
    ClawWarpEnv,
    GoalType,
    ObjectType,
    ActionType,
    RewardType,
)
from dmanip.envs.allegro_env import AllegroWarpEnv


def get_env_policy(ckpt, policy="shac", env_kwargs={}, n_eps=1, env_name="ClawWarpEnv"):
    ckpt_dir = os.path.dirname(ckpt)
    num_envs = env_kwargs.pop("num_envs", 1)
    if policy == "shac":
        config_path = os.path.join(ckpt_dir, "cfg.yaml")
        with open(config_path) as f:
            cfg = yaml.load(f, Loader=yaml.Loader)
        cfg["params"]["config"]["multi_gpu"] = False
        cfg["params"]["diff_env"]["name"] = "ClawWarpEnv"
        cfg["params"]["config"]["player"]["games_num"] = n_eps
        cfg["params"]["config"]["player"]["render"] = env_kwargs.get("render", False)
        cfg["params"]["config"]["num_actors"] = num_envs
        cfg["params"]["diff_env"] = cfg["params"].get("diff_env", {})
        env_kwargs = {
            "goal_type": env_kwargs["goal_type"],
            "rew_kw": env_kwargs["rew_kw"],
            "object_type": env_kwargs["object_type"],
            "no_grad": env_kwargs["no_grad"],
            "render": env_kwargs["render"],
        }
        cfg["params"]["diff_env"].update(env_kwargs)
        cfg["params"]["diff_env"] = run_utils.parse_diff_env_kwargs(cfg["params"]["diff_env"])
        cfg["params"]["diff_env"]["name"] = env_name
        cfg["params"]["general"] = {
            "general": False,
            "checkpoint": ckpt,
            "seed": 0,
            "device": "cuda:0",
            "render": env_kwargs.get("render", False),
            "train": False,
            "play": True,
        }

        agent = SHAC(cfg)
        agent.load(ckpt)

        @torch.no_grad()
        def pi(o):
            if agent.obs_rms:
                o = agent.obs_rms.normalize(o)
            ac = agent.actor(o, deterministic=True)
            return ac

    elif policy == "ppo":
        config_dir = os.path.dirname(ckpt_dir)
        env_kwargs.pop("no_grad", True)
        cfg_path = os.path.join(config_dir, "cfg.yaml")
        with open(cfg_path, "r") as f:
            cfg = yaml.load(f, Loader=yaml.Loader)
        cfg["params"]["config"]["player"]["games_num"] = 1
        cfg["params"]["config"]["num_actors"] = num_envs
        cfg["params"]["diff_env"] = cfg["params"].get("diff_env", {})
        cfg["params"]["diff_env"].update(env_kwargs)
        cfg["params"]["diff_env"]["name"] = env_name
        cfg["params"]["general"] = {"logdir": "", "general": False}
        run_utils.register_env(cfg, render=args.render, rl_device="cuda:0")

        runner = Runner()
        runner.load(cfg)
        agent = runner.create_player()
        agent.restore(ckpt)

        def pi(o):
            return agent.get_action(o, is_deterministic=True)

    return agent.env, pi


def get_action(t=None, state=None, period=60):
    # actions are left/right proximal/distal joints acts
    # action = np.zeros(4).astype(np.float32)
    a_i = np.sin(t / period)
    action = np.array([a_i * 0.5, a_i, -a_i * 0.8, -a_i * 0.75]).astype(np.float32) * 0.15
    # action[idx] = np.sin(t / 100 * np.pi)
    # action = np.array([0, 0, 0, 0]).astype(np.float32)
    # action = np.array([-0.75, 0.6, 0.75, -0.6]).astype(np.float32)
    return action


def main(
    n_envs,
    env_name="ClawWarpEnv",
    n_steps=250,
    n_eps=1,
    policy="random",
    debug=False,
    debug_cuda=False,
    debug_cpu=False,
    debug_ftips=False,
    path=None,
    debug_actions=None,
    diff_sim=False,
    shac_ckpt=None,
    ppo_ckpt=None,
    object_type="octprism",
    render=False,
    action_type=1,
    stochastic_init=False,
):
    print(f"Running {policy} policy")
    device = "cuda"
    if debug_cuda:
        wp.config.mode = "debug"
        wp.config.print_launches = True
        wp.config.verify_cuda = True
    if debug_cpu:
        wp.config.mode = "debug"
        # wp.config.print_launches = True
        device = "cpu"

    if path is not None and render:
        datetime = run_utils.get_time_stamp()
        stage_path = f"outputs/{datetime}-{path}.usd"
        print("rendering usd to:", stage_path)
    else:
        stage_path = None
    if debug_actions:
        np_actions = np.load(debug_actions)
        if debug_actions.endswith(".npz"):
            np_actions = np_actions["actions"]
        assert np.prod(np_actions.shape) == n_steps * n_envs * 4, "debug actions shape: {}".format(np_actions.shape)
        debug_ac = torch.as_tensor(
            np_actions.reshape(n_steps, n_envs, -1),
            dtype=torch.float32,
            device=device,
        )
    else:
        debug_ac = None

    env_kwargs = dict(
        num_envs=n_envs,
        render=render,
        debug=debug,
        device=device,
        stochastic_init=stochastic_init,
        stage_path=stage_path,
        goal_type=GoalType.ORIENTATION,
        # goal_type=GoalType.TRAJECTORY_ORIENTATION,
        # goal_type=GoalType.TRAJECTORY_ORIENTATION_TORQUE,
        # goal_type=GoalType.TRAJECTORY_POSE_WRENCH,
        goal_path="goal_traj-sliding-2.npy",
        object_type=ObjectType[object_type.upper()],
        episode_length=n_steps + 1,  # so auto-reset doesn't trigger at the end
        no_grad=not (diff_sim),
        rew_kw=RewardConfig(0.0, 0.0, 1.0, 0, 0, reward_type=RewardType.L2),
        action_type=ActionType(action_type),
    )

    if shac_ckpt:
        env, pi = get_env_policy(
            shac_ckpt,
            policy="shac",
            env_kwargs=env_kwargs,
            n_eps=n_eps,
            env_name=env_name,
        )
    elif ppo_ckpt:
        env, pi = get_env_policy(
            ppo_ckpt,
            policy="ppo",
            env_kwargs=env_kwargs,
            n_eps=n_eps,
            env_name=env_name,
        )
    else:
        if env_name == "ClawWarpEnv":
            env = ClawWarpEnv(ClawWarpConfig(**env_kwargs))
        else:
            env = AllegroWarpEnv(AllegroWarpConfig(**env_kwargs))
        pi = None

    for ep in range(n_eps):
        profile = None  # {}
        with wp.ScopedTimer("episode", active=False, detailed=False, dict=profile):
            rewards = collect_episode_rewards(env, n_steps, pi, policy, o, debug_ftips, debug_ac)
            if profile:
                print(profile)
            if debug:
                if path is None:
                    save_path = f"claw_warp_debug-ep-{ep}.npz"
                else:
                    save_path = f"{path}-ep-{ep}.npz"
                print("saving debug log:", save_path)
                env.save_log(save_path)

def collect_episode_rewards(env, n_steps, pi, policy, o, debug_ftips, debug_ac):
    o = env.reset()
    prev_q = env.extras["object_joint_pos"]
    net_qdelta = 0.0
    rewards = []
    n_envs = env.num_envs
    for t in range(n_steps):
        if debug_ac is not None:
            ac = debug_ac[t]
            if ac.shape[0] != n_envs:
                ac = np.tile(ac.reshape(1, -1), (n_envs, 1))
        elif pi is not None:
            ac = pi(o).detach()
        else:
            if policy == "random":
                ac = np.stack([env.action_space.sample() for i in range(n_envs)])
            elif policy == "zero":
                ac = np.zeros((env.num_envs, env.num_actions)).astype(np.float32)
            elif policy == "ring_finger":
                ac = np.zeros((env.num_envs, env.num_actions)).astype(np.float32)
                joint_type = env.model.joint_type.numpy()
                joint_id = env.model.joint_q_start.numpy()

                a_i = np.sin(t / 50)
                ac[:, -8:-4] = np.array([[a_i, a_i, -a_i, -a_i]] * env.num_envs)
            else:
                ac = get_action(t, o)
                if t >= 200:
                    ac *= 0
                if env_name == "AllegroWarpEnv":
                    ac = np.tile(np.concatenate([ac, ac, ac, ac], axis=-1), (n_envs, 1))

            ac = torch.tensor(ac).to(env.device)
        o, r, d, info = env.step(ac)
        if debug_ftips:
            scale = 0.1 if env_name == "ClawWarpEnv" else 0.001
            render_fingertips(env, scale)

        net_qdelta += torch.abs(info["object_joint_pos"] - prev_q).detach().cpu().numpy().sum().item()
        prev_q = info["object_joint_pos"]
        if t % 100 == 0:
            mean_rew = r.mean(dim=0).item()
            min_rew = r.min()
            max_rew = r.max()
            log = f"Step {t} | Net Ori change: {net_qdelta:.3f}"  # noqa
            log += f" | Mean rew: {mean_rew:.3f} | Min rew: {min_rew:.3f}"
            log += f" | Max rew: {max_rew:.3f}"
            print(log)
        rewards.append(r)
    return rewards






if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_steps", "-ns", type=int, default=250)
    parser.add_argument("--env_name", type=str, default="AllegroWarpEnv")
    parser.add_argument("--n_eps", "-neps", type=int, default=1)
    parser.add_argument("--policy", "-p", default="sine", type=str)
    parser.add_argument("--num_envs", "-ne", type=int, default=1)
    parser.add_argument("--debug", "-d", action="store_true")
    parser.add_argument("--debug_cuda", "-dc", action="store_true")
    parser.add_argument("--debug_cpu", "-dcp", action="store_true")
    parser.add_argument("--debug_ftips", "-dft", action="store_true")
    parser.add_argument("--debug_actions", type=str)
    parser.add_argument("--diff_sim", "-ds", action="store_true")
    parser.add_argument("--path", "-sp", type=str)
    parser.add_argument("--shac_ckpt", type=str)
    parser.add_argument("--ppo_ckpt", type=str)
    parser.add_argument("--object", "-o", default="octprism", type=str)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--stochastic_init", "-si", action="store_true")
    parser.add_argument("--action_type", "-at", type=int, default=1)
    args = parser.parse_args()
    main(
        args.num_envs,
        args.env_name,
        int(args.n_steps),
        int(args.n_eps),
        args.policy,
        args.debug,
        args.debug_cuda,
        args.debug_cpu,
        args.debug_ftips,
        args.path,
        args.debug_actions,
        args.diff_sim,
        args.shac_ckpt,
        args.ppo_ckpt,
        args.object,
        args.render,
        args.action_type,
        args.stochastic_init,
    )
