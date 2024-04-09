import sys, os

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_dir)

import time

import argparse

import torch
import random
import numpy as np
import copy
from hydra.experimental import initialize, compose
import matplotlib.pyplot as plt
import warp as wp
import yaml
import dmanip.utils.torch_utils as tu
from rl_games.torch_runner import Runner, _restore, _override_sigma

from dmanip import envs

# df.config.verify_fp = True

# df.config.check_grad = True


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


parser = argparse.ArgumentParser("")
parser.add_argument("--env", type=str, default="ReposeTask")
parser.add_argument("--stochastic-policy", action="store_true")
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--length", type=int, default=16)
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--finite-diff-check", default=False, action="store_true")
parser.add_argument("--ppo", default=False, action="store_true")

args = parser.parse_args()

seed = 0
set_seed(seed)

device = args.device
num_envs = args.num_envs
length = args.length

env_fn = getattr(envs, args.env)

env_kwargs = dict(
    num_envs=num_envs,
    seed=seed,
    no_grad=False,
    stochastic_init=False,
    device=device,
    render=False,
)
if args.env == "ReposeCube":
    env_kwargs["goal_rot_seed"] = 120


# evaluate the central value and gradient
single_env_kwargs = dict(
    num_envs=1,
    seed=seed,
    no_grad=False,
    stochastic_init=False,
    device=device,
    render=True,
)
if args.env == "ReposeCube":
    single_env_kwargs["goal_rot_seed"] = 120

env = env_fn(**single_env_kwargs)

env.clear_grad()
obs = env.reset()

num_actions = env.num_actions

actions = torch.zeros((length, 1, num_actions), dtype=torch.float32, device=device, requires_grad=True)

# compute action sequence from policy
obs_rms = None
env_state = None
if args.ppo:
    from .run_task import init_runner_and_cfg

    assert args.checkpoint is not None

    # load an omegaconf from cfg/task/repose_task.yaml using hydra compose
    with initialize(config_path="cfg"):
        cfg = compose(
            config_name="run_task",
            overrides=["alg=ppo", "task=repose", "task.env.no_grad=false"],
        )
    runner = init_runner_and_cfg(cfg)
    env.reset()
    # env_state = env.get_checkpoint()

elif args.checkpoint is not None:
    checkpoint = torch.load(args.checkpoint)
    if isinstance(checkpoint, dict):
        policy = checkpoint[1].to(args.device)
        obs_rms = checkpoint[4].to(args.device)
        # collect action sequence
        with torch.no_grad():
            obs = env.reset()
            for i in range(length):
                if obs_rms is not None:
                    obs = obs_rms.normalize(obs)
                u = policy(obs[0, :], deterministic=True)
                actions[i, 0, :] = u
                obs, _, _, _ = env.step(torch.tanh(u.unsqueeze(0).repeat((env.num_environments, 1))))
    else:
        actions = checkpoint

    # env_state = env.get_checkpoint()
else:
    # actions = torch.zeros((length, 1, num_actions), dtype = torch.float32, device = device, requires_grad = True)
    actions = torch.randn((length, 1, num_actions), dtype=torch.float32, device=device, requires_grad=True)

actions.requires_grad = True

# test forward and backward
print("start forward")

loss_total = torch.tensor(0.0, dtype=torch.float32, device=device)

env.clear_grad()
env.reset()
for i in range(length):
    obs, reward, _, _ = env.step(torch.tanh(actions[i, :, :]))
    loss_total -= reward.mean()

print("start backward")

loss_total.backward()

grad_norm = tu.grad_norm([actions])
max_grad = torch.abs(actions.grad).max()
print("loss = ", loss_total, ", grad norm = ", grad_norm, ", max grad = ", max_grad)

step_id = 0
action_id = 1

grad_0 = actions.grad[step_id][0][action_id]
grad = actions.grad.clone()

print("param = ", actions[step_id, 0, action_id])

# evaluate function space
env = env_fn(**env_kwargs)

# xs = np.arange(-1e-1, 1e-1, 1e-4, dtype=np.float32)
# xs = np.arange(-1e-1, 1e-1, 1e-3, dtype = np.float32)
# xs = np.arange(-1., 1., 1e-4, dtype = np.float32)
# xs = np.arange(-1., 1., 1e-3, dtype = np.float32)
xs = np.arange(-1e-3, 1e-3, 1e-6, dtype=np.float32)

losses = np.zeros(len(xs), dtype=np.float32)


def evaluate(eval_actions):
    eval_losses = torch.zeros(num_envs, dtype=torch.float, device=device)
    env.reset()
    for i in range(length):
        u = eval_actions[i, :, :]
        _, rew, _, _ = env.step(torch.tanh(u))
        eval_losses -= rew

    return eval_losses


with torch.no_grad():
    i = 0
    while i < len(xs):
        print("{} / {}".format(i, len(xs)))
        j = min(len(xs), i + num_envs)
        eval_actions = torch.zeros((length, num_envs, num_actions), dtype=torch.float32, device=device)
        for idx in range(i, j):
            eval_actions[:, idx - i, :] = copy.deepcopy(actions.squeeze(1))
            eval_actions[step_id, idx - i, action_id] += xs[idx]
        losses_seg = evaluate(eval_actions)
        losses[i:j] = losses_seg[: j - i].detach().cpu().numpy()
        i = j

# import IPython
# IPython.embed()
print("f = ", loss_total.item(), ", grad = ", grad_0.item(), "fd grad = ", (losses[-1] - losses[0]) / (xs[-1] - xs[0]))
print("max - min = ", np.max(losses) - np.min(losses))

np.save(open("function_space_open_loop.npy", "wb"), [xs, losses])

fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(xs, losses)
ax.set_xlabel("d$theta$")
ax.set_ylabel("loss")
plt.show()

# compare to finite difference
if args.finite_diff_check:
    grad = grad.squeeze(1)
    grad_unit = grad / grad.norm()
    env = env_fn(**env_kwargs)

    eps_list = []
    abs_error_list = []
    rel_error_list = []
    print("grad.norm = ", grad.norm().item())
    with torch.no_grad():
        eps = 4e-1
        # eps = 1e-2
        for i in range(15):
            eval_actions = torch.zeros((length, num_envs, num_actions), dtype=torch.float32, device=device)
            loss_pos = torch.zeros((length, num_actions), dtype=torch.float32, device=device)
            loss_neg = torch.zeros((length, num_actions), dtype=torch.float32, device=device)

            round = (2 * num_actions * length - 1) // num_envs + 1
            for j in range(round):
                idx_start = j * num_envs
                idx_end = min((j + 1) * num_envs, 2 * num_actions * length)
                for k in range(idx_start, idx_end):
                    eval_actions[:, k - idx_start, :] = copy.deepcopy(actions.squeeze(1))
                    step_idx = k // 2 // num_actions
                    action_idx = (k // 2) % num_actions
                    if k % 2 == 0:
                        eval_actions[step_idx, k - idx_start, action_idx] += eps
                    else:
                        eval_actions[step_idx, k - idx_start, action_idx] -= eps
                losses = evaluate(eval_actions)
                for k in range(idx_start, idx_end):
                    step_idx = k // 2 // num_actions
                    action_idx = (k // 2) % num_actions
                    if k % 2 == 0:
                        loss_pos[step_idx, action_idx] = losses[k - idx_start]
                    else:
                        loss_neg[step_idx, action_idx] = losses[k - idx_start]

            grad_fd = (loss_pos - loss_neg) / (eps * 2.0)

            abs_error = (grad_fd - grad).norm()
            rel_error = abs_error / max(grad_fd.norm(), grad.norm())
            grad_fd_unit = grad_fd / grad_fd.norm()
            dot = (grad_fd_unit * grad_unit).sum()

            print("eps = {}, abs error = {}, rel error = {}, dot = {}".format(eps, abs_error, rel_error, dot))

            eps_list.append(eps)
            abs_error_list.append(abs_error.item())
            rel_error_list.append(rel_error.item())

            eps /= 2.0

    eps_list = np.array(eps_list)
    abs_error_list = np.array(abs_error_list)
    rel_error_list = np.array(rel_error_list)

    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    ax[0].plot(eps_list, abs_error_list)
    ax[0].set_xlabel("eps")
    ax[0].set_ylabel("abs error")
    ax[1].plot(eps_list, rel_error_list)
    ax[1].set_xlabel("eps")
    ax[1].set_ylabel("rel error")

    plt.show()
