import sys, os
import time

import argparse

import torch
import random
import numpy as np
import copy
import matplotlib.pyplot as plt

# import dflex as df
import warp as wp
from shac.utils.running_mean_std import RunningMeanStd
import shac.models.actor
import shac.utils.torch_utils as tu

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
parser.add_argument("--params_id", type=int, default=0)
parser.add_argument("--row", type=int, default=0)
parser.add_argument("--col", type=int, default=0)
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--finite-diff-check", default=False, action="store_true")

args = parser.parse_args()

seed = 0
set_seed(seed)

device = args.device
num_envs = args.num_envs
length = args.length

env_fn = getattr(envs, args.env)

# evaluate the central value and gradient
env = env_fn(
    num_envs=1,
    seed=seed,
    no_grad=False,
    stochastic_init=False,
    device=device,
    render=True,
)

env.clear_grad()
obs = env.reset()

num_obs = env.num_obs
num_actions = env.num_actions

# compute action sequence from policy
obs_rms = None
if args.checkpoint is not None:
    checkpoint = torch.load(args.checkpoint)
    policy = checkpoint[1].to(args.device)
    obs_rms = checkpoint[4].to(args.device)
else:
    cfg_network = {"actor_mlp": {"units": [64, 64], "activation": "elu"}}
    # cfg_network = {'actor_mlp': {'units': [512, 256, 256], 'activation': 'elu'}}
    if args.stochastic_policy:
        policy = models.actor.ActorStochasticMLP(num_obs, num_actions, cfg_network, device=args.device)
    else:
        policy = models.actor.ActorDeterministicMLP(num_obs, num_actions, cfg_network, device=args.device)
    obs_rms = RunningMeanStd(device=args.device)


def evaluate(eval_policies):
    eval_losses = torch.zeros(len(eval_policies), dtype=torch.float, device=args.device)

    env.clear_grad()
    env.reset()
    obs = env.initialize_trajectory()
    for i in range(length):
        if obs_rms is not None:
            obs = obs_rms.normalize(obs)
        actions = torch.zeros((env.num_environments, num_actions), dtype=torch.float, device=args.device)
        for j in range(len(eval_policies)):
            actions[j, :] = eval_policies[j](obs[j, :], deterministic=True)
        obs, rew, _, _ = env.step(torch.tanh(actions))
        eval_losses -= rew[: len(eval_policies)]

    return eval_losses


# test forward and backward
params_id = args.params_id
row = args.row
col = args.col

loss = evaluate([policy])
loss.backward()

grad_norm = tu.grad_norm(policy.parameters())
max_grad = 0.0
for params in policy.parameters():
    if params.grad is not None:
        max_grad = max(torch.abs(params.grad).max(), max_grad)
print("loss = ", loss, ", grad norm = ", grad_norm, ", max grad = ", max_grad)

params = list(policy.parameters())
grad = params[params_id].grad.clone()

grad_0 = grad[row, col]

# evaluate function space
env = env_fn(
    num_envs=num_envs,
    seed=seed,
    no_grad=False,
    stochastic_init=False,
    device=device,
    render=False,
)

# xs = np.arange(-1e-1, 1e-1, 1e-4, dtype = np.float32)
xs = np.arange(-1e-2, 1e-2, 1e-4, dtype=np.float32)
# xs = np.arange(-1e-3, 1e-3, 1e-6, dtype = np.float32)
# xs = np.arange(-1e-5, 1e-5, 1e-6, dtype = np.float32)

losses = np.zeros(len(xs), dtype=np.float32)

with torch.no_grad():
    i = 0
    while i < len(xs):
        print("{} / {}".format(i, len(xs)))
        j = min(len(xs), i + args.num_envs)
        policies = []
        for idx in range(i, j):
            policies.append(copy.deepcopy(policy))
            params = list(policies[-1].parameters())
            params[params_id].data[row][col] += xs[idx]
        losses_seg = evaluate(policies)
        losses[i:j] = losses_seg.detach().cpu().numpy()
        i = j

print("f = ", loss.item(), ", grad = ", grad_0.item(), "fd grad = ", (losses[-1] - losses[0]) / (xs[-1] - xs[0]))
print("max - min = ", np.max(losses) - np.min(losses))

np.save(open("function_space_open_loop.npy", "wb"), [xs, losses])

fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(xs, losses)
ax.set_xlabel("d$theta$")
ax.set_ylabel("loss")
plt.show()

# compare to finite difference
if args.finite_diff_check:
    grad_unit = grad / grad.norm()

    env = env_fn(
        num_envs=num_envs,
        seed=seed,
        no_grad=False,
        stochastic_init=False,
        device=device,
        render=False,
    )

    eps_list = []
    abs_error_list = []
    rel_error_list = []
    nrows = grad.shape[0]
    ncols = grad.shape[1]
    print("grad.norm = ", grad.norm().item())
    with torch.no_grad():
        eps = 4e-1
        for i in range(10):
            loss_pos = torch.zeros((nrows, ncols), dtype=torch.float32, device=device)
            loss_neg = torch.zeros((nrows, ncols), dtype=torch.float32, device=device)

            round = (2 * nrows * ncols - 1) // num_envs + 1
            for j in range(round):
                print("round {}/{}".format(j, round), end="\r")
                policies = []
                idx_start = j * num_envs
                idx_end = min((j + 1) * num_envs, 2 * nrows * ncols)
                for k in range(idx_start, idx_end):
                    policies.append(copy.deepcopy(policy))
                    row_k = (k // 2) // ncols
                    col_k = (k // 2) % ncols
                    params = list(policies[-1].parameters())
                    if k % 2 == 0:
                        params[params_id].data[row_k][col_k] += eps
                    else:
                        params[params_id].data[row_k][col_k] -= eps
                losses = evaluate(policies)
                for k in range(idx_start, idx_end):
                    row_k = (k // 2) // ncols
                    col_k = (k // 2) % ncols
                    if k % 2 == 0:
                        loss_pos[row_k, col_k] = losses[k - idx_start]
                    else:
                        loss_neg[row_k, col_k] = losses[k - idx_start]

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
