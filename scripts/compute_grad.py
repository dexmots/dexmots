from collections import defaultdict
import os

import hydra
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
import numpy as np
from omegaconf import DictConfig
import torch
from tqdm import trange

import warp as wp


def check_grad(fn, inputs, eps=1e-6, atol=1e-4, rtol=1e-6):
    if inputs.grad is not None:
        inputs.grad.zero_()
    out = fn(inputs)
    out.backward()
    analytical = inputs.grad.clone()
    x2, x1 = inputs + eps, inputs - eps
    numerical = (fn(x2) - fn(x1)) / (2 * eps)
    assert torch.allclose(
        numerical, analytical, rtol, atol
    ), "numerical gradient was: {}, analytical was: {}".format(numerical, analytical)
    return (numerical, analytical)


def run_env(env, actions, checkpoint=None):
    n = len(actions)
    env.reset()
    if checkpoint:
        env.clear_grad(checkpoint)
    r_agg = torch.zeros(
        (n, env.num_envs),
        dtype=torch.float32,
        device=env.device,
    )
    for i in range(n):
        o, r, d, info = env.step(actions[i])
        r_agg[i, :] = r
    loss = -r_agg.sum(dim=0)  # first sum over timesteps, preserves env dim
    return loss, info


def to_np(tensor):
    return tensor.detach().cpu().numpy()


def run_opt(actions, lr, num_opt_steps):

    opt_hist = defaultdict(list)

    with trange(num_opt_steps) as t:
        for i in t:
            loss, info = run_env(actions)
            loss.sum(dim=0).backward()  # negative mean reward
            assert actions.grad is not None
            assert not torch.isclose(actions.grad, torch.zeros_like(actions)).all()
            grad_np = to_np(actions.grad)  # num_steps x num_envs x num_acts
            grad_np_norm = np.linalg.norm(grad_np, axis=2).mean()
            loss_np = loss.detach().cpu().numpy()
            best_loss = loss_np.min()
            # index of env with best actions
            best_ac_idx = loss_np.argmin()
            t.set_description(
                "best-loss=%.3f, grad norm=%.3f" % (best_loss, grad_np_norm)
            )
            with torch.no_grad():
                opt_hist["actions_grad"].append(grad_np[best_ac_idx])
                opt_hist["loss"].append(best_loss)
                # add additional info to log from environment if desired
                opt_hist["q_err"].append(to_np(info["q_err"]))

                # update actions to minimize loss (maximize reward)
                # clamp action gradients to [-1, 1]
                clipped_grad = actions.grad.clamp_(-1, 1)
                actions.sub_(lr * clipped_grad)
                actions.grad.zero_()
                opt_hist["actions"].append(to_np(actions)[:, best_ac_idx])

    return opt_hist, loss, actions



@hydra.main(config_path="cfg", config_name="tune_grad.yaml")
def main(cfg: DictConfig):
    device = "cuda"  # defaults to cuda
    if cfg.debug_cuda:
        wp.config.mode = "debug"
        # wp.config.print_launches = True
        wp.config.verify_cuda = True
        # wp.config.verify_fp = True
        device = "cuda"
    elif cfg.debug_cpu:
        wp.config.mode = "debug"
        # wp.config.print_launches = True
        device = "cpu"

    if cfg.task.env_name == "ClawWarpEnv":
        env_config = cfg.task
        logdir = HydraConfig.get()["runtime"]["output_dir"]
        logdir = os.path.join(logdir, cfg.general.logdir)

        env = instantiate(env_config, logdir=logdir)
    elif cfg.env_name == "AllegroWarpEnv":
        env_kw = AllegroWarpConfig(
            no_grad=False,
            goal_type=GoalType.ORIENTATION,
            goal_path="goal_traj-sliding-2.npy",
            rew_kw=RewardConfig(
                c_act=0.1, c_finger=0.2, c_q=10.0, c_pos=0.0, c_ft=10.0
            ),
            # rew_kw=RewardConfig(
            #     c_act=0, c_finger=100.0, c_q=10.0, c_pos=0.0, c_ft=10.0
            # ),
            num_envs=cfg.num_envs,
            debug=cfg.debug,
            device=device,
            action_type=ActionType.POSITION,
        )
        env = instantiate(env_config, logdir=logdir)
    if cfg.debug_cuda or cfg.debug_cpu:
        env.capture_graph = False



    num_steps = cfg.num_steps
    num_opt_steps = cfg.num_opt_steps
    if cfg.debug_actions:
        np_actions = np.load(cfg.debug_actions)
        if cfg.debug_actions.endswith(".npz"):
            np_actions = np_actions["actions"].reshape(-1, cfg.num_envs, 4)[
                :num_steps
            ]
        assert (
            np_actions.shape[0] == num_steps
        ), f"Expected debug_actions num_steps={num_steps}, got {len(np_actions)}"
        if np_actions.shape[1] != env.num_envs:
            np_actions = np_actions.repeat(env.num_envs, axis=1)
        ac = torch.tensor(np_actions, device=env.device, requires_grad=True)
    else:
        np_actions = (
            np.random.rand(num_steps, env.num_envs, env.num_actions).astype(np.float32)
            - 0.5
        )
        # np_actions = np.ones(
        #     (num_steps, env.num_envs, env.num_actions), dtype=np.float32
        # )
        ac = torch.tensor(np_actions, requires_grad=True, device=env.device)
    if cfg.gradcheck:
        # , atol=1e-3, rtol=1e-5)
        torch.autograd.gradcheck(run_env, (env, ac,), eps=1e-3, atol=cfg.atol, rtol=cfg.rtol)  
        return

    lr = cfg.lr or 5e-2
    opt_hist, loss_final, ac_final = run_opt(ac, lr, num_opt_steps)

    if cfg.render:
        env.visualize = True
        print("saving rendered viz to outputs/compute_grad_test.usd")
        env.initialize_renderer("outputs/compute_grad_test.usd")
        env.reset()
        for i in range(num_steps):
            env.step(ac_final[i])

    if cfg.debug:
        print("saving debug opt-hist to debug_compute_grad.npz")
        np.savez(
            "debug_compute_grad.npz", **{k: np.array(opt_hist[k]) for k in opt_hist}
        )

    if cfg.save_act:
        print("saving actions to debug_action.npy")
        env.reset()
        np.save("debug_action.npy", ac.detach().cpu().numpy())

    if cfg.save_fig:
        print("saving grad/ac update animations to ac-{grad|ref}.gif")
        acs = np.array(opt_hist["actions"]).reshape(num_opt_steps, -1)
        grads = np.array(opt_hist["actions_grad"]).reshape(num_opt_steps, -1)

        create_animation(
            acs, None, "./", "ac-ref.gif", title="ac-ref", ylim=(-1.0, 1.0)  # acs[0],
        )
        create_animation(
            grads,
            None,  # grads[0],
            "./",
            "ac-grad.gif",
            title="ac-grad",
        )

if __name__ == "__main__":
    main()
