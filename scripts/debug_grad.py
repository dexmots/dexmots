import hydra
import torch
import yaml
import os
import numpy as np
import copy

from omegaconf import OmegaConf, DictConfig
from hydra.utils import instantiate
from dmanip.utils.run_utils import RLGPUEnvAlgoObserver
from dmanip.utils import torch_utils as tu
from rl_games.torch_runner import Runner, _restore, _override_sigma
from dmanip.envs.environment import RenderMode
from dmanip.utils.common import run_env, HandType, collect_rollout, to_numpy
from dmanip.utils.run_utils import get_time_stamp
from train import register_envs
from dmanip.envs.wrappers import Monitor

# register custom resolvers
import dmanip.utils.hydra_resolvers


def get_policy(cfg, env):
    if cfg.alg is None or cfg.alg.name == "default":
        return None
    num_act = 16 if cfg.task.env.hand_type == HandType.ALLEGRO else 24
    if "_target" in cfg.alg:
        # Run with hydra
        cfg.task.env.no_grad = cfg.debug_grad
        traj_optimizer = instantiate(cfg.alg, env_config=cfg.task.env, logdir=cfg.logdir)

        if cfg.general.checkpoint:
            traj_optimizer.load(cfg.general.checkpoint)
        return lambda obs, t: traj_optimizer.actor(obs, deterministic=cfg.alg.stochastic_edval)
    elif cfg.alg.name == "zero":
        return lambda x, t: torch.zeros((x.shape[0], num_act), device=x.device).requires_grad_(cfg.debug_grad)
    elif cfg.alg.name == "random":
        return (
            lambda x, t: (2 * torch.rand((x.shape[0], num_act), device=x.device) - 1)
            .clamp_(-1.0, 1.0)
            .requires_grad_(cfg.debug_grad)
        )
    elif cfg.alg.name == "sine":
        period = cfg.task.env.episode_length // 2
        return lambda x, t: torch.sin(
            torch.ones((x.shape[0], num_act)).to(x) * t * 2 * np.pi * 1 / period
        ).requires_grad_(cfg.debug_grad)
    elif cfg.alg.name in ["ppo", "sac"]:
        agent_args = OmegaConf.to_container(cfg.general, resolve=True)
        agent_args.update(
            {
                "value_size": 1,
                "observation_space": env.observation_space,
                "action_space": env.action_space,
                "vec_env": env,
            }
        )
        runner = init_runner_and_cfg(cfg)

        agent = runner.algo_factory.create(runner.algo_name, base_name="run", params=runner.params)
        _restore(agent, agent_args)
        _override_sigma(agent, agent_args)
        deterministic = cfg.alg.params.config.player.get("deterministic", False)

        def get_action(x, t):
            res_dict = agent.get_action_values({"obs": x})
            if deterministic:
                return res_dict["mus"].requires_grad_(cfg.debug_grad)
            return res_dict["actions"]

        return get_action


def init_runner_and_cfg(cfg):
    cfg_full = OmegaConf.to_container(cfg, resolve=True)
    cfg_eval = cfg_full["alg"]
    cfg_eval["params"]["general"] = cfg_full["general"]
    cfg_eval["params"]["seed"] = cfg_full["general"]["seed"]
    cfg_eval["params"]["render"] = cfg_full["render"]
    cfg_eval["params"]["diff_env"] = cfg_full["task"]["env"]
    cfg_eval["params"]["config"].update({"minibatch_size": cfg.num_envs * cfg.alg.params.config.horizon_length})
    env_name = cfg_full["task"]["name"]

    if env_name.split("_")[0] == "df":
        cfg_eval["params"]["config"]["env_name"] = env_type = "dflex"
    elif env_name.split("_")[0] == "warp":
        cfg_eval["params"]["config"]["env_name"] = env_type = "warp"

    if cfg.wrapper is not None:
        register_envs(cfg.wrapper, env_type)
    else:
        register_envs(cfg.task.env, env_type)
    # add observer to score keys
    if cfg_eval["params"]["config"].get("score_keys"):
        algo_observer = RLGPUEnvAlgoObserver()
    else:
        algo_observer = None
    runner = Runner(algo_observer)
    runner.load(cfg_eval)
    return runner


@hydra.main(config_path="cfg", config_name="debug_grad.yaml")
def run(cfg: DictConfig):
    # override cfg.task.env.use_graph_capture to false
    cfg.task.env.use_graph_capture = False
    if cfg.wrapper:
        cfg.wrapper.env.use_graph_capture = False
    cfg_full = OmegaConf.to_container(cfg, resolve=True)
    cfg_yaml = yaml.dump(cfg_full)
    print("Run Params:")
    print(cfg_yaml)

    traj_optimizer = None
    # instantiate the environment if not instantiated by policy class
    # TODO: fix env instantiation so multiple wrappers can be used
    env = instantiate(
        cfg.task.env, _convert_="partial", logdir=cfg.logdir, num_envs=1, stochastic_init=False, no_grad=False
    )
    env.opengl_render_settings["headless"] = cfg.headless
    if cfg.headless:
        env = Monitor(env, "outputs/videos/{}".format(get_time_stamp()))

    eval_env = instantiate(
        cfg.task.env, _convert_="partial", logdir=cfg.logdir, stochastic_init=False, goal_rot_seed=120, no_grad=False
    )

    policy = get_policy(cfg, env)
    eval_env.reset()
    eval_checkpoint = eval_env.get_checkpoint()
    checkpoint = env.get_checkpoint()
    actions, _, _, _, _ = collect_rollout(env, cfg_full["num_steps"], policy, checkpoint=checkpoint)
    actions = [to_numpy(ac) for ac in actions]
    scripted_actions = torch.tensor(actions, requires_grad=cfg.debug_grad, device=env.device)
    scripted_policy = lambda x, t: scripted_actions[t]
    _, _, _, _, losses = collect_rollout(env, cfg_full["num_steps"], scripted_policy, checkpoint=checkpoint)
    loss_total = torch.cat(losses).sum()
    print("Total Loss:", loss_total)
    print("start backward")

    loss_total.backward()
    grad_norm = tu.grad_norm([scripted_actions])
    max_grad = torch.abs(scripted_actions.grad).max()
    print("loss = ", loss_total, ", grad norm = ", grad_norm, ", max grad = ", max_grad)

    step_id = 0
    action_id = 1

    grad_0 = scripted_actions.grad[step_id][0][action_id]
    grad = scripted_actions.grad.clone()

    def evaluate(eval_actions):
        scripted_policy = lambda x, t: eval_actions[t]
        eval_env = instantiate(
            cfg.task.env, _convert_="partial", logdir=cfg.logdir, stochastic_init=False, goal_rot_seed=120, no_grad=True
        )
        _, _, _, _, losses = collect_rollout(
            eval_env, cfg_full["num_steps"], scripted_policy, checkpoint=eval_checkpoint
        )
        return torch.stack(losses, dim=0).sum(dim=0)

    print("param = ", scripted_actions[step_id, 0, action_id])
    xs = np.arange(-1e-1, 1e-1, 1e-4, dtype=np.float32)
    # xs = np.arange(-1e-1, 1e-1, 1e-3, dtype=np.float32)
    # xs = np.arange(-1., 1., 1e-4, dtype = np.float32)
    # xs = np.arange(-1., 1., 1e-3, dtype = np.float32)
    # xs = np.arange(-1e-5, 1e-5, 1e-6, dtype = np.float32)
    losses = np.zeros(len(xs), dtype=np.float32)

    with torch.no_grad():
        i = 0
        while i < len(xs):
            print("{} / {}".format(i, len(xs)))
            j = min(len(xs), i + cfg.num_envs)
            eval_actions = torch.zeros((cfg.num_steps, cfg.num_envs, 16), dtype=torch.float32, device=env.device)
            for idx in range(i, j):
                eval_actions[:, idx - i, :] = copy.deepcopy(scripted_actions.squeeze(1))
                eval_actions[step_id, idx - i, action_id] += xs[idx]
            losses_seg = evaluate(eval_actions)
            losses[i:j] = losses_seg[: j - i].detach().cpu().numpy()
            i = j

    print(
        "f = ", loss_total.item(), ", grad = ", grad_0.item(), "fd grad = ", (losses[-1] - losses[0]) / (xs[-1] - xs[0])
    )
    print("max - min = ", np.max(losses) - np.min(losses))

    # actions, states, rewards, _ = collect_rollout(
    #     env, cfg_full["num_steps"], scripted_policy, checkpoint=checkpoint
    # )


if __name__ == "__main__":
    run()
