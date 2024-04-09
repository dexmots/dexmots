import hydra
import torch
import yaml
import os
import numpy as np

from omegaconf import OmegaConf, DictConfig
from hydra.utils import instantiate
from dmanip.utils.run_utils import RLGPUEnvAlgoObserver
from rl_games.torch_runner import Runner, _restore, _override_sigma
from dmanip.envs.environment import RenderMode
from dmanip.utils.common import run_env, HandType
from dmanip.utils.run_utils import get_time_stamp
from train import register_envs
from dmanip.envs.wrappers import Monitor

# register custom resolvers
import dmanip.utils.hydra_resolvers


def get_policy(cfg, env):
    if cfg.alg is None or cfg.alg.name == "default":
        return None
    num_act = 16 if cfg.task.env.hand_type == HandType.ALLEGRO else 24
    if cfg.alg.name == "zero":
        return lambda x, t: torch.zeros((x.shape[0], num_act), device=x.device)
    if cfg.alg.name == "random":
        return lambda x, t: (2 * torch.rand((x.shape[0], num_act), device=x.device) - 1).clamp_(-1.0, 1.0)
    if cfg.alg.name == "sine":
        period = cfg.task.env.episode_length // 2
        return lambda x, t: torch.sin(torch.ones((x.shape[0], num_act)).to(x) * t * 2 * np.pi * 1 / period)
    if cfg.alg.name in ["ppo", "sac"]:
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
        deterministic = cfg.alg.params.config.get("deterministic", False)

        def get_action(x, t):
            res_dict = agent.get_action_values(x)
            return res_dict["actions"]

        return get_action


def init_runner_and_cfg(cfg):
    if "no_grad" in cfg.task.env:
        cfg.task.env.no_grad = True
    cfg_full = OmegaConf.to_container(cfg, resolve=True)
    cfg_eval = cfg_full["alg"]
    cfg_eval["params"]["general"] = cfg_full["general"]
    cfg_eval["params"]["seed"] = cfg_full["general"]["seed"]
    cfg_eval["params"]["render"] = cfg_full["render"]
    cfg_eval["params"]["diff_env"] = cfg_full["task"]["env"]
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


@hydra.main(config_path="cfg", config_name="run_task.yaml")
def run(cfg: DictConfig):
    cfg_full = OmegaConf.to_container(cfg, resolve=True)
    cfg_yaml = yaml.dump(cfg_full)
    print("Run Params:")
    print(cfg_yaml)

    traj_optimizer = None
    # get a policy
    if "_target_" in cfg.alg:
        # Run with hydra
        # check if omegaconf object cfg.task.env has no_grad attribute and if so set it to true
        if "no_grad" in cfg.task.env:
            cfg.task.env.no_grad = True

        traj_optimizer = instantiate(cfg.alg, env_config=cfg.task.env)

        if cfg.general.checkpoint:
            traj_optimizer.load(cfg.general.checkpoint)

        traj_optimizer.run(cfg.num_rollouts)

    elif cfg.alg.name in ["default", "random", "zero", "sine"] or cfg.debug_grad:
        if "no_grad" in cfg.task.env:
            cfg.task.env.no_grad = cfg.debug_grad

        # instantiate the environment
        env = instantiate(cfg.task.env, _convert_="partial")

        env.opengl_render_settings["headless"] = cfg.headless
        if cfg.headless:
            env = Monitor(env, "outputs/videos/{}".format(get_time_stamp()))

        policy = get_policy(cfg, env)
        run_env(env, policy, cfg_full["num_steps"], cfg_full["num_rollouts"], use_grad=cfg.debug_grad)

    elif cfg.alg.name in ["ppo", "sac"]:
        runner = init_runner_and_cfg(cfg)
        runner.reset()
        traj_optimizer = runner.run(cfg_full["general"])
        if cfg.wrapper is not None:
            traj_optimizer.env.close()


if __name__ == "__main__":
    run()
