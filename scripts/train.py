import traceback
import hydra, os, wandb, yaml, torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from dmanip.utils.common import *
from dmanip.utils.run_utils import RLGPUEnvAlgoObserver, RLGPUEnv
from dmanip import envs
from gym import wrappers
from rl_games.torch_runner import Runner
from rl_games.common import env_configurations, vecenv
from dmanip.utils import hydra_resolvers


def register_envs(env_config, env_type="warp"):
    def create_dflex_env(**kwargs):
        # create env without grads since PPO doesn't need them
        if isinstance(env_config, dict):
            env_config.pop("name")
        else:
            env = instantiate(env_config)
        print("num_envs = ", env.num_envs)
        print("num_actions = ", env.num_actions)
        print("num_obs = ", env.num_obs)

        frames = kwargs.pop("frames", 1)
        if frames > 1:
            env = wrappers.FrameStack(env, frames, False)
        return env

    def create_warp_env(**kwargs):
        # create env without grads since PPO doesn't need them
        if isinstance(env_config, dict):
            env_config.pop("name")
        else:
            env = instantiate(env_config)

        print("num_envs = ", env.num_envs)
        print("num_actions = ", env.num_actions)
        print("num_obs = ", env.num_obs)

        frames = kwargs.pop("frames", 1)
        if frames > 1:
            env = wrappers.FrameStack(env, frames, False)

        return env

    if env_type == "dflex":
        vecenv.register(
            "DFLEX",
            lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs),
        )
        env_configurations.register(
            "dflex",
            {
                "env_creator": lambda **kwargs: create_dflex_env(**kwargs),
                "vecenv_type": "DFLEX",
            },
        )
    if env_type == "warp":
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


def create_wandb_run(wandb_cfg, job_config, run_id=None):
    """
    Creates and initializes a run in Weights & Biases (wandb).

    Args:
        wandb_cfg (dict): Configuration for wandb.
        job_config (dict): Configuration for the job.
        run_id (str, optional): ID of the run to resume. Defaults to None.

    Returns:
        WandbRun: The initialized wandb run.
    """
    # Get environment name from job_config
    env_name = job_config["task"]["env"]["_target_"].split(".")[-1]

    try:
        # Get algorithm name from job_config
        alg_name = job_config["alg"]["_target_"].split(".")[-1]
    except:
        # Use default algorithm name if not found in job_config
        alg_name = "PPO"

    try:
        # Multirun config
        job_id = HydraConfig().get().job.num
        name = f"{alg_name}_{env_name}_sweep_{job_id}"
        notes = wandb_cfg.get("notes", None)
    except:
        # Normal (singular) run config
        name = f"{alg_name}_{env_name}"
        notes = wandb_cfg["notes"]  # force user to make notes

    return wandb.init(
        project=wandb_cfg.project,
        config=job_config,
        group=wandb_cfg.group,
        entity=wandb_cfg.entity,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
        name=name,
        notes=notes,
        id=run_id,
        resume=run_id is not None,
    )


cfg_path = os.path.dirname(__file__)
cfg_path = os.path.join(cfg_path, "cfg")


@hydra.main(config_path="cfg", config_name="train.yaml")
def train(cfg: DictConfig):
    if cfg.debug:
        import warp as wp

        wp.config.mode = "debug"
        wp.config.verify_cuda = True
        wp.config.print_launches = True

    torch.set_default_dtype(torch.float32)

    cfg_full = OmegaConf.to_container(cfg, resolve=True)

    if cfg.general.run_wandb:
        run = create_wandb_run(cfg.wandb, cfg_full)
    else:
        run = None

    cfg_yaml = yaml.dump(cfg_full["alg"])
    resume_model = cfg.resume_model
    if os.path.exists("exp_config.yaml"):
        loaded_config = yaml.load(open("exp_config.yaml", "r"))
        params, wandb_id = loaded_config["params"], loaded_config["wandb_id"]
        resume_model = "restore_checkpoint.zip"
        assert os.path.exists(resume_model), "restore_checkpoint.zip does not exist!"
    else:
        defaults = HydraConfig.get().runtime.choices

        params = yaml.safe_load(cfg_yaml)
        params["defaults"] = {k: defaults[k] for k in ["alg"]}

        wandb_id = run.id if run != None else None
        save_dict = dict(wandb_id=wandb_id, params=params)
        yaml.dump(save_dict, open("exp_config.yaml", "w"))
        print("Alg Config:")
        print(cfg_yaml)
        print("Task Config:")
        print(yaml.dump(cfg_full["task"]))

    logdir = HydraConfig.get()["runtime"]["output_dir"]
    if cfg.general.logdir:
        cfg.general.logdir = os.path.join(logdir, cfg.general.logdir)

    if "_target_" in cfg.alg:
        # Run with hydra
        cfg.task.env.no_grad = not cfg.general.train

        traj_optimizer = instantiate(cfg.alg, env_config=cfg.task.env)

        if cfg.general.checkpoint:
            traj_optimizer.load(cfg.general.checkpoint)

        if cfg.general.train:
            traj_optimizer.train()
        else:
            traj_optimizer.run(cfg.task.player.games_num)

    elif cfg.alg.name in ["ppo", "sac"]:
        cfg_train = cfg_full["alg"]
        cfg_train["params"]["config"][
            "num_actors"
        ] = (
            cfg.task.env.num_envs
        )  # This is the only number that works, at least for humanoid. Maybe because task yaml specifies 64 envs?
        cfg_train["params"]["general"] = cfg_full["general"]
        cfg_train["params"]["seed"] = cfg_full["general"]["seed"]
        cfg_train["params"]["render"] = cfg_full["render"]
        env_name = cfg_train["params"]["config"]["env_name"]

        if env_name.split("_")[0] == "df":
            cfg_train["params"]["config"]["env_name"] = env_type = "dflex"
        elif env_name.split("_")[0] == "warp":
            cfg_train["params"]["config"]["env_name"] = env_type = "warp"

        cfg_train["params"]["diff_env"] = cfg_full["task"]["env"]
        # cfg_train["params"]["diff_env"]["use_graph_capture"] = True
        cfg_train["params"]["diff_env"]["name"] = env_name

        # save config
        if cfg_train["params"]["general"]["train"]:
            cfg_train["params"]["general"]["logdir"] = logdir
            os.makedirs(logdir, exist_ok=True)
            # save config
            yaml.dump(cfg_train, open(os.path.join(logdir, "cfg.yaml"), "w"))
        # register envs

        cfg.task.env.no_grad = True
        register_envs(cfg.task.env, env_type)

        # add observer to score keys
        if cfg_train["params"]["config"].get("score_keys"):
            algo_observer = RLGPUEnvAlgoObserver()
        else:
            algo_observer = None
        runner = Runner(algo_observer)
        runner.load(cfg_train)
        runner.reset()
        runner.run(cfg_train["params"]["general"])
        # if running first with train=True with render=True
        if cfg.general.train and cfg.general.play:
            cfg_train["params"]["render"] = True
            runner.load_config(cfg_train)
            runner.run_play(cfg_train["params"]["general"])

    if cfg.general.run_wandb:
        wandb.finish()


if __name__ == "__main__":
    train()
