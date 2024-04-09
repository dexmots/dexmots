import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
import yaml

from dmanip.envs import HandObjectTask, ObjectTask, ReposeTask
from dmanip.envs.environment import RenderMode
from dmanip.utils import builder as bu
from dmanip.utils import warp_utils as wpu
from dmanip.utils.common import collect_rollout, run_env
import dmanip.utils.hydra_resolvers
from dmanip.optim import Adam
import warp as wp


@hydra.main(config_path="cfg", config_name="tune_hand.yaml")
def run(cfg: DictConfig):
    cfg_full = OmegaConf.to_container(cfg, resolve=True)
    cfg_yaml = yaml.dump(cfg_full)
    # params = yaml.safe_load(cfg_yaml)
    print("Run Params:")
    print(cfg_yaml)

    # instantiate the environment
    if cfg.task.name.lower() == "repose_task":
        env = instantiate(cfg.task.env, no_grad=False, use_autograd=False, _convert_="partial")
    elif cfg.task.name.lower() == "hand_object_task":
        env = instantiate(cfg.task.env, no_grad=False, use_autograd=False, _convert_="partial")
    elif cfg.task.name.lower() == "object_task":
        env = instantiate(cfg.task.env, no_grad=False, use_autograd=False, _convert_="partial")
    optimize_actions(env, cfg)


def optimize_actions(env, cfg):
    lr = cfg.lr
    num_opt_steps = cfg.num_opt_steps
    num_steps = cfg.num_steps

    assert env.requires_grad
    # reward_params = {"hand_target_obs_l1_dist": (bu.l1_dist, ["hand_pos", "target_pos"], 1.0)}
    # env = HandObjectTask(
    #     num_envs=1,
    #     num_obs=0,
    #     episode_length=1000,
    #     seed=0,
    #     no_grad=False,
    #     render=True,
    #     stochastic_init=False,
    #     device="cuda",
    #     render_mode=RenderMode.OPENGL,
    #     stage_path=None,
    #     object_type=None,
    #     object_id=0,
    #     stiffness=5000.0,
    #     damping=0.5,
    #     reward_params=rew_params,
    # )

    env.reset()
    body_q_target = wp.clone(env.state_0.body_q)
    body_q_target.requires_grad = False
    joint_q_target = env.warp_actions
    joint_q_target.requires_grad = False
    loss = wp.zeros(1, dtype=wp.float32, device=env.device, requires_grad=True)

    def compute_joint_q_loss():
        wp.launch(
            kernel=wpu.l1_loss,
            inputs=[joint_q_target, env._joint_q, env.joint_target_indices],
            outputs=[loss],
            device=env.device,
            dim=env.num_acts * env.num_envs,
        )

    def compute_body_q_loss():
        wp.launch(
            kernel=wpu.l1_xform_loss,
            inputs=[body_q_target, env.state_0.body_q],
            outputs=[loss],
            device=env.device,
            dim=env.model.body_count,
        )

    joint_target_indices = env.env_joint_target_indices

    upper = env.model.joint_limit_upper.numpy().reshape(env.num_envs, -1)[0, joint_target_indices]
    lower = env.model.joint_limit_lower.numpy().reshape(env.num_envs, -1)[0, joint_target_indices]
    joint_start = env.start_joint_q.cpu().numpy()[:, joint_target_indices]
    target = joint_start + 0.1 * (upper - lower)
    action = torch.tensor(target, device=str(env.device))
    if isinstance(env, HandObjectTask):
        stiffness = env.hand_target_ke
        damping = env.hand_target_kd
    else:
        stiffness = env.model.joint_target_ke
        damping = env.model.joint_target_kd
    stiffness.requires_grad = True
    damping.requires_grad = True
    optimizer = Adam([stiffness, damping], lr=lr)

    pi = None  # lambda x, y: action

    def pi(obs, t):
        del obs
        act = np.sin(np.ones_like(upper) * t / 150) * 0.8  # [-1, 1]
        joint_q_targets = act * (upper - lower) / 2 + (upper - lower) / 2
        action = joint_start.copy()
        action[:, :] = joint_q_targets
        return torch.tensor(action, device=str(env.device))

    for i in range(num_opt_steps):
        loss.zero_()
        tape = wp.Tape()
        with tape:
            actions, states, rewards, _ = collect_rollout(env, num_steps, pi, loss_fn=compute_body_q_loss)
        tape.backward(loss=loss)
        optimizer.step([stiffness.grad, damping.grad])
        tape.zero()
        if cfg.save_log and i % cfg.save_freq == 0:
            np.savez(
                f"{env.env_name}_dof_rollout-{i}",
                actions=np.asarray(actions),
                states=np.asarray(states),
                rewards=np.asarray(rewards),
            )
    print(f"optimized stiffness/damping params for {env.object_type}:", stiffness, damping)


if __name__ == "__main__":
    run()
