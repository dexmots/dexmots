from typing import Tuple

import numpy as np
import os
import torch
import time

from ..utils import torch_utils as tu
from .environment import RenderMode
from .hand_env import HandObjectTask
from ..utils.common import ActionType, HandType, ObjectType, run_env, profile, to_numpy
from ..utils.rewards import action_penalty, l2_dist, reach_bonus, rot_dist


class ReposeTask(HandObjectTask):
    # use_tiled_rendering: bool = True
    old_obs_keys = ["hand_joint_pos", "hand_joint_vel", "object_pos", "target_pos"]
    obs_keys = [
        "hand_joint_pos",
        "hand_joint_vel",
        "object_pos",
        "target_pos",
        "object_rot",
        "target_rot",
        "object_quat_err",
    ]  # 16 + 16 + 3 + 3 + 4 + 4 + 4
    ige_obs_keys = [
        "hand_joint_pos",
        "hand_joint_vel",
        "hand_joint_torque",
        "object_pos",
        "object_rot",
        "object_lin_vel",
        "object_ang_vel",
        "target_pos",
        "target_rot",
        "object_quat_err",
        "actions",
    ]
    debug_visualization = False
    reset_position_noise: float = 0.01
    reset_dof_position_noise: float = 0.2
    xpbd_settings = dict(
        iterations=10,
        joint_linear_relaxation=1.0,
        joint_angular_relaxation=0.45,
        rigid_contact_relaxation=1.0,
        rigid_contact_con_weighting=True,
    )

    def __init__(
        self,
        num_envs,
        num_obs=50,
        episode_length=500,
        action_type: ActionType = ActionType.POSITION,
        seed=0,
        no_grad=True,
        render=True,
        stochastic_init=True,
        stochastic_goal_and_pos=False,
        device="cuda",
        render_mode=RenderMode.OPENGL,
        logdir=None,
        stiffness=1000.0,
        damping=0.1,
        reward_params=None,
        hand_type: HandType = HandType.ALLEGRO,
        hand_start_position: Tuple = (0.1, 0.3, 0.0),
        hand_start_orientation: Tuple = (-np.pi / 2, np.pi * 0.75, np.pi / 2),
        use_autograd: bool = True,
        use_graph_capture: bool = True,
        reach_threshold: float = 0.1,
        fall_dist: float = 0.24,
        goal_rot_seed: int = None,
        use_old_obs: bool = False,
        headless=False,
        rotate_z=True,
        reset_goals=False,
        rigid_contact_relaxation=1.0,
    ):
        object_type = ObjectType.REPOSE_CUBE
        object_id = 0
        if use_old_obs:  # TODO: deprecate
            self.obs_keys = self.old_obs_keys
            num_obs = 38
        if self.debug_visualization:
            self.env_offset = (0.5, 0.0, 0.5)

        self.xpbd_settings["rigid_contact_relaxation"] = rigid_contact_relaxation
        stage_path = None
        if logdir:
            stage_path = os.path.join(logdir, "ReposeCube")
        super().__init__(
            num_envs=num_envs,
            num_obs=num_obs,
            episode_length=episode_length,
            action_type=action_type,
            seed=seed,
            no_grad=no_grad,
            render=render,
            stochastic_init=stochastic_init,
            device=device,
            render_mode=render_mode,
            stage_path=stage_path,
            object_type=object_type,
            object_id=object_id,
            stiffness=stiffness,
            damping=damping,
            reward_params=reward_params,
            hand_type=hand_type,
            hand_start_position=hand_start_position,
            hand_start_orientation=hand_start_orientation,
            load_grasps=False,
            grasp_id=None,
            use_autograd=use_autograd,
            use_graph_capture=use_graph_capture,
            headless=headless,
        )
        self.reset_goals = reset_goals
        if self.reset_goals:
            self.consecutive_successes = torch.zeros_like(self.reset_buf)
        self.reward_extras["reach_threshold"] = reach_threshold
        self.reward_extras["fall_dist"] = self.fall_dist = fall_dist
        # stay in center of hand
        self.stochastic_goal_and_pos = stochastic_goal_and_pos
        self.goal_rot_seed = goal_rot_seed
        if self.goal_rot_seed:
            assert self.stochastic_init is False
            rs = np.random.RandomState(self.goal_rot_seed)
            sample_rot = torch.tensor(rs.rand(1, 3), device=self.device, dtype=torch.float)
            x, y, z = torch.eye(3, device=self.device)
            goal_rot_x = tu.quat_from_angle_axis(sample_rot[:, 0] * np.pi, x)
            if rotate_z:
                goal_rot_z = tu.quat_from_angle_axis(sample_rot[:, 1] * np.pi, z)
                self.goal_rot = self.default_goal_rot = tu.quat_mul(goal_rot_x, goal_rot_z).repeat(num_envs, 1)
            else:
                goal_rot_y = tu.quat_from_angle_axis(sample_rot[:, 1] * np.pi, y)
                self.goal_rot = self.default_goal_rot = tu.quat_mul(goal_rot_x, goal_rot_y).repeat(num_envs, 1)
            # TODO: make this a parameter
        else:
            self.goal_rot = self.default_goal_rot = (
                tu.to_torch([0.0, 0.0, 0.0, 1.0], device=self.device).view(1, 4).repeat(self.num_envs, 1)
            )
        self.goal_pos = self.default_goal_pos = (
            tu.to_torch([0.0, 0.32, 0.0], device=self.device).view(1, 3).repeat(self.num_envs, 1)
        )
        self._prev_rot_dist = None
        # self.simulate_params["ag_return_body"] = False
        # self.simulate_params["ag_return_vel"] = False

    def reset(self, env_ids=None, force_reset=True):
        self._prev_rot_dist = None
        if self.reset_goals:
            if env_ids is None:
                env_ids = np.arange(self.num_envs)
            self.consecutive_successes[env_ids] = 0
        return super().reset(env_ids=env_ids, force_reset=force_reset)

    def reset_goal_rot(self, env_ids):
        if self.goal_rot_seed:
            self.goal_rot[env_ids] = self.default_goal_rot[env_ids]
        else:
            assert self.stochastic_init, "Must be stochastic init to reset goal rot"
            sample_rot = tu.torch_rand_float(-1.0, 1.0, (len(env_ids), 2), device=self.device)
            x, _, z = torch.eye(3, device=self.device)
            goal_rot_x = tu.quat_from_angle_axis(sample_rot[:, 0], x)
            goal_rot_z = tu.quat_from_angle_axis(sample_rot[:, 1], z)
            self.goal_rot[env_ids] = tu.quat_mul(goal_rot_x, goal_rot_z)

    def get_stochastic_init(self, env_ids, joint_q, joint_qd):
        self.reset_goal_rot(env_ids)

        if self.stochastic_goal_and_pos:
            object_pos_init = self.start_joint_q.view(self.num_envs, -1)[
                env_ids, self.object_joint_start : self.object_joint_start + 2
            ]  # only x-y
            joint_q[env_ids, self.object_joint_start : self.object_joint_start + 2] += (
                torch.rand_like(object_pos_init) * self.reset_position_noise
            )
            hand_pos_init = self.start_joint_q.view(self.num_envs, -1)[env_ids, : self.object_joint_start]
            joint_q[env_ids, : self.object_joint_start] += (
                torch.rand_like(hand_pos_init) * self.reset_dof_position_noise
            )
        return joint_q[env_ids], joint_qd[env_ids]

    def get_checkpoint(self, save_path=None):
        checkpoint = super().get_checkpoint()
        checkpoint["target_pos"] = self.goal_pos
        checkpoint["target_rot"] = self.goal_rot

        if save_path is not None:
            print("saving checkpoint to", save_path)
            torch.save(checkpoint, save_path)
        return checkpoint

    def load_checkpoint(self, checkpoint_data={}, ckpt_path=None):
        super().load_checkpoint(checkpoint_data=checkpoint_data, ckpt_path=ckpt_path)
        if ckpt_path is not None:
            print("loading checkpoint from {}".format(ckpt_path))
            checkpoint_data = torch.load(ckpt_path)
        self.goal_pos = checkpoint_data["target_pos"][: self.num_envs]
        self.goal_rot = checkpoint_data["target_rot"][: self.num_envs]

    def _get_object_pose(self):
        joint_q = self.joint_q.view(self.num_envs, -1)

        pose = {}
        if self.object_model.floating:
            object_joint_pos = joint_q[:, self.object_joint_start : self.object_joint_start + 3]
            object_joint_quat = joint_q[:, self.object_joint_start + 3 : self.object_joint_start + 7]
            if self.use_tiled_rendering:
                pose["position"] = object_joint_pos  #  + tu.to_torch(self.object_model.base_pos).view(1, 3)
            else:
                pose["position"] = object_joint_pos - self.env_offsets
            start_quat = tu.to_torch(self.object_model.base_ori).view(1, 4).repeat(self.num_envs, 1)
            pose["orientation"] = tu.quat_mul(object_joint_quat, start_quat)
        elif self.object_model.base_joint == "px, py, px":
            pose["position"] = joint_q[:, self.object_joint_start : self.object_joint_start + 3]
        elif self.object_model.base_joint == "rx, ry, rx":
            pose["orientation"] = joint_q[:, self.object_joint_start : self.object_joint_start + 3]
        elif self.object_model.base_joint == "px, py, pz, rx, ry, rz":
            pose["position"] = joint_q[:, self.object_joint_start : self.object_joint_start + 3]
            pose["orientation"] = joint_q[:, self.object_joint_start + 3 : self.object_joint_start + 6]
        return pose

    def _get_object_vel(self):
        joint_qd = self.joint_qd.view(self.num_envs, -1)

        vel = {}
        if self.object_model.floating or self.object_model.base_joint == "px, py, pz, rx, ry, rz":
            object_lin_vel = joint_qd[:, self.object_joint_start : self.object_joint_start + 3]
            object_ang_vel = joint_qd[:, self.object_joint_start + 3 : self.object_joint_start + 6]
            vel["linear"] = object_lin_vel
            vel["angular"] = object_ang_vel
        elif self.object_model.base_joint == "px, py, px":
            pass  # TODO handle of position control only
        elif self.object_model.base_joint == "rx, ry, rx":
            pass  # TODO handle of rotation control only
        return vel

    def _pre_step(self):
        rot_dist = torch.zeros_like(self.rew_buf) if self._prev_rot_dist is None else self._prev_rot_dist
        self.reward_extras["prev_rot_dist"] = torch.where(
            self.progress_buf == 0,
            torch.zeros_like(rot_dist),
            rot_dist,
        )

    def _post_step(self):
        self._prev_rot_dist = self.extras["obs_dict"].get("rot_dist", None)
        if self.reset_goals and self.extras.get("is_success") is not None:
            success_ids = self.extras["is_success"].nonzero(as_tuple=False).flatten()
            if self.reset_goals and len(success_ids) > 0:
                if self.debug_visualization:
                    print(
                        "num_success:",
                        success_ids,
                        "avg ep len:",
                        self.progress_buf[success_ids].float().mean(),
                    )
                self.consecutive_successes[success_ids] += 1
                self.reset_goal_rot(success_ids)

    def _check_early_termination(self, obs_dict):
        # check if object is dropped / ejected
        # object_body_pos = obs_dict["object_pos"]
        termination = obs_dict["object_pose_err"].flatten() > self.fall_dist
        # Do not set termination=true when success, instead, reset goal pos
        # success = self._check_success(obs_dict)
        # termination = termination | success
        self.termination_buf = self.termination_buf | termination
        return termination

    def _check_success(self, obs_dict):
        return obs_dict["rot_dist"] < self.reward_extras["reach_threshold"]

    def _get_obs_dict(self):
        obs_dict = super()._get_obs_dict()
        obs_dict["target_pos"] = self.goal_pos
        obs_dict["target_rot"] = self.goal_rot
        object_pose = self._get_object_pose()
        obs_dict["object_pos"] = object_pose["position"]
        obs_dict["object_rot"] = object_pose["orientation"]
        obs_dict["object_pose_err"] = l2_dist(obs_dict["object_pos"], obs_dict["target_pos"]).view(self.num_envs, 1)
        obs_dict["object_quat_err"] = tu.quat_mul(
            obs_dict["object_rot"], tu.quat_conjugate(obs_dict["target_rot"])
        )  # 4
        obs_dict["rot_dist"] = torch.abs(rot_dist(obs_dict["object_rot"], obs_dict["target_rot"]))

        self.extras["obs_dict"] = obs_dict

        # log score keys, copy to avoid being overwritten
        self.extras["object_pose_err"] = obs_dict["object_pose_err"].view(self.num_envs).clone()
        self.extras["object_rot_err"] = (
            obs_dict["rot_dist"].view(self.num_envs).clone()
        )  # clone so correct values are logged
        self.extras["is_success"] = self._check_success(obs_dict).view(self.num_envs).clone()
        if self.reset_goals:
            self.extras["consecutive_successes"] = self.consecutive_successes.clone()
        if self.action_type is ActionType.TORQUE:
            self.extras["net_energy"] = torch.bmm(
                obs_dict["hand_joint_vel"].unsqueeze(1), self.actions.unsqueeze(2)
            ).squeeze()
        else:
            self.extras["net_energy"] = torch.zeros_like(self.rew_buf)
        return obs_dict

    def render(self, **kwargs):
        ret = super().render(**kwargs)
        if self.debug_visualization:
            if self.render_mode == RenderMode.OPENGL:
                time.sleep(self.sim_dt * 10)
            obs_dict = self.extras.get("obs_dict", {})
            # if "object_pos" in obs_dict:
            #     points = [to_numpy(obs_dict["object_pos"])]
            #     self.renderer.render_points("debug_markers", points, radius=0.15)
        return ret


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--num_episodes", "-ne", type=int, default=1)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--norender", action="store_true")
    args = parser.parse_args()

    reach_bonus = lambda x, y: torch.where(x < y, torch.ones_like(x), torch.zeros_like(x))
    reward_params = {
        "object_pos_err": (l2_dist, ("target_pos", "object_pos"), -10.0),
        # "rot_reward": (rot_reward, ("object_rot", "target_rot"), 1.0),
        "action_penalty": (action_penalty, ("action",), -0.0002),
        "reach_bonus": (reach_bonus, ("object_rot_err", "reach_threshold"), 250.0),
    }
    if args.profile or args.norender:
        render_mode = RenderMode.NONE
    else:
        render_mode = RenderMode.OPENGL
    env = ReposeTask(
        num_envs=args.num_envs,
        num_obs=38,
        episode_length=1000,
        reward_params=reward_params,
        render_mode=render_mode,
    )
    if args.profile:
        profile(env)
    else:
        # env.load_camera_params()
        run_env(env, pi=None, num_episodes=args.num_episodes, logdir="outputs/")
