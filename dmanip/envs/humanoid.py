# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import math
import os
import sys

import torch

from .environment import Environment, RenderMode, IntegratorType
from .warp_env import WarpEnv
import numpy as np
import warp as wp

np.set_printoptions(precision=5, linewidth=256, suppress=True)

try:
    from pxr import Usd
except ModuleNotFoundError:
    print("No pxr package")

from ..utils import torch_utils as tu


class HumanoidEnv(WarpEnv):
    render_mode: RenderMode = RenderMode.OPENGL
    integrator_type: IntegratorType = IntegratorType.XPBD

    def __init__(
        self,
        render=False,
        device="cuda:0",
        num_envs=10,
        seed=0,
        episode_length=1000,
        no_grad=True,
        stochastic_init=False,
        stage_path=None,
        env_name="HumanoidEnv",
    ):
        num_obs = 76
        num_act = 21

        super().__init__(
            num_envs,
            num_obs,
            num_act,
            episode_length,
            seed,
            no_grad,
            render,
            stochastic_init,
            device,
            env_name=env_name,
            render_mode=self.render_mode,
            stage_path=stage_path,
        )
        self.num_joint_q, self.num_joint_qd = 28, 27
        self.init_sim()

        # initialize some data used later on
        # todo - switch to z-up
        self.start_pos = self.start_joint_q[:, :3]
        self.start_rotation = self.start_joint_q[:, 3:7]
        self.x_unit_tensor = tu.to_torch(
            [1, 0, 0], dtype=torch.float, device=str(self.device), requires_grad=False
        ).repeat((self.num_envs, 1))
        self.y_unit_tensor = tu.to_torch(
            [0, 1, 0], dtype=torch.float, device=str(self.device), requires_grad=False
        ).repeat((self.num_envs, 1))
        self.z_unit_tensor = tu.to_torch(
            [0, 0, 1], dtype=torch.float, device=str(self.device), requires_grad=False
        ).repeat((self.num_envs, 1))
        self.up_vec = self.y_unit_tensor.clone()
        self.heading_vec = self.x_unit_tensor.clone()
        self.inv_start_rot = tu.quat_conjugate(self.start_rotation)

        self.basis_vec0 = self.heading_vec.clone()
        self.basis_vec1 = self.up_vec.clone()
        self.targets = tu.to_torch([200.0, 0.0, 0.0], device=self.device, requires_grad=False).repeat(
            (self.num_envs, 1)
        )

        # other parameters
        self.termination_height = 0.74
        self.motor_strengths = [
            200,
            200,
            200,
            200,
            200,
            600,
            400,
            100,
            100,
            200,
            200,
            600,
            400,
            100,
            100,
            100,
            100,
            200,
            100,
            100,
            200,
        ]

        self.motor_scale = 0.35

        self.motor_strengths = tu.to_torch(
            self.motor_strengths,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        ).repeat((self.num_envs, 1))

        self.action_penalty = -0.002
        self.joint_vel_obs_scaling = 0.1
        self.termination_tolerance = 0.1
        self.height_rew_scale = 10.0

    def create_articulation(self, builder):
        examples_dir = os.path.split(os.path.dirname(wp.__file__))[0] + "/examples"
        wp.sim.parse_mjcf(
            os.path.join(examples_dir, "assets/nv_humanoid.xml"),
            builder,
            stiffness=0.0,
            damping=0.1,
            armature=0.007,
            armature_scale=10.0,
            contact_ke=1.0e4,
            contact_kd=1.0e2,
            contact_kf=1.0e2,
            contact_mu=0.5,
            contact_restitution=0.0,
            limit_ke=1.0e2,
            limit_kd=1.0e1,
            enable_self_collisions=True,
        )

        builder.joint_q[:7] = [
            0.0,
            1.7,
            0.0,
            *wp.quat_from_axis_angle((1.0, 0.0, 0.0), -math.pi * 0.5),
        ]

    def update_joints(self):
        self._joint_q.zero_()
        self._joint_qd.zero_()
        if self._joint_q.grad is not None:
            self._joint_q.grad.zero_()
            self._joint_qd.grad.zero_()

        if self.update_joints_graph is None and self.use_graph_capture:
            wp.capture_begin()
            wp.sim.eval_ik(self.model, self.state_0, self._joint_q, self._joint_qd)
            # self.compute_joint_q_qd()
            self.update_joints_graph = wp.capture_end()

        if self.use_graph_capture:
            wp.capture_launch(self.update_joints_graph)
        else:
            wp.sim.eval_ik(self.model, self.state_0, self._joint_q, self._joint_qd)
            # self.compute_joint_q_qd()

        self.joint_q = self.to_torch(self._joint_q).clone()
        self.joint_qd = self.to_torch(self._joint_qd)

    def to_torch(self, state_arr, num_frames=None):
        return wp.to_torch(state_arr)

    def assign_actions(self, actions):
        actions = actions.view((self.num_envs, self.num_actions))
        actions = torch.clip(actions, -1.0, 1.0)
        acts_per_env = int(self.model.joint_act.shape[0] / self.num_envs)
        joint_act = torch.zeros((self.num_envs * acts_per_env), dtype=torch.float32, device=self.device)
        act_types = {1: [True], 3: [], 4: [False] * 6, 5: [True] * 3, 6: [True] * 2}
        joint_types = self.model.joint_type.numpy()
        act_idx = np.concatenate([act_types[i] for i in joint_types])
        joint_act[act_idx] = actions.flatten()
        self.model.joint_act.assign(joint_act.detach().cpu().numpy())
        self.actions = actions.clone()

    def step(self, actions):
        self.assign_actions(actions)

        with wp.ScopedTimer("simulate", active=False, detailed=False):
            self.update()  # iterates num_frames
            self.sim_time += self.sim_dt
        self.sim_time += self.sim_dt

        self.reset_buf = torch.zeros_like(self.reset_buf)

        self.progress_buf += 1
        self.num_frames += 1

        self.calculateObservations()
        if not self.requires_grad:
            self.calculateReward()

        self.reset_buf = self.reset_buf | (self.progress_buf >= self.episode_length)
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        if self.requires_grad:
            self.obs_buf_before_reset = self.obs_buf.clone()
            self.extras = {
                "obs_before_reset": self.obs_buf_before_reset,
                "episode_end": self.termination_buf,
            }

        if len(env_ids) > 0:
            self.reset(env_ids)

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def reset(self, env_ids=None, force_reset=True):
        super().reset(env_ids, force_reset)  # resets state_0 and joint_q
        self.update_joints()
        self.initialize_trajectory()  # sets goal_trajectory & obs_buf
        if self.requires_grad:
            self.model.allocate_rigid_contacts(requires_grad=self.requires_grad)
            wp.sim.collide(self.model, self.state_0)
        return self.obs_buf

    def get_stochastic_init(self, env_ids, joint_q, joint_qd):
        joint_q[env_ids, 0:3] += 0.1 * (torch.rand(size=(len(env_ids), 3), device=self.device) - 0.5) * 2.0
        angle = (torch.rand(len(env_ids), device=self.device) - 0.5) * np.pi / 12.0
        axis = torch.nn.functional.normalize(torch.rand((len(env_ids), 3), device=self.device) - 0.5)
        joint_q[env_ids, 3:7] = tu.quat_mul(joint_q[env_ids, 3:7], tu.quat_from_angle_axis(angle, axis))
        joint_q[env_ids, 7:] = (
            self.joint_q.view(self.num_envs, -1)[env_ids, 7:]
            + 0.2 * (torch.rand(size=(len(env_ids), self.num_joint_q - 7), device=self.device) - 0.5) * 2.0
        )
        joint_qd[env_ids, :] = 0.5 * (torch.rand(size=(len(env_ids), self.num_joint_qd), device=self.device) - 0.5)
        return joint_q, joint_qd

    def calculateObservations(self):
        torso_pos = self.joint_q.view(self.num_envs, -1)[:, 0:3]
        torso_rot = self.joint_q.view(self.num_envs, -1)[:, 3:7]
        lin_vel = self.joint_qd.view(self.num_envs, -1)[:, 3:6]
        ang_vel = self.joint_qd.view(self.num_envs, -1)[:, 0:3]

        # convert the linear velocity of the torso from twist representation to the velocity of the center of mass in world frame
        lin_vel = lin_vel - torch.cross(torso_pos, ang_vel, dim=-1)

        to_target = self.targets + self.start_pos - torso_pos
        to_target[:, 1] = 0.0

        target_dirs = tu.normalize(to_target)
        torso_quat = tu.quat_mul(torso_rot, self.inv_start_rot)

        up_vec = tu.quat_rotate(torso_quat, self.basis_vec1)
        heading_vec = tu.quat_rotate(torso_quat, self.basis_vec0)

        self.obs_buf = torch.cat(
            [
                torso_pos[:, 1:2],  # 0
                torso_rot,  # 1:5
                lin_vel,  # 5:8
                ang_vel,  # 8:11
                self.joint_q.view(self.num_envs, -1)[:, 7:],  # 11:32
                self.joint_vel_obs_scaling * self.joint_qd.view(self.num_envs, -1)[:, 6:],  # 32:53
                up_vec[:, 1:2],  # 53:54
                (heading_vec * target_dirs).sum(dim=-1).unsqueeze(-1),  # 54:55
                self.actions.clone(),
            ],  # 55:76
            dim=-1,
        )

    def calculateReward(self):
        up_reward = 0.1 * self.obs_buf[:, 53]
        heading_reward = self.obs_buf[:, 54]

        height_diff = self.obs_buf[:, 0] - (self.termination_height + self.termination_tolerance)
        height_reward = torch.clip(height_diff, -1.0, self.termination_tolerance)
        height_reward = torch.where(height_reward < 0.0, -200.0 * height_reward * height_reward, height_reward)
        height_reward = torch.where(height_reward > 0.0, self.height_rew_scale * height_reward, height_reward)

        progress_reward = self.obs_buf[:, 5]

        self.rew_buf = (
            progress_reward
            + up_reward
            + heading_reward
            + height_reward
            + torch.sum(self.actions**2, dim=-1) * self.action_penalty
        )

        # reset agents
        # self.reset_buf = torch.where(
        #     self.obs_buf[:, 0] < self.termination_height,
        #     torch.ones_like(self.reset_buf),
        #     self.reset_buf,
        # )
        self.reset_buf = torch.where(
            self.progress_buf > self.episode_length - 1,
            torch.ones_like(self.reset_buf),
            self.reset_buf,
        )

        # an ugly fix for simulation nan values
        nan_masks = torch.logical_or(
            torch.isnan(self.obs_buf).sum(-1) > 0,
            torch.logical_or(
                torch.isnan(self.joint_q.view(self.num_environments, -1)).sum(-1) > 0,
                torch.isnan(self.joint_qd.view(self.num_environments, -1)).sum(-1) > 0,
            ),
        )
        inf_masks = torch.logical_or(
            torch.isinf(self.obs_buf).sum(-1) > 0,
            torch.logical_or(
                torch.isinf(self.joint_q.view(self.num_environments, -1)).sum(-1) > 0,
                torch.isinf(self.joint_qd.view(self.num_environments, -1)).sum(-1) > 0,
            ),
        )
        invalid_value_masks = torch.logical_or(
            (torch.abs(self.joint_q.view(self.num_environments, -1)) > 1e6).sum(-1) > 0,
            (torch.abs(self.joint_qd.view(self.num_environments, -1)) > 1e6).sum(-1) > 0,
        )
        invalid_masks = torch.logical_or(invalid_value_masks, torch.logical_or(nan_masks, inf_masks))

        self.reset_buf = torch.where(invalid_masks, torch.ones_like(self.reset_buf), self.reset_buf)

        self.rew_buf[invalid_masks] = 0.0
