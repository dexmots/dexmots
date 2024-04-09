import os

import torch
from .warp_env import WarpEnv
from .environment import RenderMode, IntegratorType

import warp as wp
import warp.sim
import numpy as np

np.set_printoptions(precision=5, linewidth=256, suppress=True)
from ..utils import torch_utils as tu
from ..utils.autograd2 import clear_arrays


class HopperEnv(WarpEnv):
    render_mode: RenderMode = RenderMode.USD
    integrator_type: IntegratorType = IntegratorType.XPBD
    tiny_render_settings = dict(scaling=30.0, mode="rgb")
    activate_ground_plane: bool = True
    use_graph_capture: bool = False
    joint_attach_ke: float = 32000.0
    joint_attach_kd: float = 100.0
    start_height = 5.0
    start_x_vel = 5.0
    start_y_vel = 1.0

    def __init__(
        self,
        render=False,
        device="cuda:0",
        num_envs=4096,
        seed=0,
        episode_length=1000,
        no_grad=True,
        stochastic_init=False,
        logdir=None,
        env_name="HopperEnv",
        graph_capture=False,
    ):
        self.use_graph_capture = graph_capture
        self.ag_return_body = False
        num_obs = 11
        num_act = 3

        self.num_joint_q = 6
        self.num_joint_qd = 6

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
            render_mode=self.render_mode,
            env_name=env_name,
            stage_path=logdir,
        )
        self.init_sim()
        self.setup_autograd_vars()
        self.prev_contact_count = np.zeros(self.num_envs, dtype=int)
        self.contact_count_changed = torch.zeros_like(self.reset_buf)
        self.contact_count = wp.clone(self.model.rigid_contact_count)

        # self.start_joint_q[:, :3] *= 0.0
        self.start_pos = self.start_joint_q[:, :3]
        self.start_pos[:, 2] += self.start_height
        self.start_vel = self.start_joint_qd[:, :3]
        self.start_vel[:, 0] += self.start_x_vel
        self.start_vel[:, 1] += self.start_y_vel
        # self.start_rotation = self.start_joint_q[:, 2:3]
        # other parameters
        self.termination_height = -0.45
        self.termination_angle = np.pi / 6.0
        self.termination_height_tolerance = 0.15
        self.termination_angle_tolerance = 0.05
        self.height_rew_scale = 1.0
        self.action_strength = 200.0
        self.action_penalty = -1e-1

        self.x_unit_tensor = tu.to_torch([1, 0, 0], dtype=torch.float, device=self.device, requires_grad=False).repeat(
            (self.num_envs, 1)
        )
        self.y_unit_tensor = tu.to_torch([0, 1, 0], dtype=torch.float, device=self.device, requires_grad=False).repeat(
            (self.num_envs, 1)
        )
        self.z_unit_tensor = tu.to_torch([0, 0, 1], dtype=torch.float, device=self.device, requires_grad=False).repeat(
            (self.num_envs, 1)
        )
        # initialize some data used later on
        # todo - switch to z-up
        self.up_vec = self.y_unit_tensor.clone()

        # initialize other data to be used later

    def create_articulation(self, builder):
        # examples_dir = os.path.split(os.path.dirname(wp.__file__))[0] + "/examples"
        asset_dir = os.path.join(os.path.dirname(__file__), "assets")
        wp.sim.parse_mjcf(
            os.path.join(asset_dir, "hopper.xml"),
            builder,
            density=1000.0,
            stiffness=0.0,
            damping=2.0,
            contact_ke=2.0e4,
            contact_kd=1.0e3,
            contact_kf=1.0e3,
            contact_mu=0.9,
            limit_ke=1.0e3,
            limit_kd=1.0e1,
            armature=1.0,
            enable_self_collisions=False,
        )  # TODO: add enable_self_collisions?

        # set initial joint positions, targets unnecessary
        builder.joint_q[3:6] = [0.0, 0.0, np.pi / 2]

        # set joint targets to rest pose in mjcf
        # builder.joint_q[3 : self.num_joint_q + 6] = [0] * 3

    def assign_actions(self, actions):
        actions = actions.view((self.num_envs, self.num_actions)) * self.action_strength
        actions = torch.clip(actions, -1.0, 1.0)
        acts_per_env = int(self.model.joint_act.shape[0] / self.num_envs)
        joint_act = torch.zeros((self.num_envs * acts_per_env), dtype=torch.float32, device=self.device)
        act_types = {
            1: [True],
            3: [],
            4: [False] * 6,
            5: [True] * 3,
            6: [True] * 2,
            8: [False] * 3,
        }
        joint_types = self.model.joint_type.numpy()
        act_idx = np.concatenate([act_types[i] for i in joint_types])
        joint_act[act_idx] = actions.flatten()
        if not self.requires_grad:
            self.model.joint_act.assign(joint_act.detach().cpu().numpy())
        self.actions = actions.clone()

    def check_contact_count(self):
        contact_count = np.zeros(self.num_envs, dtype=int)
        shape0 = self.model.rigid_contact_shape0.numpy()
        for i in range(self.model.rigid_contact_count.numpy().item()):
            if shape0[i] <= 0:
                continue
            env_idx = self.model.shape_collision_group[shape0[i]] - 1
            contact_count[env_idx] += 1
        self.contact_count_changed[:] = torch.as_tensor(contact_count != self.prev_contact_count)
        self.prev_contact_count = contact_count
        self.extras["truncation"] = self.contact_count_changed

    def step(self, actions):
        """Simulate the environment for one timestep."""
        self.assign_actions(actions)

        with wp.ScopedTimer("simulate", active=False, detailed=False):
            if self.requires_grad:
                self.record_forward_simulate(actions)
            else:
                # simulates without recording on tape
                self.update()

        self.sim_time += self.sim_dt * self.sim_substeps

        self.reset_buf = torch.zeros_like(self.reset_buf)

        self.check_contact_count()
        self.progress_buf += 1
        self.num_frames += 1

        self.calculateObservations()
        self.calculateReward()

        self._check_early_termination()  # updates reset buf & extras

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        if self.requires_grad:
            self.obs_buf_before_reset = self.obs_buf.detach().clone()
            self.extras.update(
                {
                    "obs_before_reset": self.obs_buf_before_reset,
                }
            )

        if len(env_ids) > 0:
            self.reset(env_ids)

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def get_stochastic_init(self, env_ids, joint_q, joint_qd):
        joint_q[env_ids, 0:2] = (
            joint_q[env_ids, 0:2] + 0.05 * (torch.rand(size=(len(env_ids), 2), device=self.device) - 0.5) * 2.0
        )
        joint_q[env_ids, 2] = (torch.rand(len(env_ids), device=self.device) - 0.5) * 0.1
        joint_q[env_ids, 3:] = (
            joint_q[env_ids, 3:]
            + 0.05 * (torch.rand(size=(len(env_ids), self.num_joint_q - 3), device=self.device) - 0.5) * 2.0
        )
        joint_qd[env_ids, :] = (
            0.05 * (torch.rand(size=(len(env_ids), self.num_joint_qd), device=self.device) - 0.5) * 2.0
        )
        return joint_q[env_ids, :], joint_qd[env_ids, :]

    def reset(self, env_ids=None, force_reset=True):
        super().reset(env_ids, force_reset)  # resets state_0 and joint_q
        self.initialize_trajectory()  # sets goal_trajectory & obs_buf
        if self.requires_grad and not self.use_graph_capture:
            self.model.allocate_rigid_contacts(requires_grad=self.requires_grad)
        elif self.use_graph_capture:
            # zero rigid contact arrays allocated in model finalize
            clear_arrays(self.model, lambda k: k.startswith("rigid"))
        self.prev_contact_count = np.zeros(self.num_envs, dtype=int)
        self.contact_count_changed.fill_(0)
        wp.sim.collide(self.model, self.state_0)
        return self.obs_buf

    def calculateObservations(self):
        self.obs_buf = torch.cat(
            [
                self.joint_q.view(self.num_envs, -1)[:, 1:],
                self.joint_qd.view(self.num_envs, -1),
            ],
            dim=-1,
        )

    def calculateReward(self):
        height_diff = self.obs_buf[:, 0] - (self.termination_height + self.termination_height_tolerance)
        height_reward = torch.clip(height_diff, -1.0, 0.3)
        height_reward = torch.where(height_reward < 0.0, -200.0 * height_reward * height_reward, height_reward)
        height_reward = torch.where(height_reward > 0.0, self.height_rew_scale * height_reward, height_reward)

        angle_reward = 1.0 * (-self.obs_buf[:, 1] ** 2 / (self.termination_angle**2) + 1.0)

        progress_reward = self.obs_buf[:, 5]

        self.rew_buf = (
            progress_reward + height_reward + angle_reward + torch.sum(self.actions**2, dim=-1) * self.action_penalty
        )

    def _check_early_termination(self):
        # reset agents
        self.extras["termination"] = self.obs_buf[:, 0] < self.termination_height
        self.extras["episode_end"] = self.progress_buf > self.episode_length - 1
        self.reset_buf = torch.where(
            self.extras["episode_end"],
            torch.ones_like(self.reset_buf),
            self.reset_buf,
        )
        self.reset_buf = torch.where(
            self.extras["termination"],
            torch.ones_like(self.reset_buf),
            self.reset_buf,
        )
        return
