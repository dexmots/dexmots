import torch
import warp as wp
import os
import numpy as np

from .warp_env import WarpEnv
from .environment import RenderMode, IntegratorType
from ..utils import torch_utils as tu
from ..utils.autograd import forward_ag, assign_act, clear_grads


class CartPoleSwingUpEnv(WarpEnv):
    render_mode: RenderMode = RenderMode.USD
    integrator_type: IntegratorType = IntegratorType.EULER
    activate_ground_plane: bool = False
    use_graph_capture: bool = False

    def __init__(
        self,
        render=False,
        device="cuda",
        num_envs=1024,
        seed=0,
        episode_length=240,
        no_grad=True,
        stochastic_init=False,
        early_termination=False,
        stage_path=None,
        env_name="CartPoleSwingUpEnv",
        ag_return_body=False,
        graph_capture=False,
        start_pole_angle: float = 0.0,
        start_pole_vel: float = 0.0,
        start_cart_vel: float = 0.0,
    ):
        num_obs = 5
        num_act = 1
        if graph_capture:
            self.use_graph_capture = True

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
        self.num_joint_q = 2
        self.num_joint_qd = 2
        self.early_termination = early_termination
        self.ag_return_body = ag_return_body
        self.start_pole_angle = start_pole_angle
        self.start_pole_vel = start_pole_vel
        self.start_cart_vel = start_cart_vel
        self.init_sim()
        self.setup_autograd_vars()

        self.model.joint_attach_ke = 3.2e4
        self.model.joint_attach_kd = 50.0
        if self.use_graph_capture:
            self.graph_capture_params["bwd_model"].joint_attach_ke = 3.2e4
            self.graph_capture_params["bwd_model"].joint_attach_kd = 50.0

        self.simulate_params["ag_return_body"] = self.ag_return_body
        # action parameters
        self.action_strength = 1000.0

        # loss related
        self.pole_angle_penalty = 1.0
        self.pole_velocity_penalty = 0.1

        self.cart_position_penalty = 0.05
        self.cart_velocity_penalty = 0.1

        self.cart_action_penalty = 0.0

    def create_articulation(self, builder):
        assets_dir = os.path.join(os.path.dirname(__file__), "assets")
        wp.sim.parse_urdf(
            os.path.join(assets_dir, "cartpole.urdf"),
            builder,
            xform=wp.transform_identity(),
            floating=False,
            density=0,
            armature=0.1,
            stiffness=0.0,
            damping=0.0,
            shape_ke=1.0e4,
            shape_kd=1.0e2,
            shape_kf=1.0e2,
            shape_mu=0.5,
            limit_ke=1e2,
            limit_kd=1e1,
            enable_self_collisions=False,
        )
        builder.joint_q[-2:] = [0.0, self.start_pole_angle]
        builder.joint_qd[-2:] = [self.start_cart_vel, self.start_pole_vel]

    def assign_actions(self, actions):
        actions = actions.flatten() * self.action_strength
        # actions = torch.clip(actions, -1.0, 1.0)
        self.act_params["act"].assign(wp.from_torch(actions))
        assign_act(**self.act_params)

    def step(self, actions, profiler=None):
        with wp.ScopedTimer("simulate", active=False, detailed=False, dict=profiler):
            actions = torch.clip(actions, -1.0, 1.0)
            self.actions = actions.view(self.num_envs, -1)
            actions = self.action_strength * actions

            if self.requires_grad:
                # does this cut off grad to prev timestep?
                assert self.model.body_q.requires_grad and self.state_0.body_q.requires_grad
                # all grads should be from joint_q, not from body_q
                # with torch.no_grad():
                if not self.ag_return_body:
                    body_q, body_qd = (
                        self.body_q.detach().clone(),
                        self.body_qd.detach().clone(),
                    )
                else:
                    body_q = self.body_q.clone()
                    body_qd = self.body_qd.clone()

                ret = forward_ag(
                    self.simulate_params,
                    self.graph_capture_params,
                    self.act_params,
                    actions.flatten(),
                    body_q,
                    body_qd,
                )
                # swap states so start from correct next state
                (self.simulate_params["state_in"], self.simulate_params["state_out"],) = (
                    self.state_1,
                    self.state_0,
                )
                if self.ag_return_body:
                    self.joint_q, self.joint_qd, self.body_q, self.body_qd = ret
                else:
                    self.joint_q, self.joint_qd = ret
                    self.body_q = wp.to_torch(self.simulate_params["state_in"].body_q)
                    self.body_qd = wp.to_torch(self.simulate_params["state_in"].body_qd)

            else:
                self.assign_actions(self.actions)
                self.update()

            self.sim_time += self.sim_dt * self.sim_substeps

        self.reset_buf = torch.zeros_like(self.reset_buf)

        self.progress_buf += 1
        self.num_frames += 1

        self.calculateObservations()
        self.calculateReward()

        if self.requires_grad:
            self.obs_buf_before_reset = self.obs_buf.clone()
            self.extras = {
                "obs_before_reset": self.obs_buf_before_reset,
                "episode_end": self.termination_buf,
            }

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        if len(env_ids) > 0:
            with wp.ScopedTimer("reset", active=False, detailed=False, dict=profiler):
                self.reset(env_ids)

        if self.visualize and self.render_mode is not RenderMode.TINY:
            with wp.ScopedTimer("render", active=False, detailed=False, dict=profiler):
                self.render()

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def reset(self, env_ids=None):
        if not self.use_graph_capture:
            self.state_0, self.state_1 = (
                self.simulate_params["state_in"],
                self.simulate_params["state_out"],
            )
        return super().reset(env_ids)

    def get_stochastic_init(self, env_ids, joint_q, joint_qd):
        rand_init_q = np.pi * (torch.rand(size=(len(env_ids), self.num_joint_q), device=self.device) - 0.5)
        rand_init_qd = 0.5 * (torch.rand(size=(len(env_ids), self.num_joint_qd), device=self.device) - 0.5)
        return joint_q[env_ids] + rand_init_q, joint_qd[env_ids] + rand_init_qd

    def calculateObservations(self):
        joint_q, joint_qd = self.joint_q.view(self.num_envs, -1), self.joint_qd.view(self.num_envs, -1)
        if self.ag_return_body:
            body_q = self.body_q.view(self.num_envs, -1, 7)
            body_qd = self.body_qd.view(self.num_envs, -1, 6)
            x = body_q[:, 0:1, 0] - body_q[:, 1:2, 0]  # joint_q[:, 0:1]
            xdot = body_qd[:, 1, 3:4]  # joint_qd[:, 0:1]
        else:
            x = joint_q[:, 0:1]
            xdot = joint_qd[:, 0:1]

        theta = joint_q[:, 1:2]
        theta_dot = joint_qd[:, 1:2]

        # observations: [x, xdot, sin(theta), cos(theta), theta_dot]
        self.obs_buf = torch.cat([x, xdot, torch.sin(theta), torch.cos(theta), theta_dot], dim=-1)

    def calculateReward(self):
        joint_q = self.joint_q.view(self.num_envs, -1)
        joint_qd = self.joint_qd.view(self.num_envs, -1)
        if self.ag_return_body:
            body_q = self.body_q.view(self.num_envs, -1, 7)
            body_qd = self.body_qd.view(self.num_envs, -1, 6)
            x = body_q[:, 0, 0] - body_q[:, 1, 0]  #  joint_q[:, 0]
            xdot = body_qd[:, 1, 3]  #  joint_qd[:, 0]
        else:
            x = joint_q[:, 0]
            xdot = joint_qd[:, 0]

        theta = tu.normalize_angle(joint_q[:, 1])
        theta_dot = joint_qd[:, 1]

        self.rew_buf = (
            -torch.pow(theta, 2.0) * self.pole_angle_penalty
            - torch.pow(theta_dot, 2.0) * self.pole_velocity_penalty
            - torch.pow(x, 2.0) * self.cart_position_penalty
            - torch.pow(xdot, 2.0) * self.cart_velocity_penalty
            - torch.sum(self.actions**2, dim=-1) * self.cart_action_penalty
        )

        # reset agents
        self.reset_buf = torch.where(
            self.progress_buf > self.episode_length - 1,
            torch.ones_like(self.reset_buf),
            self.reset_buf,
        )

    def clear_grad(self, checkpoint=None):
        super().clear_grad(checkpoint)
        with torch.no_grad():
            self.body_q = self.body_q.clone()
            self.body_qd = self.body_qd.clone()
            # clear_grads(
            #     self.model, lambda x: x.startswith("joint_" or x.startswith("body_"))
            # )
            # if self.simulate_params["state_in"].body_q.grad is not None:
            #     clear_grads(self.simulate_params["state_in"])
            #     clear_grads(self.simulate_params["state_out"])
            #     if self.use_graph_capture:
            #         clear_grads(self.simulate_params["bwd_state_in"])
            #         clear_grads(self.simulate_params["bwd_state_out"])
            #     for s in self.simulate_params["state_list"]:
            #         clear_grads(s)
