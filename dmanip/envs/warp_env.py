import os
import random

from gym import spaces
import numpy as np
import torch

import warp as wp
import warp.sim
import warp.sim.render

from .environment import Environment
from dmanip.utils import dr_params
from dmanip.utils.common import RenderMode, IntegratorType
from dmanip.utils.autograd2 import forward_ag, get_compute_graph
from dmanip.utils.warp_utils import integrate_body_f


torch.set_default_dtype(torch.float32)


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


class WarpEnv(Environment):
    env_offset = (6.0, 0.0, 6.0)
    opengl_render_settings = dict(
        scaling=3.0,
        headless=False,
        screen_width=480,
        screen_height=480,
        near_plane=0.1,
        far_plane=100.0,
    )
    usd_render_settings = dict(scaling=100.0)
    use_graph_capture = False
    forward_sim_graph = None

    sim_substeps_euler = 32
    sim_substeps_xpbd = 5

    xpbd_settings = dict(
        iterations=2,
        joint_linear_relaxation=0.7,
        joint_angular_relaxation=0.5,
        rigid_contact_relaxation=1.0,
        rigid_contact_con_weighting=True,
    )
    activate_ground_plane: bool = True
    ag_return_body: bool = False

    def __init__(
        self,
        num_envs,
        num_obs,
        num_act,
        episode_length,
        seed=0,
        no_grad=True,
        render=True,
        stochastic_init=False,
        device="cuda",
        env_name="warp_env",
        render_mode=RenderMode.OPENGL,
        stage_path=None,
    ):
        self.seed = seed
        self.requires_grad = not no_grad
        self._device = device
        self.visualize = render
        if not self.visualize:
            self.render_mode = RenderMode.NONE
        else:
            self.render_mode = render_mode
        self.stochastic_init = stochastic_init

        print("Running with stochastic_init: ", self.stochastic_init)
        self.sim_time = 0.0
        self.num_frames = 0

        self.num_environments = num_envs
        self.env_name = env_name

        if stage_path is None:
            self.stage_path = f"{self.env_name}_{self.num_envs}"
        else:
            self.stage_path = stage_path

        self.sim_name = self.stage_path
        self.render_time = 0.0

        # potential imageio writer if writing to an mp4
        self.writer = None

        # initialize observation and action space
        self.num_observations = num_obs
        self.num_actions = num_act
        self.episode_length = episode_length

        set_seed(self.seed)

        self.obs_space = spaces.Box(
            np.ones(self.num_observations) * -np.Inf,
            np.ones(self.num_observations) * np.Inf,
        )
        self.act_space = spaces.Box(np.ones(self.num_actions) * -1.0, np.ones(self.num_actions) * 1.0)

        # allocate buffers
        self.obs_buf = torch.zeros(
            (self.num_envs, self.num_observations),
            device=self.device,
            dtype=torch.float,
        )
        self.rew_buf = torch.zeros(
            self.num_envs,
            device=self.device,
            dtype=torch.float,
        )
        self.actions = torch.zeros(
            (self.num_envs, self.num_actions),
            device=self.device,
            dtype=torch.float,
            requires_grad=self.requires_grad,
        )

        self.extras = {}
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long, requires_grad=False)
        # end of the episode
        self.termination_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long, requires_grad=False)
        self.progress_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long, requires_grad=False)

        self._model = None

    @property
    def device(self):
        return str(self._device)

    @device.setter
    def device(self, device):
        self._device = device

    @property
    def model(self):
        if self._model is None:
            raise NotImplementedError("Model not initialized")
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    @property
    def observation_space(self):
        return self.obs_space

    @property
    def action_space(self):
        return self.act_space

    @property
    def num_envs(self):
        return self.num_environments

    @property
    def num_acts(self):
        return self.num_actions

    @property
    def num_obs(self):
        return self.num_observations

    def setup_autograd_vars(self):
        dof_count = int(self.model.joint_act.size / self.num_envs)
        act = wp.zeros(
            self.num_envs * self.num_acts,
            dtype=self.model.joint_act.dtype,
            device=self.device,
        )
        assert dof_count * self.num_envs == self.model.joint_act.size
        self.simulate_params = {
            "model": self.model,
            "integrator": self.integrator,
            "dt": self.sim_dt,
            "substeps": self.sim_substeps,
            "state_in": self.state_0,
            "state_out": self.state_1,
            "ag_return_body": self.ag_return_body,
            "body_f": wp.zeros_like(self.state_0.body_f),
        }
        self.act_params = {
            "q_offset": 0,
            "joint_act": self.model.joint_act,
            "act": act,
            "num_envs": self.num_envs,
            "dof_count": dof_count,
            "num_acts": self.num_acts,
        }
        self.graph_capture_params = {
            "capture_graph": self.use_graph_capture,
            "model": self.model,
            "bwd_model": self.model,
        }
        self.graph_capture_params["joint_q_end"] = self._joint_q
        self.graph_capture_params["joint_qd_end"] = self._joint_qd

        if self.use_graph_capture and self.requires_grad:
            # if not self.activate_ground_plane:  # ie no contact
            backward_model = self.builder.finalize(device=self.device, integrator=self.integrator)
            # else:
            # backward_model = self.builder.finalize(
            #     device=self.device,
            # rigid_mesh_contact_max=self.rigid_mesh_contact_max,
            # requires_grad=self.requires_grad,
            # )
            backward_model.joint_q.requires_grad = True
            backward_model.joint_qd.requires_grad = True
            backward_model.joint_act.requires_grad = True
            backward_model.body_q.requires_grad = True
            backward_model.body_qd.requires_grad = True
            backward_model.ground = self.activate_ground_plane
            backward_model.joint_attach_ke = self.joint_attach_ke
            backward_model.joint_attach_kd = self.joint_attach_kd
            self.act_params["bwd_joint_act"] = backward_model.joint_act
            self.graph_capture_params["bwd_model"] = backward_model
            # persist tape across multiple calls to backward
            self.graph_capture_params["tape"] = wp.Tape()
            self.simulate_params["bwd_state_in"] = backward_model.state()
            self.simulate_params["bwd_state_out"] = backward_model.state()
            self.simulate_params["bwd_state_in"].body_q.requires_grad = True
            self.simulate_params["bwd_state_in"].body_qd.requires_grad = True
            self.simulate_params["bwd_state_in"].body_f.requires_grad = True
            self.simulate_params["bwd_state_out"].body_q.requires_grad = True
            self.simulate_params["bwd_state_out"].body_qd.requires_grad = True
            self.simulate_params["bwd_state_out"].body_f.requires_grad = True
            self.simulate_params["state_list"] = [backward_model.state() for _ in range(self.sim_substeps - 1)]
        elif self.requires_grad:
            self.simulate_params["state_list"] = [self.model.state() for _ in range(self.sim_substeps - 1)]

        for state in self.simulate_params.get("state_list", []):
            state.body_q.requires_grad = True
            state.body_qd.requires_grad = True
            state.body_f.requires_grad = True

    def init_sim(self):
        self.init()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self._joint_q = wp.zeros_like(self.model.joint_q)
        self._joint_qd = wp.zeros_like(self.model.joint_qd)
        if self.requires_grad:
            self.state_0.body_q.requires_grad = True
            self.state_1.body_q.requires_grad = True
            self.state_0.body_qd.requires_grad = True
            self.state_1.body_qd.requires_grad = True
            if self.integrator_type == IntegratorType.EULER:
                self.state_0.body_f.requires_grad = True
                self.state_1.body_f.requires_grad = True
            if self.integrator_type == IntegratorType.XPBD:
                # self.state_0.body_deltas.requires_grad = True
                # self.state_1.body_deltas.requires_grad = True
                for bd in self.state_0.body_deltas:
                    bd.requires_grad = True
                for bd in self.state_1.body_deltas:
                    bd.requires_grad = True
            self._joint_q.requires_grad = True
            self._joint_qd.requires_grad = True
        self.body_q, self.body_qd = wp.to_torch(self.state_0.body_q), wp.to_torch(self.state_0.body_qd)
        self.render_time = 0.0

        start_joint_q = wp.to_torch(self.model.joint_q)
        start_joint_qd = wp.to_torch(self.model.joint_qd)
        self.joint_q = start_joint_q.clone().detach().flatten()
        self.joint_qd = start_joint_qd.clone().detach().flatten()

        # Buffers copying initial state, with env batch dimension
        self.start_joint_q = start_joint_q.clone().view(self.num_envs, -1).detach()
        self.start_joint_qd = start_joint_qd.clone().view(self.num_envs, -1).detach()

    def calculateObservations(self):
        """
        Calculate the observations for the current state
        """
        raise NotImplementedError

    def calculateReward(self):
        """
        Calculate the reward for the current state
        """
        raise NotImplementedError

    def get_stochastic_init(self, env_ids, joint_q, joint_qd):
        """
        Get the rand initial state for the environment
        """
        pass

    def reset(self, env_ids=None, force_reset=True):
        if env_ids is None:
            if force_reset:
                env_ids = np.arange(self.num_envs, dtype=int)
        if env_ids is not None:
            # fixed start state
            if self.joint_q is not None:
                joint_q = self.joint_q.clone().view(self.num_envs, -1)
                joint_qd = self.joint_qd.clone().view(self.num_envs, -1)
                joint_q[env_ids, :] = self.start_joint_q[env_ids, :].clone()
                joint_qd[env_ids, :] = self.start_joint_qd[env_ids, :].clone()
            else:
                joint_q, joint_qd = (
                    self.start_joint_q.clone(),
                    self.start_joint_qd.clone(),
                )

            with torch.no_grad():
                if self.stochastic_init:
                    (
                        joint_q[env_ids],
                        joint_qd[env_ids],
                    ) = self.get_stochastic_init(env_ids, joint_q, joint_qd)

            # assign model joint_q/qd from start pos/randomized pos
            # this effects body_q when eval_fk is called later
            self.joint_q, self.joint_qd = joint_q.view(-1), joint_qd.view(-1)
            self._joint_q.assign(wp.from_torch(self.joint_q.detach()))
            self._joint_qd.assign(wp.from_torch(self.joint_qd.detach()))

            # requires_grad is properly set in clear_grad()
            self.model.joint_act.zero_()
            self.state_0.clear_forces()
            self.state_1.clear_forces()

            # only check body state requires_grad, assumes rest of state is correct
            assert self.state_0.body_q.requires_grad == self.requires_grad
            assert self.state_0.body_qd.requires_grad == self.requires_grad
            if self.integrator_type == IntegratorType.EULER:
                assert self.state_0.body_f.requires_grad == self.requires_grad
            if self.integrator_type == IntegratorType.XPBD and self.requires_grad:
                # assert self.state_1.body_deltas.requires_grad
                for bd in self.state_1.body_deltas:
                    assert bd.requires_grad == self.requires_grad

            # updates state body positions after reset
            wp.sim.eval_fk(self.model, self._joint_q, self._joint_qd, None, self.state_0)
            self.body_q = wp.to_torch(self.state_0.body_q).requires_grad_(self.requires_grad)
            self.body_qd = wp.to_torch(self.state_0.body_qd).requires_grad_(self.requires_grad)
            self.simulate_params["state_in"] = self.state_0
            self.simulate_params["state_out"] = self.state_1
            # reset progress buffer (i.e. episode done flag)
            self.progress_buf[env_ids] = 0
            self.num_frames = 0
            self.calculateObservations()

        return self.obs_buf

    def update_joints(self):
        self.joint_q = wp.to_torch(self._joint_q).clone()
        self.joint_qd = wp.to_torch(self._joint_qd).clone()

    def record_forward_simulate(self, actions):
        """
        Runs forward simulation of the environment with autograd enabled
        If self.use_graph_capture, records the graph for the forward simulation
        """
        # does this cut off grad to prev timestep?
        assert self.model.body_q.requires_grad and self.state_0.body_q.requires_grad
        # all grads should be from joint_q, not from body_q
        with torch.no_grad():
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
        if not self.use_graph_capture:
            (
                self.simulate_params["state_in"],
                self.simulate_params["state_out"],
            ) = (
                self.state_1,
                self.state_0,
            )
        if not self.simulate_params["ag_return_body"]:
            self.joint_q, self.joint_qd = ret
            self.body_q = wp.to_torch(self.simulate_params["state_out"].body_q)
            self.body_qd = wp.to_torch(self.simulate_params["state_out"].body_qd)
        else:
            self.joint_q, self.joint_qd, self.body_q, self.body_qd = ret

        self.body_f = wp.to_torch(self.simulate_params["body_f"])

    def _pre_step(self):
        """Method to store relevant data before stepping the simulation"""
        pass

    def _post_step(self):
        """Method to store relevant data after stepping the simulation and computing observation and reward"""
        pass

    def update(self):
        """Overrides Environment.update() for forward simulation with graph capture"""

        # simulates with graph capture if selected
        def forward():
            for _ in range(self.sim_substeps):
                self.state_0.clear_forces()
                if self.activate_ground_plane:
                    wp.sim.collide(self.model, self.state_0)
                # if not self.use_graph_capture:
                #     self.state_1 = self.model.state(self.requires_grad)
                # simulation step
                self.state_1 = self.integrator.simulate(self.model, self.state_0, self.state_1, self.sim_dt)
                if not self.use_graph_capture:
                    self.state_0, self.state_1 = self.state_1, self.state_0
                else:
                    self.state_0.body_q.assign(self.state_1.body_q)
                    self.state_0.body_qd.assign(self.state_1.body_qd)

            self.simulate_params["body_f"].zero_()
            integrate_body_f(
                self.model,
                self.state_1.body_qd,
                self.state_0.body_q,
                self.state_0.body_qd,
                self.simulate_params["body_f"],
                self.frame_dt,
            )
            wp.sim.eval_ik(self.model, self.state_0, self._joint_q, self._joint_qd)
            # wp.sim.eval_ik(self.model, self.state_0, self.state_0.joint_q, self.state_0.joint_qd)
            # self._joint_q.assign(self.state_0.joint_q)
            # self._joint_qd.assign(self.state_0.joint_qd)

        if self.use_graph_capture:
            if self.forward_sim_graph is None:
                self.forward_sim_graph = get_compute_graph(forward)
            wp.capture_launch(self.forward_sim_graph)
        else:
            forward()

        self.body_q = wp.to_torch(self.state_0.body_q)
        self.body_qd = wp.to_torch(self.state_0.body_qd)
        self.body_f = wp.to_torch(self.simulate_params["body_f"])
        self.update_joints()
        return self.forward_sim_graph

    def clear_grad(self, checkpoint=None):
        """
        cut off the gradient from the current state to previous states
        """
        if checkpoint is None and (self.progress_buf > 0).any():
            checkpoint = self.get_checkpoint()
        else:
            checkpoint = None
        with torch.no_grad():
            if checkpoint is not None:
                self.load_checkpoint(checkpoint)
            self.actions = self.actions.clone()
            if self.actions.grad is not None:
                self.actions.grad.zero()
            current_joint_act = wp.to_torch(self.model.joint_act).detach().clone()
            if self.act_params["act"].grad is not None:
                self.act_params["act"].grad.zero_()
            # grads will not be assigned since variables are detached
            self.model.joint_act.assign(wp.from_torch(current_joint_act))
            if self.model.joint_q.grad is not None:
                self.model.joint_q.grad.zero_()
                self.model.joint_qd.grad.zero_()
                self.model.joint_act.grad.zero_()
            if self.state_0.body_q.grad is not None:
                self.state_0.body_q.grad.zero_()
                self.state_0.body_qd.grad.zero_()
                if self.integrator_type == IntegratorType.EULER:
                    self.state_0.body_f.grad.zero_()
                if self.integrator_type == IntegratorType.XPBD and self.requires_grad:
                    # if self.state_0.body_deltas.grad is not None:
                    #     self.state_0.body_deltas.zero_()
                    # if self.state_1.body_deltas.grad is not None:
                    #     self.state_1.body_deltas.zero_()

                    for bd in self.state_0.body_deltas:
                        if bd.grad is not None:
                            bd.grad.zero_()
                    for bd in self.state_1.body_deltas:
                        if bd.grad is not None:
                            bd.grad.zero_()
            if self.state_1.body_q.grad is not None:
                self.state_1.body_q.grad.zero_()
                self.state_1.body_qd.grad.zero_()
                if self.integrator_type == IntegratorType.EULER:
                    self.state_1.body_f.grad.zero_()

        if self.requires_grad:
            assert self.model.joint_q.requires_grad, "joint_q requires_grad not set"
            assert self.model.joint_qd.requires_grad, "joint_qd requires_grad not set"
            assert self.model.joint_act.requires_grad, "joint_act requires_grad not set"

    def step(self, act):
        """
        Step the simulation forward by one timestep
        """
        raise NotImplementedError

    def render(self, mode="human"):
        if self.render_mode == RenderMode.OPENGL:
            self.renderer.paused = False
        if mode == "rgb_array":
            pixels = wp.zeros(
                (self.renderer.screen_height, self.renderer.screen_width, 3),
                dtype=np.float,
            )
        if self.visualize and self.renderer:
            with wp.ScopedTimer("render", False):
                self.render_time += self.frame_dt
                self.renderer.begin_frame(self.render_time)
                # render next_state (swapped with state 0 in update())
                self.renderer.render(self.state_0)
                self.renderer.end_frame()
                if self.render_mode == RenderMode.USD:
                    self.renderer.save()
                if mode == "rgb_array":
                    self.renderer.get_pixels(pixels, split_up_tiles=False)
                    return pixels.numpy()

    def get_checkpoint(self, save_path=None):
        checkpoint = {}
        self.update_joints()
        joint_q, joint_qd = self.joint_q.detach(), self.joint_qd.detach()
        body_q, body_qd = self.body_q.detach(), self.body_qd.detach()
        # assert np.all(joint_q.cpu().numpy() == self._joint_q.numpy())
        # assert np.all(joint_qd.cpu().numpy() == self._joint_qd.numpy())
        # assert np.all(body_q.cpu().numpy() == self.simulate_params["state_in"].body_q.numpy())
        # assert np.all(body_qd.cpu().numpy() == self.simulate_params["state_in"].body_qd.numpy())
        checkpoint["joint_q"] = joint_q.clone()
        checkpoint["joint_qd"] = joint_qd.clone()
        checkpoint["body_q"] = body_q.clone()
        checkpoint["body_qd"] = body_qd.clone()
        checkpoint["obs_buf"] = self.obs_buf.clone()
        checkpoint["actions"] = self.actions.clone()
        checkpoint["progress_buf"] = self.progress_buf.clone()
        checkpoint["reset_buf"] = self.reset_buf.clone()
        if save_path:
            print("saving checkpoint to", save_path)
            torch.save(checkpoint, save_path)
        return checkpoint

    def load_checkpoint(self, checkpoint_data={}, ckpt_path=None):
        if ckpt_path is not None:
            print("loading checkpoint from {}".format(ckpt_path))
            checkpoint_data = torch.load(ckpt_path)
        joint_q = checkpoint_data["joint_q"]
        joint_qd = checkpoint_data["joint_qd"]
        self.joint_q = joint_q.flatten().requires_grad_(self.requires_grad)
        self.joint_qd = joint_qd.flatten().requires_grad_(self.requires_grad)
        self._joint_q.assign(wp.from_torch(self.joint_q))
        self._joint_qd.assign(wp.from_torch(self.joint_qd))
        num_bodies_per_env = int(self.model.body_count / self.num_envs)
        body_q = checkpoint_data["body_q"].view(-1, num_bodies_per_env, 7)
        body_qd = checkpoint_data["body_qd"].view(-1, num_bodies_per_env, 6)
        self.body_q = body_q[: self.num_envs].view(-1, 7).requires_grad_(self.requires_grad)
        self.body_qd = body_qd[: self.num_envs].view(-1, 6).requires_grad_(self.requires_grad)
        self.simulate_params["state_in"].body_q.assign(wp.from_torch(self.body_q))
        self.simulate_params["state_in"].body_qd.assign(wp.from_torch(self.body_qd))
        # assumes self.num_envs <= number of actors in checkpoint
        self.actions[:] = checkpoint_data["actions"][: self.num_envs]
        self.progress_buf[:] = checkpoint_data["progress_buf"][: self.num_envs].view(-1)
        self.reset_buf[:] = checkpoint_data["reset_buf"][: self.num_envs].view(-1)
        self.num_frames = self.progress_buf[0].item()

    # def set_env_state(self, env_state):
    # self.load_checkpoint(env_state)
    # return

    def initialize_trajectory(self):
        self.clear_grad()
        self.calculateObservations()
        return self.obs_buf

    def close(self):
        if self.writer is not None:
            self.writer.close()
        if self.render_mode is RenderMode.USD:
            self.renderer.app.window.set_request_exit()

    def save_camera_params(self, path=None):
        if path is None:
            path = "default_camera_params.npz"
        params = {
            "model": self.renderer._model_matrix,
            "projection": self.renderer._projection_matrix,
            "view": self.renderer._view_matrix,
        }
        np.savez(path, **params)

    def load_camera_params(self, params=None):
        if params is None:
            if os.path.exists("default_camera_params.npz"):
                params = np.load("default_camera_params.npz")
            self.save_params = params
        elif isinstance(params, str):
            if os.path.exists(params):
                params = np.load(params)
        else:
            return
        self.renderer._model_matrix = params["model"]
        self.renderer._projection_matrix = params["projection"]
        self.renderer._view_matrix = params["view"]


class HybridWarpEnv(WarpEnv):
    def __init__(
        self,
        num_envs,
        num_obs,
        num_act,
        episode_length,
        seed=0,
        no_grad=True,
        render=True,
        stochastic_init=True,
        device="cuda",
        env_name="warp_env",
        render_mode=RenderMode.OPENGL,
        stage_path=None,
        hybrid_params=dr_params.HybridParams(),
    ):
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
            env_name,
            render_mode,
            stage_path,
        )
        self.hybrid_params = hybrid_params


class DRWarpEnv(WarpEnv):
    def __init__(
        self,
        num_envs,
        num_obs,
        num_act,
        episode_length,
        seed=0,
        no_grad=True,
        render=True,
        stochastic_init=True,
        device="cuda",
        env_name="warp_env",
        render_mode=RenderMode.OPENGL,
        stage_path=None,
        randomization_params=dr_params.RandomizationParams(),
        randomize=True,
    ):
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
            env_name,
            render_mode,
            stage_path,
        )
        self.randomization_params = randomization_params
        self.randomize = randomize
        self.rng = np.random.RandomState(seed)

    def get_stochastic_init(self, env_ids, joint_q, joint_qd):
        """Get stochastic initializations for the environment."""
        dr_params = self.randomization_params.sample_params()
        randomized_values = self.apply_randomizations(dr_params, env_ids)
        if "joint_q" in randomized_values:
            joint_q = randomized_values["joint_q"]
        if "joint_qd" in randomized_values:
            joint_qd = randomized_values["joint_qd"]
        return joint_q, joint_qd

    def apply_randomizations(self, dr_params, env_ids=None):
        """Apply physics randomization to the environment params.

        Only applies randomizations on resets

        Args:
            dr_params: dict containing params for domain randomization to use.
            env_ids: selective randomisation of environments
        """
        if not self.randomize:
            return

        params = self.randomize_params.sample_params(include_list=dr_params)
        if env_ids is None:
            env_ids = torch.ones(self.num_envs, device=self.device)

        if "physics_params" in params:
            bwd_model = self.graph_capture_params["bwd_model"]
            for param in params["physics_params"]:
                if param == "gravity":
                    self.model.gravity[self.model.gravity.nonzero()] = params["physics_params"][param]
                    bwd_model.gravity[self.model.gravity.nonzero()] = params["physics_params"][param]
                if param == "rigid_contact_torsional_friction":
                    self.model.rigid_contact_torsional_friction = params["physics_params"][param]
                    bwd_model.rigid_contact_torsional_friction = params["physics_params"][param]
                if param == "rigid_contact_rolling_friction":
                    self.model.rigid_contact_rolling_friction = params["physics_params"][param]
                    bwd_model.rigid_contact_rolling_friction = params["physics_params"][param]

        if "joint_params" in params:
            joint_params = self.randomize_joints(params["joint_params"], env_ids)
        else:
            joint_params = {}
        return joint_params

    def randomize_joints(self):
        return {}
