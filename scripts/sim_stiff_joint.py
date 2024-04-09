# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Sim Stiff Joint Torque
#
# Shows how to apply an external torque to a rigid body causing
# rotate opposing joint frictional force
#
###########################################################################

import os
import os.path as osp
import pdb
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import tyro
import math
import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation as R
from tqdm import trange
from warp.sim.integrator_euler import integrate_bodies
from warp.tests.grad_utils import check_kernel_jacobian

import dmanip
from dmanip.utils import plotting, safe_save, gen_unique_filename, parse_mesh
from dmanip.utils import warp_utils as wpu
from dmanip.utils.warp_utils import PI
from dmanip.utils import builder as bu
from dmanip.utils.common import ObjectType

# wp.config.mode = "debug"
# wp.config.verify_fp = True

root_path = os.path.dirname(os.path.abspath(dmanip.__file__))

wp.init()


# wp.set_device("cpu")


TARGET = wp.constant(wp.quat(0.0, -0.70710678, 0.0, -0.70710678))
TARGET_THETA = wp.constant(np.pi * 2.50)
TARGET_POS = wp.constant(wp.vec3(0.0, 0.0, 1.0))

e2 = wp.constant(wp.vec3(0.0, 1.0, 0.0))


@wp.kernel
def loss_l2(
    states: wp.array(dtype=wp.float32),
    targets: wp.array(dtype=wp.float32),
    state_dim: int,
    body_qd: wp.array(dtype=wp.spatial_vector),
    actions: wp.array(dtype=wp.float32),
    loss: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    diff = states[i * state_dim] - targets[i * state_dim]
    qd = wp.spatial_top(body_qd[-1])
    qd1 = wp.cw_mul(qd, e2)
    l = diff * diff + wp.dot(qd1, qd1) + actions[i] ** 2.0 * 1e-4
    wp.atomic_add(loss, 0, l)


@wp.kernel
def loss_linf(
    states: wp.array(dtype=wp.float32),
    targets: wp.array(dtype=wp.float32),
    state_dim: int,
    body_qd: wp.array(dtype=wp.spatial_vector),
    actions: wp.array(dtype=wp.float32),
    loss: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    diff = states[i * state_dim] - targets[i * state_dim]
    l = wp.abs(diff)
    wp.atomic_max(loss, 0, l)


@wp.kernel
def save_state(
    q: wp.array(dtype=float), start_index: int, states: wp.array(dtype=float)
):
    states[start_index] = q[0]


@wp.kernel
def update_rotation_count(
    body_q: wp.array(dtype=wp.transform),
    joint_type: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_X_p: wp.array(dtype=wp.transform),
    joint_axis: wp.array(dtype=wp.vec3),
    joint_max_rotations: float,
    joint_q_prev: wp.array(dtype=float),
    # outputs
    joint_rotation_count: wp.array(dtype=float),
):
    """Check if joint has rotated past range [-2pi, 2pi], and incr/decrement rotation count"""
    tid = wp.tid()
    if joint_type[tid] != wp.sim.JOINT_REVOLUTE:
        return

    X_wp = joint_X_p[tid]
    X_wc = body_q[tid]
    c_parent = joint_parent[tid]
    axis = joint_axis[tid]

    if c_parent >= 0:
        X_wp = body_q[c_parent] * X_wp

    q = wpu.compute_joint_q(X_wp, X_wc, axis, 0.0)
    # increment/decrement number of rotations when sign flips btwn q_prev and q
    rotated = wp.select(
        wp.abs(joint_q_prev[0] - q) > 3.0 * PI,
        0.0,
        (wp.sign(joint_q_prev[0]) - wp.sign(q)) / 2.0,
    )
    joint_rotation_count[0] = wp.clamp(
        joint_rotation_count[0] + rotated, -joint_max_rotations, joint_max_rotations
    )
    # joint_q_prev[0] = q


@wp.kernel
def compute_obj_yaw(
    body_q: wp.array(dtype=wp.transform),
    rotation_count: float,
    # outputs
    obj_yaw: wp.array(dtype=float),
):
    obj_yaw[0] = wpu.yaw_from_Xform(body_q[1], 1) + 2.0 * PI * rotation_count


@wp.kernel
def get_joint_q(
    body_q: wp.array(dtype=wp.transform),
    joint_type: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_X_p: wp.array(dtype=wp.transform),
    joint_axis: wp.array(dtype=wp.vec3),
    joint_rotation_count: float,
    # outputs
    joint_q: wp.array(dtype=float),
):
    tid = wp.tid()
    type = joint_type[tid]
    axis = joint_axis[tid]

    if (
        type != wp.sim.JOINT_REVOLUTE
        and type != wpu.JOINT_REVOLUTE_TIGHT
        and type != wpu.JOINT_REVOLUTE_SPRING
    ):
        return

    c_child = tid
    c_parent = joint_parent[tid]
    X_wp = joint_X_p[tid]
    X_wc = body_q[c_child]

    if c_parent >= 0:
        X_wp = body_q[c_parent] * X_wp

    q = wpu.compute_joint_q(X_wp, X_wc, axis, joint_rotation_count)
    joint_q[0] = q


@dataclass(frozen=True)
class AdamStepLRConfig:
    # Optimizer learning rate
    lr: float = 1e-1

    # Optimizer grad clip by value
    grad_clip: Optional[float] = None

    # Step size for LR Scheduler, if None fixed LR used
    step_freq: Optional[int] = None

    # Minimum learning rate for LR Scheduler
    min_lr: float = 1e-5

    # Gamma for LR Scheduler, i.e. lr = lr * gamma
    gamma: float = 0.5


@dataclass(frozen=True)
class ScrewTightenConfig:
    # Render output file path
    stage_name: str = "sim_stiff_joint.usd"

    # Number of seconds to render
    duration: float = 2.5

    # Number of envs to run in parallel
    num_envs: int = 1

    # Static friction coefficient
    mu_static: float = 2.0

    # Dynamic friction coefficent
    mu_dynamic: float = 1.5

    # Maximum stiction displacement
    max_stiction_displacement: float = np.pi / 16.0

    # Maximum number of rotations to allow (# of revolutions, not radians)
    max_rotations: float = 2.0

    # Target orientation (theta in pi-radians)
    target_theta: float = 1.25

    # Stiction transition velocity, units are in rad/sec
    stiction_transition_vel: float = 0.75

    # Joint torque value
    action: float = 0.5

    # Path to debug actions npy file (for testing), must be length duration * 60
    debug_actions: Optional[str] = None

    # Joint damping
    target_kd: float = 0.0

    # Whether or not to optimize
    optimize: bool = False

    # Number of steps to optimize (if optimizing subtrajectories)
    optimize_steps: Optional[int] = None

    # Number of iterations to step optimizer
    num_iter: int = 250

    # Whether or not to show plot at the end of a run
    plot: bool = False

    # Whether or not to save log data
    save_log: bool = False

    # Whether or not to render the run and save to usd
    render: bool = True

    # Name for run
    run_id: Optional[str] = 'sim_stiff_joint'

    # LR Scheduler Config
    opt_config: AdamStepLRConfig = AdamStepLRConfig()


class ScrewTightenEnv:
    def __init__(self, config: ScrewTightenConfig):
        # Sim + render initialization
        self.setup_sim(config)

        # Revolute friction dynamics
        self.setup_friction_dynamics(config)

        # generate run_id using current time
        run_id = config.run_id or datetime.now().strftime("%y%m%d-%H%M%S")

        self.save_dir = osp.abspath(f"./outputs/{run_id}")
        print("output dir:", self.save_dir)
        if not osp.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # initialize builder and setup model and renderer (if render is True)
        self.setup_model(config)

        # State contains a) joint angle b) number of rotations
        self.state_dim = 1  # 2
        self.action_dim = 1
        self.log = defaultdict(list)

        if config.optimize or config.debug_actions is not None:
            # use sigmoid ref trajectory, to dampen acceleration
            sigmoid = lambda z: 1 / (1 + np.exp(-z))
            ref_traj = np.linspace(-10.0, 10, self.sim_frames)
            ref_traj = sigmoid(ref_traj) * np.pi * config.target_theta
            # rotations = (2 * np.pi + ref_traj) // (np.pi * 4)
            # ref_traj = np.stack([ref_traj, rotations], axis=1)  # (T, 2)
            self.ref_traj = wp.array(
                ref_traj.flatten(),
                dtype=wp.float32,
                device=self.model.device,
                requires_grad=True,
            )

    def setup_model(self, config: ScrewTightenConfig):
        builder = wp.sim.ModelBuilder(gravity=0.0)
        builder.add_articulation()
        # root link
        # root_link = builder.add_body(parent=-1, origin=wp.transform_identity())
        pos = (0.0, 0.0, 0.0)
        joint_axis = (0.0, 0.0, 1.0)
        ori = wp.quat_from_axis_angle(wp.vec3(1, 0, 0), -np.pi/2)
        contact_ke = 1.0e3
        contact_kd = 100.0
        body_link = bu.add_joint(builder, pos, ori, wp.sim.JOINT_REVOLUTE, joint_axis)
        bu.add_object(
            builder,
            body_link,
            ObjectType.OCTPRISM,
            contact_ke=contact_ke,
            contact_kd=contact_kd,
            # xform=wp.transform(wp.vec3(0.15, 0, 0), wp.quat_from_axis_angle(wp.vec3(1, 0, 0), -np.pi/2)),
        )

        if self.num_envs > 1:
            art_builder = builder
            builder = wp.sim.ModelBuilder(gravity=0.0)
            for i in range(self.num_envs):
                env_pos = np.array([i * 2, 0.0, 0.0])
                env_rot = wp.quat_identity()
                builder.add_rigid_articulation(
                    art_builder,
                    xform=wp.transform(env_pos, env_rot),
                )

        self.model = builder.finalize(requires_grad=True)
        self.joint_idx = [
            idx for idx, val in enumerate(builder.joint_name) if val == "obj_joint"
        ][0]
        self.model.ground = False
        self.integrator = wp.sim.SemiImplicitIntegrator()
        self.joint_axis_idx = 1  # np.where(joint_axis)[0].item()

        if not config.optimize:
            self.state_0 = self.model.state()
            self.state_1 = self.model.state()
        stage_path = osp.join(self.save_dir, config.stage_name)
        print(f"rendering to {stage_path}")
        if config.render:
            self.renderer = wp.sim.render.SimRenderer(self.model, stage_path)
        else:
            self.renderer = None

    def setup_sim(self, config: ScrewTightenConfig):
        self.sim_fps = 60.0
        self.sim_substeps = 8
        self.sim_duration = config.duration
        self.sim_frames = int(self.sim_duration * self.sim_fps)
        self.sim_dt = (1.0 / self.sim_fps) / self.sim_substeps
        self.sim_time = 0.0
        self.sim_render = True
        self.sim_iterations = 1
        self.sim_relaxation = 1.0
        self.num_envs = config.num_envs

    def setup_friction_dynamics(self, config: ScrewTightenConfig):
        # http://www2.me.rochester.edu/courses/ME204/nx_help/index.html#uid:id563011
        self.mu_static = config.mu_static
        self.mu_dynamic = config.mu_dynamic
        self.stiction_transition_vel = config.stiction_transition_vel
        self.max_stiction_displacement = config.max_stiction_displacement
        self.max_rotations = config.max_rotations
        self.target_theta = config.target_theta
        self.friction_out = wp.array(
            [0.0, 0.0, 0.0, 0.0, 0.0], dtype=float, requires_grad=True
        )

    def update(self, action, rotation_count=None, render=False):

        with wp.ScopedTimer("simulate", active=False, detailed=False):
            # self.model.joint_act.assign([0.0] * 6 + [action])
            action = wp.array([action], dtype=float)
            rotation_count = rotation_count or wp.array([0.0])
            self.state_1, friction_out, body_f = self.simulate(
                self.state_0, action, 0, rotation_count, next_state=self.state_1
            )
            if render:
                self.sim_time += self.sim_dt * self.sim_substeps
            self.log_values(
                self.state_0,
                friction_out.numpy(),
                rotation_count.numpy()[0],
                body_f.numpy(),
            )
            self.state_0, self.state_1 = self.state_1, self.state_0

    def log_values(self, state, friction_out, rotation_count, body_f):
        q = self.compute_joint_q(state, requires_grad=False).numpy()[0]
        joint_friction = friction_out[0]
        t_net = friction_out[1]
        t_app = friction_out[2]
        dist = friction_out[3]
        q_full = friction_out[4]
        self.log["joint pos."].append(q)
        self.log["joint pos. (+rotations)"].append(q_full)
        self.log["friction"].append(joint_friction)
        self.log["net torque pre- fric."].append(t_net)
        self.log["applied torque pre- fric."].append(t_app)
        self.log["distance"].append(dist)
        self.log["body rot."].append(wpu.yaw_from_quat(state.body_q.numpy()[-1, -4:]))
        self.log["body angular vel."].append(state.body_qd.numpy()[-1, 1])
        self.log["net torque"].append(body_f[-1, self.joint_axis_idx])
        self.log["rotations"].append(rotation_count)

    def simulate(
        self,
        state: wp.sim.State,
        action: wp.array,
        action_index: int,
        rotation_count: wp.array,
        requires_grad=False,
        next_state: Optional[wp.sim.State] = None,
    ):
        """
        Simulates joint friction for sim_substeps
        """

        prev_q = self.compute_joint_q(state, requires_grad=requires_grad)
        friction_out = wp.array(np.zeros(5), dtype=float, requires_grad=requires_grad)
        for substep in range(self.sim_substeps):
            if requires_grad:
                next_state = self.model.state(requires_grad=True)
            elif next_state is None:
                next_state = state
            # apply generalized torques to rigid body here, instead of planar joints
            self.model.joint_act = wp.zeros_like(
                self.model.joint_act, requires_grad=True
            )
            wp.launch(
                kernel=wpu.revolute_tight_friction,
                dim=self.model.joint_count,
                inputs=[
                    state.body_q,
                    state.body_qd,
                    action,
                    action_index,
                    self.model.joint_q,
                    self.model.joint_qd_start,
                    self.model.joint_type,
                    self.model.joint_parent,
                    self.model.joint_X_p,
                    self.model.joint_axis,
                    self.model.joint_target,
                    self.model.joint_limit_lower,
                    self.model.joint_limit_upper,
                    self.model.joint_target_ke,
                    self.model.joint_target_kd,
                    self.model.joint_limit_ke,
                    self.model.joint_limit_kd,
                    rotation_count,
                    self.max_rotations,
                    self.mu_static,
                    self.mu_dynamic,
                    self.stiction_transition_vel,
                    self.max_stiction_displacement,
                ],
                outputs=[self.model.joint_act, friction_out],
                device=self.model.device,
            )
            next_state = self.integrate_step(
                state, next_state, action, action_index, rotation_count
            )
            # updates rotation count
            wp.launch(
                kernel=update_rotation_count,
                dim=self.model.joint_count,
                inputs=[
                    state.body_q,
                    self.model.joint_type,
                    self.model.joint_parent,
                    self.model.joint_X_p,
                    self.model.joint_axis,
                    self.max_rotations,
                    prev_q,
                ],
                outputs=[rotation_count],
                device=self.model.device,
            )
            # updates prev joint_q (for rotation count)
            self.compute_joint_q(state, joint_q=prev_q, requires_grad=requires_grad)
            # swap next state and prev state, body_f in prev_state should be zeroed
            next_state, state = state, next_state
            body_f = wp.clone(state.body_f)
            next_state.clear_forces()
        return state, friction_out, body_f

    def forward(
        self,
        actions: wp.array,
        start_state: Optional[wp.sim.State] = None,
        rotation_count: Optional[wp.array] = None,
        requires_grad: bool = False,
        loss=None,
        render=False,
        log=False,
        loss_fn=None,
    ):
        """
        Advances the system dynamics given the rigid-body state in maximal coordinates and generalized joint torques [body_q, body_qd, tau].
        Simulates for the set number of substeps and returns the next state in maximal and (optional) generalized coordinates [body_q_next, body_qd_next, joint_q_next, joint_qd_next].
        """
        if loss_fn is None:
            loss_fn = loss_l2
        actions.requires_grad = requires_grad
        state = self.model.state(requires_grad=requires_grad)
        if start_state is not None:
            state.body_q = wp.clone(start_state.body_q)
            state.body_qd = wp.clone(start_state.body_qd)
            state.body_f = wp.clone(start_state.body_f)
        rotation_count = rotation_count or wp.array(
            [0.0], dtype=float, requires_grad=requires_grad
        )
        episode_renderer = None

        if render:
            # set up Usd renderer
            stage_path = gen_unique_filename(osp.join(self.save_dir, config.stage_name))
            episode_renderer = None
            # wp.sim.render.SimRenderer(self.model, stage_path, scaling=100.)
            self.sim_time = 0.0
            self.render(state=state, renderer=episode_renderer)

        states = wp.zeros(
            len(self.ref_traj) * self.num_envs,
            dtype=wp.float32,
            device=self.model.device,
            requires_grad=requires_grad,
        )

        for i in range(len(self.ref_traj)):

            # simulate
            next_state, friction_out, body_f = self.simulate(
                state,
                actions,
                i * self.action_dim,
                rotation_count,
                requires_grad=requires_grad,
            )

            # save state
            wp.launch(
                save_state,
                dim=self.num_envs,
                inputs=[
                    self.compute_joint_q(
                        next_state, rotation_count, requires_grad=requires_grad
                    ),
                    # rotation_count,
                    i * self.state_dim,
                ],
                outputs=[states],
                device=self.model.device,
            )

            # update state
            old_state, state = state, next_state

            if render:
                self.sim_time += self.sim_dt * self.sim_substeps
                self.render(
                    state=state,
                    renderer=episode_renderer,
                )

            if log:
                self.log_values(
                    next_state,
                    friction_out.numpy(),
                    rotation_count.numpy()[0],
                    body_f.numpy(),
                )

        # compute loss
        if loss is None:
            loss = wp.zeros(
                1,
                dtype=wp.float32,
                device=self.model.device,
                requires_grad=requires_grad,
            )

        wp.launch(
            loss_fn,
            dim=(self.num_envs * len(self.ref_traj)),
            inputs=[
                states,
                self.ref_traj,
                self.state_dim,
                state.body_qd,
                actions,
            ],
            outputs=[loss],
            device=self.model.device,
        )

        return states, next_state

    def run(
        self,
        num_iter=250,
        start_state=None,
        start_action=None,
        start_rotations=0.0,
        check_backward=False,
        use_linf=False,
    ):
        loss_fn = loss_l2 if not use_linf else loss_linf

        # optimization variables
        state_err = wp.array(
            [0.0], dtype=float, device=self.model.device, requires_grad=True
        )
        if start_action is not None:
            actions = wp.array(
                start_action[: len(self.ref_traj) * self.action_dim * self.num_envs],
                dtype=float,
                device=self.model.device,
                requires_grad=True,
            )
        else:
            actions = wp.array(
                np.array([x for x in self.ref_traj.numpy()] * self.num_envs),
                dtype=wp.float32,
                device=self.model.device,
                requires_grad=True,
            )

        self.model.joint_q.requires_grad = True
        optimizer = wpu.AdamStepLR([actions], **config.opt_config.__dict__)

        progress = trange(num_iter, desc="Optimizing")
        target_q = self.ref_traj.numpy()[-1]
        opt_states = []  # list of states at each iteration

        for i in progress:

            rotation_count = wp.array(
                [start_rotations], dtype=float, requires_grad=False
            )
            tape = wp.Tape()
            with tape:
                states, state = self.forward(
                    actions,
                    start_state,
                    rotation_count,
                    requires_grad=True,
                    loss=state_err,
                    loss_fn=loss_fn,
                )
            # pdb.set_trace()

            opt_states.append(states.numpy())

            if check_backward:
                check_kernel_jacobian(
                    loss_fn,
                    1,
                    inputs=[states, self.ref_traj, state.body_qd, actions],
                    plot_jac_on_fail=True,
                    atol=1e-4,
                    rtol=1e-6,
                    outputs=[state_err],
                )

            loss = state_err.numpy()[0]
            rotations = int(rotation_count.numpy()[0])
            joint_q = self.compute_joint_q(state, rotation_count).numpy()[0]
            progress.set_description(
                f"loss: {loss:.3f}, rotations: {rotations}, joint_q: {joint_q:.3f}, target: {target_q:.3f}, act norm: {np.linalg.norm(actions.numpy()):.3f}"
            )
            self.log["losses"].append(state_err.numpy()[0])
            tape.backward(loss=state_err)
            # print("action grad", actions.grad.numpy())
            assert not np.isnan(actions.grad.numpy()).any(), "NaN in gradient"
            action_grad = np.abs(actions.grad.numpy())
            optimizer.step([actions.grad])
            tape.zero()
            state_err.zero_()
            if loss < 5e-3:
                print(f"Early stopping at {i} iters, loss: {loss:.3f}")
                break

        if config.plot and len(opt_states) > 1:
            plotting.create_animation(
                np.array(opt_states),
                self.ref_traj.numpy(),
                self.save_dir,
                "opt_states.gif",
                xlabel="timesteps",
                ylabel="joint q",
                title="Agent-agnostic trajopt",
                ylim=(-1, 8),
            )

        self.log["|dL/da_t|"] += list(action_grad)
        return actions, state

    def integrate_step(self, state_0, state_1, actions, action_index, rotation_count):
        """Computes body forces, integrates bodies, and swaps in/out state"""
        # Computes state.body_f from joint positions/act
        if actions is not None:
            wp.launch(
                kernel=wpu.assign_body_f_torque,
                inputs=[actions, action_index, self.joint_axis_idx],
                outputs=[state_0.body_f],
                dim=1,
            )
        wp.launch(
            kernel=wpu.eval_body_joints,
            dim=1,
            inputs=[
                state_0.body_q,
                state_0.body_qd,
                self.model.body_com,
                self.model.joint_q_start,
                self.model.joint_qd_start,
                self.model.joint_type,
                self.model.joint_parent,
                self.model.joint_X_p,
                self.model.joint_X_c,
                self.model.joint_axis,
                self.model.joint_target,
                self.model.joint_act,
                self.model.joint_target_ke,
                self.model.joint_target_kd,
                self.model.joint_limit_lower,
                self.model.joint_limit_upper,
                self.model.joint_limit_ke,
                self.model.joint_limit_kd,
                self.model.joint_attach_ke,
                self.model.joint_attach_kd,
                rotation_count.numpy()[0],
                self.joint_idx,
            ],
            outputs=[state_0.body_f],
            device=self.model.device,
        )
        # Integrates body position/velocity, updates state_out
        wp.launch(
            kernel=integrate_bodies,
            dim=self.model.body_count,
            inputs=[
                state_0.body_q,
                state_0.body_qd,
                state_0.body_f,
                self.model.body_com,
                self.model.body_mass,
                self.model.body_inertia,
                self.model.body_inv_mass,
                self.model.body_inv_inertia,
                self.model.gravity,
                0.05,  # euler integrator angular damping term
                self.sim_dt,
            ],
            outputs=[state_1.body_q, state_1.body_qd],
            device=self.model.device,
        )
        return state_1

    def compute_joint_q(
        self, state, rotation_count=None, requires_grad=False, joint_q=None
    ):
        if joint_q is None:
            joint_q = wp.array(
                np.zeros(self.num_envs),
                dtype=float,
                requires_grad=requires_grad,
            )
        if rotation_count is not None:
            rotation_count = rotation_count.numpy()[0]
        else:
            rotation_count = 0.0
        wp.launch(
            kernel=get_joint_q,
            inputs=[
                state.body_q,
                self.model.joint_type,
                self.model.joint_parent,
                self.model.joint_X_p,
                self.model.joint_axis,
                rotation_count,
            ],
            outputs=[joint_q],
            dim=self.model.joint_count,
        )
        # pdb.set_trace()
        # wp.launch(
        #     kernel=compute_obj_yaw,
        #     dim=1,
        #     inputs=[state.body_q, rotation_count],
        #     outputs=[joint_q],
        # )
        return joint_q

    def render(self, state=None, renderer=None, is_live=False):
        renderer = renderer or self.renderer
        with wp.ScopedTimer("render", active=False):
            time = 0.0 if is_live else self.sim_time

            renderer.begin_frame(time)
            renderer.render(state)
            obj_Xform = state.body_q.numpy()[-1]
            obj_pos, obj_rot = obj_Xform[:3], obj_Xform[-4:]
            start_pt = wp.vec3(*obj_pos)
            quat = R.from_euler(
                "xyz", [0, self.target_theta * np.pi, 0], degrees=False
            ).as_quat()
            plotting.render_line(
                renderer,
                start_pt,
                quat,
                length=1.0,
                color=(1.0, 0.0, 0.0),
                name="target_angle",
            )
            plotting.render_line(
                renderer,
                start_pt,
                wp.quat(*obj_rot),
                length=1.0,
                color=(0.0, 1.0, 0.0),
                name="curr_angle",
            )
            renderer.end_frame()


def main(config):
    env = ScrewTightenEnv(config)

    if config.optimize:
        ref_traj = env.ref_traj.numpy()
        if config.optimize_steps:
            start_action, start_state = None, None
            rotation_count = wp.array([0.0], dtype=float)
            actions, states = [], []
            for i in range(math.ceil(env.sim_frames / config.optimize_steps)):
                # slice reference trajectory starting from current state
                env.ref_traj = wp.array(
                    ref_traj[
                        config.optimize_steps * i : config.optimize_steps * (i + 1)
                    ],
                    requires_grad=False,
                    device=env.model.device,
                )
                # optimize actions from current state, w.o. reference to rotation count
                start_rotations = rotation_count.numpy()[0]
                ac, _ = env.run(
                    num_iter=config.num_iter,
                    start_state=start_state,
                    start_action=start_action,
                    start_rotations=start_rotations,
                    use_linf=True,
                )
                actions.append(ac.numpy())
                # compute loss for subtrajectory from current state
                loss = wp.array([0.0], dtype=float, device=env.model.device)
                # set rotations to what it is at end of previous subtrajectory
                s, start_state = env.forward(
                    ac, start_state, rotation_count, loss=loss, log=True
                )
                states.append(s.numpy())
                start_action = ac.numpy()
                print(f"step {i} loss {loss.numpy()[0]:.3f}")
                valid_log_keys = [k for k in env.log if len(env.log[k]) > 0]
                if config.save_log:
                    log_data = {k: np.array(env.log[k]) for k in valid_log_keys}
                    safe_save(f"{env.save_dir}/log_data.npz", log_data)
                plotting.plot_logs(env.log, env.save_dir)
                for k in valid_log_keys:
                    env.log[k] = []

            best_actions = np.concatenate(actions)
            best_states = np.concatenate(states)
        else:
            start_action = None
            if config.debug_actions:
                start_action = np.load(config.debug_actions)
                if start_action.shape[1] == 2:  # loaded goal_traj
                    start_action = start_action[:, 0]
            best_actions, _ = env.run(
                num_iter=config.num_iter, start_action=start_action
            )
            best_states, _ = env.forward(best_actions, render=config.render, log=True)
            best_states = best_states.numpy()
            best_actions = best_actions.numpy()
        goal_traj = np.stack([best_actions, best_states], axis=1)
        np.save(f"{env.save_dir}/goal_traj.npy", goal_traj)
        if config.plot:
            plotting.plot_states(env.save_dir, ref_traj, best_states, color="green")
    elif config.debug_actions is not None and osp.exists(config.debug_actions):
        from warp.tests.grad_utils import check_kernel_jacobian

        actions = wp.array(
            np.load(config.debug_actions),
            dtype=float,
            device=env.model.device,
            requires_grad=True,
        )
        assert (
            len(actions) == env.sim_frames
        ), "actions must be same length as sim_frames"
        states, state = env.forward(actions, requires_grad=True, log=True, render=True)
        loss = wp.array([0.0], dtype=float, device=env.model.device, requires_grad=True)
        check_kernel_jacobian(
            loss_l2,
            1,
            inputs=[states, env.ref_traj, env.state_dim, state.body_qd, actions],
            outputs=[loss],
        )
    else:  # just simulate a trajectory with fixed action in config.action
        action = config.action
        rotation_count = wp.array([0.0], dtype=float, device=env.model.device)
        for i in range(env.sim_frames):
            env.update(action, rotation_count, True)
            env.render(env.state_0)
        if config.plot:
            plotting.plot_logs(env.log, env.save_dir)

    if config.render:
        env.renderer.save()


if __name__ == "__main__":
    config = tyro.cli(ScrewTightenConfig)
    main(config)
