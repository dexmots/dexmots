import pdb

import warp as wp
import warp.sim
import torch
import numpy as np
from typing import Union
from warp.sim.model import State, Model
from inspect import getmembers
from dmanip.utils import ActionType
from dmanip.utils import warp_utils as wpu
from dmanip.utils.common import POSITION_GOAL_TYPES, ORIENTATION_GOAL_TYPES

from torch.cuda.amp import custom_fwd, custom_bwd

# from warp.tests.grad_utils import check_kernel_jacobian, check_backward_pass

def clear_grads(obj: Union[State, Model], filter=None):
    for k, v in getmembers(obj):
        if filter is not None and not filter(k):
            continue
        if isinstance(v, wp.array) and v.requires_grad and v.grad is not None:
            v.grad.zero_()
    return obj


clear_array_dtypes = {wp.float32, wp.int32, wp.vec3}


def clear_arrays(obj: Model, filter=None):
    for k, v in getmembers(obj):
        if filter is not None and not filter(k):
            continue
        if isinstance(v, wp.array) and v.dtype in clear_array_dtypes:
            if v.grad:
                v.grad.zero_()
            v.zero_()
    return obj



def get_goal_state(goal_state, goal_type, reward_coefs):
    # TODO: select goals based on: goal_idx, timestep, closeset goal, ...
    # coefs scale components of reward: act penalty, goal dist,
    goal_pos = np.zeros(2)
    goal_ori = 0.0
    goal_force = np.zeros(2)
    goal_torque = 0.0
    if "trajectory" in goal_type.name.lower():
        if "orientation" in goal_type.name.lower():
            if "torque" in goal_type.name.lower():
                goal_ori, goal_torque = goal_state[0], goal_state[1]
            else:
                goal_ori = goal_state[0]
        elif "position" in goal_type.name.lower():  # position goaltype
            if "force" in goal_type.name.lower():
                goal_pos, goal_force = goal_state[:2], goal_state[2:]
            else:
                goal_pos = goal_state
    else:
        reward_coefs[4] = 0.0  # no trajectory, so no force/torque reward
        goal_pos, goal_ori = goal_state[:2], goal_state[2]
        if "position" in goal_type.name.lower():
            reward_coefs[2] = 0.0  # no orientation, so no orientation reward
        elif "orientation" in goal_type.name.lower():
            reward_coefs[3] = 0.0  # no position, so no position reward

    goal_pos = wp.vec3(goal_pos[0], 0.15, goal_pos[1])  # y-up
    goal_ori = wp.array([goal_ori], dtype=wp.float32)
    goal_force = wp.vec3(goal_force[0], 0.0, goal_force[1])  # y-up
    goal_torque = wp.vec3(0.0, goal_torque, 0.0)  # torque around y-axis
    reward_coefs = wp.array(reward_coefs, dtype=float, requires_grad=False)

    return goal_pos, goal_ori, goal_force, goal_torque, reward_coefs


def get_body_idx(model, body_name):
    bid = []
    for i, name in enumerate(model.body_name):
        if name == body_name:
            bid.append(i)
    bid = wp.array(bid, dtype=int, device=model.device)
    return bid


def compute_obs_reward(ctx, forward_only=False, compute_reward=False):
    """Computes observation and reward by running integrator.simulate(), and compute_goal_reward kernel"""
    model = ctx.model
    joint_act = ctx.joint_act
    num_envs = ctx.oid.size
    num_acts = ctx.act.size // num_envs
    num_joints = ctx.num_joints

    wpu.assign_act(
        ctx.act,
        joint_act,
        model.joint_target_ke,
        ctx.action_type,
        num_acts=num_acts,
        num_envs=num_envs,
        num_joints=num_joints,
        q_offset=ctx.q_offset,
    )

    if compute_reward:
        num_envs = ctx.act_params["num_envs"]
        num_actions = ctx.act_params["num_acts"]
        if ctx.act_params['action_type'] is ActionType.TORQUE:
            wp.launch(
                wpu.compute_ctrl_reward,
                dim=(num_envs, num_actions),
                device=ctx.model.device,
                inputs=[ctx.act, num_actions],
                outputs=[ctx.reward_ctrl],
            )
        elif ctx.act_params['action_type'] is ActionType.POSITION:
            joint_indices = ctx.joint_indices
            wp.launch(
                wpu.compute_ctrl_pos_reward,
                dim=(num_envs, num_actions),
                device=ctx.model.device,
                inputs=[ctx.act, ctx.joint_q_end, ctx.joint_target_indices, num_actions],
                outputs=[ctx.reward_ctrl],
            )

    state_temp = ctx.state_in

    for step in range(ctx.substeps):
        state_in = state_temp
        state_in.clear_forces()
        if ctx.state_list is not None and step < len(ctx.state_list) and not forward_only:
            state_temp = ctx.state_list[step]
            # model.allocate_rigid_contacts(requires_grad=True)
        elif step < ctx.substeps - 1:
            state_temp = model.state()
            # model.allocate_rigid_contacts(requires_grad=True)
        else:
            # if last step or forward, set to state_out
            state_temp = ctx.state_out

        wp.sim.collide(model, state_in)
        state_temp = ctx.integrator.simulate(
            model,
            state_in,
            state_temp,
            ctx.dt / float(ctx.substeps),
        )

    if isinstance(ctx.integrator, wp.sim.SemiImplicitIntegrator):
        ctx.body_f.assign(
            state_in.body_f
        )  # takes instantaneous force from last substep
    else:
        # captures applied joint torques
        ctx.body_f.assign(state_in.body_f)
        wpu.integrate_body_f(
            ctx.model,
            ctx.state_in.body_qd,
            ctx.state_out.body_q,
            ctx.state_out.body_qd,
            ctx.body_f,
            ctx.dt,
        )
    # Compute reward
    wp.sim.eval_ik(model, ctx.state_out, ctx.joint_q, ctx.joint_qd)

    if compute_reward:
        wp.launch(
            kernel=wpu.compute_goal_reward,
            inputs=[
                ctx.state_out.body_q,
                ctx.body_f,
                ctx.joint_q,
                ctx.fid,
                ctx.num_fingers,
                ctx.oid,
                ctx.oqid,
                ctx.offset,
                ctx.object_thickness,
                ctx.base_id,
                ctx.goal_pos,
                ctx.goal_ori,
                ctx.goal_force,
                ctx.goal_torque,
                ctx.goal_type.value,
                ctx.reward_type.value,
                ctx.reward_ctrl,
                ctx.reward_coefs,
                ctx.rotation_count,
            ],
            outputs=[ctx.reward_vec, ctx.reward_vec_prev, ctx.reward_total],
            dim=num_envs,
            device=model.device,
        )
        wp.launch(
            kernel=wpu.joint_limit_penalty,
            inputs=[
                ctx.state_out.body_q,
                model.joint_qd_start,
                model.joint_type,
                model.joint_parent,
                model.joint_X_p,
                model.joint_axis,
                model.joint_limit_lower,
                model.joint_limit_upper,
                num_joints,
            ],
            outputs=[ctx.reward_total],
            dim=model.joint_count,
            device=model.device,
        )

def get_compute_graph(func, kwargs={}, tape=None, grads={}, back_grads={}):
    if tape is not None:
        wp.capture_begin()
        with tape:
            func(**kwargs)
        # copy grads from temp buffer from graph_capture_params to dest var
        for bw_temp, adj_output in grads.items():
            adj_output.assign(bw_temp)
        tape.backward()
        # copy grads back to temp buffer so tape can be zeroed in graph capture
        for bw_temp, adj_input in back_grads.items():
            bw_temp.assign(tape.gradients[adj_input])
        tape.zero()
        graph = wp.capture_end()
        return graph
    wp.capture_begin()
    func(**kwargs)
    return wp.capture_end()




class ComputeStateAndReward(torch.autograd.Function):
    forward_graph = None

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(
        ctx,
        model,
        integrator,
        dt,
        substeps,
        joint_act,  # warp array to store joint_action
        act_warp,
        body_f_warp,
        joint_q_warp,
        joint_qd_warp,
        joint_target_indices,
        act,
        body_q,
        body_qd,
        joint_q,
        joint_qd,
        action_type,
        goal_type,
        goal_state_vars,
        reward_type,
        reward_ctrl,
        reward_total,
        reward_vec,
        reward_vec_prev,
        state_in,
        state_out,
        state_list,
        q_offset,
        num_joints,
        capture_graph,
        compute_reward
    ):
        ctx.model = model
        ctx.joint_target_indices = joint_target_indices
        ctx.act = act_warp
        ctx.act_pt = act
        ctx.act.assign(wp.from_torch(ctx.act_pt.detach()))

        # warp array to store joint_action, either joint target or joint_torque
        ctx.joint_act = joint_act
        ctx.joint_q = joint_q_warp
        ctx.joint_q.assign(wp.from_torch(joint_q))
        ctx.joint_qd = joint_qd_warp
        ctx.joint_qd.assign(wp.from_torch(joint_qd))

        ctx.body_q_pt = body_q
        ctx.body_qd_pt = body_qd
        ctx.body_f = body_f_warp
        ctx.state_in = state_in
        ctx.state_out = state_out
        ctx.state_list = state_list
        # ctx.state_in.body_q.assign(wp.from_torch(ctx.body_q_pt, dtype=wp.transform))
        # ctx.state_in.body_qd.assign(
        #     wp.from_torch(ctx.body_qd_pt, dtype=wp.spatial_vector)
        # )

        ctx.q_offset = q_offset
        ctx.num_joints = num_joints

        # get indices for bodies and joints
        ctx.fid = goal_state_vars["fid"]
        ctx.oid = goal_state_vars["oid"]
        ctx.oqid = goal_state_vars["oqid"]
        ctx.base_id = goal_state_vars["base_id"]
        ctx.offset = goal_state_vars["offset"]
        ctx.object_thickness = goal_state_vars["object_thickness"]
        ctx.num_fingers = goal_state_vars["num_fingers"]

        ctx.goal_pos = goal_state_vars["goal_pos"]
        ctx.goal_ori = goal_state_vars["goal_ori"]
        ctx.goal_force = goal_state_vars["goal_force"]
        ctx.goal_torque = goal_state_vars["goal_torque"]
        ctx.reward_coefs = goal_state_vars["reward_coefs"]
        ctx.rotation_count = goal_state_vars["rotation_count"]
        ctx.compute_reward = compute_reward
        # ctx.start_joint_q = goal_state_vars["start_joint_q"]
        # ctx.start_joint_qd = goal_state_vars["start_joint_qd"]

        ctx.goal_type = goal_type
        ctx.reward_type = reward_type
        ctx.action_type = action_type
        ctx.reward_ctrl = reward_ctrl
        ctx.reward_total = reward_total
        ctx.reward_vec = reward_vec
        ctx.reward_vec_prev = reward_vec_prev
        ctx.reward_vec_prev_pt = wp.to_torch(wp.clone(reward_vec_prev))
        ctx.capture_graph = capture_graph
        ctx.integrator = integrator
        ctx.dt = dt
        ctx.substeps = substeps

        # record gradients for act, joint_q, and joint_qd
        assert ctx.act.requires_grad
        # assert ctx.body_q.requires_grad
        # assert ctx.body_qd.requires_grad
        # ctx.joint_stiffness = model.joint_target_ke

        assert ctx.state_in.body_q.requires_grad
        assert ctx.state_in.body_qd.requires_grad
        assert ctx.state_in.body_f.requires_grad
        assert ctx.state_out.body_q.requires_grad
        assert ctx.state_out.body_qd.requires_grad
        assert ctx.state_out.body_f.requires_grad

        if ComputeStateAndReward.forward_graph is None and capture_graph:
            wp.capture_begin()
            compute_obs_reward(ctx, forward_only=True, compute_reward=ctx.compute_reward)
            ComputeStateAndReward.forward_graph = wp.capture_end()

        if capture_graph:
            ctx.tape = None
            wp.capture_launch(ComputeStateAndReward.forward_graph)
        else:
            ctx.tape = wp.Tape()
            with ctx.tape:
                compute_obs_reward(ctx, forward_only=False, compute_reward=ctx.compute_reward)

        # check_kernel_jacobian(
        #     wpu.compute_goal_reward,
        #     ctx.oid.size,
        #     inputs=[
        #         ctx.state_out.body_q,
        #         ctx.body_f,
        #         ctx.joint_q,
        #         ctx.fid,
        #         ctx.num_fingers,
        #         ctx.oid,
        #         ctx.oqid,
        #         ctx.offset,
        #         ctx.object_thickness,
        #         ctx.base_id,
        #         ctx.goal_pos,
        #         ctx.goal_ori,
        #         ctx.goal_force,
        #         ctx.goal_torque,
        #         ctx.goal_type.value,
        #         ctx.reward_type.value,
        #         ctx.reward_ctrl,
        #         ctx.reward_coefs,
        #         ctx.rotation_count,
        #     ],
        #     outputs=[ctx.reward_vec, ctx.reward_vec_prev, ctx.reward_total],
        # )

        # check_backward_pass(
        #     ctx.tape,
        #     visualize_graph=False,
        #     track_inputs=[ctx.act, ctx.state_out.body_q, ctx.state_in.body_q],
        #     track_outputs=[ctx.reward_total],
        #     track_input_names=[
        #         "act",
        #         "body_q_next",
        #         "body_q",
        #     ],
        #     track_output_names=["reward"],
        # )
        # ctx.state_in.body_q.assign(ctx.state_out.body_q)
        # ctx.state_in.body_qd.assign(ctx.state_out.body_qd)
        body_q_end = wp.to_torch(ctx.state_out.body_q)
        body_qd_end = wp.to_torch(ctx.state_out.body_qd)
        # rew_buf = wp.to_torch(ctx.reward_total)
        body_f = wp.to_torch(ctx.body_f)
        joint_q_end = wp.to_torch(ctx.joint_q)
        joint_qd_end = wp.to_torch(ctx.joint_qd)
        # TODO: check gradients are valid in the forward pass ctx.tape.backward()
        return body_q_end, body_qd_end, body_f, joint_q_end, joint_qd_end

    @staticmethod
    @custom_bwd
    def backward(
        ctx,
        adj_body_q,
        adj_body_qd,
        # adj_reward_total,
        adj_body_f,
        adj_joint_q,
        adj_joint_qd,
    ):
        if ctx.capture_graph:
            ctx.tape = wp.Tape()
            ctx.reward_vec_prev.assign(wp.from_torch(ctx.reward_vec_prev_pt))
            ctx.state_in.body_q.assign(wp.from_torch(ctx.body_q_pt, dtype=wp.transform))
            ctx.state_in.body_qd.assign(
                wp.from_torch(ctx.body_qd_pt, dtype=wp.spatial_vector)
            )
            assert ctx.act.grad.numpy().sum() == 0
            assert ctx.state_in.body_q.grad.numpy().sum() == 0
            assert ctx.state_out.body_q.grad.numpy().sum() == 0
            assert ctx.reward_total.grad.numpy().sum() == 0
            # Do forward sim again, allocating rigid pairs and intermediate states
            ctx.act.zero_()
            ctx.reward_ctrl.zero_()
            ctx.reward_total.zero_()
            ctx.reward_vec.zero_()
            with ctx.tape:
                compute_obs_reward(ctx, forward_only=False)

        # map incoming Torch grads to our output variables
        ctx.state_out.body_q.grad = wp.from_torch(adj_body_q, dtype=wp.transform)
        ctx.state_out.body_qd.grad = wp.from_torch(adj_body_qd, dtype=wp.spatial_vector)
        # ctx.reward_total.grad = wp.from_torch(adj_reward_total)
        ctx.body_f.grad = wp.from_torch(adj_body_f, dtype=wp.spatial_vector)
        ctx.joint_q.grad.zero_()
        ctx.joint_qd.grad.zero_()
        ctx.joint_q.grad = wp.from_torch(adj_joint_q)
        ctx.joint_qd.grad = wp.from_torch(adj_joint_qd)

        if ctx.capture_graph:
            wp.capture_begin()
            ctx.tape.zero()
            ctx.tape.backward()
            ctx.backward_graph = wp.capture_end()
            wp.capture_launch(ctx.backward_graph)
        else:
            ctx.tape.backward()

        act_grad = wp.to_torch(ctx.tape.gradients[ctx.act]).clone()
        body_q_grad = wp.to_torch(ctx.tape.gradients[ctx.state_in.body_q]).clone()
        body_qd_grad = wp.to_torch(ctx.tape.gradients[ctx.state_in.body_qd]).clone()
        joint_q_grad = wp.to_torch(ctx.tape.gradients[ctx.joint_q]).clone()
        joint_qd_grad = wp.to_torch(ctx.tape.gradients[ctx.joint_qd]).clone()
        # if act_grad.abs().max() > 1.0 or act_grad.isnan().any():
        #     if act_grad.abs().max() > 1.0:
        #         print("Warning: large act_grad values", act_grad.abs().max())
        #     if act_grad.isnan().any():
        #         print("Warning: nan act_grad values")
        # assert False, "act_grad invalid: {}, #nans:{}".format(
        #     act_grad.abs().max(), act_grad.isnan().sum()
        # )
        act_grad = act_grad.clamp(-1e-2, 1e-2)
        body_qd_grad = body_qd_grad.clamp(-1, 1)
        body_q_grad = body_q_grad.clamp(-1, 1)
        joint_q_grad = joint_q_grad.clamp(-1, 1)
        joint_qd_grad = joint_qd_grad.clamp(-1, 1)
        ctx.tape.zero()    
        # return adjoint w.r.t. inputs

        return (
            None,  # model,
            None,  # integrator,
            None,  # dt,
            None,  # substeps,
            None,  # joint_act,  # warp array to store joint_action
            None,  # act_warp,  # warp array to store actions
            None,  # joint_q_warp
            None,  # joint_qd_warp
            None,  # body_f_warp,  # warp array to store body_f
            None,  # joint_target_indices
            act_grad,  # act,
            body_q_grad,  # body_q,
            body_qd_grad,  # body_qd,
            joint_q_grad,  # joint_q,
            joint_qd_grad,  # joint_qd,
            None,  # action_type,
            None,  # goal_state_vars,
            None,  # goal_type,
            None,  # reward_type,
            None,  # reward_ctrl,
            None,  # reward_total,
            None,  # reward_vec,
            None,  # reward_vec_prev,
            None,  # state_in,
            None,  # state_out,
            None,  # state_list
            None,  # q_offset
            None,  # num_joints
            None,  # capture_graph
            None,  # compute_reward
        )


forward_simulate = ComputeStateAndReward.apply
