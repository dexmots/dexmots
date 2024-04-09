import torch
from dmanip.utils.common import ActionType
import warp as wp
import numpy as np
from typing import Union
from inspect import getmembers
from warp.sim.model import State, Model
from torch.cuda.amp import custom_fwd, custom_bwd
from .warp_utils import compute_ctrl_pos_reward, compute_ctrl_reward, integrate_body_f, assign_act

from warp.tests.grad_utils import check_kernel_jacobian, check_backward_pass, plot_jacobian_comparison


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


zero_mat = wp.constant(wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))


@wp.kernel
def count_contact_copy(
    contact_count: wp.array(dtype=int),
    contact_count_copy: wp.array(dtype=int),
):
    i = wp.tid()
    contact_count_copy[i] = contact_count[i]


@wp.kernel
def assign_zero_kernel_float(arr: wp.array(dtype=float)):
    i = wp.tid()
    arr[i] = 0.0


@wp.kernel
def assign_zero_kernel_transform(arr: wp.array(dtype=wp.transform)):
    i = wp.tid()
    arr[i] = wp.transform(wp.vec3(), wp.quat())


@wp.kernel
def assign_zero_kernel_spatial_vector(arr: wp.array(dtype=wp.spatial_vector)):
    i = wp.tid()
    arr[i] = wp.spatial_vector(wp.vec3(), wp.vec3())


@wp.kernel
def assign_zero_kernel_vec3(arr: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    arr[i] = wp.vec3()


# @wp.kernel
# def assign_zero_kernel_mat33(arr: wp.array(dtype=wp.mat33)):
#     i = wp.tid()
#     arr[i] = arr[i] * zero_mat


assign_zero_kernel = {
    float: assign_zero_kernel_float,
    wp.float32: assign_zero_kernel_float,
    wp.transform: assign_zero_kernel_transform,
    wp.spatial_vector: assign_zero_kernel_spatial_vector,
    wp.vec3: assign_zero_kernel_vec3,
}


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


def build_assign_grads_map(adj_arrs, params, sim_params, act_params):
    """Builds a map of adjoint input/output variables to their corresponding pytorch gradient variables"""
    if sim_params.get("ag_return_body", False):
        adj_joint_q, adj_joint_qd, adj_body_q, adj_body_qd = adj_arrs
    else:
        adj_joint_q, adj_joint_qd = adj_arrs
        adj_body_q = adj_body_q = None

    grads = {}
    back_grads = {}
    if params["capture_graph"]:
        state_in = sim_params["bwd_state_in"]
        state_out = sim_params["bwd_state_out"]
    else:
        state_in = sim_params["state_in"]
        state_out = sim_params["state_out"]

    # assign adjoint outputs to temp vars, creating them if they don't exist
    if "adj_jq" not in params:
        params["adj_jq"] = wp.from_torch(adj_joint_q)
        params["adj_jqd"] = wp.from_torch(adj_joint_qd)
        params["adj_a"] = wp.zeros_like(act_params["act"].grad)
        params["adj_bqp"] = wp.zeros_like(state_in.body_q.grad)
        params["adj_bqdp"] = wp.zeros_like(state_in.body_qd.grad)
    else:
        params["adj_jq"].assign(wp.from_torch(adj_joint_q))
        params["adj_jqd"].assign(wp.from_torch(adj_joint_qd))
        params["adj_a"].zero_()
        params["adj_bqp"].zero_()
        params["adj_bqdp"].zero_()

    # map adjoint outputs to their corresponding gradient buffers
    grads[params["adj_jq"]] = params["joint_q_end"].grad
    grads[params["adj_jqd"]] = params["joint_qd_end"].grad
    back_grads[params["adj_a"]] = act_params["act"]
    back_grads[params["adj_bqp"]] = state_in.body_q
    back_grads[params["adj_bqdp"]] = state_in.body_qd

    # if body q is being returned, assign adjoint outputs to temp vars, creating them if they don't exist
    if sim_params.get("ag_return_body", False):
        if "adj_bq" not in params:
            params["adj_bq"] = wp.from_torch(adj_body_q, dtype=wp.transform)
            params["adj_bqd"] = wp.from_torch(adj_body_qd, dtype=wp.spatial_vector)
        else:
            params["adj_bq"].assign(wp.from_torch(adj_body_q, dtype=wp.transform))
            params["adj_bqd"].assign(wp.from_torch(adj_body_qd, dtype=wp.spatial_vector))
        grads[params["adj_bq"]] = state_out.body_q.grad
        grads[params["adj_bqd"]] = state_out.body_qd.grad

    return grads, back_grads


def forward_simulate(ctx, forward=False, requires_grad=False):
    joint_q_end = ctx.graph_capture_params["joint_q_end"]
    joint_qd_end = ctx.graph_capture_params["joint_qd_end"]
    if forward or not ctx.capture_graph:
        model = ctx.model
        state_temp = ctx.state_in
        state_out = ctx.state_out
        joint_act = ctx.joint_act
    else:
        model = ctx.backward_model
        state_temp = ctx.bwd_state_in
        state_out = ctx.bwd_state_out
        joint_act = ctx.bwd_joint_act

    num_envs = ctx.act_params["num_envs"]
    num_acts = ctx.act_params["num_acts"]  # ctx.act.size // num_envs
    action_type = ctx.act_params["action_type"]
    joint_stiffness = ctx.act_params["joint_stiffness"]
    joint_indices = ctx.act_params["joint_target_indices"]

    if action_type != ActionType.POSITION_DELTA:
        joint_act.zero_()
    assign_act(ctx.act, joint_act, joint_stiffness, action_type, num_acts, num_envs, joint_indices=joint_indices)
    for step in range(ctx.substeps):
        state_in = state_temp
        state_in.clear_forces()
        if forward or step == ctx.substeps - 1:
            # reuse state_out for last substep
            state_temp = state_out
        elif not ctx.capture_graph and ctx.state_list is None:
            state_temp = model.state()
        else:
            state_temp = ctx.state_list[step]
        if model.ground:
            # if not forward and step == 0:
            #     model.allocate_rigid_contacts()
            wp.sim.collide(model, state_in)
        state_temp = ctx.integrator.simulate(
            model,
            state_in,
            state_temp,
            ctx.dt,
        )
    if isinstance(ctx.integrator, wp.sim.SemiImplicitIntegrator):
        ctx.body_f.assign(state_in.body_f)  # takes instantaneous force from last substep
    else:
        # captures applied joint torques
        ctx.body_f.assign(state_in.body_f)
        integrate_body_f(
            ctx.model,
            ctx.state_in.body_qd,
            ctx.state_out.body_q,
            ctx.state_out.body_qd,
            ctx.body_f,
            ctx.dt * ctx.substeps,
        )
    wp.sim.eval_ik(model, state_out, joint_q_end, joint_qd_end)


class IntegratorSimulate(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(
        ctx,
        simulate_params,
        graph_capture_params,
        act_params,
        action,
        body_q,
        body_qd,
    ):
        ctx.model = simulate_params["model"]
        ctx.backward_model = graph_capture_params["bwd_model"]
        ctx.integrator = simulate_params["integrator"]
        ctx.dt, ctx.substeps = simulate_params["dt"], simulate_params["substeps"]
        ctx.state_in = simulate_params["state_in"]
        ctx.state_out = simulate_params["state_out"]
        ctx.state_list = simulate_params.get("state_list", None)
        ctx.return_body = simulate_params.get("ag_return_body", False)
        ctx.body_f = simulate_params["body_f"]
        ctx.act = act_params["act"]
        ctx.joint_act = act_params["joint_act"]
        ctx.act_pt = action
        ctx.act.assign(wp.from_torch(ctx.act_pt.flatten()))
        ctx.body_q_pt = body_q.clone()
        ctx.body_qd_pt = body_qd.clone()
        ctx.joint_q_end = graph_capture_params["joint_q_end"]
        ctx.joint_qd_end = graph_capture_params["joint_qd_end"]
        ctx.capture_graph = graph_capture_params["capture_graph"]

        ctx.act_params = act_params
        ctx.simulate_params = simulate_params
        ctx.graph_capture_params = graph_capture_params

        # if using graph capture, need to assign states as in/out buffers cannot be swapped
        if ctx.capture_graph:
            ctx.state_in.body_q.assign(wp.from_torch(ctx.body_q_pt, dtype=wp.transform))
            ctx.state_in.body_qd.assign(wp.from_torch(ctx.body_qd_pt, dtype=wp.spatial_vector))

        # record gradients for act, joint_q, and joint_qd
        ctx.act.requires_grad = True
        ctx.joint_q_end.requires_grad = True
        ctx.joint_qd_end.requires_grad = True
        ctx.state_in.body_q.requires_grad = True
        ctx.state_in.body_qd.requires_grad = True
        ctx.state_out.body_q.requires_grad = True
        ctx.state_out.body_qd.requires_grad = True

        if ctx.capture_graph:
            ctx.tape = graph_capture_params["tape"]
            ctx.bwd_joint_act = act_params["bwd_joint_act"]
            ctx.bwd_state_in = simulate_params["bwd_state_in"]
            ctx.bwd_state_out = simulate_params["bwd_state_out"]
            if "forward_graph" not in graph_capture_params:
                graph_capture_params["forward_graph"] = get_compute_graph(
                    forward_simulate, {"ctx": ctx, "forward": True}
                )
            ctx.forward_graph = graph_capture_params["forward_graph"]
            wp.capture_launch(ctx.forward_graph)
        else:
            ctx.tape = wp.Tape()
            with ctx.tape:
                forward_simulate(ctx, forward=False, requires_grad=True)
            # ctx.tape.backward()
            # for name, grad in [("act", ctx.act.grad), ("body_q_in", ctx.state_in.body_q.grad)]:
            #     if np.abs(grad.numpy()).max() > 1e3:
            #         print("name", name, "grad max", np.abs(grad.numpy()).max())
            # check_backward_pass(
            #     ctx.tape,
            #     track_input_names=["body_q_in", "action"],
            #     track_output_names=["joint_q_out", "body_q_out"],
            #     track_inputs=[ctx.state_in.body_q, ctx.joint_act],
            #     track_outputs=[ctx.joint_q_end, ctx.state_out.body_q],
            #     visualize_graph=True,
            #     check_kernel_jacobians=False,
            #     # check_input_output_jacobian=False,
            #     # plot_jac_on_fail=True,
            # )

        joint_q_end = wp.to_torch(ctx.graph_capture_params["joint_q_end"]).clone()
        joint_qd_end = wp.to_torch(ctx.graph_capture_params["joint_qd_end"]).clone()
        if ctx.return_body:
            body_q, body_qd = (
                wp.to_torch(ctx.state_out.body_q).clone(),
                wp.to_torch(ctx.state_out.body_qd).clone(),
            )
            return (joint_q_end, joint_qd_end, body_q, body_qd)
        else:
            return (joint_q_end, joint_qd_end)

    @staticmethod
    @custom_bwd
    def backward(ctx, *adj_arrs):
        grads, back_grads = build_assign_grads_map(
            adj_arrs, ctx.graph_capture_params, ctx.simulate_params, ctx.act_params
        )
        # map incoming Torch grads to our output variables
        state_in = ctx.state_in
        state_out = ctx.state_out
        act = ctx.act
        if ctx.capture_graph:
            state_in = ctx.bwd_state_in
            state_out = ctx.bwd_state_out
            state_in.body_q.assign(wp.from_torch(ctx.body_q_pt, dtype=wp.transform))
            state_in.body_qd.assign(wp.from_torch(ctx.body_qd_pt, dtype=wp.spatial_vector))
            act.zero_()
            act.grad.zero_()
            act.assign(wp.from_torch(ctx.act_pt.detach()))
            assert act.grad.numpy().sum() == 0
            assert state_in.body_q.grad.numpy().sum() == 0
            assert state_out.body_q.grad.numpy().sum() == 0
            # Do forward sim again, allocating rigid pairs and intermediate states
            if "bwd_forward_graph" not in ctx.graph_capture_params:
                ctx.graph_capture_params["bwd_forward_graph"] = get_compute_graph(
                    forward_simulate,
                    {"ctx": ctx, "requires_grad": True, "forward": False},
                    ctx.tape,
                    grads,
                    back_grads,
                )
            # else:
            ctx.bwd_forward_graph = ctx.graph_capture_params["bwd_forward_graph"]
            wp.capture_launch(ctx.bwd_forward_graph)
            for s in ctx.simulate_params.get("state_list", []):
                clear_grads(s)
            clear_grads(state_in)
            clear_grads(state_out)
            clear_grads(ctx.model)
            clear_grads(ctx.backward_model)
            (
                ctx.graph_capture_params["bwd_state_in"],
                ctx.graph_capture_params["bwd_state_out"],
            ) = (state_out, state_in)

            # with tape:  # check if graph capture works for this
            #     forward_simulate(ctx, forward=False, requires_grad=True)
            joint_act_grad = wp.to_torch(ctx.graph_capture_params["adj_a"]).clone()
            # if joint_act_grad.max() > 1.0:
            #     __import__("ipdb").set_trace()
            body_q_grad = wp.to_torch(ctx.graph_capture_params["adj_bqp"]).clone()
            body_qd_grad = wp.to_torch(ctx.graph_capture_params["adj_bqdp"]).clone()
            ctx.tape.zero()
            for s in ctx.simulate_params.get("state_list", []):
                clear_grads(s)
            clear_grads(ctx.simulate_params.get("state_in"))
            clear_grads(ctx.simulate_params.get("state_out"))
            clear_grads(ctx.simulate_params.get("bwd_state_in"))
            clear_grads(ctx.simulate_params.get("bwd_state_out"))
            clear_grads(ctx.graph_capture_params.get("bwd_model"))
            ctx.joint_q_end.grad.zero_()
            ctx.joint_qd_end.grad.zero_()
        else:
            if ctx.return_body:
                adj_joint_q, adj_joint_qd, adj_body_q, adj_body_qd = adj_arrs
                state_out.body_q.grad.assign(wp.from_torch(adj_body_q, dtype=wp.transform))
                state_out.body_qd.grad.assign(wp.from_torch(adj_body_qd, dtype=wp.spatial_vector))
            else:
                adj_joint_q, adj_joint_qd = adj_arrs
                adj_body_q, adj_body_qd = None, None

            ctx.graph_capture_params["joint_q_end"].grad.assign(wp.from_torch(adj_joint_q))
            ctx.graph_capture_params["joint_qd_end"].grad.assign(wp.from_torch(adj_joint_qd))

            ctx.tape.backward()
            joint_act_grad = wp.to_torch(ctx.tape.gradients[act]).clone()
            if ctx.simulate_params.get("jacobian_norm", None) is not None:
                jac_norm = joint_act_grad.norm(dim=-1, keepdim=True)
                jac_norm = torch.where(
                    jac_norm > ctx.simulate_params["jacobian_norm"],
                    jac_norm.clamp(min=1e-6),
                    torch.ones_like(jac_norm) * ctx.simulate_params["jacobian_norm"],
                )
                joint_act_grad /= jac_norm
            # Unnecessary copying of grads, grads should already be recorded by context
            body_q_grad = wp.to_torch(ctx.tape.gradients[state_in.body_q]).clone()
            body_qd_grad = wp.to_torch(ctx.tape.gradients[state_in.body_qd]).clone()
            ctx.tape.zero()

        # return adjoint w.r.t. inputs
        return (
            None,  # simulate_params,
            None,  # graph_capture_params,
            None,  # act_params,
            joint_act_grad,  # action,
            body_q_grad,  # body_q,
            body_qd_grad,  # body_qd,
        )


forward_ag = IntegratorSimulate.apply
