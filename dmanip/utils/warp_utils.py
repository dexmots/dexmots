"""Warp helper functions"""

import numpy as np
import torch
import warp as wp
import warp.sim
import warp.sim.render

from scipy.spatial.transform import Rotation
from warp.optim import Adam
from warp.sim.integrator_euler import eval_joint_force
from warp.sim.utils import quat_decompose, quat_twist

from warp.sim.integrator_euler import (
    compute_forces,
    integrate_bodies,
    integrate_particles,
    SemiImplicitIntegrator,
)

######### Warp Constants

PI = wp.constant(np.pi)
GOAL_POS = wp.constant(0)
GOAL_ORI = wp.constant(1)
GOAL_POSE = wp.constant(2)
GOAL_POS_TRAJ = wp.constant(3)
GOAL_ORI_TRAJ = wp.constant(4)
GOAL_POS_FORCE = wp.constant(5)
GOAL_ORI_TORQUE = wp.constant(6)
GOAL_POS_WRENCH = wp.constant(7)

ACTION_TORQUE = wp.constant(0)
ACTION_POSITION = wp.constant(1)
ACTION_JOINT_STIFFNESS = wp.constant(2)

JOINT_REVOLUTE_SPRING = wp.constant(7)
JOINT_REVOLUTE_TIGHT = wp.constant(8)

REWARD_DELTA = wp.constant(0)
REWARD_EXP = wp.constant(1)
REWARD_L2 = wp.constant(2)

######### Warp Functions


@wp.func
def quat_ang_err(q1: wp.quat, q2: wp.quat):
    """Calculate angular velocity between two quaternions"""
    q1_inv = wp.quat_inverse(q1)
    q2_rel = q1_inv * q2
    r_err = wp.normalize(q2_rel)
    ang_err = wp.normalize(wp.vec3(r_err[0], r_err[1], r_err[2])) * wp.acos(r_err[3]) * 2.0
    return wp.abs(ang_err[0]) + wp.abs(ang_err[1]) + wp.abs(ang_err[2])


@wp.func
def quat_ang_vel(q1: wp.quat, q2: wp.quat, dt: float):
    return (2.0 / dt) * wp.vec3(
        q1[0] * q2[1] - q1[1] * q2[0] - q1[2] * q2[3] + q1[3] * q2[2],
        q1[0] * q2[2] + q1[1] * q2[3] - q1[2] * q2[0] - q1[3] * q2[1],
        q1[0] * q2[3] - q1[1] * q2[2] + q1[2] * q2[1] - q1[3] * q2[0],
    )


@wp.func
def eval_joint_torque(
    q: float,
    qd: float,
    target: float,
    target_ke: float,
    target_kd: float,
    act: float,
    limit_lower: float,
    limit_upper: float,
    limit_ke: float,
    limit_kd: float,
):
    limit_f = 0.0

    # compute limit forces, damping only active when limit is violated
    if q < limit_lower:
        limit_f = limit_ke * (limit_lower - q) - limit_kd * min(qd, 0.0)

    if q > limit_upper:
        limit_f = limit_ke * (limit_upper - q) - limit_kd * max(qd, 0.0)

    # joint dynamics
    t_total = target_ke * (q - target) + target_kd * qd + act - limit_f
    return t_total


@wp.func
def compute_joint_q(X_wp: wp.transform, X_wc: wp.transform, axis: wp.vec3, rotation_count: float):
    # child transform and moment arm
    q_p = wp.transform_get_rotation(X_wp)
    q_c = wp.transform_get_rotation(X_wc)
    # angular error pos
    r_err = wp.quat_inverse(q_p) * q_c

    # swing twist decomposition
    twist = quat_twist(axis, r_err)

    q = (
        wp.acos(twist[3]) * 2.0 * wp.sign(wp.dot(axis, wp.vec3(twist[0], twist[1], twist[2])))
    ) + 4.0 * PI * rotation_count
    return q


@wp.func
def compute_joint_qd(
    X_wp: wp.transform,
    X_wc: wp.transform,
    w_p: wp.vec3,
    w_c: wp.vec3,
    axis: wp.vec3,
):
    axis_p = wp.transform_vector(X_wp, axis)
    # angular error vel
    w_err = w_c - w_p
    qd = wp.dot(w_err, axis_p)
    return qd


@wp.func
def yaw_from_Xform(xform: wp.transform, axis: int):
    quat = wp.transform_get_rotation(xform)
    yaw = quat_decompose(quat)[axis]  # around z-axis
    if yaw < 0.0:
        yaw += 2.0 * PI  # put yaw in range [0, 3pi]
    return yaw


@wp.func
def theta_from_quat_wp(quat: wp.quat, axis: int):
    yaw = quat_decompose(quat)[axis]  # around z-axis
    if yaw < 0.0:
        yaw += 2.0 * PI  # put yaw in range [0, 3pi]
    return yaw


######### Warp Kernels


@wp.kernel
def l1_loss(
    a: wp.array(dtype=float),
    b: wp.array(dtype=float),
    idx: wp.array(dtype=int),
    # outputs:
    loss: wp.array(dtype=float),
):
    tid = wp.tid()
    i = idx[tid]
    wp.atomic_add(loss, 0, wp.abs(a[i] - b[i]))


@wp.kernel
def l1_xform_loss(
    a: wp.array(dtype=wp.transform),
    b: wp.array(dtype=wp.transform),
    # outputs:
    loss: wp.array(dtype=float),
):
    i = wp.tid()
    a_pos = wp.transform_get_translation(a[i])
    b_pos = wp.transform_get_translation(b[i])
    wp.atomic_add(loss, 0, wp.length(a_pos - b_pos))


@wp.kernel
def revolute_spring(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    joint_qd_start: wp.array(dtype=int),
    joint_type: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_X_p: wp.array(dtype=wp.transform),
    joint_axis: wp.array(dtype=wp.vec3),
    joint_limit_lower: wp.array(dtype=float),
    joint_limit_upper: wp.array(dtype=float),
    # outputs
    joint_target_kd_out: wp.array(dtype=float),
    joint_target_out: wp.array(dtype=float),
):
    tid = wp.tid()
    type = joint_type[tid]
    qd_start = joint_qd_start[tid]
    axis = joint_axis[tid]

    if type != JOINT_REVOLUTE_SPRING:
        return

    c_child = tid
    c_parent = joint_parent[tid]
    X_wp = joint_X_p[tid]
    X_wc = body_q[c_child]
    limit_lower = joint_limit_lower[qd_start]
    limit_upper = joint_limit_upper[qd_start]
    w_p = wp.vec3()
    w_c = wp.spatial_top(body_qd[c_child])

    if c_parent >= 0:
        X_wp = body_q[c_parent] * X_wp
        twist_p = body_qd[c_parent]
        w_p = wp.spatial_top(twist_p)

    q = compute_joint_q(X_wp, X_wc, axis, 0.0)

    # child transform and moment arm
    lim_range = limit_upper - limit_lower
    # distance from range center
    dist = limit_lower + lim_range / 2.0 - q
    # target_kd increases linearly from target_ke -> 2 * target_ke
    target_kd = joint_target_kd_out[tid]
    joint_target_kd_out[tid] = target_kd * (1.0 - wp.min(0.0, dist) * 2.0 / lim_range)
    # set joint target
    joint_target_out[tid] = q - 0.1 * lim_range


def compute_tight_friction(model, state_in, body_f):
    joint_q = wp.zeros_like(model.joint_q)
    joint_qd = wp.zeros_like(model.joint_qd)
    wp.launch(
        kernel=get_joint_q,
        inputs=[
            state_in.body_q,
            model.joint_q_start,
            model.joint_type_new,
            model.joint_parent,
            model.joint_X_p,
            model.joint_axis,
            0.0,
        ],
        outputs=[joint_q],
        dim=model.joint_count,
    )
    wp.launch(
        kernel=get_joint_qd,
        inputs=[
            state_in.body_q,
            state_in.body_qd,
            model.joint_qd_start,
            model.joint_type,
            model.joint_type_new,
            model.joint_parent,
            model.joint_X_p,
            model.joint_axis,
        ],
        outputs=[joint_qd],
        dim=model.joint_count,
    )
    wp.launch(
        kernel=eval_tight_friction,
        inputs=[
            joint_q,
            joint_qd,
            state_in.dt,
            model.oid,
            model.oqid,
            model.joint_mu_static,
            model.joint_mu_dynamic,
            model.stiction_transition_vel,
            model.max_stiction_displacement,
            model.upaxis,
        ],
        outputs=[body_f, model.joint_target, model.joint_act],
    )
    return


@wp.kernel
def eval_tight_friction(
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    dt: float,
    obj_id: wp.array(dtype=int),
    obj_q_id: wp.array(dtype=int),
    # eval_joint_torque used to compute joint_torque
    joint_mu_static: float,
    joint_mu_dynamic: float,
    stiction_transition_vel: float,
    max_stiction_displacement: float,
    upaxis: int,
    # outputs
    body_f: wp.array(dtype=wp.spatial_vector),
    target_theta: wp.array(dtype=float),
    joint_act: wp.array(dtype=float),
):
    tid = wp.tid()
    body_idx = obj_id[tid]
    obj_idx = obj_q_id[tid]
    q = joint_q[obj_idx]
    qd = joint_qd[obj_idx]
    target = target_theta[tid]

    dist = target - q
    # max joint displacement is a constant
    alpha = wp.abs(dist) / max_stiction_displacement
    beta = wp.abs(qd) / (1.5 * stiction_transition_vel)
    joint_mu = wp.select(
        alpha > 1.0,
        (1.0 - alpha) * 0.1 * joint_mu_static + alpha * joint_mu_static,
        wp.select(
            beta > 1.0,
            (1.0 - beta) * joint_mu_static + beta * joint_mu_dynamic,
            joint_mu_dynamic,
        ),
    )
    t_friction = joint_mu * (1.0 - wp.min(0.0, dist) / target)
    t_total = wp.spatial_top(body_f[body_idx])[2]
    # depending on whether net torque is +/-, friction should only go up to -t_total
    # ie. not apply external force
    friction = wp.select(t_total <= 0.0, wp.max(-t_total, -t_friction), wp.min(-t_total, t_friction))  # joint friction

    if upaxis == 2:
        body_friction = wp.spatial_vector(
            wp.vec3(
                0.0,
                0.0,
                friction,
            ),
            wp.vec3(),
        )
    elif upaxis == 1:
        body_friction = wp.spatial_vector(
            wp.vec3(
                0.0,
                friction,
                0.0,
            ),
            wp.vec3(),
        )

    wp.atomic_add(body_f, obj_idx, body_friction)
    wp.atomic_sub(target_theta, obj_idx, qd * dt)


@wp.kernel
def get_joint_q(
    body_q: wp.array(dtype=wp.transform),
    joint_q_start: wp.array(dtype=int),
    joint_type: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_X_p: wp.array(dtype=wp.transform),
    joint_axis: wp.array(dtype=wp.vec3),
    joint_rotation_count: float,
    # outputs
    joint_q: wp.array(dtype=float),
):
    tid = wp.tid()
    q_start = joint_q_start[tid]
    type = joint_type[tid]
    axis = joint_axis[tid]

    if type != wp.sim.JOINT_REVOLUTE and type != JOINT_REVOLUTE_TIGHT and type != JOINT_REVOLUTE_SPRING:
        return

    c_child = tid
    c_parent = joint_parent[tid]
    X_wp = joint_X_p[tid]
    X_wc = body_q[c_child]

    if c_parent >= 0:
        X_wp = body_q[c_parent] * X_wp

    q = compute_joint_q(X_wp, X_wc, axis, joint_rotation_count)
    wp.atomic_add(joint_q, q_start, q)


@wp.kernel
def get_joint_qd(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    joint_qd_start: wp.array(dtype=int),
    joint_type: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_X_p: wp.array(dtype=wp.transform),
    joint_axis: wp.array(dtype=wp.vec3),
    # outputs
    joint_qd: wp.array(dtype=float),
):
    tid = wp.tid()
    type = joint_type[tid]
    qd_start = joint_qd_start[tid]
    axis = joint_axis[tid]

    if type != wp.sim.JOINT_REVOLUTE and type != JOINT_REVOLUTE_TIGHT:
        return

    c_child = tid
    c_parent = joint_parent[tid]
    X_wp = joint_X_p[tid]
    X_wc = body_q[c_child]
    w_p = wp.vec3()  # is zero if parent is root
    w_c = wp.spatial_top(body_qd[c_child])

    if c_parent >= 0:
        X_wp = body_q[c_parent] * X_wp
        twist_p = body_qd[c_parent]
        w_p = wp.spatial_top(twist_p)

    qd = compute_joint_qd(X_wp, X_wc, w_p, w_c, axis)

    wp.atomic_add(joint_qd, qd_start, qd)


@wp.kernel
def update_rotation_count(
    body_q: wp.array(dtype=wp.transform),
    body_q_prev: wp.array(dtype=wp.transform),
    obj_id: wp.array(dtype=int),
    # joint_max_rotations: float,
    # outputs
    joint_rotation_count: wp.array(dtype=float),
):
    """Check if joint has rotated past range [-2pi, 2pi], and incr/decrement rotation count"""
    tid = wp.tid()
    oid = obj_id[tid]
    q_prev = yaw_from_Xform(body_q_prev[oid], 2)
    q = yaw_from_Xform(body_q[oid], 2)
    # increment/decrement number of rotations when sign flips btwn q_prev and q
    rotated = wp.select(
        wp.abs(q_prev - q) > 1.8 * PI,
        0.0,
        (wp.sign(q_prev) - wp.sign(q)) / 2.0,
    )
    wp.atomic_add(joint_rotation_count, tid, rotated)


@wp.kernel
def eval_body_joints(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    joint_q_start: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_type: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_X_p: wp.array(dtype=wp.transform),
    joint_X_c: wp.array(dtype=wp.transform),
    joint_axis: wp.array(dtype=wp.vec3),
    joint_target: wp.array(dtype=float),
    joint_act: wp.array(dtype=float),
    joint_target_ke: wp.array(dtype=float),
    joint_target_kd: wp.array(dtype=float),
    joint_limit_lower: wp.array(dtype=float),
    joint_limit_upper: wp.array(dtype=float),
    joint_limit_ke: wp.array(dtype=float),
    joint_limit_kd: wp.array(dtype=float),
    joint_attach_ke: float,
    joint_attach_kd: float,
    joint_rotation_count: float,
    joint_idx: int,
    body_f: wp.array(dtype=wp.spatial_vector),
):
    tid = joint_idx
    type = joint_type[tid]

    c_child = tid
    c_parent = joint_parent[tid]

    X_pj = joint_X_p[tid]
    X_cj = joint_X_c[tid]

    X_wp = X_pj
    r_p = wp.vec3()
    w_p = wp.vec3()
    v_p = wp.vec3()
    v_p = wp.vec3()

    # parent transform and moment arm
    if c_parent >= 0:
        X_wp = body_q[c_parent] * X_wp
        r_p = wp.transform_get_translation(X_wp) - wp.transform_point(body_q[c_parent], body_com[c_parent])

        twist_p = body_qd[c_parent]

        w_p = wp.spatial_top(twist_p)
        v_p = wp.spatial_bottom(twist_p) + wp.cross(w_p, r_p)

    # child transform and moment arm
    X_wc = body_q[c_child]  # *X_cj
    r_c = wp.transform_get_translation(X_wc) - wp.transform_point(body_q[c_child], body_com[c_child])

    twist_c = body_qd[c_child]

    w_c = wp.spatial_top(twist_c)
    v_c = wp.spatial_bottom(twist_c) + wp.cross(w_c, r_c)

    # joint properties (for 1D joints)
    q_start = joint_q_start[tid]
    qd_start = joint_qd_start[tid]
    axis = joint_axis[tid]

    target = joint_target[qd_start]
    target_ke = joint_target_ke[qd_start]
    target_kd = joint_target_kd[qd_start]
    limit_ke = joint_limit_ke[qd_start]
    limit_kd = joint_limit_kd[qd_start]
    limit_lower = joint_limit_lower[qd_start]
    limit_upper = joint_limit_upper[qd_start]

    act = joint_act[qd_start]

    x_p = wp.transform_get_translation(X_wp)
    x_c = wp.transform_get_translation(X_wc)

    q_p = wp.transform_get_rotation(X_wp)
    q_c = wp.transform_get_rotation(X_wc)

    # translational error
    x_err = x_c - x_p
    r_err = wp.quat_inverse(q_p) * q_c
    v_err = v_c - v_p
    w_err = w_c - w_p

    # total force/torque on the parent
    t_total = wp.vec3()
    f_total = wp.vec3()

    # reduce angular damping stiffness for stability
    angular_damping_scale = 0.01

    if type == wp.sim.JOINT_FIXED:
        ang_err = wp.normalize(wp.vec3(r_err[0], r_err[1], r_err[2])) * wp.acos(r_err[3]) * 2.0

        f_total += x_err * joint_attach_ke + v_err * joint_attach_kd
        t_total += (
            wp.transform_vector(X_wp, ang_err) * joint_attach_ke + w_err * joint_attach_kd * angular_damping_scale
        )

    if type == wp.sim.JOINT_PRISMATIC:
        # world space joint axis
        axis_p = wp.transform_vector(X_wp, joint_axis[tid])

        # evaluate joint coordinates
        q = wp.dot(x_err, axis_p)
        qd = wp.dot(v_err, axis_p)

        f_total = eval_joint_force(
            q,
            qd,
            target,
            target_ke,
            target_kd,
            act,
            limit_lower,
            limit_upper,
            limit_ke,
            limit_kd,
            axis_p,
        )

        # attachment dynamics
        ang_err = wp.normalize(wp.vec3(r_err[0], r_err[1], r_err[2])) * wp.acos(r_err[3]) * 2.0

        # project off any displacement along the joint axis
        f_total += (x_err - q * axis_p) * joint_attach_ke + (v_err - qd * axis_p) * joint_attach_kd
        t_total += (
            wp.transform_vector(X_wp, ang_err) * joint_attach_ke + w_err * joint_attach_kd * angular_damping_scale
        )

    if type == wp.sim.JOINT_REVOLUTE or type == JOINT_REVOLUTE_SPRING or type == JOINT_REVOLUTE_TIGHT:
        axis_p = wp.transform_vector(X_wp, axis)
        axis_c = wp.transform_vector(X_wc, axis)

        # swing twist decomposition
        twist = quat_twist(axis, r_err)

        q = (
            wp.acos(twist[3]) * 2.0 * wp.sign(wp.dot(axis, wp.vec3(twist[0], twist[1], twist[2])))
        ) + 4.0 * PI * joint_rotation_count

        qd = wp.dot(w_err, axis_p)

        t_total = eval_joint_force(  # checks if q + joint rotations hits joint limit
            q,
            qd,
            target,
            target_ke,
            target_kd,
            act,
            limit_lower,
            limit_upper,
            limit_ke,
            limit_kd,
            axis_p,
        )

        # attachment dynamics
        swing_err = wp.cross(axis_p, axis_c)

        f_total += x_err * joint_attach_ke + v_err * joint_attach_kd

        t_total += swing_err * joint_attach_ke + (w_err - qd * axis_p) * joint_attach_kd * angular_damping_scale

    if type == wp.sim.JOINT_BALL:
        ang_err = wp.normalize(wp.vec3(r_err[0], r_err[1], r_err[2])) * wp.acos(r_err[3]) * 2.0

        # todo: joint limits
        t_total += target_kd * w_err + target_ke * wp.transform_vector(X_wp, ang_err)
        f_total += x_err * joint_attach_ke + v_err * joint_attach_kd

    if type == wp.sim.JOINT_COMPOUND:
        q_off = wp.transform_get_rotation(X_cj)
        q_pc = wp.quat_inverse(q_off) * wp.quat_inverse(q_p) * q_c * q_off

        # decompose to a compound rotation each axis
        angles = quat_decompose(q_pc)

        # reconstruct rotation axes
        axis_0 = wp.vec3(1.0, 0.0, 0.0)
        q_0 = wp.quat_from_axis_angle(axis_0, angles[0])

        axis_1 = wp.quat_rotate(q_0, wp.vec3(0.0, 1.0, 0.0))
        q_1 = wp.quat_from_axis_angle(axis_1, angles[1])

        axis_2 = wp.quat_rotate(q_1 * q_0, wp.vec3(0.0, 0.0, 1.0))
        q_2 = wp.quat_from_axis_angle(axis_2, angles[2])

        q_w = q_p * q_off

        # joint dynamics
        t_total = wp.vec3()
        t_total += eval_joint_force(
            angles[0],
            wp.dot(wp.quat_rotate(q_w, axis_0), w_err),
            joint_target[qd_start + 0],
            joint_target_ke[qd_start + 0],
            joint_target_kd[qd_start + 0],
            joint_act[qd_start + 0],
            joint_limit_lower[qd_start + 0],
            joint_limit_upper[qd_start + 0],
            joint_limit_ke[qd_start + 0],
            joint_limit_kd[qd_start + 0],
            wp.quat_rotate(q_w, axis_0),
        )
        t_total += eval_joint_force(
            angles[1],
            wp.dot(wp.quat_rotate(q_w, axis_1), w_err),
            joint_target[qd_start + 1],
            joint_target_ke[qd_start + 1],
            joint_target_kd[qd_start + 1],
            joint_act[qd_start + 1],
            joint_limit_lower[qd_start + 1],
            joint_limit_upper[qd_start + 1],
            joint_limit_ke[qd_start + 1],
            joint_limit_kd[qd_start + 1],
            wp.quat_rotate(q_w, axis_1),
        )
        t_total += eval_joint_force(
            angles[2],
            wp.dot(wp.quat_rotate(q_w, axis_2), w_err),
            joint_target[qd_start + 2],
            joint_target_ke[qd_start + 2],
            joint_target_kd[qd_start + 2],
            joint_act[qd_start + 2],
            joint_limit_lower[qd_start + 2],
            joint_limit_upper[qd_start + 2],
            joint_limit_ke[qd_start + 2],
            joint_limit_kd[qd_start + 2],
            wp.quat_rotate(q_w, axis_2),
        )

        f_total += x_err * joint_attach_ke + v_err * joint_attach_kd

    if type == wp.sim.JOINT_UNIVERSAL:
        q_off = wp.transform_get_rotation(X_cj)
        q_pc = wp.quat_inverse(q_off) * wp.quat_inverse(q_p) * q_c * q_off

        # decompose to a compound rotation each axis
        angles = quat_decompose(q_pc)

        # reconstruct rotation axes
        axis_0 = wp.vec3(1.0, 0.0, 0.0)
        q_0 = wp.quat_from_axis_angle(axis_0, angles[0])

        axis_1 = wp.quat_rotate(q_0, wp.vec3(0.0, 1.0, 0.0))
        q_1 = wp.quat_from_axis_angle(axis_1, angles[1])

        axis_2 = wp.quat_rotate(q_1 * q_0, wp.vec3(0.0, 0.0, 1.0))
        q_2 = wp.quat_from_axis_angle(axis_2, angles[2])

        q_w = q_p * q_off

        # joint dynamics
        t_total = wp.vec3()

        # free axes
        t_total += eval_joint_force(
            angles[0],
            wp.dot(wp.quat_rotate(q_w, axis_0), w_err),
            joint_target[qd_start + 0],
            joint_target_ke[qd_start + 0],
            joint_target_kd[qd_start + 0],
            joint_act[qd_start + 0],
            joint_limit_lower[qd_start + 0],
            joint_limit_upper[qd_start + 0],
            joint_limit_ke[qd_start + 0],
            joint_limit_kd[qd_start + 0],
            wp.quat_rotate(q_w, axis_0),
        )
        t_total += eval_joint_force(
            angles[1],
            wp.dot(wp.quat_rotate(q_w, axis_1), w_err),
            joint_target[qd_start + 1],
            joint_target_ke[qd_start + 1],
            joint_target_kd[qd_start + 1],
            joint_act[qd_start + 1],
            joint_limit_lower[qd_start + 1],
            joint_limit_upper[qd_start + 1],
            joint_limit_ke[qd_start + 1],
            joint_limit_kd[qd_start + 1],
            wp.quat_rotate(q_w, axis_1),
        )

        # last axis (fixed)
        # if angles[2] < -1e-3:
        # XXX prevent numerical instability at singularity
        t_total += eval_joint_force(
            angles[2],
            wp.dot(wp.quat_rotate(q_w, axis_2), w_err),
            0.0,
            joint_attach_ke,
            joint_attach_kd * angular_damping_scale,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            wp.quat_rotate(q_w, axis_2),
        )

        f_total += x_err * joint_attach_ke + v_err * joint_attach_kd

    # write forces
    if c_parent >= 0:
        wp.atomic_add(
            body_f,
            c_parent,
            wp.spatial_vector(t_total + wp.cross(r_p, f_total), f_total),
        )

    wp.atomic_sub(body_f, c_child, wp.spatial_vector(t_total + wp.cross(r_c, f_total), f_total))


@wp.kernel
def assign_body_f_torque(
    actions: wp.array(dtype=float),
    action_index: int,
    screw_idx: int,
    body_f: wp.array(dtype=wp.spatial_vector),
):
    """Assigns the torque to the body_f array"""
    if screw_idx == 0:
        screw_top = wp.vec3(actions[action_index], 0.0, 0.0)
    elif screw_idx == 1:
        screw_top = wp.vec3(0.0, actions[action_index], 0.0)
    elif screw_idx == 2:
        screw_top = wp.vec3(0.0, 0.0, actions[action_index])
    body_f[1] = wp.spatial_vector(
        screw_top,
        wp.vec3(
            0.0,
            0.0,
            0.0,
        ),
    )


@wp.kernel
def assign_vec3(in_vec: wp.vec3, out_vec: wp.vec3):
    """Assigns vec3 array"""
    out_vec *= 0.0
    out_vec += in_vec


@wp.kernel
def apply_clamp(a: float, b: float, arr: wp.array(dtype=float)):
    tid = wp.tid()
    arr[tid] = wp.clamp(arr[tid], a, b)


e_xz = wp.constant(wp.vec3(1.0, 0.0, 1.0))
e_x = wp.constant(wp.vec3(1.0, 0.0, 0.0))
POSE_THRESHOLD = wp.constant(0.05)
WRENCH_THRESHOLD = wp.constant(0.01)
FINGER_THRESHOLD = wp.constant(0.125)  # each finger should be within this distance


@wp.kernel
def joint_limit_penalty(
    body_q: wp.array(dtype=wp.transform),
    joint_qd_start: wp.array(dtype=int),
    joint_type: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_X_p: wp.array(dtype=wp.transform),
    joint_axis: wp.array(dtype=wp.vec3),
    joint_limit_lower: wp.array(dtype=float),
    joint_limit_upper: wp.array(dtype=float),
    num_joints: int,
    # outputs
    reward_total: wp.array(dtype=float),
):
    tid = wp.tid()
    jtype = joint_type[tid]
    qd_start = joint_qd_start[tid]
    axis = joint_axis[tid]

    if jtype != JOINT_REVOLUTE_SPRING:
        return

    c_child = tid
    c_parent = joint_parent[tid]
    X_wp = joint_X_p[tid]
    X_wc = body_q[c_child]
    limit_lower = joint_limit_lower[qd_start]
    limit_upper = joint_limit_upper[qd_start]

    if c_parent >= 0:
        X_wp = body_q[c_parent] * X_wp

    q = compute_joint_q(X_wp, X_wc, axis, 0.0)
    env_id = wp.floordiv(tid, num_joints)
    reward = reward_total[env_id]
    reward = wp.select(q < limit_lower, reward, -(limit_lower - q))
    reward = wp.select(q > limit_upper, reward, -(q - limit_upper))
    reward_total[env_id] = reward


@wp.kernel
def compute_goal_reward(
    body_q: wp.array(dtype=wp.transform),
    body_f: wp.array(dtype=wp.spatial_vector),
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    finger_id: wp.array(dtype=int),
    num_fingers: int,
    obj_id: wp.array(dtype=int),
    obj_q_id: wp.array(dtype=int),
    offset: wp.vec3,
    object_thickness: float,
    base_id: wp.array(dtype=int),
    goal_pos: wp.vec3,
    goal_q: wp.quat,
    goal_force: wp.vec3,
    goal_torque: wp.vec3,
    goal_type: int,
    reward_type: int,
    reward_ctrl: wp.array(dtype=float),
    reward_coef: wp.array(dtype=float),
    rotation_count: wp.array(dtype=float),
    # outputs
    reward_vec: wp.array(dtype=float),
    reward_vec_prev: wp.array(dtype=float),
    reward_total: wp.array(dtype=float),
):
    """Computes the reward for the goal task"""
    tid = wp.tid()
    obj_idx = obj_id[tid]
    q_idx = obj_q_id[tid]
    rotations = rotation_count[tid]
    obj_center = wp.transform_get_translation(body_q[obj_idx])
    finger_dist = float(0.0)
    for f_i in range(num_fingers):
        f_idx = finger_id[f_i + tid * num_fingers]
        finger_xform = body_q[f_idx]
        finger_dist = finger_dist + (
            wp.length(wp.cw_mul(obj_center - wp.transform_point(finger_xform, offset), e_xz)) - object_thickness
        )
    origin = wp.transform_get_translation(body_q[obj_idx - 5])
    # Projects object pos onto the xz plane, and computes distance to goal
    obj_pos = wp.cw_mul(obj_center - origin, e_xz)
    pos_dist = wp.length(goal_pos - obj_pos)

    # if POSE-WRENCH goal, use object quaternion
    if goal_type == GOAL_POS_WRENCH:
        obj_quat = wp.transform_get_rotation(body_q[obj_idx])  # quaternion
        q_dist = quat_ang_err(obj_quat, goal_q)
    else:
        if goal_type == GOAL_POS or goal_type == GOAL_POS_TRAJ or goal_type == GOAL_POS_FORCE or goal_type == GOAL_POSE:
            q = yaw_from_Xform(body_q[obj_idx], 2)
        else:
            q = joint_q[q_idx]
            qd = joint_qd[q_idx]
        goal_q_yaw = theta_from_quat_wp(goal_q, 2)
        q_dist = wp.abs(goal_q_yaw - q)
        q_vel = wp.abs(qd)
        wp.atomic_add(reward_ctrl, 0, q_vel)  # penalize joint velocity

    # allow q_dist to wrap around, since yaw is always in (0, 2pi)
    if goal_type == GOAL_POS_WRENCH:
        q_dist = quat_ang_err(obj_quat, goal_q)
    else:
        goal_q_yaw = theta_from_quat_wp(goal_q, 2)  # + 2.0 * PI * rotations
        q_dist = wp.abs(goal_q_yaw - q)

    if goal_type == GOAL_POS_FORCE:
        wrench_dist = wp.length(goal_force - wp.spatial_bottom(body_f[obj_idx]))
    elif goal_type == GOAL_ORI_TORQUE:
        wrench_dist = wp.length(goal_torque - wp.spatial_top(body_f[obj_idx]))

    if goal_type == GOAL_POS or goal_type == GOAL_POS_TRAJ or goal_type == GOAL_POS_FORCE:
        succ = pos_dist < POSE_THRESHOLD
    elif goal_type == GOAL_ORI or goal_type == GOAL_ORI_TRAJ or goal_type == GOAL_ORI_TORQUE:
        succ = q_dist < POSE_THRESHOLD

    # task_bonus = 0.0
    # task_bonus = wp.select(succ, task_bonus, task_bonus + 2.0)
    # if goal_type == GOAL_POS_FORCE or goal_type == GOAL_ORI_TORQUE:
    #     task_bonus = wp.select(
    #         succ and wrench_dist < WRENCH_THRESHOLD,
    #         task_bonus,
    #         task_bonus + 10.0,
    #     )
    # # # keep fingers near object for bonus
    # task_bonus = wp.select(
    #     succ and finger_dist / float(num_fingers) < FINGER_THRESHOLD,
    #     task_bonus,
    #     task_bonus + 1.0,
    # )
    
    wp.atomic_sub(reward_vec, 5 * tid, reward_ctrl[tid])
    wp.atomic_add(reward_vec, 5 * tid + 1, finger_dist)
    wp.atomic_add(reward_vec, 5 * tid + 2, q_dist)
    wp.atomic_add(reward_vec, 5 * tid + 3, pos_dist)
    wp.atomic_add(reward_vec, 5 * tid + 4, wrench_dist)

    c_act = reward_coef[0]
    c_finger = reward_coef[1]
    c_q = reward_coef[2]
    c_pos = reward_coef[3]
    c_ft = reward_coef[4]

    r_i = 5 * tid
    if reward_type == REWARD_L2:
        # all distances are already l2/l1, take distance to error threshold to
        # make rewards "positive"
        rew_ctrl = reward_vec[r_i]
        rew_finger = FINGER_THRESHOLD - reward_vec[r_i + 1] / float(num_fingers)
        rew_q = POSE_THRESHOLD - reward_vec[r_i + 2]
        rew_pos = POSE_THRESHOLD - reward_vec[r_i + 3]
        rew_ft = WRENCH_THRESHOLD - reward_vec[r_i + 4]
    elif reward_type == REWARD_DELTA:
        # ac penalties are negative, delts should be positive when ac penalty
        # decreases, i.e. curr - prev > 0
        rew_ctrl = reward_vec[r_i] - reward_vec_prev[r_i]
        # dists are positive and should tend -> 0, so prev - curr > 0
        rew_finger = reward_vec_prev[r_i + 1] - reward_vec[r_i + 1]
        rew_q = reward_vec_prev[r_i + 2] - reward_vec[r_i + 2]
        rew_pos = reward_vec_prev[r_i + 3] - reward_vec[r_i + 3]
        rew_ft = reward_vec_prev[r_i + 4] - reward_vec[r_i + 4]
    elif reward_type == REWARD_EXP:
        rew_ctrl = reward_vec[r_i]
        rew_finger = wp.exp(-reward_vec[r_i + 1])
        rew_q = wp.exp(-reward_vec[r_i + 2])
        rew_pos = wp.exp(-reward_vec[r_i + 3])
        rew_ft = wp.exp(-reward_vec[r_i + 4])

    wp.atomic_add(
        reward_total,
        tid,
        c_act * rew_ctrl + c_finger * rew_finger + c_q * rew_q + c_pos * rew_pos + c_ft * rew_ft 
    )
    # + task_bonus,

    # neg-exponential rewards, r_i in (0, 1)
    # wp.atomic_add(
    #     reward_total,
    #     tid,
    #     c_act * reward_vec[r_i]  # control penalty
    #     + c_finger * wp.exp(-reward_vec[r_i + 1])  # finger distance, in cm
    #     # orientation distance in (0, 2pi), shift to (0, 360)
    #     + c_q * wp.exp(-reward_vec[r_i + 2])  # q_err in degrees
    #     + c_pos * wp.exp(-reward_vec[r_i + 3])  # position distance
    #     + c_ft * wp.exp(-reward_vec[r_i + 4]),  # force/torque distance
    # )
    # l2-norm rewards, r_i in (-inf, 0)
    # wp.atomic_add(
    #     reward_total,
    #     tid,
    #     c_act * reward_vec[r_i]  # control penalty
    #     + c_finger * (reward_vec[r_i + 1])  # lfinger distance, in cm
    #     # orientation distance in (0, 2pi), shift to (0, 360)
    #     + c_q * wp.exp(-reward_vec[r_i + 2])  # q_err in degrees
    #     + c_pos * wp.exp(-reward_vec[r_i + 3])  # position distance
    #     + c_ft * wp.exp(-reward_vec[r_i + 4]),  # force/torque distance
    # )
    # delta-rewards, r_i in (0, +inf)
    # wp.atomic_add(
    #     reward_total,
    #     tid,
    #     # (neg) control penalty, should give positive reward when increased, so negate
    #     (c_act * (reward_vec[r_i] - reward_vec_prev[r_i]))
    #     # lfinger distance, in cm
    #     + (c_finger * (reward_vec_prev[r_i + 1] - reward_vec[r_i + 1]))
    #     # orientation distance in (0, 2pi), shift to (0, 360)
    #     + (c_q * (reward_vec_prev[r_i + 2] - reward_vec[r_i + 2]))
    #     # position distance
    #     + (c_pos * (reward_vec_prev[r_i + 3] - reward_vec[r_i + 3]))
    #     # force/torque distance
    #     + (c_ft * (reward_vec_prev[r_i + 4] - reward_vec[r_i + 4])) 
    # )
    # + task_bonus,


@wp.kernel
def compute_ctrl_reward(
    actions: wp.array(dtype=float),
    num_acts: int,
    # outputs
    reward: wp.array(dtype=wp.float32),
):
    i, j = wp.tid()
    tid = i * num_acts + j

    reward_ctrl = actions[tid] * actions[tid]
    wp.atomic_add(reward, i, reward_ctrl)


@wp.kernel
def compute_ctrl_pos_reward(
    actions: wp.array(dtype=float),
    joint_q_prev: wp.array(dtype=float),
    act_index: wp.array(dtype=int),
    num_acts: int,
    # outputs
    reward: wp.array(dtype=wp.float32),
):
    i, j = wp.tid()
    tid = i * num_acts + j

    qpos_diff = actions[tid] - joint_q_prev[act_index[tid]]
    reward_ctrl = qpos_diff  * qpos_diff
    wp.atomic_add(reward, i, reward_ctrl)



@wp.kernel
def assign_body_q_qd_kernel(
    body_q_in: wp.array(dtype=wp.transform),
    body_qd_in: wp.array(dtype=wp.spatial_vector),
    body_q_out: wp.array(dtype=wp.transform),
    body_qd_out: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()
    body_q_out[tid] = body_q_in[tid]
    body_qd_out[tid] = body_qd_in[tid]


@wp.kernel
def integrate_body_f_kernel(
    body_q: wp.array(dtype=wp.transform),
    body_qd_prev: wp.array(dtype=wp.spatial_vector),
    body_qd: wp.array(dtype=wp.spatial_vector),
    dt: float,
    m: wp.array(dtype=float),
    I: wp.array(dtype=wp.mat33),
    body_f: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()

    # positions
    q = body_q[tid]
    qd = body_qd[tid]
    qdprev = body_qd_prev[tid]
    f = body_f[tid]

    # masses
    mass = m[tid]
    inertia = I[tid]

    # unpack transform
    r0 = wp.transform_get_rotation(q)

    # unpack spatial twist
    w0 = wp.spatial_top(qd)
    v0 = wp.spatial_bottom(qd)
    wprev = wp.spatial_top(qdprev)
    vprev = wp.spatial_bottom(qdprev)

    # unpack spatial wrench
    t0 = wp.spatial_top(f)
    f0 = wp.spatial_bottom(f)

    wb = wp.quat_rotate_inv(r0, w0)
    wbprev = wp.quat_rotate_inv(r0, wprev)
    wbdiff = (wb - wbprev) / dt
    tb = t0 + inertia * wbdiff + wp.cross(wb, inertia * wb)

    t1 = wp.quat_rotate(r0, tb)
    f1 = f0 + mass * (v0 - vprev) / dt
    body_f[tid] = wp.spatial_vector(t1, f1)


@wp.kernel
def assign_act_kernel(
    act: wp.array(dtype=float),
    action_type: int,
    num_acts: int,
    num_joints: int,
    q_offset: int,
    # outputs
    joint_act: wp.array(dtype=float),
    joint_stiffness: wp.array(dtype=float),
):
    i, j = wp.tid()
    joint_act_idx = i * num_joints + j + q_offset  # skip object joint
    act_idx = i * num_acts + j
    wp.atomic_add(joint_act, joint_act_idx, act[act_idx])
    if action_type == ACTION_JOINT_STIFFNESS:
        wp.atomic_add(joint_stiffness, joint_act_idx, act[act_idx + num_acts // 2])


@wp.kernel
def assign_act_indexed_kernel(
    act_src: wp.array(dtype=float),
    act_index: wp.array(dtype=int),
    # outputs
    action_target: wp.array(dtype=float),
):
    tid = wp.tid()
    target_idx = act_index[tid]
    wp.atomic_add(action_target, target_idx, act_src[tid])


######### Warp Helpers


def integrate_body_f(model, body_qd_in, body_q_out, body_qd_out, body_f, dt):
    wp.launch(
        integrate_body_f_kernel,
        dim=len(body_f),
        device=model.device,
        inputs=[
            body_q_out,
            body_qd_in,
            body_qd_out,
            dt,
            model.body_mass,
            model.body_inertia,
        ],
        outputs=[body_f],
    )


def assign_act(
    act,
    joint_act,
    joint_stiffness,
    action_type,
    num_acts=4,
    num_envs=None,
    num_joints=6,
    q_offset=1,
    joint_indices=None,
):
    assert np.prod(act.shape) == num_envs * num_acts, f"act shape {act.shape} is not {num_envs} x {num_acts}"
    act_count = num_acts
    if action_type.value is None:
        print("Warning: attemted to assign action when action_type is None")
        return

    if action_type.value == ACTION_JOINT_STIFFNESS:
        # num_acts should correspond to number of joint_act dims, not include joint_ke
        act_count = num_acts // 2
    if joint_indices is not None:
        assert (
            joint_indices.size == act.size
        ), f"Expected joint_indices.size == act.size, got {joint_indices.size}, {act.size}"
        wp.launch(
            kernel=assign_act_indexed_kernel,
            dim=act.size,
            device=joint_act.device,
            inputs=[act, joint_indices],
            outputs=[joint_act],
        )
    else:
        if action_type.value != ACTION_JOINT_STIFFNESS and joint_stiffness is None:
            joint_stiffness = wp.zeros_like(joint_act, requires_grad=False)
        wp.launch(
            kernel=assign_act_kernel,
            dim=(num_envs, act_count),
            device=joint_act.device,
            inputs=[act, action_type.value, num_acts, num_joints, q_offset],
            outputs=[joint_act, joint_stiffness],
        )
    return


class IntegratorEuler(SemiImplicitIntegrator):
    def simulate(self, model, state_in, state_out, dt, requires_grad=False):
        with wp.ScopedTimer("simulate", False):
            particle_f = None
            body_f = None

            if state_in.body_count:
                body_f = state_in.body_f

            compute_forces(model, state_in, particle_f, body_f, requires_grad=requires_grad)
            compute_tight_friction(model, state_in, body_f)
            body_f.zero_()
            compute_forces(model, state_in, particle_f, body_f, requires_grad=requires_grad)

            # -------------------------------------
            # integrate bodies

            if model.body_count:
                wp.launch(
                    kernel=integrate_bodies,
                    dim=model.body_count,
                    inputs=[
                        state_in.body_q,
                        state_in.body_qd,
                        state_in.body_f,
                        model.body_com,
                        model.body_mass,
                        model.body_inertia,
                        model.body_inv_mass,
                        model.body_inv_inertia,
                        model.gravity,
                        self.angular_damping,
                        dt,
                    ],
                    outputs=[state_out.body_q, state_out.body_qd],
                    device=model.device,
                )

            return state_out


class AdamStepLR(Adam):
    """A subclass inheriting from wp.optim.Adam, with added learning rate
    scheduling as implemented in Pytorch, and gradient clipping:
    https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html
    """

    def __init__(
        self,
        params=None,
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-08,
        step_freq=5,
        gamma=0.1,
        last_epoch=-1,
        min_lr=1e-5,
        grad_clip=None,
    ):
        super().__init__(params, lr, betas, eps)
        self.step_freq = step_freq
        self.gamma = gamma
        self.last_epoch = last_epoch
        self.grad_clip = grad_clip
        self.min_lr = min_lr

    def step(self, grads):
        self.last_epoch += 1
        if self.step_freq is not None:
            if self.last_epoch > 0 and self.last_epoch % self.step_freq == 0:
                lr_prev = self.lr
                if self.min_lr:
                    self.lr = max(self.lr * self.gamma, self.min_lr)
                else:
                    self.lr *= self.gamma
                print(f"Step {self.last_epoch}: updating LR {lr_prev} -> {self.lr}")
        if self.grad_clip is not None:
            for grad in grads:
                wp.launch(
                    kernel=apply_clamp,
                    inputs=[-self.grad_clip, self.grad_clip],
                    outputs=[grad],
                    dim=len(grad),
                )
        return super().step(grads)


def yaw_from_quat(quat, axis=1):
    return Rotation.from_quat(quat).as_euler("xyz")[..., axis]
