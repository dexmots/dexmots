import numpy as np
import warp as wp
import torch
from tcdm.envs import suite
from tcdm.envs import traj_abspath
from .common import ObjectType
from .warp_utils import integrate_body_f_kernel
import scipy.ndimage as filters
from .rotation3d import (
    quat_identity_like,
    quat_mul_norm,
    quat_inverse,
    quat_angle_axis,
)


def get_tcdm_trajectory(object_type):
    assert object_type in TCDM_TRAJ_PATHS, f"object type {object_type} not found"
    traj_path = TCDM_TRAJ_PATHS[object_type]
    print("loading goal trajectory from", traj_path)
    trajectory = np.load(traj_abspath(traj_path), allow_pickle=True)
    return_dict = dict(
        position=trajectory["object_translation"],
        orientation=trajectory["object_orientation"],
    )
    trajectory_wrench = generate_wrenches(return_dict, object_type)
    torque, force = trajectory_wrench[:, :3], trajectory_wrench[:, 3:]
    return_dict["position"] = return_dict["position"][1:]
    return_dict["orientation"] = return_dict["orientation"][1:]
    return_dict["force"] = force
    return_dict["torque"] = torque
    return trajectory, return_dict


def get_obj_pose_vel(env):
    physics = env.physics
    object_name = env.task.object_name
    obj_com = physics.named.data.xipos[object_name].copy()
    obj_rot = physics.named.data.xquat[object_name].copy()
    obj_vel = physics.data.object_velocity(object_name, "body")
    obj_vel = obj_vel.reshape((1, 6))
    pose = np.concatenate([obj_com, obj_rot])
    return pose, obj_vel


def compute_angular_velocity(r, dt, gaussian_filter=True):

    # assume the second last dimension is the time axis
    r = torch.as_tensor(r)
    diff_quat_data = quat_identity_like(r)
    diff_quat_data[..., :-1, :] = quat_mul_norm(r[..., 1:, :], quat_inverse(r[..., :-1, :]))
    diff_angle, diff_axis = quat_angle_axis(diff_quat_data)
    angular_velocity = diff_axis * diff_angle.unsqueeze(-1) / dt
    if gaussian_filter:
        angular_velocity = filters.gaussian_filter1d(angular_velocity.numpy(), 2, axis=-2, mode="nearest")
    return angular_velocity


def compute_linear_velocity(x, dt, gaussian_filter=True):

    vel = (x[1:] - x[:-1]) / dt
    vel = np.concatenate([[[0, 0, 0]], vel])
    if gaussian_filter:
        lin_vel = filters.gaussian_filter1d(vel, 2, axis=-2, mode="nearest")
    return lin_vel


def get_suite(object_type):
    object, task = TCDM_OBJ_NAMES[object_type].split("-")
    env = suite.load(object, task)
    return env


def generate_wrenches(traj, object_type):
    env = get_suite(object_type)
    object_name = env.task.object_name
    rot, pos = traj["orientation"], traj["position"]
    dt = env.control_timestep()
    w0 = compute_angular_velocity(rot, dt)
    v0 = compute_linear_velocity(pos, dt)
    qd = np.hstack([w0, v0])
    q = np.hstack([pos, rot])
    qd_prev, qd_next = qd[:-1], qd[1:]

    m = env.physics.named.model.body_mass[object_name]
    I = env.physics.named.model.body_inertia[object_name]
    I = np.diag(I)
    m_wp = wp.array(np.array([m for _ in range(len(qd_next))]), dtype=float)
    I_wp = wp.array(np.array([I for _ in range(len(qd_next))]), dtype=wp.mat33)

    print("mass", m)
    print("inertia", I.round(6))

    q_wp = wp.array(q[1:], dtype=wp.transform)
    qd_in_wp = wp.array(qd_prev, dtype=wp.spatial_vector)
    qd_out_wp = wp.array(qd_next, dtype=wp.spatial_vector)
    body_f = wp.zeros_like(qd_in_wp)
    wp.launch(
        integrate_body_f_kernel,
        dim=len(body_f),
        inputs=[
            q_wp,
            qd_in_wp,
            qd_out_wp,
            dt,
            m_wp,
            I_wp,
        ],
        outputs=[body_f],
    )
    return body_f.numpy()


TCDM_MESH_PATHS = {ObjectType.TCDM_STAPLER: "meshes/objects/stapler/stapler.stl"}
TCDM_TRAJ_PATHS = {ObjectType.TCDM_STAPLER: "stapler_lift.npz"}
TCDM_OBJ_NAMES = {ObjectType.TCDM_STAPLER: "stapler-lift"}
