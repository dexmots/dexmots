import os
import numpy as np
import warp as wp
import torch
from typing import Tuple, Optional
from .obj_env import ObjectTask
from ..utils import builder as bu
from ..utils.rewards import l1_dist
from ..utils.warp_utils import assign_act
from ..utils.common import (
    HandType,
    ObjectType,
    ActionType,
    HAND_ACT_COUNT,
    joint_coord_map,
    load_grasps_npy,
    supported_joint_types,
    run_env,
)
from ..utils.torch_utils import to_torch, scale
from ..utils import torch_utils as tu
from .environment import RenderMode


class HandObjectTask(ObjectTask):
    obs_keys = ["hand_joint_pos", "hand_joint_vel"]
    collapse_joints: bool = True
    xpbd_settings = dict(
        iterations=10,
        joint_linear_relaxation=1.0,
        joint_angular_relaxation=0.45,
        rigid_contact_relaxation=2.0,
        rigid_contact_con_weighting=True,
    )

    rigid_contact_margin = 0.001
    rigid_mesh_contact_max = 100

    def __init__(
        self,
        num_envs,
        num_obs,
        episode_length,
        action_type: ActionType = ActionType.TORQUE,
        seed=0,
        no_grad=True,
        render=True,
        stochastic_init=False,
        device="cuda",
        render_mode=RenderMode.OPENGL,
        stage_path=None,
        object_type: Optional[ObjectType] = None,
        object_id=0,
        stiffness=1000.0,
        damping=0.1,
        reward_params=None,
        hand_type: HandType = HandType.ALLEGRO,
        hand_start_position: Tuple = (0.1, 0.3, 0.0),
        hand_start_orientation: Tuple = (-np.pi / 2 * 3, np.pi * 1.25, np.pi / 2 * 3),
        load_grasps: bool = False,
        grasp_id: int = -1,
        use_autograd: bool = True,
        use_graph_capture: bool = True,
        goal_joint_pos=None,
        fix_position: bool = True,
        fix_orientation: bool = True,
        headless: bool = False,
    ):
        self.fix_position = fix_position
        self.fix_orientation = fix_orientation
        env_name = hand_type.name + "Env"
        self.hand_start_position = hand_start_position
        self.hand_start_orientation = hand_start_orientation
        self.hand_type = hand_type
        self.grasp_joint_q = None

        if load_grasps:
            self.grasps = load_grasps_npy(object_type, object_id, hand_type)
            self.grasp = None
            if grasp_id:
                self.grasp = self.grasps[grasp_id]
                self.grasp_joint_q = to_torch(
                    np.stack([self.grasp.joint_pos for _ in range(num_envs)], axis=0),
                    device=device,
                )
                pos, ori = bu.get_object_xform(object_type, object_id, base_joint="rx, ry, rz, px, py, pz")
                ori, pos = tu.tf_combine(*[to_torch(x) for x in [ori, pos, self.grasp.xform[3:], self.grasp.xform[:3]]])
                # ori = self.grasp.xform[3:]
                # ori = tu.quat_mul(to_torch(tuple(wp.quat_rpy(*self.hand_start_orientation))), to_torch(ori))
                # pos = tu.tf_apply(*[to_torch(x) for x in [ori, pos, self.grasp.xform[:3]]])

                self.hand_start_position = (pos[0], pos[1], pos[2])
                self.hand_start_orientation = (ori[0], ori[1], ori[2], ori[3])
        else:
            self.grasp = None
            self.grasps = None

        stochastic_init = stochastic_init
        self.hand_stiffness = stiffness
        self.hand_damping = damping
        # self.gravity = 0.0

        super().__init__(
            num_envs=num_envs,
            num_obs=num_obs,
            num_act=HAND_ACT_COUNT[(hand_type, action_type)],
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
            stiffness=0.0,
            damping=damping,
            reward_params=reward_params,
            env_name=env_name,
            use_autograd=use_autograd,
            use_graph_capture=use_graph_capture,
            goal_joint_pos=goal_joint_pos,
            headless=headless,
        )

        # contact truncation requires keeping track of num contacts per env
        # self.prev_contact_count = np.zeros(self.num_envs, dtype=int)
        # self.contact_count_changed = torch.zeros_like(self.reset_buf)

        print("gravity", self.model.gravity, self.gravity)

        self.hand_target_ke = self.model.joint_target_ke
        self.hand_target_kd = self.model.joint_target_kd
        self.simulate_params["ag_return_body"] = True

    @property
    def base_joint(self):
        base_joint = ""
        if self.fix_position and self.fix_orientation:
            base_joint = None
        elif self.fix_orientation:
            base_joint += "rx, ry, rz "
        elif self.fix_position:
            base_joint = "px, py, pz"
        else:
            base_joint = ""
        if base_joint:
            return base_joint
        else:
            return None

    @property
    def floating_hand(self):
        return not (self.fix_position and self.fix_orientation)

    def init_sim(self):
        super().init_sim()
        # create mapping from body name to index
        self.body_name_to_idx, self.joint_name_to_idx = {}, {}
        for i, body_name in enumerate(self.model.body_name):
            body_ind = self.body_name_to_idx.get(body_name, [])
            body_ind.append(i)
            self.body_name_to_idx[body_name] = body_ind

        # create mapping from joint name to index
        for i, joint_name in enumerate(self.model.joint_name):
            joint_ind = self.joint_name_to_idx.get(joint_name, [])
            joint_ind.append(i)
            self.joint_name_to_idx[joint_name] = joint_ind

        self.body_name_to_idx = {k: np.array(v) for k, v in self.body_name_to_idx.items()}
        self.joint_name_to_idx = {k: np.array(v) for k, v in self.joint_name_to_idx.items()}

    def compute_position_delta_action(self, actions, hand_pos=None):
        if hand_pos is None:
            hand_pos = self.joint_q.view(self.num_envs, -1)[:, self.env_joint_target_indices]
        lower, upper = self.action_bounds
        actions = 0.1 * actions.clamp(-1, 1) * (upper - lower)
        actions = torch.clamp(hand_pos + actions, lower, upper)  # TODO: parameterize action_scale in config
        return actions

    def assign_actions(self, actions):
        # overrides ObjectTask.assign_actions
        actions = actions.reshape(self.num_envs, -1)
        num_joints = self.num_joint_q  # - bu.OBJ_NUM_JOINTS.get(self.object_type, 0)
        if self.num_acts < num_joints and self.floating_hand:
            # assert (
            #     num_joints == self.num_acts + 7
            # ), "num acts should equal should equal num joint q + 7 (free joint dof)"
            hand_xform = wp.from_torch(self.hand_init_xform)
            # in case of multiple free joints per env, take first
            xform_indices = np.concatenate(
                list(
                    map(
                        lambda x: list(
                            range(
                                self.model.joint_axis_start[x[0]],
                                self.model.joint_axis_start[x[0] + 1],
                            )
                        ),
                        filter(lambda x: x[1] == wp.sim.JOINT_FREE, self.model.joint_type),
                    )
                )
            ).reshape(self.num_envs, -1)[:, 0]
            assign_act(
                hand_xform,
                self.model.joint_target,
                self.model.joint_target_ke,
                self.action_type,
                7,
                self.num_envs,
                joint_indices=xform_indices,
            )

        if self.action_type == ActionType.POSITION_DELTA:
            actions = self.compute_position_delta_action(actions)
            self.reward_extras["target_qpos"] = actions
            self.reward_extras["prev_hand_qpos"] = self.joint_q.view(self.num_envs, -1)[
                :, self.env_joint_target_indices
            ].clone()
        else:
            lower, upper = self.action_bounds
            actions = scale(actions.clamp(-1, 1), lower, upper)
            self.reward_extras["target_qpos"] = actions

        actions = actions.flatten()

        if self.action_type == ActionType.TORQUE:
            self.model.joint_act.zero_()
            joint_acts = self.model.joint_act
        else:
            joint_acts = self.model.joint_target

        joint_acts.zero_()
        # assigns act
        self.warp_actions.assign(wp.from_torch(actions.flatten()))
        # env manages grad tape, and action setting happens with autograd utils
        if self.use_autograd and self.requires_grad:
            return
        assign_act(
            self.warp_actions,
            joint_acts,
            self.model.joint_target_ke,
            self.action_type,
            self.num_actions,
            self.num_envs,
            joint_indices=self.joint_target_indices,
        )

    def _check_contact_count(self):
        contact_count = np.zeros(self.num_envs, dtype=int)
        shape0 = wp.to_torch(self.model.rigid_contact_shape0)
        for i in range(self.model.rigid_contact_count.numpy().item()):
            if shape0[i] <= 0:
                continue
            env_idx = self.model.shape_collision_group[shape0[i]] - 1
            contact_count[env_idx] += 1
        self.contact_count_changed[:] = torch.as_tensor(contact_count != self.prev_contact_count)
        self.prev_contact_count = contact_count
        self.extras["truncation"] = self.contact_count_changed

    def _get_obs_dict(self):
        joint_q, joint_qd = self.joint_q.view(self.num_envs, -1), self.joint_qd.view(self.num_envs, -1)
        joint_act = wp.to_torch(self.model.joint_act, requires_grad=self.requires_grad).view(self.num_envs, -1)
        obs_dict = {}
        obs_dict["hand_joint_pos"] = joint_q[:, self.env_joint_target_indices]
        obs_dict["hand_joint_vel"] = joint_qd[:, self.env_joint_target_indices]
        obs_dict["hand_joint_act"] = joint_act[:, self.env_joint_target_indices]
        if self.object_type is not None:
            obs_dict["object_joint_pos"] = joint_q[
                :,
                self.object_joint_start : self.object_joint_start + self.object_num_joint_q,
            ]
            obs_dict["object_lin_vel"] = joint_qd[:, self.object_joint_start : self.object_joint_start + 3]
            obs_dict["object_ang_vel"] = joint_qd[:, self.object_joint_start + 3 : self.object_joint_start + 6]
            obs_dict["goal_joint_pos"] = self.goal_joint_pos.view(self.num_envs, -1)

        if self.grasp is not None:
            obs_dict["init_hand_qpos"] = to_torch(self.grasp.joint_pos).view(1, -1).repeat(self.num_envs, 1)
        else:
            obs_dict["init_hand_qpos"] = self.start_joint_q[:, self.env_joint_target_indices]
        obs_dict["hand_qpos"] = self.joint_q.view(self.num_envs, -1)[:, self.env_joint_target_indices]
        obs_dict["action"] = self.actions.view(self.num_envs, -1)
        if self.action_type == ActionType.POSITION:
            obs_dict["target_qpos"] = self.actions.view(self.num_envs, -1)
        if self.action_type == ActionType.POSITION_DELTA:
            if self.reward_extras.get("target_qpos") is not None:
                obs_dict["target_qpos"] = self.reward_extras["target_qpos"]
            else:
                hand_qpos = self.start_joint_q[:, self.env_joint_target_indices]
                obs_dict["target_qpos"] = self.compute_position_delta_action(self.actions, hand_qpos)
        self.extras["obs_dict"] = obs_dict
        return obs_dict

    def apply_grasps(self, joint_q, xform):
        # update both joint and shape transforms according to xform
        if self.model.joint_type.numpy()[0] == wp.sim.JOINT_FREE:
            # if hand base joint is free aka floating
            print("assigning joint q to set hand base xform")
            base_pos = joint_q[:, self.hand_joint_start : self.hand_joint_start + 3]
            base_pos = tu.tf_apply(xform[:, 3:], xform[:, :3], base_pos)
            joint_q[:, self.hand_joint_start : self.hand_joint_start + 3] = base_pos
            joint_q[:, self.hand_joint_start + 3 : self.hand_joint_start + 7] = xform[:, 3:]
        else:
            # TODO: Implement as a warp kernel in warp_utils
            # else if hand base is either fixed or D6
            # apply transforms to the shape bodies where parent body is -1 (i.e. world frame)
            print("assigning shape_transform to hand base xform")
            shape_body = self.model.shape_body.numpy()
            shape_transform = self.model.shape_transform.numpy()
            grasp_idx = 0

            for s, b in enumerate(shape_body):
                # only affects hand shapes attached to world frame
                if b == -1 and s % self.env_num_shapes < self.hand_num_shapes:
                    # apply offset transform to root bodies
                    shape_xform = to_torch(shape_transform[s], device=self.device)
                    shape_quat_new, shape_pos_new = tu.tf_combine(
                        xform[grasp_idx, 3:],
                        xform[grasp_idx, :3],
                        shape_xform[3:],
                        shape_xform[:3],
                    )
                    shape_transform[s] = torch.cat([shape_pos_new, shape_quat_new]).cpu().numpy()
                    grasp_idx = s // self.env_num_shapes
            self.model.shape_transform.assign(shape_transform)

            # apply transform to the joint_X_p where parent body is -1 (i.e. world frame)
            print("assigning joint_X_p to hand base xform")
            joint_X_p = self.model.joint_X_p.numpy()
            joint_parent = self.model.joint_parent.numpy()
            grasp_idx = 0
            for i in range(len(joint_X_p)):
                if i % self.env_joint_count < self.hand_joint_count and joint_parent[i] == -1:
                    joint_p_quat, joint_p_pos = tu.tf_combine(
                        xform[grasp_idx, 3:],
                        xform[grasp_idx, :3],
                        *[to_torch(x) for x in [joint_X_p[i][3:], joint_X_p[i][:3]]],
                    )
                    joint_X_p[i] = torch.cat([joint_p_pos, joint_p_quat]).cpu().numpy()
                    grasp_idx = i // self.env_joint_count
            # joint_X_p = self.model.joint_X_p.numpy().reshape(self.num_envs, -1, 7)
            # joint_X_p[:, 0] = xform
            self.model.joint_X_p.assign(joint_X_p)
        return joint_q

    def sample_grasps(self, env_ids, joint_q, joint_qd):
        grasp_joint_q = self.start_joint_q[:, self.env_joint_target_indices].clone()

        if self.grasps is None:
            return
        if env_ids is not None:
            num_envs = len(env_ids)
        else:
            env_ids = np.arange(self.num_envs)
            num_envs = self.num_envs
        sampled_grasps = [self.grasps[i] for i in np.random.randint(len(self.grasps), size=num_envs)]
        grasp_xform = to_torch(
            np.stack([sampled_grasp.xform for sampled_grasp in sampled_grasps], axis=0),
            device=self.device,
        )
        if self.object_model.base_joint is not None:
            # apply object base xform to grasp xform
            object_base_joint = to_torch(self.model.joint_X_p.numpy()[self.joint_name_to_idx["base_joint"]][env_ids])
            ori, pos = tu.tf_combine(
                object_base_joint[:, 3:],
                object_base_joint[:, :3],
                grasp_xform[:, 3:],
                grasp_xform[:, :3],
            )
        else:
            pos, ori = grasp_xform[:, :3], grasp_xform[:, 3:]

        # hand_init_xform = torch.cat([pos, ori], dim=-1)
        self.hand_init_q = hand_init_q = to_torch(
            np.stack([sampled_grasp.joint_pos for sampled_grasp in sampled_grasps], axis=0),
            device=self.device,
        )
        grasp_joint_q[env_ids] = hand_init_q

        joint_q[:, self.env_joint_target_indices] = grasp_joint_q
        # joint_q = self.apply_grasps(joint_q, hand_init_xform)
        return joint_q, joint_qd

    def get_stochastic_init(self, env_ids, joint_q, joint_qd):
        # need to set the base joint of each env to sampled grasp xform
        # then set each joint target pos to grasp.

        joint_q, joint_qd = joint_q[env_ids], joint_qd[env_ids]
        if self.grasps:
            joint_q, joint_qd = self.sample_grasps(env_ids, joint_q, joint_qd)

        return joint_q, joint_qd

    def reset(self, env_ids=None, force_reset=True):
        if self.grasps is not None and self.stochastic_init:
            self.sample_grasps(env_ids)
        return super().reset(env_ids=env_ids, force_reset=force_reset)

    def remesh(self, builder):
        remeshed_path = os.path.join(os.path.split(__file__)[0], "assets", "remeshed", self.hand_type.name)
        if self.object_type:
            remeshed_path += "_" + self.object_type.name
            if self.object_id:
                remeshed_path += "_" + str(self.object_id)
        remeshed_path += ".npz"
        if os.path.exists(remeshed_path):
            print(f"loading mesh_data from {remeshed_path}")
            mesh_data = np.load(remeshed_path, allow_pickle=True)
        else:
            mesh_data = None

        mesh_obj = {}
        for i, mesh in enumerate(builder.shape_geo_src):
            if isinstance(mesh, wp.sim.Mesh):
                if mesh_data is None or hash(mesh) != mesh_data.get(f"hash-{i}"):
                    # force remesh if any mesh hash does not match
                    mesh_data = None
                    mesh_obj[f"hash-{i}"] = hash(mesh)
                    mesh.remesh(visualize=False)
                    mesh_obj[f"vertices-{i}"] = mesh.vertices
                    mesh_obj[f"indices-{i}"] = mesh.indices
                    mesh_obj[f"mass-{i}"] = mesh.mass
                    mesh_obj[f"com-{i}"] = mesh.com
                    mesh_obj[f"I-{i}"] = mesh.I
                else:
                    mesh.vertices, mesh.indices = (
                        mesh_data[f"vertices-{i}"],
                        mesh_data[f"indices-{i}"],
                    )
                    mesh.mass, mesh.com, mesh.I = (
                        mesh_data[f"mass-{i}"],
                        mesh_data[f"com-{i}"],
                        mesh_data[f"I-{i}"],
                    )

        if mesh_data is None:
            print("saving remeshed data to ", remeshed_path)
            np.savez(remeshed_path, **mesh_obj)

    def create_articulation(self, builder):
        self.builder = builder

        if self.hand_type == HandType.ALLEGRO:
            bu.create_allegro_hand(
                builder,
                self.action_type,
                stiffness=self.hand_stiffness,
                damping=self.hand_damping,
                base_joint=self.base_joint,
                hand_start_position=self.hand_start_position,
                hand_start_orientation=self.hand_start_orientation,
                collapse_joints=self.collapse_joints,
                floating_base=self.floating_hand,
            )
        elif self.hand_type == HandType.SHADOW:
            bu.create_shadow_hand(
                builder,
                self.action_type,
                stiffness=self.hand_stiffness,
                damping=self.hand_damping,
                base_joint=self.base_joint,
                hand_start_position=self.hand_start_position,
                hand_start_orientation=self.hand_start_orientation,
                collapse_joints=self.collapse_joints,
                floating_base=self.floating_hand,
            )
        else:
            raise NotImplementedError("Hand type not supported:", self.hand_type)

        self.hand_joint_names = builder.joint_name[:]
        self.hand_joint_count = builder.joint_count
        self.hand_num_bodies = builder.body_count
        self.hand_num_shapes = builder.shape_count
        self.num_joint_q += len(builder.joint_q)  # num_joint_q is equivalent to joint_axis_count
        self.num_joint_qd += len(builder.joint_qd)
        valid_joint_types = supported_joint_types[self.action_type]
        hand_env_joint_mask = list(
            map(
                lambda x: x[0],
                filter(lambda x: x[1] in valid_joint_types, enumerate(builder.joint_type)),
            )
        )
        if len(hand_env_joint_mask) > 0:
            joint_indices = []
            for i in hand_env_joint_mask:
                joint_start, axis_count = (
                    builder.joint_q_start[i],
                    joint_coord_map[builder.joint_type[i]],
                )
                # joint_start, axis_count = builder.joint_axis_start[i], joint_coord_map[builder.joint_type[i]]
                joint_indices.append(np.arange(joint_start, joint_start + axis_count))
            joint_indices = np.concatenate(joint_indices)
        else:
            joint_indices = []

        if self.object_type:
            object_kwargs = dict()
            if self.grasp:
                object_kwargs["base_joint"] = "rx, ry, rz, px, py, pz"
            object_articulation_builder = builder  # wp.sim.ModelBuilder()
            if self.object_type is not ObjectType.REPOSE_CUBE:
                object_kwargs["use_mesh_extents"] = self.object_type not in [ObjectType.SPRAY_BOTTLE]
            super().create_articulation(object_articulation_builder, remesh=False, **object_kwargs)
            # self.object_num_joint_axis = object_articulation_builder.joint_axis_count - self.hand_num_joint_axis
            # self.object_num_joint_axis = object_articulation_builder.joint_axis_count
            # self.object_joint_start = self.hand_num_joint_axis
            # self.asset_builders.insert(0, object_articulation_builder)

        try:
            self.remesh(builder)
        except ImportError as e:
            print("Skipped remeshing due to ImportError:", e)
            print("Likely due to missing wildmeshing dependency")
            pass

        self.env_num_shapes = builder.shape_count
        self.env_joint_count = builder.joint_count
        self.env_joint_mask = np.array(hand_env_joint_mask)
        self.env_joint_target_indices = joint_indices
        self.hand_joint_start = joint_indices[0]
        assert self.num_acts == len(joint_indices), "num_act must match number of joint control indices"
        self.env_num_joints = len(joint_indices)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--hand_type", type=str, default="allegro")
    parser.add_argument("--action_type", type=str, default="position")
    parser.add_argument("--object_type", type=str, default=None)
    parser.add_argument("--object_id", type=int, default=None)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--norender", action="store_false", dest="render")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--num_obs", type=int, default=36)
    parser.add_argument("--episode_length", type=int, default=1000)
    parser.add_argument("--stiffness", type=float, default=5000.0)
    parser.add_argument("--damping", type=float, default=10.0)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--grasp_file", type=str, default="")
    parser.add_argument("--grasp_id", type=int, default=-1)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--num_steps", default=100, type=int)
    parser.set_defaults(render=True)

    args = parser.parse_args()
    if args.debug:
        wp.config.mode = "debug"
        wp.config.print_launches = True
        wp.config.verify_cuda = True

    if args.object_type is None:
        object_type = None
    else:
        object_type = ObjectType[args.object_type.upper()]

    rew_params = {"hand_joint_pos_err": (l1_dist, ("target_qpos", "hand_qpos"), 1.0)}
    HandObjectTask.profile = args.profile

    env = HandObjectTask(
        args.num_envs,
        args.num_obs,
        args.episode_length,
        action_type=ActionType[args.action_type.upper()],
        object_type=object_type,
        object_id=args.object_id,
        hand_type=HandType[args.hand_type.upper()],
        render=args.render,
        load_grasps=args.grasp_file,
        grasp_id=args.grasp_id,
        stiffness=args.stiffness,
        damping=args.damping,
        reward_params=rew_params,
        headless=args.headless,
    )
    if args.headless and args.render:
        from .wrappers import Monitor

        env = Monitor(env, "outputs/videos")
    run_env(env, num_steps=args.num_steps)
    env.close()
