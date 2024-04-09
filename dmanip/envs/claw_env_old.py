from collections import OrderedDict
import math

import numpy as np
import os.path as osp
from shac.utils import torch_utils as tu
import torch

import warp as wp
import warp.sim
from typing import Tuple

from ..utils import warp_utils as wpu
from ..utils import builder as bu
from ..utils.autograd import forward_simulate 
from ..utils.common import ORIENTATION_GOAL_TYPES, POSITION_GOAL_TYPES, ActionType, IntegratorType, ObjectType, GoalType, RewardType, clear_state_grads, to_numpy
from ..utils.common import joint_coord_map, supported_joint_types
from .environment import RenderMode
from .warp_env import WarpEnv
from .obj_env import ObjectTask

np.set_printoptions(precision=3)
sigmoid = lambda z: 1 / (1 + np.exp(-z))
CLAW_OBS_DICT = {
        GoalType.ORIENTATION: 29,
        GoalType.POSITION: 31,
        GoalType.POSE: 32,
        GoalType.TRAJECTORY_ORIENTATION: 29,
        GoalType.TRAJECTORY_POSITION: 31,
        GoalType.TRAJECTORY_ORIENTATION_TORQUE: 31,
        GoalType.TRAJECTORY_POSITION_FORCE: 32,
    }


class ClawWarpEnvOld(WarpEnv):
    integrator_type: IntegratorType = IntegratorType.EULER
    body_names = [
        "root",
        "right_finger",
        "proximal_body_right_finger",
        "distal_body_right_finger",
        "left_finger",
        "proximal_body_left_finger",
        "distal_body_left_finger",
        "object",
    ]
    num_rigid_contacts_per_env_mesh = 4100
 
    # c_act, c_finger, c_q, c_pos, c_f/t
    reward_coefs = np.array([1e-3, 0.2, 10, 0, 0], dtype=np.float32)
    capture_graph = False
    sub_length = 16

    sim_substeps_euler = 24
    sim_substeps_xpbd = 5
    termination_threshold = np.pi * 3 / 4

    def __init__(
        self,
        num_envs,
        episode_length=500,
        action_type: ActionType = ActionType.POSITION,
        object_type: ObjectType = ObjectType.OCTPRISM,
        goal_type: GoalType = GoalType.ORIENTATION,
        reward_type: RewardType = RewardType.DELTA,
        seed=0,
        no_grad=True,
        render=True,
        stochastic_init=True,
        device="cuda",
        action_strength: float = 10.,
        goal_path: str = None,
        render_mode=RenderMode.OPENGL,
        logdir=None,
        object_id=0,
        debug=False,
        integrator_type=IntegratorType.EULER
    ):
        self.integrator_type = integrator_type
        num_obs = CLAW_OBS_DICT[goal_type]
        if action_type is ActionType.TORQUE:
            num_act = 4
        elif action_type is ActionType.POSITION:
            num_act = 4
        elif action_type is ActionType.VARIABLE_STIFFNESS:
            num_act = 8
        self.state_tensors = OrderedDict()
        super().__init__(
            num_envs=num_envs,
            num_obs=num_obs,
            num_act=num_act,
            render_mode=render_mode,
            episode_length=episode_length,
            no_grad=no_grad,
            seed=seed,
            render=render,
            stochastic_init=stochastic_init,
            device=device,
            env_name="ClawEnv",
        )
        self.goal_type = goal_type
        self.reward_type = reward_type
        self.action_type = action_type
        if action_type in [ActionType.POSITION, ActionType.VARIABLE_STIFFNESS]:
            self.action_strength = 1.0
        else:
            print("action strength:", action_strength)
            self.action_strength = action_strength
        self.debug = debug
        self.prev_steps = []
        self.object_type = object_type
        self.object_id = object_id
        self.init_sim()
        self.setup_autograd_vars()
        self.state_list = self.simulate_params.get("state_list", None)
        self.post_finalize()

        # load goal trajectory if necessary
        self.goal_path = goal_path
        self.goal_trajectory = None
        if "trajectory" in self.goal_type.name.lower():
            self.load_goal_trajectory(goal_path)

        self.calculate_reward_graph = None
        self.update_joints_graph = None
        self.forward_sim_graph = None
        self.rew_buf = torch.zeros(self.num_envs, device=self.device)

        fid = np.stack(
            [
                self.body_name_to_idx["distal_body_left_finger"],
                self.body_name_to_idx["distal_body_right_finger"],
            ],
            axis=1,
        ).reshape(-1)
        fid = wp.array(
            fid,
            device=self.device,
            dtype=int,
        )
        oid = wp.array(self.body_name_to_idx["object"], device=self.device, dtype=int)
        oqid = wp.array(self.joint_name_to_idx["obj_joint"], device=self.device, dtype=int)
        # joint_type_new = self.model.joint_type.numpy()[:]
        # for i, _ in enumerate(joint_type_new):
        #     if i in self.joint_name_to_idx["obj_joint"]:
        #         joint_type_new[i] = wpu.JOINT_REVOLUTE_TIGHT

        base_id = wp.array(self.body_name_to_idx["root"], device=self.device, dtype=int)
        self.goal_state_vars = {
            "num_fingers": 2,
            "offset": wp.vec3(0.225 * 2, 0.0, 0.0),
            "object_thickness": 0.155,
            "oid": oid,
            "oqid": oqid,
            "base_id": base_id,
            "fid": fid,
            "rotation_count": wp.array(np.zeros(self.num_envs), dtype=float, device=self.device),
            "start_joint_q": wp.from_torch(self.start_joint_q.flatten()),
            "start_joint_qd": wp.from_torch(self.start_joint_qd.flatten()),
            "reward_coefs": wp.array(self.reward_coefs, dtype=float, device=self.device, requires_grad=False),
            "goal_type": self.goal_type,
            "reward_type": self.reward_type,
        }

        # Create state array buffers and graph capture instances
        # if self.requires_grad:
            # self.state_0 = self.model.state(requires_grad=True)
            # self.state_1 = self.model.state(requires_grad=True)
            # if self.capture_graph:
            # self.state_list = [self.model.state() for _ in range(self.sim_substeps - 1)]
            # else:
            #     self.state_list = None
        if not self.requires_grad:
            self.state_list = [self.model.state() for _ in range(self.sim_substeps - 1)]
            # self.state_list = None
            # if self.capture_graph:
            # call allocate and collide once
            wp.sim.collide(self.model, self.state_0)

        self.warp_body_f = wp.zeros_like(self.state_0.body_f)
        self.body_f = None
        # -----------------------
        # set up Usd renderer
        if self.visualize:
            self.initialize_renderer(logdir)
        self.obs_buf_before_reset = None

    @property
    def goal_step(self):
        return self.goal_trajectory_idx[self.num_frames]

    @property
    def goal_state(self):
        """Returns full goal state, including trajectory state"""
        if "trajectory" in self.goal_type.name.lower():
            return self.goal_trajectory[self.goal_step]
        return self._goal_state

    def init_sim(self):
        self.num_joint_q = self.num_joint_qd = 0
        super().init_sim()
        # create custom warp buffers for actions, rewards
        # actions buffer in warp to use for action assignment
        self.warp_actions = wp.array(
            np.zeros(self.num_envs * self.num_actions),
            dtype=float,
            device=self.device,
            requires_grad=self.requires_grad,
        )

        self.reward_ctrl = wp.array(
            np.zeros(self.num_envs),
            dtype=float,
            device=self.device,
            requires_grad=self.requires_grad,
        )
        self.reward_total = wp.array(
            np.zeros(self.num_envs),
            dtype=float,
            device=self.device,
            requires_grad=self.requires_grad,
        )
        # shape: (num_envs, 5) for 5 reward components, feeds into reward_total
        self.reward_vec = wp.array(
            np.zeros(self.num_envs * 5),
            dtype=float,
            device=self.device,
            requires_grad=self.requires_grad,
        )
        self.reward_vec_prev = wp.array(
            np.zeros(self.num_envs * 5),
            dtype=float,
            device=self.device,
            requires_grad=self.requires_grad,
        )

        self.model.allocate_rigid_contacts(requires_grad=self.requires_grad)
        wp.sim.collide(self.model, self.state_0)

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

    def create_articulation(self, builder):
        self.builder = builder
        # aux root joint, 4 finger joints, 1 object joint
        object_free_joint = self.goal_type in POSITION_GOAL_TYPES

        self.num_robot_joint_q = 5
        self.num_obj_joint_q = 7 * int(object_free_joint) + (1 - object_free_joint)
        self.num_joint_q = self.num_robot_joint_q + self.num_obj_joint_q

        model_file = "claw_only_warp.xml"
        model_path = osp.join(osp.dirname(__file__), "assets", model_file)

        # assigns joint target_ke and depending on if action type is pos/torque
        stiffness = 0.0 if self.action_type is ActionType.TORQUE else 10.0
        damping = 0.0 if self.action_type is ActionType.TORQUE else 1.5
        contact_ke = 1.0e3
        contact_kd = 100.0
        if self.integrator_type is IntegratorType.EULER:
            armature = 0.07
            armature_scale = 10.0
        else:
            armature = 0.0
            armature_scale = 1.0
        wp.sim.parse_mjcf(
            model_path,
            builder,
            density=100.0,
            stiffness=stiffness,
            damping=damping,
            contact_ke=contact_ke,
            contact_kd=contact_kd,
            contact_kf=100,
            contact_mu=0.5,
            limit_ke=100,
            limit_kd=10,
            armature=armature,  # comment if using XPBD
            armature_scale=armature_scale,  # comment if using XPBD
            # parse_meshes=True,
            enable_self_collisions=True,
        )
        if self.goal_type in ORIENTATION_GOAL_TYPES:
            builder.joint_q[-5:-1] = [0.824, -1.365, -0.824, 1.365]
        else:
            builder.joint_q[-12:-8] = [0.824, -1.365, -0.824, 1.365]

        self.hand_joint_names = [name for name in builder.joint_name[:] if 'finger' in name]
        self.hand_joint_mask = [i for i, name in enumerate(builder.joint_name[:]) if 'finger' in name]
        self.hand_joint_count = builder.joint_count
        self.hand_num_bodies = builder.body_count
        self.hand_num_shapes = builder.shape_count
        # self.num_joint_q += len(builder.joint_q)  # num_joint_q is equivalent to joint_axis_count
        # self.num_joint_qd += len(builder.joint_qd)
        valid_joint_types = supported_joint_types[self.action_type]
        hand_env_joint_mask = list(
            map(lambda x: x[0], filter(lambda x: x[1] in valid_joint_types, enumerate(builder.joint_type)))
        )

        # load object mesh
        pos = (0.8, 0.0, 0.0)  # Z-up, negative when rotated
        ori = (0.0, 0.0, 0.0, 1.0)
        joint_type = wp.sim.JOINT_FREE
        if self.goal_type in ORIENTATION_GOAL_TYPES:
            joint_type = wp.sim.JOINT_REVOLUTE  # JOINT_REVOLUTE_TIGHT
        body_link = bu.add_joint(builder, pos, ori, joint_type)
        bu.add_object(
            builder, body_link, self.object_type, contact_ke=contact_ke, contact_kd=contact_kd, scale=1.0
        )

        if not self.stochastic_init:
            self._goal_state = torch.as_tensor(
                np.array([pos[0], pos[1], 0.3]),
                device=self.device
                # np.array([pos[0], pos[1], wpu.yaw_from_quat(ori)]), device=self.device
            )
        else:
            if self.goal_type in POSITION_GOAL_TYPES:
                # pull object closer
                self._goal_state = torch.as_tensor(np.array([pos[0] - 0.1, pos[1], 0.0]), device=self.device)
            elif self.goal_type in ORIENTATION_GOAL_TYPES:
                # rotate object in place
                self._goal_state = torch.as_tensor(np.array([pos[0], pos[1], 0.0]), device=self.device)
        self.builder.gravity = 0.0

        if "mesh" in self.object_type.name:
            rigid_contact_max = self.num_rigid_contacts_per_env_mesh
        else:
            rigid_contact_max = None

        self.builder.num_rigid_contacts_per_env = rigid_contact_max

    def post_finalize(self):
        if self.debug:
            print(
                "num contacts:",
                self.model.rigid_contact_max,
                "count_contacts:",
                self.model.count_contact_points(),
            )
            print(f"after model.find_shape_contact_pairs(), {self.model.shape_contact_pair_count} contacts counted")
            print(f"{self.model.shape_collision_filter_pairs}")
            print(f"{self.model.shape_contact_pairs.numpy()}")
        self.joint_limit_lower = wp.to_torch(self.model.joint_limit_lower).view(self.num_envs, -1)[
            :, 1:5
        ]  # removes base and object joints
        self.joint_limit_upper = wp.to_torch(self.model.joint_limit_upper).view(self.num_envs, -1)[:, 1:5]
        self.model.ground = True
        if self.integrator_type == IntegratorType.EULER:
            self.model.joint_attach_ke = 100 # 1e3
            self.model.joint_attach_kd = 50 # 1e2
        else:
            self.model.joint_attach_ke = 100
            self.model.joint_attach_kd = 10

        # create mappings for joint and body indices
        self.body_name_to_idx, self.joint_name_to_idx = {}, {}
        for i, body_name in enumerate(self.model.body_name):
            body_ind = self.body_name_to_idx.get(body_name, [])
            body_ind.append(i)
            self.body_name_to_idx[body_name] = body_ind

        for i, joint_name in enumerate(self.model.joint_name):
            joint_ind = self.joint_name_to_idx.get(joint_name, [])
            joint_ind.append(i)
            self.joint_name_to_idx[joint_name] = joint_ind

        self.body_name_to_idx = {k: np.array(v) for k, v in self.body_name_to_idx.items()}
        self.joint_name_to_idx = {k: np.array(v) for k, v in self.joint_name_to_idx.items()}

        joint_axis_start = self.model.joint_axis_start.numpy()
        joint_types = self.model.joint_type.numpy()
        joint_target_indices = np.concatenate(
            [
                np.arange(joint_idx, joint_idx + joint_coord_map[joint_type])
                for joint_idx, joint_type in zip(joint_axis_start, joint_types)
            ]
            ).reshape(self.num_envs, -1)[:, self.hand_joint_mask].flatten()
        self.joint_target_indices = wp.array(joint_target_indices, dtype=int, device=self.device)

        # initialize integrator, states, and starting joint states
        if self.integrator_type is IntegratorType.EULER:
            self.integrator = wp.sim.SemiImplicitIntegrator()
        else:
            self.integrator = wp.sim.XPBDIntegrator(
                iterations=2, rigid_contact_relaxation=0.8
            )
        start_joint_q = wp.to_torch(self.model.joint_q)
        start_joint_qd = wp.to_torch(self.model.joint_qd)
        joint_X_p = wp.to_torch(self.model.joint_X_p)

        # Buffers copying initial state, with env batch dimension
        self.start_joint_q = start_joint_q.clone().view(self.num_envs, -1)
        self.start_joint_qd = start_joint_qd.clone().view(self.num_envs, -1)
        self.start_joint_X_p = joint_X_p.clone().view(self.num_envs, -1, 7)
        obj_q = self.start_joint_X_p[:, -1, [0, 2]]
        obj_theta = torch.zeros((self.num_envs, 1), device=self.device)
        self.start_obj_state = torch.cat([obj_q, obj_theta], dim=1)
        self.start_joint_stiffness = wp.clone(self.model.joint_target_ke)
        self.start_joint_stiffness_pt = wp.to_torch(self.start_joint_stiffness).view(self.num_envs, -1)

        # Buffers to compute robot joint pos and vel
        self._joint_q = wp.zeros_like(self.model.joint_q)
        self._joint_qd = wp.zeros_like(self.model.joint_qd)
        self.joint_q = None
        self.log = []

    def initialize_trajectory(self):
        self.clear_grad()
        self.calculateObservations()
        return self.obs_buf

    def load_goal_trajectory(self, goal_path=None, env_ids=None):
        if env_ids is None:
            env_ids = np.arange(self.num_envs)
        if goal_path is not None and osp.exists(goal_path):
            # loads goal trajectory (obj_force, obj_pos), shape: (T, 2)
            gt = torch.tensor(np.load(goal_path), device=self.device)
        elif (
            self.goal_type is GoalType.TRAJECTORY_ORIENTATION
            or self.goal_type is GoalType.TRAJECTORY_ORIENTATION_TORQUE
        ):
            start = self.start_obj_state[:, 2].cpu().numpy()
            end = self._goal_state[2].cpu().numpy()
            gt = sigmoid(np.linspace(-5, 5, 100))[:, None] @ (end - start) + start
            # (T, n_envs)
            self.friction_out = wp.array(
                np.zeros(self.num_envs * 5),
                dtype=float,
                device=self.device,
                requires_grad=self.requires_grad,
            )
        elif self.goal_type is GoalType.TRAJECTORY_POSITION:
            start, end = self.start_obj_state, self._goal_state
            gt_x = sigmoid(np.linspace(-5, 5, 100))[:, None] @ (end[0] - start[:, 0]) + start[:, 0]
            gt_y = sigmoid(np.linspace(-5, 5, 100))[:, None] @ (end[1] - start[:, 1]) + start[:, 1]
            # goal_x = np.linspace(start[:, 0], end[0], self.episode_length)
            # goal_y = np.linspace(
            #     start[:, 1], end[1], self.episode_length
            # )  # (T, n_envs)
            gt = np.stack([gt_x, gt_y], axis=-1)  # (T, n_envs, 2)

        if self.goal_trajectory is None:
            self.goal_trajectory = gt
        elif len(gt.shape) > 1 and gt.shape[1] == self.num_envs:
            self.goal_trajectory[:, env_ids] = gt[:, env_ids]

        # setups goal index to map goals to current timestep
        if goal_path is not None and "trajectory" in self.goal_type.name.lower():
            assert not self.stochastic_init, "stochastic init not supported for trajectory goals, goal_type: {}".format(
                self.goal_type.name
            )
            self.goal_trajectory_idx = np.array(
                [math.floor(x) for x in np.linspace(0, self.goal_trajectory.shape[0] - 1, self.episode_length + 1)]
            )
            self.goal_trajectory_idx = np.clip(
                (self.goal_trajectory_idx // self.sub_length + 1) * self.sub_length,
                0,
                self.goal_trajectory_idx[-1],
            )

    def _get_goal_state(self, goal_type):
        # TODO: select goals based on: goal_idx, timestep, closeset goal, ...
        # coefs scale components of reward: act penalty, goal dist,
        goal_pos = np.zeros(2)
        goal_ori = 0.0
        goal_force = np.zeros(2)
        goal_torque = 0.0
        goal_state = self.goal_state
        if isinstance(goal_state, torch.Tensor):
            goal_state = to_numpy(goal_state)
        if goal_type.name.startswith("TRAJ"):
            if goal_type in ORIENTATION_GOAL_TYPES:
                if goal_type.name.endswith("TORQUE"):
                    goal_torque, theta = goal_state[..., 0], goal_state[..., 1]
                else:
                    theta = goal_state[0]
                goal_ori = wp.quat_rpy(0, 0, theta)
            elif "position" in goal_type.name.lower():  # position goaltype
                if goal_type.name.endswith("FORCE"):
                    goal_force, goal_pos = goal_state[:2], goal_state[2:]
                else:
                    goal_pos = goal_state
        else:
            goal_pos, theta = goal_state[:2], goal_state[2]
            goal_ori = wp.quat_rpy(0, 0, theta)

        goal_pos = wp.vec3(goal_pos[0], 0.15, goal_pos[1])  # y-up
        goal_force = wp.vec3(goal_force[0], goal_force[1], 0.0)  # y-up
        goal_torque = wp.vec3(0.0, 0.0, goal_torque)  # torque around y-axis
        self.goal_state_vars['reward_coefs'].assign(self.reward_coefs)

        self.goal_state_vars.update(
            {
                "goal_pos": goal_pos,
                "goal_ori": goal_ori,
                "goal_force": goal_force,
                "goal_torque": goal_torque,
            }
        )
        return self.goal_state_vars

    def _get_goal_obs(self):
        """Returns goal observation as torch.Tensor with shape (num_envs, goal_dim)"""
        goal_dict = OrderedDict()
        goal_state = self.goal_state
        if self.goal_type.name.startswith("TRAJ"):
            if self.goal_type in ORIENTATION_GOAL_TYPES:  # orientation goal-type
                if self.goal_type.name.endswith("TORQUE"):
                    goal_dict["goal_ori"] = goal_state[..., [1]]
                    goal_dict["goal_torque"] = goal_state[..., [0]]
                else:
                    goal_dict["goal_ori"] = goal_state[:1]
            elif self.goal_type in POSITION_GOAL_TYPES:  # position goal-type
                if self.goal_type.name.endswith("FORCE"):
                    goal_dict["goal_pos"] = goal_state[2:]
                    goal_dict["goal_force"] = goal_state[:2]
                else:
                    goal_dict["goal_pos"] = goal_state  # goal state is pos only
        else:
            if self.goal_type in POSITION_GOAL_TYPES:  # position goal-type
                goal_dict["goal_pos"] = goal_state[:2]
            elif self.goal_type in ORIENTATION_GOAL_TYPES:  # position goal-type
                goal_dict["goal_ori"] = goal_state[2:]
            elif self.goal_type.name.lower() == "both":
                goal_dict["goal_pos"], goal_dict["goal_ori"] = (
                    goal_state[:2],
                    goal_state[2:],
                )

        goal = torch.cat(
            [torch.as_tensor(goal_dict[k], dtype=torch.float32, device=self.device) for k in goal_dict.keys()]
        )
        if len(goal.shape) != 2:  # repeat goal for all envs if needed
            goal = goal.view(1, -1).repeat(self.num_envs, 1)

        goal = goal.view(self.num_envs, -1)
        return goal

    def _get_joint_pos_vel(self, joint_name, return_vel=False):
        joint_index = self.joint_name_to_idx[joint_name]
        if self.joint_q is None:
            self.update_joints()
        joint_pos = self.joint_q[joint_index]
        if return_vel:
            joint_vel = self.joint_qd[joint_index]
            return joint_pos, joint_vel
        return joint_pos

    def _get_body_pos_vel(self, body_name, return_vel=False, return_quat=False):
        """Warning: adds extra stacking dimension to body_pos and body_vel"""
        body_index = self.body_name_to_idx[body_name]
        assert len(body_index) == self.num_envs
        end = 7 if return_quat else 3
        body_pos = self.body_q[body_index, :end].view(self.num_envs, end)
        if return_vel:
            body_vel = self.body_qd[body_index].view(self.num_envs, 6)
            return body_pos, body_vel
        return body_pos

    def _get_body_f(self, body_name, body_f):
        body_index = self.body_name_to_idx[body_name]
        # returns angular, linear vel
        if self.body_f is None:
            self.body_f = self.to_torch(body_f)
        body_f = self.body_f[body_index, :].reshape(self.num_envs, -1)
        return body_f

    def _get_fingertip_pos(self):
        fingertip_xform1 = self._get_body_pos_vel("distal_body_left_finger", return_quat=True)

        fingertip_xform2 = self._get_body_pos_vel("distal_body_right_finger", return_quat=True)

        fingertip_offset = torch.tensor([[0.225 * 2, 0.0, 0.0] for _ in range(self.num_envs)], device=self.device)
        fingertip_pos1 = tu.tf_apply(fingertip_xform1[:, 3:], fingertip_xform1[:, :3], fingertip_offset)
        fingertip_pos2 = tu.tf_apply(fingertip_xform2[:, 3:], fingertip_xform2[:, :3], fingertip_offset)
        return fingertip_pos1, fingertip_pos2

    def to_torch(self, state_arr, num_frames=None):
        num_frames = num_frames or self.num_frames
        if (state_arr, num_frames) not in self.state_tensors:
            self.state_tensors[(state_arr, num_frames)] = wp.to_torch(state_arr)
        return self.state_tensors[(state_arr, num_frames)]

    def write_log(self):
        # print("Model Root transform:", self.model.joint_X_p.numpy()[0])

        ftip_1, ftip_2 = (
            self.extras["left_fingertip_pos"],
            self.extras["right_fingertip_pos"],
        )
        # prev_reward_total = 0.0
        # if len(self.log) > 0:
        #     prev_reward_total = self.log[-1]["reward_total"] * 0.99
        log_item = {
            "joint_q": to_numpy(self.joint_q).reshape(self.num_envs, -1),
            "joint_qd": to_numpy(self.joint_qd).reshape(self.num_envs, -1),
            "actions": to_numpy(self.actions).reshape(self.num_envs, -1),
            "left_fingertip_pos": to_numpy(ftip_1).reshape(self.num_envs, -1),
            "right_fingertip_pos": to_numpy(ftip_2).reshape(self.num_envs, -1),
            "reward_total": to_numpy(self.rew_buf).reshape(self.num_envs, -1),
            "reward_vec": self.reward_vec.numpy().reshape(self.num_envs, -1),
        }
        if "object" in self.body_name_to_idx:
            obj_pos, obj_vel = (
                self.extras["object_body_pos"],
                self.extras["object_body_vel"],
            )
            obj_pos, obj_vel = to_numpy(obj_pos), to_numpy(obj_vel)
            obj_body_f = to_numpy(self._get_body_f("object", self.warp_body_f))
            log_item.update(
                {
                    "object_body_q": obj_pos,
                    "object_body_qd_ang": obj_vel[..., :3],
                    "object_body_qd_lin": obj_vel[..., 3:],
                    "object_body_f": obj_body_f,
                }
            )

        self.log.append(log_item)
        # Console logging frequency every 10 steps
        if self.num_frames % 10 == 0:
            for k in log_item:
                print(f"{k}: {log_item[k][0, :]}")
        return

    def _get_obs_dict(self):
        # TODO: fix joint representations
        joint_q, joint_qd = (
            self.joint_q.view(self.num_envs, -1),
            self.joint_qd.view(self.num_envs, -1),
        )
        # TODO: handle joint representation if object prismatic joints present
        x = joint_q[:, 1:5]
        xdot = joint_qd[:, 1:5]
        if self.goal_type in ORIENTATION_GOAL_TYPES:
            theta = joint_q[:, 5:6]
            theta_dot = joint_qd[:, 5:6]
        elif self.goal_type in POSITION_GOAL_TYPES:
            theta = joint_q[:, [5, 7]]
            theta_dot = joint_qd[:, [8, 10]]  # x-y vel
        obj_pos, obj_vel = self._get_body_pos_vel("object", return_vel=True)
        obj_f = self._get_body_f("object", self.warp_body_f)[:, 1:2]
        rewards = self.to_torch(self.reward_vec).view(self.num_envs, -1)
        q_err = rewards[:, 2:3]
        pos_err = rewards[:, 3:4]
        force_err = rewards[:, 4:5]

        goal = self._get_goal_obs()

        fingertip_1, fingertip_2 = self._get_fingertip_pos()
        obs_dict = OrderedDict(
            [
                ("robot_joint_pos", x),  # (num_envs, 4)
                ("object_joint_pos", theta),  # (num_envs, 1)
                ("object_body_pos", obj_pos[:, [0, 2]]),  # (num_envs, 2)
                ("left_fingertip_pos", fingertip_1),  # (num_envs, 3)
                ("right_fingertip_pos", fingertip_2),  # (num_envs, 3)
                ("robot_joint_vel", xdot),  # (num_envs, 4)
                ("object_joint_vel", theta_dot),  # (num_envs, 1)
                ("object_body_vel", obj_vel),  #  (num_envs, 6)
                ("object_body_f", obj_f),  #  (num_envs, 1)
                ("object_ori_err", q_err),  #  (num_envs, 1)
                ("object_pos_err", pos_err),  #  (num_envs, 1)
                ("object_force_err", force_err),  #  (num_envs, 1)
                ("goal", goal),  # (num_envs, len(goal_state))
            ]
        )
        self.extras['obs_dict'] = obs_dict
        return obs_dict

    def update_joints(self):
        self._joint_q.zero_()
        self._joint_qd.zero_()
        if self._joint_q.grad is not None:
            self._joint_q.grad.zero_()
            self._joint_qd.grad.zero_()

        if self.capture_graph:
            if self.update_joints_graph is None: 
                wp.capture_begin()
                wp.sim.eval_ik(self.model, self.state_0, self._joint_q, self._joint_qd)
                # self.compute_joint_q_qd()
                self.update_joints_graph = wp.capture_end()
            wp.capture_launch(self.update_joints_graph)
        else:
            wp.sim.eval_ik(self.model, self.state_0, self._joint_q, self._joint_qd)
            # self.compute_joint_q_qd()

        self.joint_q = wp.to_torch(self._joint_q)
        self.joint_qd = wp.to_torch(self._joint_qd)

    def calculateObservations(self):
        obs_dict = self._get_obs_dict()
        self.obs_buf = torch.cat(
            (
                torch.sin(obs_dict['robot_joint_pos']),  # :4
                torch.cos(obs_dict['robot_joint_pos']),  # :4
                obs_dict['object_joint_pos'],  # 4:5
                obs_dict['object_body_pos'],  # 5:7
                obs_dict['left_fingertip_pos'],  # 7:10
                obs_dict['right_fingertip_pos'],  # 10:13
                obs_dict['robot_joint_vel'],  # 13:17
                obs_dict['object_joint_vel'],  # 17:18
                obs_dict['object_body_vel'],  # 18:24
                obs_dict['goal'],  # 24:24+len(goal)
                # self.actions,
                # self.progress_buf[:, None],  # 25+len(goal):26+len(goal)
            ),
            dim=1,
        ).view(self.num_envs, -1)

        # TODO: handle nan errors, currently penalized with negative rew + reset
        try:
            assert self.obs_buf.isnan().sum() == 0, "NaN in observation"
        except AssertionError:
            pass  
            # print("resetting env_ids:", env_ids.nonzero())
        if self.goal_type in ORIENTATION_GOAL_TYPES:
            reset_q = 0.4 if self.goal_type.name.lower().startswith("traj") else 1.25
            self.reset_buf = self.reset_buf | (obs_dict['object_ori_err'].squeeze() > reset_q)
        self.extras.update(obs_dict)
        # observations: [x, theta, obj_pos, xdot, theta_dot, obj_vel]

    def calculate_reward_kernels(self, goal_state_vars):
        wp.launch(
            kernel=wpu.compute_ctrl_reward,
            dim=(self.num_envs, self.num_actions),
            inputs=[self.warp_actions, self.num_actions],
            outputs=[self.reward_ctrl],
            device=self.device,
        )
        wp.launch(
            kernel=wpu.compute_goal_reward,
            inputs=[
                self.state_0.body_q,
                self.warp_body_f,
                self._joint_q,
                self._joint_qd,
                goal_state_vars["fid"],
                goal_state_vars["num_fingers"],
                goal_state_vars["oid"],
                goal_state_vars["oqid"],
                goal_state_vars["offset"],
                goal_state_vars["object_thickness"],
                goal_state_vars["base_id"],
                goal_state_vars["goal_pos"],
                goal_state_vars["goal_ori"],
                goal_state_vars["goal_force"],
                goal_state_vars["goal_torque"],
                goal_state_vars["goal_type"].value,
                goal_state_vars["reward_type"].value,
                self.reward_ctrl,
                goal_state_vars["reward_coefs"],
                goal_state_vars["rotation_count"],
            ],
            outputs=[self.reward_vec, self.reward_vec_prev, self.reward_total],
            dim=self.num_envs,
            device=self.device,
        )
        wp.launch(
            kernel=wpu.joint_limit_penalty,
            inputs=[
                self.state_0.body_q,
                self.model.joint_qd_start,
                self.model.joint_type,
                self.model.joint_parent,
                self.model.joint_X_p,
                self.model.joint_axis,
                self.model.joint_limit_lower,
                self.model.joint_limit_upper,
                6,  # num joints
            ],
            outputs=[self.reward_total],
            dim=self.model.joint_count,
            device=self.model.device,
        )

    def forward_simulate(self, joint_act):
        wpu.assign_act(
            self.warp_actions,
            joint_act,
            self.model.joint_target_ke,
            self.action_type,
            num_acts=self.num_actions,
            num_envs=self.num_envs,
            num_joints=self.num_joint_q,
            q_offset=1,
        )
        state_in = self.state_1
        state_out = self.state_0
        for step in range(self.sim_substeps):
            state_in.clear_forces()
            if not self.capture_graph and self.state_list is None:
                state_in, state_out = state_out, self.model.state()
            else:
                state_in = state_out
                if step == self.sim_substeps - 1:  # final step
                    state_out = self.state_1
                else:
                    state_out = self.state_list[step]
            # updates ground/rigid contacts
            wp.sim.collide(self.model, state_in)
            state_out = self.integrator.simulate(
                self.model,
                state_in,
                state_out,
                self.sim_dt,
            )
        if self.capture_graph:
            self.state_0.body_q.assign(self.state_1.body_q)
            self.state_0.body_qd.assign(self.state_1.body_qd)
        else:
            self.state_0, self.state_1 = self.state_1, self.state_0

        if self.integrator_type == IntegratorType.EULER:
            self.warp_body_f.assign(self.state_0.body_f)
        else:
            wpu.integrate_body_f(self.model, self.state_1.body_qd,
                                 self.state_0.body_q, self.state_0.body_qd,
                                 self.warp_body_f, self.sim_dt * self.sim_substeps)

        wp.sim.eval_ik(self.model, self.state_1, self._joint_q, self._joint_qd)

    def calculateReward(self):
        # computes reward in reward kernel
        self.reward_ctrl.zero_()
        self.reward_total.zero_()
        self.reward_vec_prev.assign(self.reward_vec)
        self.reward_vec.zero_()
        goal_state_vars = self._get_goal_state(self.goal_type)

        if self.capture_graph:
            if self.calculate_reward_graph is None:
                wp.capture_begin()
                self.calculate_reward_kernels(goal_state_vars)
                self.calculate_reward_graph = wp.capture_end()
            wp.capture_launch(self.calculate_reward_graph)
        else:
            self.calculate_reward_kernels(goal_state_vars)

        self.rew_buf = wp.to_torch(self.reward_total)
        reward_vec = self.reward_vec.numpy().copy()
        # reward_total = self.reward_total.numpy().copy()
        # rew_vec = wp.to_torch(self.reward_vec)
        # self.rew_buf = (rew_vec.view(-1, 5) * torch.tensor(self.reward_coefs, device=self.device)).sum(axis=-1)
        rewards = torch.as_tensor(reward_vec, device=self.device).view(self.num_envs, -1)

        self.extras["control_pen"] = rewards[:, 0]
        self.extras["ftip_err"] = rewards[:, 1]
        self.extras["q_err"] = rewards[:, 2]
        self.extras["pos_err"] = rewards[:, 3]
        self.extras["force_err"] = rewards[:, 4]

    def _calculateRewardGoal(self):
        # DEPRECATED: old gc-reward function
        joint_pos = self.extras["object_joint_pos"]
        body_pos = self.extras["object_body_pos"]
        ftip1 = self.extras["left_fingertip_pos"]
        ftip2 = self.extras["right_fingertip_pos"]

        goal_state = torch.as_tensor(self.goal_state).to(self.device)
        if self.goal_type is GoalType.POSITION:
            reward_dist = -torch.linalg.norm(goal_state[:2] - body_pos, axis=1)
        elif self.goal_type is GoalType.ORIENTATION:
            reward_dist = -torch.abs(goal_state[2] - joint_pos)
        elif self.goal_type is GoalType.POSE:
            reward_dist = -torch.linalg.norm(goal_state - torch.cat([body_pos, joint_pos]))
        elif self.goal_type is GoalType.TRAJECTORY_POSITION:
            goal_pos = self.goal_trajectory[self.goal_step]
            reward_dist = -torch.linalg.norm(goal_pos[:2] - body_pos)
        elif self.goal_type is GoalType.TRAJECTORY_ORIENTATION:
            goal_ori = self.goal_trajectory[self.goal_step]
            reward_dist = -torch.linalg.norm(goal_ori - joint_pos)
        elif self.goal_type is GoalType.TRAJECTORY_ORIENTATION_TORQUE:
            # ori and torque info
            goal_ori_torque = self.goal_trajectory[self.goal_step]
            tor = self._get_body_f("object", self.warp_body_f)[:, 1:2]
            ori_torque = torch.cat([joint_pos, tor], dim=1)
            reward_dist = -torch.linalg.norm(goal_ori_torque - ori_torque)
        elif self.goal_type is GoalType.TRAJECTORY_POSITION_FORCE:
            goal_pos_force = self.goal_trajectory[self.goal_step]
            # x-z force (ignore y, remember y-up)
            force = self.to_torch(self.state_1.body_f).view(self.num_envs, -1)[:, [3, 5]]
            pos_force = torch.cat([body_pos, force], dim=1)
            reward_dist = -torch.linalg.norm(goal_pos_force - pos_force)

        reward_near = -torch.linalg.norm(ftip1[:, [0, 2]] - body_pos) - torch.linalg.norm(ftip2[:, [0, 2]] - body_pos)
        reward_ctrl = -torch.linalg.norm(self.actions)
        reward_total = reward_dist + 0.1 * reward_ctrl + 0.5 * reward_near
        self.rew_buf = reward_total.view(self.num_envs)
        reward_dict = OrderedDict(
            [
                ("reward_dist", reward_dist),
                ("reward_near", reward_near),
                ("reward_ctrl", reward_ctrl),
                ("reward_total", reward_total),
            ]
        )
        self.extras.update(reward_dict)

    def get_stochastic_init(self, env_ids, joint_q, joint_qd):
        joint_X_p = self.start_joint_X_p.clone().view(self.num_envs, -1, 7)

        n = len(env_ids)
        if self.goal_type in POSITION_GOAL_TYPES:
            # object should have free joint
            assert joint_q.shape == (n, self.num_joint_q)
            # move closer to origin, with random L/R z translation
            start_obj_state = torch.stack(
                [
                    -torch.rand(n, device=self.device) * 0.2,  # x-translation
                    torch.rand(n, device=self.device) * 0.2 - 0.1,  # z-translation
                    torch.zeros(n, device=self.device),  # theta
                ],
                dim=1,
            )
            obj_Xform_w = torch.cat(
                [
                    start_obj_state[env_ids, 0:1],
                    start_obj_state[env_ids, 1:2],
                    torch.zeros_like(start_obj_state[env_ids, 0:1], device=self.device),
                    torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=self.device).repeat((n, 1)),
                ],
                dim=1,
            )
            # TODO: hardcoded object index as last joint index
            # obj_X_p = joint_q[:, -7:]  # obj parent link transform
            # obj_Xform_r, obj_Xform_t = tu.tf_combine(
            #     obj_Xform_w[:, 3:], obj_Xform_w[:, :3], obj_X_p[:, 3:], obj_X_p[:, :3]
            # )
            joint_q[env_ids, -7:] = obj_Xform_w
        else:
            # rand_orientation = torch.rand((n, 1), device=self.device) * 0.2 + 0.1
            # obj_orientation = (
            #     to_torch(np.random.choice([-1, 1], n)[:, None], self.device)
            #     * rand_orientation
            # )
            obj_orientation = (torch.rand((n, 1), device=self.device) - 0.5) * np.pi# *3/4
            start_obj_state = torch.cat(
                [
                    joint_X_p[env_ids, -1, 0:1],
                    joint_X_p[env_ids, -1, 2:3],
                    obj_orientation,
                ],
                dim=1,
            )
            joint_q[env_ids, -1] = start_obj_state[:, 2]
        if self.debug:
            print(
                f"Stochastic init- Starting position: ",
                start_obj_state[:10, :2],
                "Starting orientation: ",
                start_obj_state[:10, 2:],
            )

        # object state contains x, y, theta (orientation)
        self.start_obj_state = start_obj_state
        return joint_q[env_ids], joint_qd[env_ids]

    def reset(self, env_ids=None, force_reset=True):
        self.state_tensors.clear()

        # creating state_1 if no_grad enabled for warp_step and calculateObservation
        # if not self.capture_graph and not self.requires_grad:
        #     self.state_0 = self.model.state()
        #     self.state_1 = self.model.state()

        super().reset(env_ids, force_reset)  # resets state_0 and joint_q
        if "trajectory" in self.goal_type.name.lower():
            self.load_goal_trajectory(self.goal_path, env_ids)
        self.update_joints()
        self.warp_actions.zero_()
        self.reward_vec_prev.zero_()
        self.goal_state_vars["rotation_count"].zero_()
        self.warp_body_f.zero_()
        self.initialize_trajectory()  # sets goal_trajectory & obs_buf
        self.log.clear()
        self.prev_steps = []
        return self.obs_buf

    def clear_grad(self, checkpoint=None):
        self.update_joints()
        super().clear_grad(checkpoint)
        if hasattr(self, "body_q"):
            with torch.no_grad():
                self.joint_q = self.joint_q.clone()
                self.joint_qd = self.joint_qd.clone()
                self.body_q = self.body_q.clone()
                self.body_qd = self.body_qd.clone()
                self.actions = self.actions.clone()
                self.rew_buf = self.rew_buf.clone() * 0
                self.warp_actions.zero_()
                self.warp_body_f.zero_()
                if self.requires_grad:
                    self.warp_actions.grad.zero_()
                    self.warp_body_f.grad.zero_()
                    self.reward_ctrl.grad.zero_()
                    self.reward_total.grad.zero_()
                    self.reward_vec.grad.zero_()
                    self.reward_vec_prev.grad.zero_()
                    clear_state_grads(self.state_0)
                    clear_state_grads(self.state_1)
                    if self.state_list is not None:
                        [clear_state_grads(s) for s in self.state_list]

    def assign_actions(self, actions):
        if self.action_type is ActionType.TORQUE:
            actions = torch.clamp(actions, -1, 1.0) * self.action_strength
        elif self.action_type is ActionType.POSITION:
            jlu, jll = self.joint_limit_upper, self.joint_limit_lower
            actions = (jll - jlu) * actions
            # q = self.joint_q.view(self.num_envs, -1)[:, 1:5]
            actions = torch.clamp(actions, jll, jlu)
        else:
            jlu, jll = self.joint_limit_upper, self.joint_limit_lower
            act_idx = int(self.num_actions / 2)
            actions[:, :act_idx] *= jlu - jll
            # q = self.joint_q.view(self.num_envs, -1)[:, 1:5]
            actions[:, :act_idx] = torch.clamp(actions[:, :act_idx], jll, jlu)
            # assign joint_stiffnesses to add to current joint stiffness
            actions[:, act_idx:] *= 0.25 * self.start_joint_stiffness_pt[:, 1 : 1 + self.num_acts // 2]
            self.model.joint_target_ke.assign(self.start_joint_stiffness)

    def assign_scale_actions(self, actions):
        if self.action_type is ActionType.TORQUE:
            joint_act = self.model.joint_act
            actions = torch.clamp(actions, -1.0, 1.0) * self.action_strength
        elif self.action_type is ActionType.POSITION:
            joint_act = self.model.joint_target
            jlu, jll = self.joint_limit_upper, self.joint_limit_lower
            actions = (jll - jlu) * actions
            # q = self.joint_q.view(self.num_envs, -1)[:, 1:5]
            actions = torch.clamp(actions, jll, jlu)
            # joint_act.assign(self._joint_q)
        else:
            joint_act = self.model.joint_target
            jlu, jll = self.joint_limit_upper, self.joint_limit_lower
            stiff_scale = 0.25 * self.start_joint_stiffness_pt[:, : self.num_acts // 2]
            action_scale = torch.cat([(jlu - jll), stiff_scale], dim=-1)
            actions = actions * action_scale
            # q = self.joint_q.view(self.num_envs, -1)[:, 1:-1]
            # act_stiffness in [-0.5 joint stiff, +0.5 joint stiff]
            self.model.joint_target_ke.assign(self.start_joint_stiffness)
        joint_act.zero_()
        return actions, joint_act

    def autograd_step(self, actions):
        # Steps model in autograd kernel to allow gradients to flow
        actions, joint_act = self.assign_scale_actions(actions)
        self.actions = actions
        joint_act.zero_()

        actions = actions.flatten()
        # Refresh warp arrays to preserve gradients across multiple steps
        with torch.no_grad():
            rew_prev = self.rew_buf.clone()

        self.warp_actions.zero_()
        self.reward_ctrl.zero_()
        self.reward_total.zero_()
        self.reward_vec_prev.assign(self.reward_vec)
        self.reward_vec.zero_()
        with wp.ScopedTimer("simulate_autograd", active=False, detailed=False):
            # copying body_q/qd from previous state to pass grad thru prev state
            body_q = self.body_q.clone()
            body_qd = self.body_qd.clone()
            goal_state_vars = self._get_goal_state(self.goal_type)
            compute_reward = False
            (
                self.body_q[:],
                self.body_qd[:],
                # self.rew_buf,
                self.body_f,
                self.joint_q,
                self.joint_qd,
            ) = forward_simulate(
                self.model,
                self.integrator,
                self.sim_dt,
                self.sim_substeps,
                joint_act,
                self.warp_actions,
                self.warp_body_f,
                self._joint_q,
                self._joint_qd,
                self.joint_target_indices,
                actions,
                body_q,
                body_qd,
                self.joint_q,
                self.joint_qd,
                self.action_type,
                self.goal_type,
                goal_state_vars,
                self.reward_type,
                self.reward_ctrl,
                self.reward_total,
                self.reward_vec,
                self.reward_vec_prev,
                self.state_0,
                self.state_1,
                self.state_list,
                1,
                self.num_joint_q,
                self.capture_graph,
                compute_reward
            )
            # self.rew_buf = self.rew_buf - rew_prev  # subtract previous reward
            # if self.num_frames > 9*16:
            #     print("act:", actions.view(self.num_envs, -1)[0])
            #     print("state_in body_q after step:", self.state_0.body_q.numpy()[0])
            #     print("body_qd max after step:", self.body_q.max())
            #     print("state_out body_q after step:", self.state_1.body_q.numpy()[0])
            self.num_frames += 1  # iterate num_frames after calculating reward
            # updates body_q tensors
            # self.state_tensors[(self.state_1.body_q, self.num_frames)] = self.body_q
            # self.state_tensors[(self.state_1.body_qd, self.num_frames)] = self.body_qd
            # self.state_tensors[(self.reward_total, self.num_frames)] = self.rew_buf
            # self.state_tensors[(self.warp_body_f, self.num_frames)] = self.body_f
            # self.state_tensors[(self._joint_q, self.num_frames)] = self.joint_q
            # self.state_tensors[(self._joint_qd, self.num_frames)] = self.joint_qd

        # self.state_0.body_q.assign(self.state_1.body_q)
        # self.state_0.body_qd.assign(self.state_1.body_qd)
        # self.state_0.body_f.assign(self.state_1.body_f)

        # with torch.no_grad():
        #     self.actions = actions.view(self.num_envs, -1)
            # updates self.joint_q, self.joint_qd
        # rewards = torch.as_tensor(self.reward_vec_prev.numpy().copy(), device=self.device).view(self.num_envs, -1)
        # self.extras["control_pen"] = rewards[:, 0]
        # self.extras["ftip_err"] = rewards[:, 1]
        # self.extras["q_err"] = rewards[:, 2]
        # self.extras["pos_err"] = rewards[:, 3]
        # self.extras["force_err"] = rewards[:, 4]

    def warp_step(self, actions, requires_grad=False):
        """Does XPBD integrator step"""
        if requires_grad:
            self.autograd_step(actions)
        else:
            actions, joint_act = self.assign_scale_actions(actions)
            joint_act.zero_()
            self.warp_actions.zero_()
            self.warp_actions.assign(wp.from_torch(actions.flatten()))
            self.actions = actions.view(self.num_envs, -1)
            if self.capture_graph:
                if self.forward_sim_graph is None:
                    wp.capture_begin()
                    self.forward_simulate(joint_act)
                    self.forward_sim_graph = wp.capture_end()
                wp.capture_launch(self.forward_sim_graph)
            else:

                self.forward_simulate(joint_act)
                self.state_0, self.state_1 = self.state_1, self.state_0

            if self.debug:
                try:
                    assert np.isnan(self.state_0.body_f.numpy()).sum() == 0
                    assert np.isnan(self.state_1.body_q.numpy()).sum() == 0
                    assert np.isnan(self.state_1.body_qd.numpy()).sum() == 0
                except AssertionError as e:
                    print("Warning:", e)

            self.num_frames += 1  # iterate num_frames after calculating reward
            self.update_joints()
            # self.joint_q = wp.to_torch(self._joint_q)
            # self.joint_qd = wp.to_torch(self._joint_qd)
            self.body_q = wp.to_torch(self.state_0.body_q)
            self.body_qd = wp.to_torch(self.state_0.body_qd)

    def _check_early_termination(self):
        too_far_away = self.extras['obs_dict']["object_ori_err"] > self.termination_threshold
        termination_buf = too_far_away.flatten()
        self.extras['termination'] = termination_buf
        if termination_buf.sum() > 0:
            print(f"early termination: ep_len {self.progress_buf[termination_buf]}")
            print(f"act norm: {self.actions.norm(dim=1)[termination_buf]}")
        assert len(termination_buf) == (self.num_envs), termination_buf.shape

        nan_env_ids = self.obs_buf.isnan().sum(dim=1) > 0
        if self.obs_buf_before_reset is not None and nan_env_ids.sum() > 0:
            self.obs_buf[nan_env_ids] = self.obs_buf[nan_env_ids] * 0 + self.obs_buf_before_reset[nan_env_ids]

        return termination_buf


    def step(self, actions):
        assert (
            np.prod(actions.shape) == self.num_envs * self.num_actions
        ), f"actions should have compatible shape ({self.num_envs}, {self.num_actions}), got {actions.shape}"

        self.reset_buf = torch.zeros_like(self.reset_buf)
        if self.debug:
            self.prev_steps.append(self.extras)
        del self.extras
        self.extras = OrderedDict()

        with wp.ScopedTimer("simulate", active=False, detailed=False):
            self.warp_step(actions, self.requires_grad)  # iterates num_frames
            self.sim_time += self.sim_dt

        self.progress_buf += 1

        self.calculateObservations()
        # if not self.requires_grad:
        self.calculateReward()
        
        truncation = self.progress_buf >= self.episode_length
        termination_buf  = self._check_early_termination()
        self.reset_buf = self.reset_buf | truncation
        self.reset_buf = self.reset_buf | termination_buf
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset(env_ids)

        # logging will make env run much slower
        if self.debug:
            self.write_log()

        if self.requires_grad:
            self.obs_buf_before_reset = self.obs_buf.clone()
            self.extras["obs_before_reset"] = self.obs_buf_before_reset
            # self.extras["episode_end"] = self.extras["termination"]
            self.extras["truncation"] = truncation

        # Automatically renders if initialized with render=True
        if self.visualize:
            self.render()


        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def save_log(self, path):
        log = OrderedDict([(k, np.stack([l[k] for l in self.log])) for k in self.log[0]])
        np.savez(path, **log)
        self.log.clear()
