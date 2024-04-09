import pdb
from collections import OrderedDict

import numpy as np
import torch
import warp as wp
import math
import warp.sim
from dmanip.config import AllegroWarpConfig
from dmanip.envs.environment import compute_env_offsets
from dmanip.utils import warp_utils as wpu
from dmanip.utils import autograd as ag
from dmanip.utils import builder as bu
from dmanip.utils import warp_utils as wpu
from dmanip.utils.common import *
from shac.utils import torch_utils as tu

# from .warp_env import WarpEnv
from .warp_env import WarpEnv


np.set_printoptions(precision=3)


class AllegroWarpEnv(WarpEnv):
    num_rigid_contacts_per_env_mesh = 35000
    num_obs_dict = {
        GoalType.ORIENTATION: 56,  # yaw-only
        GoalType.POSITION: 57,  # x-y only
        GoalType.POSE: 58,  # x-y, yaw
        GoalType.TRAJECTORY_ORIENTATION: 56,
        GoalType.TRAJECTORY_POSITION: 57,  # TODO: scale by sub traj window size
        GoalType.TRAJECTORY_ORIENTATION_TORQUE: 58,
        GoalType.TRAJECTORY_POSITION_FORCE: 59,
        GoalType.TRAJECTORY_POSE_WRENCH: 74,  # 4D ORIENTATION + POSITION + WRENCH
    }
    # bonus obs size for tcdm observations
    tcdm_num_obs_dict = {
        GoalType.ORIENTATION: 9,  # quat orientation + 6DoF velocity
        GoalType.POSITION: 7,  # xyz
        GoalType.POSE: 10,  # pos and quat
        GoalType.TRAJECTORY_ORIENTATION: 17,
        GoalType.TRAJECTORY_POSITION: 7,
        GoalType.TRAJECTORY_ORIENTATION_TORQUE: 11,  # quat ori + 3DoF torque
        GoalType.TRAJECTORY_POSITION_FORCE: 8,
        GoalType.TRAJECTORY_POSE_WRENCH: 15,  # for 3d base,
    }
    # c_act, c_finger, c_q, c_pos, c_f/t
    reward_coefs = np.array([1e-2, 1, 10, 0, 0], dtype=np.float32)

    env_offset = (6.0, 0.0, 6.0)
    tiny_render_settings = dict(scaling=15.0)
    usd_render_settings = dict(scaling=200.0)

    sim_substeps_euler = 64
    sim_substeps_xpbd = 10

    xpbd_settings = dict(
        iterations=10,
        joint_linear_relaxation=1.0,
        joint_angular_relaxation=0.45,
        rigid_contact_relaxation=0.8,
        rigid_contact_con_weighting=True,
    )
    use_graph_capture = False
    obs_list_names = []
    sub_length = 8

    def __init__(
        self, 
        num_envs: int = 1,
        episode_length: int = 600,
        no_grad: bool = True,
        seed: int = 42,
        render: bool = False,
        device: str = "cuda",
        stochastic_init: bool = False,
        action_type: ActionType = ActionType.POSITION,
        object_type: ObjectType = ObjectType.OCTPRISM,
        goal_type: GoalType = GoalType.ORIENTATION,
        reward_type: RewardType = RewardType.DELTA,
        debug: bool = False,
        stage_path: Optional[str] = None,
        goal_path: Optional[str] = None,
        action_strength: float = 0.1,
    ):
        num_obs = self.num_obs_dict[goal_type]
        if object_type.name.startswith("TCDM"):
            # if non-planar goal type
            num_obs += self.tcdm_num_obs_dict[goal_type]
        if action_type in [
            ActionType.TORQUE,
            ActionType.POSITION,
            # ActionType.DELTA_POSITION,
        ]:
            num_act = 16
        elif action_type is ActionType.VARIABLE_STIFFNESS:
            num_act = 32
        self.state_tensors = {}
        super().__init__(
            num_envs=num_envs,
            num_obs=num_obs,
            num_act=num_act,
            episode_length=episode_length,
            no_grad=no_grad,
            seed=seed,
            render=render,
            stochastic_init=stochastic_init,
            device=device,
            env_name="Allegro-{}".format(object_type.name.lower()),
        )
        self.goal_type = goal_type
        self.reward_type = reward_type
        self.action_type = action_type
        print("setting action type to:", action_type)
        if action_type is ActionType.POSITION or action_type is ActionType.VARIABLE_STIFFNESS:
            self.action_strength = 0.05
        else:
            self.action_strength = action_strength

        print("action strength", self.action_strength)
        self.debug = debug
        self.prev_steps = []

        # 9 is where TCDM objects begin
        self.object_type = object_type
        self.tcdm_object = self.object_type.name.startswith("TCDM")

        self.goal_path = goal_path
        self.goal_trajectory = None
        self.initialize_sim()

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
        # shape: (num_envs, 6) for 6 reward components, feeds into reward_total
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
        # actions buffer in warp to use for action assignment
        self.warp_actions = wp.array(
            np.zeros(self.num_envs * self.num_actions),
            dtype=float,
            device=self.device,
            requires_grad=self.requires_grad,
        )
        fid = np.stack(
            [
                self.body_name_to_idx["index_link_3"],
                self.body_name_to_idx["middle_link_3"],
                self.body_name_to_idx["ring_link_3"],
                self.body_name_to_idx["thumb_link_3"],
            ],
            axis=1,
        ).reshape(-1)
        fid = wp.array(
            fid,
            device=self.device,
            dtype=int,
        )
        oid = wp.array(self.body_name_to_idx["object"], device=self.device, dtype=int)
        oqid = self.model.joint_q_start.numpy()[self.joint_name_to_idx["obj_joint"]]
        oqid = wp.array(oqid, device=self.device, dtype=int)
        base_id = wp.array(
            self.body_name_to_idx["palm_link"],
            device=self.device,
            dtype=int,
        )
        self.goal_state_vars = {
            "num_fingers": 4,
            "offset": wp.vec3(0.0584, 0.0, 0.0),
            "object_thickness": 0.062,
            "oid": oid,
            "oqid": oqid,
            "base_id": base_id,
            "rotation_count": wp.array(np.zeros(self.num_envs), dtype=float, device=self.device),
            "fid": fid,
            "start_joint_q": wp.from_torch(self.start_joint_q.flatten()),
            "start_joint_qd": wp.from_torch(self.start_joint_qd.flatten()),
            "reward_coefs": wp.array(self.reward_coefs, dtype=float, device=self.device),
        }

        # Create state array buffers and graph capture instances
        self.update_joints_graph = None
        self.forward_sim_graph = None
        if self.requires_grad:
            self.state_0 = self.model.state(requires_grad=True)
            self.state_1 = self.model.state(requires_grad=True)
            self.state_list = [self.model.state(requires_grad=True) for _ in range(self.sim_substeps - 1)]
            self.model.allocate_rigid_contacts()
            wp.sim.collide(self.model, self.state_0)
            self.warp_body_f = wp.zeros_like(self.state_0.body_f)
            self.warp_body_f.requires_grad = True
            self.body_f = None

        else:
            self.state_0 = self.model.state()
            self.state_1 = self.model.state()
            self.warp_body_f = wp.zeros_like(self.state_0.body_f)
            self.model.allocate_rigid_contacts()
            wp.sim.collide(self.model, self.state_0)

        self.setup_autograd_vars()
        # -----------------------
        # set up Usd renderer
        if self.visualize:
            self.initialize_renderer(stage_path)

    @property
    def goal_step(self):
        return self.goal_trajectory_idx[self.num_frames]

    @property
    def goal_state(self):
        """Returns full goal state, including trajectory state"""
        if "trajectory" in self.goal_type.name.lower():
            return self.goal_trajectory[self.goal_step]
        return self._goal_state

    def initialize_sim(self):
        # sim_dt defaults to 1.0 / 60.0
        self.sim_dt = 1.0 / 60.0
        self.sim_substeps = 6  # 4 if XPBD, 24 if Euler
        if self.debug:
            self.render_freq = 1
        wp.init()

        articulation_builder = wp.sim.ModelBuilder(gravity=0.0)

        # Currently, 3D tasks only use TCDM objects
        self.floating_base = self.tcdm_object

        bu.create_allegro_hand(articulation_builder, self.action_type, self.floating_base)

        # aux root joint, 16 finger joints, 1-7 object joint
        object_free_joint = self.goal_type in POSITION_GOAL_TYPES
        joint_type = wp.sim.JOINT_FREE if object_free_joint else wp.sim.JOINT_REVOLUTE

        self.num_robot_joint_q = 16 + 7 * int(self.floating_base)
        self.num_robot_joint_qd = 16 + 6 * int(self.floating_base)
        self.num_obj_joint_q = 7 * int(object_free_joint) + (1 - object_free_joint)
        self.num_obj_joint_qd = 6 * int(object_free_joint) + (1 - object_free_joint)
        self.num_joint_q = self.num_robot_joint_q + self.num_obj_joint_q
        self.num_joint_qd = self.num_robot_joint_qd + self.num_obj_joint_qd

        # load object
        self.object_model = bu.OBJ_MODELS[self.object_type]()
        self.object_model.joint_type = joint_type
        self.object_model.create_articulation(articulation_builder)
        pos = self.object_model.base_pos

        self.builder = wp.sim.ModelBuilder()
        env_offsets = compute_env_offsets(self.num_envs, self.env_offset, "y")
        for i in range(self.num_envs):
            xform = wp.transform(env_offsets[i], wp.quat_identity())
            self.builder.add_rigid_articulation(articulation_builder, xform=xform)

        if self.tcdm_object:
            self.load_goal_trajectory(self.goal_path)
            self._goal_state = self.goal_trajectory[-1]
        elif not self.stochastic_init:
            self._goal_state = torch.as_tensor(np.array([pos[0], pos[1], 0.3]), device=self.device)
            if self.goal_type.name.lower().startswith("trajectory") and self.goal_trajectory is None:
                self.load_goal_trajectory(self.goal_path)
        else:
            if self.goal_type in POSITION_GOAL_TYPES:
                # pull object closer
                self._goal_state = torch.as_tensor(np.array([pos[0] - 0.1, pos[1], 0.0]), device=self.device)
            elif self.goal_type in ORIENTATION_GOAL_TYPES:
                # rotate object in place
                self._goal_state = torch.as_tensor(np.array([pos[0], pos[1], 0.0]), device=self.device)
            if self.goal_type.name.lower().startswith("trajectory") and self.goal_trajectory is None:
                self.load_goal_trajectory(self.goal_path)

        self.builder.gravity = 0.0
        # if self.object_type.value > ObjectType.OCTPRISM.value:
        #     self.num_rigid_contacts_per_env_mesh = 140000
        # self.builder.num_rigid_contacts_per_env = self.num_rigid_contacts_per_env_mesh
        self.model = self.builder.finalize(
            self.device, requires_grad=self.requires_grad, rigid_mesh_contact_max=self.num_rigid_contacts_per_env_mesh
        )
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
            :, : self.num_robot_joint_q - 7 * self.floating_base
        ]
        self.joint_limit_upper = wp.to_torch(self.model.joint_limit_upper).view(self.num_envs, -1)[
            :, : self.num_robot_joint_q - 7 * self.floating_base
        ]

        self.model.ground = False
        self.model.joint_attach_ke = 32000.0
        self.model.joint_attach_kd = 50.0

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

        # initialize integrator, states, and starting joint states
        self.integrator = wp.sim.XPBDIntegrator(**self.xpbd_settings)
        start_joint_q = wp.to_torch(self.model.joint_q)
        start_joint_qd = wp.to_torch(self.model.joint_qd)
        joint_X_p = wp.to_torch(self.model.joint_X_p)

        # only stores a single copy of the initial state
        self.start_joint_q = start_joint_q.clone().view(self.num_envs, -1)
        self.start_joint_qd = start_joint_qd.clone().view(self.num_envs, -1)
        self.start_joint_X_p = joint_X_p.clone().view(self.num_envs, -1, 7)
        obj_q = self.start_joint_X_p[:, -1, [0, 2]]
        obj_theta = torch.zeros((self.num_envs, 1), device=self.device)
        self.start_obj_state = torch.cat([obj_q, obj_theta], dim=1)
        self.start_joint_stiffness = wp.clone(self.model.joint_target_ke)
        self.start_joint_stiffness_pt = wp.to_torch(self.start_joint_stiffness).view(self.num_envs, -1)

        # Buffers to compute robot joint pos and vel
        self._joint_q = wp.zeros_like(self.model.joint_q, requires_grad=self.requires_grad)
        self._joint_qd = wp.zeros_like(self.model.joint_qd, requires_grad=self.requires_grad)
        self.joint_q, self.joint_qd = None, None
        self.log = []

    def initialize_trajectory(self):
        self.clear_grad()
        self.calculateObservations()
        return self.obs_buf

    def load_goal_trajectory(self, goal_path=None, env_ids=None):
        if env_ids is None:
            env_ids = np.arange(self.num_envs)
            # based on OBJECT/GOAL type, assign temporary variable gt = np array of size (traj_len, goal_dim)

        # different goal trajectory loading process for TCDM objects
        if self.object_type.name.startswith("TCDM"):
            trajectory = self.object_model.dex_trajectory
            orientation_goals = trajectory["orientation"]  # (traj_len, 3)
            translation_goals = trajectory["position"]  # (traj_len, 3)
            force_goals = trajectory["force"]  # (traj_len, 3)
            torque_goals = trajectory["torque"]  # (traj_len, 3)
            gt = np.hstack((translation_goals, orientation_goals, force_goals, torque_goals))
        else:
            if goal_path is not None and osp.exists(goal_path):
                # loads goal trajectory (obj_force, obj_pos), shape: (T, 2)
                print("loading goal trajectory from", goal_path)
                gt = np.load(goal_path)
            # if no goal trajectory is provided, generate one
            elif (
                self.goal_type is GoalType.TRAJECTORY_ORIENTATION
                or self.goal_type is GoalType.TRAJECTORY_ORIENTATION_TORQUE
            ):
                sigmoid = lambda z: 1 / (1 + np.exp(-z))
                start = self.start_obj_state[:, 2:3].cpu().numpy()  # (n_envs, 1)
                end = self._goal_state[2].cpu().numpy()  # (1,)
                gt = sigmoid(np.linspace(-5, 5, self.episode_length))[:, None]
                # (T, n_envs)
                gt = gt @ (end - start) + start
            elif self.goal_type is GoalType.TRAJECTORY_POSITION:
                start, end = self.start_obj_state, self._goal_state
                goal_x = np.linspace(start[:, 0], end[0], self.episode_length)
                goal_y = np.linspace(start[:, 1], end[1], self.episode_length)  # (T, n_envs)
                gt = np.stack([goal_x, goal_y], axis=-1)  # (T, n_envs, 2)
            else:
                return

        if self.goal_trajectory is None:
            self.goal_trajectory = gt

        if self.goal_trajectory is not None and gt.shape[1] == self.num_envs:
            self.goal_trajectory[:, env_ids] = gt[:, env_ids]

        # goal_index maps goal indices to current timestep
        if self.goal_trajectory is not None and "trajectory" in self.goal_type.name.lower():
            assert not self.stochastic_init, "stochastic init not supported for trajectory goals, goal_type: {}".format(
                self.goal_type.name
            )
            self.goal_trajectory_idx = np.array(
                [math.floor(x) for x in np.linspace(0, len(self.goal_trajectory) - 1, self.episode_length + 1)]
            )
            # Use sub_length-step goal keypoints for trajectory goals
            self.goal_trajectory_idx = np.clip(
                (self.goal_trajectory_idx // self.sub_length + 1) * self.sub_length,
                0,
                self.goal_trajectory_idx[-1],
            )

    def get_goal_state(self, goal_type):
        # TODO: select goals based on: goal_idx, timestep, closeset goal, ...
        # coefs scale components of reward: act penalty, goal dist,
        goal_pos = np.zeros(2)
        goal_ori = (0, 0, 0, 1)
        goal_force = np.zeros(2)
        goal_torque = 0.0

        goal_state = to_numpy(self.goal_state)
        # if TCDM object and using 3D trajectory/floating hand
        if self.floating_base:
            goal_pos = wp.vec3(*goal_state[:3])
            goal_ori = wp.quat(*goal_state[3:7])
            goal_force = wp.vec3(*goal_state[7:10])
            goal_torque = wp.vec3(*goal_state[10:13])
        else:
            if "trajectory" in goal_type.name.lower():
                if "orientation" in goal_type.name.lower():
                    if "torque" in goal_type.name.lower():
                        goal_torque, theta = goal_state[..., 0], goal_state[..., 1]
                    else:
                        theta = goal_state[0]
                    goal_ori = wp.quat_rpy(0, 0, theta)
                elif "position" in goal_type.name.lower():  # position goaltype
                    if "force" in goal_type.name.lower():
                        goal_force, goal_pos = goal_state[:2], goal_state[2:]
                    else:
                        goal_pos = goal_state
            else:
                goal_pos, theta = goal_state[:2], goal_state[2]
                goal_ori = wp.quat_rpy(0, 0, theta)
            goal_pos = wp.vec3(goal_pos[0], 0.15, goal_pos[1])  # y-up
            goal_force = wp.vec3(goal_force[0], goal_force[1], 0.0)  # y-up
            goal_torque = wp.vec3(0.0, 0.0, goal_torque)  # torque around y-axis

        self.goal_state_vars.update(
            {
                "goal_pos": goal_pos,
                "goal_ori": goal_ori,
                "goal_force": goal_force,
                "goal_torque": goal_torque,
            }
        )
        return self.goal_state_vars

    def get_goal_obs(self):
        """Returns goal observation as torch.Tensor with shape (num_envs, goal_dim)"""
        goal_dict = OrderedDict()
        goal_state = self.goal_state
        if self.goal_type == GoalType.TRAJECTORY_POSE_WRENCH:
            goal_dict["goal_pos"] = goal_state[:3]
            goal_dict["goal_ori"] = goal_state[3:7]
            goal_dict["goal_force"] = goal_state[7:10]
            goal_dict["goal_torque"] = goal_state[10:13]
        elif "trajectory" in self.goal_type.name.lower():
            if "orientation" in self.goal_type.name.lower():  # orientation goal-type
                if "torque" in self.goal_type.name.lower():
                    goal_dict["goal_ori"] = goal_state[..., [1]]
                    goal_dict["goal_torque"] = goal_state[..., [0]]
                else:
                    goal_dict["goal_ori"] = np.array([goal_state[0]])
            elif "position" in self.goal_type.name.lower():  # position goal-type
                if "force" in self.goal_type.name.lower():
                    goal_dict["goal_pos"] = goal_state[2:]
                    goal_dict["goal_force"] = goal_state[:2]
                else:
                    goal_dict["goal_pos"] = goal_state  # goal state is pos only
        else:
            if "position" in self.goal_type.name.lower():
                goal_dict["goal_pos"] = goal_state[:2]
            elif "orientation" in self.goal_type.name.lower():
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

    def get_joint_pos_vel(self, joint_name, return_vel=False):
        joint_index = self.joint_name_to_idx[joint_name]
        if self.joint_q is None:
            self.update_joints()
        joint_pos = self.joint_q[joint_index]
        if return_vel:
            joint_vel = self.joint_qd[joint_index]
            return joint_pos, joint_vel
        return joint_pos

    def get_body_pos_vel(self, body_name, return_vel=False, return_quat=False):
        """Warning: adds extra stacking dimension to body_pos and body_vel"""
        body_index = self.body_name_to_idx[body_name]
        assert len(body_index) == self.num_envs
        end = 7 if return_quat else 3
        body_pos = self.body_q[body_index, :end].view(self.num_envs, end)
        if return_vel:
            body_vel = self.body_qd[body_index].view(self.num_envs, 6)
            return body_pos, body_vel
        return body_pos

    def get_body_f(self, body_name):
        body_index = self.body_name_to_idx[body_name]
        # returns angular, linear vel
        body_f = self.to_torch(self.warp_body_f)[body_index, :].reshape(self.num_envs, -1)
        return body_f

    def get_fingertip_pos(self):
        fingertip_xform1 = self.get_body_pos_vel("index_link_3", return_quat=True)
        fingertip_xform2 = self.get_body_pos_vel("middle_link_3", return_quat=True)
        fingertip_xform3 = self.get_body_pos_vel("ring_link_3", return_quat=True)
        fingertip_xform4 = self.get_body_pos_vel("thumb_link_3", return_quat=True)

        fingertip_offset = torch.tensor([[0.0584, 0.0075, 0.0] for _ in range(self.num_envs)], device=self.device)
        fingertip_pos1 = tu.tf_apply(fingertip_xform1[:, 3:], fingertip_xform1[:, :3], fingertip_offset)
        fingertip_pos2 = tu.tf_apply(fingertip_xform2[:, 3:], fingertip_xform2[:, :3], fingertip_offset)
        fingertip_pos3 = tu.tf_apply(fingertip_xform3[:, 3:], fingertip_xform3[:, :3], fingertip_offset)
        fingertip_pos4 = tu.tf_apply(fingertip_xform4[:, 3:], fingertip_xform4[:, :3], fingertip_offset)
        return fingertip_pos1, fingertip_pos2, fingertip_pos3, fingertip_pos4

    def to_torch(self, state_arr):
        # num_frames = num_frames or self.num_frames
        if state_arr not in self.state_tensors:
            self.state_tensors[state_arr] = wp.to_torch(state_arr)
        return self.state_tensors[state_arr]

    def write_log(self):
        # print("Model Root transform:", self.model.joint_X_p.numpy()[0])

        ftip_1, ftip_2, ftip_3, ftip_4 = (
            self.extras["index_fingertip_pos"],
            self.extras["middle_fingertip_pos"],
            self.extras["ring_fingertip_pos"],
            self.extras["thumb_fingertip_pos"],
        )
        # prev_reward_total = 0.0
        # if len(self.log) > 0:
        #     prev_reward_total = self.log[-1]["reward_total"] * 0.99
        log_item = {
            "joint_q": to_numpy(self.joint_q).reshape(self.num_envs, -1),
            "joint_qd": to_numpy(self.joint_qd).reshape(self.num_envs, -1),
            "actions": to_numpy(self.actions).reshape(self.num_envs, -1),
            "index_fingertip_pos": to_numpy(ftip_1).reshape(self.num_envs, -1),
            "middle_fingertip_pos": to_numpy(ftip_2).reshape(self.num_envs, -1),
            "ring_fingertip_pos": to_numpy(ftip_3).reshape(self.num_envs, -1),
            "thumb_fingertip_pos": to_numpy(ftip_4).reshape(self.num_envs, -1),
            "reward_total": to_numpy(self.rew_buf).reshape(self.num_envs, -1),
            "reward_vec": self.reward_vec.numpy().reshape(self.num_envs, -1),
        }
        if "object" in self.body_name_to_idx:
            obj_pos, obj_vel = (
                self.extras["object_body_pos"],
                self.extras["object_body_vel"],
            )
            obj_pos, obj_vel = to_numpy(obj_pos), to_numpy(obj_vel)
            obj_body_f = to_numpy(self.get_body_f("object"))
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

    def update_joints(self):
        self._joint_q.zero_()
        self._joint_qd.zero_()
        if self._joint_q.grad is not None:
            self._joint_q.grad.zero_()
            self._joint_qd.grad.zero_()

        if self.use_graph_capture:
            if self.update_joints_graph is None:
                wp.capture_begin()
                wp.sim.eval_ik(self.model, self.state_0, self._joint_q, self._joint_qd)
                self.update_joints_graph = wp.capture_end()
            wp.capture_launch(self.update_joints_graph)
        else:
            wp.sim.eval_ik(self.model, self.state_0, self._joint_q, self._joint_qd)

        self.joint_q = self.to_torch(self._joint_q).clone()
        self.joint_qd = self.to_torch(self._joint_qd).clone()

    def calculateObservations(self):
        joint_q = self.joint_q.view(self.num_envs, -1)
        joint_qd = self.joint_qd.view(self.num_envs, -1)
        x = joint_q[:, 0 : self.num_robot_joint_q]
        theta = joint_q[:, self.num_robot_joint_q :]  # Xform if joint_type == free
        if self.goal_type is GoalType.TRAJECTORY_POSE_WRENCH:
            obj_theta = self.get_body_pos_vel("object", return_quat=True)[:, 3:]
        elif self.goal_type in POSITION_GOAL_TYPES:
            obj_theta = self.get_body_pos_vel("object", return_quat=True)[:, 3:]
            theta = to_torch(wpu.yaw_from_quat(to_numpy(obj_theta), 2), device=self.device)[:, None]

        xdot = joint_qd[:, 0 : self.num_robot_joint_qd]
        theta_dot = joint_qd[:, self.num_robot_joint_qd :]
        obj_pos, obj_vel = self.get_body_pos_vel("object", return_vel=True)
        obj_f = self.get_body_f("object")

        goal = self.get_goal_obs()
        rewards = self.to_torch(self.reward_vec).view(self.num_envs, -1)
        q_err = rewards[:, 2:3]
        pos_err = rewards[:, 3:4]
        force_err = rewards[:, 4:5]

        fingertip_1, fingertip_2, fingertip_3, fingertip_4 = self.get_fingertip_pos()
        if self.floating_base:
            obj_torque = obj_f[:, :3]
            obj_force = obj_f[:, 3:]
        else:
            obj_torque = obj_f[:, 1:2]
            obj_force = obj_f[:, [3, 5]]

        obs_dict = OrderedDict(
            [
                ("robot_joint_pos", x),  # (num_envs, 16 or 23)
                ("object_joint_pos", theta),  # (num_envs, 1 or 7)
                ("object_body_pos", obj_pos),  # (num_envs, 3)
                ("index_fingertip_pos", fingertip_1),  # (num_envs, 3)
                ("middle_fingertip_pos", fingertip_2),  # (num_envs, 3)
                ("ring_fingertip_pos", fingertip_3),  # (num_envs, 3)
                ("thumb_fingertip_pos", fingertip_4),  # (num_envs, 3)
                ("robot_joint_vel", xdot),  # (num_envs, 16 or 23)
                ("object_joint_vel", theta_dot),  # (num_envs, 1)
                ("object_body_vel", obj_vel),  #  (num_envs, 6)
                ("object_body_torque", obj_torque),  #  (num_envs, 1)
                ("object_body_f", obj_force),  #  (num_envs, 2)
                ("object_q_err", q_err),  #  (num_envs, 2)
                ("object_pos_err", pos_err),  #  (num_envs, 2)
                ("object_force_err", force_err),  #  (num_envs, 2)
                (
                    "action",
                    self.actions.view(self.num_envs, -1),
                ),  #  (num_envs, num_acts)
                ("goal", goal),  # (num_envs, len(goal_state))
                ("timestep", self.progress_buf),
            ]
        )

        for k in obs_dict:
            if obs_dict[k].isnan().any():
                assert False, "NaN in {}".format(k)
                # print("NaN in {}".format(k))
                # with torch.no_grad():
                #     env_mask = obs_dict[k].isnan().any(dim=1)
                #     self.reset_buf = self.reset_buf | env_mask
                #     obs_dict[k][env_mask] = 0.0
                #     rew_vec_mask = to_numpy(self.reset_buf.view(-1, 1).repeat(1, 5))
                #     reward_vec = self.reward_vec.numpy()
                #     reward_vec[rew_vec_mask] = 0.0
                #     self.reward_vec.assign(reward_vec)
                #     self.rew_buf[self.reset_buf] = -10.0

        if self.goal_type in ORIENTATION_GOAL_TYPES:
            reset_q = 0.69 if self.goal_type.name.lower().startswith("traj") else 0.5
            self.reset_buf = self.reset_buf | (q_err.squeeze() > reset_q)

        if self.obs_list_names:
            obs_list_names = self.obs_list_names
        elif self.goal_type == GoalType.TRAJECTORY_POSE_WRENCH:
            obs_list_names = [
                "robot_joint_pos",
                "index_fingertip_pos",
                "middle_fingertip_pos",
                "ring_fingertip_pos",
                "thumb_fingertip_pos",
                "robot_joint_vel",
                "object_joint_pos",
                "object_joint_vel",
                "object_body_torque",
                "object_body_f",
                "goal",
            ]
        elif self.goal_type == GoalType.TRAJECTORY_ORIENTATION_TORQUE:
            obs_list_names = [
                "robot_joint_pos",
                "object_joint_pos",
                "object_body_pos",
                "index_fingertip_pos",
                "middle_fingertip_pos",
                "ring_fingertip_pos",
                "thumb_fingertip_pos",
                "robot_joint_vel",
                "object_joint_vel",
                "object_body_vel",
                "object_body_torque",
                "goal",
            ]
        elif self.goal_type == GoalType.TRAJECTORY_POSITION_FORCE:
            obs_list_names = [
                "robot_joint_pos",
                "object_joint_pos",
                "object_body_pos",
                "index_fingertip_pos",
                "middle_fingertip_pos",
                "ring_fingertip_pos",
                "thumb_fingertip_pos",
                "robot_joint_vel",
                "object_joint_vel",
                "object_body_vel",
                "object_body_f",
                "goal",
            ]
        else:
            obs_list_names = [
                "robot_joint_pos",
                "object_joint_pos",
                "object_body_pos",
                "index_fingertip_pos",
                "middle_fingertip_pos",
                "ring_fingertip_pos",
                "thumb_fingertip_pos",
                "robot_joint_vel",
                "object_joint_vel",
                "object_body_vel",
                "goal",
            ]
        obs_list = [obs_dict[k] for k in obs_list_names]
        self.obs_buf = torch.cat(
            obs_list,
            dim=1,
        ).view(self.num_envs, -1)
        assert self.obs_buf.isnan().sum() == 0, "NaN in observation"
        self.extras.update(obs_dict)
        # observations: [x, theta, obj_pos, xdot, theta_dot, obj_vel]

    def calculate_reward_kernels(self, goal_state_vars):
        wp.launch(
            kernel=wpu.compute_ctrl_reward,
            inputs=[self.warp_actions, self.num_actions],
            outputs=[self.reward_ctrl],
            dim=(self.num_envs, self.num_actions),
            device=self.device,
        )
        wp.launch(
            kernel=wpu.compute_goal_reward,
            inputs=[
                self.state_0.body_q,
                self.warp_body_f,
                self._joint_q,
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
                self.goal_type.value,
                self.reward_type.value,
                self.reward_ctrl,
                goal_state_vars["reward_coefs"],
                goal_state_vars["rotation_count"],
                # goal_state_vars["start_joint_q"],
                # goal_state_vars["start_joint_qd"],
            ],
            outputs=[self.reward_vec, self.reward_vec_prev, self.reward_total],
            dim=self.num_envs,
            device=self.device,
        )
        # wp.launch(
        #     kernel=wpu.joint_limit_penalty,
        #     inputs=[
        #         self.state_0.body_q,
        #         self.model.joint_qd_start,
        #         self.model.joint_type,
        #         self.model.joint_parent,
        #         self.model.joint_X_p,
        #         self.model.joint_axis,
        #         self.model.joint_limit_lower,
        #         self.model.joint_limit_upper,
        #         self.num_joint_q,  # num_joints
        #     ],
        #     outputs=[self.reward_total],
        #     dim=self.model.joint_count,
        #     device=self.model.device,
        # )

    def forward_simulate(self, joint_act):
        wpu.assign_act(
            self.warp_actions,
            joint_act,
            self.model.joint_target_ke,
            self.action_type,
            num_acts=self.num_actions,
            num_envs=self.num_envs,
            num_joints=self.num_joint_q,
            q_offset=0,  # skip first 7 joint dims if hand floating
        )
        if self.floating_base:
            # set base velocity to zero unless lifting after grasping object
            pass

        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.state_1.clear_forces()
            # updates ground/rigid contacts
            if not self.use_graph_capture:
                self.model.allocate_rigid_contacts()
            wp.sim.collide(self.model, self.state_0)
            self.state_1 = self.integrator.simulate(
                self.model,
                self.state_0,  # state_in
                self.state_1,  # state_out
                self.sim_dt / self.sim_substeps,
            )
            self.state_0, self.state_1 = self.state_1, self.state_0
            # print("new_state", self.state_1.body_f.numpy())
            # print("old_state", self.state_0.body_f.numpy())
        self.warp_body_f.assign(self.state_1.body_f)
        # wpu.integrate_body_f(
        #     self.model,
        #     self.state_0.body_q,
        #     self.state_1.body_qd,
        #     self.state_0.body_qd,
        #     self.warp_body_f,
        #     self.sim_dt / self.sim_substeps,
        # )

    def calculateReward(self):
        # computes reward in reward kernel
        self.reward_ctrl.zero_()
        self.reward_total.zero_()
        self.reward_vec_prev.assign(self.reward_vec)
        self.reward_vec.zero_()
        goal_state_vars = self.get_goal_state(self.goal_type)

        self.calculate_reward_kernels(goal_state_vars)

        self.rew_buf = wp.to_torch(self.reward_total)
        if self.rew_buf.isnan().sum() != 0:
            # print("NaN in reward")
            assert False, "NaN in reward"
        self.rew_buf[self.reset_buf] = -10.0
        rewards = torch.as_tensor(self.reward_vec.numpy(), device=self.device).view(self.num_envs, -1)
        self.extras["control_pen"] = rewards[:, 0]
        self.extras["ftip_err"] = rewards[:, 1]
        self.extras["q_err"] = rewards[:, 2]
        self.extras["pos_err"] = rewards[:, 3]
        self.extras["force_err"] = rewards[:, 4]

    def get_stochastic_init(self, env_ids, joint_q, joint_qd):
        joint_X_p = self.start_joint_X_p.clone().view(self.num_envs, -1, 7)

        n = len(env_ids)
        # import pdb
        #
        # pdb.set_trace()
        if self.goal_type in POSITION_GOAL_TYPES:
            # object should have free joint
            assert joint_q.shape == (n, self.num_joint_q)

            # move closer to origin, with random L/R z translation
            start_obj_state = torch.stack(
                [
                    -torch.ones(n, device=self.device) * 0.2,  # x-translation
                    torch.ones(n, device=self.device) * 0.2 - 0.1,  # z-translation
                    torch.zeros(n, device=self.device),  # theta
                ],
                dim=1,
            )
            # ori = tuple(x for x in wp.quat_rpy(np.pi / 2, 0.0, 0.0))
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
            # obj_X_p = joint_X_p[env_ids, -1]  # obj parent link transform
            # obj_Xform_r, obj_Xform_t = tu.tf_combine(
            #     obj_Xform_w[:, 3:], obj_Xform_w[:, :3], obj_X_p[:, 3:], obj_X_p[:, :3]
            # )
            # obj_pose = torch.cat([obj_Xform_t, obj_Xform_r], dim=1)
            # joint_X_p[env_ids, -1] = obj_pose
            self.model.joint_X_p.assign(wp.from_torch(joint_X_p))
            joint_q[env_ids, -7:] = obj_Xform_w
        else:
            rand_orientation = torch.rand((n, 1), device=self.device) * 0.275 + 0.025
            obj_orientation = to_torch(np.random.choice([-1, 1], n)[:, None], self.device) * rand_orientation
            # obj_orientation = torch.rand((n, 1), device=self.device) * 0.6 - 0.3
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
                start_obj_state[..., :2],
                "Starting orientation: ",
                start_obj_state[..., 2:],
            )

        # object state contains x, y, theta (orientation)
        self.start_obj_state = start_obj_state
        return joint_q[env_ids], joint_qd[env_ids]

    def reset(self, env_ids=None, force_reset=True):
        # self.state_tensors.clear()

        # creating state_1 if no_grad enabled for warp_step and calculateObservation
        if not self.requires_grad:
            self.state_1 = self.model.state(requires_grad=False)

        # if self.goal_type.name.lower().startswith("trajectory"):
        #     self.load_goal_trajectory(self.goal_path, env_ids)
        super().reset(env_ids, force_reset)  # resets state_0 and joint_q, and joint_act
        self._joint_q.assign(wp.from_torch(self.joint_q))
        self._joint_qd.assign(wp.from_torch(self.joint_qd))
        self.warp_actions.zero_()
        self.reward_vec_prev.zero_()
        self.goal_state_vars["rotation_count"].zero_()
        self.warp_body_f.zero_()
        # sets goal_trajectory, obs_buf, and updates joints
        self.initialize_trajectory()
        if self.requires_grad:
            self.model.allocate_rigid_contacts(requires_grad=self.requires_grad)
            wp.sim.collide(self.model, self.state_0)
        self.log.clear()
        self.prev_steps = []
        return self.obs_buf

    def clear_grad(self, checkpoint=None):
        self.update_joints()
        self.body_q = wp.to_torch(self.state_0.body_q)
        self.body_qd = wp.to_torch(self.state_0.body_qd)
        super().clear_grad(checkpoint)
        if hasattr(self, "body_q"):
            with torch.no_grad():
                self.joint_q = self.joint_q.clone()
                self.joint_qd = self.joint_qd.clone()
                self.body_q = self.body_q.clone()
                self.body_qd = self.body_qd.clone()
                self.actions = self.actions.clone()
                if self.actions.grad is not None:
                    self.actions.grad.zero()
                self.warp_actions.zero_()
                if self.requires_grad:
                    self.warp_body_f.zero_()
                    self.warp_actions.grad.zero_()
                    self.warp_body_f.grad.zero_()
                # clear_state_grads(self.state_0)
                # clear_state_grads(self.state_1)
                # [clear_state_grads[s] for s in self.state_list]
                if self.requires_grad:
                    self.reward_ctrl.grad.zero_()
                    self.reward_total.grad.zero_()
                    self.reward_vec.grad.zero_()

    def assign_scale_actions(self, actions):
        if self.action_type is ActionType.TORQUE:
            joint_act = self.model.joint_act
            actions = torch.clamp(actions, -1.0, 1.0) * self.action_strength
            self.model.joint_act.zero_()
        elif self.action_type is ActionType.POSITION:
            # control position directly by changing joint_target
            # stiffness, damping = 5000, 10
            joint_act = self.model.joint_target
            jlu, jll = self.joint_limit_upper, self.joint_limit_lower
            joint_limit_range = (jlu - jll) * 0.4  # limit range to  +/- 0.1 from edge
            joint_limit_center = (jlu - jll) * 0.5 + jll  # center
            actions = joint_limit_range * actions + joint_limit_center
            actions = torch.clamp(actions, jll, jlu).flatten()
            self.model.joint_target.zero_()
        elif self.action_type is ActionType.POSITION:  # is ActionType.DELTA_POSITION:
            # control position directly
            joint_act = self.model.joint_target
            jlu, jll = self.joint_limit_upper, self.joint_limit_lower
            joint_limit_range = (jlu - jll) * self.action_strength  # delta size
            # actions +/-
            actions = joint_limit_range * actions  # * 0.25
            q = self.joint_q.view(self.num_envs, -1)[:, : self.num_robot_joint_q]
            actions = torch.clamp(actions, jll - q, jlu - q).flatten()
            self.model.joint_target.assign(self._joint_q)
        else:
            joint_act = self.model.joint_target
            jlu, jll = self.joint_limit_upper, self.joint_limit_lower
            # stiffness +/- 0.15 default stiffness
            joint_limit_range = (jlu - jll) * 0.4
            joint_limit_center = (jlu - jll) * 0.5 + jll
            stiff_scale = 0.25 * self.start_joint_stiffness_pt[:, : self.num_acts // 2]
            action_scale = torch.cat([joint_limit_range, stiff_scale], dim=-1)
            act_range = (
                torch.cat([jll, -stiff_scale], dim=-1),
                torch.cat([jlu, stiff_scale], dim=-1),
            )
            action_shift = torch.cat(
                [joint_limit_center, torch.zeros_like(stiff_scale)],
                dim=-1,
            )
            actions = action_scale * actions + action_shift
            actions = torch.clamp(actions, act_range[0], act_range[1])
            # act_stiffness in [-0.15 joint stiff, +0.15 joint stiff]
            # if self.num_frames % 5 == 0:
            self.model.joint_target_ke.assign(self.start_joint_stiffness)
            self.model.joint_target.zero_()
        return actions, joint_act

    def autograd_step(self, actions):
        # Steps model in autograd kernel to allow gradients to flow
        actions, joint_act = self.assign_scale_actions(actions)

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
            joint_q = self.joint_q.clone()
            joint_qd = self.joint_qd.clone()
            goal_state_vars = self.get_goal_state(self.goal_type)
            # print("body_q before step:", self.body_q[0])
            # print("state_in body_q before step:", self.state_0.body_q.numpy()[0])
            (
                self.body_q[:],
                self.body_qd[:],
                self.rew_buf,
                self.body_f,
                self.joint_q,
                self.joint_qd,
            ) = ag.forward_simulate(
                self.model,
                self.integrator,
                self.sim_dt,
                self.sim_substeps,
                joint_act,
                self.warp_actions,
                self.warp_body_f,
                self._joint_q,
                self._joint_qd,
                actions,
                body_q,
                body_qd,
                joint_q,
                joint_qd,
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
                0,
                self.num_joint_q,
                self.use_graph_capture,
            )
            # print("act:", actions.view(self.num_envs, -1)[0])
            # print("state_in body_q after step:", self.state_0.body_q.numpy()[0])
            # print("body_q after step:", self.body_q[0])
            # print("state_out body_q after step:", self.state_1.body_q.numpy()[0])
            self.num_frames += 1  # iterate num_frames after calculating reward
            # updates body_q tensors
            self.state_tensors[self.state_1.body_q] = self.body_q
            self.state_tensors[self.state_1.body_qd] = self.body_qd
            self.state_tensors[self.reward_total] = self.rew_buf
            self.state_tensors[self.warp_body_f] = self.body_f

        # self.state_0.body_q.assign(self.state_1.body_q)
        # self.state_0.body_qd.assign(self.state_1.body_qd)
        # self.state_0.body_f.assign(self.state_1.body_f)

        with torch.no_grad():
            self.actions = actions.view(self.num_envs, -1)
            # updates self.joint_q, self.joint_qd

        rewards = torch.as_tensor(self.reward_vec.numpy(), device=self.device).view(self.num_envs, -1)
        self.extras["control_pen"] = rewards[:, 0]
        self.extras["ftip_err"] = rewards[:, 1]
        self.extras["q_err"] = rewards[:, 2]
        self.extras["pos_err"] = rewards[:, 3]
        self.extras["force_err"] = rewards[:, 4]

    def warp_step(self, actions, requires_grad=False):
        """Does XPBD integrator step"""
        if requires_grad:
            self.autograd_step(actions)
        else:
            actions, joint_act = self.assign_scale_actions(actions)
            if self.forward_sim_graph is None and self.use_graph_capture:
                wp.capture_begin()
                self.forward_simulate(joint_act)
                self.forward_sim_graph = wp.capture_end()

            self.actions = actions.view(self.num_envs, -1)
            self.warp_actions.assign(wp.from_torch(self.actions.flatten()))
            if self.use_graph_capture:
                wp.capture_launch(self.forward_sim_graph)
            else:
                self.forward_simulate(joint_act)

            if self.debug:
                try:
                    assert np.isnan(self.state_0.body_f.numpy()).sum() == 0
                    assert np.isnan(self.state_1.body_q.numpy()).sum() == 0
                    assert np.isnan(self.state_1.body_qd.numpy()).sum() == 0
                except AssertionError as e:
                    print("Warning:", e)

            self.num_frames += 1  # iterate num_frames after calculating reward
            self.body_q = self.to_torch(self.state_0.body_q)
            self.body_qd = self.to_torch(self.state_0.body_qd)
            self.body_f = self.to_torch(self.warp_body_f)
            self.update_joints()
        # self.state_0.body_f.assign(self.state_1.body_deltas)

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
        if not self.requires_grad:
            self.calculateReward()

        self.reset_buf = self.reset_buf | (self.progress_buf >= self.episode_length)
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset(env_ids)

        # logging will make env run much slower
        if self.debug:
            self.write_log()
        # clear joint_q joint_qd to recompute next step
        # self.joint_q = None
        # self.joint_qd = None

        if self.requires_grad:
            self.obs_buf_before_reset = self.obs_buf.clone()
            self.extras["obs_before_reset"] = self.obs_buf_before_reset
            self.extras["episode_end"] = self.termination_buf

        # clear state tensors unless in debug mode
        if not self.debug:
            self.state_tensors.clear()

        # Automatically renders if initialized with render=True
        if self.visualize:
            self.render()

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def save_log(self, path):
        log = OrderedDict([(k, np.stack([l[k] for l in self.log])) for k in self.log[0]])
        np.savez(path, **log)
        self.log.clear()
