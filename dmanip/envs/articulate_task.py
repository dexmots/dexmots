import torch
import numpy as np
from typing import Tuple, Optional
from .hand_env import HandObjectTask
from .environment import RenderMode
from ..utils import torch_utils as tu
from ..utils.rewards import l1_dist, action_penalty
from ..utils.common import ActionType, HandType, ObjectType, run_env


class ArticulateTask(HandObjectTask):
    obs_keys = [
        "hand_joint_pos",  # in [16,20]
        "object_joint_pos",  # in [1,2,3]
        "object_body_pos",  # 3
        "fingertip_pos",  # in [12, 15]
        "hand_joint_vel",  # in [16,20]
        "object_joint_vel",  # in [1,2,3]
        "object_body_vel",  # 3
        # "object_body_torque",  # 3
        # "object_body_f",  # 3
        "object_joint_pos_err",  # in [1,2,3]
        # "object_pos_err",  # 3
    ]
    collapse_joints: bool = True

    def __init__(
        self,
        num_envs,
        num_obs=1,
        episode_length=500,
        action_type: ActionType = ActionType.POSITION,
        seed=0,
        no_grad=True,
        render=True,
        stochastic_init=True,
        device="cuda",
        render_mode=RenderMode.OPENGL,
        stage_path=None,
        object_type: Optional[ObjectType] = None,
        object_id=0,
        stiffness=1000.0,
        damping=0.5,
        reward_params=None,
        hand_type: HandType = HandType.ALLEGRO,
        hand_start_position: Tuple = (0.1, 0.3, -0.6),
        hand_start_orientation: Tuple = (-np.pi / 2 * 3, np.pi * 0.75, np.pi / 2 * 3),
        load_grasps: bool = False,
        grasp_id: int = 0,
        use_autograd: bool = True,
        reach_threshold=0.1,
        goal_joint_pos=None,
        headless=False
    ):
        if reward_params is None:
            reward_params = {
                    "action_penalty": (action_penalty, ["action"], 1e-3),
                    "object_joint_pos_err_penalty": (l1_dist, ["object_joint_pos_err"], 1e-3),
                    }
        reward_params_dict = {k: v for k, v in reward_params.items() if k != "reach_bonus"}
        self.reach_bonus = reward_params.get(
            "reach_bonus", (lambda x: torch.zeros_like(x), ("object_joint_pos_err",), 100.0)
        )
        assert object_type is not None, "object_type must be specified for articulate_task"
        # unset stochastic init if grasp_id not specified
        if load_grasps:
            stochastic_init = (self.grasps is not None and grasp_id is None) and stochastic_init
        super().__init__(
            num_envs=num_envs,
            num_obs=num_obs,
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
            stiffness=stiffness,
            damping=damping,
            reward_params=reward_params_dict,
            hand_type=hand_type,
            hand_start_position=hand_start_position,
            hand_start_orientation=hand_start_orientation,
            load_grasps=load_grasps,
            grasp_id=grasp_id,
            use_autograd=use_autograd,
            goal_joint_pos=goal_joint_pos,
            headless=headless
        )
        self.reward_extras["reach_threshold"] = reach_threshold

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

    def _get_obs_dict(self):
        obs_dict = super()._get_obs_dict()
        object_pose, object_vel = self.get_body_pos_vel("base", return_vel=True)
        obs_dict["object_body_pos"] = object_pose[:, :3]
        obs_dict["object_body_vel"] = object_vel
        obs_dict["fingertip_pos"] = self._get_fingertip_pos()
        object_joint_pos = self.joint_q.view(self.num_envs, -1)[:, self.object_joint_target_indices]
        obs_dict["object_joint_pos_err"] = object_joint_pos - self.goal_joint_pos

        self.extras["obs_dict"] = obs_dict
        early_termination = self._check_early_termination(obs_dict)
        self.reset_buf = self.reset_buf | early_termination
        return obs_dict

    def _get_fingertip_pos(self):
        fingertip_xform1 = self.get_body_pos_vel("index_link_3", return_quat=True)
        fingertip_xform2 = self.get_body_pos_vel("middle_link_3", return_quat=True)
        fingertip_xform3 = self.get_body_pos_vel("ring_link_3", return_quat=True)
        fingertip_xform4 = self.get_body_pos_vel("thumb_link_3", return_quat=True)

        fingertip_offset = torch.tensor([[0.0584, 0.0075, 0.0] for _ in range(self.num_envs)], device=str(self.device))
        fingertip_pos1 = tu.tf_apply(fingertip_xform1[:, 3:], fingertip_xform1[:, :3], fingertip_offset)
        fingertip_pos2 = tu.tf_apply(fingertip_xform2[:, 3:], fingertip_xform2[:, :3], fingertip_offset)
        fingertip_pos3 = tu.tf_apply(fingertip_xform3[:, 3:], fingertip_xform3[:, :3], fingertip_offset)
        fingertip_pos4 = tu.tf_apply(fingertip_xform4[:, 3:], fingertip_xform4[:, :3], fingertip_offset)
        fingertip_pos = torch.cat([fingertip_pos1, fingertip_pos2, fingertip_pos3, fingertip_pos4], dim=1)
        return fingertip_pos


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
    parser.add_argument("--episode_length", type=int, default=1000)
    parser.add_argument("--stiffness", type=float, default=5000.0)
    parser.add_argument("--damping", type=float, default=10.0)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--grasp_file", type=str, default="")
    parser.add_argument("--grasp_id", type=int, default=None)
    parser.add_argument("--headless", action="store_true")
    parser.set_defaults(render=True)

    args = parser.parse_args()
    if args.debug:
        import warp as wp

        wp.config.mode = "debug"
        wp.config.print_launches = True
        wp.config.verify_cuda = True

    if args.object_type is None:
        object_type = ObjectType.SPRAY_BOTTLE
    else:
        object_type = ObjectType[args.object_type.upper()]

    if args.headless:
        ArticulateTask.opengl_render_settings["headless"] = True

    rew_params = {"hand_joint_pos_err": (l1_dist, ("target_qpos", "hand_qpos"), 1.0)}
    ArticulateTask.profile = args.profile

    env = ArticulateTask(
        args.num_envs,
        56,
        1000,
        action_type=ActionType[args.action_type.upper()],
        object_type=object_type,
        object_id=args.object_id,
        hand_type=HandType[args.hand_type.upper()],
        render=args.render,
        load_grasp=args.grasp_file,
        grasp_id=args.grasp_id,
        stiffness=args.stiffness,
        damping=args.damping,
        reward_params=rew_params,
    )
    if args.headless and args.render:
        from .wrappers import Monitor

        env = Monitor(env, "outputs/videos")
    run_env(env, num_steps=50)
    env.close()
