from dmanip.utils.common import ActionType, HandType
import warp as wp
import numpy as np
import torch

from dataclasses import dataclass
from typing import Tuple, Union, Optional


@dataclass
class ObjectParams:
    joint_q_stddev: Union[float, np.ndarray] = 0.0
    joint_qd_stddev: Union[float, np.ndarray] = 0.0
    joint_q_bias: Union[float, np.ndarray] = 0.0
    joint_qd_bias: Union[float, np.ndarray] = 0.0
    joint_q_min: Union[float, np.ndarray] = 0.0
    joint_q_max: Union[float, np.ndarray] = 1.0


@dataclass
class PhysicsParams:
    gravity: Union[float, np.ndarray] = np.array([0.0, -9.81, 0.0])
    joint_target_ke: Union[float, np.ndarray] = 0.0
    stiffness: Union[float, np.ndarray] = 5000.0
    damping: Union[float, np.ndarray] = 10.0
    contact_kf: Union[float, np.ndarray] = 0.0
    limit_ke: Union[float, np.ndarray] = 1.0e4
    limit_kd: Union[float, np.ndarray] = 1.0e1
    rigid_contact_margin: Union[float, np.ndarray] = 0.1
    rigid_contact_torsional_friction: Union[float, np.ndarray] = 0.5
    rigid_contact_rolling_friction: Union[float, np.ndarray] = 0.001


@dataclass
class HandParams:
    joint_q_stddev: Union[float, np.ndarray] = 0.0
    joint_qd_stddev: Union[float, np.ndarray] = 0.0
    joint_q_bias: Union[float, np.ndarray] = 0.0
    joint_qd_bias: Union[float, np.ndarray] = 0.0
    joint_q_min: Union[float, np.ndarray] = 0.0
    joint_q_max: Union[float, np.ndarray] = 1.0


@dataclass
class AllegroHandParams(HandParams):
    joint_q_min = np.array([-0.2, -0.2, -0.2, -0.2, -0.2, -0.2])
    joint_q_max = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2])


class RandomizationParams:
    def __init__(self, hand_params=HandParams(), object_params=ObjectParams(), physics_params=PhysicsParams()):
        self.hand_params = hand_params
        self.object_params = object_params
        self.physics_params = physics_params

    def sample_params(self, include_list=[], exclude_list=[]):
        params = {}
        sampled_params = {}
        for p in self.hand_params:
            include_var = p in include_list or len(include_list) == 0
            if include_var and p not in exclude_list:
                params[p] = self.hand_params[p]
        sampled_params["hand_params"] = params

        params = {}
        for p in self.object_params:
            include_var = p in include_list or len(include_list) == 0
            if include_var and p not in exclude_list:
                params[p] = self.object_params[p]
        sampled_params["object_params"] = params

        params = {}
        for p in self.physics_params:
            include_var = p in include_list or len(include_list) == 0
            if include_var and p not in exclude_list:
                params[p] = self.physics_params[p]
        sampled_params["physics_params"] = params

        return sampled_params


class HybridParams:
    def __init__(self, hand_params=HandParams(), object_params=None, physics_params=None, device="cuda:0", hand_type: HandType = HandType.ALLEGRO, action_type: ActionType= ActionType.POSITION):
        self.device = device
        self.hand_params = hand_params
        self.object_params = object_params
        self.physics_params = physics_params
        self.hand_type = hand_type
        self.action_type = action_type
        self.init_params()

    def init_params(self):
        pass
