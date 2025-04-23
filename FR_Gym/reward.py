import math
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation as R
from loguru import logger


def judge_success(self, height_threshold=0.4):
    """Determine if the target has been successfully lifted above a certain height."""
    target_pos = p.getBasePositionAndOrientation(self.target)[0]
    self.success = target_pos[2] > height_threshold


def cal_success_reward(self, distance):
    """
    Calculate success reward or penalties based on collision or successful grasp.
    Rewards are less extreme to improve training stability.
    """
    gripper_joint_indices = [8, 9]
    target_contact_points = p.getContactPoints(bodyA=self.fr5, bodyB=self.target)
    table_contact_points = p.getContactPoints(bodyA=self.fr5, bodyB=self.table)
    targettable_contact_points = p.getContactPoints(bodyA=self.fr5, bodyB=self.targettable)

    other_contact = any(cp[3] not in gripper_joint_indices for cp in target_contact_points)
    table_contact = len(table_contact_points) > 0
    targettable_contact = len(targettable_contact_points) > 0

    obstacle_collision = any(
        len(p.getContactPoints(bodyA=self.fr5, bodyB=obs_id)) > 0
        for obs_id in getattr(self, 'obstacle_ids', [])
    )

    reward = 0

    if self.success and self.step_num <= 100:
        reward = 500  # Reduced to prevent overpowering signal
        self.terminated = True
        logger.info(f"[Success] Object lifted at step {self.step_num}, distance: {distance:.4f}")

    elif obstacle_collision:
        reward = -10  # Softer penalty to maintain exploration
        self.terminated = True
        logger.info(f"[Failure] Obstacle collision at step {self.step_num}")

    elif targettable_contact or table_contact:
        reward = -5
        self.terminated = True
        logger.info(f"[Failure] Collision with base/table at step {self.step_num}")

    elif other_contact:
        reward = -5
        self.terminated = True
        logger.info(f"[Failure] Non-gripper contact with target at step {self.step_num}")

    elif self.step_num > 100:
        reward = -5
        self.terminated = True
        logger.info(f"[Failure] Exceeded step limit: {self.step_num}")

    return reward


def cal_dis_reward(self, distance):
    """Reward for moving closer to the target object."""
    if not hasattr(self, 'distance_last'):
        self.distance_last = distance
        return 0.0
    reward = 10 * (self.distance_last - distance)
    self.distance_last = distance
    return reward


def cal_pose_reward(self):
    """Reward for maintaining a good vertical grasping orientation."""
    gripper_orientation = p.getLinkState(self.fr5, 7)[1]
    euler = R.from_quat(gripper_orientation).as_euler('xyz', degrees=True)
    deviation = pow(euler[0] + 90, 2) + pow(euler[1], 2) + pow(euler[2], 2)
    return -deviation * 0.01  # Weaker regularization


def cal_grip_force_reward(self):
    """Reward proper gripping force to encourage adequate but not excessive force."""
    force_left = np.linalg.norm(p.getJointState(self.fr5, 8)[2])
    force_right = np.linalg.norm(p.getJointState(self.fr5, 9)[2])
    total_force = force_left + force_right

    if 1.5 < total_force < 5.0:
        return 5.0
    elif total_force >= 5.0:
        return -2.0
    elif total_force <= 1.5:
        return -1.0
    return 0.0


def get_distance(self):
    """Compute the Euclidean distance from the gripper center to the target object."""
    gripper_tip_pos = np.array(p.getLinkState(self.fr5, 6)[0])
    orientation = R.from_quat(p.getLinkState(self.fr5, 7)[1])
    offset = np.array([0, 0, 0.15])
    gripper_center = gripper_tip_pos + orientation.apply(offset)
    target_pos = np.array(p.getBasePositionAndOrientation(self.target)[0])
    return np.linalg.norm(gripper_center - target_pos)


def grasp_reward(self):
    """Aggregate all rewards and return the total reward and detailed info."""
    info = {}
    distance = get_distance(self)
    judge_success(self)

    success_reward = cal_success_reward(self, distance)
    distance_reward = cal_dis_reward(self, distance)
    pose_reward = cal_pose_reward(self)
    grip_force_reward = cal_grip_force_reward(self)
    efficiency_penalty = -0.01 * self.step_num  # Lower time penalty to support long episodes

    total_reward = (
        success_reward +
        distance_reward +
        pose_reward +
        grip_force_reward +
        efficiency_penalty
    )

    self.truncated = False
    self.reward = total_reward

    info['reward'] = total_reward
    info['is_success'] = self.success
    info['step_num'] = self.step_num
    info['success_reward'] = int(self.success)
    info['distance_reward'] = distance_reward
    info['pose_reward'] = pose_reward
    info['grip_force_reward'] = grip_force_reward
    info['efficiency_penalty'] = efficiency_penalty

    return total_reward, info

