"""
Basic, hardcoded parameterized primitive motions
"""
import numpy as np

import robosuite as suite
from robosuite import load_controller_config

import pdb

class PrimitiveSkill():
    def __init__(
        self,
        env
    ):
        self.env = env
        self.controller_config = env.robot_configs[0]["controller_config"]
        self.controller_type = self.controller_config["type"]
        if self.controller_type == "OSC_POSE":
            self.action_dim = 6
        elif self.controller_type == "OSC_POSITION":
            self.action_dim = 3
        else:
            raise Exception("Controller type must be OSC_POSE or OSC_POSITION")

    def move_to_pos(self, env, obs, goal_pos, gripper_closed, robot_id=0):
        """
        Moves end effector to target position (x, y, z) coordinate in a straight line path.
        Cannot be interrupted until target position is reached.
        Controller type must be OSC_POSE or OSC_POSITION.

        Args:
            env: environment to step in
            obs: current observation
            goal_pos (3-tuple or array of floats): target position 
            gripper_closed (bool): whether gripper should be closed during the motion

        Returns:
            obs, reward, done, info from environment's step function
        """

        thresh = 0.005
        speed = 0.15
        slow_speed = 0.1
        slow_dist = 0.02 # slow down when the end effector is slow_dist away from goal
        gripper_action = 1 if gripper_closed else -1

        eef_pos = obs[f"robot{robot_id}_eef_pos"]
        error = goal_pos - eef_pos
        env.render()

        while np.any(np.abs(error) > thresh):
            print("eef pos ", eef_pos)
            print("goal pos ", goal_pos)
            print("error ", error)

            # pdb.set_trace()
            # if close to goal, reduce speed
            if np.abs(np.linalg.norm(error)) < slow_dist:
                speed = slow_speed

            action = speed * (error / np.linalg.norm(error)) # unit vector in direction of goal * speed
            
            # adjust output dimension
            if self.action_dim == 6: 
                action = np.concatenate([action, np.zeros(3)])

            action = np.append(action, gripper_action)
            # pdb.set_trace()
            print("action ", action)
            print("\n")

            # take step in environment 
            obs, reward, done, info = env.step(action)
            eef_pos = obs[f"robot{robot_id}_eef_pos"]
            error = goal_pos - eef_pos
            # pdb.set_trace()
            env.render()

        return obs, reward, done, info

        
