"""
Basic, hardcoded parameterized primitive motions
"""
import numpy as np

import pdb

class PrimitiveSkill():
    def __init__(
        self,
        env,
        home_pos=None,
        thresh=0.001,
        return_all_states=False,
    ):

        """
        Args:
            env: environment to take steps in
            home_pos (3-tuple):
                position in 3d space for end effector to return to after each primitive action (excluding move_to)
                if not specified, position of the end effector when the PrimitiveSkill class is initialized is used
            thresh (float): how close the end effector position must be to the goal for the primitive action to complete
            return_all_states (bool): if True, retun list of all obs, reward, done, info encountered during primitive action; if False, only return last state
        """
        self.env = env
        self.controller_config = env.robot_configs[0]["controller_config"]
        self.controller_type = self.controller_config["type"]
        if self.controller_type == "OSC_POSE":
            self.action_dim = 6
        elif self.controller_type == "OSC_POSITION":
            self.action_dim = 3
        else:
            raise Exception("Controller type must be OSC_POSE or OSC_POSITION")
        
        if home_pos is None:
            self.home_pos = env._eef_xpos
        else:
            self.home_pos = home_pos

        self.return_all_states = return_all_states
        self.thresh = thresh 

    def move_to_pos(self, obs, goal_pos, gripper_closed, robot_id=0, speed=0.15):
        """
        Moves end effector to target position (x, y, z) coordinate in a straight line path.
        Cannot be interrupted until target position is reached.
        Controller type must be OSC_POSE or OSC_POSITION.

        Args:
            obs: current observation
            goal_pos (3-tuple or array of floats): target position 
            gripper_closed (bool): whether gripper should be closed during the motion

        Returns:
            obs, reward, done, info from environment's step function
        """

        slow_speed = 0.1
        slow_dist = 0.02 # slow down when the end effector is slow_dist away from goal
        gripper_action = 1 if gripper_closed else -1

        if self.return_all_states:
            obs_list = []
            reward_list = []
            done_list = []
            info_list = []

        eef_pos = obs[f"robot{robot_id}_eef_pos"]
        error = goal_pos - eef_pos

        if self.env.has_renderer:
            self.env.render()

        while np.any(np.abs(error) > self.thresh):
            print("eef pos ", eef_pos)
            print("goal pos ", goal_pos)
            print("error ", error)

            # if close to goal, reduce speed
            if np.abs(np.linalg.norm(error)) < slow_dist:
                speed = slow_speed

            action = speed * (error / np.linalg.norm(error)) # unit vector in direction of goal * speed
            
            # adjust output dimension
            if self.action_dim == 6: 
                action = np.concatenate([action, np.zeros(3)])

            action = np.append(action, gripper_action)

            # take step in environment 
            obs, reward, done, info = self.env.step(action)

            if self.return_all_states:
                obs_list.append(obs)
                reward_list.append(reward)
                done_list.append(done)
                info_list.append(info)

            eef_pos = obs[f"robot{robot_id}_eef_pos"]
            error = goal_pos - eef_pos

            if self.env.has_renderer:
                self.env.render()
        
        # make sure there is something to return
        obs, reward, done, info = self.env.step(np.append(np.zeros(self.action_dim), gripper_action))
        obs_list.append(obs)
        reward_list.append(reward)
        done_list.append(done)
        info_list.append(info)

        if self.return_all_states:
            return obs_list, reward_list, done_list, info_list
        
        return obs, reward, done, info

    def pick(self, obs, goal_pos=None, obj_id=None, lift_height=None, robot_id=0, speed=0.15):
        """
        Moves end effector to above object to be grasped, moves down to grasp, moves up, then back to the home position.
        Grip position can be specified by either 3d location (goal_pos) or id of object to pick up (obj_id).

        Args:
            obs: current observation
            goal_pos (3-tuple or array of floats): location to grasp
            obj_id: object to pick
            lift_height (float): how high to lift the object after grasp. if none, uses home_pos height
            robot_id (int): id of robot to be controlled
            speed (float): how fast the end effector will move (0,1]
        
        Returns:
            obs, reward, done, info from environment's step function
        """
        # make sure one of goal_pos or obj_id is provided
        assert not goal_pos is None and obj_id is None, "Either goal_pos or obj_id must be given"
        assert not goal_pos is not None and obj_id is not None, "Cannot provide both goal_pos and obj_id"

        # if specified by obj_id
        if obj_id is not None:
            raise NotImplementedError

        if lift_height is None:
            lift_height = self.home_pos[2]
        above_pos = (goal_pos[0], goal_pos[1], lift_height)

        # move to above grip site
        obs, reward, done, info = self.move_to_pos(
            obs=obs,
            goal_pos=above_pos,
            gripper_closed=False,
            robot_id=robot_id,
            speed=speed
        )

        # move down to grip site
        obs, reward, done, info = self.move_to_pos(
            obs=obs,
            goal_pos=goal_pos,
            gripper_closed=False,
            robot_id=robot_id,
            speed=speed
        )

        # grip
        action = np.append(np.zeros(self.action_dim), 1)
        for _ in range(15):
            obs, reward, done, info = self.env.step(action)
            if self.env.has_renderer:
                self.env.render()

        # move up
        obs, reward, done, info = self.move_to_pos(
            obs=obs,
            goal_pos=above_pos,
            gripper_closed=True,
            robot_id=robot_id,
            speed=speed
        )

        # move to home position
        obs, reward, done, info = self.move_to_pos(
            obs=obs,
            goal_pos=self.home_pos,
            gripper_closed=True,
            robot_id=robot_id,
            speed=speed
        )

        return obs, reward, done, info
    
    def place(self, obs, goal_pos=None, obj_id=None, lift_height=None, robot_id=0, speed=0.15):
        """
        Moves end effector to above position to place, moves down to goal position and drops object, move up, then back to the home position,
        Position to release object can be specified as 3d position (goal_pos) or id of object to release the object onto (obj_id)

        Args:
            obs: current observation
            goal_pos (3-tuple or array of floats): location to grasp
            obj_id: id of object to place an object onto 
            lift_height (float): height of waypoint. if none, uses home_pos height
            robot_id (int): id of robot to be controlled
            speed (float): how fast the end effector will move (0,1]
        
        Returns:
            obs, reward, done, info from environment's step function
        """
        # make sure one of goal_pos or obj_id is provided
        assert not goal_pos is None and obj_id is None, "Either goal_pos or obj_id must be given"
        assert not goal_pos is not None and obj_id is not None, "Cannot provide both goal_pos and obj_id"

        # if specified by obj_id
        if obj_id is not None:
            raise NotImplementedError

        if lift_height is None:
            lift_height = self.home_pos[2]
        above_pos = (goal_pos[0], goal_pos[1], lift_height)

        # move to above grip site
        obs, reward, done, info = self.move_to_pos(
            obs=obs,
            goal_pos=above_pos,
            gripper_closed=True,
            robot_id=robot_id,
            speed=speed
        )

        # move down to grip site
        obs, reward, done, info = self.move_to_pos(
            obs=obs,
            goal_pos=goal_pos,
            gripper_closed=True,
            robot_id=robot_id,
            speed=speed
        )

        # release object
        action = np.append(np.zeros(self.action_dim), -1)
        for _ in range(15):
            obs, reward, done, info = self.env.step(action)
            if self.env.has_renderer:
                self.env.render()

        # move up
        obs, reward, done, info = self.move_to_pos(
            obs=obs,
            goal_pos=above_pos,
            gripper_closed=False,
            robot_id=robot_id,
            speed=speed
        )

        # move to home position
        obs, reward, done, info = self.move_to_pos(
            obs=obs,
            goal_pos=self.home_pos,
            gripper_closed=False,
            robot_id=robot_id,
            speed=speed
        )

        return obs, reward, done, info
    
    def pull_drawer(self, obs, drawer_obj, pull_dist, robot_id=0, speed=0.15):
        """
        
        """
        pass