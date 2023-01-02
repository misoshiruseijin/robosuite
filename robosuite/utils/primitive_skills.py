"""
Basic, hardcoded parameterized primitive motions

NOTE Currently, only move_to_pos works with option return_all_states = True
"""
import numpy as np

import pdb

class PrimitiveSkill():
    def __init__(
        self,
        env,
        home_pos=None,
        home_wrist_ori=0.781,
        return_all_states=False,
    ):

        """
        Args:
            env: environment to take steps in
            home_pos (3-tuple):
                position in 3d space for end effector to return to after each primitive action (excluding move_to)
                if not specified, position of the end effector when the PrimitiveSkill class is initialized is used
            home_wrist_ori (float): wrist orientation at home position in radians
            return_all_states (bool): if True, retun list of all obs, reward, done, info encountered during primitive action; if False, only return last state
        """
        self.env = env

        if home_pos is None:
            self.home_pos = env._eef_xpos
        else:
            self.home_pos = home_pos

        self.home_wrist_ori = self._wrap_to_pi(home_wrist_ori)
        self.return_all_states = return_all_states



    def gripper_release(self):
        """
        Releases gripper

        Args:
            None
        
        Returns:
            obs, reward, done, info from environment's step function
        """
        action = np.zeros(self.env.aciton_dim)
        action[-1] = -1
            
        for _ in range(10):
            obs, reward, done, info = self.env.step(action)
        
        return obs, reward, done, info

    def move_to_pos(self, obs, goal_pos, gripper_closed, wrist_ori=None, info_list=None, robot_id=0, speed=0.15, thresh=0.001, wrist_thresh=0.1):
        """
        Moves end effector to target position (x, y, z) coordinate in a straight line path.
        Cannot be interrupted until target position is reached.
        Controller type must be OSC_POSE or OSC_POSITION.

        Args:
            obs: current observation
            goal_pos (3-tuple or array of floats): target position 
            gripper_closed (bool): whether gripper should be closed during the motion
            wrist_ori (float): goal wrist orientation in radians wrt. home_wrist_ori (control mode must be OSC_POSE to use this parameter)

        Returns:
            obs, reward, done, info from environment's step function (or lists of these if self.return_all_states is True)
        """

        if wrist_ori is not None:
            assert self.env.action_dim == 7, "Control mode must be OSC_POSE to use wrist orientation parameter"
            self.env.modify_observable(observable_name=f"robot0_joint_pos", attribute="active", modifier=True)

        slow_speed = 0.15
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
        
        while np.any(np.abs(error) > thresh):
            print("eef pos ", eef_pos)
            print("goal pos ", goal_pos)
            # print("error ", error)

            # if close to goal, reduce speed
            if np.abs(np.linalg.norm(error)) < slow_dist:
                speed = slow_speed

            action = np.zeros(self.env.action_dim)
            action[:3] = speed * (error / np.linalg.norm(error)) # unit vector in direction of goal * speed            
            action[-1] = gripper_action

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
        
        # rotate wrist
        action = np.zeros(self.env.action_dim)
        if wrist_ori is not None:
            goal_ori = self._wrap_to_pi(wrist_ori) # wrt home_wrist_ori
            cur_ori = self._wrap_to_pi(obs[f"robot{robot_id}_joint_pos"][-1]) - self.home_wrist_ori # wrt home_wrist_ori
            error = goal_ori - cur_ori
            while abs(error) >= wrist_thresh:
                cur_ori = self._wrap_to_pi(obs[f"robot{robot_id}_joint_pos"][-1]) - self.home_wrist_ori
                error = goal_ori - cur_ori
                action[-2] = -np.sign(error) * 0.2
                if error < 0.1:
                    action[-2] = -np.sign(error) * 0.1      
                obs, reward, done, info = self.env.step(action)
                if self.env.has_renderer:
                    self.env.render()

        # make sure there is something to return
        action = np.zeros(self.env.action_dim)
        action[-1] = gripper_action
        obs, reward, done, info = self.env.step(action)
        
        if self.return_all_states:
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)

        if self.return_all_states:
            return obs_list, reward_list, done_list, info_list
        
        return obs, reward, done, info

    def move_to_pos_xy(self, obs, goal_pos, gripper_closed, wrist_ori=None, info_list=None, robot_id=0, speed=0.15, thresh=0.001):
        """
        Moves end effector to target position (x, y) coordinate in a straight line path.
        Cannot be interrupted until target position is reached.
        Controller type must be OSC_POSE or OSC_POSITION.

        Args:
            obs: current observation
            goal_pos (2-tuple or array of floats): target position 
            gripper_closed (bool): whether gripper should be closed during the motion

        Returns:
            obs, reward, done, info from environment's step function (or lists of these if self.return_all_states is True)
        """

        goal_pos = np.append(goal_pos, 0)
        return self.move_to_pos(obs=obs, goal_pos=goal_pos, gripper_closed=gripper_closed, wrist_ori=wrist_ori, robot_id=robot_id, speed=speed, thresh=thresh)

    def pick(self, obs, goal_pos=None, wrist_ori=None, obj_id=None, waypoint_height=None, robot_id=0, speed=0.15):
        """
        Moves end effector to above object to be grasped, moves down to grasp, moves up, then back to the home position.
        Grip position can be specified by either 3d location (goal_pos) or id of object to pick up (obj_id).

        Args:
            obs: current observation
            goal_pos (3-tuple or array of floats): location to grasp
            wrist_ori (float): goal wrist orientation in radians (control mode must be OSC_POSE to use this parameter)
            obj_id: object to pick
            waypoint_height (float): how high to lift the object after grasp. if none, uses home_pos height
            robot_id (int): id of robot to be controlled
            speed (float): how fast the end effector will move (0,1]
        
        Returns:
            obs, reward, done, info from environment's step function (or lists of these if self.return_all_states is True)
        """

        # make sure one of goal_pos or obj_id is provided
        assert not (goal_pos is None and obj_id is None), "Either goal_pos or obj_id must be given"
        assert not (goal_pos is not None and obj_id is not None), "Cannot provide both goal_pos and obj_id"

        # if specified by obj_id
        if obj_id is not None:
            # get object center position
            obj_name = self.env.sim.model.body_id2name(obj_id)
            goal_pos = self.env.sim.data.get_body_xpos(obj_name)
            grip_pos = self.env.sim.data.get_body_xpos(obj_name)

            # offset z
            goal_pos += np.array([0, 0, self.env.body_id2obj[obj_id].body_half_size[2] - 0.015])
            print("goal position ", goal_pos)

        if waypoint_height is None:
            waypoint_height = self.home_pos[2]
        above_pos = (goal_pos[0], goal_pos[1], waypoint_height)

        # move to above grip site
        obs, reward, done, info = self.move_to_pos(obs=obs, goal_pos=above_pos, wrist_ori=wrist_ori, gripper_closed=False, robot_id=robot_id, speed=speed)

        # move down to grip site
        obs, reward, done, info = self.move_to_pos(obs=obs, goal_pos=goal_pos, wrist_ori=wrist_ori, gripper_closed=False, robot_id=robot_id, speed=speed)

        # grip
        action = np.zeros(self.env.action_dim)
        action[-1] = 1
        for _ in range(15):
            obs, reward, done, info = self.env.step(action)
            if self.env.has_renderer:
                self.env.render()

        # move up
        obs, reward, done, info = self.move_to_pos(obs=obs, goal_pos=above_pos, wrist_ori=0.0, gripper_closed=True, robot_id=robot_id, speed=speed)

        # move to home position
        obs, reward, done, info = self.move_to_pos(obs=obs, goal_pos=self.home_pos, wrist_ori=0.0, gripper_closed=True, robot_id=robot_id, speed=speed)

        return obs, reward, done, info
    
    def place(self, obs, goal_pos=None, wrist_ori=None, obj_id=None, waypoint_height=None, robot_id=0, speed=0.15):
        """
        Moves end effector to above position to place, moves down to goal position and drops object, move up, then back to the home position,
        Position to release object can be specified as 3d position (goal_pos) or id of object to release the object onto (obj_id)

        Args:
            obs: current observation
            goal_pos (3-tuple or array of floats): location to grasp
            obj_id: id of object to place an object onto 
            waypoint_height (float): height of waypoint. if none, uses home_pos height
            robot_id (int): id of robot to be controlled
            speed (float): how fast the end effector will move (0,1]
        
        Returns:
            obs, reward, done, info from environment's step function (or lists of these if self.return_all_states is True)
        """
        # make sure one of goal_pos or obj_id is provided
        assert not (goal_pos is None and obj_id is None), "Either goal_pos or obj_id must be given"
        assert not (goal_pos is not None and obj_id is not None), "Cannot provide both goal_pos and obj_id"

        # if specified by obj_id
        if obj_id is not None:
            # get object center position
            obj_name = self.env.sim.model.body_id2name(obj_id)
            goal_pos = self.env.sim.data.get_body_xpos(obj_name)
            # offset z
            goal_pos += np.array([0, 0, self.env.body_id2obj[obj_id].body_half_size[2] - 0.015])

        if waypoint_height is None:
            waypoint_height = self.home_pos[2]
        above_pos = (goal_pos[0], goal_pos[1], waypoint_height)

        # move to above goal position
        obs, reward, done, info = self.move_to_pos(obs=obs, goal_pos=above_pos, wrist_ori=wrist_ori, gripper_closed=True, robot_id=robot_id, speed=speed)

        # move down to goal position
        obs, reward, done, info = self.move_to_pos(obs=obs, goal_pos=goal_pos, wrist_ori=wrist_ori, gripper_closed=True, robot_id=robot_id, speed=speed)

        # release object
        action = np.zeros(self.env.action_dim)
        action[-1] = -1
        for _ in range(15):
            obs, reward, done, info = self.env.step(action)
            if self.env.has_renderer:
                self.env.render()

        # move up
        obs, reward, done, info = self.move_to_pos(obs=obs, goal_pos=above_pos, wrist_ori=0.0, gripper_closed=False, robot_id=robot_id, speed=speed)

        # move to home position
        obs, reward, done, info = self.move_to_pos(obs=obs, goal_pos=self.home_pos, wrist_ori=0.0, gripper_closed=False, robot_id=robot_id, speed=speed)

        return obs, reward, done, info
    
    def open_drawer(self, obs, drawer_obj_id, pull_dist, pull_direction=(1,-1), waypoint_height=None, robot_id=0, speed=0.15):
        """
        Grips drawer handle and opens drawer by pull_dist (delta), then returns to home position. 
        
        Args:
            obs: current observation
            drawer_obj_id: id of drawer object to act on (object must be CabinetObject or LargeCabinetObject type) 
            pull_dist (float): amount to pull
            pull_direction (2-tuple of ints): first element is axis to pull along (0 = x, 1 = y),
                second element is direction to pull (1 = positive, -1 = negative) - e.g. (1, -1) means pull in negative y direction
            waypoint_height (float): height of waypoint. if none, uses home_pos height
            robot_id (int): id of robot to be controlled
            speed (float): how fast the end effector will move (0,1]

        Returns:
            obs, reward, done, info from environment's step function (or lists of these if self.return_all_states is True)
        """
        
        # make sure provided object is a drawer
        # NOTE: Environment must have dict named obj_body_id storing {obj_name : obj_id}. The drawer object must have "drawer" in it's name
        obj_name = list(self.env.obj_body_id.keys())[list(self.env.obj_body_id.values()).index(drawer_obj_id)]
        assert "drawer" in obj_name, "Object with id 'drawer_obj_id' must be CabinetObject or LargeCabinetObject and name must include 'drawer'"

        # get position of drawer handle site
        handle_pos = self.env.sim.data.get_site_xpos(obj_name + "_handle_site")

        if waypoint_height is None:
            waypoint_height = self.home_pos[2]
        
        above_pos1 = (handle_pos[0], handle_pos[1], waypoint_height)
        pull_pos = np.copy(handle_pos)
        pull_pos[pull_direction[0]] += pull_direction[1] * pull_dist
        above_pos2 = np.copy(pull_pos)
        above_pos2[2] = waypoint_height

        # move to above handle
        obs, reward, done, info = self.move_to_pos(obs=obs, goal_pos=above_pos1, gripper_closed=False, robot_id=robot_id, speed=speed)

        # move down
        obs, reward, done, info = self.move_to_pos(obs=obs, goal_pos=handle_pos, gripper_closed=False, robot_id=robot_id, speed=speed,)

        # grip
        action = np.zeros(self.env.action_dim)
        action[-1] = 1
        for _ in range(15):
            obs, reward, done, info = self.env.step(action)
            if self.env.has_renderer:
                self.env.render()

        # pull
        obs, reward, done, info = self.move_to_pos(obs=obs, goal_pos=pull_pos, gripper_closed=True, robot_id=robot_id, speed=speed, thresh=0.003)

        # stop
        for _ in range(15):
            obs, reward, done, info = self.env.step(action)
            if self.env.has_renderer:
                self.env.render()

        # release handle
        action[-1] = -1
        for _ in range(15):
            obs, reward, done, info = self.env.step(action)
            if self.env.has_renderer:
                self.env.render()

        # move up
        obs, reward, done, info = self.move_to_pos(obs=obs, goal_pos=above_pos2, gripper_closed=False, robot_id=robot_id, speed=speed)

        # return to home
        obs, reward, done, info = self.move_to_pos(obs=obs, goal_pos=self.home_pos, gripper_closed=False, robot_id=robot_id, speed=speed)

        return obs, reward, done, info

    def close_drawer(self, obs, drawer_obj_id, pull_dist, pull_direction=(1,1), waypoint_height=None, robot_id=0, speed=0.15):
        """
        Grips drawer handle and closes drawer by pull_dist (delta), then returns to home position. 
        
        Args:
            obs: current observation
            drawer_obj_id: id of drawer object to act on (object must be CabinetObject or LargeCabinetObject type) 
            pull_dist (float): amount to pull
            pull_direction (2-tuple of ints): first element is axis to pull along (0 = x, 1 = y),
                second element is direction to pull (1 = positive, -1 = negative) - e.g. (1, -1) means pull in negative y direction
            waypoint_height (float): height of waypoint. if none, uses home_pos height
            robot_id (int): id of robot to be controlled
            speed (float): how fast the end effector will move (0,1]

        Returns:
            obs, reward, done, info from environment's step function (or lists of these if self.return_all_states is True)
        """
        
        # make sure provided object is a drawer
        # NOTE: Environment must have dict named obj_body_id storing {obj_name : obj_id}. The drawer object must have "drawer" in it's name
        obj_name = list(self.env.obj_body_id.keys())[list(self.env.obj_body_id.values()).index(drawer_obj_id)]
        assert "drawer" in obj_name, "Object with id 'drawer_obj_id' must be CabinetObject or LargeCabinetObject and name must include 'drawer'"

        # get position of drawer handle site
        handle_pos = self.env.sim.data.get_site_xpos(obj_name + "_handle_site")

        if waypoint_height is None:
            waypoint_height = self.home_pos[2]
        
        above_pos1 = (handle_pos[0], handle_pos[1], waypoint_height)
        pull_pos = np.copy(handle_pos)
        pull_pos[pull_direction[0]] += pull_direction[1] * pull_dist
        above_pos2 = np.copy(pull_pos)
        above_pos2[2] = waypoint_height

        # move to above handle
        obs, reward, done, info = self.move_to_pos(obs=obs, goal_pos=above_pos1, gripper_closed=False, robot_id=robot_id, speed=speed)

        # move down
        obs, reward, done, info = self.move_to_pos(obs=obs, goal_pos=handle_pos, gripper_closed=False, robot_id=robot_id, speed=speed,)

        # grip
        action = np.zeros(self.env.action_dim)
        action[-1] = 1
        for _ in range(15):
            obs, reward, done, info = self.env.step(action)
            if self.env.has_renderer:
                self.env.render()

        # push
        obs, reward, done, info = self.move_to_pos(obs=obs, goal_pos=pull_pos, gripper_closed=True, robot_id=robot_id, speed=speed, thresh=0.003)

        # stop
        for _ in range(15):
            obs, reward, done, info = self.env.step(action)
            if self.env.has_renderer:
                self.env.render()

        # release handle
        action[-1] = -1
        for _ in range(15):
            obs, reward, done, info = self.env.step(action)
            if self.env.has_renderer:
                self.env.render()

        # move up
        obs, reward, done, info = self.move_to_pos(obs=obs, goal_pos=above_pos2, gripper_closed=False, robot_id=robot_id, speed=speed)

        # return to home
        obs, reward, done, info = self.move_to_pos(obs=obs, goal_pos=self.home_pos, gripper_closed=False, robot_id=robot_id, speed=speed)

        return obs, reward, done, info

    def _wrap_to_pi(self, angles):
        """
        normalize angle in rad to range [-pi, pi]
        """
        pi2 = 2 * np.pi
        result = np.fmod( np.fmod(angles, pi2) + pi2, pi2)
        if result > np.pi:
            result = result - pi2
        if result < -np.pi:
            result = result + pi2
        return result