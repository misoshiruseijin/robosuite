"""
Basic, hardcoded parameterized primitive motions

Suggested wrist angle range [-0.5pi, 0.5pi]

NOTE Currently, only move_to_pos works with option return_all_states = True
"""
import numpy as np
from robosuite.utils.transform_utils import *

import pdb

class PrimitiveSkillGlobal():
    def __init__(
        self,
        skill_indices=None,
        home_pos=(0, 0, 1.011),
        waypoint_height=1.011,
        aff_pos_thresh=None,
    ):

        """
        Args:
            skill_indices (dict): assigns one-hot vector indices to primitive skill names.
                skill names can be selected from TODO            
            home_pos (3-tuple):
                position in 3d space for end effector to return to after each primitive action (excluding move_to)
                if not specified, position of the end effector when the PrimitiveSkill class is initialized is used
            home_wrist_ori (float): wrist orientation at home position in radians
            waypoint_height (float): height of waypoint used in skills such as pick, place
            aff_pos_thresh (float): position threshold used in affordance score calculation
                encourage skills to choose reach position that is within this threshold of keypoints
        """
        self.home_pos = home_pos
        self.waypoint_height = waypoint_height

        self.skill_names = [
            "move_to",
            "pick",
            "place",
            "push",
            "gripper_release",
            "gripper_close",
            "atomic",
        ]

        self.name_to_skill = {
            "move_to" : self._move_to,
            "gripper_release" : self._gripper_release,
            "gripper_close" : self._gripper_close,
            "pick" : self._pick,
            "place" : self._place,
            "push" : self._push,
            "atomic": self._atomic,
        }

        self.max_steps = {
            "move_to" : 20,
            "gripper_release" : 4,
            "gripper_close" : 4,
            "pick" : 50,
            "place" : 50,
            "push" : 100,
        }

        self.name_to_num_params = {
            "move_to" : 5,
            "gripper_release" : 0,
            "gripper_close" : 0,
            "pick" : 4,
            "place" : 4,
            "push" : 8,
        }

        self.skill_indices = skill_indices
        if not skill_indices:
            self.skill_indices = {
                0 : "move_to",
                1 : "pick",
                2 : "place",
                3 : "push",
                4 : "gripper_release",
                5 : "gripper_close",
            }

        for key in self.skill_indices.keys():
            assert self.skill_indices[key] in self.skill_names, f"skill {self.skill_indices[key]} is undefined. skill name must be one of {self.skill_names}"

        self.n_skills = len(self.skill_indices)
        self.max_num_params = max([self.name_to_num_params[skill_name] for skill_name in self.skill_indices.values()])

        self.steps = 0 # number of steps spent on one primitive skill
        self.grip_steps = 0
        self.phase = 0 # keeps track of phase in multi-step skills
        self.prev_success = False 
        self.skill_failed = False

        self.workspace_bounds_x = (-0.2, 0.2)
        self.workspace_bounds_y = (-0.4, 0.4)
        self.workspace_bounds_z = (0.83, 1.3)
        self.yaw_bounds = (-0.5*np.pi, 0.5*np.pi)

        if aff_pos_thresh is not None:
            self.aff_pos_thresh = aff_pos_thresh
        else:
            self.aff_pos_thresh = {
                "move_to" : 0.05,
                "pick" : 0.03,
                "place" : 0.03,
                "push" : 0.1,
            }
        self.aff_tanh_scaling = 1.0

    def get_action(self, action, obs):
        """
        Args:
            action (tuple): one-hot vector for skill selection concatenated with skill parameters
                one-hot vector dimension must be same as self.n_skills. skill parameter can have variable dimension

        Returns:
            action (7-tuple): action commands for simulation environment - (position commands, orientation commands, gripper command)    
            skill_done (bool): True if goal skill completed successfully or if max allowed steps is reached
        """
        # choose right skill
        skill_idx = np.argmax(action[:self.n_skills])
        skill = self.name_to_skill[self.skill_indices[skill_idx]]
        
        # extract params
        params = action[self.n_skills:]
        return skill(obs, params)

    def get_keypoints_dict(self):
        
        keypoints = {key : None for key in self.skill_indices.values()}
        return keypoints

    def compute_affordance_reward(self, action, keypoint_dict):
        """
        Computes afforance reward given action and keypoints

        Args:
            action (array): action
            keypoint_dict (dict) : maps skill name to keypoints. keypoints can be None or list of coordinates
                "None" indicates that the skill is relevant at any position (choosing this skill is never penalized regardless of position parameters)
                Empty list indicates that the skill is not relevant at any position (choosing this skill is always penalized regardless of position parameters)

        Returns:
            affordance_reward (float) : affordance reward for choosing given action
        """
        skill_idx = np.argmax(action[:self.n_skills])
        skill_name = self.skill_indices[skill_idx]
        keypoints = keypoint_dict[skill_name] # keypoints for chosen skill
        reach_pos = action[self.n_skills:self.n_skills + 3] # component of params corresponding to reach position
        if keypoints is None:
            return 1.0

        if len(keypoints) == 0:
            return 0.0

        aff_centers = np.stack(keypoints)
        dist = np.clip(np.abs(aff_centers - reach_pos) - self.aff_pos_thresh[skill_name], 0, None)
        min_dist = np.min(np.sum(dist, axis=1))
        aff_reward = 1.0 - np.tanh(self.aff_tanh_scaling * min_dist)
        return aff_reward

    def _atomic(self, obs, params, robot_id=0):
        action = np.array(params)
        return action, True

    def _move_to(self, obs, params, robot_id=0, thresh=0.005, yaw_thresh=0.15, count_steps=True):
        
        """
        Moves end effector to goal position and orientation.
        Args:
            obs: observation dict from environment
            params (tuple of floats): [goal_pos, goal_yaw, gripper_command]
                goal_pos (3-tuple): goal end effector position (x, y, z) in world
                gripper_command (float): gripper is closed if > 0, opened if <= 0 
                goal_yaw (float): goal yaw angle for end effector
            robot_id (int): specifies which robot's observations are used (if more than one robots exist in environment)
            thresh (float): how close end effector position must be to the goal for skill to be complete
            yaw_thresh (float): how close end effector yaw angle must be to the goal value for skill to be complete
            count_steps (bool): if True, steps taken in this skill will be counted (set to False if calling this skill from within another skill) 
        Returns:
            action: 7d action commands for simulation environment - (position commands, orientation commands, gripper command)
            skill_done: True if goal skill completed successfully or if max allowed steps is reached
        """

        goal_pos = params[:3]
        goal_yaw = params[3]
        gripper_action = 1 if params[4] > 0 else -1

        max_steps = self.max_steps["move_to"]
        
        skill_done = False
        success = False

        eef_pos = obs[f"robot{robot_id}_eef_pos"]
        cur_ori = _quat2euler(obs[f"robot{robot_id}_eef_quat"])
        cur_yaw = cur_ori[-1]
        eef_quat = obs[f"robot{robot_id}_eef_quat"]
        pos_error = goal_pos - eef_pos
        yaw_error = goal_yaw - cur_yaw
        # print("pos error", pos_error)
        # print("yaw error", yaw_error)

        pos_reached = np.all(np.abs(pos_error) < thresh)
        yaw_reached = np.abs(yaw_error) < yaw_thresh

        ori_action = np.append(_roll_pitch_correction(eef_quat), 0.9*yaw_error)

        action = np.concatenate([goal_pos, ori_action, np.array([gripper_action])])

        # max steps reached - skill done with fail
        if count_steps and (self.steps > max_steps):
            print("Max steps for primitive reached: ", max_steps)
            print(f"Goal was {params}\nReached {eef_pos}, {cur_yaw}")
            success = False
            skill_done = True

        # goal is reached - skill done with success
        if (pos_reached and yaw_reached):
            success = True
            skill_done = True
            if count_steps:
                self.steps = 0
        else:
            if count_steps:
                self.steps += 1

        return action, skill_done, success

    def _gripper_release(self, obs, params=(), robot_id=0):
        """
        Opens gripper

        Args:
            obs: observation dict from environment - not used
            params (tuple of floats): not used
        
        Returns:
            action: 7d action commands for simulation environment - (position commands, orientation commands, gripper command)
            skill_done: True if goal skill completed successfully or if max allowed steps is reached
        """
        max_steps = self.max_steps["gripper_release"]
        eef_pos = obs[f"robot{robot_id}_eef_pos"]
        action = np.concatenate([eef_pos, np.zeros(3), np.array([-1])])

        if self.grip_steps < max_steps:
            self.grip_steps += 1
            return action, False, False
        
        self.grip_steps = 0
        return action, True, True

    def _gripper_close(self, obs, params=(), robot_id=0):
        """
        Closes gripper

        Args:
            obs: observation dict from environment - not used
            params (tuple of floats): not used
        
        Returns:
            action: 7d action commands for simulation environment - (position commands, orientation commands, gripper command)
            skill_done: True if goal skill completed successfully or if max allowed steps is reached
        """
        max_steps = self.max_steps["gripper_close"]
        eef_pos = obs[f"robot{robot_id}_eef_pos"]
        action = np.concatenate([eef_pos, np.zeros(3), np.array([1])])

        if self.grip_steps < max_steps:
            self.grip_steps += 1
            return action, False, False
        
        self.grip_steps = 0
        return action, True, True

    def _pick(self, obs, params, robot_id=0, thresh=0.005, yaw_thresh=0.15):       
        """
        Picks up an object at a target position and returns to home position.
        Args:
            obs: observation dict from environment
            params (tuple of floats): [goal_pos, goal_yaw]
                goal_pos (3-tuple): goal end effector position (x, y, z) in world
                goal_yaw (tuple): goal yaw angle for end effector
            robot_id (int): specifies which robot's observations are used (if more than one robots exist in environment)
            speed (float): controls magnitude of position commands. Values over ~0.75 is not recommended
            thresh (float): how close end effector position must be to the goal for skill to be complete
            yaw_thresh (float): how close end effector yaw angle must be to the goal value for skill to be complete
            normalized input (bool): set to True if input parameters are normalized to [-1, 1]. set to False to use raw params
        Returns:
            action: 7d action commands for simulation environment - (position commands, orientation commands, gripper command)
            skill_done: True if goal skill completed successfully or if max allowed steps is reached
        """
        goal_pos = params[:3]
        goal_yaw = params[3]
        max_steps = self.max_steps["pick"]

        above_pos = (goal_pos[0], goal_pos[1], self.waypoint_height)

        skill_failed = False
        success = False

        if self.prev_success:
            self.phase += 1
            self.prev_success = False

        if not self.phase == 4 and self.steps > max_steps:
            self.phase = 4
            skill_failed = True
            print("max steps for pick reached:", max_steps)

        # phase 0: move to above grip site
        if self.phase == 0:
            params = np.concatenate([above_pos, np.array([goal_yaw, -1])])
            action, skill_done, self.prev_success = self._move_to(obs=obs, params=params, robot_id=robot_id, thresh=thresh, yaw_thresh=yaw_thresh, count_steps=False)

        # phase 1: move down to grip site
        if self.phase == 1:
            params = np.concatenate([goal_pos, np.array([goal_yaw, -1])])
            action, skill_done, self.prev_success = self._move_to(obs=obs, params=params, robot_id=robot_id, thresh=thresh, yaw_thresh=yaw_thresh, count_steps=False)

        # phase 2: grip
        if self.phase == 2:
            action, skill_done, self.prev_success = self._gripper_close(obs=obs)

        # phase 3: lift
        if self.phase == 3:
            params = np.concatenate([above_pos, np.array([goal_yaw, 1])])
            action, skill_done, self.prev_success = self._move_to(obs=obs, params=params, robot_id=robot_id, thresh=thresh, yaw_thresh=yaw_thresh, count_steps=False)

        # phase 4: move to home
        if self.phase == 4:
            self.steps = 0
            params = np.concatenate([self.home_pos, np.array([0, 1])])
            if skill_failed:
                params[-1] = -1
            action, skill_done, self.prev_success = self._move_to(obs=obs, params=params, robot_id=robot_id, thresh=thresh, yaw_thresh=yaw_thresh, count_steps=False)
            if self.prev_success:
                self.steps = 0
                self.phase = 0
                self.prev_success = False
                success = True
                return action, True, success
            
            return action, False, success
        
        self.steps += 1

        return action, False, success

    def _place(self, obs, params, robot_id=0, speed=0.3, thresh=0.005, yaw_thresh=0.15, max_steps=40):       
        """
        Places an object at a target position and returns to home position.
        Args:
            obs: observation dict from environment
            params (tuple of floats): [goal_pos, goal_yaw]
                goal_pos (3-tuple): goal end effector position (x, y, z) in world
                goal_yaw (tuple): goal yaw angle for end effector
            robot_id (int): specifies which robot's observations are used (if more than one robots exist in environment)
            speed (float): controls magnitude of position commands. Values over ~0.75 is not recommended
            thresh (float): how close end effector position must be to the goal for skill to be complete
            yaw_thresh (float): how close end effector yaw angle must be to the goal value for skill to be complete
            normalized input (bool): set to True if input parameters are normalized to [-1, 1]. set to False to use raw params
        Returns:
            action: 7d action commands for simulation environment - (position commands, orientation commands, gripper command)
            skill_done: True if goal skill completed successfully or if max allowed steps is reached
        """

        goal_pos = params[:3]
        goal_yaw = params[3]
        max_steps = self.max_steps["place"]

        above_pos = (goal_pos[0], goal_pos[1], self.waypoint_height)

        skill_failed = False
        success = False

        if self.prev_success:
            self.phase += 1
            self.prev_success = False

        if not self.phase == 4 and self.steps > max_steps:
            self.phase = 4
            skill_failed = True
            print("max steps for place reached:", max_steps)

        # phase 0: move to above place site
        if self.phase == 0:
            params = np.concatenate([above_pos, np.array([goal_yaw, 1])])
            action, skill_done, self.prev_success = self._move_to(obs=obs, params=params, robot_id=robot_id, thresh=thresh, yaw_thresh=yaw_thresh, count_steps=False)

        # phase 1: move down to place site
        if self.phase == 1:
            params = np.concatenate([goal_pos, np.array([goal_yaw, 1])])
            action, skill_done, self.prev_success = self._move_to(obs=obs, params=params, robot_id=robot_id, thresh=thresh, yaw_thresh=yaw_thresh, count_steps=False)

        # phase 2: release
        if self.phase == 2:
            action, skill_done, self.prev_success = self._gripper_release(obs=obs)

        # phase 3: lift
        if self.phase == 3:
            params = np.concatenate([above_pos, np.array([goal_yaw, -1])])
            action, skill_done, self.prev_success = self._move_to(obs=obs, params=params, robot_id=robot_id, thresh=thresh, yaw_thresh=yaw_thresh, count_steps=False)

        # phase 4: move to home
        if self.phase == 4:
            self.steps = 0
            params = np.concatenate([self.home_pos, np.array([0, -1])])
            action, skill_done, self.prev_success = self._move_to(obs=obs, params=params, robot_id=robot_id, thresh=thresh, yaw_thresh=yaw_thresh, count_steps=False)
            if self.prev_success:
                self.steps = 0
                self.phase = 0
                self.prev_success = False
                success = True
                return action, True, success
            
            return action, False, success
    
        self.steps += 1

        return action, False, success

    def _push(self, obs, params, robot_id=0, speed=0.3, thresh=0.005, yaw_thresh=0.15, max_steps=60):
        """
        Moves end effector to above push starting position, moves down to start position, moves to goal position, up, then back to the home position,
        Positions are defined in world coordinates

        Args:
            obs: current observation
            params:
                start_pos (3-tuple or array of floats): world coordinate location to start push
                end_pos (3-tuple or array of floats): world coordinate location to end push
                wrist_yaw (float): wrist joint angle to keep while pushing
                gripper_closed (bool): if True, keeps gripper closed during pushing 
            robot_id (int): id of robot to be controlled
            speed (float): how fast the end effector will move (0,1]
        
        Returns:
            obs, reward, done, info from environment's step function (or lists of these if self.return_all_states is True)
        """
        start_pos = params[:3]
        end_pos = params[3:6]
        goal_yaw = params[6]
        gripper_action = 1 if params[7] > 0 else -1
        max_steps = self.max_steps["push"]

        above_start_pos = (start_pos[0], start_pos[1], self.waypoint_height)
        above_end_pos = (end_pos[0], end_pos[1], self.waypoint_height)

        skill_failed = False
        success = False

        if self.prev_success:
            self.phase += 1
            self.prev_success = False

        if not self.phase == 4 and self.steps > max_steps:
            self.phase = 4
            skill_failed = True
            print("max steps for push reached:", max_steps)

        # phase 0: move to above start pos
        if self.phase == 0:
            params = np.concatenate([above_start_pos, np.array([goal_yaw, gripper_action])])
            action, skill_done, self.prev_success = self._move_to(obs=obs, params=params, robot_id=robot_id, thresh=thresh, yaw_thresh=yaw_thresh, count_steps=False)

        # phase 1: move down to place site
        if self.phase == 1:
            params = np.concatenate([start_pos, np.array([goal_yaw, gripper_action])])
            action, skill_done, self.prev_success = self._move_to(obs=obs, params=params, robot_id=robot_id, thresh=thresh, yaw_thresh=yaw_thresh, count_steps=False)

        # phase 2: move to end pos
        if self.phase == 2:
            self.grip_steps = 0
            params = np.concatenate([end_pos, np.array([goal_yaw, gripper_action])])
            action, skill_done, self.prev_success = self._move_to(obs=obs, params=params, robot_id=robot_id, thresh=thresh, yaw_thresh=yaw_thresh, count_steps=False)

        # phase 3: move to above end pos
        if self.phase == 3:
            self.grip_steps = 0
            params = np.concatenate([above_end_pos, np.array([goal_yaw, gripper_action])])
            action, skill_done, self.prev_success = self._move_to(obs=obs, params=params, robot_id=robot_id, thresh=thresh, yaw_thresh=yaw_thresh, count_steps=False)

        # phase 4: move to home
        if self.phase == 4:
            self.steps = 0
            params = np.concatenate([self.home_pos, np.array([0, gripper_action])])
            action, skill_done, self.prev_success = self._move_to(obs=obs, params=params, robot_id=robot_id, thresh=thresh, yaw_thresh=yaw_thresh, count_steps=False)
            if self.prev_success:
                self.steps = 0
                self.phase = 0
                self.prev_success = False
                success = True
                return action, True, success
            
            return action, False, success
        
        self.steps += 1

        return action, False, success
class PrimitiveSkillDelta():
    def __init__(
        self,
        skill_indices=None,
        home_pos=(0, 0, 1.011),
        home_wrist_ori=0.0,
        waypoint_height=1.011,
        aff_pos_thresh=None,
        use_yaw=False,
    ):

        """
        Args:
            skill_indices (dict): assigns one-hot vector indices to primitive skill names.
                skill names can be selected from TODO            
            home_pos (3-tuple):
                position in 3d space for end effector to return to after each primitive action (excluding move_to)
                if not specified, position of the end effector when the PrimitiveSkill class is initialized is used
            home_wrist_ori (float): wrist orientation at home position in radians
            waypoint_height (float): height of waypoint used in skills such as pick, place
        """
        self.home_pos = home_pos
        self.home_wrist_ori = _wrap_to_pi(home_wrist_ori)
        self.waypoint_height = waypoint_height
        self.use_yaw = use_yaw

        self.skill_names = [
            "move_to",
            "pick",
            "place",
            "push",
            "gripper_release",
            "gripper_close",
            "atomic",
        ]

        self.name_to_skill = {
            "move_to" : self._move_to,
            "gripper_release" : self._gripper_release,
            "gripper_close" : self._gripper_close,
            "pick" : self._pick,
            "place" : self._place,
            "push" : self._push,
            "atomic": self._atomic,
        }
        
        self.max_steps = {
            "move_to" : 250,
            "gripper_release" : 10,
            "gripper_close" : 10,
            "pick" : 250,
            "place" : 250,
            "push" : 250,
        }
        
        self.name_to_num_params = {
            "move_to" : 5 if self.use_yaw else 4,
            "gripper_release" : 0,
            "gripper_close" : 0,
            "pick" : 4 if self.use_yaw else 3,
            "place" : 4 if self.use_yaw else 3,
            "push" : 8 if self.use_yaw else 7,
            "atomic": 7,
        }

        self.skill_indices = skill_indices
        if not skill_indices:
            self.skill_indices = {
                0 : "move_to",
                1 : "pick",
                2 : "place",
                3 : "push",
                4 : "gripper_release",
                5 : "gripper_close",
                6 : "atomic",
            }

        for key in self.skill_indices.keys():
            assert self.skill_indices[key] in self.skill_names, f"skill {self.skill_indices[key]} is undefined. skill name must be one of {self.skill_names}"

        self.n_skills = len(self.skill_indices)
        self.max_num_params = max([self.name_to_num_params[skill_name] for skill_name in self.skill_indices.values()])

        self.steps = 0 # number of steps spent on one primitive skill
        self.grip_steps = 0
        self.phase = 0 # keeps track of phase in multi-step skills
        self.prev_success = False 
        self.skill_failed = False

        self.workspace_bounds_x = (-0.2, 0.2)
        self.workspace_bounds_y = (-0.4, 0.4)
        self.workspace_bounds_z = (0.83, 1.3)
        self.yaw_bounds = (-0.5*np.pi, 0.5*np.pi)

        if aff_pos_thresh is not None:
            self.aff_pos_thresh = aff_pos_thresh
        else:
            self.aff_pos_thresh = {
                "move_to" : 0.05,
                "pick" : 0.01,
                "place" : 0.01,
                "push" : 0.1,
            }
        self.aff_tanh_scaling = 1.0

    def get_action(self, action, obs):
        """
        Args:
            action (tuple): one-hot vector for skill selection concatenated with skill parameters
                one-hot vector dimension must be same as self.n_skills. skill parameter can have variable dimension

        Returns:
            action (7-tuple): action commands for simulation environment - (position commands, orientation commands, gripper command)    
            skill_done (bool): True if goal skill completed successfully or if max allowed steps is reached
        """
        # choose right skill
        skill_idx = np.argmax(action[:self.n_skills])
        skill = self.name_to_skill[self.skill_indices[skill_idx]]
        
        # extract params
        params = action[self.n_skills:]
        return skill(obs, params)

    def get_keypoints_dict(self):
        
        keypoints = {key : None for key in self.skill_indices.values()}
        return keypoints
    
    def compute_affordance_reward(self, action, keypoint_dict):
        """
        Computes afforance reward given action and keypoints

        Args:
            action (array): action
            keypoint_dict (dict) : maps skill name to keypoints. keypoints can be None or list of coordinates
                "None" indicates that the skill is relevant at any position (choosing this skill is never penalized regardless of position parameters)
                Empty list indicates that the skill is not relevant at any position (choosing this skill is always penalized regardless of position parameters)

        Returns:
            affordance_reward (float) : affordance reward for choosing given action
        """
        skill_idx = np.argmax(action[:self.n_skills])
        skill_name = self.skill_indices[skill_idx]
        keypoints = keypoint_dict[skill_name] # keypoints for chosen skill
        reach_pos = action[self.n_skills:self.n_skills + 3] # component of params corresponding to reach position
        if keypoints is None:
            return 1.0

        if len(keypoints) == 0:
            return 0.0

        aff_centers = np.stack(keypoints)
        dist = np.clip(np.abs(aff_centers - reach_pos) - self.aff_pos_thresh[skill_name], 0, None)
        min_dist = np.min(np.sum(dist, axis=1))
        aff_reward = 1.0 - np.tanh(self.aff_tanh_scaling * min_dist)
        return aff_reward

    def _atomic(self, obs, params, robot_id=0):
        action = np.array(params)
        return action, True
        
    def _move_to(self, obs, params, robot_id=0, speed=0.3, thresh=0.005, yaw_thresh=0.1, slow_speed=0.15, count_steps=False):
        
        """
        Moves end effector to goal position and orientation.
        Args:
            obs: observation dict from environment
            params (tuple of floats): [goal_pos, goal_yaw, gripper_command]
                goal_pos (3-tuple): goal end effector position (x, y, z) in world
                gripper_command (float): gripper is closed if > 0, opened if <= 0 
                goal_yaw (tuple): goal yaw angle for end effector
            robot_id (int): specifies which robot's observations are used (if more than one robots exist in environment)
            speed (float): controls magnitude of position commands. Values over ~0.75 is not recommended
            thresh (float): how close end effector position must be to the goal for skill to be complete
            yaw_thresh (float): how close end effector yaw angle must be to the goal value for skill to be complete
            normalized input (bool): set to True if input parameters are normalized to [-1, 1]. set to False to use raw params
        Returns:
            action: 7d action commands for simulation environment - (position commands, orientation commands, gripper command)
            skill_done: True if goal skill completed successfully or if max allowed steps is reached
        """
        goal_pos = params[:3]
        # goal_yaw = params[3]
        # gripper_action = params[4]
        if self.use_yaw:
            goal_yaw = params[3]
            gripper_action = 1 if params[4] > 0 else -1

        else:
            goal_yaw = 0.0
            gripper_action = 1 if params[3] > 0 else -1

        max_steps = self.max_steps["move_to"]

        ori_speed = 0.2
        slow_dist = 0.02 # slow down when the end effector is slow_dist away from goal

        skill_done = False
        success = False

        eef_pos = obs[f"robot{robot_id}_eef_pos"]
        eef_quat = obs[f"robot{robot_id}_eef_quat"]
        pos_error = goal_pos - eef_pos

        if goal_yaw:
            goal_ori = _wrap_to_pi(goal_yaw)
        else:
            goal_ori = self.home_wrist_ori

        cur_ori = _quat2euler(obs[f"robot{robot_id}_eef_quat"])
        cur_yaw = cur_ori[-1]
        yaw_error = goal_ori - cur_yaw
        pos_reached = np.all(np.abs(pos_error) < thresh)
        yaw_reached = np.abs(yaw_error) < yaw_thresh
        
        # set goal reached condition depending on use_yaw parameter
        if self.use_yaw:
            goal_reached = pos_reached and yaw_reached
        else:
            goal_reached = pos_reached

        # if close to goal, reduce speed
        if np.abs(np.linalg.norm(pos_error)) < slow_dist:
            speed = slow_speed
        if abs(yaw_error) < 0.75:
            ori_speed = 0.05
        pos_action = speed * (pos_error / np.linalg.norm(pos_error)) # unit vector in direction of goal * speed
        ori_action = np.append(_roll_pitch_correction(eef_quat), np.sign(yaw_error) * ori_speed)
        action = np.concatenate([pos_action, ori_action, np.array([gripper_action])])

        # max steps reached - skill done with fail
        if count_steps and (self.steps > max_steps):
            print("Max steps for primitive reached: ", max_steps)
            print(f"Goal was {params}\nReached {eef_pos}, {cur_yaw}")
            success = False
            skill_done = True

        # goal is reached - skill done with success
        if goal_reached:
            success = True
            skill_done = True
            if count_steps:
                self.steps = 0
        else:
            if count_steps:
                self.steps += 1

        return action, skill_done, success

    def _gripper_release(self, obs={}, params=(), robot_id=0):
        """
        Opens gripper

        Args:
            obs: observation dict from environment - not used
            params (tuple of floats): not used
        
        Returns:
            action: 7d action commands for simulation environment - (position commands, orientation commands, gripper command)
            skill_done: True if goal skill completed successfully or if max allowed steps is reached
        """
        max_steps = self.max_steps["gripper_release"]
        action = np.array([0, 0, 0, 0, 0, 0, -1])

        if self.grip_steps < max_steps:
            self.grip_steps += 1
            return action, False, False
        
        self.grip_steps = 0
        return action, True, True

    def _gripper_close(self, obs={}, params=(), robot_id=0):
        """
        Closes gripper

        Args:
            obs: observation dict from environment - not used
            params (tuple of floats): not used
        
        Returns:
            action: 7d action commands for simulation environment - (position commands, orientation commands, gripper command)
            skill_done: True if goal skill completed successfully or if max allowed steps is reached
        """
        max_steps = self.max_steps["gripper_close"]
        action = np.array([0, 0, 0, 0, 0, 0, 1])

        if self.grip_steps < max_steps:
            self.grip_steps += 1
            return action, False, False
        
        self.grip_steps = 0
        return action, True, True

    def _pick(self, obs, params, robot_id=0, speed=0.3, thresh=0.005, yaw_thresh=0.05):       
        """
        Picks up an object at a target position and returns to home position.
        Args:
            obs: observation dict from environment
            params (tuple of floats): [goal_pos, goal_yaw]
                goal_pos (3-tuple): goal end effector position (x, y, z) in world
                goal_yaw (tuple): goal yaw angle for end effector
            robot_id (int): specifies which robot's observations are used (if more than one robots exist in environment)
            speed (float): controls magnitude of position commands. Values over ~0.75 is not recommended
            thresh (float): how close end effector position must be to the goal for skill to be complete
            yaw_thresh (float): how close end effector yaw angle must be to the goal value for skill to be complete
            normalized input (bool): set to True if input parameters are normalized to [-1, 1]. set to False to use raw params
        Returns:
            action: 7d action commands for simulation environment - (position commands, orientation commands, gripper command)
            skill_done: True if goal skill completed successfully or if max allowed steps is reached
        """
        goal_pos = params[:3]
        if self.use_yaw:
            goal_yaw = params[3]
        else:
            goal_yaw = 0.0
    
        max_steps = self.max_steps["pick"]

        above_pos = (goal_pos[0], goal_pos[1], self.waypoint_height)

        skill_failed = False
        success = False

        if self.prev_success:
            self.phase += 1
            self.prev_success = False

        # if max steps reached go to rehoming phase
        if not self.phase == 4 and self.steps > max_steps:
            self.phase = 4
            skill_failed = True
            print("max steps for pick reached:", max_steps)

        # phase 0: move to above grip site
        if self.phase == 0:
            if self.use_yaw:
                params = np.concatenate([above_pos, np.array([goal_yaw, -1])])
            else:
                params = np.concatenate([above_pos, [-1]])
            action, skill_done, self.prev_success = self._move_to(obs=obs, params=params, robot_id=robot_id, speed=speed, thresh=thresh, yaw_thresh=yaw_thresh, count_steps=False)

        # phase 1: move down to grip site
        if self.phase == 1:
            if self.use_yaw:
                params = np.concatenate([goal_pos, np.array([goal_yaw, -1])])
            else:
                params = np.concatenate([goal_pos, [-1]])
            action, skill_done, self.prev_success = self._move_to(obs=obs, params=params, robot_id=robot_id, speed=speed, thresh=thresh, yaw_thresh=yaw_thresh, count_steps=False)

        # phase 2: grip
        if self.phase == 2:
            action, skill_done, self.prev_success = self._gripper_close()

        # phase 3: lift
        if self.phase == 3:
            if self.use_yaw:
                params = np.concatenate([above_pos, np.array([goal_yaw, 1])])
            else:
                params = np.concatenate([above_pos, [1]])
            action, skill_done, self.prev_success = self._move_to(obs=obs, params=params, robot_id=robot_id, speed=speed, thresh=thresh, yaw_thresh=yaw_thresh, count_steps=False)

        # phase 4: move to home
        if self.phase == 4:
            self.grip_steps = 0
            if self.use_yaw:
                params = np.concatenate([self.home_pos, np.array([0, 1])])
            else:
                params = np.concatenate([self.home_pos, [1]])

            if skill_failed:
                params[-1] = -1
            action, skill_done, self.prev_success = self._move_to(obs=obs, params=params, robot_id=robot_id, speed=speed, thresh=thresh, yaw_thresh=yaw_thresh, count_steps=False)
            if self.prev_success:
                self.steps = 0
                self.phase = 0
                self.prev_success = False
                success = True
                return action, True, success

            return action, False, success
        
        self.steps += 1

        return action, False, success

    def _place(self, obs, params, robot_id=0, speed=0.3, thresh=0.005, yaw_thresh=0.05):       
        """
        Places an object at a target position and returns to home position.
        Args:
            obs: observation dict from environment
            params (tuple of floats): [goal_pos, goal_yaw]
                goal_pos (3-tuple): goal end effector position (x, y, z) in world
                goal_yaw (tuple): goal yaw angle for end effector
            robot_id (int): specifies which robot's observations are used (if more than one robots exist in environment)
            speed (float): controls magnitude of position commands. Values over ~0.75 is not recommended
            thresh (float): how close end effector position must be to the goal for skill to be complete
            yaw_thresh (float): how close end effector yaw angle must be to the goal value for skill to be complete
            normalized input (bool): set to True if input parameters are normalized to [-1, 1]. set to False to use raw params
        Returns:
            action: 7d action commands for simulation environment - (position commands, orientation commands, gripper command)
            skill_done: True if goal skill completed successfully or if max allowed steps is reached
        """
        goal_pos = params[:3]        
        if self.use_yaw:
            goal_yaw = params[3]
        else:
            goal_yaw = 0.0
            
        max_steps = self.max_steps["place"]

        above_pos = (goal_pos[0], goal_pos[1], self.waypoint_height)

        skill_failed = False
        success = False

        if self.prev_success:
            self.phase += 1
            self.prev_success = False
        
        if not self.phase == 4 and self.steps > max_steps:
            self.phase = 4
            skill_failed = True
            print("max steps for place reached:", max_steps)

        # phase 0: move to above place site
        if self.phase == 0:
            if self.use_yaw:
                params = np.concatenate([above_pos, np.array([goal_yaw, 1])])
            else:
                params = np.concatenate([above_pos, [1]])
            action, skill_done, self.prev_success = self._move_to(obs=obs, params=params, robot_id=robot_id, speed=speed, thresh=thresh, yaw_thresh=yaw_thresh, count_steps=False)

        # phase 1: move down to drop site
        if self.phase == 1:
            if self.use_yaw:
                params = np.concatenate([goal_pos, np.array([goal_yaw, 1])])
            else:
                params = np.concatenate([goal_pos, [1]])
                
            action, skill_done, self.prev_success = self._move_to(obs=obs, params=params, robot_id=robot_id, speed=speed, thresh=thresh, yaw_thresh=yaw_thresh, count_steps=False)

        # phase 2: release
        if self.phase == 2:
            action, skill_done, self.prev_success = self._gripper_release()

        # phase 3: lift
        if self.phase == 3:
            if self.use_yaw:
                params = np.concatenate([above_pos, np.array([goal_yaw, -1])])
            else:
                params = np.concatenate([above_pos, [-1]])
            action, skill_done, self.prev_success = self._move_to(obs=obs, params=params, robot_id=robot_id, speed=speed, thresh=thresh, yaw_thresh=yaw_thresh, count_steps=False)

        # phase 4: move to home
        if self.phase == 4:
            self.grip_steps = 0
            self.steps = 0
            failed = self.skill_failed
            if self.use_yaw:
                params = np.concatenate([self.home_pos, np.array([0, -1])])
            else:
                params = np.concatenate([self.home_pos, [-1]])
                
            action, skill_done, self.prev_success = self._move_to(obs=obs, params=params, robot_id=robot_id, speed=speed, thresh=thresh, yaw_thresh=yaw_thresh, count_steps=False)
            
            if self.prev_success:
                self.steps = 0
                self.phase = 0
                self.prev_success = False
                success = True
                return action, True, success
            return action, False, success

        self.steps += 1
        
        return action, False, success

    def _push(self, obs, params, robot_id=0, speed=0.3, thresh=0.005, yaw_thresh=0.05):
        """
        Moves end effector to above push starting position, moves down to start position, moves to goal position, up, then back to the home position,
        Positions are defined in world coordinates

        Args:
            obs: current observation
            params:
                start_pos (3-tuple or array of floats): world coordinate location to start push
                end_pos (3-tuple or array of floats): world coordinate location to end push
                wrist_yaw (float): wrist joint angle to keep while pushing
                gripper_closed (bool): if True, keeps gripper closed during pushing 
            robot_id (int): id of robot to be controlled
            speed (float): how fast the end effector will move (0,1]
        
        Returns:
            obs, reward, done, info from environment's step function (or lists of these if self.return_all_states is True)
        """
        start_pos = params[:3]
        end_pos = params[3:6]
        if self.use_yaw:
            goal_yaw = params[6]
            gripper_action = 1 if params[7] > 0 else -1
        else:
            goal_yaw = 0.0
            gripper_action = 1 if params[6] > 0 else -1

        max_steps = self.max_steps["push"]

        above_start_pos = (start_pos[0], start_pos[1], self.waypoint_height)
        above_end_pos = (end_pos[0], end_pos[1], self.waypoint_height)

        skill_failed = False
        success = False
        
        if self.prev_success:
            self.phase += 1
            self.prev_success = False

        if not self.phase == 4 and self.steps > max_steps:
            self.phase = 4
            skill_failed = True
            print("max steps for push reached:", max_steps)

        # phase 0: move to above start pos
        if self.phase == 0:
            params = np.concatenate([above_start_pos, np.array([goal_yaw, gripper_action])])
            action, skill_done, self.prev_success = self._move_to(obs=obs, params=params, robot_id=robot_id, speed=speed, thresh=thresh, yaw_thresh=yaw_thresh, count_steps=False)
        
        # phase 1: move down to start pos
        if self.phase == 1:
            params = np.concatenate([start_pos, np.array([goal_yaw, gripper_action])])
            action, skill_done, self.prev_success = self._move_to(obs=obs, params=params, robot_id=robot_id, speed=speed, thresh=thresh, yaw_thresh=yaw_thresh, count_steps=False)

        # phase 2: move to goal pos
        if self.phase == 2:
            params = np.concatenate([end_pos, np.array([goal_yaw, gripper_action])])
            action, skill_done, self.prev_success = self._move_to(obs=obs, params=params, robot_id=robot_id, speed=speed, thresh=0.03, yaw_thresh=yaw_thresh, count_steps=False, slow_speed=0.3)

        # phase 3: move to above end pos
        if self.phase == 3:
            params = np.concatenate([above_end_pos, np.array([goal_yaw, gripper_action])])
            action, skill_done, self.prev_success = self._move_to(obs=obs, params=params, robot_id=robot_id, speed=speed, thresh=thresh, yaw_thresh=yaw_thresh, count_steps=False)

        # phase 4: move to home
        if self.phase == 4:
            self.steps = 0
            params = np.concatenate([self.home_pos, np.array([0, -1])])
            action, skill_done, self.prev_success = self._move_to(obs=obs, params=params, robot_id=robot_id, speed=speed, thresh=thresh, yaw_thresh=yaw_thresh, count_steps=False)
            
            if self.prev_success:
                self.steps = 0
                self.phase = 0
                self.prev_success = False
                success = True
                return action, True, success
            return action, False, success

        self.steps += 1
        
        return action, False, success


# class PrimitiveSkillDelta():
#     def __init__(
#         self,
#         skill_indices=None,
#         home_pos=(0, 0, 1.011),
#         home_wrist_ori=0.0,
#         waypoint_height=1.011,
#         aff_pos_thresh=None,
#     ):

#         """
#         Args:
#             skill_indices (dict): assigns one-hot vector indices to primitive skill names.
#                 skill names can be selected from TODO            
#             home_pos (3-tuple):
#                 position in 3d space for end effector to return to after each primitive action (excluding move_to)
#                 if not specified, position of the end effector when the PrimitiveSkill class is initialized is used
#             home_wrist_ori (float): wrist orientation at home position in radians
#             waypoint_height (float): height of waypoint used in skills such as pick, place
#         """
#         self.home_pos = home_pos
#         self.home_wrist_ori = _wrap_to_pi(home_wrist_ori)
#         self.waypoint_height = waypoint_height

#         self.skill_names = [
#             "move_to",
#             "pick",
#             "place",
#             "push",
#             "gripper_release",
#             "gripper_close",
#             "atomic",
#         ]

#         self.name_to_skill = {
#             "move_to" : self._move_to,
#             "gripper_release" : self._gripper_release,
#             "gripper_close" : self._gripper_close,
#             "pick" : self._pick,
#             "place" : self._place,
#             "push" : self._push,
#             "atomic": self._atomic,
#         }
        
#         self.max_steps = {
#             "move_to" : 250,
#             "gripper_release" : 10,
#             "gripper_close" : 10,
#             "pick" : 250,
#             "place" : 250,
#             "push" : 250,
#         }
        
#         self.name_to_num_params = {
#             "move_to" : 5,
#             "gripper_release" : 0,
#             "gripper_close" : 0,
#             "pick" : 4,
#             "place" : 4,
#             "push" : 8,
#             "atomic": 7,
#         }

#         self.skill_indices = skill_indices
#         if not skill_indices:
#             self.skill_indices = {
#                 0 : "move_to",
#                 1 : "pick",
#                 2 : "place",
#                 3 : "push",
#                 4 : "gripper_release",
#                 5 : "gripper_close",
#                 6 : "atomic",
#             }

#         for key in self.skill_indices.keys():
#             assert self.skill_indices[key] in self.skill_names, f"skill {self.skill_indices[key]} is undefined. skill name must be one of {self.skill_names}"

#         self.n_skills = len(self.skill_indices)
#         self.max_num_params = max([self.name_to_num_params[skill_name] for skill_name in self.skill_indices.values()])

#         self.steps = 0 # number of steps spent on one primitive skill
#         self.grip_steps = 0
#         self.phase = 0 # keeps track of phase in multi-step skills
#         self.prev_success = False 
#         self.skill_failed = False

#         self.workspace_bounds_x = (-0.2, 0.2)
#         self.workspace_bounds_y = (-0.4, 0.4)
#         self.workspace_bounds_z = (0.83, 1.3)
#         self.yaw_bounds = (-0.5*np.pi, 0.5*np.pi)

#         if aff_pos_thresh is not None:
#             self.aff_pos_thresh = aff_pos_thresh
#         else:
#             self.aff_pos_thresh = {
#                 "move_to" : 0.05,
#                 "pick" : 0.01,
#                 "place" : 0.01,
#                 "push" : 0.1,
#             }
#         self.aff_tanh_scaling = 1.0

#     def get_action(self, action, obs):
#         """
#         Args:
#             action (tuple): one-hot vector for skill selection concatenated with skill parameters
#                 one-hot vector dimension must be same as self.n_skills. skill parameter can have variable dimension

#         Returns:
#             action (7-tuple): action commands for simulation environment - (position commands, orientation commands, gripper command)    
#             skill_done (bool): True if goal skill completed successfully or if max allowed steps is reached
#         """
#         # choose right skill
#         skill_idx = np.argmax(action[:self.n_skills])
#         skill = self.name_to_skill[self.skill_indices[skill_idx]]
        
#         # extract params
#         params = action[self.n_skills:]
#         return skill(obs, params)

#     def get_keypoints_dict(self):
        
#         keypoints = {key : None for key in self.skill_indices.values()}
#         return keypoints
    
#     def compute_affordance_reward(self, action, keypoint_dict):
#         """
#         Computes afforance reward given action and keypoints

#         Args:
#             action (array): action
#             keypoint_dict (dict) : maps skill name to keypoints. keypoints can be None or list of coordinates
#                 "None" indicates that the skill is relevant at any position (choosing this skill is never penalized regardless of position parameters)
#                 Empty list indicates that the skill is not relevant at any position (choosing this skill is always penalized regardless of position parameters)

#         Returns:
#             affordance_reward (float) : affordance reward for choosing given action
#         """
#         skill_idx = np.argmax(action[:self.n_skills])
#         skill_name = self.skill_indices[skill_idx]
#         keypoints = keypoint_dict[skill_name] # keypoints for chosen skill
#         reach_pos = action[self.n_skills:self.n_skills + 3] # component of params corresponding to reach position
#         if keypoints is None:
#             return 1.0

#         if len(keypoints) == 0:
#             return 0.0

#         aff_centers = np.stack(keypoints)
#         dist = np.clip(np.abs(aff_centers - reach_pos) - self.aff_pos_thresh[skill_name], 0, None)
#         min_dist = np.min(np.sum(dist, axis=1))
#         aff_reward = 1.0 - np.tanh(self.aff_tanh_scaling * min_dist)
#         return aff_reward

#     def _atomic(self, obs, params, robot_id=0):
#         action = np.array(params)
#         return action, True
        
#     def _move_to(self, obs, params, robot_id=0, speed=0.3, thresh=0.005, yaw_thresh=0.1, slow_speed=0.15, count_steps=False):
        
#         """
#         Moves end effector to goal position and orientation.
#         Args:
#             obs: observation dict from environment
#             params (tuple of floats): [goal_pos, goal_yaw, gripper_command]
#                 goal_pos (3-tuple): goal end effector position (x, y, z) in world
#                 gripper_command (float): gripper is closed if > 0, opened if <= 0 
#                 goal_yaw (tuple): goal yaw angle for end effector
#             robot_id (int): specifies which robot's observations are used (if more than one robots exist in environment)
#             speed (float): controls magnitude of position commands. Values over ~0.75 is not recommended
#             thresh (float): how close end effector position must be to the goal for skill to be complete
#             yaw_thresh (float): how close end effector yaw angle must be to the goal value for skill to be complete
#             normalized input (bool): set to True if input parameters are normalized to [-1, 1]. set to False to use raw params
#         Returns:
#             action: 7d action commands for simulation environment - (position commands, orientation commands, gripper command)
#             skill_done: True if goal skill completed successfully or if max allowed steps is reached
#         """
#         goal_pos = params[:3]
#         goal_yaw = params[3]
#         gripper_action = 1 if params[4] > 0 else -1

#         max_steps = self.max_steps["move_to"]

#         ori_speed = 0.2
#         slow_dist = 0.02 # slow down when the end effector is slow_dist away from goal

#         skill_done = False
#         success = False

#         eef_pos = obs[f"robot{robot_id}_eef_pos"]
#         eef_quat = obs[f"robot{robot_id}_eef_quat"]
#         pos_error = goal_pos - eef_pos

#         if goal_yaw:
#             goal_ori = _wrap_to_pi(goal_yaw)
#         else:
#             goal_ori = self.home_wrist_ori
#         cur_ori = _quat2euler(obs[f"robot{robot_id}_eef_quat"])
#         cur_yaw = cur_ori[-1]
#         yaw_error = goal_ori - cur_yaw
#         pos_reached = np.all(np.abs(pos_error) < thresh)
#         yaw_reached = np.abs(yaw_error) < yaw_thresh

#         # if close to goal, reduce speed
#         if np.abs(np.linalg.norm(pos_error)) < slow_dist:
#             speed = slow_speed
#         if abs(yaw_error) < 0.75:
#             ori_speed = 0.05
#         pos_action = speed * (pos_error / np.linalg.norm(pos_error)) # unit vector in direction of goal * speed
#         ori_action = np.append(_roll_pitch_correction(eef_quat), np.sign(yaw_error) * ori_speed)
#         action = np.concatenate([pos_action, ori_action, np.array([gripper_action])])

#         # max steps reached - skill done with fail
#         if count_steps and (self.steps > max_steps):
#             print("Max steps for primitive reached: ", max_steps)
#             print(f"Goal was {params}\nReached {eef_pos}, {cur_yaw}")
#             success = False
#             skill_done = True

#         # goal is reached - skill done with success
#         if (pos_reached and yaw_reached):
#             success = True
#             skill_done = True
#             if count_steps:
#                 self.steps = 0
#         else:
#             if count_steps:
#                 self.steps += 1

#         return action, skill_done, success

#     def _gripper_release(self, obs={}, params=(), robot_id=0):
#         """
#         Opens gripper

#         Args:
#             obs: observation dict from environment - not used
#             params (tuple of floats): not used
        
#         Returns:
#             action: 7d action commands for simulation environment - (position commands, orientation commands, gripper command)
#             skill_done: True if goal skill completed successfully or if max allowed steps is reached
#         """
#         max_steps = self.max_steps["gripper_release"]
#         action = np.array([0, 0, 0, 0, 0, 0, -1])

#         if self.grip_steps < max_steps:
#             self.grip_steps += 1
#             return action, False, False
        
#         self.grip_steps = 0
#         return action, True, True

#     def _gripper_close(self, obs={}, params=(), robot_id=0):
#         """
#         Closes gripper

#         Args:
#             obs: observation dict from environment - not used
#             params (tuple of floats): not used
        
#         Returns:
#             action: 7d action commands for simulation environment - (position commands, orientation commands, gripper command)
#             skill_done: True if goal skill completed successfully or if max allowed steps is reached
#         """
#         max_steps = self.max_steps["gripper_close"]
#         action = np.array([0, 0, 0, 0, 0, 0, 1])

#         if self.grip_steps < max_steps:
#             self.grip_steps += 1
#             return action, False, False
        
#         self.grip_steps = 0
#         return action, True, True

#     def _pick(self, obs, params, robot_id=0, speed=0.3, thresh=0.005, yaw_thresh=0.05):       
#         """
#         Picks up an object at a target position and returns to home position.
#         Args:
#             obs: observation dict from environment
#             params (tuple of floats): [goal_pos, goal_yaw]
#                 goal_pos (3-tuple): goal end effector position (x, y, z) in world
#                 goal_yaw (tuple): goal yaw angle for end effector
#             robot_id (int): specifies which robot's observations are used (if more than one robots exist in environment)
#             speed (float): controls magnitude of position commands. Values over ~0.75 is not recommended
#             thresh (float): how close end effector position must be to the goal for skill to be complete
#             yaw_thresh (float): how close end effector yaw angle must be to the goal value for skill to be complete
#             normalized input (bool): set to True if input parameters are normalized to [-1, 1]. set to False to use raw params
#         Returns:
#             action: 7d action commands for simulation environment - (position commands, orientation commands, gripper command)
#             skill_done: True if goal skill completed successfully or if max allowed steps is reached
#         """
#         goal_pos = params[:3]
#         goal_yaw = params[3]
#         max_steps = self.max_steps["pick"]

#         above_pos = (goal_pos[0], goal_pos[1], self.waypoint_height)

#         skill_failed = False
#         success = False

#         if self.prev_success:
#             self.phase += 1
#             self.prev_success = False

#         # if max steps reached go to rehoming phase
#         if not self.phase == 4 and self.steps > max_steps:
#             self.phase = 4
#             skill_failed = True
#             print("max steps for pick reached:", max_steps)

#         # phase 0: move to above grip site
#         if self.phase == 0:
#             params = np.concatenate([above_pos, np.array([goal_yaw, -1])])
#             action, skill_done, self.prev_success = self._move_to(obs=obs, params=params, robot_id=robot_id, speed=speed, thresh=thresh, yaw_thresh=yaw_thresh, count_steps=False)

#         # phase 1: move down to grip site
#         if self.phase == 1:
#             params = np.concatenate([goal_pos, np.array([goal_yaw, -1])])
#             action, skill_done, self.prev_success = self._move_to(obs=obs, params=params, robot_id=robot_id, speed=speed, thresh=thresh, yaw_thresh=yaw_thresh, count_steps=False)

#         # phase 2: grip
#         if self.phase == 2:
#             action, skill_done, self.prev_success = self._gripper_close()

#         # phase 3: lift
#         if self.phase == 3:
#             params = np.concatenate([above_pos, np.array([goal_yaw, 1])])
#             action, skill_done, self.prev_success = self._move_to(obs=obs, params=params, robot_id=robot_id, speed=speed, thresh=thresh, yaw_thresh=yaw_thresh, count_steps=False)

#         # phase 4: move to home
#         if self.phase == 4:
#             self.grip_steps = 0
#             params = np.concatenate([self.home_pos, np.array([0, 1])])
#             if skill_failed:
#                 params[-1] = -1
#             action, skill_done, self.prev_success = self._move_to(obs=obs, params=params, robot_id=robot_id, speed=speed, thresh=thresh, yaw_thresh=yaw_thresh, count_steps=False)
#             if self.prev_success:
#                 self.steps = 0
#                 self.phase = 0
#                 self.prev_success = False
#                 success = True
#                 return action, True, success

#             return action, False, success
        
#         self.steps += 1

#         return action, False, success

#     def _place(self, obs, params, robot_id=0, speed=0.3, thresh=0.005, yaw_thresh=0.05):       
#         """
#         Places an object at a target position and returns to home position.
#         Args:
#             obs: observation dict from environment
#             params (tuple of floats): [goal_pos, goal_yaw]
#                 goal_pos (3-tuple): goal end effector position (x, y, z) in world
#                 goal_yaw (tuple): goal yaw angle for end effector
#             robot_id (int): specifies which robot's observations are used (if more than one robots exist in environment)
#             speed (float): controls magnitude of position commands. Values over ~0.75 is not recommended
#             thresh (float): how close end effector position must be to the goal for skill to be complete
#             yaw_thresh (float): how close end effector yaw angle must be to the goal value for skill to be complete
#             normalized input (bool): set to True if input parameters are normalized to [-1, 1]. set to False to use raw params
#         Returns:
#             action: 7d action commands for simulation environment - (position commands, orientation commands, gripper command)
#             skill_done: True if goal skill completed successfully or if max allowed steps is reached
#         """
#         goal_pos = params[:3]
#         goal_yaw = params[3]
#         max_steps = self.max_steps["place"]

#         above_pos = (goal_pos[0], goal_pos[1], self.waypoint_height)

#         skill_failed = False
#         success = False

#         if self.prev_success:
#             self.phase += 1
#             self.prev_success = False
        
#         if not self.phase == 4 and self.steps > max_steps:
#             self.phase = 4
#             skill_failed = True
#             print("max steps for place reached:", max_steps)

#         # phase 0: move to above place site
#         if self.phase == 0:
#             params = np.concatenate([above_pos, np.array([goal_yaw, 1])])
#             action, skill_done, self.prev_success = self._move_to(obs=obs, params=params, robot_id=robot_id, speed=speed, thresh=thresh, yaw_thresh=yaw_thresh, count_steps=False)

#         # phase 1: move down to drop site
#         if self.phase == 1:
#             params = np.concatenate([goal_pos, np.array([goal_yaw, 1])])
#             action, skill_done, self.prev_success = self._move_to(obs=obs, params=params, robot_id=robot_id, speed=speed, thresh=thresh, yaw_thresh=yaw_thresh, count_steps=False)

#         # phase 2: release
#         if self.phase == 2:
#             action, skill_done, self.prev_success = self._gripper_release()

#         # phase 3: lift
#         if self.phase == 3:
#             params = np.concatenate([above_pos, np.array([goal_yaw, -1])])
#             action, skill_done, self.prev_success = self._move_to(obs=obs, params=params, robot_id=robot_id, speed=speed, thresh=thresh, yaw_thresh=yaw_thresh, count_steps=False)

#         # phase 4: move to home
#         if self.phase == 4:
#             self.grip_steps = 0
#             self.steps = 0
#             failed = self.skill_failed
#             params = np.concatenate([self.home_pos, np.array([0, -1])])
#             action, skill_done, self.prev_success = self._move_to(obs=obs, params=params, robot_id=robot_id, speed=speed, thresh=thresh, yaw_thresh=yaw_thresh, count_steps=False)
            
#             if self.prev_success:
#                 self.steps = 0
#                 self.phase = 0
#                 self.prev_success = False
#                 success = True
#                 return action, True, success
#             return action, False, success

#         self.steps += 1
        
#         return action, False, success

#     def _push(self, obs, params, robot_id=0, speed=0.3, thresh=0.005, yaw_thresh=0.05):
#         """
#         Moves end effector to above push starting position, moves down to start position, moves to goal position, up, then back to the home position,
#         Positions are defined in world coordinates

#         Args:
#             obs: current observation
#             params:
#                 start_pos (3-tuple or array of floats): world coordinate location to start push
#                 end_pos (3-tuple or array of floats): world coordinate location to end push
#                 wrist_yaw (float): wrist joint angle to keep while pushing
#                 gripper_closed (bool): if True, keeps gripper closed during pushing 
#             robot_id (int): id of robot to be controlled
#             speed (float): how fast the end effector will move (0,1]
        
#         Returns:
#             obs, reward, done, info from environment's step function (or lists of these if self.return_all_states is True)
#         """
#         start_pos = params[:3]
#         end_pos = params[3:6]
#         goal_yaw = params[6]
#         gripper_action = 1 if params[7] > 0 else -1
#         max_steps = self.max_steps["push"]

#         above_start_pos = (start_pos[0], start_pos[1], self.waypoint_height)
#         above_end_pos = (end_pos[0], end_pos[1], self.waypoint_height)

#         skill_failed = False
#         success = False
        
#         if self.prev_success:
#             self.phase += 1
#             self.prev_success = False

#         if not self.phase == 4 and self.steps > max_steps:
#             self.phase = 4
#             skill_failed = True
#             print("max steps for push reached:", max_steps)

#         # phase 0: move to above start pos
#         if self.phase == 0:
#             params = np.concatenate([above_start_pos, np.array([goal_yaw, gripper_action])])
#             action, skill_done, self.prev_success = self._move_to(obs=obs, params=params, robot_id=robot_id, speed=speed, thresh=thresh, yaw_thresh=yaw_thresh, count_steps=False)
        
#         # phase 1: move down to start pos
#         if self.phase == 1:
#             params = np.concatenate([start_pos, np.array([goal_yaw, gripper_action])])
#             action, skill_done, self.prev_success = self._move_to(obs=obs, params=params, robot_id=robot_id, speed=speed, thresh=thresh, yaw_thresh=yaw_thresh, count_steps=False)

#         # phase 2: move to goal pos
#         if self.phase == 2:
#             params = np.concatenate([end_pos, np.array([goal_yaw, gripper_action])])
#             action, skill_done, self.prev_success = self._move_to(obs=obs, params=params, robot_id=robot_id, speed=speed, thresh=0.03, yaw_thresh=yaw_thresh, count_steps=False, slow_speed=0.3)

#         # phase 3: move to above end pos
#         if self.phase == 3:
#             params = np.concatenate([above_end_pos, np.array([goal_yaw, gripper_action])])
#             action, skill_done, self.prev_success = self._move_to(obs=obs, params=params, robot_id=robot_id, speed=speed, thresh=thresh, yaw_thresh=yaw_thresh, count_steps=False)

#         # phase 4: move to home
#         if self.phase == 4:
#             self.steps = 0
#             params = np.concatenate([self.home_pos, np.array([0, -1])])
#             action, skill_done, self.prev_success = self._move_to(obs=obs, params=params, robot_id=robot_id, speed=speed, thresh=thresh, yaw_thresh=yaw_thresh, count_steps=False)
            
#             if self.prev_success:
#                 self.steps = 0
#                 self.phase = 0
#                 self.prev_success = False
#                 success = True
#                 return action, True, success
#             return action, False, success

#         self.steps += 1
        
#         return action, False, success


def _wrap_to_pi(angles):
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

def _wrap_to_2pi(angle):
    """
    normalize angle in rad to range [0, 2pi]
    """
    return angle % (2 * np.pi)

def _quat_to_yaw(quat):
    """
    Given quaternion, returns yaw [rad]
    """
    x = quat[0]
    y = quat[1]
    z = quat[2]
    w = quat[3]
    quat = quat / np.linalg.norm(quat)
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

def _roll_pitch_correction(quat):
    """
    Given current (roll, pitch) and desired (roll, pitch), returns roll and pitch actions to maintain desired angles
    """
    rp_des = np.array([np.pi, 0]) # desired roll and pitch to maintain
    rp = _quat2euler(quat)[:-1]
    action = 1.5 * (rp_des - rp)
    # print("rp ", rp)
    # print(action)
    return action

def _quat2euler(quat):
    rpy = mat2euler(quat2mat(quat))
    return np.array([_wrap_to_2pi(rpy[0]), _wrap_to_pi(rpy[1]), _wrap_to_pi(rpy[2])])

class PrimitiveSkill_Old():
    # NOTE: this class is outdated

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

        self.home_wrist_ori = _wrap_to_pi(home_wrist_ori)
        self.return_all_states = return_all_states

        self.move_to_external_call = False # controls whether step count is reset after move_to_pos call
        self.ignore_z = False # set to True when calling move_to_xy
        self.steps = 0 # number of steps spent on one primitive skill
        self.max_steps = 150 # number of steps each primitive can last

    def gripper_release(self):
        """
        Releases gripper

        Args:
            None
        
        Returns:
            obs, reward, done, info from environment's step function
        """
        action = np.zeros(self.env.action_dim)
        action[-1] = -1
            
        for _ in range(15):
            obs, reward, done, info = self.env.step(action)
            if self.env.has_renderer:
                self.env.render()
        
        return obs, reward, done, info

    def move_to_pos(self, obs, goal_pos, gripper_closed, wrist_ori=None, info_list=None, robot_id=0, speed=0.15, thresh=0.001, wrist_thresh=0.05):
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
            # print(self.steps)
            # break if max steps is exceeded
            if self.steps >= self.max_steps:
                print("Reached max steps for move to - terminating move_to")
                break

            # print("eef pos ", eef_pos)
            # print("goal pos ", goal_pos)
            # print("error ", error)
            # print(np.abs(error) > thresh)

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

            self.steps += 1
        
        # rotate wrist
        action = np.zeros(self.env.action_dim)
        if wrist_ori is not None:
            goal_ori = _wrap_to_pi(wrist_ori)
            cur_ori = 2 * np.arccos(_wrap_to_pi(obs[f"robot{robot_id}_eef_quat"][0])) # extract yaw from quat
            error = goal_ori - cur_ori

            while abs(error) >= wrist_thresh:
                # break if max steps is exceeded
                if self.steps >= self.max_steps:
                    print(f"max steps {self.max_steps} for primitive reached. terminating")
                    break    
                cur_ori = _wrap_to_pi(obs[f"robot{robot_id}_joint_pos"][-1]) - self.home_wrist_ori
                error = goal_ori - cur_ori
                action[-2] = -np.sign(error) * 0.2
                if error < 0.1:
                    action[-2] = -np.sign(error) * 0.1      
                obs, reward, done, info = self.env.step(action)
                if self.env.has_renderer:
                    self.env.render()
                
                self.steps += 1

        # make sure there is something to return
        action = np.zeros(self.env.action_dim)
        action[-1] = gripper_action
        obs, reward, done, info = self.env.step(action)
        self.steps += 1
        
        if self.return_all_states:
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)

        self.steps = 0

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

        goal_pos = np.append(goal_pos, obs[f"robot{robot_id}_eef_pos"][2])
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
        self.move_to_external_call = True

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

        self.move_to_external_call = False

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
        self.move_to_external_call = True

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

        self.move_to_external_call = False

        return obs, reward, done, info
    
    def push(self, obs, start_pos, end_pos, wrist_ori=None, waypoint_height=None, gripper_closed=False, robot_id=0, speed=0.15):
        """
        Moves end effector to above push starting position, moves down to start position, moves to goal position, up, then back to the home position,
        Positions are defined in world coordinates

        Args:
            obs: current observation
            start_pos (3-tuple or array of floats): world coordinate location to start push
            end_pos (3-tuple or array of floats): world coordinate location to end push
            wrist_ori (float): wrist joint angle to keep while pushing
            waypoint_height (float): height of waypoints. if none, uses home_pos height
            gripper_closed (bool): if True, keeps gripper closed during pushing 
            robot_id (int): id of robot to be controlled
            speed (float): how fast the end effector will move (0,1]
        
        Returns:
            obs, reward, done, info from environment's step function (or lists of these if self.return_all_states is True)
        """
        self.move_to_external_call = True

        if waypoint_height is None:
            waypoint_height = self.home_pos[2]
        start_above_pos = (start_pos[0], start_pos[1], waypoint_height)
        end_above_pos = (end_pos[0], end_pos[1], waypoint_height)

        # move to above start position
        obs, reward, done, info = self.move_to_pos(obs=obs, goal_pos=start_above_pos, wrist_ori=wrist_ori, gripper_closed=gripper_closed, robot_id=robot_id, speed=speed)

        # move down to start position
        obs, reward, done, info = self.move_to_pos(obs=obs, goal_pos=start_pos, wrist_ori=wrist_ori, gripper_closed=gripper_closed, robot_id=robot_id, speed=speed)

        # push
        obs, reward, done, info = self.move_to_pos(obs=obs, goal_pos=end_pos, wrist_ori=wrist_ori, gripper_closed=gripper_closed, robot_id=robot_id, speed=speed)

        # move to above end position
        obs, reward, done, info = self.move_to_pos(obs=obs, goal_pos=end_above_pos, wrist_ori=0.0, gripper_closed=gripper_closed, robot_id=robot_id, speed=speed)

        # move to home position
        obs, reward, done, info = self.move_to_pos(obs=obs, goal_pos=self.home_pos, wrist_ori=0.0, gripper_closed=False, robot_id=robot_id, speed=speed)

        self.move_to_external_call = False

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
        self.move_to_external_call = True
        
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

        self.move_to_external_call = False

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
        self.move_to_external_call = True
        
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

        self.move_to_external_call = False

        return obs, reward, done, info
