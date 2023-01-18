import numpy as np

import robosuite.utils.transform_utils as T

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject

from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.primitive_skills import PrimitiveSkill
from robosuite.utils.transform_utils import quat2yaw

from robosuite.controllers import load_controller_config
import pdb
import time

from collections import OrderedDict


class POCReaching(SingleArmEnv):
    """
    This class corresponds to the POC 2D reaching task for a single robot arm.
    Task: move to targetA -> release gripper -> move to targetB

    Args:
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
            Note: Must be a single single-arm robot!

        env_configuration (str): Specifies how to position the robots within the environment (default is "default").
            For most single arm environments, this argument has no impact on the robot setup.

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        gripper_types (str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
            with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
            overrides the default gripper. Should either be single str if same gripper type is to be used for all
            robots or else it should be a list of the same length as "robots" param

        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param

            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.

        table_full_size (3-tuple): x, y, and z dimensions of the table.

        table_friction (3-tuple): the three mujoco friction parameters for
            the table.

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        use_object_obs (bool): if True, include object (cube) information in
            the observation.

        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized

        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering

        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse

        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).

        control_freq (float): many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.

        horizon (int): Every episode lasts for exactly @horizon timesteps.

        ignore_done (bool): True if never terminating the environment (ignore @horizon).

        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables

        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.

            :Note: At least one camera must be specified if @use_camera_obs is True.

            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).

        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.

        camera_segmentations (None or str or list of str or list of list of str): Camera segmentation(s) to use
            for each camera. Valid options are:

                `None`: no segmentation sensor used
                `'instance'`: segmentation at the class-instance level
                `'class'`: segmentation at the class level
                `'element'`: segmentation at the per-geom level

            If not None, multiple types of segmentations can be specified. A [list of str / str or None] specifies
            [multiple / a single] segmentation(s) to use for all cameras. A list of list of str specifies per-camera
            segmentation setting(s) to use.

        target_half_size (2-tuple): (x,y) half size of target area 

        target_positions (list of 2-tuple): 1 or 2 (x,y) position of target area

        random_init (bool): If True, initial end-effector position is set randomly (position is selected so that eef is
            within workspace boundaries and at least 0.15m away from edge of the target). If False, end-effector position
            starts at the default home position. 

        random_target (bool): If True, value of target_position is ignored and target is placed randomly on the table.

        normalized_params (bool): only relevant when use_skills=True
            Set to True if input skill parameters are normalized to [-1,1]. Set to False if input skill parameters are raw values

    Raises:
        AssertionError: [Invalid number of robots specified]
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(0.65, 0.8, 0.15),
        table_friction=(100, 100, 100),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="frontview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mujoco",
        renderer_config=None,
        target_half_size=(0.05, 0.05, 0.001), # target width, height, thickness
        random_init=True,
        # random_target=False,
        use_skills=False, 
        normalized_params=True, 
    ):

        targetA_position = np.array([0.1, 0.3])
        targetB_position = np.array([0.1, -0.3]) # target position (height above the table)

        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))

        # reward configuration
        self.reward_scale = reward_scale

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # target
        self.target_half_size = target_half_size
        self.targetA_position = targetA_position + np.array(self.table_offset[:2])
        self.targetB_position = targetB_position + np.array(self.table_offset[:2])

        # initial eef position
        self.initial_eef_pos = None

        # whether to use random eef position and target position
        self.random_init = random_init
        # self.random_target = random_target

        # workspace boundaries
        self.workspace_x = (-0.2, 0.2)
        self.workspace_y = (-0.4, 0.4)
        self.workspace_z = (0.83, 1.3)
        self.yaw_bounds = (-0.5*np.pi, 0.5*np.pi)

        # random initialization area
        self.random_init_x = 0.9 * np.array([self.workspace_x[0], 0])
        self.random_init_y = 0.9 * np.array(self.workspace_y)

        self.reset_ready = False # hack to fix target initialized in wrong position issue

        # flags for intermediate goals
        self.reached_targetA = False # reached first target with gripper closed
        self.released_at_A = False # opened gripper at first target
        self.reached_targetB = False # reached second target with gripper open

        # keep track of gripper release/close actions
        self.gripper_state = 1 # 1 is closed, -1 is open
        self.prev_gripper_state = 1
        self.gripper_close_cnt = 0
        self.gripper_open_cnt = 0

        # primitive skill mode
        self.use_skills = use_skills  
        self.skill = PrimitiveSkill(
            skill_indices={
                0 : "move_to_xy",
                1 : "gripper_release",
                # TODO - add atomic action
            }
        )
        self.num_skills = self.skill.n_skills
        self.normalized_params = normalized_params

        # whether reward has been given - completion reward is only given once
        self.reward_given = False

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
        )

        self.cur_obs = self.reset()         

    def reward(self, action=None): ### TODO ###
        """
        Reward function for the task.

        Sparse un-normalized reward:

            - a discrete reward of 10.0 is provided if entire task is complete

        Note that the final reward is normalized and scaled by
        reward_scale / 10 as well so that the max score is equal to reward_scale

        Args:
            action (np array): [NOT USED]

        Returns:
            float: reward value
        """
        reward = 0.0
        
        # check if task is complete
        if self._check_success() and not self.reward_given:
            reward = 10.0
            self.reward_given = True

        # check intermediate goals
        else:   
            if not self.reached_targetA:
                if (
                    self._check_in_region(self.targetA_position, self.target_half_size, self._eef_xpos[:2])
                    and self.gripper_state == 1 # TODO - change "gripper is closed" condition?
                ): 
                    # first goal (move to targetA with gripper closed) complete
                    self.reached_targetA = True
                    print("-----REACHED TARGET A-----")

            elif self.reached_targetA and not self.released_at_A and self.gripper_state == -1: # TODO - change "gripper is released" condition?
                # second goal (release gripper at targetA) complete
                self.released_at_A = True
                print("-----RELEASED AT TARGET A-----")


            elif self.reached_targetA and self.released_at_A and not self.reached_targetB: # TODO - add gripper condition?
                if self._check_in_region(self.targetB_position, self.target_half_size, self._eef_xpos):
                    # final goal (move to targetB) complete
                    self.reached_targetB = True
                    print("-----REACHED TARGET B-----")

        # Scale reward if requested
        if self.reward_scale is not None:
            reward *= self.reward_scale / 10

        return reward

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """

        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])
        
        self.targetA = BoxObject(
            name="targetA",
            size=self.target_half_size,
            rgba=(1,0,0,1),
        )
        self.targetB = BoxObject(
            name="targetB",
            size=self.target_half_size,
            rgba=(0,0,1,1),
        )

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.targetA, self.targetB],
        )

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.targetA_body_id = self.sim.model.body_name2id(self.targetA.root_body)
        self.targetB_body_id = self.sim.model.body_name2id(self.targetB.root_body)

    def _setup_observables(self):
            """
            Sets up observables to be used for this environment. Creates object-based observables if enabled

            Returns:
                OrderedDict: Dictionary mapping observable names to its corresponding Observable object
            """
            observables = super()._setup_observables()

            # low-level object information
            if self.use_object_obs:
                # Get robot prefix and define observables modality
                pf = self.robots[0].robot_model.naming_prefix
                modality = "object"

                @sensor(modality=modality)
                def eef_xy(obs_cache):
                    return (
                        obs_cache[f"{pf}eef_pos"][:2] 
                        if f"{pf}eef_pos" in obs_cache
                        else np.zeros(2)
                    )

                @sensor(modality=modality)
                def eef_xy_gripper(obs_cache):
                    return (
                        np.append(np.array(obs_cache[f"{pf}eef_pos"][:2]), self.gripper_state)
                        if f"{pf}eef_pos" in obs_cache
                        else np.zeros(3)
                    )  

                @sensor(modality=modality)
                def eef_yaw(obs_cache):
                    return (
                        quat2yaw(obs_cache[f"{pf}eef_quat"])
                        if f"{pf}eef_quat" in obs_cache
                        else 0
                    )

                @sensor(modality=modality)
                def targetA_pos(obs_cache):
                    return self.targetA_position

                @sensor(modality=modality)
                def targetB_pos(obs_cache):
                    return self.targetB_position

                sensors = [targetA_pos, targetB_pos, eef_xy, eef_xy_gripper, eef_yaw]
                names = [s.__name__ for s in sensors]

                # Create observables
                for name, s in zip(names, sensors):
                    observables[name] = Observable(
                        name=name,
                        sensor=s,
                        sampling_rate=self.control_freq,
                    )
    
            return observables

    def _reset_internal(self):
        # print("START reset internal")
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        targetA_xpos = np.concatenate((self.targetA_position, np.array([self.table_offset[2]+self.target_half_size[2], 0, 0, 0, 1])))
        targetB_xpos = np.concatenate((self.targetB_position, np.array([self.table_offset[2]+self.target_half_size[2], 0, 0, 0, 1])))
        print(targetA_xpos, targetB_xpos)
        self.sim.data.set_joint_qpos(self.targetA.joints[0], targetA_xpos)
        self.sim.data.set_joint_qpos(self.targetB.joints[0], targetB_xpos)

        self.reset_ready = True

        # print("TARGET PLACED AT: ", self.target_position)
        # print("END reset internal")

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the cube.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # Color the gripper visualization site according to its distance to the cube
        if vis_settings["grippers"]:
            self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=self.targetA)

    def _check_success(self): 
        """
        Check if task is complete: end effector moved to targetA -> released gripper at targetA -> moved to targetB

        Returns:
            bool: True if entire task is complete
        """

        return self.reached_targetA and self.released_at_A and self.reached_targetB

    def _detected_gripper_release(self):

        """
        Returns: True if gripper has been open for 10 or more consequtive steps
        """
        return self.gripper_open_cnt >= 10

    def _detected_gripper_close(self):
        """
        Returns: True if gripper has been closed for 10 or more consequtive steps
        """
        return self.gripper_close_cnt >= 10
 
    def _check_terminated(self):
        """
        Check if the task has completed one way or another. The following conditions lead to termination:

            - Task completion

        Returns:
            bool: True if episode is terminated
        """

        terminated = False

        # Prematurely terminate if task is success
        if self._check_success():
            print("~~~~~~~~~~~~~~ TASK COMPLETE ~~~~~~~~~~~~~~")
            terminated = True

        return terminated
    
    def _check_in_region(self, region_center, region_bounds, coord):
        """
        Check if input coordinate is inside the target

        Args
            region_center (array): 3d coordiante of region center
            region_bounds (array): defines bounds of region
                box (half size)
            coord (array): 2d coordinate to check
        """
        in_x = region_center[0] - region_bounds[0] < coord[0] < region_center[0] + region_bounds[0]
        in_y = region_center[1] - region_bounds[1] < coord[1] < region_center[1] + region_bounds[1]
        return in_x and in_y

    def _check_action_in_bounds(self, action):

        sf = 2 # safety factor to prevent robot from moving out of bounds
        x_in_bounds = self.workspace_x[0] < self._eef_xpos[0] + sf * action[0] / self.control_freq < self.workspace_x[1]
        y_in_bounds = self.workspace_y[0] < self._eef_xpos[1] + sf * action[1] / self.control_freq < self.workspace_y[1]
        return x_in_bounds and y_in_bounds

    def step(self, action):

        # if using primitive skills
        if self.use_skills:
            skill_done = False
            # TODO - sum rewards?
            obs = self.cur_obs
            total_reward = 0

            if self.normalized_params: # scale parameters if input params are normalized values
                action[self.num_skills:] = self._scale_params(action[self.num_skills:])
            
            self.prev_gripper_state = self.gripper_state

            while not skill_done:
                action_ll, skill_done = self.skill.get_action(action, obs)
                obs, reward, done, info = super().step(action_ll)
                total_reward += reward
                if self.has_renderer:
                    self.render()
                self.gripper_state = action_ll[-1]
            self.cur_obs = obs

            return self.cur_obs, total_reward, done, info
        
        # if using low level action commands
        else:
            action_in_bounds = self._check_action_in_bounds(action)

            # ignore orientation inputs
            action[3:-1] = 0

            self.prev_gripper_state = self.gripper_state
            self.gripper_state = action[-1]

            if self.gripper_state == 1 and self.prev_gripper_state == 1:
                self.gripper_close_cnt += 1
                self.gripper_open_cnt = 0
            elif self.gripper_state == -1 and self.prev_gripper_state == -1:
                self.gripper_open_cnt += 1
                self.gripper_close_cnt = 0

            # if end effector position is off the table, ignore the action
            if not action_in_bounds:
                action[:-1] = 0
                print("Action out of bounds")
            
            return super().step(action)

    def step_no_count(self, action):
        """
        Modified version of step in base environment. Used for random initialization. 
        Returns observations, but actions taken using this step function does not affect the number of steps, time, etc.
        """

        # Since the env.step frequency is slower than the mjsim timestep frequency, the internal controller will output
        # multiple torque commands in between new high level action commands. Therefore, we need to denote via
        # 'policy_step' whether the current step we're taking is simply an internal update of the controller,
        # or an actual policy update
        policy_step = True

        # Loop through the simulation at the model timestep rate until we're ready to take the next policy step
        # (as defined by the control frequency specified at the environment level)
        for i in range(int(self.control_timestep / self.model_timestep)):
            self.sim.forward()
            self._pre_action(action, policy_step)
            self.sim.step()
            self._update_observables()
            policy_step = False

        reward = 0
        done = False
        info = {}

        if self.viewer is not None and self.renderer != "mujoco":
            self.viewer.update()

        observations = self.viewer._get_observations() if self.viewer_get_obs else self._get_observations()
        return observations, reward, done, info
    
    def reset(self):
        print("Resetting....")
        observations = super().reset()

        # wait for reset_internal to run
        while not self.reset_ready:
            pass

        # start with gripper closed
        action = np.zeros(self.action_dim)
        action[-1] = 1
        for _ in range(5):
            self.step_no_count(action)

        # sample random eef initial position
        if self.random_init:
            initial_eef_pos = np.concatenate((
                np.random.uniform(self.random_init_x[0], self.random_init_x[1], 1),
                np.random.uniform(self.random_init_y[0], self.random_init_y[1], 1)
            ))
            # move eef to start position
            thresh = 0.005
            while np.any(np.abs(self._eef_xpos[:2] - initial_eef_pos) > thresh):
                action = 4 * (initial_eef_pos - self._eef_xpos[:2]) / np.linalg.norm(initial_eef_pos - self._eef_xpos[:2])
                action = np.concatenate((action, np.array([0, 0, 0, 0, -1])))
                observations, reward, done, info = self.step_no_count(action)
                # print("error to initial pos ", initial_pos - self._eef_xpos)

        self.reset_ready = False
        # print(f"EEF position {self.initial_eef_pos} sampled based on target {self.target_position}")
        # print("End reset")
        return observations

    def _post_action(self, action):
        """
        In addition to super method, add additional info if requested

        Args:
            action (np.array): Action to execute within the environment

        Returns:
            3-tuple:

                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) info about current env step
        """
        reward, done, info = super()._post_action(action)

        # allow episode to finish early if allowed
        done = done or self._check_terminated()

        return reward, done, info

    def _scale_params(self, params):
        """
        Scales normalized parameter ([-1, 1]) to appropriate raw values
        """
        params[0] = params[0] * 0.5 * (self.workspace_x[1] - self.workspace_x[0]) - np.mean(self.workspace_x)
        params[1] = params[1] * 0.5 * (self.workspace_y[1] - self.workspace_y[0]) - np.mean(self.workspace_y)
        params[2] = params[2] * 0.5 * (self.workspace_z[1] - self.workspace_z[0]) - np.mean(self.workspace_z)
        params[3] = params[3] * 0.5 * (self.yaw_bounds[1] - self.yaw_bounds[0]) - np.mean(self.yaw_bounds)
        return params