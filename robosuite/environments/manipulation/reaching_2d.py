import numpy as np

import robosuite.utils.transform_utils as T

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject

from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.transform_utils import *

from robosuite.utils.primitive_skills import PrimitiveSkill

import pdb
import time

class Reaching2D(SingleArmEnv):
    """
    This class corresponds to the 2D reaching task for a single robot arm.

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

        target_position (2-tuple): (x,y) position of target area

        random_init (bool): If True, initial end-effector position is set randomly (position is selected so that eef is
            within workspace boundaries and at least 0.15m away from edge of the target). If False, end-effector position
            starts at the default home position. 

        random_target (bool): If True, value of target_position is ignored and target is placed randomly on the table.

        
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
        target_position=(0.1, 0.15), # target position (height above the table)
        random_init=False,
        random_target=False,
        use_skills=False,
        normalized_params=True,
    ):

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
        self.target_position = target_position + self.table_offset[:2]

        # initial eef position
        self.initial_eef_pos = None

        # whether to use random eef position and target position
        self.random_init = random_init
        self.random_target = random_target

        # workspace boundaries
        self.workspace_x = (-0.2, 0.2)
        self.workspace_y = (-0.4, 0.4)
        self.workspace_z = (0.83, 1.3)
        self.yaw_bounds = (-0.5*np.pi, 0.5*np.pi)

        self.reset_ready = False # hack to fix target initialized in wrong position issue

        # primitive skill mode 
        self.use_skills = use_skills  
        self.skill = PrimitiveSkill(
            skill_indices={
                0 : "move_to",
                1 : "gripper_release",
            }
        )
        self.num_skills = self.skill.n_skills
        self.normalized_params = normalized_params

        # gripper state
        self.gripper_state = -1 # 1 is closed, -1 is opened

        # flags
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

    def reward(self, action=None):
        """
        Reward function for the task.

        Sparse un-normalized reward:

            - a discrete reward of 10.0 is provided if the gripper ends up in the target region

        Note that the final reward is normalized and scaled by
        reward_scale / 10.0 as well so that the max score is equal to reward_scale

        Args:
            action (np array): [NOT USED]

        Returns:
            float: reward value
        """
        reward = 0.0

        # sparse completion reward
        if self._check_success() and not self.reward_given:
            reward = 10.0
            if not self.use_skills:
                self.reward_given = True
                print("~~~~~~~~in target~~~~~~~~~~~~~~")
        
        # Scale reward if requested
        if self.reward_scale is not None:
            reward *= self.reward_scale / 10.0

        return reward

    def _reward(self, action=None):
        """
        Reward function for the task.

        Sparse un-normalized reward:

            - a discrete reward of 10.0 is provided if the gripper ends up in the target region

        Note that the final reward is normalized and scaled by
        reward_scale / 10.0 as well so that the max score is equal to reward_scale

        Args:
            action (np array): [NOT USED]

        Returns:
            float: reward value
        """
        reward = 0.0

        # sparse completion reward
        if self._check_success() and not self.reward_given:
            reward = 10.0
            self.reward_given = True
            print("~~~~~~~~in target~~~~~~~~~~~~~~")
        
        # Scale reward if requested
        if self.reward_scale is not None:
            reward *= self.reward_scale / 10.0

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

        # Initialize target object
        if self.random_target: # sample target position if using random target initialization
            self.target_position = np.concatenate((
                np.random.uniform(self.workspace_x[0] + self.target_half_size[0], self.workspace_x[1] - self.target_half_size[0], 1), # x
                np.random.uniform(self.workspace_y[0] + self.target_half_size[0], self.workspace_y[1] - self.target_half_size[1], 1), # y
            ))
        
        self.target = BoxObject(
            name="target",
            size=self.target_half_size,
            rgba=(1,0,0,1),
        )

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.target],
        )

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.target_body_id = self.sim.model.body_name2id(self.target.root_body)

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
                def eef_xyz(obs_cache):
                    return (
                        obs_cache[f"{pf}eef_pos"] 
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
                def eef_ori_mat(obs_cache):
                    # return None
                    return self.sim.data.site_xmat[self.sim.model.site_name2id("gripper0_grip_site")]

                @sensor(modality=modality)
                def gripper_state(obs_cache):
                    return self.gripper_state

                @sensor(modality=modality)
                def target_pos(obs_cache):
                    return self.target_position

                sensors = [target_pos, eef_xyz, eef_yaw, gripper_state, eef_ori_mat]
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
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        target_xpos = np.concatenate((self.target_position, np.array([self.table_offset[2]+self.target_half_size[2], 0, 0, 0, 1])))
        # target_xpos[2] = self.table_offset[2] + self.target.base_half_size[1]
        self.sim.data.set_joint_qpos(self.target.joints[0], target_xpos)

        self.reset_ready = True

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
            self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=self.target)

    def _check_success(self):
        """
        Check if ball has been placed into basket.

        Returns:
            bool: True if ball is in basket
        """
       
        ##### Success = center point of end effector is within target cylinder #####
        success = self._check_in_region(
            region_center = self.target_position,
            region_bounds = self.target_half_size,
            coord = self._eef_xpos
        )

        return success

    def _check_terminated(self):
        """
        Check if the task has completed one way or another. The following conditions lead to termination:

            - Task completion

        Returns:
            bool: True if episode is terminated
        """

        terminated = False

        # Prematurely terminate if task is success
        if self._check_success() and not self.use_skills:
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

        sf = 3 # safety factor to prevent robot from moving out of bounds
        x_in_bounds = self.workspace_x[0] < self._eef_xpos[0] + sf * action[0] / self.control_freq < self.workspace_x[1]
        y_in_bounds = self.workspace_y[0] < self._eef_xpos[1] + sf * action[1] / self.control_freq < self.workspace_y[1]
        z_in_bounds = self.workspace_z[0] < self._eef_xpos[2] + sf * action[2] / self.control_freq < self.workspace_z[1]
        
        if not (x_in_bounds and y_in_bounds and z_in_bounds):
            print(f"Action {action} out of bounds at pos {self._eef_xpos}")
            print(x_in_bounds, y_in_bounds, z_in_bounds)
        return x_in_bounds and y_in_bounds and z_in_bounds

    def step(self, action):

        # if using primitive skills
        if self.use_skills:
            done, skill_done = False, False
            obs = self.cur_obs
            # total_reward = 0
            num_timesteps = 0
            if self.normalized_params: # scale parameters if input params are normalized values
                action[self.num_skills:] = self._scale_params(action[self.num_skills:])
            
            while not done and not skill_done:
                action_ll, skill_done = self.skill.get_action(action, obs)
                obs, reward, done, info = super().step(action_ll)
                num_timesteps += 1
                if self.has_renderer:
                    self.render()
            if done:
                print(f"=====horizon reached {self.timestep}=====")
                print(f"final position {self._eef_xpos}")

            self.cur_obs = obs
            # print("eef_pos, yaw", obs["eef_xyz"], obs["eef_yaw"])
            info = {"num_timesteps": num_timesteps}

            reward = self._reward()
            if reward > 0 and not skill_done:
                reward = 0.0
                print("Reached target by accident... reward = 0")
            if self._check_success(): # check if termination condition (success) is met
                done = True

            return self.cur_obs, reward, done, info

        # if using low level inputs        
        else:
            # if input action dimension is 5, input is assumed to be [x, y, z, yaw, gripper]
            if action.shape[0] == 5:
                action = np.concatenate([action[:3], np.zeros(2), action[3:]])

            # action_in_bounds = self._check_action_in_bounds(action)

            # if end effector position is off the table, ignore the action
            # if not action_in_bounds:
            #     action[:-1] = 0
                # print(f"Action {action} out of bounds at pos {self._eef_xpos}")
            
            self.gripper_state = action[-1]
            
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

        # reset flags
        self.reward_given = False

        # wait for reset_internal to run
        while not self.reset_ready:
            pass

        if self.random_init:
            # sample random position inside workspace
            self.initial_eef_pos = np.concatenate((
                0.9*np.random.uniform(self.workspace_x[0], self.workspace_x[1], 1),
                0.9*np.random.uniform(self.workspace_y[0], self.workspace_y[1], 1),
            ))

            # sample again if start position is already inside or too close to the target
            thresh = np.array([0.15, 0.15]) # how far the starting position must be from target bounds 

            while (self._check_in_region(self.target_position[:2],self.target_half_size[:2] + thresh, self.initial_eef_pos)):
                self.initial_eef_pos = np.concatenate((
                    0.9*np.random.uniform(self.workspace_x[0], self.workspace_x[1], 1),
                    0.9*np.random.uniform(self.workspace_y[0], self.workspace_y[1], 1),
                ))

            # move the eef to the sampled initial position
            thresh = 0.005
            while np.any(np.abs(self._eef_xpos[:2] - self.initial_eef_pos) > thresh):
                action = 4 * (self.initial_eef_pos - self._eef_xpos[:2]) / np.linalg.norm(self.initial_eef_pos - self._eef_xpos[:2])
                action = np.concatenate((action, np.array([0, 0, 0, 0, -1])))
                observations, reward, done, info = self.step_no_count(action)
                # print("error to initial pos ", initial_pos - self._eef_xpos)
        
        self.cur_obs = observations
        self.reset_ready = False

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

    def _scale_params(self, params): # TODO - update after deciding which primitive skills to use
        """
        Scales normalized parameter ([-1, 1]) to appropriate raw values
        """
        params[0] = ( ((params[0] + 1) / 2 ) * (self.workspace_x[1] - self.workspace_x[0]) ) + self.workspace_x[0]
        params[1] = ( ((params[1] + 1) / 2 ) * (self.workspace_y[1] - self.workspace_y[0]) ) + self.workspace_y[0]
        params[2] = ( ((params[2] + 1) / 2 ) * (self.workspace_z[1] - self.workspace_z[0]) ) + self.workspace_z[0]
        params[3] = ( ((params[3] + 1) / 2 ) * (self.yaw_bounds[1] - self.yaw_bounds[0]) ) + self.yaw_bounds[0]

        return params